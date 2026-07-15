"""Flat molecular-complex records used by the ESMFold2 public API.

The folding model operates on tokens and a single atom table.  This module owns
that representation, its protein-only bridge, mmCIF I/O, structure metrics, and
the compact wire format.  It deliberately has no dependency on the upstream
Biohub package; the pinned submodule is used only by differential tests.
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from subprocess import check_output
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import biotite.structure as bs
import biotite.structure.io.pdbx as pdbx
import brotli
import msgpack
import numpy as np
import torch
from biotite.structure.io.pdbx import (
    CIFCategory,
    CIFColumn,
    CIFData,
    CIFFile,
    set_structure,
)

from . import esmfold2_residue_constants as residue_constants
from .esmfold2_metrics import compute_lddt, compute_rmsd
from .esmfold2_mmcif_parsing import PLDDT_B_FACTOR_SCALE, round_mmcif_columns
from .esmfold2_protein_complex import ProteinComplex, ProteinComplexMetadata


@dataclass
class MolecularComplexResult:
    """One folded complex and the optional model outputs associated with it."""

    complex: MolecularComplex
    plddt: torch.Tensor | None = None
    ptm: float | None = None
    iptm: float | None = None
    pae: torch.Tensor | None = None
    distogram: torch.Tensor | None = None
    pair_chains_iptm: torch.Tensor | None = None
    output_embedding_sequence: torch.Tensor | None = None
    output_embedding_pair_pooled: torch.Tensor | None = None
    residue_index: torch.Tensor | None = None
    entity_id: torch.Tensor | None = None
    sae_features: np.ndarray | None = None  # X has shape (l, n_features).
    ttt_metrics: dict[str, Any] | None = None


@dataclass
class MolecularComplexMetadata:
    """Entity and chain labels carried with a molecular complex."""

    entity_lookup: dict[int, str]
    chain_lookup: dict[int, str]
    assembly_composition: dict[str, list[str]] | None = None


@dataclass
class Molecule:
    """The atom slice represented by one model token."""

    token: str
    token_idx: int
    atom_positions: np.ndarray  # P has shape (n_atoms, 3).
    atom_elements: np.ndarray  # E has shape (n_atoms,).
    atom_names: np.ndarray | None = None  # N has shape (n_atoms,) when present.
    atom_hetero: np.ndarray | None = None  # M has shape (n_atoms,) when present.
    residue_type: int = 0
    molecule_type: int = 0
    confidence: float = 0.0


_NUCLEOTIDE_NAMES = frozenset({"A", "T", "G", "C", "U", "DA", "DT", "DG", "DC"})
_SERIALIZED_ARRAYS = frozenset(
    {
        "atom_positions",
        "atom_elements",
        "atom_names",
        "atom_hetero",
        "token_to_atoms",
        "chain_id",
        "plddt",
    }
)


def _assert_table_lengths(complex_value: MolecularComplex) -> None:
    """Check that token and atom annotations align with their tables."""
    n_tokens = len(complex_value.sequence)
    n_atoms = len(complex_value.atom_positions)
    token_tables = {
        "token_to_atoms": complex_value.token_to_atoms,
        "chain_id": complex_value.chain_id,
        "plddt": complex_value.plddt,
    }
    for label, values in token_tables.items():
        assert values.shape[0] == n_tokens, f"{label} shape {values.shape} != {n_tokens} tokens"
    for label, values in (
        ("atom_names", complex_value.atom_names),
        ("atom_hetero", complex_value.atom_hetero),
    ):
        if values is not None:
            assert values.shape[0] == n_atoms, f"{label} shape {values.shape} != {n_atoms} atoms"


def _flat_protein_atoms(
    protein: ProteinComplex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten the populated atom37 entries of a protein complex."""
    positions: list[np.ndarray] = []
    elements: list[str] = []
    names: list[str] = []
    hetero: list[bool] = []
    spans: list[tuple[int, int]] = []

    for sequence_index, residue in enumerate(protein.sequence):
        if residue == "|":
            continue
        start = len(positions)
        mask = protein.atom37_mask[sequence_index]
        residue_positions = protein.atom37_positions[sequence_index]
        for atom_index in np.flatnonzero(mask):
            atom_name = residue_constants.atom_types[int(atom_index)]
            positions.append(residue_positions[atom_index])
            elements.append(atom_name[0] if atom_name else "C")
            names.append(atom_name)
            hetero.append(False)
        spans.append((start, len(positions)))

    return (
        np.asarray(positions, dtype=np.float32),
        np.asarray(elements, dtype=object),
        np.asarray(names, dtype=object),
        np.asarray(hetero, dtype=bool),
        np.asarray(spans, dtype=np.int32),
    )


def _protein_sequence_and_indices(
    complex_value: MolecularComplex,
) -> tuple[list[int], str, np.ndarray, np.ndarray]:
    protein_indices = [
        index
        for index, token in enumerate(complex_value.sequence)
        if token in residue_constants.restype_3to1
    ]
    if not protein_indices:
        raise ValueError("No protein tokens found in MolecularComplex")

    chain_ids = complex_value.chain_id[protein_indices]
    confidences = complex_value.plddt[protein_indices]
    sequence: list[str] = []
    previous_chain: Any = None
    for index, chain_id in zip(protein_indices, chain_ids, strict=True):
        if previous_chain is not None and chain_id != previous_chain:
            sequence.append("|")
        sequence.append(residue_constants.restype_3to1[complex_value.sequence[index]])
        previous_chain = chain_id
    return protein_indices, "".join(sequence), chain_ids, confidences


def _atom37_from_flat(
    complex_value: MolecularComplex, protein_indices: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    n_residues = len(protein_indices)
    positions = np.full((n_residues, 37, 3), np.nan, dtype=np.float32)
    mask = np.zeros((n_residues, 37), dtype=bool)
    if complex_value.atom_names is None:
        return positions, mask

    for residue_index, token_index in enumerate(protein_indices):
        start, stop = complex_value.token_to_atoms[token_index]
        seen: set[str] = set()
        for atom_name, atom_position in zip(
            complex_value.atom_names[start:stop],
            complex_value.atom_positions[start:stop],
            strict=True,
        ):
            normalized = str(atom_name).upper().strip()
            if normalized in seen:
                continue
            seen.add(normalized)
            atom37_index = residue_constants.atom_order.get(normalized)
            if atom37_index is not None:
                positions[residue_index, atom37_index] = atom_position
                mask[residue_index, atom37_index] = True
    return positions, mask


def _expand_protein_rows(
    sequence: str,
    protein_chain_ids: np.ndarray,
    confidences: np.ndarray,
    compact_positions: np.ndarray,
    compact_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Insert empty rows at chain separators in a protein representation."""
    n_positions = len(sequence)
    expanded = {
        "chain_id": np.full(n_positions, -1, dtype=np.int64),
        "entity_id": np.full(n_positions, -1, dtype=np.int64),
        "sym_id": np.zeros(n_positions, dtype=np.int64),
        "residue_index": np.zeros(n_positions, dtype=np.int64),
        "insertion_code": np.asarray([""] * n_positions, dtype=object),
        "confidence": np.zeros(n_positions, dtype=np.float32),
        "atom37_positions": np.full((n_positions, 37, 3), np.nan, dtype=np.float32),
        "atom37_mask": np.zeros((n_positions, 37), dtype=bool),
    }
    next_residue: dict[Any, int] = {}
    compact_index = 0
    for sequence_index, residue in enumerate(sequence):
        if residue == "|":
            continue
        chain_id = protein_chain_ids[compact_index]
        residue_number = next_residue.get(chain_id, 0) + 1
        next_residue[chain_id] = residue_number
        expanded["chain_id"][sequence_index] = chain_id
        expanded["entity_id"][sequence_index] = chain_id
        expanded["residue_index"][sequence_index] = residue_number
        expanded["confidence"][sequence_index] = confidences[compact_index]
        expanded["atom37_positions"][sequence_index] = compact_positions[compact_index]
        expanded["atom37_mask"][sequence_index] = compact_mask[compact_index]
        compact_index += 1
    return expanded


def _read_cif(source: str) -> CIFFile:
    if os.path.exists(source):
        return pdbx.CIFFile.read(source)
    return pdbx.CIFFile.read(io.StringIO(source))


def _read_structure(cif_file: CIFFile) -> Any:
    try:
        return pdbx.get_structure(cif_file, model=1, extra_fields=["b_factor"])
    except (KeyError, ValueError):
        try:
            return pdbx.get_structure(cif_file)
        except Exception:
            return pdbx.get_structure(cif_file, model=None)


def _column_array(category: Any, name: str) -> np.ndarray:
    column = category[name]
    if hasattr(column, "as_array"):
        return column.as_array(str)
    return np.asarray(list(column), dtype=str)


def _label_asym_ids(cif_file: CIFFile, n_structure_atoms: int) -> list[str] | None:
    """Return label-asym identifiers after applying Biohub's atom filters."""
    block = cif_file.block
    if "atom_site" not in block or "label_asym_id" not in block["atom_site"]:
        return None
    atom_site = block["atom_site"]
    labels = _column_array(atom_site, "label_asym_id")
    keep = np.ones(len(labels), dtype=bool)
    if "pdbx_PDB_model_num" in atom_site:
        keep &= _column_array(atom_site, "pdbx_PDB_model_num") == "1"
    if "label_alt_id" in atom_site:
        keep &= np.isin(_column_array(atom_site, "label_alt_id"), [".", "?", "", "A"])
    filtered = labels[keep]
    return filtered.tolist() if len(filtered) == n_structure_atoms else None


def _entity_metadata(cif_file: CIFFile) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    try:
        category = cif_file.block["entity"]
        if "id" not in category or "type" not in category:
            return result
        for entity_id, entity_type in zip(category["id"], category["type"], strict=False):
            result[entity_id] = entity_type
    except Exception:
        return {}
    return result


def _group_structure_atoms(
    structure: Any, labels: list[str] | None
) -> dict[str, dict[tuple[int, str], dict[str, Any]]]:
    grouped: dict[str, dict[tuple[int, str], dict[str, Any]]] = {}
    for atom_index, atom in enumerate(structure):
        chain = labels[atom_index] if labels is not None else atom.chain_id
        residues = grouped.setdefault(chain, {})
        key = (atom.res_id, atom.res_name)
        record = residues.setdefault(
            key,
            {"atoms": [], "res_name": atom.res_name, "is_hetero": atom.hetero},
        )
        record["atoms"].append(atom)
    return grouped


def _flatten_structure_groups(
    grouped: dict[str, dict[tuple[int, str], dict[str, Any]]],
) -> tuple[
    list[str],
    list[np.ndarray],
    list[str],
    list[str],
    list[bool],
    list[tuple[int, int]],
    list[float],
    list[int],
    dict[str, int],
]:
    tokens: list[str] = []
    positions: list[np.ndarray] = []
    elements: list[str] = []
    names: list[str] = []
    hetero: list[bool] = []
    spans: list[tuple[int, int]] = []
    confidences: list[float] = []
    token_chains: list[int] = []
    chain_numbers = {chain: index for index, chain in enumerate(sorted(grouped))}

    for chain in sorted(grouped):
        for residue_key in sorted(grouped[chain]):
            record = grouped[chain][residue_key]
            if record["res_name"] == "HOH":
                continue
            atoms = record["atoms"]
            tokens.append(record["res_name"])
            token_chains.append(chain_numbers[chain])
            start = len(positions)
            positions.extend(atom.coord for atom in atoms)
            elements.extend(atom.element for atom in atoms)
            names.extend(atom.atom_name for atom in atoms)
            hetero.extend(atom.hetero for atom in atoms)
            spans.append((start, len(positions)))
            b_factor = getattr(atoms[0], "b_factor", 50.0) if atoms else 50.0
            confidences.append(min(b_factor / PLDDT_B_FACTOR_SCALE, 1.0))
    return (
        tokens,
        positions,
        elements,
        names,
        hetero,
        spans,
        confidences,
        token_chains,
        chain_numbers,
    )


def _chain_entity_maps(
    complex_value: MolecularComplex,
) -> tuple[dict[str, list[str]], dict[str, int], dict[int, tuple[str, ...]]]:
    chains: dict[str, list[str]] = {}
    for token_index, numeric_chain in enumerate(complex_value.chain_id):
        numeric = int(numeric_chain)
        label = complex_value.metadata.chain_lookup.get(numeric, chr(65 + numeric))
        chains.setdefault(label, []).append(complex_value.sequence[token_index])

    sequence_entities: dict[tuple[str, ...], int] = {}
    chain_entities: dict[str, int] = {}
    entity_sequences: dict[int, tuple[str, ...]] = {}
    for label, sequence in chains.items():
        key = tuple(sequence)
        entity_id = sequence_entities.get(key)
        if entity_id is None:
            entity_id = len(sequence_entities) + 1
            sequence_entities[key] = entity_id
            entity_sequences[entity_id] = key
        chain_entities[label] = entity_id
    return chains, chain_entities, entity_sequences


def _cif_column(values: list[str]) -> CIFColumn:
    return CIFColumn(data=CIFData(array=np.asarray(values), dtype=np.str_))


def _add_entity_categories(
    cif_file: CIFFile,
    complex_value: MolecularComplex,
    entity_sequences: dict[int, tuple[str, ...]],
) -> None:
    ids: list[str] = []
    types: list[str] = []
    descriptions: list[str] = []
    for entity_id in sorted(entity_sequences):
        sequence = entity_sequences[entity_id]
        protein = any(token in residue_constants.restype_3to1 for token in sequence)
        nucleic = any(token in _NUCLEOTIDE_NAMES for token in sequence)
        ids.append(str(entity_id))
        types.append("polymer" if protein or nucleic else "non-polymer")
        if protein:
            descriptions.append(f"Polymer entity {entity_id} (protein)")
        elif nucleic:
            descriptions.append(f"Polymer entity {entity_id} (nucleic acid)")
        else:
            descriptions.append(f"Non-polymer entity {entity_id}")

    if ids:
        cif_file.block["entity"] = CIFCategory(
            name="entity",
            columns={
                "id": _cif_column(ids),
                "type": _cif_column(types),
                "pdbx_description": _cif_column(descriptions),
            },
        )

    _, chain_entities, _ = _chain_entity_maps(complex_value)
    if chain_entities:
        labels = sorted(chain_entities)
        cif_file.block["struct_asym"] = CIFCategory(
            name="struct_asym",
            columns={
                "id": _cif_column(labels),
                "entity_id": _cif_column([str(chain_entities[label]) for label in labels]),
            },
        )

    entity_chains: dict[int, list[str]] = {}
    for chain, entity_id in chain_entities.items():
        entity_chains.setdefault(entity_id, []).append(chain)
    polymer_rows: list[tuple[str, str, str, str]] = []
    residue_rows: list[tuple[str, str, str, str]] = []
    for entity_id in sorted(entity_sequences):
        sequence = entity_sequences[entity_id]
        protein = any(token in residue_constants.restype_3to1 for token in sequence)
        nucleic = any(token in _NUCLEOTIDE_NAMES for token in sequence)
        if not (protein or nucleic):
            continue
        if protein:
            polymer_type = "polypeptide(L)"
            canonical = "".join(
                residue_constants.restype_3to1.get(token, "(X)") for token in sequence
            )
        else:
            polymer_type = (
                "polyribonucleotide"
                if "U" in sequence
                else (
                    "polydeoxyribonucleotide"
                    if any(token in {"DA", "DT", "DG", "DC"} for token in sequence)
                    else "polyribonucleotide"
                )
            )
            nucleotide_letters = {"DA": "A", "DT": "T", "DG": "G", "DC": "C"}
            canonical = "".join(nucleotide_letters.get(token, token) for token in sequence)
        strand_ids = ",".join(sorted(entity_chains.get(entity_id, []))) or "?"
        polymer_rows.append((str(entity_id), polymer_type, strand_ids, canonical))
        residue_rows.extend(
            (str(entity_id), str(number), token, "n")
            for number, token in enumerate(sequence, start=1)
        )

    if polymer_rows:
        columns = list(zip(*polymer_rows, strict=True))
        cif_file.block["entity_poly"] = CIFCategory(
            name="entity_poly",
            columns={
                "entity_id": _cif_column(list(columns[0])),
                "type": _cif_column(list(columns[1])),
                "pdbx_strand_id": _cif_column(list(columns[2])),
                "pdbx_seq_one_letter_code_can": _cif_column(list(columns[3])),
            },
        )
    if residue_rows:
        columns = list(zip(*residue_rows, strict=True))
        cif_file.block["entity_poly_seq"] = CIFCategory(
            name="entity_poly_seq",
            columns={
                "entity_id": _cif_column(list(columns[0])),
                "num": _cif_column(list(columns[1])),
                "mon_id": _cif_column(list(columns[2])),
                "hetero": _cif_column(list(columns[3])),
            },
        )


def _fallback_atom_names(token: str, count: int) -> list[str]:
    if token in residue_constants.restype_3to1:
        names = list(residue_constants.residue_atoms.get(token, ["N", "CA", "C", "O"]))[:count]
        names.extend(f"X{index + 1}" for index in range(len(names), count))
        return names
    return [f"C{index + 1}" for index in range(count)]


def _as_atom_array(complex_value: MolecularComplex, chain_entities: dict[str, int]) -> bs.AtomArray:
    n_atoms = len(complex_value.atom_positions)
    atom_array = bs.AtomArray(length=n_atoms)
    atom_array.coord = complex_value.atom_positions
    residue_ids = np.zeros(n_atoms, dtype=np.int32)
    chain_labels = np.empty(n_atoms, dtype=object)
    residue_names = np.empty(n_atoms, dtype=object)
    hetero = np.zeros(n_atoms, dtype=bool)
    b_factors = np.zeros(n_atoms, dtype=np.float32)
    atom_names = np.empty(n_atoms, dtype=object)
    entity_ids = np.zeros(n_atoms, dtype=np.int32)
    next_residue: dict[Any, int] = {}

    for token_index, (start, stop) in enumerate(complex_value.token_to_atoms):
        token = complex_value.sequence[token_index]
        numeric_chain = complex_value.chain_id[token_index]
        numeric = int(numeric_chain)
        chain = complex_value.metadata.chain_lookup.get(numeric, chr(65 + numeric))
        residue_id = next_residue.get(numeric_chain, 0) + 1
        next_residue[numeric_chain] = residue_id
        count = int(stop - start)
        names = (
            list(complex_value.atom_names[start:stop])
            if complex_value.atom_names is not None
            else _fallback_atom_names(token, count)
        )
        residue_ids[start:stop] = residue_id
        chain_labels[start:stop] = chain
        residue_names[start:stop] = token
        hetero[start:stop] = (
            complex_value.atom_hetero[start:stop]
            if complex_value.atom_hetero is not None
            else token not in residue_constants.restype_3to1
        )
        b_factors[start:stop] = complex_value.plddt[token_index] * PLDDT_B_FACTOR_SCALE
        atom_names[start:stop] = names
        entity_ids[start:stop] = chain_entities.get(chain, 1)

    atom_array.res_id = residue_ids
    atom_array.chain_id = np.asarray(chain_labels, dtype="U16")
    atom_array.res_name = np.asarray(residue_names, dtype="U8")
    atom_array.hetero = hetero
    atom_array.atom_name = np.asarray(atom_names, dtype="U4")
    atom_array.add_annotation("b_factor", dtype=float)
    atom_array.b_factor = b_factors
    atom_array.add_annotation("occupancy", dtype=float)
    atom_array.occupancy = np.ones(n_atoms, dtype=np.float32)
    atom_array.add_annotation("entity_id", dtype=int)
    atom_array.entity_id = entity_ids
    if complex_value.atom_elements is not None and len(complex_value.atom_elements) == n_atoms:
        atom_array.element = np.asarray(complex_value.atom_elements, dtype="U4")
    else:
        atom_array.element = bs.infer_elements(atom_array)
    return atom_array


def _repair_label_entity_ids(cif_file: CIFFile, chain_entities: dict[str, int]) -> None:
    if "atom_site" not in cif_file.block:
        return
    atom_site = cif_file.block["atom_site"]
    if "label_asym_id" not in atom_site or "label_entity_id" not in atom_site:
        return
    labels = _column_array(atom_site, "label_asym_id").tolist()
    if labels:
        atom_site["label_entity_id"] = _cif_column(
            [str(chain_entities.get(label, 1)) for label in labels]
        )


def _centroid_tensors(
    mobile: MolecularComplex,
    target: MolecularComplex,
    *,
    retain_missing: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(mobile) != len(target):
        raise ValueError(
            f"Complexes must have the same number of tokens: {len(mobile)} vs {len(target)}"
        )
    mobile_centers: list[np.ndarray] = []
    target_centers: list[np.ndarray] = []
    valid: list[bool] = []
    for token_index in range(len(mobile)):
        mobile_start, mobile_stop = mobile.token_to_atoms[token_index]
        target_start, target_stop = target.token_to_atoms[token_index]
        mobile_atoms = mobile.atom_positions[mobile_start:mobile_stop]
        target_atoms = target.atom_positions[target_start:target_stop]
        present = len(mobile_atoms) > 0 and len(target_atoms) > 0
        if not present and not retain_missing:
            continue
        if present:
            mobile_centers.append(mobile_atoms.mean(axis=0))
            target_centers.append(target_atoms.mean(axis=0))
        else:
            mobile_centers.append(np.full(3, np.nan))
            target_centers.append(np.full(3, np.nan))
        valid.append(present)
    if not any(valid):
        metric = "LDDT" if retain_missing else "RMSD"
        raise ValueError(f"No valid atoms found for {metric} computation")
    return (
        torch.from_numpy(np.stack(mobile_centers)).unsqueeze(0),
        torch.from_numpy(np.stack(target_centers)).unsqueeze(0),
        torch.as_tensor(valid, dtype=torch.bool).unsqueeze(0),
    )


@dataclass(frozen=True)
class MolecularComplex:
    """A token sequence backed by one contiguous atom table.

    P stores atom coordinates with shape (n_atoms, 3).  Token span ``i`` is
    ``P[token_to_atoms[i, 0]:token_to_atoms[i, 1]]``.
    """

    id: str
    sequence: list[str]
    atom_positions: np.ndarray  # P has shape (n_atoms, 3).
    atom_elements: np.ndarray  # E has shape (n_atoms,).
    token_to_atoms: np.ndarray  # I has shape (n_tokens, 2).
    chain_id: np.ndarray  # C has shape (n_tokens,).
    plddt: np.ndarray  # S has shape (n_tokens,).
    metadata: MolecularComplexMetadata
    atom_names: np.ndarray | None = None  # N has shape (n_atoms,) when present.
    atom_hetero: np.ndarray | None = None  # M has shape (n_atoms,) when present.

    def __post_init__(self) -> None:
        _assert_table_lengths(self)

    def __len__(self) -> int:
        return len(self.sequence)

    def __getitem__(self, idx: int) -> Molecule:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Token index {idx} out of range for {len(self)} tokens")
        start, stop = self.token_to_atoms[idx]
        return Molecule(
            token=self.sequence[idx],
            token_idx=idx,
            atom_positions=self.atom_positions[start:stop],
            atom_elements=self.atom_elements[start:stop],
            atom_names=None if self.atom_names is None else self.atom_names[start:stop],
            atom_hetero=(None if self.atom_hetero is None else self.atom_hetero[start:stop]),
            residue_type=0,
            molecule_type=0,
            confidence=self.plddt[idx],
        )

    @property
    def atom_coordinates(self) -> np.ndarray:
        """Return P, the flat atom-coordinate table with shape (n_atoms, 3)."""
        return self.atom_positions

    @classmethod
    def from_protein_complex(cls, pc: ProteinComplex) -> MolecularComplex:
        positions, elements, names, hetero, spans = _flat_protein_atoms(pc)
        residue_positions = [index for index, value in enumerate(pc.sequence) if value != "|"]
        metadata = MolecularComplexMetadata(
            entity_lookup={key: str(value) for key, value in pc.metadata.entity_lookup.items()},
            chain_lookup=pc.metadata.chain_lookup,
            assembly_composition=pc.metadata.assembly_composition,
        )
        return cls(
            id=pc.id,
            sequence=[
                residue_constants.restype_1to3.get(pc.sequence[index], "UNK")
                for index in residue_positions
            ],
            atom_positions=positions,
            atom_elements=elements,
            token_to_atoms=spans,
            chain_id=np.asarray(pc.chain_id[residue_positions], dtype=np.int64),
            plddt=np.asarray(pc.confidence[residue_positions], dtype=np.float32),
            metadata=metadata,
            atom_names=names,
            atom_hetero=hetero,
        )

    def to_protein_complex(self) -> ProteinComplex:
        protein_indices, sequence, chain_ids, confidences = _protein_sequence_and_indices(self)
        compact_positions, compact_mask = _atom37_from_flat(self, protein_indices)
        arrays = _expand_protein_rows(
            sequence, chain_ids, confidences, compact_positions, compact_mask
        )
        unique_chains = np.unique(chain_ids)
        metadata = ProteinComplexMetadata(
            entity_lookup={int(chain): int(chain) for chain in unique_chains},
            chain_lookup={
                int(chain): self.metadata.chain_lookup.get(int(chain), chr(65 + int(chain)))
                for chain in unique_chains
            },
            assembly_composition=self.metadata.assembly_composition,
        )
        return ProteinComplex(
            id=self.id,
            sequence=sequence,
            entity_id=arrays["entity_id"],
            chain_id=arrays["chain_id"],
            sym_id=arrays["sym_id"],
            residue_index=arrays["residue_index"],
            insertion_code=arrays["insertion_code"],
            atom37_positions=arrays["atom37_positions"],
            atom37_mask=arrays["atom37_mask"],
            confidence=arrays["confidence"],
            metadata=metadata,
        )

    @classmethod
    def from_mmcif(cls, inp: str, id: str | None = None) -> MolecularComplex:
        cif_file = _read_cif(inp)
        structure = _read_structure(cif_file)
        if TYPE_CHECKING:
            structure: Any = structure
        labels = _label_asym_ids(cif_file, len(structure))
        grouped = _group_structure_atoms(structure, labels)
        (
            tokens,
            positions,
            elements,
            names,
            hetero,
            spans,
            confidences,
            token_chains,
            chain_numbers,
        ) = _flatten_structure_groups(grouped)
        n_tokens = len(tokens)
        if positions:
            position_array = np.asarray(positions, dtype=np.float32)
            element_array = np.asarray(elements, dtype=object)
            name_array = np.asarray(names, dtype=object)
            hetero_array = np.asarray(hetero, dtype=bool)
            span_array = np.asarray(spans, dtype=np.int32)
            chain_array = np.asarray(token_chains, dtype=np.int64)
        else:
            position_array = np.zeros((0, 3), dtype=np.float32)
            element_array = np.zeros(0, dtype=object)
            name_array = np.zeros(0, dtype=object)
            hetero_array = np.zeros(0, dtype=bool)
            span_array = np.zeros((n_tokens, 2), dtype=np.int32)
            chain_array = (
                np.asarray(token_chains, dtype=np.int64)
                if token_chains
                else np.zeros(n_tokens, dtype=np.int64)
            )
        complex_id = id or (Path(inp).stem if os.path.exists(inp) else "complex_from_string")
        return cls(
            id=complex_id,
            sequence=tokens,
            atom_positions=position_array,
            atom_elements=element_array,
            token_to_atoms=span_array,
            chain_id=chain_array,
            plddt=np.asarray(confidences, dtype=np.float32),
            metadata=MolecularComplexMetadata(
                entity_lookup=_entity_metadata(cif_file),
                chain_lookup={number: chain for chain, number in chain_numbers.items()},
                assembly_composition=None,
            ),
            atom_names=name_array,
            atom_hetero=hetero_array,
        )

    def _get_entity_mapping(
        self,
    ) -> tuple[dict[str, list[str]], dict[str, int], dict[int, tuple[str, ...]]]:
        return _chain_entity_maps(self)

    def _add_entity_information(
        self, cif_file: CIFFile, entity_sequences: dict[int, tuple[str, ...]]
    ) -> None:
        _add_entity_categories(cif_file, self, entity_sequences)

    def to_mmcif(self) -> str:
        _, chain_entities, entity_sequences = _chain_entity_maps(self)
        atom_array = _as_atom_array(self, chain_entities)
        cif_file = CIFFile()
        set_structure(cif_file, atom_array, data_block=self.id)
        _repair_label_entity_ids(cif_file, chain_entities)
        _add_entity_categories(cif_file, self, entity_sequences)
        round_mmcif_columns(cif_file)
        output = io.StringIO()
        cif_file.write(output)
        return output.getvalue()

    def dockq(self, native: MolecularComplex) -> Any:
        try:
            mobile = self.to_protein_complex().normalize_chain_ids_for_pdb()
            target = native.to_protein_complex().normalize_chain_ids_for_pdb()
        except ValueError as error:
            raise ValueError(
                f"Cannot convert MolecularComplex to ProteinComplex for DockQ: {error}"
            ) from None
        try:
            return mobile.dockq(target)
        except Exception:
            return self._compute_dockq_manual(native)

    def _compute_dockq_manual(self, native: MolecularComplex) -> Any:
        try:
            mobile = self.to_protein_complex().normalize_chain_ids_for_pdb()
            target = native.to_protein_complex().normalize_chain_ids_for_pdb()
        except ValueError as error:
            raise ValueError(
                f"Cannot convert MolecularComplex to ProteinComplex for DockQ: {error}"
            ) from None
        with TemporaryDirectory() as directory:
            mobile_path = Path(directory) / "self.pdb"
            target_path = Path(directory) / "native.pdb"
            mobile.to_pdb(mobile_path)
            target.to_pdb(target_path)
            try:
                raw_output = check_output(["DockQ", str(mobile_path), str(target_path)])
                output = raw_output.decode()
                score: float | None = None
                for line in output.split("\n"):
                    if "Total DockQ" in line:
                        match = re.search(r"Total DockQ.*: ([\d.]+)", line)
                        if match:
                            score = float(match.group(1))
                            break
                if score is None:
                    for line in output.split("\n"):
                        if line.startswith("DockQ") and ":" in line:
                            try:
                                score = float(line.split(":")[1].strip())
                                break
                            except (ValueError, IndexError):
                                continue
                if score is None:
                    raise ValueError("Could not parse DockQ score from output")
                return {"total_dockq": score, "raw_output": output, "aligned": self}
            except FileNotFoundError:
                raise RuntimeError(
                    "DockQ is not installed. Please install DockQ to use this method."
                ) from None
            except Exception as error:
                raise RuntimeError(f"DockQ computation failed: {error}") from error

    def rmsd(self, target: MolecularComplex, **kwargs: Any) -> float:
        mobile, reference, mask = _centroid_tensors(self, target, retain_missing=False)
        value = compute_rmsd(
            mobile=mobile,
            target=reference,
            atom_exists_mask=mask,
            reduction="batch",
            **kwargs,
        )
        return float(value)

    def lddt_ca(self, target: MolecularComplex, **kwargs: Any) -> float:
        mobile, reference, mask = _centroid_tensors(self, target, retain_missing=True)
        value = compute_lddt(
            all_atom_pred_pos=mobile,
            all_atom_positions=reference,
            all_atom_mask=mask,
            per_residue=False,
            **kwargs,
        )
        return float(value)

    def state_dict(self) -> dict[str, Any]:
        state = dict(vars(self))
        for key, value in tuple(state.items()):
            if isinstance(value, MolecularComplexMetadata):
                state[key] = asdict(value)
            elif isinstance(value, np.ndarray):
                if value.dtype == np.int64:
                    value = value.astype(np.int32)
                elif value.dtype in (np.dtype(np.float64), np.dtype(np.float32)):
                    value = value.astype(np.float16)
                state[key] = value.tolist()
        return state

    def to_blob(self) -> bytes:
        return brotli.compress(msgpack.dumps(self.state_dict()), quality=5)

    @classmethod
    def from_state_dict(cls, dct: dict[str, Any]) -> MolecularComplex:
        for key, value in tuple(dct.items()):
            if isinstance(value, list) and key in _SERIALIZED_ARRAYS:
                dct[key] = np.asarray(value)
        for key, value in tuple(dct.items()):
            if not isinstance(value, np.ndarray):
                continue
            if key in {"atom_positions", "plddt"}:
                dct[key] = value.astype(np.float32)
            elif key == "token_to_atoms":
                dct[key] = value.astype(np.int32)
            elif key == "chain_id":
                dct[key] = value.astype(np.int64)
        dct["metadata"] = MolecularComplexMetadata(**dct["metadata"])
        if "chain_id" not in dct:
            dct["chain_id"] = np.zeros(len(dct["sequence"]), dtype=np.int64)
        return cls(**dct)

    @classmethod
    def from_blob(cls, input: Path | str | io.BytesIO | bytes) -> MolecularComplex:
        if isinstance(input, (Path, str)):
            payload = Path(input).read_bytes()
        elif isinstance(input, io.BytesIO):
            payload = input.getvalue()
        else:
            payload = input
        state = msgpack.loads(brotli.decompress(payload), strict_map_key=False)
        return cls.from_state_dict(state)
