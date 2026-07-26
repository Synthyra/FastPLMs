"""Convert ESMFold2 coordinate tensors into molecular-complex records."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import groupby
from typing import Any

import numpy as np
import torch

from .esmfold2_constants import ELEMENT_NUMBER_TO_SYMBOL, MOL_TYPE_NONPOLYMER
from .esmfold2_molecular_complex import MolecularComplex, MolecularComplexMetadata


def get_element_symbol(atomic_number: int) -> str:
    """Map a training-time atomic number to a chemical symbol."""

    return ELEMENT_NUMBER_TO_SYMBOL.get(atomic_number, "X")


def _decode_atom_name(encoded_name: Any) -> str:
    values = encoded_name.tolist() if hasattr(encoded_name, "tolist") else encoded_name
    return "".join(chr(int(value) + 32) for value in values if int(value)).strip()


@dataclass
class _ComplexRecords:
    sequence: list[str] = field(default_factory=list)
    chain_ids: list[int] = field(default_factory=list)
    token_to_atoms: list[list[int]] = field(default_factory=list)
    confidence: list[float] = field(default_factory=list)
    positions: list[list[float]] = field(default_factory=list)
    elements: list[str] = field(default_factory=list)
    atom_names: list[str] = field(default_factory=list)
    atom_hetero: list[bool] = field(default_factory=list)
    chain_lookup: dict[int, str] = field(default_factory=dict)
    entity_lookup: dict[int, str] = field(default_factory=dict)

    def add_token(
        self,
        *,
        residue_name: str,
        asym_id: int,
        plddt: float,
        atoms: Iterable[tuple[list[float], str, str]],
        hetero: bool,
    ) -> None:
        atom_start = len(self.positions)
        for position, element, atom_name in atoms:
            self.positions.append(position)
            self.elements.append(element)
            self.atom_names.append(atom_name)
            self.atom_hetero.append(hetero)
        self.sequence.append(residue_name)
        self.chain_ids.append(asym_id)
        self.confidence.append(plddt)
        self.token_to_atoms.append([atom_start, len(self.positions)])

    def build(self, complex_id: str) -> MolecularComplex:
        return MolecularComplex(
            id=complex_id,
            sequence=self.sequence,
            atom_positions=np.asarray(self.positions, dtype=np.float32).reshape(-1, 3),
            atom_elements=np.asarray(self.elements, dtype=object),
            token_to_atoms=np.asarray(self.token_to_atoms, dtype=np.int32).reshape(-1, 2),
            chain_id=np.asarray(self.chain_ids, dtype=np.int64),
            plddt=np.asarray(self.confidence, dtype=np.float32),
            atom_names=np.asarray(self.atom_names, dtype=object),
            atom_hetero=np.asarray(self.atom_hetero, dtype=bool),
            metadata=MolecularComplexMetadata(
                entity_lookup=self.entity_lookup,
                chain_lookup=self.chain_lookup,
                assembly_composition=None,
            ),
        )


def build_molecular_complex_from_features(
    coords: torch.Tensor,
    plddt: torch.Tensor,
    atom_mask: torch.Tensor,
    ref_element: torch.Tensor,
    ref_atom_name_chars: torch.Tensor,
    chain_infos: list[Any],
    complex_id: str,
) -> MolecularComplex:
    """Decode model features into one complex without intermediate structure files.

    Protein, DNA, and RNA tokens are grouped by residue index. Ligand atom
    tokens are collapsed into one non-polymer residue per chain.
    """

    M = atom_mask.bool().cpu().numpy()
    X = coords.float().cpu().numpy()
    atom_names = ref_atom_name_chars.cpu().numpy()
    elements = ref_element.cpu().numpy()
    confidence = plddt.float().cpu().numpy()
    records = _ComplexRecords()

    def decode_atoms(tokens: Iterable[Any]):
        for token in tokens:
            for atom_index in range(token.atom_start, token.atom_start + token.atom_count):
                if M[atom_index]:
                    yield (
                        X[atom_index].tolist(),
                        get_element_symbol(int(elements[atom_index])),
                        _decode_atom_name(atom_names[atom_index]),
                    )

    for chain in chain_infos:
        is_nonpolymer = chain.mol_type == MOL_TYPE_NONPOLYMER
        records.chain_lookup[chain.asym_id] = chain.chain_id
        records.entity_lookup[chain.entity_id] = "non-polymer" if is_nonpolymer else "polymer"

        if is_nonpolymer:
            mean_confidence = (
                float(np.mean([confidence[token.token_index] for token in chain.tokens]))
                if chain.tokens
                else 0.0
            )
            records.add_token(
                residue_name=chain.tokens[0].residue_name if chain.tokens else "LIG",
                asym_id=chain.asym_id,
                plddt=mean_confidence,
                atoms=decode_atoms(chain.tokens),
                hetero=True,
            )
            continue

        residue_groups = groupby(chain.tokens, key=lambda token: token.residue_index)
        for _residue_index, group in residue_groups:
            residue_tokens = list(group)
            records.add_token(
                residue_name=residue_tokens[0].residue_name,
                asym_id=chain.asym_id,
                plddt=float(np.mean([confidence[token.token_index] for token in residue_tokens])),
                atoms=decode_atoms(residue_tokens),
                hetero=False,
            )

    return records.build(complex_id)


def build_molecular_complex(
    structure: Any,
    coords: torch.Tensor,
    plddt: torch.Tensor,
    complex_id: str,
) -> MolecularComplex:
    """Decode coordinates using the atom and residue arrays of a prepared structure."""

    records = _ComplexRecords()
    coordinate_index = 0
    confidence_index = 0

    for chain in structure.chains:
        asym_id = int(chain["asym_id"])
        mol_type = int(chain["mol_type"])
        is_nonpolymer = mol_type == MOL_TYPE_NONPOLYMER
        records.chain_lookup[asym_id] = str(chain["name"])
        records.entity_lookup[int(chain["entity_id"])] = (
            "non-polymer" if is_nonpolymer else "polymer"
        )

        residue_start = int(chain["res_idx"])
        residue_stop = residue_start + int(chain["res_num"])
        for residue in structure.residues[residue_start:residue_stop]:
            atom_start = int(residue["atom_idx"])
            atom_stop = atom_start + int(residue["atom_num"])
            decoded_atoms: list[tuple[list[float], str, str]] = []
            for atom in structure.atoms[atom_start:atom_stop]:
                if not atom["is_present"]:
                    continue
                decoded_atoms.append(
                    (
                        coords[coordinate_index].tolist(),
                        get_element_symbol(int(atom["element"].item())),
                        _decode_atom_name(atom["name"]),
                    )
                )
                coordinate_index += 1

            records.add_token(
                residue_name=str(residue["name"]),
                asym_id=asym_id,
                plddt=float(plddt[confidence_index].item()),
                atoms=decoded_atoms,
                hetero=is_nonpolymer,
            )
            confidence_index += 1

    return records.build(complex_id)


__all__ = [
    "build_molecular_complex",
    "build_molecular_complex_from_features",
    "get_element_symbol",
]
