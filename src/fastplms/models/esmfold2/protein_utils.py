"""Protein-only ESMFold2 featurization without the Biohub runtime package.

Input is one amino-acid sequence. The transformation expands each residue into
the checkpoint atom schema, pads atoms to a multiple of 32, and emits batched
token, atom, and single-sequence MSA tensors. Reference coordinates are loaded
lazily from a provenance-bearing declarative package asset.
"""

from __future__ import annotations

import json
import torch

from functools import cache
from importlib.resources import files
from typing import Any
from torch import Tensor

from .esmfold2_constants import (
    CHARGED_ATOMS,
    ELEMENT_TO_ATOMIC_NUM,
    ESM_PROTEIN_VOCAB,
    MOL_TYPE_PROTEIN,
    PROTEIN_1TO3,
    PROTEIN_HEAVY_ATOMS,
    PROTEIN_RESIDUE_TO_RES_TYPE,
    PROTEIN_UNK_RES_TYPE,
)


_GEOMETRY_ASSET = "protein_reference_geometry.json"
_GEOMETRY_SCHEMA = "fastplms.esmfold2.reference_geometry.v1"


@cache
def _reference_geometry() -> dict[str, dict[str, tuple[float, float, float]]]:
    resource = files(__package__).joinpath(_GEOMETRY_ASSET)
    with resource.open(mode="r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if (
        payload.get("schema") != _GEOMETRY_SCHEMA
        or payload.get("dtype") != "float32"
        or payload.get("provenance", {}).get("manifest_family") != "esmfold2"
    ):
        raise RuntimeError("The ESMFold2 reference-geometry asset has invalid provenance.")

    raw_residues = payload.get("residues")
    if not isinstance(raw_residues, dict):
        raise RuntimeError("The ESMFold2 reference-geometry asset has no residue table.")
    geometry: dict[str, dict[str, tuple[float, float, float]]] = {}
    for residue, atom_positions in raw_residues.items():
        if not isinstance(residue, str) or not isinstance(atom_positions, dict):
            raise RuntimeError("The ESMFold2 reference-geometry residue table is malformed.")
        geometry[residue] = {}
        for atom_name, position in atom_positions.items():
            if (
                not isinstance(atom_name, str)
                or not isinstance(position, list)
                or len(position) != 3
            ):
                raise RuntimeError("The ESMFold2 reference-geometry atom table is malformed.")
            geometry[residue][atom_name] = tuple(float(value) for value in position)

    expected_residues = set(PROTEIN_HEAVY_ATOMS) - {"MSE"}
    if set(geometry) != expected_residues:
        raise RuntimeError("The ESMFold2 reference-geometry residue set is incomplete.")
    for residue, atom_names in PROTEIN_HEAVY_ATOMS.items():
        if residue == "MSE":
            continue
        if set(geometry[residue]) != set(atom_names):
            raise RuntimeError(f"Reference geometry differs from the atom schema for {residue}.")
    return geometry


def _encode_atom_name(atom_name: str) -> tuple[int, int, int, int]:
    padded = atom_name.ljust(4)[:4]
    return tuple(ord(character) - 32 if character != " " else 0 for character in padded)


def _padded_atom_count(actual_count: int) -> int:
    return max(32, ((actual_count + 31) // 32) * 32)


def _residue_records(sequence: str) -> tuple[list[dict[str, Any]], list[int], list[int], list[int]]:
    geometry = _reference_geometry()
    atoms: list[dict[str, Any]] = []
    residue_types: list[int] = []
    input_ids: list[int] = []
    representative_atoms: list[int] = []

    for token_index, residue_letter in enumerate(sequence):
        residue_name = PROTEIN_1TO3.get(residue_letter, "UNK")
        atom_names = PROTEIN_HEAVY_ATOMS[residue_name]
        atom_start = len(atoms)
        for atom_name in atom_names:
            atoms.append(
                {
                    "token_index": token_index,
                    "name": atom_name,
                    "element": atom_name[0],
                    "charge": CHARGED_ATOMS.get((residue_name, atom_name), 0),
                    "position": geometry[residue_name][atom_name],
                }
            )

        representative_name = "CB" if "CB" in atom_names else "CA"
        representative_atoms.append(atom_start + atom_names.index(representative_name))
        residue_types.append(PROTEIN_RESIDUE_TO_RES_TYPE.get(residue_name, PROTEIN_UNK_RES_TYPE))
        input_ids.append(ESM_PROTEIN_VOCAB.get(residue_letter, ESM_PROTEIN_VOCAB["X"]))

    return atoms, residue_types, input_ids, representative_atoms


def prepare_protein_features(sequence: str) -> dict[str, Tensor]:
    """Build the protein-only feature mapping consumed by ESMFold2.

    Every tensor includes a leading batch dimension. Biological tokens have
    length ``l``; atom tensors have length ``n_atoms``, where ``n_atoms`` is the
    smallest multiple of 32 covering all heavy atoms.
    """

    if not sequence:
        raise ValueError("sequence must be non-empty")

    atoms, residue_types, input_ids, representative_atoms = _residue_records(sequence)
    sequence_length = len(sequence)  # l
    n_atoms = _padded_atom_count(len(atoms))

    ref_pos = torch.zeros((n_atoms, 3), dtype=torch.float32)  # (n_atoms, 3)
    ref_element = torch.zeros(n_atoms, dtype=torch.int64)  # (n_atoms,)
    ref_charge = torch.zeros(n_atoms, dtype=torch.int8)  # (n_atoms,)
    ref_atom_name_chars = torch.zeros((n_atoms, 4), dtype=torch.int64)  # (n_atoms, 4)
    ref_space_uid = torch.zeros(n_atoms, dtype=torch.int64)  # (n_atoms,)
    atom_attention_mask = torch.zeros(n_atoms, dtype=torch.bool)  # (n_atoms,)
    atom_to_token = torch.zeros(n_atoms, dtype=torch.int64)  # (n_atoms,)

    for atom_index, atom in enumerate(atoms):
        token_index = atom["token_index"]
        ref_pos[atom_index] = torch.tensor(atom["position"], dtype=torch.float32)  # (3,)
        ref_element[atom_index] = ELEMENT_TO_ATOMIC_NUM[atom["element"]]  # ()
        ref_charge[atom_index] = atom["charge"]  # ()
        ref_atom_name_chars[atom_index] = torch.tensor(
            _encode_atom_name(atom["name"]), dtype=torch.int64
        )  # (4,)
        ref_space_uid[atom_index] = token_index  # ()
        atom_attention_mask[atom_index] = True  # ()
        atom_to_token[atom_index] = token_index  # ()

    residue_type_tensor = torch.tensor(residue_types, dtype=torch.int64)  # (l,)
    msa = residue_type_tensor.unsqueeze(0)  # (1, l)
    features = {
        "token_index": torch.arange(sequence_length, dtype=torch.int64),  # (l,)
        "residue_index": torch.arange(sequence_length, dtype=torch.int64),  # (l,)
        "asym_id": torch.zeros(sequence_length, dtype=torch.int64),  # (l,)
        "sym_id": torch.zeros(sequence_length, dtype=torch.int64),  # (l,)
        "entity_id": torch.ones(sequence_length, dtype=torch.int64),  # (l,)
        "mol_type": torch.full((sequence_length,), MOL_TYPE_PROTEIN, dtype=torch.int64),  # (l,)
        "res_type": residue_type_tensor,  # (l,)
        "input_ids": torch.tensor(input_ids, dtype=torch.int64),  # (l,)
        "token_bonds": torch.zeros((sequence_length, sequence_length, 1), dtype=torch.float32),  # (l, l, 1)
        "token_attention_mask": torch.ones(sequence_length, dtype=torch.bool),  # (l,)
        "ref_pos": ref_pos,  # (n_atoms, 3)
        "ref_element": ref_element,  # (n_atoms,)
        "ref_charge": ref_charge,  # (n_atoms,)
        "ref_atom_name_chars": ref_atom_name_chars,  # (n_atoms, 4)
        "ref_space_uid": ref_space_uid,  # (n_atoms,)
        "atom_attention_mask": atom_attention_mask,  # (n_atoms,)
        "atom_to_token": atom_to_token,  # (n_atoms,)
        "distogram_atom_idx": torch.tensor(representative_atoms, dtype=torch.int64),  # (l,)
        "msa": msa,  # (1, l)
        "msa_attention_mask": torch.ones_like(msa, dtype=torch.bool),  # (1, l)
        "has_deletion": torch.zeros_like(msa, dtype=torch.bool),  # (1, l)
        "deletion_value": torch.zeros_like(msa, dtype=torch.float32),  # (1, l)
        "deletion_mean": torch.zeros(sequence_length, dtype=torch.float32),  # (l,)
    }
    return {name: tensor.unsqueeze(0) for name, tensor in features.items()}  # each shape -> (1, *shape)


__all__ = ["prepare_protein_features"]
