"""Deterministic candidate-side contracts for Boltz2 feature preparation.

Live parity against the pinned Boltz package runs in the native reference
service. These tests keep inexpensive shape, dtype, padding, and repeatability
failures close to the FastPLMs implementation.
"""

from __future__ import annotations

import pytest
import torch

from fastplms.models.boltz.minimal_featurizer import build_boltz2_features
from fastplms.models.boltz.minimal_structures import ProteinStructureTemplate

pytestmark = pytest.mark.structure

SEQUENCE = "ACDEFGHIKLMNPQRSTVWY"
REQUIRED_FEATURES = {
    "atom_pad_mask",
    "atom_to_token",
    "coords",
    "disto_center",
    "frames_idx",
    "msa",
    "msa_mask",
    "ref_pos",
    "res_type",
    "residue_index",
    "token_index",
    "token_pad_mask",
    "token_to_center_atom",
    "token_to_rep_atom",
}


def _build_with_seed(
    seed: int,
) -> tuple[dict[str, torch.Tensor], ProteinStructureTemplate]:
    torch.manual_seed(seed)
    return build_boltz2_features(SEQUENCE)


def test_boltz2_feature_preparation_is_seed_reproducible() -> None:
    first, first_template = _build_with_seed(17)
    second, second_template = _build_with_seed(17)

    assert first.keys() == second.keys()
    for name in first:
        assert torch.equal(first[name], second[name]), name
    assert first_template == second_template


def test_boltz2_single_chain_feature_contract() -> None:
    features, template = _build_with_seed(23)
    n_residues = len(SEQUENCE)

    assert features.keys() >= REQUIRED_FEATURES
    assert template.sequence == SEQUENCE
    assert template.num_residues == n_residues
    assert features["token_index"].shape == (1, n_residues)
    assert features["residue_index"].shape == (1, n_residues)
    assert features["token_pad_mask"].shape == (1, n_residues)
    assert features["res_type"].shape[:2] == (1, n_residues)
    assert features["msa"].shape[:3] == (1, 1, n_residues)

    n_atoms_padded = features["atom_pad_mask"].shape[-1]
    assert n_atoms_padded >= template.num_atoms
    assert n_atoms_padded % 32 == 0
    assert features["ref_pos"].shape == (1, n_atoms_padded, 3)
    assert features["coords"].shape == (1, 1, n_atoms_padded, 3)
    assert features["atom_to_token"].shape == (1, n_atoms_padded, n_residues)
    assert torch.equal(
        features["atom_pad_mask"][0, : template.num_atoms],
        torch.ones(template.num_atoms),
    )
    assert not features["atom_pad_mask"][0, template.num_atoms :].bool().any()


def test_boltz2_feature_dtypes_are_explicit() -> None:
    features, _ = _build_with_seed(29)

    integer_features = (
        "atom_backbone_feat",
        "atom_to_token",
        "contact_conditioning",
        "frames_idx",
        "msa",
        "ref_atom_name_chars",
        "ref_chirality",
        "ref_element",
        "res_type",
        "residue_index",
        "r_set_to_rep_atom",
        "token_index",
        "token_to_center_atom",
        "token_to_rep_atom",
    )
    for name in integer_features:
        assert features[name].dtype == torch.long
    for name in ("atom_resolved_mask", "frame_resolved_mask", "has_deletion"):
        assert features[name].dtype == torch.bool
    for name in ("ref_pos", "coords", "token_pad_mask"):
        assert features[name].dtype == torch.float32


def test_boltz2_sequence_only_observation_features_are_empty() -> None:
    features, _ = _build_with_seed(31)

    assert torch.count_nonzero(features["coords"]) == 0
    assert torch.count_nonzero(features["disto_center"]) == 0
    assert torch.count_nonzero(features["disto_coords_ensemble"]) == 0
    assert torch.all(features["disto_target"][..., 0] == 1)
    assert torch.count_nonzero(features["disto_target"][..., 1:]) == 0


def test_boltz2_canonical_charge_and_chirality_tables() -> None:
    features, template = _build_with_seed(37)

    def atom_index(residue_name: str, atom_name: str) -> int:
        for index, (name, residue_index) in enumerate(
            zip(template.atom_names, template.atom_residue_index, strict=True)
        ):
            if name == atom_name and template.residue_names[residue_index] == residue_name:
                return index
        raise AssertionError(f"Missing {residue_name} {atom_name} atom.")

    for residue_name, atom_name in (("HIS", "ND1"), ("LYS", "NZ")):
        index = atom_index(residue_name, atom_name)
        assert features["ref_charge"][0, index].item() == 1.0
    for residue_name, atom_name in (("ALA", "CA"), ("ILE", "CB"), ("THR", "CB")):
        index = atom_index(residue_name, atom_name)
        assert features["ref_chirality"][0, index].item() == 2
