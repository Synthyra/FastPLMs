"""Deterministic output-contract tests for ESMFold2 leaf utilities."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fastplms.models.esmfold2.esmfold2_constants import (
    MOL_TYPE_NONPOLYMER,
    MOL_TYPE_PROTEIN,
)
from fastplms.models.esmfold2.esmfold2_output import (
    build_molecular_complex_from_features,
)
from fastplms.models.esmfold2.modeling_esmfold2_common import LanguageModelShim

pytestmark = pytest.mark.structure


def _encoded_name(name: str) -> list[int]:
    return [ord(character) - 32 if character != " " else 0 for character in name.ljust(4)]


def _token(
    token_index: int,
    residue_index: int,
    residue_name: str,
    atom_start: int,
    atom_count: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        token_index=token_index,
        residue_index=residue_index,
        residue_name=residue_name,
        atom_start=atom_start,
        atom_count=atom_count,
    )


def test_feature_output_groups_modified_residues_and_ligand_atoms() -> None:
    polymer_tokens = [
        _token(0, 0, "ALA", 0, 2),
        _token(1, 1, "GLY", 2, 1),
        _token(2, 1, "GLY", 3, 1),
    ]
    ligand_tokens = [
        _token(3, 0, "LIG", 4, 1),
        _token(4, 0, "LIG", 5, 1),
    ]
    chain_infos = [
        SimpleNamespace(
            asym_id=0,
            entity_id=0,
            chain_id="A",
            mol_type=MOL_TYPE_PROTEIN,
            tokens=polymer_tokens,
        ),
        SimpleNamespace(
            asym_id=1,
            entity_id=1,
            chain_id="B",
            mol_type=MOL_TYPE_NONPOLYMER,
            tokens=ligand_tokens,
        ),
    ]
    X = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    atom_names = torch.tensor([_encoded_name(name) for name in ("N", "CA", "C", "O", "C1", "N1")])

    complex_record = build_molecular_complex_from_features(
        coords=X,
        plddt=torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9]),
        atom_mask=torch.ones(6, dtype=torch.bool),
        ref_element=torch.tensor([7, 6, 6, 8, 6, 7]),
        ref_atom_name_chars=atom_names,
        chain_infos=chain_infos,
        complex_id="fixture",
    )

    assert complex_record.id == "fixture"
    assert complex_record.sequence == ["ALA", "GLY", "LIG"]
    np.testing.assert_array_equal(
        complex_record.token_to_atoms,
        np.array([[0, 2], [2, 4], [4, 6]], dtype=np.int32),
    )
    np.testing.assert_array_equal(complex_record.chain_id, np.array([0, 0, 1]))
    np.testing.assert_allclose(complex_record.plddt, np.array([0.1, 0.4, 0.8]))
    np.testing.assert_array_equal(
        complex_record.atom_hetero,
        np.array([False, False, False, False, True, True]),
    )
    np.testing.assert_array_equal(complex_record.atom_positions, X.numpy())
    assert complex_record.atom_names.tolist() == ["N", "CA", "C", "O", "C1", "N1"]
    assert complex_record.metadata.chain_lookup == {0: "A", 1: "B"}
    assert complex_record.metadata.entity_lookup == {0: "polymer", 1: "non-polymer"}


@pytest.mark.gpu
def test_learned_sequence_projection_matches_explicit_cuda_operation() -> None:
    assert torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda")
    shim = LanguageModelShim(d_z=7, d_model=11, num_layers=3).to(
        device=device, dtype=torch.bfloat16
    )
    H = torch.randn((2, 17, 4, 11), device=device, dtype=torch.bfloat16)
    M = torch.tensor([[True] * 13 + [False] * 4, [True] * 17], device=device, dtype=torch.bool)

    projected_states = shim.base_z_linear(H)
    expected = shim.base_z_combine.softmax(dim=0) @ projected_states
    expected = expected * M.unsqueeze(-1)
    Z = shim.project_sequence(H, M)

    assert torch.equal(Z, expected)
    assert Z.shape == (2, 17, 7)
    assert Z.dtype == torch.bfloat16
