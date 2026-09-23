"""Unit tests for immutable ESMFold2 confidence caches."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from types import SimpleNamespace
from unittest.mock import Mock

from torch import nn

from fastplms.models.esmfold2.configuration_esmfold2 import ESMFold2Config
from fastplms.registry import get_model_spec
from tools.confidence import cache
from tools.confidence.cache import (
    _kabsch_aligned,
    _chain_assignment,
    _aligned_true_coordinates,
    _resolve_ambiguous_atoms,
    _single_sample_coordinates,
    confidence_inputs,
    load_cache,
)
from tools.confidence.data import _atom14_names


def _cache_tensors() -> dict[str, torch.Tensor]:
    tensors = {
        "s_inputs": torch.zeros(1, 2, 4),
        "z": torch.zeros(1, 2, 2, 3),
        "x_pred": torch.zeros(1, 4, 3),
        "true_coords": torch.zeros(4, 3),
        "resolved_mask": torch.ones(4, dtype=torch.bool),
        "token_index": torch.arange(2).view(1, 2),
        "residue_index": torch.arange(2).view(1, 2),
        "asym_id": torch.zeros(1, 2, dtype=torch.long),
        "sym_id": torch.zeros(1, 2, dtype=torch.long),
        "entity_id": torch.ones(1, 2, dtype=torch.long),
        "mol_type": torch.zeros(1, 2, dtype=torch.long),
        "token_bonds": torch.zeros(1, 2, 2, 1),
        "token_attention_mask": torch.ones(1, 2, dtype=torch.bool),
        "ref_atom_name_chars": torch.zeros(1, 4, 4, dtype=torch.long),
        "atom_attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "atom_to_token": torch.tensor([[0, 0, 1, 1]]),
        "distogram_atom_idx": torch.tensor([[0, 2]]),
        "backbone_indices": torch.tensor([[0, 1, 2], [1, 2, 3]]),
    }
    return tensors


class _PositionalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def rel_pos(self, **values: torch.Tensor) -> torch.Tensor:
        length = values["residue_index"].shape[-1]
        return torch.zeros(1, length, length, 3)

    def token_bonds(self, values: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1, 2, 2, 3)


def test_confidence_inputs_reconstructs_derived_embeddings() -> None:
    inputs = confidence_inputs(_PositionalModel(), _cache_tensors())

    assert "relative_position_encoding" in inputs
    assert "token_bonds_encoding" in inputs
    assert inputs["relative_position_encoding"].shape == (1, 2, 2, 3)
    assert inputs["token_bonds_encoding"].shape == (1, 2, 2, 3)
    assert inputs["num_diffusion_samples"] == 1


def test_kabsch_alignment_handles_nontrivial_rigid_rotation() -> None:
    predicted = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 1.0]])  # (3, 3)
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # (3, 3)
    target = (predicted - predicted.mean(0)) @ rotation.T + torch.tensor([4.0, -2.0, 1.0])
    assert torch.allclose(_kabsch_aligned(predicted, target), predicted, atol=1e-5)


def test_load_cache_rejects_derived_embeddings(tmp_path) -> None:
    from safetensors.torch import save_file

    path = tmp_path / "cache.safetensors"
    save_file(
        {**_cache_tensors(), "relative_position_encoding": torch.zeros(1, 2, 2, 3)},
        str(path),
        metadata={"schema": "fastplms.confidence.cache.v2"},
    )

    try:
        load_cache(path)
    except ValueError as error:
        assert "derived" in str(error)
    else:
        raise AssertionError("derived embeddings must not be accepted")


def test_load_cache_round_trips_tensor_contract(tmp_path) -> None:
    from safetensors.torch import save_file

    path = tmp_path / "cache.safetensors"
    save_file(
        _cache_tensors(),
        str(path),
        metadata={"schema": "fastplms.confidence.cache.v2", "model_id": "esmfold2_300"},
    )

    tensors, metadata = load_cache(path)

    assert metadata["model_id"] == "esmfold2_300"
    assert tensors["s_inputs"].shape == (1, 2, 4)
    assert tensors["resolved_mask"].dtype == torch.bool


def test_ile_branch_atoms_are_not_swapped() -> None:
    true = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])  # (2, 3)
    predicted = true.flip(0)
    resolved = torch.ones(2, dtype=torch.bool)  # (2,)
    result = _resolve_ambiguous_atoms(
        predicted, true, resolved, torch.zeros(2, dtype=torch.long), ["CG1", "CG2"], {0: "ILE"}
    )
    assert torch.equal(result, true)


def test_homodimer_assignment_ignores_padded_atoms() -> None:
    chain_a = torch.tensor([[10.0, 0.0, 0.0], [10.0, 1.0, 0.0], [10.0, 0.0, 1.0]])  # (3, 3)
    chain_b = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # (3, 3)
    true = torch.cat(
        (
            chain_a,
            torch.tensor([[100.0, 100.0, 100.0]]),
            chain_b,
            torch.tensor([[-100.0, -100.0, -100.0]]),
        )
    )
    predicted = torch.cat(
        (
            chain_b,
            torch.tensor([[1000.0, 1000.0, 1000.0]]),
            chain_a,
            torch.tensor([[-1000.0, -1000.0, -1000.0]]),
        )
    )
    atoms = [torch.tensor([0, 1, 2, 3]), torch.tensor([4, 5, 6, 7])]
    mask = torch.tensor([True, True, True, False, True, True, True, False])  # (8,)
    assignment = _chain_assignment(
        predicted, true, atoms, ["AAA", "AAA"], ["CA"] * 8, mask, torch.arange(8)
    )
    assert assignment == [1, 0]


def test_phe_aromatic_pairs_swap_jointly() -> None:
    true = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [2.0, 1.0, 1.0],
            [-1.0, 2.0, 2.0],
        ]
    )  # (7, 3)
    predicted = true[[1, 0, 3, 2, 4, 5, 6]]
    resolved = torch.ones(7, dtype=torch.bool)  # (7,)
    result = _resolve_ambiguous_atoms(
        predicted,
        true,
        resolved,
        torch.tensor([0, 0, 0, 0, 1, 1, 1]),
        ["CD1", "CD2", "CE1", "CE2", "N", "CA", "C"],
        {0: "PHE", 1: "ALA"},
    )
    assert torch.equal(result, predicted)


def test_native_zero_based_residues_map_to_complete_structure_positions(tmp_path) -> None:
    sequence = "AC"
    names = _atom14_names(sequence)
    coordinates = np.full((2, 14, 3), np.nan, dtype=np.float32)  # (2, 14, 3)
    expected, atom_names, token_indices = [], [], []
    for residue in range(2):
        for atom, name in enumerate(names[residue]):
            if name:
                xyz = np.array([residue * 100 + atom, atom * 2, atom % 3], dtype=np.float32)
                coordinates[residue, atom] = xyz
                expected.append(xyz)
                atom_names.append(str(name))
                token_indices.append(residue)
    path = tmp_path / "structure.npz"
    np.savez(
        path,
        coordinates=coordinates,
        atom_names=names,
        chain_index=np.zeros(2, dtype=np.int32),
        residue_index=np.array([1, 2]),
    )
    features = {
        "atom_to_token": torch.tensor([token_indices]),
        "atom_attention_mask": torch.ones(1, len(expected), dtype=torch.bool),
        "ref_atom_name_chars": torch.tensor(
            [[ord(char) - 32 for char in name.ljust(4)] for name in atom_names]
        ).unsqueeze(0),
    }
    chains = [
        SimpleNamespace(
            tokens=[
                SimpleNamespace(token_index=i, residue_index=i, residue_name=name)
                for i, name in enumerate(("ALA", "CYS"))
            ]
        )
    ]
    expected_tensor = torch.tensor(np.asarray(expected))
    actual, mask = _aligned_true_coordinates(
        features, chains, {"chains": [{"id": "A", "sequence": sequence}]}, path, expected_tensor
    )
    assert mask.all()
    torch.testing.assert_close(actual, expected_tensor, rtol=0, atol=0)


def test_cache_target_aligns_on_cpu_after_gpu_model_features(tmp_path, monkeypatch) -> None:
    import hashlib

    from tools.confidence import cache as cache_module

    structure = tmp_path / "structure.npz"
    np.savez(
        structure,
        coordinates=np.full((1, 14, 3), np.nan, dtype=np.float32),
        atom_names=np.asarray([["N"] + [""] * 13]),
        chain_index=np.zeros(1, dtype=np.int32),
        residue_index=np.ones(1, dtype=np.int32),
    )
    digest = hashlib.file_digest(structure.open("rb"), "sha256").hexdigest()
    record = {
        "id": "cpu-transfer",
        "split": "train",
        "structure_path": "structure.npz",
        "structure_sha256": digest,
        "chains": [{"id": "A", "sequence": "A"}],
    }
    required = {
        "token_index": torch.zeros(1, 1, dtype=torch.long),
        "residue_index": torch.zeros(1, 1, dtype=torch.long),
        "asym_id": torch.zeros(1, 1, dtype=torch.long),
        "sym_id": torch.zeros(1, 1, dtype=torch.long),
        "entity_id": torch.ones(1, 1, dtype=torch.long),
        "mol_type": torch.zeros(1, 1, dtype=torch.long),
        "token_bonds": torch.zeros(1, 1, 1, 1),
        "token_attention_mask": torch.ones(1, 1, dtype=torch.bool),
        "ref_atom_name_chars": torch.tensor([[[46, 0, 0, 0]]]),
        "atom_attention_mask": torch.ones(1, 1, dtype=torch.bool),
        "atom_to_token": torch.zeros(1, 1, dtype=torch.long),
        "distogram_atom_idx": torch.zeros(1, 1, dtype=torch.long),
    }

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))

        def prepare_structure_input(self, value, seed):
            del value, seed
            return required, [
                SimpleNamespace(
                    tokens=[SimpleNamespace(token_index=0, residue_index=0, residue_name="ALA")]
                )
            ]

        def forward(self, **kwargs):
            del kwargs
            return SimpleNamespace(
                hidden_states=(torch.zeros(1, 1, 2), torch.zeros(1, 1, 1, 2)),
                sample_atom_coords=torch.zeros(1, 1, 1, 3),
            )

    seen = {}

    def aligned(features, *args):
        seen.update(features)
        return torch.zeros(1, 3), torch.ones(1, dtype=torch.bool)

    monkeypatch.setattr(cache_module, "_aligned_true_coordinates", aligned)
    output = tmp_path / "cache.safetensors"
    cache_module.cache_target(FakeModel(), record, tmp_path, output, 17)
    assert seen and all(value.device.type == "cpu" for value in seen.values())


def test_aligned_true_coordinates_forced_swapped_homodimer(tmp_path) -> None:
    coordinates = np.full((6, 14, 3), np.nan, dtype=np.float32)  # (6, 14, 3)
    chain_a = ((10, 0, 0), (10, 1, 0), (10, 0, 1))
    chain_b = ((0, 0, 0), (0, 1, 0), (0, 0, 1))
    for row, xyz in enumerate((*chain_a, *chain_b)):
        coordinates[row, 0] = xyz
    path = tmp_path / "dimer.npz"
    np.savez(
        path,
        coordinates=coordinates,
        atom_names=np.asarray([["CA"] + [""] * 13] * 6),
        chain_index=np.asarray([0, 0, 0, 1, 1, 1]),
        residue_index=np.asarray([1, 2, 3, 1, 2, 3]),
    )
    features = {
        "atom_to_token": torch.arange(6).view(1, 6),
        "atom_attention_mask": torch.ones(1, 6, dtype=torch.bool),
        "ref_atom_name_chars": torch.tensor([[[35, 33, 0, 0]] * 6]),
    }
    chains = [
        SimpleNamespace(
            tokens=[
                SimpleNamespace(token_index=i, residue_index=i, residue_name="ALA")
                for i in range(3)
            ]
        ),
        SimpleNamespace(
            tokens=[
                SimpleNamespace(token_index=i + 3, residue_index=i, residue_name="ALA")
                for i in range(3)
            ]
        ),
    ]
    predicted = torch.tensor((*chain_b, *chain_a), dtype=torch.float32)
    actual, mask = _aligned_true_coordinates(
        features, chains, {"chains": [{"sequence": "AAA"}, {"sequence": "AAA"}]}, path, predicted
    )
    assert mask.all()
    torch.testing.assert_close(actual, predicted, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(1, 4, 3), (1, 1, 4, 3)])
def test_native_coordinate_layout_keeps_the_sample_axis(shape):
    coordinates = torch.arange(12).reshape(shape)
    assert torch.equal(_single_sample_coordinates(coordinates), coordinates.reshape(1, 4, 3))


def test_coordinate_cache_rejects_multiple_samples():
    with pytest.raises(ValueError, match="exactly one"):
        _single_sample_coordinates(torch.zeros(1, 2, 4, 3))


def test_folding_loader_preserves_frozen_base_after_release(monkeypatch):
    original = get_model_spec("esmfold2_300").fast
    published = SimpleNamespace(repo_id=original.repo_id, revision="f" * 40)
    monkeypatch.setattr(cache, "get_model_spec", lambda model_id: SimpleNamespace(
        fast=published, confidence_training_base=original,
    ))
    config_loader = Mock(return_value=SimpleNamespace())
    monkeypatch.setattr(ESMFold2Config, "from_pretrained", config_loader)
    model = Mock()
    model.to.return_value = model
    model.eval.return_value = model
    model.requires_grad_.return_value = model
    model_loader = Mock(return_value=model)
    monkeypatch.setattr(cache.ESMFold2ExperimentalModel, "from_pretrained", model_loader)
    loaded = cache.load_folding_model("esmfold2_300", device="cpu")
    assert loaded is model
    assert config_loader.call_args.kwargs["revision"] == original.revision
    assert model_loader.call_args.kwargs["revision"] == original.revision
    assert loaded._fastplms_revision == original.revision
    assert loaded._fastplms_pins == ";".join(item.encoded for item in original.files)
