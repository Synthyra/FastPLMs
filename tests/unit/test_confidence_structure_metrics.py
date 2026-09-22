"""Tests for the structure metric adapters using mocked external APIs."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from tools.confidence.structure_metrics import compute_structure_metrics


def _cache(token_count: int = 2) -> dict[str, np.ndarray]:
    atom_count = token_count * 3
    return {
        "x_pred": np.arange(atom_count * 3, dtype=float).reshape(1, 1, atom_count, 3),
        "true_coords": np.arange(atom_count * 3, dtype=float).reshape(atom_count, 3),
        "resolved_mask": np.ones(atom_count, dtype=bool),
        "backbone_indices": np.arange(atom_count).reshape(token_count, 3),
        "atom_to_token": np.arange(atom_count) // 3,
        "asym_id": np.array([1] * (token_count // 2) + [2] * (token_count - token_count // 2)),
        "ref_atom_name_chars": np.array([[ord(char) - 32 for char in "CA\0\0"]] * atom_count),
        "token_attention_mask": np.ones(token_count, dtype=bool),
    }


def test_monomer_returns_tm_and_no_dockq(monkeypatch, tmp_path) -> None:
    tmtools = types.ModuleType("tmtools")
    tmtools.tm_align = lambda *args: types.SimpleNamespace(tm_norm_chain1=0.7, tm_norm_chain2=0.6)
    monkeypatch.setitem(sys.modules, "tmtools", tmtools)
    result = compute_structure_metrics(
        _cache(), {"id": "mono", "chains": [{"id": "A", "sequence": "AC"}]}, tmp_path
    )
    assert result == {"tm_score": 0.7, "dockq": None}


def test_installed_structure_metrics_on_toy_complex(tmp_path) -> None:
    pytest.importorskip("tmtools")
    pytest.importorskip("DockQ.DockQ")
    atom_names = ["N", "CA", "C", "O"]
    coordinates = []
    for chain_offset in (0.0, 3.0):
        for residue_offset in (0.0, 1.5):
            coordinates.extend(
                [
                    [chain_offset + residue_offset, 0.0, 0.0],
                    [chain_offset + residue_offset, 1.0, 0.0],
                    [chain_offset + residue_offset, 2.0, 0.0],
                    [chain_offset + residue_offset, 2.5, 0.0],
                ]
            )
    cache = {
        "x_pred": np.asarray(coordinates, dtype=float),
        "true_coords": np.asarray(coordinates, dtype=float),
        "resolved_mask": np.ones(16, dtype=bool),
        "backbone_indices": np.asarray([[0, 1, 2], [4, 5, 6], [8, 9, 10], [12, 13, 14]]),
        "atom_to_token": np.repeat(np.arange(4), 4),
        "asym_id": np.asarray([1, 1, 2, 2]),
        "ref_atom_name_chars": np.asarray(
            [
                [ord(char) - 32 if char != " " else 0 for char in name.ljust(4)]
                for name in atom_names
            ]
            * 4
        ),
        "token_attention_mask": np.ones(4, dtype=bool),
    }
    result = compute_structure_metrics(
        cache,
        {"id": "toy", "chains": [{"id": "A", "sequence": "AC"}, {"id": "B", "sequence": "DE"}]},
        tmp_path,
    )
    assert np.isfinite(result["tm_score"])
    assert result["dockq"] == pytest.approx(1.0, abs=1e-6)


def test_dimer_calls_native_dockq(monkeypatch, tmp_path) -> None:
    tmtools = types.ModuleType("tmtools")
    tmtools.tm_align = lambda *args: types.SimpleNamespace(tm_norm_chain1=0.7, tm_norm_chain2=0.6)
    monkeypatch.setitem(sys.modules, "tmtools", tmtools)
    dockq = types.ModuleType("DockQ")
    dockq_submodule = types.ModuleType("DockQ.DockQ")
    dockq_submodule.load_PDB = lambda path: path
    dockq_submodule.run_on_all_native_interfaces = lambda model, native: (
        {("A", "B"): {"DockQ": 0.8}},
        0.8,
    )
    monkeypatch.setitem(sys.modules, "DockQ", dockq)
    monkeypatch.setitem(sys.modules, "DockQ.DockQ", dockq_submodule)
    record = {
        "id": "dimer",
        "chains": [{"id": "A", "sequence": "AC"}, {"id": "B", "sequence": "DE"}],
    }
    result = compute_structure_metrics(_cache(4), record, tmp_path)
    assert result["dockq"] == 0.8


def test_padded_tokens_are_masked() -> None:
    cache = _cache(2)
    cache["backbone_indices"] = np.concatenate(
        [cache["backbone_indices"], np.asarray([[-1, -1, -1]])], axis=0
    )
    cache["token_attention_mask"] = np.asarray([True, True, False])
    cache["asym_id"] = np.asarray([1, 1])
    cache["x_pred"] = cache["x_pred"]

    import tools.confidence.structure_metrics as structure_metrics

    predicted, truth, sequence, _ = structure_metrics._ca_coordinates(
        cache, {"chains": [{"id": "source", "sequence": "AC"}]}
    )
    assert predicted.shape == (2, 3)
    assert truth.shape == (2, 3)
    assert sequence == "AC"


def test_nonfinite_structure_coordinates_are_rejected(monkeypatch, tmp_path) -> None:
    tmtools = types.ModuleType("tmtools")
    tmtools.tm_align = lambda *args: types.SimpleNamespace(tm_norm_chain1=0.7, tm_norm_chain2=0.6)
    monkeypatch.setitem(sys.modules, "tmtools", tmtools)
    cache = _cache()
    cache["true_coords"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compute_structure_metrics(
            cache, {"id": "mono", "chains": [{"id": "A", "sequence": "AC"}]}, tmp_path
        )


def test_unresolved_nonfinite_atoms_are_omitted_from_pdb(tmp_path) -> None:
    import tools.confidence.structure_metrics as structure_metrics

    cache = _cache()
    cache["true_coords"][0, 0] = np.nan
    cache["resolved_mask"][0] = False
    path = tmp_path / "structure.pdb"
    structure_metrics._write_pdb(
        path, cache["true_coords"], cache, {"chains": [{"id": "A", "sequence": "AC"}]}
    )
    assert path.exists()
    assert "nan" not in path.read_text().lower()


def test_unmasked_padded_backbone_tokens_are_rejected() -> None:
    import tools.confidence.structure_metrics as structure_metrics

    cache = _cache(2)
    cache["backbone_indices"] = np.concatenate(
        [cache["backbone_indices"], np.asarray([[-1, -1, -1]])], axis=0
    )
    cache["token_attention_mask"] = np.asarray([True, True, True])
    with pytest.raises(ValueError, match="padded backbone"):
        structure_metrics._ca_coordinates(cache, {"chains": [{"id": "A", "sequence": "AC"}]})
