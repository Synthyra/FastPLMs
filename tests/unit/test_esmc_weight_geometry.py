"""CPU contracts for the weights-only ESMC layer-geometry analysis."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.analysis import esmc_weight_geometry as geometry

torch = pytest.importorskip("torch")


def test_state_51_maps_to_output_of_block_50() -> None:
    assert geometry.state_for_block_output(50) == 51
    assert geometry.block_producing_state(51) == 50
    assert geometry.block_producing_state(0) is None
    with pytest.raises(ValueError):
        geometry.state_for_block_output(80)
    with pytest.raises(ValueError):
        geometry.block_producing_state(81)


def _fake_inventory() -> SimpleNamespace:
    records: dict[str, geometry.TensorRecord] = {}
    for block in range(geometry.N_BLOCKS):
        for role, shape in geometry._EXPECTED_SHAPES.items():
            name = geometry._block_key(block, role)
            records[name] = geometry.TensorRecord(
                name=name,
                file="model.safetensors",
                dtype="F32",
                shape=shape,
                nbytes=math.prod(shape) * 4,
                block=block,
                role=role,
            )
    final_name = "esmc.transformer.norm.weight"
    records[final_name] = geometry.TensorRecord(
        name=final_name,
        file="model.safetensors",
        dtype="F32",
        shape=(geometry.D_MODEL,),
        nbytes=geometry.D_MODEL * 4,
        block=None,
        role=None,
    )

    def record(name: str) -> geometry.TensorRecord:
        if name not in records:
            raise geometry.AnalysisError(name)
        return records[name]

    return SimpleNamespace(records=records, record=record)


def test_exact_inventory_covers_every_block_and_final_norm() -> None:
    selected = geometry.validate_esmc_inventory(_fake_inventory())
    assert len(selected) == geometry.N_BLOCKS * len(geometry._EXPECTED_SHAPES) + 1
    assert {record.block for record in selected if record.block is not None} == set(range(80))


def test_inventory_rejects_missing_or_misshaped_tensor() -> None:
    inventory = _fake_inventory()
    inventory.records.pop(geometry._block_key(50, "attn_output"))
    with pytest.raises(geometry.AnalysisError):
        geometry.validate_esmc_inventory(inventory)

    inventory = _fake_inventory()
    name = geometry._block_key(50, "attn_output")
    inventory.records[name] = geometry.TensorRecord(
        name=name,
        file="model.safetensors",
        dtype="F32",
        shape=(3, 4),
        nbytes=48,
        block=50,
        role="attn_output",
    )
    with pytest.raises(geometry.AnalysisError, match="expected"):
        geometry.validate_esmc_inventory(inventory)


def test_fused_matrix_splits_reconstruct_qkv_and_swiglu() -> None:
    qkv = torch.arange(3 * 8 * 5, dtype=torch.float32).reshape(24, 5)
    original_d_model = geometry.D_MODEL
    original_ffn_width = geometry.FFN_WIDTH
    geometry.D_MODEL = 8
    geometry.FFN_WIDTH = 6
    try:
        pieces = [
            geometry._split_matrix(family, qkv, None, None, None)
            for family in ("q", "k", "v")
        ]
        assert torch.equal(torch.cat(pieces), qkv)
        fc1 = torch.arange(12 * 5, dtype=torch.float32).reshape(12, 5)
        gate = geometry._split_matrix("gate", None, fc1, None, None)
        value = geometry._split_matrix("value", None, fc1, None, None)
        assert torch.equal(torch.cat((gate, value)), fc1)
    finally:
        geometry.D_MODEL = original_d_model
        geometry.FFN_WIDTH = original_ffn_width


def test_known_spectrum_and_rank_metrics() -> None:
    matrix = torch.diag(torch.tensor([4.0, 3.0, 0.0]))
    result = geometry.gram_spectrum(
        matrix,
        center="operator",
        device=torch.device("cpu"),
    )
    assert torch.allclose(
        result.singular_values,
        torch.tensor([4.0, 3.0, 0.0], dtype=torch.float64),
    )
    metrics = geometry.spectrum_metrics(result.singular_values, matrix.shape)
    assert metrics["stable_rank"] == pytest.approx(25 / 16)
    assert metrics["participation_ratio"] == pytest.approx(625 / 337)
    assert metrics["rank_90"] == 2
    assert metrics["rank_95"] == 2
    assert metrics["rank_99"] == 2


def test_zero_spectrum_preserves_complete_metric_schema() -> None:
    metrics = geometry.spectrum_metrics(torch.zeros(4), (8, 4), center="rows")
    assert metrics["stable_rank"] == 0
    assert metrics["rank_95"] == 0
    assert metrics["center_mean_parameters"] == 4
    assert metrics["rank_16_relative_error"] == 0
    assert metrics["rank_16_storage_fraction"] == pytest.approx((16 * 12 + 4) / 32)


def test_spectral_metrics_are_scale_invariant() -> None:
    generator = torch.Generator().manual_seed(7)
    matrix = torch.randn(18, 9, generator=generator)
    first = geometry.gram_spectrum(matrix, center="operator", device=torch.device("cpu"))
    second = geometry.gram_spectrum(
        matrix * 17,
        center="operator",
        device=torch.device("cpu"),
    )
    first_metrics = geometry.spectrum_metrics(first.singular_values, matrix.shape)
    second_metrics = geometry.spectrum_metrics(second.singular_values, matrix.shape)
    for name in ("stable_rank", "participation_ratio", "effective_rank"):
        assert first_metrics[name] == pytest.approx(second_metrics[name], rel=1e-8)


def test_row_and_column_centered_pca_invariances() -> None:
    generator = torch.Generator().manual_seed(11)
    matrix = torch.randn(20, 12, generator=generator)
    row_offset = torch.randn(12, generator=generator)
    row_first = geometry.gram_spectrum(matrix, center="rows", device=torch.device("cpu"))
    row_second = geometry.gram_spectrum(
        matrix + row_offset,
        center="rows",
        device=torch.device("cpu"),
    )
    assert torch.allclose(row_first.singular_values, row_second.singular_values, atol=1e-7)

    column_offset = torch.randn(20, generator=generator)
    column_first = geometry.gram_spectrum(
        matrix,
        center="columns",
        device=torch.device("cpu"),
    )
    column_second = geometry.gram_spectrum(
        matrix + column_offset[:, None],
        center="columns",
        device=torch.device("cpu"),
    )
    assert torch.allclose(
        column_first.singular_values,
        column_second.singular_values,
        atol=1e-7,
    )


def test_spectrum_is_invariant_to_orthogonal_transformations() -> None:
    generator = torch.Generator().manual_seed(13)
    matrix = torch.randn(12, 8, generator=generator, dtype=torch.float64)
    left, _ = torch.linalg.qr(torch.randn(12, 12, generator=generator, dtype=torch.float64))
    right, _ = torch.linalg.qr(torch.randn(8, 8, generator=generator, dtype=torch.float64))
    transformed = left @ matrix @ right
    first = geometry.gram_spectrum(matrix, center="operator", device=torch.device("cpu"))
    second = geometry.gram_spectrum(
        transformed,
        center="operator",
        device=torch.device("cpu"),
    )
    assert torch.allclose(first.singular_values, second.singular_values, atol=1e-8)


def test_twonn_and_mle_handle_duplicates_without_infinities() -> None:
    neighbors = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.2, 1.8, 2.4, 3.2],
            [0.9, 1.7, 2.8, 4.2],
        ]
    )
    result = geometry.intrinsic_dimension_metrics(neighbors, trim_fraction=0)
    assert result["duplicate_fraction"] == pytest.approx(0.25)
    assert result["twonn_valid_points"] == 3
    assert result["twonn_dimension"] is not None
    assert math.isfinite(float(result["twonn_dimension"]))


def test_nearest_neighbors_remove_self_and_are_scale_equivariant() -> None:
    points = torch.tensor([[0.0], [1.0], [3.0], [7.0]])
    first = geometry.nearest_neighbor_distances(
        points,
        k=2,
        normalized=False,
        device=torch.device("cpu"),
        chunk_size=2,
    )
    second = geometry.nearest_neighbor_distances(
        points * 5,
        k=2,
        normalized=False,
        device=torch.device("cpu"),
        chunk_size=3,
    )
    assert torch.all(first[:, 0] > 0)
    assert torch.allclose(second, first * 5)


def test_subspace_overlap_recovers_identical_and_orthogonal_spaces() -> None:
    identity = torch.eye(6, dtype=torch.float64)
    left = identity[:, :2]
    same = identity[:, :2]
    orthogonal = identity[:, 2:4]
    assert geometry.subspace_metrics(left, same)["normalized_overlap"] == pytest.approx(1)
    assert geometry.subspace_metrics(left, orthogonal)["normalized_overlap"] == pytest.approx(0)
    assert geometry.subspace_metrics(left, same)["maximum_angle_degrees"] == pytest.approx(0)


def test_low_rank_storage_marks_expansion_above_break_even() -> None:
    values = torch.ones(2560, dtype=torch.float64)
    metrics = geometry.spectrum_metrics(values, (2560, 2560))
    assert metrics["rank_1024_is_compression"] == 1
    assert metrics["rank_2048_is_compression"] == 0


@pytest.mark.parametrize("bits", (4, 8))
def test_symmetric_quantization_handles_zero_rows_deterministically(bits: int) -> None:
    matrix = torch.tensor([[0.0, 0.0, 0.0, 0.0], [-3.0, -1.0, 1.0, 3.0]])
    first, first_metrics = geometry.symmetric_row_quantize(matrix, bits)
    second, second_metrics = geometry.symmetric_row_quantize(matrix, bits)
    assert torch.equal(first, second)
    assert torch.equal(first[0], matrix[0])
    assert first_metrics == second_metrics
    assert math.isfinite(first_metrics["relative_frobenius_error"])


def test_magnitude_and_two_of_four_sparsity_are_deterministic() -> None:
    matrix = torch.arange(1, 17, dtype=torch.float32).reshape(2, 8)
    pruned, metrics = geometry.magnitude_prune(matrix, 0.5)
    assert int(torch.count_nonzero(pruned)) == 8
    assert metrics["actual_sparsity"] == pytest.approx(0.5)

    structured, structured_metrics = geometry.structured_two_of_four(matrix)
    assert int(torch.count_nonzero(structured.reshape(2, 2, 4), dim=2).min()) == 2
    assert int(torch.count_nonzero(structured.reshape(2, 2, 4), dim=2).max()) == 2
    assert structured_metrics["actual_sparsity"] == pytest.approx(0.5)


def test_trajectory_gram_matches_explicit_flattening() -> None:
    stack = torch.tensor(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 0.0],
        ]
    )
    gram, distances, cosine, summary = geometry._trajectory_from_stack(stack)
    assert torch.allclose(gram, stack.to(torch.float64) @ stack.to(torch.float64).mT)
    assert torch.allclose(torch.diagonal(distances), torch.zeros(3, dtype=torch.float64))
    assert torch.allclose(torch.diagonal(cosine), torch.ones(3, dtype=torch.float64))
    assert summary["trajectory_rank"] <= 2


def test_anomaly_table_preserves_state_mapping_and_finds_inserted_spike() -> None:
    rows = []
    for block in range(geometry.N_BLOCKS):
        rows.append(
            {
                "block": block,
                "produced_state": block + 1,
                "family": "q",
                "metric": math.sin(block / 12) + (8.0 if block == 50 else 0.0),
            }
        )
    result = geometry.anomaly_table(rows)
    candidate = next(
        row for row in result if row["metric"] == "metric" and row["block"] == 50
    )
    assert candidate["produced_state"] == 51
    assert candidate["prespecified_candidate"] == 1
    assert abs(float(candidate["local_robust_z"])) > 5
    assert float(candidate["phase_randomized_p"]) < 0.01


def test_all_nonzero_circular_shifts_are_unique_for_80_layers() -> None:
    profile = np.arange(geometry.N_BLOCKS)
    shifted = {tuple(np.roll(profile, shift)) for shift in range(1, geometry.N_BLOCKS)}
    assert len(shifted) == 79


def test_parse_fold_argument_requires_label_and_path() -> None:
    local_path = Path("fold-checkpoint")
    label, path = geometry.parse_fold_argument(f"standard={local_path}")
    assert label == "standard"
    assert path == local_path.resolve()
    with pytest.raises(argparse.ArgumentTypeError):
        geometry.parse_fold_argument(str(local_path))


def test_safetensor_checkpoint_inventory_and_lazy_tensor_read(tmp_path: Path) -> None:
    safetensors = pytest.importorskip("safetensors.torch")
    path = tmp_path / "model.safetensors"
    expected = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    safetensors.save_file({"example.weight": expected}, path)
    checkpoint = geometry.SafetensorCheckpoint(path)
    assert checkpoint.keys() == ("example.weight",)
    assert checkpoint.record("example.weight").shape == (3, 4)
    assert checkpoint.record("example.weight").nbytes == 48
    assert torch.equal(checkpoint.tensor("example.weight"), expected)


def test_range_subset_download_reconstructs_selected_tensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    safetensors = pytest.importorskip("safetensors.torch")
    source = tmp_path / "source.safetensors"
    tensors = {
        "prefix.first": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "prefix.second": torch.arange(4, dtype=torch.float32),
        "prefix.unused": torch.ones(5, dtype=torch.float32),
    }
    safetensors.save_file(tensors, source)
    raw = source.read_bytes()

    def fake_range(_url: str, start: int, stop: int) -> tuple[bytes, dict[str, str]]:
        return raw[start : stop + 1], {"Content-Range": f"bytes {start}-{stop}/{len(raw)}"}

    monkeypatch.setattr(geometry, "_hub_range", fake_range)
    output = tmp_path / "subset" / "model.safetensors"
    metadata = geometry.download_safetensor_subset(
        repo_id="example/repo",
        revision="a" * 40,
        filename="model.safetensors",
        suffixes=("first", "second"),
        output=output,
    )
    subset = geometry.SafetensorCheckpoint(output)
    assert set(subset.keys()) == {"prefix.first", "prefix.second"}
    assert torch.equal(subset.tensor("prefix.first"), tensors["prefix.first"])
    assert torch.equal(subset.tensor("prefix.second"), tensors["prefix.second"])
    assert metadata["subset_sha256"] == geometry._sha256_file(output)


def test_resume_requires_matching_fingerprint_and_artifact_hash(tmp_path: Path) -> None:
    run = geometry.AnalysisRun(tmp_path / "run", {"seed": 3}, resume=True)
    artifact = run.output_dir / "value.txt"
    artifact.write_text("first", encoding="utf-8")
    run.write_result("test", "one", {"value": 1}, (artifact,))
    assert run.is_complete("test", "one")
    artifact.write_text("changed", encoding="utf-8")
    assert not run.is_complete("test", "one")
    different = geometry.AnalysisRun(tmp_path / "run", {"seed": 4}, resume=True)
    assert not different.is_complete("test", "one")


def test_micro_checkpoint_streaming_spectra_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    safetensors = pytest.importorskip("safetensors.torch")
    monkeypatch.setattr(geometry, "N_BLOCKS", 1)
    monkeypatch.setattr(geometry, "N_STATES", 2)
    monkeypatch.setattr(geometry, "D_MODEL", 4)
    monkeypatch.setattr(geometry, "N_HEADS", 2)
    monkeypatch.setattr(geometry, "D_HEAD", 2)
    monkeypatch.setattr(geometry, "FFN_WIDTH", 3)
    monkeypatch.setattr(geometry, "SPECTRUM_RANKS", (1, 2, 3))
    expected_shapes = {
        "attn_qkv": (12, 4),
        "attn_input_norm_weight": (4,),
        "attn_input_norm_bias": (4,),
        "attn_output": (4, 4),
        "q_norm_weight": (4,),
        "k_norm_weight": (4,),
        "ffn_fc1": (6, 4),
        "ffn_down": (4, 3),
        "ffn_norm_weight": (4,),
        "ffn_norm_bias": (4,),
    }
    monkeypatch.setattr(geometry, "_EXPECTED_SHAPES", expected_shapes)
    generator = torch.Generator().manual_seed(29)
    tensors = {
        geometry._block_key(0, role): torch.randn(shape, generator=generator)
        for role, shape in expected_shapes.items()
    }
    tensors["esmc.transformer.norm.weight"] = torch.ones(4)
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    safetensors.save_file(tensors, checkpoint_root / "model.safetensors")
    checkpoint = geometry.SafetensorCheckpoint(checkpoint_root)
    assert len(geometry.validate_esmc_inventory(checkpoint)) == len(expected_shapes) + 1

    run = geometry.AnalysisRun(tmp_path / "analysis", {"micro": True}, resume=True)
    geometry.run_spectra(
        run,
        checkpoint,
        device=torch.device("cpu"),
        accumulation_dtype="float64",
    )
    geometry.run_normalization_metrics(run, checkpoint)
    geometry.run_ffn_pair_metrics(run, checkpoint)
    rows = geometry._read_csv(run.output_dir / "tensor_metrics.csv")
    assert {row["family"] for row in rows} == set(geometry.MATRIX_FAMILIES)
    assert len(rows) == len(geometry.MATRIX_FAMILIES)
    assert len(list((run.output_dir / "spectra").glob("*.npz"))) == len(
        geometry.MATRIX_FAMILIES
    )
