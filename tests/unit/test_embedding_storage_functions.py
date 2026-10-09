"""Saving, loading and resuming embedding results, one public function at a time."""

from __future__ import annotations

import sqlite3
import pytest
import torch

from pathlib import Path

from fastplms.embeddings import EmbeddingRecord, EmbeddingResult
from fastplms.embeddings.storage import (
    append_sqlite_records,
    initialize_sqlite_run,
    load_result,
    load_sqlite_result,
    safetensors_result_exists,
    save_result,
    tensor_sha256,
    update_sqlite_run_metadata,
)


RUN = "run-fingerprint-a"


def records() -> list[EmbeddingRecord]:
    generator = torch.Generator().manual_seed(7)
    return [
        EmbeddingRecord("p1", "ACD", torch.randn(3, 4, generator=generator)),
        EmbeddingRecord("p2", "GGKL", torch.randn(4, 4, generator=generator).to(torch.bfloat16)),
        EmbeddingRecord("p3", "T", torch.randn(1, 4, generator=generator).to(torch.float16)),
    ]


def metadata(run: str = RUN) -> dict[str, object]:
    return {"run_fingerprint": run, "model": "synthetic", "complete": True}


def assert_same_records(loaded: EmbeddingResult, expected: list[EmbeddingRecord]) -> None:
    assert [(record.id, record.sequence) for record in loaded] == [(r.id, r.sequence) for r in expected]
    for record, original in zip(loaded, expected, strict=True):
        X = record.load_tensor()
        assert X.dtype == original.tensor.dtype and X.shape == original.tensor.shape
        assert X.view(torch.uint8).tolist() == original.tensor.view(torch.uint8).tolist()


@pytest.mark.parametrize("format", ["safetensors", "sqlite"])
def test_save_then_load_returns_every_tensor_bit_for_bit(tmp_path: Path, format: str) -> None:
    path = tmp_path / ("run" if format == "safetensors" else "run.sqlite")
    expected = records()
    saved = save_result(EmbeddingResult(expected, metadata()), path, format=format)
    assert_same_records(saved, expected)
    loaded = load_result(path, format=format)
    assert_same_records(loaded, expected)
    assert loaded.metadata["run_fingerprint"] == RUN
    assert loaded.metadata["model"] == "synthetic"


def test_save_defaults_to_safetensors(tmp_path: Path) -> None:
    path = tmp_path / "default"
    save_result(EmbeddingResult(records(), metadata()), path)
    assert safetensors_result_exists(path)
    assert_same_records(load_result(path), records())


def test_unknown_and_pickle_formats_are_refused(tmp_path: Path) -> None:
    embeddings = EmbeddingResult(records(), metadata())
    with pytest.raises(ValueError, match="pickle-based .pth embeddings is not supported"):
        save_result(embeddings, tmp_path / "x", format="pth")
    with pytest.raises(ValueError, match="format must be 'safetensors' or 'sqlite'"):
        save_result(embeddings, tmp_path / "x", format="parquet")
    with pytest.raises(ValueError, match="format must be 'safetensors' or 'sqlite'"):
        load_result(tmp_path / "x", format="pth")


def test_safetensors_result_exists_only_for_a_committed_run(tmp_path: Path) -> None:
    path = tmp_path / "embeddings"
    assert safetensors_result_exists(path) is False
    save_result(EmbeddingResult(records(), metadata()), path)
    assert safetensors_result_exists(path) is True
    # run.json is the commit record and snapshots the index, so a damaged index.json alone
    # leaves the run committed.
    (path / "index.json").write_text("{not json", encoding="utf-8")
    assert safetensors_result_exists(path) is True
    (path / "run.json").write_text("{not json", encoding="utf-8")
    assert safetensors_result_exists(path) is False


def test_tensor_hash_covers_dtype_shape_and_bytes() -> None:
    X = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert tensor_sha256(X) == tensor_sha256(X.clone())
    assert tensor_sha256(X) != tensor_sha256(X.reshape(3, 2))
    assert tensor_sha256(X) != tensor_sha256(X.to(torch.float64))
    assert tensor_sha256(X) != tensor_sha256(X + 1)
    assert tensor_sha256(X.t().contiguous().t()) == tensor_sha256(X)
    with pytest.raises(TypeError, match="X must be a tensor"):
        tensor_sha256([1, 2, 3])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="meta tensor"):
        tensor_sha256(torch.empty(2, device="meta"))


class TestSqliteRun:
    def test_initialize_creates_an_empty_run_and_returns_its_id(self, tmp_path: Path) -> None:
        path = tmp_path / "run.sqlite"
        assert initialize_sqlite_run(path, metadata(), resume=True) == RUN
        loaded = load_sqlite_result(path, run_id=RUN)
        assert len(loaded) == 0
        assert loaded.metadata["record_count"] == 0

    def test_initialize_requires_a_run_fingerprint(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="run_fingerprint"):
            initialize_sqlite_run(tmp_path / "run.sqlite", {"model": "x"}, resume=True)

    def test_resume_keeps_committed_records_and_a_fresh_start_drops_them(self, tmp_path: Path) -> None:
        path = tmp_path / "run.sqlite"
        initialize_sqlite_run(path, metadata(), resume=True)
        append_sqlite_records(path, RUN, 0, records()[:2])
        initialize_sqlite_run(path, metadata(), resume=True)
        assert len(load_sqlite_result(path, run_id=RUN)) == 2
        initialize_sqlite_run(path, metadata(), resume=False)
        assert len(load_sqlite_result(path, run_id=RUN)) == 0

    def test_appended_batches_extend_a_contiguous_prefix(self, tmp_path: Path) -> None:
        path = tmp_path / "run.sqlite"
        expected = records()
        initialize_sqlite_run(path, metadata(), resume=True)
        append_sqlite_records(path, RUN, 0, expected[:2])
        append_sqlite_records(path, RUN, 2, expected[2:])
        loaded = load_sqlite_result(path, run_id=RUN)
        assert_same_records(loaded, expected)
        assert loaded.metadata["record_count"] == 3

    def test_a_batch_must_start_where_the_prefix_ends(self, tmp_path: Path) -> None:
        path = tmp_path / "run.sqlite"
        initialize_sqlite_run(path, metadata(), resume=True)
        append_sqlite_records(path, RUN, 0, records()[:1])
        with pytest.raises(ValueError, match="does not match the contiguous"):
            append_sqlite_records(path, RUN, 2, records()[1:])
        with pytest.raises(ValueError, match="does not match the contiguous"):
            append_sqlite_records(path, RUN, 0, records()[1:])
        assert len(load_sqlite_result(path, run_id=RUN)) == 1

    def test_arguments_are_checked(self, tmp_path: Path) -> None:
        path = tmp_path / "run.sqlite"
        initialize_sqlite_run(path, metadata(), resume=True)
        with pytest.raises(ValueError, match="run_id"):
            append_sqlite_records(path, "", 0, [])
        with pytest.raises(TypeError, match="start_position"):
            append_sqlite_records(path, RUN, True, [])  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="start_position"):
            append_sqlite_records(path, RUN, -1, [])
        with pytest.raises(TypeError, match="list of EmbeddingRecord"):
            append_sqlite_records(path, RUN, 0, [("p1", "ACD")])  # type: ignore[list-item]

    def test_appending_to_an_unknown_run_fails(self, tmp_path: Path) -> None:
        path = tmp_path / "run.sqlite"
        initialize_sqlite_run(path, metadata(), resume=True)
        with pytest.raises(KeyError, match="Missing SQLite embedding run other"):
            append_sqlite_records(path, "other", 0, records()[:1])

    def test_replacement_metadata_restarts_the_run_under_the_same_id(self, tmp_path: Path) -> None:
        path = tmp_path / "run.sqlite"
        initialize_sqlite_run(path, metadata(), resume=True)
        append_sqlite_records(path, RUN, 0, records()[:2])
        replacement = {**metadata(), "model": "replacement"}
        append_sqlite_records(path, RUN, 0, records()[:1], replace_metadata=replacement)
        loaded = load_sqlite_result(path, run_id=RUN)
        assert len(loaded) == 1
        assert loaded.metadata["model"] == "replacement"
        with pytest.raises(ValueError, match="must match the SQLite run ID"):
            append_sqlite_records(path, RUN, 0, [], replace_metadata=metadata("another-run"))

    def test_update_metadata_rewrites_the_run_record_with_the_stored_count(self, tmp_path: Path) -> None:
        path = tmp_path / "run.sqlite"
        initialize_sqlite_run(path, metadata(), resume=True)
        append_sqlite_records(path, RUN, 0, records()[:2])
        update_sqlite_run_metadata(path, RUN, {**metadata(), "elapsed_seconds": 1.5, "outputs": ["dropped"]})
        loaded = load_sqlite_result(path, run_id=RUN)
        assert loaded.metadata["elapsed_seconds"] == 1.5
        assert loaded.metadata["record_count"] == 2
        assert "outputs" not in loaded.metadata

    def test_update_metadata_of_a_missing_run_fails_and_changes_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / "run.sqlite"
        initialize_sqlite_run(path, metadata(), resume=True)
        with pytest.raises(KeyError, match="Missing SQLite embedding run absent"):
            update_sqlite_run_metadata(path, "absent", metadata("absent"))
        with sqlite3.connect(path) as connection:
            assert connection.execute("SELECT COUNT(*) FROM runs").fetchone() == (1,)
