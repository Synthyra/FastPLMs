"""Conversion moves an old cache into a feature store and fails loudly when a row does not survive."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastplms.features import (
    CSR,
    DENSE,
    RAGGED,
    RAGGED_TOPK,
    ConversionMismatch,
    FeatureReader,
    FeatureStore,
    SparseRow,
    StoredFeature,
    TopKRow,
    conversion_fingerprint,
    convert_rows,
    describe_file,
)
from fastplms.features.store import COMMIT_FILE, SEGMENTS_DIRECTORY


SEQUENCES = tuple("M" + "ACDEFGHIK"[: 1 + i % 9] + "L" * i for i in range(12))
ORIGIN = {"format": "fixture", "files": [{"name": "old.bin", "bytes": 1, "sha256": "0" * 64}]}


def dense_rows(count: int, dtype: torch.dtype = torch.float32) -> list[tuple[str, torch.Tensor]]:
    return [(SEQUENCES[i], (torch.arange(4) + 7 * i).to(dtype)) for i in range(count)]  # count rows, each a sequence with a (4,) vector


def open_dense(tmp_path, dtype: torch.dtype = torch.float32) -> FeatureStore:
    return FeatureStore.open(tmp_path, StoredFeature("converted", DENSE, 4, dtype))


def test_a_dense_cache_converts_and_reads_back_exactly(tmp_path) -> None:
    store = open_dense(tmp_path)
    old = dense_rows(6)
    receipt = convert_rows(store, lambda: iter(old), origin=ORIGIN)
    assert (receipt.source_rows, receipt.written, receipt.skipped, receipt.verified) == (6, 6, 0, 6)
    with FeatureReader.open(store.directory) as reader:
        for (_sequence, row), stored in zip(old, reader.read([s for s, _ in old]), strict=True):
            assert torch.equal(stored, row)
    marker = next((store.directory / SEGMENTS_DIRECTORY).glob(f"*/{COMMIT_FILE}"))
    assert conversion_fingerprint(ORIGIN) in str(marker)
    assert '"origin"' in marker.read_text(encoding="utf-8")


@pytest.mark.parametrize("layout", [RAGGED, CSR, RAGGED_TOPK])
def test_every_layout_converts(tmp_path, layout) -> None:
    if layout == RAGGED:
        spec = StoredFeature("converted-ragged", RAGGED, 3, torch.bfloat16)
        old = [(SEQUENCES[i], torch.full((1 + i, 3), float(i)).to(torch.bfloat16)) for i in range(5)]
    elif layout == CSR:
        spec = StoredFeature("converted-csr", CSR, 16, torch.float16, positions=True)
        old = [
            (SEQUENCES[i], SparseRow(
                torch.tensor([i, i + 3], dtype=torch.int64), torch.tensor([1.0, 2.5 + i]).half(),
                torch.tensor([i, 4], dtype=torch.int64),
            ))
            for i in range(5)
        ]
    else:
        spec = StoredFeature("converted-topk", RAGGED_TOPK, 32, torch.float32, sparse_count=2)
        old = [
            (SEQUENCES[i], TopKRow(
                torch.tensor([[i, i + 1]] * (1 + i), dtype=torch.int64), torch.full((1 + i, 2), 0.25),
            ))
            for i in range(5)
        ]
    store = FeatureStore.open(tmp_path, spec)
    receipt = convert_rows(store, lambda: iter(old), origin=ORIGIN)
    assert receipt.written == 5 and receipt.verified == 5


def test_a_sparse_row_with_no_entries_converts_and_verifies(tmp_path) -> None:
    """A decoder built on numpy hands out empty rows whose tensors have stride zero.

    They hold no bytes, so the comparison must treat them as equal instead of refusing to view them.
    """

    def numpy_row(indices: list[int], values: list[float], positions: list[int]) -> SparseRow:
        return SparseRow(
            torch.from_numpy(np.array(indices, dtype=np.int32)),
            torch.from_numpy(np.array(values, dtype=np.float16)),
            torch.from_numpy(np.array(positions, dtype=np.int16)),
        )

    spec = StoredFeature("converted-empty", CSR, 16, torch.float16, positions=True)
    old = [
        (SEQUENCES[0], numpy_row([], [], [])),
        (SEQUENCES[1], numpy_row([1, 5], [1.0, 2.0], [3, 4])),
        (SEQUENCES[2], numpy_row([], [], [])),
    ]

    receipt = convert_rows(FeatureStore.open(tmp_path, spec), lambda: iter(old), origin=ORIGIN)

    assert (receipt.written, receipt.verified) == (3, 3)
    with FeatureReader.open(tmp_path / spec.key) as reader:
        rows = reader.read_sparse([SEQUENCES[0], SEQUENCES[1]])
    assert rows[0].indices.numel() == 0 and rows[1].indices.tolist() == [1, 5]


def test_a_lossy_dtype_is_refused(tmp_path) -> None:
    store = open_dense(tmp_path, torch.float16)
    old = [(SEQUENCES[0], torch.tensor([0.1, 0.2, 0.3, 0.4]))]  # float32 values float16 cannot hold
    with pytest.raises(ConversionMismatch, match="would change them"):
        convert_rows(store, lambda: iter(old), origin=ORIGIN)


def test_a_wider_source_that_fits_is_stored_exactly(tmp_path) -> None:
    store = open_dense(tmp_path, torch.float16)
    old = [(SEQUENCES[0], torch.tensor([0.5, 1.0, 2.0, 4.0]))]  # exactly representable
    assert convert_rows(store, lambda: iter(old), origin=ORIGIN).verified == 1


def test_a_source_that_changes_between_decodes_is_caught(tmp_path) -> None:
    store = open_dense(tmp_path)
    calls = {"count": 0}

    def drifting():
        calls["count"] += 1
        bump = 0.0 if calls["count"] == 1 else 1.0
        return iter([(SEQUENCES[0], torch.arange(4).float() + bump)])

    with pytest.raises(ConversionMismatch, match="differs from the source"):
        convert_rows(store, drifting, origin=ORIGIN)


def test_a_source_with_a_different_row_count_the_second_time_is_caught(tmp_path) -> None:
    store = open_dense(tmp_path)
    calls = {"count": 0}

    def shrinking():
        calls["count"] += 1
        return iter(dense_rows(4 if calls["count"] == 1 else 3))

    with pytest.raises(ConversionMismatch, match="the first time"):
        convert_rows(store, shrinking, origin=ORIGIN)


def test_an_empty_source_is_refused(tmp_path) -> None:
    with pytest.raises(ConversionMismatch, match="decoded no rows"):
        convert_rows(open_dense(tmp_path), lambda: iter(()), origin=ORIGIN)


def test_a_sequence_repeated_with_different_rows_is_refused(tmp_path) -> None:
    store = open_dense(tmp_path)
    old = [(SEQUENCES[0], torch.zeros(4)), (SEQUENCES[0], torch.ones(4))]
    with pytest.raises(ConversionMismatch, match="repeats a sequence"):
        convert_rows(store, lambda: iter(old), origin=ORIGIN)


def test_a_repeated_sequence_with_the_same_row_is_written_once(tmp_path) -> None:
    store = open_dense(tmp_path)
    old = [(SEQUENCES[0], torch.zeros(4)), (SEQUENCES[1], torch.ones(4)), (SEQUENCES[0], torch.zeros(4))]
    receipt = convert_rows(store, lambda: iter(old), origin=ORIGIN)
    assert (receipt.source_rows, receipt.written, receipt.skipped) == (3, 2, 1)


def test_segments_commit_as_they_fill_and_a_rerun_resumes(tmp_path) -> None:
    store = open_dense(tmp_path)
    old = dense_rows(10)
    # A segment closes at the first window boundary at or past its row budget: 6 rows, then 4.
    first = convert_rows(store, lambda: iter(old), origin=ORIGIN, rows_per_segment=4, window_rows=3)
    assert len(first.segments) == 2 and first.written == 10
    second = convert_rows(store, lambda: iter(old), origin=ORIGIN, rows_per_segment=4, window_rows=3)
    assert (second.written, second.skipped, second.verified, second.segments) == (0, 10, 10, ())


def test_an_interrupted_conversion_resumes_from_its_committed_segments(tmp_path) -> None:
    store = open_dense(tmp_path)
    old = dense_rows(10)
    calls = {"count": 0}

    def dying():
        calls["count"] += 1
        for position, pair in enumerate(old):
            if calls["count"] == 1 and position == 6:
                raise RuntimeError("the decoder died")
            yield pair

    with pytest.raises(RuntimeError, match="decoder died"):
        convert_rows(store, dying, origin=ORIGIN, rows_per_segment=4, window_rows=2)
    assert len(store) == 4  # one segment committed, the open one left uncommitted
    resumed = convert_rows(store, dying, origin=ORIGIN, rows_per_segment=4, window_rows=2)
    assert resumed.written == 6 and resumed.verified == 10


def test_rows_already_in_the_store_that_differ_fail_the_comparison(tmp_path) -> None:
    store = open_dense(tmp_path)
    with store.segment("fresh") as writer:
        writer.append([SEQUENCES[0]], [torch.full((4,), 9.0)])
    with pytest.raises(ConversionMismatch, match="differs from the source"):
        convert_rows(store, lambda: iter(dense_rows(3)), origin=ORIGIN)


def test_describe_file_records_content_not_metadata(tmp_path) -> None:
    path = tmp_path / "old.bin"
    path.write_bytes(b"abc")
    described = describe_file(path)
    assert described["bytes"] == 3 and described["name"] == "old.bin"
    assert described["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
