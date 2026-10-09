"""``write_rows`` commits a window of rows as one segment named by what it holds."""

from __future__ import annotations

import json
import pytest
import torch

from fastplms.features import (
    CSR,
    DENSE,
    RAGGED_TOPK,
    FeatureReader,
    FeatureStore,
    SparseRow,
    StoredFeature,
    TopKRow,
    write_rows,
)
from fastplms.features.store import COMMIT_FILE, SEGMENTS_DIRECTORY


SEQUENCES = tuple("M" + "ACDEFGHIK"[: 1 + i % 9] + "L" * i for i in range(10))


def segment_names(store: FeatureStore) -> list[str]:
    markers = (store.directory / SEGMENTS_DIRECTORY).glob(f"*/{COMMIT_FILE}")
    return sorted(marker.parent.name for marker in markers)


def test_a_window_of_dense_rows_commits_and_reads_back(tmp_path) -> None:
    store = FeatureStore.open(tmp_path, StoredFeature("window", DENSE, 3, torch.float32))
    rows = {sequence: torch.arange(3, dtype=torch.float32) + index for index, sequence in enumerate(SEQUENCES[:4])}

    assert write_rows(store, rows) == 4

    with FeatureReader.open(store.directory) as reader:
        for sequence, stored in zip(rows, reader.read(list(rows)), strict=True):
            assert torch.equal(stored, rows[sequence])
    assert store.missing(SEQUENCES[:5]) == (SEQUENCES[4],)


def test_the_segment_name_depends_only_on_the_sequences_it_holds(tmp_path) -> None:
    first = FeatureStore.open(tmp_path / "first", StoredFeature("window", DENSE, 3, torch.float32))
    second = FeatureStore.open(tmp_path / "second", StoredFeature("window", DENSE, 3, torch.float32))
    window = {sequence: torch.zeros(3) for sequence in SEQUENCES[:3]}

    write_rows(first, window)
    write_rows(second, {sequence: torch.ones(3) for sequence in SEQUENCES[:3]})
    write_rows(second, {SEQUENCES[5]: torch.ones(3)})

    assert segment_names(first)[0] in segment_names(second)
    assert len(segment_names(first)) == 1 and len(segment_names(second)) == 2


def test_an_empty_window_writes_nothing(tmp_path) -> None:
    store = FeatureStore.open(tmp_path, StoredFeature("window", DENSE, 3, torch.float32))

    assert write_rows(store, {}) == 0
    assert segment_names(store) == []


def test_a_sequence_the_store_already_holds_is_refused(tmp_path) -> None:
    store = FeatureStore.open(tmp_path, StoredFeature("window", DENSE, 3, torch.float32))
    write_rows(store, {SEQUENCES[0]: torch.zeros(3)})

    with pytest.raises(ValueError):
        write_rows(store, {SEQUENCES[0]: torch.ones(3), SEQUENCES[1]: torch.ones(3)})
    assert store.missing(SEQUENCES[:2]) == (SEQUENCES[1],)


def test_metadata_is_recorded_in_the_commit_marker(tmp_path) -> None:
    store = FeatureStore.open(tmp_path, StoredFeature("window", DENSE, 3, torch.float32))

    write_rows(store, {SEQUENCES[0]: torch.zeros(3)}, metadata={"window": 7})

    marker = next((store.directory / SEGMENTS_DIRECTORY).glob(f"*/{COMMIT_FILE}"))
    assert json.loads(marker.read_text(encoding="utf-8"))["metadata"] == {"window": 7}


def test_a_small_tensor_budget_splits_a_window_into_parts_of_one_segment(tmp_path) -> None:
    store = FeatureStore.open(tmp_path, StoredFeature("window", DENSE, 64, torch.float32))
    rows = {sequence: torch.full((64,), float(index)) for index, sequence in enumerate(SEQUENCES[:6])}

    write_rows(store, rows, max_tensor_bytes=2 * 64 * 4)

    assert len(segment_names(store)) == 1
    with FeatureReader.open(store.directory) as reader:
        assert [float(row[0]) for row in reader.read(list(rows))] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_sparse_and_top_k_windows_commit(tmp_path) -> None:
    sparse = FeatureStore.open(tmp_path / "sparse", StoredFeature("window-csr", CSR, 16, torch.float16, positions=True))
    top_k = FeatureStore.open(
        tmp_path / "topk", StoredFeature("window-topk", RAGGED_TOPK, 32, torch.float32, sparse_count=2)
    )

    write_rows(sparse, {
        SEQUENCES[0]: SparseRow(torch.tensor([1, 4], dtype=torch.int32), torch.tensor([1.0, 2.0]).half(), torch.tensor([3, 5], dtype=torch.int16)),
        SEQUENCES[1]: SparseRow(torch.tensor([2], dtype=torch.int32), torch.tensor([0.5]).half(), torch.tensor([0], dtype=torch.int16)),
    })
    write_rows(top_k, {
        SEQUENCES[2]: TopKRow(torch.tensor([[0, 1], [2, 3]], dtype=torch.int32), torch.full((2, 2), 0.25)),
    })

    with FeatureReader.open(sparse.directory) as reader:
        assert reader.read_sparse([SEQUENCES[1], SEQUENCES[0]])[1].indices.tolist() == [1, 4]
    with FeatureReader.open(top_k.directory) as reader:
        assert reader.read_topk([SEQUENCES[2]])[0].indices.shape == (2, 2)
