"""A complex must reserve every component chain across adaptation splits."""

import pytest

from tools.confidence.splits import select_disjoint


def test_complex_second_chain_excludes_training_target():
    records = [
        {"id": "test", "pdb_id": "1aaa", "split": "final_test", "chains": [{}, {}]},
        {"id": "train-overlap", "pdb_id": "2aaa", "split": "train", "chains": [{}]},
        {"id": "train-clean", "pdb_id": "3aaa", "split": "train", "chains": [{}]},
    ]
    clusters = {
        ("test", 0): "a",
        ("test", 1): "b",
        ("train-overlap", 0): "b",
        ("train-clean", 0): "c",
    }
    counts = {"train": {1: 1}, "validation": {}, "final_test": {2: 1}}
    selected = select_disjoint(records, clusters, counts)
    assert [r["id"] for r in selected] == ["test", "train-clean"]


def test_insufficient_independent_test_targets_fail():
    with pytest.raises(ValueError, match="disjoint"):
        select_disjoint([], {}, {"train": {}, "validation": {}, "final_test": {1: 64}})


def test_final_holdout_and_training_use_distinct_pool_clusters():
    records = [
        {"id": str(i), "pdb_id": f"{i}abc", "split": "pool", "chains": [{}]} for i in range(5)
    ]
    clusters = {(str(i), 0): str(i) for i in range(5)}
    selected = select_disjoint(
        records, clusters, {"train": {1: 2}, "validation": {}, "final_test": {1: 2}}
    )
    assert len({item["id"] for item in selected}) == 4
    assert [item["split"] for item in selected] == ["final_test", "final_test", "train", "train"]
    assert all(item["split"] == "pool" for item in records)
