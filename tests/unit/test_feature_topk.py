"""Sparse residue storage preserves crop inputs and fails on corruption, without densifying."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import pickle
import sqlite3
import pytest
import torch

from concurrent.futures import ProcessPoolExecutor
from safetensors.torch import load_file, save_file

from fastplms.features import DENSE, RAGGED_TOPK, FeatureStore, StoredFeature, TopKRow


def spec(**changes):
    fields = dict(key="residue-codes", layout=RAGGED_TOPK, width=16384,
                  sparse_count=3, dtype=torch.float32)
    return StoredFeature(**(fields | changes))


def row(length, dtype=torch.float32):
    # Unsorted codes, zeros and repeated codes across residues are intentional.
    indices = torch.tensor([[9000, 0, 16383]], dtype=torch.int64).repeat(length, 1)
    values = torch.arange(length * 3, dtype=dtype).reshape(length, 3) / 4
    return TopKRow(indices, values)


def build(root, dtype=torch.float32):
    store = FeatureStore.open(root, spec(dtype=dtype))
    with store.segment("first") as writer:
        writer.append(["AC", "DEF"], [row(2, dtype), row(3, dtype)],
                      row_metadata=[{"residues": [0, 1]}, {"residues": [0, 1, 2]}])
    return store


def read_worker(store):
    rows = store.read_topk(["DEF", "AC", "DEF"])
    return [(item.indices.tolist(), item.values.tolist()) for item in rows]


def rewrite_part(store, change):
    """Adversarial source: rewrite bytes and internal checksums to test structural validation."""
    part = store.directory / "segments/first/part-00000.safetensors"
    tensors = {name: value.clone() for name, value in load_file(str(part)).items()}
    marker = part.with_name("run.json")
    payload = json.loads(marker.read_text())
    change(tensors, payload["parts"][0])
    save_file(tensors, str(part))
    payload["parts"][0]["sha256"] = hashlib.sha256(part.read_bytes()).hexdigest()
    payload["parts"][0]["tensor_bytes"] = sum(
        value.numel() * value.element_size() for value in tensors.values()
    )
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    payload["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    marker.write_text(json.dumps(payload))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_exact_roundtrip_order_zeros_duplicates_and_residue_metadata(tmp_path, dtype):
    store = build(tmp_path, dtype)
    assert StoredFeature.from_payload(store.spec.payload()) == store.spec
    readonly = FeatureStore.read_only(store.directory)
    request = ["DEF", "AC", "DEF"]
    for actual, length in zip(readonly.read_topk(request), (3, 2, 3), strict=True):
        expected = row(length, dtype)
        assert torch.equal(actual.indices, expected.indices) and actual.indices.dtype == torch.int32
        assert torch.equal(actual.values, expected.values) and actual.values.dtype == dtype
    assert readonly.residue_counts(request) == [3, 2, 3]
    assert readonly.row_metadata(request) == [
        {"residues": [0, 1, 2]}, {"residues": [0, 1]}, {"residues": [0, 1, 2]},
    ]
    # Returned rows have independent ownership, including repeated requests.
    values = readonly.read_topk(request)
    values[0].values.zero_()
    values[0].indices.zero_()
    assert torch.equal(values[2].values, row(3, dtype).values)
    assert torch.equal(readonly.read_topk(["DEF"])[0].indices, row(3).indices)


@pytest.mark.parametrize("change", [
    {"sparse_count": None}, {"sparse_count": True}, {"sparse_count": 0},
    {"sparse_count": 16385}, {"sparse_count": 3.0}, {"sparse_count": "3"},
    {"layout": DENSE}, {"width": 2**31 + 1}, {"positions": True},
])
def test_invalid_physical_descriptor_is_rejected(change):
    with pytest.raises(ValueError):
        spec(**change)


def test_legacy_descriptor_bytes_and_topk_feature_identity_are_distinct(tmp_path):
    legacy = StoredFeature("dense", DENSE, 8, torch.float32)
    assert "sparse_count" not in legacy.payload()
    assert set(legacy.payload()) == {
        "format", "key", "layout", "width", "dtype", "positions", "descriptor",
    }
    store = build(tmp_path)
    with pytest.raises(ValueError):
        FeatureStore.open(tmp_path, spec(sparse_count=2))
    assert store.read_topk([]) == []
    with pytest.raises(KeyError, match="no row"):
        store.read_topk(["MISSING"])
    with pytest.raises(ValueError, match="read_topk"):
        store.read(["AC"])
    with pytest.raises(ValueError, match="csr"):
        store.read_sparse(["AC"])
    dense = FeatureStore.open(tmp_path, legacy)
    with pytest.raises(ValueError, match="ragged_topk"):
        dense.read_topk([])


@pytest.mark.parametrize("bad,match", [
    (TopKRow(torch.tensor([[0., 1., 2.]]), torch.ones(1, 3)), "signed integer"),
    (TopKRow(torch.tensor([[0, 1, 2]], dtype=torch.uint8), torch.ones(1, 3)), "signed integer"),
    (TopKRow(torch.tensor([[0, 1]]), torch.ones(1, 2)), "shape"),
    (TopKRow(torch.tensor([0, 1, 2]), torch.ones(3)), "shape"),
    (TopKRow(torch.tensor([[0, 1, 2]]), torch.ones(3, 1)), "shape"),
    (TopKRow(torch.tensor([[0, 1, 2]]), torch.ones(1, 3, dtype=torch.int64)), "floating"),
    (TopKRow(torch.tensor([[0, 1, 2]]), torch.tensor([[0., float("nan"), 1.]])), "finite"),
    (TopKRow(torch.tensor([[0, 1, 2]]), torch.tensor([[0., float("inf"), 1.]])), "finite"),
    (TopKRow(torch.tensor([[0, 1, 16384]]), torch.ones(1, 3)), "outside"),
    (TopKRow(torch.tensor([[0, 1, -1]]), torch.ones(1, 3)), "outside"),
    (TopKRow(torch.tensor([[0, 2, 2]]), torch.ones(1, 3)), "unique"),
])
def test_invalid_rows_never_commit(tmp_path, bad, match):
    store = FeatureStore.open(tmp_path, spec())
    with pytest.raises(ValueError, match=match), store.segment("invalid") as writer:
        writer.append(["AC"], [bad])
    assert len(store) == 0
    assert not list(store.directory.glob("segments/*/run.json"))


def test_float16_overflow_and_wrong_row_type_are_rejected(tmp_path):
    store = FeatureStore.open(tmp_path, spec(dtype=torch.float16))
    bad = TopKRow(torch.tensor([[0, 1, 2]]), torch.full((1, 3), 100000.))
    with pytest.raises(ValueError, match="range"), store.segment("overflow") as writer:
        writer.append(["AC"], [bad])
    with pytest.raises(TypeError, match="TopKRow"), store.segment("wrong-type") as writer:
        writer.append(["AC"], [torch.zeros(1, 3)])
    with pytest.raises(TypeError, match="TopKRow"), store.segment("wrong-bounded") as writer:
        writer.append_bounded(["AC"], [torch.zeros(1, 3)], max_tensor_bytes=1024)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_bounded_parts_scale_with_k_not_codebook(tmp_path, dtype):
    store = FeatureStore.open(tmp_path, spec(dtype=dtype))
    lengths = [0, 2, 5, 1, 3, 2, 4]
    sequences = ["A" * (i + 1) for i in range(len(lengths))]
    rows = [row(length, dtype) for length in lengths]
    entry_size = 4 + torch.empty(0, dtype=dtype).element_size()
    budget = 16 + 5 * 3 * entry_size
    with store.segment("bounded") as writer:
        parts = writer.append_bounded(sequences, rows, max_tensor_bytes=budget)
    assert len(parts) > 1
    marker = json.loads((store.directory / "segments/bounded/run.json").read_text())
    for part in marker["parts"]:
        path = store.directory / f"segments/bounded/part-{part['part']:05d}.safetensors"
        tensors = load_file(str(path))
        measured = sum(value.numel() * value.element_size() for value in tensors.values())
        assert measured == part["tensor_bytes"] <= budget
        assert tensors["indices"].shape[1] == tensors["values"].shape[1] == 3
    for actual, expected in zip(store.read_topk(sequences), rows, strict=True):
        assert torch.equal(actual.indices, expected.indices)
        assert torch.equal(actual.values, expected.values)
    with pytest.raises(ValueError, match="exceeds"), store.segment("oversized") as writer:
        writer.append_bounded(["CC"], [row(6, dtype)], max_tensor_bytes=budget)
    assert store.missing(["CC"]) == ("CC",)


def test_boundary_codebook_index_and_full_topk_are_preserved(tmp_path):
    feature = spec(width=2**31, sparse_count=1)
    store = FeatureStore.open(tmp_path, feature)
    with store.segment("largest-index") as writer:
        writer.append(["AC"], [TopKRow(torch.tensor([[2**31 - 1]]), torch.tensor([[0.]]))])
    assert store.read_topk(["AC"])[0].indices.item() == 2**31 - 1
    full = FeatureStore.open(tmp_path, spec(key="full", width=3))
    with full.segment("full-k") as writer:
        writer.append(["AC"], [TopKRow(torch.tensor([[2, 1, 0]]), torch.tensor([[3., 2., 1.]]))])
    assert full.read_topk(["AC"])[0].indices.tolist() == [[2, 1, 0]]


@pytest.mark.parametrize("mutation,match", [
    (lambda t, p: t["indices"].__setitem__((0, 0), 16384), "outside"),
    (lambda t, p: t["indices"].__setitem__((0, 0), 0), "unique"),
    (lambda t, p: t["values"].__setitem__((0, 0), float("nan")), "finite"),
    (lambda t, p: t.__setitem__("indices", t["indices"].long()), "dtype"),
    (lambda t, p: t.__setitem__("indices", t["indices"][:, :2].contiguous()), "shape"),
    (lambda t, p: t["offsets"].__setitem__(1, 3), "residue counts"),
    (lambda t, p: t["offsets"].__setitem__(0, 1), "offsets"),
    (lambda t, p: t.__setitem__("positions", torch.zeros(5, 3, dtype=torch.int16)), "tensor names"),
])
def test_self_consistent_corrupt_part_fails_physical_validation(tmp_path, mutation, match):
    store = build(tmp_path)
    rewrite_part(store, mutation)
    with pytest.raises(ValueError, match=match):
        FeatureStore.read_only(store.directory).read_topk(["AC"])
    with pytest.raises(ValueError, match=match):
        store.reindex()


def test_independent_pins_catch_valid_replacement_values(tmp_path):
    store = build(tmp_path)
    pins = store.content_pins(["AC"])
    pinned = FeatureStore.read_only(store.directory, content_pins=pins)
    rewrite_part(store, lambda t, p: t["values"].__setitem__((0, 0), 123.))
    assert FeatureStore.read_only(store.directory).read_topk(["AC"])[0].values[0, 0] == 123
    with pytest.raises(ValueError, match="independent content pin"):
        pinned.read_topk(["AC"])


def test_reindex_readonly_pickle_and_spawned_readers(tmp_path):
    store = build(tmp_path)
    expected = read_worker(store)
    pins = store.content_pins(["AC", "DEF"])
    # Explicit writable recovery is still required; read-only workers never repair a missing index.
    (store.directory / "index.sqlite").unlink()
    with pytest.raises(sqlite3.OperationalError, match="unable to open"):
        len(FeatureStore.read_only(store.directory))
    assert not (store.directory / "index.sqlite").exists()
    store.reindex()
    reader = FeatureStore.read_only(store.directory, content_pins=pins)
    assert read_worker(pickle.loads(pickle.dumps(reader))) == expected
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
        assert list(pool.map(read_worker, [reader, reader])) == [expected, expected]
    with pytest.raises(PermissionError), reader.segment("forbidden"):
        pytest.fail("A read-only store entered a write transaction")
