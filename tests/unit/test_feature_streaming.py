"""Bounded feature delivery, lossless part limits, and independent read-only handles."""

from __future__ import annotations

import gc
import hashlib
import json
import multiprocessing
import pickle
import sqlite3
import weakref
import pytest
import torch

from safetensors import safe_open

from fastplms.embeddings import HiddenTap, TapRunReceipt, embed_dataset, embed_into_features
from fastplms.embeddings.batches import TapExecutor
from fastplms.features import CSR, DENSE, RAGGED, FeatureStore, SparseRow, StoredFeature
from fastplms.features.store import SegmentWriter
from fastplms.models.esm_plusplus.modeling_esm_plusplus import ESMplusplusConfig, ESMplusplusModel


def tiny_model():
    torch.manual_seed(5)
    return ESMplusplusModel(ESMplusplusConfig(
        hidden_size=16, num_attention_heads=2, num_hidden_layers=2, attn_backend="sdpa",
    )).eval()


def contents(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*") if p.is_file()}


def read_worker(store):
    """Spawned workers receive a handle containing paths and descriptors, never connections."""
    return store.read(["CC", "A", "CC"])[0].tolist(), len(store)


def test_sink_matches_memory_order_fingerprint_and_only_keeps_one_window(monkeypatch):
    model = tiny_model()
    sequences = ["AC" * (index + 1) for index in range(11)]
    taps = [HiddenTap("mean", -1, ("mean", "var")), HiddenTap("rows", -1)]
    options = dict(batch_size=2, batch_window_size=4, max_length=32, truncate=True)
    expected = embed_dataset(model, sequences, taps=taps, **options)
    previous, received, identities = [], [], []
    original = TapExecutor.run_window
    def check_release(self, records, *, window_start):
        gc.collect()
        assert all(ref() is None for ref in previous)
        return original(self, records, window_start=window_start)
    monkeypatch.setattr(TapExecutor, "run_window", check_release)
    def consume(records, identity):
        assert len(records) <= 4
        for actual in records:
            reference = expected[len(received)]
            assert actual.sequence == reference.sequence and actual.id == reference.id
            for name in actual.tensors:
                torch.testing.assert_close(
                    actual.tensors[name], reference.tensors[name], rtol=0, atol=0,
                )
                previous.append(weakref.ref(actual.tensors[name]))
            received.append(actual.sequence)
        identities.append(identity)
    receipt = embed_dataset(model, iter(sequences), taps=taps, tap_sink=consume, **options)
    gc.collect()
    assert isinstance(receipt, TapRunReceipt) and receipt.record_count == len(sequences)
    assert receipt.metadata["run_fingerprint"] == expected.metadata["run_fingerprint"]
    assert receipt.metadata["batching"]["input_storage"] == "disk-spool"
    assert receipt.metadata["storage_format"] == "tap-sink"
    assert received == sequences and len(identities) == 3
    assert all(ref() is None for ref in previous)


def test_sink_failure_stops_delivery_and_restores_training_mode():
    model = tiny_model().train()
    calls = []
    def fail(records, identity):
        calls.append(len(records))
        raise RuntimeError("destination failure")
    with pytest.raises(RuntimeError, match="destination failure"):
        embed_dataset(model, ["AC"] * 9, taps=[HiddenTap("mean", -1, "mean")],
                      tap_sink=fail, batch_size=2, batch_window_size=2)
    assert calls == [2] and model.training


@pytest.mark.parametrize("sink,taps", [(object(), [HiddenTap("x", -1)]), (lambda *_: None, None)])
def test_invalid_sink_is_refused_before_inference(sink, taps):
    with pytest.raises(ValueError, match="tap_sink"):
        embed_dataset(tiny_model(), ["AC"], taps=taps, tap_sink=sink)


@pytest.mark.parametrize("layout", [DENSE, CSR, RAGGED])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_bounded_parts_roundtrip_exactly_with_shuffled_duplicate_requests(tmp_path, layout, dtype):
    spec = StoredFeature("feature", layout, width=7, dtype=dtype, positions=layout == CSR)
    store = FeatureStore.open(tmp_path, spec)
    sequences = ["A" * (i + 1) for i in range(17)]
    rows = []
    budget = 128 if layout == DENSE else 288
    for i in range(17):
        if layout == CSR:
            count = i % 4
            rows.append(SparseRow(torch.arange(count), torch.arange(count, dtype=dtype) + 0.25,
                                  torch.arange(count, dtype=torch.int64) * 10000))
        else:
            shape = (7,) if layout == DENSE else (i % 5, 7)
            rows.append(torch.full(shape, i + 0.25, dtype=dtype))
    with store.segment("bounded") as writer:
        parts = writer.append_bounded(sequences, rows, max_tensor_bytes=budget,
                                      row_metadata=[{"original": i} for i in range(17)])
    assert len(parts) > 1
    marker = json.loads((store.directory / "segments/bounded/run.json").read_text())
    for part in marker["parts"]:
        with safe_open(store.directory / f"segments/bounded/part-{part['part']:05d}.safetensors",
                       framework="pt", device="cpu") as tensor_file:
            payload = sum(tensor_file.get_tensor(name).numel() * tensor_file.get_tensor(name).element_size()
                          for name in list(tensor_file.keys()))
        assert payload == part["tensor_bytes"] <= budget
    order = [16, 0, 6, 16, 3]
    request = [sequences[i] for i in order]
    readonly = FeatureStore.read_only(store.directory)
    assert readonly.row_metadata(request) == [{"original": i} for i in order]
    actual = readonly.read_sparse(request) if layout == CSR else readonly.read(request)
    for value, i in zip(actual, order, strict=True):
        if layout == CSR:
            assert torch.equal(value.indices, rows[i].indices)
            assert torch.equal(value.positions, rows[i].positions)
            assert torch.equal(value.values, rows[i].values) and value.values.dtype == dtype
        else:
            assert torch.equal(value, rows[i]) and value.dtype == dtype


def test_streamed_store_writes_before_next_window_without_publishing_early(tmp_path, monkeypatch):
    model = tiny_model()
    sequences = ["AC" * (i + 1) for i in range(9)]
    feature = StoredFeature("residues", RAGGED, 16, torch.float32)
    original = TapExecutor.run_window
    window_sizes = []
    def observe(self, records, *, window_start):
        if window_start:
            assert list(tmp_path.glob("*/segments/*/part-*.safetensors"))
            assert not list(tmp_path.glob("*/segments/*/run.json"))
            assert len(FeatureStore.read_only(tmp_path / feature.key)) == 0
        window_sizes.append(len(records))
        return original(self, records, window_start=window_start)
    monkeypatch.setattr(TapExecutor, "run_window", observe)
    stores = embed_into_features(model, sequences, tmp_path, {"rows": feature},
                                 taps=[HiddenTap("rows", -1)], batch_size=2,
                                 batch_window_size=4, max_part_bytes=1400)
    assert window_sizes == [4, 4, 1]
    assert stores["rows"].rows == 9 and len(stores["rows"].parts) > 3
    monkeypatch.setattr(
        model, "_embed_taps", lambda *_a, **_k: pytest.fail("cached forward"),
    )
    assert embed_into_features(model, sequences, tmp_path, {"rows": feature},
                               taps=[HiddenTap("rows", -1)], max_part_bytes=1400) == {}


def test_oversized_row_leaves_no_commit_or_indexed_rows(tmp_path):
    store = FeatureStore.open(tmp_path, StoredFeature("huge", DENSE, 8, torch.float32))
    with (pytest.raises(ValueError, match="exceeds max_tensor_bytes"),
          store.segment("attempt") as writer):
        writer.append_bounded(["A"], [torch.ones(8)], max_tensor_bytes=31)
    assert len(store) == 0 and not list(store.directory.glob("segments/*/run.json"))


def test_bounded_append_cannot_hide_duplicates_across_part_boundaries(tmp_path):
    store = FeatureStore.open(tmp_path, StoredFeature("x", DENSE, 8, torch.float32))
    with pytest.raises(ValueError, match="repeats"), store.segment("attempt") as writer:
        writer.append_bounded(["A", "CC", "A"], [torch.ones(8)] * 3, max_tensor_bytes=32)
    assert len(store) == 0 and not list(store.directory.glob("segments/*/part-*"))


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_invalid_part_budget_is_refused_without_files(tmp_path, bad):
    with pytest.raises(ValueError, match="max_part_bytes"):
        embed_into_features(tiny_model(), ["A"], tmp_path, {}, taps=[], max_part_bytes=bad)
    assert not list(tmp_path.iterdir())


def test_read_only_has_no_write_paths_or_repair_side_effects(tmp_path):
    store = FeatureStore.open(tmp_path / "store # % ü", StoredFeature("x", DENSE, 2, torch.float32))
    with store.segment("seed") as writer:
        writer.append(["A", "CC"], [torch.tensor([1., 2.]), torch.tensor([3., 4.])])
    readonly = FeatureStore.read_only(store.directory)
    before = contents(tmp_path)
    for operation in (readonly.reindex, readonly.sweep, readonly._ensure_index,
                      readonly._index_uncounted_segments):
        with pytest.raises(PermissionError, match="read-only"):
            operation()
    with pytest.raises(PermissionError, match="read-only"), readonly.segment("forbidden"):
        pytest.fail("read-only writer was entered")
    with pytest.raises(PermissionError, match="read-only"):
        SegmentWriter(readonly, store.directory / "forbidden", "forbidden", {})
    with (readonly._connect() as connection,
          pytest.raises(sqlite3.OperationalError, match="readonly")):
        connection.execute("DELETE FROM rows")
    copied = pickle.loads(pickle.dumps(readonly))
    assert read_worker(copied) == ([3., 4.], 2)
    with multiprocessing.get_context("spawn").Pool(2) as pool:
        assert pool.map(read_worker, [readonly, readonly]) == [([3., 4.], 2)] * 2
    assert contents(tmp_path) == before


def test_read_only_succeeds_without_write_permissions(tmp_path):
    store = FeatureStore.open(tmp_path, StoredFeature("x", DENSE, 2, torch.float32))
    with store.segment("seed") as writer:
        writer.append(["A"], [torch.tensor([1., 2.])])
    paths = [store.directory, *store.directory.rglob("*")]
    modes = {path: path.stat().st_mode for path in paths}
    before = contents(tmp_path)
    try:
        for path in paths:
            path.chmod(0o555 if path.is_dir() else 0o444)
        readonly = FeatureStore.read_only(store.directory)
        assert len(readonly) == 1 and readonly.read(["A"])[0].tolist() == [1., 2.]
    finally:
        for path, mode in modes.items():
            path.chmod(mode)
    assert contents(tmp_path) == before


def test_missing_or_corrupt_read_only_index_never_becomes_an_empty_cache(tmp_path):
    spec = StoredFeature("x", DENSE, 2, torch.float32)
    directory = tmp_path / spec.key
    directory.mkdir()
    (directory / "feature.json").write_text(json.dumps(spec.payload()))
    readonly = FeatureStore.read_only(directory)
    before = contents(tmp_path)
    with pytest.raises(sqlite3.OperationalError):
        readonly.missing(["A"])
    assert contents(tmp_path) == before and not (directory / "index.sqlite").exists()
    (directory / "index.sqlite").write_bytes(b"not a database")
    with pytest.raises(sqlite3.DatabaseError):
        len(readonly)
    assert (directory / "index.sqlite").read_bytes() == b"not a database"


@pytest.mark.parametrize("indices,positions", [([1, 1], [0, 1]), ([1, 2], [-1, 0]),
                                              ([1, 2], [0, 32768]), ([1., 2.], [0, 1])])
def test_sparse_integer_encoding_never_wraps_truncates_or_loses_duplicates(
    tmp_path, indices, positions,
):
    store = FeatureStore.open(tmp_path, StoredFeature("x", CSR, 8, torch.float32, positions=True))
    with pytest.raises(ValueError), store.segment("bad") as writer:
        row = SparseRow(torch.tensor(indices), torch.ones(2), torch.tensor(positions))
        writer.append(["A"], [row])
    assert not store.segments() and len(store) == 0
