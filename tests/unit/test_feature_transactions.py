"""Real process death, concurrent writers, and independently checked segment recovery."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import shutil
import pytest
import torch

from pathlib import Path
from safetensors.torch import save_file

from fastplms.embeddings import HiddenTap, embed_into_features
from fastplms.features import (
    CSR, DENSE, RAGGED, FeatureStore, SparseRow, StoredFeature, partition_sequences,
)
from fastplms.features import store as storage
from fastplms.models.esm_plusplus.modeling_esm_plusplus import ESMplusplusConfig, ESMplusplusModel


SEQUENCES = ("AC", "DDD", "EEEE")
BOUNDARIES = (
    "before_part_write", "after_part_write", "after_metadata_write", "before_index_update",
    "after_index_update", "before_commit_marker", "after_commit_marker", "after_index_commit",
)


def spec(layout=DENSE, width=4):
    return StoredFeature("test_feature", layout, width, torch.float32, positions=layout == CSR)


def row_values(sequences, layout, offset=0):
    if layout == DENSE:
        return [torch.arange(4).float() + len(s) + offset for s in sequences]
    if layout == RAGGED:
        return [torch.arange(len(s) * 4).reshape(len(s), 4).float() + offset for s in sequences]
    return [SparseRow(torch.tensor([1, 3]), torch.tensor([len(s) + offset, 2.0]),
                      torch.tensor([0, 1])) for s in sequences]


class IdentityPanel:
    """Opaque software-fixture sidecars; scientific contracts have their own regression lane."""

    def validate(self, *args):
        pass

    def validate_cached(self, name, store, sequences):
        if sequences:
            store.row_metadata(sequences)

    def bind_rows(self, name, records):
        return [{"sequence_sha256": storage.sequence_digest(record.sequence)} for record in records]


def write_process(root, mode, layout, boundary, output):
    """Exit after observed I/O, without Python context cleanup or lock release handlers."""
    torch.set_num_threads(1)
    seen = 0
    original = storage._transaction_event
    def observe(stage, path):
        nonlocal seen
        original(stage, path)
        if stage != boundary:
            return
        seen += 1
        # For part boundaries, kill during the second write, never before any data exists.
        if stage in BOUNDARIES[:3] and seen != 2:
            return
        files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in path.iterdir() if p.is_file()}
        (Path(root) / "interruption.json").write_text(json.dumps({
            "boundary": stage, "occurrence": seen, "segment": str(path), "files": files,
            "marker_exists": (path / "run.json").exists(), "pid": os.getpid(),
        }))
        os._exit(73)
    storage._transaction_event = observe
    if mode == "streamed":
        torch.manual_seed(17)
        model = ESMplusplusModel(ESMplusplusConfig(
            hidden_size=16, num_attention_heads=2, num_hidden_layers=2, attn_backend="sdpa",
        )).eval()
        taps = [HiddenTap("mean", -1, "mean")]
        calls = []
        handle = model.transformer.register_forward_hook(lambda *_: calls.append(1))
        embed_into_features(
            model, SEQUENCES, root, {"mean": spec(width=16)}, taps=taps, contract=IdentityPanel(),
            batch_size=1, batch_window_size=1, max_part_bytes=64,
        )
        handle.remove()
    else:
        store = FeatureStore.open(root, spec(layout))
        if store.missing(SEQUENCES):
            with store.segment("attempt", {"panel": "process_death"}) as writer:
                rows = row_values(SEQUENCES, layout)
                identities = [{"source": s} for s in SEQUENCES]
                if mode == "bounded":
                    writer.append_bounded(SEQUENCES, rows, max_tensor_bytes=100,
                                          row_metadata=identities)
                else:
                    for sequence, row, identity in zip(SEQUENCES, rows, identities, strict=True):
                        writer.append([sequence], [row], row_metadata=[identity])
        calls = []
    output.put({"status": "passed", "encoder_calls": len(calls)})


def read_process(root, layout, width, output):
    store = FeatureStore.open(root, spec(layout, width))
    present = [sequence for sequence in SEQUENCES if store.address(sequence) is not None]
    output.put({"rows": len(store), "sequences": present,
                "values": [row.tolist() for row in store.read(present)],
                "metadata": store.row_metadata(present) if present else []})


def invoke(target, *args, exitcode=0):
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    process = context.Process(target=target, args=(*args, output))
    process.start()
    process.join(60)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("Feature-store child exceeded 60 seconds.")
    assert process.exitcode == exitcode
    returned = output.get(timeout=5) if exitcode == 0 else None
    output.close()
    return returned


@pytest.mark.parametrize("boundary", BOUNDARIES)
@pytest.mark.parametrize("mode,layout", [
    ("direct", DENSE), ("bounded", RAGGED), ("streamed", DENSE),
])
def test_real_process_death_recovers_exact_clean_rows(tmp_path, mode, layout, boundary):
    crashed, clean = tmp_path / "crashed", tmp_path / "clean"
    invoke(write_process, crashed, mode, layout, boundary, exitcode=73)
    observed = json.loads((crashed / "interruption.json").read_text())
    assert any(name.endswith(".safetensors") for name in observed["files"])
    committed = boundary in ("after_commit_marker", "after_index_commit")
    assert observed["marker_exists"] is committed
    width = 16 if mode == "streamed" else 4
    recovered = invoke(read_process, crashed, layout, width)
    assert recovered["rows"] == (3 if committed else 0)
    resumed = invoke(write_process, crashed, mode, layout, None)
    invoke(write_process, clean, mode, layout, None)
    actual = invoke(read_process, crashed, layout, width)
    expected = invoke(read_process, clean, layout, width)
    assert actual == expected and actual["rows"] == 3
    if mode == "streamed":
        assert resumed["encoder_calls"] == (0 if committed else 3)
    (tmp_path / "recovery.json").write_text(json.dumps({
        "interruption": observed, "recovered": recovered, "resumed": resumed,
        "actual": actual, "clean": expected,
    }, indent=2))


def competing_writer(root, fingerprint, sequences, ready, release, output):
    torch.set_num_threads(1)
    try:
        store = FeatureStore.open(root, spec())
        with store.segment(fingerprint) as writer:
            writer.append(sequences, row_values(sequences, DENSE))
            ready.put(fingerprint)
            if not release.wait(30):
                raise TimeoutError("Test did not release staged writer.")
        output.put({"writer": fingerprint, "status": "committed"})
    except (ValueError, BlockingIOError) as error:
        output.put({"writer": fingerprint, "status": type(error).__name__, "message": str(error)})


@pytest.mark.parametrize("overlap", [False, True])
def test_two_processes_stage_together_and_commit_without_replacement(tmp_path, overlap):
    context = multiprocessing.get_context("spawn")
    ready, output, release = context.Queue(), context.Queue(), context.Event()
    groups = [("AC", "DDD"), ("AC" if overlap else "GGG", "EEEE")]
    workers = [context.Process(target=competing_writer,
               args=(tmp_path, f"writer-{i}", sequences, ready, release, output))
               for i, sequences in enumerate(groups)]
    for worker in workers:
        worker.start()
    try:
        assert {ready.get(timeout=30) for _ in workers} == {"writer-0", "writer-1"}
        store = FeatureStore.open(tmp_path, spec())
        assert len(store) == 0 and store.sweep() == ()
        release.set()
        answers = [output.get(timeout=30) for _ in workers]
    finally:
        release.set()
        for worker in workers:
            worker.join(30)
            if worker.is_alive():
                worker.kill()
                worker.join()
            assert worker.exitcode == 0
    assert sum(a["status"] == "committed" for a in answers) == (1 if overlap else 2)
    if overlap:
        assert sum(a["status"] == "ValueError" for a in answers) == 1
    reopened = FeatureStore.open(tmp_path, spec())
    assert len(reopened) == (2 if overlap else 4)
    assert len(reopened.segments()) == (1 if overlap else 2)
    assert reopened.reindex() == len(reopened)
    (tmp_path / "concurrency.json").write_text(json.dumps(answers, indent=2))


def test_same_fingerprint_has_one_owner_and_sweep_skips_it(tmp_path):
    context = multiprocessing.get_context("spawn")
    ready, output, release = context.Queue(), context.Queue(), context.Event()
    worker = context.Process(target=competing_writer,
                             args=(tmp_path, "same", SEQUENCES, ready, release, output))
    worker.start()
    try:
        assert ready.get(timeout=30) == "same"
        store = FeatureStore.open(tmp_path, spec())
        assert store.sweep() == ()
        with pytest.raises(BlockingIOError, match="active writer"), store.segment("same"):
            pass
    finally:
        release.set()
        worker.join(30)
        if worker.is_alive():
            worker.kill()
            worker.join()
    assert worker.exitcode == 0 and output.get(timeout=5)["status"] == "committed"


def use_transferred_writer(writer, output):
    errors = []
    for operation in (writer.commit, writer.abandon,
                      lambda: writer.append(["DDD"], row_values(["DDD"], DENSE))):
        try:
            operation()
        except RuntimeError as error:
            errors.append(str(error))
    output.put(errors)


def test_a_spawned_copy_of_a_writer_cannot_write_commit_or_abandon(tmp_path):
    store = FeatureStore.open(tmp_path, spec())
    with store.segment("owner") as writer:
        writer.append(["AC"], row_values(["AC"], DENSE))
        errors = invoke(use_transferred_writer, writer)
        assert len(errors) == 3 and all("process that opened" in error for error in errors)
        writer.append(["DDD"], row_values(["DDD"], DENSE))
    assert len(store) == 2


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX descriptor inheritance only")
def test_forked_context_cleanup_does_not_unlock_the_parent_writer(tmp_path):
    store = FeatureStore.open(tmp_path, spec())
    context = store.segment("owner")
    writer = context.__enter__()
    try:
        writer.append(["AC"], row_values(["AC"], DENSE))
        child = os.fork()
        if child == 0:
            try:
                error = RuntimeError("child discards inherited context")
                context.__exit__(RuntimeError, error, None)
            except BaseException:  # noqa: broad-except  a forked child leaves with a status code on any failure
                os._exit(74)
            os._exit(0)
        _, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        with pytest.raises(BlockingIOError), store.segment("owner"):
            pass
        writer.append(["DDD"], row_values(["DDD"], DENSE))
    finally:
        context.__exit__(None, None, None)
    assert len(store) == 2


def committed_store(root, layout=DENSE, fingerprint="seed", sequences=SEQUENCES):
    store = FeatureStore.open(root, spec(layout))
    with store.segment(fingerprint) as writer:
        writer.append(sequences, row_values(sequences, layout),
                      row_metadata=[{"source": sequence} for sequence in sequences])
    return store


@pytest.mark.parametrize("damage", [
    "missing", "data", "sidecar", "marker", "count", "address", "schema",
])
def test_corrupt_segment_never_reads_or_enters_a_rebuilt_index(tmp_path, damage):
    store = committed_store(tmp_path)
    segment = store.directory / "segments/seed"
    marker = segment / "run.json"
    payload = json.loads(marker.read_text())
    if damage == "missing":
        (segment / "part-00000.safetensors").unlink()
    elif damage == "data":
        path = segment / "part-00000.safetensors"
        path.write_bytes(path.read_bytes()[:-1] + b"x")
    elif damage == "sidecar":
        (segment / "part-00000.rows.json.gz").write_bytes(b"broken")
    elif damage == "marker":
        marker.write_text("{")
    elif damage == "schema":
        payload.pop("transaction_schema")
        marker.write_text(json.dumps(payload))
    else:
        payload["rows" if damage == "count" else "fingerprint"] = 100
        marker.write_text(json.dumps(payload))
    before = (store.directory / "index.sqlite").read_bytes()
    for action in (lambda: store.read(SEQUENCES), store.reindex,
                   lambda: FeatureStore.open(tmp_path, spec())):
        with pytest.raises((ValueError, FileNotFoundError)):
            action()
    assert (store.directory / "index.sqlite").read_bytes() == before


@pytest.mark.parametrize("layout", [DENSE, CSR, RAGGED])
def test_recovery_checks_shapes_even_when_new_checksums_are_consistent(tmp_path, layout):
    store = committed_store(tmp_path, layout)
    segment = store.directory / "segments/seed"
    part_path, marker = segment / "part-00000.safetensors", segment / "run.json"
    tensors = store._load_part("seed", 0)
    tensors["values"] = tensors["values"].to(torch.float64)
    save_file(tensors, part_path)
    payload = json.loads(marker.read_text())
    payload["parts"][0]["sha256"] = hashlib.sha256(part_path.read_bytes()).hexdigest()
    payload.pop("manifest_sha256")
    payload["manifest_sha256"] = storage.json_sha256(payload)
    marker.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="dtype"):
        store.reindex()


def test_partial_index_recovers_but_changed_addresses_require_explicit_reindex(tmp_path):
    store = committed_store(tmp_path)
    with store._connect() as connection:
        connection.execute("DELETE FROM rows WHERE row = 1")
        connection.commit()
    assert len(FeatureStore.open(tmp_path, spec())) == 3
    with store._connect() as connection:
        connection.execute("UPDATE rows SET row = 90 WHERE row = 1")
        connection.commit()
    with pytest.raises(ValueError, match="index disagrees"):
        FeatureStore.open(tmp_path, spec())
    assert store.reindex() == 3
    for actual, expected in zip(store.read(SEQUENCES), row_values(SEQUENCES, DENSE), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_corrupt_derived_database_is_rebuilt_from_verified_segments(tmp_path):
    store = committed_store(tmp_path)
    (store.directory / "index.sqlite").write_bytes(b"not a database")
    assert len(FeatureStore.open(tmp_path, spec())) == 3


@pytest.mark.parametrize("overlap", [False, True])
def test_imported_shards_accept_only_disjoint_verified_rows(tmp_path, overlap):
    first = committed_store(tmp_path / "first", fingerprint="first", sequences=("AC",))
    second = committed_store(tmp_path / "second", fingerprint="second",
                             sequences=("AC" if overlap else "DDD",))
    shutil.copytree(second.directory / "segments/second", first.directory / "segments/second")
    if overlap:
        with pytest.raises(ValueError, match="conflicting"):
            first.reindex()
    else:
        assert first.reindex() == 2
        assert first.read(["DDD"])[0].tolist() == [3., 4., 5., 6.]


def test_duplicates_across_appends_and_closed_writer_are_refused(tmp_path):
    store = FeatureStore.open(tmp_path, spec())
    with store.segment("attempt") as writer:
        writer.append(["AC"], row_values(["AC"], DENSE))
        with pytest.raises(ValueError, match="repeats"):
            writer.append(["AC"], row_values(["AC"], DENSE))
    with pytest.raises(RuntimeError, match="committed"):
        writer.append(["DDD"], row_values(["DDD"], DENSE))


def test_part_io_failure_cannot_commit_a_staged_prefix(tmp_path, monkeypatch):
    store = FeatureStore.open(tmp_path, spec())
    with pytest.raises(RuntimeError, match="closed or failed"), store.segment("attempt") as writer:
        writer.append(["AC"], row_values(["AC"], DENSE))
        def fail(stage, path):
            if stage == "after_part_write":
                raise OSError("destination failed")
        monkeypatch.setattr(storage, "_transaction_event", fail)
        with pytest.raises(OSError, match="destination failed"):
            writer.append(["DDD"], row_values(["DDD"], DENSE))
    assert len(store) == 0 and not list(store.directory.glob("segments/*/run.json"))
    assert store.sweep() == ("attempt",)


def test_interrupted_multifeature_extraction_reuses_the_committed_feature(tmp_path, monkeypatch):
    torch.manual_seed(23)
    model = ESMplusplusModel(ESMplusplusConfig(
        hidden_size=16, num_attention_heads=2, num_hidden_layers=2, attn_backend="sdpa",
    )).eval()
    features = {name: StoredFeature(name, DENSE, 16, torch.float32) for name in ("mean", "max")}
    taps = [HiddenTap(name, -1, name) for name in features]
    def fail(stage, path):
        if stage == "after_commit_marker":
            raise RuntimeError("interrupted between features")
    monkeypatch.setattr(storage, "_transaction_event", fail)
    with pytest.raises(RuntimeError, match="between features"):
        embed_into_features(model, SEQUENCES, tmp_path, features, taps=taps,
                            batch_size=1, batch_window_size=1)
    first = FeatureStore.open(tmp_path, features["mean"])
    second = FeatureStore.open(tmp_path, features["max"])
    assert len(first) == 3 and len(second) == 0
    parts_before = {str(p): p.read_bytes() for p in first.directory.glob("segments/*/*")}
    monkeypatch.setattr(storage, "_transaction_event", lambda *args: None)
    receipts = embed_into_features(model, SEQUENCES, tmp_path, features, taps=taps,
                                   batch_size=1, batch_window_size=1)
    assert set(receipts) == {"max"}
    assert parts_before == {str(p): p.read_bytes() for p in first.directory.glob("segments/*/*")}
    clean = tmp_path / "clean"
    embed_into_features(model, SEQUENCES, clean, features, taps=taps,
                        batch_size=1, batch_window_size=1)
    for feature in features.values():
        recovered = FeatureStore.read_only(tmp_path / feature.key)
        reference = FeatureStore.read_only(clean / feature.key)
        for actual, expected in zip(
            recovered.read(SEQUENCES), reference.read(SEQUENCES), strict=True,
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_deterministic_partition_deduplicates_and_covers_inventory():
    sequences = ["AC" * i for i in range(1, 80)]
    groups = [partition_sequences(sequences * 2, shard=i, shards=5) for i in range(5)]
    assert len({s for group in groups for s in group}) == sum(map(len, groups)) == len(sequences)
    for i, group in enumerate(groups):
        assert group == partition_sequences(sequences, shard=i, shards=5)
        assert set(group) == set(partition_sequences(reversed(sequences), shard=i, shards=5))
        assert all(int(hashlib.sha256(s.encode()).hexdigest(), 16) % 5 == i for s in group)
