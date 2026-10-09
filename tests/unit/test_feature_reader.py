"""Random access preserves verified store rows and rejects invalid inputs."""

from __future__ import annotations

import json
import os
import pickle
import threading
import pytest
import torch

from concurrent.futures import ThreadPoolExecutor

from fastplms.features import (
    CSR,
    DENSE,
    RAGGED,
    RAGGED_TOPK,
    FeatureReader,
    FeatureStore,
    SparseRow,
    StoredFeature,
    TopKRow,
)
from fastplms.features.store import COMMIT_FILE, PART_TEMPLATE, SEGMENTS_DIRECTORY


SEQUENCES = ("MKTFFVAVLALALATA", "MAAAAGGGKKLL", "MWWWCCYY", "MPPPEEE", "MQQRRSS")


def make_store(tmp_path, layout: str, **overrides: object) -> tuple[FeatureStore, list]:
    """A store of two segments, the second holding two parts, so a reader crosses every boundary."""

    if layout == DENSE:
        spec = StoredFeature("reader-dense", DENSE, 4, torch.float32)
        rows = [torch.arange(4, dtype=torch.float32) + 10 * i for i in range(len(SEQUENCES))]
    elif layout == RAGGED:
        spec = StoredFeature("reader-ragged", RAGGED, 3, torch.bfloat16)
        rows = [
            torch.full((2 + i, 3), float(i), dtype=torch.bfloat16)
            for i in range(len(SEQUENCES))
        ]
    elif layout == CSR:
        spec = StoredFeature(
            "reader-csr", CSR, 16, torch.float16, positions=bool(overrides.get("positions", True)),
        )
        rows = [
            SparseRow(
                indices=torch.tensor([i, i + 5], dtype=torch.int32),
                values=torch.tensor([1.5 + i, 2.5], dtype=torch.float16),
                positions=torch.tensor([i, 2], dtype=torch.int16) if spec.positions else None,
            )
            for i in range(len(SEQUENCES))
        ]
    else:
        spec = StoredFeature("reader-topk", RAGGED_TOPK, 32, torch.float32, sparse_count=2)
        rows = [
            TopKRow(
                torch.tensor([[i, i + 1]] * (1 + i), dtype=torch.int32),
                torch.full((1 + i, 2), 0.5 + i),
            )
            for i in range(len(SEQUENCES))
        ]
    store = FeatureStore.open(tmp_path, spec)
    with store.segment("first") as writer:
        writer.append(SEQUENCES[:2], rows[:2])
    with store.segment("second") as writer:
        writer.append(SEQUENCES[2:4], rows[2:4])
        writer.append(SEQUENCES[4:], rows[4:])
    return store, rows


def same(left, right) -> bool:
    if isinstance(left, SparseRow):
        return (
            torch.equal(left.indices, right.indices) and torch.equal(left.values, right.values)
            and (left.positions is None) == (right.positions is None)
            and (left.positions is None or torch.equal(left.positions, right.positions))
        )
    if isinstance(left, TopKRow):
        return torch.equal(left.indices, right.indices) and torch.equal(left.values, right.values)
    return left.dtype == right.dtype and torch.equal(left, right)


@pytest.mark.parametrize("layout", [DENSE, RAGGED])
def test_reading_matches_the_store_for_tensor_layouts(tmp_path, layout) -> None:
    store, rows = make_store(tmp_path, layout)
    order = [SEQUENCES[3], SEQUENCES[0], SEQUENCES[4], SEQUENCES[0]]
    with FeatureReader(store) as reader:
        got = reader.read(order)
    for sequence, row in zip(order, got, strict=True):
        assert same(row, rows[SEQUENCES.index(sequence)])
    assert all(same(a, b) for a, b in zip(got, store.read(order), strict=True))


@pytest.mark.parametrize("positions", [True, False])
def test_reading_matches_the_store_for_csr(tmp_path, positions) -> None:
    store, rows = make_store(tmp_path, CSR, positions=positions)
    with FeatureReader(store) as reader:
        sparse = reader.read_sparse(list(SEQUENCES))
        dense = reader.read(list(SEQUENCES))
    assert all(same(a, b) for a, b in zip(sparse, rows, strict=True))
    assert all(same(a, b) for a, b in zip(sparse, store.read_sparse(list(SEQUENCES)), strict=True))
    assert all(same(a, b) for a, b in zip(dense, store.read(list(SEQUENCES)), strict=True))


@pytest.mark.parametrize("layout", [DENSE, RAGGED, CSR])
def test_returned_duplicate_rows_cannot_mutate_each_other_or_future_reads(tmp_path, layout) -> None:
    store, rows = make_store(tmp_path, layout)
    selected = [SEQUENCES[0], SEQUENCES[0]]
    with FeatureReader(store) as reader:
        actual = reader.read_sparse(selected) if layout == CSR else reader.read(selected)
        if layout == CSR:
            actual[0].values.fill_(999)
            actual[0].indices.fill_(9)
            actual[0].positions.fill_(9)
            assert same(actual[1], rows[0])
            assert same(reader.read_sparse(selected[:1])[0], rows[0])
        else:
            actual[0].fill_(999)
            assert same(actual[1], rows[0])
            assert same(reader.read(selected[:1])[0], rows[0])
    reread = store.read_sparse(selected[:1]) if layout == CSR else store.read(selected[:1])
    assert same(reread[0], rows[0])


@pytest.mark.parametrize("positions", [True, False])
def test_read_csr_gathers_rows_in_the_order_asked(tmp_path, positions) -> None:
    store, rows = make_store(tmp_path, CSR, positions=positions)
    order = [SEQUENCES[3], SEQUENCES[0], SEQUENCES[4], SEQUENCES[0]]

    with FeatureReader(store) as reader:
        block = reader.read_csr(order)

    assert block.indptr.dtype == torch.int64 and block.indptr.tolist() == [0, 2, 4, 6, 8]
    assert block.indices.dtype == torch.int32 and block.values.dtype == torch.float16
    expected = [rows[SEQUENCES.index(sequence)] for sequence in order]
    assert torch.equal(block.indices, torch.cat([row.indices for row in expected]))
    assert torch.equal(block.values, torch.cat([row.values for row in expected]))
    if positions:
        assert block.positions.dtype == torch.int16
        assert torch.equal(block.positions, torch.cat([row.positions for row in expected]))
    else:
        assert block.positions is None


def test_read_csr_of_nothing_and_of_empty_rows(tmp_path) -> None:
    spec = StoredFeature("reader-empty", CSR, 8, torch.float32)
    store = FeatureStore.open(tmp_path, spec)
    empty = SparseRow(torch.zeros(0, dtype=torch.int32), torch.zeros(0, dtype=torch.float32), None)
    full = SparseRow(torch.tensor([3], dtype=torch.int32), torch.tensor([2.0]), None)
    with store.segment("rows") as writer:
        writer.append(SEQUENCES[:3], [empty, full, empty])

    with FeatureReader(store) as reader:
        none = reader.read_csr([])
        block = reader.read_csr(list(SEQUENCES[:3]))

    assert none.indptr.tolist() == [0] and none.indices.numel() == 0
    assert none.values.dtype == torch.float32
    assert block.indptr.tolist() == [0, 0, 1, 1]
    assert block.indices.tolist() == [3] and block.values.tolist() == [2.0]


def test_read_csr_refuses_a_feature_that_is_not_csr(tmp_path) -> None:
    store, _ = make_store(tmp_path, DENSE)
    with FeatureReader(store) as reader, pytest.raises(ValueError, match="read_sparse needs a csr"):
        reader.read_csr([SEQUENCES[0]])


def test_reading_matches_the_store_for_topk_and_refuses_dense_read(tmp_path) -> None:
    store, rows = make_store(tmp_path, RAGGED_TOPK)
    with FeatureReader(store) as reader:
        got = reader.read_topk(list(SEQUENCES))
        with pytest.raises(ValueError, match="read_topk"):
            reader.read(list(SEQUENCES))
    assert all(same(a, b) for a, b in zip(got, rows, strict=True))


def test_wrong_reader_for_the_layout_is_refused(tmp_path) -> None:
    store, _ = make_store(tmp_path, DENSE)
    with FeatureReader(store) as reader:
        with pytest.raises(ValueError, match="read_sparse needs a csr"):
            reader.read_sparse([SEQUENCES[0]])
        with pytest.raises(ValueError, match="read_topk needs a ragged_topk"):
            reader.read_topk([SEQUENCES[0]])


def test_membership_and_counts(tmp_path) -> None:
    store, _ = make_store(tmp_path, RAGGED)
    with FeatureReader.open(store.directory) as reader:
        assert len(reader) == len(SEQUENCES)
        assert SEQUENCES[0] in reader and "MNOPE" not in reader
        assert reader.missing(["MNOPE", SEQUENCES[1], "MNOPE", "MZZZ"]) == ("MNOPE", "MZZZ")
        assert reader.residue_counts([SEQUENCES[2], SEQUENCES[0]]) == [4, 2]


def test_a_sequence_the_feature_lacks_raises(tmp_path) -> None:
    store, _ = make_store(tmp_path, DENSE)
    with FeatureReader(store) as reader, pytest.raises(KeyError, match="no row"):
        reader.read(["MNOPE"])


def test_a_part_changed_after_commit_is_refused_on_first_touch(tmp_path) -> None:
    store, _ = make_store(tmp_path, DENSE)
    part = store.directory / SEGMENTS_DIRECTORY / "first" / PART_TEMPLATE.format(0)
    part_bytes = bytearray(part.read_bytes())
    part_bytes[-1] ^= 0x01
    part.write_bytes(bytes(part_bytes))
    with FeatureReader.open(store.directory) as reader:
        assert reader.read([SEQUENCES[2]])  # the second segment is untouched
        with pytest.raises(ValueError, match="digest does not match"):
            reader.read([SEQUENCES[0]])


def test_verify_pays_for_verification_up_front(tmp_path) -> None:
    store, _ = make_store(tmp_path, DENSE)
    with FeatureReader.open(store.directory) as reader:
        assert reader.verify() == 3
        assert reader.verify([SEQUENCES[0], SEQUENCES[1]]) == 1
    part = store.directory / SEGMENTS_DIRECTORY / "second" / PART_TEMPLATE.format(1)
    part.write_bytes(part.read_bytes()[:-1] + b"\x00")
    with FeatureReader.open(store.directory) as reader, pytest.raises(
        ValueError, match="digest does not match",
    ):
        reader.verify()


def test_pins_are_honoured(tmp_path) -> None:
    store, rows = make_store(tmp_path, DENSE)
    pins = store.content_pins([SEQUENCES[0]])
    with FeatureReader.open(store.directory, content_pins=pins) as reader:
        assert same(reader.read([SEQUENCES[0]])[0], rows[0])
        with pytest.raises(ValueError, match="outside the pinned"):
            reader.read([SEQUENCES[3]])


def test_a_reader_pickles_as_the_store_it_reads(tmp_path) -> None:
    store, rows = make_store(tmp_path, DENSE)
    with FeatureReader.open(store.directory) as reader:
        reader.read([SEQUENCES[0]])
        clone = pickle.loads(pickle.dumps(reader))
    with clone:
        assert same(clone.read([SEQUENCES[1]])[0], rows[1])


def test_threads_read_through_their_own_connections(tmp_path) -> None:
    store, rows = make_store(tmp_path, DENSE)
    ready = threading.Barrier(len(SEQUENCES))
    with FeatureReader.open(store.directory) as reader:
        def work(position: int) -> None:
            ready.wait(timeout=10)
            for _ in range(20):
                assert same(reader.read([SEQUENCES[position]])[0], rows[position])

        with ThreadPoolExecutor(max_workers=len(SEQUENCES)) as pool:
            list(pool.map(work, range(len(SEQUENCES))))
        assert len({key[1] for key in reader._connections}) == len(SEQUENCES)


def counted_verification(reader: FeatureReader, monkeypatch) -> list[tuple[str, int]]:
    """Record each part the store verifies for `reader`, as (segment fingerprint, part number)."""

    verified: list[tuple[str, int]] = []
    original = reader.store._verified_part

    def record(fingerprint, part):
        verified.append((fingerprint, part["part"]))
        return original(fingerprint, part)

    monkeypatch.setattr(reader.store, "_verified_part", record)
    return verified


@pytest.mark.parametrize("workers", [2, 8])
def test_parallel_verification_checks_each_part_once(tmp_path, monkeypatch, workers) -> None:
    store, rows = make_store(tmp_path, DENSE)
    with FeatureReader.open(store.directory) as reader:
        verified = counted_verification(reader, monkeypatch)
        assert reader.verify(workers=workers) == 3
        assert len(verified) == 3 and len(set(verified)) == 3
        assert reader.verify(workers=workers) == 3 and len(verified) == 3
        assert all(
            same(reader.read([sequence])[0], row)
            for sequence, row in zip(SEQUENCES, rows, strict=True)
        )
        assert len(verified) == 3


def test_threads_that_meet_on_one_unverified_part_verify_it_once(tmp_path, monkeypatch) -> None:
    store, rows = make_store(tmp_path, DENSE)
    with FeatureReader.open(store.directory) as reader:
        verified = counted_verification(reader, monkeypatch)
        barrier = threading.Barrier(8)
        results: list[bool] = []

        def work() -> None:
            barrier.wait()
            results.append(same(reader.read([SEQUENCES[0]])[0], rows[0]))

        threads = [threading.Thread(target=work) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert results == [True] * 8 and len(verified) == 1


def test_parallel_verification_raises_a_corrupt_part_and_needs_a_worker(tmp_path) -> None:
    store, _ = make_store(tmp_path, DENSE)
    part = store.directory / SEGMENTS_DIRECTORY / "second" / PART_TEMPLATE.format(1)
    part.write_bytes(part.read_bytes()[:-1] + b"\x00")
    with FeatureReader.open(store.directory) as reader:
        with pytest.raises(ValueError, match="digest does not match"):
            reader.verify(workers=4)
        with pytest.raises(ValueError, match="at least one worker"):
            reader.verify(workers=0)


def test_a_closed_reader_refuses_reads(tmp_path) -> None:
    store, _ = make_store(tmp_path, DENSE)
    reader = FeatureReader.open(store.directory)
    reader.close()
    with pytest.raises(ValueError, match="closed"):
        reader.read([SEQUENCES[0]])


def test_addresses_locate_rows_without_verifying_a_part(tmp_path, monkeypatch) -> None:
    store, _ = make_store(tmp_path, RAGGED)
    with FeatureReader.open(store.directory) as reader:
        verified = counted_verification(reader, monkeypatch)
        found = reader.addresses([SEQUENCES[4], SEQUENCES[0], SEQUENCES[4]])
        assert [address.residues for address in found] == [2 + 4, 2, 2 + 4]
        assert found[0] == found[2] and verified == []
        with pytest.raises(KeyError, match="no row"):
            reader.addresses(["MNOPE"])


@pytest.mark.parametrize("layout", [DENSE, RAGGED, CSR, RAGGED_TOPK])
def test_a_receipt_spares_the_next_reader_the_verification(tmp_path, monkeypatch, layout) -> None:
    store, rows = make_store(tmp_path, layout)
    receipt = tmp_path / "receipts" / "feature.json"
    with FeatureReader(store, receipt=receipt) as reader:
        verified = counted_verification(reader, monkeypatch)
        assert reader.verify() == 3 and len(verified) == 3
    assert receipt.is_file()
    with FeatureReader(store, receipt=receipt) as trusting:
        verified = counted_verification(trusting, monkeypatch)
        assert trusting.verify() == 3 and verified == []
        readers = {RAGGED_TOPK: trusting.read_topk, CSR: trusting.read_sparse}
        read = readers.get(layout, trusting.read)
        got = read(SEQUENCES)
        assert all(same(one, row) for one, row in zip(got, rows, strict=True)) and verified == []
    monkeypatch.undo()  # a store with a local function patched in cannot pickle
    with pickle.loads(pickle.dumps(FeatureReader(store, receipt=receipt))) as clone:
        verified = counted_verification(clone, monkeypatch)  # a spawned worker keeps the receipt
        assert clone.verify() == 3 and verified == []


def test_a_receipt_never_vouches_for_a_changed_part_or_another_descriptor(tmp_path) -> None:
    store, _ = make_store(tmp_path, RAGGED)
    receipt = tmp_path / "receipt.json"
    with FeatureReader(store, receipt=receipt) as reader:
        reader.verify()
    other = StoredFeature("reader-ragged", RAGGED, 3, torch.float32)  # same receipt path
    elsewhere = FeatureReader(FeatureStore.open(tmp_path / "other", other), receipt=receipt)
    assert elsewhere._receipt._trusted == {}
    part = store.directory / SEGMENTS_DIRECTORY / "first" / PART_TEMPLATE.format(0)
    original = part.read_bytes()
    part.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))  # same size, new modification time
    with FeatureReader(store, receipt=receipt) as reader, pytest.raises(ValueError, match="digest"):
        reader.verify()


def test_a_deep_reader_catches_a_rewrite_the_receipt_cannot_see(tmp_path) -> None:
    store, _ = make_store(tmp_path, DENSE)
    receipt = tmp_path / "receipt.json"
    with FeatureReader(store, receipt=receipt) as reader:
        reader.verify()
    part = store.directory / SEGMENTS_DIRECTORY / "first" / PART_TEMPLATE.format(0)
    stamp = part.stat()
    part_bytes = bytearray(part.read_bytes())
    part_bytes[-1] ^= 0x01
    part.write_bytes(bytes(part_bytes))
    os.utime(part, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))  # same size, same time
    with FeatureReader(store, receipt=receipt, trust_receipt=False) as deep, pytest.raises(
        ValueError, match="digest",
    ):
        deep.verify()


def test_a_pinned_store_never_trusts_a_receipt(tmp_path, monkeypatch) -> None:
    store, _ = make_store(tmp_path, DENSE)
    receipt = tmp_path / "receipt.json"
    with FeatureReader(store, receipt=receipt) as reader:
        reader.verify()
    pins = store.content_pins(SEQUENCES)
    pinned_store = FeatureStore.read_only(store.directory, content_pins=pins)
    with FeatureReader(pinned_store, receipt=receipt) as pinned:
        assert pinned._receipt is None
        verified = counted_verification(pinned, monkeypatch)
        assert pinned.verify() == 3 and len(verified) == 3


def test_an_unwritable_receipt_location_warns_and_the_reader_still_verifies(tmp_path) -> None:
    store, rows = make_store(tmp_path, DENSE)
    blocker = tmp_path / "blocked"
    blocker.write_text("a file where the receipt directory should be")
    with pytest.warns(RuntimeWarning) as captured, FeatureReader(
        store, receipt=blocker / "feature.json",
    ) as reader:
        assert reader.verify() == 3
        assert same(reader.read([SEQUENCES[0]])[0], rows[0])
    assert any("Could not save" in str(warning.message) for warning in captured)


@pytest.mark.parametrize("layout", [DENSE, RAGGED, CSR, RAGGED_TOPK])
@pytest.mark.parametrize("workers", [1, 4])
def test_pinning_reuses_verified_layouts_and_matches_strict_store_pins(tmp_path, monkeypatch, layout, workers) -> None:
    store, _ = make_store(tmp_path, layout)
    expected = store.content_pins(SEQUENCES)
    with FeatureReader(store) as reader:
        reader.verify(SEQUENCES)
        repeated = counted_verification(reader, monkeypatch)
        assert reader.content_pins(SEQUENCES, workers=workers) == expected
        assert repeated == []


@pytest.mark.parametrize("workers", [1, 4])
def test_pinning_rejects_a_same_size_same_time_rewrite_after_verification(tmp_path, workers) -> None:
    store, _ = make_store(tmp_path, DENSE)
    with FeatureReader(store) as reader:
        reader.verify(SEQUENCES)
        part = store.directory / SEGMENTS_DIRECTORY / "first" / PART_TEMPLATE.format(0)
        stamp = part.stat()
        part_bytes = bytearray(part.read_bytes())
        part_bytes[-1] ^= 1
        part.write_bytes(part_bytes)
        os.utime(part, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        with pytest.raises(ValueError, match="bytes changed"):
            reader.content_pins(SEQUENCES, workers=workers)


def test_pinning_rejects_a_marker_changed_after_verification(tmp_path) -> None:
    store, _ = make_store(tmp_path, DENSE)
    with FeatureReader(store) as reader:
        reader.verify(SEQUENCES)
        marker = store.directory / SEGMENTS_DIRECTORY / "first" / COMMIT_FILE
        payload = json.loads(marker.read_text())
        payload["metadata"]["changed"] = True
        marker.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="commit marker changed"):
            reader.content_pins(SEQUENCES)


def test_pinning_an_old_selection_is_stable_after_an_unrelated_append(tmp_path) -> None:
    store, _ = make_store(tmp_path, DENSE)
    with FeatureReader(store) as reader:
        before = reader.content_pins(SEQUENCES, workers=4)
        with store.segment("unrelated") as writer:
            writer.append(["MNEW"], [torch.zeros(4)])
        assert reader.content_pins(SEQUENCES, workers=4) == before
