"""The row encodings, locks, receipts and addresses of the feature store, one public name at a time."""

from __future__ import annotations

import dataclasses
import json
import pytest
import torch

from pathlib import Path

from fastplms.features import (
    CSR,
    DENSE,
    RAGGED,
    RAGGED_TOPK,
    ConversionReceipt,
    CsrRows,
    FeatureReader,
    FeatureStore,
    RowAddress,
    SegmentReceipt,
    SparseRow,
    StoredFeature,
    TopKRow,
    convert_rows,
    dtype_name,
    value_dtype,
)
from fastplms.features.layouts import (
    encode_csr,
    encode_dense,
    encode_ragged,
    encode_topk_rows,
    row_count,
    row_tensor_bytes,
    tensor_names,
    validate_topk,
)
from fastplms.features.receipts import SCHEMA, PartReceipt
from fastplms.features.store import COMMIT_FILE, PART_TEMPLATE, SEGMENTS_DIRECTORY
from fastplms.features.transactions import file_lock, publish_file, sync_directory


SEQUENCES = ("MKTAYI", "GGGS", "WWCCPP")


class TestValueDtypes:
    @pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
    def test_a_name_and_its_dtype_round_trip(self, dtype: torch.dtype) -> None:
        assert value_dtype(dtype_name(dtype)) is dtype

    def test_names_are_the_torch_names_without_the_prefix(self) -> None:
        assert dtype_name(torch.bfloat16) == "bfloat16"

    def test_a_dtype_no_layout_stores_is_refused(self) -> None:
        with pytest.raises(ValueError, match="float64, float32, float16, or bfloat16"):
            dtype_name(torch.int32)
        with pytest.raises(ValueError, match="Unknown feature value dtype 'int8'"):
            value_dtype("int8")


class TestSparseRow:
    def test_from_dense_keeps_exactly_the_non_zero_entries_and_their_positions(self) -> None:
        vector = torch.tensor([0.0, 2.5, 0.0, -1.0, 0.0])
        positions = torch.tensor([9, 7, 8, 6, 5], dtype=torch.int16)
        row = SparseRow.from_dense(vector, positions)
        assert row.indices.tolist() == [1, 3]
        assert row.indices.dtype == torch.int32
        assert row.values.tolist() == [2.5, -1.0]
        assert row.positions is not None and row.positions.tolist() == [7, 6]
        assert torch.equal(row.to_dense(5), vector)

    def test_a_vector_without_positions_gives_a_row_without_positions(self) -> None:
        assert SparseRow.from_dense(torch.tensor([0.0, 1.0])).positions is None

    def test_an_all_zero_vector_gives_an_empty_row(self) -> None:
        row = SparseRow.from_dense(torch.zeros(4))
        assert row.indices.numel() == 0 and row.to_dense(4).tolist() == [0.0] * 4

    def test_shapes_are_checked(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional vector"):
            SparseRow.from_dense(torch.zeros(2, 2))
        with pytest.raises(ValueError, match="positions must have the vector's shape"):
            SparseRow.from_dense(torch.zeros(3), torch.zeros(2, dtype=torch.int16))


class TestEncodeDense:
    def test_rows_stack_in_the_declared_dtype(self) -> None:
        tensors = encode_dense([torch.arange(3.0), torch.ones(3)], 3, torch.bfloat16)
        assert set(tensors) == {"values"}
        assert tensors["values"].dtype == torch.bfloat16
        assert tensors["values"].tolist() == [[0.0, 1.0, 2.0], [1.0, 1.0, 1.0]]

    def test_a_row_of_the_wrong_width_names_its_position(self) -> None:
        with pytest.raises(ValueError, match=r"must have shape \(3,\); row 1"):
            encode_dense([torch.zeros(3), torch.zeros(2)], 3, torch.float32)


class TestEncodeRagged:
    def test_offsets_name_each_span_of_the_concatenated_values(self) -> None:
        tensors = encode_ragged([torch.ones(2, 3), torch.zeros(1, 3), torch.full((3, 3), 2.0)], 3, torch.float16)
        assert tensors["offsets"].tolist() == [0, 2, 3, 6]
        assert tensors["offsets"].dtype == torch.int64
        assert tensors["values"].shape == (6, 3) and tensors["values"].dtype == torch.float16

    def test_no_rows_give_an_empty_block(self) -> None:
        tensors = encode_ragged([], 4, torch.float32)
        assert tensors["offsets"].tolist() == [0] and tensors["values"].shape == (0, 4)

    def test_a_block_of_the_wrong_width_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"\(r_i, 3\); row 0"):
            encode_ragged([torch.zeros(2, 4)], 3, torch.float32)


class TestEncodeCsr:
    def rows(self, positions: bool) -> list[SparseRow]:
        return [
            SparseRow(
                torch.tensor([1, 4], dtype=torch.int32),
                torch.tensor([0.5, 1.5]),
                torch.tensor([2, 3], dtype=torch.int16) if positions else None,
            ),
            SparseRow(
                torch.tensor([], dtype=torch.int32),
                torch.tensor([]),
                torch.tensor([], dtype=torch.int16) if positions else None,
            ),
            SparseRow(
                torch.tensor([0], dtype=torch.int64),
                torch.tensor([9.0]),
                torch.tensor([7], dtype=torch.int64) if positions else None,
            ),
        ]

    def test_indptr_names_the_span_of_each_row(self) -> None:
        tensors = encode_csr(self.rows(positions=True), 8, torch.float32)
        assert tensors["indptr"].tolist() == [0, 2, 2, 3]
        assert tensors["indices"].tolist() == [1, 4, 0] and tensors["indices"].dtype == torch.int32
        assert tensors["values"].tolist() == [0.5, 1.5, 9.0]
        assert tensors["positions"].tolist() == [2, 3, 7] and tensors["positions"].dtype == torch.int16

    def test_a_segment_without_positions_stores_none(self) -> None:
        assert "positions" not in encode_csr(self.rows(positions=False), 8, torch.float32)

    def test_no_rows_give_empty_typed_tensors(self) -> None:
        tensors = encode_csr([], 8, torch.float16)
        assert tensors["indptr"].tolist() == [0]
        assert tensors["indices"].dtype == torch.int32 and tensors["values"].dtype == torch.float16

    @pytest.mark.parametrize(
        ("row", "message"),
        [
            (SparseRow(torch.tensor([8]), torch.tensor([1.0]), None), "outside 0..7"),
            (SparseRow(torch.tensor([-1]), torch.tensor([1.0]), None), "outside 0..7"),
            (SparseRow(torch.tensor([1, 1]), torch.tensor([1.0, 2.0]), None), "unique within each row"),
            (SparseRow(torch.tensor([1, 2]), torch.tensor([1.0]), None), "equal length"),
            (SparseRow(torch.tensor([1.0]), torch.tensor([1.0]), None), "signed integer"),
            (SparseRow(torch.tensor([1]), torch.tensor([1.0]), torch.tensor([1, 2])), "positions for"),
            (SparseRow(torch.tensor([1]), torch.tensor([1.0]), torch.tensor([40000])), "position exceeds"),
        ],
    )
    def test_malformed_rows_are_refused(self, row: SparseRow, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            encode_csr([row], 8, torch.float32)

    def test_a_batch_cannot_mix_rows_with_and_without_positions(self) -> None:
        mixed = [self.rows(True)[0], self.rows(False)[0]]
        with pytest.raises(ValueError, match="mixes both"):
            encode_csr(mixed, 8, torch.float32)


class TestTopK:
    def codes(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor([[0, 3], [2, 1]], dtype=torch.int32), torch.tensor([[1.0, 0.0], [2.0, 3.0]])  # indices (2, 2) int32, values (2, 2)

    def test_valid_codes_pass(self) -> None:
        indices, values = self.codes()
        validate_topk(indices, values, 4, 2)

    @pytest.mark.parametrize(
        ("indices", "values", "message"),
        [
            (torch.zeros(2, 2), torch.zeros(2, 2), "signed integer"),
            (torch.zeros(2, 3, dtype=torch.int32), torch.zeros(2, 3), "both have shape"),
            (torch.zeros(2, 2, dtype=torch.int32), torch.zeros(2, 1), "both have shape"),
            (torch.zeros(2, 2, dtype=torch.int32), torch.zeros(2, 2, dtype=torch.int64), "finite floating point"),
            (torch.zeros(2, 2, dtype=torch.int32), torch.full((2, 2), float("inf")), "finite floating point"),
            (torch.tensor([[0, 4]], dtype=torch.int32), torch.zeros(1, 2), "outside 0..3"),
            (torch.tensor([[-1, 0]], dtype=torch.int32), torch.zeros(1, 2), "outside 0..3"),
            (torch.tensor([[2, 2]], dtype=torch.int32), torch.zeros(1, 2), "unique within each residue"),
        ],
    )
    def test_malformed_codes_are_refused(self, indices: torch.Tensor, values: torch.Tensor, message: str) -> None:
        # indices, values: (r, k) for r residues and k codes each; the malformed cases vary shape, dtype or range
        with pytest.raises(ValueError, match=message):
            validate_topk(indices, values, 4, 2)

    def test_rows_concatenate_with_offsets_and_keep_zero_entries_in_order(self) -> None:
        indices, values = self.codes()
        rows = [TopKRow(indices, values), TopKRow(indices[:1], values[:1])]
        tensors = encode_topk_rows(rows, 4, 2, torch.bfloat16)
        assert tensors["offsets"].tolist() == [0, 2, 3]
        assert tensors["indices"].shape == (3, 2) and tensors["indices"].dtype == torch.int32
        assert tensors["values"].dtype == torch.bfloat16
        assert tensors["values"][0].tolist() == [1.0, 0.0]

    def test_no_rows_give_empty_typed_tensors(self) -> None:
        tensors = encode_topk_rows([], 4, 2, torch.float32)
        assert tensors["indices"].shape == (0, 2) and tensors["values"].shape == (0, 2)

    def test_conversion_past_the_dtype_range_and_foreign_rows_are_refused(self) -> None:
        indices, _ = self.codes()
        huge = torch.full((2, 2), 1e30)
        with pytest.raises(ValueError, match="exceeded the stored dtype range"):
            encode_topk_rows([TopKRow(indices, huge)], 4, 2, torch.float16)
        with pytest.raises(TypeError, match="requires TopKRow"):
            encode_topk_rows([torch.zeros(2, 2)], 4, 2, torch.float32)  # type: ignore[list-item]


class TestRowBookkeeping:
    def test_row_count_reads_the_index_tensor_of_each_layout(self) -> None:
        assert row_count(DENSE, {"values": torch.zeros(5, 3)}) == 5
        assert row_count(CSR, {"indptr": torch.zeros(4, dtype=torch.int64)}) == 3
        assert row_count(RAGGED, {"offsets": torch.zeros(2, dtype=torch.int64)}) == 1
        assert row_count(RAGGED_TOPK, {"offsets": torch.zeros(1, dtype=torch.int64)}) == 0
        with pytest.raises(ValueError, match="Unknown feature layout 'rows'"):
            row_count("rows", {})

    def test_tensor_names_follow_the_layout_and_positions(self) -> None:
        assert tensor_names(DENSE, positions=False) == ("values",)
        assert tensor_names(CSR, positions=True) == ("indptr", "indices", "values", "positions")
        assert tensor_names(CSR, positions=False) == ("indptr", "indices", "values")
        assert tensor_names(RAGGED, positions=False) == ("offsets", "values")
        assert tensor_names(RAGGED_TOPK, positions=False) == ("offsets", "indices", "values")
        with pytest.raises(ValueError, match="Unknown feature layout"):
            tensor_names("rows", positions=False)

    def test_encoded_byte_sizes_match_the_tensors_actually_written(self) -> None:
        dense = torch.zeros(6)
        assert row_tensor_bytes(dense, DENSE, 6, torch.float16, positions=False) == 12
        ragged = torch.zeros(4, 3)
        assert row_tensor_bytes(ragged, RAGGED, 3, torch.float32, positions=False) == 8 + 4 * 3 * 4
        sparse = SparseRow(torch.tensor([1, 2], dtype=torch.int32), torch.ones(2), None)
        assert row_tensor_bytes(sparse, CSR, 8, torch.float32, positions=False) == 8 + 2 * (4 + 4)
        assert row_tensor_bytes(sparse, CSR, 8, torch.float32, positions=True) == 8 + 2 * (4 + 4 + 2)
        topk = TopKRow(torch.zeros(3, 2, dtype=torch.int32), torch.zeros(3, 2))
        assert row_tensor_bytes(topk, RAGGED_TOPK, 8, torch.bfloat16, positions=False) == 8 + 6 * (4 + 2)

    def test_the_wrong_row_type_for_a_layout_is_a_type_error(self) -> None:
        with pytest.raises(TypeError, match="requires a TopKRow"):
            row_tensor_bytes(torch.zeros(1, 1), RAGGED_TOPK, 4, torch.float32, positions=False)
        with pytest.raises(TypeError, match="requires a SparseRow"):
            row_tensor_bytes(torch.zeros(4), CSR, 4, torch.float32, positions=False)
        with pytest.raises(TypeError, match="requires tensor rows"):
            row_tensor_bytes(SparseRow(torch.zeros(0), torch.zeros(0), None), DENSE, 4, torch.float32, positions=False)


class TestFileLockAndPublication:
    def test_a_second_owner_is_refused_while_the_lock_is_held_and_admitted_after(self, tmp_path: Path) -> None:
        lock = tmp_path / "nested" / "writer.lock"
        with file_lock(lock):
            assert lock.is_file()
            with (
                pytest.raises(BlockingIOError, match="already has an active writer"),
                file_lock(lock, wait=False),
            ):
                pytest.fail("the lock was granted twice")
        with file_lock(lock, wait=False):
            pass
        assert lock.is_file(), "a lock file stays in place so that no other inode can be locked at its path"

    def test_publishing_renames_the_staged_file_and_keeps_its_bytes(self, tmp_path: Path) -> None:
        staged = tmp_path / "part.writing"
        staged.write_bytes(b"payload")
        publish_file(staged, tmp_path / "part.safetensors")
        assert not staged.exists()
        assert (tmp_path / "part.safetensors").read_bytes() == b"payload"
        sync_directory(tmp_path)

    def test_publishing_can_leave_the_directory_flush_to_the_caller(self, tmp_path: Path) -> None:
        staged = tmp_path / "other.writing"
        staged.write_bytes(b"x")
        publish_file(staged, tmp_path / "other.bin", sync_parent=False)
        assert (tmp_path / "other.bin").read_bytes() == b"x"


def dense_store(root: Path) -> FeatureStore:
    store = FeatureStore.open(root, StoredFeature("layout-dense", DENSE, 2, torch.float32))
    with store.segment("first") as writer:
        writer.append(SEQUENCES[:2], [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])])
    with store.segment("second") as writer:
        writer.append(SEQUENCES[2:], [torch.tensor([5.0, 6.0])])
    return store


class TestStoreRecords:
    def test_an_address_says_where_a_row_sits(self, tmp_path: Path) -> None:
        store = dense_store(tmp_path)
        address = store.address(SEQUENCES[2])
        assert address == RowAddress(segment="second", part=0, row=0, residues=0)
        assert store.address("NOTHERE") is None
        with pytest.raises(dataclasses.FrozenInstanceError):
            address.row = 3  # type: ignore[misc]

    def test_segment_receipts_list_committed_segments_oldest_first(self, tmp_path: Path) -> None:
        store = dense_store(tmp_path)
        receipts = store.segments()
        assert all(isinstance(receipt, SegmentReceipt) for receipt in receipts)
        assert {receipt.fingerprint for receipt in receipts} == {"first", "second"}
        by_name = {receipt.fingerprint: receipt for receipt in receipts}
        assert by_name["first"].rows == 2 and by_name["second"].rows == 1
        assert by_name["first"].parts == (0,)
        assert [receipt.committed_at for receipt in receipts] == sorted(receipt.committed_at for receipt in receipts)

    def test_reserved_part_numbers_are_consecutive_and_unique(self, tmp_path: Path) -> None:
        store = FeatureStore.open(tmp_path, StoredFeature("layout-reserve", DENSE, 2, torch.float32))
        with store.segment("run") as writer:
            numbers = [writer.reserve_part() for _ in range(3)]
            assert numbers == [0, 1, 2]
            writer.append(SEQUENCES[:1], [torch.zeros(2)])
            assert writer.reserve_part() == 4, "an append takes the number after the reservations"


class TestPartReceipt:
    def committed_part(self, store: FeatureStore, segment: str) -> dict[str, object]:
        marker = store.directory / SEGMENTS_DIRECTORY / segment / COMMIT_FILE
        return json.loads(marker.read_text(encoding="utf-8"))["parts"][0]

    def test_a_saved_receipt_vouches_for_the_same_files_and_nothing_else(self, tmp_path: Path) -> None:
        store = dense_store(tmp_path / "store")
        payload = store.spec.payload()
        path = tmp_path / "receipt.json"
        receipt = PartReceipt(path, store.directory, payload)
        committed = self.committed_part(store, "first")
        described = receipt.describe("first", 0, committed)
        assert described["sha256"] == committed["sha256"]
        assert described["size"] == (store.directory / SEGMENTS_DIRECTORY / "first" / PART_TEMPLATE.format(0)).stat().st_size
        assert receipt.trusts("first", 0, described) is False
        receipt.record("first", 0, described)
        receipt.save()
        assert json.loads(path.read_text(encoding="utf-8"))["schema"] == SCHEMA
        assert receipt.trusts("first", 0, described) is True
        assert PartReceipt(path, store.directory, payload).trusts("first", 0, described) is True
        assert PartReceipt(path, store.directory, payload, trust=False).trusts("first", 0, described) is False
        assert PartReceipt(path, store.directory, payload).trusts("first", 0, {**described, "size": 1}) is False

    def test_a_receipt_for_another_descriptor_is_ignored(self, tmp_path: Path) -> None:
        store = dense_store(tmp_path / "store")
        path = tmp_path / "receipt.json"
        receipt = PartReceipt(path, store.directory, store.spec.payload())
        described = receipt.describe("first", 0, self.committed_part(store, "first"))
        receipt.record("first", 0, described)
        receipt.save()
        other = {**store.spec.payload(), "width": 3}
        assert PartReceipt(path, store.directory, other).trusts("first", 0, described) is False

    def test_an_unreadable_receipt_warns_and_trusts_nothing(self, tmp_path: Path) -> None:
        store = dense_store(tmp_path / "store")
        path = tmp_path / "receipt.json"
        path.write_text("{broken", encoding="utf-8")
        with pytest.warns(RuntimeWarning, match="unreadable verification receipt"):
            receipt = PartReceipt(path, store.directory, store.spec.payload())
        assert receipt.trusts("first", 0, {}) is False

    def test_saving_with_nothing_verified_writes_no_file(self, tmp_path: Path) -> None:
        store = dense_store(tmp_path / "store")
        path = tmp_path / "receipt.json"
        PartReceipt(path, store.directory, store.spec.payload()).save()
        assert not path.exists()

    def test_the_reader_writes_a_receipt_that_the_next_reader_trusts(self, tmp_path: Path) -> None:
        store = dense_store(tmp_path / "store")
        path = tmp_path / "reader-receipt.json"
        with FeatureReader(store, receipt=path) as reader:
            assert reader.read([SEQUENCES[0]])[0].tolist() == [1.0, 2.0]
        assert path.is_file()
        recorded = json.loads(path.read_text(encoding="utf-8"))
        assert set(recorded["parts"]) == {"first/0"}


class TestCsrRowsAndConversion:
    def test_a_csr_block_is_gathered_in_the_order_asked(self, tmp_path: Path) -> None:
        spec = StoredFeature("layout-csr", CSR, 8, torch.float32, positions=True)
        store = FeatureStore.open(tmp_path, spec)
        rows = [
            SparseRow(torch.tensor([1, 4], dtype=torch.int32), torch.tensor([0.5, 1.5]), torch.tensor([2, 3], dtype=torch.int16)),
            SparseRow(torch.tensor([0], dtype=torch.int32), torch.tensor([9.0]), torch.tensor([5], dtype=torch.int16)),
        ]
        with store.segment("run") as writer:
            writer.append(SEQUENCES[:2], rows)
        with FeatureReader.open(store.directory) as reader:
            block = reader.read_csr([SEQUENCES[1], SEQUENCES[0], SEQUENCES[1]])
        assert isinstance(block, CsrRows)
        assert block.indptr.tolist() == [0, 1, 3, 4]
        assert block.indices.tolist() == [0, 1, 4, 0]
        assert block.values.tolist() == [9.0, 0.5, 1.5, 9.0]
        assert block.positions is not None and block.positions.tolist() == [5, 2, 3, 5]

    def test_a_conversion_reports_what_it_wrote_and_what_it_compared(self, tmp_path: Path) -> None:
        store = FeatureStore.open(tmp_path, StoredFeature("layout-convert", DENSE, 2, torch.float32))
        old = [(sequence, torch.full((2,), float(index))) for index, sequence in enumerate(SEQUENCES)]
        origin = {"format": "synthetic", "rows": len(old)}
        first = convert_rows(store, lambda: iter(old), origin=origin)
        assert isinstance(first, ConversionReceipt)
        assert (first.source_rows, first.written, first.skipped, first.verified) == (3, 3, 0, 3)
        assert len(first.segments) == 1
        again = convert_rows(store, lambda: iter(old), origin=origin)
        assert (again.written, again.skipped, again.verified, again.segments) == (0, 3, 3, ())
