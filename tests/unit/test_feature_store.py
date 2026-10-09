"""The feature store's contract: identity, the three layouts, commit visibility, and repair."""

from __future__ import annotations

import json
import sqlite3
import pytest
import torch

from fastplms.features import (
    CSR,
    DENSE,
    RAGGED,
    StoredFeature,
    FeatureStore,
    SparseRow,
    features_in,
    open_feature,
    sequence_digest,
)
from fastplms.features.store import COMMIT_FILE, PART_TEMPLATE, SEGMENTS_DIRECTORY


KEY = "esmc_600m__0123abcd__l30__mean__float32__maxlen-none__0123456789abcdef"
SEQUENCES = ("MKTFFVAVLALALATA", "MAAAAGGGKKLL", "MWWWCCYY")


def dense_feature(**overrides: object) -> StoredFeature:
    fields: dict[str, object] = {
        "key": KEY,
        "layout": DENSE,
        "width": 4,
        "dtype": torch.float32,
        "descriptor": {"model": "Synthyra/ESMplusplus_large", "layer": 30, "pooling": ["mean"]},
    }
    fields.update(overrides)
    return StoredFeature(**fields)  # type: ignore[arg-type]


def dense_rows(count: int, width: int = 4) -> list[torch.Tensor]:
    return [torch.arange(width, dtype=torch.float32) + position for position in range(count)]  # count tensors of (width,)


class TestIdentity:
    def test_digest_is_the_sequence_bytes(self) -> None:
        import hashlib

        assert sequence_digest("MKT") == hashlib.sha256(b"MKT").hexdigest()

    def test_one_residue_apart_is_a_different_row(self) -> None:
        assert sequence_digest("MKTA") != sequence_digest("MKTG")

    def test_an_empty_sequence_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            sequence_digest("")

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"key": "not a filename"}, "filename-safe"),
            ({"layout": "sparse"}, "layout must be one of"),
            ({"width": 0}, "positive integer"),
            ({"dtype": torch.int32}, "float64, float32"),
            ({"positions": True}, "Only a csr feature"),
        ],
    )
    def test_a_spec_refuses_what_it_cannot_store(self, overrides: dict, message: str) -> None:
        with pytest.raises((ValueError, TypeError), match=message):
            dense_feature(**overrides)

    def test_a_spec_round_trips_through_its_payload(self) -> None:
        spec = dense_feature()
        assert StoredFeature.from_payload(spec.payload()) == spec

    def test_a_foreign_directory_is_refused(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="format"):
            StoredFeature.from_payload({"format": "something-else", "key": KEY})


class TestDenseFeature:
    def test_append_then_read_returns_the_same_rows(self, tmp_path) -> None:
        store = open_feature(tmp_path, dense_feature())
        rows = dense_rows(3)
        with store.segment("run0000") as writer:
            writer.append(SEQUENCES, rows)
        for sequence, row in zip(SEQUENCES, rows, strict=True):
            assert torch.equal(store.read([sequence])[0], row)

    def test_read_follows_the_order_asked(self, tmp_path) -> None:
        store = open_feature(tmp_path, dense_feature())
        rows = dense_rows(3)
        with store.segment("run0000") as writer:
            writer.append(SEQUENCES, rows)
        backwards = list(reversed(SEQUENCES))
        read = store.read(backwards)
        assert [row.tolist() for row in read] == [row.tolist() for row in reversed(rows)]

    def test_missing_names_only_what_is_absent(self, tmp_path) -> None:
        store = open_feature(tmp_path, dense_feature())
        assert store.missing(SEQUENCES) == SEQUENCES
        with store.segment("run0000") as writer:
            writer.append(SEQUENCES[:2], dense_rows(2))
        assert store.missing(SEQUENCES) == (SEQUENCES[2],)
        assert len(store) == 2

    def test_missing_deduplicates_and_keeps_order(self, tmp_path) -> None:
        store = open_feature(tmp_path, dense_feature())
        assert store.missing(["MKT", "MAA", "MKT"]) == ("MKT", "MAA")

    def test_reading_an_absent_sequence_raises(self, tmp_path) -> None:
        store = open_feature(tmp_path, dense_feature())
        with pytest.raises(KeyError, match="no row"):
            store.read(["MKT"])

    def test_the_first_absent_sequence_in_order_is_the_one_named(self, tmp_path) -> None:
        store = open_feature(tmp_path, dense_feature())
        with store.segment("run0000") as writer:
            writer.append(SEQUENCES, dense_rows(3))
        with pytest.raises(KeyError, match=r"sequence of 5 residues"):
            store.read([SEQUENCES[0], "MKTAA", "MKTAAAAAAA", SEQUENCES[1]])

    def test_a_sequence_asked_twice_is_read_twice_in_place(self, tmp_path) -> None:
        store = open_feature(tmp_path, dense_feature())
        rows = dense_rows(3)
        with store.segment("run0000") as writer:
            writer.append(SEQUENCES, rows)
        asked = [SEQUENCES[2], SEQUENCES[0], SEQUENCES[2]]
        read = store.read(asked)
        assert [row.tolist() for row in read] == [rows[2].tolist(), rows[0].tolist(), rows[2].tolist()]

    def test_locating_rows_opens_the_index_once_however_many_are_asked(self, tmp_path, monkeypatch) -> None:
        store = open_feature(tmp_path, dense_feature())
        many = [f"MK{'A' * length}" for length in range(1, 1200)]
        with store.segment("run0000") as writer:
            writer.append(many, dense_rows(len(many)))
        real_connect = sqlite3.connect
        opened: list[object] = []

        def counting_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
            opened.append(args)
            return real_connect(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(sqlite3, "connect", counting_connect)
        store.content_pins(many[:3])
        few = len(opened)
        opened.clear()
        pins = store.content_pins(many)
        assert len(opened) == few == 1
        assert len(pins) == 3  # the descriptor, the one commit marker, the one part

    def test_a_wrong_width_row_is_refused(self, tmp_path) -> None:
        store = open_feature(tmp_path, dense_feature())
        with store.segment("run0000") as writer:
            with pytest.raises(ValueError, match=r"shape \(4,\)"):
                writer.append(["MKT"], [torch.zeros(3)])
            writer.append(["MKT"], dense_rows(1))

    def test_a_row_is_stored_in_the_declared_dtype(self, tmp_path) -> None:
        store = open_feature(tmp_path, dense_feature(dtype=torch.float16))
        with store.segment("run0000") as writer:
            writer.append(["MKT"], [torch.tensor([0.5, 1.5, 2.5, 3.5], dtype=torch.float32)])
        assert store.read(["MKT"])[0].dtype is torch.float16


class TestSparseFeature:
    def spec(self, *, positions: bool) -> StoredFeature:
        return StoredFeature(
            key=KEY, layout=CSR, width=16, dtype=torch.float16, positions=positions
        )

    def rows(self, *, positions: bool) -> list[SparseRow]:
        return [
            SparseRow(
                indices=torch.tensor([1, 5]),
                values=torch.tensor([0.25, 0.5]),
                positions=torch.tensor([3, 7]) if positions else None,
            ),
            SparseRow(
                indices=torch.tensor([15]),
                values=torch.tensor([1.0]),
                positions=torch.tensor([0]) if positions else None,
            ),
        ]

    def test_compressed_rows_round_trip(self, tmp_path) -> None:
        store = open_feature(tmp_path, self.spec(positions=True))
        with store.segment("run0000") as writer:
            writer.append(SEQUENCES[:2], self.rows(positions=True))
        read = store.read_sparse(list(SEQUENCES[:2]))
        assert read[0].indices.tolist() == [1, 5]
        assert read[0].values.tolist() == [0.25, 0.5]
        assert read[0].positions is not None and read[0].positions.tolist() == [3, 7]
        assert read[1].indices.tolist() == [15]

    def test_read_densifies_a_sparse_row(self, tmp_path) -> None:
        store = open_feature(tmp_path, self.spec(positions=False))
        with store.segment("run0000") as writer:
            writer.append(SEQUENCES[:2], self.rows(positions=False))
        dense = store.read([SEQUENCES[0]])[0]
        assert dense.shape == (16,)
        assert dense[1].item() == 0.25
        assert dense[5].item() == 0.5
        assert dense.sum().item() == pytest.approx(0.75)

    def test_a_code_outside_the_codebook_is_refused(self, tmp_path) -> None:
        store = open_feature(tmp_path, self.spec(positions=False))
        with store.segment("run0000") as writer, pytest.raises(ValueError, match="outside 0..15"):
            writer.append(["MKT"], [SparseRow(torch.tensor([16]), torch.tensor([1.0]), None)])

    def test_positions_must_match_the_spec(self, tmp_path) -> None:
        store = open_feature(tmp_path, self.spec(positions=True))
        with store.segment("run0000") as writer, pytest.raises(ValueError, match="stores argmax positions"):
            writer.append(SEQUENCES[:2], self.rows(positions=False))

    def test_a_batch_cannot_mix_rows_with_and_without_positions(self, tmp_path) -> None:
        store = open_feature(tmp_path, self.spec(positions=True))
        mixed = [self.rows(positions=True)[0], self.rows(positions=False)[1]]
        with store.segment("run0000") as writer, pytest.raises(ValueError, match="mixes both"):
            writer.append(SEQUENCES[:2], mixed)

    def test_read_sparse_needs_a_sparse_feature(self, tmp_path) -> None:
        store = open_feature(tmp_path, dense_feature())
        with pytest.raises(ValueError, match="needs a csr feature"):
            store.read_sparse(["MKT"])


class TestRaggedFeature:
    def spec(self) -> StoredFeature:
        return StoredFeature(key=KEY, layout=RAGGED, width=8, dtype=torch.bfloat16)

    def test_each_sequence_keeps_its_own_residue_count(self, tmp_path) -> None:
        store = open_feature(tmp_path, self.spec())
        rows = [torch.ones((len(sequence), 8)) * index for index, sequence in enumerate(SEQUENCES)]
        with store.segment("run0000") as writer:
            writer.append(SEQUENCES, rows)
        for sequence, row in zip(SEQUENCES, rows, strict=True):
            read = store.read([sequence])[0]
            assert read.shape == (len(sequence), 8)
            assert torch.equal(read.to(torch.float32), row.to(torch.bfloat16).to(torch.float32))

    def test_residue_counts_are_indexed_for_length_bucketing(self, tmp_path) -> None:
        store = open_feature(tmp_path, self.spec())
        rows = [torch.zeros((len(sequence), 8)) for sequence in SEQUENCES]
        with store.segment("run0000") as writer:
            writer.append(SEQUENCES, rows)
        assert store.residue_counts(list(SEQUENCES)) == [len(s) for s in SEQUENCES]

    def test_a_wrong_width_block_is_refused(self, tmp_path) -> None:
        store = open_feature(tmp_path, self.spec())
        with store.segment("run0000") as writer, pytest.raises(ValueError, match=r"shape \(r_i, 8\)"):
            writer.append(["MKT"], [torch.zeros((3, 4))])


class TestSegments:
    def test_a_run_streams_several_parts_into_one_segment(self, tmp_path) -> None:
        store = open_feature(tmp_path, dense_feature())
        with store.segment("run0000") as writer:
            assert writer.append(SEQUENCES[:1], dense_rows(1)) == 0
            assert writer.append(SEQUENCES[1:], dense_rows(2)) == 1
        receipts = store.segments()
        assert [receipt.fingerprint for receipt in receipts] == ["run0000"]
        assert receipts[0].parts == (0, 1)
        assert receipts[0].rows == 3
        assert len(store) == 3

    def test_nothing_is_visible_before_the_commit_marker(self, tmp_path) -> None:
        store = open_feature(tmp_path, dense_feature())
        context = store.segment("run0000")
        writer = context.__enter__()
        writer.append(SEQUENCES, dense_rows(3))
        assert len(store) == 0
        assert store.missing(SEQUENCES) == SEQUENCES
        assert store.segments() == ()
        writer.commit()
        assert len(store) == 3
        context.__exit__(None, None, None)

    def test_a_dead_run_leaves_a_segment_that_sweep_removes(self, tmp_path) -> None:
        store = open_feature(tmp_path, dense_feature())
        with pytest.raises(RuntimeError, match="interrupted"), store.segment("run0000") as writer:
            writer.append(SEQUENCES, dense_rows(3))
            assert store.sweep() == ()
            raise RuntimeError("interrupted")
        assert store.sweep() == ("run0000",)
        assert store.missing(SEQUENCES) == SEQUENCES
        assert not (tmp_path / KEY / SEGMENTS_DIRECTORY / "run0000").exists()

    def test_sweep_keeps_committed_segments(self, tmp_path) -> None:
        store = open_feature(tmp_path, dense_feature())
        with store.segment("run0000") as writer:
            writer.append(SEQUENCES, dense_rows(3))
        assert store.sweep() == ()
        assert len(store) == 3

    def test_a_committed_segment_cannot_be_written_twice(self, tmp_path) -> None:
        store = open_feature(tmp_path, dense_feature())
        with store.segment("run0000") as writer:
            writer.append(SEQUENCES, dense_rows(3))
        with pytest.raises(FileExistsError, match="already committed"), store.segment("run0000"):
            pass

    def test_a_sequence_cannot_be_stored_twice(self, tmp_path) -> None:
        store = open_feature(tmp_path, dense_feature())
        with store.segment("run0000") as writer:
            writer.append(SEQUENCES, dense_rows(3))
        with store.segment("run0001") as writer:
            with pytest.raises(ValueError, match="already have a row"):
                writer.append(SEQUENCES[:1], dense_rows(1))
            writer.append(["MQQQ"], dense_rows(1))

    def test_a_batch_cannot_repeat_a_sequence(self, tmp_path) -> None:
        store = open_feature(tmp_path, dense_feature())
        with store.segment("run0000") as writer:
            with pytest.raises(ValueError, match="repeats 1 sequences"):
                writer.append(["MKT", "MKT"], dense_rows(2))
            writer.append(["MKT"], dense_rows(1))

    def test_a_run_can_abandon_its_rows(self, tmp_path) -> None:
        store = open_feature(tmp_path, dense_feature())
        with store.segment("run0000") as writer:
            writer.append(SEQUENCES, dense_rows(3))
            writer.abandon()
        assert len(store) == 0
        assert store.segments() == ()

    def test_metadata_travels_with_the_segment(self, tmp_path) -> None:
        store = open_feature(tmp_path, dense_feature())
        with store.segment("run0000", {"machine": "gh200", "batch_size": 32}) as writer:
            writer.append(SEQUENCES, dense_rows(3))
        assert store.segments()[0].metadata == {"machine": "gh200", "batch_size": 32}


class TestOpening:
    def test_reopening_finds_the_rows(self, tmp_path) -> None:
        with open_feature(tmp_path, dense_feature()).segment("run0000") as writer:
            writer.append(SEQUENCES, dense_rows(3))
        reopened = open_feature(tmp_path, dense_feature())
        assert len(reopened) == 3
        assert reopened.missing(SEQUENCES) == ()

    def test_a_different_feature_under_one_key_is_refused(self, tmp_path) -> None:
        open_feature(tmp_path, dense_feature())
        with pytest.raises(ValueError, match="different feature under key"):
            open_feature(tmp_path, dense_feature(width=8))

    def test_read_only_takes_the_spec_from_disk(self, tmp_path) -> None:
        with open_feature(tmp_path, dense_feature()).segment("run0000") as writer:
            writer.append(SEQUENCES, dense_rows(3))
        store = FeatureStore.read_only(tmp_path / KEY)
        assert store.spec == dense_feature()
        assert torch.equal(store.read([SEQUENCES[0]])[0], dense_rows(1)[0])

    def test_features_in_lists_every_store_under_a_root(self, tmp_path) -> None:
        open_feature(tmp_path, dense_feature())
        open_feature(tmp_path, dense_feature(key=f"{KEY}-two"))
        assert sorted(store.spec.key for store in features_in(tmp_path)) == [KEY, f"{KEY}-two"]

    def test_the_descriptor_is_recorded_beside_the_rows(self, tmp_path) -> None:
        open_feature(tmp_path, dense_feature())
        payload = json.loads((tmp_path / KEY / "feature.json").read_text(encoding="utf-8"))
        assert payload["descriptor"]["model"] == "Synthyra/ESMplusplus_large"
        assert payload["layout"] == DENSE


class TestRepair:
    def test_reindex_rebuilds_the_index_from_the_segments(self, tmp_path) -> None:
        store = open_feature(tmp_path, dense_feature())
        with store.segment("run0000") as writer:
            writer.append(SEQUENCES, dense_rows(3))
        (tmp_path / KEY / "index.sqlite").unlink()
        rebuilt = open_feature(tmp_path, dense_feature())
        assert len(rebuilt) == 3
        assert torch.equal(rebuilt.read([SEQUENCES[1]])[0], dense_rows(2)[1])

    def test_reindex_is_idempotent(self, tmp_path) -> None:
        store = open_feature(tmp_path, dense_feature())
        with store.segment("run0000") as writer:
            writer.append(SEQUENCES, dense_rows(3))
        assert store.reindex() == 3
        assert store.reindex() == 3

    def test_a_committed_segment_the_index_lacks_is_picked_up_on_open(self, tmp_path) -> None:
        store = open_feature(tmp_path, dense_feature())
        with store.segment("run0000") as writer:
            writer.append(SEQUENCES, dense_rows(3))
        with store._connect() as connection:
            connection.execute("DELETE FROM rows")
            connection.commit()
        assert len(open_feature(tmp_path, dense_feature())) == 3

    def test_the_commit_marker_is_written_after_the_parts(self, tmp_path) -> None:
        store = open_feature(tmp_path, dense_feature())
        segment = tmp_path / KEY / SEGMENTS_DIRECTORY / "run0000"
        with store.segment("run0000") as writer:
            writer.append(SEQUENCES, dense_rows(3))
            assert (segment / PART_TEMPLATE.format(0)).exists()
            assert not (segment / COMMIT_FILE).exists()
        assert (segment / COMMIT_FILE).exists()
