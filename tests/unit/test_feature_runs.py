"""Persisting taps: one pass fills several feature stores, and a second pass runs no model.

Every check runs on a tiny randomly initialized ESM++ on CPU. What a store returns must equal what
the same tap returned in memory, because a project reading the store has to get the numbers the
model produced.
"""

from __future__ import annotations

import pytest
import torch

from pathlib import Path
from torch import nn

from fastplms.embeddings import (
    HiddenTap,
    ReducedTap,
    TapBatch,
    embed_dataset,
    embed_into_features,
)
from fastplms.features import CSR, DENSE, RAGGED, StoredFeature, FeatureStore, open_feature
from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
    ESMplusplusConfig,
    ESMplusplusModel,
)


HIDDEN_SIZE = 32  # d
N_LAYERS = 3  # hidden states 0..3
SEQUENCES = ["MKTAYIAKQR", "GG", "MSTNPKPQRKTKRNT"]
DIGEST = "0123456789abcdef"


def model() -> nn.Module:
    torch.manual_seed(0)
    return ESMplusplusModel(
        ESMplusplusConfig(
            hidden_size=HIDDEN_SIZE,
            num_attention_heads=4,
            num_hidden_layers=N_LAYERS,
            attn_backend="sdpa",
        )
    ).eval()


def key(name: str) -> str:
    return f"esmpp__{name}__{DIGEST}"


def pooled_feature() -> StoredFeature:
    return StoredFeature(key=key("mean"), layout=DENSE, width=HIDDEN_SIZE, dtype=torch.float32)


def residues_feature() -> StoredFeature:
    return StoredFeature(key=key("residues"), layout=RAGGED, width=HIDDEN_SIZE, dtype=torch.float32)


def sparse_feature() -> StoredFeature:
    return StoredFeature(key=key("codes"), layout=CSR, width=HIDDEN_SIZE, dtype=torch.float32)


def top_two(batch: TapBatch) -> torch.Tensor:
    """A stand-in sparse reducer: the two largest residue-mean codes, exact zeros elsewhere.

    It mimics what a top-k sparse autoencoder leaves behind, which is what the csr layout stores.
    """

    mask = batch.residue_mask.unsqueeze(-1)  # (b, l, 1)
    mean = (batch.X * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (b, d)
    kept = mean.abs().topk(2, dim=-1).indices  # (b, 2)
    sparse = torch.zeros_like(mean)  # (b, d)
    return sparse.scatter(1, kept, mean.gather(1, kept))  # (b, d)


class TestOnePassFillsSeveralStores:
    def test_three_taps_fill_three_stores(self, tmp_path: Path) -> None:
        taps = [
            HiddenTap("mean", layer=-1, pooling="mean"),
            HiddenTap("residues", layer=-1),
            ReducedTap("codes", layer=2, reduce=top_two, identity={"reducer": "top_two"}),
        ]
        receipts = embed_into_features(
            model(),
            SEQUENCES,
            tmp_path,
            {"mean": pooled_feature(), "residues": residues_feature(), "codes": sparse_feature()},
            taps=taps,
        )
        assert sorted(receipts) == ["codes", "mean", "residues"]
        assert all(receipt.rows == len(SEQUENCES) for receipt in receipts.values())
        assert {receipt.fingerprint for receipt in receipts.values()} == {
            next(iter(receipts.values())).fingerprint
        }

    def test_what_the_store_returns_equals_what_the_tap_returned(self, tmp_path: Path) -> None:
        taps = [
            HiddenTap("mean", layer=-1, pooling="mean"),
            HiddenTap("residues", layer=-1),
            ReducedTap("codes", layer=2, reduce=top_two, identity={"reducer": "top_two"}),
        ]
        in_memory = embed_dataset(model(), SEQUENCES, taps=list(taps))
        embed_into_features(
            model(),
            SEQUENCES,
            tmp_path,
            {"mean": pooled_feature(), "residues": residues_feature(), "codes": sparse_feature()},
            taps=taps,
        )
        stores = {
            "mean": open_feature(tmp_path, pooled_feature()),
            "residues": open_feature(tmp_path, residues_feature()),
            "codes": open_feature(tmp_path, sparse_feature()),
        }
        for record in in_memory:
            for name, store in stores.items():
                stored = store.read([record.sequence])[0]
                assert torch.equal(stored, record.tensors[name]), name

    def test_residue_rows_keep_each_sequence_length(self, tmp_path: Path) -> None:
        embed_into_features(
            model(),
            SEQUENCES,
            tmp_path,
            {"residues": residues_feature()},
            taps=[HiddenTap("residues", layer=-1)],
        )
        store = open_feature(tmp_path, residues_feature())
        assert store.residue_counts(SEQUENCES) == [len(sequence) for sequence in SEQUENCES]
        for sequence in SEQUENCES:
            assert store.read([sequence])[0].shape == (len(sequence), HIDDEN_SIZE)

    def test_a_sparse_tap_is_stored_compressed(self, tmp_path: Path) -> None:
        embed_into_features(
            model(),
            SEQUENCES,
            tmp_path,
            {"codes": sparse_feature()},
            taps=[ReducedTap("codes", layer=2, reduce=top_two, identity={"reducer": "top_two"})],
        )
        store = open_feature(tmp_path, sparse_feature())
        for row in store.read_sparse(SEQUENCES):
            assert row.indices.numel() == 2
            assert row.positions is None


class TestOnlyWhatIsMissingIsEmbedded:
    def test_a_second_call_runs_no_model(self, tmp_path: Path) -> None:
        taps = [HiddenTap("mean", layer=-1, pooling="mean")]
        embed_into_features(model(), SEQUENCES, tmp_path, {"mean": pooled_feature()}, taps=taps)

        class Refuses(nn.Module):
            embedding_tap_support = True
            embedding_tap_state_count = N_LAYERS + 1

            def forward(self, *args: object, **kwargs: object) -> None:
                raise AssertionError("The model must not run when every sequence is stored.")

        assert (
            embed_into_features(Refuses(), SEQUENCES, tmp_path, {"mean": pooled_feature()}, taps=taps)
            == {}
        )

    def test_only_the_new_sequences_get_rows(self, tmp_path: Path) -> None:
        taps = [HiddenTap("mean", layer=-1, pooling="mean")]
        embed_into_features(model(), SEQUENCES[:2], tmp_path, {"mean": pooled_feature()}, taps=taps)
        receipts = embed_into_features(
            model(), SEQUENCES, tmp_path, {"mean": pooled_feature()}, taps=taps
        )
        assert receipts["mean"].rows == 1
        store = open_feature(tmp_path, pooled_feature())
        assert len(store) == len(SEQUENCES)
        assert len(store.segments()) == 2

    def test_a_store_that_lags_the_others_catches_up(self, tmp_path: Path) -> None:
        pooled = HiddenTap("mean", layer=-1, pooling="mean")
        residues = HiddenTap("residues", layer=-1)
        embed_into_features(model(), SEQUENCES, tmp_path, {"mean": pooled_feature()}, taps=[pooled])
        receipts = embed_into_features(
            model(),
            SEQUENCES,
            tmp_path,
            {"mean": pooled_feature(), "residues": residues_feature()},
            taps=[pooled, residues],
        )
        assert sorted(receipts) == ["residues"]
        assert receipts["residues"].rows == len(SEQUENCES)
        assert len(open_feature(tmp_path, pooled_feature())) == len(SEQUENCES)

    def test_repeated_sequences_are_stored_once(self, tmp_path: Path) -> None:
        receipts = embed_into_features(
            model(),
            [*SEQUENCES, SEQUENCES[0]],
            tmp_path,
            {"mean": pooled_feature()},
            taps=[HiddenTap("mean", layer=-1, pooling="mean")],
        )
        assert receipts["mean"].rows == len(SEQUENCES)


class TestRefusals:
    def test_features_must_name_every_tap(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="exactly the taps"):
            embed_into_features(
                model(),
                SEQUENCES,
                tmp_path,
                {"mean": pooled_feature()},
                taps=[HiddenTap("mean", layer=-1, pooling="mean"), HiddenTap("other", layer=1)],
            )

    def test_a_pooled_tap_cannot_fill_a_ragged_feature(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="is ragged"):
            embed_into_features(
                model(),
                SEQUENCES,
                tmp_path,
                {"mean": StoredFeature(key=key("r"), layout=RAGGED, width=HIDDEN_SIZE, dtype=torch.float32)},
                taps=[HiddenTap("mean", layer=-1, pooling="mean")],
            )

    def test_a_residue_tap_cannot_fill_a_dense_feature(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="is dense"):
            embed_into_features(
                model(),
                SEQUENCES,
                tmp_path,
                {"residues": pooled_feature()},
                taps=[HiddenTap("residues", layer=-1)],
            )

    def test_a_feature_keeping_positions_says_no_tap_carries_them(self, tmp_path: Path) -> None:
        spec = StoredFeature(
            key=key("codes"), layout=CSR, width=HIDDEN_SIZE, dtype=torch.float32, positions=True
        )
        with pytest.raises(ValueError, match="no tap carries them"):
            embed_into_features(
                model(),
                SEQUENCES,
                tmp_path,
                {"codes": spec},
                taps=[ReducedTap("codes", layer=2, reduce=top_two, identity={"r": 1})],
            )

    def test_no_sequences_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="at least one sequence"):
            embed_into_features(
                model(), [], tmp_path, {"mean": pooled_feature()},
                taps=[HiddenTap("mean", layer=-1, pooling="mean")],
            )


class TestProvenance:
    def test_the_segment_is_named_by_the_run_fingerprint(self, tmp_path: Path) -> None:
        taps = [HiddenTap("mean", layer=-1, pooling="mean")]
        in_memory = embed_dataset(model(), SEQUENCES, taps=list(taps))
        receipts = embed_into_features(
            model(), SEQUENCES, tmp_path, {"mean": pooled_feature()}, taps=taps
        )
        assert receipts["mean"].fingerprint == in_memory.metadata["run_fingerprint"]

    def test_the_caller_metadata_and_input_fingerprint_are_recorded(self, tmp_path: Path) -> None:
        receipts = embed_into_features(
            model(),
            SEQUENCES,
            tmp_path,
            {"mean": pooled_feature()},
            taps=[HiddenTap("mean", layer=-1, pooling="mean")],
            metadata={"machine": "wings_laptop"},
        )
        recorded = receipts["mean"].metadata
        assert recorded["machine"] == "wings_laptop"
        assert isinstance(recorded["input_fingerprint"], str) and recorded["input_fingerprint"]

    def test_a_store_written_by_a_run_reads_without_the_model(self, tmp_path: Path) -> None:
        embed_into_features(
            model(),
            SEQUENCES,
            tmp_path,
            {"mean": pooled_feature()},
            taps=[HiddenTap("mean", layer=-1, pooling="mean")],
        )
        store = FeatureStore.read_only(tmp_path / key("mean"))
        assert store.read([SEQUENCES[0]])[0].shape == (HIDDEN_SIZE,)
