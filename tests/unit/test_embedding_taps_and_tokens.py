"""Tap plans, token rows, the batch executor and the output state, on synthetic models and tokenizers.

Symbols: b sequences of a batch; l token columns; d hidden width; r residues of one sequence; n kept rows.
"""

from __future__ import annotations

import pytest
import torch

from pathlib import Path
from types import SimpleNamespace
from torch import nn

from fastplms.embeddings import (
    EmbeddingBatch,
    EmbeddingInput,
    EmbeddingMixin,
    EmbeddingRecord,
    HiddenTap,
    Pooler,
    ReducedTap,
    ResidueVocabulary,
    SparseResidueTap,
    StreamingTap,
    TapBatch,
    embed_dataset,
    plan_token_batches,
)
from fastplms.embeddings.batches import (
    BatchExecutor,
    canonical_residue_ids,
    select_hidden_state_embeddings,
)
from fastplms.embeddings.output import EmbeddingOutput
from fastplms.embeddings.taps import RowSelection, TapPlan, plan_taps
from fastplms.embeddings.token_batches import HostBatch, build_host_batch
from fastplms.embeddings.token_runs import distinct_with_digests
from fastplms.embeddings.tokens import check_canonical_text
from fastplms.features import sequence_digest


LETTERS = "ACDEFGHIKLMNPQRSTVWY"


class LetterTokenizer:
    """One token per residue: ids 0..3 are specials, then the twenty letters."""

    all_special_ids = [0, 1, 2, 3]
    pad_token_id, cls_token_id, eos_token_id, unk_token_id = 0, 1, 2, 3

    def __init__(self) -> None:
        self.vocabulary = {"<pad>": 0, "<cls>": 1, "<eos>": 2, "<unk>": 3}
        self.vocabulary.update({letter: 4 + index for index, letter in enumerate(LETTERS)})

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocabulary)

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [self.vocabulary.get(token, self.vocabulary["<unk>"]) for token in tokens]

    def __call__(self, texts: list[str], add_special_tokens: bool = True) -> dict[str, list[list[int]]]:
        return {"input_ids": [[1, *self.convert_tokens_to_ids(list(text)), 2] for text in texts]}


class TwoWideModel(nn.Module):
    """Hidden states are the residue code and its position, so every row is predictable."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(model_type="synthetic", _name_or_path="synthetic/checkpoint", _commit_hash="abc")
        self.calls: list[list[str]] = []

    def _embedding_batch(self, sequences: list[str]) -> EmbeddingBatch:
        self.calls.append(list(sequences))
        b, length = len(sequences), max(map(len, sequences)) + 2
        X = torch.zeros(b, length, 2)  # (b, l, 2)
        M = torch.zeros(b, length, dtype=torch.bool)  # (b, l)
        for row, sequence in enumerate(sequences):
            for column, residue in enumerate(sequence, start=1):
                X[row, column] = torch.tensor([float(ord(residue)), float(column)])
                M[row, column] = True
        return EmbeddingBatch(X=X, residue_mask=M)


def executor(model: nn.Module, *, full: bool, batch_size: int = 2, pooler: Pooler | None = None, **overrides) -> BatchExecutor:
    settings = {
        "model": model, "batch_size": batch_size, "max_tokens_per_batch": None, "max_length": None, "truncate": False,
        "model_kwargs": {}, "hidden_state_source": "encoder", "normalized_decoder_inputs": None,
        "decoder_input_ids": None, "decoder_attention_mask": None, "_embedding_batch_fn": None, "tokenizer": None,
        "store_all_hidden_states": False, "full_embeddings": full, "dtype": None, "pooler": pooler,
        "attention_backend": None, "need_attentions": False,
    }
    settings.update(overrides)
    return BatchExecutor(**settings)


class TestSelectHiddenStateEmbeddings:
    def states(self) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        hidden = tuple(torch.full((2, 3, 4), float(layer)) for layer in range(3))  # (b, l, d) each
        return hidden[-1] + 0.5, hidden  # (2, 3, 4) = (b, l, d) final state, then three (b, l, d) layer states

    def test_the_default_is_the_last_hidden_state_without_needing_hidden_states(self) -> None:
        last, _ = self.states()
        assert select_hidden_state_embeddings(last, None) is last

    def test_an_index_picks_one_state_and_negative_indices_count_back(self) -> None:
        last, hidden = self.states()
        assert select_hidden_state_embeddings(last, hidden, hidden_state_index=1) is hidden[1]
        assert select_hidden_state_embeddings(last, hidden, hidden_state_index=-2) is hidden[-2]

    def test_storing_all_states_stacks_them_on_a_new_axis_in_model_order(self) -> None:
        last, hidden = self.states()
        stacked = select_hidden_state_embeddings(last, hidden, store_all_hidden_states=True)
        assert stacked.shape == (2, 3, 3, 4)  # (b, n, l, d)
        assert [float(stacked[0, layer, 0, 0]) for layer in range(3)] == [0.0, 1.0, 2.0]

    def test_a_missing_hidden_state_tuple_is_an_error_when_one_is_needed(self) -> None:
        last, _ = self.states()
        with pytest.raises(ValueError, match="store_all_hidden_states requires model hidden states"):
            select_hidden_state_embeddings(last, None, store_all_hidden_states=True)
        with pytest.raises(ValueError, match="hidden_state_index requires model hidden states"):
            select_hidden_state_embeddings(last, (), hidden_state_index=0)


class TestCanonicalText:
    @pytest.mark.parametrize("text", ["MKV", "A", "ZZZ"])
    def test_uppercase_ascii_letters_pass_unchanged(self, text: str) -> None:
        check_canonical_text(text)

    @pytest.mark.parametrize("text", ["", "mkv", "MK V", "MK1", "MK*", "MKÉ", "mKV"])
    def test_anything_else_is_refused_without_being_normalized(self, text: str) -> None:
        with pytest.raises(ValueError, match="already normalized uppercase protein"):
            check_canonical_text(text)

    def test_residue_ids_are_one_non_special_token_per_letter(self) -> None:
        tokenizer = LetterTokenizer()
        assert canonical_residue_ids("ACD", tokenizer) == [4, 5, 6]

    def test_a_letter_the_tokenizer_maps_to_a_special_id_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no non-special tokenizer representation"):
            canonical_residue_ids("BJ", LetterTokenizer())

    def test_text_that_is_not_canonical_never_reaches_the_tokenizer(self) -> None:
        with pytest.raises(ValueError, match="already normalized"):
            canonical_residue_ids("acd", LetterTokenizer())


class TestHostBatch:
    def test_ids_rows_and_the_flat_selection_describe_the_padded_grid(self) -> None:
        vocabulary = ResidueVocabulary(LetterTokenizer())
        batch = build_host_batch(vocabulary, ["AC", "D"])
        assert isinstance(batch, HostBatch)
        assert batch.input_ids.tolist() == [[1, 4, 5, 2], [1, 6, 2, 0]]
        assert batch.rows.tolist() == [4, 3]
        assert batch.owner.tolist() == [0, 0, 0, 0, 1, 1, 1]
        assert batch.flat_index.tolist() == [0, 1, 2, 3, 4, 5, 6]

    def test_the_selection_skips_padding_columns_of_shorter_rows(self) -> None:
        vocabulary = ResidueVocabulary(LetterTokenizer())
        batch = build_host_batch(vocabulary, ["D", "ACD"])
        grid = batch.input_ids.reshape(-1)
        assert grid[batch.flat_index].tolist() == [1, 6, 2, 1, 4, 5, 6, 2]
        assert (batch.input_ids == vocabulary.pad_id).sum() == 2

    def test_an_unmappable_residue_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no non-special tokenizer representation"):
            build_host_batch(ResidueVocabulary(LetterTokenizer()), ["BZ"])

    def test_the_batch_plan_keeps_every_sequence_exactly_once(self) -> None:
        lengths = [5, 1, 9, 2]
        plans = list(plan_token_batches(lengths, max_sequences=2, max_tokens=64, window=4))
        assert sorted(index for plan in plans for index in plan) == [0, 1, 2, 3]
        assert all(len(plan) <= 2 for plan in plans)


class TestDistinctWithDigests:
    def test_repeats_collapse_in_first_seen_order_with_their_digests(self) -> None:
        ordered, keys = distinct_with_digests(["BB", "AA", "BB", "CC"], None)
        assert ordered == ["BB", "AA", "CC"]
        assert keys == [sequence_digest(text) for text in ordered]

    def test_supplied_digests_are_used_for_the_first_occurrence(self) -> None:
        ordered, keys = distinct_with_digests(["AA", "BB", "AA"], ["k1", "k2", "k3"])
        assert (ordered, keys) == (["AA", "BB"], ["k1", "k2"])

    def test_one_digest_per_sequence_is_required(self) -> None:
        with pytest.raises(ValueError, match="one key per sequence"):
            distinct_with_digests(["AA", "BB"], ["only-one"])

    def test_nothing_gives_nothing(self) -> None:
        assert distinct_with_digests([], None) == ([], [])


def hidden(name: str, layer: int, **fields) -> HiddenTap:
    return HiddenTap(name, layer, **fields)


class TestTapPlan:
    def reducer(self, batch: TapBatch) -> torch.Tensor:
        return batch.X.sum(dim=(1, 2))  # (b,)

    def test_layers_resolve_negative_indices_and_each_property_summarizes_the_plan(self) -> None:
        reduced = ReducedTap("sum", 1, self.reducer, {"kind": "sum"})
        streaming = StreamingTap("stream", (0, 2), lambda: None, {"kind": "stream"})  # type: ignore[arg-type, return-value]
        plan = plan_taps(
            [hidden("a", -1, pooling=("mean", "max")), hidden("b", 2), reduced, streaming], 4,
        )
        assert isinstance(plan, TapPlan)
        assert plan.layers == (3, 2, 1, 2)
        assert plan.captured_layers == (1, 2, 3)
        assert plan.streamed_layers == (0, 2)
        assert plan.deepest_layer == 3
        assert plan.pooling_names == frozenset({"mean", "max"})

    def test_captured_layers_are_distinct_and_exclude_streaming_taps(self) -> None:
        streaming = StreamingTap("stream", (0,), lambda: None, {"kind": "stream"})  # type: ignore[arg-type, return-value]
        plan = plan_taps([hidden("a", 1), hidden("b", -3), streaming], 4)
        assert plan.captured_layers == (1,)
        assert plan.pooling_names == frozenset()

    def test_identity_lists_each_tap_by_kind_in_plan_order(self) -> None:
        reduced = ReducedTap("sum", 1, self.reducer, {"kind": "sum"})
        sparse = SparseResidueTap(
            "codes", 2, lambda batch: [], {"kind": "codes"}, codebook_size=16, sparse_count=4,
        )
        plan = plan_taps([hidden("a", -1, pooling="mean", dtype=torch.bfloat16), reduced, sparse], 4)
        identity = plan.identity()
        assert [entry["kind"] for entry in identity] == ["hidden", "reduced", "sparse_residue"]
        assert identity[0] == {"name": "a", "kind": "hidden", "layer": 3, "pooling": ["mean"], "dtype": "bfloat16"}
        assert identity[1]["identity"] == {"kind": "sum"}
        assert (identity[2]["codebook_size"], identity[2]["sparse_count"]) == (16, 4)

    def test_a_plan_rejects_bad_taps_before_inference(self) -> None:
        with pytest.raises(TypeError, match="sequence of HiddenTap"):
            plan_taps("not taps", 4)
        with pytest.raises(ValueError, match="at least one tap"):
            plan_taps([], 4)
        with pytest.raises(ValueError, match="repeated"):
            plan_taps([hidden("a", 0), hidden("a", 1)], 4)
        with pytest.raises(ValueError, match="outside this model's hidden states -4..3"):
            plan_taps([hidden("a", 4)], 4)
        with pytest.raises(TypeError, match="found str"):
            plan_taps(["a"], 4)


class TestRowSelection:
    def test_gather_returns_the_kept_rows_in_sequence_order_without_padding(self) -> None:
        X = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)  # (b, l, d)
        mask = torch.tensor([[True, True, False], [False, True, True]])  # (b, l)
        batch = TapBatch(X=X, token_mask=mask, residue_mask=mask)
        selection = batch.selection()
        assert isinstance(selection, RowSelection)
        assert selection.counts == (2, 2)
        assert selection.owner.tolist() == [0, 0, 1, 1]
        assert selection.flat_index.tolist() == [0, 1, 4, 5]
        assert torch.equal(selection.gather(X), X[mask])

    def test_the_executors_selection_is_used_when_supplied(self) -> None:
        X = torch.zeros(1, 2, 3)
        mask = torch.ones(1, 2, dtype=torch.bool)
        rows = RowSelection(torch.tensor([1]), torch.tensor([0]), (1,), torch.tensor([1]))
        batch = TapBatch(X=X, token_mask=mask, residue_mask=mask, rows=rows)
        assert batch.selection() is rows


class TestBatchExecutor:
    def test_full_embeddings_are_the_residue_rows_of_each_record_in_source_order(self) -> None:
        model = TwoWideModel()
        records = [EmbeddingInput("long", "ACDE"), EmbeddingInput("short", "GK")]
        results, slices = executor(model, full=True, batch_size=1).run_window(records, window_start=0)
        assert slices == {}
        assert [record.id for record in results] == ["long", "short"]
        assert results[0].tensor.shape == (4, 2) and results[1].tensor.shape == (2, 2)
        assert results[1].tensor[:, 0].tolist() == [float(ord("G")), float(ord("K"))]

    def test_pooled_output_concatenates_the_poolers_and_reports_their_slices(self) -> None:
        model = TwoWideModel()
        pooler = Pooler(("mean", "max"))
        records = [EmbeddingInput("a", "AC"), EmbeddingInput("b", "DEF")]
        results, slices = executor(model, full=False, pooler=pooler).run_window(records, window_start=0)
        assert slices == {"mean": (0, 2), "max": (2, 4)}
        mean_a = (ord("A") + ord("C")) / 2
        assert results[0].tensor.tolist() == [mean_a, 1.5, float(ord("C")), 2.0]

    def test_a_window_inside_a_longer_run_keeps_its_own_positions(self) -> None:
        model = TwoWideModel()
        records = [EmbeddingInput("x", "AA"), EmbeddingInput("y", "CCC")]
        results, _ = executor(model, full=True).run_window(records, window_start=7)
        assert [record.id for record in results] == ["x", "y"]

    def test_truncation_crops_the_text_the_model_sees(self) -> None:
        model = TwoWideModel()
        executor(model, full=True, max_length=3, truncate=True).run_window([EmbeddingInput("a", "ACDEFG")], window_start=0)
        assert model.calls == [["ACD"]]

    def test_pooling_without_a_pooler_is_an_error(self) -> None:
        with pytest.raises(RuntimeError, match="without an initialized pooler"):
            executor(TwoWideModel(), full=False).run_window([EmbeddingInput("a", "AC")], window_start=0)

    def test_the_embedding_batch_must_be_finite_binary_and_matching(self) -> None:
        class BadBatches(TwoWideModel):
            def __init__(self, kind: str) -> None:
                super().__init__()
                self.kind = kind

            def _embedding_batch(self, sequences: list[str]) -> EmbeddingBatch:
                batch = super()._embedding_batch(sequences)
                if self.kind == "nan":
                    batch.X[0, 1, 0] = float("nan")
                elif self.kind == "mask":
                    batch.residue_mask[...] = False
                elif self.kind == "shape":
                    return EmbeddingBatch(X=batch.X[:, :-1], residue_mask=batch.residue_mask)
                elif self.kind == "int":
                    return EmbeddingBatch(X=batch.X.long(), residue_mask=batch.residue_mask)
                return batch

        window = [EmbeddingInput("a", "AC")]
        expectations = {
            "nan": "non-finite output",
            "mask": "biological residue",
            "shape": "must provide X with shape",
            "int": "floating-point",
        }
        for kind, message in expectations.items():
            with pytest.raises((ValueError, TypeError), match=message):
                executor(BadBatches(kind), full=True).run_window(window, window_start=0)

    def test_a_dtype_conversion_applies_to_the_returned_rows(self) -> None:
        results, _ = executor(TwoWideModel(), full=True, dtype=torch.float64).run_window(
            [EmbeddingInput("a", "AC")], window_start=0,
        )
        assert results[0].tensor.dtype == torch.float64


class TestEmbeddingOutput:
    def build(self, records: list[EmbeddingInput], **overrides) -> EmbeddingOutput:
        settings = {
            "output": None, "format": "safetensors", "resume": False, "shard_size": 1024,
            "run_fingerprint": "run-a", "input_fingerprint": "inputs-a", "model_state_fingerprint": None,
            "model_state_fingerprint_source": "none", "pooler": None, "pooling_names": (),
        }
        settings.update(overrides)
        return EmbeddingOutput(records, **settings)

    def results(self) -> list[EmbeddingRecord]:
        return [EmbeddingRecord("a", "AC", torch.ones(2, 2)), EmbeddingRecord("b", "D", torch.zeros(1, 2))]

    def test_an_in_memory_run_collects_windows_and_describes_each_tensor(self) -> None:
        records = [EmbeddingInput("a", "AC"), EmbeddingInput("b", "D")]
        output = self.build(records)
        assert output.completed is None and output.start_position == 0
        output.append(0, self.results()[:1])
        output.append(1, self.results()[1:])
        assert output.output_descriptors is not None
        assert [d["id"] for d in output.output_descriptors] == ["a", "b"]
        assert output.output_descriptors[0]["shape"] == (2, 2) and output.output_descriptors[0]["dtype"] == "float32"
        finished_records = output.finish({"run_fingerprint": "run-a", "complete": True})
        assert [record.id for record in finished_records] == ["a", "b"]

    @pytest.mark.parametrize("format", ["safetensors", "sqlite"])
    def test_a_finished_run_resumes_as_complete_and_a_changed_run_is_refused(self, tmp_path: Path, format: str) -> None:
        path = tmp_path / ("run.sqlite" if format == "sqlite" else "run")
        records = [EmbeddingInput("a", "AC"), EmbeddingInput("b", "D")]
        output = self.build(records, output=path, format=format)
        output.append(0, self.results())
        output.finish({"run_fingerprint": "run-a", "fingerprint_schema_version": _schema(), "complete": True})
        resumed = self.build(records, output=path, format=format, resume=True)
        assert resumed.completed is not None and len(resumed.completed) == 2
        with pytest.raises(ValueError, match="different run fingerprint"):
            self.build(records, output=path, format=format, resume=True, run_fingerprint="run-b")
        with pytest.raises(ValueError, match="not an ordered prefix"):
            self.build([EmbeddingInput("a", "AC")], output=path, format=format, resume=True)
        with pytest.raises(ValueError, match="not an ordered prefix"):
            self.build([EmbeddingInput("z", "AC"), EmbeddingInput("b", "D")], output=path, format=format, resume=True)


def _schema() -> int:
    from fastplms.embeddings.identity import _RUN_FINGERPRINT_SCHEMA_VERSION

    return _RUN_FINGERPRINT_SCHEMA_VERSION


class TestPoolerSlicesAndMixin:
    def test_output_slices_give_each_pooler_one_interval_in_request_order(self) -> None:
        assert Pooler(("max", "mean", "var")).output_slices(5) == {"max": (0, 5), "mean": (5, 10), "var": (10, 15)}

    @pytest.mark.parametrize("width", [0, -3])
    def test_a_non_positive_width_is_refused(self, width: int) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            Pooler("mean").output_slices(width)

    @pytest.mark.parametrize("width", [True, 2.0, "2"])
    def test_a_non_integer_width_is_a_type_error(self, width: object) -> None:
        with pytest.raises(TypeError, match="positive integer"):
            Pooler("mean").output_slices(width)  # type: ignore[arg-type]

    def test_the_mixin_delegates_to_embed_dataset(self) -> None:
        class Embedder(TwoWideModel, EmbeddingMixin):
            pass

        model = Embedder()
        via_mixin = model.embed_dataset(["AC", "DE"], full_embeddings=True, batch_size=2)
        direct = embed_dataset(model, ["AC", "DE"], full_embeddings=True, batch_size=2)
        assert [r.sequence for r in via_mixin] == [r.sequence for r in direct] == ["AC", "DE"]
        for left, right in zip(via_mixin, direct, strict=True):
            assert torch.equal(left.load_tensor(), right.load_tensor())


class TestProtocolsAreStructural:
    def test_a_plain_class_with_the_methods_satisfies_the_token_and_feature_contracts(self) -> None:
        from fastplms.embeddings.feature_runs import FeatureRunContract
        from fastplms.embeddings.taps import LayerAccumulator
        from fastplms.embeddings.token_runs import TokenFeatureContract

        class Accumulator:
            def update(self, layer: int, batch: TapBatch) -> None:
                self.last = layer

            def finish(self) -> torch.Tensor:
                return torch.zeros(1, 1, 1)  # (1, 1, 1)

        class Contract:
            def validate(self, *args: object) -> None: ...
            def validate_cached(self, *args: object) -> None: ...
            def bind_rows(self, name, records): return []
            def row_identities(self, name, sequences, digests): return [{} for _ in sequences]
            def check_row_keys(self, sequences, digests) -> None: ...
            def before_commit(self) -> None: ...

        accumulator: LayerAccumulator = Accumulator()
        contract: FeatureRunContract = Contract()
        token_contract: TokenFeatureContract = Contract()
        accumulator.update(2, TapBatch(torch.zeros(1, 1, 1), torch.ones(1, 1, dtype=torch.bool), torch.ones(1, 1, dtype=torch.bool)))
        assert accumulator.finish().shape == (1, 1, 1)
        assert contract.before_commit() is None and token_contract.check_row_keys([], []) is None
        assert token_contract.row_identities("t", ["AA"], ["d"]) == [{}]
