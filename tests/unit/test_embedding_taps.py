"""One-pass taps: several hidden-state outputs from one ESM++ forward pass per batch.

Every check runs on a tiny randomly initialized ESM++ on CPU. A tap must equal the
single-output ``embed_dataset`` run that selects the same hidden state and pooling, and a pass
that stops early must equal a full-depth pass on every state it captures.
"""

from __future__ import annotations

import pytest
import torch

import fastplms.models.esm_plusplus.modeling_esm_plusplus as esmpp_module

from contextlib import contextmanager
from pathlib import Path
from torch import nn

from fastplms.embeddings import (
    HiddenTap,
    ReducedTap,
    StreamingTap,
    TapBatch,
    TapRecord,
    TapResult,
    embed_dataset,
)
from fastplms.models.esm3.modeling_esm3 import FastESM3Config, FastESM3Model
from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
    ESMplusplusConfig,
    ESMplusplusForMaskedLM,
    ESMplusplusForSequenceClassification,
    ESMplusplusForTokenClassification,
    ESMplusplusModel,
)


N_LAYERS = 3  # hidden states 0..3; state 3 is the final normalized state
HIDDEN_SIZE = 32  # d
SEQUENCES = ["MKTAYIAKQR", "GG", "MSTNPKPQRKTKRNT", "ACD", "GG"]
ESMPP_CLASSES = (
    ESMplusplusModel,
    ESMplusplusForMaskedLM,
    ESMplusplusForSequenceClassification,
    ESMplusplusForTokenClassification,
)


def _model(model_class: type[nn.Module] = ESMplusplusModel) -> nn.Module:
    torch.manual_seed(0)
    config = ESMplusplusConfig(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=4,
        num_hidden_layers=N_LAYERS,
        attn_backend="sdpa",
    )
    return model_class(config).eval()


def _residue_sum(batch: TapBatch) -> torch.Tensor:
    # batch.X: (b, l, d); batch.residue_mask: (b, l)
    return (batch.X * batch.residue_mask.unsqueeze(-1)).sum(dim=1)  # (b, d)


def _tokens(model: nn.Module, sequences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = model.tokenizer(sequences, return_tensors="pt", padding=True)
    return encoded["input_ids"], encoded["attention_mask"]  # each: (b, l)


@pytest.mark.parametrize("model_class", ESMPP_CLASSES)
def test_each_hidden_tap_equals_its_single_output_run(model_class: type[nn.Module]) -> None:
    model = _model(model_class)
    taps = [
        HiddenTap("last", layer=-1),
        HiddenTap("last_meanvar", layer=-1, pooling=("mean", "var")),
        HiddenTap("final_by_index", layer=N_LAYERS, pooling="mean"),
        HiddenTap("mid_max", layer=1, pooling="max"),
        HiddenTap("second_to_last_rows", layer=-2),
        HiddenTap("embedding_rows", layer=0),
        HiddenTap("cls", layer=2, pooling=("cls",)),
    ]
    single_output_runs = {
        "last": {"full_embeddings": True},
        "last_meanvar": {"pooling": ("mean", "var")},
        "final_by_index": {"pooling": "mean", "hidden_state_index": N_LAYERS},
        "mid_max": {"pooling": "max", "hidden_state_index": 1},
        "second_to_last_rows": {"full_embeddings": True, "hidden_state_index": -2},
        "embedding_rows": {"full_embeddings": True, "hidden_state_index": 0},
        "cls": {"pooling": "cls", "hidden_state_index": 2},
    }

    features = embed_dataset(model, SEQUENCES, batch_size=2, taps=taps)

    for name, options in single_output_runs.items():
        reference = embed_dataset(model, SEQUENCES, batch_size=2, **options)
        for record, expected in zip(features, reference, strict=True):
            assert torch.equal(record.tensors[name], expected.load_tensor()), name


def test_tap_results_keep_input_order_duplicates_and_residue_shapes() -> None:
    features = embed_dataset(
        _model(),
        [("b", "GG"), ("a", "MKTAYIAKQR"), ("b", "GG")],
        batch_size=2,
        taps=[HiddenTap("rows", layer=1), HiddenTap("pooled", layer=1, pooling=("mean", "max"))],
    )

    assert isinstance(features, TapResult)
    assert [record.id for record in features] == ["b", "a", "b"]
    assert [record.sequence for record in features] == ["GG", "MKTAYIAKQR", "GG"]
    assert [tuple(record.tensors["rows"].shape) for record in features] == [
        (2, HIDDEN_SIZE),
        (10, HIDDEN_SIZE),
        (2, HIDDEN_SIZE),
    ]
    assert all(record.tensors["pooled"].shape == (2 * HIDDEN_SIZE,) for record in features)
    # Length bucketing pads the two copies differently, so they agree to rounding.
    torch.testing.assert_close(features[0].tensors["rows"], features[2].tensors["rows"])
    with pytest.raises(TypeError):
        features[0].tensors["rows"] = torch.zeros(1)  # type: ignore[index]


def test_tap_metadata_records_the_plan_depth_and_pool_slices() -> None:
    features = embed_dataset(
        _model(),
        ["ACD"],
        taps=[
            HiddenTap("mid", layer=1, pooling=("mean", "var")),
            ReducedTap("summed", layer=-3, reduce=_residue_sum, identity={"reducer": "sum"}),
        ],
    )

    assert features.metadata["taps"] == {
        "plan": [
            {
                "name": "mid", "kind": "hidden", "layer": 1,
                "pooling": ["mean", "var"], "dtype": None,
            },
            {"name": "summed", "kind": "reduced", "layer": 1, "identity": {"reducer": "sum"}},
        ],
        "stop_after_layer": 1,
        "pool_slices": {"mid": {"mean": (0, HIDDEN_SIZE), "var": (HIDDEN_SIZE, 2 * HIDDEN_SIZE)}},
    }
    assert features.metadata["pooling"] == []
    assert features.metadata["layer"] is None
    assert features.metadata["storage_format"] == "memory"
    assert features.metadata["descriptor_index"] == "not-recorded"


def test_a_plan_runs_one_forward_pass_per_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model()
    calls: list[tuple[int, tuple[int, ...]]] = []
    original = model._embed_taps

    def counting(input_ids, attention_mask, layers):
        calls.append((int(input_ids.shape[0]), layers))
        return original(input_ids, attention_mask, layers)

    def single_output(*args, **kwargs):
        raise AssertionError("a tap plan must not run the single-output path")

    monkeypatch.setattr(model, "_embed_taps", counting)
    monkeypatch.setattr(model, "_embed", single_output)

    embed_dataset(
        model,
        SEQUENCES,
        batch_size=2,
        taps=[
            HiddenTap("rows", layer=1),
            HiddenTap("pooled", layer=-1, pooling="mean"),
            ReducedTap("summed", layer=1, reduce=_residue_sum, identity={"reducer": "sum"}),
        ],
    )

    assert calls == [(2, (1, N_LAYERS)), (2, (1, N_LAYERS)), (1, (1, N_LAYERS))]


def test_mixed_tap_dtypes_start_from_the_same_original_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model().double()
    sequences = ["MKTAYIAKQR", "ACD", "G"]
    input_ids, attention_mask = _tokens(model, sequences)
    with torch.inference_mode():
        reference = model._embed_taps(input_ids, attention_mask, (N_LAYERS,))[N_LAYERS]
    calls = []
    original = model._embed_taps

    def counting(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "_embed_taps", counting)
    embeddings = embed_dataset(
        model, sequences, batch_size=3, dtype=torch.float16,
        taps=[
            HiddenTap("rows", -1, dtype=torch.bfloat16),
            HiddenTap("pooled", -1, ("mean", "var"), dtype=torch.float32),
            HiddenTap("inherited", -1),
        ],
    )
    assert calls == [True]
    for i, (record, sequence) in enumerate(zip(embeddings, sequences, strict=True)):
        residues = reference[i, 1 : len(sequence) + 1]
        assert torch.equal(record.tensors["rows"], residues.bfloat16())
        assert torch.equal(record.tensors["inherited"], residues.half())
        # This fails if the shared state is first quantized through the run's FP16 dtype.
        fp32 = residues.float()
        expected = torch.cat((fp32.mean(dim=0), fp32.var(dim=0, correction=0)))
        torch.testing.assert_close(record.tensors["pooled"], expected, rtol=1e-6, atol=1e-7)
        assert record.tensors["pooled"].dtype == torch.float32
    assert embeddings.metadata["pooling_semantics"]["variance_correction"] == 0
    assert embeddings.metadata["pooling_semantics"]["version"] == 2


def test_tap_crop_matches_explicit_prefix_and_unpadded_singletons() -> None:
    model = _model().double()
    sequences = ["MKTAYIAKQR", "G", "ACDEF"]
    taps = [HiddenTap("rows", -1), HiddenTap("pooled", -1, ("mean", "var"))]
    cropped = embed_dataset(model, sequences, taps=taps, dtype=None, batch_size=3, max_length=3)
    prefixes = embed_dataset(
        model, [sequence[:3] for sequence in sequences], taps=taps, dtype=None, batch_size=1
    )
    for actual, expected in zip(cropped, prefixes, strict=True):
        for name in ("rows", "pooled"):
            torch.testing.assert_close(
                actual.tensors[name], expected.tensors[name], rtol=1e-12, atol=1e-12
            )
    assert [record.tensors["rows"].shape[0] for record in cropped] == [3, 1, 3]
    assert torch.count_nonzero(cropped[1].tensors["pooled"][HIDDEN_SIZE:]) == 0
    assert cropped.metadata["retained_positions"] == (
        "biological_residues_in_input_order_after_optional_prefix_crop_before_forward"
    )


def test_tap_rejects_output_cast_overflow_and_overlength_before_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    calls = []

    def overflowing(input_ids, attention_mask, layers):
        calls.append(True)
        return {layer: torch.full((*input_ids.shape, HIDDEN_SIZE), 100_000.0) for layer in layers}

    monkeypatch.setattr(model, "_embed_taps", overflowing)
    with pytest.raises(ValueError, match="max_length=2"):
        embed_dataset(model, ["ACD"], taps=[HiddenTap("rows", -1)], max_length=2, truncate=False)
    assert calls == []
    with pytest.raises(ValueError, match="dtype conversion produced non-finite"):
        embed_dataset(model, ["ACD"], taps=[HiddenTap("rows", -1, dtype=torch.float16)])
    assert calls == [True]


@pytest.mark.parametrize("layer", range(N_LAYERS + 1))
def test_an_early_stop_equals_the_full_pass_and_runs_no_later_block(layer: int) -> None:
    model = _model()
    input_ids, attention_mask = _tokens(model, ["MKTAYIAKQR", "ACD"])  # each: (2, 12)
    ran: list[int | str] = []
    for index, block in enumerate(model.transformer.blocks):
        block.register_forward_hook(lambda module, args, output, index=index: ran.append(index))
    model.transformer.norm.register_forward_hook(lambda module, args, output: ran.append("norm"))

    with torch.inference_mode():
        captured = model._embed_taps(input_ids, attention_mask, (layer,))  # each: (2, 12, d)
        stopped_calls = list(ran)
        reference = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        ).hidden_states  # N_LAYERS + 1 states, each (2, 12, d)

    assert torch.equal(captured[layer], reference[layer])
    # No block after the stop runs, and the final norm runs only for the final state.
    expected_calls: list[int | str] = list(range(layer))
    if layer == N_LAYERS:
        expected_calls.append("norm")
    assert stopped_calls == expected_calls


def test_a_stopped_stack_equals_full_depth_capture_for_every_captured_state() -> None:
    model = _model()
    input_ids, attention_mask = _tokens(model, SEQUENCES)  # each: (5, 17)
    with torch.inference_mode():
        x = model.embed(input_ids)  # (5, 17, d)
        stopped = model.transformer(
            x=x,
            attention_mask=attention_mask,
            capture_layers=(0, 2),
            stop_after_layer=2,
        )
        full = model.transformer(
            x=x,
            attention_mask=attention_mask,
            capture_layers=(0, 2, N_LAYERS),
        )

    assert stopped.last_hidden_state is None
    assert set(stopped.captured_hidden_states) == {0, 2}
    for index in (0, 2):
        assert torch.equal(
            stopped.captured_hidden_states[index],
            full.captured_hidden_states[index],
        )
    assert torch.equal(full.captured_hidden_states[N_LAYERS], full.last_hidden_state)


def test_captures_leave_the_sae_state_contract_unchanged() -> None:
    model = _model()
    input_ids, attention_mask = _tokens(model, ["ACD"])  # each: (1, 5)
    with torch.inference_mode():
        x = model.embed(input_ids)  # (1, 5, d)
        sae_only = model.transformer(x=x, attention_mask=attention_mask, sae_layers=(1, N_LAYERS))
        both = model.transformer(
            x=x,
            attention_mask=attention_mask,
            sae_layers=(1, N_LAYERS),
            capture_layers=(0, 2),
        )

    assert set(sae_only.sae_hidden_states) == set(both.sae_hidden_states) == {1, N_LAYERS}
    assert sae_only.captured_hidden_states is None
    assert set(both.captured_hidden_states) == {0, 2}
    for index in (1, N_LAYERS):
        assert torch.equal(sae_only.sae_hidden_states[index], both.sae_hidden_states[index])


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"stop_after_layer": N_LAYERS + 1}, "stop_after_layer must be"),
        ({"stop_after_layer": -1}, "stop_after_layer must be"),
        ({"capture_layers": (2,), "stop_after_layer": 1}, "outside the states 0..1"),
        ({"capture_layers": (N_LAYERS + 1,)}, "outside the states"),
        ({"sae_layers": (2,), "stop_after_layer": 1}, "outside the states 0..1"),
        ({"output_hidden_states": True, "stop_after_layer": 1}, "only captured hidden states"),
        ({"output_attentions": True, "stop_after_layer": 1}, "only captured hidden states"),
    ],
)
def test_the_stack_rejects_captures_it_does_not_compute(options: dict, match: str) -> None:
    model = _model()
    x = torch.zeros(1, 4, HIDDEN_SIZE)  # (1, 4, d)

    with pytest.raises(ValueError, match=match):
        model.transformer(x=x, attention_mask=torch.ones(1, 4, dtype=torch.bool), **options)


def test_taps_follow_the_fp8_padding_context_and_trim(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model()
    input_ids, attention_mask = _tokens(model, ["MKT"])  # each: (1, 5)
    with torch.inference_mode():
        unpadded = model._embed_taps(input_ids, attention_mask, (1, N_LAYERS))  # each: (1, 5, d)

    model._esmc_fp8 = True
    seen_lengths: list[int] = []
    entered: list[bool] = []
    original_forward = model.transformer.forward

    def recording_forward(*args, **kwargs):
        seen_lengths.append(int(kwargs["x"].shape[1]))
        return original_forward(*args, **kwargs)

    @contextmanager
    def recording_context(enabled: bool, device: torch.device):
        entered.append(enabled)
        yield

    monkeypatch.setattr(model.transformer, "forward", recording_forward)
    monkeypatch.setattr(esmpp_module, "_esmplusplus_fp8_context", recording_context)

    with torch.inference_mode():
        captured = model._embed_taps(input_ids, attention_mask, (1, N_LAYERS))  # each: (1, 5, d)

    assert seen_lengths == [16], "FP8 pads the token axis to a multiple of 16"
    assert entered == [True]
    assert {layer: tuple(state.shape) for layer, state in captured.items()} == {
        1: (1, 5, HIDDEN_SIZE),
        N_LAYERS: (1, 5, HIDDEN_SIZE),
    }
    for layer in (1, N_LAYERS):
        torch.testing.assert_close(captured[layer], unpadded[layer])


def test_a_reducer_receives_the_tapped_state_and_both_masks() -> None:
    model = _model()
    seen: list[TapBatch] = []

    def reduce(batch: TapBatch) -> torch.Tensor:
        seen.append(batch)
        return _residue_sum(batch)  # (b, d)

    features = embed_dataset(
        model,
        SEQUENCES,
        batch_size=len(SEQUENCES),
        taps=[ReducedTap("summed", layer=1, reduce=reduce, identity={"reducer": "sum"})],
    )
    rows = embed_dataset(
        model,
        SEQUENCES,
        batch_size=len(SEQUENCES),
        full_embeddings=True,
        hidden_state_index=1,
    )

    (batch,) = seen
    assert batch.X.shape == (len(SEQUENCES), 17, HIDDEN_SIZE)  # longest: 15 residues, BOS, EOS
    assert batch.token_mask.dtype == batch.residue_mask.dtype == torch.bool
    # Length bucketing orders the batch longest first; ties keep input order.
    assert batch.residue_mask.sum(dim=1).tolist() == [15, 10, 3, 2, 2]
    assert (batch.token_mask.sum(dim=1) - batch.residue_mask.sum(dim=1)).tolist() == [2] * 5
    for record, expected in zip(features, rows, strict=True):
        torch.testing.assert_close(record.tensors["summed"], expected.load_tensor().sum(dim=0))


@pytest.mark.parametrize(
    ("reduce", "error", "match"),
    [
        (lambda batch: batch.X[:1].sum(dim=1), ValueError, "one row per sequence"),
        (lambda batch: batch.X.sum(), ValueError, "one row per sequence"),
        (lambda batch: [0.0] * batch.X.shape[0], TypeError, "must return a Tensor"),
        (lambda batch: torch.full((batch.X.shape[0], 2), float("nan")), ValueError, "non-finite"),
    ],
)
def test_reducer_outputs_are_validated(reduce, error: type[Exception], match: str) -> None:
    tap = ReducedTap("bad", layer=1, reduce=reduce, identity={"reducer": "bad"})

    with pytest.raises(error, match=match):
        embed_dataset(_model(), ["ACD", "GG"], batch_size=2, taps=[tap])


def test_the_run_fingerprint_records_the_tap_plan() -> None:
    model = _model()

    def fingerprint(taps: list) -> str:
        return embed_dataset(model, ["ACD", "GG"], taps=taps).metadata["run_fingerprint"]

    base = fingerprint([HiddenTap("a", layer=1, pooling="mean")])
    assert fingerprint([HiddenTap("a", layer=1, pooling="mean")]) == base
    assert fingerprint([HiddenTap("a", layer=1 - (N_LAYERS + 1), pooling="mean")]) == base, (
        "a negative index naming the same state is the same plan"
    )
    changed = [
        [HiddenTap("a", layer=2, pooling="mean")],
        [HiddenTap("a", layer=1, pooling="max")],
        [HiddenTap("a", layer=1, pooling="mean", dtype=torch.bfloat16)],
        [HiddenTap("a", layer=1)],
        [HiddenTap("b", layer=1, pooling="mean")],
        [HiddenTap("a", layer=1, pooling="mean"), HiddenTap("b", layer=0)],
        [ReducedTap("a", layer=1, reduce=_residue_sum, identity={"reducer": "sum"})],
        [ReducedTap("a", layer=1, reduce=_residue_sum, identity={"reducer": "sum", "version": 2})],
    ]
    fingerprints = [fingerprint(taps) for taps in changed]
    assert len({base, *fingerprints}) == len(changed) + 1
    single_output = embed_dataset(model, ["ACD", "GG"], pooling="mean", hidden_state_index=1)
    assert single_output.metadata["run_fingerprint"] != base


def _unconsumed_inputs(state: dict[str, bool]):
    state["consumed"] = True
    yield "ACD"


@pytest.mark.parametrize(
    ("taps", "options", "error", "match"),
    [
        ([], {}, ValueError, "at least one tap"),
        ([HiddenTap("a", layer=0), HiddenTap("a", layer=1)], {}, ValueError, "unique"),
        ([HiddenTap("a", layer=N_LAYERS + 1)], {}, ValueError, "outside this model's hidden"),
        ([HiddenTap("a", layer=-(N_LAYERS + 2))], {}, ValueError, "outside this model's hidden"),
        ("last", {}, TypeError, "sequence of HiddenTap"),
        ([object()], {}, TypeError, "HiddenTap, ReducedTap, StreamingTap or SparseResidueTap"),
        ([HiddenTap("a", layer=1)], {"pooling": "mean"}, ValueError, "combined with pooling"),
        ([HiddenTap("a", layer=1)], {"full_embeddings": True}, ValueError, "full_embeddings"),
        ([HiddenTap("a", layer=1)], {"hidden_state_index": 1}, ValueError, "hidden_state_index"),
        (
            [HiddenTap("a", layer=1)],
            {"store_all_hidden_states": True},
            ValueError,
            "store_all_hidden_states",
        ),
        ([HiddenTap("a", layer=1)], {"output_scale": 2}, ValueError, "no model keyword arguments"),
    ],
)
def test_invalid_plans_fail_before_consuming_inputs(
    taps: object,
    options: dict,
    error: type[Exception],
    match: str,
) -> None:
    state = {"consumed": False}

    with pytest.raises(error, match=match):
        embed_dataset(_model(), _unconsumed_inputs(state), taps=taps, **options)
    assert state["consumed"] is False


def test_output_points_at_the_feature_store_instead_of_writing_tap_records(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="embed_into_features"):
        embed_dataset(_model(), ["ACD"], taps=[HiddenTap("a", layer=1)], output=tmp_path / "taps")
    assert not (tmp_path / "taps").exists()


def test_other_families_reject_taps_instead_of_falling_back() -> None:
    model = FastESM3Model(
        FastESM3Config(
            hidden_size=8,
            num_attention_heads=2,
            num_vector_heads=2,
            num_hidden_layers=1,
            attn_backend="eager",
        )
    ).eval()

    with pytest.raises(ValueError, match="FastESM3Model does not support taps="):
        embed_dataset(model, ["ACD"], taps=[HiddenTap("a", layer=-1)])


@pytest.mark.parametrize(
    ("build", "error", "match"),
    [
        (lambda: HiddenTap("", layer=0), ValueError, "non-empty string"),
        (lambda: HiddenTap("a", layer=True), TypeError, "integer hidden-state index"),
        (lambda: HiddenTap("a", layer=0, pooling="parti"), ValueError, "attention graph"),
        (lambda: HiddenTap("a", layer=0, pooling=("mean", "mean")), ValueError, "Duplicate"),
        (lambda: HiddenTap("a", layer=0, pooling=("median2",)), ValueError, "Unknown pooling"),
        (lambda: HiddenTap("a", layer=0, dtype=torch.int64), ValueError, "hidden tap dtype"),
        (lambda: HiddenTap("a", layer=0, dtype="float32"), ValueError, "hidden tap dtype"),
        (lambda: ReducedTap("a", 0, None, {"reducer": "x"}), TypeError, "callable reduce"),
        (lambda: ReducedTap("a", 0, _residue_sum, {}), TypeError, "non-empty identity"),
        (lambda: ReducedTap("a", 0, _residue_sum, {"fn": _residue_sum}), TypeError, "a function"),
        (lambda: ReducedTap("a", 0, _residue_sum, {"x": float("nan")}), ValueError, "non-finite"),
        (lambda: ReducedTap("a", 0, _residue_sum, {1: "x"}), TypeError, "non-string key"),
    ],
)
def test_taps_validate_their_own_fields(build, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        build()


def test_a_reduced_tap_keeps_a_private_copy_of_its_identity() -> None:
    identity = {"reducer": "sum", "settings": [1, 2]}
    tap = ReducedTap("a", layer=0, reduce=_residue_sum, identity=identity)
    identity["reducer"] = "changed"
    identity["settings"].append(3)

    assert tap.identity == {"reducer": "sum", "settings": [1, 2]}


def test_tap_records_validate_their_fields() -> None:
    with pytest.raises(ValueError, match="id"):
        TapRecord("", "ACD", {"a": torch.zeros(1)})
    with pytest.raises(TypeError, match="non-empty mapping"):
        TapRecord("a", "ACD", {})
    with pytest.raises(TypeError, match="map tap names to Tensor or TopKRow values"):
        TapRecord("a", "ACD", {"a": [0.0]})


@pytest.mark.parametrize("model_class", ESMPP_CLASSES)
def test_streaming_stops_early_and_multiple_reducers_share_one_pass(model_class) -> None:
    model = _model(model_class)
    class Sum:
        def __init__(self):
            self.value = None
        def update(self, layer, batch):
            self.value = batch.X.clone() if self.value is None else self.value + batch.X
        def finish(self):
            return self.value
    calls = []
    handles = [
        block.register_forward_pre_hook(lambda module, args, i=i: calls.append(i))
        for i, block in enumerate(model.transformer.blocks)
    ]
    try:
        embeddings = embed_dataset(model, ["ACDE"], taps=[
            StreamingTap("sum", (0, 2), Sum, {"reducer": "sum"}),
            StreamingTap("first", (0,), Sum, {"reducer": "first"}),
        ])
    finally:
        for handle in handles:
            handle.remove()
    assert calls == [0, 1]
    ids, mask = _tokens(model, ["ACDE"])
    with torch.inference_mode():
        expected = model(
            input_ids=ids, attention_mask=mask, output_hidden_states=True,
        ).hidden_states
    torch.testing.assert_close(embeddings[0].tensors["sum"], (expected[0] + expected[2])[0, 1:-1])
    torch.testing.assert_close(embeddings[0].tensors["first"], expected[0][0, 1:-1])


def test_streaming_trims_alignment_padding_before_callback(monkeypatch):
    model = _model()
    ids, mask = _tokens(model, ["ACD"])
    model._esmc_fp8 = True
    @contextmanager
    def fake_context(enabled, device):
        yield
    monkeypatch.setattr(esmpp_module, "_esmplusplus_fp8_context", fake_context)
    states = {}
    with torch.inference_mode():
        retained = model._embed_taps(
            ids, mask, (), stream_layers=(0, 3),
            state_consumer=lambda layer, X: states.update({layer: X.clone()}),
        )
    assert retained == {}
    assert {layer: tuple(X.shape) for layer, X in states.items()} == {
        0: (1, 5, HIDDEN_SIZE), 3: (1, 5, HIDDEN_SIZE),
    }


@pytest.mark.parametrize("options", [
    {"stream_layers": (0,)}, {"state_consumer": lambda layer, X: None},
    {"stream_layers": (3,), "state_consumer": lambda layer, X: None, "stop_after_layer": 1},
])
def test_stack_rejects_incomplete_stream_requests(options):
    model = _model()
    with pytest.raises(ValueError):
        model.transformer(torch.zeros(1, 4, HIDDEN_SIZE), **options)
