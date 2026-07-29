"""Fast CPU contracts for E1 retrieval caching and load context."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Barrier
from typing import Any
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from fastplms.models.e1 import modeling_e1 as e1_modeling
from fastplms.models.e1.attention import _get_unpad_data, _packed_lengths_cache_key
from fastplms.models.e1.cache import DynamicCache, KVCache
from fastplms.models.e1.modeling_e1 import (
    Attention,
    AttentionLayerType,
    E1Config,
    E1ForMaskedLM,
    E1ForSequenceClassification,
    E1ForTokenClassification,
    E1Model,
)


@dataclass
class _CacheableE1Output(ModelOutput):
    logits: torch.Tensor | None = None
    last_hidden_state: torch.Tensor | None = None
    embeddings: torch.Tensor | None = None
    token_embeddings: torch.Tensor | None = None
    past_key_values: DynamicCache | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None


def _dynamic_cache(sequence_length: int) -> DynamicCache:
    cache = DynamicCache()
    states = torch.arange(sequence_length * 2, dtype=torch.float32).reshape(  # (b=1, l, h=1, d_h=2)
        1, sequence_length, 1, 2
    )
    cache.update(states, states + 1, layer_idx=0)
    return cache


def _cache_batch(sequence_length: int = 5, context_length: int = 3) -> dict[str, Any]:
    token_values = torch.arange(sequence_length).unsqueeze(0)
    return {
        "context": ["ACD"],
        "context_len": [context_length],
        "use_cache": True,
        "input_ids": token_values.clone(),
        "within_seq_position_ids": token_values.clone(),
        "global_position_ids": token_values.clone(),
        "sequence_ids": token_values.clone(),
        "labels": token_values.clone(),
    }


def test_e1_cache_miss_slices_every_sequence_aligned_output_alias() -> None:
    cache = KVCache(cache_size=1)
    batch = _cache_batch()
    output_values = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1)
    outputs = _CacheableE1Output(
        logits=output_values + 10,
        last_hidden_state=output_values + 20,
        embeddings=output_values + 30,
        token_embeddings=output_values + 40,
        past_key_values=_dynamic_cache(sequence_length=5),
        hidden_states=(output_values + 50, output_values + 60),
    )

    cache.after_forward(batch, outputs)

    for field_name in cache.tensor_input_field_names:
        assert batch[field_name].shape[1] == 2
    for field_name in cache.tensor_output_field_names:
        value = outputs[field_name]
        assert value is not None
        assert value.shape[1] == 2
    assert isinstance(outputs.hidden_states, tuple)
    assert all(hidden_state.shape[1] == 2 for hidden_state in outputs.hidden_states)
    assert cache.cache_dict["ACD"].get_seq_length() == 3


def test_e1_cache_hit_does_not_slice_target_outputs_twice() -> None:
    cache = KVCache(cache_size=1)
    miss_batch = _cache_batch()
    miss_outputs = _CacheableE1Output(
        last_hidden_state=torch.zeros(1, 5, 2),
        past_key_values=_dynamic_cache(sequence_length=5),
    )
    cache.after_forward(miss_batch, miss_outputs)

    hit_batch = _cache_batch()
    cache.before_forward(hit_batch)
    assert hit_batch["input_ids"].shape[1] == 2
    assert hit_batch["past_key_values"] is cache.cache_dict["ACD"]

    hit_outputs = _CacheableE1Output(
        last_hidden_state=torch.ones(1, 2, 2),
        past_key_values=cache.cache_dict["ACD"],
    )
    cache.after_forward(hit_batch, hit_outputs)

    assert hit_outputs.last_hidden_state is not None
    assert hit_outputs.last_hidden_state.shape[1] == 2


def test_e1_from_pretrained_tokenizer_context_is_thread_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent loads must not exchange source, revision, cache, or token settings."""

    barrier = Barrier(2)

    def fake_from_pretrained(
        cls: type[E1ForMaskedLM],
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del pretrained_model_name_or_path, model_args, kwargs
        barrier.wait(timeout=5)
        observed = cls._tokenizer_kwargs_from_config(E1Config())
        barrier.wait(timeout=5)
        return observed

    monkeypatch.setattr(
        PreTrainedModel,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )

    load_specs = (
        ("model-a", "cache-a", "revision-a", "token-a"),
        ("model-b", "cache-b", "revision-b", "token-b"),
    )

    def load(spec: tuple[str, str, str, str]) -> dict[str, Any]:
        source, cache_dir, revision, token = spec
        return E1ForMaskedLM.from_pretrained(
            source,
            local_files_only=True,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        observed = list(executor.map(load, load_specs))

    for load_spec, tokenizer_kwargs in zip(load_specs, observed, strict=True):
        source, cache_dir, revision, token = load_spec
        assert tokenizer_kwargs == {
            "tokenizer_source": source,
            "local_files_only": True,
            "cache_dir": cache_dir,
            "revision": revision,
            "token": token,
        }


def test_e1_lazy_tokenizer_uses_resolved_weight_commit_per_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Moving weight revisions must resolve to the tokenizer's immutable commit."""

    load_barrier = Barrier(2)
    access_barrier = Barrier(2)
    commits = {
        "model-a": "a" * 40,
        "model-b": "b" * 40,
    }

    def fake_from_pretrained(
        cls: type[E1ForMaskedLM],
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ) -> E1ForMaskedLM:
        del model_args, kwargs
        config = _tiny_e1_config()
        config._commit_hash = commits[pretrained_model_name_or_path]
        load_barrier.wait(timeout=5)
        return cls(config)

    requests: list[dict[str, Any]] = []

    class RecordingPreparer:
        def __init__(self, *, data_prep_config: Any, **kwargs: Any) -> None:
            del data_prep_config
            access_barrier.wait(timeout=5)
            requests.append(kwargs)

    monkeypatch.setattr(
        PreTrainedModel,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )
    monkeypatch.setattr(e1_modeling, "E1BatchPreparer", RecordingPreparer)

    load_specs = (
        ("model-a", "cache-a", None, "token-a"),
        ("model-b", "cache-b", "main", "token-b"),
    )

    def load(spec: tuple[str, str, str | None, str]) -> E1ForMaskedLM:
        source, cache_dir, revision, token = spec
        return E1ForMaskedLM.from_pretrained(
            source,
            local_files_only=True,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        models = list(executor.map(load, load_specs))
    with ThreadPoolExecutor(max_workers=2) as executor:
        preparers = list(executor.map(lambda model: model.prep_tokens, models))

    assert all(isinstance(preparer, RecordingPreparer) for preparer in preparers)
    observed = {request["tokenizer_source"]: request for request in requests}
    for source, cache_dir, _requested_revision, token in load_specs:
        assert observed[source] == {
            "tokenizer_source": source,
            "local_files_only": True,
            "cache_dir": cache_dir,
            "revision": commits[source],
            "token": token,
        }


def _tiny_e1_config() -> E1Config:
    config = E1Config(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_num_sequences=4,
        max_num_positions_within_seq=32,
        max_num_positions_global=64,
        attn_backend="sdpa",
        dtype="float32",
        num_labels=3,
    )
    config.output_hidden_states = True
    config.use_cache = True
    return config


def _tiny_e1_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 5, 6, 2]], dtype=torch.long),
        "within_seq_position_ids": torch.arange(4).unsqueeze(0),
        "global_position_ids": torch.arange(4).unsqueeze(0),
        "sequence_ids": torch.zeros(1, 4, dtype=torch.long),
    }


def test_e1_sdpa_scalar_validation_runs_outside_torch_compile(
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = E1Model(_tiny_e1_config()).eval()
    first_batch = _tiny_e1_batch()
    second_batch = {
        **_tiny_e1_batch(),
        "within_seq_position_ids": torch.tensor([[0, 1, 0, 1]]),
        "sequence_ids": torch.tensor([[0, 0, 1, 1]]),
    }

    def run(
        input_ids: torch.Tensor,
        within_seq_position_ids: torch.Tensor,
        global_position_ids: torch.Tensor,
        sequence_ids: torch.Tensor,
    ) -> torch.Tensor:
        return model(
            input_ids=input_ids,
            within_seq_position_ids=within_seq_position_ids,
            global_position_ids=global_position_ids,
            sequence_ids=sequence_ids,
            use_cache=False,
            output_hidden_states=False,
        ).last_hidden_state

    with torch.inference_mode():
        expected_first = run(**first_batch)
        expected_second = run(**second_batch)

    compiled_graphs: list[torch.fx.GraphModule] = []

    def counting_backend(
        graph_module: torch.fx.GraphModule,
        example_inputs: list[torch.Tensor],
    ) -> Callable[..., object]:
        del example_inputs
        compiled_graphs.append(graph_module)
        return graph_module.forward

    compiled_run = torch.compile(run, backend=counting_backend, dynamic=False)
    try:
        with torch.inference_mode():
            actual_first = compiled_run(**first_batch)
            first_compile_count = len(compiled_graphs)
            actual_second = compiled_run(**second_batch)

            assert first_compile_count > 0
            assert len(compiled_graphs) == first_compile_count
            torch.testing.assert_close(actual_first, expected_first)
            torch.testing.assert_close(actual_second, expected_second)
            with pytest.raises(ValueError, match="Sequence ids must be in the range"):
                compiled_run(
                    **{
                        **first_batch,
                        "sequence_ids": torch.full(
                            (1, 4),
                            model.config.max_num_sequences,
                        ),
                    }
                )
    finally:
        torch.compiler.reset()

    captured = capsys.readouterr()
    assert "Graph break from `Tensor.item()`" not in captured.err


def test_e1_cached_query_and_packed_metadata_run_outside_torch_compile(
    capsys: pytest.CaptureFixture[str],
) -> None:
    attention = _tiny_attention(AttentionLayerType.GLOBAL)
    compiled_cached_ids = torch.compile(
        attention._cached_global_sequence_ids,
        backend="eager",
        dynamic=False,
    )

    def packed_metadata(
        sequence_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int, tuple[int, ...]]:
        indices, cumulative_lengths, maximum_length = _get_unpad_data(sequence_ids)
        sequence_lengths = cumulative_lengths[1:] - cumulative_lengths[:-1]
        cache_key = _packed_lengths_cache_key(sequence_lengths, sequence_lengths)
        return indices, cumulative_lengths, maximum_length, cache_key

    compiled_metadata = torch.compile(packed_metadata, backend="eager", dynamic=False)
    try:
        query_ids, key_ids = compiled_cached_ids(torch.tensor([[2, 2]]), 5)
        assert torch.equal(query_ids, torch.tensor([[2, 2]]))
        assert torch.equal(key_ids, torch.tensor([[2, 2, 2, 2, 2]]))
        with pytest.raises(ValueError, match="must start with a non-padding"):
            compiled_cached_ids(torch.tensor([[-1, 2]]), 5)

        indices, cumulative_lengths, maximum_length, cache_key = compiled_metadata(
            torch.tensor([[0, 0, 1, -1], [0, 0, 0, -1]])
        )
        assert torch.equal(indices, torch.tensor([0, 1, 2, 4, 5, 6]))
        assert torch.equal(cumulative_lengths, torch.tensor([0, 2, 3, 6], dtype=torch.int32))
        assert maximum_length == 3
        assert cache_key == (2, 1, 3, -1, 2, 1, 3)
    finally:
        torch.compiler.reset()

    captured = capsys.readouterr()
    assert "Graph break from `Tensor.item()`" not in captured.err


def test_e1_config_round_trip_preserves_cache_policy(tmp_path: Path) -> None:
    config = _tiny_e1_config()

    config.save_pretrained(tmp_path)
    restored = E1Config.from_pretrained(tmp_path, local_files_only=True)

    assert restored.use_cache is True


def test_e1_encoder_embedding_filters_training_only_preparer_fields() -> None:
    model = E1Model(_tiny_e1_config()).eval()

    with torch.inference_mode():
        hidden, token_mask = model._embed(["ACD", "G"], return_attention_mask=True)

    assert hidden.shape[:2] == token_mask.shape
    assert token_mask.sum(dim=-1).tolist() == [7, 5]


def _assert_nested_output_close(actual: Any, expected: Any) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual, expected)
        return
    if isinstance(expected, DynamicCache):
        assert isinstance(actual, DynamicCache)
        assert len(actual.key_cache) == len(expected.key_cache)
        assert len(actual.value_cache) == len(expected.value_cache)
        for actual_tensor, expected_tensor in zip(
            actual.key_cache + actual.value_cache,
            expected.key_cache + expected.value_cache,
            strict=True,
        ):
            torch.testing.assert_close(actual_tensor, expected_tensor)
        return
    if isinstance(expected, (tuple, list)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_output_close(actual_item, expected_item)
        return
    assert actual == expected


@pytest.mark.parametrize(
    "model_class",
    [E1Model, E1ForMaskedLM, E1ForSequenceClassification, E1ForTokenClassification],
)
def test_e1_public_models_honor_config_output_flags_and_return_dict(
    model_class: type[PreTrainedModel],
) -> None:
    model = model_class(_tiny_e1_config()).eval()
    batch = _tiny_e1_batch()

    with torch.inference_mode():
        structured = model(**batch, return_dict=True)
        tuple_output = model(**batch, return_dict=False)

    assert isinstance(structured, ModelOutput)
    assert structured.hidden_states is not None
    assert structured.past_key_values is not None
    assert isinstance(tuple_output, tuple)
    assert len(tuple_output) == len(structured.to_tuple())
    assert isinstance(tuple_output[0], torch.Tensor)
    assert tuple_output[0].shape == structured.to_tuple()[0].shape


@pytest.mark.parametrize(
    "model_class",
    [E1ForMaskedLM, E1ForSequenceClassification, E1ForTokenClassification],
)
def test_e1_loss_bearing_head_tuples_start_with_loss_then_logits(
    model_class: type[PreTrainedModel],
) -> None:
    model = model_class(_tiny_e1_config()).eval()
    batch = _tiny_e1_batch()
    if model_class is E1ForSequenceClassification:
        labels = torch.tensor([1], dtype=torch.long)
    elif model_class is E1ForTokenClassification:
        labels = batch["input_ids"].remainder(model.config.num_labels)
    else:
        labels = batch["input_ids"].clone()

    structured = model(**batch, labels=labels, return_dict=True)
    tuple_output = model(**batch, labels=labels, return_dict=False)

    assert structured.loss is not None
    torch.testing.assert_close(tuple_output[0], structured.loss)
    torch.testing.assert_close(tuple_output[1], structured.logits)
    if model_class is E1ForMaskedLM:
        assert structured.mlm_loss is not None
        assert structured.to_tuple()[2] is structured.hidden_states
    elif model_class is E1ForSequenceClassification:
        assert structured.to_tuple()[2] is structured.past_key_values
    else:
        assert structured.to_tuple()[2] is structured.hidden_states
    assert len(tuple_output) == len(structured.to_tuple())
    for actual, expected in zip(tuple_output, structured.to_tuple(), strict=True):
        _assert_nested_output_close(actual, expected)


@pytest.mark.parametrize(
    "model_class",
    [E1Model, E1ForMaskedLM, E1ForSequenceClassification, E1ForTokenClassification],
)
def test_e1_public_forwards_reject_unknown_arguments(
    model_class: type[PreTrainedModel],
) -> None:
    model = model_class(_tiny_e1_config()).eval()
    with pytest.raises(TypeError, match="unexpected_argument"):
        model(**_tiny_e1_batch(), unexpected_argument=True)


def test_e1_base_model_rejects_misaligned_biological_indices() -> None:
    model = E1Model(_tiny_e1_config()).eval()
    batch = _tiny_e1_batch()

    with pytest.raises(ValueError, match="Cannot specify both"):
        model(**batch, inputs_embeds=torch.zeros(1, 4, model.config.hidden_size))
    with pytest.raises(ValueError, match="sequence_ids must have shape"):
        model(**{**batch, "sequence_ids": torch.zeros(2, 2, dtype=torch.long)})
    with pytest.raises(ValueError, match="Global position ids must be in the range"):
        model(
            **{
                **batch,
                "global_position_ids": torch.full((1, 4), model.config.max_num_positions_global),
            }
        )
    with pytest.raises(ValueError, match="Sequence ids must be in the range"):
        model(
            **{
                **batch,
                "sequence_ids": torch.full((1, 4), model.config.max_num_sequences),
            }
        )


def test_e1_masked_lm_resizes_input_and_output_embeddings_together() -> None:
    model = E1ForMaskedLM(_tiny_e1_config()).eval()

    resized_input = model.resize_token_embeddings(39)

    assert resized_input.num_embeddings == 39
    assert model.get_input_embeddings().num_embeddings == 39
    assert model.get_output_embeddings().out_features == 39
    assert model.config.vocab_size == 39
    assert model.vocab_size == 39

    batch = _tiny_e1_batch()
    batch["input_ids"][0, 1] = 38
    with torch.inference_mode():
        output = model(**batch)
    assert output.logits.shape == (1, 4, 39)


def test_e1_legacy_backend_setter_rejects_unadvertised_backends() -> None:
    model = E1Model(_tiny_e1_config()).eval()

    with pytest.raises(ValueError, match="E1 does not support 'eager'"):
        model.attn_backend = "eager"


def _tiny_attention(layer_type: AttentionLayerType) -> Attention:
    config = E1Config(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_num_sequences=8,
        max_num_positions_within_seq=32,
        max_num_positions_global=32,
        global_attention_every_n_layers=(1 if layer_type == AttentionLayerType.GLOBAL else 0),
        dtype="float32",
        attn_backend="sdpa",
    )
    attention = Attention(config, layer_idx=0).eval()
    assert attention.layer_type == layer_type
    return attention


@pytest.mark.parametrize(
    "layer_type",
    (AttentionLayerType.WITHIN_SEQ, AttentionLayerType.GLOBAL),
)
def test_e1_cached_sdpa_preserves_layer_attention_semantics(
    layer_type: AttentionLayerType,
) -> None:
    torch.manual_seed(11)
    attention = _tiny_attention(layer_type)
    query = torch.randn(1, 2, 2, 4)
    key = torch.randn(1, 5, 2, 4)
    value = torch.randn(1, 5, 2, 4)
    sequence_ids = torch.tensor([[1, 1]])

    actual, _ = attention._sdpa_attn(
        query,
        key,
        value,
        sequence_ids=sequence_ids,
        effective_layer_type=layer_type,
        is_cache_prefilled=True,
    )

    expected_key = key[:, -2:] if layer_type == AttentionLayerType.WITHIN_SEQ else key
    expected_value = value[:, -2:] if layer_type == AttentionLayerType.WITHIN_SEQ else value
    mask = attention._cached_attention_mask_4d(
        sequence_ids,
        key.shape[1],
        layer_type,
    )
    expected_heads = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        expected_key.transpose(1, 2),
        expected_value.transpose(1, 2),
        attn_mask=mask,
    )
    expected = expected_heads.transpose(1, 2).reshape(1, 2, 8)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("layer_type", "expected_path", "expected_kv_length"),
    (
        (AttentionLayerType.WITHIN_SEQ, "dense", 2),
        (AttentionLayerType.GLOBAL, "packed", 5),
    ),
)
def test_e1_cached_flex_dispatch_keeps_or_discards_context_by_layer(
    monkeypatch: pytest.MonkeyPatch,
    layer_type: AttentionLayerType,
    expected_path: str,
    expected_kv_length: int,
) -> None:
    attention = _tiny_attention(layer_type)
    query = torch.randn(1, 2, 2, 4)
    key = torch.randn(1, 5, 2, 4)
    value = torch.randn(1, 5, 2, 4)
    observed: list[tuple[str, int]] = []

    def fake_dense(q, k, v, **kwargs):
        del v, kwargs
        observed.append(("dense", k.shape[1]))
        return torch.zeros_like(q)

    def fake_packed(q, k, v, **kwargs):
        del v, kwargs
        observed.append(("packed", k.shape[1]))
        return torch.zeros_like(q)

    monkeypatch.setattr(e1_modeling, "flex_attention_func", fake_dense)
    monkeypatch.setattr(e1_modeling, "varlen_flex_attention_func", fake_packed)

    attention._flex_attn(
        query,
        key,
        value,
        sequence_ids=torch.tensor([[1, 1]]),
        effective_layer_type=layer_type,
        is_cache_prefilled=True,
    )

    assert observed == [(expected_path, expected_kv_length)]
