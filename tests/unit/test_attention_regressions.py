"""Focused CPU regressions for attention dispatch, masks, and caches."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest
import torch
from transformers.models.esm.configuration_esm import EsmConfig

import fastplms.models.ankh.modeling_ankh as ankh_module
import fastplms.models.e1.modeling_e1 as e1_module
import fastplms.models.esm2.modeling_fastesm as esm2_module
import fastplms.models.esm3.modeling_esm3 as esm3_module
import fastplms.models.esm_plusplus.modeling_esm_plusplus as esmpp_module
from fastplms.attention import AttentionBackend, _core, _kernel_lock
from fastplms.models.ankh.modeling_ankh import (
    AnkhSelfAttention,
    FastAnkhConfig,
    FastAnkhForMaskedLMExtension,
)
from fastplms.models.dplm.modeling_dplm import (
    DPLMConfig,
    ModifiedEsmEncoder,
)
from fastplms.models.dplm2.modeling_dplm2 import (
    DPLM2Config,
)
from fastplms.models.dplm2.modeling_dplm2 import (
    ModifiedEsmEncoder as DPLM2ModifiedEsmEncoder,
)
from fastplms.models.dplm2.modeling_dplm2 import (
    ModifiedEsmSelfAttention as DPLM2ModifiedEsmSelfAttention,
)
from fastplms.models.e1.attention import build_block_causal_mask_4d
from fastplms.models.e1.modeling_e1 import (
    FAST_E1_ENCODER,
    E1Config,
)
from fastplms.models.e1.modeling_e1 import (
    Attention as E1Attention,
)
from fastplms.models.e1.modeling_e1 import (
    AttentionArgs as E1AttentionArgs,
)
from fastplms.models.e1.modeling_e1 import (
    AttentionLayerType as E1AttentionLayerType,
)
from fastplms.models.esm2.modeling_fastesm import (
    EsmEncoder,
    FastEsmConfig,
    FastEsmPreTrainedModel,
)
from fastplms.models.esm2.modeling_fastesm import (
    EsmSelfAttention as FastEsmSelfAttention,
)
from fastplms.models.esm3.modeling_esm3 import FastESM3Config, FastESM3Model
from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
    MultiHeadAttention as ESMplusplusMultiHeadAttention,
)
from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
    TransformerStack,
)
from fastplms.models.esmfold.modeling_fast_esmfold import (
    EsmSelfAttention as EsmFoldSelfAttention,
)
from fastplms.models.esmfold.modeling_fast_esmfold import (
    FastEsmEncoder as EsmFoldEncoder,
)
from fastplms.models.ttt import LoraInjectedLinear


@pytest.mark.parametrize(
    "invalid_mask",
    (
        torch.ones(2, 4, 1, dtype=torch.bool),
        torch.ones(2, 3, dtype=torch.bool),
        torch.ones(1, 4, dtype=torch.bool),
    ),
)
def test_attention_masks_require_exact_batch_sequence_shape(
    invalid_mask: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match=r"attention_mask.*shape"):
        _core.get_attention_mask(
            AttentionBackend.EAGER,
            batch_size=2,
            seq_len=4,
            device=torch.device("cpu"),
            attention_mask=invalid_mask,
        )


@pytest.mark.parametrize("backend", tuple(AttentionBackend))
def test_attention_masks_reject_rows_without_valid_keys_before_dispatch(
    backend: AttentionBackend,
) -> None:
    with pytest.raises(
        ValueError,
        match="attention_mask must keep at least one valid key per batch row",
    ):
        _core.get_attention_mask(
            backend,
            batch_size=2,
            seq_len=4,
            device=torch.device("cpu"),
            attention_mask=torch.tensor(((1, 1, 0, 0), (0, 0, 0, 0))),
        )


def test_esmplusplus_rejects_empty_attention_rows_without_fallback_or_mutation() -> None:
    stack = TransformerStack(d_model=8, n_heads=2, n_layers=1, attn_backend="sdpa").eval()
    configured_backend = stack.attention_backend

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(
            ValueError,
            match="attention_mask must keep at least one valid key per batch row",
        ):
            stack(
                torch.randn(2, 4, 8),
                attention_mask=torch.tensor(((1, 1, 0, 0), (0, 0, 0, 0))),
                output_attentions=True,
            )

    assert captured == []
    assert stack.attention_backend == configured_backend


def _assert_masked_output_attentions_fallback(encoder: torch.nn.Module) -> None:
    hidden_states = torch.randn(1, 4, 8)
    attention_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
    configured_backend = encoder.attention_backend
    backend_name = configured_backend.value

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        output = encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=True,
        )

    assert len(captured) == 1
    assert issubclass(captured[0].category, RuntimeWarning)
    warning = str(captured[0].message)
    assert "output_attentions=True" in warning
    assert repr(backend_name) in warning
    assert "using 'eager'" in warning
    assert encoder.attention_backend == configured_backend
    assert output.attentions is not None
    assert len(output.attentions) == len(encoder.layer)
    for attention_weights in output.attentions:
        assert attention_weights is not None
        assert torch.count_nonzero(attention_weights[..., 2:]) == 0


def test_esm2_output_attentions_fallback_preserves_padding_mask() -> None:
    encoder = EsmEncoder(
        FastEsmConfig(
            vocab_size=16,
            hidden_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            position_embedding_type="absolute",
            attn_backend="flex_attention",
        )
    ).eval()
    _assert_masked_output_attentions_fallback(encoder)


def test_dplm_output_attentions_fallback_preserves_padding_mask() -> None:
    encoder = ModifiedEsmEncoder(
        DPLMConfig(
            vocab_size=16,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            position_embedding_type="absolute",
            attn_backend="flash_attention_3",
        )
    ).eval()
    _assert_masked_output_attentions_fallback(encoder)


def test_dplm2_output_attentions_fallback_is_call_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = DPLM2Config(
        vocab_size=64,
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=16,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        position_embedding_type="absolute",
        attn_backend="sdpa",
    )
    encoder = DPLM2ModifiedEsmEncoder(config).eval()
    hidden_states = torch.randn(2, 4, 8)
    attention_mask = torch.tensor(((1, 1, 1, 0), (1, 1, 0, 0)), dtype=torch.long)
    configured_backend = encoder.attention_backend
    configured_layer_backends = tuple(layer.attention.self.attn_backend for layer in encoder.layer)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        fallback_output = encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=True,
        )

    assert len(captured) == 1
    assert issubclass(captured[0].category, RuntimeWarning)
    warning = str(captured[0].message)
    assert "output_attentions=True" in warning
    assert "requested 'sdpa'" in warning
    assert "using 'eager'" in warning
    assert "call only" in warning
    assert config.attn_backend == "sdpa"
    assert encoder.attention_backend == configured_backend
    assert (
        tuple(layer.attention.self.attn_backend for layer in encoder.layer)
        == configured_layer_backends
    )
    assert torch.isfinite(fallback_output.last_hidden_state).all()
    assert fallback_output.attentions is not None
    assert len(fallback_output.attentions) == 2
    invalid_keys = attention_mask[:, None, None, :].logical_not()
    for attention_weights in fallback_output.attentions:
        assert attention_weights is not None
        assert attention_weights.shape == (2, 2, 4, 4)
        assert torch.isfinite(attention_weights).all()
        masked_weights = attention_weights.masked_select(invalid_keys.expand_as(attention_weights))
        assert torch.equal(masked_weights, torch.zeros_like(masked_weights))

    dispatches: list[DPLM2ModifiedEsmSelfAttention] = []
    original_sdpa = DPLM2ModifiedEsmSelfAttention._sdpa_attn

    def record_sdpa(self, *args, **kwargs):
        dispatches.append(self)
        return original_sdpa(self, *args, **kwargs)

    monkeypatch.setattr(DPLM2ModifiedEsmSelfAttention, "_sdpa_attn", record_sdpa)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        subsequent_output = encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=False,
        )

    assert dispatches == [layer.attention.self for layer in encoder.layer]
    assert torch.isfinite(subsequent_output.last_hidden_state).all()
    assert subsequent_output.attentions is None
    assert config.attn_backend == "sdpa"
    assert encoder.attention_backend == configured_backend
    assert (
        tuple(layer.attention.self.attn_backend for layer in encoder.layer)
        == configured_layer_backends
    )


def test_e1_output_attentions_fallback_preserves_block_causal_mask_and_backend() -> None:
    config = E1Config(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_num_sequences=4,
        max_num_positions_within_seq=16,
        max_num_positions_global=32,
        global_attention_every_n_layers=1,
        attn_backend="flex_attention",
        dtype="float32",
    )
    attention = E1Attention(config, layer_idx=0).eval()
    assert attention.layer_type == E1AttentionLayerType.GLOBAL
    configured_backend = attention.attn_backend
    sequence_ids = torch.tensor(((0, 0, 1, 1),), dtype=torch.long)
    attention_args = E1AttentionArgs(block_causal_mask_4d=build_block_causal_mask_4d(sequence_ids))
    query = torch.randn(1, 4, 2, 4)
    key = torch.randn(1, 4, 2, 4)
    value = torch.randn(1, 4, 2, 4)

    with pytest.warns(
        RuntimeWarning,
        match=r"requested 'flex_attention'.*using 'eager'.*call only",
    ) as fallback_warnings:
        output, weights, s_max = attention._attn(
            query,
            key,
            value,
            sequence_ids=sequence_ids,
            attention_args=attention_args,
            output_attentions=True,
        )

    assert len(fallback_warnings) == 1
    assert attention.attn_backend == configured_backend
    assert output.shape == (1, 4, 8)
    assert torch.isfinite(output).all()
    assert weights is not None
    assert weights.shape == (1, 2, 4, 4)
    assert torch.isfinite(weights).all()
    assert torch.count_nonzero(weights[:, :, :2, 2:]) == 0
    assert torch.count_nonzero(weights[:, :, 2:, :2]) == weights[:, :, 2:, :2].numel()
    assert s_max is None


def test_e1_public_fallback_warns_once_masks_padding_and_preserves_flex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = E1Config(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_num_sequences=4,
        max_num_positions_within_seq=16,
        max_num_positions_global=32,
        global_attention_every_n_layers=2,
        attn_backend="flex_attention",
        dtype="float32",
    )
    model = FAST_E1_ENCODER(config).eval()
    configured_backend = model._attn_backend
    configured_layer_backends = tuple(
        layer.norm_attn_norm.self_attn.attn_backend for layer in model.layers
    )
    inputs_embeds = torch.randn(2, 5, 8)
    within_positions = torch.tensor(((0, 1, 0, 1, -1), (0, 1, 2, -1, -1)))
    global_positions = torch.tensor(((0, 1, 2, 3, -1), (0, 1, 2, -1, -1)))
    sequence_ids = torch.tensor(((0, 0, 1, 1, -1), (0, 0, 0, -1, -1)))

    with pytest.warns(
        RuntimeWarning,
        match=r"requested 'flex_attention'.*using 'eager'.*call only",
    ) as fallback_warnings:
        fallback_output = model(
            inputs_embeds=inputs_embeds,
            within_seq_position_ids=within_positions,
            global_position_ids=global_positions,
            sequence_ids=sequence_ids,
            output_attentions=True,
        )

    assert len(fallback_warnings) == 1
    assert model._attn_backend == configured_backend
    assert config.attn_backend == "flex_attention"
    assert (
        tuple(layer.norm_attn_norm.self_attn.attn_backend for layer in model.layers)
        == configured_layer_backends
    )
    assert torch.isfinite(fallback_output.last_hidden_state).all()
    assert fallback_output.attentions is not None
    expected_masks = (
        e1_module.build_within_seq_mask_4d(sequence_ids),
        e1_module.build_block_causal_mask_4d(sequence_ids),
    )
    for attention_weights, expected_mask in zip(
        fallback_output.attentions,
        expected_masks,
        strict=True,
    ):
        assert torch.isfinite(attention_weights).all()
        expanded_mask = expected_mask.expand_as(attention_weights)
        masked_weights = attention_weights.masked_select(~expanded_mask)
        assert torch.equal(masked_weights, torch.zeros_like(masked_weights))

    within_block_mask = object()
    global_block_mask = object()
    flex_dispatches: list[dict[str, object]] = []
    monkeypatch.setattr(
        e1_module,
        "create_within_seq_block_mask",
        lambda _sequence_ids: within_block_mask,
    )
    monkeypatch.setattr(
        e1_module,
        "create_block_causal_mask_optimized",
        lambda _sequence_ids: global_block_mask,
    )

    def fake_flex_attention(query, _key, _value, **kwargs):
        flex_dispatches.append(kwargs)
        return query

    monkeypatch.setattr(e1_module, "flex_attention_func", fake_flex_attention)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        optimized_output = model(
            inputs_embeds=inputs_embeds,
            within_seq_position_ids=within_positions,
            global_position_ids=global_positions,
            sequence_ids=sequence_ids,
            output_attentions=False,
        )

    assert torch.isfinite(optimized_output.last_hidden_state).all()
    assert [dispatch["block_mask"] for dispatch in flex_dispatches] == [
        within_block_mask,
        global_block_mask,
    ]
    assert [dispatch["mask_semantics"] for dispatch in flex_dispatches] == [
        E1AttentionLayerType.WITHIN_SEQ.value,
        E1AttentionLayerType.GLOBAL.value,
    ]
    assert model._attn_backend == configured_backend
    assert (
        tuple(layer.norm_attn_norm.self_attn.attn_backend for layer in model.layers)
        == configured_layer_backends
    )


def test_esm3_sdpa_fallback_is_call_scoped_and_preserves_padding_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FastESM3Model(
        FastESM3Config(
            hidden_size=8,
            num_attention_heads=2,
            num_vector_heads=2,
            num_hidden_layers=2,
            attn_backend="sdpa",
        )
    ).eval()
    input_ids = torch.tensor(((0, 3, 4, 2, 1), (0, 6, 2, 1, 1)))
    attention_mask = input_ids.ne(1)
    configured_stack_backend = model.esm3.transformer.attention_backend
    configured_layer_backends = tuple(
        block.attn.attn_backend for block in model.esm3.transformer.blocks
    )

    with pytest.warns(
        RuntimeWarning,
        match=r"requested 'sdpa'.*using 'eager'.*call only",
    ) as fallback_warnings:
        fallback_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )

    assert len(fallback_warnings) == 1
    assert model.attn_backend == "sdpa"
    assert model.config._attn_implementation == "sdpa"
    assert model.esm3.transformer.attention_backend == configured_stack_backend
    assert (
        tuple(block.attn.attn_backend for block in model.esm3.transformer.blocks)
        == configured_layer_backends
    )
    assert torch.isfinite(fallback_output.last_hidden_state).all()
    assert fallback_output.attentions is not None
    invalid_keys = ~attention_mask[:, None, None, :]
    for attention_weights in fallback_output.attentions:
        assert torch.isfinite(attention_weights).all()
        masked_weights = attention_weights.masked_select(invalid_keys.expand_as(attention_weights))
        assert torch.equal(masked_weights, torch.zeros_like(masked_weights))

    sdpa_masks: list[torch.Tensor] = []

    def fake_sdpa(query, _key, _value, **kwargs):
        sdpa_masks.append(kwargs["attn_mask"].detach().clone())
        return query

    monkeypatch.setattr(esm3_module.F, "scaled_dot_product_attention", fake_sdpa)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        optimized_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
        )

    assert torch.isfinite(optimized_output.last_hidden_state).all()
    expected_padding_mask = attention_mask[:, None, None, :]
    assert len(sdpa_masks) == 2
    for observed_mask in sdpa_masks:
        assert torch.equal(observed_mask, expected_padding_mask)
    assert model.attn_backend == "sdpa"
    assert model.esm3.transformer.attention_backend == configured_stack_backend
    assert (
        tuple(block.attn.attn_backend for block in model.esm3.transformer.blocks)
        == configured_layer_backends
    )


def test_esm3_sequence_id_grouping_combines_with_public_padding_mask() -> None:
    model = FastESM3Model(
        FastESM3Config(
            hidden_size=8,
            num_attention_heads=2,
            num_vector_heads=2,
            num_hidden_layers=1,
            attn_backend="eager",
        )
    ).eval()
    input_ids = torch.tensor(((0, 3, 4, 2, 1), (0, 6, 2, 1, 1)))
    attention_mask = input_ids.ne(1)
    sequence_id = torch.tensor(((0, 0, 1, 1, -1), (0, 1, 1, -1, -1)))

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        sequence_id=sequence_id,
        output_attentions=True,
    )

    assert output.attentions is not None
    expected_mask = (
        sequence_id.unsqueeze(-1).eq(sequence_id.unsqueeze(-2)).unsqueeze(1)
        & attention_mask[:, None, None, :]
    )
    attention_weights = output.attentions[0]
    assert torch.isfinite(attention_weights).all()
    masked_weights = attention_weights.masked_select(~expected_mask.expand_as(attention_weights))
    assert torch.equal(masked_weights, torch.zeros_like(masked_weights))
    assert torch.count_nonzero(attention_weights[0, :, :2, 2:4]) == 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("attention_mask", torch.ones(2, 4), r"attention_mask must have shape"),
        ("attention_mask", torch.tensor(((1, 1, 2), (1, 0, 0))), r"only boolean or 0/1"),
        ("sequence_id", torch.zeros(1, 3, dtype=torch.long), r"sequence_id must have shape"),
    ),
)
def test_esm3_rejects_malformed_padding_and_sequence_masks(
    field: str,
    value: torch.Tensor,
    message: str,
) -> None:
    model = FastESM3Model(
        FastESM3Config(
            hidden_size=8,
            num_attention_heads=2,
            num_vector_heads=2,
            num_hidden_layers=1,
            attn_backend="eager",
        )
    ).eval()
    inputs = {
        "input_ids": torch.tensor(((0, 3, 2), (0, 4, 2))),
        field: value,
    }

    with pytest.raises(ValueError, match=message):
        model(**inputs)


def test_dplm_sdpa_output_attentions_fallback_preserves_cross_attention_mask_and_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = ModifiedEsmEncoder(
        DPLMConfig(
            vocab_size=16,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            position_embedding_type="absolute",
            is_decoder=True,
            add_cross_attention=True,
            attn_backend="sdpa",
        )
    ).eval()
    cross_attention = encoder.layer[0].crossattention.self
    configured_backend = encoder.attention_backend
    additive_encoder_mask = torch.tensor([[[[0.0, 0.0, -10_000.0, -10_000.0]]]])
    observed: dict[str, torch.Tensor] = {}
    original_manual_attention = cross_attention._manual_attn

    def record_cross_attention(query, key, value, attention_mask_4d=None, output_s_max=False):
        result = original_manual_attention(
            query,
            key,
            value,
            attention_mask_4d,
            output_s_max,
        )
        observed["mask"] = attention_mask_4d.detach().clone()
        observed["weights"] = result[1].detach().clone()
        return result

    monkeypatch.setattr(cross_attention, "_manual_attn", record_cross_attention)
    with pytest.warns(
        RuntimeWarning,
        match=r"requested 'sdpa'.*using 'eager'.*call only",
    ) as fallback_warnings:
        output = encoder(
            torch.randn(1, 3, 8),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            encoder_hidden_states=torch.randn(1, 4, 8),
            encoder_attention_mask=additive_encoder_mask,
            output_attentions=True,
        )

    assert len(fallback_warnings) == 1
    assert encoder.attention_backend == configured_backend
    assert encoder.layer[0].attention.self.attn_backend == configured_backend
    assert cross_attention.attn_backend == configured_backend
    assert torch.equal(observed["mask"], additive_encoder_mask)
    assert torch.equal(observed["weights"][..., 2:], torch.zeros_like(observed["weights"][..., 2:]))
    assert torch.isfinite(output.last_hidden_state).all()


def test_esmfold_output_attentions_fallback_preserves_padding_mask() -> None:
    encoder = EsmFoldEncoder(
        EsmConfig(
            vocab_size=16,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            position_embedding_type="absolute",
            attn_backend="flex_attention",
        )
    ).eval()
    _assert_masked_output_attentions_fallback(encoder)


def test_esmfold_flex_training_rejects_unimplemented_attention_dropout() -> None:
    attention = EsmFoldSelfAttention(
        EsmConfig(
            hidden_size=8,
            num_attention_heads=2,
            attention_probs_dropout_prob=0.1,
            position_embedding_type="absolute",
            attn_backend="flex_attention",
        )
    ).train()
    heads = torch.randn(1, 2, 3, 4)

    with pytest.raises(RuntimeError, match=r"inference-only.*dropout.*eager or SDPA"):
        attention._attn(heads, heads, heads)


def test_esmplusplus_chain_masks_fail_closed_without_assertions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = TransformerStack(d_model=8, n_heads=2, n_layers=1, attn_backend="sdpa")

    with pytest.raises(ValueError, match=r"sequence_id must have shape \(2, 4\)"):
        stack._sequence_id_attention_masks(
            sequence_id=torch.ones(2, 4, 1, dtype=torch.bool),
            batch_size=2,
            seq_len=4,
            device=torch.device("cpu"),
        )

    stack.attention_backend = AttentionBackend.FLASH_ATTENTION_3
    with pytest.raises(ValueError, match="only supports boolean sequence_id"):
        stack._sequence_id_attention_masks(
            sequence_id=torch.zeros(2, 4, dtype=torch.long),
            batch_size=2,
            seq_len=4,
            device=torch.device("cpu"),
        )

    stack.attention_backend = AttentionBackend.FLEX_ATTENTION
    monkeypatch.setattr(_core, "create_block_mask", None)
    with pytest.raises(RuntimeError, match="create_block_mask is unavailable"):
        stack._sequence_id_attention_masks(
            sequence_id=torch.ones(2, 4, dtype=torch.bool),
            batch_size=2,
            seq_len=4,
            device=torch.device("cpu"),
        )


def test_flash_attention_2_dense_and_varlen_use_autograd_wrappers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[dict[str, object]] = []

    def low_level(*_args, **_kwargs):
        raise AssertionError("low-level FlashAttention entry points must not be called")

    def dense(**kwargs):
        observed.append(kwargs)
        return kwargs["q"] + kwargs["k"] + kwargs["v"]

    def varlen(**kwargs):
        observed.append(kwargs)
        return kwargs["q"] + kwargs["k"] + kwargs["v"]

    kernel = SimpleNamespace(
        fwd=low_level,
        varlen_fwd=low_level,
        flash_attn_func=dense,
        flash_attn_varlen_func=varlen,
    )
    monkeypatch.setattr(
        _core,
        "_ensure_flash_kernels_loaded",
        lambda _implementation: (kernel, "flash_attn2"),
    )
    query = torch.randn(1, 3, 2, 4, requires_grad=True)
    key = torch.randn(1, 3, 2, 4, requires_grad=True)
    value = torch.randn(1, 3, 2, 4, requires_grad=True)

    dense_output = _core._kernels_flash_forward(
        query,
        key,
        value,
        implementation="flash_attention_2",
    )
    dense_output.sum().backward()
    assert all(tensor.grad is not None for tensor in (query, key, value))
    assert observed[-1]["dropout_p"] == 0.0

    flat_query = query.detach().reshape(3, 2, 4).requires_grad_()
    flat_key = key.detach().reshape(3, 2, 4).requires_grad_()
    flat_value = value.detach().reshape(3, 2, 4).requires_grad_()
    cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)
    varlen_output = _core._kernels_flash_varlen_forward(
        flat_query,
        flat_key,
        flat_value,
        cu_seqlens,
        cu_seqlens,
        3,
        3,
        implementation="flash_attention_2",
    )
    varlen_output.sum().backward()
    assert all(tensor.grad is not None for tensor in (flat_query, flat_key, flat_value))
    assert observed[-1]["dropout_p"] == 0.0


def test_flash_attention_2_dense_and_varlen_preserve_lora_and_input_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatch_count = {"dense": 0, "varlen": 0}

    def dense(**kwargs):
        dispatch_count["dense"] += 1
        return kwargs["q"] + kwargs["k"] + kwargs["v"]

    def varlen(**kwargs):
        dispatch_count["varlen"] += 1
        return kwargs["q"] + kwargs["k"] + kwargs["v"]

    kernel = SimpleNamespace(
        flash_attn_func=dense,
        flash_attn_varlen_func=varlen,
    )
    monkeypatch.setattr(
        _core,
        "_ensure_flash_kernels_loaded",
        lambda _implementation: (kernel, "flash_attn2"),
    )
    monkeypatch.setattr(
        _core,
        "_validate_kernels_flash_device",
        lambda query, _key, _value, _implementation: query.device,
    )
    monkeypatch.setattr(
        _core,
        "_validate_kernels_flash_dtype",
        lambda query, _key, _value, _implementation: query.dtype,
    )

    projections = torch.nn.ModuleList(
        [
            LoraInjectedLinear(
                torch.nn.Linear(8, 8, bias=False),
                rank=2,
                alpha=1.0,
            )
            for _ in range(3)
        ]
    )
    down_weights = torch.arange(1, 17, dtype=torch.float32).reshape(2, 8) / 16
    with torch.no_grad():
        for projection in projections:
            projection.linear.weight.copy_(torch.eye(8))
            projection.lora_down.weight.copy_(down_weights)
            projection.lora_up.weight.fill_(0.05)

    def project(hidden_states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(projection(hidden_states).reshape(2, 4, 2, 4) for projection in projections)

    dense_input = (torch.arange(1, 65, dtype=torch.float32).reshape(2, 4, 8) / 64).requires_grad_()
    padded_input = (
        torch.arange(65, 129, dtype=torch.float32).reshape(2, 4, 8) / 128
    ).requires_grad_()
    dense_output = _core.kernels_flash_attention_func(
        *project(dense_input),
        implementation="flash_attention_2",
    )
    attention_mask = torch.tensor(
        [[True, True, True, True], [True, True, False, False]],
    )
    padded_output = _core.kernels_flash_attention_func(
        *project(padded_input),
        attention_mask_2d=attention_mask,
        implementation="flash_attention_2",
    )

    (dense_output.sum() + padded_output.sum()).backward()

    assert dispatch_count == {"dense": 1, "varlen": 1}
    for hidden_states in (dense_input, padded_input):
        assert hidden_states.grad is not None
        assert torch.isfinite(hidden_states.grad).all()
        assert torch.count_nonzero(hidden_states.grad) > 0
    assert torch.count_nonzero(padded_input.grad[1, 2:]) == 0
    for projection in projections:
        assert projection.linear.weight.grad is None
        for parameter in (projection.lora_down.weight, projection.lora_up.weight):
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert torch.count_nonzero(parameter.grad) > 0


def test_flash_attention_2_rejects_low_level_only_kernel_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = SimpleNamespace(
        fwd=lambda **kwargs: kwargs["q"],
        varlen_fwd=lambda **kwargs: kwargs["q"],
    )
    monkeypatch.setattr(_core, "load_locked_kernel", lambda *_args: kernel)

    with pytest.raises(RuntimeError, match="autograd-enabled flash_attn_func"):
        _core._load_kernels_flash("flash_attention_2")


@pytest.mark.parametrize("varlen", (False, True))
def test_flash_attention_3_preserves_internal_type_errors(
    monkeypatch: pytest.MonkeyPatch,
    varlen: bool,
) -> None:
    failure = TypeError("internal FlashAttention kernel failure")
    calls = 0

    def fail(**_kwargs):
        nonlocal calls
        calls += 1
        raise failure

    kernel = SimpleNamespace(
        flash_attn_func=fail,
        flash_attn_varlen_func=fail,
    )
    monkeypatch.setattr(
        _core,
        "_ensure_flash_kernels_loaded",
        lambda _implementation: (kernel, "flash_attn3"),
    )
    query = torch.randn(1, 3, 2, 4)

    with pytest.raises(TypeError) as captured:
        if varlen:
            flat_query = query.reshape(3, 2, 4)
            cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)
            _core._kernels_flash_varlen_forward(
                flat_query,
                flat_query,
                flat_query,
                cu_seqlens,
                cu_seqlens,
                3,
                3,
                implementation="flash_attention_3",
            )
        else:
            _core._kernels_flash_forward(
                query,
                query,
                query,
                implementation="flash_attention_3",
            )

    assert captured.value is failure
    assert calls == 1


@pytest.mark.parametrize("variable", ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))
def test_kernel_loader_honors_all_offline_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setenv(variable, "yes")

    assert _kernel_lock._offline_mode()


def test_public_flex_cache_cleanup_is_scoped_and_complete() -> None:
    _core._compiled_flex_attention[("compiled",)] = object()
    _core._flex_block_masks[("mask",)] = object()

    _core.clear_flex_attention_caches()

    assert not _core._compiled_flex_attention
    assert not _core._flex_block_masks


def _ankh_attention(dropout_rate: float) -> AnkhSelfAttention:
    config = FastAnkhConfig(
        vocab_size=16,
        d_model=8,
        d_kv=4,
        d_ff=16,
        num_heads=2,
        num_layers=1,
        dropout_rate=dropout_rate,
        attn_backend="sdpa",
    )
    return AnkhSelfAttention(config)


@pytest.mark.parametrize(
    ("training", "expected_calls"),
    ((True, [(0.25, True)]), (False, [])),
)
def test_ankh_output_attentions_eager_fallback_honors_attention_dropout(
    monkeypatch: pytest.MonkeyPatch,
    training: bool,
    expected_calls: list[tuple[float, bool]],
) -> None:
    attention = _ankh_attention(dropout_rate=0.25).train(training)
    attention.attn_backend = AttentionBackend.SDPA
    dropout_calls: list[tuple[float, bool]] = []

    def record_dropout(
        tensor: torch.Tensor,
        *,
        p: float,
        training: bool,
    ) -> torch.Tensor:
        dropout_calls.append((p, training))
        return tensor

    monkeypatch.setattr(ankh_module.F, "dropout", record_dropout)
    hidden_states = torch.randn(1, 3, 8)

    with pytest.warns(RuntimeWarning, match="output_attentions=True"):
        output, attention_weights, _ = attention(
            hidden_states,
            output_attentions=True,
        )

    assert output.shape == hidden_states.shape
    assert attention_weights is not None
    assert dropout_calls == expected_calls


@pytest.mark.parametrize(
    ("training", "expected_dropout"),
    ((True, 0.25), (False, 0.0)),
)
def test_ankh_sdpa_receives_training_attention_dropout(
    monkeypatch: pytest.MonkeyPatch,
    training: bool,
    expected_dropout: float,
) -> None:
    attention = _ankh_attention(dropout_rate=0.25).train(training)
    observed_dropout: list[float] = []

    def record_sdpa(
        query: torch.Tensor,
        _key: torch.Tensor,
        _value: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        observed_dropout.append(kwargs["dropout_p"])
        return query

    monkeypatch.setattr(ankh_module.F, "scaled_dot_product_attention", record_sdpa)
    heads = torch.randn(1, 2, 3, 4)

    output = attention._sdpa_attn(heads, heads, heads, None)

    assert output.shape == (1, 3, 8)
    assert observed_dropout == [expected_dropout]


def test_ankh_ttt_missing_input_uses_optimization_safe_validation() -> None:
    model = object.__new__(FastAnkhForMaskedLMExtension)

    with pytest.raises(ValueError, match="either seq or input_ids"):
        model._ttt_tokenize()


def test_esm2_attention_validates_head_divisibility_without_assertions() -> None:
    config = FastEsmConfig(
        hidden_size=10,
        num_attention_heads=3,
        attention_probs_dropout_prob=0.0,
        attn_backend="eager",
    )

    with pytest.raises(ValueError, match="not a multiple"):
        FastEsmSelfAttention(config)


def test_esm2_legacy_backend_setter_uses_explicit_validation() -> None:
    model = FastEsmPreTrainedModel(FastEsmConfig(attn_backend="sdpa"))

    with pytest.raises(ValueError, match="does not support"):
        model.attn_backend = "not_an_attention_backend"


def test_bool_to_additive_mask_rejects_non_boolean_input_explicitly() -> None:
    with pytest.raises(TypeError, match="requires a bool tensor"):
        _core.bool_to_additive_mask(torch.ones(1, 2), torch.float32)


def test_esm2_flex_unavailability_raises_explicit_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention = FastEsmSelfAttention(
        FastEsmConfig(
            hidden_size=8,
            num_attention_heads=2,
            attention_probs_dropout_prob=0.0,
            attn_backend="sdpa",
        )
    )
    attention.attn_backend = AttentionBackend.FLEX_ATTENTION
    monkeypatch.setattr(esm2_module, "flex_attention", None)
    heads = torch.randn(1, 2, 3, 4)

    with pytest.raises(RuntimeError, match="Flex attention is not available"):
        attention._flex_attn(heads, heads, heads)


def test_esmplusplus_flex_unavailability_raises_explicit_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention = ESMplusplusMultiHeadAttention(
        d_model=8,
        n_heads=2,
        attn_backend="sdpa",
    )
    attention.attn_backend = AttentionBackend.FLEX_ATTENTION
    monkeypatch.setattr(esmpp_module, "flex_attention", None)
    heads = torch.randn(1, 2, 3, 4)

    with pytest.raises(RuntimeError, match="Flex attention is not available"):
        attention._flex_attn(heads, heads, heads)
