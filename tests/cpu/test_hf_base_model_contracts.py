"""Hugging Face backbone, prefix-loading, and encoder-only CPU contracts."""

from __future__ import annotations

import json
import pytest
import torch
from collections.abc import Callable
from pathlib import Path
from transformers import PretrainedConfig, PreTrainedModel

from fastplms.models.ankh.modeling_ankh import FastAnkhConfig, FastAnkhModel
from fastplms.models.dplm.modeling_dplm import (
    FAST_DPLM_ENCODER,
    DPLMConfig,
    DPLMForMaskedLM,
    DPLMForSequenceClassification,
    DPLMForTokenClassification,
    DPLMModel,
)
from fastplms.models.dplm2.modeling_dplm2 import (
    FAST_DPLM2_ENCODER,
    DPLM2Config,
    DPLM2ForMaskedLM,
    DPLM2ForSequenceClassification,
    DPLM2ForTokenClassification,
    DPLM2Model,
)
from fastplms.models.esm2.modeling_fastesm import (
    FAST_ESM_ENCODER,
    FastEsmConfig,
    FastEsmForMaskedLM,
    FastEsmForSequenceClassification,
    FastEsmForTokenClassification,
    FastEsmModel,
)
from fastplms.models.esm3.modeling_esm3 import FastESM3Config, FastESM3Model
from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
    ESMplusplusConfig,
    ESMplusplusModel,
)


def _esm2_config() -> FastEsmConfig:
    return FastEsmConfig(
        vocab_size=16,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=16,
        pad_token_id=1,
        mask_token_id=5,
        num_labels=3,
        position_embedding_type="absolute",
        add_pooling_layer=False,
        attn_backend="eager",
        use_cache=False,
    )


def _dplm_config() -> DPLMConfig:
    return DPLMConfig(
        vocab_size=33,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=16,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=32,
        num_labels=3,
        position_embedding_type="rotary",
        add_pooling_layer=False,
        attn_backend="eager",
        use_cache=False,
    )


def _dplm2_config() -> DPLM2Config:
    return DPLM2Config(
        vocab_size=64,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=16,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=32,
        num_labels=3,
        position_embedding_type="rotary",
        add_pooling_layer=False,
        attn_backend="sdpa",
        use_cache=False,
    )


def _esmc_config() -> ESMplusplusConfig:
    return ESMplusplusConfig(
        vocab_size=16,
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=1,
        dropout=0.0,
        pad_token_id=1,
        mask_token_id=5,
        attn_backend="eager",
    )


def _ankh_config() -> FastAnkhConfig:
    return FastAnkhConfig(
        vocab_size=16,
        d_model=8,
        d_kv=4,
        d_ff=16,
        num_heads=2,
        num_layers=1,
        num_decoder_layers=1,
        dropout_rate=0.0,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
        attn_backend="eager",
        use_cache=False,
    )


def _esm3_config() -> FastESM3Config:
    return FastESM3Config(
        hidden_size=8,
        num_attention_heads=2,
        num_vector_heads=2,
        num_hidden_layers=1,
        attn_backend="eager",
    )


_BASE_MODEL_CASES: tuple[
    tuple[
        str,
        Callable[[], PretrainedConfig],
        type[PreTrainedModel],
        type[PreTrainedModel],
    ],
    ...,
] = (
    ("esm2-base", _esm2_config, FAST_ESM_ENCODER, FastEsmModel),
    ("esm2-mlm", _esm2_config, FAST_ESM_ENCODER, FastEsmForMaskedLM),
    (
        "esm2-sequence",
        _esm2_config,
        FAST_ESM_ENCODER,
        FastEsmForSequenceClassification,
    ),
    ("esm2-token", _esm2_config, FAST_ESM_ENCODER, FastEsmForTokenClassification),
    ("dplm-base", _dplm_config, FAST_DPLM_ENCODER, DPLMModel),
    ("dplm-mlm", _dplm_config, FAST_DPLM_ENCODER, DPLMForMaskedLM),
    (
        "dplm-sequence",
        _dplm_config,
        FAST_DPLM_ENCODER,
        DPLMForSequenceClassification,
    ),
    ("dplm-token", _dplm_config, FAST_DPLM_ENCODER, DPLMForTokenClassification),
    ("dplm2-base", _dplm2_config, FAST_DPLM2_ENCODER, DPLM2Model),
    ("dplm2-mlm", _dplm2_config, FAST_DPLM2_ENCODER, DPLM2ForMaskedLM),
    (
        "dplm2-sequence",
        _dplm2_config,
        FAST_DPLM2_ENCODER,
        DPLM2ForSequenceClassification,
    ),
    (
        "dplm2-token",
        _dplm2_config,
        FAST_DPLM2_ENCODER,
        DPLM2ForTokenClassification,
    ),
)


def _assert_exact_state(actual: torch.nn.Module, expected: torch.nn.Module) -> None:
    actual_state = actual.state_dict()  # parameter name -> checkpoint-shaped tensor
    expected_state = expected.state_dict()  # parameter name -> checkpoint-shaped tensor
    assert actual_state.keys() == expected_state.keys()
    for name, expected_tensor in expected_state.items():
        torch.testing.assert_close(
            actual_state[name],
            expected_tensor,
            rtol=0.0,
            atol=0.0,
            msg=lambda message, name=name: f"{name}: {message}",
        )


@pytest.mark.parametrize(
    ("case_id", "config_factory", "encoder_class", "wrapper_class"),
    _BASE_MODEL_CASES,
    ids=[case[0] for case in _BASE_MODEL_CASES],
)
def test_advertised_wrappers_expose_and_prefix_load_the_esm_base_model(
    case_id: str,
    config_factory: Callable[[], PretrainedConfig],
    encoder_class: type[PreTrainedModel],
    wrapper_class: type[PreTrainedModel],
    tmp_path: Path,
) -> None:
    del case_id
    encoder = encoder_class(config_factory()).eval()
    encoder_dir = tmp_path / "encoder"
    encoder.save_pretrained(encoder_dir, safe_serialization=True)

    wrapper, loading_info = wrapper_class.from_pretrained(
        encoder_dir,
        local_files_only=True,
        output_loading_info=True,
    )
    wrapper.eval()

    assert wrapper.base_model is wrapper.esm
    assert wrapper.base_model is not wrapper
    _assert_exact_state(wrapper.base_model, encoder)
    assert not loading_info["unexpected_keys"]
    assert not any(key.startswith("esm.") for key in loading_info["missing_keys"])
    assert not loading_info["mismatched_keys"]

    wrapper_dir = tmp_path / "wrapper"
    wrapper.save_pretrained(wrapper_dir, safe_serialization=True)
    reloaded = wrapper_class.from_pretrained(wrapper_dir, local_files_only=True).eval()
    assert reloaded.base_model is reloaded.esm
    assert reloaded.base_model is not reloaded
    _assert_exact_state(reloaded, wrapper)


_PACKAGE_RESAVE_CASES: tuple[
    tuple[str, Callable[[], PretrainedConfig], type[PreTrainedModel], bool], ...
] = (
    ("esm2", _esm2_config, FastEsmModel, False),
    ("dplm", _dplm_config, DPLMModel, False),
    ("dplm2", _dplm2_config, DPLM2Model, False),
    ("esmc", _esmc_config, ESMplusplusModel, False),
    ("ankh", _ankh_config, FastAnkhModel, False),
    ("esm3", _esm3_config, FastESM3Model, True),
)


@pytest.mark.parametrize(
    ("case_id", "config_factory", "model_class", "expected_remote_code"),
    _PACKAGE_RESAVE_CASES,
    ids=[case[0] for case in _PACKAGE_RESAVE_CASES],
)
def test_package_models_use_hf_local_semantics_across_save_resave(
    case_id: str,
    config_factory: Callable[[], PretrainedConfig],
    model_class: type[PreTrainedModel],
    expected_remote_code: bool,
    tmp_path: Path,
) -> None:
    del case_id
    model = model_class(config_factory()).eval()
    assert model.is_remote_code() is expected_remote_code

    first_path = tmp_path / "first"
    model.save_pretrained(first_path, safe_serialization=True)
    reloaded = model_class.from_pretrained(first_path, local_files_only=True).eval()
    assert reloaded.is_remote_code() is expected_remote_code
    _assert_exact_state(reloaded, model)

    second_path = tmp_path / "second"
    reloaded.save_pretrained(second_path, safe_serialization=True)
    resaved = model_class.from_pretrained(second_path, local_files_only=True).eval()
    assert resaved.is_remote_code() is expected_remote_code
    _assert_exact_state(resaved, model)
    for checked_model in (model, reloaded, resaved):
        assert None not in (getattr(checked_model.config, "auto_map", None) or {})


def test_dplm_rejects_mixed_batch_attention_mask_broadcasting() -> None:
    model = DPLMModel(_dplm_config()).eval()
    input_ids = torch.tensor([[0, 6, 2, 1], [0, 7, 8, 2]])  # (b=2, l=4)

    for malformed_mask in (
        torch.ones(1, 4, dtype=torch.long),
        torch.ones(2, 3, dtype=torch.long),
    ):
        with pytest.raises(
            ValueError,
            match=r"attention_mask must have shape \(2, 4\)",
        ):
            model(input_ids=input_ids, attention_mask=malformed_mask)


def test_dplm2_config_rejects_decoder_and_cross_attention_modes() -> None:
    for unsupported_flag in ("is_decoder", "add_cross_attention"):
        with pytest.raises(ValueError, match="DPLM2 is encoder-only"):
            DPLM2Config(**{unsupported_flag: True})


def test_dplm2_legacy_cache_config_warns_once_and_new_artifacts_disable_cache(
    tmp_path: Path,
) -> None:
    with pytest.warns(UserWarning, match="normalizing use_cache to False") as warning_records:
        config = DPLM2Config(use_cache=True)

    assert len(warning_records) == 1
    assert config.use_cache is False
    assert config.is_decoder is False
    assert config.add_cross_attention is False

    config.save_pretrained(tmp_path)
    serialized = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert serialized["use_cache"] is False
    assert serialized["is_decoder"] is False
    assert serialized["add_cross_attention"] is False

    reloaded = DPLM2Config.from_pretrained(tmp_path, local_files_only=True)
    assert reloaded.use_cache is False


def test_dplm2_public_forwards_reject_cache_and_cross_attention_arguments() -> None:
    input_ids = torch.tensor([[0, 6, 7, 2]])  # (b=1, l=4)
    for model_class in (
        DPLM2Model,
        DPLM2ForMaskedLM,
        DPLM2ForSequenceClassification,
        DPLM2ForTokenClassification,
    ):
        model = model_class(_dplm2_config()).eval()
        for argument, value in (
            ("use_cache", True),
            ("past_key_values", ((torch.zeros(1), torch.zeros(1)),)),
            ("encoder_hidden_states", torch.zeros(1, 2, 8)),
        ):
            with pytest.raises(TypeError, match=argument):
                model(input_ids=input_ids, **{argument: value})


def test_esmc_sequence_id_is_authoritative_for_chain_and_padding_masks() -> None:
    model = ESMplusplusModel(_esmc_config()).eval()
    input_ids = torch.tensor([[0, 3, 4, 5, 1, 1]])  # (b=1, l=6)
    sequence_id = torch.tensor([[0, 0, 1, 1, -1, -1]])  # (b, l)

    with torch.inference_mode():
        expected = model(  # (b, l, d=8)
            input_ids=input_ids,
            sequence_id=sequence_id,
        ).last_hidden_state
        actual = model(  # (b, l, d)
            input_ids=input_ids,
            sequence_id=sequence_id,
            # The official Biohub contract ignores this mask whenever
            # sequence_id is present; sequence_id itself carries padding.
            attention_mask=torch.zeros_like(input_ids),
        ).last_hidden_state

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    mask_2d, mask_4d, block_mask = model.transformer._sequence_id_attention_masks(
        sequence_id,
        batch_size=1,
        seq_len=6,
        device=input_ids.device,
    )  # (b, l), (b, 1, l, l), None for eager attention
    assert torch.equal(mask_2d, sequence_id.ge(0))
    assert torch.equal(
        mask_4d,
        sequence_id[:, None, :, None].eq(sequence_id[:, None, None, :]),
    )
    assert block_mask is None
