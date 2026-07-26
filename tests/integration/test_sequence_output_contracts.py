"""Hugging Face output contracts for the ESM2 and ESMC families."""

from __future__ import annotations

import pytest
import torch
from transformers.modeling_outputs import (
    MaskedLMOutput,
    ModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

from fastplms.models.esm2.modeling_fastesm import (
    FastEsmConfig,
    FastEsmForMaskedLM,
    FastEsmForSequenceClassification,
    FastEsmForTokenClassification,
    FastEsmModel,
)
from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
    ESMplusplusConfig,
    ESMplusplusForMaskedLM,
    ESMplusplusForSequenceClassification,
    ESMplusplusForTokenClassification,
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
        attn_backend="eager",
        return_dict=False,
        output_hidden_states=True,
    )


def _esmc_config() -> ESMplusplusConfig:
    return ESMplusplusConfig(
        vocab_size=16,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        dropout=0.0,
        pad_token_id=1,
        mask_token_id=5,
        num_labels=3,
        attn_backend="eager",
        return_dict=False,
        output_hidden_states=True,
    )


def _assert_nested_close(actual, expected) -> None:
    if torch.is_tensor(expected):
        assert torch.is_tensor(actual)
        torch.testing.assert_close(actual, expected)
        return
    if isinstance(expected, (tuple, list)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected, strict=True):
            _assert_nested_close(actual_value, expected_value)
        return
    assert actual == expected


@pytest.mark.parametrize(
    ("model_class", "config_factory", "kind", "output_class"),
    (
        (FastEsmModel, _esm2_config, "base", ModelOutput),
        (FastEsmForMaskedLM, _esm2_config, "mlm", MaskedLMOutput),
        (
            FastEsmForSequenceClassification,
            _esm2_config,
            "sequence",
            SequenceClassifierOutput,
        ),
        (
            FastEsmForTokenClassification,
            _esm2_config,
            "token",
            TokenClassifierOutput,
        ),
        (ESMplusplusModel, _esmc_config, "base", ModelOutput),
        (ESMplusplusForMaskedLM, _esmc_config, "mlm", MaskedLMOutput),
        (
            ESMplusplusForSequenceClassification,
            _esmc_config,
            "sequence",
            SequenceClassifierOutput,
        ),
        (
            ESMplusplusForTokenClassification,
            _esmc_config,
            "token",
            TokenClassifierOutput,
        ),
    ),
)
def test_sequence_models_honor_config_and_explicit_output_controls(
    model_class: type[torch.nn.Module],
    config_factory,
    kind: str,
    output_class: type[ModelOutput],
) -> None:
    model = model_class(config_factory()).eval()
    # input_ids: (2, 5)
    input_ids = torch.tensor([[0, 3, 4, 2, 1], [0, 6, 2, 1, 1]])
    # attention_mask: (b, l)
    attention_mask = input_ids.ne(1)
    labels = {
        "base": None,
        "mlm": input_ids.masked_fill(~attention_mask, -100),
        "sequence": torch.tensor([1, 2]),
        "token": input_ids.remainder(3).masked_fill(~attention_mask, -100),
    }[kind]
    full_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "output_attentions": True,
        "output_hidden_states": True,
        "output_s_max": True,
    }
    if labels is not None:
        full_kwargs["labels"] = labels

    with torch.inference_mode():
        default_output = model(input_ids=input_ids, attention_mask=attention_mask)
        default_structured = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        tuple_output = model(**full_kwargs, return_dict=False)
        structured = model(**full_kwargs, return_dict=True)
        no_states = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )

    assert isinstance(default_output, tuple)
    _assert_nested_close(default_output, default_structured.to_tuple())
    assert isinstance(structured, output_class)
    _assert_nested_close(tuple_output, structured.to_tuple())
    assert structured.hidden_states is not None
    assert structured.attentions is not None
    assert structured.s_max is not None
    assert no_states.hidden_states is None
    if kind != "base":
        assert tuple(structured.keys())[:4] == (
            "loss",
            "logits",
            "hidden_states",
            "attentions",
        )
        assert tuple(structured.keys())[4] == "s_max"


@pytest.mark.parametrize(
    ("model_class", "config_factory"),
    (
        (FastEsmModel, _esm2_config),
        (FastEsmForMaskedLM, _esm2_config),
        (FastEsmForSequenceClassification, _esm2_config),
        (FastEsmForTokenClassification, _esm2_config),
        (ESMplusplusModel, _esmc_config),
        (ESMplusplusForMaskedLM, _esmc_config),
        (ESMplusplusForSequenceClassification, _esmc_config),
        (ESMplusplusForTokenClassification, _esmc_config),
    ),
)
def test_sequence_models_reject_unexpected_forward_arguments(
    model_class: type[torch.nn.Module],
    config_factory,
) -> None:
    model = model_class(config_factory()).eval()

    with pytest.raises(TypeError, match="unexpected_contract"):
        model(
            input_ids=torch.tensor([[0, 3, 2]]),
            unexpected_contract=True,
        )


def test_esm2_resize_preserves_existing_logits_and_bias_contract() -> None:
    model = FastEsmForMaskedLM(_esm2_config()).eval()
    # input_ids: (1, 4)
    input_ids = torch.tensor([[0, 3, 4, 2]])
    with torch.inference_mode():
        original_logits = model(input_ids=input_ids, return_dict=True).logits

    model.resize_token_embeddings(19)

    with torch.inference_mode():
        resized_logits = model(input_ids=input_ids, return_dict=True).logits
    assert model.get_output_embeddings().bias is None
    assert model.lm_head.bias.shape == (19,)
    torch.testing.assert_close(resized_logits[..., :16], original_logits)
