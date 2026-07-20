"""Focused ESM++ input-mask contracts."""

from __future__ import annotations

import pytest
import torch

from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
    ESMplusplusConfig,
    ESMplusplusForMaskedLM,
    ESMplusplusModel,
    ESMplusplusForSequenceClassification,
    TransformerStack,
)


@pytest.mark.parametrize("model_class", (ESMplusplusModel, ESMplusplusForMaskedLM))
def test_esmplusplus_infers_padding_mask_from_input_ids(model_class: type) -> None:
    config = ESMplusplusConfig(
        vocab_size=16,
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=1,
        attn_backend="eager",
        pad_token_id=1,
    )
    model = model_class(config).eval()
    input_ids = torch.tensor([[0, 3, 4, 1, 1]], dtype=torch.long)
    attention_mask = input_ids.ne(config.pad_token_id)

    kwargs = {"compute_logits": False} if model_class is ESMplusplusForMaskedLM else {}
    with torch.inference_mode():
        inferred = model(input_ids=input_ids, **kwargs).last_hidden_state
        explicit = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        ).last_hidden_state

    torch.testing.assert_close(inferred, explicit, rtol=0.0, atol=0.0)


def test_esmplusplus_boolean_sequence_id_matches_biohub_equality_mask() -> None:
    stack = TransformerStack(
        d_model=16,
        n_heads=4,
        n_layers=1,
        attn_backend="eager",
    )
    sequence_id = torch.tensor([[True, True, True, False, False]])

    mask_2d, mask_4d, block_mask = stack._sequence_id_attention_masks(
        sequence_id=sequence_id,
        batch_size=1,
        seq_len=5,
        device=torch.device("cpu"),
    )

    expected = sequence_id[:, None, :, None] == sequence_id[:, None, None, :]
    assert torch.equal(mask_2d, sequence_id)
    assert torch.equal(mask_4d, expected)
    assert block_mask is None


@pytest.mark.parametrize("model_class", (ESMplusplusModel, ESMplusplusForMaskedLM))
def test_esmplusplus_embedding_helper_infers_padding_mask(model_class: type) -> None:
    config = ESMplusplusConfig(
        vocab_size=16,
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=1,
        attn_backend="eager",
        pad_token_id=1,
    )
    model = model_class(config).eval()
    input_ids = torch.tensor([[0, 3, 4, 1, 1]], dtype=torch.long)
    attention_mask = input_ids.ne(config.pad_token_id)

    with torch.inference_mode():
        inferred = model._embed(input_ids)
        explicit = model._embed(input_ids, attention_mask=attention_mask)

    torch.testing.assert_close(inferred, explicit, rtol=0.0, atol=0.0)


def _sequence_classifier_config() -> ESMplusplusConfig:
    return ESMplusplusConfig(
        vocab_size=16,
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=1,
        num_labels=3,
        attn_backend="eager",
        pad_token_id=1,
    )


def test_esmplusplus_sequence_classifier_accepts_explicit_pooling_types() -> None:
    model = ESMplusplusForSequenceClassification(
        _sequence_classifier_config(),
        pooling_types=["mean"],
    ).eval()

    assert model.pooler.names == ("mean",)
    assert model.classifier[0].in_features == 16

    with torch.inference_mode():
        output = model(input_ids=torch.tensor([[0, 3, 4, 2]], dtype=torch.long))

    assert output.logits is not None
    assert output.logits.shape == (1, 3)


def test_esmplusplus_sequence_classifier_pooling_round_trips(tmp_path) -> None:
    model = ESMplusplusForSequenceClassification(
        _sequence_classifier_config(),
        pooling_types=["mean"],
    ).eval()
    model.save_pretrained(tmp_path)

    reloaded = ESMplusplusForSequenceClassification.from_pretrained(tmp_path).eval()

    assert reloaded.config.classifier_pooling_types == ["mean"]
    assert reloaded.pooler.names == ("mean",)
    assert reloaded.classifier[0].in_features == 16
    for name, value in model.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[name], value, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("pooling_types", "exception"),
    [
        ([], ValueError),
        (("mean",), TypeError),
        (["mean", 1], TypeError),
        (["parti"], ValueError),
    ],
)
def test_esmplusplus_sequence_classifier_validates_pooling_types(
    pooling_types: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception, match="pooling_types"):
        ESMplusplusForSequenceClassification(
            _sequence_classifier_config(),
            pooling_types=pooling_types,
        )


def test_esmplusplus_sequence_classifier_is_right_padding_invariant_without_mask() -> None:
    model = ESMplusplusForSequenceClassification(_sequence_classifier_config()).eval()
    unpadded = torch.tensor([[0, 3, 4, 2]], dtype=torch.long)
    right_padded = torch.tensor([[0, 3, 4, 2, 1, 1]], dtype=torch.long)

    with torch.inference_mode():
        unpadded_logits = model(input_ids=unpadded).logits
        padded_logits = model(input_ids=right_padded).logits

    assert unpadded_logits is not None
    assert padded_logits is not None
    torch.testing.assert_close(padded_logits, unpadded_logits, rtol=1e-5, atol=1e-6)
