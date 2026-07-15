"""Focused ESM++ input-mask contracts."""

from __future__ import annotations

import pytest
import torch

from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
    ESMplusplusConfig,
    ESMplusplusForMaskedLM,
    ESMplusplusModel,
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
