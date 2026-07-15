"""DPLM rotary compatibility with the pinned Transformers release."""

from __future__ import annotations

import torch

from fastplms.models._esm_rotary import RotaryEmbedding
from fastplms.models.dplm.modeling_dplm import DPLMConfig, ModifiedEsmSelfAttention


def test_dplm_initializes_the_checkpoint_compatible_rotary_buffer() -> None:
    config = DPLMConfig(
        vocab_size=33,
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=1,
        intermediate_size=128,
        position_embedding_type="rotary",
        attn_backend="sdpa",
    )
    attention = ModifiedEsmSelfAttention(config)

    assert isinstance(attention.rotary_embeddings, RotaryEmbedding)
    assert set(attention.rotary_embeddings.state_dict()) == {"inv_freq"}
    expected = 1.0 / (
        10_000
        ** (
            torch.arange(0, attention.attention_head_size, 2).float()
            / attention.attention_head_size
        )
    )
    assert torch.equal(attention.rotary_embeddings.inv_freq, expected)


def test_dplm_rotary_forward_is_finite() -> None:
    config = DPLMConfig(
        vocab_size=33,
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=1,
        intermediate_size=128,
        position_embedding_type="rotary",
        attn_backend="sdpa",
    )
    attention = ModifiedEsmSelfAttention(config).eval()
    hidden_states = torch.randn(2, 11, config.hidden_size)
    output, weights, s_max = attention(
        hidden_states,
        attention_mask_2d=torch.ones(2, 11, dtype=torch.bool),
    )

    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all()
    assert weights is None
    assert s_max is None
