"""Shared tensor operations for Boltz2 pair-biased attention layers."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def reshape_heads(states: Tensor, num_heads: int) -> Tensor:
    """Reshape X from ``(b, l, d)`` to ``(b, l, h, d_head)``."""

    batch_size, sequence_length, width = states.shape
    if width % num_heads:
        raise ValueError(f"width {width} is not divisible by {num_heads} heads")
    return states.view(batch_size, sequence_length, num_heads, width // num_heads)


def pair_biased_attention(
    query_states: Tensor,
    key_states: Tensor,
    value_states: Tensor,
    pair_bias: Tensor,
    key_mask: Tensor,
    mask_value: float,
) -> Tensor:
    """Return pair-biased attention values with shape ``(b, l_q, h, d_head)``."""

    head_dim = query_states.shape[-1]
    with torch.autocast("cuda", enabled=False):
        # S is the FP32 attention-score tensor with shape (b, h, l_q, l_k).
        scores = torch.einsum(
            "bihd,bjhd->bhij",
            query_states.float(),
            key_states.float(),
        )
        scores = scores / math.sqrt(head_dim) + pair_bias.float()
        scores = scores + (1 - key_mask[:, None, None].float()) * -mask_value
        probabilities = scores.softmax(dim=-1)
        output = torch.einsum(
            "bhij,bjhd->bihd",
            probabilities,
            value_states.float(),
        )
    return output.to(value_states.dtype)
