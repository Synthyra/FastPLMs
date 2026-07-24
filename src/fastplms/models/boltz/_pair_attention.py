"""Shared tensor operations for Boltz2 pair-biased attention layers."""

from __future__ import annotations

import math
import torch
from torch import Tensor


def reshape_heads(states: Tensor, num_heads: int) -> Tensor:
    """Reshape X from ``(b, l, d)`` to ``(b, l, h, d_head)``."""

    # states: (b, l, d); num_heads: h
    batch_size, sequence_length, width = states.shape  # b, l, d
    if width % num_heads:
        raise ValueError(f"width {width} is not divisible by {num_heads} heads")
    # states.view(...): (b, l, h, d_h)
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

    # query_states: (b, l_q, h, d_h); key_states, value_states: (b, l_k, h, d_h)
    # pair_bias: (b, h, l_q, l_k); key_mask: (b, l_k)
    head_dim = query_states.shape[-1]  # d_h
    with torch.autocast("cuda", enabled=False):
        scores = torch.einsum(
            "bihd,bjhd->bhij",
            query_states.float(),
            key_states.float(),
        )  # (b, h, l_q, l_k)
        scores = scores / math.sqrt(head_dim) + pair_bias.float()  # (b, h, l_q, l_k)
        scores = (
            scores + (1 - key_mask[:, None, None].float()) * -mask_value
        )  # (b, h, l_q, l_k)
        probabilities = scores.softmax(dim=-1)  # (b, h, l_q, l_k)
        output = torch.einsum(
            "bhij,bjhd->bihd",
            probabilities,
            value_states.float(),
        )  # (b, l_q, h, d_h)
    return output.to(value_states.dtype)  # (b, l_q, h, d_h)
