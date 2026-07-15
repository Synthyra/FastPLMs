"""Triangular row and column attention for Boltz2 pair states."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from .vb_tri_attn_primitives import Attention, LayerNorm, Linear
from .vb_tri_attn_utils import chunk_layer, permute_final_dims


class TriangleAttention(nn.Module):
    """Apply pair-biased attention along one axis of pair tensor X."""

    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        no_heads: int,
        starting: bool = True,
        inf: float = 1e9,
    ) -> None:
        super().__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf
        self.layer_norm = LayerNorm(c_in)
        self.linear = Linear(c_in, no_heads, bias=False, init="normal")
        self.mha = Attention(c_in, c_in, c_in, c_hidden, no_heads)

    @torch.jit.ignore  # type: ignore[untyped-decorator]
    def _chunk(
        self,
        x: Tensor,
        tri_bias: Tensor,
        mask_bias: Tensor,
        mask: Tensor,
        chunk_size: int,
        use_kernels: bool = False,
    ) -> Tensor:
        """Evaluate attention in slices of the batch-like leading axes."""

        def attention_call(**inputs: Tensor) -> Tensor:
            return cast(Tensor, self.mha(**inputs, use_kernels=use_kernels))

        inputs = {
            "q_x": x,
            "kv_x": x,
            "tri_bias": tri_bias,
            "mask_bias": mask_bias,
            "mask": mask,
        }
        return cast(
            Tensor,
            chunk_layer(
                attention_call,
                inputs,
                chunk_size=chunk_size,
                no_batch_dims=len(x.shape[:-2]),
                _out=None,
            ),
        )

    def _run_attention(
        self,
        states: Tensor,
        triangle_bias: Tensor,
        mask_bias: Tensor,
        expanded_mask: Tensor,
        chunk_size: int | None,
        use_kernels: bool,
    ) -> Tensor:
        if chunk_size is not None and not use_kernels:
            return cast(
                Tensor,
                self._chunk(
                    states,
                    triangle_bias,
                    mask_bias,
                    expanded_mask,
                    chunk_size,
                    use_kernels=False,
                ),
            )
        return cast(
            Tensor,
            self.mha(
                states,
                states,
                triangle_bias,
                mask_bias,
                expanded_mask,
                use_kernels=use_kernels,
            ),
        )

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        chunk_size: int | None = None,
        use_kernels: bool = False,
    ) -> Tensor:
        """Transform X with shape ``(..., l, l, d)`` under mask M."""

        pair_mask = x.new_ones(x.shape[:-1]) if mask is None else mask
        pair_states = x
        if not self.starting:
            pair_states = pair_states.transpose(-2, -3)
            pair_mask = pair_mask.transpose(-1, -2)

        normalized = self.layer_norm(pair_states)
        expanded_mask = pair_mask[..., :, None, None, :]
        mask_bias = self.inf * (expanded_mask - 1)
        triangle_bias = permute_final_dims(self.linear(normalized), (2, 0, 1))
        triangle_bias = triangle_bias.unsqueeze(-4)
        output = self._run_attention(
            normalized,
            triangle_bias,
            mask_bias,
            expanded_mask,
            chunk_size,
            use_kernels,
        )
        return output if self.starting else output.transpose(-2, -3)


TriangleAttentionStartingNode = TriangleAttention


class TriangleAttentionEndingNode(TriangleAttention):
    """Apply triangular attention around each pair's ending node."""

    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        no_heads: int,
        inf: float = 1e9,
    ) -> None:
        super().__init__(c_in, c_hidden, no_heads, starting=False, inf=inf)
