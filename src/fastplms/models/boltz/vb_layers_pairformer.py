"""Pairformer layers for joint or pair-only Boltz2 representations."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from . import vb_const as const
from .vb_layers_attention import AttentionPairBias
from .vb_layers_attentionv2 import AttentionPairBias as AttentionPairBiasV2
from .vb_layers_dropout import get_dropout_mask
from .vb_layers_transition import Transition
from .vb_layers_triangular_mult import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from .vb_tri_attn_attention import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
)


def _pair_update(
    modules: Any,
    pair_states: Tensor,
    pair_mask: Tensor,
    chunk_size: int | None,
    use_kernels: bool,
    use_cuequiv_mul: bool,
    use_cuequiv_attn: bool,
) -> Tensor:
    """Apply multiplicative, attention, and transition updates to pair tensor Z."""

    multiplication_kernel = use_kernels or use_cuequiv_mul
    attention_kernel = use_kernels or use_cuequiv_attn
    output = pair_states
    for update in (modules.tri_mul_out, modules.tri_mul_in):
        dropout = get_dropout_mask(modules.dropout, output, modules.training)
        output = output + dropout * update(
            output,
            mask=pair_mask,
            use_kernels=multiplication_kernel,
        )

    dropout = get_dropout_mask(modules.dropout, output, modules.training)
    output = output + dropout * modules.tri_att_start(
        output,
        mask=pair_mask,
        chunk_size=chunk_size,
        use_kernels=attention_kernel,
    )
    dropout = get_dropout_mask(
        modules.dropout,
        output,
        modules.training,
        columnwise=True,
    )
    output = output + dropout * modules.tri_att_end(
        output,
        mask=pair_mask,
        chunk_size=chunk_size,
        use_kernels=attention_kernel,
    )
    return cast(Tensor, output + modules.transition_z(output))


def _triangle_chunk_size(pair_states: Tensor, training: bool) -> int | None:
    if training:
        return None
    return 128 if pair_states.shape[1] > const.chunk_size_threshold else 512


class PairformerLayer(nn.Module):
    """Update sequence tensor S and pair tensor Z once."""

    def __init__(
        self,
        token_s: int,
        token_z: int,
        num_heads: int = 16,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        post_layer_norm: bool = False,
        v2: bool = False,
    ) -> None:
        super().__init__()
        self.token_z = token_z
        self.dropout = dropout
        self.num_heads = num_heads
        self.post_layer_norm = post_layer_norm
        self.pre_norm_s = nn.LayerNorm(token_s)
        attention_class = AttentionPairBiasV2 if v2 else AttentionPairBias
        self.attention = attention_class(token_s, token_z, num_heads)
        self.tri_mul_out = TriangleMultiplicationOutgoing(token_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(token_z)
        self.tri_att_start = TriangleAttentionStartingNode(
            token_z,
            pairwise_head_width,
            pairwise_num_heads,
            inf=1e9,
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            token_z,
            pairwise_head_width,
            pairwise_num_heads,
            inf=1e9,
        )
        self.transition_s = Transition(token_s, token_s * 4)
        self.transition_z = Transition(token_z, token_z * 4)
        self.s_post_norm = nn.LayerNorm(token_s) if post_layer_norm else nn.Identity()

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        pair_mask: Tensor,
        chunk_size_tri_attn: int | None = None,
        use_kernels: bool = False,
        use_cuequiv_mul: bool = False,
        use_cuequiv_attn: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Return updated ``S: (b, l, d_s)`` and ``Z: (b, l, l, d_z)``."""

        pair_output = _pair_update(
            self,
            z,
            pair_mask,
            chunk_size_tri_attn,
            use_kernels,
            use_cuequiv_mul,
            use_cuequiv_attn,
        )
        with torch.autocast("cuda", enabled=False):
            normalized = self.pre_norm_s(s.float())
            sequence_output = s.float() + self.attention(
                s=normalized,
                z=pair_output.float(),
                mask=mask.float(),
                k_in=normalized,
            )
            sequence_output = sequence_output + self.transition_s(sequence_output)
            sequence_output = cast(Tensor, self.s_post_norm(sequence_output))
        return sequence_output, pair_output


class PairformerModule(nn.Module):
    """Run a stack of joint sequence and pair updates."""

    def __init__(
        self,
        token_s: int,
        token_z: int,
        num_blocks: int,
        num_heads: int = 16,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        post_layer_norm: bool = False,
        activation_checkpointing: bool = False,
        v2: bool = False,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__()
        self.token_z = token_z
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.num_heads = num_heads
        self.post_layer_norm = post_layer_norm
        self.activation_checkpointing = activation_checkpointing
        self.layers = nn.ModuleList(
            [
                PairformerLayer(
                    token_s,
                    token_z,
                    num_heads,
                    dropout,
                    pairwise_head_width,
                    pairwise_num_heads,
                    post_layer_norm,
                    v2,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        pair_mask: Tensor,
        use_kernels: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Return S and Z after every configured pairformer block."""

        chunk_size = _triangle_chunk_size(z, self.training)
        sequence_output, pair_output = s, z
        for layer in self.layers:
            if self.activation_checkpointing:
                sequence_output, pair_output = checkpoint(
                    layer,
                    sequence_output,
                    pair_output,
                    mask,
                    pair_mask,
                    chunk_size,
                    use_kernels,
                    use_reentrant=False,
                )
            else:
                sequence_output, pair_output = layer(
                    sequence_output,
                    pair_output,
                    mask,
                    pair_mask,
                    chunk_size,
                    use_kernels,
                )
        return sequence_output, pair_output


class PairformerNoSeqLayer(nn.Module):
    """Update pair tensor Z without a sequence track."""

    def __init__(
        self,
        token_z: int,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        post_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.token_z = token_z
        self.dropout = dropout
        self.post_layer_norm = post_layer_norm
        self.tri_mul_out = TriangleMultiplicationOutgoing(token_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(token_z)
        self.tri_att_start = TriangleAttentionStartingNode(
            token_z,
            pairwise_head_width,
            pairwise_num_heads,
            inf=1e9,
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            token_z,
            pairwise_head_width,
            pairwise_num_heads,
            inf=1e9,
        )
        self.transition_z = Transition(token_z, token_z * 4)

    def forward(
        self,
        z: Tensor,
        pair_mask: Tensor,
        chunk_size_tri_attn: int | None = None,
        use_kernels: bool = False,
        use_cuequiv_mul: bool = False,
        use_cuequiv_attn: bool = False,
    ) -> Tensor:
        """Return updated pair tensor Z with shape ``(b, l, l, d_z)``."""

        return _pair_update(
            self,
            z,
            pair_mask,
            chunk_size_tri_attn,
            use_kernels,
            use_cuequiv_mul,
            use_cuequiv_attn,
        )


class PairformerNoSeqModule(nn.Module):
    """Run a stack of pair-only updates."""

    def __init__(
        self,
        token_z: int,
        num_blocks: int,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        post_layer_norm: bool = False,
        activation_checkpointing: bool = False,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__()
        self.token_z = token_z
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.post_layer_norm = post_layer_norm
        self.activation_checkpointing = activation_checkpointing
        self.layers = nn.ModuleList(
            [
                PairformerNoSeqLayer(
                    token_z,
                    dropout,
                    pairwise_head_width,
                    pairwise_num_heads,
                    post_layer_norm,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        z: Tensor,
        pair_mask: Tensor,
        use_kernels: bool = False,
    ) -> Tensor:
        """Return Z after every configured pair-only block."""

        chunk_size = _triangle_chunk_size(z, self.training)
        output = z
        for layer in self.layers:
            if self.activation_checkpointing:
                output = checkpoint(
                    layer,
                    output,
                    pair_mask,
                    chunk_size,
                    use_kernels,
                    use_reentrant=False,
                )
            else:
                output = layer(output, pair_mask, chunk_size, use_kernels)
        return output
