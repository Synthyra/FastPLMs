"""Conditioned transformer blocks used by Boltz2 diffusion."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .vb_layers_attentionv2 import AttentionPairBias
from .vb_modules_utils import LinearNoBias, SwiGLU


class AdaLN(nn.Module):
    """Normalize activations and apply scale and shift from conditioning S."""

    def __init__(self, dim: int, dim_single_cond: int) -> None:
        super().__init__()
        self.a_norm = nn.LayerNorm(dim, elementwise_affine=False, bias=False)
        self.s_norm = nn.LayerNorm(dim_single_cond, bias=False)
        self.s_scale = nn.Linear(dim_single_cond, dim)
        self.s_bias = LinearNoBias(dim_single_cond, dim)

    def forward(self, a: Tensor, s: Tensor) -> Tensor:
        normalized = self.a_norm(a)
        conditioning = self.s_norm(s)
        output = self.s_scale(conditioning).sigmoid() * normalized + self.s_bias(conditioning)
        return cast(Tensor, output)


class ConditionedTransitionBlock(nn.Module):
    """Apply a conditioned gated transition without changing tensor shape."""

    def __init__(
        self,
        dim_single: int,
        dim_single_cond: int,
        expansion_factor: float = 2,
    ) -> None:
        super().__init__()
        self.adaln = AdaLN(dim_single, dim_single_cond)
        inner_dim = int(dim_single * expansion_factor)
        self.swish_gate = nn.Sequential(
            LinearNoBias(dim_single, inner_dim * 2),
            SwiGLU(),
        )
        self.a_to_b = LinearNoBias(dim_single, inner_dim)
        self.b_to_a = LinearNoBias(inner_dim, dim_single)

        projection = nn.Linear(dim_single_cond, dim_single)
        nn.init.zeros_(projection.weight)
        nn.init.constant_(projection.bias, -2.0)
        self.output_projection = nn.Sequential(projection, nn.Sigmoid())

    def forward(self, a: Tensor, s: Tensor) -> Tensor:
        normalized = self.adaln(a, s)
        hidden = self.swish_gate(normalized) * self.a_to_b(normalized)
        return cast(Tensor, self.output_projection(s) * self.b_to_a(hidden))


class DiffusionTransformer(nn.Module):
    """Run conditioned transformer layers over token or atom states."""

    def __init__(
        self,
        depth: int,
        heads: int,
        dim: int = 384,
        dim_single_cond: int | None = None,
        pair_bias_attn: bool = True,
        activation_checkpointing: bool = False,
        post_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        conditioning_dim = dim if dim_single_cond is None else dim_single_cond
        self.activation_checkpointing = activation_checkpointing
        self.pair_bias_attn = pair_bias_attn
        self.layers = nn.ModuleList(
            [
                DiffusionTransformerLayer(
                    heads,
                    dim,
                    conditioning_dim,
                    post_layer_norm,
                )
                for _ in range(depth)
            ]
        )

    def _split_pair_bias(self, bias: Tensor | None) -> Tensor | None:
        if not self.pair_bias_attn:
            return None
        if bias is None:
            raise ValueError("pair bias is required when pair_bias_attn=True")
        batch_size, query_length, key_length, width = bias.shape
        depth = len(self.layers)
        if depth == 0 or width % depth:
            raise ValueError("pair-bias width must be divisible by transformer depth")
        return bias.view(batch_size, query_length, key_length, depth, width // depth)

    def forward(
        self,
        a: Tensor,
        s: Tensor,
        bias: Tensor | None = None,
        mask: Tensor | None = None,
        to_keys: Callable[[Tensor], Tensor] | None = None,
        multiplicity: int = 1,
    ) -> Tensor:
        """Transform A and preserve shape ``(b * m, l, d)``."""

        layer_biases = self._split_pair_bias(bias)
        output = a
        for index, layer in enumerate(self.layers):
            layer_bias = None if layer_biases is None else layer_biases[..., index, :]
            if self.activation_checkpointing:
                output = checkpoint(
                    layer,
                    output,
                    s,
                    layer_bias,
                    mask,
                    to_keys,
                    multiplicity,
                    use_reentrant=False,
                )
            else:
                output = layer(
                    output,
                    s,
                    layer_bias,
                    mask,
                    to_keys,
                    multiplicity,
                )
        return output


class DiffusionTransformerLayer(nn.Module):
    """One adaptive-normalization attention and transition block."""

    def __init__(
        self,
        heads: int,
        dim: int = 384,
        dim_single_cond: int | None = None,
        post_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        conditioning_dim = dim if dim_single_cond is None else dim_single_cond
        self.adaln = AdaLN(dim, conditioning_dim)
        self.pair_bias_attn = AttentionPairBias(
            c_s=dim,
            num_heads=heads,
            compute_pair_bias=False,
        )
        self.output_projection_linear = nn.Linear(conditioning_dim, dim)
        nn.init.zeros_(self.output_projection_linear.weight)
        nn.init.constant_(self.output_projection_linear.bias, -2.0)
        self.output_projection = nn.Sequential(
            self.output_projection_linear,
            nn.Sigmoid(),
        )
        self.transition = ConditionedTransitionBlock(dim, conditioning_dim)
        self.post_lnorm = nn.LayerNorm(dim) if post_layer_norm else nn.Identity()

    def forward(
        self,
        a: Tensor,
        s: Tensor,
        bias: Tensor | None = None,
        mask: Tensor | None = None,
        to_keys: Callable[[Tensor], Tensor] | None = None,
        multiplicity: int = 1,
    ) -> Tensor:
        """Update activation tensor A with conditioning S and pair bias."""

        if bias is None or mask is None:
            raise ValueError("diffusion attention requires pair bias and a key mask")
        normalized = self.adaln(a, s)
        key_states = normalized
        key_mask = mask
        if to_keys is not None:
            key_states = to_keys(normalized)
            key_mask = to_keys(mask.unsqueeze(-1)).squeeze(-1)

        attended = self.pair_bias_attn(
            s=normalized,
            z=bias,
            mask=key_mask,
            multiplicity=multiplicity,
            k_in=key_states,
        )
        output = a + self.output_projection(s) * attended
        output = output + self.transition(output, s)
        return cast(Tensor, self.post_lnorm(output))


class AtomTransformer(nn.Module):
    """Apply the diffusion transformer over fixed atom-query windows."""

    def __init__(
        self,
        attn_window_queries: int,
        attn_window_keys: int,
        **diffusion_transformer_kwargs: Any,
    ) -> None:
        super().__init__()
        self.attn_window_queries = attn_window_queries
        self.attn_window_keys = attn_window_keys
        self.diffusion_transformer = DiffusionTransformer(
            **diffusion_transformer_kwargs,
        )

    def forward(
        self,
        q: Tensor,
        c: Tensor,
        bias: Tensor,
        to_keys: Callable[[Tensor], Tensor],
        mask: Tensor,
        multiplicity: int = 1,
    ) -> Tensor:
        """Transform atom tensor Q with shape ``(b, n_atoms, d)``."""

        query_window = self.attn_window_queries
        key_window = self.attn_window_keys
        batch_size, atom_count, width = q.shape
        window_count = atom_count // query_window
        query_states = q.view(batch_size * window_count, query_window, -1)
        conditioning = c.view(batch_size * window_count, query_window, -1)
        query_mask = mask.view(batch_size * window_count, query_window)
        pair_bias = bias.repeat_interleave(multiplicity, dim=0)
        pair_bias = pair_bias.view(
            pair_bias.shape[0] * window_count,
            query_window,
            key_window,
            -1,
        )

        def windowed_keys(states: Tensor) -> Tensor:
            merged = states.view(batch_size, window_count * query_window, -1)
            return to_keys(merged).view(batch_size * window_count, key_window, -1)

        output = self.diffusion_transformer(
            a=query_states,
            s=conditioning,
            bias=pair_bias,
            mask=query_mask.float(),
            multiplicity=1,
            to_keys=windowed_keys,
        )
        return cast(Tensor, output.view(batch_size, window_count * query_window, width))
