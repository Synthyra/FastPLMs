"""Pair-biased self-attention used in the Boltz2 diffusion stack."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping

from einops.layers.torch import Rearrange
from torch import Tensor, nn

from . import vb_layers_initialize as init
from ._pair_attention import pair_biased_attention, reshape_heads


class AttentionPairBias(nn.Module):
    """Attend over sequence states while adding a learned pair representation."""

    def __init__(
        self,
        c_s: int,
        c_z: int,
        num_heads: int,
        inf: float = 1e6,
        initial_norm: bool = True,
    ) -> None:
        super().__init__()
        if c_s % num_heads:
            raise ValueError(f"c_s={c_s} must be divisible by num_heads={num_heads}")

        self.c_s = c_s
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        self.inf = inf
        self.initial_norm = initial_norm
        if initial_norm:
            self.norm_s = nn.LayerNorm(c_s)

        self.proj_q = nn.Linear(c_s, c_s)
        self.proj_k = nn.Linear(c_s, c_s, bias=False)
        self.proj_v = nn.Linear(c_s, c_s, bias=False)
        self.proj_g = nn.Linear(c_s, c_s, bias=False)
        self.proj_z = nn.Sequential(
            nn.LayerNorm(c_z),
            nn.Linear(c_z, num_heads, bias=False),
            Rearrange("b ... h -> b h ..."),
        )
        self.proj_o = nn.Linear(c_s, c_s, bias=False)
        init.final_init_(self.proj_o.weight)

    def _resolve_pair_bias(
        self,
        pair_states: Tensor,
        model_cache: MutableMapping[str, Tensor] | None,
    ) -> Tensor:
        if model_cache is not None and "z" in model_cache:
            return model_cache["z"]
        pair_bias = self.proj_z(pair_states)
        if model_cache is not None:
            model_cache["z"] = pair_bias
        return pair_bias

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        k_in: Tensor | None = None,
        multiplicity: int = 1,
        to_keys: Callable[[Tensor], Tensor] | None = None,
        model_cache: MutableMapping[str, Tensor] | None = None,
    ) -> Tensor:
        """Transform S with shapes ``S: (b, l_q, d)`` and ``Z: (b, l, l, d_z)``."""

        sequence_states = self.norm_s(s) if self.initial_norm else s
        if to_keys is not None:
            key_states = to_keys(sequence_states)
            key_mask = to_keys(mask.unsqueeze(-1)).squeeze(-1)
        else:
            key_states = sequence_states if k_in is None else k_in
            key_mask = mask

        query = reshape_heads(self.proj_q(sequence_states), self.num_heads)
        key = reshape_heads(self.proj_k(key_states), self.num_heads)
        value = reshape_heads(self.proj_v(key_states), self.num_heads)
        pair_bias = self._resolve_pair_bias(z, model_cache).repeat_interleave(
            multiplicity,
            dim=0,
        )
        attended = pair_biased_attention(
            query,
            key,
            value,
            pair_bias,
            key_mask,
            self.inf,
        )
        attended = attended.reshape(s.shape[0], -1, self.c_s)
        gate = self.proj_g(sequence_states).sigmoid()
        return self.proj_o(gate * attended)
