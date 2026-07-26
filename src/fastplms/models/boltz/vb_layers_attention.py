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
        # pair_states: (b, l_q, l_k, d_z)
        if model_cache is not None and "z" in model_cache:
            return model_cache["z"]  # (b, h, l_q, l_k)
        pair_bias = self.proj_z(pair_states)  # (b, h, l_q, l_k)
        if model_cache is not None:
            model_cache["z"] = pair_bias
        return pair_bias  # (b, h, l_q, l_k)

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

        # s: (b, l_q, d); z: (b_z, l_q, l_k, d_z); mask: (b_k, l_k).
        sequence_states = self.norm_s(s) if self.initial_norm else s  # (b, l_q, d)
        if to_keys is not None:
            key_states = to_keys(sequence_states)  # (b, l_k, d)
            key_mask = to_keys(mask.unsqueeze(-1)).squeeze(-1)  # (b_k, l_k)
        else:
            key_states = sequence_states if k_in is None else k_in  # (b, l_k, d)
            key_mask = mask  # (b_k, l_k)

        query = reshape_heads(
            self.proj_q(sequence_states), self.num_heads
        )  # (b, l_q, h, d_h)
        key = reshape_heads(self.proj_k(key_states), self.num_heads)  # (b, l_k, h, d_h)
        value = reshape_heads(
            self.proj_v(key_states), self.num_heads
        )  # (b, l_k, h, d_h)
        pair_bias = self._resolve_pair_bias(z, model_cache).repeat_interleave(
            multiplicity,
            dim=0,
        )  # (b, h, l_q, l_k)
        attended = pair_biased_attention(
            query,
            key,
            value,
            pair_bias,
            key_mask,
            self.inf,
        )  # (b, l_q, h, d_h)
        attended = attended.reshape(s.shape[0], -1, self.c_s)  # (b, l_q, d)
        gate = self.proj_g(sequence_states).sigmoid()  # (b, l_q, d)
        return self.proj_o(gate * attended)  # (b, l_q, d)
