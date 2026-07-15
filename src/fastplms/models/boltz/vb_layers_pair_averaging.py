"""Pair-weighted aggregation of Boltz2 MSA states."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from . import vb_layers_initialize as init


class PairWeightedAveraging(nn.Module):
    """Update M with pair-derived attention weights while preserving its shape."""

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_h: int,
        num_heads: int,
        inf: float = 1e6,
    ) -> None:
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_h = c_h
        self.num_heads = num_heads
        self.inf = inf
        self.norm_m = nn.LayerNorm(c_m)
        self.norm_z = nn.LayerNorm(c_z)
        self.proj_m = nn.Linear(c_m, c_h * num_heads, bias=False)
        self.proj_g = nn.Linear(c_m, c_h * num_heads, bias=False)
        self.proj_z = nn.Linear(c_z, num_heads, bias=False)
        self.proj_o = nn.Linear(c_h * num_heads, c_m, bias=False)
        init.final_init_(self.proj_o.weight)

    def _attention_weights(self, pair_states: Tensor, mask: Tensor) -> Tensor:
        logits = self.proj_z(pair_states).permute(0, 3, 1, 2)
        logits = logits + (1 - mask[:, None]) * -self.inf
        return torch.softmax(logits, dim=-1)

    def _all_heads(self, msa_states: Tensor, pair_states: Tensor, mask: Tensor) -> Tensor:
        values = self.proj_m(msa_states).reshape(
            *msa_states.shape[:3],
            self.num_heads,
            self.c_h,
        )
        values = values.permute(0, 3, 1, 2, 4)
        weights = self._attention_weights(pair_states, mask)
        gate = self.proj_g(msa_states).sigmoid()
        attended = torch.einsum("bhij,bhsjd->bhsid", weights, values)
        attended = attended.permute(0, 2, 3, 1, 4)
        attended = attended.reshape(*attended.shape[:3], self.num_heads * self.c_h)
        return self.proj_o(gate * attended)

    def _one_head(self, msa_states: Tensor, pair_states: Tensor, mask: Tensor, head: int) -> Tensor:
        start = head * self.c_h
        stop = start + self.c_h
        value = msa_states @ self.proj_m.weight[start:stop].T
        value = value.reshape(*value.shape[:3], 1, self.c_h).permute(0, 3, 1, 2, 4)
        logits = pair_states @ self.proj_z.weight[head : head + 1].T
        logits = logits.permute(0, 3, 1, 2)
        weights = torch.softmax(logits + (1 - mask[:, None]) * -self.inf, dim=-1)
        gate = (msa_states @ self.proj_g.weight[start:stop].T).sigmoid()
        attended = torch.einsum("bhij,bhsjd->bhsid", weights, value)
        attended = attended.permute(0, 2, 3, 1, 4).reshape(
            *msa_states.shape[:3],
            self.c_h,
        )
        return (gate * attended) @ self.proj_o.weight[:, start:stop].T

    def forward(
        self,
        m: Tensor,
        z: Tensor,
        mask: Tensor,
        chunk_heads: bool = True,
    ) -> Tensor:
        """Return updated M with shape ``(b, s, l, d_m)``."""

        msa_states = self.norm_m(m)
        pair_states = self.norm_z(z)
        if not chunk_heads or self.training:
            return self._all_heads(msa_states, pair_states, mask)

        output: Tensor | None = None
        for head in range(self.num_heads):
            contribution = self._one_head(msa_states, pair_states, mask, head)
            output = contribution if output is None else output + contribution
        if output is None:
            raise ValueError("num_heads must be positive")
        return output
