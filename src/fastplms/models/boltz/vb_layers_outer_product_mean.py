"""Outer-product aggregation from MSA states to pair states."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from . import vb_layers_initialize as init


class OuterProductMean(nn.Module):
    """Aggregate M with shape ``(b, s, l, d_m)`` into pair tensor Z."""

    def __init__(self, c_in: int, c_hidden: int, c_out: int) -> None:
        super().__init__()
        self.c_hidden = c_hidden
        self.norm = nn.LayerNorm(c_in)
        self.proj_a = nn.Linear(c_in, c_hidden, bias=False)
        self.proj_b = nn.Linear(c_in, c_hidden, bias=False)
        self.proj_o = nn.Linear(c_hidden * c_hidden, c_out)
        init.final_init_(self.proj_o.weight)
        init.final_init_(self.proj_o.bias)

    @staticmethod
    def _pair_counts(mask: Tensor) -> Tensor:
        """Count valid MSA observations for each residue pair."""

        counts: Tensor | None = None
        for start in range(0, mask.shape[1], 64):
            mask_slice = mask[:, start : start + 64]
            contribution = (mask_slice[:, :, None, :] * mask_slice[:, :, :, None]).sum(dim=1)
            counts = contribution if counts is None else counts + contribution
        if counts is None:
            raise ValueError("MSA depth must be positive")
        return counts.clamp(min=1)

    @staticmethod
    def _outer_product(left: Tensor, right: Tensor, counts: Tensor) -> Tensor:
        outer = torch.einsum("bsic,bsjd->bijcd", left, right)
        return outer.reshape(*outer.shape[:3], -1) / counts

    def _chunked_projection(
        self,
        left: Tensor,
        right: Tensor,
        counts: Tensor,
        chunk_size: int,
        target: Tensor,
    ) -> Tensor:
        output: Tensor | None = None
        for start in range(0, self.c_hidden, chunk_size):
            stop = min(start + chunk_size, self.c_hidden)
            outer = self._outer_product(left[..., start:stop], right, counts)
            weight = self.proj_o.weight[
                :,
                start * self.c_hidden : stop * self.c_hidden,
            ]
            contribution = outer.to(target) @ weight.T
            output = contribution if output is None else output + contribution
        if output is None:
            raise ValueError("c_hidden must be positive")
        return output + self.proj_o.bias

    def forward(self, m: Tensor, mask: Tensor, chunk_size: int | None = None) -> Tensor:
        """Return pair tensor Z with shape ``(b, l, l, d_z)``."""

        expanded_mask = mask.unsqueeze(-1).to(m)
        normalized = self.norm(m)
        left = self.proj_a(normalized) * expanded_mask
        right = self.proj_b(normalized) * expanded_mask

        if chunk_size is not None and not self.training:
            if chunk_size <= 0:
                raise ValueError("chunk_size must be positive")
            counts = self._pair_counts(expanded_mask)
            return self._chunked_projection(
                left,
                right,
                counts,
                chunk_size,
                normalized,
            )

        pair_mask = expanded_mask[:, :, None, :] * expanded_mask[:, :, :, None]
        counts = pair_mask.sum(dim=1).clamp(min=1)
        outer = self._outer_product(left.float(), right.float(), counts)
        return self.proj_o(outer.to(normalized))
