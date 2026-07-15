"""Stable ESM rotary embeddings independent of Transformers internals.

Transformers 5 changed both the name and call contract of its private ESM
rotary helper. FastPLMs checkpoints use the earlier two-tensor contract, so
the small mathematical primitive lives here instead of importing a private
Transformers implementation.
"""

from __future__ import annotations

import torch
from torch import nn


def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    """Rotate the final dimension of X by 90 degrees in paired subspaces."""

    first, second = tensor.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary_pos_emb(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply cached rotary factors to X with shape ``(b, h, l, d)``."""

    cos = cos[:, :, : tensor.shape[-2], :]
    sin = sin[:, :, : tensor.shape[-2], :]
    return tensor * cos + _rotate_half(tensor) * sin


class RotaryEmbedding(nn.Module):
    """Apply rotary position embeddings to query and key tensors."""

    inv_freq: torch.Tensor

    def __init__(self, dim: int) -> None:
        super().__init__()
        frequencies = 1.0 / (10_000 ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        # Keep this persistent to preserve the historical checkpoint schema.
        self.register_buffer("inv_freq", frequencies)
        self._seq_len_cached: int | None = None
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None

    def _update_cos_sin_tables(
        self,
        tensor: torch.Tensor,
        seq_dimension: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = tensor.shape[seq_dimension]
        cache_stale = (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != tensor.device
        )
        if cache_stale:
            self._seq_len_cached = seq_len
            positions = torch.arange(seq_len, device=tensor.device).type_as(self.inv_freq)
            angles = torch.outer(positions, self.inv_freq)
            angles = torch.cat((angles, angles), dim=-1).to(tensor.device)
            self._cos_cached = angles.cos()[None, None, :, :]
            self._sin_cached = angles.sin()[None, None, :, :]

        assert self._cos_cached is not None
        assert self._sin_cached is not None
        return self._cos_cached, self._sin_cached

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self._update_cos_sin_tables(key, seq_dimension=-2)
        return (
            apply_rotary_pos_emb(query, cos, sin).to(dtype=query.dtype),
            apply_rotary_pos_emb(key, cos, sin).to(dtype=key.dtype),
        )
