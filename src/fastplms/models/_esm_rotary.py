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

    # tensor: (..., d)
    first, second = tensor.chunk(2, dim=-1)  # (..., d / 2), (..., d / 2)
    return torch.cat((-second, first), dim=-1)  # (..., d)


def apply_rotary_pos_emb(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply cached rotary factors to X with shape ``(b, h, l, d)``."""

    # tensor: (b, h, l, d); cos, sin: (1, 1, l_cache, d)
    cos = cos[:, :, : tensor.shape[-2], :]  # (1, 1, l, d)
    sin = sin[:, :, : tensor.shape[-2], :]  # (1, 1, l, d)
    return tensor * cos + _rotate_half(tensor) * sin  # (b, h, l, d)


class RotaryEmbedding(nn.Module):
    """Apply rotary position embeddings to query and key tensors."""

    inv_freq: torch.Tensor

    def __init__(self, dim: int) -> None:
        super().__init__()
        frequencies = 1.0 / (  # (d / 2,)
            10_000 ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
        )
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
        # tensor: (..., l, d)
        seq_len = tensor.shape[seq_dimension]
        # Row t depends only on t and ``inv_freq``, so the first l rows of a
        # longer table equal a table built at length l. A shorter request can
        # therefore reuse the cache, provided ``inv_freq`` still has the dtype
        # the cache was built from. A length change used to rebuild from the
        # current ``inv_freq``; the dtype test preserves that after ``.to(dtype)``.
        # Autograd must save these factors, so inference-created tables cannot
        # be reused in a gradient-enabled forward, even when the module is in eval mode.
        cached_prefix_is_valid = (
            self._cos_cached is not None
            and self._sin_cached is not None
            and self._seq_len_cached is not None
            and self._cos_cached.device == tensor.device
            and not (
                torch.is_grad_enabled()
                and (self._cos_cached.is_inference() or self._sin_cached.is_inference())
            )
            and (
                seq_len == self._seq_len_cached
                or (
                    seq_len < self._seq_len_cached and self._cos_cached.dtype == self.inv_freq.dtype
                )
            )
        )
        if not cached_prefix_is_valid:
            self._seq_len_cached = seq_len
            positions = torch.arange(seq_len, device=tensor.device).type_as(  # (l,)
                self.inv_freq
            )
            angles = torch.outer(positions, self.inv_freq)  # (l, d / 2)
            angles = torch.cat((angles, angles), dim=-1).to(tensor.device)  # (l, d)
            self._cos_cached = angles.cos()[None, None, :, :]  # (1, 1, l, d)
            self._sin_cached = angles.sin()[None, None, :, :]  # (1, 1, l, d)

        assert self._cos_cached is not None
        assert self._sin_cached is not None
        # Callers still receive exactly l rows: (1, 1, l, d), (1, 1, l, d)
        return self._cos_cached[:, :, :seq_len, :], self._sin_cached[:, :, :seq_len, :]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # query, key: (b, h, l, d)
        cos, sin = self._update_cos_sin_tables(  # (1, 1, l, d), (1, 1, l, d)
            key,
            seq_dimension=-2,
        )
        return (
            apply_rotary_pos_emb(query, cos, sin).to(dtype=query.dtype),  # (b, h, l, d)
            apply_rotary_pos_emb(key, cos, sin).to(dtype=key.dtype),  # (b, h, l, d)
        )
