"""Contracts for E1 rotary-position buffers."""

from __future__ import annotations

import torch

from fastplms.models.e1.modeling_e1 import RotaryPositionalEmbedding


def test_e1_rotary_lazily_initializes_after_meta_materialization() -> None:
    """Meta-device construction cannot leave uninitialized trigonometric caches."""

    with torch.device("meta"):
        rotary = RotaryPositionalEmbedding(dim=8, max_position_embeddings=16)
    rotary = rotary.to_empty(device="cpu")

    Q = torch.randn(2, 5, 3, 8)
    K = torch.randn(2, 5, 1, 8)
    position_ids = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, -1]])
    Q_rotated, K_rotated = rotary(Q, K, position_ids)

    assert torch.isfinite(Q_rotated).all()
    assert torch.isfinite(K_rotated).all()
    assert rotary.max_seq_len_cached == 5
    assert rotary.cos_cached.shape == (5, 8)
    assert rotary.sin_cached.shape == (5, 8)
    assert rotary.cos_cached.abs().max() <= 1
    assert rotary.sin_cached.abs().max() <= 1
