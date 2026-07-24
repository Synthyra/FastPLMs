"""Contracts for ESM3 rotary-position buffers."""

from __future__ import annotations

import torch

from fastplms.models.esm3.modeling_esm3 import RotaryEmbedding


def test_esm3_rotary_rebuilds_frequency_after_meta_materialization() -> None:
    """A meta-loaded non-persistent frequency buffer must not contain garbage."""

    with torch.device("meta"):
        rotary = RotaryEmbedding(dim=8)
    rotary = rotary.to_empty(device="cpu")

    Q = torch.randn(2, 5, 3, 8)  # (b=2, l=5, h=3, d_h=8)
    K = torch.randn(2, 5, 3, 8)  # (b=2, l=5, h=3, d_h=8)
    Q_rotated, K_rotated = rotary(Q, K)  # each: (b=2, l=5, h=3, d_h=8)

    expected_inv_freq = rotary._compute_inv_freq(torch.device("cpu"))  # (d_h / 2=4,)
    assert torch.equal(rotary.inv_freq, expected_inv_freq)
    assert torch.isfinite(Q_rotated).all()
    assert torch.isfinite(K_rotated).all()
    assert rotary._cos_cached is not None
    assert rotary._sin_cached is not None
    assert rotary._cos_cached.abs().max() <= 1
    assert rotary._sin_cached.abs().max() <= 1
