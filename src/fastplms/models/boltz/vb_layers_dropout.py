"""Broadcast dropout masks used by the Boltz pair stack.

The mask is sampled in ``float32`` even when the pair tensor uses a lower
precision dtype.  This matches the checkpoint implementation while keeping
the broadcast axis explicit.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _broadcast_mask_shape(Z: Tensor, *, columnwise: bool) -> tuple[int, ...]:
    """Return the row-wise or column-wise pair-mask shape for ``Z``."""

    if Z.ndim != 4:
        raise ValueError(f"pair tensor must have four dimensions, got {Z.ndim}")
    b, rows, columns, _ = Z.shape
    return (b, 1, columns, 1) if columnwise else (b, rows, 1, 1)


def get_dropout_mask(
    dropout: float,
    z: Tensor,
    training: bool,
    columnwise: bool = False,
) -> Tensor:
    """Sample an inverted-dropout mask that broadcasts over pair channels.

    ``Z`` is the pair tensor with shape ``(b, n, n, d)``.  Row-wise masks
    vary along the first residue axis; column-wise masks vary along the
    second.  Evaluation returns an all-one mask while retaining the same RNG
    call pattern as training.
    """

    probability = float(dropout) if training else 0.0
    sample_shape = _broadcast_mask_shape(z, columnwise=columnwise)
    keep = (
        torch.rand(sample_shape, dtype=torch.float32, device=z.device) >= probability
    )  # (b, 1, n, 1) or (b, n, 1, 1)
    return keep * (1.0 / (1.0 - probability))  # same broadcast-mask shape
