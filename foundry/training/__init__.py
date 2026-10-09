"""Model-agnostic training helpers: seeding, gradient norms and clipping, precision names, and paired-batch masks (`pair_masks`).

`set_seed` seeds Python's, NumPy's, and torch's generators, and `torch.manual_seed` seeds every
device torch has. `AutoGradClipper` clips to a percentile of the gradient norms it has seen, once
it has seen ten, adapted from https://github.com/pseeth/autoclip.
"""

from __future__ import annotations

import random
import numpy as np
import torch

from torch import nn


__all__ = ["AutoGradClipper", "clip_grad_norm", "precision_map", "set_seed"]

precision_map: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}
"""Precision names, as trainers' arguments spell them, to torch dtypes."""


def set_seed(seed: int) -> None:
    """Seed Python's, NumPy's, and torch's generators, torch's on every device."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clip_grad_norm(model: nn.Module, max_norm: float) -> float:
    """Clip `model`'s gradients to a global L2 norm of `max_norm`, and return the norm before."""
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item()


class AutoGradClipper:
    """Clips gradients to a percentile of the global norms observed so far.

    Each call records the current norm. From the tenth call on, it clips to the
    `clip_percentile` percentile of the last `history_length` norms and returns that value;
    before then it leaves the gradients alone and returns 0.0.
    """

    def __init__(self, model: nn.Module, clip_percentile: float = 10, history_length: int = 1000000) -> None:
        self.model = model
        self.clip_percentile = clip_percentile
        self.history_length = history_length
        self.grad_history: list[float] = []

    def clip_gradients(self) -> float:
        """Record the current norm, clip once ten are recorded, and return the clip value or 0.0."""
        self.grad_history.append(_get_grad_norm(self.model))
        if len(self.grad_history) > self.history_length:
            self.grad_history = self.grad_history[-self.history_length :]

        if len(self.grad_history) >= 10:
            clip_value = np.percentile(self.grad_history, self.clip_percentile)  # ()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
            return clip_value
        return 0.0


def _get_grad_norm(model: nn.Module) -> float:
    """The global L2 norm of `model`'s gradients, skipping parameters without one."""
    total_norm = 0.0
    for parameter in model.parameters():  # (...)
        if parameter.grad is not None:  # grad: (...), the parameter's shape
            total_norm += parameter.grad.data.norm(2).item() ** 2  # norm: ()
    return total_norm**0.5
