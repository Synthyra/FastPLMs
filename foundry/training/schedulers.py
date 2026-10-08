"""Learning-rate schedules for training runs."""

from __future__ import annotations

import math
import torch

from typing import Any


class WarmupCosineScheduler:
    """Two-phase learning-rate schedule: linear warmup, then cosine decay to zero.

    The phases split `total_steps`: the first `warmup_fraction` of them (5% by default, at least one
    step) rise linearly from 0 to the optimizer's learning rate, and the rest follow
    `0.5 * (1 + cos(pi * fraction))` down to 0. `step(current_step)` sets the rate for an absolute
    step, so a resumed run lands where it stopped; without an argument it advances by one.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_fraction: float = 0.05,
    ) -> None:
        assert total_steps > 0, f"total_steps must be positive, got {total_steps}"
        assert 0.0 <= warmup_fraction < 1.0, f"warmup_fraction must be in [0, 1), got {warmup_fraction}"
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_fraction = warmup_fraction

        self.warmup_steps = max(1, int(total_steps * warmup_fraction))

        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.last_step = 0
        self._last_lr = [0.0 for _ in self.base_lrs]

        self._apply_lrs_for_step(0)

    def _factor_for_step(self, current_step: int) -> float:
        if current_step < self.warmup_steps:
            return float(current_step) / float(self.warmup_steps)
        decay_steps = self.total_steps - self.warmup_steps
        if decay_steps <= 0:
            return 1.0
        progress = min(current_step - self.warmup_steps, decay_steps)
        fraction = float(progress) / float(decay_steps)
        return 0.5 * (1.0 + math.cos(math.pi * fraction))

    def _apply_lrs_for_step(self, current_step: int) -> None:
        factor = self._factor_for_step(current_step)
        updated_lrs = []
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups, strict=True):
            lr = base_lr * factor
            group["lr"] = lr
            updated_lrs.append(lr)
        self._last_lr = updated_lrs
        self.last_step = int(current_step)

    def step(self, current_step: int | None = None) -> None:
        """Set every group's rate for `current_step`, or for the step after the last one."""
        if current_step is None:
            current_step = self.last_step + 1
        self._apply_lrs_for_step(int(current_step))

    def get_last_lr(self) -> list[float]:
        """A copy of the rates the last step set, one per parameter group."""
        return list(self._last_lr)

    def state_dict(self) -> dict[str, Any]:
        """Plain numbers and lists, so a checkpoint holding it loads with `weights_only=True`."""
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "warmup_fraction": self.warmup_fraction,
            "base_lrs": list(self.base_lrs),
            "last_step": self.last_step,
            "last_lr": list(self._last_lr),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore a `state_dict` and re-apply the rate of its last step."""
        self.total_steps = int(state_dict["total_steps"])
        self.warmup_steps = int(state_dict["warmup_steps"])
        self.warmup_fraction = float(state_dict["warmup_fraction"])
        self.base_lrs = [float(value) for value in state_dict["base_lrs"]]
        self.last_step = int(state_dict["last_step"])
        self._last_lr = [float(value) for value in state_dict["last_lr"]]
        self._apply_lrs_for_step(self.last_step)
