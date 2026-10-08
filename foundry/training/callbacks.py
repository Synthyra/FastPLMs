"""Mutable training state and the callback hooks that observe it."""

from __future__ import annotations

import torch

from dataclasses import dataclass
from typing import Any, Protocol
from torch import nn


@dataclass
class TrainerState:
    """What a run has done so far; `vars(state)` is what a checkpoint stores."""

    global_step: int = 0
    epoch: int = 0
    max_steps: int = -1
    max_epochs: int = -1
    best_metric: float = float("inf")
    best_metric_step: int = 0
    train_loss: float = 0.0

    patience_counter: int = 0
    should_stop: bool = False

    final_valid_metrics: dict[str, float] | None = None
    test_metrics: dict[str, float] | None = None


class TrainerCallback:
    """Base callback class. Override any hooks you need.

    Every hook receives the current `TrainerState`, the model, and keyword arguments from the trainer.
    """

    def on_train_begin(self, state: TrainerState, model: nn.Module, **kwargs: Any) -> None:
        """The loop is about to start."""

    def on_epoch_begin(self, state: TrainerState, model: nn.Module, **kwargs: Any) -> None:
        """An epoch is about to start; `state.epoch` already counts it."""

    def on_step_begin(self, state: TrainerState, model: nn.Module, **kwargs: Any) -> None:
        """The first micro-step of an optimizer step is about to run."""

    def on_before_backward(self, state: TrainerState, model: nn.Module, loss: torch.Tensor | None = None, **kwargs: Any) -> None:
        """`loss`, already divided by the accumulation count, is about to be backpropagated."""

    def on_after_backward(self, state: TrainerState, model: nn.Module, **kwargs: Any) -> None:
        """A micro-step's gradients are accumulated."""

    def on_before_optimizer_step(
        self, state: TrainerState, model: nn.Module, optimizer: torch.optim.Optimizer | None = None, **kwargs: Any
    ) -> None:
        """The gradients are final and not yet clipped."""

    def on_step_end(self, state: TrainerState, model: nn.Module, **kwargs: Any) -> None:
        """The optimizer and scheduler stepped; `state.global_step` counts the step."""

    def on_evaluate(self, state: TrainerState, model: nn.Module, metrics: dict[str, float] | None = None, **kwargs: Any) -> None:
        """A validation pass finished with `metrics`."""

    def on_save(self, state: TrainerState, model: nn.Module, path: str | None = None, **kwargs: Any) -> None:
        """A best checkpoint was written to `path`."""

    def on_log(self, state: TrainerState, model: nn.Module, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """A training window was averaged into `logs`."""

    def on_epoch_end(self, state: TrainerState, model: nn.Module, **kwargs: Any) -> None:
        """An epoch finished or the run stopped inside it."""

    def on_train_end(self, state: TrainerState, model: nn.Module, **kwargs: Any) -> None:
        """Training and the final evaluation are over."""


class CallbackHandler:
    """Dispatches callback events to a list of `TrainerCallback` instances, in order."""

    def __init__(self, callbacks: list[TrainerCallback] | None = None) -> None:
        self.callbacks: list[TrainerCallback] = list(callbacks) if callbacks else []

    def add(self, callback: TrainerCallback) -> None:
        """Append `callback` after the ones already registered."""
        self.callbacks.append(callback)

    def dispatch(self, event: str, state: TrainerState, model: nn.Module, **kwargs: Any) -> None:
        """Call the hook named `event` on every callback; an unknown name raises AttributeError."""
        for callback in self.callbacks:
            method = getattr(callback, event)
            method(state=state, model=model, **kwargs)


class HasTestLoader(Protocol):
    """What `PeriodicTestEvaluation` needs of a trainer."""

    test_loader: Any
    is_main_process: bool

    def evaluate(self, data_loader: Any, prefix: str = "valid") -> dict[str, float]: ...


class PeriodicTestEvaluation(TrainerCallback):
    """Evaluate the test loader after every validation pass, for a run that watches test as it trains.

    Only the main process evaluates, as the Atlas trainers' `--eval_test` always did. Never select a
    checkpoint on what this reports: selection reads validation.
    """

    def __init__(self, trainer: HasTestLoader) -> None:
        self._trainer = trainer

    def on_evaluate(self, state: TrainerState, model: nn.Module, metrics: dict[str, float] | None = None, **kwargs: Any) -> None:
        if not self._trainer.is_main_process:
            return
        print("--- Test evaluation (periodic) ---")
        self._trainer.evaluate(self._trainer.test_loader, prefix="test")
