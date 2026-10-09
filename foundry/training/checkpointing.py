"""Training checkpoints on disk: one best file, one latest file, and the payload both hold.

`Checkpointer` is what `Trainer` calls. `StateCheckpointer` is the default. A checkpoint is
a plain dictionary of tensors, numbers and strings, so it loads with `weights_only=True`; no class
of any project is pickled into it.
"""

from __future__ import annotations

import torch

from pathlib import Path
from typing import Any, Protocol
from torch import nn

from foundry.serialization.atomic import atomic_replace
from foundry.training.callbacks import TrainerState
from foundry.training.config import TrainingArguments


BEST_FILENAME = "best_model.pt"
LATEST_FILENAME = "last_checkpoint.pt"


class Checkpointer(Protocol):
    """The calls `Trainer` makes on whatever holds the run's checkpoints."""

    @property
    def best_model_path(self) -> str | None: ...

    @property
    def latest_checkpoint_path(self) -> str | None: ...

    def save(
        self,
        model: nn.Module,
        step: int,
        is_best: bool = True,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        trainer_state: TrainerState | None = None,
        args: TrainingArguments | None = None,
        filename: str | None = None,
    ) -> str: ...

    def save_latest(
        self,
        model: nn.Module,
        step: int,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        trainer_state: TrainerState | None = None,
        args: TrainingArguments | None = None,
    ) -> str: ...

    def load_checkpoint(
        self,
        checkpoint: str,
        model: nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
    ) -> dict[str, Any]: ...

    def load_best(self, model: nn.Module, device: torch.device) -> None: ...


class StateCheckpointer:
    """Saves and loads the run's `best_model.pt` and `last_checkpoint.pt` in `save_dir`.

    A payload holds `model_state_dict` and `step`, and when given them `optimizer_state_dict`,
    `scheduler_state_dict`, `trainer_state` (`vars(TrainerState)`) and `training_args` (the arguments
    without credentials). A directory that already holds either file is picked up, so a restarted
    run resumes from it.
    """

    def __init__(self, save_dir: str) -> None:
        self.save_dir = Path(save_dir)
        self._best_model_path: str | None = None
        self._latest_checkpoint_path: str | None = None

        self.save_dir.mkdir(parents=True, exist_ok=True)
        best_path = self.save_dir / BEST_FILENAME
        latest_path = self.save_dir / LATEST_FILENAME
        if best_path.exists():
            self._best_model_path = str(best_path)
        if latest_path.exists():
            self._latest_checkpoint_path = str(latest_path)

    @property
    def best_model_path(self) -> str | None:
        """Path of the best saved checkpoint, or None when none exists."""
        return self._best_model_path

    @property
    def latest_checkpoint_path(self) -> str | None:
        """Path of the most recent resumable checkpoint, or None when none exists."""
        return self._latest_checkpoint_path

    def save(
        self,
        model: nn.Module,
        step: int,
        is_best: bool = True,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        trainer_state: TrainerState | None = None,
        args: TrainingArguments | None = None,
        filename: str | None = None,
    ) -> str:
        """Write `model`'s state dict and the rest of the payload, and return the path.

        The file is `best_model.pt` when `is_best`, else `step_<step>.pt`, unless `filename` names
        it. `model` is the unwrapped module, whose keys carry no `module.` or `_orig_mod.` prefix.
        """
        if filename is None:
            filename = BEST_FILENAME if is_best else f"step_{step}.pt"
        path = self.save_dir / filename

        payload: dict[str, Any] = {"model_state_dict": model.state_dict(), "step": int(step)}
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler_state_dict"] = scheduler.state_dict()
        if trainer_state is not None:
            payload["trainer_state"] = dict(vars(trainer_state))
        if args is not None:
            payload["training_args"] = args.to_safe_dict()

        with atomic_replace(path) as temporary:
            torch.save(payload, temporary)

        if is_best:
            self._best_model_path = str(path)
        if filename == LATEST_FILENAME:
            self._latest_checkpoint_path = str(path)
        return str(path)

    def save_latest(
        self,
        model: nn.Module,
        step: int,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        trainer_state: TrainerState | None = None,
        args: TrainingArguments | None = None,
    ) -> str:
        """Write the resumable `last_checkpoint.pt`."""
        return self.save(
            model=model,
            step=step,
            is_best=False,
            optimizer=optimizer,
            scheduler=scheduler,
            trainer_state=trainer_state,
            args=args,
            filename=LATEST_FILENAME,
        )

    def resolve_checkpoint_path(self, checkpoint: str) -> str:
        """The file `checkpoint` names: empty for the latest (else the best), a directory for the latest it holds (else its best), or a path."""
        if checkpoint == "":
            if self._latest_checkpoint_path is not None:
                return self._latest_checkpoint_path
            assert self._best_model_path is not None, "No checkpoint is available to resume from."
            return self._best_model_path

        candidate = Path(checkpoint)
        if not candidate.is_absolute():
            save_dir_candidate = self.save_dir / checkpoint
            if save_dir_candidate.exists():
                candidate = save_dir_candidate

        if candidate.is_dir():
            latest_path = candidate / LATEST_FILENAME
            best_path = candidate / BEST_FILENAME
            if latest_path.exists():
                return str(latest_path)
            assert best_path.exists(), f"No checkpoint file found in directory: {candidate}"
            return str(best_path)

        assert candidate.exists(), f"Checkpoint path not found: {candidate}"
        return str(candidate)

    def load_checkpoint(
        self,
        checkpoint: str,
        model: nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
    ) -> dict[str, Any]:
        """Load a checkpoint's weights into `model`, and its optimizer and scheduler state when present.

        A file that is a bare state dict is accepted. The payload comes back with `resolved_path`
        added, so a caller can say which file it read.
        """
        resolved_path = self.resolve_checkpoint_path(checkpoint)
        payload = torch.load(resolved_path, map_location=device)
        if "model_state_dict" not in payload:
            payload = {"model_state_dict": payload}

        model.load_state_dict(payload["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in payload:
            scheduler.load_state_dict(payload["scheduler_state_dict"])

        payload["resolved_path"] = resolved_path
        return payload

    def load_best(self, model: nn.Module, device: torch.device) -> None:
        """Load the best checkpoint's weights into `model` in place."""
        assert self._best_model_path is not None, "No best checkpoint has been saved yet."
        self.load_checkpoint(self._best_model_path, model=model, device=device)
