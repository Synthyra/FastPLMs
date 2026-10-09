"""The one trainer: a loop over patches, batches and optimizer steps, with its options.

`Trainer` owns what every training job here repeats: seeding, distributed setup, placing the model
(precision, autocast, compile, DDP), the loop (patch groups, gradient accumulation, clipping,
divergence checks, the scheduler), evaluation, early stopping, checkpoints, and the end-of-run
evaluation. A subclass provides the model, the data and the two steps, and overrides a hook where
its task differs. The options:

| Option | Where | Does |
|---|---|---|
| Patching | `args.patch_size`, `args.batch_size` | The loader yields patches; a step's batch is `batch_size // patch_size` of them, forwarded together; `build_train_prefetcher` cuts the groups |
| Batching | `args.grad_accum` | Batches accumulated per optimizer step, divided across ranks; DDP syncs only on the last |
| Precision | `precision=` | The dtype the model is moved to: `fp32`, `bf16` or `fp16` |
| Autocast | `autocast=` | `bf16` runs each forward under `torch.autocast`, with the weights kept as `precision` says |
| Prefetch | `args.bugfix`, `args.prefetch_queue_size` | A thread prepares the next group behind a bounded queue; `bugfix` runs it on the caller's thread |
| Compile | `compile_mode=` | `auto` compiles on CUDA only; `off` never |
| Clipping | `args.max_grad_norm`, `args.use_autoclip` | A fixed global norm, or a percentile of the norms seen |

See `foundry/training/README.md` for the contract of every hook.
"""

from __future__ import annotations

import gc
import math
import time
import torch

from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Protocol
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from foundry.training import AutoGradClipper, clip_grad_norm, precision_map, set_seed
from foundry.training.callbacks import CallbackHandler, TrainerCallback, TrainerState
from foundry.training.checkpointing import Checkpointer, StateCheckpointer
from foundry.training.compilation import CompileMode, compile_enabled_for_environment, normalize_compile_mode
from foundry.training.config import TrainingArguments
from foundry.training.distributed import (
    DistributedState,
    barrier,
    broadcast_value,
    cleanup_distributed,
    gather_objects,
    init_distributed,
    reduce_scalar,
    unwrap_model,
    wrap_model_ddp,
)
from foundry.training.exceptions import TrainingDivergedError
from foundry.training.metric_logger import LocalMetricLogger, MetricLogger, MetricSink
from foundry.training.prefetch import PatchAccumulator, PatchGroups, build_prefetcher
from foundry.training.schedulers import WarmupCosineScheduler


ACCEPTED_AUTOCAST = ("bf16",)
"""The autocast dtypes supported: bf16 needs no loss scaling, which a float16 autocast would."""


class Scheduler(Protocol):
    """What the loop calls on a learning-rate schedule: it is stepped to an absolute step."""

    def step(self, current_step: int | None = None) -> None: ...

    def get_last_lr(self) -> list[float]: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...


@dataclass
class TrainWindow:
    """The micro-step losses and scalar metrics since the last logged window."""

    losses: list[float] = field(default_factory=list)
    metrics: dict[str, list[float]] = field(default_factory=dict)

    def clear(self) -> None:
        self.losses = []
        self.metrics = {}


class TrainerHooks:
    """The hooks of `Trainer` that do nothing until a subclass overrides them."""

    def initialize_external_services(self) -> None:
        """Run before seeding, on every rank, then all ranks wait: a login, a cache warm-up."""

    def prepare_training_resources(self) -> None:
        """Runs after the distributed state exists and the seed is set, before the model and data are built."""

    def warmup_compiled_model(self) -> None:
        """Run after compile and DDP wrapping, to trace the graph before the loop."""

    def after_final_evaluation(self, final_valid: dict[str, float], test_metrics: dict[str, float]) -> None:
        """Runs after the best checkpoint is evaluated on validation and test, before the results are published."""

    def publish_results(
        self,
        final_valid: dict[str, float] | None,
        test_metrics: dict[str, float] | None,
        checkpoint_choice: str = "best",
    ) -> None:
        """Export or upload the finished model, on the main process."""


class Trainer(TrainerHooks, ABC):
    """The training loop. Subclasses implement `get_model`, `get_datasets`, `get_data_loaders`, `train_step` and `eval_step`.

    A step receives `patches`, the list of `patch_accum` items the loader yielded, and returns
    `(loss, metrics)`: an unscaled scalar loss and a dict of scalars to log (the loop divides the loss
    by the accumulation count). `metrics` may also carry lists, strings and flat dicts of primitives,
    which only the per-micro-step log keeps. An evaluation step receives one batch as `[batch]` and
    returns `(loss, outputs)`; the outputs of every batch are gathered across ranks and handed to
    `compute_metrics`.

    Optional hooks, with their defaults: `get_optimizer` (AdamW over the trainable parameters),
    `get_scheduler` (`WarmupCosineScheduler`), `get_loss_fn`, `prepare_training_resources`,
    `place_model_on_device`, `compile_model` (static shapes), `warmup_compiled_model`,
    `build_patch_accumulator`, `transform_patch_group`, `estimate_patch_groups_per_epoch`,
    `eval_interval` (`args.eval_epochs` epochs), `initial_best_metric` (the worst value),
    `get_eval_step_kwargs`, and the reporting and ending hooks listed in the README.
    """

    default_precision: str = "fp32"
    """The precision a subclass runs in unless the constructor says otherwise."""

    verbose_sum_keys: frozenset[str] = frozenset()
    """Step metrics a verbose per-step line sums over its micro-steps instead of averaging."""

    def __init__(
        self,
        args: TrainingArguments,
        compute_metrics: Callable[[list[dict[str, Any]]], dict[str, float]] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        *,
        precision: str | None = None,
        autocast: str | None = None,
        compile_mode: str = "auto",
        metric_sink: MetricSink | None = None,
    ) -> None:
        args.validate()
        self.precision = precision or self.default_precision
        assert self.precision in precision_map, f"precision must be one of {sorted(precision_map)}, got {self.precision!r}"
        assert autocast is None or autocast in ACCEPTED_AUTOCAST, f"autocast must be None or one of {ACCEPTED_AUTOCAST}, got {autocast!r}"
        self.autocast = autocast
        self.compile_mode: CompileMode = normalize_compile_mode(compile_mode)
        self.metric_sink = metric_sink

        self.args = args
        self.compute_metrics = compute_metrics
        self.callback_handler = CallbackHandler(callbacks)
        self.state = TrainerState()

        self.dist: DistributedState | None = None

        self.model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Scheduler | None = None
        self.loss_fn: nn.Module | None = None

        self.train_loader: DataLoader | None = None
        self.valid_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None
        self.train_dataset: Dataset | None = None
        self.valid_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

        self.grad_clipper: AutoGradClipper | None = None

        self.logger: MetricLogger | None = None
        self.checkpointer: Checkpointer | None = None

        self._is_prepared = False
        self._skip_distributed_cleanup = False
        self._best_saved_this_session = False

    @abstractmethod
    def get_model(self) -> nn.Module:
        """An initialised model, not yet on the device: `prep_for_training` places, compiles and wraps it."""
        ...

    @abstractmethod
    def get_datasets(self) -> tuple[Dataset, Dataset, Dataset]:
        """`(train, valid, test)`, built after the model and the distributed state exist."""
        ...

    @abstractmethod
    def get_data_loaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """`(train, valid, test)` loaders over the datasets; under DDP the train loader needs a distributed sampler."""
        ...

    @abstractmethod
    def train_step(self, patches: list[Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        """Forward one batch (a list of patches) and return its scalar loss `()` and its metrics."""
        ...

    @abstractmethod
    def eval_step(self, patches: list[Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        """Forward one evaluation batch (`[batch]`) and return its scalar loss `()` and the outputs `compute_metrics` reads."""
        ...

    def place_model_on_device(self, model: nn.Module) -> nn.Module:
        """Move `model` to the device in the run's precision. Override to keep some parameters as they are."""
        return model.to(device=self.device, dtype=precision_map[self.precision])

    def should_compile_model(self, model: nn.Module) -> tuple[bool, str]:
        """Whether to `torch.compile` `model`, and why not when not: `auto` compiles on CUDA only."""
        del model
        return compile_enabled_for_environment(self.device, self.compile_mode)

    def get_optimizer(self) -> torch.optim.Optimizer:
        """AdamW at `args.lr` over the parameters that require a gradient."""
        trainable = [parameter for parameter in self._raw_model.parameters() if parameter.requires_grad]
        return torch.optim.AdamW(trainable, lr=self.args.lr)

    def get_scheduler(self, optimizer: torch.optim.Optimizer, num_training_steps: int) -> Scheduler:
        """A `WarmupCosineScheduler` over `num_training_steps`."""
        return WarmupCosineScheduler(optimizer=optimizer, total_steps=num_training_steps)

    def get_loss_fn(self) -> nn.Module | None:
        """A loss module for `train_step` to use through `self.loss_fn`, or None."""
        return None

    def get_eval_step_kwargs(self, prefix: str) -> dict[str, Any]:
        """Extra keyword arguments for `eval_step` in the pass named `prefix` (`valid`, `valid_final`, `test`)."""
        return {}

    def build_patch_accumulator(self, data_iter: Any) -> PatchGroups:
        """How the loader's patches are cut into groups: `patch_accum` consecutive patches."""
        return PatchAccumulator(data_iter, self.args.patch_accum)

    def transform_patch_group(self, patches: list[Any]) -> list[Any]:
        """Prepare a group on the prefetch thread, for example by reading cached embeddings. Identity by default."""
        return patches

    def prefetch_queue_size(self) -> int:
        """Groups the prefetch thread may hold ahead of the step."""
        return self.args.prefetch_queue_size

    def build_train_prefetcher(self, data_iter: Any) -> PatchGroups:
        """The accumulator behind a prefetcher: threaded, or on the caller's thread under `args.bugfix`."""
        return build_prefetcher(
            self.build_patch_accumulator(data_iter),
            asynchronous=not self.args.bugfix,
            queue_size=self.prefetch_queue_size(),
            transform_fn=self.transform_patch_group,
        )

    def estimate_patch_groups_per_epoch(self) -> int:
        """Groups the train loader yields in an epoch, which sizes the schedule when `max_steps` is not set."""
        return len(self.train_loader) // self.args.patch_accum

    def build_logger(self) -> MetricLogger:
        """Where metrics go: files in `args.save_dir`, and `metric_sink` when given."""
        return LocalMetricLogger(self.args.save_dir, self.is_main_process, sink=self.metric_sink)

    def build_checkpointer(self) -> Checkpointer:
        """Where checkpoints go: `args.save_dir`."""
        return StateCheckpointer(self.args.save_dir)

    def average_window(self, window: TrainWindow) -> dict[str, float]:
        """The window's mean loss and the mean of each metric's finite values (NaN when none is finite)."""
        averaged: dict[str, float] = {}
        if window.losses:
            averaged["loss"] = sum(window.losses) / len(window.losses)
        for name, values in window.metrics.items():
            if values:
                finite_values = [value for value in values if math.isfinite(value)]
                averaged[name] = sum(finite_values) / len(finite_values) if finite_values else float("nan")
        return averaged

    def train_reduce_op(self, name: str) -> str:
        """How a logged training metric combines across ranks: `avg` or `sum`."""
        del name
        return "avg"

    def finalize_train_metrics(self, reduced: dict[str, float]) -> dict[str, float]:
        """Derive metrics from the rank-reduced window, such as a rate from summed counts. Unchanged by default."""
        return reduced

    def progress_postfix(self, reduced: dict[str, float]) -> dict[str, str]:
        """What the progress bar shows after a logged window."""
        return {"loss": f"{reduced['loss']:.4f}", "lr": f"{reduced['lr']:.2e}"}

    def describe_metrics(self, prefix: str, metrics: dict[str, float]) -> list[str]:
        """Extra console lines under a metrics line, such as a confusion matrix. None by default."""
        del prefix, metrics
        return []

    def resolve_tracked_metric(self, metrics: dict[str, float]) -> tuple[str, float]:
        """The name and value of `args.tracked_metric` in `metrics`; raises when it is absent."""
        tracked = self.args.tracked_metric
        assert tracked in metrics, f"Tracked metric '{tracked}' not found in eval metrics. Available: {list(metrics.keys())}"
        return tracked, metrics[tracked]

    def eval_interval(self) -> int:
        """Optimizer steps between validation passes: `args.eval_epochs` epochs' worth, and at least one."""
        steps_per_epoch = max(1, self.state.max_steps // max(1, self.state.max_epochs))
        return max(1, int(steps_per_epoch * self.args.eval_epochs))

    def initial_best_metric(self) -> float:
        """The tracked metric before any validation pass: the worst value, so the first pass improves on it."""
        return float("-inf") if self.args.tracked_metric_direction == "max" else float("inf")

    def compile_model(self, model: nn.Module) -> nn.Module:
        """`model` compiled with static shapes, once `should_compile_model` has agreed. Override for dynamic shapes."""
        return torch.compile(model, dynamic=False)

    @property
    def _raw_model(self) -> nn.Module:
        assert self.model is not None, "model not initialised yet"
        return unwrap_model(self.model)

    @property
    def device(self) -> torch.device:
        assert self.dist is not None, "distributed state not initialised"
        return self.dist.device

    @property
    def is_distributed(self) -> bool:
        return self.dist is not None and self.dist.is_distributed

    @property
    def is_main_process(self) -> bool:
        return self.dist is None or self.dist.is_main_process

    @property
    def world_size(self) -> int:
        return self.dist.world_size if self.dist is not None else 1

    @property
    def _local_grad_accum(self) -> int:
        """Batches this rank accumulates per optimizer step: `grad_accum` divided across the ranks."""
        world_size = self.world_size
        grad_accum = self.args.grad_accum
        if world_size == 1:
            return grad_accum
        assert grad_accum % world_size == 0, f"grad_accum ({grad_accum}) must be divisible by world_size ({world_size})"
        return grad_accum // world_size

    @property
    def _global_effective_batch_size(self) -> int:
        return self.args.batch_size * self._local_grad_accum * self.world_size

    def _autocast(self) -> AbstractContextManager[Any]:
        if self.autocast is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)

    def prep_for_training(self) -> None:
        """Set up the run: distributed state, seed, model, data, optimizer, logging, state, and a resume when asked."""
        self._print("[pipeline] prep_for_training() starting.")

        self.dist = init_distributed(self.args.auto_distributed_backend)
        self._print(f"Distributed: {self.is_distributed}  |  world_size: {self.world_size}  |  device: {self.device}")

        self.initialize_external_services()
        barrier(self.dist)
        set_seed(self.args.seed)

        self.prepare_training_resources()
        barrier(self.dist)

        self._build_model()
        self._build_data()
        self._print(
            "[pipeline] dataset splits: "
            f"train={len(self.train_dataset)} | valid={len(self.valid_dataset)} | test={len(self.test_dataset)}"
        )
        num_training_steps = self._build_optimization()
        self._build_state(num_training_steps)

        if self.args.resume_from_checkpoint:
            self._resume(self.args.resume_from_checkpoint)
        else:
            self._print("[pipeline] no resume_from_checkpoint path provided.")

        self._print(
            f"Effective batch size per optimizer step: "
            f"{self.args.patch_size} patch * {self.args.patch_accum} accum * "
            f"{self._local_grad_accum} local_grad_accum * {self.world_size} GPUs = "
            f"{self._global_effective_batch_size}"
        )

        self._is_prepared = True

    def _build_model(self) -> None:
        """Get the model, place it, compile it when asked, wrap it in DDP, and attach the auto-clipper."""
        self.model = self.place_model_on_device(self.get_model())

        should_compile, skip_reason = self.should_compile_model(self.model)
        if should_compile:
            self.model = self.compile_model(self.model)
            self._print("Model compiled with torch.compile")
        else:
            self._print(f"Skipping torch.compile: {skip_reason}")

        if self.is_distributed:
            self.model = wrap_model_ddp(self.model, self.dist.local_rank)
            self._print("Model wrapped with DDP")

        if should_compile:
            self.warmup_compiled_model()

        if self.args.use_autoclip:
            self.grad_clipper = AutoGradClipper(
                self._raw_model,
                clip_percentile=self.args.clip_percentile,
                history_length=self.args.clip_history_length,
            )
            self._print(f"AutoGradClipper enabled (percentile={self.args.clip_percentile})")

    def _build_data(self) -> None:
        self.train_dataset, self.valid_dataset, self.test_dataset = self.get_datasets()
        self.train_loader, self.valid_loader, self.test_loader = self.get_data_loaders()

    def _build_optimization(self) -> int:
        """Build the optimizer, schedule and loss; return the number of steps the schedule spans."""
        num_training_steps = self._estimate_training_steps()
        self._print(f"[pipeline] estimated training steps: {num_training_steps}")
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler(self.optimizer, num_training_steps)
        self.loss_fn = self.get_loss_fn()
        self.optimizer.zero_grad()
        return num_training_steps

    def _build_state(self, num_training_steps: int) -> None:
        """Open the logger and the checkpointer, and start the state at step 0."""
        self.logger = self.build_logger()
        self.checkpointer = self.build_checkpointer()

        self.state.max_steps = self.args.max_steps if self.args.max_steps > 0 else num_training_steps
        self.state.max_epochs = self.args.num_epochs
        self.state.best_metric = self.initial_best_metric()

    def _resume(self, checkpoint: str) -> None:
        payload = self.checkpointer.load_checkpoint(
            checkpoint,
            model=self._raw_model,
            device=self.device,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        if "trainer_state" in payload:
            for key, value in payload["trainer_state"].items():
                if hasattr(self.state, key):
                    setattr(self.state, key, value)
            if self.state.epoch > 0 and self.state.global_step < self.state.max_steps:
                # The loader's position is not stored, so the current epoch replays from its start
                # instead of silently skipping what was left of it.
                self.state.epoch -= 1
        self._print(f"Resumed checkpoint from {payload['resolved_path']}")
        barrier(self.dist)

    def _estimate_training_steps(self) -> int:
        if self.args.max_steps > 0:
            return self.args.max_steps

        patch_groups_per_epoch = self.estimate_patch_groups_per_epoch()
        steps_per_epoch = patch_groups_per_epoch // max(1, self._local_grad_accum)
        estimated_steps = max(1, steps_per_epoch * self.args.num_epochs)
        self._print(
            f"[pipeline] _estimate_training_steps() computed {estimated_steps} from patch_groups={patch_groups_per_epoch} "
            f"({steps_per_epoch} steps/epoch * {self.args.num_epochs} epochs)"
        )
        return estimated_steps

    def run(self) -> None:
        """Prepare, then train."""
        self.prep_for_training()
        self.train()

    @staticmethod
    def _shutdown_loader(loader: DataLoader | None) -> None:
        """Stop a DataLoader's persistent worker processes now instead of when it is garbage collected."""
        if loader is None:
            return
        iterator = getattr(loader, "_iterator", None)
        if iterator is None:
            return
        iterator._shutdown_workers()
        loader._iterator = None

    def reset_for_trial(self, args: TrainingArguments) -> None:
        """Rebuild the model, optimizer, loaders and state under new `args`, keeping the distributed state.

        For an in-process loop over trials that wants to share its expensive setup. Call
        `prep_for_training` once first; afterwards `train` can run at once. Callbacks are dropped.
        """
        assert self.dist is not None, "Must call prep_for_training() before reset_for_trial()"
        self._print("[pipeline] reset_for_trial() starting.")

        self._shutdown_loader(self.train_loader)
        self._shutdown_loader(self.valid_loader)
        self._shutdown_loader(self.test_loader)

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.grad_clipper = None

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

        # Compiled graphs from trials with other configurations would otherwise accumulate.
        torch._dynamo.reset()

        args.validate()
        self.args = args
        set_seed(self.args.seed)

        self._build_model()
        self._build_data()
        num_training_steps = self._build_optimization()

        if self.logger is not None:
            self.logger.finish()
        self.state = TrainerState()
        self._build_state(num_training_steps)
        self._best_saved_this_session = False

        self.callback_handler = CallbackHandler([])

        self._is_prepared = True
        self._print("[pipeline] reset_for_trial() complete.")

    def _prepare_sampler_epoch(self) -> None:
        """Tell the train loader's sampler which epoch begins, so a distributed run reshuffles it."""
        batch_sampler = self.train_loader.batch_sampler
        if self.is_distributed:
            if hasattr(batch_sampler, "set_epoch"):
                batch_sampler.set_epoch(self.state.epoch)
            elif hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(self.state.epoch)

    def train(self) -> nn.Module:
        """Run the loop to the step or epoch limit or an early stop, evaluate the best checkpoint, and return the unwrapped model."""
        assert self._is_prepared, "Call prep_for_training() before train()"

        self._dispatch("on_train_begin")

        while True:
            window = TrainWindow()
            progress = self._build_progress(self.state.max_steps, "Training")
            if self.state.global_step > 0:
                progress.update(self.state.global_step)

            try:
                while not self._should_stop():
                    self._run_epoch(window, progress)
                    if self._should_stop():
                        break

                break

            except KeyboardInterrupt:
                choice = self._handle_interrupt(progress)
                if choice == "continue":
                    self.state.should_stop = False
                    continue
                final_valid, test_metrics = self._post_training_eval("best")
                self.publish_results(final_valid, test_metrics, "best")
                self._post_training_cleanup()
                return self._raw_model
            finally:
                progress.close()

        final_valid, test_metrics = self._post_training_eval("best")
        self.publish_results(final_valid, test_metrics, "best")
        self._post_training_cleanup()
        return self._raw_model

    def _run_epoch(self, window: TrainWindow, progress: tqdm) -> None:
        """One pass over the train loader, which ends early when a micro-step asks the loop to stop."""
        self.state.epoch += 1
        self._dispatch("on_epoch_begin")

        self._prepare_sampler_epoch()

        self.model.train()
        prefetcher = self.build_train_prefetcher(iter(self.train_loader))

        micro_step = 0

        try:
            while True:
                patches, exhausted = prefetcher.next()
                if not patches:
                    break

                micro_step += 1
                if self._run_micro_step(patches, exhausted, micro_step, window, progress):
                    break
        finally:
            prefetcher.shutdown()

        self._dispatch("on_epoch_end")

    def _run_micro_step(
        self,
        patches: list[Any],
        exhausted: bool,
        micro_step: int,
        window: TrainWindow,
        progress: tqdm,
    ) -> bool:
        """Forward and backward one batch; on the last of an optimizer step's batches, step. True when the epoch should end."""
        local_grad_accum = self._local_grad_accum
        is_sync = (micro_step % local_grad_accum == 0) or exhausted

        # Gradient sync waits for the last micro-step of an optimizer step.
        sync_context: AbstractContextManager[Any] = nullcontext()
        if self.is_distributed and isinstance(self.model, nn.parallel.DistributedDataParallel) and not is_sync:
            sync_context = self.model.no_sync()

        if is_sync:
            self._dispatch("on_step_begin")

        with sync_context:
            with self._autocast():
                loss, step_metrics = self.train_step(patches)  # loss: ()
            scaled_loss = loss / float(local_grad_accum)  # ()
            self._dispatch("on_before_backward", loss=scaled_loss)
            scaled_loss.backward()
            self._dispatch("on_after_backward")

        loss_value = float(loss.detach())
        window.losses.append(loss_value)

        if self.is_main_process:
            self.logger.log_micro_step(self._micro_step_payload(loss_value, step_metrics, micro_step, is_sync))

        for name, value in step_metrics.items():
            # Lists, strings and dicts live in the micro-step log only; windows hold numbers.
            if isinstance(value, (bool, int, float)):
                number = float(value)
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                number = float(value.item())
            else:
                continue
            window.metrics.setdefault(name, []).append(number)

        self._check_loss(loss_value, window)

        if is_sync and self._optimizer_step(window, progress):
            return True

        return self._should_stop()

    def _micro_step_payload(
        self, loss_value: float, step_metrics: dict[str, Any], micro_step: int, is_sync: bool
    ) -> dict[str, Any]:
        """The forensic record of one forward pass: where it sat in the accumulation, its loss, then its metrics."""
        local_grad_accum = self._local_grad_accum
        payload: dict[str, Any] = {
            "event": "micro_step",
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "micro_step": micro_step,
            "micro_step_within_accum": ((micro_step - 1) % local_grad_accum) + 1,
            "grad_accum_microsteps": local_grad_accum,
            "is_sync": bool(is_sync),
        }
        if "n_examples" in step_metrics:
            payload["is_full_batch"] = int(step_metrics["n_examples"]) == int(self.args.batch_size)
        payload["batch_size_configured"] = int(self.args.batch_size)
        payload["timestamp"] = time.time()
        payload["loss"] = loss_value
        payload.update(step_metrics)
        return payload

    def _check_loss(self, loss_value: float, window: TrainWindow) -> None:
        """Raise `TrainingDivergedError` for a non-finite loss, or a spike over the best of the warmup window."""
        if self.args.divergence_nan_check and not math.isfinite(loss_value):
            raise TrainingDivergedError(
                reason="non-finite training loss",
                step=self.state.global_step,
                value=loss_value,
                kind="nan_loss" if math.isnan(loss_value) else "inf_loss",
            )

        if (
            self.args.divergence_spike_factor > 0
            and self.state.global_step > self.args.divergence_warmup_steps
            and len(window.losses) >= self.args.divergence_warmup_steps
        ):
            warmup_slice = list(window.losses)[: self.args.divergence_warmup_steps]
            finite_baseline = [value for value in warmup_slice if math.isfinite(value)]
            if finite_baseline:
                baseline = min(finite_baseline)
                threshold = self.args.divergence_spike_factor * max(baseline, 1e-8)
                if loss_value > threshold:
                    raise TrainingDivergedError(
                        reason=f"loss spike ({loss_value:.3f} > {self.args.divergence_spike_factor}x baseline {baseline:.3f})",
                        step=self.state.global_step,
                        value=loss_value,
                        baseline=baseline,
                        kind="loss_spike",
                    )

    def _clip(self) -> float:
        """Clip the accumulated gradients as configured and return the norm: before a fixed clip, the clip value of auto-clip, else 0."""
        if self.grad_clipper is not None:
            return self.grad_clipper.clip_gradients()
        if self.args.max_grad_norm > 0:
            return clip_grad_norm(self.model, self.args.max_grad_norm)
        return 0.0

    def _optimizer_step(self, window: TrainWindow, progress: tqdm) -> bool:
        """Clip, step the optimizer and schedule, then log and evaluate on their cadences. True when training should stop."""
        local_grad_accum = self._local_grad_accum
        self._dispatch("on_before_optimizer_step", optimizer=self.optimizer)

        grad_norm = self._clip()

        if self.args.divergence_grad_norm_max > 0:
            grad_norm_value = float(grad_norm)
            if not math.isfinite(grad_norm_value) or grad_norm_value > self.args.divergence_grad_norm_max:
                raise TrainingDivergedError(
                    reason=f"grad_norm={grad_norm_value}",
                    step=self.state.global_step,
                    value=grad_norm_value,
                    kind="grad_nan",
                )

        self.optimizer.step()
        self.state.global_step += 1
        self.scheduler.step(current_step=self.state.global_step)
        self.optimizer.zero_grad()

        self._dispatch("on_step_end")
        progress.update(1)

        steps_per_epoch = max(1, self.state.max_steps // max(1, self.state.max_epochs))
        epoch_frac = self.state.global_step / steps_per_epoch

        if self.args.verbose:
            step_metrics = self._verbose_reduce_step(
                window.losses[-local_grad_accum:],
                {name: values[-local_grad_accum:] for name, values in window.metrics.items()},
            )
            step_metrics["grad_norm"] = grad_norm
            step_metrics["lr"] = float(self.scheduler.get_last_lr()[0])
            step_metrics["epoch_frac"] = epoch_frac
            step_metrics["grad_accum_microsteps"] = float(local_grad_accum)
            if self.is_main_process:
                self._emit_metrics(prefix="train_verbose", metrics=step_metrics, step=self.state.global_step)

        if self.state.global_step % self.args.log_every == 0:
            self._log_window(window, progress, grad_norm, epoch_frac)

        if self.state.global_step % self.eval_interval() == 0:
            return self._evaluate_and_check_patience()

        return False

    def _log_window(self, window: TrainWindow, progress: tqdm, grad_norm: float, epoch_frac: float) -> None:
        """Average the window, reduce it across ranks, report it, and start a new window."""
        averaged = self.average_window(window)
        averaged["grad_norm"] = grad_norm
        averaged["lr"] = float(self.scheduler.get_last_lr()[0])
        averaged["epoch_frac"] = epoch_frac
        reduced = {
            name: reduce_scalar(value, self.device, self.is_distributed, op=self.train_reduce_op(name))
            for name, value in averaged.items()
        }
        reduced = self.finalize_train_metrics(reduced)

        if self.is_main_process:
            progress.set_postfix(self.progress_postfix(reduced))

        self.state.train_loss = reduced["loss"]
        self._dispatch("on_log", logs=reduced)
        self._emit_metrics(prefix="train", metrics=reduced, step=self.state.global_step)

        window.clear()

    def _evaluate_and_check_patience(self) -> bool:
        """Evaluate validation, update the best checkpoint and the patience count, and say whether to stop."""
        valid_metrics = self.evaluate(self.valid_loader, prefix="valid")
        self._dispatch("on_evaluate", metrics=valid_metrics)

        if self.is_main_process:
            self._update_patience(valid_metrics)

        should_stop = broadcast_value(self.state.should_stop, self.device, self.is_distributed, src=0)
        self.state.should_stop = bool(should_stop)
        if self.state.should_stop:
            self._print(f"Early stopping at step {self.state.global_step}")
            return True
        return False

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, prefix: str = "valid") -> dict[str, float]:
        """One pass over `data_loader` in eval mode: the mean loss, then `compute_metrics` on the outputs gathered from every rank."""
        assert self._is_prepared, "Call prep_for_training() before evaluate()"

        was_training = self.model.training
        self.model.eval()

        local_loss_sum = 0.0
        local_loss_count = 0
        all_outputs: list[dict[str, Any]] = []

        eval_progress = tqdm(
            total=len(data_loader),
            desc=f"{prefix} eval",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            disable=not self.is_main_process,
        )
        step_kwargs = self.get_eval_step_kwargs(prefix)

        try:
            for batch in data_loader:
                with self._autocast():
                    loss, outputs = self.eval_step([batch], **step_kwargs)
                local_loss_sum += float(loss.detach())
                local_loss_count += 1
                all_outputs.append(outputs)
                eval_progress.update(1)
        finally:
            eval_progress.close()

        global_loss_sum = reduce_scalar(local_loss_sum, self.device, self.is_distributed, op="sum")
        global_loss_count = reduce_scalar(float(local_loss_count), self.device, self.is_distributed, op="sum")
        average_loss = global_loss_sum / max(1.0, global_loss_count)

        metrics: dict[str, float] = {"loss": average_loss}

        if self.compute_metrics is not None:
            merged: list[dict[str, Any]] = all_outputs
            if self.is_distributed:
                gathered = gather_objects(all_outputs, self.world_size, self.is_distributed)
                if self.is_main_process:
                    merged = []
                    for rank_outputs in gathered:
                        merged.extend(rank_outputs)
            if self.is_main_process:
                metrics.update(self.compute_metrics(merged))

        if self.is_main_process:
            self._emit_metrics(prefix=prefix, metrics=metrics, step=self.state.global_step)

        if was_training:
            self.model.train()

        barrier(self.dist)
        return metrics

    def _update_patience(self, metrics: dict[str, float]) -> None:
        """Save the latest checkpoint, and the best one when the tracked metric improved; else count toward `patience`."""
        tracked, value = self.resolve_tracked_metric(metrics)

        if self.args.save_latest_checkpoint:
            self.checkpointer.save_latest(
                model=self._raw_model,
                step=self.state.global_step,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                trainer_state=self.state,
                args=self.args,
            )

        if self.args.tracked_metric_direction == "max":
            improved = value > self.state.best_metric
        else:
            improved = value < self.state.best_metric

        if improved:
            self.state.best_metric = value
            self.state.best_metric_step = self.state.global_step
            self.state.patience_counter = 0
            self._print(
                f"[patience] Improved {tracked} = {value:.6f} at step {self.state.global_step} "
                f"(best metric step={self.state.best_metric_step}, patience={self.state.patience_counter}/{self.args.patience})"
            )
            path = self.checkpointer.save(
                self._raw_model,
                self.state.global_step,
                is_best=True,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                trainer_state=self.state,
                args=self.args,
            )
            self._best_saved_this_session = True
            self._dispatch("on_save", path=path)
        else:
            self.state.patience_counter += 1
            self._print(f"No improvement in {tracked} ({self.state.patience_counter}/{self.args.patience})")
            if self.state.patience_counter >= self.args.patience:
                self.state.should_stop = True

    def _handle_interrupt(self, progress: tqdm) -> str:
        """Ask on the main process whether to stop and run the final evaluation (`evaluate`) or carry on (`continue`)."""
        progress.close()
        choice = 0

        if self.is_main_process:
            print("\n[interrupt] Stop training and run final eval? (y/n) [n]: ", end="", flush=True)
            try:
                answer = input().strip().lower()
                choice = 1 if answer in ("y", "yes") else 0
            except (KeyboardInterrupt, EOFError):
                print("\n[interrupt] Second interrupt received. Stopping.")
                choice = 1

        choice = broadcast_value(choice, self.device, self.is_distributed, src=0)
        return "evaluate" if choice == 1 else "continue"

    def _post_training_eval(self, checkpoint_choice: str = "best") -> tuple[dict[str, float], dict[str, float]]:
        """Load the best (or latest) checkpoint and evaluate validation as `valid_final` and test as `test`."""
        barrier(self.dist)

        if checkpoint_choice == "best" and self.checkpointer.best_model_path and self._best_saved_this_session:
            self.checkpointer.load_best(self._raw_model, self.device)
            self._print("Best model loaded for final evaluation")
        elif checkpoint_choice == "best" and self.checkpointer.best_model_path:
            self._print("Skipping best checkpoint load: no checkpoint was saved this session (stale checkpoint from a previous run)")
        elif checkpoint_choice == "latest" and self.checkpointer.latest_checkpoint_path:
            self.checkpointer.load_checkpoint(
                self.checkpointer.latest_checkpoint_path,
                model=self._raw_model,
                device=self.device,
            )
            self._print("Latest checkpoint loaded for final evaluation")

        barrier(self.dist)

        self.state.global_step += 1
        self._print("=== Final validation ===")
        final_valid = self.evaluate(self.valid_loader, prefix="valid_final")
        self._print("=== Test evaluation ===")
        test_metrics = self.evaluate(self.test_loader, prefix="test")

        self.after_final_evaluation(final_valid, test_metrics)

        self.state.final_valid_metrics = dict(final_valid) if final_valid is not None else None
        self.state.test_metrics = dict(test_metrics) if test_metrics is not None else None
        return final_valid, test_metrics

    def _post_training_cleanup(self) -> None:
        self._dispatch("on_train_end")
        if self.logger is not None:
            self.logger.finish()
        if not self._skip_distributed_cleanup:
            cleanup_distributed(self.dist)

    def _should_stop(self) -> bool:
        reached_steps = self.args.max_steps > 0 and self.state.global_step >= self.args.max_steps
        reached_epochs = self.args.max_steps <= 0 and self.state.epoch > self.args.num_epochs
        return self.state.should_stop or reached_steps or reached_epochs

    def _dispatch(self, event: str, **kwargs: Any) -> None:
        self.callback_handler.dispatch(event, self.state, self.model, **kwargs)

    def _emit_metrics(self, prefix: str, metrics: dict[str, float], step: int) -> None:
        """Print one metrics line (and `describe_metrics` lines under it) and log the metrics, on the main process."""
        if not self.is_main_process:
            return

        ordered = ", ".join(
            f"{name}={float(value):.6f}" if isinstance(value, (int, float, bool)) else f"{name}={value}"
            for name, value in metrics.items()
        )
        self._print(f"[{prefix}] step={step} | {ordered}")
        for line in self.describe_metrics(prefix, metrics):
            self._print(line)

        if self.logger is not None:
            self.logger.log(metrics, step, prefix=prefix, epoch=self.state.epoch)

    def _print(self, message: str) -> None:
        if self.is_main_process:
            print(message)

    def _build_progress(self, total: int, desc: str) -> tqdm:
        return tqdm(
            total=total,
            desc=desc,
            unit="step",
            dynamic_ncols=True,
            leave=True,
            disable=not self.is_main_process,
        )

    def _verbose_reduce_step(self, loss_values: list[float], metric_values: dict[str, list[float]]) -> dict[str, float]:
        """One optimizer step's micro-step values reduced to one number each: summed for `verbose_sum_keys`, else averaged."""
        reduced: dict[str, float] = {}
        if loss_values:
            reduced["loss"] = sum(loss_values) / len(loss_values)
        for name, values in metric_values.items():
            if not values:
                continue
            finite = [value for value in values if math.isfinite(value)]
            if not finite:
                reduced[name] = float("nan")
            elif name in self.verbose_sum_keys:
                reduced[name] = float(sum(finite))
            else:
                reduced[name] = sum(finite) / len(finite)
        return reduced
