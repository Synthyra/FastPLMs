"""The training run's configuration record and its command-line round trip."""

import argparse
import sys

from dataclasses import dataclass, field, fields
from typing import Any, Self, get_type_hints

from foundry.logging import redact


@dataclass
class TrainingArguments:
    """The configuration of one `foundry.training.Trainer` run, grouped by concern.

    A subclass adds fields (a project's own settings) or overrides defaults (a project's recipe), and
    `to_safe_dict`, `from_dict`, `add_argparse_args` and `from_argparse` follow it. `extra` carries
    any argparse value that is not a field, so a project's flags travel with the arguments.

    Batching has three levels. The loader yields *patches* of `patch_size` rows. A step's batch is
    `patch_accum = batch_size // patch_size` patches, which the model forwards together. An optimizer
    step accumulates `grad_accum` batches (divided across the ranks of a distributed run).
    """

    # --- Core training hyper-parameters ---
    lr: float = 1e-4
    num_epochs: int = 10
    max_steps: int = -1
    batch_size: int = 64
    patch_size: int = 64
    grad_accum: int = 1
    max_grad_norm: float = -1
    use_autoclip: bool = False
    clip_percentile: int = 10
    clip_history_length: int = 1000000
    seed: int = 42

    # --- Divergence / NaN detection ---
    divergence_nan_check: bool = True
    divergence_spike_factor: float = 0.0
    divergence_warmup_steps: int = 100
    divergence_grad_norm_max: float = 0.0

    # --- Evaluation / patience ---
    eval_epochs: float = 1.0
    log_every: int = 100
    patience: int = 3
    tracked_metric: str = "loss"
    tracked_metric_direction: str = "min"
    verbose: bool = False

    # --- Data loading ---
    num_workers: int = 4
    prefetch_factor: int = 4
    prefetch_queue_size: int = 4
    pin_memory: bool = True

    # --- Checkpointing ---
    save_dir: str = "checkpoints"
    resume_from_checkpoint: str = ""
    save_latest_checkpoint: bool = True
    hf_save_path: str = ""
    skip_hub_upload: bool = False

    # --- Logging ---
    # No field holds a credential: W&B records the command line these fields become, so
    # huggingface_hub and wandb read HF_TOKEN and WANDB_API_KEY from the environment.
    wandb_project: str = ""
    run_name: str = ""

    # --- Misc ---
    bugfix: bool = False
    distributed_backend: str = ""

    # --- Extra user-defined fields ---
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def wandb_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        return self.hf_save_path.replace("/", "_")

    @property
    def patch_accum(self) -> int:
        """Patches in one batch: `batch_size // patch_size`, which must divide exactly."""
        assert self.batch_size % self.patch_size == 0, (
            f"batch_size ({self.batch_size}) must be divisible by patch_size ({self.patch_size})"
        )
        return self.batch_size // self.patch_size

    @property
    def effective_batch_size_per_step(self) -> int:
        return self.batch_size * self.grad_accum

    @property
    def auto_distributed_backend(self) -> str:
        """`distributed_backend` when set, else `gloo` on Windows and `nccl` elsewhere."""
        if self.distributed_backend:
            return self.distributed_backend
        return "gloo" if sys.platform == "win32" else "nccl"

    def validate(self) -> None:
        """Raise AssertionError for a value the trainer cannot run with."""
        assert self.lr > 0, f"lr must be positive, got {self.lr}"
        assert self.batch_size > 0, f"batch_size must be positive, got {self.batch_size}"
        assert self.patch_size > 0, f"patch_size must be positive, got {self.patch_size}"
        assert self.grad_accum > 0, f"grad_accum must be positive, got {self.grad_accum}"
        assert self.num_workers >= 0, f"num_workers must be non-negative, got {self.num_workers}"
        assert self.prefetch_factor > 0, f"prefetch_factor must be positive, got {self.prefetch_factor}"
        assert self.prefetch_queue_size > 0, f"prefetch_queue_size must be positive, got {self.prefetch_queue_size}"
        assert self.num_epochs > 0 or self.max_steps > 0, "Either num_epochs or max_steps must be positive"
        assert self.tracked_metric_direction in ("min", "max"), (
            f"tracked_metric_direction must be 'min' or 'max', got {self.tracked_metric_direction}"
        )
        _ = self.patch_accum

    @classmethod
    def add_argparse_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """One flag per scalar field: `--name value`, `--name` for a false bool, `--no_name` for a true one."""
        hints = get_type_hints(cls)
        for f in fields(cls):
            arg_name = f"--{f.name}"
            field_type = hints[f.name]
            if field_type is bool:
                if f.default:
                    parser.add_argument(f"--no_{f.name}", action="store_false", dest=f.name)
                else:
                    parser.add_argument(arg_name, action="store_true")
            elif field_type in (int, float, str):
                parser.add_argument(arg_name, type=field_type, default=f.default)
        return parser

    @classmethod
    def from_argparse(cls, args: argparse.Namespace) -> Self:
        """The arguments a parsed namespace holds: fields by name, every other value in `extra`."""
        known_fields = {f.name for f in fields(cls)} - {"extra"}
        kwargs: dict[str, Any] = {}
        extra: dict[str, Any] = {}

        for name, value in vars(args).items():
            if name in known_fields:
                kwargs[name] = value
            else:
                extra[name] = value

        instance = cls(**kwargs, extra=extra)
        instance.validate()
        return instance

    def to_dict(self) -> dict[str, Any]:
        """Every field by name."""
        return {f.name: getattr(self, f.name) for f in fields(self.__class__)}

    def to_safe_dict(self) -> dict[str, Any]:
        """`to_dict()` without credential-shaped keys at any depth, `extra` included.

        Use for uploads, logging, and checkpoints. `foundry.logging.redact` decides what is a
        credential: `hf_token` and `wandb_token` go, and a setting such as `token_budget` stays.
        """
        return redact(self.to_dict())

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Self:
        """Rebuild from `to_dict()` or `to_safe_dict()`: unknown top-level keys join `extra`, which `values` must carry."""
        known_fields = {f.name for f in fields(cls)} - {"extra"}
        kwargs: dict[str, Any] = {}
        extra: dict[str, Any] = {}

        assert "extra" in values, "Missing required key in TrainingArguments dict: 'extra'"
        top_level_extra = values["extra"]
        assert isinstance(top_level_extra, dict), f"Expected 'extra' to be dict, got {type(top_level_extra)}"

        for name, value in values.items():
            if name == "extra":
                continue
            if name in known_fields:
                kwargs[name] = value
            else:
                extra[name] = value

        if top_level_extra:
            extra.update(top_level_extra)

        instance = cls(**kwargs, extra=extra)
        instance.validate()
        return instance
