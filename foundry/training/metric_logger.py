"""Where a trainer's metrics go: files under the run's directory, and optionally a run record.

`MetricLogger` is what `Trainer` calls. `LocalMetricLogger` is the default: it appends every row to
`metrics.log` and every micro-step to `micro_steps.jsonl` in `save_dir`, and forwards the scalars to
a sink when given one, normally a `foundry.logging.Run`, which owns the W&B run and its key budget.
It never starts a W&B run itself ([the W&B convention](../../docs/conventions/wandb.md): one run per
job, through `foundry.logging`).
"""

from __future__ import annotations

import json
import numpy as np
import torch

from collections.abc import Mapping
from pathlib import Path
from typing import IO, Any, Protocol


class MetricLogger(Protocol):
    """The three calls `Trainer` makes; every one is a no-op off the main process."""

    def log(self, metrics: Mapping[str, Any], step: int, prefix: str = "", epoch: int | None = None) -> None: ...

    def log_micro_step(self, payload: dict[str, Any]) -> None: ...

    def finish(self) -> None: ...


class MetricSink(Protocol):
    """Anything that takes a step's scalars, such as `foundry.logging.Run`."""

    def log(self, values: dict[str, Any], *, step: int) -> None: ...


class LocalMetricLogger:
    """Appends metrics to `save_dir/metrics.log` and micro-steps to `save_dir/micro_steps.jsonl`.

    Both files are opened in append mode, so a resumed run keeps its history. Only the main process
    writes. Scalars (numbers, bools, one-element tensors) are kept; other values are skipped with one
    notice per key. Keys ending in `_std` stay in the files and are not sent to the sink, since a
    per-subsample deviation is a diagnostic and not a chart.
    """

    def __init__(self, save_dir: str, is_main_process: bool, sink: MetricSink | None = None) -> None:
        self.is_main_process = is_main_process
        self.sink = sink
        self._log_file: IO[str] | None = None
        self._micro_step_file: IO[str] | None = None
        self._skipped_metric_keys: set[str] = set()

        if self.is_main_process:
            directory = Path(save_dir)
            directory.mkdir(parents=True, exist_ok=True)
            self._log_file = (directory / "metrics.log").open("a", encoding="utf-8")
            self._micro_step_file = (directory / "micro_steps.jsonl").open("a", encoding="utf-8")
            self._write_log({"event": "logger_init", "save_dir": save_dir})

    def log(self, metrics: Mapping[str, Any], step: int, prefix: str = "", epoch: int | None = None) -> None:
        """Append one row of scalar `metrics`, each key under `prefix/`, and send the sink the same row."""
        if not self.is_main_process:
            return

        if prefix:
            metrics = {f"{prefix}/{name}": value for name, value in metrics.items()}

        safe_metrics = self._scalars(metrics)
        if not safe_metrics:
            return

        self._write_log({"step": step, "epoch": epoch, "prefix": prefix, "metrics": safe_metrics})

        if self.sink is not None:
            charted = {name: value for name, value in safe_metrics.items() if not name.endswith("_std")}
            if charted:
                self.sink.log(charted, step=step)

    def log_micro_step(self, payload: dict[str, Any]) -> None:
        """Append one JSON line holding `payload`, whose lists and flat dicts of primitives are kept whole."""
        if not self.is_main_process or self._micro_step_file is None:
            return
        self._micro_step_file.write(json.dumps(_micro_step_record(payload), sort_keys=True) + "\n")
        self._micro_step_file.flush()

    def finish(self) -> None:
        """Close both files. The sink is the caller's to finish."""
        if not self.is_main_process:
            return

        if self._log_file is not None:
            self._log_file.flush()
            self._log_file.close()
            self._log_file = None

        if self._micro_step_file is not None:
            self._micro_step_file.flush()
            self._micro_step_file.close()
            self._micro_step_file = None

    def _scalars(self, metrics: Mapping[str, Any]) -> dict[str, float | bool]:
        scalars: dict[str, float | bool] = {}
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    scalars[name] = float(value.item())
                else:
                    self._skip_non_scalar(name, value)
            elif isinstance(value, bool):
                scalars[name] = bool(value)
            elif isinstance(value, (np.integer, np.floating, np.bool_, int, float)):
                scalars[name] = float(value)
            else:
                self._skip_non_scalar(name, value)
        return scalars

    def _skip_non_scalar(self, key: str, value: Any) -> None:
        if key in self._skipped_metric_keys:
            return
        print(f"[LocalMetricLogger] Skipping non-scalar metric '{key}' of type {type(value).__name__}.")
        self._skipped_metric_keys.add(key)

    def _write_log(self, payload: dict[str, Any]) -> None:
        if self._log_file is None:
            return
        self._log_file.write(json.dumps(payload, sort_keys=True) + "\n")
        self._log_file.flush()


def _micro_step_record(payload: dict[str, Any]) -> dict[str, Any]:
    """`payload` with tensors and NumPy scalars as Python numbers; a value that JSON cannot hold raises."""
    record: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            assert value.numel() == 1, f"micro_step payload field '{key}' is a non-scalar tensor; only scalar tensors are supported."
            record[key] = float(value.item())
        elif isinstance(value, bool):
            record[key] = bool(value)
        elif isinstance(value, (np.integer, np.floating, np.bool_)):
            record[key] = float(value)
        elif isinstance(value, (int, float, str)) or value is None:
            record[key] = value
        elif isinstance(value, list):
            for index, element in enumerate(value):
                assert isinstance(element, (int, float, bool, str)), (
                    f"micro_step payload field '{key}'[{index}] has unsupported type {type(element).__name__}; "
                    "only primitives allowed in lists."
                )
            record[key] = list(value)
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                assert isinstance(sub_key, str), (
                    f"micro_step payload field '{key}' has non-string dict key {sub_key!r} "
                    f"(type {type(sub_key).__name__}); only string keys allowed."
                )
                assert isinstance(sub_value, (int, float, bool, str)) or sub_value is None, (
                    f"micro_step payload field '{key}'[{sub_key!r}] has unsupported type "
                    f"{type(sub_value).__name__}; only primitives allowed in dict values."
                )
            record[key] = dict(value)
        else:
            raise TypeError(f"micro_step payload field '{key}' has unsupported type {type(value).__name__}.")
    return record
