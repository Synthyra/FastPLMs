"""One W&B run inside an experiment, mirrored to durable local records.

An experiment is the recorded intent and a run is the W&B object inside it, so a run's group is
its experiment id, `YYYY-MM-DD_slug`, which `ws capture` reads the group back by. Promoted from
contact_esmc6b's `DurableExperiment`, with base_model_distillation's step metric and planned
stops. Every run:

- sends W&B its config with credential-shaped keys removed and the source revision added, and
  keeps W&B's own git capture off, which would otherwise record remote URLs;
- resumes the same W&B run from `run_file` after a restart, and refuses to guess when local
  history exists without it;
- appends every logged row to `history.jsonl`, flushed and fsynced, so a lost connection or a
  killed container keeps what was logged;
- writes `run_summary.json` when it finishes: identity, status, revision, config hash, and the
  last finite value of every scalar metric, which it also mirrors into the W&B summary.

It also holds W&B to docs/conventions/wandb.md, so a run stays a handful of readable charts:

- only real scalars reach W&B, under keys of at most three `/` segments, and at most `max_keys`
  of them per run; a nested mapping or an id-shaped key segment is refused, because each key is a
  chart. Per-entity values go to `table`, which is one panel however many rows it has;
- tuning, evaluation, diagnostic, packaging and release jobs stay local unless the caller passes
  `mode` explicitly: their records are `history.jsonl` and `run_summary.json`, not a W&B run;
- W&B's system charts are off unless `system_metrics=True`;
- W&B's own files land in `directory` itself when it is named `wandb`, else in `directory/wandb/`,
  never in a nested `wandb/wandb/` (`wandb_root`).

And it does the bookkeeping a training job otherwise repeats: it loads the secrets file when an
online run finds no W&B key, `batches` pilots the training loop (`foundry.pilot`), and `finish`
adds what the job cost, as summary-only `cost/` keys outside the budget.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import re
import sys
import time
import uuid

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, TypedDict, TypeVar

from .credentials import redact
from .organization import experiment_group, organization_tags, role_for_job
from .revision import Revision, revision_of


if TYPE_CHECKING:
    from foundry.pilot import Pilot


HISTORY_FILE = "history.jsonl"
SUMMARY_FILE = "run_summary.json"
TABLE_DIRECTORY = "tables"
PILOT_FILE = "pilot.json"
# Set to 1 to make a job a pilot: its training loop stops after the timed window and it makes no W&B run.
PILOT_VARIABLE = "WS_PILOT"
# Not `revision`, which configs already use for the model or dataset revision they load.
REVISION_KEY = "source_revision"
# Stops that end a run as planned: a residue target, a wall-clock or budget cap, a deadline.
PLANNED_STATUSES = frozenset({"completed", "time_stop", "budget_stop", "deadline"})
STATUSES = PLANNED_STATUSES | {"failed"}
# Each key is a chart. The default fits train, validation and test curves of one model; a run
# that needs more says why in its experiment README and passes `max_keys`, never past the ceiling.
MAX_KEYS = 40
KEY_CEILING = 100
# `section/metric` or `section/part/metric`, as W&B groups panels by the first segment. A dot only
# joins digits (`coverage@0.95`), since a dotted path such as `methods.plain.test` is a nested report.
SEGMENT = r"(?:[A-Za-z0-9_@\-]|(?<=[0-9])\.(?=[0-9]))+"
KEY = re.compile(rf"{SEGMENT}(?:/{SEGMENT}){{0,2}}")
# A hash or uuid inside a key means one chart per entity: a cluster, a sequence, a checkpoint.
ENTITY_SEGMENT = re.compile(r"[0-9a-f]{12,}")
# Roles whose numbers belong in local records and results, not in a W&B run of their own.
LOCAL_ROLES = frozenset({"tuning", "evaluation", "diagnostic", "package", "release"})
# Hugging Face Trainer log names that are throughput or bookkeeping, not a result.
TRAINER_NOISE = re.compile(r"(?:.*_runtime|.*_per_second|total_flos|epoch|train_loss|.*_model_preparation_time)")
# What a job cost, written by the run itself into the summary only, so they draw no chart and sit outside the budget.
COST_KEYS = frozenset({"cost/seconds", "cost/peak_gpu_gib", "cost/step_seconds", "cost/input_wait_share", "cost/projected_hours"})
GIB = 2**30
T = TypeVar("T")


class RunIdentity(TypedDict):
    """Which W&B run a directory holds: the whole of `run.json`, and the start of `run_summary.json`."""

    id: str
    url: str | None
    project: str
    entity: str | None
    group: str | None
    name: str
    job_type: str | None
    mode: str


def wandb_root(directory: Path) -> Path:
    """The folder to hand `wandb.init(dir=...)`, so that W&B's own files land in `directory/wandb/` once.

    W&B always stages a run in a `wandb/` folder under the folder it is given. A run directory named
    `wandb`, as `track` and most callers name it, is already that folder, and handing it over as is
    nested a second `wandb/` that pushed a 2026-10-05 run past Windows' 260-character path limit.
    """
    return directory.parent if directory.name == "wandb" else directory


class Run:
    """A resumable W&B run whose logged rows also land, durably, in its directory.

    `run` is the W&B run itself, for artifacts and anything else this class does not wrap.
    With `step_metric`, rows are plotted against that metric instead of W&B's own step, so a
    retried container that restarts from an earlier step is not dropped. `revision` defaults to
    the code of the file that opens the run (`revision_of`). Used as a context manager, the run
    finishes `completed`, or `failed` when the block raises, unless it was finished inside.
    `pilot` defaults to the `WS_PILOT` environment variable; a pilot makes no W&B run and keeps its
    records in `pilots/<time>/` under `directory`, so the full run after it starts clean.
    """

    def __init__(
        self,
        directory: Path,
        config: Mapping[str, Any],
        *,
        project: str,
        revision: Revision | None = None,
        group: str | None = None,
        job_type: str | None = None,
        entity: str | None = None,
        name: str | None = None,
        tags: Sequence[str] = (),
        mode: str | None = None,
        step_metric: str | None = None,
        run_file: str = "run.json",
        reinit: str | None = None,
        max_keys: int = MAX_KEYS,
        system_metrics: bool = False,
        pilot: bool | None = None,
    ) -> None:
        clock_started = time.perf_counter()
        import wandb

        group = group if group is not None else experiment_group(directory, config)
        if group is not None and (not group or "/" in group):
            raise ValueError(f"W&B group {group!r} must be an experiment id without '/', which `ws capture` cannot parse.")
        if not 0 < max_keys <= KEY_CEILING:
            raise ValueError(f"max_keys must be in 1..{KEY_CEILING}; received {max_keys}. Put the rest in a table.")
        self.piloting = pilot if pilot is not None else os.environ.get(PILOT_VARIABLE, "").lower() in {"1", "true", "yes"}
        name = name or directory.name
        if self.piloting:
            directory = directory / "pilots" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        if mode is None and (self.piloting or role_for_job(job_type) in LOCAL_ROLES):
            kind = "pilot" if self.piloting else f"{job_type} job"
            print(f"W&B: a {kind} keeps its record in {directory}, not a W&B run (docs/conventions/wandb.md).", file=sys.stderr)
            mode = "disabled"
        mode = mode or os.environ.get("WANDB_MODE") or "online"
        if mode == "online" and not os.environ.get("WANDB_API_KEY"):
            from foundry.secrets_env import load_secrets_env

            load_secrets_env()
        if mode == "online" and not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError("WANDB_API_KEY is required for an online run, and no secrets file set it (foundry.secrets_env).")
        revision = revision if revision is not None else revision_of(Path(sys._getframe(1).f_code.co_filename))

        directory.mkdir(parents=True, exist_ok=True)
        self.directory, self.revision, self.step_metric, self.max_keys = directory, revision, step_metric, max_keys
        self.run_path, self.history_path = directory / run_file, directory / HISTORY_FILE
        if self.history_path.exists() and not self.run_path.exists():
            raise ValueError(f"{self.history_path} exists without {run_file}, so the run it belongs to is unknown.")
        resuming = self.run_path.exists()
        earlier = json.loads(self.run_path.read_text(encoding="utf-8")) if resuming else {}
        if earlier.get("mode") == "disabled" and mode != "disabled":
            raise ValueError(f"{directory} holds a local-only record, which W&B cannot resume {mode}; use a new directory.")
        run_id = earlier["id"] if resuming else uuid.uuid4().hex[:8]
        # The budget counts what an earlier attempt of this run already charted, and the cost its time.
        self.keys: set[str] = _history_keys(self.history_path) - {step_metric} if resuming else set()
        self.clock_started, self.earlier_seconds = clock_started, _earlier_seconds(directory / SUMMARY_FILE) if resuming else 0.0
        self.pilot: Pilot | None = None
        self.costs: dict[str, float] = {}

        self.config = redact(config)
        if REVISION_KEY in self.config:
            raise ValueError(f"config already has a {REVISION_KEY!r} key, which the run records itself.")
        self.config_sha256 = hashlib.sha256(_canonical(self.config).encode("utf-8")).hexdigest()
        self.run = wandb.init(
            entity=entity,
            project=project,
            id=run_id,
            resume="must" if resuming else "allow",
            group=group,
            job_type=job_type,
            name=name,
            tags=organization_tags(tags, job_type),
            config={**self.config, REVISION_KEY: revision.to_dict()},
            dir=str(wandb_root(directory)),
            mode=mode,
            settings=wandb.Settings(init_timeout=90, disable_git=True, x_disable_stats=not system_metrics),
            **({"reinit": reinit} if reinit is not None else {}),
        )
        if self.run is None or (mode != "disabled" and self.run.id != run_id):
            raise RuntimeError(f"W&B did not open run {run_id}.")
        if step_metric is not None:
            self.run.define_metric(step_metric)
            self.run.define_metric("*", step_metric=step_metric)

        self.identity: RunIdentity = {
            "id": run_id, "url": self.run.url, "project": project, "entity": entity,
            "group": group, "name": name, "job_type": job_type, "mode": mode,
        }
        _atomic_json(self.run_path, self.identity)
        self.started = _now()
        self.summary_path: Path | None = None
        self.latest: dict[str, float] = {}
        self.stream = self.history_path.open("a", encoding="utf-8", newline="\n")

    def __enter__(self) -> Run:
        return self

    def __exit__(self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None) -> None:
        if self.summary_path is None:
            self.finish(status="completed" if kind is None else "failed")

    def log(self, metrics: Mapping[str, Any], step: int) -> None:
        """Append `{"step": step, **metrics}` to the history, then send its real scalars to W&B."""
        charted = self._charted(metrics)
        self.stream.write(json.dumps({"step": step, **metrics}, sort_keys=True, default=_json_number) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.latest.update({name: float(value) for name, value in charted.items() if math.isfinite(value)})
        if self.step_metric is None:
            self.run.log(charted, step=step)
        else:
            self.run.log({**charted, self.step_metric: step})

    def table(self, name: str, rows: Sequence[Mapping[str, Any]]) -> Path:
        """Log rows as one W&B table, one panel and one key of the budget, and keep them locally.

        Anything keyed by an entity, such as per-task, per-class, per-cluster or per-method
        scores, belongs here rather than in metric names.
        """
        import wandb

        self._admit({name})
        columns = list(dict.fromkeys(column for row in rows for column in row))
        path = self.directory / TABLE_DIRECTORY / f"{name.replace('/', '__')}.json"
        path.parent.mkdir(exist_ok=True)
        _atomic_json(path, {"name": name, "columns": columns, "rows": [dict(row) for row in rows]})
        self.run.log({name: wandb.Table(columns=columns, data=[[row.get(column) for column in columns] for row in rows])})
        return path

    def batches(self, batches: Iterable[T], *, total_steps: int | None = None, items: int | Callable[[Any], int] | None = None,
                item_unit: str = "items") -> Iterator[T]:
        """Yield `batches` under a `foundry.pilot.Pilot` timed from when the run opened.

        A pilot stops after the timed window; a full run goes on unmeasured, so its first minutes can
        be compared with its pilot. Either way the report goes to `pilot.json`, and its step time,
        input wait and projected hours to the summary. Only the first loop is watched, so a later
        epoch passes straight through. `total_steps` defaults to `len(batches)`.
        """
        if self.pilot is not None:
            yield from batches
            return
        from foundry.pilot import Pilot

        if total_steps is None and hasattr(batches, "__len__"):
            total_steps = len(batches)
        self.pilot = Pilot(total_steps=total_steps, items=items, item_unit=item_unit, stop=self.piloting,
                           receipt=self.directory / PILOT_FILE, started=self.clock_started)
        for batch in self.pilot.watch(batches):
            yield batch
            self._record_pilot()

    def finish(self, *, status: str = "completed", summary: Mapping[str, Any] | None = None) -> Path:
        """Close the run and write `run_summary.json`. A planned stop exits 0, `failed` exits 1."""
        if status not in STATUSES:
            raise ValueError(f"status must be one of {sorted(STATUSES)}; received {status!r}.")
        if self.summary_path is not None:
            raise RuntimeError(f"{self.identity['name']} already finished; finish a run once.")
        self._record_pilot()
        costs = {"cost/seconds": self.earlier_seconds + time.perf_counter() - self.clock_started, **_peak_gpu(), **self.costs}
        given = {name: value for name, value in self._charted(summary or {}).items() if math.isfinite(value)}
        scalars = {**self.latest, **costs, **given}
        exit_code = 0 if status in PLANNED_STATUSES else 1
        record = {
            **self.identity, "status": status, "exit_code": exit_code, "started": self.started, "finished": _now(),
            REVISION_KEY: self.revision.to_dict(), "config_sha256": self.config_sha256, "summary": scalars,
        }
        # Local records first: they must survive a W&B call that fails.
        self.stream.close()
        self.summary_path = self.directory / SUMMARY_FILE
        _atomic_json(self.summary_path, record)
        self.run.summary.update(scalars)
        self.run.finish(exit_code=exit_code)
        return self.summary_path

    def trainer_callback(self) -> TrainerMirror:
        """A Hugging Face Trainer callback that logs through this run; pair it with `report_to="none"`."""
        return TrainerMirror(self)

    def _record_pilot(self) -> None:
        """Copy the pilot's numbers into the costs once its window has closed."""
        if self.costs or self.pilot is None or not self.pilot.closed:
            return
        report = self.pilot.report
        self.costs = {"cost/step_seconds": report["step_seconds_median"], "cost/input_wait_share": report["input_wait_share"]}
        if report["projected_seconds"] is not None:
            self.costs["cost/projected_hours"] = report["projected_seconds"] / 3600

    def _charted(self, metrics: Mapping[str, Any]) -> dict[str, float]:
        """The metrics W&B charts, after checking their form and the run's key budget."""
        nested = sorted(name for name, value in metrics.items() if isinstance(value, Mapping))
        if nested:
            raise TypeError(f"{nested} hold mappings; log scalars, and put per-entity values in Run.table (docs/conventions/wandb.md).")
        charted = {name: float(value) for name, value in metrics.items() if _is_real(value)}
        self._admit(charted.keys())
        return charted

    def _admit(self, names: Iterable[str]) -> None:
        new = set(names) - self.keys - COST_KEYS
        malformed = sorted(name for name in new if not KEY.fullmatch(name) or ENTITY_SEGMENT.search(name))
        if malformed:
            raise ValueError(
                f"W&B keys {malformed[:5]} are not `section/metric` with at most three segments and no ids; "
                "put per-entity values in Run.table (docs/conventions/wandb.md)."
            )
        if len(self.keys) + len(new) > self.max_keys:
            raise ValueError(
                f"{self.identity['name']} would chart {len(self.keys) + len(new)} keys, over its budget of {self.max_keys}; "
                f"new: {sorted(new)[:8]}. Log headline metrics, keep config out of metrics, and put per-entity values in "
                "Run.table (docs/conventions/wandb.md)."
            )
        self.keys |= new


class TrainerMirror:
    """A Hugging Face `TrainerCallback` by duck type, so `foundry.logging` needs no transformers.

    The Trainer calls every `on_*` event on each callback; this one acts on `on_log` and ignores
    the rest. `eval_*` becomes `val/*`, `test_*` becomes `test/*`, everything else `train/*`,
    and throughput, epoch and FLOP counters are dropped.
    """

    def __init__(self, run: Run) -> None:
        self.run = run

    def on_log(self, args: Any, state: Any, control: Any, logs: Mapping[str, Any] | None = None, **_: Any) -> None:
        metrics = trainer_metrics(logs or {})
        if state.is_world_process_zero and metrics:
            self.run.log(metrics, step=state.global_step)

    def __getattr__(self, event: str) -> Callable[..., None]:
        if event.startswith("on_"):
            return _ignore_event
        raise AttributeError(event)


def _ignore_event(*_: Any, **__: Any) -> None:
    return None


def trainer_metrics(logs: Mapping[str, Any]) -> dict[str, Any]:
    """A Hugging Face Trainer log row under the workspace's `train/`, `val/` and `test/` sections."""
    metrics: dict[str, Any] = {}
    for name, value in logs.items():
        if TRAINER_NOISE.fullmatch(name):
            continue
        if name.startswith("eval_"):
            metrics[f"val/{name.removeprefix('eval_')}"] = value
        elif name.startswith("test_"):
            metrics[f"test/{name.removeprefix('test_')}"] = value
        else:
            metrics[f"train/{name.removeprefix('train_')}"] = value
    return metrics


def _peak_gpu() -> dict[str, float]:
    """`cost/peak_gpu_gib` when the job has used CUDA through torch, which this module never imports itself."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return {}
    return {"cost/peak_gpu_gib": torch.cuda.max_memory_allocated() / GIB}


def _earlier_seconds(path: Path) -> float:
    """What earlier attempts of a resumed run already spent, from the summary the last one wrote."""
    if not path.exists():
        return 0.0
    return float(json.loads(path.read_text(encoding="utf-8"))["summary"].get("cost/seconds", 0.0))


def _history_keys(path: Path) -> set[str]:
    """The keys an earlier attempt sent W&B, read back from its history."""
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return {name for row in rows for name, value in row.items() if name != "step" and _is_real(value)}


def _is_real(value: object) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _json_number(value: object) -> int | float:
    """numpy scalars, which json cannot serialize, as Python numbers; anything else still fails."""
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _canonical(values: Mapping[str, Any]) -> str:
    return json.dumps(values, sort_keys=True, separators=(",", ":"), default=str)


def _atomic_json(path: Path, record: Mapping[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
