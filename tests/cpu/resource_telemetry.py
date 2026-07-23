"""Linux process-tree memory accounting for the mandatory CPU gate."""

from __future__ import annotations

import errno
import os
import threading
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


class MemoryEvidenceError(RuntimeError):
    """Raised when process memory evidence is absent or internally inconsistent."""


_TRANSIENT_PROC_ERRNOS = frozenset({errno.ENOENT, errno.ESRCH})
_KIBIBYTE = 1024


def _validated_pid(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise MemoryEvidenceError(f"invalid process identifier: {value!r}")
    return value


def _read_proc_text(path: Path) -> str | None:
    """Read one procfs file, treating a vanished process as a sampling race."""

    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError):
        return None
    except PermissionError as error:
        raise MemoryEvidenceError(f"permission denied while reading {path}") from error
    except OSError as error:
        if error.errno in _TRANSIENT_PROC_ERRNOS:
            return None
        raise MemoryEvidenceError(f"cannot read procfs memory evidence from {path}") from error


def _direct_child_pids(proc_root: Path, process_id: int) -> set[int]:
    """Return children created by any thread in one Linux process."""

    task_root = proc_root / str(process_id) / "task"
    try:
        task_directories = tuple(task_root.iterdir())
    except (FileNotFoundError, ProcessLookupError):
        return set()
    except PermissionError as error:
        raise MemoryEvidenceError(f"permission denied while reading {task_root}") from error
    except OSError as error:
        if error.errno in _TRANSIENT_PROC_ERRNOS:
            return set()
        raise MemoryEvidenceError(f"cannot enumerate process tasks in {task_root}") from error

    children: set[int] = set()
    for task_directory in task_directories:
        if not task_directory.name.isdigit():
            continue
        child_text = _read_proc_text(task_directory / "children")
        if child_text is None:
            continue
        for raw_child in child_text.split():
            try:
                child_id = int(raw_child)
            except ValueError as error:
                raise MemoryEvidenceError(
                    f"invalid child process identifier {raw_child!r} in {task_directory}"
                ) from error
            children.add(_validated_pid(child_id))
    return children


def collect_process_tree_pids(
    proc_root: Path,
    root_pids: Iterable[int],
) -> tuple[int, ...]:
    """Collect overlapping Linux process roots and descendants exactly once."""

    roots = tuple(sorted({_validated_pid(process_id) for process_id in root_pids}))
    if not roots:
        raise MemoryEvidenceError("process-tree sampling requires at least one root PID")
    observed: set[int] = set()
    pending = list(reversed(roots))
    while pending:
        process_id = pending.pop()
        if process_id in observed:
            continue
        observed.add(process_id)
        pending.extend(
            child_id
            for child_id in sorted(
                _direct_child_pids(proc_root, process_id),
                reverse=True,
            )
            if child_id not in observed
        )
    return tuple(sorted(observed))


def _parse_kib_field(text: str, field: str, *, path: Path) -> int | None:
    for line in text.splitlines():
        name, separator, raw_value = line.partition(":")
        if not separator or name != field:
            continue
        parts = raw_value.split()
        if len(parts) != 2 or parts[1] != "kB":
            raise MemoryEvidenceError(f"invalid {field} value in {path}: {raw_value!r}")
        try:
            kibibytes = int(parts[0])
        except ValueError as error:
            raise MemoryEvidenceError(
                f"non-integer {field} value in {path}: {parts[0]!r}"
            ) from error
        if kibibytes < 0:
            raise MemoryEvidenceError(f"negative {field} value in {path}: {kibibytes}")
        return kibibytes * _KIBIBYTE
    return None


def _read_rss_bytes(proc_root: Path, process_id: int) -> int | None:
    process_root = proc_root / str(process_id)
    status_path = process_root / "status"
    status_text = _read_proc_text(status_path)
    if status_text is None:
        return None
    rss_bytes = _parse_kib_field(status_text, "VmRSS", path=status_path)
    if rss_bytes is not None:
        return rss_bytes

    # A zombie can omit VmRSS. statm still provides an exact zero-resident
    # record and avoids incorrectly treating a zero-footprint process as an
    # unavailable PSS sample.
    statm_path = process_root / "statm"
    statm_text = _read_proc_text(statm_path)
    if statm_text is None:
        return None
    fields = statm_text.split()
    if len(fields) < 2:
        raise MemoryEvidenceError(f"invalid statm record in {statm_path}")
    try:
        resident_pages = int(fields[1])
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, TypeError, ValueError) as error:
        raise MemoryEvidenceError(f"cannot decode resident pages in {statm_path}") from error
    if resident_pages < 0 or page_size <= 0:
        raise MemoryEvidenceError(f"invalid resident-page accounting in {statm_path}")
    return resident_pages * page_size


def _read_pss_bytes(proc_root: Path, process_id: int) -> tuple[int | None, str | None]:
    path = proc_root / str(process_id) / "smaps_rollup"
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError):
        return None, "smaps_rollup disappeared during sampling"
    except PermissionError:
        return None, "smaps_rollup permission denied"
    except OSError as error:
        if error.errno in _TRANSIENT_PROC_ERRNOS:
            return None, "smaps_rollup disappeared during sampling"
        return None, f"smaps_rollup read failed with errno {error.errno}"
    pss_bytes = _parse_kib_field(text, "Pss", path=path)
    if pss_bytes is None:
        return None, "smaps_rollup contained no Pss field"
    return pss_bytes, None


def sample_process_tree_memory(
    *,
    root_pids: Iterable[int],
    proc_root: Path = Path("/proc"),
) -> dict[str, object]:
    """Measure one concurrent, PID-deduplicated Linux process-tree snapshot."""

    proc_root = proc_root.resolve()
    roots = tuple(sorted({_validated_pid(process_id) for process_id in root_pids}))
    process_ids = collect_process_tree_pids(proc_root, roots)
    processes: list[dict[str, object]] = []
    fallback_reasons: set[str] = set()
    for process_id in process_ids:
        rss_bytes = _read_rss_bytes(proc_root, process_id)
        if rss_bytes is None:
            continue
        pss_bytes, pss_error = _read_pss_bytes(proc_root, process_id)
        process_root = proc_root / str(process_id)
        if pss_bytes is None and not process_root.exists():
            # Do not combine RSS from a process that exited before its PSS
            # could be observed. A subsequent sample accounts its replacement.
            continue
        if pss_bytes is None and rss_bytes == 0:
            pss_bytes = 0
            pss_error = None
        if pss_error is not None:
            fallback_reasons.add(f"pid {process_id}: {pss_error}")
        processes.append(
            {
                "pid": process_id,
                "rss_bytes": rss_bytes,
                "pss_bytes": pss_bytes,
            }
        )

    observed_pids = [int(process["pid"]) for process in processes]
    if not observed_pids or not set(roots).intersection(observed_pids):
        raise MemoryEvidenceError(
            "procfs sampling did not capture any requested process-tree root"
        )
    aggregate_rss = sum(int(process["rss_bytes"]) for process in processes)
    pss_complete = all(isinstance(process["pss_bytes"], int) for process in processes)
    aggregate_pss = (
        sum(int(process["pss_bytes"]) for process in processes)
        if pss_complete
        else None
    )
    return {
        "root_pids": list(roots),
        "process_ids": observed_pids,
        "process_count": len(observed_pids),
        "pid_accounting": "unique-live-process-id",
        "aggregate_rss_bytes": aggregate_rss,
        "aggregate_pss_bytes": aggregate_pss,
        "pss_complete": pss_complete,
        "pss_fallback_reasons": sorted(fallback_reasons),
        "processes": processes,
    }


class ConcurrentProcessTreeSampler:
    """Sample a Linux controller process tree at a fixed high frequency."""

    def __init__(
        self,
        *,
        root_pids: Iterable[int] | None = None,
        sample_interval_seconds: float = 0.05,
        proc_root: Path = Path("/proc"),
    ) -> None:
        roots = (os.getpid(),) if root_pids is None else tuple(root_pids)
        self.root_pids = tuple(sorted({_validated_pid(process_id) for process_id in roots}))
        if not self.root_pids:
            raise MemoryEvidenceError("concurrent memory sampling requires a root PID")
        if sample_interval_seconds <= 0:
            raise MemoryEvidenceError("memory sample interval must be positive")
        self.sample_interval_seconds = float(sample_interval_seconds)
        self.proc_root = proc_root.resolve()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._stopped = False
        self._errors: list[str] = []
        self._sample_count = 0
        self._pss_complete_sample_count = 0
        self._pss_complete_for_all_samples = True
        self._pss_fallback_reasons: set[str] = set()
        self._observed_process_ids: set[int] = set()
        self._max_process_count = 0
        self._peak_concurrent_rss_bytes = 0
        self._peak_concurrent_pss_bytes = 0
        self._peak_rss_snapshot: dict[str, object] | None = None
        self._peak_pss_snapshot: dict[str, object] | None = None
        self._last_sample_started: float | None = None
        self._max_sample_gap_seconds = 0.0

    def _record_snapshot(
        self,
        snapshot: dict[str, object],
        *,
        sample_started: float,
    ) -> None:
        with self._lock:
            if self._last_sample_started is not None:
                self._max_sample_gap_seconds = max(
                    self._max_sample_gap_seconds,
                    sample_started - self._last_sample_started,
                )
            self._last_sample_started = sample_started
            self._sample_count += 1
            process_ids = {int(process_id) for process_id in snapshot["process_ids"]}
            self._observed_process_ids.update(process_ids)
            self._max_process_count = max(self._max_process_count, len(process_ids))
            aggregate_rss = int(snapshot["aggregate_rss_bytes"])
            if aggregate_rss > self._peak_concurrent_rss_bytes:
                self._peak_concurrent_rss_bytes = aggregate_rss
                self._peak_rss_snapshot = snapshot
            if snapshot["pss_complete"] is True:
                self._pss_complete_sample_count += 1
                aggregate_pss = int(snapshot["aggregate_pss_bytes"])
                if aggregate_pss > self._peak_concurrent_pss_bytes:
                    self._peak_concurrent_pss_bytes = aggregate_pss
                    self._peak_pss_snapshot = snapshot
            else:
                self._pss_complete_for_all_samples = False
                self._pss_fallback_reasons.update(snapshot["pss_fallback_reasons"])

    def sample_now(self) -> dict[str, object]:
        if not self._started or self._stopped:
            raise MemoryEvidenceError("concurrent memory sampler is not running")
        sample_started = time.monotonic()
        snapshot = sample_process_tree_memory(
            root_pids=self.root_pids,
            proc_root=self.proc_root,
        )
        self._record_snapshot(snapshot, sample_started=sample_started)
        return snapshot

    def _run(self) -> None:
        while not self._stop_event.wait(self.sample_interval_seconds):
            try:
                self.sample_now()
            except Exception as error:  # pragma: no cover - exercised through stop()
                with self._lock:
                    self._errors.append(f"{type(error).__name__}: {error}")
                self._stop_event.set()
                return

    def start(self) -> None:
        if self._started:
            raise MemoryEvidenceError("concurrent memory sampler was already started")
        self._started = True
        try:
            self.sample_now()
        except Exception:
            self._started = False
            raise
        self._thread = threading.Thread(
            target=self._run,
            name="fastplms-cpu-memory-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> dict[str, object]:
        if not self._started:
            raise MemoryEvidenceError("concurrent memory sampler was never started")
        if self._stopped:
            raise MemoryEvidenceError("concurrent memory sampler was already stopped")
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 10 * self.sample_interval_seconds))
            if self._thread.is_alive():
                with self._lock:
                    self._errors.append("sampler thread did not stop")
        if not self._errors:
            try:
                self.sample_now()
            except Exception as error:
                with self._lock:
                    self._errors.append(f"{type(error).__name__}: {error}")
        self._stopped = True
        with self._lock:
            if self._errors:
                raise MemoryEvidenceError(
                    "concurrent process-tree sampling failed: " + "; ".join(self._errors)
                )
            if self._sample_count <= 0 or self._peak_rss_snapshot is None:
                raise MemoryEvidenceError("concurrent process-tree sampler produced no evidence")
            return {
                "available": True,
                "measurement": "linux-procfs-concurrent-process-tree",
                "root_pids": list(self.root_pids),
                "sample_interval_seconds": self.sample_interval_seconds,
                "sample_count": self._sample_count,
                "max_sample_gap_seconds": round(self._max_sample_gap_seconds, 6),
                "observed_process_ids": sorted(self._observed_process_ids),
                "max_concurrent_process_count": self._max_process_count,
                "pid_accounting": "each live descendant PID is counted once per sample",
                "peak_concurrent_rss_bytes": self._peak_concurrent_rss_bytes,
                "peak_concurrent_pss_bytes": (
                    self._peak_concurrent_pss_bytes
                    if self._pss_complete_sample_count
                    else None
                ),
                "pss_complete_sample_count": self._pss_complete_sample_count,
                "pss_complete_for_all_samples": self._pss_complete_for_all_samples,
                "pss_fallback_reasons": sorted(self._pss_fallback_reasons),
                "peak_rss_snapshot": self._peak_rss_snapshot,
                "peak_pss_snapshot": self._peak_pss_snapshot,
            }


def select_concurrent_memory_gate(evidence: Mapping[str, Any]) -> dict[str, object]:
    """Prefer complete concurrent PSS evidence, or conservatively gate on RSS."""

    sample_count = evidence.get("sample_count")
    peak_rss = evidence.get("peak_concurrent_rss_bytes")
    if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count <= 0:
        raise MemoryEvidenceError("concurrent memory evidence has no samples")
    if not isinstance(peak_rss, int) or isinstance(peak_rss, bool) or peak_rss <= 0:
        raise MemoryEvidenceError("concurrent memory evidence has no positive RSS peak")
    if evidence.get("pss_complete_for_all_samples") is True:
        peak_pss = evidence.get("peak_concurrent_pss_bytes")
        if not isinstance(peak_pss, int) or isinstance(peak_pss, bool) or peak_pss <= 0:
            raise MemoryEvidenceError(
                "PSS evidence is marked complete but has no positive concurrent peak"
            )
        return {
            "metric": "proportional-set-size",
            "source": "/proc/<pid>/smaps_rollup:Pss",
            "peak_bytes": peak_pss,
            "fallback_used": False,
            "fallback_is_conservative": False,
            "fallback_reasons": [],
        }

    reasons = evidence.get("pss_fallback_reasons")
    normalized_reasons = (
        [str(reason) for reason in reasons]
        if isinstance(reasons, list) and reasons
        else ["one or more concurrent samples lacked complete PSS evidence"]
    )
    return {
        "metric": "resident-set-size",
        "source": "/proc/<pid>/status:VmRSS",
        "peak_bytes": peak_rss,
        "fallback_used": True,
        "fallback_is_conservative": True,
        "fallback_reasons": normalized_reasons,
    }


def _peak_rss_bytes(scope: int) -> int:
    """Return Linux ``getrusage`` peak RSS in bytes for one resource scope."""

    try:
        import resource

        peak = int(resource.getrusage(scope).ru_maxrss)
    except (ImportError, OSError, ValueError) as error:
        raise MemoryEvidenceError("getrusage peak RSS is unavailable") from error
    if peak < 0:
        raise MemoryEvidenceError(f"getrusage returned a negative peak RSS: {peak}")
    return peak * 1024 if os.name != "nt" else peak


def capture_process_memory(process_id: str, *, role: str) -> dict[str, object]:
    """Capture one process and the largest child it has already waited for."""

    try:
        import resource
    except ImportError as error:
        raise MemoryEvidenceError("getrusage is required by the CPU contract") from error
    if not process_id:
        raise MemoryEvidenceError("process memory evidence requires a process identifier")
    if role not in {"controller", "worker"}:
        raise MemoryEvidenceError(f"unsupported process memory role: {role!r}")
    process_peak = _peak_rss_bytes(resource.RUSAGE_SELF)
    children_peak = _peak_rss_bytes(resource.RUSAGE_CHILDREN)
    return {
        "process_id": process_id,
        "pid": os.getpid(),
        "role": role,
        "process_peak_rss_bytes": process_peak,
        "waited_children_peak_rss_bytes": children_peak,
    }


def _validated_record(record: Mapping[str, Any], *, role: str) -> dict[str, object]:
    process_id = record.get("process_id")
    if not isinstance(process_id, str) or not process_id:
        raise MemoryEvidenceError("process memory evidence has no process_id")
    if record.get("role") != role:
        raise MemoryEvidenceError(
            f"process {process_id!r} reported role {record.get('role')!r}, expected {role!r}"
        )
    process_pid = _validated_pid(record.get("pid"))
    values: dict[str, int] = {}
    for name in ("process_peak_rss_bytes", "waited_children_peak_rss_bytes"):
        value = record.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise MemoryEvidenceError(
                f"process {process_id!r} has invalid {name}: {value!r}"
            )
        if name == "process_peak_rss_bytes" and value == 0:
            raise MemoryEvidenceError(
                f"process {process_id!r} reported no process peak RSS"
            )
        values[name] = value
    return {
        "process_id": process_id,
        "pid": process_pid,
        "role": role,
        **values,
    }


def aggregate_process_memory(
    controller: Mapping[str, Any],
    workers: Iterable[Mapping[str, Any]],
) -> dict[str, object]:
    """Build a temporal upper bound from nonconcurrent getrusage maxima.

    Under xdist, the controller's ``RUSAGE_CHILDREN`` contains the same workers
    that report their own usage, so it is evidence only and is not added. Each
    worker's waited-child peak is added because isolated AutoClass probes are
    separate processes and are absent from that worker's ``RUSAGE_SELF``.
    In serial mode, the controller's waited children are added directly.
    """

    controller_record = _validated_record(controller, role="controller")
    worker_records = [_validated_record(record, role="worker") for record in workers]
    worker_records.sort(key=lambda record: str(record["process_id"]))
    worker_ids = [str(record["process_id"]) for record in worker_records]
    if len(worker_ids) != len(set(worker_ids)):
        raise MemoryEvidenceError("duplicate xdist worker memory evidence")
    worker_pids = [int(record["pid"]) for record in worker_records]
    if len(worker_pids) != len(set(worker_pids)):
        raise MemoryEvidenceError("duplicate xdist worker operating-system PIDs")

    controller_self = int(controller_record["process_peak_rss_bytes"])
    if worker_records:
        accounted_controller_children = 0
        accounted_workers = sum(
            int(record["process_peak_rss_bytes"])
            + int(record["waited_children_peak_rss_bytes"])
            for record in worker_records
        )
        accounting_mode = "xdist-conservative-process-tree-upper-bound"
    else:
        accounted_controller_children = int(
            controller_record["waited_children_peak_rss_bytes"]
        )
        accounted_workers = 0
        accounting_mode = "serial-conservative-process-tree-upper-bound"
    aggregate = controller_self + accounted_controller_children + accounted_workers
    return {
        "available": True,
        "measurement": "getrusage.ru_maxrss",
        "units": "bytes",
        "accounting_mode": accounting_mode,
        "budget_enforced": False,
        "interpretation": (
            "Diagnostic temporal upper bound only: component maxima can occur at "
            "different times and must not enforce the concurrent memory budget."
        ),
        "double_counting_policy": (
            "Exclude controller RUSAGE_CHILDREN when workers report themselves; "
            "include each worker RUSAGE_SELF and RUSAGE_CHILDREN exactly once."
        ),
        "aggregate_peak_rss_bytes": aggregate,
        "temporal_upper_bound_rss_bytes": aggregate,
        "controller": controller_record,
        "workers": worker_records,
    }
