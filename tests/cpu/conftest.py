"""Hermetic policy for the mandatory CPU contract lane."""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import platform
import random
import signal
import time
import warnings
from collections.abc import Iterator
from importlib.metadata import version
from pathlib import Path
from typing import Any

from tests.cpu.resource_telemetry import (
    ConcurrentProcessTreeSampler,
    MemoryEvidenceError,
    aggregate_process_memory,
    capture_process_memory,
    select_concurrent_memory_gate,
)

if os.environ.get("FASTPLMS_CPU_BOOTSTRAPPED") != "1" or not getattr(
    builtins,
    "_fastplms_cpu_process_bootstrapped",
    False,
):
    raise RuntimeError("tests/cpu requires the Python-startup hermetic bootstrap")
_CPU_CACHE_ROOT = Path(os.environ["FASTPLMS_CPU_CACHE_ROOT"]).resolve()
_CACHE_NAMES = tuple(
    sorted(
        {
            "HF_HOME",
            "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "TRANSFORMERS_CACHE",
            "HF_DATASETS_CACHE",
            "TORCH_HOME",
            "TORCH_EXTENSIONS_DIR",
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
            "XDG_CACHE_HOME",
        }
    )
)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402


def _install_checkpoint_loader_guards() -> None:
    """Block native checkpoint loaders that can bypass Python's open wrappers."""

    if getattr(builtins, "_fastplms_cpu_checkpoint_loader_guard", False):
        return
    assert_path = getattr(builtins, "_fastplms_cpu_assert_portable_path", None)
    if not callable(assert_path):
        raise RuntimeError("CPU checkpoint path guard was not installed at Python startup")
    original_torch_load = torch.load

    def guarded_torch_load(file: object, *args: Any, **kwargs: Any) -> Any:
        assert_path(file)
        return original_torch_load(file, *args, **kwargs)

    torch.load = guarded_torch_load  # type: ignore[assignment]
    try:
        import safetensors
        import safetensors.torch
    except ImportError:
        pass
    else:
        original_safe_open = safetensors.safe_open
        original_load_file = safetensors.torch.load_file

        def guarded_safe_open(filename: object, *args: Any, **kwargs: Any) -> Any:
            assert_path(filename)
            return original_safe_open(filename, *args, **kwargs)

        def guarded_load_file(filename: object, *args: Any, **kwargs: Any) -> Any:
            assert_path(filename)
            return original_load_file(filename, *args, **kwargs)

        safetensors.safe_open = guarded_safe_open  # type: ignore[assignment]
        safetensors.torch.safe_open = guarded_safe_open  # type: ignore[assignment]
        safetensors.torch.load_file = guarded_load_file  # type: ignore[assignment]
    builtins.__dict__["_fastplms_cpu_checkpoint_loader_guard"] = True


_install_checkpoint_loader_guards()
torch.set_num_interop_threads(1)

_FORBIDDEN_MARKERS = {
    "artifact",
    "benchmark",
    "checkpoint",
    "compliance",
    "gpu",
    "large",
    "network",
    "packaging",
    "reference",
    "slow",
}
_MAX_TEST_SECONDS = 10.0
_MAX_SUITE_SECONDS = 300.0
_MAX_SUITE_RSS_BYTES = 4 * 1024**3
_MEMORY_SAMPLE_INTERVAL_SECONDS = 0.05
_WORKSPACE = Path(__file__).resolve().parents[2]
_TELEMETRY_PATH = Path(
    os.environ.get(
        "FASTPLMS_CPU_TELEMETRY_PATH",
        str(_WORKSPACE / "artifacts" / "telemetry" / "cpu-contract.json"),
    )
).resolve()


class CpuContractTimeoutError(TimeoutError):
    """Raised while a CPU contract still owns the per-test wall-clock budget."""


def _timeout_test(_signum: int, _frame: object) -> None:
    raise CpuContractTimeoutError(
        f"CPU contract exceeded its {_MAX_TEST_SECONDS:.0f}s execution budget."
    )


def _fail_session(session: pytest.Session, message: str) -> None:
    session.exitstatus = pytest.ExitCode.TESTS_FAILED
    warnings.warn(pytest.PytestWarning(message), stacklevel=2)


def _cache_snapshot() -> dict[str, object]:
    records: list[dict[str, object]] = []
    seen: set[Path] = set()
    for name in _CACHE_NAMES:
        root = Path(os.environ[name]).resolve()
        if root in seen:
            continue
        seen.add(root)
        files = sorted(path for path in root.rglob("*") if path.is_file())
        digest = hashlib.sha256()
        total_bytes = 0
        for path in files:
            size = path.stat().st_size
            total_bytes += size
            relative = path.relative_to(_CPU_CACHE_ROOT).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(size.to_bytes(8, "big"))
        records.append(
            {
                "root": root.relative_to(_CPU_CACHE_ROOT).as_posix(),
                "files": len(files),
                "bytes": total_bytes,
                "inventory_sha256": digest.hexdigest(),
            }
        )
    return {"roots": sorted(records, key=lambda record: str(record["root"]))}


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def pytest_sessionstart(session: pytest.Session) -> None:
    session.config._fastplms_cpu_started = time.monotonic()  # type: ignore[attr-defined]
    session.config._fastplms_cpu_durations = []  # type: ignore[attr-defined]
    session.config._fastplms_cpu_phase_durations = {}  # type: ignore[attr-defined]
    session.config._fastplms_cpu_outcomes = {}  # type: ignore[attr-defined]
    session.config._fastplms_cache_before = _cache_snapshot()  # type: ignore[attr-defined]
    if not hasattr(session.config, "workerinput"):
        session.config._fastplms_worker_memory = {}  # type: ignore[attr-defined]
        session.config._fastplms_worker_memory_errors = []  # type: ignore[attr-defined]
    if (
        not hasattr(session.config, "workerinput")
        and os.environ.get("FASTPLMS_CPU_CACHE_STARTED_EMPTY") != "1"
    ):
        raise pytest.UsageError(
            "CPU contract bootstrap did not attest an empty fresh cache root"
        )
    if not hasattr(session.config, "workerinput"):
        sampler = ConcurrentProcessTreeSampler(
            root_pids=(os.getpid(),),
            sample_interval_seconds=_MEMORY_SAMPLE_INTERVAL_SECONDS,
        )
        try:
            sampler.start()
        except MemoryEvidenceError as error:
            raise pytest.UsageError(
                f"CPU contract concurrent memory sampler is unavailable: {error}"
            ) from error
        session.config._fastplms_concurrent_memory_sampler = sampler  # type: ignore[attr-defined]


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    del exitstatus
    started = session.config._fastplms_cpu_started  # type: ignore[attr-defined]
    elapsed = time.monotonic() - started
    is_worker = hasattr(session.config, "workerinput")
    worker_output = getattr(session.config, "workeroutput", None)
    if isinstance(worker_output, dict):
        worker_id = str(session.config.workerinput.get("workerid", ""))  # type: ignore[attr-defined]
        try:
            worker_output["fastplms_memory_evidence"] = capture_process_memory(
                worker_id,
                role="worker",
            )
        except MemoryEvidenceError as error:
            worker_output["fastplms_memory_error"] = str(error)
            _fail_session(session, f"CPU contract memory evidence unavailable: {error}")
        worker_output["fastplms_test_durations"] = list(
            session.config._fastplms_cpu_durations  # type: ignore[attr-defined]
        )
    if not is_worker and elapsed > _MAX_SUITE_SECONDS:
        _fail_session(
            session,
            f"CPU contract budget exceeded: {elapsed:.2f}s > {_MAX_SUITE_SECONDS:.0f}s",
        )
    if not is_worker:
        worker_memory = getattr(session.config, "_fastplms_worker_memory", {})
        diagnostic_errors = list(
            getattr(session.config, "_fastplms_worker_memory_errors", [])
        )
        expected_workers = _expected_xdist_workers(session.config)
        if expected_workers != len(worker_memory):
            diagnostic_errors.append(
                "CPU contract expected "
                f"{expected_workers} xdist worker memory records, received {len(worker_memory)}"
            )
        try:
            controller_memory = capture_process_memory("controller", role="controller")
            temporal_upper_bound = aggregate_process_memory(
                controller_memory,
                worker_memory.values(),
            )
        except MemoryEvidenceError as error:
            diagnostic_errors.append(str(error))
            temporal_upper_bound = {
                "available": False,
                "errors": sorted(set(diagnostic_errors)),
            }
        if diagnostic_errors:
            temporal_upper_bound["available"] = False
            temporal_upper_bound["errors"] = sorted(set(diagnostic_errors))
            warnings.warn(
                pytest.PytestWarning(
                    "CPU contract temporal-upper-bound diagnostic is incomplete: "
                    + "; ".join(sorted(set(diagnostic_errors)))
                ),
                stacklevel=2,
            )

        concurrent_errors: list[str] = []
        try:
            sampler = session.config._fastplms_concurrent_memory_sampler  # type: ignore[attr-defined]
            concurrent_memory = sampler.stop()
            memory_gate = select_concurrent_memory_gate(concurrent_memory)
        except (AttributeError, MemoryEvidenceError) as error:
            concurrent_errors.append(str(error))
            concurrent_memory = {
                "available": False,
                "errors": sorted(set(concurrent_errors)),
            }
            memory_gate = {
                "available": False,
                "errors": sorted(set(concurrent_errors)),
            }
        memory_evidence = {
            "available": not concurrent_errors,
            "gate": memory_gate,
            "concurrent_process_tree": concurrent_memory,
            "temporal_upper_bound": temporal_upper_bound,
        }
        if concurrent_errors:
            _fail_session(
                session,
                "CPU contract concurrent memory evidence unavailable: "
                + "; ".join(concurrent_errors),
            )
        selected_peak = memory_gate.get("peak_bytes")
        if memory_gate.get("fallback_used") is True:
            warnings.warn(
                pytest.PytestWarning(
                    "CPU contract PSS evidence was incomplete; enforcing the 4 GiB "
                    "budget with conservative concurrent RSS instead: "
                    + "; ".join(memory_gate.get("fallback_reasons", []))
                ),
                stacklevel=2,
            )
        if isinstance(selected_peak, int) and selected_peak > _MAX_SUITE_RSS_BYTES:
            _fail_session(
                session,
                "CPU contract concurrent memory budget exceeded "
                f"({memory_gate.get('metric', 'unavailable')}): "
                f"{selected_peak / 1024**3:.2f} GiB > 4 GiB",
            )
        durations = list(
            getattr(session.config, "_fastplms_cpu_durations", [])
        ) + list(getattr(session.config, "_fastplms_worker_durations", []))
        durations.sort(key=lambda record: str(record["nodeid"]))
        cache_before = session.config._fastplms_cache_before  # type: ignore[attr-defined]
        _atomic_json(
            _TELEMETRY_PATH,
            {
                "schema_version": 3,
                "report": "fastplms-cpu-contract",
                "source_revision": os.environ.get("GITHUB_SHA", "unbound-local-source"),
                "runtime": {
                    "python": platform.python_version(),
                    "torch": version("torch"),
                    "transformers": version("transformers"),
                    "platform": platform.platform(),
                },
                "budgets": {
                    "suite_seconds": _MAX_SUITE_SECONDS,
                    "test_seconds": _MAX_TEST_SECONDS,
                    "concurrent_physical_memory_bytes": _MAX_SUITE_RSS_BYTES,
                },
                "observed": {
                    "suite_seconds": round(elapsed, 6),
                    "concurrent_peak_memory_bytes": selected_peak,
                    "concurrent_peak_memory_metric": memory_gate.get("metric"),
                    "peak_concurrent_rss_bytes": concurrent_memory.get(
                        "peak_concurrent_rss_bytes"
                    ),
                    "peak_concurrent_pss_bytes": concurrent_memory.get(
                        "peak_concurrent_pss_bytes"
                    ),
                    "temporal_upper_bound_rss_bytes": temporal_upper_bound.get(
                        "temporal_upper_bound_rss_bytes"
                    ),
                    "memory": memory_evidence,
                    "tests": durations,
                },
                "cache": {
                    "started_empty": True,
                    "before_collection": cache_before,
                    "after_session": _cache_snapshot(),
                },
            },
        )


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node: Any, error: object) -> None:
    worker_output = getattr(node, "workeroutput", {})
    worker_memory = getattr(node.config, "_fastplms_worker_memory", None)
    memory_errors = getattr(node.config, "_fastplms_worker_memory_errors", None)
    if worker_memory is None or memory_errors is None:
        return
    memory_record = worker_output.get("fastplms_memory_evidence")
    if isinstance(memory_record, dict):
        worker_id = str(memory_record.get("process_id", ""))
        if not worker_id or worker_id in worker_memory:
            memory_errors.append(f"duplicate or missing worker memory identity: {worker_id!r}")
        else:
            worker_memory[worker_id] = memory_record
    else:
        detail = worker_output.get("fastplms_memory_error") or error or "missing record"
        memory_errors.append(f"xdist worker memory evidence unavailable: {detail}")
    durations = getattr(node.config, "_fastplms_worker_durations", None)
    if durations is None:
        durations = []
        node.config._fastplms_worker_durations = durations
    durations.extend(worker_output.get("fastplms_test_durations", []))


def _expected_xdist_workers(config: pytest.Config) -> int:
    raw_value = config.getoption("numprocesses", default=0)
    if raw_value in (None, 0, "0"):
        return 0
    if raw_value == "auto":
        raw_value = os.environ.get("PYTEST_XDIST_AUTO_NUM_WORKERS", "")
    try:
        count = int(raw_value)
    except (TypeError, ValueError) as error:
        raise pytest.UsageError(
            f"Cannot determine configured xdist worker count from {raw_value!r}"
        ) from error
    if count < 0:
        raise pytest.UsageError(f"Invalid xdist worker count: {count}")
    return count


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Make directory ownership positive and reject expensive resource markers."""

    for item in items:
        item.add_marker(pytest.mark.cpu_contract)
        forbidden = sorted(
            mark.name for mark in item.iter_markers() if mark.name in _FORBIDDEN_MARKERS
        )
        if forbidden:
            raise pytest.UsageError(
                f"CPU contract {item.nodeid} carries forbidden markers: {forbidden}"
            )
        if item.get_closest_marker("skip") or item.get_closest_marker("skipif"):
            raise pytest.UsageError(f"CPU contract {item.nodeid} may not be skipped")
        if item.get_closest_marker("xfail"):
            raise pytest.UsageError(f"CPU contract {item.nodeid} may not be xfailed")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> Iterator[None]:
    """Bound the complete setup/call/teardown protocol for one contract."""

    del item, nextitem
    if (
        not hasattr(signal, "SIGALRM")
        or not hasattr(signal, "ITIMER_REAL")
        or not hasattr(signal, "setitimer")
    ):
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_test)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, _MAX_TEST_SECONDS)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0.0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[object]) -> Iterator[None]:
    """Turn dynamic import/runtime skips into hard failures."""

    outcome = yield
    report = outcome.get_result()
    if report.skipped:
        report.outcome = "failed"
        report.longrepr = "Mandatory CPU contracts may not skip at runtime."
    phase_durations = item.config._fastplms_cpu_phase_durations  # type: ignore[attr-defined]
    total_duration = float(phase_durations.get(report.nodeid, 0.0)) + report.duration
    phase_durations[report.nodeid] = total_duration
    outcomes = item.config._fastplms_cpu_outcomes  # type: ignore[attr-defined]
    if report.outcome != "passed":
        outcomes[report.nodeid] = report.outcome
    if total_duration > _MAX_TEST_SECONDS:
        report.outcome = "failed"
        outcomes[report.nodeid] = "failed"
        report.longrepr = (
            f"CPU contract exceeded its {_MAX_TEST_SECONDS:.0f}s budget: "
            f"{total_duration:.2f}s across setup/call/teardown"
        )
    if report.when == "teardown":
        item.config._fastplms_cpu_durations.append(  # type: ignore[attr-defined]
            {
                "nodeid": report.nodeid,
                "seconds": round(total_duration, 6),
                "outcome": outcomes.get(report.nodeid, "passed"),
            }
        )


@pytest.fixture(autouse=True)
def _hermetic_cpu(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Deny network access and keep tiny numerical tests deterministic."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    try:
        yield
    finally:
        torch.set_num_threads(previous_threads)
