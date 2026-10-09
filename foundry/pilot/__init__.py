"""Measure a job's first steps, so its whole run is projected and its waste found before it is paid for.

Iterate `Pilot.watch(batches)` in place of `batches` in a training loop, an embedding pass, or a data pipeline. After a
few warmup steps the pilot times each step, splitting the time the loop waits for its next batch from the time it
computes, and records the device's peak memory, the GPU's kernel and power utilization from `nvidia-smi`, the host's
resident memory, busy cores and storage traffic, and how long the job took to reach its first step. It projects the
whole run from those steps, prints what it measured with what looks wasteful, and can save the numbers as a receipt.
`docs/conventions/compute.md` is the procedure that asks for it.
"""

from __future__ import annotations

import itertools
import math
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
import psutil

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, TypedDict, TypeVar

from foundry.serialization import write_json_dict_atomic


try:
    import torch
except ImportError:  # A data pipeline without torch still gets its timing, host and storage numbers.
    torch = None


__all__ = ["DeviceSample", "Pilot", "PilotReport", "assess", "describe", "parse_nvidia_smi"]

T = TypeVar("T")

# What a pilot is judged against. Each threshold names a waste, and the finding it raises says what to try.
INPUT_WAIT_LIMIT = 0.10  # Share of a step spent waiting for the next batch.
GPU_KERNEL_FLOOR = 90.0  # Percent of the time a kernel runs; Modal's utilization guide calls 90 realistic.
MEMORY_HEADROOM_SHARE = 0.50  # A peak under half the device leaves room for a larger batch or more work on it.
MEMORY_CEILING_SHARE = 0.95  # A peak over this is one longer batch, or some fragmentation, from running out.
STALL_RATIO = 3.0  # A 90th percentile step this many times the median is a periodic stall.
STARTUP_SHARE_LIMIT = 0.10  # Setup before the first step should be a small part of the run it serves.
IDLE_CORE_SHARE = 0.50  # A job without a GPU that keeps fewer than half its cores busy is mostly waiting.
RESIDENT_SAMPLE_SECONDS = 0.25
NVIDIA_SMI_MILLISECONDS = 500
NVIDIA_SMI_FIELDS = ("utilization.gpu", "memory.used", "memory.total", "power.draw", "power.limit")
MIB = 2**20
GIB = 2**30


class DeviceSample(TypedDict):
    """One `nvidia-smi` reading; a field the device does not report is None."""

    kernel_percent: float | None
    memory_used_bytes: int | None
    memory_total_bytes: int | None
    power_watts: float | None
    power_limit_watts: float | None


class PilotReport(TypedDict):
    """What a pilot measured over its window of steps, and what it projects for the whole run.

    `gpu_*` fields come from `nvidia-smi`, which reads the whole device, other processes included; `device_peak_*`
    come from torch's allocator and count this process alone. `gpu_sampler` says why a GPU job has no readings. On a
    GH200 `gpu_memory_used_peak_bytes` includes the page cache of the HBM node, which the kernel gives back
    (`environments/gh200_suite/README.md`), so judge memory by `device_peak_*` there.
    """

    warmup_steps: int
    steps: int
    first_step_seconds: float
    step_seconds_mean: float
    step_seconds_median: float
    step_seconds_p90: float
    input_wait_share: float
    items_per_second: float | None
    item_unit: str
    device_name: str | None
    device_memory_bytes: int | None
    device_peak_allocated_bytes: int | None
    device_peak_reserved_bytes: int | None
    gpu_sampler: str
    gpu_readings: int
    gpu_kernel_percent_mean: float | None
    gpu_power_share_mean: float | None
    gpu_memory_used_peak_bytes: int | None
    host_resident_peak_bytes: int
    host_cores_busy: float
    host_cores: int
    storage_read_bytes_per_second: float | None
    storage_write_bytes_per_second: float | None
    total_steps: int | None
    projected_seconds: float | None
    budget_seconds: float | None
    findings: list[str]


def parse_nvidia_smi(line: str) -> DeviceSample:
    """Read one line of `nvidia-smi --query-gpu=<NVIDIA_SMI_FIELDS> --format=csv,noheader,nounits`.

    A bracketed value such as `[N/A]` or `[Not Supported]` reads as None; any other value that is not a number raises.
    """
    values = [_reading(value) for value in line.split(",")]
    if len(values) != len(NVIDIA_SMI_FIELDS):
        raise ValueError(f"expected {len(NVIDIA_SMI_FIELDS)} nvidia-smi fields, read {line.strip()!r}")
    kernel, used, total, power, limit = values
    return DeviceSample(
        kernel_percent=kernel,
        memory_used_bytes=None if used is None else int(used * MIB),
        memory_total_bytes=None if total is None else int(total * MIB),
        power_watts=power,
        power_limit_watts=limit,
    )


def _reading(value: str) -> float | None:
    text = value.strip()
    if text.startswith("[") or text in ("", "N/A"):
        return None
    return float(text)


class _NvidiaSmi:
    """Reads one GPU from `nvidia-smi` every NVIDIA_SMI_MILLISECONDS until stopped.

    `status` is "sampled", "nvidia-smi not found", or how `nvidia-smi` failed, so a GPU job without readings says why
    instead of looking idle.
    """

    def __init__(self, device: str) -> None:
        self.readings: list[DeviceSample] = []
        self.status = "sampled"
        self._process: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        executable = shutil.which("nvidia-smi")
        if executable is None:
            self.status = "nvidia-smi not found"
            return
        command = [
            executable, f"--id={device}", "--query-gpu=" + ",".join(NVIDIA_SMI_FIELDS),
            "--format=csv,noheader,nounits", f"--loop-ms={NVIDIA_SMI_MILLISECONDS}",
        ]
        self._process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for line in self._process.stdout:
            if not line.strip():
                continue
            try:
                self.readings.append(parse_nvidia_smi(line))
            except ValueError as error:  # "No devices were found" and the like arrive on stdout.
                self.status = f"nvidia-smi printed an unreadable line: {error}"
                return

    def stop(self) -> None:
        """End the sampling, and record in `status` a failure of `nvidia-smi` before it was stopped."""
        if self._process is None:
            return
        exited = self._process.poll()
        if exited is None:
            self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        assert self._reader is not None and self._process.stdout is not None and self._process.stderr is not None
        self._reader.join(timeout=10)
        message = self._process.stderr.read().strip().splitlines()
        self._process.stdout.close()
        self._process.stderr.close()
        if exited is not None and exited != 0:
            self.status = f"nvidia-smi exited with code {exited}" + (f": {message[0]}" if message else "")
        elif self.status == "sampled" and not self.readings:
            self.status = "nvidia-smi gave no readings"


class Pilot:
    """Times the first steps of a loop, projects the whole run, and names what looks wasteful.

    Create it first thing in the job, so that its first-step time counts loading the model, staging the data and
    verifying it, then iterate `watch(batches)` in place of `batches`. The first `warmup` steps are not timed:
    compilation, autotuning and the allocator's first growth happen there. The next `steps` are. With `stop` the loop
    ends when they have run, which is a pilot; without it the loop runs on unmeasured, so the first minutes of a full
    run print the numbers its pilot printed. On a CUDA device each step ends with `torch.cuda.synchronize()`, so a
    step is charged for the device work it queued.

    `total_steps` is the whole run's step count, for the projection. `items` counts what a step processes, as a fixed
    number or a function of the batch (sequences, tokens, rows), named by `item_unit`. `budget_seconds` is how long the
    run should take, which the projection is checked against. `receipt` is a JSON path the report is written to.
    `started` is the `clock` reading the job began at, for a pilot created after that, as `foundry.logging.Run`'s is.
    """

    def __init__(
        self,
        *,
        steps: int = 20,
        warmup: int = 3,
        total_steps: int | None = None,
        items: int | Callable[[Any], int] | None = None,
        item_unit: str = "items",
        budget_seconds: float | None = None,
        stop: bool = True,
        receipt: Path | None = None,
        clock: Callable[[], float] = time.perf_counter,
        started: float | None = None,
    ) -> None:
        if steps < 1 or warmup < 1:
            raise ValueError("a pilot times at least one step, after at least one warmup step")
        self.steps = steps
        self.warmup = warmup
        self.total_steps = total_steps
        self.items = items
        self.item_unit = item_unit
        self.budget_seconds = budget_seconds
        self.stop = stop
        self.receipt = receipt
        self._clock = clock
        self._created = clock() if started is None else started
        self._first_step_seconds: float | None = None
        self._waits: list[float] = []
        self._bodies: list[float] = []
        self._counts: list[int] = []
        self._window_opened: float | None = None
        self._cpu_start: dict[int, float] = {}
        self._storage_start: tuple[int, int] | None = None
        self._gpu: _NvidiaSmi | None = None
        self._report: PilotReport | None = None
        self._watching = False
        self._process = psutil.Process()
        self._cuda = _cuda_index()
        if self._cuda is not None:
            torch.cuda.reset_peak_memory_stats(self._cuda)
        self._resident_peak = self._process.memory_info().rss
        self._resident_stop = threading.Event()
        self._resident_sampler = threading.Thread(target=self._sample_resident, daemon=True)
        self._resident_sampler.start()

    @property
    def closed(self) -> bool:
        """Whether the window has closed, so `report` is ready."""
        return self._report is not None

    @property
    def report(self) -> PilotReport:
        """What the pilot measured, once its window has closed."""
        if self._report is None:
            raise RuntimeError("the pilot's window has not closed: iterate watch() past its steps first")
        return self._report

    def watch(self, batches: Iterable[T]) -> Iterator[T]:
        """Yield `batches`, timing the `steps` that follow the `warmup`; see the class docstring."""
        if self._watching:
            raise RuntimeError("a pilot watches one loop")
        self._watching = True
        iterator = iter(batches)
        window = self.warmup + self.steps
        arrived = previous_end = self._created
        try:
            for index in itertools.count():
                if index > 0:
                    self._synchronize()
                    previous_end = self._clock()
                    if self.warmup < index <= window:
                        self._bodies.append(previous_end - arrived)
                if index == window:
                    break
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                arrived = self._clock()
                if index == 0:
                    self._first_step_seconds = arrived - self._created
                if index == self.warmup:
                    self._open_window(arrived)
                if index >= self.warmup:
                    self._waits.append(arrived - previous_end)
                    self._counts.append(self._count(batch))
                yield batch
        finally:
            self._close()
        if not self.stop:
            yield from iterator

    def _count(self, batch: Any) -> int:
        if self.items is None:
            return 0
        if isinstance(self.items, int):
            return self.items
        return self.items(batch)

    def _synchronize(self) -> None:
        if self._cuda is not None:
            torch.cuda.synchronize(self._cuda)

    def _sample_resident(self) -> None:
        while not self._resident_stop.wait(RESIDENT_SAMPLE_SECONDS):
            self._resident_peak = max(self._resident_peak, self._process.memory_info().rss)

    def _open_window(self, opened: float) -> None:
        print(f"pilot: timing {self.steps} steps after {self.warmup} warmup", file=sys.stderr, flush=True)
        self._window_opened = opened
        self._cpu_start = _cpu_seconds(self._process)
        self._storage_start = _storage_bytes(self._process)
        if self._cuda is not None:
            self._gpu = _NvidiaSmi(_nvidia_smi_device(self._cuda))

    def _close(self) -> None:
        if self._report is not None:
            return
        closed = self._clock()
        self._resident_stop.set()
        self._resident_sampler.join()
        if self._gpu is not None:
            self._gpu.stop()
        self._report = self._measure(closed)
        print(describe(self._report), file=sys.stderr, flush=True)
        if self.receipt is not None:
            write_json_dict_atomic(self.receipt, dict(self._report))

    def _measure(self, closed: float) -> PilotReport:
        """The report of the window that closed at `closed`, with its findings."""
        # A loop left early has fetched a batch whose step never finished, so the waits can outnumber the bodies by one.
        steps = [wait + body for wait, body in zip(self._waits, self._bodies, strict=False)]
        measured = len(steps)
        step_total = sum(steps)
        window = closed - self._window_opened if self._window_opened is not None else 0.0

        device_name = device_memory = peak_allocated = peak_reserved = None
        if self._cuda is not None:
            properties = torch.cuda.get_device_properties(self._cuda)
            device_name, device_memory = properties.name, properties.total_memory
            peak_allocated = torch.cuda.max_memory_allocated(self._cuda)
            peak_reserved = torch.cuda.max_memory_reserved(self._cuda)

        readings = self._gpu.readings if self._gpu is not None else []
        kernels = [reading["kernel_percent"] for reading in readings if reading["kernel_percent"] is not None]
        powers = [
            reading["power_watts"] / reading["power_limit_watts"] for reading in readings
            if reading["power_watts"] is not None and reading["power_limit_watts"]
        ]
        used = [reading["memory_used_bytes"] for reading in readings if reading["memory_used_bytes"] is not None]
        if self._gpu is not None:
            sampler = self._gpu.status
        else:
            sampler = "no CUDA device" if self._cuda is None else "the window never opened"

        storage_end = _storage_bytes(self._process)
        read_rate = write_rate = None
        if self._storage_start is not None and storage_end is not None and window > 0:
            read_rate = (storage_end[0] - self._storage_start[0]) / window
            write_rate = (storage_end[1] - self._storage_start[1]) / window
        busy = _cores_busy(self._cpu_start, _cpu_seconds(self._process), window) if self._window_opened else 0.0

        mean = step_total / measured if measured else math.nan
        projected = None
        if self.total_steps is not None and measured:
            projected = (self._first_step_seconds or 0.0) + self.total_steps * mean
        report = PilotReport(
            warmup_steps=self.warmup,
            steps=measured,
            first_step_seconds=self._first_step_seconds if self._first_step_seconds is not None else math.nan,
            step_seconds_mean=mean,
            step_seconds_median=statistics.median(steps) if measured else math.nan,
            step_seconds_p90=_percentile(steps, 0.9) if measured else math.nan,
            input_wait_share=sum(self._waits[:measured]) / step_total if step_total > 0 else 0.0,
            items_per_second=sum(self._counts[:measured]) / step_total if self.items is not None and step_total > 0 else None,
            item_unit=self.item_unit,
            device_name=device_name,
            device_memory_bytes=device_memory,
            device_peak_allocated_bytes=peak_allocated,
            device_peak_reserved_bytes=peak_reserved,
            gpu_sampler=sampler,
            gpu_readings=len(readings),
            gpu_kernel_percent_mean=statistics.fmean(kernels) if kernels else None,
            gpu_power_share_mean=statistics.fmean(powers) if powers else None,
            gpu_memory_used_peak_bytes=max(used) if used else None,
            host_resident_peak_bytes=self._resident_peak,
            host_cores_busy=busy,
            host_cores=_usable_cores(),
            storage_read_bytes_per_second=read_rate,
            storage_write_bytes_per_second=write_rate,
            total_steps=self.total_steps,
            projected_seconds=projected,
            budget_seconds=self.budget_seconds,
            findings=[],
        )
        report["findings"] = assess(report)
        return report


def assess(report: PilotReport) -> list[str]:
    """Name what in `report` looks wasteful, each with the number that shows it and what to try."""
    if report["steps"] == 0:
        return [f"no step was timed: the loop ended before its {report['warmup_steps']} warmup steps and one more"]
    findings = []
    if report["input_wait_share"] > INPUT_WAIT_LIMIT:
        findings.append(
            f"input-bound: {report['input_wait_share']:.0%} of each step waits for the next batch; give the loader "
            "workers and pinned memory, prefetch, and read prepared inputs from local disk, not item by item from a volume"
        )
    kernel = report["gpu_kernel_percent_mean"]
    # A device that waits for input idles for that share of the step, so only the idling the waits leave unexplained counts.
    if kernel is not None and kernel < (1 - report["input_wait_share"]) * GPU_KERNEL_FLOOR:
        findings.append(
            f"the GPU ran kernels {kernel:.0f}% of the time, idling beyond its waits for input: look for host work in "
            "the step (.item(), .cpu(), printing a tensor), batches too small for the device, and many small kernels"
        )
    total = report["device_memory_bytes"]
    reserved = report["device_peak_reserved_bytes"]
    if total and reserved is not None:
        share = reserved / total
        if share < MEMORY_HEADROOM_SHARE:
            findings.append(
                f"device memory peaked at {share:.0%} of {total / GIB:.0f} GiB: raise the batch, or put more of the "
                "work on this device, until the peak nears 85 percent"
            )
        elif share > MEMORY_CEILING_SHARE:
            findings.append(
                f"device memory peaked at {share:.0%} of {total / GIB:.0f} GiB: a longer batch or fragmentation will "
                "run it out, so leave headroom"
            )
    median = report["step_seconds_median"]
    if median > 0 and report["step_seconds_p90"] > STALL_RATIO * median:
        findings.append(
            f"uneven steps: the 90th percentile is {report['step_seconds_p90'] / median:.1f} times the median, a "
            "periodic stall such as a checkpoint or log write, a synchronizing metric, or a slow shard"
        )
    projected = report["projected_seconds"]
    setup = report["first_step_seconds"]
    if projected and setup > STARTUP_SHARE_LIMIT * projected:
        findings.append(
            f"setup took {_duration(setup)}, {setup / projected:.0%} of the projected run: load, stage and verify "
            "once, on a CPU container, and reuse the result"
        )
    cores, busy = report["host_cores"], report["host_cores_busy"]
    if report["device_name"] is None and cores >= 4 and busy < IDLE_CORE_SHARE * cores:
        findings.append(f"{busy:.1f} of {cores} cores busy: parallelize the stage, or run it on a smaller machine")
    budget = report["budget_seconds"]
    if projected is not None and budget is not None and projected > budget:
        findings.append(
            f"projected {_duration(projected)}, over the {_duration(budget)} budget: find what the run computes twice, "
            "or checks that no input can change, before launching it"
        )
    return findings


def describe(report: PilotReport) -> str:
    """The report as a few lines, for the job's log and for the experiment's launch record."""
    if report["steps"] == 0:
        return "pilot: " + "; ".join(report["findings"])
    rate = report["items_per_second"]
    lines = [
        f"pilot: {report['steps']} steps after {report['warmup_steps']} warmup, {report['step_seconds_mean']:.3g} s a step "
        f"(median {report['step_seconds_median']:.3g}, p90 {report['step_seconds_p90']:.3g})"
        + ("" if rate is None else f", {rate:,.0f} {report['item_unit']}/s"),
        f"  first step {_duration(report['first_step_seconds'])} after the pilot began; "
        f"waiting for input {report['input_wait_share']:.0%} of each step",
    ]
    if report["device_name"] is not None:
        allocated, reserved = report["device_peak_allocated_bytes"] or 0, report["device_peak_reserved_bytes"] or 0
        lines.append(
            f"  {report['device_name']}: peak {reserved / GIB:.1f} GiB reserved, {allocated / GIB:.1f} GiB allocated, "
            f"of {(report['device_memory_bytes'] or 0) / GIB:.1f} GiB"
        )
        kernel, power = report["gpu_kernel_percent_mean"], report["gpu_power_share_mean"]
        if kernel is None:
            lines.append(f"  GPU utilization not read: {report['gpu_sampler']}")
        else:
            lines.append(
                f"  GPU: kernels running {kernel:.0f}% of the time"
                + ("" if power is None else f", drawing {power:.0%} of its power limit")
                + f" ({report['gpu_readings']} readings)"
            )
    storage = ""
    if report["storage_read_bytes_per_second"] is not None and report["storage_write_bytes_per_second"] is not None:
        storage = (
            f"; storage read {report['storage_read_bytes_per_second'] / MIB:.1f} MiB/s, "
            f"written {report['storage_write_bytes_per_second'] / MIB:.1f} MiB/s"
        )
    lines.append(
        f"  host: {report['host_resident_peak_bytes'] / GIB:.1f} GiB resident at peak, "
        f"{report['host_cores_busy']:.1f} of {report['host_cores']} cores busy{storage}"
    )
    projected, total_steps, budget = report["projected_seconds"], report["total_steps"], report["budget_seconds"]
    if projected is not None and total_steps is not None:
        lines.append(
            f"  projected: {_duration(projected)} for {total_steps:,} steps"
            + ("" if budget is None else f", against a budget of {_duration(budget)}")
        )
    lines.append("findings:" + ("" if report["findings"] else " none"))
    lines.extend(f"  - {finding}" for finding in report["findings"])
    return "\n".join(lines)


def _cuda_index() -> int | None:
    """The CUDA device this process computes on, or None without torch or a CUDA device."""
    if torch is None or not torch.cuda.is_available():
        return None
    return torch.cuda.current_device()


def _nvidia_smi_device(index: int) -> str:
    """The `nvidia-smi --id` of CUDA device `index`, which CUDA_VISIBLE_DEVICES renumbers when it is set."""
    visible = [entry.strip() for entry in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if entry.strip()]
    return visible[index] if index < len(visible) else str(index)


def _usable_cores() -> int:
    """The cores this process may run on, which a container's CPU set limits, else the machine's count."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _cpu_seconds(process: psutil.Process) -> dict[int, float]:
    """User and system CPU seconds of `process` and each live descendant, such as loader workers, by pid."""
    seconds = {}
    for member in [process, *process.children(recursive=True)]:
        try:
            times = member.cpu_times()
        except psutil.NoSuchProcess:  # A worker that exited between the listing and the read.
            continue
        seconds[member.pid] = times.user + times.system
    return seconds


def _cores_busy(start: dict[int, float], end: dict[int, float], seconds: float) -> float:
    """Cores kept busy over `seconds`, from CPU seconds by pid; a process new since `start` counts from zero."""
    if seconds <= 0:
        return 0.0
    return sum(value - start.get(pid, 0.0) for pid, value in end.items()) / seconds


def _storage_bytes(process: psutil.Process) -> tuple[int, int] | None:
    """Bytes `process` and its live descendants have read and written through storage, or None where unreported."""
    read = written = 0
    for member in [process, *process.children(recursive=True)]:
        try:
            counters = member.io_counters()
        except psutil.NoSuchProcess:
            continue
        except (AttributeError, NotImplementedError, psutil.AccessDenied):  # No per-process I/O on this platform.
            return None
        read += counters.read_bytes
        written += counters.write_bytes
    return read, written


def _percentile(values: list[float], share: float) -> float:
    """The nearest-rank percentile of `values` at `share`, 0.9 for the 90th."""
    ordered = sorted(values)
    return ordered[max(math.ceil(share * len(ordered)) - 1, 0)]


def _duration(seconds: float) -> str:
    """`seconds` in the two largest units that matter: 48.2 s, 12 min 5 s, 4 h 12 min, 3 d 4 h."""
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, second = divmod(round(seconds), 60)
    hours, minute = divmod(minutes, 60)
    days, hour = divmod(hours, 24)
    if days:
        return f"{days} d {hour} h"
    if hours:
        return f"{hours} h {minute} min"
    return f"{minutes} min {second} s"
