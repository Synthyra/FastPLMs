---
id: foundry.pilot
kind: library_module
status: active
aliases: [foundry.pilot]
---
# foundry.pilot

Times the first steps of a job, projects the whole run, and names what looks wasteful, so that a run's length, its peak
memory and its bottleneck are known before it is paid for. [The compute convention](../../docs/conventions/compute.md)
says when a pilot is due and what to do with what it finds.

| Name | Does |
|---|---|
| `Pilot(steps=20, warmup=3, total_steps=None, items=None, item_unit="items", budget_seconds=None, stop=True, receipt=None, started=None)` | Starts timing at creation, or at `started` (a `time.perf_counter()` reading), so create it first thing in the job: the time to the first step then counts model loading, data staging and verification. `foundry.logging.Run.batches` creates one for you |
| `Pilot.watch(batches)` | Yields `batches`. Times the `steps` after the `warmup`, then ends the loop (`stop=True`, a pilot) or lets it run on unmeasured (`stop=False`, the first minutes of a full run). Prints the report to stderr when the window closes, and writes `receipt` as JSON |
| `Pilot.report` | The `PilotReport`: step time (mean, median, 90th percentile), the share of each step spent waiting for input, items per second, device memory peaks, GPU kernel and power utilization, host memory, busy cores and storage rates, the projected run and the findings |
| `assess(report)` | The findings: input-bound, GPU kernels under 90 percent of the time that its input waits leave free, device memory under half or over 95 percent, uneven steps, setup over a tenth of the run, idle cores on a job without a GPU, a projection over the budget |
| `describe(report)` | The report as the few lines a launch record quotes |
| `parse_nvidia_smi(line)` | One `nvidia-smi --query-gpu` line as a `DeviceSample`, with `[N/A]` read as None |

```python
from foundry.pilot import Pilot

pilot = Pilot(total_steps=epochs * len(loader), items=lambda batch: batch["input_ids"].shape[0],
              item_unit="sequences", budget_seconds=2 * 3600, receipt=run_directory / "pilot.json")
model = load_model()  # counted in the time to the first step
for batch in pilot.watch(loader):
    train_step(model, batch)
```

## What it measures, and its limits

- **Waiting and computing.** A step runs from the arrival of its batch to the arrival of the next. The time spent in the
  loader's `next()` is the wait; the rest is the step's own work. On a CUDA device each step ends with
  `torch.cuda.synchronize()`, so queued kernels are charged to the step that queued them and the wait is time the device
  sat idle. The synchronization costs a little throughput, so a pilot's projection errs long.
- **GPU utilization** is `nvidia-smi`'s: the share of time at least one kernel ran, read every half second during the
  window, for the whole device and so for other processes on it too. It shows idling, not efficiency: a device busy
  with small kernels reads 100 percent. Power drawn against the limit is the better sign of saturation, where the device
  reports it.
- **Device memory** is torch's allocator for this process: the reserved peak is what a larger batch has to fit beside.
- **Host** numbers come from `psutil`: the main process's resident peak, the CPU seconds of the process and its live
  workers over the window, and the bytes they moved through storage. A network volume's reads may not show as storage on
  every kernel; the wait share is the measure that counts.
- **The projection** is the time to the first step plus `total_steps` mean steps. Evaluation, checkpoints and anything
  outside the watched loop are not in it.

## Why it starts shared

Code reaches `foundry` on its second consumer ([the record](../../docs/decisions/workspace/2026-09-21_promotion_on_the_second_consumer.md)).
This module starts here, as `foundry.embedding` did, because [the compute convention](../../docs/conventions/compute.md)
asks every long job for a pilot, and about sixty project files already measure some part of one by hand: peak memory
with `torch.cuda.max_memory_allocated`, utilization with their own `nvidia-smi` samplers (`property_lowcode`'s
`monitor_parallel_probe_hardware.py`, `dual_triangle_attention`'s `GPUMonitor`), stage peaks with `property_lowcode`'s
`StageRecorder`. Those projects are `converted: false` and keep their copies until they are converted.

`torch` is optional at run time: a data pipeline without it gets timing, host and storage numbers.

Tests: `tests/tier1_unit/test_pilot.py`, on the CPU with a hand-moved clock.
