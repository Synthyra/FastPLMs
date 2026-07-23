"""Deterministic contracts for concurrent Linux process-tree memory evidence."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.cpu.resource_telemetry import (
    ConcurrentProcessTreeSampler,
    MemoryEvidenceError,
    sample_process_tree_memory,
    select_concurrent_memory_gate,
)


def _write_fake_process(
    proc_root: Path,
    process_id: int,
    *,
    children: tuple[int, ...],
    rss_kib: int,
    pss_kib: int,
) -> None:
    process_root = proc_root / str(process_id)
    task_root = process_root / "task" / str(process_id)
    task_root.mkdir(parents=True)
    (task_root / "children").write_text(
        " ".join(str(child_id) for child_id in children),
        encoding="utf-8",
    )
    (process_root / "status").write_text(
        f"Name:\tpython\nVmRSS:\t{rss_kib} kB\n",
        encoding="utf-8",
    )
    (process_root / "smaps_rollup").write_text(
        f"Pss:\t{pss_kib} kB\n",
        encoding="utf-8",
    )


def test_overlapping_worker_and_child_roots_are_deduplicated(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    _write_fake_process(
        proc_root,
        100,
        children=(200, 300),
        rss_kib=100,
        pss_kib=50,
    )
    _write_fake_process(
        proc_root,
        200,
        children=(400,),
        rss_kib=200,
        pss_kib=100,
    )
    _write_fake_process(
        proc_root,
        300,
        children=(),
        rss_kib=300,
        pss_kib=150,
    )
    _write_fake_process(
        proc_root,
        400,
        children=(),
        rss_kib=400,
        pss_kib=200,
    )

    snapshot = sample_process_tree_memory(
        proc_root=proc_root,
        # These roots deliberately overlap: 200 and 400 are descendants of
        # 100, and 400 is also a descendant of 200.
        root_pids=(100, 200, 400, 200),
    )

    assert snapshot["root_pids"] == [100, 200, 400]
    assert snapshot["process_ids"] == [100, 200, 300, 400]
    assert snapshot["process_count"] == 4
    assert snapshot["aggregate_rss_bytes"] == 1_000 * 1024
    assert snapshot["aggregate_pss_bytes"] == 500 * 1024
    assert snapshot["pss_complete"] is True
    assert len(snapshot["processes"]) == len(set(snapshot["process_ids"]))


def test_sampler_captures_a_live_child_process() -> None:
    assert sys.platform.startswith("linux")
    script = textwrap.dedent(
        """
        import os
        import sys

        payload = bytearray(8 * 1024 * 1024)
        for offset in range(0, len(payload), 4096):
            payload[offset] = 1
        print(os.getpid(), flush=True)
        sys.stdin.buffer.read(1)
        """
    )
    sampler = ConcurrentProcessTreeSampler(
        root_pids=(os.getpid(),),
        sample_interval_seconds=0.01,
    )
    sampler.start()
    child = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    summary: dict[str, object]
    snapshot: dict[str, object]
    try:
        assert child.stdout is not None
        child_id = int(child.stdout.readline().decode("utf-8").strip())
        assert child_id == child.pid
        snapshot = sampler.sample_now()
        assert child_id in snapshot["process_ids"]
        child_record = next(
            process for process in snapshot["processes"] if process["pid"] == child_id
        )
        assert child_record["rss_bytes"] >= 8 * 1024**2
    finally:
        if child.poll() is None and child.stdin is not None:
            try:
                child.stdin.write(b"x")
                child.stdin.flush()
            except BrokenPipeError:
                pass
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=3)
        for stream in (child.stdin, child.stdout, child.stderr):
            if stream is not None:
                stream.close()
        summary = sampler.stop()

    assert child.returncode == 0
    assert child.pid in summary["observed_process_ids"]
    assert summary["sample_count"] >= 3
    gate = select_concurrent_memory_gate(summary)
    assert gate["peak_bytes"] > 0
    if gate["fallback_used"]:
        assert gate["fallback_is_conservative"] is True
        assert gate["metric"] == "resident-set-size"
    else:
        assert gate["metric"] == "proportional-set-size"


def test_pss_is_preferred_and_incomplete_pss_falls_back_to_concurrent_rss() -> None:
    preferred = select_concurrent_memory_gate(
        {
            "sample_count": 4,
            "peak_concurrent_rss_bytes": 900,
            "peak_concurrent_pss_bytes": 600,
            "pss_complete_for_all_samples": True,
            "pss_fallback_reasons": [],
        }
    )
    assert preferred == {
        "metric": "proportional-set-size",
        "source": "/proc/<pid>/smaps_rollup:Pss",
        "peak_bytes": 600,
        "fallback_used": False,
        "fallback_is_conservative": False,
        "fallback_reasons": [],
    }

    fallback = select_concurrent_memory_gate(
        {
            "sample_count": 4,
            "peak_concurrent_rss_bytes": 900,
            "peak_concurrent_pss_bytes": 600,
            "pss_complete_for_all_samples": False,
            "pss_fallback_reasons": ["pid 123: smaps_rollup permission denied"],
        }
    )
    assert fallback["metric"] == "resident-set-size"
    assert fallback["peak_bytes"] == 900
    assert fallback["fallback_used"] is True
    assert fallback["fallback_is_conservative"] is True
    assert fallback["fallback_reasons"] == [
        "pid 123: smaps_rollup permission denied"
    ]

    with pytest.raises(MemoryEvidenceError, match="no positive RSS peak"):
        select_concurrent_memory_gate(
            {
                "sample_count": 1,
                "peak_concurrent_rss_bytes": None,
                "pss_complete_for_all_samples": False,
            }
        )
