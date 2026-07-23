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
    state: str = "S",
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
    # Fields after comm begin at field 3 (state); field 22 is starttime.
    stat_fields = [state, *(["0"] * 18), str(process_id * 10)]
    (process_root / "stat").write_text(
        f"{process_id} (python) {' '.join(stat_fields)}\n",
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


def test_process_exit_during_pss_read_uses_only_that_process_rss_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tests.cpu.resource_telemetry as telemetry

    proc_root = tmp_path / "proc"
    _write_fake_process(
        proc_root,
        100,
        children=(200,),
        rss_kib=100,
        pss_kib=50,
    )
    _write_fake_process(
        proc_root,
        200,
        children=(),
        rss_kib=200,
        pss_kib=100,
    )
    original_read_pss = telemetry._read_pss_bytes

    def disappearing_pss(root: Path, process_id: int) -> tuple[int | None, str | None]:
        if process_id == 200:
            process_root = root / str(process_id)
            for path in sorted(process_root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                else:
                    path.rmdir()
            process_root.rmdir()
            return None, "smaps_rollup disappeared during sampling"
        return original_read_pss(root, process_id)

    monkeypatch.setattr(telemetry, "_read_pss_bytes", disappearing_pss)
    snapshot = sample_process_tree_memory(root_pids=(100,), proc_root=proc_root)

    assert snapshot["process_ids"] == [100, 200]
    assert snapshot["pss_complete"] is False
    assert snapshot["aggregate_pss_bytes"] is None
    assert snapshot["aggregate_hybrid_bytes"] == 250 * 1024
    child = next(process for process in snapshot["processes"] if process["pid"] == 200)
    assert child["accounted_bytes"] == 200 * 1024
    assert child["accounting_metric"] == "resident-set-size-fallback"
    assert snapshot["transient_process_event_count"] == 1
    assert "pid 200: exited before PSS sampling completed" in snapshot[
        "transient_process_events"
    ]
    assert snapshot["pss_fallback_reasons"] == [
        "pid 200: smaps_rollup disappeared during sampling"
    ]


def test_zombie_without_smaps_is_zero_resident_and_keeps_pss_complete(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    _write_fake_process(
        proc_root,
        100,
        children=(200,),
        rss_kib=100,
        pss_kib=50,
    )
    _write_fake_process(
        proc_root,
        200,
        children=(),
        rss_kib=999,
        pss_kib=999,
        state="Z",
    )
    (proc_root / "200" / "smaps_rollup").unlink()

    snapshot = sample_process_tree_memory(root_pids=(100,), proc_root=proc_root)

    zombie = next(process for process in snapshot["processes"] if process["pid"] == 200)
    assert zombie["rss_bytes"] == 0
    assert zombie["pss_bytes"] == 0
    assert zombie["accounted_bytes"] == 0
    assert zombie["state"] == "Z"
    assert snapshot["pss_complete"] is True


def test_persistent_pss_denial_uses_only_denied_process_rss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tests.cpu.resource_telemetry as telemetry

    proc_root = tmp_path / "proc"
    _write_fake_process(
        proc_root,
        100,
        children=(200,),
        rss_kib=100,
        pss_kib=50,
    )
    _write_fake_process(
        proc_root,
        200,
        children=(),
        rss_kib=200,
        pss_kib=100,
    )
    original_read_pss = telemetry._read_pss_bytes

    def denied_pss(root: Path, process_id: int) -> tuple[int | None, str | None]:
        if process_id == 200:
            return None, "smaps_rollup permission denied"
        return original_read_pss(root, process_id)

    monkeypatch.setattr(telemetry, "_read_pss_bytes", denied_pss)
    snapshot = sample_process_tree_memory(root_pids=(100,), proc_root=proc_root)
    gate = select_concurrent_memory_gate(
        {
            "sample_count": 1,
            "peak_concurrent_rss_bytes": snapshot["aggregate_rss_bytes"],
            "peak_concurrent_hybrid_bytes": snapshot["aggregate_hybrid_bytes"],
            "peak_concurrent_pss_bytes": None,
            "pss_complete_for_all_samples": False,
            "pss_fallback_reasons": snapshot["pss_fallback_reasons"],
        }
    )

    assert snapshot["aggregate_rss_bytes"] == 300 * 1024
    assert snapshot["aggregate_hybrid_bytes"] == 250 * 1024
    assert gate["metric"] == "per-process-pss-rss-hybrid"
    assert gate["peak_bytes"] == 250 * 1024
    assert gate["fallback_used"] is True
    assert gate["fallback_is_conservative"] is True


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
        assert gate["metric"] == "per-process-pss-rss-hybrid"
    else:
        assert gate["metric"] == "proportional-set-size"


def test_pss_is_preferred_and_incomplete_pss_uses_per_process_hybrid() -> None:
    preferred = select_concurrent_memory_gate(
        {
            "sample_count": 4,
            "peak_concurrent_rss_bytes": 900,
            "peak_concurrent_hybrid_bytes": 600,
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
            "peak_concurrent_hybrid_bytes": 700,
            "peak_concurrent_pss_bytes": 600,
            "pss_complete_for_all_samples": False,
            "pss_fallback_reasons": ["pid 123: smaps_rollup permission denied"],
        }
    )
    assert fallback["metric"] == "per-process-pss-rss-hybrid"
    assert fallback["peak_bytes"] == 700
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
                "peak_concurrent_hybrid_bytes": None,
                "pss_complete_for_all_samples": False,
            }
        )


def test_memory_over_budget_forces_nonzero_pytest_exit(tmp_path: Path) -> None:
    test_file = tmp_path / "test_synthetic_gate.py"
    test_file.write_text("def test_passes():\n    assert True\n", encoding="utf-8")
    script = textwrap.dedent(
        f"""
        import pytest

        from tests.cpu.conftest import _enforce_concurrent_memory_budget

        class SyntheticMemoryGate:
            @pytest.hookimpl(trylast=True)
            def pytest_sessionfinish(self, session, exitstatus):
                del exitstatus
                _enforce_concurrent_memory_budget(
                    session,
                    {{
                        "metric": "per-process-pss-rss-hybrid",
                        "peak_bytes": 4 * 1024**3 + 1,
                    }},
                )

        raise SystemExit(
            pytest.main(
                [{str(test_file)!r}, "-q", "-p", "no:cacheprovider"],
                plugins=[SyntheticMemoryGate()],
            )
        )
        """
    )
    workspace = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(workspace), environment.get("PYTHONPATH", ""))
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=workspace,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=8,
    )

    assert completed.returncode == int(pytest.ExitCode.TESTS_FAILED)
    assert "concurrent memory budget exceeded" in completed.stdout + completed.stderr
