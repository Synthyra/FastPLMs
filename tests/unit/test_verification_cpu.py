"""Portable verification preserves failures and isolates CPU-tier pytest policies."""

from __future__ import annotations

import subprocess
import sys

from pathlib import Path

from tools.verification import worker
from tools.verification.cpu import upstream_legal_files


ROOT = Path(__file__).resolve().parents[2]


def test_verification_includes_canonical_upstream_licenses() -> None:
    import tomllib

    with (ROOT / "src/fastplms/models.toml").open("rb") as stream:
        manifest = tomllib.load(stream)
    files = upstream_legal_files(ROOT)
    assert {str(Path(name).parent).replace("\\", "/") for name in files} == {
        source["path"] for source in manifest["upstreams"]
    }
    assert all((ROOT / name).is_file() for name in files)


def test_cpu_policy_is_collected_separately() -> None:
    batches = worker.test_batches(ROOT)
    assert any("test_confidence_" in path for path in batches["unit-integration"])
    assert "tests/unit/test_verification_cpu.py" in batches["unit-integration"]
    for name, paths in batches.items():
        assert all((ROOT / path.split("::", 1)[0]).is_file() for path in paths)
        assert all(path.startswith("tests/cpu/") == (name == "cpu-contracts") for path in paths)


def test_failure_keeps_outputs_and_later_batches(tmp_path, monkeypatch) -> None:
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert command[4:6] == ["-m", "not gpu"]
        junit = Path(command[-1].split("=", 1)[1])
        junit.write_text('<testsuite tests="1" failures="1"/>', encoding="utf-8")
        return subprocess.CompletedProcess(command, 1 if len(calls) == 1 else 0, "stdout", "stderr")

    monkeypatch.setattr(worker.subprocess, "run", run)
    result = worker.run_batches(ROOT, tmp_path)
    assert result["status"] == "failed"
    assert len(calls) == 3
    assert [batch["exit_code"] for batch in result["batches"]] == [1, 0, 0]
    assert result["batches"][0]["stdout"] == "stdout"
    assert result["batches"][0]["stderr"] == "stderr"
    assert 'failures="1"' in result["batches"][0]["junit_xml"]


def test_timeout_preserves_partial_output(tmp_path, monkeypatch) -> None:
    def run(command, **kwargs):
        raise subprocess.TimeoutExpired(
            command, kwargs["timeout"], output=b"partial", stderr=b"failure"
        )

    monkeypatch.setattr(worker.subprocess, "run", run)
    result = worker.run_batches(ROOT, tmp_path)
    assert result["status"] == "failed"
    assert result["batches"][0]["status"] == "timed_out"
    assert result["batches"][0]["exit_code"] is None
    assert result["batches"][0]["stdout"] == "partial"
    assert result["batches"][0]["junit_xml"] is None


def test_exhausted_budget_does_not_start_subprocess(tmp_path, monkeypatch) -> None:
    def run(*args, **kwargs):
        raise AssertionError("Budget-exhausted batch must not launch")

    monkeypatch.setattr(worker.subprocess, "run", run)
    result = worker.run_batches(ROOT, tmp_path, maximum_seconds=0)
    assert result["status"] == "failed"
    assert all(batch["status"] == "not_run" for batch in result["batches"])


def test_help_does_not_need_site_packages() -> None:
    result = subprocess.run(
        [sys.executable, "-S", "-m", "tools.verification.cpu", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--output-root" in result.stdout
