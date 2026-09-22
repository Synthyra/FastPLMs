"""Remote job exit codes and shell path validation without an SSH connection."""

import shlex
import subprocess
import sys
import pytest

from pathlib import Path

from tools.confidence import ssh


@pytest.mark.parametrize(
    "job", ["../escape", "a/b", "a;b", "a b", "$(id)", "-option", "", "a" * 81]
)
def test_invalid_job_names_fail_before_connecting(job, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid job name reached SSH")

    monkeypatch.setattr(ssh.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="Job names"):
        ssh.wait("unused", Path("unused"), job, 100)
    with pytest.raises(ValueError, match="Job names"):
        ssh.start("unused", Path("unused"), "valid", [], job)


@pytest.mark.parametrize("status", [0, 7, 137])
def test_wait_propagates_recorded_job_status(status, tmp_path, monkeypatch):
    log = tmp_path / "train-300.log"
    log.write_text(f"retained output\nLoading weights: hidden progress\n[exit status {status}]\n")
    monkeypatch.setattr(ssh, "REMOTE_LOGS", shlex.quote(str(tmp_path)))
    run = subprocess.run
    results = []

    def execute_remote_locally(command, **kwargs):
        assert command[0] == "ssh"
        result = run(["bash", "-c", command[-1]], capture_output=True, text=True, timeout=5)
        results.append(result)
        if kwargs.get("check"):
            result.check_returncode()
        return result

    monkeypatch.setattr(ssh.subprocess, "run", execute_remote_locally)
    if status:
        with pytest.raises(subprocess.CalledProcessError) as error:
            ssh.wait("unused", Path("unused"), "train-300", 1000)
        assert error.value.returncode == status
    else:
        ssh.wait("unused", Path("unused"), "train-300", 1000)
    assert "retained output" in results[0].stdout
    assert "Loading weights" not in results[0].stdout


def test_wait_cli_preserves_failed_job_exit_code(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["ssh", "--host", "unused", "--identity", "unused", "wait", "train"]
    )

    def failed(*args):
        raise subprocess.CalledProcessError(7, ["ssh"])

    monkeypatch.setattr(ssh, "wait", failed)
    with pytest.raises(SystemExit) as error:
        ssh.main()
    assert error.value.code == 7


def test_wait_rejects_nonpositive_tail_size():
    with pytest.raises(ValueError, match="positive"):
        ssh.wait("unused", Path("unused"), "train", 0)
