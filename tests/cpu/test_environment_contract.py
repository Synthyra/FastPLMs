"""Environment invariants for the mandatory CPU confidence lane."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import huggingface_hub
import huggingface_hub.file_download
import pytest
import safetensors.torch
import torch
import transformers

from tests.cpu.resource_telemetry import aggregate_process_memory


def test_locked_cpu_runtime() -> None:
    assert sys.version_info[:2] == (3, 12)
    assert torch.__version__.split("+", maxsplit=1)[0] == "2.13.0"
    assert transformers.__version__ == "5.13.0"
    assert not torch.cuda.is_available()
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["HF_DATASETS_OFFLINE"] == "1"
    assert os.environ["PYTEST_XDIST_AUTO_NUM_WORKERS"] == "4"
    assert os.environ["FASTPLMS_CPU_BOOTSTRAPPED"] == "1"
    assert os.environ["FASTPLMS_CPU_CACHE_STARTED_EMPTY"] == "1"
    assert torch.get_num_threads() == 1
    assert torch.get_num_interop_threads() == 1
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert os.environ[variable] == "1"


def test_all_runtime_caches_are_fresh_and_task_scoped() -> None:
    variables = (
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
    )
    paths = [Path(os.environ[variable]).resolve() for variable in variables]
    assert all(path.is_dir() for path in paths)
    common_root = Path(os.path.commonpath(paths))
    assert common_root.name.startswith("fastplms-cpu-contract-cache-")


def test_network_guard_is_active() -> None:
    with pytest.raises(RuntimeError, match="Network access is forbidden"):
        socket.getaddrinfo("huggingface.co", 443)
    with (
        socket.socket() as client,
        pytest.raises(RuntimeError, match="Network access is forbidden"),
    ):
        client.connect_ex(("huggingface.co", 443))
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
        with pytest.raises(RuntimeError, match="Network access is forbidden"):
            client.sendto(b"blocked", ("127.0.0.1", 9))
        if hasattr(client, "sendmsg"):
            with pytest.raises(RuntimeError, match="Network access is forbidden"):
                client.sendmsg([b"blocked"], [], 0, ("127.0.0.1", 9))
    with pytest.raises(RuntimeError, match="Network access is forbidden"):
        huggingface_hub.hf_hub_download("org/model", "config.json")
    with pytest.raises(RuntimeError, match="Network access is forbidden"):
        huggingface_hub.file_download.hf_hub_download("org/model", "config.json")


def test_optimized_subprocess_inherits_socket_and_hub_guards() -> None:
    script = textwrap.dedent(
        """
        import os
        import socket

        import huggingface_hub

        assert os.environ["FASTPLMS_CPU_BOOTSTRAPPED"] == "1"
        for operation in (
            lambda: socket.getaddrinfo("huggingface.co", 443),
            lambda: huggingface_hub.hf_hub_download("org/model", "config.json"),
        ):
            try:
                operation()
            except RuntimeError as error:
                assert "Network access is forbidden" in str(error)
            else:
                raise AssertionError("Inherited CPU network guard was not active")
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
            for operation in (
                lambda: client.sendto(b"blocked", ("127.0.0.1", 9)),
                lambda: client.sendmsg([b"blocked"], [], 0, ("127.0.0.1", 9)),
            ):
                try:
                    operation()
                except RuntimeError as error:
                    assert "Network access is forbidden" in str(error)
                else:
                    raise AssertionError("Inherited CPU UDP guard was not active")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-O", "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env=dict(os.environ),
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert completed.returncode == 0, completed.stderr


def test_container_process_spawns_are_blocked_at_python_startup() -> None:
    operations = (
        lambda: subprocess.run(["docker", "version"], check=False),
        lambda: subprocess.Popen(["podman", "info"]),
        lambda: subprocess.run("docker compose version", shell=True, check=False),
        lambda: subprocess.run(["sh", "-lc", "buildx version"], check=False),
        lambda: os.system("sudo docker buildx version"),
    )
    for operation in operations:
        with pytest.raises(RuntimeError, match="Container execution is forbidden"):
            operation()


def test_repository_checkpoint_reads_are_structurally_blocked() -> None:
    workspace = Path(__file__).resolve().parents[2]
    checkpoints = sorted((workspace / "tests/goldens").glob("*.safetensors"))
    assert checkpoints, "The release tree must contain pinned golden tensors"
    checkpoint = checkpoints[0]

    for operation in (
        lambda: checkpoint.open("rb"),
        lambda: torch.load(checkpoint, map_location="cpu"),
        lambda: safetensors.torch.load_file(checkpoint, device="cpu"),
    ):
        with pytest.raises(RuntimeError, match="checkpoint path"):
            operation()


def test_reference_and_submodule_reads_are_blocked_during_test_execution() -> None:
    blocked = Path(__file__).resolve().parents[2] / "vendor" / "upstream" / "README.md"
    with pytest.raises(RuntimeError, match="submodule/reference path"):
        blocked.open(encoding="utf-8")


def test_open_guard_resolves_relative_paths_against_dir_fd(tmp_path: Path) -> None:
    harmless = tmp_path / "official"
    harmless.mkdir()
    parent_fd = os.open(tmp_path, os.O_RDONLY)
    try:
        harmless_fd = os.open("official", os.O_RDONLY, dir_fd=parent_fd)
        os.close(harmless_fd)
    finally:
        os.close(parent_fd)

    workspace = Path(__file__).resolve().parents[2]
    workspace_fd = os.open(workspace, os.O_RDONLY)
    try:
        with pytest.raises(RuntimeError, match="submodule/reference path"):
            os.open("official", os.O_RDONLY, dir_fd=workspace_fd)
    finally:
        os.close(workspace_fd)


def test_resource_accounting_includes_probe_children_without_double_counting_workers() -> None:
    evidence = aggregate_process_memory(
        {
            "process_id": "controller",
            "pid": 100,
            "role": "controller",
            "process_peak_rss_bytes": 100,
            # This contains the same workers below and must remain unaccounted.
            "waited_children_peak_rss_bytes": 10_000,
        },
        (
            {
                "process_id": "gw1",
                "pid": 102,
                "role": "worker",
                "process_peak_rss_bytes": 400,
                "waited_children_peak_rss_bytes": 500,
            },
            {
                "process_id": "gw0",
                "pid": 101,
                "role": "worker",
                "process_peak_rss_bytes": 200,
                "waited_children_peak_rss_bytes": 300,
            },
        ),
    )

    assert evidence["aggregate_peak_rss_bytes"] == 1_500
    assert evidence["temporal_upper_bound_rss_bytes"] == 1_500
    assert evidence["accounting_mode"] == "xdist-conservative-process-tree-upper-bound"
    assert evidence["budget_enforced"] is False
    assert [record["process_id"] for record in evidence["workers"]] == ["gw0", "gw1"]
