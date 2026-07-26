"""Contracts for the non-canonical Python source-support matrix."""

from __future__ import annotations

from pathlib import Path

from tools.remote.python_matrix import (
    CANONICAL_GPU_PYTHON,
    OFFLINE_SMOKE_ENVIRONMENT,
    PYTHON_SUPPORT_VERSIONS,
    build_dependency_install_command,
    build_smoke_environment,
)
from tools.remote.run import SUITES


ROOT = Path(__file__).resolve().parents[2]


def test_python_support_versions_match_the_executed_matrix() -> None:
    assert CANONICAL_GPU_PYTHON == "3.12"
    assert PYTHON_SUPPORT_VERSIONS == ("3.11", "3.13", "3.14")
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.12.3"


def test_matrix_installs_declared_source_dependencies_with_cpu_constraints() -> None:
    command = build_dependency_install_command(
        "uv",
        Path("/environment/bin/python"),
        ROOT,
    )

    assert command == (
        "uv",
        "pip",
        "install",
        "--python",
        "/environment/bin/python",
        "--torch-backend=cpu",
        "-r",
        str(ROOT / "requirements/profiles/runtime.in"),
        "-c",
        str(ROOT / "requirements/constraints/validation.txt"),
    )


def test_matrix_smoke_is_offline_cpu_only_and_source_isolated() -> None:
    environment = build_smoke_environment(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HUB_OFFLINE": "0",
            "PYTHONPATH": "/workspace/src",
        }
    )

    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["HF_DATASETS_OFFLINE"] == "1"
    assert environment["PYTHONPATH"] == ""
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["UV_TORCH_BACKEND"] == "cpu"
    assert set(OFFLINE_SMOKE_ENVIRONMENT).issubset(environment)


def test_remote_matrix_runs_source_members_in_parallel_without_raw_logs() -> None:
    source = (ROOT / "tools/remote/python_matrix.py").read_text(encoding="utf-8")

    assert "ThreadPoolExecutor" in source
    assert "max_workers=min(4, len(versions))" in source
    assert "_output_fingerprint" in source
    assert '"stdout": _output_fingerprint' in source
    assert '"stderr": _output_fingerprint' in source
    assert 'stage = "dependency-install"' in source
    assert 'stage = "offline-cpu-source-smoke"' in source
    assert 'str(python),\n                "-I",' in source
    assert "uv build" not in source
    assert ".whl" not in source


def test_remote_matrix_reuses_the_single_candidate_image() -> None:
    suite = SUITES["python-matrix"]

    assert suite.bake_targets == ("candidate",)
    assert suite.pre_commands == ()
    assert suite.command == (
        "sudo",
        "docker",
        "compose",
        "-f",
        "docker/compose.yaml",
        "run",
        "--rm",
        "candidate",
        "python",
        "-m",
        "tools.remote.python_matrix",
        "--output",
        "artifacts/python-matrix.json",
        "--junit-output",
        "artifacts/junit/python-matrix.xml",
    )


def test_python_matrix_is_documented() -> None:
    remote = (ROOT / "tools/remote/README.md").read_text(encoding="utf-8")
    testing = (ROOT / "docs/testing.md").read_text(encoding="utf-8")

    for document in (remote, testing):
        assert "python-matrix" in document
        assert "3.11" in document
        assert "3.13" in document
        assert "3.14" in document
        assert "Python 3.12" in document
