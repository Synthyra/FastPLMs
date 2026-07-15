"""Contracts for the non-canonical Python package-support matrix."""

from __future__ import annotations

import tomllib
from pathlib import Path

from tools.remote.python_matrix import (
    CANONICAL_GPU_PYTHON,
    OFFLINE_SMOKE_ENVIRONMENT,
    PYTHON_SUPPORT_VERSIONS,
    UV_SYNC_ARGUMENTS,
    build_smoke_environment,
    build_sync_command,
)
from tools.remote.run import SUITES

ROOT = Path(__file__).resolve().parents[2]


def test_python_support_metadata_matches_the_executed_matrix() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    classifiers = set(project["classifiers"])

    assert project["requires-python"] == ">=3.11,<3.15"
    assert CANONICAL_GPU_PYTHON == "3.12"
    assert PYTHON_SUPPORT_VERSIONS == ("3.11", "3.13", "3.14")
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.12.3"
    assert {
        f"Programming Language :: Python :: {python}" for python in ("3.11", "3.12", "3.13", "3.14")
    }.issubset(classifiers)


def test_matrix_sync_is_frozen_core_only_and_non_editable() -> None:
    command = build_sync_command("uv", Path("/workspace"), "3.13")

    assert UV_SYNC_ARGUMENTS == ("sync", "--frozen", "--no-dev", "--no-editable")
    assert command == (
        "uv",
        "sync",
        "--frozen",
        "--no-dev",
        "--no-editable",
        "--project",
        "/workspace",
        "--python",
        "3.13",
    )
    assert "--extra" not in command


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
    assert environment["PYTHONPATH"] == ""
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert set(OFFLINE_SMOKE_ENVIRONMENT).issubset(environment)


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
