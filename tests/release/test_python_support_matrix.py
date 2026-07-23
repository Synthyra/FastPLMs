"""Contracts for the non-canonical Python package-support matrix."""

from __future__ import annotations

import tomllib
from pathlib import Path

from tools.remote.python_matrix import (
    CANONICAL_GPU_PYTHON,
    LOCKED_RUNTIME_REQUIREMENTS,
    OFFLINE_SMOKE_ENVIRONMENT,
    PYTHON_SUPPORT_VERSIONS,
    build_smoke_environment,
    build_wheel_install_command,
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


def test_matrix_installs_one_built_wheel_with_the_locked_cpu_runtime() -> None:
    command = build_wheel_install_command(
        "uv",
        Path("/environment/bin/python"),
        Path("/wheel/fastplms-1.0.0-py3-none-any.whl"),
    )

    assert LOCKED_RUNTIME_REQUIREMENTS == ("torch==2.13.0", "transformers==5.13.0")
    assert command == (
        "uv",
        "pip",
        "install",
        "--python",
        "/environment/bin/python",
        "--torch-backend=cpu",
        "torch==2.13.0",
        "transformers==5.13.0",
        "/wheel/fastplms-1.0.0-py3-none-any.whl",
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


def test_remote_matrix_runs_members_in_parallel_without_raw_subprocess_logs() -> None:
    source = (ROOT / "tools/remote/python_matrix.py").read_text(encoding="utf-8")

    assert "ThreadPoolExecutor" in source
    assert "max_workers=min(4, len(versions))" in source
    assert "_output_fingerprint" in source
    assert '"stdout": _output_fingerprint' in source
    assert '"stderr": _output_fingerprint' in source
    assert 'shared_stage = "wheel-build"' in source
    assert 'shared_stage = "wheel-inventory"' in source
    assert 'str(python),\n                "-I",' in source


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


def test_every_pr_workflow_parallelizes_required_cpu_and_package_gates() -> None:
    workflow = (ROOT / ".github/workflows/cpu-contracts.yml").read_text(encoding="utf-8")

    jobs = (
        "cpu-contracts:",
        "static:",
        "runtime-import-closure:",
        "docs-licenses:",
        "distributions:",
        "extras-resolution:",
        "wheel-smoke:",
    )
    for job in jobs:
        assert job in workflow
    assert 'python: ["3.11", "3.12", "3.13", "3.14"]' in workflow
    assert workflow.count("persist-credentials: false") == len(jobs)
    assert workflow.count("submodules: false") == len(jobs)
    assert "tests/cpu -m cpu_contract -n auto" in workflow
    assert '"torch==2.13.0" "transformers==5.13.0"' in workflow
    assert 'HF_HUB_OFFLINE: "1"' in workflow
    assert 'TRANSFORMERS_OFFLINE: "1"' in workflow
    assert 'PYTHONNOUSERSITE: "1"' in workflow
    assert "mypy --python-version 3.12" in workflow
    assert '"mypy==1.20.2" "ruff==0.15.21"' in workflow
    assert "tools.artifacts.generate_docs --check" in workflow
    assert "test_model_card_licenses.py" in workflow
    assert "tools.remote.distribution_inspect" in workflow
    assert '--source-root "$GITHUB_WORKSPACE/src"' in workflow
    assert "extras-resolution (${{ matrix.extra }}, 3.12)" in workflow
    assert "uv pip compile pyproject.toml" in workflow
    assert "--python-platform x86_64-unknown-linux-gnu" in workflow
    assert '--torch-backend="${{ matrix.torch_backend }}"' in workflow
    for extra in ("cpu", "structure", "binder", "cueq", "reporting", "flash", "fp8", "train"):
        assert f"- extra: {extra}" in workflow
    assert "cuequivariance-ops-torch-cu13" in workflow
    assert "transformer-engine-cu13" in workflow
    assert "torch_backend: cu130" in workflow
    resolution_job = workflow.split("  extras-resolution:", maxsplit=1)[1].split(
        "  wheel-smoke:", maxsplit=1
    )[0]
    assert "--group validation" in resolution_job
    assert "uv pip install" not in resolution_job
    assert "import " not in resolution_job
    assert "kernels download" not in workflow
    assert "import transformer_engine" not in workflow
    assert "tools/remote/python_support_smoke.py" in workflow
    distributions_job = workflow.split("  distributions:", maxsplit=1)[1].split(
        "  extras-resolution:", maxsplit=1
    )[0]
    assert "uv venv --python 3.12 .sdist-smoke-venv" in distributions_job
    assert "dist/*.tar.gz" in distributions_job
    assert "artifacts/sdist-smoke-3.12.json" in distributions_job
    assert "docker" not in workflow.lower()


def test_cpu_contract_install_is_frozen_to_the_checked_lock() -> None:
    workflow = (ROOT / ".github/workflows/cpu-contracts.yml").read_text(encoding="utf-8")
    cpu_job = workflow.split("  cpu-contracts:", maxsplit=1)[1].split("  static:", maxsplit=1)[0]

    assert "uv lock --check" in cpu_job
    assert "uv sync --frozen --python 3.12 --no-default-groups" in cpu_job
    assert "--group validation" in cpu_job
    for extra in ("cpu", "dev", "structure", "train"):
        assert f"--extra {extra}" in cpu_job
    assert "uv pip install" not in cpu_job
    assert cpu_job.index("uv lock --check") < cpu_job.index("uv sync --frozen")

    testing = (ROOT / "docs/testing.md").read_text(encoding="utf-8")
    assert "uv lock --check" in testing
    assert "uv sync --frozen" in testing
    assert "PyTorch CPU index" in testing
    assert "`UV_TORCH_BACKEND` does not affect" in testing


def test_hopper_workflow_represents_real_validation_tiers_and_fails_unconfigured() -> None:
    workflow = (ROOT / ".github/workflows/h100-validation.yml").read_text(encoding="utf-8")

    assert "schedule:" in workflow
    for tier in ("golden-smoke", "nightly", "release-candidate", "benchmark-capture"):
        assert tier in workflow
    for suite in (
        "gpu-golden-smoke",
        "nightly",
        "compliance",
        "artifact",
        "structure",
        "benchmark",
    ):
        assert suite in workflow
    assert "FASTPLMS_H100_KNOWN_HOSTS" in workflow
    assert "exact GH200/aarch64 validation host" in workflow
    assert "never reports an unconfigured Hopper/SM90 tier as green" in workflow
    assert "Legacy secret names remain stable" in workflow
    assert "'h100-validation' || 'h100-nightly'" in workflow
    assert "revision:" in workflow
    assert "^[0-9a-f]{40}$" in workflow
    assert '"$REQUESTED_REVISION" != "$WORKFLOW_REVISION"' in workflow
    assert "ACTUAL_REVISION=$(git rev-parse HEAD)" in workflow
    assert '"$ACTUAL_REVISION" != "$EXPECTED_REVISION"' in workflow
    assert "ref: ${{ github.event_name == 'workflow_dispatch'" in workflow
    assert "max-parallel: 1" in workflow
    assert "group: fastplms-h100-validation" in workflow
    assert "submodules: false" in workflow
    assert "persist-credentials: false" in workflow
    assert "git submodule update --init --recursive" in workflow
    for suite in (
        "compliance",
        "structure",
        "artifact",
        "benchmark",
        "benchmark-capture",
        "nightly",
    ):
        assert f"matrix.suite == '{suite}'" in workflow
    assert "--accept-new-host-key" not in workflow
    assert "pull_request:" not in workflow
    assert "pull_request_target:" not in workflow
