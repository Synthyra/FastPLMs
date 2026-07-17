"""Portable remote-runner policy tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.remote.run import (
    REMOTE_CLEANUP_SCRIPT,
    SENSITIVE_SUFFIXES,
    SUITES,
    _is_sensitive,
    _require_clean_repository,
    _run_report,
    remote_cleanup_command,
)


def test_required_remote_suites_are_available() -> None:
    assert {"check", "compliance", "structure", "feature", "artifact", "benchmark"}.issubset(SUITES)


def test_sensitive_files_are_never_archived() -> None:
    assert _is_sensitive(Path(".secrets.env"))
    assert _is_sensitive(Path("credentials.json"))
    for directory in (".agents", ".claude", ".codex"):
        assert _is_sensitive(Path(directory) / "workspace-state.json")
    for suffix in SENSITIVE_SUFFIXES:
        assert _is_sensitive(Path(f"identity{suffix}"))
    assert _is_sensitive(Path("vendor/upstream/example/.git/config"))


def test_connection_details_are_runtime_only() -> None:
    source = (Path(__file__).resolve().parents[2] / "tools" / "remote" / "run.py").read_text(
        encoding="utf-8"
    )
    assert 'parser.add_argument("--host", required=True' in source
    assert 'parser.add_argument("--identity", required=True' in source
    assert ".ssh/" not in source


def test_recursive_cleanup_is_verified_by_remote_realpath() -> None:
    command = remote_cleanup_command(
        "/home/ubuntu/fastplms-runs",
        "/home/ubuntu/fastplms-runs/20260714T120000Z-1234abcd",
    )
    assert command[:2] == ("sh", "-c")
    assert command[2] == REMOTE_CLEANUP_SCRIPT
    assert 'base=$(realpath -e -- "$1")' in REMOTE_CLEANUP_SCRIPT
    assert 'workspace=$(realpath -e -- "$2")' in REMOTE_CLEANUP_SCRIPT
    assert '"$base"/*' in REMOTE_CLEANUP_SCRIPT
    assert 'test "$workspace" != "$base"' in REMOTE_CLEANUP_SCRIPT
    assert 'rm -rf -- "$workspace"' in REMOTE_CLEANUP_SCRIPT


def test_remote_runner_retrieves_the_complete_artifact_tree() -> None:
    source = (Path(__file__).resolve().parents[2] / "tools" / "remote" / "run.py").read_text(
        encoding="utf-8"
    )
    assert "remote_workspace}/artifacts/." in source


def test_remote_runner_rejects_a_dirty_exact_head(monkeypatch, tmp_path: Path) -> None:
    def dirty_status(*args, **kwargs):
        del args, kwargs
        return subprocess.CompletedProcess([], 0, stdout="?? scratch.py\n")

    monkeypatch.setattr(subprocess, "run", dirty_status)

    with pytest.raises(RuntimeError, match="clean Git worktree"):
        _require_clean_repository(tmp_path)


def test_remote_run_report_is_machine_readable_and_secret_free() -> None:
    suite = SUITES["unit"]
    report = _run_report(
        run_id="20260714T120000Z-1234abcd",
        suite_name="unit",
        suite=suite,
        started_at="2026-07-14T12:00:00+00:00",
        finished_at="2026-07-14T12:01:00+00:00",
        source_archive_sha256="a" * 64,
        git_revision="b" * 40,
        submodule_revisions={"vendor/upstream/example": "c" * 40},
        execution_environment={
            "host_kernel": "Linux 6.8.0 x86_64",
            "docker_server": {"Version": "28.0.0"},
            "gpus": ["NVIDIA H100, 580.1"],
            "images": {"candidate": {"id": "sha256:" + "d" * 64}},
        },
        failure_phase=None,
        failure=None,
        artifact_retrieval_returncode=0,
        cleanup_status="succeeded",
    )
    assert report["status"] == "passed"
    assert report["schema_version"] == 2
    assert report["git_revision"] == "b" * 40
    assert report["submodule_revisions"] == {"vendor/upstream/example": "c" * 40}
    environment = report["execution_environment"]
    assert isinstance(environment, dict)
    assert environment["images"]["candidate"]["id"] == "sha256:" + "d" * 64
    assert report["suite_contract"] == {
        "bake_targets": list(suite.bake_targets),
        "pre_commands": [list(command) for command in suite.pre_commands],
        "command": list(suite.command),
    }
    serialized = str(report).lower()
    assert "identity" not in serialized
    assert "private-key" not in serialized
    assert "gpu-host" not in serialized


def test_remote_run_report_records_failure_without_exception_text() -> None:
    failure = subprocess.CalledProcessError(
        7,
        ["ssh", "-i", "/secret/private-key", "user@gpu-host"],
    )
    report = _run_report(
        run_id="20260714T120000Z-1234abcd",
        suite_name="unit",
        suite=SUITES["unit"],
        started_at="2026-07-14T12:00:00+00:00",
        finished_at="2026-07-14T12:01:00+00:00",
        source_archive_sha256="a" * 64,
        git_revision="b" * 40,
        submodule_revisions={},
        execution_environment=None,
        failure_phase="suite",
        failure=failure,
        artifact_retrieval_returncode=0,
        cleanup_status="succeeded",
    )
    assert report["failure"] == {
        "phase": "suite",
        "type": "CalledProcessError",
        "returncode": 7,
    }
    serialized = str(report)
    assert "/secret/private-key" not in serialized
    assert "user@gpu-host" not in serialized


def test_compliance_runs_native_services_before_candidate_comparison() -> None:
    suite = SUITES["compliance"]
    commands = "\n".join(" ".join(command) for command in suite.pre_commands)
    assert "tools.artifacts.build_all" in commands
    assert "tools.remote.prepare_references" in commands
    for service in (
        "reference-esm2",
        "reference-biohub-esm",
        "reference-e1",
        "reference-dplm",
        "reference-ankh",
    ):
        assert service in suite.bake_targets
        assert service in commands
    assert "references" not in suite.bake_targets
    for service in ("reference-esmfold", "reference-esmfold2"):
        assert service in suite.bake_targets
        assert service in commands
    assert "reference-boltz2" not in suite.bake_targets
    assert "tests.structure.support.boltz2_bundle" not in commands
    command = " ".join(suite.command)
    assert " fp8 " in f" {command} "
    assert "tests/parity/test_native_results.py" in command
    fp8_stack_test = (
        "tests/release/test_validation_stack.py::"
        "test_fp8_validation_stack_uses_the_cuda13_transformer_engine_core"
    )
    assert fp8_stack_test in command
    assert f"--deselect={fp8_stack_test}" not in command
    assert "tests/structure/test_boltz2_folding_compliance.py" not in command
    assert "tests/structure/test_esmfold_folding_compliance.py" in command
    assert "tests/structure/test_esmfold2_folding_compliance.py" in command
    assert "tests/structure/test_esmfold2_fp8_compliance.py" in command


def test_check_runs_artifacts_and_manifest_selected_live_representatives() -> None:
    suite = SUITES["check"]
    commands = [" ".join(command) for command in suite.pre_commands]
    joined = "\n".join(commands)
    assert "candidate-artifact" in suite.bake_targets
    assert "candidate-structure" in suite.bake_targets
    assert "tools.artifacts.build_all" in joined
    assert "kernels download /workspace" in joined
    assert joined.index("tools.artifacts.build_all") < joined.index("kernels download /workspace")
    assert joined.index("kernels download /workspace") < joined.index(
        "tests/release/test_published_automodel.py"
    )
    assert "tests/release/test_published_automodel.py" in joined
    assert "tests/release/test_manifest_readiness.py" in joined
    assert "tools.remote.prepare_references" in joined
    for service in (
        "reference-esm2",
        "reference-biohub-esm",
        "reference-e1",
        "reference-dplm",
        "reference-ankh",
    ):
        assert service in suite.bake_targets
        matching = [command for command in commands if service in command]
        assert matching
        assert all("--deep-only" in command for command in matching)
    assert "test_native_representatives_all_backends" in joined
    command = " ".join(suite.command)
    assert " structure " in f" {command} "
    assert "tests/unit" in command
    assert "tests/integration" in command
    assert "tests/release" in command


def test_unit_suite_uses_the_structure_dependency_superset() -> None:
    suite = SUITES["unit"]
    assert suite.bake_targets == ("candidate-structure",)
    assert " structure " in f" {' '.join(suite.command)} "


def test_artifact_suite_builds_every_artifact_before_offline_probe() -> None:
    suite = SUITES["artifact"]
    assert "candidate" in suite.bake_targets
    assert "candidate-artifact" in suite.bake_targets
    assert any("tools.artifacts.build_all" in command for command in suite.pre_commands)
    assert any(
        "kernels download /workspace" in " ".join(command)
        for command in suite.pre_commands
    )


def test_release_suite_aggregates_exact_head_artifact_reference_and_gpu_gates() -> None:
    suite = SUITES["release"]
    commands = [" ".join(command) for command in suite.pre_commands]
    joined = "\n".join(commands)
    for target in (
        "candidate",
        "candidate-structure",
        "candidate-artifact",
        "reference-esmfold",
        "reference-esmfold2",
    ):
        assert target in suite.bake_targets
    assert joined.index("tools.artifacts.build_all") < joined.index("kernels download /workspace")
    assert joined.index("kernels download /workspace") < joined.index(
        "tests/release/test_published_automodel.py"
    )
    assert "tests.parity.support.native_reference" in joined
    assert "tests.structure.support.esmfold2_bundle" in joined
    assert "--precision bf16" in joined
    assert "--precision fp8" not in joined
    assert "tests.structure.support.boltz2_bundle" not in joined
    assert "tools.remote.python_matrix" in joined
    command = " ".join(suite.command)
    assert " structure " in f" {command} "
    assert "candidate-fp8" not in suite.bake_targets
    assert "reference-boltz2" not in suite.bake_targets
    assert "--ignore=tests/structure/test_structure_models.py" in command
    assert "--ignore=tests/structure/test_esmfold2_fp8_compliance.py" in command
    fp8_stack_test = (
        "tests/release/test_validation_stack.py::"
        "test_fp8_validation_stack_uses_the_cuda13_transformer_engine_core"
    )
    assert f"--deselect={fp8_stack_test}" in command
    assert "test_boltz2_live_folding_matches_pinned_official" in command
    assert "test_esmfold2_isolated_bf16_and_fp8_folding_compliance" not in command
    for path in ("tests/unit", "tests/integration", "tests/release", "tests/structure"):
        assert path in command
    assert "tests/parity/test_native_results.py" in command
    for path in (
        "tests/parity/test_esmfold2_common_parity.py",
        "tests/parity/test_esmfold2_protein_data_parity.py",
        "tests/parity/test_esmfold2_reimplemented_source_parity.py",
        "tests/parity/test_esmfold2_residue_config_parity.py",
        "tests/parity/test_esmfold2_source_slice3_parity.py",
        "tests/parity/test_esmfold2_source_slice4_parity.py",
    ):
        assert path in command
    assert "tests/parity/test_model_parity.py" not in command
    assert "tests/parity/test_ankh_seq2seq_parity.py" not in command
    assert "tests/parity/test_e1_source_independence_parity.py" not in command
    assert "not artifact" in command


def test_feature_suite_does_not_install_or_select_fp8() -> None:
    suite = SUITES["feature"]
    assert suite.bake_targets == ("candidate-structure",)
    command = " ".join(suite.command)
    assert " structure " in f" {command} "
    assert " fp8 " not in f" {command} "
    assert "tests/integration/test_dplm_generation.py" in command
    assert "tests/integration/test_esm3.py" in command
    assert "tests/release/test_conversion_tools.py" in command


def test_integration_suite_uses_the_structure_dependency_image() -> None:
    suite = SUITES["integration"]
    assert suite.bake_targets == ("candidate-structure",)
    command = " ".join(suite.command)
    assert " structure " in f" {command} "
    assert "tests/integration" in command


def test_structure_suite_produces_isolated_folding_bundles_before_gating() -> None:
    suite = SUITES["structure"]
    commands = [" ".join(command) for command in suite.pre_commands]
    joined = "\n".join(commands)
    assert "reference-boltz2" in suite.bake_targets
    assert "reference-esmfold" in suite.bake_targets
    assert "reference-esmfold2" in suite.bake_targets
    assert "tests.structure.support.boltz2_bundle prepare" in joined
    assert "reference-boltz2 python -m tests.structure.support.boltz2_bundle" in joined
    assert "tests.structure.support.boltz2_bundle produce-reference" in joined
    assert "tests.structure.support.boltz2_bundle produce-candidate" in joined
    assert "tests.structure.support.esmfold_bundle prepare" in joined
    assert "reference-esmfold python -m tests.structure.support.esmfold_bundle" in joined
    assert "produce-reference --exchange-root /exchange" in joined
    assert "produce-candidate --exchange-root /workspace/artifacts/reference" in joined
    assert "tests.structure.support.esmfold2_bundle prepare" in joined
    suite_command = " ".join(suite.command)
    assert "tests/structure" in suite_command
    assert "tests/parity/test_boltz_source_refactor.py" in suite_command
    assert "--ignore=tests/structure/test_structure_models.py" in suite_command
    assert "-m structure" in suite_command
    assert "structure and gpu" not in suite_command
    esmfold2_commands = [
        command for command in commands if "tests.structure.support.esmfold2_bundle" in command
    ]
    assert any(
        "reference-esmfold2" in command and "produce-reference" in command and "--all" in command
        for command in esmfold2_commands
    )
    assert any(
        "produce-candidate" in command and "--all" in command and "--precision bf16" in command
        for command in esmfold2_commands
    )
    assert any(
        "produce-candidate" in command and "--all" in command and "--precision fp8" in command
        for command in esmfold2_commands
    )
    runner_source = (Path(__file__).resolve().parents[2] / "tools" / "remote" / "run.py").read_text(
        encoding="utf-8"
    )
    assert "_ESMFOLD2_MODEL_IDS" not in runner_source
