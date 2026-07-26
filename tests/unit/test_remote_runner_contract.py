"""Portable remote-runner policy tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
import pytest
from pathlib import Path

from tools.remote.run import (
    REMOTE_CLEANUP_SCRIPT,
    SENSITIVE_SUFFIXES,
    SUITES,
    RemoteRunner,
    RunnerConfig,
    _artifact_tree_summary,
    _host_hardware_preflight,
    _is_sensitive,
    _kernel_capability_preflight,
    _reference_container_image_identity,
    _require_clean_repository,
    _require_matching_archive_digest,
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


def test_uploaded_source_archive_digest_is_verified_against_local_bytes() -> None:
    expected = hashlib.sha256(b"archive").hexdigest()
    _require_matching_archive_digest(f"{expected}  source.tar.gz\n", expected)

    with pytest.raises(RuntimeError, match="differs from local bytes"):
        _require_matching_archive_digest(f"{'0' * 64}  source.tar.gz\n", expected)
    with pytest.raises(RuntimeError, match="differs from local bytes"):
        _require_matching_archive_digest("", expected)


def test_remote_archive_inventory_is_tracked_only() -> None:
    source = (Path(__file__).resolve().parents[2] / "tools" / "remote" / "run.py").read_text(
        encoding="utf-8"
    )
    git_files = source.split("def _git_files", maxsplit=1)[1].split(
        "def _require_clean_repository", maxsplit=1
    )[0]
    assert '"--cached"' in git_files
    assert '"--others"' not in git_files
    assert '"--exclude-standard"' not in git_files


def test_connection_details_are_runtime_only() -> None:
    source = (Path(__file__).resolve().parents[2] / "tools" / "remote" / "run.py").read_text(
        encoding="utf-8"
    )
    assert 'parser.add_argument("--host", required=True' in source
    assert 'parser.add_argument("--identity", required=True' in source
    assert ".ssh/" not in source
    assert source.count('"IdentitiesOnly=yes"') == 2
    assert "secrets.token_hex(8)" in source


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


@pytest.mark.parametrize(
    ("failure_phase", "failing_command"),
    (
        ("create-remote-workspace", "mkdir"),
        ("upload-source-archive", "scp"),
        ("verify-source-archive", "sha256sum"),
        ("extract-source-archive", "tar"),
        ("remove-source-archive", "rm"),
    ),
)
def test_remote_staging_failures_are_reported_and_cleaned(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_phase: str,
    failing_command: str,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    identity = tmp_path / "identity"
    identity.write_text("test", encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    revision = "a" * 40

    monkeypatch.setattr("tools.remote.run._require_clean_repository", lambda _repository: None)
    monkeypatch.setattr("tools.remote.run._git_head_revision", lambda _repository: revision)

    def write_archive(_repository: Path, destination: Path) -> dict[str, dict[str, object]]:
        destination.write_bytes(b"archive")
        return {}

    monkeypatch.setattr("tools.remote.run.create_source_archive", write_archive)

    runner = RemoteRunner(
        RunnerConfig(
            host="gpu-host",
            identity=identity,
            repository=repository,
            suite="unit",
            artifacts=artifacts,
        )
    )
    monkeypatch.setattr(
        runner,
        "_capture_host_hardware",
        lambda: _host_hardware_preflight(
            "aarch64",
            "NVIDIA GH200 480GB, GPU-test, 580.1, 97871\n",
        ),
    )
    monkeypatch.setattr(runner, "_remote_base", lambda: "/remote/fastplms-runs")
    ssh_commands: list[tuple[str, ...]] = []

    def run_ssh(command, *, capture=False, timeout_seconds=None):
        del capture, timeout_seconds
        value = tuple(command)
        ssh_commands.append(value)
        if failing_command in {"mkdir", "sha256sum", "tar", "rm"} and value[0] == failing_command:
            raise subprocess.CalledProcessError(7, value)
        stdout = ""
        if value[0] == "sha256sum":
            stdout = hashlib.sha256(b"archive").hexdigest() + "  source.tar.gz\n"
        return subprocess.CompletedProcess(value, 0, stdout=stdout)

    monkeypatch.setattr(runner, "_ssh", run_ssh)

    def run_local(command, *, check, **kwargs):
        del kwargs
        value = tuple(command)
        if "-r" in value:
            assert check is False
            return subprocess.CompletedProcess(value, 1)
        if failing_command == "scp":
            assert check is True
            raise subprocess.CalledProcessError(6, value)
        return subprocess.CompletedProcess(value, 0)

    monkeypatch.setattr(subprocess, "run", run_local)

    with pytest.raises(subprocess.CalledProcessError):
        runner.run()

    cleanup = remote_cleanup_command(
        "/remote/fastplms-runs",
        f"/remote/fastplms-runs/{runner.run_id}",
    )
    assert cleanup in ssh_commands
    report_path = artifacts / runner.run_id / "remote-run.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["failure"]["phase"] == failure_phase
    assert report["remote_cleanup"] == "succeeded"


def test_remote_runner_rejects_a_dirty_exact_head(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
    assert report["schema_version"] == 5
    assert report["git_revision"] == "b" * 40
    assert report["submodule_revisions"] == {"vendor/upstream/example": "c" * 40}
    environment = report["execution_environment"]
    assert isinstance(environment, dict)
    assert environment["images"]["candidate"]["id"] == "sha256:" + "d" * 64
    assert report["suite_contract"] == {
        "bake_targets": list(suite.bake_targets),
        "pre_commands": [list(command) for command in suite.pre_commands],
        "command": list(suite.command),
        "required_paths": list(suite.required_paths),
        "biohub_reference_targets": [],
        "reference_targets": [],
        "host_hardware_binding_required": True,
        "attention_backends": [],
        "kernel_downloads_allowed": False,
        "same_host_candidate_reference_required": False,
        "timeouts_seconds": {
            "control": 300,
            "transfer": 1_800,
            "build": suite.build_timeout_seconds,
            "pre_command": suite.pre_command_timeout_seconds,
            "command": suite.command_timeout_seconds,
        },
    }
    assert report["phase_durations_seconds"] == {}
    assert report["cache_telemetry"] == {}
    assert report["artifact_inventory"] is None
    assert report["host_hardware_preflight"] is None
    assert report["kernel_capability_preflight"] is None
    serialized = str(report).lower()
    assert "identity" not in serialized
    assert "private-key" not in serialized
    assert "gpu-host" not in serialized


def test_retrieved_artifact_inventory_is_content_addressed_and_secret_free(
    tmp_path: Path,
) -> None:
    (tmp_path / "junit").mkdir()
    (tmp_path / "junit" / "report.xml").write_text("<testsuite/>", encoding="utf-8")
    first = _artifact_tree_summary(tmp_path)
    second = _artifact_tree_summary(tmp_path)

    assert first == second
    assert first["status"] == "captured"
    assert first["file_count"] == 1
    assert first["total_bytes"] == len("<testsuite/>")
    assert len(str(first["tree_sha256"])) == 64
    assert "report.xml" not in str(first)

    (tmp_path / "credentials.json").write_text("secret", encoding="utf-8")
    with pytest.raises(RuntimeError, match="sensitive path"):
        _artifact_tree_summary(tmp_path)


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
    assert "kernels download" not in commands
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
    assert "biohub-biotraj-wheel" in suite.bake_targets


def test_host_hardware_preflight_binds_exact_gh200_arm64_identity() -> None:
    preflight = _host_hardware_preflight(
        "aarch64\n",
        "NVIDIA GH200 480GB, GPU-1234, 580.1, 97871\n",
    )

    assert preflight["status"] == "passed"
    assert preflight["uname_machine"] == "aarch64"
    assert preflight["architecture"] == "arm64"
    assert preflight["container_platform"] == "linux/arm64"
    assert preflight["gpus"] == [
        {
            "name": "NVIDIA GH200 480GB",
            "uuid": "GPU-1234",
            "driver_version": "580.1",
            "memory_total_mib": 97871,
        }
    ]
    assert len(str(preflight["identity_sha256"])) == 64


def test_every_suite_accepts_the_bound_gh200_hardware_contract() -> None:
    preflight = _host_hardware_preflight(
        "aarch64",
        "NVIDIA GH200 480GB, GPU-1234, 580.1, 97871\n",
    )
    assert preflight["status"] == "passed"
    assert all(suite.bake_targets for suite in SUITES.values())
    for suite in SUITES.values():
        if {"reference-biohub-esm", "reference-esmfold2"}.intersection(
            suite.bake_targets
        ):
            assert "biohub-biotraj-wheel" in suite.bake_targets
    for suite_name in ("compliance", "structure", "release"):
        assert {
            "reference-biohub-esm",
            "reference-esmfold2",
        }.intersection(SUITES[suite_name].bake_targets)


def test_gh200_kernel_policy_is_explicit_no_download_and_fail_closed() -> None:
    hardware = _host_hardware_preflight(
        "aarch64",
        "NVIDIA GH200 480GB, GPU-1234, 580.1, 97871\n",
    )
    policy = _kernel_capability_preflight(
        hardware,
        ("eager", "sdpa", "flex_attention"),
    )

    assert policy["status"] == "passed"
    assert policy["network_downloads"] is False
    assert policy["source_builds"] is False
    assert policy["selected_backends"] == ["eager", "sdpa", "flex_attention"]
    backends = policy["backends"]
    assert backends["flash_attention_2"]["status"] == "prior_focused_evidence_only"
    assert backends["flash_attention_2"]["selected"] is False
    assert backends["flash_attention_3"]["status"] == "unavailable"
    assert backends["flash_attention_3"]["selected"] is False

    rejected = _kernel_capability_preflight(hardware, ("sdpa", "flash_attention_3"))
    assert rejected["status"] == "failed"
    assert "flash_attention_3" in str(rejected["reason"])


def test_reference_container_identity_is_stable_and_excludes_ephemeral_fields() -> None:
    digest = "sha256:" + "a" * 64
    identity = _reference_container_image_identity(
        {
            "container_platform": "linux/arm64",
            "docker_server": {
                "Version": "28.0.0",
                "ApiVersion": "1.48",
                "Arch": "arm64",
                "Os": "linux",
                "Name": "ephemeral-hostname",
            },
            "docker_buildx": "github.com/docker/buildx v0.25.0 deadbeef",
            "images": {
                "reference-biohub-esm": {
                    "tag": "local/ephemeral:tag",
                    "id": digest,
                    "content_digest": digest,
                    "created": "2026-07-22T00:00:00Z",
                    "os": "linux",
                    "architecture": "arm64",
                    "resolved_platform": "linux/arm64",
                }
            },
        }
    )

    assert identity == {
        "schema_version": 1,
        "resolved_platform": "linux/arm64",
        "docker_server": {
            "Version": "28.0.0",
            "ApiVersion": "1.48",
            "Os": "linux",
            "Arch": "arm64",
        },
        "docker_buildx": "github.com/docker/buildx v0.25.0 deadbeef",
        "images": {
            "reference-biohub-esm": {
                "content_digest": digest,
                "image_id": digest,
                "os": "linux",
                "architecture": "arm64",
                "resolved_platform": "linux/arm64",
            }
        },
    }
    assert "tag" not in str(identity).lower()
    assert "created" not in str(identity).lower()
    assert "ephemeral-hostname" not in str(identity)


def test_gh200_hardware_binding_happens_before_archive_or_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    identity = tmp_path / "identity"
    identity.write_text("test", encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    revision = "a" * 40
    archive_called = False
    hardware = _host_hardware_preflight(
        "aarch64",
        "NVIDIA GH200 480GB, GPU-test, 580.1, 97871\n",
    )

    monkeypatch.setattr("tools.remote.run._require_clean_repository", lambda _root: None)
    monkeypatch.setattr("tools.remote.run._git_head_revision", lambda _root: revision)

    def stop_after_preflight(_repository: Path, _destination: Path):
        nonlocal archive_called
        archive_called = True
        raise RuntimeError("stop after hardware preflight")

    monkeypatch.setattr("tools.remote.run.create_source_archive", stop_after_preflight)
    runner = RemoteRunner(
        RunnerConfig(
            host="gpu-host",
            identity=identity,
            repository=repository,
            suite="compliance",
            artifacts=artifacts,
        )
    )
    monkeypatch.setattr(runner, "_remote_base", lambda: "/remote/fastplms-runs")
    monkeypatch.setattr(runner, "_capture_host_hardware", lambda: hardware)

    with pytest.raises(RuntimeError, match="stop after hardware preflight"):
        runner.run()

    assert archive_called
    report = json.loads(
        (artifacts / runner.run_id / "remote-run.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "failed"
    assert report["failure"]["phase"] == "create-source-archive"
    assert report["host_hardware_preflight"] == hardware


def test_check_uses_candidate_goldens_without_artifacts_or_live_references() -> None:
    suite = SUITES["check"]
    assert suite.bake_targets == ("candidate-structure",)
    pre_commands = "\n".join(" ".join(command) for command in suite.pre_commands)
    command = " ".join(suite.command)
    assert "tests/unit" in command
    assert "tests/integration" in command
    assert "tests/release" in command
    assert "tools.artifacts.build_all" not in pre_commands
    assert "tests/release/test_published_automodel.py" not in pre_commands
    assert "kernels download" not in pre_commands
    assert "tests/integration/test_official_goldens.py" in pre_commands
    assert "tests/structure/test_structure_official_goldens.py" in pre_commands
    assert "reference-" not in pre_commands
    assert "tests.parity.support.native_reference" not in pre_commands
    assert suite.attention_backends == ("eager", "sdpa", "flex_attention")


def test_remote_builds_targets_together_and_enforces_remote_timeouts() -> None:
    source = (Path(__file__).resolve().parents[2] / "tools" / "remote" / "run.py").read_text(
        encoding="utf-8"
    )
    assert "*suite.bake_targets" in source
    assert 'f"*.platform={host_hardware_preflight[' in source
    assert 'f"*.platform={container_platform}"' in source
    assert "value.get(\"Architecture\") != expected_architecture" in source
    assert "Remote host hardware identity changed during the build" in source
    assert '"--kill-after=30s"' in source
    assert "timeout_seconds=suite.build_timeout_seconds" in source
    assert "timeout_seconds=suite.pre_command_timeout_seconds" in source
    assert "timeout_seconds=suite.command_timeout_seconds" in source
    assert "timeout=_TRANSFER_TIMEOUT_SECONDS" in source
    assert 'cache_telemetry["before_build"]' in source
    assert 'cache_telemetry["after_run"]' in source
    assert "kernels download" not in source
    assert "persist-reference-container-identity" in source
    assert "artifacts/reference/environment/container-images.json" in source


def test_gpu_golden_smoke_uses_checked_in_results_without_reference_images() -> None:
    suite = SUITES["gpu-golden-smoke"]
    assert suite.bake_targets == ("candidate-structure",)
    assert suite.pre_commands == ()
    command = " ".join(suite.command)
    assert "test_release_hopper_sm90_gpu_is_available_without_running_a_model" in command
    assert "tests/integration/test_official_goldens.py" in command
    assert "tests/structure/test_structure_official_goldens.py" in command
    assert "gpu and not large" in command
    assert "reference-" not in command


def test_nightly_uses_candidate_goldens_features_artifacts_fp8_and_throughput() -> None:
    suite = SUITES["nightly"]
    assert suite.bake_targets == (
        "candidate",
        "candidate-structure",
        "candidate-fp8",
        "candidate-artifact",
    )
    pre_commands = "\n".join(" ".join(command) for command in suite.pre_commands)
    command = " ".join(suite.command)
    assert "tools.artifacts.build_all" in pre_commands
    assert "tests/release/test_published_automodel.py" in pre_commands
    assert "test_esmfold2_fp8_compliance.py" in pre_commands
    assert "artifacts/benchmarks/nightly-h100.json" in pre_commands
    assert "--artifact-root dist/hub" in pre_commands
    assert pre_commands.index("tools.artifacts.build_all") < pre_commands.index(
        "artifacts/benchmarks/nightly-h100.json"
    )
    assert "test_official_goldens.py" in pre_commands
    assert "test_structure_official_goldens.py" in pre_commands
    assert "test_flash_attention_backends.py" not in command
    assert "kernels download" not in pre_commands
    assert "--backends eager sdpa flex_attention" in pre_commands
    assert "test_fine_tuning_example.py" in command
    assert "reference-" not in pre_commands


def test_benchmark_requires_a_tracked_baseline_and_capture_is_descriptive() -> None:
    gated = SUITES["benchmark"]
    capture = SUITES["benchmark-capture"]
    gated_command = " ".join(gated.command)
    capture_command = " ".join(capture.command)
    assert gated.required_paths == ("benchmarks/baselines/h100.json",)
    assert "--baseline benchmarks/baselines/h100.json" in gated_command
    assert "--baseline" not in capture_command
    assert "h100-baseline-candidate.json" in capture_command
    for suite, command in ((gated, gated_command), (capture, capture_command)):
        assert suite.bake_targets == ("candidate", "candidate-fp8")
        pre_commands = "\n".join(" ".join(item) for item in suite.pre_commands)
        assert "tools.artifacts.build_all" in pre_commands
        assert "--benchmark-suite" in pre_commands
        assert "kernels download" not in pre_commands
        assert "--artifact-root dist/hub" in command
        assert "--backends eager sdpa flex_attention" in command
        assert "--junit-output artifacts/junit/" in command
        assert suite.attention_backends == ("eager", "sdpa", "flex_attention")
        assert suite.pre_command_timeout_seconds >= 14_400


def test_live_release_and_benchmark_suites_enforce_hopper_sm90_hardware() -> None:
    compliance = " ".join(SUITES["compliance"].command)
    benchmark = " ".join(SUITES["benchmark"].command)

    assert "test_release_hopper_sm90_gpu_is_available_without_running_a_model" in compliance
    assert "benchmarks/baselines/h100.json" in benchmark  # Legacy compatibility filename.
    source = (Path(__file__).resolve().parents[2] / "benchmarks" / "suite.py").read_text(
        encoding="utf-8"
    )
    assert "validate_hopper_sm90_environment(environment)" in source
    assert '"validated_hopper_sm90_exact_device"' in source


def test_unit_suite_uses_the_structure_dependency_superset() -> None:
    suite = SUITES["unit"]
    assert suite.bake_targets == ("candidate-structure",)
    assert " structure " in f" {' '.join(suite.command)} "


def test_artifact_suite_builds_every_artifact_before_offline_probe() -> None:
    suite = SUITES["artifact"]
    assert "candidate" in suite.bake_targets
    assert "candidate-artifact" in suite.bake_targets
    assert any("tools.artifacts.build_all" in command for command in suite.pre_commands)
    assert not any("kernels download" in " ".join(command) for command in suite.pre_commands)
    assert "not test_local_artifact_locked_flash_backend" in " ".join(suite.command)


def test_release_suite_aggregates_exact_head_artifact_reference_and_gpu_gates() -> None:
    suite = SUITES["release"]
    commands = [" ".join(command) for command in suite.pre_commands]
    joined = "\n".join(commands)
    for target in (
        "candidate",
        "candidate-structure",
        "candidate-fp8",
        "candidate-artifact",
        "biohub-biotraj-wheel",
        "reference-esmfold",
        "reference-esmfold2",
    ):
        assert target in suite.bake_targets
    assert "kernels download" not in joined
    assert joined.index("tools.artifacts.build_all") < joined.index(
        "tests/release/test_published_automodel.py"
    )
    assert "tests.parity.support.native_reference" in joined
    assert "tests.structure.support.esmfold2_bundle" in joined
    assert "--precision bf16" in joined
    assert "--precision fp8" not in joined
    assert "tests.structure.support.boltz2_bundle" not in joined
    assert "tools.remote.python_matrix" in joined
    assert "artifacts/benchmarks/release-h100.json" in joined
    assert "--artifact-root dist/hub" in joined
    assert "--backends eager sdpa flex_attention" in joined
    assert "--junit-output artifacts/junit/release-benchmark.xml" in joined
    assert joined.index("tools.artifacts.build_all") < joined.index(
        "artifacts/benchmarks/release-h100.json"
    )
    command = " ".join(suite.command)
    assert " structure " in f" {command} "
    assert "reference-boltz2" not in suite.bake_targets
    assert "--ignore=tests/structure/test_structure_models.py" in command
    assert "--ignore=tests/structure/test_esmfold2_fp8_compliance.py" in command
    assert "--ignore=tests/integration/test_flash_attention_backends.py" in command
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
