"""Synchronize a clean workspace and run FastPLMs containers over SSH.

The runner accepts the SSH host and identity only at invocation time. It does
not read credential files, copy ignored files, or persist workstation details
in the repository.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from tools.source_provenance import (
    ARCHIVE_PROVENANCE_NAME,
    render_archive_provenance,
    tracked_tree_digest,
)

HOST_PATTERN = re.compile(r"^[A-Za-z0-9_.@:\-]+$")
RUN_PATTERN = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$")
GIT_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
REMOTE_CLEANUP_SCRIPT = """set -eu
base=$(realpath -e -- "$1")
workspace=$(realpath -e -- "$2")
case "$workspace" in
    "$base"/*) ;;
    *) echo "refusing cleanup outside managed remote base" >&2; exit 64 ;;
esac
test "$workspace" != "$base"
rm -rf -- "$workspace"
"""
SENSITIVE_NAMES = {
    ".agents",
    ".claude",
    ".codex",
    ".env",
    ".secrets.env",
    "credentials",
    "credentials.json",
    "id_rsa",
    "id_ed25519",
}
SENSITIVE_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}


@dataclass(frozen=True)
class Suite:
    """Images to build and the command executed in the remote workspace."""

    bake_targets: tuple[str, ...]
    command: tuple[str, ...]
    pre_commands: tuple[tuple[str, ...], ...] = ()


def _run_report(
    *,
    run_id: str,
    suite_name: str,
    suite: Suite,
    started_at: str,
    finished_at: str,
    source_archive_sha256: str,
    git_revision: str,
    submodule_revisions: Mapping[str, str],
    execution_environment: Mapping[str, object] | None,
    failure_phase: str | None,
    failure: BaseException | None,
    artifact_retrieval_returncode: int,
    cleanup_status: str,
) -> dict[str, object]:
    """Build a secret-free machine report for one remote invocation."""

    failure_record: dict[str, object] | None = None
    if failure is not None:
        failure_record = {"phase": failure_phase, "type": type(failure).__name__}
        if isinstance(failure, subprocess.CalledProcessError):
            failure_record["returncode"] = failure.returncode
    elif artifact_retrieval_returncode != 0:
        failure_record = {
            "phase": "artifact-retrieval",
            "type": "ArtifactRetrievalError",
            "returncode": artifact_retrieval_returncode,
        }
    passed = failure_record is None and cleanup_status in {"succeeded", "retained"}
    return {
        "schema_version": 2,
        "run_id": run_id,
        "suite": suite_name,
        "status": "passed" if passed else "failed",
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "source_archive_sha256": source_archive_sha256,
        "git_revision": git_revision,
        "submodule_revisions": dict(sorted(submodule_revisions.items())),
        "execution_environment": (
            dict(execution_environment) if execution_environment is not None else None
        ),
        "artifact_retrieval": {
            "returncode": artifact_retrieval_returncode,
            "status": ("succeeded" if artifact_retrieval_returncode == 0 else "failed"),
        },
        "remote_cleanup": cleanup_status,
        "failure": failure_record,
        "suite_contract": {
            "bake_targets": list(suite.bake_targets),
            "pre_commands": [list(command) for command in suite.pre_commands],
            "command": list(suite.command),
        },
    }


def _compose_run(service: str, *command: str) -> tuple[str, ...]:
    return (
        "sudo",
        "docker",
        "compose",
        "-f",
        "docker/compose.yaml",
        "run",
        "--rm",
        service,
        *command,
    )


_BUILD_ARTIFACTS = _compose_run(
    "candidate",
    "python",
    "-m",
    "tools.artifacts.build_all",
    "--output-root",
    "dist/hub",
    "--source-root",
    "/workspace",
)
_DOWNLOAD_KERNELS = _compose_run(
    "candidate",
    "kernels",
    "download",
    "/workspace",
)
_PREPARE_REFERENCES = _compose_run(
    "candidate",
    "python",
    "-m",
    "tools.remote.prepare_references",
    "--output-root",
    "artifacts/reference",
)
_SEQUENCE_REFERENCE_CONTAINERS = (
    "reference-esm2",
    "reference-biohub-esm",
    "reference-e1",
    "reference-dplm",
    "reference-ankh",
)
_RUN_NATIVE_REFERENCES = tuple(
    _compose_run(
        container,
        "python",
        "-m",
        "tests.parity.support.native_reference",
        "--request-dir",
        f"/exchange/requests/{container}",
        "--output-dir",
        "/exchange/results",
    )
    for container in _SEQUENCE_REFERENCE_CONTAINERS
)
_RUN_NATIVE_REPRESENTATIVES = tuple(
    _compose_run(
        container,
        "python",
        "-m",
        "tests.parity.support.native_reference",
        "--request-dir",
        f"/exchange/requests/{container}",
        "--output-dir",
        "/exchange/results",
        "--deep-only",
    )
    for container in _SEQUENCE_REFERENCE_CONTAINERS
)
_RUN_CHECK_ARTIFACTS = _compose_run(
    "artifact",
    "python",
    "-m",
    "pytest",
    "tests/release/test_published_automodel.py",
    "tests/release/test_manifest_readiness.py",
    "-m",
    "artifact",
    "--junitxml=artifacts/junit/check-artifact.xml",
)
_RUN_CHECK_NATIVE_ASSERTIONS = _compose_run(
    "candidate",
    "python",
    "-m",
    "pytest",
    "tests/parity/test_native_results.py::test_native_representatives_all_backends",
    "--junitxml=artifacts/junit/check-native.xml",
)
_RUN_PYTHON_MATRIX = _compose_run(
    "candidate",
    "python",
    "-m",
    "tools.remote.python_matrix",
    "--output",
    "artifacts/python-matrix.json",
    "--junit-output",
    "artifacts/junit/python-matrix.xml",
)
_PREPARE_BOLTZ2_BUNDLE = _compose_run(
    "structure",
    "python",
    "-m",
    "tests.structure.support.boltz2_bundle",
    "prepare",
    "--exchange-root",
    "/workspace/artifacts/reference",
)
_RUN_BOLTZ2_REFERENCE = _compose_run(
    "reference-boltz2",
    "python",
    "-m",
    "tests.structure.support.boltz2_bundle",
    "produce-reference",
    "--exchange-root",
    "/exchange",
)
_RUN_BOLTZ2_CANDIDATE = _compose_run(
    "structure",
    "python",
    "-m",
    "tests.structure.support.boltz2_bundle",
    "produce-candidate",
    "--exchange-root",
    "/workspace/artifacts/reference",
)
_PREPARE_ESMFOLD_BUNDLE = _compose_run(
    "structure",
    "python",
    "-m",
    "tests.structure.support.esmfold_bundle",
    "prepare",
    "--exchange-root",
    "/workspace/artifacts/reference",
)
_RUN_ESMFOLD_REFERENCES = tuple(
    _compose_run(
        "reference-esmfold",
        "python",
        "-m",
        "tests.structure.support.esmfold_bundle",
        "produce-reference",
        "--exchange-root",
        "/exchange",
        "--precision",
        precision,
    )
    for precision in ("fp32", "bf16")
)
_RUN_ESMFOLD_CANDIDATES = tuple(
    _compose_run(
        "structure",
        "python",
        "-m",
        "tests.structure.support.esmfold_bundle",
        "produce-candidate",
        "--exchange-root",
        "/workspace/artifacts/reference",
        "--precision",
        precision,
    )
    for precision in ("fp32", "bf16")
)
_PREPARE_ESMFOLD2_BUNDLES = _compose_run(
    "structure",
    "python",
    "-m",
    "tests.structure.support.esmfold2_bundle",
    "prepare",
    "--exchange-root",
    "/workspace/artifacts/reference",
)
_RUN_ESMFOLD2_REFERENCE = _compose_run(
    "reference-esmfold2",
    "python",
    "-m",
    "tests.structure.support.esmfold2_bundle",
    "produce-reference",
    "--exchange-root",
    "/exchange",
    "--all",
)
_RUN_ESMFOLD2_CANDIDATES = tuple(
    _compose_run(
        "fp8" if precision == "fp8" else "structure",
        "python",
        "-m",
        "tests.structure.support.esmfold2_bundle",
        "produce-candidate",
        "--exchange-root",
        "/workspace/artifacts/reference",
        "--all",
        "--precision",
        precision,
    )
    for precision in ("bf16", "fp8")
)
_RUN_STRUCTURE_REFERENCES = (
    _PREPARE_BOLTZ2_BUNDLE,
    _RUN_BOLTZ2_REFERENCE,
    _RUN_BOLTZ2_CANDIDATE,
    _PREPARE_ESMFOLD_BUNDLE,
    *_RUN_ESMFOLD_REFERENCES,
    *_RUN_ESMFOLD_CANDIDATES,
    _PREPARE_ESMFOLD2_BUNDLES,
    _RUN_ESMFOLD2_REFERENCE,
    *_RUN_ESMFOLD2_CANDIDATES,
)

_RUN_RELEASE_STRUCTURE_REFERENCES = (
    _PREPARE_ESMFOLD_BUNDLE,
    *_RUN_ESMFOLD_REFERENCES,
    *_RUN_ESMFOLD_CANDIDATES,
    _PREPARE_ESMFOLD2_BUNDLES,
    _RUN_ESMFOLD2_REFERENCE,
    _RUN_ESMFOLD2_CANDIDATES[0],
)

# These source-parity modules are self-contained in the candidate environment.
# Direct model parity plus ANKH and E1 parity remain in the isolated native
# reference workflow because their official dependencies conflict with it.
_RELEASE_LOCAL_PARITY_TESTS = (
    "tests/parity/test_esmfold2_common_parity.py",
    "tests/parity/test_esmfold2_protein_data_parity.py",
    "tests/parity/test_esmfold2_reimplemented_source_parity.py",
    "tests/parity/test_esmfold2_residue_config_parity.py",
    "tests/parity/test_esmfold2_source_slice3_parity.py",
    "tests/parity/test_esmfold2_source_slice4_parity.py",
)


SUITES = {
    "check": Suite(
        (
            "candidate",
            "candidate-structure",
            "candidate-artifact",
            *_SEQUENCE_REFERENCE_CONTAINERS,
        ),
        (
            "sudo",
            "docker",
            "compose",
            "-f",
            "docker/compose.yaml",
            "run",
            "--rm",
            "structure",
            "python",
            "-m",
            "pytest",
            "tests/unit",
            "tests/integration",
            "tests/release",
            "-m",
            "not gpu and not slow and not structure and not artifact",
            "--junitxml=artifacts/junit/check.xml",
        ),
        pre_commands=(
            _BUILD_ARTIFACTS,
            _DOWNLOAD_KERNELS,
            _RUN_CHECK_ARTIFACTS,
            _PREPARE_REFERENCES,
            *_RUN_NATIVE_REPRESENTATIVES,
            _RUN_CHECK_NATIVE_ASSERTIONS,
        ),
    ),
    "unit": Suite(
        ("candidate-structure",),
        (
            "sudo",
            "docker",
            "compose",
            "-f",
            "docker/compose.yaml",
            "run",
            "--rm",
            "structure",
            "python",
            "-m",
            "pytest",
            "tests/unit",
            "--junitxml=artifacts/junit/unit.xml",
        ),
    ),
    "integration": Suite(
        ("candidate-structure",),
        (
            "sudo",
            "docker",
            "compose",
            "-f",
            "docker/compose.yaml",
            "run",
            "--rm",
            "structure",
            "python",
            "-m",
            "pytest",
            "tests/integration",
            "--junitxml=artifacts/junit/integration.xml",
        ),
    ),
    "compliance": Suite(
        (
            "candidate",
            "candidate-structure",
            "candidate-fp8",
            "reference-esm2",
            "reference-biohub-esm",
            "reference-e1",
            "reference-dplm",
            "reference-ankh",
            "reference-esmfold",
            "reference-esmfold2",
        ),
        (
            "sudo",
            "docker",
            "compose",
            "-f",
            "docker/compose.yaml",
            "run",
            "--rm",
            "fp8",
            "python",
            "-m",
            "pytest",
            "tests/parity/test_native_results.py",
            (
                "tests/release/test_validation_stack.py::"
                "test_fp8_validation_stack_uses_the_cuda13_transformer_engine_core"
            ),
            "tests/structure/test_esmfold_folding_compliance.py",
            "tests/structure/test_esmfold2_folding_compliance.py",
            "tests/structure/test_esmfold2_fp8_compliance.py",
            "--junitxml=artifacts/junit/compliance.xml",
        ),
        pre_commands=(
            _BUILD_ARTIFACTS,
            _PREPARE_REFERENCES,
            *_RUN_NATIVE_REFERENCES,
            *_RUN_RELEASE_STRUCTURE_REFERENCES,
        ),
    ),
    "structure": Suite(
        (
            "candidate-structure",
            "candidate-fp8",
            "reference-boltz2",
            "reference-esmfold",
            "reference-esmfold2",
        ),
        (
            "sudo",
            "docker",
            "compose",
            "-f",
            "docker/compose.yaml",
            "run",
            "--rm",
            "structure",
            "python",
            "-m",
            "pytest",
            "tests/structure",
            "tests/parity/test_boltz_source_refactor.py",
            "--ignore=tests/structure/test_structure_models.py",
            "-m",
            "structure",
            "--junitxml=artifacts/junit/structure.xml",
        ),
        pre_commands=_RUN_STRUCTURE_REFERENCES,
    ),
    "feature": Suite(
        ("candidate-structure",),
        (
            "sudo",
            "docker",
            "compose",
            "-f",
            "docker/compose.yaml",
            "run",
            "--rm",
            "structure",
            "python",
            "-m",
            "pytest",
            "tests/integration/test_binder_design.py",
            "tests/integration/test_dplm_generation.py",
            "tests/integration/test_e1_rag.py",
            "tests/integration/test_esm3.py",
            "tests/integration/test_ttt.py",
            "tests/release/test_conversion_tools.py",
            "--junitxml=artifacts/junit/feature.xml",
        ),
    ),
    "artifact": Suite(
        ("candidate", "candidate-artifact"),
        (
            "sudo",
            "docker",
            "compose",
            "-f",
            "docker/compose.yaml",
            "run",
            "--rm",
            "artifact",
            "python",
            "-m",
            "pytest",
            "tests/release",
            "-m",
            "artifact",
            "--junitxml=artifacts/junit/artifact.xml",
        ),
        pre_commands=(_BUILD_ARTIFACTS, _DOWNLOAD_KERNELS),
    ),
    "benchmark": Suite(
        ("candidate-fp8",),
        (
            "sudo",
            "docker",
            "compose",
            "-f",
            "docker/compose.yaml",
            "run",
            "--rm",
            "benchmark",
            "--output",
            "artifacts/benchmarks/h100.json",
        ),
    ),
    "release": Suite(
        (
            "candidate",
            "candidate-structure",
            "candidate-artifact",
            *_SEQUENCE_REFERENCE_CONTAINERS,
            "reference-esmfold",
            "reference-esmfold2",
        ),
        (
            "sudo",
            "docker",
            "compose",
            "-f",
            "docker/compose.yaml",
            "run",
            "--rm",
            "structure",
            "python",
            "-m",
            "pytest",
            "tests/unit",
            "tests/integration",
            "tests/release",
            "tests/parity/test_native_results.py",
            *_RELEASE_LOCAL_PARITY_TESTS,
            "tests/structure",
            "tests/parity/test_boltz_source_refactor.py",
            "--ignore=tests/structure/test_structure_models.py",
            "--ignore=tests/structure/test_esmfold2_fp8_compliance.py",
            (
                "--deselect=tests/release/test_validation_stack.py::"
                "test_fp8_validation_stack_uses_the_cuda13_transformer_engine_core"
            ),
            (
                "--deselect=tests/structure/test_boltz2_folding_compliance.py::"
                "test_boltz2_live_folding_matches_pinned_official"
            ),
            "-m",
            "not artifact",
            "--junitxml=artifacts/junit/release.xml",
        ),
        pre_commands=(
            _BUILD_ARTIFACTS,
            _DOWNLOAD_KERNELS,
            _RUN_CHECK_ARTIFACTS,
            _PREPARE_REFERENCES,
            *_RUN_NATIVE_REFERENCES,
            *_RUN_RELEASE_STRUCTURE_REFERENCES,
            _RUN_PYTHON_MATRIX,
        ),
    ),
    "python-matrix": Suite(
        ("candidate",),
        _RUN_PYTHON_MATRIX,
    ),
}


@dataclass(frozen=True)
class RunnerConfig:
    """Runtime-only remote connection and execution settings."""

    host: str
    identity: Path
    repository: Path
    suite: str = "check"
    artifacts: Path = Path("artifacts/remote")
    accept_new_host_key: bool = False
    keep_remote: bool = False
    remote_parent: str | None = None


def _run_id(repository: Path) -> str:
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    material = f"{repository.resolve()}:{timestamp}:{os.getpid()}".encode()
    return f"{timestamp}-{hashlib.sha256(material).hexdigest()[:8]}"


def _is_sensitive(path: PurePosixPath) -> bool:
    lowered = tuple(part.lower() for part in path.parts)
    return (
        any(part in SENSITIVE_NAMES for part in lowered)
        or path.suffix.lower() in SENSITIVE_SUFFIXES
        or ".git" in lowered
        or "__pycache__" in lowered
    )


def _git_files(repository: Path) -> list[Path]:
    command = [
        "git",
        "-c",
        f"safe.directory={repository.resolve().as_posix()}",
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
    ]
    completed = subprocess.run(command, cwd=repository, check=True, capture_output=True)
    return [Path(raw.decode()) for raw in completed.stdout.split(b"\0") if raw]


def _require_clean_repository(repository: Path) -> None:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository.resolve().as_posix()}",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip():
        raise RuntimeError(
            "Remote runs require a clean Git worktree so the reported revision "
            "identifies the exact source."
        )


def _git_head_revision(repository: Path) -> str:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository.resolve().as_posix()}",
            "rev-parse",
            "HEAD",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    if GIT_REVISION_PATTERN.fullmatch(revision) is None:
        raise RuntimeError(f"Git returned an invalid HEAD revision: {revision!r}")
    return revision


def _gitlink_revision(repository: Path, relative_root: Path) -> str:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository.resolve().as_posix()}",
            "ls-files",
            "--stage",
            "-z",
            "--",
            relative_root.as_posix(),
        ],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    records = [record for record in completed.stdout.split(b"\0") if record]
    if len(records) != 1:
        raise RuntimeError(f"Expected one Git-link record for {relative_root.as_posix()!r}")
    try:
        metadata, encoded_path = records[0].split(b"\t", 1)
        mode, revision, stage = metadata.decode("ascii").split()
        recorded_path = encoded_path.decode()
    except (UnicodeDecodeError, ValueError) as error:
        raise RuntimeError(
            f"Could not parse Git-link record for {relative_root.as_posix()!r}"
        ) from error
    if (
        mode != "160000"
        or stage != "0"
        or recorded_path != relative_root.as_posix()
        or GIT_REVISION_PATTERN.fullmatch(revision) is None
    ):
        raise RuntimeError(f"Invalid Git-link record for {relative_root.as_posix()!r}")
    return revision


def _submodule_files(
    repository: Path,
    submodule: Path,
    relative_root: Path,
) -> tuple[list[tuple[Path, Path]], dict[str, object]]:
    git_metadata = submodule / ".git"
    if not (git_metadata.exists() or git_metadata.is_symlink()):
        raise RuntimeError(
            f"Submodule {relative_root.as_posix()!r} is not initialized. Run "
            "'git submodule update --init --recursive'."
        )
    safe_directory = f"safe.directory={submodule.resolve().as_posix()}"
    head = subprocess.run(
        ["git", "-c", safe_directory, "rev-parse", "HEAD"],
        cwd=submodule,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    gitlink = _gitlink_revision(repository, relative_root)
    if GIT_REVISION_PATTERN.fullmatch(head) is None or head != gitlink:
        raise RuntimeError(
            f"Submodule {relative_root.as_posix()!r} is at {head!r}, "
            f"but its Git link records {gitlink!r}."
        )
    status = subprocess.run(
        ["git", "-c", safe_directory, "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=submodule,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(
            f"Submodule {relative_root.as_posix()!r} has modified tracked files; "
            "source archives require the exact pinned tree."
        )
    completed = subprocess.run(
        ["git", "-c", safe_directory, "ls-files", "-z"],
        cwd=submodule,
        check=True,
        capture_output=True,
    )
    output: list[tuple[Path, Path]] = []
    tracked_files: list[str] = []
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        child = Path(raw.decode())
        source = submodule / child
        if not (source.is_file() or source.is_symlink()):
            raise RuntimeError(
                f"Tracked submodule path is unavailable or unsupported: "
                f"{(relative_root / child).as_posix()}"
            )
        child_name = child.as_posix()
        if _is_sensitive(PurePosixPath(child_name)):
            raise RuntimeError(
                f"Submodule tracks forbidden source path: {(relative_root / child).as_posix()}"
            )
        output.append((source, relative_root / child))
        tracked_files.append(child_name)
    tracked_files.sort()
    record: dict[str, object] = {
        "file_count": len(tracked_files),
        "gitlink_revision": gitlink,
        "head_revision": head,
        "tracked_files": tracked_files,
        "tree_sha256": tracked_tree_digest(submodule, tracked_files),
    }
    return output, record


def create_source_archive(
    repository: Path,
    destination: Path,
) -> dict[str, dict[str, object]]:
    """Archive tracked/untracked source plus initialized submodule tracked files."""

    repository = repository.resolve()
    files: list[tuple[Path, Path]] = []
    provenance: dict[str, dict[str, object]] = {}
    for relative in _git_files(repository):
        source = repository / relative
        posix = PurePosixPath(relative.as_posix())
        if (
            _is_sensitive(posix)
            or posix.as_posix() == ARCHIVE_PROVENANCE_NAME
            or not source.exists()
        ):
            continue
        if source.is_file():
            files.append((source, relative))
        elif posix.parts[:2] == ("vendor", "upstream"):
            submodule_files, record = _submodule_files(repository, source, relative)
            files.extend(submodule_files)
            provenance[posix.as_posix()] = record

    seen: set[str] = set()
    with tarfile.open(destination, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for source, relative in sorted(files, key=lambda item: item[1].as_posix()):
            archive_name = relative.as_posix()
            if archive_name in seen or _is_sensitive(PurePosixPath(archive_name)):
                continue
            seen.add(archive_name)
            archive.add(source, arcname=archive_name, recursive=False)
        provenance_bytes = render_archive_provenance(provenance)
        provenance_info = tarfile.TarInfo(ARCHIVE_PROVENANCE_NAME)
        provenance_info.size = len(provenance_bytes)
        provenance_info.mode = 0o644
        provenance_info.mtime = 0
        provenance_info.uid = 0
        provenance_info.gid = 0
        provenance_info.uname = ""
        provenance_info.gname = ""
        archive.addfile(provenance_info, io.BytesIO(provenance_bytes))
    return provenance


def remote_cleanup_command(remote_base: str, remote_workspace: str) -> tuple[str, ...]:
    """Build a fail-closed, remote-realpath-verified cleanup command."""

    base = PurePosixPath(remote_base)
    workspace = PurePosixPath(remote_workspace)
    if not base.is_absolute() or not workspace.is_absolute():
        raise ValueError("Remote cleanup paths must be absolute")
    if ".." in base.parts or ".." in workspace.parts:
        raise ValueError("Remote cleanup paths may not contain '..'")
    return (
        "sh",
        "-c",
        REMOTE_CLEANUP_SCRIPT,
        "fastplms-cleanup",
        str(base),
        str(workspace),
    )


class RemoteRunner:
    """Run one isolated Docker suite and retrieve its artifacts."""

    def __init__(self, config: RunnerConfig) -> None:
        if config.host.startswith("-") or not HOST_PATTERN.fullmatch(config.host):
            raise ValueError("SSH host contains unsupported characters")
        if config.suite not in SUITES:
            raise ValueError(f"Unknown suite {config.suite!r}")
        if not config.identity.is_file():
            raise FileNotFoundError(f"SSH identity does not exist: {config.identity}")
        self.config = config
        self.run_id = _run_id(config.repository)
        if not RUN_PATTERN.fullmatch(self.run_id):
            raise AssertionError("Generated invalid run ID")

    @property
    def ssh_prefix(self) -> list[str]:
        options = [
            "ssh",
            "-i",
            str(self.config.identity),
            "-o",
            "BatchMode=yes",
        ]
        if self.config.accept_new_host_key:
            options.extend(["-o", "StrictHostKeyChecking=accept-new"])
        return options

    @property
    def scp_prefix(self) -> list[str]:
        options = ["scp", "-i", str(self.config.identity), "-o", "BatchMode=yes"]
        if self.config.accept_new_host_key:
            options.extend(["-o", "StrictHostKeyChecking=accept-new"])
        return options

    def _ssh(
        self,
        command: Sequence[str],
        *,
        capture: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*self.ssh_prefix, self.config.host, shlex.join(command)],
            check=True,
            text=True,
            capture_output=capture,
        )

    def _ssh_at(
        self,
        workspace: str,
        command: Sequence[str],
        *,
        capture: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Run one command from ``workspace`` without interpolating user shell text."""

        script = f"cd {shlex.quote(workspace)} && exec {shlex.join(command)}"
        return self._ssh(("sh", "-lc", script), capture=capture)

    def _execution_environment(
        self,
        workspace: str,
        suite: Suite,
    ) -> dict[str, object]:
        """Capture exact built image IDs and stable host runtime identities."""

        bake = self._ssh_at(
            workspace,
            (
                "sudo",
                "docker",
                "buildx",
                "bake",
                "-f",
                "docker/docker-bake.hcl",
                "--print",
                *suite.bake_targets,
            ),
            capture=True,
        )
        bake_plan = json.loads(bake.stdout)
        target_plan = bake_plan.get("target")
        if not isinstance(target_plan, dict):
            raise RuntimeError("Docker Bake did not return a target plan")

        images: dict[str, object] = {}
        for target in suite.bake_targets:
            raw_target = target_plan.get(target)
            if not isinstance(raw_target, dict):
                raise RuntimeError(f"Docker Bake omitted target {target!r}")
            tags = raw_target.get("tags")
            if not isinstance(tags, list) or not tags or not isinstance(tags[0], str):
                raise RuntimeError(f"Docker Bake target {target!r} has no image tag")
            inspected = self._ssh(
                ("sudo", "docker", "image", "inspect", tags[0]),
                capture=True,
            )
            values = json.loads(inspected.stdout)
            if not isinstance(values, list) or len(values) != 1:
                raise RuntimeError(f"Docker returned invalid image identity for {target!r}")
            value = values[0]
            images[target] = {
                "tag": tags[0],
                "id": value["Id"],
                "repo_digests": value.get("RepoDigests") or [],
                "created": value["Created"],
                "os": value["Os"],
                "architecture": value["Architecture"],
            }

        docker_server = self._ssh(
            ("sudo", "docker", "version", "--format", "{{json .Server}}"),
            capture=True,
        )
        try:
            gpu = self._ssh(
                (
                    "nvidia-smi",
                    "--query-gpu=name,driver_version",
                    "--format=csv,noheader",
                ),
                capture=True,
            )
            gpus = [line.strip() for line in gpu.stdout.splitlines() if line.strip()]
        except subprocess.CalledProcessError:
            gpus = []
        return {
            "host_kernel": self._ssh(("uname", "-srm"), capture=True).stdout.strip(),
            "docker_server": json.loads(docker_server.stdout),
            "gpus": gpus,
            "images": images,
        }

    def _remote_base(self) -> str:
        if self.config.remote_parent is not None:
            parent = PurePosixPath(self.config.remote_parent)
            if not parent.is_absolute() or ".." in parent.parts:
                raise ValueError("--remote-parent must be an absolute path without '..'")
            return str(parent)
        completed = self._ssh(("pwd",), capture=True)
        home = PurePosixPath(completed.stdout.strip())
        if not home.is_absolute() or ".." in home.parts:
            raise RuntimeError("Could not determine a safe remote home directory")
        return str(home / "fastplms-runs")

    def run(self) -> Path:
        started_at = dt.datetime.now(dt.UTC).isoformat()
        _require_clean_repository(self.config.repository)
        git_revision = _git_head_revision(self.config.repository)
        remote_base = self._remote_base()
        remote_workspace = str(PurePosixPath(remote_base) / self.run_id)
        if not remote_workspace.startswith(remote_base.rstrip("/") + "/"):
            raise AssertionError("Remote workspace escaped its managed parent")
        output = self.config.artifacts / self.run_id
        output.mkdir(parents=True, exist_ok=False)
        source_archive_sha256 = ""
        submodule_revisions: dict[str, str] = {}
        execution_environment: dict[str, object] | None = None

        with tempfile.TemporaryDirectory(prefix="fastplms-remote-") as temporary:
            archive = Path(temporary) / "source.tar.gz"
            provenance = create_source_archive(self.config.repository, archive)
            submodule_revisions = {
                path: str(record["head_revision"])
                for path, record in provenance.items()
            }
            _require_clean_repository(self.config.repository)
            if _git_head_revision(self.config.repository) != git_revision:
                raise RuntimeError("Git HEAD changed while the remote source archive was built.")
            with archive.open("rb") as stream:
                source_archive_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
            self._ssh(("mkdir", "-p", remote_workspace))
            subprocess.run(
                [
                    *self.scp_prefix,
                    str(archive),
                    f"{self.config.host}:{remote_workspace}/source.tar.gz",
                ],
                check=True,
            )
            self._ssh(("tar", "-xzf", f"{remote_workspace}/source.tar.gz", "-C", remote_workspace))
            self._ssh(("rm", f"{remote_workspace}/source.tar.gz"))

        suite = SUITES[self.config.suite]
        phase = "initialize"
        retrieval_returncode = -1
        cleanup_status = "retained" if self.config.keep_remote else "pending"
        cleanup_failure: BaseException | None = None
        try:
            phase = "initialize-artifacts"
            self._ssh(("mkdir", "-p", f"{remote_workspace}/artifacts/junit"))
            for target in suite.bake_targets:
                phase = f"build:{target}"
                self._ssh_at(
                    remote_workspace,
                    (
                        "sudo",
                        "docker",
                        "buildx",
                        "bake",
                        "-f",
                        "docker/docker-bake.hcl",
                        target,
                        "--load",
                    ),
                )
            phase = "capture-environment"
            execution_environment = self._execution_environment(remote_workspace, suite)
            for index, command in enumerate(suite.pre_commands):
                phase = f"pre-command:{index}"
                self._ssh_at(remote_workspace, command)
            phase = "suite"
            self._ssh_at(remote_workspace, suite.command)
            phase = "complete"
        finally:
            active_failure = sys.exception()
            remote_artifacts = f"{self.config.host}:{remote_workspace}/artifacts/."
            retrieval = subprocess.run(
                [*self.scp_prefix, "-r", remote_artifacts, str(output)],
                check=False,
            )
            retrieval_returncode = retrieval.returncode
            try:
                if not self.config.keep_remote:
                    self._ssh(remote_cleanup_command(remote_base, remote_workspace))
                    cleanup_status = "succeeded"
            except BaseException as error:
                cleanup_failure = error
                cleanup_status = "failed"
                if active_failure is None:
                    raise
            finally:
                report_failure = active_failure or cleanup_failure
                report = _run_report(
                    run_id=self.run_id,
                    suite_name=self.config.suite,
                    suite=suite,
                    started_at=started_at,
                    finished_at=dt.datetime.now(dt.UTC).isoformat(),
                    source_archive_sha256=source_archive_sha256,
                    git_revision=git_revision,
                    submodule_revisions=submodule_revisions,
                    execution_environment=execution_environment,
                    failure_phase=(phase if active_failure is not None else "cleanup"),
                    failure=report_failure,
                    artifact_retrieval_returncode=retrieval_returncode,
                    cleanup_status=cleanup_status,
                )
                (output / "remote-run.json").write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        if retrieval_returncode != 0:
            raise RuntimeError(f"Remote artifacts could not be retrieved for run {self.run_id}")
        return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="SSH destination, for example user@gpu-host")
    parser.add_argument("--identity", required=True, type=Path, help="SSH private-key path")
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--suite", choices=tuple(SUITES), default="check")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/remote"))
    parser.add_argument("--accept-new-host-key", action="store_true")
    parser.add_argument("--keep-remote", action="store_true")
    parser.add_argument(
        "--remote-parent",
        help="Optional absolute managed directory on the remote host",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = RunnerConfig(
        host=arguments.host,
        identity=arguments.identity,
        repository=arguments.repository,
        suite=arguments.suite,
        artifacts=arguments.artifacts,
        accept_new_host_key=arguments.accept_new_host_key,
        keep_remote=arguments.keep_remote,
        remote_parent=arguments.remote_parent,
    )
    output = RemoteRunner(config).run()
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
