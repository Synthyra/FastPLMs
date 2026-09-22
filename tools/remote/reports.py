"""Secret-free execution receipts and retrieved-artifact inventories."""

from __future__ import annotations

import hashlib
import subprocess

from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from tools.execution.source import excluded_from_upload as _is_sensitive

from .contracts import (
    _BIOHUB_REFERENCE_TARGETS,
    _CONTROL_TIMEOUT_SECONDS,
    _TRANSFER_TIMEOUT_SECONDS,
    Suite,
)


_ARTIFACT_TREE_DOMAIN = b"fastplms-remote-artifact-tree-v1\0"


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
    phase_durations_seconds: Mapping[str, float] | None = None,
    cache_telemetry: Mapping[str, object] | None = None,
    artifact_inventory: Mapping[str, object] | None = None,
    host_hardware_preflight: Mapping[str, object] | None = None,
    kernel_capability_preflight: Mapping[str, object] | None = None,
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
        "schema_version": 5,
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
        "phase_durations_seconds": {
            key: round(value, 3) for key, value in sorted((phase_durations_seconds or {}).items())
        },
        "cache_telemetry": dict(cache_telemetry or {}),
        "artifact_inventory": (
            dict(artifact_inventory) if artifact_inventory is not None else None
        ),
        "host_hardware_preflight": (
            dict(host_hardware_preflight)
            if host_hardware_preflight is not None
            else None
        ),
        "kernel_capability_preflight": (
            dict(kernel_capability_preflight)
            if kernel_capability_preflight is not None
            else None
        ),
        "failure": failure_record,
        "suite_contract": {
            "bake_targets": list(suite.bake_targets),
            "pre_commands": [list(command) for command in suite.pre_commands],
            "command": list(suite.command),
            "required_paths": list(suite.required_paths),
            "biohub_reference_targets": sorted(
                _BIOHUB_REFERENCE_TARGETS.intersection(suite.bake_targets)
            ),
            "reference_targets": sorted(
                target for target in suite.bake_targets if target.startswith("reference-")
            ),
            "host_hardware_binding_required": True,
            "attention_backends": list(suite.attention_backends),
            "kernel_downloads_allowed": False,
            "same_host_candidate_reference_required": bool(
                any(target.startswith("reference-") for target in suite.bake_targets)
            ),
            "timeouts_seconds": {
                "control": _CONTROL_TIMEOUT_SECONDS,
                "transfer": _TRANSFER_TIMEOUT_SECONDS,
                "build": suite.build_timeout_seconds,
                "pre_command": suite.pre_command_timeout_seconds,
                "command": suite.command_timeout_seconds,
            },
        },
    }


def _artifact_tree_summary(root: Path) -> dict[str, object]:
    """Hash retrieved artifacts without recording potentially sensitive contents."""

    root = root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"Retrieved artifact root does not exist: {root}")
    digest = hashlib.sha256()
    digest.update(_ARTIFACT_TREE_DOMAIN)
    file_count = 0
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        relative_name = path.relative_to(root).as_posix()
        relative = PurePosixPath(relative_name)
        if path.is_symlink():
            raise RuntimeError("Retrieved artifacts may not contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError("Retrieved artifacts contain a non-regular entry")
        if _is_sensitive(relative):
            raise RuntimeError("Retrieved artifacts contain a sensitive path")
        content = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                content.update(chunk)
                size += len(chunk)
        for value in (
            relative_name.encode("utf-8"),
            size.to_bytes(8, "big"),
            content.digest(),
        ):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
        file_count += 1
        total_bytes += size
    return {
        "status": "captured",
        "file_count": file_count,
        "total_bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
    }
