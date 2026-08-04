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
import re
import secrets
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from tools.source_record import (
    ARCHIVE_PROVENANCE_NAME,
    archive_root_record,
    render_archive_provenance,
    tracked_tree_digest,
)


HOST_PATTERN = re.compile(r"^[A-Za-z0-9_.@:\-]+$")
RUN_PATTERN = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{16}$")
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
_ARTIFACT_TREE_DOMAIN = b"fastplms-remote-artifact-tree-v1\0"
_CONTROL_TIMEOUT_SECONDS = 300
_TRANSFER_TIMEOUT_SECONDS = 1_800
_BIOHUB_REFERENCE_TARGETS = frozenset({"reference-biohub-esm", "reference-esmfold2"})
_BIOHUB_BUILD_TARGET = "biohub-biotraj-wheel"
_MACHINE_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_GH200_RELEASE_BACKENDS = ("eager", "sdpa", "flex_attention")
_FLASH_ATTENTION_2_REVISION = "db6b51744f0cd7061386442c09df890fc6d9f47e"
_FLASH_ATTENTION_3_REVISION = "43f0bd269777115d94ff826e0d113ce9c1c9087b"
_REFERENCE_IMAGE_IDENTITY_PATH = (
    "artifacts/reference/environment/container-images.json"
)
_WRITE_JSON_SCRIPT = """import pathlib, sys
path = pathlib.Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_suffix(path.suffix + '.tmp')
temporary.write_text(sys.argv[2] + '\\n', encoding='utf-8')
temporary.replace(path)
"""


@dataclass(frozen=True)
class Suite:
    """Images to build and the command executed in the remote workspace."""

    bake_targets: tuple[str, ...]
    command: tuple[str, ...]
    pre_commands: tuple[tuple[str, ...], ...] = ()
    required_paths: tuple[str, ...] = ()
    build_timeout_seconds: int = 7_200
    pre_command_timeout_seconds: int = 7_200
    command_timeout_seconds: int = 7_200
    attention_backends: tuple[str, ...] = ()


def _normalized_host_architecture(machine: str) -> str:
    """Normalize trusted ``uname -m`` aliases without guessing unknown machines."""

    value = machine.strip().lower()
    if _MACHINE_PATTERN.fullmatch(value) is None:
        return "unknown"
    if value in {"amd64", "x86_64"}:
        return "amd64"
    if value in {"aarch64", "arm64"}:
        return "arm64"
    return value


def _host_hardware_preflight(machine: str, gpu_output: str) -> dict[str, object]:
    """Return one exact, platform-neutral host architecture and GPU binding."""

    uname_machine = machine.strip().lower()
    architecture = _normalized_host_architecture(uname_machine)
    if architecture == "unknown":
        raise RuntimeError("Remote uname returned an invalid machine architecture")
    gpus: list[dict[str, object]] = []
    seen_uuids: set[str] = set()
    for raw_line in gpu_output.splitlines():
        if not raw_line.strip():
            continue
        fields = [field.strip() for field in raw_line.split(",")]
        if len(fields) != 4 or any(not field for field in fields):
            raise RuntimeError("nvidia-smi returned an invalid GPU identity record")
        name, uuid, driver_version, raw_memory = fields
        if uuid in seen_uuids:
            raise RuntimeError("nvidia-smi returned a duplicate GPU UUID")
        try:
            memory_total_mib = int(raw_memory)
        except ValueError as error:
            raise RuntimeError("nvidia-smi returned invalid total GPU memory") from error
        if memory_total_mib <= 0:
            raise RuntimeError("nvidia-smi returned non-positive total GPU memory")
        seen_uuids.add(uuid)
        gpus.append(
            {
                "name": name,
                "uuid": uuid,
                "driver_version": driver_version,
                "memory_total_mib": memory_total_mib,
            }
        )
    if not gpus:
        raise RuntimeError("Remote validation requires an identifiable NVIDIA GPU")
    gpus.sort(key=lambda item: str(item["uuid"]))
    if architecture not in {"amd64", "arm64"}:
        raise RuntimeError(
            f"Remote validation does not declare an OCI platform for {architecture!r}"
        )
    identity = {
        "uname_machine": uname_machine,
        "architecture": architecture,
        "container_platform": f"linux/{architecture}",
        "gpus": gpus,
    }
    identity_sha256 = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"status": "passed", **identity, "identity_sha256": identity_sha256}


def _kernel_capability_preflight(
    host_hardware: Mapping[str, object],
    requested_backends: Sequence[str],
) -> dict[str, object]:
    """Resolve the no-download attention matrix for one exact native platform."""

    platform_name = host_hardware.get("container_platform")
    architecture = host_hardware.get("architecture")
    if platform_name != "linux/arm64" or architecture != "arm64":
        return {
            "schema_version": 1,
            "status": "failed",
            "policy": "gh200-native-no-flash-download-v1",
            "platform": platform_name,
            "selected_backends": list(requested_backends),
            "network_downloads": False,
            "source_builds": False,
            "reason": "The current release kernel policy is bound to native GH200 linux/arm64.",
            "backends": {},
        }

    requested = tuple(requested_backends)
    if len(set(requested)) != len(requested):
        return {
            "schema_version": 1,
            "status": "failed",
            "policy": "gh200-native-no-flash-download-v1",
            "platform": platform_name,
            "selected_backends": list(requested),
            "network_downloads": False,
            "source_builds": False,
            "reason": "The requested attention matrix contains duplicate backends.",
            "backends": {},
        }

    backend_records: dict[str, dict[str, object]] = {
        "eager": {
            "status": "available",
            "selected": "eager" in requested,
            "provider": "torch",
            "reason": "Framework eager attention is available without an external kernel.",
        },
        "sdpa": {
            "status": "available",
            "selected": "sdpa" in requested,
            "provider": "torch",
            "reason": "Framework SDPA is available without an external kernel.",
        },
        "flex_attention": {
            "status": "available",
            "selected": "flex_attention" in requested,
            "provider": "torch",
            "reason": "Framework Flex Attention is available without an external kernel.",
        },
        "flash_attention_2": {
            "status": "prior_focused_evidence_only",
            "selected": False,
            "provider": "kernels-community/flash-attn2",
            "revision": _FLASH_ATTENTION_2_REVISION,
            "reason": (
                "The GH200 release matrix reuses prior revision-pinned focused FA2 "
                "evidence; it does not download, build, or execute FA2 in this run."
            ),
        },
        "flash_attention_3": {
            "status": "unavailable",
            "selected": False,
            "provider": "kernels-community/flash-attn3",
            "revision": _FLASH_ATTENTION_3_REVISION,
            "reason": (
                "The manifest-pinned FA3 kernel has no validated linux/arm64 artifact "
                "for the current GH200 release image."
            ),
        },
    }
    unavailable = [
        backend
        for backend in requested
        if backend not in backend_records
        or backend_records[backend]["status"] != "available"
    ]
    return {
        "schema_version": 1,
        "status": "failed" if unavailable else "passed",
        "policy": "gh200-native-no-flash-download-v1",
        "platform": platform_name,
        "selected_backends": list(requested),
        "excluded_backends": [
            backend for backend in backend_records if backend not in requested
        ],
        "network_downloads": False,
        "source_builds": False,
        "reason": (
            "Requested backends are unavailable under the native GH200 policy: "
            + ", ".join(unavailable)
            if unavailable
            else None
        ),
        "backends": backend_records,
    }


def _reference_container_image_identity(
    execution_environment: Mapping[str, object],
) -> dict[str, object]:
    """Return the stable image/runtime identity shared with reference containers."""

    platform_name = execution_environment.get("container_platform")
    if not isinstance(platform_name, str) or not platform_name.startswith("linux/"):
        raise RuntimeError("Execution environment has no resolved Linux platform")
    raw_images = execution_environment.get("images")
    if not isinstance(raw_images, Mapping) or not raw_images:
        raise RuntimeError("Execution environment has no built-image identity map")
    images: dict[str, dict[str, str]] = {}
    for raw_name, raw_identity in sorted(raw_images.items(), key=lambda item: str(item[0])):
        if not isinstance(raw_name, str) or not isinstance(raw_identity, Mapping):
            raise RuntimeError("Execution environment contains an invalid image identity")
        content_digest = raw_identity.get("content_digest")
        image_id = raw_identity.get("id")
        os_name = raw_identity.get("os")
        architecture = raw_identity.get("architecture")
        resolved_platform = raw_identity.get("resolved_platform")
        if (
            not isinstance(content_digest, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", content_digest) is None
            or image_id != content_digest
            or os_name != "linux"
            or architecture != platform_name.split("/")[1]
            or resolved_platform != platform_name
        ):
            raise RuntimeError(f"Built image {raw_name!r} has an invalid stable identity")
        images[raw_name] = {
            "content_digest": content_digest,
            "image_id": content_digest,
            "os": os_name,
            "architecture": architecture,
            "resolved_platform": platform_name,
        }

    raw_server = execution_environment.get("docker_server")
    if not isinstance(raw_server, Mapping):
        raise RuntimeError("Execution environment has no Docker server identity")
    server_fields = (
        "Version",
        "ApiVersion",
        "MinAPIVersion",
        "GitCommit",
        "Os",
        "Arch",
        "KernelVersion",
    )
    docker_server = {
        field: raw_server[field]
        for field in server_fields
        if isinstance(raw_server.get(field), (str, int, float, bool))
    }
    required_server_fields = {"Version", "ApiVersion", "Os", "Arch"}
    if not required_server_fields.issubset(docker_server):
        raise RuntimeError("Docker server identity is missing required stable fields")
    if docker_server["Os"] != "linux" or docker_server["Arch"] != platform_name.split("/")[1]:
        raise RuntimeError("Docker server identity differs from the resolved native platform")
    buildx = execution_environment.get("docker_buildx")
    if not isinstance(buildx, str) or not buildx.strip():
        raise RuntimeError("Execution environment has no Docker Buildx identity")
    return {
        "schema_version": 1,
        "resolved_platform": platform_name,
        "docker_server": docker_server,
        "docker_buildx": buildx.strip(),
        "images": images,
    }


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
_BUILD_BENCHMARK_ARTIFACTS = _compose_run(
    "candidate",
    "python",
    "-m",
    "tools.artifacts.build_all",
    "--benchmark-suite",
    "--output-root",
    "dist/hub",
    "--source-root",
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
_RUN_CHECK_ARTIFACTS = _compose_run(
    "artifact",
    "python",
    "-m",
    "pytest",
    "tests/release/test_published_automodel.py",
    "tests/release/test_manifest_readiness.py",
    "-m",
    "artifact",
    "-k",
    "not test_local_artifact_locked_flash_backend",
    "--junitxml=artifacts/junit/check-artifact.xml",
)
_RUN_CHECK_GOLDENS = _compose_run(
    "structure",
    "python",
    "-m",
    "pytest",
    "tests/integration/test_official_goldens.py",
    "tests/structure/test_structure_official_goldens.py",
    "-m",
    "gpu and not large",
    "--junitxml=artifacts/junit/check-goldens.xml",
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
_RUN_NIGHTLY_FP8 = _compose_run(
    "fp8",
    "python",
    "-m",
    "pytest",
    "tests/structure/test_esmfold2_fp8_compliance.py",
    "--junitxml=artifacts/junit/nightly-fp8.xml",
)
_RUN_NIGHTLY_SEQUENCE_GOLDENS = _compose_run(
    "structure",
    "python",
    "-m",
    "pytest",
    "tests/integration/test_official_goldens.py",
    "--junitxml=artifacts/junit/nightly-sequence-goldens.xml",
)
_RUN_NIGHTLY_STRUCTURE_GOLDENS = _compose_run(
    "structure",
    "python",
    "-m",
    "pytest",
    "tests/structure/test_structure_official_goldens.py",
    "--junitxml=artifacts/junit/nightly-structure-goldens.xml",
)
_RUN_NIGHTLY_BENCHMARK = _compose_run(
    "benchmark",
    "--artifact-root",
    "dist/hub",
    "--backends",
    *_GH200_RELEASE_BACKENDS,
    "--output",
    "artifacts/benchmarks/nightly-h100.json",
    "--junit-output",
    "artifacts/junit/nightly-benchmark.xml",
)
_RUN_RELEASE_BENCHMARK = _compose_run(
    "benchmark",
    "--artifact-root",
    "dist/hub",
    "--backends",
    *_GH200_RELEASE_BACKENDS,
    "--output",
    "artifacts/benchmarks/release-h100.json",
    "--junit-output",
    "artifacts/junit/release-benchmark.xml",
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
            "tests/integration",
            "tests/release",
            "-m",
            "not gpu and not slow and not structure and not artifact",
            "--junitxml=artifacts/junit/check.xml",
        ),
        pre_commands=(
            _RUN_CHECK_GOLDENS,
        ),
        attention_backends=_GH200_RELEASE_BACKENDS,
    ),
    "gpu-golden-smoke": Suite(
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
            (
                "tests/release/test_validation_stack.py::"
                "test_release_hopper_sm90_gpu_is_available_without_running_a_model"
            ),
            "tests/integration/test_official_goldens.py",
            "tests/structure/test_structure_official_goldens.py",
            "-m",
            "gpu and not large",
            "--junitxml=artifacts/junit/gpu-golden-smoke.xml",
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
            _BIOHUB_BUILD_TARGET,
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
                "test_release_hopper_sm90_gpu_is_available_without_running_a_model"
            ),
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
        attention_backends=_GH200_RELEASE_BACKENDS,
    ),
    "structure": Suite(
        (
            "candidate-structure",
            "candidate-fp8",
            _BIOHUB_BUILD_TARGET,
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
            "-k",
            "not test_local_artifact_locked_flash_backend",
            "--junitxml=artifacts/junit/artifact.xml",
        ),
        pre_commands=(_BUILD_ARTIFACTS,),
    ),
    "benchmark": Suite(
        ("candidate", "candidate-fp8"),
        (
            "sudo",
            "docker",
            "compose",
            "-f",
            "docker/compose.yaml",
            "run",
            "--rm",
            "benchmark",
            "--artifact-root",
            "dist/hub",
            "--backends",
            *_GH200_RELEASE_BACKENDS,
            "--output",
            "artifacts/benchmarks/h100-current.json",
            "--baseline",
            "benchmarks/baselines/h100.json",
            "--junit-output",
            "artifacts/junit/benchmark.xml",
        ),
        pre_commands=(_BUILD_BENCHMARK_ARTIFACTS,),
        required_paths=("benchmarks/baselines/h100.json",),
        pre_command_timeout_seconds=14_400,
        attention_backends=_GH200_RELEASE_BACKENDS,
    ),
    "benchmark-capture": Suite(
        ("candidate", "candidate-fp8"),
        (
            "sudo",
            "docker",
            "compose",
            "-f",
            "docker/compose.yaml",
            "run",
            "--rm",
            "benchmark",
            "--artifact-root",
            "dist/hub",
            "--backends",
            *_GH200_RELEASE_BACKENDS,
            "--output",
            "artifacts/benchmarks/h100-baseline-candidate.json",
            "--junit-output",
            "artifacts/junit/benchmark-capture.xml",
        ),
        pre_commands=(_BUILD_BENCHMARK_ARTIFACTS,),
        pre_command_timeout_seconds=14_400,
        attention_backends=_GH200_RELEASE_BACKENDS,
    ),
    "nightly": Suite(
        (
            "candidate",
            "candidate-structure",
            "candidate-fp8",
            "candidate-artifact",
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
            "tests/integration/test_backend_consistency.py",
            "tests/integration/test_binder_design.py",
            "tests/integration/test_dplm_generation.py",
            "tests/integration/test_e1_rag.py",
            "tests/integration/test_esm3.py",
            "tests/integration/test_ttt.py",
            "tests/unit/test_fine_tuning_example.py",
            "--junitxml=artifacts/junit/nightly-features.xml",
        ),
        pre_commands=(
            _BUILD_ARTIFACTS,
            _RUN_CHECK_ARTIFACTS,
            _RUN_NIGHTLY_SEQUENCE_GOLDENS,
            _RUN_NIGHTLY_STRUCTURE_GOLDENS,
            _RUN_NIGHTLY_FP8,
            _RUN_NIGHTLY_BENCHMARK,
        ),
        pre_command_timeout_seconds=14_400,
        command_timeout_seconds=21_600,
        attention_backends=_GH200_RELEASE_BACKENDS,
    ),
    "release": Suite(
        (
            "candidate",
            "candidate-structure",
            "candidate-fp8",
            "candidate-artifact",
            _BIOHUB_BUILD_TARGET,
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
            "--ignore=tests/integration/test_flash_attention_backends.py",
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
            _RUN_CHECK_ARTIFACTS,
            _PREPARE_REFERENCES,
            *_RUN_NATIVE_REFERENCES,
            *_RUN_RELEASE_STRUCTURE_REFERENCES,
            _RUN_PYTHON_MATRIX,
            _RUN_RELEASE_BENCHMARK,
        ),
        pre_command_timeout_seconds=21_600,
        attention_backends=_GH200_RELEASE_BACKENDS,
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


def _run_id(_repository: Path) -> str:
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{secrets.token_hex(8)}"


def _is_sensitive(path: PurePosixPath) -> bool:
    lowered = tuple(part.lower() for part in path.parts)
    return (
        any(part in SENSITIVE_NAMES for part in lowered)
        or path.suffix.lower() in SENSITIVE_SUFFIXES
        or ".git" in lowered
        or "__pycache__" in lowered
    )


def _require_matching_archive_digest(output: str, expected_sha256: str) -> None:
    fields = output.split()
    if (
        re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        or not fields
        or fields[0] != expected_sha256
    ):
        raise RuntimeError("Uploaded source archive SHA-256 differs from local bytes")


def _git_files(repository: Path) -> list[Path]:
    command = [
        "git",
        "-c",
        f"safe.directory={repository.resolve().as_posix()}",
        "ls-files",
        "-z",
        "--cached",
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


def _require_clean_tracked_repository(repository: Path) -> None:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository.resolve().as_posix()}",
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
            "--ignore-submodules=all",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip():
        raise RuntimeError(
            "Source archives require clean tracked root files so the content "
            "attestation identifies exact bytes."
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


def _is_tracked_file(repository: Path, relative_name: str) -> bool:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository.resolve().as_posix()}",
            "ls-files",
            "--error-unmatch",
            "--",
            relative_name,
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0 and completed.stdout.strip() == relative_name


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
    """Archive tracked source plus initialized, pinned submodule tracked files."""

    repository = repository.resolve()
    _require_clean_tracked_repository(repository)
    head_revision = _git_head_revision(repository)
    files: list[tuple[Path, Path]] = []
    root_tracked_files: list[str] = []
    provenance: dict[str, dict[str, object]] = {}
    for relative in _git_files(repository):
        source = repository / relative
        posix = PurePosixPath(relative.as_posix())
        if _is_sensitive(posix):
            raise RuntimeError(f"Repository tracks forbidden source path: {posix.as_posix()!r}")
        if posix.as_posix() == ARCHIVE_PROVENANCE_NAME:
            raise RuntimeError("Repository may not track the generated source provenance marker")
        if not source.exists() and not source.is_symlink():
            if posix.parts[:2] == ("vendor", "upstream"):
                continue
            raise RuntimeError(f"Tracked source path is unavailable: {posix.as_posix()!r}")
        if source.is_file() or source.is_symlink():
            files.append((source, relative))
            root_tracked_files.append(posix.as_posix())
        elif posix.parts[:2] == ("vendor", "upstream"):
            git_metadata = source / ".git"
            if not (git_metadata.exists() or git_metadata.is_symlink()):
                continue
            submodule_files, record = _submodule_files(repository, source, relative)
            files.extend(submodule_files)
            provenance[posix.as_posix()] = record
        else:
            raise RuntimeError(f"Tracked source path has an unsupported type: {posix.as_posix()!r}")

    root_record = archive_root_record(
        repository,
        root_tracked_files,
        head_revision=head_revision,
    )

    seen: set[str] = set()
    with tarfile.open(
        destination,
        "w:gz",
        format=tarfile.PAX_FORMAT,
        dereference=False,
    ) as archive:
        for source, relative in sorted(files, key=lambda item: item[1].as_posix()):
            archive_name = relative.as_posix()
            if archive_name in seen or _is_sensitive(PurePosixPath(archive_name)):
                continue
            seen.add(archive_name)
            archive.add(source, arcname=archive_name, recursive=False)
        provenance_bytes = render_archive_provenance(provenance, root=root_record)
        provenance_info = tarfile.TarInfo(ARCHIVE_PROVENANCE_NAME)
        provenance_info.size = len(provenance_bytes)
        provenance_info.mode = 0o644
        provenance_info.mtime = 0
        provenance_info.uid = 0
        provenance_info.gid = 0
        provenance_info.uname = ""
        provenance_info.gname = ""
        archive.addfile(provenance_info, io.BytesIO(provenance_bytes))
    _require_clean_tracked_repository(repository)
    if _git_head_revision(repository) != head_revision:
        destination.unlink(missing_ok=True)
        raise RuntimeError("Repository revision changed while creating the source archive")
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
            "-o",
            "IdentitiesOnly=yes",
        ]
        if self.config.accept_new_host_key:
            options.extend(["-o", "StrictHostKeyChecking=accept-new"])
        return options

    @property
    def scp_prefix(self) -> list[str]:
        options = [
            "scp",
            "-i",
            str(self.config.identity),
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
        ]
        if self.config.accept_new_host_key:
            options.extend(["-o", "StrictHostKeyChecking=accept-new"])
        return options

    def _ssh(
        self,
        command: Sequence[str],
        *,
        capture: bool = False,
        timeout_seconds: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        effective_timeout = (
            _CONTROL_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds + 60
        )
        return subprocess.run(
            [*self.ssh_prefix, self.config.host, shlex.join(command)],
            check=True,
            text=True,
            capture_output=capture,
            timeout=effective_timeout,
        )

    def _ssh_at(
        self,
        workspace: str,
        command: Sequence[str],
        *,
        capture: bool = False,
        timeout_seconds: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run one cancellable command from ``workspace``.

        GNU ``timeout`` terminates the remote process group, while the slightly
        longer local SSH timeout prevents a disconnected client from waiting
        forever if the remote host becomes unresponsive.
        """

        script = f"cd {shlex.quote(workspace)} && exec {shlex.join(command)}"
        remote_command: tuple[str, ...]
        if timeout_seconds is None:
            remote_command = ("sh", "-lc", script)
        else:
            remote_command = (
                "timeout",
                "--signal=TERM",
                "--kill-after=30s",
                f"{timeout_seconds}s",
                "sh",
                "-lc",
                script,
            )
        return self._ssh(
            remote_command,
            capture=capture,
            timeout_seconds=timeout_seconds,
        )

    def _capture_host_hardware(self) -> dict[str, object]:
        """Capture the exact native architecture and NVIDIA devices before Docker."""

        machine = self._ssh(
            ("uname", "-m"),
            capture=True,
            timeout_seconds=_CONTROL_TIMEOUT_SECONDS,
        ).stdout
        gpu_output = self._ssh(
            (
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ),
            capture=True,
            timeout_seconds=_CONTROL_TIMEOUT_SECONDS,
        ).stdout
        return _host_hardware_preflight(machine, gpu_output)

    def _docker_cache_telemetry(self) -> dict[str, object]:
        """Return stable Docker disk/cache counters without command output text."""

        try:
            completed = self._ssh(
                (
                    "sudo",
                    "docker",
                    "system",
                    "df",
                    "--format",
                    "{{json .}}",
                ),
                capture=True,
                timeout_seconds=60,
            )
            records: list[dict[str, object]] = []
            allowed_fields = {"Type", "TotalCount", "Active", "Size", "Reclaimable"}
            for line in completed.stdout.splitlines():
                if not line.strip():
                    continue
                raw_record = json.loads(line)
                if not isinstance(raw_record, dict):
                    raise ValueError("Docker cache telemetry record is not an object")
                records.append(
                    {
                        str(key): value
                        for key, value in raw_record.items()
                        if key in allowed_fields and isinstance(value, (str, int, float))
                    }
                )
        except (OSError, ValueError, subprocess.SubprocessError):
            return {"status": "unavailable"}
        return {"status": "captured", "records": records}

    def _execution_environment(
        self,
        workspace: str,
        suite: Suite,
        host_hardware: Mapping[str, object],
    ) -> dict[str, object]:
        """Capture exact built image IDs and stable host runtime identities."""

        current_hardware = self._capture_host_hardware()
        if current_hardware != host_hardware:
            raise RuntimeError("Remote host hardware identity changed during the build")
        container_platform = str(host_hardware["container_platform"])

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
                "--set",
                f"*.platform={container_platform}",
                *suite.bake_targets,
            ),
            capture=True,
            timeout_seconds=_CONTROL_TIMEOUT_SECONDS,
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
            platforms = raw_target.get("platforms")
            if platforms != [container_platform]:
                raise RuntimeError(
                    f"Docker Bake target {target!r} resolved unexpected platforms: {platforms!r}"
                )
            tags = raw_target.get("tags")
            if not isinstance(tags, list) or not tags or not isinstance(tags[0], str):
                raise RuntimeError(f"Docker Bake target {target!r} has no image tag")
            inspected = self._ssh(
                ("sudo", "docker", "image", "inspect", tags[0]),
                capture=True,
                timeout_seconds=_CONTROL_TIMEOUT_SECONDS,
            )
            values = json.loads(inspected.stdout)
            if (
                not isinstance(values, list)
                or len(values) != 1
                or not isinstance(values[0], dict)
            ):
                raise RuntimeError(f"Docker returned invalid image identity for {target!r}")
            value = values[0]
            image_id = value.get("Id")
            if (
                not isinstance(image_id, str)
                or re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None
            ):
                raise RuntimeError(f"Docker returned invalid image digest for {target!r}")
            expected_os, expected_architecture = container_platform.split("/", maxsplit=1)
            if (
                value.get("Os") != expected_os
                or value.get("Architecture") != expected_architecture
            ):
                raise RuntimeError(
                    f"Built image {target!r} does not match native platform "
                    f"{container_platform!r}"
                )
            images[target] = {
                "tag": tags[0],
                "id": image_id,
                "repo_digests": value.get("RepoDigests") or [],
                "created": value["Created"],
                "os": value["Os"],
                "architecture": value["Architecture"],
                "resolved_platform": container_platform,
                "content_digest": image_id,
            }

        docker_server = self._ssh(
            ("sudo", "docker", "version", "--format", "{{json .Server}}"),
            capture=True,
            timeout_seconds=_CONTROL_TIMEOUT_SECONDS,
        )
        docker_buildx = self._ssh(
            ("sudo", "docker", "buildx", "version"),
            capture=True,
            timeout_seconds=_CONTROL_TIMEOUT_SECONDS,
        )
        try:
            gpu = self._ssh(
                (
                    "nvidia-smi",
                    "--query-gpu=name,driver_version",
                    "--format=csv,noheader",
                ),
                capture=True,
                timeout_seconds=_CONTROL_TIMEOUT_SECONDS,
            )
            gpus = [line.strip() for line in gpu.stdout.splitlines() if line.strip()]
        except subprocess.CalledProcessError:
            gpus = []
        return {
            "host_hardware": dict(host_hardware),
            "container_platform": container_platform,
            "host_kernel": self._ssh(
                ("uname", "-srm"),
                capture=True,
                timeout_seconds=_CONTROL_TIMEOUT_SECONDS,
            ).stdout.strip(),
            "docker_server": json.loads(docker_server.stdout),
            "docker_buildx": docker_buildx.stdout.strip(),
            "gpus": gpus,
            "images": images,
        }

    def _persist_reference_container_identity(
        self,
        workspace: str,
        execution_environment: Mapping[str, object],
    ) -> dict[str, object]:
        """Persist stable image identities before any native reference executes."""

        identity = _reference_container_image_identity(execution_environment)
        payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        self._ssh_at(
            workspace,
            ("python3", "-c", _WRITE_JSON_SCRIPT, _REFERENCE_IMAGE_IDENTITY_PATH, payload),
            timeout_seconds=_CONTROL_TIMEOUT_SECONDS,
        )
        return identity

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
        suite = SUITES[self.config.suite]
        for relative_name in suite.required_paths:
            relative = PurePosixPath(relative_name)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"Suite has an unsafe required path: {relative_name!r}")
            required = self.config.repository.joinpath(*relative.parts)
            if (
                not required.is_file()
                or required.is_symlink()
                or not _is_tracked_file(self.config.repository, relative_name)
            ):
                capture_hint = (
                    " Run --suite benchmark-capture to produce a descriptive candidate report; "
                    "review and commit an immutable baseline separately."
                    if self.config.suite == "benchmark"
                    else ""
                )
                raise RuntimeError(
                    f"Suite {self.config.suite!r} requires tracked file {relative_name!r}."
                    + capture_hint
                )
        remote_base = self._remote_base()
        remote_workspace = str(PurePosixPath(remote_base) / self.run_id)
        if not remote_workspace.startswith(remote_base.rstrip("/") + "/"):
            raise AssertionError("Remote workspace escaped its managed parent")
        output = self.config.artifacts / self.run_id
        output.mkdir(parents=True, exist_ok=False)
        source_archive_sha256 = ""
        submodule_revisions: dict[str, str] = {}
        execution_environment: dict[str, object] | None = None
        phase = "initialize"
        phase_started = time.monotonic()
        phase_durations_seconds: dict[str, float] = {}
        cache_telemetry: dict[str, object] = {}
        artifact_inventory: dict[str, object] | None = None
        host_hardware_preflight: dict[str, object] | None = None
        kernel_capability_preflight: dict[str, object] | None = None
        retrieval_returncode = -1
        cleanup_status = "retained" if self.config.keep_remote else "pending"
        cleanup_failure: BaseException | None = None
        inventory_failure: BaseException | None = None
        remote_workspace_touched = False
        remote_workspace_created = False

        def start_phase(next_phase: str) -> None:
            nonlocal phase, phase_started
            phase_durations_seconds[phase] = (
                phase_durations_seconds.get(phase, 0.0) + time.monotonic() - phase_started
            )
            phase = next_phase
            phase_started = time.monotonic()

        try:
            start_phase("host-hardware-preflight")
            host_hardware_preflight = self._capture_host_hardware()
            start_phase("kernel-capability-preflight")
            kernel_capability_preflight = _kernel_capability_preflight(
                host_hardware_preflight,
                suite.attention_backends,
            )
            if kernel_capability_preflight["status"] != "passed":
                raise RuntimeError(str(kernel_capability_preflight["reason"]))
            with tempfile.TemporaryDirectory(prefix="fastplms-remote-") as temporary:
                start_phase("create-source-archive")
                archive = Path(temporary) / "source.tar.gz"
                provenance = create_source_archive(self.config.repository, archive)
                submodule_revisions = {
                    path: str(record["head_revision"]) for path, record in provenance.items()
                }
                _require_clean_repository(self.config.repository)
                if _git_head_revision(self.config.repository) != git_revision:
                    raise RuntimeError(
                        "Git HEAD changed while the remote source archive was built."
                    )
                with archive.open("rb") as stream:
                    source_archive_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()

                start_phase("create-remote-workspace")
                remote_workspace_touched = True
                self._ssh(("mkdir", "-p", remote_workspace))
                remote_workspace_created = True

                start_phase("upload-source-archive")
                subprocess.run(
                    [
                        *self.scp_prefix,
                        str(archive),
                        f"{self.config.host}:{remote_workspace}/source.tar.gz",
                    ],
                    check=True,
                    timeout=_TRANSFER_TIMEOUT_SECONDS,
                )
                start_phase("verify-source-archive")
                remote_digest_output = self._ssh(
                    ("sha256sum", f"{remote_workspace}/source.tar.gz"),
                    capture=True,
                    timeout_seconds=_CONTROL_TIMEOUT_SECONDS,
                ).stdout
                _require_matching_archive_digest(
                    remote_digest_output,
                    source_archive_sha256,
                )
                start_phase("extract-source-archive")
                self._ssh(
                    ("tar", "-xzf", f"{remote_workspace}/source.tar.gz", "-C", remote_workspace)
                )
                start_phase("remove-source-archive")
                self._ssh(("rm", f"{remote_workspace}/source.tar.gz"))

            start_phase("initialize-artifacts")
            self._ssh(("mkdir", "-p", f"{remote_workspace}/artifacts/junit"))
            start_phase("capture-cache-before-build")
            cache_telemetry["before_build"] = self._docker_cache_telemetry()
            start_phase("build")
            self._ssh_at(
                remote_workspace,
                (
                    "sudo",
                    "docker",
                    "buildx",
                    "bake",
                    "-f",
                    "docker/docker-bake.hcl",
                    "--set",
                    f"*.platform={host_hardware_preflight['container_platform']}",
                    *suite.bake_targets,
                    "--load",
                ),
                timeout_seconds=suite.build_timeout_seconds,
            )
            start_phase("capture-environment")
            execution_environment = self._execution_environment(
                remote_workspace,
                suite,
                host_hardware_preflight,
            )
            start_phase("persist-reference-container-identity")
            execution_environment["reference_container_identity"] = (
                self._persist_reference_container_identity(
                    remote_workspace,
                    execution_environment,
                )
            )
            for index, command in enumerate(suite.pre_commands):
                start_phase(f"pre-command:{index}")
                self._ssh_at(
                    remote_workspace,
                    command,
                    timeout_seconds=suite.pre_command_timeout_seconds,
                )
            start_phase("suite")
            self._ssh_at(
                remote_workspace,
                suite.command,
                timeout_seconds=suite.command_timeout_seconds,
            )
            start_phase("complete")
        finally:
            active_failure = sys.exception()
            failure_phase = phase if active_failure is not None else None
            start_phase("capture-cache-after-run")
            if remote_workspace_created and "before_build" in cache_telemetry:
                cache_telemetry["after_run"] = self._docker_cache_telemetry()
            start_phase("artifact-retrieval")
            if remote_workspace_created:
                remote_artifacts = f"{self.config.host}:{remote_workspace}/artifacts/."
                try:
                    retrieval = subprocess.run(
                        [*self.scp_prefix, "-r", remote_artifacts, str(output)],
                        check=False,
                        timeout=_TRANSFER_TIMEOUT_SECONDS,
                    )
                    retrieval_returncode = retrieval.returncode
                except subprocess.TimeoutExpired:
                    retrieval_returncode = 124
                if retrieval_returncode == 0:
                    start_phase("artifact-inventory")
                    try:
                        artifact_inventory = _artifact_tree_summary(output)
                    except BaseException as error:
                        inventory_failure = error
                        artifact_inventory = {
                            "status": "failed",
                            "error_type": type(error).__name__,
                        }
            try:
                start_phase("cleanup")
                if not self.config.keep_remote and remote_workspace_touched:
                    self._ssh(remote_cleanup_command(remote_base, remote_workspace))
                    cleanup_status = "succeeded"
                elif not self.config.keep_remote:
                    cleanup_status = "succeeded"
            except BaseException as error:
                cleanup_failure = error
                cleanup_status = "failed"
                if active_failure is None:
                    raise
            finally:
                start_phase("report")
                report_failure = active_failure or cleanup_failure or inventory_failure
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
                    failure_phase=(
                        failure_phase
                        if active_failure is not None
                        else (
                            "cleanup"
                            if cleanup_failure is not None
                            else ("artifact-inventory" if inventory_failure is not None else None)
                        )
                    ),
                    failure=report_failure,
                    artifact_retrieval_returncode=retrieval_returncode,
                    cleanup_status=cleanup_status,
                    phase_durations_seconds=phase_durations_seconds,
                    cache_telemetry=cache_telemetry,
                    artifact_inventory=artifact_inventory,
                    host_hardware_preflight=host_hardware_preflight,
                    kernel_capability_preflight=kernel_capability_preflight,
                )
                report_path = output / "remote-run.json"
                temporary_report = output / ".remote-run.json.tmp"
                temporary_report.write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                temporary_report.replace(report_path)
        if retrieval_returncode != 0:
            raise RuntimeError(f"Remote artifacts could not be retrieved for run {self.run_id}")
        if inventory_failure is not None:
            raise RuntimeError(
                f"Remote artifacts failed inventory validation for run {self.run_id}"
            ) from inventory_failure
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


def main(argv: Sequence[str] | None = None) -> int:
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
