"""Capture and validate the complete native Biohub reference environment."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tools.remote.biohub_reference_lock import (
    BiohubReferenceLockError,
    expected_installed_inventory,
    load_biohub_reference_lock_contract,
    normalize_distribution_version,
    verify_biohub_reference_lock_contract,
    verify_current_installed_inventory,
    verify_current_pip_check,
)


REFERENCE_ENVIRONMENT_SCHEMA_VERSION = 2
BIOHUB_BUILD_TARGET = "biohub-biotraj-wheel"
BIOHUB_REFERENCE_TARGETS = frozenset({"reference-biohub-esm", "reference-esmfold2"})
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class BiohubReferenceEnvironmentError(RuntimeError):
    """The native runtime or persisted image identity differs from its lock."""


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise BiohubReferenceEnvironmentError(f"Unable to hash {path}.") from error


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _digest_object(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _load_canonical_json(path: Path) -> tuple[dict[str, object], str]:
    try:
        serialized = path.read_bytes()
        value: Any = json.loads(serialized.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BiohubReferenceEnvironmentError(
            f"Unable to read reference container identity: {path}"
        ) from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise BiohubReferenceEnvironmentError("Reference container identity must be a JSON object.")
    normalized = {str(key): item for key, item in value.items()}
    if serialized != _canonical_json_bytes(normalized):
        raise BiohubReferenceEnvironmentError("Reference container identity is not canonical JSON.")
    return normalized, hashlib.sha256(serialized).hexdigest()


def _validated_image_identity(value: object, *, target: str) -> dict[str, str]:
    fields = {"content_digest", "image_id", "os", "architecture", "resolved_platform"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise BiohubReferenceEnvironmentError(
            f"Container image identity fields differ for {target!r}."
        )
    digest = value["content_digest"]
    if not isinstance(digest, str) or _IMAGE_ID.fullmatch(digest) is None:
        raise BiohubReferenceEnvironmentError(f"Container image digest is invalid for {target!r}.")
    if (
        value["image_id"] != digest
        or value["os"] != "linux"
        or value["architecture"] != "arm64"
        or value["resolved_platform"] != "linux/arm64"
    ):
        raise BiohubReferenceEnvironmentError(f"Container image platform differs for {target!r}.")
    return {field: str(value[field]) for field in sorted(fields)}


def _validated_container_identity(value: object) -> dict[str, object]:
    fields = {"schema_version", "resolved_platform", "docker_server", "docker_buildx", "images"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise BiohubReferenceEnvironmentError("Reference container identity fields differ.")
    if value["schema_version"] != 1 or value["resolved_platform"] != "linux/arm64":
        raise BiohubReferenceEnvironmentError("Reference container platform is not linux/arm64.")
    buildx = value["docker_buildx"]
    if not isinstance(buildx, str) or not buildx.strip():
        raise BiohubReferenceEnvironmentError("Docker Buildx identity is missing.")
    server = value["docker_server"]
    allowed_server_fields = {
        "Version",
        "ApiVersion",
        "MinAPIVersion",
        "GitCommit",
        "Os",
        "Arch",
        "KernelVersion",
    }
    required_server_fields = {"Version", "ApiVersion", "Os", "Arch"}
    if (
        not isinstance(server, Mapping)
        or not required_server_fields.issubset(server)
        or not set(server).issubset(allowed_server_fields)
        or server.get("Os") != "linux"
        or server.get("Arch") not in {"arm64", "aarch64"}
        or any(not isinstance(item, (str, int, float, bool)) for item in server.values())
    ):
        raise BiohubReferenceEnvironmentError("Docker server identity is invalid.")
    images = value["images"]
    if not isinstance(images, Mapping) or not images:
        raise BiohubReferenceEnvironmentError("Reference container image map is missing.")
    normalized_images: dict[str, dict[str, str]] = {}
    for raw_target, identity in images.items():
        if not isinstance(raw_target, str) or not raw_target:
            raise BiohubReferenceEnvironmentError("Reference container target name is invalid.")
        normalized_images[raw_target] = _validated_image_identity(identity, target=raw_target)
    return {
        "schema_version": 1,
        "resolved_platform": "linux/arm64",
        "docker_server": {str(key): server[key] for key in sorted(server)},
        "docker_buildx": buildx.strip(),
        "images": {key: normalized_images[key] for key in sorted(normalized_images)},
    }


def _driver_version() -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise BiohubReferenceEnvironmentError(
            "Biohub reference evidence requires the NVIDIA driver version."
        ) from error
    versions = {line.strip() for line in completed.stdout.splitlines() if line.strip()}
    if len(versions) != 1:
        raise BiohubReferenceEnvironmentError("NVIDIA driver version is ambiguous.")
    return versions.pop()


def _runtime_identity(installed_inventory: Mapping[str, str]) -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise BiohubReferenceEnvironmentError("Biohub reference evidence requires CUDA.")
    system = platform.system().lower()
    machine = platform.machine()
    implementation = platform.python_implementation()
    python_version = platform.python_version()
    torch_version = normalize_distribution_version("torch", torch.__version__)
    cuda_runtime = str(torch.version.cuda or "")
    properties = torch.cuda.get_device_properties(0)
    gpu_name = properties.name
    capability = list(torch.cuda.get_device_capability(0))
    if (
        system != "linux"
        or machine != "aarch64"
        or implementation != "CPython"
        or not python_version.startswith("3.12.")
        or torch_version != installed_inventory.get("torch")
        or not cuda_runtime.startswith("13.0")
        or gpu_name != "NVIDIA GH200 480GB"
        or capability != [9, 0]
    ):
        raise BiohubReferenceEnvironmentError("Active runtime differs from the GH200 lock target.")
    uname = platform.uname()
    return {
        "operating_system": system,
        "architecture": machine,
        "python_implementation": implementation,
        "python_version": python_version,
        "torch": torch_version,
        "cuda_runtime": cuda_runtime,
        "cuda_driver": _driver_version(),
        "gpu": {
            "name": gpu_name,
            "capability": capability,
            "total_memory_bytes": int(properties.total_memory),
        },
        "uname": {
            "system": uname.system,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
        },
    }


def capture_biohub_reference_environment(
    repository_root: Path,
    contract_path: Path,
    container_identity_path: Path,
    *,
    reference_target: str,
) -> dict[str, object]:
    """Capture the locked dependency, image, hardware, and runtime identity."""

    if reference_target not in BIOHUB_REFERENCE_TARGETS:
        raise BiohubReferenceEnvironmentError(
            f"Unsupported Biohub reference target: {reference_target!r}."
        )
    try:
        contract = load_biohub_reference_lock_contract(contract_path)
        locks = verify_biohub_reference_lock_contract(repository_root, contract_path)
        inventory = verify_current_installed_inventory(
            repository_root,
            contract_path,
            profile="final",
        )
        pip_check = verify_current_pip_check(repository_root, contract_path)
    except BiohubReferenceLockError as error:
        raise BiohubReferenceEnvironmentError(str(error)) from error
    container_identity, manifest_sha256 = _load_canonical_json(container_identity_path)
    normalized_container = _validated_container_identity(container_identity)
    images = normalized_container["images"]
    if not isinstance(images, Mapping):
        raise BiohubReferenceEnvironmentError(
            "Reference container identity images must be a mapping."
        )
    missing_targets = {BIOHUB_BUILD_TARGET, reference_target}.difference(images)
    if missing_targets:
        raise BiohubReferenceEnvironmentError(
            f"Reference container identity omits targets: {sorted(missing_targets)}."
        )
    payload: dict[str, object] = {
        "schema_version": REFERENCE_ENVIRONMENT_SCHEMA_VERSION,
        "contract": contract.contract,
        "contract_sha256": _sha256(contract_path),
        "target": asdict(contract.target),
        "build_container": asdict(contract.container),
        "locks": locks,
        "biotraj": asdict(contract.biotraj),
        "installed_inventory": inventory,
        "installed_inventory_sha256": _digest_object(inventory),
        "pip_check": pip_check,
        "reference_container_target": reference_target,
        "container_identity": normalized_container,
        "container_identity_sha256": manifest_sha256,
        "runtime": _runtime_identity(inventory),
    }
    return validate_biohub_reference_environment_evidence(
        payload,
        repository_root=repository_root,
        contract_path=contract_path,
    )


def validate_biohub_reference_environment_evidence(
    value: object,
    *,
    repository_root: Path,
    contract_path: Path,
) -> dict[str, object]:
    """Validate portable Biohub environment evidence against checked-in locks."""

    fields = {
        "schema_version",
        "contract",
        "contract_sha256",
        "target",
        "build_container",
        "locks",
        "biotraj",
        "installed_inventory",
        "installed_inventory_sha256",
        "pip_check",
        "reference_container_target",
        "container_identity",
        "container_identity_sha256",
        "runtime",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise BiohubReferenceEnvironmentError("Biohub reference environment fields differ.")
    if value["schema_version"] != REFERENCE_ENVIRONMENT_SCHEMA_VERSION:
        raise BiohubReferenceEnvironmentError("Unsupported Biohub environment schema version.")
    try:
        contract = load_biohub_reference_lock_contract(contract_path)
        expected_locks = verify_biohub_reference_lock_contract(repository_root, contract_path)
        expected_inventory = expected_installed_inventory(
            repository_root,
            contract_path,
            profile="final",
        )
    except BiohubReferenceLockError as error:
        raise BiohubReferenceEnvironmentError(str(error)) from error
    expected_static = {
        "contract": contract.contract,
        "contract_sha256": _sha256(contract_path),
        "target": asdict(contract.target),
        "build_container": asdict(contract.container),
        "locks": expected_locks,
        "biotraj": asdict(contract.biotraj),
        "installed_inventory": expected_inventory,
        "installed_inventory_sha256": _digest_object(expected_inventory),
        "pip_check": {
            "status": "accepted-platform-exception",
            "returncode": 1,
            "diagnostics": [
                exception.accepted_diagnostic
                for exception in contract.pip_check_platform_exceptions
            ],
            "accepted_platform_exceptions": [
                asdict(exception) for exception in contract.pip_check_platform_exceptions
            ],
        },
    }
    for field, expected in expected_static.items():
        if value[field] != expected:
            raise BiohubReferenceEnvironmentError(
                f"Biohub reference environment {field} differs from the lock."
            )
    target = value["reference_container_target"]
    if not isinstance(target, str) or target not in BIOHUB_REFERENCE_TARGETS:
        raise BiohubReferenceEnvironmentError("Biohub reference container target is invalid.")
    container = _validated_container_identity(value["container_identity"])
    if value["container_identity"] != container:
        raise BiohubReferenceEnvironmentError("Container identity is not canonically ordered.")
    container_digest = value["container_identity_sha256"]
    if (
        not isinstance(container_digest, str)
        or _SHA256.fullmatch(container_digest) is None
        or container_digest != _digest_object(container)
    ):
        raise BiohubReferenceEnvironmentError("Container identity digest differs.")
    images = container["images"]
    if not isinstance(images, Mapping):
        raise BiohubReferenceEnvironmentError(
            "Reference container identity images must be a mapping."
        )
    if BIOHUB_BUILD_TARGET not in images or target not in images:
        raise BiohubReferenceEnvironmentError("Required Biohub container image is absent.")
    runtime = value["runtime"]
    runtime_fields = {
        "operating_system",
        "architecture",
        "python_implementation",
        "python_version",
        "torch",
        "cuda_runtime",
        "cuda_driver",
        "gpu",
        "uname",
    }
    if not isinstance(runtime, Mapping) or set(runtime) != runtime_fields:
        raise BiohubReferenceEnvironmentError("Biohub runtime identity fields differ.")
    if (
        runtime["operating_system"] != "linux"
        or runtime["architecture"] != "aarch64"
        or runtime["python_implementation"] != "CPython"
        or not isinstance(runtime["python_version"], str)
        or not runtime["python_version"].startswith("3.12.")
        or runtime["torch"] != expected_inventory["torch"]
        or not isinstance(runtime["cuda_runtime"], str)
        or not runtime["cuda_runtime"].startswith("13.0")
        or not isinstance(runtime["cuda_driver"], str)
        or not runtime["cuda_driver"].strip()
    ):
        raise BiohubReferenceEnvironmentError("Biohub runtime differs from the target policy.")
    gpu = runtime["gpu"]
    if (
        not isinstance(gpu, Mapping)
        or set(gpu) != {"name", "capability", "total_memory_bytes"}
        or gpu["name"] != "NVIDIA GH200 480GB"
        or gpu["capability"] != [9, 0]
        or isinstance(gpu["total_memory_bytes"], bool)
        or not isinstance(gpu["total_memory_bytes"], int)
        or gpu["total_memory_bytes"] <= 0
    ):
        raise BiohubReferenceEnvironmentError("Biohub GPU identity differs from GH200/SM90.")
    uname = runtime["uname"]
    if (
        not isinstance(uname, Mapping)
        or set(uname) != {"system", "release", "version", "machine"}
        or uname["system"] != "Linux"
        or uname["machine"] != "aarch64"
        or any(not isinstance(item, str) or not item for item in uname.values())
    ):
        raise BiohubReferenceEnvironmentError("Biohub uname identity is invalid.")
    return {field: value[field] for field in sorted(fields)}


__all__ = [
    "BIOHUB_BUILD_TARGET",
    "BIOHUB_REFERENCE_TARGETS",
    "REFERENCE_ENVIRONMENT_SCHEMA_VERSION",
    "BiohubReferenceEnvironmentError",
    "capture_biohub_reference_environment",
    "validate_biohub_reference_environment_evidence",
]
