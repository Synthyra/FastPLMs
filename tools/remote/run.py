"""Synchronize a clean workspace and run FastPLMs containers over SSH.

The runner accepts the SSH host and identity only at invocation time. It does
not read credential files or persist workstation details in the repository.
Ignored evidence is copied only when its bytes match the tracked evidence manifest.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import secrets
import shlex
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from tools.execution.source import (
    SENSITIVE_SUFFIXES as SENSITIVE_SUFFIXES,
    excluded_from_upload as _is_sensitive,  # noqa: F401  Historical import surface.
)

from .contracts import (
    _CONTROL_TIMEOUT_SECONDS,
    _TRANSFER_TIMEOUT_SECONDS,
    RunnerConfig as RunnerConfig,
    Suite as Suite,
)
from .reports import _artifact_tree_summary as _artifact_tree_summary, _run_report as _run_report
from .source import (
    _git_head_revision,
    _is_tracked_file,
    _require_clean_repository as _require_clean_repository,
    _require_matching_archive_digest as _require_matching_archive_digest,
    create_source_archive as create_source_archive,
)
from .suites import SUITES as SUITES, _RELEASE_LOCAL_PARITY_TESTS as _RELEASE_LOCAL_PARITY_TESTS


HOST_PATTERN = re.compile(r"^[A-Za-z0-9_.@:\-]+$")
RUN_PATTERN = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{16}$")
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
_MACHINE_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
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
    if platform_name not in {"linux/arm64", "linux/amd64"} or platform_name != (
        f"linux/{architecture}"
    ):
        return {
            "schema_version": 1,
            "status": "failed",
            "policy": "native-no-flash-download-v1",
            "platform": platform_name,
            "selected_backends": list(requested_backends),
            "network_downloads": False,
            "source_builds": False,
            "reason": "Validation requires a native Linux Docker platform.",
            "backends": {},
        }

    requested = tuple(requested_backends)
    if len(set(requested)) != len(requested):
        return {
            "schema_version": 1,
            "status": "failed",
            "policy": "native-no-flash-download-v1",
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
                "The release matrix reuses prior revision-pinned focused FA2 "
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
                "for the current release image."
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
        "policy": "native-no-flash-download-v1",
        "platform": platform_name,
        "selected_backends": list(requested),
        "excluded_backends": [
            backend for backend in backend_records if backend not in requested
        ],
        "network_downloads": False,
        "source_builds": False,
        "reason": (
            "Requested backends are unavailable under the native container policy: "
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


def _run_id(_repository: Path) -> str:
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{secrets.token_hex(8)}"


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
