"""Typed suite and connection contracts shared by SSH orchestration and reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_CONTROL_TIMEOUT_SECONDS = 300
_TRANSFER_TIMEOUT_SECONDS = 1_800
_BIOHUB_REFERENCE_TARGETS = frozenset({"reference-biohub-esm", "reference-esmfold2"})
_BIOHUB_BUILD_TARGET = "biohub-biotraj-wheel"
_PORTABLE_RELEASE_BACKENDS = ("eager", "sdpa", "flex_attention")


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
