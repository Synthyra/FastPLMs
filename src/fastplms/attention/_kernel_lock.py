"""Resolve and validate Hugging Face kernels before importing their binaries."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Any


def require_kernels_package() -> None:
    """Fail early when the precompiled-kernel runtime is not installed."""
    try:
        import kernels  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "Precompiled FlashAttention requires the FastPLMs 'flash' extra."
        ) from error


def _kernel_lock_path() -> Path:
    """Return the lock from an artifact, checkout, or installed distribution."""
    source_path = Path(__file__).resolve()
    candidates = (
        source_path.parents[1] / "kernels.lock",
        source_path.parents[3] / "kernels.lock",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    try:
        distribution = importlib.metadata.distribution("fastplms")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("FastPLMs was installed without kernels.lock.") from error
    for relative in distribution.files or ():
        if relative.name != "kernels.lock":
            continue
        candidate = Path(distribution.locate_file(relative))
        if candidate.is_file():
            return candidate
    raise RuntimeError("The installed FastPLMs distribution does not contain kernels.lock.")


def _locked_entry(lock_path: Path, repository: str) -> dict[str, Any]:
    try:
        data = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unable to read the packaged kernel lock: {lock_path}") from error
    if not isinstance(data, list):
        raise RuntimeError("kernels.lock must contain a JSON list.")
    if any(not isinstance(entry, dict) for entry in data):
        raise RuntimeError("Every kernels.lock entry must be a JSON object.")
    matches = [entry for entry in data if entry.get("repo_id") == repository]
    if len(matches) != 1:
        raise RuntimeError(
            f"kernels.lock must contain exactly one entry for {repository!r}; "
            f"found {len(matches)}."
        )
    return matches[0]


def load_locked_kernel(repository: str, revision: str) -> object:
    """Download, hash-validate, then import one immutable precompiled kernel."""
    require_kernels_package()
    try:
        from kernels import get_local_kernel, install_kernel
        from kernels.lockfile import KernelLock
    except ImportError as error:
        raise RuntimeError(
            "Precompiled FlashAttention requires the FastPLMs 'flash' extra."
        ) from error

    lock_path = _kernel_lock_path()
    kernel_lock = KernelLock.from_json(_locked_entry(lock_path, repository))
    if kernel_lock.sha != revision:
        raise RuntimeError(
            f"The typed manifest pins {repository}@{revision}, but kernels.lock pins "
            f"{kernel_lock.sha}."
        )

    # `install_kernel` downloads data without importing it and validates the
    # selected build against the tracked variant hash. Only then is the exact
    # validated path imported directly.
    validated_path = install_kernel(
        repository,
        revision=kernel_lock.sha,
        variant_locks=kernel_lock.variants,
    )
    return get_local_kernel(validated_path)
