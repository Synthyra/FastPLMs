"""Resolve and validate Hugging Face kernels before importing their binaries."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def require_kernels_package() -> None:
    """Fail early when the precompiled-kernel runtime is not installed."""
    try:
        import kernels  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "Precompiled FlashAttention requires requirements/features/flash.in."
        ) from error


def _kernel_lock_path() -> Path:
    """Return the kernel lock from a Hub artifact or source checkout."""
    source_path = Path(__file__).resolve()
    for candidate in (
        source_path.parents[1] / "kernels.lock",
        source_path.parents[3] / "kernels.lock",
    ):
        if candidate.is_file():
            return candidate

    raise RuntimeError(
        "kernels.lock is missing from the Hugging Face artifact or source checkout."
    )


def _locked_entry(lock_path: Path, repository: str) -> dict[str, Any]:
    try:
        lock_entries = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unable to read the kernel lock: {lock_path}") from error
    if not isinstance(lock_entries, list):
        raise RuntimeError("kernels.lock must contain a JSON list.")
    if any(not isinstance(entry, dict) for entry in lock_entries):
        raise RuntimeError("Every kernels.lock entry must be a JSON object.")
    matches = [entry for entry in lock_entries if entry.get("repo_id") == repository]
    if len(matches) != 1:
        raise RuntimeError(
            f"kernels.lock must contain exactly one entry for {repository!r}; found {len(matches)}."
        )
    return matches[0]


def _offline_mode() -> bool:
    """Return whether Hub access was explicitly disabled for this process."""

    enabled_values = {"1", "on", "true", "yes"}
    return any(
        os.environ.get(name, "").strip().lower() in enabled_values
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def _offline_snapshot_path(repository: str, revision: str) -> Path:
    """Locate one exact, possibly sparse, kernel snapshot without using Hub APIs."""

    try:
        from huggingface_hub import constants
        from huggingface_hub.file_download import repo_folder_name
    except ImportError as error:
        raise RuntimeError("Offline kernel loading requires huggingface-hub.") from error

    cache_root = Path(os.environ.get("KERNELS_CACHE") or constants.HF_HUB_CACHE).resolve()
    repository_root = (
        cache_root / repo_folder_name(repo_id=repository, repo_type="kernel")
    ).resolve()
    snapshot = repository_root / "snapshots" / revision
    if not snapshot.is_dir():
        raise RuntimeError(
            f"The exact offline kernel snapshot {repository}@{revision} is not cached under "
            f"{cache_root}. Run `kernels download` before enabling offline mode."
        )
    if repository_root not in snapshot.resolve().parents:
        raise RuntimeError(f"Refusing kernel snapshot outside its cache repository: {snapshot}")
    return snapshot


def _load_offline_locked_kernel(
    repository: str,
    revision: str,
    variant_locks: dict[str, object],
) -> object:
    """Validate and import the one compatible variant from a sparse Hub snapshot."""
    snapshot = _offline_snapshot_path(repository, revision)
    build_root = snapshot / "build"
    if not build_root.is_dir():
        raise RuntimeError(f"The cached kernel snapshot has no build directory: {snapshot}")

    cached_names = sorted(entry.name for entry in build_root.iterdir() if entry.is_dir())
    unexpected = sorted(set(cached_names).difference(variant_locks))
    if unexpected:
        raise RuntimeError(
            f"The cached {repository}@{revision} snapshot contains unlocked variants: "
            f"{', '.join(unexpected)}"
        )

    try:
        from kernels import get_local_kernel
        from kernels.utils import validate_kernel
        from kernels.variants import get_variants_local, resolve_variants
    except ImportError as error:
        raise RuntimeError(
            "Precompiled FlashAttention requires requirements/features/flash.in."
        ) from error

    cached_variants = get_variants_local(build_root)
    parsed_names = {variant.variant_str for variant in cached_variants}
    invalid = sorted(set(cached_names).difference(parsed_names))
    if invalid:
        raise RuntimeError(
            f"The cached {repository}@{revision} snapshot contains invalid variants: "
            f"{', '.join(invalid)}"
        )

    compatible, _ = resolve_variants(cached_variants)
    if len(compatible) != 1:
        names = ", ".join(variant.variant_str for variant in compatible) or "none"
        raise RuntimeError(
            f"Expected exactly one compatible cached variant for {repository}@{revision}; "
            f"found {names}."
        )
    variant_name = compatible[0].variant_str
    variant_lock = variant_locks.get(variant_name)
    expected_hash = getattr(variant_lock, "hash", None)
    if not isinstance(expected_hash, str) or not expected_hash.startswith("sha256-"):
        raise RuntimeError(f"The kernel lock for {variant_name} has no valid SHA-256 digest.")

    # Hash validation deliberately happens before import. This operates on the
    # sparse snapshot produced by `kernels download` and avoids Hub 1.23's
    # full-snapshot completeness check in offline mode.
    validate_kernel(repo_path=snapshot, variant=variant_name, hash=expected_hash)
    return get_local_kernel(build_root / variant_name)


def load_locked_kernel(repository: str, revision: str) -> object:
    """Download, hash-validate, then import one immutable precompiled kernel."""
    require_kernels_package()
    try:
        from kernels import get_local_kernel, install_kernel
        from kernels.lockfile import KernelLock
    except ImportError as error:
        raise RuntimeError(
            "Precompiled FlashAttention requires requirements/features/flash.in."
        ) from error

    lock_path = _kernel_lock_path()
    kernel_lock = KernelLock.from_json(_locked_entry(lock_path, repository))
    if kernel_lock.sha != revision:
        raise RuntimeError(
            f"The typed manifest pins {repository}@{revision}, but kernels.lock pins "
            f"{kernel_lock.sha}."
        )

    if _offline_mode():
        return _load_offline_locked_kernel(repository, revision, kernel_lock.variants)

    # `install_kernel` downloads data without importing it and validates the
    # selected build against the tracked variant hash. Only then is the exact
    # validated path imported directly. Offline mode uses the sparse-cache
    # resolver above because Hub 1.23 rejects partial snapshots as incomplete.
    validated_path = install_kernel(
        repository,
        revision=kernel_lock.sha,
        variant_locks=kernel_lock.variants,
    )
    return get_local_kernel(validated_path)
