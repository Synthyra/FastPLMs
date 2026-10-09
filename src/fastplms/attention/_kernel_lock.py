"""Resolve and validate Hugging Face kernels before importing their binaries."""

from __future__ import annotations

import hashlib
import json
import os
import re

from pathlib import Path
from typing import Any


# kernels.lock records, per locked build variant, one SHA-256 over the variant's relative
# file paths and the Git or Git LFS object IDs of their contents. kernels 0.17 no longer
# writes or checks these digests, so this module checks them before anything is imported.
_VARIANT_HASH_TYPE = "git_lfs_concat"
_VARIANT_DIGEST = re.compile(r"sha256-[0-9a-f]{64}")
# The Hub cache names a Git blob by its 40-character SHA-1 object ID and Git LFS content
# by its 64-character SHA-256.
_GIT_OBJECT_ID_LENGTH = 40
_LFS_OBJECT_ID_LENGTH = 64


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


def _locked_variant_digests(entry: dict[str, Any]) -> dict[str, str]:
    """Each locked build variant of one kernels.lock entry, mapped to its digest."""
    variants = entry.get("variants")
    if not isinstance(variants, dict) or not variants:
        raise RuntimeError(f"kernels.lock locks no build variants for {entry.get('repo_id')!r}.")
    digests: dict[str, str] = {}
    for variant_name, variant_lock in variants.items():
        expected_hash = variant_lock.get("hash") if isinstance(variant_lock, dict) else None
        if (
            not isinstance(expected_hash, str)
            or _VARIANT_DIGEST.fullmatch(expected_hash) is None
            or variant_lock.get("hash_type") != _VARIANT_HASH_TYPE
        ):
            raise RuntimeError(f"The kernel lock for {variant_name} has no valid SHA-256 digest.")
        digests[variant_name] = expected_hash
    return digests


def _git_blob_object_id(contents: bytes) -> bytes:
    """Return the SHA-1 object ID Git assigns to a blob with these contents."""
    return hashlib.sha1(b"blob %d\0" % len(contents) + contents).digest()


def validate_variant_digest(snapshot: Path, variant_name: str, expected_hash: str) -> None:
    """Check one cached build variant against its kernels.lock digest before import.

    Snapshot files link into the Hub cache's content-addressed blobs. Each linked file
    contributes its path relative to the variant, then the object ID its contents hash to:
    a Git blob ID when the cache names the blob by SHA-1, a Git LFS SHA-256 otherwise.
    Files that are not links are skipped, because importing a kernel writes bytecode
    beside it.
    """
    variant_root = snapshot / "build" / variant_name
    linked_files: list[tuple[bytes, Path]] = []
    for directory, _, file_names in os.walk(variant_root):
        for file_name in file_names:
            path = Path(directory) / file_name
            if path.is_symlink():
                relative_name = path.relative_to(variant_root).as_posix().encode("utf-8")
                linked_files.append((relative_name, path))

    digest = hashlib.sha256()
    for relative_name, path in sorted(linked_files):
        contents = path.read_bytes()
        object_id_length = len(path.resolve().name)
        if object_id_length == _GIT_OBJECT_ID_LENGTH:
            object_id = _git_blob_object_id(contents)
        elif object_id_length == _LFS_OBJECT_ID_LENGTH:
            object_id = hashlib.sha256(contents).digest()
        else:
            raise RuntimeError(f"Unexpected Hub cache blob name behind {path}.")
        digest.update(relative_name)
        digest.update(object_id)

    received_hash = f"sha256-{digest.hexdigest()}"
    if received_hash != expected_hash:
        raise RuntimeError(
            f"The cached kernel variant {variant_name} hashes to {received_hash}, but "
            f"kernels.lock records {expected_hash}."
        )


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
            f"{cache_root}. Load the kernel once with Hub access before enabling offline mode."
        )
    if repository_root not in snapshot.resolve().parents:
        raise RuntimeError(f"Refusing kernel snapshot outside its cache repository: {snapshot}")
    return snapshot


def _load_offline_locked_kernel(
    repository: str,
    revision: str,
    variant_digests: dict[str, str],
) -> object:
    """Validate and import the one compatible variant from a sparse Hub snapshot."""
    snapshot = _offline_snapshot_path(repository, revision)
    build_root = snapshot / "build"
    if not build_root.is_dir():
        raise RuntimeError(f"The cached kernel snapshot has no build directory: {snapshot}")

    cached_names = sorted(entry.name for entry in build_root.iterdir() if entry.is_dir())
    unexpected = sorted(set(cached_names).difference(variant_digests))
    if unexpected:
        raise RuntimeError(
            f"The cached {repository}@{revision} snapshot contains unlocked variants: "
            f"{', '.join(unexpected)}"
        )

    try:
        from kernels import get_local_kernel
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

    # Hash validation deliberately happens before import. It reads the sparse snapshot
    # directly, so no Hub API judges whether the snapshot is complete.
    validate_variant_digest(snapshot, variant_name, variant_digests[variant_name])
    return get_local_kernel(build_root / variant_name)


def _compatible_locked_variants(repository: str, variant_names: list[str]) -> list[str]:
    """Return the locked build variants `kernels` can load on this system, preferred first."""
    try:
        from kernels.variants import parse_variant, resolve_variants
    except ImportError as error:
        raise RuntimeError(
            "Precompiled FlashAttention requires requirements/features/flash.in."
        ) from error

    try:
        locked_variants = [parse_variant(variant_name) for variant_name in variant_names]
    except ValueError as error:
        raise RuntimeError(f"kernels.lock contains an invalid {repository} variant.") from error
    compatible, _ = resolve_variants(locked_variants)
    return [variant.variant_str for variant in compatible]


def _preferred_locked_variant(repository: str, revision: str, variant_names: list[str]) -> str:
    """Return the locked build variant `kernels` prefers on this system."""
    compatible = _compatible_locked_variants(repository, variant_names)
    if not compatible:
        raise RuntimeError(
            f"kernels.lock locks no build of {repository}@{revision} for this system; "
            f"locked variants: {', '.join(sorted(variant_names))}."
        )
    return compatible[0]


def _pinned_variant_digests(repository: str, revision: str) -> dict[str, str]:
    """Return the locked build digests of a kernel whose kernels.lock entry pins `revision`."""
    entry = _locked_entry(_kernel_lock_path(), repository)
    locked_revision = entry.get("sha")
    if locked_revision != revision:
        raise RuntimeError(
            f"The typed manifest pins {repository}@{revision}, but kernels.lock pins "
            f"{locked_revision}."
        )
    return _locked_variant_digests(entry)


def locked_variant_for_this_system(repository: str, revision: str) -> str | None:
    """Return the locked build `kernels` would load here, or None when kernels.lock pins none.

    Only kernels.lock is read, so a caller can tell a platform the lock does not cover
    apart from a download, digest, or import failure.
    """
    variant_digests = _pinned_variant_digests(repository, revision)
    compatible = _compatible_locked_variants(repository, list(variant_digests))
    return compatible[0] if compatible else None


def load_locked_kernel(repository: str, revision: str) -> object:
    """Download, hash-validate, then import one immutable precompiled kernel."""
    require_kernels_package()
    try:
        from huggingface_hub import snapshot_download
        from kernels import get_local_kernel
    except ImportError as error:
        raise RuntimeError(
            "Precompiled FlashAttention requires requirements/features/flash.in."
        ) from error

    variant_digests = _pinned_variant_digests(repository, revision)

    if _offline_mode():
        return _load_offline_locked_kernel(repository, revision, variant_digests)

    # Only a locked build can be selected, and only its files are downloaded, at the
    # immutable revision. The download imports nothing; the digest check runs first.
    variant_name = _preferred_locked_variant(repository, revision, list(variant_digests))
    snapshot = Path(
        snapshot_download(
            repository,
            repo_type="kernel",
            revision=revision,
            allow_patterns=[f"build/{variant_name}/*"],
            cache_dir=os.environ.get("KERNELS_CACHE") or None,
        )
    )
    validate_variant_digest(snapshot, variant_name, variant_digests[variant_name])
    return get_local_kernel(snapshot / "build" / variant_name)
