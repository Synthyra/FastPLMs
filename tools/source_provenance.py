"""Verify source-only archives without carrying Git metadata.

The portable remote runner intentionally excludes every ``.git`` entry because
Git configuration can contain credentials or workstation-specific paths. This
module defines the small, non-secret attestation that replaces those entries:
the parent Git-link revision, the checked-out submodule revision, and a
digest over the exact tracked tree copied into the archive.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

ARCHIVE_PROVENANCE_NAME = ".fastplms-source-provenance.json"
ARCHIVE_PROVENANCE_SCHEMA = 1
_HEX_DIGEST_LENGTH = 64
_HEX_REVISION_LENGTH = 40
_TREE_DOMAIN = b"fastplms-tracked-submodule-tree-v1\0"


class SourceProvenanceError(RuntimeError):
    """Raised when a source archive cannot prove its submodule identity."""


def _frame(digest: Any, label: bytes, payload: bytes) -> None:
    digest.update(len(label).to_bytes(4, "big"))
    digest.update(label)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _normalized_paths(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise SourceProvenanceError("tracked_files must be a sequence of paths")
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise SourceProvenanceError("tracked_files contains an invalid path")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
            raise SourceProvenanceError(f"Non-portable tracked path: {value!r}")
        if any(part.lower() == ".git" for part in path.parts):
            raise SourceProvenanceError(f"Git metadata is forbidden in source archives: {value!r}")
        normalized.append(value)
    ordered = tuple(sorted(normalized))
    if len(ordered) != len(set(ordered)):
        raise SourceProvenanceError("tracked_files contains duplicate paths")
    return ordered


def _safe_symlink_target(root: Path, path: Path) -> str:
    target = os.readlink(path)
    target_path = Path(target)
    if target_path.is_absolute():
        raise SourceProvenanceError(f"Absolute symlink is forbidden: {path}")
    resolved_root = root.resolve()
    resolved_target = (path.parent / target_path).resolve(strict=False)
    try:
        resolved_target.relative_to(resolved_root)
    except ValueError as error:
        raise SourceProvenanceError(f"Symlink escapes its submodule tree: {path}") from error
    return target


def tracked_tree_digest(root: Path, tracked_files: Sequence[str]) -> str:
    """Hash exact tracked files, including portable in-tree symlink targets."""

    root = root.resolve()
    if not root.is_dir():
        raise SourceProvenanceError(f"Tracked tree does not exist: {root}")
    paths = _normalized_paths(tracked_files)
    digest = hashlib.sha256()
    digest.update(_TREE_DOMAIN)
    for relative_name in paths:
        path = root.joinpath(*PurePosixPath(relative_name).parts)
        try:
            mode = path.lstat().st_mode
        except OSError as error:
            raise SourceProvenanceError(f"Tracked path is missing: {path}") from error
        _frame(digest, b"path", relative_name.encode("utf-8"))
        if stat.S_ISLNK(mode):
            target = _safe_symlink_target(root, path)
            _frame(digest, b"type", b"symlink")
            _frame(digest, b"target", target.encode("utf-8"))
            continue
        if not stat.S_ISREG(mode):
            raise SourceProvenanceError(f"Tracked path is not a regular file: {path}")
        content = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                content.update(chunk)
                size += len(chunk)
        _frame(digest, b"type", b"file")
        _frame(digest, b"size", size.to_bytes(8, "big"))
        _frame(digest, b"content_sha256", content.digest())
    return digest.hexdigest()


def actual_tree_paths(root: Path) -> tuple[str, ...]:
    """Return every file or symlink present in an extracted submodule tree."""

    root = root.resolve()
    if not root.is_dir():
        raise SourceProvenanceError(f"Archived submodule does not exist: {root}")
    result: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part.lower() == ".git" for part in relative.parts):
            raise SourceProvenanceError(f"Archived submodule contains Git metadata: {path}")
        if path.is_symlink() or path.is_file():
            result.append(relative.as_posix())
    return tuple(sorted(result))


def render_archive_provenance(submodules: Mapping[str, Mapping[str, object]]) -> bytes:
    """Render a deterministic, credential-free archive provenance record."""

    value = {
        "schema_version": ARCHIVE_PROVENANCE_SCHEMA,
        "submodules": {path: dict(record) for path, record in sorted(submodules.items())},
    }
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _load_record(source_root: Path) -> dict[str, Any]:
    path = source_root / ARCHIVE_PROVENANCE_NAME
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise SourceProvenanceError(
            f"Git-free source tree is missing archive provenance: {path}"
        ) from error
    if not stat.S_ISREG(mode):
        raise SourceProvenanceError(f"Archive provenance is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SourceProvenanceError(
            f"Git-free source tree is missing valid archive provenance: {path}"
        ) from error
    if not isinstance(value, dict) or set(value) != {"schema_version", "submodules"}:
        raise SourceProvenanceError("Archive provenance has an invalid top-level schema")
    if value["schema_version"] != ARCHIVE_PROVENANCE_SCHEMA:
        raise SourceProvenanceError("Archive provenance schema version is unsupported")
    if not isinstance(value["submodules"], dict):
        raise SourceProvenanceError("Archive provenance submodules must be a table")
    return value


def _valid_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_archived_submodule(
    source_root: Path,
    *,
    relative_path: str,
    expected_revision: str,
) -> None:
    """Validate one Git-free submodule against its archived attestation."""

    normalized = PurePosixPath(relative_path)
    if (
        normalized.is_absolute()
        or ".." in normalized.parts
        or normalized.as_posix() != relative_path
    ):
        raise SourceProvenanceError(f"Invalid archived submodule path: {relative_path!r}")
    if not _valid_hex(expected_revision, _HEX_REVISION_LENGTH):
        raise SourceProvenanceError(f"Invalid expected revision: {expected_revision!r}")

    provenance = _load_record(source_root.resolve())
    raw_record = provenance["submodules"].get(relative_path)
    expected_fields = {
        "file_count",
        "gitlink_revision",
        "head_revision",
        "tracked_files",
        "tree_sha256",
    }
    if not isinstance(raw_record, dict) or set(raw_record) != expected_fields:
        raise SourceProvenanceError(
            f"Archive provenance is missing a complete record for {relative_path!r}"
        )
    if (
        raw_record["gitlink_revision"] != expected_revision
        or raw_record["head_revision"] != expected_revision
    ):
        raise SourceProvenanceError(
            f"Archived submodule {relative_path!r} does not match {expected_revision}"
        )
    tracked_files = raw_record["tracked_files"]
    if not isinstance(tracked_files, list):
        raise SourceProvenanceError(f"Archived submodule {relative_path!r} has no file list")
    normalized_files = _normalized_paths(tracked_files)
    file_count = raw_record["file_count"]
    if isinstance(file_count, bool) or not isinstance(file_count, int):
        raise SourceProvenanceError(f"Archived submodule {relative_path!r} has invalid file_count")
    if file_count != len(normalized_files):
        raise SourceProvenanceError(f"Archived submodule {relative_path!r} file_count differs")
    expected_tree = raw_record["tree_sha256"]
    if not _valid_hex(expected_tree, _HEX_DIGEST_LENGTH):
        raise SourceProvenanceError(f"Archived submodule {relative_path!r} has invalid digest")

    source_root = source_root.resolve()
    checkout = source_root
    for part in normalized.parts:
        checkout /= part
        if checkout.is_symlink():
            raise SourceProvenanceError(
                f"Archived submodule path traverses a symlink: {relative_path!r}"
            )
    try:
        checkout.resolve(strict=False).relative_to(source_root)
    except ValueError as error:
        raise SourceProvenanceError(
            f"Archived submodule escapes the source tree: {relative_path!r}"
        ) from error
    actual_files = actual_tree_paths(checkout)
    if actual_files != normalized_files:
        missing = sorted(set(normalized_files).difference(actual_files))
        extra = sorted(set(actual_files).difference(normalized_files))
        raise SourceProvenanceError(
            f"Archived submodule {relative_path!r} file inventory differs; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    actual_digest = tracked_tree_digest(checkout, normalized_files)
    if actual_digest != expected_tree:
        raise SourceProvenanceError(
            f"Archived submodule {relative_path!r} tracked-tree digest differs"
        )


__all__ = [
    "ARCHIVE_PROVENANCE_NAME",
    "ARCHIVE_PROVENANCE_SCHEMA",
    "SourceProvenanceError",
    "actual_tree_paths",
    "render_archive_provenance",
    "tracked_tree_digest",
    "validate_archived_submodule",
]
