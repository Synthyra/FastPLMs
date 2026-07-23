"""Verify source-only archives without carrying Git metadata.

The portable remote runner intentionally excludes every ``.git`` entry because
Git configuration can contain credentials or workstation-specific paths. This
module defines the small, non-secret attestations that replace those entries:
exact root tracked paths, modes, sizes, symlink targets, and content digests,
plus parent Git-link and checked-out submodule revisions. Root archive metadata
is a content attestation, not proof of a Git commit; Git-free builders therefore
use a content-addressed runtime revision.
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
ARCHIVE_PROVENANCE_SCHEMA = 2
_HEX_DIGEST_LENGTH = 64
_HEX_REVISION_LENGTH = 40
_TREE_DOMAIN = b"fastplms-tracked-submodule-tree-v1\0"
_ROOT_TREE_DOMAIN = b"fastplms-tracked-root-tree-v1\0"


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
        if (
            not path.parts
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != value
            or "\\" in value
            or any(":" in part for part in path.parts)
        ):
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
    if "\\" in target or any(":" in part for part in PurePosixPath(target).parts):
        raise SourceProvenanceError(f"Non-portable symlink target is forbidden: {path}")
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


def _tracked_path(root: Path, relative_name: str) -> Path:
    """Join one portable path while rejecting traversal through parent links."""

    candidate = root
    parts = PurePosixPath(relative_name).parts
    for part in parts[:-1]:
        candidate /= part
        if candidate.is_symlink():
            raise SourceProvenanceError(
                f"Tracked path traverses a symlink: {relative_name!r}"
            )
    return candidate / parts[-1]


def tracked_tree_digest(root: Path, tracked_files: Sequence[str]) -> str:
    """Hash exact tracked files, including portable in-tree symlink targets."""

    root = root.resolve()
    if not root.is_dir():
        raise SourceProvenanceError(f"Tracked tree does not exist: {root}")
    paths = _normalized_paths(tracked_files)
    digest = hashlib.sha256()
    digest.update(_TREE_DOMAIN)
    for relative_name in paths:
        path = _tracked_path(root, relative_name)
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


def tracked_root_inventory(
    root: Path,
    tracked_files: Sequence[str],
) -> dict[str, dict[str, object]]:
    """Describe exact root tracked bytes, portable modes, and symlink targets."""

    root = root.resolve()
    if not root.is_dir():
        raise SourceProvenanceError(f"Tracked root does not exist: {root}")
    result: dict[str, dict[str, object]] = {}
    for relative_name in _normalized_paths(tracked_files):
        path = _tracked_path(root, relative_name)
        try:
            mode = path.lstat().st_mode
        except OSError as error:
            raise SourceProvenanceError(f"Tracked root path is missing: {path}") from error
        if stat.S_ISLNK(mode):
            result[relative_name] = {
                "mode": "120000",
                "target": _safe_symlink_target(root, path),
            }
            continue
        if not stat.S_ISREG(mode):
            raise SourceProvenanceError(f"Tracked root path is not a regular file: {path}")
        content = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                content.update(chunk)
                size += len(chunk)
        result[relative_name] = {
            "mode": "100755" if mode & stat.S_IXUSR else "100644",
            "size": size,
            "sha256": content.hexdigest(),
        }
    return result


def root_inventory_digest(inventory: Mapping[str, Mapping[str, object]]) -> str:
    """Hash a strict, path-ordered root inventory without trusting JSON rendering."""

    paths = _normalized_paths(tuple(inventory))
    digest = hashlib.sha256()
    digest.update(_ROOT_TREE_DOMAIN)
    for relative_name in paths:
        record = inventory[relative_name]
        _frame(digest, b"path", relative_name.encode("utf-8"))
        mode = record.get("mode")
        if mode == "120000" and set(record) == {"mode", "target"}:
            target = record.get("target")
            if not isinstance(target, str):
                raise SourceProvenanceError(
                    f"Tracked root symlink has invalid target: {relative_name!r}"
                )
            _frame(digest, b"mode", b"120000")
            _frame(digest, b"target", target.encode("utf-8"))
            continue
        if mode not in {"100644", "100755"} or set(record) != {
            "mode",
            "size",
            "sha256",
        }:
            raise SourceProvenanceError(
                f"Tracked root file has invalid metadata: {relative_name!r}"
            )
        size = record.get("size")
        sha256 = record.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not _valid_hex(sha256, _HEX_DIGEST_LENGTH)
        ):
            raise SourceProvenanceError(
                f"Tracked root file has invalid size or digest: {relative_name!r}"
            )
        _frame(digest, b"mode", str(mode).encode("ascii"))
        _frame(digest, b"size", size.to_bytes(8, "big"))
        _frame(digest, b"content_sha256", bytes.fromhex(str(sha256)))
    return digest.hexdigest()


def archive_root_record(
    root: Path,
    tracked_files: Sequence[str],
    *,
    head_revision: str,
) -> dict[str, object]:
    """Create the content attestation embedded in one Git-free source archive."""

    if not _valid_hex(head_revision, _HEX_REVISION_LENGTH):
        raise SourceProvenanceError(f"Invalid root revision: {head_revision!r}")
    inventory = tracked_root_inventory(root, tracked_files)
    return {
        "head_revision": head_revision,
        "file_count": len(inventory),
        "files": inventory,
        "tree_sha256": root_inventory_digest(inventory),
    }


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


def render_archive_provenance(
    submodules: Mapping[str, Mapping[str, object]],
    *,
    root: Mapping[str, object],
) -> bytes:
    """Render a deterministic, credential-free archive provenance record."""

    value = {
        "schema_version": ARCHIVE_PROVENANCE_SCHEMA,
        "root": dict(root),
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
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "root",
        "submodules",
    }:
        raise SourceProvenanceError("Archive provenance has an invalid top-level schema")
    if value["schema_version"] != ARCHIVE_PROVENANCE_SCHEMA:
        raise SourceProvenanceError("Archive provenance schema version is unsupported")
    if not isinstance(value["submodules"], dict):
        raise SourceProvenanceError("Archive provenance submodules must be a table")
    if not isinstance(value["root"], dict):
        raise SourceProvenanceError("Archive provenance root must be a table")
    return value


def _valid_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_archived_root(
    source_root: Path,
) -> tuple[str, dict[str, dict[str, object]]]:
    """Validate all attested root tracked bytes and return their content inventory.

    The recorded Git revision is diagnostic transport metadata. Callers must not
    treat it as commit authentication because the extracted tree contains no Git
    objects. Content-addressed callers should derive identity from validated
    payload bytes instead.
    """

    if source_root.is_symlink():
        raise SourceProvenanceError("Archived source root may not be a symlink")
    source_root = source_root.resolve()
    provenance = _load_record(source_root)
    raw_root = provenance["root"]
    expected_fields = {"head_revision", "file_count", "files", "tree_sha256"}
    if not isinstance(raw_root, dict) or set(raw_root) != expected_fields:
        raise SourceProvenanceError("Archive provenance has an invalid root record")
    head_revision = raw_root["head_revision"]
    if not _valid_hex(head_revision, _HEX_REVISION_LENGTH):
        raise SourceProvenanceError("Archive provenance has an invalid root revision")
    raw_files = raw_root["files"]
    if not isinstance(raw_files, dict):
        raise SourceProvenanceError("Archive provenance root files must be a table")
    normalized = _normalized_paths(tuple(raw_files))
    if set(normalized) != set(raw_files):
        raise SourceProvenanceError("Archive provenance root file paths are invalid")
    inventory: dict[str, dict[str, object]] = {}
    for relative_name in normalized:
        raw_record = raw_files[relative_name]
        if not isinstance(raw_record, dict):
            raise SourceProvenanceError(
                f"Archive provenance root file record is invalid: {relative_name!r}"
            )
        inventory[relative_name] = dict(raw_record)
    file_count = raw_root["file_count"]
    if (
        isinstance(file_count, bool)
        or not isinstance(file_count, int)
        or file_count != len(inventory)
    ):
        raise SourceProvenanceError("Archive provenance root file_count differs")
    expected_tree = raw_root["tree_sha256"]
    if not _valid_hex(expected_tree, _HEX_DIGEST_LENGTH):
        raise SourceProvenanceError("Archive provenance root digest is invalid")
    encoded_tree = root_inventory_digest(inventory)
    if encoded_tree != expected_tree:
        raise SourceProvenanceError("Archive provenance root inventory digest differs")
    actual_inventory = tracked_root_inventory(source_root, normalized)
    if actual_inventory != inventory:
        differing = sorted(
            name for name in normalized if actual_inventory.get(name) != inventory.get(name)
        )
        raise SourceProvenanceError(
            "Archived root tracked bytes or modes differ: " + ", ".join(differing[:10])
        )
    actual_tree = root_inventory_digest(actual_inventory)
    if actual_tree != expected_tree:
        raise SourceProvenanceError("Archived root tracked-tree digest differs")
    return str(head_revision), inventory


def validate_archived_submodule(
    source_root: Path,
    *,
    relative_path: str,
    expected_revision: str,
) -> None:
    """Validate one Git-free submodule against its archived attestation."""

    normalized = PurePosixPath(relative_path)
    if (
        not normalized.parts
        or normalized.is_absolute()
        or ".." in normalized.parts
        or normalized.as_posix() != relative_path
        or "\\" in relative_path
        or any(":" in part for part in normalized.parts)
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
    "archive_root_record",
    "render_archive_provenance",
    "root_inventory_digest",
    "tracked_root_inventory",
    "tracked_tree_digest",
    "validate_archived_root",
    "validate_archived_submodule",
]
