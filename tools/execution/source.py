"""Freeze allowlisted source bytes and identify exactly what a worker receives."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath


SENSITIVE_NAMES = frozenset(
    {
        ".agents",
        ".claude",
        ".codex",
        ".git",
        ".netrc",
        ".npmrc",
        ".pypirc",
        ".git-credentials",
        ".envrc",
        "credentials",
        "credentials.json",
        "id_rsa",
        "id_ed25519",
    }
)
SENSITIVE_SUFFIXES = frozenset({".pem", ".key", ".p12", ".pfx"})
_BUILD_DIRECTORIES = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"})
_SNAPSHOT_DOMAIN = b"fastplms-source-snapshot-v1\0"


def excluded_from_upload(path: Path | PurePosixPath) -> bool:
    """Exclude credentials and generated caches at every directory depth."""
    for part in path.parts:
        name = part.lower()
        if name in SENSITIVE_NAMES or name in _BUILD_DIRECTORIES:
            return True
        if name == ".env" or name.startswith(".env.") or name.endswith(".env"):
            return True
        if PurePosixPath(name).suffix in SENSITIVE_SUFFIXES:
            return True
    return path.suffix.lower() == ".pyc"


def _is_link(path: Path) -> bool:
    metadata = path.lstat()
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


@dataclass(frozen=True)
class SourceFile:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class SourceSnapshot:
    root: Path
    files: tuple[SourceFile, ...]
    tree_sha256: str

    def to_dict(self) -> dict[str, object]:
        """Return a portable inventory without workstation paths."""
        return {
            "schema_version": 1,
            "tree_sha256": self.tree_sha256,
            "file_count": len(self.files),
            "total_bytes": sum(entry.size for entry in self.files),
            "files": [asdict(entry) for entry in self.files],
        }


def require_regular_source(repository: Path, relative: Path) -> Path:
    """Reject source links before opening their contents, including parent links."""
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("Source paths must be relative and may not contain '..'")
    source = repository
    for part in relative.parts:
        source /= part
        if _is_link(source):
            raise RuntimeError(f"Source path traverses a symlink: {relative.as_posix()}")
    if not source.resolve().is_relative_to(repository.resolve()):
        raise RuntimeError(f"Source path leaves the repository: {relative.as_posix()}")
    if not stat.S_ISREG(source.stat().st_mode):
        raise RuntimeError(f"Source path is not a regular file: {relative.as_posix()}")
    return source


def _selected_source_files(
    repository: Path,
    directories: Sequence[str],
    files: Sequence[str],
    exclude: Callable[[Path], bool],
) -> list[Path]:
    def require_readable_directory(error: OSError) -> None:
        raise error

    selected: set[Path] = set()
    for name in (*directories, *files):
        relative = Path(name)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("Source paths must be relative and may not contain '..'")
        if exclude(relative):
            continue
        # Check directory roots and their ancestors before os.walk can follow a link.
        for parent in (*reversed(relative.parents), relative):
            source = repository / parent
            if _is_link(source):
                raise RuntimeError(f"Source path traverses a symlink: {name}")
        if name in files:
            require_regular_source(repository, relative)
            selected.add(relative)
            continue
        directory = repository / relative
        if not directory.is_dir():
            raise FileNotFoundError(f"Source directory is unavailable: {name}")
        for current, children, file_names in os.walk(
            directory, followlinks=False, onerror=require_readable_directory,
        ):
            parent = Path(current).relative_to(repository)
            children[:] = sorted(child for child in children if not exclude(parent / child))
            for child in children:
                path = repository / parent / child
                if _is_link(path):
                    raise RuntimeError(f"Source path traverses a symlink: {parent / child}")
            for child in file_names:
                path = parent / child
                if not exclude(path):
                    require_regular_source(repository, path)
                    selected.add(path)
    return sorted(selected, key=lambda path: path.as_posix())


def stage_source_snapshot(
    repository: Path,
    destination: Path,
    *,
    directories: Sequence[str],
    files: Sequence[str],
    exclude: Callable[[Path], bool] = excluded_from_upload,
) -> SourceSnapshot:
    """Copy approved files once; hashes describe the copied bytes, including dirty edits."""
    repository = repository.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Source snapshot already exists: {destination}")
    for directory in directories:
        if destination.is_relative_to((repository / directory).resolve()):
            raise ValueError("Source snapshot destination must be outside selected inputs")
    selected = _selected_source_files(
        repository,
        directories,
        files,
        lambda path: excluded_from_upload(path) or exclude(path),
    )
    destination.mkdir(parents=True)
    entries = []
    digest = hashlib.sha256(_SNAPSHOT_DOMAIN)
    for relative in selected:
        source = require_regular_source(repository, relative)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        content = hashlib.sha256()
        size = 0
        with source.open("rb") as source_stream, target.open("xb") as target_stream:
            while chunk := source_stream.read(1024 * 1024):
                target_stream.write(chunk)
                content.update(chunk)
                size += len(chunk)
        shutil.copymode(source, target)
        entry = SourceFile(relative.as_posix(), size, content.hexdigest())
        entries.append(entry)
        for value in (entry.path.encode(), size.to_bytes(8, "big"), content.digest()):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    return SourceSnapshot(destination, tuple(entries), digest.hexdigest())
