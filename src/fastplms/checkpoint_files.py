"""Verify local registry-pinned checkpoint files without importing release tooling."""

from __future__ import annotations

import hashlib

from pathlib import Path, PurePosixPath

from .digests import FILE_READ_BYTES, file_sha256
from .registry import CheckpointSource


class ArtifactError(RuntimeError):
    """Raised when an artifact cannot be built or validated safely."""


def hash_file(path: Path, algorithm: str = "sha256") -> str:
    """Return a normal SHA-256 or Git-blob SHA-1 digest for one file."""
    if algorithm == "sha256":
        return file_sha256(path)
    if algorithm != "git-sha1":
        raise ArtifactError(f"Unsupported digest algorithm: {algorithm!r}")
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as handle:
        while chunk := handle.read(FILE_READ_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(snapshot: Path, source: CheckpointSource) -> None:
    """Verify every manifest-pinned file in a local checkpoint snapshot."""
    snapshot = snapshot.resolve()
    if not snapshot.is_dir():
        raise ArtifactError(f"Checkpoint snapshot does not exist: {snapshot}")
    failures: list[str] = []
    for expected in source.files:
        path = snapshot.joinpath(*PurePosixPath(expected.path).parts)
        if not path.is_file():
            failures.append(f"missing {expected.path}")
            continue
        actual = hash_file(path, expected.algorithm)
        if actual != expected.digest:
            failures.append(
                f"{expected.path}: expected {expected.encoded}, "
                f"received {expected.algorithm}:{actual}"
            )
    if failures:
        detail = "\n  - ".join(failures)
        raise ArtifactError(f"Checkpoint verification failed for {source.repo_id}:\n  - {detail}")
