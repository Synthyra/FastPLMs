"""The SHA-256 of a file's bytes, read in chunks so a large file never sits in memory whole."""

from __future__ import annotations

import hashlib
import os

from pathlib import Path


__all__ = ["sha256_file"]


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1 << 20) -> str:
    """The lowercase hexadecimal SHA-256 of the file at `path`, read `chunk_size` bytes at a time.

    `path` is opened as `Path(path)`, so a value that is not a path, such as a file descriptor,
    raises `TypeError`. A `chunk_size` that is not positive raises `ValueError` before the file is
    opened: a read of zero bytes would end the loop at once and return the digest of nothing.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
