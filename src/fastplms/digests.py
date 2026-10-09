"""SHA-256 digests of files and JSON values, the identities FastPLMs records and compares.

This file exists twice, byte for byte: here and as ``features/digests.py``. ``features`` loads as a standalone
package (``foundry.embedding.private_store``) and imports nothing outside itself, so it cannot reach this
module. ``tests/tier1_unit/test_features_package_is_self_contained.py`` fails when the two differ.
"""

from __future__ import annotations

import hashlib

from pathlib import Path
from typing import Any

from .json_files import compact_json


FILE_READ_BYTES = 1024 * 1024


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 of a file's bytes, read in blocks so a checkpoint never sits in memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(FILE_READ_BYTES):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any, *, ensure_ascii: bool = True, allow_nan: bool = True) -> str:
    """Return the SHA-256 of ``value`` in its compact, key-sorted JSON form (``compact_json``)."""

    encoded = compact_json(value, ensure_ascii=ensure_ascii, allow_nan=allow_nan).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
