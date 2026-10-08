"""Write the JSON and byte files the FastPLMs tools produce, in one step and readable by whoever collects them.

Each writer stages a temporary file beside the target and renames it into place (``fastplms.atomic_files``), so a reader
sees the old file or the new one. The staging file is owner-only, so the writer then gives the target the permission
bits that ``open(path, "w")`` would have given it: a container that runs as root writes its reports into a bind mount,
and the ssh user who copies them out reads them through those bits.
"""

from __future__ import annotations

import os

from pathlib import Path
from typing import Any

from fastplms.atomic_files import write_bytes_atomically, write_text_atomically
from fastplms.json_files import indented_json


def write_stored_json(
    path: Path,
    value: Any,
    *,
    sort_keys: bool = True,
    ensure_ascii: bool = True,
    allow_nan: bool = True,
    newline: str | None = None,
) -> None:
    """Replace ``path`` with the two-space-indented JSON of ``value``, creating missing parent directories.

    ``newline`` is the argument of ``open``: ``None`` writes the platform line separator and ``"\\n"`` writes a bare
    line feed.
    """

    text = indented_json(value, ensure_ascii=ensure_ascii, allow_nan=allow_nan, sort_keys=sort_keys)
    write_text_atomically(path, text, newline=newline, create_parent=True)
    _give_plain_file_mode(path)


def write_stored_bytes(path: Path, payload: bytes) -> None:
    """Replace ``path`` with ``payload``, creating missing parent directories."""

    write_bytes_atomically(path, payload, create_parent=True)
    _give_plain_file_mode(path)


def _give_plain_file_mode(path: Path) -> None:
    """Set the permission bits of ``path`` to what a plain ``open(path, "w")`` gives a new file under the umask."""

    umask = os.umask(0)
    os.umask(umask)
    path.chmod(0o666 & ~umask)
