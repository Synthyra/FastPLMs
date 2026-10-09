"""Replace a file in one step: readers see the old bytes or the new bytes, never a partial write."""

from __future__ import annotations

import os
import tempfile

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any


def write_bytes_atomically(path: Path, payload: bytes, *, create_parent: bool = False) -> None:
    """Write ``payload`` to a temporary file beside ``path``, flush it to disk, then rename it over ``path``.

    A failed write removes the temporary file and leaves ``path`` untouched. The parent directory must
    exist unless ``create_parent`` is true. The temporary file sits in the same directory so the rename
    never crosses a file system.
    """

    with _staged_handle(path, "wb", create_parent=create_parent) as handle:
        handle.write(payload)


def write_text_atomically(
    path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    newline: str | None = None,
    create_parent: bool = False,
) -> None:
    """Write ``text`` as ``write_bytes_atomically`` does.

    ``newline`` is the argument of ``open``: ``None`` translates ``"\\n"`` to the platform separator and
    ``"\\n"`` writes it unchanged.
    """

    with _staged_handle(
        path, "w", create_parent=create_parent, encoding=encoding, newline=newline
    ) as handle:
        handle.write(text)


@contextmanager
def _staged_handle(
    path: Path, mode: str, *, create_parent: bool, **open_arguments: Any
) -> Iterator[IO[Any]]:
    """Yield a handle to a temporary file; on a clean exit flush it and rename it over ``path``."""

    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, mode, **open_arguments) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
