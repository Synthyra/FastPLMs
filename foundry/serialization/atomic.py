"""Atomic file writes: a reader sees the old file or the whole new one, never part of one.

Every writer here fills a hidden temporary sibling in the target's directory, flushes it to disk,
renames it over the target with `os.replace`, and removes the temporary file when anything fails,
so a crash or an exception leaves the target as it was. Writing in place is unsafe in a Dropbox
tree, where the sync client can upload a half-written file, and a rename in the same directory
is atomic where a rename across volumes is not.

`atomic_replace` hands the temporary path to any writer that takes a path, such as Parquet,
NumPy, PyTorch or safetensors, and the typed writers on top of it cover bytes, text and JSON.

Text is encoded and written as given, with no newline translation: `"\\n"` stays one byte on
Windows, where `Path.write_text` and a text-mode `open` write `"\\r\\n"`.

Standard library only; `foundry.serialization` re-exports these names.
"""

from __future__ import annotations

import json
import os
import time
import uuid

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from foundry.compression import PathLike


# Windows refuses a rename over a file that Dropbox, an indexer or an antivirus scan holds for a
# moment after it changes, so a replace there retries on `PermissionError` before it gives up.
RETRY_REPLACE = os.name == "nt"
REPLACE_ATTEMPTS = 20
REPLACE_WAIT_SECONDS = 0.25


@contextmanager
def atomic_replace(path: PathLike, *, durable: bool = True) -> Iterator[Path]:
    """Yield a temporary sibling of `path` for a writer to fill, then rename it over `path`.

    The parent directory is created. When the block exits cleanly the temporary file is flushed
    to disk (`durable=True`) and renamed onto `path`. When the block raises, or the rename fails,
    the temporary file is removed and `path` is untouched. A block that writes no file raises
    `FileNotFoundError`. `durable=False` skips the flush for a file rewritten often enough that
    its durability is not worth the wait. The directory entry is not flushed.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        yield temporary
        if durable:
            _flush_file(temporary)
        _replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_bytes_atomic(path: PathLike, content: bytes, *, durable: bool = True) -> None:
    """Replace `path` with `content` atomically."""
    with atomic_replace(path, durable=durable) as temporary:
        temporary.write_bytes(content)


def write_text_atomic(path: PathLike, text: str, *, encoding: str = "utf-8", durable: bool = True) -> None:
    """Replace `path` with `text` encoded as `encoding`, atomically and without newline translation."""
    write_bytes_atomic(path, text.encode(encoding), durable=durable)


def write_json_atomic(
    path: PathLike,
    payload: Any,
    *,
    sort_keys: bool = True,
    allow_nan: bool = True,
    ensure_ascii: bool = True,
    default: Callable[[Any], Any] | None = None,
    durable: bool = True,
) -> None:
    """Replace `path` with `payload` as JSON indented by 2, keys sorted, ending in one newline.

    The payload is serialized before any file is touched, so one that does not serialize raises
    `TypeError` or `ValueError` and leaves `path` as it was. This is not `write_json_dict_atomic`,
    which also writes `.gz`, takes only a dictionary, keeps insertion order and ends no line.
    """
    text = json.dumps(payload, indent=2, sort_keys=sort_keys, allow_nan=allow_nan, ensure_ascii=ensure_ascii, default=default)
    write_text_atomic(path, text + "\n", durable=durable)


def _replace(temporary: Path, target: Path) -> None:
    if not RETRY_REPLACE:
        os.replace(temporary, target)
        return
    for attempt in range(1, REPLACE_ATTEMPTS + 1):
        try:
            os.replace(temporary, target)
            return
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS:
                raise
            time.sleep(REPLACE_WAIT_SECONDS)


def _flush_file(path: Path) -> None:
    """Force a finished file's bytes to disk, whichever writer produced it."""
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


__all__ = ["atomic_replace", "write_bytes_atomic", "write_json_atomic", "write_text_atomic"]
