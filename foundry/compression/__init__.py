"""Gzip-transparent file openers, and a zip archive of a directory.

A path ending in `.gz`, in any letter case, opens through `gzip`; any other path opens
directly. A caller then reads and writes a cache the same way however it was stored. A gzip
write defaults to level 1, `GZIP_COMPRESSLEVEL`, which trades file size for speed on the large
caches the dataset builders and Atlas write.
"""

from __future__ import annotations

import gzip
import os
import shutil

from typing import IO


GZIP_COMPRESSLEVEL = 1

PathLike = str | os.PathLike[str]


def is_gzip_path(path: PathLike) -> bool:
    return str(path).lower().endswith(".gz")


def open_text_maybe_gzip(
    path: PathLike, mode: str = "rt", encoding: str | None = None, newline: str | None = None
) -> IO[str]:
    """Open `path` in a text mode, which must name `t` explicitly.

    `newline` is `open`'s: left as None, a write ends each line in the platform's separator, a carriage
    return and line feed on Windows; "\\n" writes line feeds only.
    """
    assert "b" not in mode, "open_text_maybe_gzip expects a text mode"
    assert "t" in mode, "open_text_maybe_gzip expects a text mode including 't'"

    if is_gzip_path(path):
        return gzip.open(path, mode, encoding=encoding, newline=newline)
    return open(path, mode, encoding=encoding, newline=newline)


def open_binary_maybe_gzip(path: PathLike, mode: str = "rb", compresslevel: int | None = None) -> IO[bytes]:
    """Open `path` in a binary mode. `compresslevel` applies only to a gzip write."""
    assert "b" in mode, "open_binary_maybe_gzip expects a binary mode"

    if not is_gzip_path(path):
        return open(path, mode)
    if any(flag in mode for flag in "wax"):
        return gzip.open(path, mode, compresslevel=GZIP_COMPRESSLEVEL if compresslevel is None else compresslevel)
    return gzip.open(path, mode)


def zip_directory(output_dir: PathLike, archive_suffix: str = "_archive") -> str:
    """Write `<output_dir><archive_suffix>.zip` beside the directory and return its path."""
    directory = os.path.abspath(output_dir)
    return shutil.make_archive(f"{directory}{archive_suffix}", "zip", directory)


__all__ = [
    "GZIP_COMPRESSLEVEL",
    "PathLike",
    "is_gzip_path",
    "open_binary_maybe_gzip",
    "open_text_maybe_gzip",
    "zip_directory",
]
