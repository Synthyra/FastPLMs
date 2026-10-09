"""FASTA parsing and writing with explicit stream ownership."""

from __future__ import annotations

import gzip
import io

from collections.abc import Generator, Iterable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import NamedTuple, TextIO

from ...embeddings.inputs import FastaDialect, scan_fasta_lines
from .esmfold2_utils_types import PathOrBuffer


# Lines are used as given and `#` lines are comments. A header keeps its full text, and sequence data
# before the first header is skipped.
ESMFOLD2_FASTA = FastaDialect(
    strip_lines=False,
    comment_prefix="#",
    first_word_header=False,
    squeeze_sequence_whitespace=False,
    orphan_message=None,
    empty_message="Found no sequences in input",
)


class FastaEntry(NamedTuple):
    """One FASTA record in source order."""

    header: str
    sequence: str


def parse_fasta(text: str) -> Generator[FastaEntry, None, None]:
    """Yield records from FASTA text without normalizing sequence symbols."""

    for record in scan_fasta_lines(text.splitlines(), ESMFOLD2_FASTA, source="input"):
        yield FastaEntry(record.header, record.sequence)


def _open_reader(source: PathOrBuffer) -> AbstractContextManager[TextIO]:
    if isinstance(source, io.TextIOBase):
        return nullcontext(source)
    path = Path(source)
    if path.suffix.lower() == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open(mode="r", encoding="utf-8")


def read_sequences(source: PathOrBuffer) -> Generator[FastaEntry, None, None]:
    """Read FASTA records while leaving caller-owned streams open."""

    with _open_reader(source) as handle:
        yield from parse_fasta(handle.read())


def read_first_sequence(source: PathOrBuffer) -> FastaEntry:
    """Return the first FASTA record from a path or text stream."""

    return next(read_sequences(source))


def count_fasta_sequences(path: str | Path) -> int:
    """Count FASTA headers without parsing sequence bodies."""

    source = Path(path)
    if not source.exists():
        return 0
    with source.open(encoding="utf-8") as handle:
        return sum(line.startswith(">") for line in handle)


def append_fasta_sequence(header: str, sequence: str, path: str | Path) -> None:
    """Append one record, inserting a separator if the file lacks a final newline."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    needs_separator = (
        destination.exists()
        and destination.stat().st_size > 0
        and destination.read_bytes()[-1:] != b"\n"
    )
    with destination.open(mode="a", encoding="utf-8") as handle:
        if needs_separator:
            handle.write("\n")
        handle.write(f">{header}\n{sequence}\n")


def _open_writer(destination: PathOrBuffer) -> AbstractContextManager[TextIO]:
    if isinstance(destination, io.TextIOBase):
        return nullcontext(destination)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open(mode="w", encoding="utf-8")


def write_sequences(sequences: Iterable[tuple[str, str]], destination: PathOrBuffer) -> None:
    """Write records with one blank-line-free separator between entries."""

    with _open_writer(destination) as handle:
        _write_records(handle, sequences)


def _write_records(handle: TextIO, sequences: Iterable[tuple[str, str]]) -> None:
    for index, (header, sequence) in enumerate(sequences):
        if index:
            handle.write("\n")
        handle.write(f">{header}\n{sequence}")


__all__ = [
    "FastaEntry",
    "append_fasta_sequence",
    "count_fasta_sequences",
    "parse_fasta",
    "read_first_sequence",
    "read_sequences",
    "write_sequences",
]
