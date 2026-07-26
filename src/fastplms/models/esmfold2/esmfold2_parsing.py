"""FASTA parsing and writing with explicit stream ownership."""

from __future__ import annotations

import gzip
import io
from collections.abc import Generator, Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import NamedTuple, TextIO

from .esmfold2_utils_types import PathOrBuffer


class FastaEntry(NamedTuple):
    """One FASTA record in source order."""

    header: str
    sequence: str


def parse_fasta(text: str) -> Generator[FastaEntry, None, None]:
    """Yield records from FASTA text without normalizing sequence symbols."""

    header: str | None = None
    sequence_lines: list[str] = []
    found_record = False

    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith(">"):
            if header is not None:
                found_record = True
                yield FastaEntry(header, "".join(sequence_lines))
            header = line[1:].strip()
            sequence_lines.clear()
        elif header is not None:
            sequence_lines.append(line)

    if header is not None:
        found_record = True
        yield FastaEntry(header, "".join(sequence_lines))
    if not found_record:
        raise ValueError("Found no sequences in input")


def _open_reader(source: PathOrBuffer):
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


def _open_writer(destination: PathOrBuffer):
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
