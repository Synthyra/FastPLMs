"""Normalize ordered inputs and plan bounded windows without retaining a full stream."""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
import tempfile
import weakref

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, overload

from .types import EmbeddingInput


class FastaRecord(NamedTuple):
    """One FASTA record: its header text and its sequence lines joined."""

    header: str
    sequence: str


@dataclass(frozen=True)
class FastaDialect:
    """How one caller reads FASTA lines. FastPLMs has three readers, and each keeps its rules here.

    ``strip_lines``: strip each line before use; otherwise lines are used as given, so a space-only line
        is sequence data.
    ``comment_prefix``: lines that start with it are skipped, or ``None`` for no comments.
    ``first_word_header``: the header is its first whitespace-delimited word, not the whole header text.
    ``squeeze_sequence_whitespace``: remove all whitespace inside a sequence line.
    ``orphan_message``: raised as ``ValueError`` for sequence data before the first header, formatted
        with ``line_number`` and ``source``; ``None`` skips such lines.
    ``empty_message``: raised as ``ValueError`` when the input holds no record, formatted with
        ``source``; ``None`` allows empty input.
    """

    strip_lines: bool
    comment_prefix: str | None
    first_word_header: bool
    squeeze_sequence_whitespace: bool
    orphan_message: str | None
    empty_message: str | None


EMBEDDING_FASTA = FastaDialect(
    strip_lines=True,
    comment_prefix=None,
    first_word_header=True,
    squeeze_sequence_whitespace=True,
    orphan_message="Sequence data precedes the first FASTA header on line {line_number}.",
    empty_message="No FASTA records found in {source}.",
)


def scan_fasta_lines(
    lines: Iterable[str], dialect: FastaDialect, *, source: str
) -> Iterator[FastaRecord]:
    """Yield the records of FASTA ``lines`` in order, one record at a time.

    ``source`` names the input in the dialect's messages. A record is yielded when the next header or
    the end of input shows it is complete, so an error raised later in the input surfaces after the
    records before it.
    """

    header: str | None = None
    sequence_parts: list[str] = []
    found_record = False
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip() if dialect.strip_lines else raw_line
        if not line or (dialect.comment_prefix is not None and line.startswith(dialect.comment_prefix)):
            continue

        if line.startswith(">"):
            if header is not None:
                found_record = True
                yield FastaRecord(header, "".join(sequence_parts))
            header = line[1:].strip()
            if dialect.first_word_header:
                # An empty header has no first word and raises IndexError here.
                header = header.split(maxsplit=1)[0]
            sequence_parts = []
        elif header is not None:
            sequence_parts.append("".join(line.split()) if dialect.squeeze_sequence_whitespace else line)
        elif dialect.orphan_message is not None:
            raise ValueError(dialect.orphan_message.format(line_number=line_number, source=source))

    if header is not None:
        found_record = True
        yield FastaRecord(header, "".join(sequence_parts))
    if not found_record and dialect.empty_message is not None:
        raise ValueError(dialect.empty_message.format(source=source))


def iter_fasta(path: str | Path) -> Iterator[EmbeddingInput]:
    """Yield FASTA records in source order without reading the file into memory."""

    with Path(path).open("r", encoding="utf-8") as handle:
        for record in scan_fasta_lines(handle, EMBEDDING_FASTA, source=str(path)):
            yield EmbeddingInput(record.header, record.sequence)


def parse_fasta(path: str | Path) -> list[EmbeddingInput]:
    """Parse FASTA records while preserving identifiers, order, and duplicates."""

    return list(iter_fasta(path))


def _normalize_input_item(
    position: int,
    item: str | EmbeddingInput | tuple[str, str],
) -> EmbeddingInput:
    if isinstance(item, EmbeddingInput):
        return item
    if isinstance(item, str):
        return EmbeddingInput(str(position), item)
    if isinstance(item, tuple) and len(item) == 2:
        return EmbeddingInput(str(item[0]), str(item[1]))
    raise TypeError(
        "inputs must contain sequences, EmbeddingInput values, or (id, sequence) tuples."
    )


class _SpoolFiles:
    """A spool's directory and SQLite connection, released together: the connection first.

    Windows cannot delete a file that an open connection holds. The cycle collector runs weakref
    finalizers before any ``__del__``, so a spool freed as cyclic garbage, as one referenced from
    a raised exception's traceback is, would have ``tempfile.TemporaryDirectory``'s own finalizer
    remove the directory while the connection was still open. One finalizer owns both instead.
    """

    def __init__(self) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="fastplms-inputs-"))
        self.connection: sqlite3.Connection | None = None

    def release(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.directory.exists():
            shutil.rmtree(self.directory)


class _InputSpool(Sequence[EmbeddingInput]):
    """Immutable disk-backed normalized inputs with an incremental digest."""

    def __init__(
        self,
        values: Iterable[str | EmbeddingInput | tuple[str, str]],
    ) -> None:
        self._files = _SpoolFiles()
        # Called by close(), or by garbage collection however the spool is freed; at most once.
        self._release = weakref.finalize(self, self._files.release)
        self.path = self._files.directory / "inputs.sqlite"
        digest = hashlib.sha256()
        count = 0
        pending: list[tuple[int, str, str]] = []
        try:
            connection = self._files.connection = sqlite3.connect(self.path)
            connection.execute(
                "CREATE TABLE inputs ("
                "position INTEGER PRIMARY KEY, input_id TEXT NOT NULL, sequence TEXT NOT NULL)"
            )
            for position, item in enumerate(values):
                record = _normalize_input_item(position, item)
                for value in (record.id, record.sequence):
                    encoded = value.encode("utf-8")
                    digest.update(len(encoded).to_bytes(8, "big"))
                    digest.update(encoded)
                pending.append((position, record.id, record.sequence))
                count += 1
                if len(pending) == 1_024:
                    connection.executemany("INSERT INTO inputs VALUES (?, ?, ?)", pending)
                    pending.clear()
            if pending:
                connection.executemany("INSERT INTO inputs VALUES (?, ?, ?)", pending)
            if count == 0:
                raise ValueError("inputs must contain at least one sequence.")
            connection.commit()
            connection.close()
            self._files.connection = sqlite3.connect(
                f"{self.path.resolve().as_uri()}?mode=ro",
                uri=True,
            )
        except BaseException:
            self.close()
            raise
        digest.update(count.to_bytes(8, "big"))
        self.input_fingerprint = digest.hexdigest()
        self._count = count

    def _require_connection(self) -> sqlite3.Connection:
        connection = self._files.connection
        if connection is None:
            raise RuntimeError("Input spool is closed.")
        return connection

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[EmbeddingInput]:
        cursor = self._require_connection().execute(
            "SELECT input_id, sequence FROM inputs ORDER BY position"
        )
        while rows := cursor.fetchmany(1_024):
            for input_id, sequence in rows:
                yield EmbeddingInput(input_id, sequence)

    @overload
    def __getitem__(self, index: int, /) -> EmbeddingInput: ...

    @overload
    def __getitem__(self, index: slice, /) -> list[EmbeddingInput]: ...

    def __getitem__(self, index: int | slice) -> EmbeddingInput | list[EmbeddingInput]:
        connection = self._require_connection()

        if isinstance(index, slice):
            start, stop, step = index.indices(self._count)
            if step != 1:
                return [self[position] for position in range(start, stop, step)]
            rows = connection.execute(
                "SELECT input_id, sequence FROM inputs "
                "WHERE position >= ? AND position < ? ORDER BY position",
                (start, stop),
            ).fetchall()
            return [EmbeddingInput(input_id, sequence) for input_id, sequence in rows]
        position = index + self._count if index < 0 else index
        if position < 0 or position >= self._count:
            raise IndexError(index)
        row = connection.execute(
            "SELECT input_id, sequence FROM inputs WHERE position = ?", (position,)
        ).fetchone()
        if row is None:
            raise IndexError(index)
        return EmbeddingInput(row[0], row[1])

    def close(self) -> None:
        self._release()


def _normalize_inputs(
    inputs: (Iterable[str | EmbeddingInput | tuple[str, str]] | Mapping[str, str] | str | Path),
    *,
    disk_backed: bool,
) -> Sequence[EmbeddingInput]:
    is_fasta_path = isinstance(inputs, Path)
    if isinstance(inputs, str):
        try:
            is_fasta_path = Path(inputs).is_file()
        except OSError:
            is_fasta_path = False
    should_spool = disk_backed or is_fasta_path or not isinstance(inputs, (str, Sequence, Mapping))
    values: Iterable[str | EmbeddingInput | tuple[str, str]]
    if isinstance(inputs, Path):
        values = iter_fasta(inputs)
    elif isinstance(inputs, str):
        values = iter_fasta(inputs) if is_fasta_path else [inputs]
    elif isinstance(inputs, Mapping):
        values = inputs.items()
    else:
        values = inputs
    if should_spool:
        return _InputSpool(values)
    records: list[EmbeddingInput] = []
    for position, item in enumerate(values):
        records.append(_normalize_input_item(position, item))
    if not records:
        raise ValueError("inputs must contain at least one sequence.")
    return records


def _validate_untruncated_lengths(
    records: Sequence[EmbeddingInput],
    *,
    max_length: int | None,
    truncate: bool,
) -> None:
    """Fail before inference when a biological-residue limit would be exceeded."""

    if max_length is None or truncate:
        return
    for position, record in enumerate(records):
        residue_count = len(record.sequence)
        if residue_count > max_length:
            raise ValueError(
                f"Input at position {position} with id {record.id!r} has "
                f"{residue_count} biological residues, exceeding max_length={max_length} "
                "while truncate=False."
            )


def _planned_batches(
    records: Sequence[EmbeddingInput],
    positions: range,
    *,
    batch_size: int,
    max_tokens_per_batch: int | None,
    max_length: int | None,
    truncate: bool,
) -> Iterator[list[int]]:
    """Length-bucket one bounded window while retaining stable output positions."""

    def effective_length(position: int) -> int:
        length = len(records[position].sequence)
        return min(length, max_length) if truncate and max_length is not None else length

    ordered = sorted(positions, key=lambda position: (-effective_length(position), position))
    batch: list[int] = []
    longest = 0
    for position in ordered:
        length = effective_length(position)
        if max_tokens_per_batch is not None and length > max_tokens_per_batch:
            raise ValueError(
                f"Input at position {position} has {length} residues, exceeding "
                f"max_tokens_per_batch={max_tokens_per_batch}."
            )
        candidate_longest = max(longest, length)
        exceeds_tokens = (
            max_tokens_per_batch is not None
            and candidate_longest * (len(batch) + 1) > max_tokens_per_batch
        )
        if batch and (len(batch) >= batch_size or exceeds_tokens):
            yield batch
            batch = []
            longest = 0
        batch.append(position)
        longest = max(longest, length)
    if batch:
        yield batch
