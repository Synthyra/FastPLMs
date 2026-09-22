"""Normalize ordered inputs and plan bounded windows without retaining a full stream."""

from __future__ import annotations

import hashlib
import sqlite3
import tempfile

from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import overload

from .types import EmbeddingInput


def iter_fasta(path: str | Path) -> Iterator[EmbeddingInput]:
    """Yield FASTA records in source order without reading the file into memory."""

    identifier: str | None = None
    sequence_parts: list[str] = []
    found_record = False
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    found_record = True
                    yield EmbeddingInput(identifier, "".join(sequence_parts))
                identifier = line[1:].strip().split(maxsplit=1)[0]
                if not identifier:
                    raise ValueError(f"Missing FASTA identifier on line {line_number}.")
                sequence_parts = []
            else:
                if identifier is None:
                    raise ValueError(
                        f"Sequence data precedes the first FASTA header on line {line_number}."
                    )
                sequence_parts.append("".join(line.split()))
    if identifier is not None:
        found_record = True
        yield EmbeddingInput(identifier, "".join(sequence_parts))
    if not found_record:
        raise ValueError(f"No FASTA records found in {path}.")


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


class _InputSpool(Sequence[EmbeddingInput]):
    """Immutable disk-backed normalized inputs with an incremental digest."""

    def __init__(
        self,
        values: Iterable[str | EmbeddingInput | tuple[str, str]],
    ) -> None:
        self._temporary: tempfile.TemporaryDirectory[str] | None = tempfile.TemporaryDirectory(
            prefix="fastplms-inputs-"
        )
        self.path = Path(self._temporary.name) / "inputs.sqlite"
        self._connection: sqlite3.Connection | None = sqlite3.connect(self.path)
        self._connection.execute(
            "CREATE TABLE inputs ("
            "position INTEGER PRIMARY KEY, input_id TEXT NOT NULL, sequence TEXT NOT NULL)"
        )
        digest = hashlib.sha256()
        count = 0
        pending: list[tuple[int, str, str]] = []
        try:
            for position, item in enumerate(values):
                record = _normalize_input_item(position, item)
                for value in (record.id, record.sequence):
                    encoded = value.encode("utf-8")
                    digest.update(len(encoded).to_bytes(8, "big"))
                    digest.update(encoded)
                pending.append((position, record.id, record.sequence))
                count += 1
                if len(pending) == 1_024:
                    self._connection.executemany("INSERT INTO inputs VALUES (?, ?, ?)", pending)
                    pending.clear()
            if pending:
                self._connection.executemany("INSERT INTO inputs VALUES (?, ?, ?)", pending)
            if count == 0:
                raise ValueError("inputs must contain at least one sequence.")
            self._connection.commit()
            self._connection.close()
            self._connection = sqlite3.connect(
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
        if self._connection is None:
            raise RuntimeError("Input spool is closed.")
        return self._connection

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
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()
            self._connection = None
        temporary = getattr(self, "_temporary", None)
        if temporary is not None:
            temporary.cleanup()
            self._temporary = None

    def __del__(self) -> None:
        self.close()


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
