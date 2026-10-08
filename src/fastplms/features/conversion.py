"""Checked conversion of a cache another format holds into a feature store.

A project that already holds embeddings in its own format keeps them by converting them, not by
re-embedding. ``convert_rows`` takes the old cache as a function that decodes it, writes the rows
through the store's ordinary segment writer, then decodes the old cache a second time and compares
every row it produced with what the store now returns. Comparison is bit-exact on the stored
representation, so a converter never has to argue that two floats are close enough.

Nothing is repaired or skipped quietly. A row the store's dtype cannot hold exactly, a sequence
absent after the commit, a row that differs, a source that repeats a sequence with different rows,
and a source that changes between its two decodes each raise ``ConversionMismatch``. The old cache is
never modified or deleted: the move is undone by removing the segments this call names, and the
origin recorded in each commit marker says which files those rows came from.

What a project supplies is only the decoder of its old format. The decoder yields
``(sequence, row)`` pairs, where a row is a tensor for ``dense`` and ``ragged`` features, a
``SparseRow`` for ``csr``, and a ``TopKRow`` for ``ragged_topk``. It takes no arguments, because it
is called twice.
"""

from __future__ import annotations

import json
import torch

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from torch import Tensor

from .digests import file_sha256, json_sha256
from .layouts import CSR, RAGGED_TOPK, SparseRow, TopKRow
from .reader import FeatureReader
from .store import (
    COMMIT_FILE,
    SEGMENTS_DIRECTORY,
    FeatureStore,
    StoredFeature,
    sequence_digest,
)


Row = Tensor | SparseRow | TopKRow


class ConversionMismatch(ValueError):
    """The store does not hold exactly what the old cache held."""


@dataclass(frozen=True, slots=True)
class ConversionReceipt:
    """What one call did: the segments it committed and how many rows it compared."""

    segments: tuple[str, ...]
    source_rows: int
    written: int
    skipped: int
    verified: int


def describe_file(path: str | Path) -> dict[str, Any]:
    """A file as an origin records it: its name, size, and content digest."""

    location = Path(path)
    return {
        "name": location.name,
        "bytes": location.stat().st_size,
        "sha256": file_sha256(location),
    }


def conversion_fingerprint(origin: Mapping[str, Any]) -> str:
    """The stable name of the conversion of this origin, from which segment names derive."""

    return "convert-" + json_sha256(dict(origin), allow_nan=False)[:16]


def convert_rows(
    store: FeatureStore,
    source: Callable[[], Iterable[tuple[str, Row]]],
    *,
    origin: Mapping[str, Any],
    rows_per_segment: int = 100_000,
    window_rows: int = 1_024,
    max_tensor_bytes: int = 256 * 1024**2,
) -> ConversionReceipt:
    """Write the decoded rows into ``store``, then prove the store holds them.

    ``origin`` is plain data naming where the rows came from, normally the old cache's format, the
    decoder, and ``describe_file`` of each file. It is recorded in every segment's commit marker and
    names the conversion, so calling this again with the same origin resumes an interrupted
    conversion: committed segments are kept, sequences already in the store are not written twice,
    and the comparison runs over the whole source. A long conversion commits a segment once it holds
    ``rows_per_segment`` new rows, rounded up to a window, so an interruption loses at most one
    segment.

    Rows the store held before the call are compared too. A store that mixes converted rows with
    rows embedded afresh therefore fails here, which is the point: the old numbers are not the
    store's numbers.
    """

    if rows_per_segment < 1 or window_rows < 1:
        raise ValueError("rows_per_segment and window_rows must be positive.")
    fingerprint = conversion_fingerprint(origin)
    ordinal = _committed_segments(store, fingerprint)
    segments: list[str] = []
    source_rows = 0
    written = 0

    def counted() -> Iterator[tuple[str, Row]]:
        nonlocal source_rows
        for pair in source():
            source_rows += 1
            yield pair

    windows = _unique_windows(store.spec, counted(), window_rows)
    pending = next(windows, None)
    while pending is not None:
        name = f"{fingerprint}-{ordinal:05d}"
        metadata = {"conversion": {"origin": dict(origin), "ordinal": ordinal}}
        staged: set[str] = set()  # digests written into this segment, which is not yet committed
        with store.segment(name, metadata) as writer:
            while pending is not None and len(staged) < rows_per_segment:
                absent = set(store.missing([sequence for sequence, _ in pending]))
                fresh = [
                    pair for pair in pending
                    if pair[0] in absent and sequence_digest(pair[0]) not in staged
                ]
                if fresh:
                    writer.append_bounded(
                        [sequence for sequence, _ in fresh], [row for _, row in fresh],
                        max_tensor_bytes=max_tensor_bytes,
                    )
                    staged.update(sequence_digest(sequence) for sequence, _ in fresh)
                pending = next(windows, None)
        if staged:
            segments.append(name)
            ordinal += 1
            written += len(staged)

    if source_rows == 0:
        raise ConversionMismatch("The source decoded no rows; check the path and the decoder.")
    verified = _verify(store, source, window_rows)
    if verified != source_rows:
        raise ConversionMismatch(
            f"The source decoded {source_rows} rows the first time and {verified} the second."
        )
    return ConversionReceipt(
        segments=tuple(segments), source_rows=source_rows, written=written,
        skipped=source_rows - written, verified=verified,
    )


def _committed_segments(store: FeatureStore, fingerprint: str) -> int:
    directory = store.directory / SEGMENTS_DIRECTORY
    return sum(
        1 for marker in directory.glob(f"{fingerprint}-*/{COMMIT_FILE}") if marker.is_file()
    )


def _windows(rows: Iterable[tuple[str, Row]], size: int) -> Iterator[list[tuple[str, Row]]]:
    window: list[tuple[str, Row]] = []
    for pair in rows:
        window.append(pair)
        if len(window) == size:
            yield window
            window = []
    if window:
        yield window


def _unique_windows(
    spec: StoredFeature, rows: Iterable[tuple[str, Row]], size: int,
) -> Iterator[list[tuple[str, Row]]]:
    """Windows in which each sequence appears once.

    A sequence repeated inside a window keeps its first row, and is an error if the rows differ.
    A repeat in a later window is skipped when its sequence is already written, and the comparison
    after the commit raises if its row differs from the one that was.
    """

    for window in _windows(rows, size):
        unique: dict[str, tuple[str, Row]] = {}
        for sequence, row in window:
            digest = sequence_digest(sequence)
            first = unique.setdefault(digest, (sequence, row))
            if first[1] is not row and _row_bytes(first[1], spec) != _row_bytes(row, spec):
                raise ConversionMismatch(
                    f"The source repeats a sequence (sha256 {digest[:12]}) with different rows."
                )
        yield list(unique.values())


def _verify(
    store: FeatureStore, source: Callable[[], Iterable[tuple[str, Row]]], window_rows: int,
) -> int:
    """Decode the source again and compare each row with a fresh read of the committed store."""

    spec = store.spec
    verified = 0
    with FeatureReader.open(store.directory) as reader:
        for window in _windows(source(), window_rows):
            sequences = [sequence for sequence, _ in window]
            absent = reader.missing(sequences)
            if absent:
                raise ConversionMismatch(
                    f"{len(absent)} converted sequences are absent from {spec.key!r} after commit."
                )
            for (sequence, row), stored in zip(window, _read(reader, sequences), strict=True):
                if _row_bytes(row, spec) != _row_bytes(stored, spec):
                    raise ConversionMismatch(
                        f"Feature {spec.key!r} differs from the source for the sequence with "
                        f"sha256 {sequence_digest(sequence)[:12]}."
                    )
            verified += len(window)
    return verified


def _read(reader: FeatureReader, sequences: list[str]) -> list[Row]:
    layout = reader.spec.layout
    if layout == CSR:
        return list(reader.read_sparse(sequences))
    if layout == RAGGED_TOPK:
        return list(reader.read_topk(sequences))
    return list(reader.read(sequences))


def _row_bytes(row: Row, spec: StoredFeature) -> bytes:
    """A row as the store holds it, as bytes: shapes, then each tensor's raw storage.

    Values are cast to the feature's dtype first, and a cast that changes any value raises, so
    equal bytes mean equal stored rows.
    """

    # row: (w,) dense, (r_i, d) ragged; a SparseRow holds (nnz,) tensors and a TopKRow (r_i, k) tensors
    if isinstance(row, SparseRow):
        tensors = [row.indices.to(torch.int32), _stored_values(row.values, spec.dtype)]  # (nnz,), (nnz,)
        if row.positions is not None:
            tensors.append(row.positions.to(torch.int16))  # (nnz,)
    elif isinstance(row, TopKRow):
        tensors = [row.indices.to(torch.int32), _stored_values(row.values, spec.dtype)]  # (r_i, k), (r_i, k)
    else:
        tensors = [_stored_values(row, spec.dtype)]  # (w,) or (r_i, d)
    shapes = json.dumps([list(tensor.shape) for tensor in tensors]).encode("utf-8")
    return shapes + b"".join(_raw_bytes(tensor) for tensor in tensors)


def _stored_values(values: Tensor, dtype: torch.dtype) -> Tensor:
    # values: (...), the shape of one stored tensor; the cast keeps it
    cast = values.detach().to("cpu").to(dtype)  # (...)
    if values.dtype != dtype and _raw_bytes(cast.to(values.dtype)) != _raw_bytes(values.detach().cpu()):
        raise ConversionMismatch(
            f"Storing {values.dtype} values as {dtype} would change them; choose a wider dtype."
        )
    return cast  # (...)


def _raw_bytes(tensor: Tensor) -> bytes:
    # tensor: (...)
    flat = tensor.detach().to("cpu").contiguous().reshape(-1)  # (n,), n = tensor.numel()
    # An empty tensor can carry a zero stride, which the byte view refuses, and has no bytes to compare.
    return b"" if flat.numel() == 0 else flat.view(torch.uint8).numpy().tobytes()


__all__ = [
    "ConversionMismatch",
    "ConversionReceipt",
    "convert_rows",
    "conversion_fingerprint",
    "describe_file",
]
