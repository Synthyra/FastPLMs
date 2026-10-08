"""Write the rows of many streams behind the model: one bounded queue, one ingest thread, parallel part writes.

The embedding loop runs on the device. This writer takes each finished batch from pinned host buffers,
packs rows into parts of about ``part_bytes``, writes and hashes each part in one pass on a pool thread,
and commits every stream together once ``segment_bytes`` of parts exist, so a killed run loses at most
one segment. ``submit`` blocks only when the queued batches exceed ``queue_bytes``, which is what keeps
the device from outrunning the disk.

Symbols: b sequences of a batch; n token rows of a batch (sum of l_i + 2, with l_i the residues of
sequence i after the crop); w stored columns of a stream; k SAE codes kept per token; c SAE codebook.
Layouts of one batch, by stream:

- ragged       values (n, w): row 0 CLS, rows 1..l_i residues, row l_i + 1 EOS of each sequence, in order.
- ragged_topk  indices, values (n, k): the top-k SAE codes of every token row.
- dense        values (b, w): one pooled vector per sequence.
- csr          values (b, c) dense on arrival, compressed here to (nnz,) indices and values.
"""

from __future__ import annotations

import threading
import torch

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any
from torch import Tensor

from .layouts import CSR, DENSE, INDEX_DTYPE, OFFSET_DTYPE, RAGGED, RAGGED_TOPK
from .store import FeatureStore, SegmentReceipt, SegmentWriter


@dataclass(eq=False)
class PackedBatch:
    """One batch of finished rows for every stream, on the host, as the executor hands them over.

    ``streams`` maps a stream to its host tensors by layout: ragged ``{"values": (n, w)}``, ragged
    top-k ``{"values": (n, k), "indices": (n, k)}``, dense ``{"values": (b, w)}``, csr
    ``{"values": (b, c)}`` dense, which the writer compresses. ``rows`` counts the stored rows of each
    sequence in a ragged stream (l_i + 2 when the special tokens are kept). ``wait`` blocks until the
    device copies landed and the batch passed its finite check, and raises if it did not.
    """

    sequences: tuple[str, ...]  # (b,) the exact text of each row, for the row identities
    digests: tuple[str, ...]  # (b,) SHA-256 row keys, hashed once by the caller
    rows: tuple[int, ...]  # (b,) stored rows per sequence for ragged streams
    streams: Mapping[str, Mapping[str, Tensor]]  # per stream: values (n, w) | (n, k) with indices (n, k) | (b, w) | (b, c)
    wait: Callable[[], None]
    nbytes: int = field(init=False)

    def __post_init__(self) -> None:
        self.nbytes = sum(
            tensor.numel() * tensor.element_size() for group in self.streams.values() for tensor in group.values()
        )


@dataclass(eq=False)
class _StreamBuffer:
    """Rows of one stream waiting to fill a part."""

    sequences: list[str] = field(default_factory=list)
    digests: list[str] = field(default_factory=list)
    rows: list[int] = field(default_factory=list)  # stored rows per sequence, as in PackedBatch
    nnz: list[int] = field(default_factory=list)  # csr entries per sequence, zero for other layouts
    tensors: dict[str, list[Tensor]] = field(default_factory=dict)  # per name: slices (n_i, w) or (b_i, w), joined at flush
    nbytes: int = 0


def _row_bytes(layout: str, tensors: Mapping[str, Tensor], rows: Sequence[int], nnz: Sequence[int]) -> list[int]:
    """Encoded payload of each sequence's row, its offset included, which the part budget counts."""
    # tensors: (b, w) per stream when dense, (nnz,) when csr, (n, w) when ragged; n = sum(rows) token rows.
    if layout == DENSE:
        per_row = sum(tensor.shape[1] * tensor.element_size() for tensor in tensors.values())
        return [per_row] * len(rows)
    if layout == CSR:
        per_entry = sum(tensor.element_size() for tensor in tensors.values())
        return [8 + count * per_entry for count in nnz]
    per_token = sum(tensor.shape[1] * tensor.element_size() for tensor in tensors.values())
    return [8 + count * per_token for count in rows]


class AsyncFeatureWriter:
    """Take finished batches from the embedding loop and commit them as rolling segments of every stream.

    ``stores`` maps a stream to its opened store, whose descriptor carries the layout. ``wanted`` maps a
    stream to the digests it lacks, so a rerun fills a lagging stream without rewriting the others.
    ``row_records(stream, sequences, digests)`` returns each row's persisted identity. ``before_commit`` runs once
    per committed group, before the first stream's marker, and may raise to refuse the commit.
    """

    def __init__(
        self,
        stores: Mapping[str, FeatureStore],
        *,
        fingerprint: str,
        metadata: Mapping[str, Any],
        wanted: Mapping[str, frozenset[str]],
        row_records: Callable[[str, Sequence[str], Sequence[str]], Sequence[Mapping[str, Any]]],
        part_bytes: int,
        segment_bytes: int,
        queue_bytes: int,
        before_commit: Callable[[], None] | None = None,
        workers: int = 4,
        progress: Callable[[int], None] | None = None,
        verify_staged: bool = False,
    ) -> None:
        if min(part_bytes, segment_bytes, queue_bytes, workers) < 1:
            raise ValueError("part_bytes, segment_bytes, queue_bytes and workers must be positive.")
        self._stores = dict(stores)
        self._fingerprint = fingerprint
        self._metadata = dict(metadata)
        self._wanted = dict(wanted)
        self._row_records = row_records
        self._part_bytes = part_bytes
        self._segment_bytes = segment_bytes
        self._queue_bytes = queue_bytes
        self._before_commit = before_commit
        self._progress = progress
        self._verify_staged = verify_staged
        self._buffers = {name: _StreamBuffer() for name in self._stores}
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="feature-part")
        # Parts in flight, written or waiting for a pool thread; more would only queue host memory.
        self._slots = threading.Semaphore(workers + 2)
        self._futures: deque[Future[None]] = deque()
        self._condition = threading.Condition()
        self._queue: deque[PackedBatch | None] = deque()
        self._queued_bytes = 0
        self._error: BaseException | None = None
        # Held around each commit, so an abort lands before a commit starts or after it ends, never during one.
        self._commit_lock = threading.Lock()
        self._aborted = False
        self._segment_index = 0
        self._segment_written = 0
        self._receipts: dict[str, list[SegmentReceipt]] = {name: [] for name in self._stores}
        self._stack = ExitStack()
        self._writers: dict[str, SegmentWriter] = {}
        self._checked = False
        self._open_segments()
        self._thread = threading.Thread(target=self._ingest, name="feature-ingest", daemon=True)
        self._thread.start()

    def submit(self, batch: PackedBatch) -> None:
        """Queue one batch, blocking while the queue holds more than ``queue_bytes`` of finished rows."""
        with self._condition:
            while self._error is None and self._queued_bytes and self._queued_bytes + batch.nbytes > self._queue_bytes:
                self._condition.wait()
            self._raise_error()
            self._queue.append(batch)
            self._queued_bytes += batch.nbytes
            self._condition.notify_all()

    def close(self) -> dict[str, list[SegmentReceipt]]:
        """Write what is queued, commit the last segment, and return every committed segment by stream."""
        with self._condition:
            self._queue.append(None)
            self._condition.notify_all()
        self._thread.join()
        self._pool.shutdown(wait=True)
        if self._error is not None:
            self._discard()
            raise self._error
        return self._receipts

    def abort(self) -> None:
        """Stop without committing: queued rows are dropped and open segments stay uncommitted for ``sweep``."""
        with self._commit_lock:  # a commit in flight finishes; none starts after this
            self._aborted = True
        with self._condition:
            self._error = self._error or RuntimeError("The feature writer was aborted.")
            self._queue.clear()
            self._condition.notify_all()
        self._thread.join()
        self._pool.shutdown(wait=True)
        self._discard()

    def _discard(self) -> None:
        """Close the open segments without committing them; a closed writer is skipped by its context."""
        with self._commit_lock:  # one closer at a time, and never during a commit
            for writer in self._writers.values():
                writer.closed = True
            self._stack.close()

    def _raise_error(self) -> None:
        if self._error is not None:
            raise self._error

    def _open_segments(self) -> None:
        name = f"{self._fingerprint}-{self._segment_index:05d}"
        for stream, store in self._stores.items():
            self._writers[stream] = self._stack.enter_context(store.segment(
                name, {**self._metadata, "segment_index": self._segment_index},
                before_commit=self._check_once, verify_staged=self._verify_staged,
            ))

    def _check_once(self) -> None:
        """Run the caller's pre-commit check for the first stream of a group; its siblings reuse the verdict."""
        if not self._checked:
            if self._before_commit is not None:
                self._before_commit()
            self._checked = True

    def _ingest(self) -> None:
        try:
            while True:
                with self._condition:
                    while not self._queue and self._error is None:
                        self._condition.wait()
                    if self._error is not None:
                        return
                    batch = self._queue.popleft()
                if batch is None:
                    self._commit_group()
                    return
                batch.wait()  # the device copies landed and the finite check passed
                for stream in self._stores:
                    self._add(stream, batch)
                with self._condition:
                    self._queued_bytes -= batch.nbytes
                    self._condition.notify_all()
                if self._progress is not None:
                    self._progress(len(batch.sequences))
                if self._segment_written >= self._segment_bytes:
                    self._commit_group(reopen=True)
        # A worker thread has no caller to raise to: keep the error for `_raise_error` on the caller's thread.
        except BaseException as error:  # noqa: broad-except
            with self._condition:
                self._error = self._error or error
                self._condition.notify_all()

    def _add(self, stream: str, batch: PackedBatch) -> None:
        layout = self._stores[stream].spec.layout
        keep = [index for index, digest in enumerate(batch.digests) if digest in self._wanted[stream]]
        if not keep:
            return
        tensors = batch.streams[stream]  # values (n, w) | (n, k) with indices (n, k) | (b, w) | (b, c), by layout
        rows = list(batch.rows)  # (b,) stored rows per sequence: l_i + 2 when CLS and EOS are kept
        if layout == CSR:
            tensors, nnz = _compress_csr(tensors["values"])  # indices, values (nnz,); nnz per sequence (b,)
        else:
            nnz = [0] * len(rows)
        if len(keep) != len(rows):
            tensors, rows, nnz = _select_rows(layout, tensors, rows, nnz, keep)
        sequences = [batch.sequences[index] for index in keep]  # (m,) m sequences this stream lacks
        digests = [batch.digests[index] for index in keep]  # (m,)
        sizes = _row_bytes(layout, tensors, rows, nnz)  # (m,) payload bytes per sequence
        if any(size > self._part_bytes for size in sizes):
            raise ValueError("A feature row exceeds max_part_bytes; increase the explicit part budget.")
        buffer = self._buffers[stream]
        start = 0
        while start < len(sequences):
            # Take rows until the next would overflow the part, flush, and go on; a flushed buffer fits any row.
            room = self._part_bytes - buffer.nbytes
            stop, used = start, 0
            while stop < len(sequences) and used + sizes[stop] <= room:
                used += sizes[stop]
                stop += 1
            if stop == start:
                self._flush(stream)
                buffer = self._buffers[stream]
                continue
            _take(buffer, layout, tensors, sequences, digests, rows, nnz, start, stop, used)
            self._segment_written += used  # buffered rows count toward the segment, so a small segment commits per batch
            if buffer.nbytes >= self._part_bytes:
                self._flush(stream)
                buffer = self._buffers[stream]
            start = stop

    def _flush(self, stream: str) -> None:
        buffer = self._buffers[stream]
        if not buffer.sequences:
            return
        self._buffers[stream] = _StreamBuffer()
        # A failed part surfaces at the next flush, not only at the end of the segment.
        while self._futures and self._futures[0].done():
            self._futures.popleft().result()
        writer = self._writers[stream]
        part = writer.reserve_part()
        self._slots.acquire()
        future = self._pool.submit(self._write_part, stream, writer, part, buffer)
        future.add_done_callback(lambda _: self._slots.release())
        self._futures.append(future)

    def _write_part(self, stream: str, writer: SegmentWriter, part: int, buffer: _StreamBuffer) -> None:
        spec = self._stores[stream].spec
        tensors = _pack(spec.layout, buffer)  # offsets (b + 1,) and values (n, w), or (b, w), or indptr (b + 1,) and (nnz,)
        identities = self._row_records(stream, buffer.sequences, buffer.digests)  # (b,) one identity per sequence
        residues = buffer.rows if spec.layout in (RAGGED, RAGGED_TOPK) else [0] * len(buffer.sequences)  # (b,) stored rows
        writer.append_packed(part, buffer.digests, tensors, residues, row_metadata=identities)

    def _drain(self) -> None:
        """Wait for every part in flight, surfacing the first write error."""
        while self._futures:
            self._futures.popleft().result()

    def _commit_group(self, *, reopen: bool = False) -> None:
        """Flush every stream, commit its open segment, and optionally open the next segment group."""
        for stream in self._stores:
            self._flush(stream)
        self._drain()
        self._checked = False
        for stream, writer in self._writers.items():
            with self._commit_lock:
                if self._aborted:
                    raise RuntimeError("The feature writer was aborted before its segment committed.")
                if writer.parts:
                    self._receipts[stream].append(writer.commit())
                else:
                    writer.abandon()
        self._stack.close()
        if reopen:
            self._stack = ExitStack()
            self._segment_index += 1
            self._segment_written = 0
            self._open_segments()


def _compress_csr(dense: Tensor) -> tuple[dict[str, Tensor], list[int]]:
    """Keep the nonzero codes of each sequence's row, in row-major order, as indices and values."""
    # dense: (b, c) float32, zero where no code fired anywhere in the sequence
    present = dense != 0  # (b, c)
    counts = present.sum(dim=1).tolist()  # (b,) nnz per sequence
    columns = present.nonzero()[:, 1].to(INDEX_DTYPE)  # (nnz,)
    return {"indices": columns, "values": dense[present]}, [int(count) for count in counts]  # values (nnz,)


def _select_rows(
    layout: str, tensors: Mapping[str, Tensor], rows: Sequence[int], nnz: Sequence[int], keep: Sequence[int],
) -> tuple[dict[str, Tensor], list[int], list[int]]:
    """Gather the sequences at ``keep`` from packed batch tensors, for a rerun that fills only some rows."""
    # tensors: (b, w) when dense, (n, w) when ragged, (nnz,) when csr; the kept rows keep the layout.
    if layout == DENSE:
        index = torch.tensor(list(keep), dtype=torch.int64)  # (m,) m kept sequences
        picked = {name: tensor[index] for name, tensor in tensors.items()}
        return picked, [rows[i] for i in keep], [0] * len(keep)  # (m, w) per stream, m kept sequences
    spans = rows if layout in (RAGGED, RAGGED_TOPK) else nnz
    pieces = {name: torch.split(tensor, list(spans)) for name, tensor in tensors.items()}  # (span_i, ...) per sequence
    selected = {name: torch.cat([parts[i] for i in keep]) for name, parts in pieces.items()}
    return selected, [rows[i] for i in keep], [nnz[i] for i in keep]  # (n_kept, w) or (nnz_kept,) per stream


def _take(
    buffer: _StreamBuffer, layout: str, tensors: Mapping[str, Tensor], sequences: Sequence[str],
    digests: Sequence[str], rows: Sequence[int], nnz: Sequence[int], start: int, stop: int, used: int,
) -> None:
    """Move sequences ``[start, stop)`` of a batch into the stream's part buffer, slicing the packed tensors."""
    # tensors: (b, w) when dense, (n, w) when ragged, (nnz,) when csr; rows low:high of each go to the buffer.
    buffer.sequences.extend(sequences[start:stop])
    buffer.digests.extend(digests[start:stop])
    buffer.rows.extend(rows[start:stop])
    buffer.nnz.extend(nnz[start:stop])
    if layout == DENSE:
        low, high = start, stop  # one row per sequence
    else:
        spans = rows if layout in (RAGGED, RAGGED_TOPK) else nnz
        low, high = sum(spans[:start]), sum(spans[:stop])  # first and last packed row of the slice
    for name, tensor in tensors.items():
        buffer.tensors.setdefault(name, []).append(tensor[low:high])
    buffer.nbytes += used


def _pack(layout: str, buffer: _StreamBuffer) -> dict[str, Tensor]:
    """Concatenate a part's batches into the layout's tensors, with offsets naming each sequence's span."""
    joined = {name: torch.cat(parts) for name, parts in buffer.tensors.items()}  # (n, ...) one copy per tensor
    if layout == DENSE:
        return joined  # values (b, w)
    spans = torch.tensor(buffer.rows if layout in (RAGGED, RAGGED_TOPK) else buffer.nnz, dtype=OFFSET_DTYPE)  # (b,)
    offsets = torch.zeros(len(spans) + 1, dtype=OFFSET_DTYPE)  # (b + 1,)
    offsets[1:] = torch.cumsum(spans, dim=0)
    return {("indptr" if layout == CSR else "offsets"): offsets, **joined}  # offsets (b + 1,); values (n, w) or (nnz,)


__all__ = ["AsyncFeatureWriter", "PackedBatch"]
