"""Embed canonical proteins with CLS and EOS kept: one pass per token-budget batch, no host stall, pinned outputs.

A canonical run keeps the special tokens. A protein of l residues, after the N-terminal crop, is l + 2 rows in
every per-token stream: row 0 CLS, rows 1..l residues, row l + 1 EOS. Padding is the only masked position, so a
pooled vector averages all l + 2 rows. A legacy residue-only store has l rows (b, l, d) instead; the two never
mix, because a v2 descriptor says ``special_tokens: kept``.

The executor never reads a device value while it builds or runs a batch. Lengths are known on the host, so the
attention mask, the row selection, and every offset come from host arrays, and the finite check is one device
flag read after the outputs land. That keeps the device queue full while the previous batch is written.

A run with a ``BatchGeometry`` gives every sequence one batch shape, whatever its companions: its l + 2 tokens round
up to a bucket of T columns, and every batch of that bucket holds exactly ``rows(T)`` sequences padded to T. A
GEMM's reduction order follows its shape, so a fixed shape per sequence is what makes a stored row independent of
the batch that made it. Without a geometry, batches follow the token budget and pad to their longest member.

Symbols: b sequences of a batch; l residues of one sequence after the crop; m = max(l) + 2 padded token columns,
or the bucket T of a geometry batch; n = sum(l_i + 2) attended token rows; d hidden width; c SAE codebook; k SAE
codes kept per token; w stored columns of a stream.
"""

from __future__ import annotations

import numpy as np
import torch

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from torch import Tensor

from .pooling import pool_token_rows
from .taps import HiddenTap, ReducedTap, RowSelection, SparseResidueTap, StreamingTap, TapBatch, TapPlan
from .tokens import ResidueVocabulary
from ..features.async_writer import PackedBatch
from ..features.layouts import INDEX_DTYPE


# CLS before the residues and EOS after them.
SPECIAL_TOKEN_ROWS = 2
# The N-terminal crop in residues: with CLS and EOS it fills a 2,048-token context. The one place the engine names it.
CANONICAL_MAX_RESIDUES = 2046
# A geometry run's batch algorithm and its rule for a bucket's last, partial batch.
GEOMETRY_ALGORITHM = "bucketed_fixed_shape_v1"
PARTIAL_BATCH_POLICY = "repeat_last_sequence_discard_duplicate_outputs_v1"


@dataclass(frozen=True, slots=True)
class BatchGeometry:
    """Fixed batch shapes, so a sequence meets the same kernels in whatever batch it runs.

    A sequence of l residues after the crop needs l + 2 token columns and runs in the bucket
    T = ``bucket_tokens`` * ceil((l + 2) / ``bucket_tokens``), at most ``max_columns``. Every batch of bucket T holds
    exactly ``rows(T)`` sequences, as many as ``token_budget`` padded token rows hold and at most ``max_rows``, each
    padded to T columns; a bucket's last batch repeats its last sequence to fill and discards the copies' outputs.
    """

    bucket_tokens: int
    token_budget: int  # padded token rows one batch may hold
    max_rows: int  # sequences one batch may hold
    max_columns: int = CANONICAL_MAX_RESIDUES + SPECIAL_TOKEN_ROWS

    def __post_init__(self) -> None:
        for name in ("bucket_tokens", "token_budget", "max_rows", "max_columns"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"BatchGeometry.{name} must be a positive integer; received {value!r}.")
        if self.max_columns % self.bucket_tokens:
            raise ValueError("BatchGeometry.max_columns must be a multiple of bucket_tokens.")
        if self.token_budget < self.max_columns:
            raise ValueError("BatchGeometry.token_budget must hold one sequence of max_columns tokens.")

    def columns(self, residues: int) -> int:
        """The bucket T of a sequence of ``residues`` after the crop: its l + 2 tokens rounded up to the bucket width."""
        tokens = residues + SPECIAL_TOKEN_ROWS
        if residues < 1 or tokens > self.max_columns:
            raise ValueError(
                f"A sequence of {residues} residues needs {tokens} token columns; this geometry holds 3 to {self.max_columns}."
            )
        return -(-tokens // self.bucket_tokens) * self.bucket_tokens

    def rows(self, columns: int) -> int:
        """Sequences in every batch of bucket ``columns``: as many as the token budget holds, at most ``max_rows``."""
        if columns % self.bucket_tokens or not 0 < columns <= self.max_columns:
            raise ValueError(f"{columns} is not a bucket of this geometry.")
        return min(self.max_rows, self.token_budget // columns)

    def shapes(self) -> dict[int, int]:
        """Every bucket T with the row count rows(T) of its batches."""
        return {
            columns: self.rows(columns)
            for columns in range(self.bucket_tokens, self.max_columns + 1, self.bucket_tokens)
        }

    def describe(self) -> dict[str, int | str]:
        """The geometry as a run record and a feature contract state it."""
        return {
            "algorithm": GEOMETRY_ALGORITHM, "bucket_tokens": self.bucket_tokens, "token_budget": self.token_budget,
            "max_rows": self.max_rows, "max_columns": self.max_columns, "partial_batch": PARTIAL_BATCH_POLICY,
        }


def plan_geometry_batches(
    lengths: Sequence[int], digests: Sequence[str], geometry: BatchGeometry,
) -> Iterator[tuple[int, ...]]:
    """Batches of one bucket each: the longest bucket first, members in row-key order, ``rows(T)`` to a batch.

    ``lengths`` holds the residues l of each sequence after the crop and ``digests`` their row keys. A bucket's last
    batch may hold fewer sequences; the executor fills it to ``rows(T)``. The longest bucket first makes an
    out-of-memory failure show on the first batch, and key order makes the plan independent of the input's order.
    Yields indices into ``lengths``.
    """
    if len(lengths) != len(digests):
        raise ValueError("plan_geometry_batches needs one row key per length.")
    buckets: dict[int, list[int]] = {}
    for index, residues in enumerate(lengths):
        buckets.setdefault(geometry.columns(residues), []).append(index)
    for columns in sorted(buckets, reverse=True):
        members = sorted(buckets[columns], key=lambda index: digests[index])
        size = geometry.rows(columns)
        for start in range(0, len(members), size):
            yield tuple(members[start:start + size])


def plan_token_batches(
    lengths: Sequence[int], *, max_sequences: int, max_tokens: int, window: int,
) -> Iterator[tuple[int, ...]]:
    """Group sequences into batches of similar length under a padded-token budget.

    ``lengths`` holds the residues l of each sequence after the crop. Sequences are sorted longest first
    inside each window of ``window`` consecutive sequences, so a batch pads to its first member: it holds
    at most ``max_sequences`` sequences and ``b * (l_first + 2) <= max_tokens`` padded token rows. Longest
    first also makes an out-of-memory failure show on the first batch. Yields indices into ``lengths``.
    """
    if min(max_sequences, max_tokens, window) < 1:
        raise ValueError("max_sequences, max_tokens and window must be positive.")
    for start in range(0, len(lengths), window):
        order = sorted(range(start, min(start + window, len(lengths))), key=lambda index: (-lengths[index], index))
        batch: list[int] = []
        for index in order:
            if lengths[index] + SPECIAL_TOKEN_ROWS > max_tokens:
                raise ValueError(
                    f"A sequence of {lengths[index]} residues needs {lengths[index] + SPECIAL_TOKEN_ROWS} token rows, "
                    f"more than max_tokens={max_tokens}."
                )
            padded_rows = (len(batch) + 1) * (lengths[batch[0]] + SPECIAL_TOKEN_ROWS) if batch else 0
            if batch and (len(batch) + 1 > max_sequences or padded_rows > max_tokens):
                yield tuple(batch)
                batch = []
            batch.append(index)
        if batch:
            yield tuple(batch)


@dataclass(frozen=True, slots=True)
class HostBatch:
    """The integer arrays of one batch, built on the host from known lengths."""

    input_ids: np.ndarray  # (b, m) int64, CLS, residue ids, EOS, then padding
    rows: np.ndarray  # (b,) int64, l_i + 2 attended tokens of each sequence
    flat_index: np.ndarray  # (n,) int64, each attended token's position in the row-major (b * m) grid
    owner: np.ndarray  # (n,) int64, the sequence of each attended token


def build_host_batch(vocabulary: ResidueVocabulary, texts: Sequence[str], *, columns: int | None = None) -> HostBatch:
    """Token ids, lengths, and the row selection of ``texts`` (already cropped), with no device value.

    ``columns`` pads every sequence to that many token columns (a geometry batch's bucket T); None pads to the
    longest sequence.
    """
    encoded = [vocabulary.encode(text) for text in texts]  # b arrays of (l_i + 2,)
    rows = np.fromiter((len(ids) for ids in encoded), dtype=np.int64, count=len(encoded))  # (b,)
    longest = int(rows.max())
    if columns is not None and columns < longest:
        raise ValueError(f"A batch padded to {columns} columns cannot hold a sequence of {longest} tokens.")
    m = longest if columns is None else columns
    input_ids = np.full((len(encoded), m), vocabulary.pad_id, dtype=np.int64)  # (b, m)
    for index, ids in enumerate(encoded):
        input_ids[index, : len(ids)] = ids
    owner = np.repeat(np.arange(len(encoded), dtype=np.int64), rows)  # (n,)
    starts = np.cumsum(rows) - rows  # (b,) first packed row of each sequence
    within = np.arange(int(rows.sum()), dtype=np.int64) - np.repeat(starts, rows)  # (n,) token index in its sequence
    return HostBatch(input_ids, rows, owner * m + within, owner)


def _to_device(arrays: Sequence[np.ndarray], device: torch.device) -> list[Tensor]:
    """Copy several host arrays to the device in one transfer from pinned memory, without blocking the host."""
    # arrays: (b, m), (b,), (n,), (n,) int64 for the executor's batch; each comes back with its own shape.
    flat = np.concatenate([array.reshape(-1) for array in arrays])  # (total,) int64
    if device.type == "cuda":
        staging = torch.empty(flat.shape[0], dtype=torch.int64, pin_memory=True)  # (total,)
        staging.numpy()[:] = flat
        moved = staging.to(device, non_blocking=True)  # (total,)
    else:
        moved = torch.from_numpy(flat).to(device)  # (total,)
    pieces, cursor = [], 0
    for array in arrays:
        pieces.append(moved[cursor : cursor + array.size].view(array.shape))
        cursor += array.size
    return pieces  # views shaped like arrays, e.g. (b, m), (b,), (n,), (n,), of one device buffer


class TokenTapExecutor:
    """Run a tap plan over canonical proteins and hand each batch to the writer as pinned host tensors.

    ``plan`` holds the taps. Every tap sees all attended tokens, CLS and EOS included, so a hidden tap keeps
    ``(n, d)`` rows and a pooled tap averages l + 2 rows. ``max_residues`` is the N-terminal crop. ``dtype`` is
    the run dtype a tap converts to unless it names its own, and None keeps the model's dtype. ``geometry`` runs
    every batch at its bucket's fixed shape (``BatchGeometry``); ``fixed_batch_size`` fills every batch to that
    many sequences but pads it to its longest member. A run uses one of the two at most.
    """

    def __init__(
        self, model: Any, plan: TapPlan, *, vocabulary: ResidueVocabulary, max_residues: int | None,
        dtype: torch.dtype | None,
        fixed_batch_size: int | None = None,
        geometry: BatchGeometry | None = None,
    ) -> None:
        if getattr(model, "embedding_tap_support", False) is not True:
            raise ValueError(f"{type(model).__name__} does not support one-pass taps.")
        if any(isinstance(tap, StreamingTap) for tap in plan.taps) and getattr(
            model, "embedding_streaming_tap_support", False
        ) is not True:
            raise ValueError("This model does not support streaming hidden-state taps.")
        if max_residues is not None and max_residues < 1:
            raise ValueError("max_residues must be positive.")
        if fixed_batch_size is not None and (type(fixed_batch_size) is not int or fixed_batch_size < 1):
            raise ValueError("fixed_batch_size must be a positive integer.")
        if geometry is not None:
            if fixed_batch_size is not None:
                raise ValueError("A run takes its batch shapes from a geometry or a fixed batch size, not both.")
            if max_residues is None or max_residues + SPECIAL_TOKEN_ROWS > geometry.max_columns:
                raise ValueError("A geometry run needs a crop whose l + 2 tokens fit the geometry's widest bucket.")
        self.model = model
        self.plan = plan
        self.vocabulary = vocabulary
        self.max_residues = max_residues
        self.dtype = dtype
        self.fixed_batch_size = fixed_batch_size
        self.geometry = geometry
        self.device = next(model.parameters()).device

    def crop(self, sequence: str) -> str:
        """The N-terminal crop: the first ``max_residues`` residues, so l <= max_residues and l + 2 tokens."""
        return sequence if self.max_residues is None else sequence[: self.max_residues]

    def batch_shape(self, texts: Sequence[str]) -> tuple[int, int] | None:
        """The (rows, columns) a geometry runs ``texts`` (already cropped) at; None without a geometry."""
        if self.geometry is None:
            return None
        buckets = {self.geometry.columns(len(text)) for text in texts}
        if len(buckets) != 1:
            raise ValueError(f"A geometry batch holds sequences of one bucket; these fall in {sorted(buckets)}.")
        (columns,) = buckets
        rows = self.geometry.rows(columns)
        if len(texts) > rows:
            raise ValueError(f"Bucket {columns} runs {rows} sequences to a batch; received {len(texts)}.")
        return rows, columns

    def run_batch(self, sequences: Sequence[str], digests: Sequence[str]) -> PackedBatch:
        """Embed one batch and start the copies of its outputs to the host; returns without waiting for them."""
        count = len(sequences)
        if not count or len(digests) != count:
            raise ValueError("A batch needs at least one sequence and one digest per sequence.")
        if self.fixed_batch_size is not None and count > self.fixed_batch_size:
            raise ValueError("The input exceeds the fixed physical batch size.")
        texts = [self.crop(sequence) for sequence in sequences]
        shape = self.batch_shape(texts)
        columns = None
        if shape is not None:
            rows, columns = shape
            texts += [texts[-1]] * (rows - count)
        elif self.fixed_batch_size is not None:
            texts += [texts[-1]] * (self.fixed_batch_size - count)
        host = build_host_batch(self.vocabulary, texts, columns=columns)
        input_ids, rows, flat_index, owner = _to_device(
            (host.input_ids, host.rows, host.flat_index, host.owner), self.device,
        )  # (b, m), (b,), (n,), (n,)
        m = input_ids.shape[1]
        # (b, m): true on CLS, residues and EOS; padding is the only masked position.
        token_mask = torch.arange(m, device=self.device).unsqueeze(0) < rows.unsqueeze(1)  # (b, m)
        selection = RowSelection(flat_index, owner, tuple(int(count) for count in host.rows), rows)
        cache: dict[Any, Any] = {}

        def batch_for(X: Tensor) -> TapBatch:
            # X: (b, m, d) one layer's hidden state; padding columns stay in X and are masked by token_mask (b, m).
            return TapBatch(X=X, token_mask=token_mask, residue_mask=token_mask, rows=selection, cache=cache)

        streaming = tuple(tap for tap in self.plan.taps if isinstance(tap, StreamingTap))
        accumulators = {tap.name: tap.begin() for tap in streaming}
        stream_layers = self.plan.streamed_layers

        def consume(layer: int, X: Tensor) -> None:
            # X: (b, m, d), borrowed until this callback returns
            borrowed = batch_for(X)
            for tap in streaming:
                if layer in tap.layers:
                    accumulators[tap.name].update(layer, borrowed)

        states = self.model._embed_taps(
            input_ids, token_mask, self.plan.captured_layers, stream_layers=stream_layers,
            state_consumer=consume if streaming else None, assume_valid_mask=True,
        )  # {layer: (b, m, d)}
        outputs: dict[str, dict[str, Tensor]] = {}
        for tap, layer in zip(self.plan.taps, self.plan.layers, strict=True):
            if isinstance(tap, StreamingTap):
                Y = accumulators[tap.name].finish()  # (b, m, w)
                if tap.dtype is not None:
                    Y = Y.to(tap.dtype)  # (b, m, w)
                if tap.pooling is None:
                    outputs[tap.name] = {"values": selection.gather(Y)}  # (n, w)
                else:
                    outputs[tap.name] = {"values": pool_token_rows(Y, token_mask, tap.pooling)}  # (b, p * w)
                continue
            dtype = tap.dtype if isinstance(tap, HiddenTap) and tap.dtype is not None else self.dtype
            X = states[layer]  # (b, m, d)
            if dtype is not None:
                X = X.to(dtype)  # (b, m, d)
            if isinstance(tap, SparseResidueTap):
                if tap.reduce_packed is None:
                    raise ValueError(f"Tap {tap.name!r} has no packed reducer; it cannot keep special tokens.")
                packed = tap.reduce_packed(batch_for(X))  # indices, values (n, k)
                outputs[tap.name] = {  # indices and values: (n, k)
                    "indices": packed.indices.to(INDEX_DTYPE),
                    "values": packed.values,
                }
            elif isinstance(tap, ReducedTap):
                outputs[tap.name] = {"values": tap.reduce(batch_for(X))}  # (b, w)
            elif tap.pooling is None:
                outputs[tap.name] = {"values": selection.gather(X)}  # (n, d)
            else:
                outputs[tap.name] = {"values": pool_token_rows(X, token_mask, tap.pooling)}  # (b, p * d)
        if len(texts) != count:
            token_rows = int(host.rows[:count].sum())
            for tap in self.plan.taps:
                per_token = isinstance(tap, SparseResidueTap) or (
                    isinstance(tap, (HiddenTap, StreamingTap)) and tap.pooling is None)
                retained = token_rows if per_token else count
                outputs[tap.name] = {name: tensor[:retained] for name, tensor in outputs[tap.name].items()}
        return self._deliver(outputs, sequences, digests, tuple(int(rows) for rows in host.rows[:count]))

    def _deliver(
        self, outputs: Mapping[str, Mapping[str, Tensor]], sequences: Sequence[str], digests: Sequence[str],
        rows: tuple[int, ...],
    ) -> PackedBatch:
        """Start the device-to-host copies, record one event after them, and check finiteness on the device."""
        # outputs: (b, w) pooled, (n, w) per token row, (n, k) top-k indices, per tap; n = sum(l_i + 2).
        finite = [torch.isfinite(tensor).all() for group in outputs.values() for tensor in group.values()
                  if tensor.is_floating_point()]  # () per floating tensor
        everything_finite = torch.stack(finite).all() if finite else torch.ones((), dtype=torch.bool, device=self.device)  # ()
        if self.device.type != "cuda":
            hosts = {name: dict(group) for name, group in outputs.items()}
            verdict = everything_finite

            def wait() -> None:
                if not bool(verdict):
                    raise ValueError("A tap produced a non-finite value.")
        else:
            hosts = {name: {key: _pinned_copy(tensor) for key, tensor in group.items()} for name, group in outputs.items()}
            flag = _pinned_copy(everything_finite)
            event = torch.cuda.Event()
            event.record()

            def wait() -> None:
                event.synchronize()
                if not bool(flag):
                    raise ValueError("A tap produced a non-finite value.")

        return PackedBatch(tuple(sequences), tuple(digests), rows, hosts, wait)


def _pinned_copy(tensor: Tensor) -> Tensor:
    """A pinned host tensor the copy of ``tensor`` is queued into on the current stream, without waiting."""
    # tensor: (...) any shape; the host copy has the same shape and dtype.
    host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)  # (...)
    host.copy_(tensor.detach(), non_blocking=True)
    return host  # (...) the shape of tensor


__all__ = [
    "CANONICAL_MAX_RESIDUES",
    "GEOMETRY_ALGORITHM",
    "PARTIAL_BATCH_POLICY",
    "SPECIAL_TOKEN_ROWS",
    "BatchGeometry",
    "HostBatch",
    "TokenTapExecutor",
    "build_host_batch",
    "plan_geometry_batches",
    "plan_token_batches",
]
