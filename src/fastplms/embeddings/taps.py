"""Tap plans: several hidden-state outputs from one forward pass per batch.

A tap names one hidden state and what to keep from it. ``HiddenTap`` keeps the rows the mask selects
(biological residues in a residue run; CLS, residues and EOS in a token run) or pools them
(``Pooler`` in a residue run, ``pool_token_rows`` in a token run). ``ReducedTap`` hands the state to a
caller's reducer, such as a sparse-autoencoder encoder and its pooling. ``StreamingTap`` reduces layers
as they arrive without saving their full hidden states. A plan runs one forward pass per
batch that stops once the deepest tapped state exists.

Layer indices follow the FastPLMs hidden-state order: index ``i`` is the input to block ``i``,
index ``n`` (the block count) is the final normalized state, and negative indices count back
from it, so ``-1`` is the final state.

Symbols: b sequences of a batch; l token columns of the padded batch (CLS, residues, EOS, padding); d hidden
width; n attended rows of a batch; r residues of one sequence. In a residue run the mask is false on CLS, EOS and
padding, so a sequence is r rows of an ``(n, d)`` output. In a token run (canonical) it is false on padding only, so a
sequence is r + 2 rows (row 0 CLS, rows 1..r residues, row r + 1 EOS) in every ``(n, d)`` output and every pooling.
"""

from __future__ import annotations

import json
import math
import torch

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol
from torch import Tensor

from types import MappingProxyType
from .pooling import Pooler
from ..features.layouts import TopKRow


@dataclass(frozen=True, slots=True)
class RowSelection:
    """The rows of a ``(b, l)`` grid that a tap keeps, known on the host so no device value is read.

    ``flat_index`` holds each kept row's position in the row-major ``(b * l)`` grid, ``owner`` the
    sequence it belongs to, and ``counts`` how many rows each sequence keeps. Gathering with an index
    built from host-known lengths never stalls the host the way boolean indexing does.
    """

    flat_index: Tensor  # (n,) int64 on the device, n = sum of counts
    owner: Tensor  # (n,) int64 on the device, the sequence of each kept row
    counts: tuple[int, ...]  # (b,) kept rows per sequence, on the host
    sizes: Tensor  # (b,) int64 on the device, the same counts

    def gather(self, X: Tensor) -> Tensor:
        """The kept rows of ``X`` in sequence order, without padding."""
        # X: (b, l, w) -> (n, w)
        return X.reshape(-1, X.shape[-1]).index_select(0, self.flat_index)  # (n, w)


@dataclass(frozen=True, slots=True)
class TapBatch:
    """One batch of a tapped hidden state, as a ``ReducedTap`` reducer receives it.

    ``X`` has shape ``(b, l, d)``. ``token_mask`` has shape ``(b, l)`` and marks every attended
    token, BOS and EOS included. ``residue_mask`` has shape ``(b, l)`` and marks the rows that
    ``HiddenTap`` keeps and pools: the biological residues, or, when the run keeps the special
    tokens, every attended token (``l`` then counts CLS and EOS). All three share X's device.
    ``rows`` is the host-known selection of ``residue_mask`` when the executor has one, and
    ``cache`` lets taps of one batch share an intermediate such as an SAE encoding.
    """

    X: Tensor
    token_mask: Tensor
    residue_mask: Tensor
    rows: RowSelection | None = None
    cache: dict[Any, Any] = field(default_factory=dict)

    def selection(self) -> RowSelection:
        """The kept rows, taken from the executor when it supplied them, else read from the mask."""
        if self.rows is not None:
            return self.rows
        kept = self.residue_mask.bool()  # (b, l)
        owner, position = kept.nonzero(as_tuple=True)  # (n,), (n,)
        sizes = kept.sum(dim=1)  # (b,)
        return RowSelection(owner * kept.shape[1] + position, owner, tuple(sizes.tolist()), sizes)


@dataclass(frozen=True, slots=True)
class HiddenTap:
    """One hidden state, kept as ragged rows or pooled.

    ``pooling=None`` keeps one ``(r_i, d)`` tensor of mask-selected rows per sequence: ``r_i`` biological
    residues in a residue run, ``l_i + 2`` rows (CLS, residues, EOS) in a token run. Pooler
    names give one pooled vector per sequence over those same rows, concatenated in request order. ``parti`` needs the
    attention graph of a full pass, so a tap rejects it. ``dtype`` overrides the run's dtype
    for this output, starting from the original captured state; None inherits the run dtype.
    """

    name: str
    layer: int
    pooling: str | Sequence[str] | None = None
    dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        _require_name_and_layer(self.name, self.layer)
        if self.dtype is not None and self.dtype not in (
            torch.float16, torch.bfloat16, torch.float32, torch.float64
        ):
            raise ValueError(
                "A hidden tap dtype must be float16, bfloat16, float32, float64, "
                "or None to inherit the run dtype."
            )
        if self.pooling is None:
            return
        names = Pooler(self.pooling).names  # validates names and rejects duplicates
        if "parti" in names:
            raise ValueError(
                f"Tap {self.name!r} cannot pool with 'parti', which needs the attention graph "
                "of a full forward pass."
            )
        object.__setattr__(self, "pooling", names)


@dataclass(frozen=True, slots=True)
class ReducedTap:
    """One hidden state reduced by a caller's function.

    ``reduce`` maps a ``TapBatch`` to a tensor with one row per sequence, shape ``(b, ...)``.
    ``identity`` describes the reducer in plain data: strings, numbers, booleans, None, lists,
    and string-keyed mappings. The run fingerprint records it, so two reducers with equal
    identities must return equal outputs.
    """

    name: str
    layer: int
    reduce: Callable[[TapBatch], Tensor]
    identity: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_name_and_layer(self.name, self.layer)
        if not callable(self.reduce):
            raise TypeError(f"Tap {self.name!r} needs a callable reduce.")
        if not isinstance(self.identity, Mapping) or not self.identity:
            raise TypeError(f"Tap {self.name!r} needs a non-empty identity mapping.")
        _require_plain_data(self.identity, f"identity of tap {self.name!r}")
        # A private canonical copy, so later changes to the caller's mapping cannot change the
        # fingerprint of this tap.
        canonical = json.loads(json.dumps(dict(self.identity), sort_keys=True, allow_nan=False))
        object.__setattr__(self, "identity", MappingProxyType(canonical))


@dataclass(frozen=True, slots=True)
class SparseResidueTap:
    """Reduce one state to sparse codes per kept row, in sequence and row order.

    The reducer receives the same masks as dense taps and returns one ``TopKRow`` per sequence.
    Its output must retain every row the mask keeps: the biological residues in a residue run, all
    ``l_i + 2`` attended tokens in a token run. Both integer indices and floating
    values remain sparse through extraction and persistence.
    """

    name: str
    layer: int
    reduce: Callable[[TapBatch], Sequence[TopKRow]]
    identity: Mapping[str, Any]
    codebook_size: int
    sparse_count: int
    # A run that keeps CLS and EOS reads every sequence's codes as one packed (n, k) pair, n = sum(l_i + 2),
    # and splits nothing: the token executor calls this instead of ``reduce``. Not part of the identity.
    reduce_packed: Callable[[TapBatch], TopKRow] | None = None

    def __post_init__(self) -> None:
        checked = ReducedTap(self.name, self.layer, self.reduce, self.identity)
        object.__setattr__(self, "identity", checked.identity)
        if (type(self.codebook_size) is not int or not 1 <= self.codebook_size <= 2**31
                or type(self.sparse_count) is not int
                or not 1 <= self.sparse_count <= self.codebook_size):
            raise ValueError(
                "Sparse residue taps require integer 1 <= sparse_count <= codebook_size <= 2**31."
            )


class LayerAccumulator(Protocol):
    """Batch-local state for a streaming reduction. Never mutate the borrowed hidden state."""

    def update(self, layer: int, batch: TapBatch) -> None: ...

    def finish(self) -> Tensor:
        """Return a token-aligned tensor of shape (b, l, c)."""
        ...


@dataclass(frozen=True, slots=True)
class StreamingTap:
    """Reduce selected layers as they arrive, retaining only the accumulator's own state.

    ``begin`` creates a fresh accumulator for each batch. ``update`` borrows each original
    hidden state, before the run's output dtype conversion, in ascending layer order. ``finish``
    returns token-aligned residue features; the engine applies its biological mask and restores
    input order. Reducers own their arithmetic and describe it in ``identity``. They must not
    retain or mutate borrowed states. No callback is installed on the model between calls.

    ``pooling`` (token runs only) pools the finished rows of each sequence over every attended token, as a
    pooled ``HiddenTap`` does, so one value per sequence is kept instead of one per token; ``dtype`` converts
    the finished rows first (float32 for float32 moments of a 16-bit reducer).
    """

    name: str
    layers: tuple[int, ...]
    begin: Callable[[], LayerAccumulator]
    identity: Mapping[str, Any]
    required_state_count: int | None = None
    pooling: str | Sequence[str] | None = None
    dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        if self.dtype is not None and self.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError("A streaming tap dtype must be float16, bfloat16, float32, float64, or None.")
        if self.pooling is not None:
            names = Pooler(self.pooling).names  # validates names and rejects duplicates
            if "parti" in names:
                raise ValueError(f"Tap {self.name!r} cannot pool with 'parti', which needs the attention graph.")
            object.__setattr__(self, "pooling", names)
        layers = tuple(self.layers)
        if not layers or any(type(layer) is not int or layer < 0 for layer in layers):
            raise ValueError("Streaming layers must be nonempty nonnegative integer indices.")
        if tuple(sorted(set(layers))) != layers:
            raise ValueError("Streaming layers must be distinct and ascending.")
        object.__setattr__(self, "layers", layers)
        if self.required_state_count is not None and (
            type(self.required_state_count) is not int or self.required_state_count <= layers[-1]
        ):
            raise ValueError("required_state_count must include every streamed layer.")
        # Reuse the reducer identity validation and detached canonical copy.
        checked = ReducedTap(self.name, layers[-1], self.begin, self.identity)
        object.__setattr__(self, "identity", checked.identity)

    @property
    def layer(self) -> int:
        """The deepest required state, for the existing early-stop plan."""
        return self.layers[-1]


Tap = HiddenTap | ReducedTap | StreamingTap | SparseResidueTap


@dataclass(frozen=True, slots=True)
class TapPlan:
    """Validated taps, each layer resolved to a hidden-state index in ``0..n``."""

    taps: tuple[Tap, ...]
    layers: tuple[int, ...]

    @property
    def captured_layers(self) -> tuple[int, ...]:
        """The distinct hidden states the forward pass must record, in ascending order."""

        return tuple(sorted({
            layer for tap, layer in zip(self.taps, self.layers, strict=True)
            if not isinstance(tap, StreamingTap)
        }))

    @property
    def streamed_layers(self) -> tuple[int, ...]:
        return tuple(sorted({
            layer for tap in self.taps if isinstance(tap, StreamingTap) for layer in tap.layers
        }))

    @property
    def deepest_layer(self) -> int:
        """The hidden state after which the forward pass stops."""

        return max(self.layers)

    @property
    def pooling_names(self) -> frozenset[str]:
        """Every pooler name the plan's hidden taps request."""

        return frozenset(
            name
            for tap in self.taps
            if isinstance(tap, HiddenTap) and tap.pooling is not None
            for name in tap.pooling
        )

    def identity(self) -> list[dict[str, Any]]:
        """Every tap in plan order, as the run fingerprint and metadata record it."""

        described: list[dict[str, Any]] = []
        for tap, layer in zip(self.taps, self.layers, strict=True):
            if isinstance(tap, HiddenTap):
                pooling = None if tap.pooling is None else list(tap.pooling)
                described.append(
                    {
                        "name": tap.name,
                        "kind": "hidden",
                        "layer": layer,
                        "pooling": pooling,
                        "dtype": (
                            str(tap.dtype).removeprefix("torch.") if tap.dtype is not None else None
                        ),
                    }
                )
            elif isinstance(tap, StreamingTap):
                record = {
                    "name": tap.name, "kind": "streaming", "layers": list(tap.layers),
                    "identity": dict(tap.identity),
                    "required_state_count": tap.required_state_count,
                }
                if tap.pooling is not None:  # absent for a per-token tap, so its existing fingerprint holds
                    record.update(pooling=list(tap.pooling),
                                  dtype=None if tap.dtype is None else str(tap.dtype).removeprefix("torch."))
                described.append(record)
            elif isinstance(tap, SparseResidueTap):
                described.append({
                    "name": tap.name, "kind": "sparse_residue", "layer": layer,
                    "identity": dict(tap.identity), "codebook_size": tap.codebook_size,
                    "sparse_count": tap.sparse_count,
                })
            else:
                described.append(
                    {
                        "name": tap.name,
                        "kind": "reduced",
                        "layer": layer,
                        "identity": dict(tap.identity),
                    }
                )
        return described


def plan_taps(taps: object, state_count: int) -> TapPlan:
    """Validate ``taps`` against a model that exposes ``state_count`` hidden states.

    ``taps`` is the caller's ``embed_dataset`` argument, checked here rather than trusted.
    """

    if isinstance(taps, (str, bytes)) or not isinstance(taps, Sequence):
        raise TypeError(
            "taps must be a sequence of HiddenTap, ReducedTap, StreamingTap "
            "or SparseResidueTap values."
        )
    if not taps:
        raise ValueError("taps must contain at least one tap.")
    checked: list[Tap] = []
    for tap in taps:
        if not isinstance(tap, (HiddenTap, ReducedTap, StreamingTap, SparseResidueTap)):
            raise TypeError(
                "taps must contain HiddenTap, ReducedTap, StreamingTap or SparseResidueTap values; "
                f"found {type(tap).__name__}."
            )
        checked.append(tap)
    names = [tap.name for tap in checked]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"Tap names must be unique; repeated: {repeated}.")
    layers: list[int] = []
    for tap in checked:
        if isinstance(tap, StreamingTap) and tap.required_state_count not in (None, state_count):
            raise ValueError(
                f"Tap {tap.name!r} requires {tap.required_state_count} hidden states, "
                f"not {state_count}."
            )
        if not -state_count <= tap.layer < state_count:
            raise ValueError(
                f"Tap {tap.name!r} names layer {tap.layer}, outside this model's hidden states "
                f"{-state_count}..{state_count - 1}. Index i is the input to block i, and "
                f"{state_count - 1} or -1 is the final normalized state."
            )
        layers.append(tap.layer % state_count)
    return TapPlan(taps=tuple(checked), layers=tuple(layers))


def _require_name_and_layer(name: object, layer: object) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("A tap name must be a non-empty string.")
    if not isinstance(layer, int) or isinstance(layer, bool):
        raise TypeError(f"Tap {name!r} layer must be an integer hidden-state index.")


def _require_plain_data(value: object, where: str) -> None:
    """Reject content whose serialized form could differ between runs of one reducer."""

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"The {where} holds a non-finite number.")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"The {where} has a non-string key {key!r}.")
            _require_plain_data(item, where)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _require_plain_data(item, where)
        return
    raise TypeError(
        f"The {where} holds a {type(value).__name__}. An identity holds only strings, numbers, "
        "booleans, None, lists, and string-keyed mappings, so its fingerprint is stable."
    )


__all__ = [
    "HiddenTap",
    "LayerAccumulator",
    "ReducedTap",
    "RowSelection",
    "SparseResidueTap",
    "StreamingTap",
    "Tap",
    "TapBatch",
    "TapPlan",
    "plan_taps",
]
