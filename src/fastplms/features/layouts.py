"""The row layouts a feature segment stores, as memory-mappable safetensors tensors.

A feature is one value per sequence, and the layouts differ in what that value is:

- ``dense``: a fixed-width vector, ``values`` of shape ``(n, w)``. Pooled embeddings.
- ``csr``: a sparse fixed-width vector, compressed by row. Max-pooled sparse-autoencoder codes,
  where a row holds at most one entry per code that fired anywhere in the sequence. ``positions``
  carries the argmax residue of each entry, which is what makes a code interpretable as "here",
  and is optional because a pooling that discards it has nothing to store.
- ``ragged``: per-row values, ``values`` of shape ``(sum r_i, d)`` with ``offsets`` naming each
  sequence's span. Hidden states, where ``r_i`` is the sequence's stored row count: its biological residue
  count ``l`` in a residue-only (``feature_spec_v1``) store, and ``l + 2`` in a canonical store, whose rows are
  CLS, the ``l`` residues, then EOS.
- ``ragged_topk``: per-row sparse codes, ``indices`` and ``values`` of shape
  ``(sum r_i, k)`` with sequence ``offsets``, ``r_i`` as above. All k entries, including zeros, retain their
  order. Unlike pooled csr positions, these rows retain every stored row and support crop-local pooling.

Rows are addressed individually and never sliced by column, so compressed-sparse-row is the right
sparse layout: a batch materializes with one gather and no search. The index dtypes follow
enzyme_loop's measured store: int32 code indices, int16 residue positions, and int64 row offsets,
which keep an entry to eight bytes against four bytes per column dense.

Every layout stores its values in the feature's declared dtype, so a segment is a lossless record
of what the model produced at that precision, and no reader has to guess.
"""

from __future__ import annotations

import torch

from collections.abc import Sequence
from dataclasses import dataclass
from torch import Tensor


DENSE = "dense"
CSR = "csr"
RAGGED = "ragged"
RAGGED_TOPK = "ragged_topk"
LAYOUT_NAMES = (DENSE, CSR, RAGGED, RAGGED_TOPK)

INDEX_DTYPE = torch.int32
POSITION_DTYPE = torch.int16
OFFSET_DTYPE = torch.int64

VALUE_DTYPES = (torch.float64, torch.float32, torch.float16, torch.bfloat16)
DTYPE_NAMES = {dtype: str(dtype).removeprefix("torch.") for dtype in VALUE_DTYPES}
NAME_DTYPES = {name: dtype for dtype, name in DTYPE_NAMES.items()}


@dataclass(frozen=True, slots=True)
class TopKRow:
    """One sequence's sparse residue codes: integer indices and values, both ``(r_i, k)``.

    The store validates the declared codebook and k. Zero-valued entries and code order are
    retained exactly; there is deliberately no implicit residue-by-codebook densification.
    """

    indices: Tensor
    values: Tensor


@dataclass(frozen=True, slots=True)
class SparseRow:
    """One compressed row: the codes that fired, their values, and where each peaked.

    ``indices`` and ``values`` have shape ``(nnz_i,)``. ``positions`` has the same shape when the
    segment stores them, and is None when it does not.
    """

    indices: Tensor
    values: Tensor
    positions: Tensor | None

    def to_dense(self, width: int) -> Tensor:
        """The row as a ``(width,)`` vector, zero where no code fired."""

        # self.indices, self.values: (nnz,), the entries that fired
        dense = torch.zeros(width, dtype=self.values.dtype)  # (w,)
        dense[self.indices.to(torch.int64)] = self.values
        return dense  # (w,)

    @classmethod
    def from_dense(cls, vector: Tensor, positions: Tensor | None = None) -> SparseRow:
        """Compress a ``(w,)`` vector by keeping its exactly non-zero entries.

        Top-k sparse-autoencoder pooling leaves exact zeros where a code never fired, so this
        drops nothing a reader could want. It is not a threshold: a code that fired weakly is
        kept. ``positions`` is the full ``(w,)`` argmax residue vector, gathered at the same
        entries.
        """

        # vector: (w,); positions: (w,) or None
        if vector.ndim != 1:
            raise ValueError(f"A sparse row comes from a one-dimensional vector; received {tuple(vector.shape)}.")
        kept = torch.nonzero(vector, as_tuple=False).flatten()  # (nnz,)
        if positions is not None and positions.shape != vector.shape:
            raise ValueError(
                f"positions must have the vector's shape {tuple(vector.shape)}; received "
                f"{tuple(positions.shape)}."
            )
        return cls(
            indices=kept.to(INDEX_DTYPE),  # (nnz,)
            values=vector[kept],  # (nnz,)
            positions=None if positions is None else positions[kept],  # (nnz,)
        )


def dtype_name(dtype: torch.dtype) -> str:
    """The stored name of a value dtype, rejecting one no layout stores."""

    if dtype not in DTYPE_NAMES:
        raise ValueError(
            f"A feature stores float64, float32, float16, or bfloat16 values; received {dtype}."
        )
    return DTYPE_NAMES[dtype]


def value_dtype(name: str) -> torch.dtype:
    """The dtype a stored name means."""

    if name not in NAME_DTYPES:
        raise ValueError(f"Unknown feature value dtype {name!r}; expected one of {list(NAME_DTYPES)}.")
    return NAME_DTYPES[name]


def encode_dense(rows: Sequence[Tensor], width: int, dtype: torch.dtype) -> dict[str, Tensor]:
    """Stack ``(w,)`` rows into one ``values`` tensor of shape ``(n, w)``."""

    # rows: (w,) each, n tensors; w = width
    stacked = torch.empty((len(rows), width), dtype=dtype)  # (n, w)
    for position, row in enumerate(rows):
        vector = row.detach().to("cpu")  # (w,)
        if vector.ndim != 1 or vector.shape[0] != width:
            raise ValueError(
                f"A dense feature row must have shape ({width},); row {position} has "
                f"{tuple(vector.shape)}."
            )
        stacked[position] = vector.to(dtype)
    return {"values": stacked}  # {"values": (n, w)}


def row_tensor_bytes(
    row: Tensor | SparseRow | TopKRow, layout: str, width: int, dtype: torch.dtype,
    *, positions: bool,
) -> int:
    """Exact per-row encoded payload, including its offset but not the part's initial offset."""
    # row: (w,) dense, (r_i, d) ragged, or a SparseRow or TopKRow
    value_size = torch.empty(0, dtype=dtype).element_size()
    if layout == RAGGED_TOPK:
        if not isinstance(row, TopKRow):
            raise TypeError("Ragged top-k payload sizing requires a TopKRow.")
        return 8 + row.values.numel() * (4 + value_size)
    if layout == CSR:
        if not isinstance(row, SparseRow):
            raise TypeError("CSR payload sizing requires a SparseRow.")
        return 8 + row.values.numel() * (4 + value_size + (2 if positions else 0))
    if not isinstance(row, Tensor):
        raise TypeError("Dense and ragged payload sizing requires tensor rows.")
    return width * value_size if layout == DENSE else 8 + row.numel() * value_size


def encode_csr(
    rows: Sequence[SparseRow], width: int, dtype: torch.dtype
) -> dict[str, Tensor]:
    """Concatenate sparse rows, with ``indptr`` naming each row's span.

    Every row must agree about positions: either all carry them or none does, because one segment
    stores one tensor set.
    """

    # rows[i].indices, .values, .positions: (nnz_i,); the segment holds n rows and nnz = sum(nnz_i) entries
    with_positions = [row.positions is not None for row in rows]
    if any(with_positions) and not all(with_positions):
        raise ValueError(
            "Either every sparse row carries argmax positions or none does; this batch mixes both."
        )
    indptr = torch.zeros(len(rows) + 1, dtype=OFFSET_DTYPE)  # (n + 1,)
    indices: list[Tensor] = []
    values: list[Tensor] = []
    positions: list[Tensor] = []
    for position, row in enumerate(rows):
        if row.indices.dtype not in (torch.int16, torch.int32, torch.int64):
            raise ValueError("Sparse code indices must be signed integer tensors.")
        row_indices = row.indices.detach().to("cpu").to(torch.int64)  # (nnz_i,)
        row_values = row.values.detach().to("cpu")  # (nnz_i,)
        if row_indices.ndim != 1 or row_values.shape != row_indices.shape:
            raise ValueError(
                f"Sparse row {position} needs one-dimensional indices and values of equal length; "
                f"received {tuple(row_indices.shape)} and {tuple(row_values.shape)}."
            )
        if row_indices.numel() and (int(row_indices.min()) < 0 or int(row_indices.max()) >= width):
            raise ValueError(
                f"Sparse row {position} names a code outside 0..{width - 1}."
            )
        if row_indices.numel() and int(row_indices.max()) > torch.iinfo(INDEX_DTYPE).max:
            raise ValueError("Sparse code index exceeds the stored integer range.")
        if row_indices.unique().numel() != row_indices.numel():
            raise ValueError("Sparse code indices must be unique within each row.")
        indptr[position + 1] = int(indptr[position]) + row_indices.numel()
        indices.append(row_indices.to(INDEX_DTYPE))
        values.append(row_values.to(dtype))
        if row.positions is not None:
            if row.positions.dtype not in (torch.int16, torch.int32, torch.int64):
                raise ValueError("Sparse residue positions must be signed integer tensors.")
            row_positions = row.positions.detach().to("cpu")  # (nnz_i,)
            if row_positions.shape != row_indices.shape:
                raise ValueError(
                    f"Sparse row {position} has {row_positions.numel()} positions for "
                    f"{row_indices.numel()} entries."
                )
            if row_positions.numel() and (
                int(row_positions.min()) < 0
                or int(row_positions.max()) > torch.iinfo(POSITION_DTYPE).max
            ):
                raise ValueError("Sparse residue position exceeds the stored integer range.")
            positions.append(row_positions.to(POSITION_DTYPE))

    tensors = {
        "indptr": indptr,  # (n + 1,)
        "indices": _concatenate(indices, INDEX_DTYPE),  # (nnz,)
        "values": _concatenate(values, dtype),  # (nnz,)
    }
    if positions:
        tensors["positions"] = _concatenate(positions, POSITION_DTYPE)  # (nnz,)
    return tensors  # indptr (n + 1,); indices, values and positions (nnz,)


def encode_ragged(rows: Sequence[Tensor], width: int, dtype: torch.dtype) -> dict[str, Tensor]:
    """Concatenate ``(r_i, d)`` residue blocks, with ``offsets`` naming each sequence's span."""

    # rows: (r_i, d) each, n tensors; d = width
    offsets = torch.zeros(len(rows) + 1, dtype=OFFSET_DTYPE)  # (n + 1,)
    blocks: list[Tensor] = []
    for position, row in enumerate(rows):
        block = row.detach().to("cpu")  # (r_i, d)
        if block.ndim != 2 or block.shape[1] != width:
            raise ValueError(
                f"A ragged feature row must have shape (r_i, {width}); row {position} has "
                f"{tuple(block.shape)}."
            )
        offsets[position + 1] = int(offsets[position]) + block.shape[0]
        blocks.append(block.to(dtype))
    values = (  # (sum r_i, d)
        torch.cat(blocks, dim=0) if blocks else torch.empty((0, width), dtype=dtype)
    )
    return {"offsets": offsets, "values": values}  # {"offsets": (n + 1,), "values": (sum r_i, d)}


def validate_topk(indices: Tensor, values: Tensor, width: int, count: int) -> None:
    """Reject malformed residue codes before writing and when reading committed tensors."""
    # indices, values: (r, k), with k = count; r residues of one sequence, or of every sequence of a part
    if indices.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError("Top-k code indices must be signed integer tensors.")
    if (indices.ndim != 2 or indices.shape[1] != count or values.shape != indices.shape):
        raise ValueError(f"Top-k indices and values must both have shape (residues, {count}).")
    if not values.is_floating_point() or not bool(torch.isfinite(values).all()):
        raise ValueError("Top-k values must be finite floating point tensors.")
    # Python integer bounds avoid narrowing 2**31 to int32 (or 16384 to int16).
    if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= width):
        raise ValueError(f"Top-k code index is outside 0..{width - 1}.")
    # Sorting validates uniqueness without changing the stored (residues,k) order.
    ordered = indices.sort(dim=1).values  # (r, k)
    if bool((ordered[:, 1:] == ordered[:, :-1]).any()):
        raise ValueError("Top-k indices must be unique within each residue.")


def encode_topk_rows(
    rows: Sequence[TopKRow], width: int, count: int, dtype: torch.dtype,
) -> dict[str, Tensor]:
    """Encode sparse residue rows without allocating a residue-by-codebook tensor."""
    # rows[i].indices, .values: (r_i, k), with k = count
    offsets = torch.zeros(len(rows) + 1, dtype=OFFSET_DTYPE)  # (n + 1,)
    indices, values = [], []
    for position, row in enumerate(rows):
        if not isinstance(row, TopKRow):
            raise TypeError("A ragged top-k feature requires TopKRow values.")
        row_indices = row.indices.detach().to("cpu")  # (r_i, k)
        row_values = row.values.detach().to("cpu")  # (r_i, k)
        validate_topk(row_indices, row_values, width, count)
        converted = row_values.to(dtype)  # (r_i, k)
        if not bool(torch.isfinite(converted).all()):
            raise ValueError("Top-k value conversion exceeded the stored dtype range.")
        offsets[position + 1] = int(offsets[position]) + row_values.shape[0]
        indices.append(row_indices.to(INDEX_DTYPE))
        values.append(converted)
    return {
        "offsets": offsets,  # (n + 1,)
        "indices": torch.cat(indices) if indices else torch.empty((0, count), dtype=INDEX_DTYPE),
        "values": torch.cat(values) if values else torch.empty((0, count), dtype=dtype),
    }  # offsets (n + 1,); indices and values (sum r_i, k)


def row_count(layout: str, tensors: dict[str, Tensor]) -> int:
    """How many sequences a segment's tensors hold."""

    # tensors: (n, w) values for dense, (n + 1,) indptr or offsets otherwise
    if layout == DENSE:
        return int(tensors["values"].shape[0])
    if layout == CSR:
        return int(tensors["indptr"].shape[0]) - 1
    if layout in (RAGGED, RAGGED_TOPK):
        return int(tensors["offsets"].shape[0]) - 1
    raise ValueError(f"Unknown feature layout {layout!r}; expected one of {list(LAYOUT_NAMES)}.")


def tensor_names(layout: str, *, positions: bool) -> tuple[str, ...]:
    """The tensors a segment of this layout holds, in a stable order."""

    if layout == DENSE:
        return ("values",)
    if layout == CSR:
        return ("indptr", "indices", "values", "positions") if positions else (
            "indptr", "indices", "values",
        )
    if layout == RAGGED:
        return ("offsets", "values")
    if layout == RAGGED_TOPK:
        return ("offsets", "indices", "values")
    raise ValueError(f"Unknown feature layout {layout!r}; expected one of {list(LAYOUT_NAMES)}.")


def _concatenate(parts: Sequence[Tensor], dtype: torch.dtype) -> Tensor:
    # parts: (n_i, ...) each, with one shared trailing shape
    if not parts:
        return torch.empty(0, dtype=dtype)  # (0,)
    return torch.cat(list(parts), dim=0)  # (sum n_i, ...)


__all__ = [
    "CSR",
    "DENSE",
    "DTYPE_NAMES",
    "INDEX_DTYPE",
    "LAYOUT_NAMES",
    "NAME_DTYPES",
    "OFFSET_DTYPE",
    "POSITION_DTYPE",
    "RAGGED",
    "RAGGED_TOPK",
    "VALUE_DTYPES",
    "SparseRow",
    "TopKRow",
    "dtype_name",
    "encode_csr",
    "encode_dense",
    "encode_ragged",
    "encode_topk_rows",
    "row_count",
    "row_tensor_bytes",
    "tensor_names",
    "validate_topk",
    "value_dtype",
]
