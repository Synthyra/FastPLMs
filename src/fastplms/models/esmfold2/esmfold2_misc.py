"""Small tensor, sequence, and annotation utilities used by ESMFold2.

The helpers in this module are deliberately free of model state. Importing the
module therefore performs no device selection, compilation, or remote access.
"""

from __future__ import annotations

import numpy as np
import torch
import zstandard

from collections import defaultdict
from collections.abc import Generator, Iterable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import is_dataclass
from io import BytesIO
from typing import Any, Protocol, TypeVar, runtime_checkable
from warnings import warn

from .esmfold2_constants_esm3 import CHAIN_BREAK_STR
from .esmfold2_utils_types import FunctionAnnotation


MAX_SUPPORTED_DISTANCE = 1e6

TSequence = TypeVar("TSequence", bound=Sequence)


@runtime_checkable
class Concatable(Protocol):
    """Protocol for sequence-like records with a class-level concatenator."""

    @classmethod
    def concat(cls, objs: list[Concatable]) -> Concatable: ...


def fp32_autocast_context(
    device_type: str,
) -> AbstractContextManager[Any]:  # type: ignore
    """Return a context that keeps numerically sensitive work in FP32."""

    if device_type == "mps":
        return nullcontext()
    if device_type == "cpu":
        return torch.amp.autocast(device_type, enabled=False)  # type: ignore
    if device_type == "cuda":
        return torch.amp.autocast(device_type, dtype=torch.float32)  # type: ignore
    raise ValueError(f"Unsupported device type: {device_type}")


def maybe_tensor(value, convert_none_to_nan: bool = False) -> torch.Tensor | None:
    """Convert an optional array-like value to a tensor."""
    # value: array-like arbitrary shape, or a list of identically shaped tensors.

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value  # value.shape
    if isinstance(value, list) and all(isinstance(element, torch.Tensor) for element in value):
        return torch.stack(value)  # (n_values, *element_shape)
    if convert_none_to_nan:
        value = np.asarray(value, dtype=np.float32)  # shape inferred from the nested input
        value = np.where(value is None, np.nan, value)  # value.shape
    return torch.tensor(value)  # shape inferred from the array-like input


def maybe_list(value, convert_nan_to_none: bool = False) -> list | None:
    """Convert an optional tensor or NumPy array to nested Python lists."""
    # value: arbitrary-shaped array/tensor; element order and nesting are retained.

    if value is None:
        return None
    if not convert_nan_to_none:
        return value.tolist()
    if isinstance(value, torch.Tensor):
        nan_mask = torch.isnan(value).cpu().numpy()  # value.shape
        array = value.cpu().numpy().astype(object)  # value.shape
    elif isinstance(value, np.ndarray):
        nan_mask = np.isnan(value)  # value.shape
        array = value.astype(object)  # value.shape
    else:
        raise TypeError("maybe_list can only work with torch.tensor or np.ndarray.")
    array[nan_mask] = None  # (n_nan,) selected elements
    return array.tolist()


def replace_inf(data):
    """Replace infinite array values by the ESM API sentinel value."""
    # data: array-like arbitrary shape; the returned list retains its nesting.

    if data is None:
        return None
    array = np.asarray(data, dtype=np.float32)  # shape inferred from data
    return np.where(np.isinf(array), 1000, array).tolist()


def slice_python_object_as_numpy(
    obj: TSequence,
    idx: int | list[int] | slice | np.ndarray,
) -> TSequence:
    """Apply NumPy-style scalar, mask, or index-array slicing to Python data."""

    normalized_idx: list[int] | slice | np.ndarray = (
        [int(idx)] if np.isscalar(idx) else idx  # type: ignore[arg-type]
    )

    if isinstance(normalized_idx, np.ndarray) and normalized_idx.dtype == bool:
        selected = [obj[position] for position in np.flatnonzero(normalized_idx)]
    elif isinstance(normalized_idx, slice):
        selected = obj[normalized_idx]
    else:
        selected = [obj[position] for position in normalized_idx]

    if isinstance(obj, str) and isinstance(selected, list):
        return "".join(selected)  # type: ignore[return-value]
    return obj.__class__(selected)  # type: ignore[call-arg,return-value]


def slice_any_object(
    obj: TSequence,
    idx: int | list[int] | slice | np.ndarray,
) -> TSequence:
    """Slice tensors, arrays, dataclasses, and ordinary Python sequences."""
    # Array shape is caller-defined; idx follows that array type's indexing rules.

    if isinstance(obj, (np.ndarray, torch.Tensor)) or is_dataclass(obj):
        return obj[idx]  # type: ignore[index,return-value]; shape determined by NumPy/Torch indexing when obj is an array
    return slice_python_object_as_numpy(obj, idx)


def join_lists(
    lists: Sequence[Sequence[Any]],
    separator: Sequence[Any] | None = None,
) -> list[Any]:
    """Join lists, inserting all elements of ``separator`` between inputs."""

    if len(lists) == 0:
        return []
    joined = list(lists[0])
    for values in lists[1:]:
        if separator:
            joined.extend(separator)
        joined.extend(values)
    return joined


def iterate_with_intermediate(
    lists: Iterable,
    intermediate,
) -> Generator[Any, None, None]:
    """Yield an intermediate value between consecutive input values."""

    iterator = iter(lists)
    yield next(iterator)
    for value in iterator:
        yield intermediate
        yield value


def concat_objects(objs: Sequence[Any], separator: Any | None = None):
    """Concatenate one supported homogeneous collection."""

    if not objs:
        raise ValueError("objs must contain at least one value.")
    first = objs[0]
    if isinstance(first, Concatable):
        return first.__class__.concat(objs)
    if isinstance(first, str):
        if not isinstance(separator, str):
            raise TypeError("separator must be a string when joining strings.")
        return separator.join(objs)
    if isinstance(first, list):
        return join_lists(objs, None if separator is None else [separator])
    if isinstance(first, np.ndarray):
        pieces = (
            objs
            if separator is None
            else list(iterate_with_intermediate(objs, np.array([separator])))
        )  # arrays with a common trailing shape, interleaved with a separator if supplied
        return np.concatenate(pieces)  # (sum of leading lengths, *common_trailing_shape)
    if isinstance(first, torch.Tensor):
        pieces = (
            objs
            if separator is None
            else list(iterate_with_intermediate(objs, torch.tensor([separator])))
        )  # tensors with a common trailing shape, interleaved with a separator if supplied
        return torch.cat(pieces)  # type: ignore[arg-type]; (sum of leading lengths, *common_trailing_shape)
    raise TypeError(type(first))


def rbf(values: torch.Tensor, v_min: float, v_max: float, n_bins: int = 16) -> torch.Tensor:
    """Encode values against evenly spaced radial basis centers."""
    # values: arbitrary shape; the output appends a final n_bins axis.

    centers = torch.linspace(
        v_min,
        v_max,
        n_bins,
        dtype=values.dtype,
        device=values.device,
    )  # (n_bins,)
    centers = centers.reshape((1,) * values.ndim + (-1,))  # (1, ..., 1, n_bins); values.ndim leading singleton axes
    standardized = (values.unsqueeze(-1) - centers) / ((v_max - v_min) / n_bins)  # (*values.shape, n_bins)
    return torch.exp(-(standardized**2))  # (*values.shape, n_bins)


def batched_gather(
    data: torch.Tensor, inds: torch.Tensor, dim: int = 0, no_batch_dims: int = 0
) -> torch.Tensor:
    """Gather along one data dimension while retaining leading batch axes."""
    # data/inds ranks are caller-defined. The first no_batch_dims axes index together;
    # remaining advanced-index axes broadcast, while slice axes retain their data lengths.

    batch_indices = []
    index_rank = len(inds.shape)
    for axis, size in enumerate(data.shape[:no_batch_dims]):
        shape = (1,) * axis + (-1,) + (1,) * (index_rank - axis - 1)
        batch_indices.append(torch.arange(size).view(*shape))  # index-rank tensor; only the batch axis has length size
    tail = [slice(None)] * (len(data.shape) - no_batch_dims)
    tail[dim - no_batch_dims if dim >= 0 else dim] = inds
    return data[tuple(batch_indices + tail)]  # broadcast advanced-index shape plus retained data slice axes


def node_gather(s: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Gather node features for each row of an edge-index tensor."""
    # s: (..., n_nodes, d); edges: (..., n_nodes, n_neighbors).

    return batched_gather(
        s.unsqueeze(-3),
        edges,
        -2,
        no_batch_dims=len(s.shape) - 1,
    )  # (..., n_nodes, n_neighbors, d)


def knn_graph(
    coords: torch.Tensor,
    coord_mask: torch.Tensor,
    padding_mask: torch.Tensor,
    sequence_id: torch.Tensor,
    *,
    no_knn: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build nearest-neighbor edges, using sequence distance for missing geometry."""
    # coords: (..., l, 3); masks: (..., l). With sequence_id, inputs use batch shape (b, l).

    length = coords.shape[-2]
    coords = coords.nan_to_num()  # (..., l, 3)
    missing_pair = ~(coord_mask[..., None, :] & coord_mask[..., :, None])  # (..., l, l)
    excluded_pair = padding_mask[..., None, :] | padding_mask[..., :, None]  # (..., l, l)
    if sequence_id is not None:
        excluded_pair |= sequence_id.unsqueeze(1) != sequence_id.unsqueeze(2)  # (b, l, l)

    distances = (coords.unsqueeze(-2) - coords.unsqueeze(-3)).norm(dim=-1)  # (..., l, l)
    residue_index = torch.arange(length, device=coords.device)  # (l,)
    sequence_distance = (residue_index.unsqueeze(-1) - residue_index.unsqueeze(-2)).abs()  # (l, l)
    if not (distances[~missing_pair] < MAX_SUPPORTED_DISTANCE).all():
        raise ValueError(
            "Coordinate pairwise distances exceed max supported distance "
            f"({MAX_SUPPORTED_DISTANCE}). "
        )

    rank_distance = sequence_distance.to(distances.dtype).mul(1e2).add(MAX_SUPPORTED_DISTANCE)  # (l, l)
    rank_distance = rank_distance.where(missing_pair, distances)  # (..., l, l)
    rank_distance = rank_distance.masked_fill(excluded_pair, torch.inf)  # (..., l, l)
    sorted_distance, sorted_edge = rank_distance.sort(dim=-1, descending=False)  # each (..., l, l)
    width = min(no_knn, length)
    return sorted_edge[..., :width], sorted_distance[..., :width].isfinite()  # each (..., l, min(no_knn, l))


def stack_variable_length_tensors(
    sequences: Sequence[torch.Tensor],
    constant_value: int | float = 0,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Pad arbitrary tensor dimensions to their maxima, then stack."""
    # Each sequence has the same rank; its axis lengths may differ. Padding uses each axis maximum.

    output_shape = [
        len(sequences),
        *np.max([sequence.shape for sequence in sequences], axis=0).tolist(),
    ]
    output = torch.full(
        output_shape,
        constant_value,
        dtype=sequences[0].dtype if dtype is None else dtype,
        device=sequences[0].device,
    )  # (n_sequences, *axiswise_maximum_shape)
    for destination, source in zip(output, sequences, strict=True):
        destination[tuple(slice(size) for size in source.shape)] = source  # source.shape slice of destination
    return output  # (n_sequences, *axiswise_maximum_shape)


def binpack(
    tensor: torch.Tensor,
    sequence_id: torch.Tensor | None,
    pad_value: int | float,
) -> torch.Tensor:
    """Scatter a sequence-major tensor into the packed layout described by IDs."""
    # tensor: (n_unpacked_sequences, max_length, ...); sequence_id: (b, packed_length) or None.

    if sequence_id is None:
        return tensor  # tensor.shape
    sequence_counts = sequence_id.max(dim=-1).values + 1  # (b,)
    output = torch.full(
        sequence_id.shape + tensor.shape[2:],
        fill_value=pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )  # (b, packed_length, *tensor.shape[2:])
    source_index = 0
    for batch_index, (batch_ids, count) in enumerate(
        zip(sequence_id, sequence_counts, strict=True)
    ):
        for seqid in range(count):
            selection = batch_ids == seqid  # (packed_length,)
            output[batch_index, selection] = tensor[source_index, : selection.sum()]  # (n_selected, *tensor.shape[2:]) selected rows
            source_index += 1
    return output  # (b, packed_length, *tensor.shape[2:])


def unbinpack(
    tensor: torch.Tensor,
    sequence_id: torch.Tensor | None,
    pad_value: int | float,
) -> torch.Tensor:
    """Restore sequence-major rows from a packed tensor and its sequence IDs."""
    # tensor: (b, packed_length, ...); sequence_id: (b, packed_length) or None.

    if sequence_id is None:
        return tensor  # tensor.shape
    rows = []
    sequence_counts = sequence_id.max(dim=-1).values + 1  # (b,)
    for batch_index, (batch_ids, count) in enumerate(
        zip(sequence_id, sequence_counts, strict=True)
    ):
        for seqid in range(count):
            rows.append(tensor[batch_index, batch_ids == seqid])  # (selected_sequence_length, *tensor.shape[2:])
    return stack_variable_length_tensors(rows, pad_value)  # (n_unpacked_sequences, max_sequence_length, *tensor.shape[2:])


def merge_ranges(
    ranges: list[range],
    merge_gap_max: int | None = None,
) -> list[range]:
    """Merge overlapping or sufficiently close ranges in positional order."""

    maximum_gap = 0 if merge_gap_max is None else merge_gap_max
    if not isinstance(maximum_gap, int) or isinstance(maximum_gap, bool):
        raise TypeError("merge_gap_max must be an integer or None.")
    if maximum_gap < 0:
        raise ValueError(f"merge_gap_max must be non-negative, got {maximum_gap}.")
    merged: list[range] = []
    for current in sorted(ranges, key=lambda item: item.start):
        if not merged or merged[-1].stop + maximum_gap < current.start:
            merged.append(current)
            continue
        previous = merged[-1]
        merged[-1] = range(previous.start, max(previous.stop, current.stop))
    return merged


def merge_annotations(
    annotations: list[FunctionAnnotation],
    merge_gap_max: int | None = None,
) -> list[FunctionAnnotation]:
    """Merge overlapping annotations independently for each label."""

    grouped: dict[str, list[range]] = defaultdict(list)
    for annotation in annotations:
        grouped[annotation.label].append(range(annotation.start, annotation.end + 1))
    result = []
    for label, spans in grouped.items():
        result.extend(
            FunctionAnnotation(label=label, start=span.start, end=span.stop - 1)
            for span in merge_ranges(spans, merge_gap_max=merge_gap_max)
        )
    return result


def get_chainbreak_boundaries_from_sequence(
    sequence: Sequence[str],
) -> np.ndarray:
    """Return half-open chain intervals split by chain-break tokens."""

    boundaries = [0]
    final_index = len(sequence) - 1
    for index, residue in enumerate(sequence):
        if residue != CHAIN_BREAK_STR:
            continue
        if index == final_index:
            raise ValueError(
                "Encountered chain break token at end of sequence, this is unexpected."
            )
        if index == final_index - 1:
            warn(
                "Encountered chain break token at penultimate position, this is unexpected.",
                stacklevel=2,
            )
        boundaries.extend((index, index + 1))
    boundaries.append(len(sequence))
    assert len(boundaries) % 2 == 0
    return np.asarray(boundaries).reshape(-1, 2)  # (n_chains, 2)


def deserialize_tensors(data: bytes) -> Any:
    """Decompress a tensor-only Torch payload onto CPU."""

    decompressed = zstandard.ZstdDecompressor().decompress(data)
    return torch.load(
        BytesIO(decompressed),
        map_location="cpu",
        weights_only=True,
    )
