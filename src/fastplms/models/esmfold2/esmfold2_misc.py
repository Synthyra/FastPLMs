"""Small tensor, sequence, and annotation utilities used by ESMFold2.

The helpers in this module are deliberately free of model state. Importing the
module therefore performs no device selection, compilation, or remote access.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Generator, Iterable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import is_dataclass
from io import BytesIO
from typing import Any, Protocol, TypeVar, runtime_checkable
from warnings import warn

import numpy as np
import torch
import zstandard

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

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, list) and all(isinstance(element, torch.Tensor) for element in value):
        return torch.stack(value)
    if convert_none_to_nan:
        value = np.asarray(value, dtype=np.float32)
        value = np.where(value is None, np.nan, value)
    return torch.tensor(value)


def maybe_list(value, convert_nan_to_none: bool = False) -> list | None:
    """Convert an optional tensor or NumPy array to nested Python lists."""

    if value is None:
        return None
    if not convert_nan_to_none:
        return value.tolist()
    if isinstance(value, torch.Tensor):
        nan_mask = torch.isnan(value).cpu().numpy()
        array = value.cpu().numpy().astype(object)
    elif isinstance(value, np.ndarray):
        nan_mask = np.isnan(value)
        array = value.astype(object)
    else:
        raise TypeError("maybe_list can only work with torch.tensor or np.ndarray.")
    array[nan_mask] = None
    return array.tolist()


def replace_inf(data):
    """Replace infinite array values by the ESM API sentinel value."""

    if data is None:
        return None
    array = np.asarray(data, dtype=np.float32)
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

    if isinstance(obj, (np.ndarray, torch.Tensor)) or is_dataclass(obj):
        return obj[idx]  # type: ignore[index,return-value]
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
        )
        return np.concatenate(pieces)
    if isinstance(first, torch.Tensor):
        pieces = (
            objs
            if separator is None
            else list(iterate_with_intermediate(objs, torch.tensor([separator])))
        )
        return torch.cat(pieces)  # type: ignore[arg-type]
    raise TypeError(type(first))


def rbf(values, v_min, v_max, n_bins=16):
    """Encode values against evenly spaced radial basis centers."""

    centers = torch.linspace(
        v_min,
        v_max,
        n_bins,
        dtype=values.dtype,
        device=values.device,
    )
    centers = centers.reshape((1,) * values.ndim + (-1,))
    standardized = (values.unsqueeze(-1) - centers) / ((v_max - v_min) / n_bins)
    return torch.exp(-(standardized**2))


def batched_gather(data, inds, dim=0, no_batch_dims=0):
    """Gather along one data dimension while retaining leading batch axes."""

    batch_indices = []
    index_rank = len(inds.shape)
    for axis, size in enumerate(data.shape[:no_batch_dims]):
        shape = (1,) * axis + (-1,) + (1,) * (index_rank - axis - 1)
        batch_indices.append(torch.arange(size).view(*shape))
    tail = [slice(None)] * (len(data.shape) - no_batch_dims)
    tail[dim - no_batch_dims if dim >= 0 else dim] = inds
    return data[tuple(batch_indices + tail)]


def node_gather(s: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Gather node features for each row of an edge-index tensor."""

    return batched_gather(
        s.unsqueeze(-3),
        edges,
        -2,
        no_batch_dims=len(s.shape) - 1,
    )


def knn_graph(
    coords: torch.Tensor,
    coord_mask: torch.Tensor,
    padding_mask: torch.Tensor,
    sequence_id: torch.Tensor,
    *,
    no_knn: int,
):
    """Build nearest-neighbor edges, using sequence distance for missing geometry."""

    length = coords.shape[-2]
    coords = coords.nan_to_num()
    missing_pair = ~(coord_mask[..., None, :] & coord_mask[..., :, None])
    excluded_pair = padding_mask[..., None, :] | padding_mask[..., :, None]
    if sequence_id is not None:
        excluded_pair |= sequence_id.unsqueeze(1) != sequence_id.unsqueeze(2)

    distances = (coords.unsqueeze(-2) - coords.unsqueeze(-3)).norm(dim=-1)
    residue_index = torch.arange(length, device=coords.device)
    sequence_distance = (residue_index.unsqueeze(-1) - residue_index.unsqueeze(-2)).abs()
    if not (distances[~missing_pair] < MAX_SUPPORTED_DISTANCE).all():
        raise ValueError(
            "Coordinate pairwise distances exceed max supported distance "
            f"({MAX_SUPPORTED_DISTANCE}). "
        )

    rank_distance = sequence_distance.to(distances.dtype).mul(1e2).add(MAX_SUPPORTED_DISTANCE)
    rank_distance = rank_distance.where(missing_pair, distances)
    rank_distance = rank_distance.masked_fill(excluded_pair, torch.inf)
    sorted_distance, sorted_edge = rank_distance.sort(dim=-1, descending=False)
    width = min(no_knn, length)
    return sorted_edge[..., :width], sorted_distance[..., :width].isfinite()


def stack_variable_length_tensors(
    sequences: Sequence[torch.Tensor],
    constant_value: int | float = 0,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Pad arbitrary tensor dimensions to their maxima, then stack."""

    output_shape = [
        len(sequences),
        *np.max([sequence.shape for sequence in sequences], axis=0).tolist(),
    ]
    output = torch.full(
        output_shape,
        constant_value,
        dtype=sequences[0].dtype if dtype is None else dtype,
        device=sequences[0].device,
    )
    for destination, source in zip(output, sequences, strict=True):
        destination[tuple(slice(size) for size in source.shape)] = source
    return output


def binpack(
    tensor: torch.Tensor,
    sequence_id: torch.Tensor | None,
    pad_value: int | float,
):
    """Scatter a sequence-major tensor into the packed layout described by IDs."""

    if sequence_id is None:
        return tensor
    sequence_counts = sequence_id.max(dim=-1).values + 1
    output = torch.full(
        sequence_id.shape + tensor.shape[2:],
        fill_value=pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    source_index = 0
    for batch_index, (batch_ids, count) in enumerate(
        zip(sequence_id, sequence_counts, strict=True)
    ):
        for seqid in range(count):
            selection = batch_ids == seqid
            output[batch_index, selection] = tensor[source_index, : selection.sum()]
            source_index += 1
    return output


def unbinpack(
    tensor: torch.Tensor,
    sequence_id: torch.Tensor | None,
    pad_value: int | float,
):
    """Restore sequence-major rows from a packed tensor and its sequence IDs."""

    if sequence_id is None:
        return tensor
    rows = []
    sequence_counts = sequence_id.max(dim=-1).values + 1
    for batch_index, (batch_ids, count) in enumerate(
        zip(sequence_id, sequence_counts, strict=True)
    ):
        for seqid in range(count):
            rows.append(tensor[batch_index, batch_ids == seqid])
    return stack_variable_length_tensors(rows, pad_value)


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
    return np.asarray(boundaries).reshape(-1, 2)


def deserialize_tensors(data: bytes) -> Any:
    """Decompress a tensor-only Torch payload onto CPU."""

    decompressed = zstandard.ZstdDecompressor().decompress(data)
    return torch.load(
        BytesIO(decompressed),
        map_location="cpu",
        weights_only=True,
    )
