"""Tensor-tree and memory-bounded execution helpers for pair attention.

These utilities implement the small subset of OpenFold-style chunking needed
by the local Boltz pair stack.  The implementation is independent, but keeps
the public function names expected by converted checkpoints.
"""

from __future__ import annotations

import torch
from collections.abc import Callable, Sequence
from functools import partial
from math import prod
from typing import Any


def add(left: torch.Tensor, right: torch.Tensor, inplace: bool) -> torch.Tensor:
    """Add ``right`` to ``left``, optionally reusing ``left`` storage."""

    # left/right: broadcast-compatible shapes; result has their broadcast shape.
    if inplace:
        left += right  # same shape as left
        return left  # same shape as left
    return left + right  # broadcast shape


def permute_final_dims(tensor: torch.Tensor, inds: Sequence[int]) -> torch.Tensor:
    """Permute only the final ``len(inds)`` axes of ``tensor``."""

    final_count = len(inds)
    leading = list(range(tensor.ndim - final_count))
    final = [tensor.ndim - final_count + index for index in inds]
    return tensor.permute(leading + final)  # (..., d_0, ..., d_n)


def is_fp16_enabled() -> bool:
    """Return whether CUDA autocast currently targets ``float16``."""

    return torch.is_autocast_enabled() and torch.get_autocast_dtype("cuda") == torch.float16


def dict_map(
    fn: Callable[[Any], Any],
    dic: dict[Any, Any],
    leaf_type: type | tuple[type, ...],
) -> dict[Any, Any]:
    """Apply ``fn`` to leaves in a nested dictionary tree."""

    return {key: tree_map(fn, value, leaf_type) for key, value in dic.items()}


def tree_map(
    fn: Callable[[Any], Any],
    tree: Any,
    leaf_type: type | tuple[type, ...],
) -> Any:
    """Map a function over dict, list, and tuple containers."""

    if isinstance(tree, leaf_type):
        return fn(tree)
    if isinstance(tree, dict):
        return dict_map(fn, tree, leaf_type)
    if isinstance(tree, list):
        return [tree_map(fn, item, leaf_type) for item in tree]
    if isinstance(tree, tuple):
        return tuple(tree_map(fn, item, leaf_type) for item in tree)
    raise ValueError(f"tree type {type(tree)!r} is not supported")


tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)


def flatten_final_dims(tensor: torch.Tensor, no_dims: int) -> torch.Tensor:
    """Collapse the final ``no_dims`` axes into one axis."""

    # tensor: (..., d_0, ..., d_n)
    return tensor.reshape(*tensor.shape[:-no_dims], -1)  # (..., prod(final dims))


def _fetch_dims(tree: Any) -> list[torch.Size]:
    """Collect tensor shapes from a supported tree."""

    if isinstance(tree, torch.Tensor):
        return [tree.shape]
    if isinstance(tree, dict):
        children = tree.values()
    elif isinstance(tree, (list, tuple)):
        children = tree
    else:
        raise ValueError(f"tree type {type(tree)!r} is not supported")

    shapes: list[torch.Size] = []
    for child in children:
        shapes.extend(_fetch_dims(child))
    return shapes


@torch.jit.ignore
def _flat_idx_to_idx(flat_idx: int, dims: tuple[int, ...]) -> tuple[int, ...]:
    """Convert a row-major flat index into an index tuple."""

    coordinates = [0] * len(dims)
    remainder = flat_idx
    for axis in range(len(dims) - 1, -1, -1):
        remainder, coordinates[axis] = divmod(remainder, dims[axis])
    return tuple(coordinates)


def _ravel_index(index: Sequence[int], dims: Sequence[int]) -> int:
    """Convert a row-major index tuple to a flat index."""

    flat = 0
    for coordinate, size in zip(index, dims, strict=True):
        flat = flat * size + coordinate
    return flat


def _cover_flat_interval(
    start: int,
    stop: int,
    dims: tuple[int, ...],
) -> list[tuple[slice, ...]]:
    """Cover ``[start, stop)`` with ordered contiguous tensor slices."""

    if not dims:
        return [tuple()]
    if len(dims) == 1:
        return [(slice(start, stop),)]

    child_size = prod(dims[1:])
    first_child = start // child_size
    last_child = (stop - 1) // child_size
    if first_child == last_child:
        tail = _cover_flat_interval(
            start % child_size,
            (stop - 1) % child_size + 1,
            dims[1:],
        )
        prefix = slice(first_child, first_child + 1)
        return [(prefix, *item) for item in tail]

    slices: list[tuple[slice, ...]] = []
    start_offset = start % child_size
    first_full_child = first_child
    if start_offset:
        prefix = slice(first_child, first_child + 1)
        slices.extend(
            (prefix, *item) for item in _cover_flat_interval(start_offset, child_size, dims[1:])
        )
        first_full_child += 1

    stop_offset = stop % child_size
    full_stop = last_child if stop_offset else last_child + 1
    if first_full_child < full_stop:
        slices.append((slice(first_full_child, full_stop),))

    if stop_offset:
        prefix = slice(last_child, last_child + 1)
        slices.extend((prefix, *item) for item in _cover_flat_interval(0, stop_offset, dims[1:]))
    return slices


@torch.jit.ignore
def _get_minimal_slice_set(
    start: Sequence[int],
    end: Sequence[int],
    dims: Sequence[int],
    start_edges: Sequence[bool] | None = None,
    end_edges: Sequence[bool] | None = None,
) -> list[tuple[slice, ...]]:
    """Return ordered slices covering the inclusive row-major interval.

    ``start_edges`` and ``end_edges`` remain accepted for compatibility.  The
    interval decomposition derives the same information directly.
    """

    del start_edges, end_edges
    shape = tuple(dims)
    if not shape:
        return [tuple()]
    flat_start = _ravel_index(start, shape)
    flat_stop = _ravel_index(end, shape) + 1
    return _cover_flat_interval(flat_start, flat_stop, shape)


@torch.jit.ignore
def _chunk_slice(
    tensor: torch.Tensor,
    flat_start: int,
    flat_end: int,
    no_batch_dims: int,
) -> torch.Tensor:
    """Slice a flattened batch interval without flattening the full tensor."""

    batch_shape = tuple(tensor.shape[:no_batch_dims])
    start = _flat_idx_to_idx(flat_start, batch_shape)
    end = _flat_idx_to_idx(flat_end - 1, batch_shape)
    pieces = [
        tensor[item] for item in _get_minimal_slice_set(start, end, batch_shape)
    ]  # each: (*covered_batch_shape, *feature_shape)
    feature_shape = tuple(tensor.shape[no_batch_dims:])
    return torch.cat(
        [piece.reshape(-1, *feature_shape) for piece in pieces]
    )  # (flat_end - flat_start, *feature_shape)


def _prepare_input(
    tensor: torch.Tensor,
    batch_shape: tuple[int, ...],
    no_batch_dims: int,
    *,
    low_mem: bool,
) -> torch.Tensor:
    # tensor: (*input_batch_shape, *feature_shape)
    feature_shape = tuple(tensor.shape[no_batch_dims:])
    if low_mem:
        return tensor.expand(batch_shape + feature_shape)  # (*batch_shape, *feature_shape)
    if any(size != 1 for size in tensor.shape[:no_batch_dims]):
        tensor = tensor.expand(
            batch_shape + feature_shape
        )  # (*batch_shape, *feature_shape)
    return tensor.reshape(-1, *feature_shape)  # (flat_batch, *feature_shape)


def _write_chunk(
    destination: Any,
    source: Any,
    start: int,
    stop: int,
    *,
    add_into_out: bool,
) -> None:
    if isinstance(source, dict):
        for key, value in source.items():
            _write_chunk(
                destination[key],
                value,
                start,
                stop,
                add_into_out=add_into_out,
            )
        return
    if isinstance(source, (tuple, list)):
        for destination_item, source_item in zip(destination, source, strict=True):
            _write_chunk(
                destination_item,
                source_item,
                start,
                stop,
                add_into_out=add_into_out,
            )
        return
    if not isinstance(source, torch.Tensor):
        raise ValueError(f"output type {type(source)!r} is not supported")
    if add_into_out:
        destination[start:stop] += source  # (stop - start, *feature_shape)
    else:
        destination[start:stop] = source  # (stop - start, *feature_shape)


def chunk_layer(
    layer: Callable[..., Any],
    inputs: dict[str, Any],
    chunk_size: int,
    no_batch_dims: int,
    low_mem: bool = False,
    _out: Any = None,
    _add_into_out: bool = False,
) -> Any:
    """Apply ``layer`` to flattened batch chunks and reassemble its output."""

    if not inputs:
        raise ValueError("at least one input is required")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    leading_shapes = [shape[:no_batch_dims] for shape in _fetch_dims(inputs)]
    batch_shape = tuple(max(sizes) for sizes in zip(*leading_shapes, strict=True))
    prepare = partial(
        _prepare_input,
        batch_shape=batch_shape,
        no_batch_dims=no_batch_dims,
        low_mem=low_mem,
    )
    prepared_inputs = tensor_tree_map(
        prepare, inputs
    )  # each tensor: (flat_batch, *feature_shape), or broadcast batch in low-memory mode

    output = None
    if _out is not None:
        output = tensor_tree_map(
            lambda tensor: tensor.reshape(-1, *tensor.shape[no_batch_dims:]),
            _out,
        )  # each tensor: (flat_batch, *feature_shape)

    flat_batch_size = prod(batch_shape)
    for start in range(0, flat_batch_size, chunk_size):
        stop = min(flat_batch_size, start + chunk_size)
        if low_mem:
            select = partial(
                _chunk_slice,
                flat_start=start,
                flat_end=stop,
                no_batch_dims=len(batch_shape),
            )
        else:

            def select(
                tensor: torch.Tensor,
                start: int = start,
                stop: int = stop,
            ) -> torch.Tensor:
                return tensor if tensor.shape[0] == 1 else tensor[start:stop]

        input_chunk = tensor_tree_map(
            select, prepared_inputs
        )  # each tensor: (chunk, *feature_shape)
        output_chunk = layer(
            **input_chunk
        )  # each output tensor: (chunk, *output_feature_shape)

        if output is None:
            output = tensor_tree_map(
                lambda tensor: tensor.new_zeros((flat_batch_size, *tensor.shape[1:])),
                output_chunk,
            )  # each tensor: (flat_batch, *output_feature_shape)
        _write_chunk(
            output,
            output_chunk,
            start,
            stop,
            add_into_out=_add_into_out,
        )

    return tensor_tree_map(
        lambda tensor: tensor.reshape(batch_shape + tuple(tensor.shape[1:])),
        output,
    )  # each tensor: (*batch_shape, *output_feature_shape)
    # tensor: (..., d_0, ..., d_n); only the named final dimensions are reordered.
    # tensor: (*batch_shape, *feature_shape)
