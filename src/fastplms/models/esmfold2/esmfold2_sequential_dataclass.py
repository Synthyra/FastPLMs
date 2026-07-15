"""Dataclass support for aligned residue-level fields."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import Field, dataclass, fields, replace
from typing import Any, Self

import numpy as np

from .esmfold2_misc import concat_objects, slice_any_object

Index = int | list[int] | slice | np.ndarray


def _is_sequential(field: Field[Any]) -> bool:
    return bool(field.metadata.get("sequence", False))


def _sequence_axis(field: Field[Any]) -> int:
    axis = int(field.metadata.get("sequence_dim", 0))
    if axis not in (0, 1):
        raise NotImplementedError("SequentialDataclass supports sequence_dim values zero and one.")
    return axis


def _slice_value(value: Any, index: Index, axis: int) -> Any:
    if axis == 0:
        return slice_any_object(value, index)
    sliced = [slice_any_object(track, index) for track in value]
    return value.__class__(sliced)


def _iter_sequence_lengths(value: Any, axis: int) -> Iterable[int]:
    if axis == 0:
        yield len(value)
    else:
        yield from (len(track) for track in value)


@dataclass(frozen=True)
class SequentialDataclass(ABC):
    """Keep dataclass fields aligned along a shared residue dimension.

    A subclass marks aligned fields with ``metadata={"sequence": True}``.
    ``sequence_dim`` may be zero for a direct sequence or one for a collection
    of aligned tracks. ``join_token`` is passed to the package concatenation
    helper when instances are joined.
    """

    def __post_init__(self) -> None:
        expected = len(self)
        for field in fields(self):
            if not _is_sequential(field) or field.name == "complex":
                continue
            value = getattr(self, field.name)
            if value is None:
                continue
            for actual in _iter_sequence_lengths(value, _sequence_axis(field)):
                if actual != expected:
                    raise ValueError(
                        f"Mismatch in sequence length for field: {field.name}. "
                        f"Expected {expected}, received {actual}"
                    )

    @abstractmethod
    def __len__(self) -> int:
        """Return the shared sequence length."""

        raise NotImplementedError

    def __getitem__(self, index: Index) -> Self:
        """Apply one sequence index to every aligned field."""

        normalized_index: Index = [index] if isinstance(index, int) else index
        updates: dict[str, Any] = {}
        for field in fields(self):
            if not _is_sequential(field):
                continue
            value = getattr(self, field.name)
            if value is not None:
                updates[field.name] = _slice_value(value, normalized_index, _sequence_axis(field))
        return replace(self, **updates)

    @classmethod
    def concat(cls, items: list[Self], **overrides: Any) -> Self:
        """Join aligned fields and retain non-sequential values from the first item."""

        if not items:
            raise ValueError("SequentialDataclass.concat requires at least one item.")

        updates: dict[str, Any] = {}
        for field in fields(cls):
            if not _is_sequential(field):
                continue
            first_value = getattr(items[0], field.name)
            if first_value is None:
                continue
            values = [getattr(item, field.name) for item in items]
            join_token = field.metadata.get("join_token")
            if _sequence_axis(field) == 0:
                updates[field.name] = concat_objects(values, join_token)
            else:
                tracks = [concat_objects(track, join_token) for track in zip(*values, strict=True)]
                updates[field.name] = first_value.__class__(tracks)

        updates.update(overrides)
        return replace(items[0], **updates)


__all__ = ["SequentialDataclass"]
