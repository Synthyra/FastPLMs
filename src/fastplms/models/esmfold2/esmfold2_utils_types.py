"""Small public types shared by the ESMFold2 structure utilities.

These definitions are intentionally independent of cloud-storage packages. Any
path object implementing :class:`os.PathLike` is accepted by the runtime file
helpers, including cloud-path implementations installed by an application.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import TypeAlias

PathLike: TypeAlias = str | os.PathLike[str]
PathOrBuffer: TypeAlias = PathLike | io.TextIOBase


@dataclass(slots=True)
class FunctionAnnotation:
    """A residue-range annotation using one-based inclusive coordinates."""

    label: str
    start: int
    end: int

    def to_tuple(self) -> tuple[str, int, int]:
        """Return the serialization order used by annotation tokenizers."""

        return (self.label, self.start, self.end)

    def __len__(self) -> int:
        """Return the number of annotated residues."""

        return self.end - self.start + 1


__all__ = ["FunctionAnnotation", "PathLike", "PathOrBuffer"]
