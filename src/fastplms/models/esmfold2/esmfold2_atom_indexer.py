"""Name-based views into an atom-axis property."""

from __future__ import annotations

from operator import attrgetter
from typing import Any

import numpy as np

from .esmfold2_protein_structure import index_by_atom_name


class AtomIndexer:
    """Select named atoms from one property of a structure-like object.

    The wrapper intentionally remains small because ``ProteinChain.atom37`` and
    related public properties expose it directly.
    """

    __slots__ = ("_get_property", "dim", "property", "structure")

    def __init__(self, structure: Any, property: str, dim: int):
        self.structure = structure
        self.property = property
        self.dim = dim
        self._get_property = attrgetter(property)

    def __getitem__(self, atom_names: str | list[str]) -> np.ndarray:
        values = self._get_property(self.structure)
        return index_by_atom_name(values, atom_names, dim=self.dim)
