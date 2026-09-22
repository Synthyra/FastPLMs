"""Rigid alignment for structure dataclasses."""

from __future__ import annotations

import numpy as np
import torch

from dataclasses import Field, replace
from typing import Any, ClassVar, Protocol, TypeVar
from torch import Tensor

from .esmfold2_protein_structure import compute_affine_and_rmsd


class Alignable(Protocol):
    """Minimum structure interface accepted by :class:`Aligner`."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    @property
    def atom37_positions(self) -> np.ndarray: ...

    @property
    def atom37_mask(self) -> np.ndarray: ...

    def __len__(self) -> int: ...


AlignableT = TypeVar("AlignableT", bound=Alignable)


def _coordinate_batch(structure: Alignable) -> Tensor:
    # l is the residue count; the atom37 table has three Cartesian coordinates.
    return torch.as_tensor(structure.atom37_positions, dtype=torch.double).unsqueeze(0)  # (1, l, 37, 3)


def _shared_atom_mask(mobile: Alignable, target: Alignable, backbone_only: bool) -> Tensor:
    shared = np.asarray(mobile.atom37_mask, dtype=bool) & np.asarray(
        target.atom37_mask,
        dtype=bool,
    )  # (l, 37)
    if backbone_only:
        shared = shared.copy()  # (l, 37)
        shared[:, 3:] = False  # (l, 34); retain N, CA, C.
    return torch.from_numpy(shared).unsqueeze(0)  # (1, l, 37)


class Aligner:
    """Fit a mobile structure onto a target with masked Kabsch alignment."""

    def __init__(
        self,
        mobile: Alignable,
        target: Alignable,
        only_use_backbone: bool = False,
        use_reflection: bool = False,
    ) -> None:
        if len(mobile) != len(target):
            raise AssertionError("mobile and target must contain the same residue count")

        mobile_coordinates = _coordinate_batch(mobile)  # (1, l, 37, 3)
        target_coordinates = _coordinate_batch(target)  # (1, l, 37, 3)
        if use_reflection:
            target_coordinates = -target_coordinates  # (1, l, 37, 3)
        atom_mask = _shared_atom_mask(mobile, target, only_use_backbone)  # (1, l, 37)
        self._affine3D, rmsd = compute_affine_and_rmsd(
            mobile_coordinates,
            target_coordinates,
            atom_exists_mask=atom_mask,
        )  # affine shape: (1, 1); rmsd: ()
        self._rmsd = rmsd.item()

    @property
    def rmsd(self) -> float:
        return self._rmsd

    def apply(self, mobile: AlignableT) -> AlignableT:
        """Return a dataclass copy with all present atom coordinates aligned."""

        present = np.asarray(mobile.atom37_mask, dtype=bool)  # (l, 37)
        packed = torch.as_tensor(
            mobile.atom37_positions[present],
            dtype=torch.float32,
        ).unsqueeze(0)  # (1, n, 3), n is the number of present atoms.
        aligned = self._affine3D.apply(packed).squeeze(0).cpu().numpy()  # (n, 3)
        atom37_positions = np.full_like(mobile.atom37_positions, np.nan)  # (l, 37, 3)
        atom37_positions[present] = aligned  # (n, 3)
        return replace(mobile, atom37_positions=atom37_positions)
