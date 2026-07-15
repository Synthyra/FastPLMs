"""Rigid-frame normalization for atom37 coordinates."""

from __future__ import annotations

from typing import TypeVar

import numpy as np
import torch
from torch import Tensor

from . import esmfold2_residue_constants as residue_constants
from .esmfold2_affine3d import Affine3D

ArrayOrTensor = TypeVar("ArrayOrTensor", np.ndarray, Tensor)


def atom3_to_backbone_frames(bb_positions: Tensor) -> Affine3D:
    """Construct a frame from N, C-alpha, and C positions in ``X``."""

    n_position, ca_position, c_position = bb_positions.unbind(dim=-2)
    return Affine3D.from_graham_schmidt(c_position, ca_position, n_position)


def index_by_atom_name(
    atom37: ArrayOrTensor,
    atom_names: str | list[str],
    dim: int = -2,
) -> ArrayOrTensor:
    """Select one or more named atoms along an atom37 axis."""

    single_atom = isinstance(atom_names, str)
    names = [atom_names] if single_atom else atom_names
    indices = [residue_constants.atom_order[name] for name in names]
    axis = dim % atom37.ndim
    if isinstance(atom37, Tensor):
        index = torch.tensor(indices, dtype=torch.long, device=atom37.device)
        selected = torch.index_select(atom37, axis, index)
    else:
        selected = np.take(atom37, indices, axis=axis)
    return selected.squeeze(axis) if single_atom else selected  # type: ignore[return-value]


def get_protein_normalization_frame(coords: Tensor) -> Affine3D:
    """Build one frame from backbone coordinates ``X`` with shape (l, 37, 3)."""

    backbone = index_by_atom_name(coords, ["N", "CA", "C"], dim=-2)
    residue_is_valid = torch.isfinite(backbone).all(dim=-1).all(dim=-1)
    weights = residue_is_valid[..., None, None]
    coordinate_sum = backbone.masked_fill(~weights, 0).sum(dim=-3)
    count = residue_is_valid.sum(dim=-1)[..., None, None]
    mean_backbone = coordinate_sum / (count + 1e-8)
    return atom3_to_backbone_frames(mean_backbone.float())


def apply_frame_to_coords(coords: Tensor, frame: Affine3D) -> Tensor:
    """Express atom coordinates ``X`` in the inverse of ``frame``."""

    transformed = frame[..., None, None].invert().apply(coords)
    frame_is_valid = frame.trans.norm(dim=-1) > 0
    normalized = torch.where(frame_is_valid[..., None, None, None], transformed, coords)
    return normalized.masked_fill(torch.isinf(coords), torch.inf)


def normalize_coordinates(coords: Tensor) -> Tensor:
    """Normalize ``X`` with shape (..., l, 37, 3) to its backbone frame."""

    return apply_frame_to_coords(coords, get_protein_normalization_frame(coords))
