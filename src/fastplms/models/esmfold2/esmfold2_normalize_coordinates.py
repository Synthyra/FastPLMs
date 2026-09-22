"""Rigid-frame normalization for atom37 coordinates."""

from __future__ import annotations

import numpy as np
import torch

from typing import TypeVar
from torch import Tensor

from . import esmfold2_residue_constants as residue_constants
from .esmfold2_affine3d import Affine3D


ArrayOrTensor = TypeVar("ArrayOrTensor", np.ndarray, Tensor)


def atom3_to_backbone_frames(bb_positions: Tensor) -> Affine3D:
    """Construct a frame from N, C-alpha, and C positions in ``X``."""

    # bb_positions: (..., 3, 3), ordered N, CA, C on the penultimate axis.
    n_position, ca_position, c_position = bb_positions.unbind(dim=-2)  # each (..., 3)
    return Affine3D.from_graham_schmidt(c_position, ca_position, n_position)  # affine shape: (...,)


def index_by_atom_name(
    atom37: ArrayOrTensor,
    atom_names: str | list[str],
    dim: int = -2,
) -> ArrayOrTensor:
    """Select named atoms, replacing axis ``dim`` by their count (or removing it)."""

    single_atom = isinstance(atom_names, str)
    names = [atom_names] if single_atom else atom_names
    indices = [residue_constants.atom_order[name] for name in names]
    axis = dim % atom37.ndim
    if isinstance(atom37, Tensor):
        index = torch.tensor(indices, dtype=torch.long, device=atom37.device)  # (n_names,)
        selected = torch.index_select(atom37, axis, index)  # atom37.shape with axis = n_names
    else:
        selected = np.take(atom37, indices, axis=axis)  # atom37.shape with axis = n_names
    return selected.squeeze(axis) if single_atom else selected  # type: ignore[return-value]


def get_protein_normalization_frame(coords: Tensor) -> Affine3D:
    """Build one frame per batch item from coordinates of shape (..., l, 37, 3)."""

    backbone = index_by_atom_name(coords, ["N", "CA", "C"], dim=-2)  # (..., l, 3, 3)
    residue_is_valid = torch.isfinite(backbone).all(dim=-1).all(dim=-1)  # (..., l)
    weights = residue_is_valid[..., None, None]  # (..., l, 1, 1)
    coordinate_sum = backbone.masked_fill(~weights, 0).sum(dim=-3)  # (..., 3, 3)
    count = residue_is_valid.sum(dim=-1)[..., None, None]  # (..., 1, 1)
    mean_backbone = coordinate_sum / (count + 1e-8)  # (..., 3, 3)
    return atom3_to_backbone_frames(mean_backbone.float())  # affine shape: (...,)


def apply_frame_to_coords(coords: Tensor, frame: Affine3D) -> Tensor:
    """Express atom coordinates ``X`` in the inverse of ``frame``."""

    # coords: (..., l, 37, 3); frame shape: (...,).
    transformed = frame[..., None, None].invert().apply(coords)  # (..., l, 37, 3)
    frame_is_valid = frame.trans.norm(dim=-1) > 0  # (...,)
    normalized = torch.where(frame_is_valid[..., None, None, None], transformed, coords)  # (..., l, 37, 3)
    return normalized.masked_fill(torch.isinf(coords), torch.inf)  # (..., l, 37, 3)


def normalize_coordinates(coords: Tensor) -> Tensor:
    """Normalize ``X`` with shape (..., l, 37, 3) to its backbone frame."""

    return apply_frame_to_coords(coords, get_protein_normalization_frame(coords))  # (..., l, 37, 3)
