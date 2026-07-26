"""Atom selection, rigid alignment, RMSD, and GDT-TS primitives."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast  # type: ignore

from .esmfold2_affine3d import Affine3D
from .esmfold2_misc import unbinpack
from .esmfold2_normalize_coordinates import index_by_atom_name

ArrayOrTensor = TypeVar("ArrayOrTensor", np.ndarray, Tensor)


def _coordinate_operations(
    coordinates: ArrayOrTensor,
) -> tuple[Callable[[ArrayOrTensor], ArrayOrTensor], Callable[..., ArrayOrTensor]]:
    if isinstance(coordinates, np.ndarray):

        def normalize(X: ArrayOrTensor) -> ArrayOrTensor:
            return X / np.linalg.norm(X, axis=-1, keepdims=True)

        return normalize, np.cross
    return F.normalize, torch.cross  # type: ignore[return-value]


def infer_cbeta_from_atom37(
    atom37: ArrayOrTensor,
    bond_length: float = 1.522,
    bond_angle: float = 1.927,
    dihedral: float = -2.143,
) -> ArrayOrTensor:
    """Infer C-beta coordinates from backbone tensor ``X``.

    The scalar keyword arguments encode the bond length, bond angle, and
    dihedral in radians used by the checkpoint's training geometry.
    """

    n_position = index_by_atom_name(atom37, "N", dim=-2)
    ca_position = index_by_atom_name(atom37, "CA", dim=-2)
    c_position = index_by_atom_name(atom37, "C", dim=-2)
    normalize, cross = _coordinate_operations(atom37)
    with np.errstate(invalid="ignore"):
        n_to_ca = n_position - ca_position
        n_to_c = n_position - c_position
    unit_n_to_ca = normalize(n_to_ca)
    normal = normalize(cross(n_to_c, unit_n_to_ca))
    basis = [unit_n_to_ca, cross(normal, unit_n_to_ca), normal]
    coefficients = [
        bond_length * np.cos(bond_angle),
        bond_length * np.sin(bond_angle) * np.cos(dihedral),
        -bond_length * np.sin(bond_angle) * np.sin(dihedral),
    ]
    offset = sum(
        vector * coefficient for vector, coefficient in zip(basis, coefficients, strict=True)
    )
    return ca_position + offset


def _unpack_alignment_inputs(
    mobile: Tensor,
    target: Tensor,
    atom_mask: Tensor | None,
    sequence_id: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    if sequence_id is None:
        return mobile, target, atom_mask
    unpacked_mobile = unbinpack(mobile, sequence_id, pad_value=torch.nan)
    unpacked_target = unbinpack(target, sequence_id, pad_value=torch.nan)
    if atom_mask is None:
        unpacked_mask = torch.isfinite(unpacked_target).all(dim=-1)
    else:
        unpacked_mask = unbinpack(atom_mask, sequence_id, pad_value=0)
    return unpacked_mobile, unpacked_target, unpacked_mask


def _flatten_atom_axes(
    mobile: Tensor,
    target: Tensor,
    atom_mask: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    b = mobile.shape[0]
    flat_mobile = mobile.view(b, -1, 3) if mobile.dim() == 4 else mobile
    flat_target = target.view(b, -1, 3) if target.dim() == 4 else target
    flat_mask = atom_mask
    if flat_mask is not None and flat_mask.dim() == 3:
        flat_mask = flat_mask.view(b, -1)
    return flat_mobile, flat_target, flat_mask


def _masked_coordinates(
    mobile: Tensor,
    target: Tensor,
    atom_mask: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    if atom_mask is None:
        atom_mask = torch.ones(
            mobile.shape[:2],
            dtype=torch.bool,
            device=mobile.device,
        )
        return mobile, target, atom_mask
    expanded_mask = atom_mask.unsqueeze(-1)
    return (
        mobile.masked_fill(~expanded_mask, 0),
        target.masked_fill(~expanded_mask, 0),
        atom_mask,
    )


@torch.no_grad()
@autocast("cuda", enabled=False)
def compute_alignment_tensors(
    mobile: Tensor,
    target: Tensor,
    atom_exists_mask: Tensor | None = None,
    sequence_id: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Center and align coordinate tensors ``X`` and ``Y``.

    Inputs have shape (b, n, 3), or (b, l, n_atoms, 3). The returned rotation
    tensor ``R`` has shape (b, 3, 3), and atom counts have shape (b, 1).
    """

    mobile, target, atom_exists_mask = _unpack_alignment_inputs(
        mobile,
        target,
        atom_exists_mask,
        sequence_id,
    )
    if mobile.shape != target.shape:
        raise AssertionError("Batch structure shapes do not match!")
    mobile, target, atom_exists_mask = _flatten_atom_axes(
        mobile,
        target,
        atom_exists_mask,
    )
    mobile, target, atom_exists_mask = _masked_coordinates(
        mobile,
        target,
        atom_exists_mask,
    )

    num_valid_atoms = atom_exists_mask.sum(dim=-1, keepdim=True)
    centroid_mobile = mobile.sum(dim=-2, keepdim=True) / num_valid_atoms.unsqueeze(-1)
    centroid_target = target.sum(dim=-2, keepdim=True) / num_valid_atoms.unsqueeze(-1)
    centroid_mobile[num_valid_atoms == 0] = 0
    centroid_target[num_valid_atoms == 0] = 0

    expanded_mask = atom_exists_mask.unsqueeze(-1)
    centered_mobile = (mobile - centroid_mobile).masked_fill(~expanded_mask, 0)
    centered_target = (target - centroid_target).masked_fill(~expanded_mask, 0)
    covariance = torch.matmul(centered_mobile.transpose(1, 2), centered_target)
    left_vectors, _, right_vectors = torch.svd(covariance)
    rotation = torch.matmul(left_vectors, right_vectors.transpose(1, 2))
    return (
        centered_mobile,
        centroid_mobile,
        centered_target,
        centroid_target,
        rotation,
        num_valid_atoms,
    )


def _validate_reduction(reduction: str, allowed: tuple[str, ...]) -> None:
    if reduction not in allowed:
        raise ValueError("Unrecognized reduction: '{reduction}'")


@torch.no_grad()
@autocast("cuda", enabled=False)
def compute_rmsd_no_alignment(
    aligned: Tensor,
    target: Tensor,
    num_valid_atoms: Tensor,
    reduction: str = "batch",
) -> Tensor:
    """Measure RMSD after alignment using a declared reduction."""

    _validate_reduction(reduction, ("per_residue", "per_sample", "batch"))
    difference = aligned - target
    if reduction == "per_residue":
        mean_squared_error = difference.square().view(difference.size(0), -1, 9).mean(-1)
    else:
        mean_squared_error = difference.square().sum(dim=(1, 2)) / num_valid_atoms.squeeze(-1)
    rmsd = torch.sqrt(mean_squared_error)
    if reduction in {"per_residue", "per_sample"}:
        return rmsd
    valid_samples = num_valid_atoms.squeeze(-1) > 0
    return rmsd.masked_fill(~valid_samples, 0).sum() / (valid_samples.sum() + 1e-8)


@torch.no_grad()
@autocast("cuda", enabled=False)
def compute_affine_and_rmsd(
    mobile: Tensor,
    target: Tensor,
    atom_exists_mask: Tensor | None = None,
    sequence_id: Tensor | None = None,
) -> tuple[Affine3D, Tensor]:
    """Fit ``X`` onto ``Y`` and return the rigid transform and batch RMSD."""

    (
        centered_mobile,
        centroid_mobile,
        centered_target,
        centroid_target,
        rotation,
        num_valid_atoms,
    ) = compute_alignment_tensors(mobile, target, atom_exists_mask, sequence_id)
    translation = torch.matmul(-centroid_mobile, rotation) + centroid_target
    affine = Affine3D.from_tensor_pair(
        translation,
        rotation.unsqueeze(dim=-3).transpose(-2, -1),
    )
    rotated_mobile = torch.matmul(centered_mobile, rotation)
    rmsd = compute_rmsd_no_alignment(
        rotated_mobile,
        centered_target,
        num_valid_atoms,
        reduction="batch",
    )
    return affine, rmsd


def compute_gdt_ts_no_alignment(
    aligned: Tensor,
    target: Tensor,
    atom_exists_mask: Tensor,
    reduction: str = "batch",
) -> Tensor:
    """Compute GDT-TS for already aligned coordinate tensors."""

    _validate_reduction(reduction, ("per_sample", "batch"))
    if atom_exists_mask is None:
        atom_exists_mask = torch.isfinite(target).all(dim=-1)
    deviation = torch.linalg.vector_norm(aligned - target, dim=-1)
    counts = atom_exists_mask.sum(dim=-1)
    score_1 = ((deviation < 1) * atom_exists_mask).sum(dim=-1) / counts
    score_2 = ((deviation < 2) * atom_exists_mask).sum(dim=-1) / counts
    score_4 = ((deviation < 4) * atom_exists_mask).sum(dim=-1) / counts
    score_8 = ((deviation < 8) * atom_exists_mask).sum(dim=-1) / counts
    score = (score_1 + score_2 + score_4 + score_8) * 0.25
    return score.mean() if reduction == "batch" else score
