"""Geometry objectives shared by Boltz diffusion and steering code.

The rigid-alignment mechanism is based on the Kabsch formulation used by
AlphaFold 3 implementations.  The implementation is maintained locally and
does not import an upstream runtime package.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as functional
from einops import einsum


def _weighted_centroid(
    coordinates: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    return (coordinates * weights).sum(dim=-2, keepdim=True) / weights.sum(
        dim=-2,
        keepdim=True,
    )


def _warn_if_alignment_is_ambiguous(
    mask: torch.Tensor,
    singular_values: torch.Tensor,
    *,
    num_points: int,
    coordinate_dim: int,
) -> None:
    if torch.any(mask.sum(dim=-1) < coordinate_dim + 1):
        warnings.warn(
            "The size of one of the point clouds is <= dim+1. "
            "`WeightedRigidAlign` cannot return a unique rotation.",
            RuntimeWarning,
            stacklevel=3,
        )
    if (singular_values.abs() <= 1e-15).any() and num_points >= coordinate_dim + 1:
        warnings.warn(
            "Excessively low rank of cross-correlation between aligned "
            "point clouds. `WeightedRigidAlign` cannot return a unique rotation.",
            RuntimeWarning,
            stacklevel=3,
        )


def weighted_rigid_align(
    true_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Align true coordinates to predicted coordinates with weighted Kabsch.

    ``true_coords`` and ``pred_coords`` have shape ``(..., n, 3)``.  The
    returned tensor is detached because alignment defines a fixed target for
    the diffusion loss.
    """

    output_shape = torch.broadcast_shapes(true_coords.shape, pred_coords.shape)
    *batch_shape, num_points, coordinate_dim = output_shape
    point_weights = (mask * weights).unsqueeze(-1)

    true_centroid = _weighted_centroid(true_coords, point_weights)
    pred_centroid = _weighted_centroid(pred_coords, point_weights)
    true_centered = true_coords - true_centroid
    pred_centered = pred_coords - pred_centroid

    covariance = einsum(
        point_weights * pred_centered,
        true_centered,
        "... n i, ... n j -> ... i j",
    )
    original_dtype = covariance.dtype
    covariance_fp32 = covariance.to(torch.float32)
    left_vectors, singular_values, right_vectors_h = torch.linalg.svd(
        covariance_fp32,
        driver="gesvd" if covariance_fp32.is_cuda else None,
    )
    right_vectors = right_vectors_h.mH
    _warn_if_alignment_is_ambiguous(
        mask,
        singular_values,
        num_points=num_points,
        coordinate_dim=coordinate_dim,
    )

    preliminary_rotation = torch.einsum(
        "... i j, ... k j -> ... i k",
        left_vectors,
        right_vectors,
    ).to(torch.float32)
    orientation = torch.eye(
        coordinate_dim,
        dtype=covariance_fp32.dtype,
        device=covariance.device,
    )[None].repeat(*batch_shape, 1, 1)
    orientation[..., -1, -1] = torch.det(preliminary_rotation)
    rotation = einsum(
        left_vectors,
        orientation,
        right_vectors,
        "... i j, ... j k, ... l k -> ... i l",
    ).to(original_dtype)

    aligned = einsum(true_centered, rotation, "... n i, ... j i -> ... n j") + pred_centroid
    aligned.detach_()
    return aligned


def _smooth_lddt_for_example(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    is_nucleotide: torch.Tensor,
    coords_mask: torch.Tensor,
    *,
    nucleic_acid_cutoff: float,
    other_cutoff: float,
) -> torch.Tensor:
    true_distances = torch.cdist(true_coords, true_coords)
    nucleotide_rows = is_nucleotide.unsqueeze(-1).expand(-1, is_nucleotide.shape[-1])
    pair_mask = nucleotide_rows * (true_distances < nucleic_acid_cutoff).float()
    pair_mask += (1 - nucleotide_rows) * (true_distances < other_cutoff).float()
    pair_mask *= 1 - torch.eye(pred_coords.shape[0], device=pred_coords.device)
    pair_mask *= coords_mask.unsqueeze(-1)
    pair_mask *= coords_mask.unsqueeze(-2)

    pair_indices = pair_mask.nonzero()
    true_pair_distances = true_distances[pair_indices[:, 0], pair_indices[:, 1]]
    pred_pair_distances = functional.pairwise_distance(
        pred_coords[pair_indices[:, 0]],
        pred_coords[pair_indices[:, 1]],
    )
    distance_error = torch.abs(true_pair_distances - pred_pair_distances)
    smooth_agreement = (
        sum(torch.sigmoid(threshold - distance_error) for threshold in (0.5, 1.0, 2.0, 4.0)) / 4.0
    )
    return smooth_agreement.sum() / (pair_indices.shape[0] + 1e-5)


def smooth_lddt_loss(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    is_nucleotide: torch.Tensor,
    coords_mask: torch.Tensor,
    nucleic_acid_cutoff: float = 30.0,
    other_cutoff: float = 15.0,
    multiplicity: int = 1,
) -> torch.Tensor:
    """Return one minus the smooth local-distance agreement.

    Coordinate tensors have shape ``(b, n, 3)``.  Sequence-level masks may
    be shared across repeated diffusion samples through ``multiplicity``.
    """

    agreements = [
        _smooth_lddt_for_example(
            pred_coords[index],
            true_coords[index],
            is_nucleotide[index // multiplicity],
            coords_mask[index // multiplicity],
            nucleic_acid_cutoff=nucleic_acid_cutoff,
            other_cutoff=other_cutoff,
        )
        for index in range(true_coords.shape[0])
    ]
    return 1.0 - torch.stack(agreements).mean(dim=0)
