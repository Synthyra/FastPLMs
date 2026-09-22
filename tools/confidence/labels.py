"""Compute atom lDDT and token PAE targets for confidence-head training."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor
from torch.nn import functional


PLDDT_BINS = 50
PAE_BINS = 64
PAE_MAX_ANGSTROM = 32.0
LDDT_CUTOFF_ANGSTROM = 15.0
LDDT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)


def _check_coordinates(predicted: Tensor, true: Tensor, resolved: Tensor) -> None:
    if predicted.ndim != 2 or predicted.shape[-1] != 3:
        raise ValueError("coordinates must have shape (atoms, 3)")
    if true.shape != predicted.shape:
        raise ValueError("predicted and true coordinates must have equal shape")
    if resolved.shape != predicted.shape[:1]:
        raise ValueError("resolved_mask must have shape (atoms,)")


def _lddt_scores(predicted: Tensor, true: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    """Return per-point lDDT scores and masks from coordinates shaped (points, 3)."""
    finite = torch.isfinite(predicted).all(-1) & torch.isfinite(true).all(-1)
    valid = valid.to(torch.bool) & finite
    true_distances = torch.cdist(true, true)
    predicted_distances = torch.cdist(predicted.nan_to_num(), predicted.nan_to_num())
    pair_mask = valid[:, None] & valid[None, :]
    pair_mask &= ~torch.eye(valid.shape[0], dtype=torch.bool, device=valid.device)
    pair_mask &= true_distances < LDDT_CUTOFF_ANGSTROM
    distance_error = (predicted_distances - true_distances).abs()
    threshold_hits = torch.stack(
        [distance_error < threshold for threshold in LDDT_THRESHOLDS], dim=-1
    ).to(torch.float32)
    pair_count = pair_mask.sum(-1)
    scores = (threshold_hits * pair_mask[..., None]).sum(-2).sum(-1)
    scores = scores / (pair_count.clamp_min(1).to(scores.dtype) * len(LDDT_THRESHOLDS))
    score_mask = pair_count > 0
    return scores, valid & score_mask


def _safe_backbone_coordinates(
    coordinates: Tensor,
    backbone_indices: Tensor,
    resolved_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    atoms = coordinates.shape[0]
    if backbone_indices.ndim != 2 or backbone_indices.shape[-1] != 3:
        raise ValueError("backbone_indices must have shape (tokens, 3)")
    safe_indices = backbone_indices.clamp(0, max(atoms - 1, 0))
    selected = coordinates[safe_indices]
    finite = torch.isfinite(selected).all(-1).all(-1)
    in_range = (backbone_indices >= 0).all(-1) & (backbone_indices < atoms).all(-1)
    resolved = resolved_mask[safe_indices].all(-1)
    return selected, finite & in_range & resolved


def _local_frames(coordinates: Tensor, frame_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Build AtlasFold-compatible source frames from N, CA, C coordinates."""
    # Mirrors AtlasFold confidence.py:get_local_frame_points at pinned commit
    # 444f376d85b9954a5f2f5f3f8b3cbcae1201ebb1.
    n_xyz, ca_xyz, c_xyz = coordinates.unbind(1)
    n_vector = n_xyz - ca_xyz
    c_vector = c_xyz - ca_xyz
    n_norm = n_vector.norm(dim=-1)
    c_norm = c_vector.norm(dim=-1)
    n_unit = n_vector / n_norm.clamp_min(torch.finfo(n_vector.dtype).eps)[..., None]
    c_unit = c_vector / c_norm.clamp_min(torch.finfo(c_vector.dtype).eps)[..., None]
    e1 = n_unit + c_unit
    e1_norm = e1.norm(dim=-1)
    e1 = e1 / e1_norm.clamp_min(torch.finfo(e1.dtype).eps)[..., None]
    e2 = c_unit - n_unit
    e2_norm = e2.norm(dim=-1)
    e2 = e2 / e2_norm.clamp_min(torch.finfo(e2.dtype).eps)[..., None]
    e3 = torch.linalg.cross(e1, e2, dim=-1)
    basis = torch.stack((e1, e2, e3), dim=-2)
    valid = frame_mask & (n_norm > 1e-6) & (c_norm > 1e-6) & (e1_norm > 1e-6) & (e2_norm > 1e-6)
    return ca_xyz, basis, valid


def compute_targets(
    predicted_coords: Tensor,
    true_coords: Tensor,
    resolved_mask: Tensor,
    atom_to_token: Tensor,
    backbone_indices: Tensor,
    token_mask: Tensor,
) -> dict[str, Tensor]:
    """Compute native pLDDT, PAE, and C-alpha lDDT labels.

    ``predicted_coords`` and ``true_coords`` have shape ``(atoms, 3)``.  The
    backbone table has shape ``(tokens, 3)`` in N, C-alpha, C order. PAE ``[i, j]``
    measures target token ``j`` in the local frame of source token ``i``.
    """
    _check_coordinates(predicted_coords, true_coords, resolved_mask)
    if atom_to_token.shape != (predicted_coords.shape[0],):
        raise ValueError("atom_to_token must have shape (atoms,)")
    if token_mask.shape != (backbone_indices.shape[0],):
        raise ValueError("token_mask must have shape (tokens,)")
    if predicted_coords.shape[0] == 0 or backbone_indices.shape[0] == 0:
        raise ValueError("confidence targets require at least one atom and token")
    if atom_to_token.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("atom_to_token must contain integer indices")
    if ((atom_to_token < 0) | (atom_to_token >= backbone_indices.shape[0])).any():
        raise ValueError("atom_to_token contains an out-of-range token index")

    predicted_coords = predicted_coords.detach().float()
    true_coords = true_coords.detach().float()
    resolved_mask = resolved_mask.detach().to(torch.bool)
    atom_to_token = atom_to_token.detach()
    backbone_indices = backbone_indices.detach().long()
    token_mask = token_mask.detach().to(torch.bool)

    atom_token_mask = token_mask[atom_to_token]
    plddt_score, plddt_mask = _lddt_scores(
        predicted_coords, true_coords, resolved_mask & atom_token_mask
    )
    plddt_mask &= atom_token_mask
    plddt_target = (plddt_score.clamp(0.0, 1.0) * PLDDT_BINS).long().clamp_max(PLDDT_BINS - 1)

    predicted_backbone, predicted_frame_mask = _safe_backbone_coordinates(
        predicted_coords, backbone_indices, resolved_mask & torch.isfinite(true_coords).all(-1)
    )
    true_backbone, true_frame_mask = _safe_backbone_coordinates(
        true_coords, backbone_indices, resolved_mask
    )
    predicted_ca = predicted_backbone[:, 1]
    true_ca = true_backbone[:, 1]
    ca_indices = backbone_indices[:, 1]
    ca_in_range = (ca_indices >= 0) & (ca_indices < predicted_coords.shape[0])
    ca_resolved = torch.zeros_like(ca_in_range)
    ca_resolved[ca_in_range] = resolved_mask[ca_indices[ca_in_range]]
    ca_finite = torch.isfinite(predicted_ca).all(-1) & torch.isfinite(true_ca).all(-1)
    ca_mask = token_mask & ca_in_range & ca_resolved & ca_finite
    lddt_ca, lddt_ca_mask = _lddt_scores(predicted_ca, true_ca, ca_mask)

    predicted_origin, predicted_basis, predicted_frame_mask = _local_frames(
        predicted_backbone, predicted_frame_mask
    )
    true_origin, true_basis, true_frame_mask = _local_frames(true_backbone, true_frame_mask)
    predicted_local = torch.einsum(
        "ijk,imk->ijm", predicted_ca[None, :, :] - predicted_origin[:, None, :], predicted_basis
    )
    true_local = torch.einsum(
        "ijk,imk->ijm", true_ca[None, :, :] - true_origin[:, None, :], true_basis
    )
    pae_error = (predicted_local - true_local).norm(dim=-1)
    pae_source_mask = predicted_frame_mask & true_frame_mask
    pae_target_mask = ca_mask
    pae_mask = (
        token_mask[:, None]
        & token_mask[None, :]
        & pae_source_mask[:, None]
        & pae_target_mask[None, :]
    )
    pae_error = pae_error.nan_to_num(posinf=PAE_MAX_ANGSTROM, neginf=0.0)
    pae_target = (pae_error.clamp(0.0, PAE_MAX_ANGSTROM) * PAE_BINS / PAE_MAX_ANGSTROM).long()
    pae_target = pae_target.clamp_max(PAE_BINS - 1)
    return {
        "plddt_target": plddt_target,
        "plddt_score": plddt_score,
        "plddt_mask": plddt_mask,
        "pae_target": pae_target,
        "pae_error": pae_error,
        "pae_mask": pae_mask,
        "lddt_ca": lddt_ca,
        "lddt_ca_mask": lddt_ca_mask,
    }


def _masked_cross_entropy(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    logits = logits.float()
    valid_count = mask.sum()
    if valid_count.item() == 0:
        return logits.sum() * 0.0
    loss = functional.cross_entropy(logits.float()[mask], target[mask], reduction="sum")
    return loss / valid_count.to(loss.dtype)


def confidence_loss(
    outputs: Mapping[str, Tensor], targets: Mapping[str, Tensor]
) -> dict[str, Tensor]:
    """Compute normalized pLDDT plus 0.1-weighted PAE cross entropy."""
    plddt_logits = outputs["plddt_logits"]
    pae_logits = outputs["pae_logits"]
    plddt_logits = (
        plddt_logits[0] if plddt_logits.ndim == 3 and plddt_logits.shape[0] == 1 else plddt_logits
    )
    pae_logits = pae_logits[0] if pae_logits.ndim == 4 and pae_logits.shape[0] == 1 else pae_logits
    plddt_loss = _masked_cross_entropy(plddt_logits, targets["plddt_target"], targets["plddt_mask"])
    pae_loss = _masked_cross_entropy(pae_logits, targets["pae_target"], targets["pae_mask"])
    total = plddt_loss + 0.1 * pae_loss
    return {"total": total, "plddt": plddt_loss, "pae": pae_loss}
