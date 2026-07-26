"""Contact, lDDT, RMSD, and GDT-TS metrics for structure validation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast  # type: ignore

from . import esmfold2_residue_constants as residue_constants
from .esmfold2_misc import binpack, unbinpack
from .esmfold2_protein_structure import (
    compute_alignment_tensors,
    compute_gdt_ts_no_alignment,
    compute_rmsd_no_alignment,
)


def _distance_matrix(positions: Tensor, eps: float) -> Tensor:
    displacement = positions[..., None, :] - positions[..., None, :, :]
    return torch.sqrt(eps + torch.sum(displacement**2, dim=-1))


def compute_lddt_from_dmat(
    dmat_pred: Tensor,
    dmat_true: Tensor,
    pairwise_mask: Tensor,
    cutoff: float | Tensor = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> Tensor:
    """Score distance matrices ``D_pred`` and ``D_true`` with shape (..., l, l)."""

    sequence_length = dmat_true.size(-1)
    identity = torch.eye(sequence_length, device=dmat_true.device)
    scored_pairs = (dmat_true < cutoff) * pairwise_mask * (1.0 - identity)
    absolute_error = torch.abs(dmat_true - dmat_pred)
    score = (
        (absolute_error < 0.5).type(absolute_error.dtype)
        + (absolute_error < 1.0).type(absolute_error.dtype)
        + (absolute_error < 2.0).type(absolute_error.dtype)
        + (absolute_error < 4.0).type(absolute_error.dtype)
    ) * 0.25
    dimensions = (-1,) if per_residue else (-2, -1)
    normalization = 1.0 / (eps + scored_pairs.sum(dim=dimensions))
    return normalization * (eps + (scored_pairs * score).sum(dim=dimensions))


def compute_lddt(
    all_atom_pred_pos: Tensor,
    all_atom_positions: Tensor,
    all_atom_mask: Tensor,
    pairwise_all_atom_mask: Tensor | None = None,
    cutoff: float | Tensor = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
    sequence_id: Tensor | None = None,
) -> Tensor:
    """Compute lDDT from coordinate tensors and atom masks."""

    expanded_mask = all_atom_mask[..., None]
    true_distances = _distance_matrix(all_atom_positions, eps)
    predicted_distances = _distance_matrix(all_atom_pred_pos, eps)
    pair_mask = expanded_mask * expanded_mask.transpose(-2, -1)
    if pairwise_all_atom_mask is not None:
        pair_mask = pair_mask * pairwise_all_atom_mask
    if sequence_id is not None:
        same_sequence = sequence_id[..., None] == sequence_id[..., None, :]
        pair_mask = pair_mask * same_sequence.type_as(pair_mask)
    return compute_lddt_from_dmat(
        predicted_distances,
        true_distances,
        pair_mask,
        cutoff=cutoff,
        eps=eps,
        per_residue=per_residue,
    )


def compute_lddt_ca(
    all_atom_pred_pos: Tensor,
    all_atom_positions: Tensor,
    all_atom_mask: Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
    sequence_id: Tensor | None = None,
) -> Tensor:
    """Compute lDDT using only C-alpha coordinates."""

    ca_index = residue_constants.atom_order["CA"]
    predicted_ca = (
        all_atom_pred_pos if all_atom_pred_pos.dim() == 3 else all_atom_pred_pos[..., ca_index, :]
    )
    return compute_lddt(
        predicted_ca,
        all_atom_positions[..., ca_index, :],
        all_atom_mask[..., ca_index],
        cutoff=cutoff,
        eps=eps,
        per_residue=per_residue,
        sequence_id=sequence_id,
    )


@torch.no_grad()
@autocast("cuda", enabled=False)
def compute_rmsd(
    mobile: Tensor,
    target: Tensor,
    atom_exists_mask: Tensor | None = None,
    sequence_id: Tensor | None = None,
    reduction: str = "batch",
) -> Tensor:
    """Align ``X`` to ``Y`` and compute RMSD."""

    centered_mobile, _, centered_target, _, rotation, counts = compute_alignment_tensors(
        mobile,
        target,
        atom_exists_mask,
        sequence_id,
    )
    rmsd = compute_rmsd_no_alignment(
        torch.matmul(centered_mobile, rotation),
        centered_target,
        counts,
        reduction=reduction,
    )
    if reduction == "per_residue" and sequence_id is not None:
        return binpack(rmsd, sequence_id, pad_value=0)
    return rmsd


def compute_gdt_ts(
    mobile: Tensor,
    target: Tensor,
    atom_exists_mask: Tensor | None = None,
    sequence_id: Tensor | None = None,
    reduction: str = "per_sample",
) -> Tensor:
    """Align ``X`` to ``Y`` and compute GDT-TS."""

    if atom_exists_mask is None:
        atom_exists_mask = torch.isfinite(target).all(dim=-1)
    centered_mobile, _, centered_target, _, rotation, _ = compute_alignment_tensors(
        mobile,
        target,
        atom_exists_mask,
        sequence_id,
    )
    if sequence_id is not None:
        atom_exists_mask = unbinpack(atom_exists_mask, sequence_id, pad_value=False)
    return compute_gdt_ts_no_alignment(
        torch.matmul(centered_mobile, rotation),
        centered_target,
        atom_exists_mask,
        reduction,
    )


def _batched_contacts(predictions: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)
    if targets.dim() == 2:
        targets = targets.unsqueeze(0)
    if predictions.size() != targets.size():
        raise ValueError(
            f"Size mismatch. Received predictions of size {predictions.size()}, "
            f"targets of size {targets.size()}"
        )
    return predictions, targets


def _valid_contact_mask(
    targets: Tensor,
    src_lengths: Tensor,
    minsep: int,
    maxsep: int | None,
) -> Tensor:
    sequence_length = targets.shape[-1]
    positions = torch.arange(sequence_length, device=targets.device)
    separation = (positions.unsqueeze(0) - positions.unsqueeze(1)).unsqueeze(0)
    valid = (separation >= minsep) & (targets >= 0)
    if maxsep is not None:
        valid &= separation < maxsep
    within_length = positions.unsqueeze(0) < src_lengths.unsqueeze(1)
    return valid & within_length.unsqueeze(1) & within_length.unsqueeze(2)


def contact_precision(
    predictions: Tensor,
    targets: Tensor,
    src_lengths: Tensor | None = None,
    minsep: int = 6,
    maxsep: int | None = None,
    override_length: int | None = None,
) -> dict[str, Tensor]:
    """Compute P@L, P@L/5, and binned area for contact probabilities."""

    predictions, targets = _batched_contacts(predictions, targets)
    batch_size, sequence_length, _ = predictions.shape
    if src_lengths is None:
        src_lengths = torch.full(
            (batch_size,),
            sequence_length,
            dtype=torch.long,
            device=predictions.device,
        )
    valid = _valid_contact_mask(targets, src_lengths, minsep, maxsep)
    masked_predictions = predictions.masked_fill(~valid, float("-inf"))
    row_index, column_index = np.triu_indices(sequence_length, minsep)
    upper_predictions = masked_predictions[:, row_index, column_index]
    upper_targets = targets[:, row_index, column_index]

    topk = sequence_length if override_length is None else max(sequence_length, override_length)
    ranked_indices = upper_predictions.argsort(dim=-1, descending=True)[:, :topk]
    batch_indices = torch.arange(batch_size, device=ranked_indices.device).unsqueeze(1)
    ranked_targets = upper_targets[batch_indices, ranked_indices]
    if ranked_targets.size(1) < topk:
        ranked_targets = F.pad(ranked_targets, [0, topk - ranked_targets.size(1)])
    cumulative_contacts = ranked_targets.type_as(predictions).cumsum(dim=-1)

    gather_lengths = src_lengths.unsqueeze(1)
    if override_length is not None:
        gather_lengths = override_length * torch.ones_like(gather_lengths)
    fractions = torch.arange(0.1, 1.1, 0.1, device=predictions.device).unsqueeze(0)
    gather_indices = (fractions * gather_lengths).type(torch.long).sub(1).clamp_min(0)
    cumulative_bins = cumulative_contacts.gather(1, gather_indices)
    precisions = cumulative_bins / (gather_indices + 1).type_as(cumulative_bins)
    return {
        "AUC": precisions.mean(dim=-1),
        "P@L": precisions[:, 9],
        "P@L5": precisions[:, 1],
    }
