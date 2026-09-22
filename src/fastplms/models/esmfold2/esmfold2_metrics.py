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
    # positions: (..., n, 3); n is the number of points in each distance matrix.
    displacement = positions[..., None, :] - positions[..., None, :, :]  # (..., n, n, 3)
    return torch.sqrt(eps + torch.sum(displacement**2, dim=-1))  # (..., n, n)


def compute_lddt_from_dmat(
    dmat_pred: Tensor,
    dmat_true: Tensor,
    pairwise_mask: Tensor,
    cutoff: float | Tensor = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> Tensor:
    """Score distance matrices ``D_pred`` and ``D_true`` with shape (..., l, l)."""
    # dmat_pred, dmat_true, pairwise_mask: (..., l, l); cutoff broadcasts with distances.

    sequence_length = dmat_true.size(-1)
    identity = torch.eye(sequence_length, device=dmat_true.device)  # (l, l)
    scored_pairs = (dmat_true < cutoff) * pairwise_mask * (1.0 - identity)  # (..., l, l)
    absolute_error = torch.abs(dmat_true - dmat_pred)  # (..., l, l)
    score = (
        (absolute_error < 0.5).type(absolute_error.dtype)
        + (absolute_error < 1.0).type(absolute_error.dtype)
        + (absolute_error < 2.0).type(absolute_error.dtype)
        + (absolute_error < 4.0).type(absolute_error.dtype)
    ) * 0.25  # (..., l, l)
    dimensions = (-1,) if per_residue else (-2, -1)
    normalization = 1.0 / (eps + scored_pairs.sum(dim=dimensions))  # (..., l) if per_residue, otherwise (...)
    return normalization * (eps + (scored_pairs * score).sum(dim=dimensions))  # (..., l) if per_residue, otherwise (...)


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
    # positions: (..., n, 3); all_atom_mask: (..., n); pairwise mask: (..., n, n).
    # sequence_id, when supplied, follows the point axes (..., n).

    expanded_mask = all_atom_mask[..., None]  # (..., n, 1)
    true_distances = _distance_matrix(all_atom_positions, eps)  # (..., n, n)
    predicted_distances = _distance_matrix(all_atom_pred_pos, eps)  # (..., n, n)
    pair_mask = expanded_mask * expanded_mask.transpose(-2, -1)  # (..., n, n)
    if pairwise_all_atom_mask is not None:
        pair_mask = pair_mask * pairwise_all_atom_mask  # (..., n, n)
    if sequence_id is not None:
        same_sequence = sequence_id[..., None] == sequence_id[..., None, :]  # (..., n, n)
        pair_mask = pair_mask * same_sequence.type_as(pair_mask)  # (..., n, n)
    return compute_lddt_from_dmat(
        predicted_distances,
        true_distances,
        pair_mask,
        cutoff=cutoff,
        eps=eps,
        per_residue=per_residue,
    )  # (..., n) if per_residue, otherwise (...)


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
    # True coordinates/mask: (..., l, n_atoms, 3) / (..., l, n_atoms).
    # Predicted rank-three input is treated as CA-only; otherwise its CA atom axis is selected.

    ca_index = residue_constants.atom_order["CA"]
    predicted_ca = (
        all_atom_pred_pos if all_atom_pred_pos.dim() == 3 else all_atom_pred_pos[..., ca_index, :]
    )  # (..., l, 3)
    return compute_lddt(
        predicted_ca,
        all_atom_positions[..., ca_index, :],
        all_atom_mask[..., ca_index],
        cutoff=cutoff,
        eps=eps,
        per_residue=per_residue,
        sequence_id=sequence_id,
    )  # (..., l) if per_residue, otherwise (...)


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
    # mobile/target: (b, n, 3) or (b, l, n_atoms, 3); masks omit xyz.
    # b_eff counts unpacked sequences when sequence_id is provided; n counts flattened atoms.

    centered_mobile, _, centered_target, _, rotation, counts = compute_alignment_tensors(
        mobile,
        target,
        atom_exists_mask,
        sequence_id,
    )  # coordinates (b_eff, n, 3), centroids (b_eff, 1, 3), rotation (b_eff, 3, 3), counts (b_eff, 1)
    rmsd = compute_rmsd_no_alignment(
        torch.matmul(centered_mobile, rotation),
        centered_target,
        counts,
        reduction=reduction,
    )  # (b_eff, n / 3) per_residue; (b_eff,) per_sample; () batch
    if reduction == "per_residue" and sequence_id is not None:
        return binpack(rmsd, sequence_id, pad_value=0)  # (b, packed_length)
    return rmsd  # shape selected by reduction above


def compute_gdt_ts(
    mobile: Tensor,
    target: Tensor,
    atom_exists_mask: Tensor | None = None,
    sequence_id: Tensor | None = None,
    reduction: str = "per_sample",
) -> Tensor:
    """Align ``X`` to ``Y`` and compute GDT-TS."""
    # mobile/target: batched xyz coordinates; masks omit xyz.
    # b_eff counts unpacked sequences when sequence_id is provided; n counts flattened atoms.

    if atom_exists_mask is None:
        atom_exists_mask = torch.isfinite(target).all(dim=-1)  # target.shape[:-1]
    centered_mobile, _, centered_target, _, rotation, _ = compute_alignment_tensors(
        mobile,
        target,
        atom_exists_mask,
        sequence_id,
    )  # coordinates (b_eff, n, 3), centroids (b_eff, 1, 3), rotation (b_eff, 3, 3), counts (b_eff, 1)
    if sequence_id is not None:
        atom_exists_mask = unbinpack(atom_exists_mask, sequence_id, pad_value=False)  # (b_eff, max_sequence_length, *original_mask.shape[2:])
    return compute_gdt_ts_no_alignment(
        torch.matmul(centered_mobile, rotation),
        centered_target,
        atom_exists_mask,
        reduction,
    )  # (b_eff,) per_sample; () batch


def _batched_contacts(predictions: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    # predictions, targets: (l, l) or (b, l, l).
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)  # (1, l, l)
    if targets.dim() == 2:
        targets = targets.unsqueeze(0)  # (1, l, l)
    if predictions.size() != targets.size():
        raise ValueError(
            f"Size mismatch. Received predictions of size {predictions.size()}, "
            f"targets of size {targets.size()}"
        )
    return predictions, targets  # each (b, l, l)


def _valid_contact_mask(
    targets: Tensor,
    src_lengths: Tensor,
    minsep: int,
    maxsep: int | None,
) -> Tensor:
    # targets: (b, l, l); src_lengths: (b,).
    sequence_length = targets.shape[-1]
    positions = torch.arange(sequence_length, device=targets.device)  # (l,)
    separation = (positions.unsqueeze(0) - positions.unsqueeze(1)).unsqueeze(0)  # (1, l, l)
    valid = (separation >= minsep) & (targets >= 0)  # (b, l, l)
    if maxsep is not None:
        valid &= separation < maxsep  # (b, l, l)
    within_length = positions.unsqueeze(0) < src_lengths.unsqueeze(1)  # (b, l)
    return valid & within_length.unsqueeze(1) & within_length.unsqueeze(2)  # (b, l, l)


def contact_precision(
    predictions: Tensor,
    targets: Tensor,
    src_lengths: Tensor | None = None,
    minsep: int = 6,
    maxsep: int | None = None,
    override_length: int | None = None,
) -> dict[str, Tensor]:
    """Compute P@L, P@L/5, and binned area for contact probabilities."""
    # predictions, targets: (l, l) or (b, l, l); src_lengths: (b,) or None.
    # n_upper_pairs counts the entries returned by triu_indices(l, minsep).

    predictions, targets = _batched_contacts(predictions, targets)  # each (b, l, l)
    batch_size, sequence_length, _ = predictions.shape
    if src_lengths is None:
        src_lengths = torch.full(
            (batch_size,),
            sequence_length,
            dtype=torch.long,
            device=predictions.device,
        )  # (b,)
    valid = _valid_contact_mask(targets, src_lengths, minsep, maxsep)  # (b, l, l)
    masked_predictions = predictions.masked_fill(~valid, float("-inf"))  # (b, l, l)
    row_index, column_index = np.triu_indices(sequence_length, minsep)  # each (n_upper_pairs,)
    upper_predictions = masked_predictions[:, row_index, column_index]  # (b, n_upper_pairs)
    upper_targets = targets[:, row_index, column_index]  # (b, n_upper_pairs)

    topk = sequence_length if override_length is None else max(sequence_length, override_length)
    ranked_indices = upper_predictions.argsort(dim=-1, descending=True)[:, :topk]  # (b, min(topk, n_upper_pairs))
    batch_indices = torch.arange(batch_size, device=ranked_indices.device).unsqueeze(1)  # (b, 1)
    ranked_targets = upper_targets[batch_indices, ranked_indices]  # (b, min(topk, n_upper_pairs))
    if ranked_targets.size(1) < topk:
        ranked_targets = F.pad(ranked_targets, [0, topk - ranked_targets.size(1)])  # (b, topk)
    cumulative_contacts = ranked_targets.type_as(predictions).cumsum(dim=-1)  # (b, topk)

    gather_lengths = src_lengths.unsqueeze(1)  # (b, 1)
    if override_length is not None:
        gather_lengths = override_length * torch.ones_like(gather_lengths)  # (b, 1)
    fractions = torch.arange(0.1, 1.1, 0.1, device=predictions.device).unsqueeze(0)  # (1, 10)
    gather_indices = (fractions * gather_lengths).type(torch.long).sub(1).clamp_min(0)  # (b, 10)
    cumulative_bins = cumulative_contacts.gather(1, gather_indices)  # (b, 10)
    precisions = cumulative_bins / (gather_indices + 1).type_as(cumulative_bins)  # (b, 10)
    return {
        "AUC": precisions.mean(dim=-1),
        "P@L": precisions[:, 9],
        "P@L5": precisions[:, 1],
    }  # each metric: (b,)
