"""Predicted-aligned-error scores and training loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from torch import Tensor

from .esmfold2_affine3d import Affine3D


_CPU_DEVICE = torch.device("cpu")


def _compute_pae_masks(mask: Tensor) -> Tensor:
    # mask: (..., l), where l is the residue count.
    residue_mask = mask.bool()  # (..., l)
    return residue_mask.unsqueeze(-1) & residue_mask.unsqueeze(-2)  # (..., l, l)


def _pae_bins(
    max_bin: float = 31,
    num_bins: int = 64,
    device: torch.device = _CPU_DEVICE,
) -> Tensor:
    """Return the representative distance for each PAE probability bin."""

    boundaries = torch.linspace(0, max_bin, steps=num_bins - 1, device=device)  # (n_bins - 1,)
    width = max_bin / (num_bins - 2)
    centers = boundaries + width / 2  # (n_bins - 1,)
    overflow_center = centers[-1:] + width  # (1,)
    return torch.cat((centers, overflow_center))  # (n_bins,)


def _masked_probabilities(logits: Tensor, pair_mask: Tensor) -> Tensor:
    # logits: (..., l, l, n_bins); pair_mask: (..., l, l).
    masked_logits = logits.masked_fill(
        ~pair_mask.unsqueeze(-1),
        torch.finfo(logits.dtype).min,
    )  # (..., l, l, n_bins)
    return masked_logits.softmax(dim=-1)  # (..., l, l, n_bins)


def masked_mean(
    mask: Tensor,
    value: Tensor,
    dim: int | tuple[int, ...] | None = None,
    eps: float = 1e-10,
) -> Tensor:
    """Average values over true entries of a broadcast-compatible mask."""

    # value has arbitrary shape; reduced_shape removes the axes named by dim.
    weights = mask.expand_as(value)  # value.shape
    weighted_sum = torch.sum(weights * value, dim=dim)  # reduced_shape
    weight_sum = torch.sum(weights, dim=dim)  # reduced_shape
    return weighted_sum / (weight_sum + eps)  # reduced_shape


def compute_predicted_aligned_error(
    logits: Tensor,
    aa_mask: Tensor,
    sequence_id: Tensor | None = None,
    max_bin: float = 31,
) -> Tensor:
    """Convert PAE logits ``X`` with shape (..., l, l, n) to distances."""

    del sequence_id
    pair_mask = _compute_pae_masks(aa_mask)  # (..., l, l)
    probabilities = _masked_probabilities(logits, pair_mask)  # (..., l, l, n_bins)
    centers = _pae_bins(max_bin, logits.shape[-1], logits.device)  # (n_bins,)
    return torch.sum(probabilities * centers, dim=-1)  # (..., l, l)


@torch.no_grad()
def compute_tm(logits: Tensor, aa_mask: Tensor, max_bin: float = 31.0) -> Tensor:
    """Estimate TM score from logits (..., l, l, n_bins) and residue mask (..., l)."""

    pair_mask = _compute_pae_masks(aa_mask)  # (..., l, l)
    sequence_lengths = aa_mask.sum(dim=-1, keepdim=True)  # (..., 1)
    centers = _pae_bins(max_bin, logits.shape[-1], logits.device)  # (n_bins,)
    distance_scale = 1.24 * (sequence_lengths.clamp_min(19) - 15) ** (1 / 3) - 1.8  # (..., 1)
    tm_weights = 1.0 / (1 + (centers / distance_scale.unsqueeze(-1)) ** 2)  # (..., 1, n_bins)
    probabilities = _masked_probabilities(logits, pair_mask)  # (..., l, l, n_bins)
    score_per_pair = torch.sum(probabilities * tm_weights.unsqueeze(-2), dim=-1)  # (..., l, l)
    score_per_anchor = masked_mean(pair_mask, score_per_pair, dim=-1)  # (..., l)
    return score_per_anchor.max(dim=-1).values  # (...,)


def _local_coordinates(frames: Affine3D) -> Tensor:
    # frames.shape: (..., l); trans: (..., l, 3).
    origins = frames.trans[..., None, :, :]  # (..., 1, l, 3)
    return frames.invert()[..., None].apply(origins)  # (..., l, l, 3)


def tm_loss(
    logits: Tensor,
    pred_affine: Tensor,
    targ_affine: Tensor,
    targ_mask: Tensor,
    tm_mask: Tensor | None = None,
    sequence_id: Tensor | None = None,
    max_bin: float = 31,
) -> Tensor:
    """Cross-entropy loss for logits (b, l, l, n_bins) and residue frames (b, l)."""

    del sequence_id
    predicted_frames = Affine3D.from_tensor(pred_affine)  # frame shape: (b, l)
    target_frames = Affine3D.from_tensor(targ_affine)  # frame shape: (b, l)
    with torch.no_grad():
        squared_error = (
            (_local_coordinates(predicted_frames) - _local_coordinates(target_frames))
            .square()
            .sum(dim=-1)
        )  # (b, l, l)
        boundaries = torch.linspace(
            0,
            max_bin,
            logits.shape[-1] - 1,
            device=logits.device,
        ).square()  # (n_bins - 1,)
        target_bins = (squared_error[..., None] > boundaries).sum(dim=-1).long()  # (b, l, l)

    cross_entropy = F.cross_entropy(
        logits.movedim(3, 1),
        target_bins,
        reduction="none",
    )  # (b, l, l)
    pair_mask = _compute_pae_masks(targ_mask)  # (b, l, l)
    loss_per_sample = masked_mean(pair_mask, cross_entropy, dim=(-1, -2))  # (b,)
    if tm_mask is None:
        return loss_per_sample.mean()  # ()
    return masked_mean(tm_mask, loss_per_sample)  # ()
