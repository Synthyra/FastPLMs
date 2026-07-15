"""Predicted-aligned-error scores and training loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .esmfold2_affine3d import Affine3D

_CPU_DEVICE = torch.device("cpu")


def _compute_pae_masks(mask: Tensor) -> Tensor:
    residue_mask = mask.bool()
    return residue_mask.unsqueeze(-1) & residue_mask.unsqueeze(-2)


def _pae_bins(
    max_bin: float = 31,
    num_bins: int = 64,
    device: torch.device = _CPU_DEVICE,
) -> Tensor:
    """Return the representative distance for each PAE probability bin."""

    boundaries = torch.linspace(0, max_bin, steps=num_bins - 1, device=device)
    width = max_bin / (num_bins - 2)
    centers = boundaries + width / 2
    overflow_center = centers[-1:] + width
    return torch.cat((centers, overflow_center))


def _masked_probabilities(logits: Tensor, pair_mask: Tensor) -> Tensor:
    masked_logits = logits.masked_fill(
        ~pair_mask.unsqueeze(-1),
        torch.finfo(logits.dtype).min,
    )
    return masked_logits.softmax(dim=-1)


def masked_mean(
    mask: Tensor,
    value: Tensor,
    dim: int | tuple[int, ...] | None = None,
    eps: float = 1e-10,
) -> Tensor:
    """Average values over true entries of a broadcast-compatible mask."""

    weights = mask.expand_as(value)
    weighted_sum = torch.sum(weights * value, dim=dim)
    weight_sum = torch.sum(weights, dim=dim)
    return weighted_sum / (weight_sum + eps)


def compute_predicted_aligned_error(
    logits: Tensor,
    aa_mask: Tensor,
    sequence_id: Tensor | None = None,
    max_bin: float = 31,
) -> Tensor:
    """Convert PAE logits ``X`` with shape (..., l, l, n) to distances."""

    del sequence_id
    pair_mask = _compute_pae_masks(aa_mask)
    probabilities = _masked_probabilities(logits, pair_mask)
    centers = _pae_bins(max_bin, logits.shape[-1], logits.device)
    return torch.sum(probabilities * centers, dim=-1)


@torch.no_grad()
def compute_tm(logits: Tensor, aa_mask: Tensor, max_bin: float = 31.0) -> Tensor:
    """Estimate TM score from pairwise PAE logits."""

    pair_mask = _compute_pae_masks(aa_mask)
    sequence_lengths = aa_mask.sum(dim=-1, keepdim=True)
    centers = _pae_bins(max_bin, logits.shape[-1], logits.device)
    distance_scale = 1.24 * (sequence_lengths.clamp_min(19) - 15) ** (1 / 3) - 1.8
    tm_weights = 1.0 / (1 + (centers / distance_scale.unsqueeze(-1)) ** 2)
    probabilities = _masked_probabilities(logits, pair_mask)
    score_per_pair = torch.sum(probabilities * tm_weights.unsqueeze(-2), dim=-1)
    score_per_anchor = masked_mean(pair_mask, score_per_pair, dim=-1)
    return score_per_anchor.max(dim=-1).values


def _local_coordinates(frames: Affine3D) -> Tensor:
    origins = frames.trans[..., None, :, :]
    return frames.invert()[..., None].apply(origins)


def tm_loss(
    logits: Tensor,
    pred_affine: Tensor,
    targ_affine: Tensor,
    targ_mask: Tensor,
    tm_mask: Tensor | None = None,
    sequence_id: Tensor | None = None,
    max_bin: float = 31,
) -> Tensor:
    """Cross-entropy loss for discretized aligned-position errors."""

    del sequence_id
    predicted_frames = Affine3D.from_tensor(pred_affine)
    target_frames = Affine3D.from_tensor(targ_affine)
    with torch.no_grad():
        squared_error = (
            (_local_coordinates(predicted_frames) - _local_coordinates(target_frames))
            .square()
            .sum(dim=-1)
        )
        boundaries = torch.linspace(
            0,
            max_bin,
            logits.shape[-1] - 1,
            device=logits.device,
        ).square()
        target_bins = (squared_error[..., None] > boundaries).sum(dim=-1).long()

    cross_entropy = F.cross_entropy(
        logits.movedim(3, 1),
        target_bins,
        reduction="none",
    )
    pair_mask = _compute_pae_masks(targ_mask)
    loss_per_sample = masked_mean(pair_mask, cross_entropy, dim=(-1, -2))
    if tm_mask is None:
        return loss_per_sample.mean()
    return masked_mean(tm_mask, loss_per_sample)
