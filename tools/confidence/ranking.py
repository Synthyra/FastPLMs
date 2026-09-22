"""Differentiable confidence scores and within-target ranking losses.

Scores reproduce the experimental ESMFold2 head's own summaries: mean expected per-atom pLDDT,
and pTM and ipTM from PAE bin probabilities. The ranking loss orders the diffusion samples of one
target by their true quality, which the per-sample cross-entropy objective does not reward.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from collections.abc import Sequence

from torch import Tensor

from .labels import PAE_MAX_ANGSTROM, PLDDT_BINS


EPSILON = 1e-8


def expected_mean_plddt(plddt_logits: Tensor, atom_mask: Tensor) -> Tensor:
    """Mean expected pLDDT in [0, 1]; logits (b, a, 50) and mask (b, a) give (b,)."""
    # b: batch items; a: atoms.
    centers = (torch.arange(PLDDT_BINS, device=plddt_logits.device) + 0.5) / PLDDT_BINS  # (50,)
    per_atom = (plddt_logits.float().softmax(-1) * centers).sum(-1)  # (b, a)
    mask = atom_mask.float()  # (b, a)
    return (per_atom * mask).sum(-1) / mask.sum(-1).clamp(min=1)  # (b,)


def expected_tm_scores(pae_logits: Tensor, asym_id: Tensor, token_mask: Tensor) -> tuple[Tensor, Tensor]:
    """pTM and ipTM from PAE logits (b, t, t, bins), as the head computes them; returns (b,), (b,)."""
    # b: batch items; t: tokens; asym_id/token_mask: (b, t).
    bins = pae_logits.shape[-1]
    width = PAE_MAX_ANGSTROM / bins
    centers = torch.arange(0.5 * width, PAE_MAX_ANGSTROM, width, device=pae_logits.device)  # (bins,)
    mask = token_mask.float()  # (b, t)
    d0 = 1.24 * (mask.sum(-1, keepdim=True).clamp(min=19) - 15) ** (1 / 3) - 1.8  # (b, 1)
    tm_per_bin = 1 / (1 + (centers / d0) ** 2)  # (b, bins)
    tm = (pae_logits.float().softmax(-1) * tm_per_bin[:, None, None, :]).sum(-1)  # (b, t, t)
    pair = mask[:, :, None] * mask[:, None, :]  # (b, t, t)
    inter_chain = pair * (asym_id[:, :, None] != asym_id[:, None, :]).float()  # (b, t, t)
    ptm = ((tm * pair).sum(-1) / (pair.sum(-1) + EPSILON)).max(-1).values  # (b,)
    iptm = ((tm * inter_chain).sum(-1) / (inter_chain.sum(-1) + EPSILON)).max(-1).values  # (b,)
    return ptm, iptm  # each (b,)


def ranking_pairs(qualities: Sequence[float], margin: float) -> list[tuple[int, int]]:
    """Ordered (better, worse) sample pairs whose true quality differs by at least `margin`."""
    return [
        (better, worse)
        for better, better_quality in enumerate(qualities)
        for worse, worse_quality in enumerate(qualities)
        if better_quality - worse_quality >= margin
    ]


def sample_ranking_loss(
    sample: int,
    score: Tensor,
    detached_scores: Tensor,
    pairs: Sequence[tuple[int, int]],
    temperature: float,
) -> Tensor:
    """Ranking loss terms that involve `sample`, with every other sample's score detached.

    Summing this over all samples of a target gives exactly the gradient of the pairwise logistic
    loss `mean(softplus(-(s_better - s_worse) / temperature))`, while each backward pass only
    holds one sample's activations. The summed value is twice that loss.
    """
    # score: (); detached_scores: (samples,); each appended loss term is ().
    terms = []
    for better, worse in pairs:
        if better == sample:
            terms.append(functional.softplus(-(score - detached_scores[worse]) / temperature))
        elif worse == sample:
            terms.append(functional.softplus(-(detached_scores[better] - score) / temperature))
    if not terms:
        return score.sum() * 0.0  # ()
    return torch.stack(terms).sum() / len(pairs)  # (participating pairs,) -> ()
