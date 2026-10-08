"""Metrics and bundle accessors the folding compliance tests share, over bundles that two isolated producers wrote."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from collections.abc import Mapping


def feature_tensors(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return the ``feature__`` tensors of a bundle with the prefix removed."""

    # tensors: (...) one tensor per bundle name, any shape
    return {  # (...) the feature__ tensors without the prefix, shapes unchanged
        name.removeprefix("feature__"): tensor
        for name, tensor in tensors.items()
        if name.startswith("feature__")
    }


def bundle_output(tensors: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    """Return the ``output__`` tensor called ``name`` or raise ``KeyError``."""

    # tensors: (...) one tensor per bundle name, any shape
    key = f"output__{name}"
    if key not in tensors:
        raise KeyError(f"Structure bundle omits required output {name!r}.")
    return tensors[key]  # (...) the named output tensor, shaped as the model emitted it


def aligned_ca_rmsd(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Return the C-alpha RMSD after the best rigid alignment of ``actual`` onto ``expected``."""

    # actual, expected: (n, 3), where n is the number of C-alpha atoms.
    actual_centered = actual.float() - actual.float().mean(dim=0, keepdim=True)  # (n, 3)
    expected_centered = expected.float() - expected.float().mean(dim=0, keepdim=True)  # (n, 3)
    covariance = actual_centered.T @ expected_centered  # (3, 3)
    left, _, right = torch.linalg.svd(covariance)
    # correction: (3, 3)
    correction = torch.eye(3, dtype=torch.float32)
    correction[-1, -1] = torch.sign(torch.det(left @ right))
    rotation = left @ correction @ right  # (3, 3)
    aligned = actual_centered @ rotation  # (n, 3)
    return torch.sqrt(torch.mean(torch.sum((aligned - expected_centered) ** 2, dim=-1))).item()


def lddt_ca(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Return the C-alpha lDDT of ``actual`` against ``expected`` over pairs closer than 15 angstrom in ``expected``."""

    # actual, expected: (n, 3), where n is the number of C-alpha atoms.
    actual_distances = torch.cdist(actual.float(), actual.float())  # (n, n)
    expected_distances = torch.cdist(expected.float(), expected.float())  # (n, n)
    # pair_mask: (n, n)
    pair_mask = expected_distances.lt(15.0)
    pair_mask.fill_diagonal_(False)
    assert pair_mask.any(), "No valid C-alpha pairs for lDDT."
    errors = (actual_distances - expected_distances).abs()  # (n, n)
    # score: (n, n)
    score = torch.stack([errors.lt(threshold).float() for threshold in (0.5, 1.0, 2.0, 4.0)]).mean(
        dim=0
    )
    return score[pair_mask].mean().item()


def probability_jsd(
    actual_logits: torch.Tensor,
    expected_logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return the mean Jensen-Shannon divergence between two logit tensors over the masked sites."""

    # actual_logits, expected_logits: (*s, c), where s contains batch/sample and atom or token-pair axes.
    # mask: (...) covers the trailing axes of s; c is the number of confidence bins.
    actual_log_prob = F.log_softmax(actual_logits.float(), dim=-1)
    expected_log_prob = F.log_softmax(expected_logits.float(), dim=-1)
    actual_prob = actual_log_prob.exp()
    expected_prob = expected_log_prob.exp()
    mean_prob = 0.5 * (actual_prob + expected_prob)
    log_mean_prob = mean_prob.clamp_min(torch.finfo(torch.float32).tiny).log()
    jsd = 0.5 * (
        (actual_prob * (actual_log_prob - log_mean_prob)).sum(dim=-1)
        + (expected_prob * (expected_log_prob - log_mean_prob)).sum(dim=-1)
    )
    while mask.ndim < jsd.ndim:
        # Prepend one singleton sample/batch axis until the mask has shape rank len(s).
        mask = mask.unsqueeze(0)
    mask = torch.broadcast_to(mask, jsd.shape)
    assert mask.any()
    return jsd[mask].mean()  # ()
