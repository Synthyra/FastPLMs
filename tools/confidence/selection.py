"""Choose confidence checkpoints using separate held-out head metrics."""

from __future__ import annotations

import math

from collections.abc import Mapping


def improves_checkpoint(
    candidate: Mapping[str, float | None],
    previous: Mapping[str, float | None],
    donor: Mapping[str, float | None],
    *,
    overfit: bool = False,
) -> bool:
    """Require improvement in both categorical heads and preserve pLDDT ranking."""
    for metric in ("plddt_ce", "pae_ce"):
        value = candidate[metric]
        reference = previous[metric]
        if value is None or reference is None or not math.isfinite(value) or value >= reference:
            return False
    if overfit:
        return True
    ranking = candidate.get("target_plddt_spearman")
    donor_ranking = donor.get("target_plddt_spearman")
    return (
        ranking is not None
        and math.isfinite(ranking)
        and (donor_ranking is None or ranking >= donor_ranking - 0.02)
    )
