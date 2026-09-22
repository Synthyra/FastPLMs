"""Serialized sample records used by the v2 confidence evaluation protocol.

These types describe saved JSON without importing training or structure code. The pilot's
target-collapsed records and acceptance rules remain in ``metrics.py``.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict


class PredictionSummary(TypedDict):
    """One head's confidence and sufficient statistics for one diffusion sample.

    Confidence values use the interval [0, 1]. Calibration vectors have ten bins and disorder
    histograms have fifty bins. Sums and counts preserve atom and residue weighting during
    target bootstrap resampling.
    """

    mean_plddt: float
    mean_plddt_labeled: NotRequired[float]
    ptm: float
    iptm: float
    plddt_ce: float
    pae_ce: float
    absolute_error_sum: float
    calibration_count: list[int]
    calibration_predicted_sum: list[float]
    calibration_true_sum: list[float]
    resolved_residue_histogram: list[int]
    unresolved_residue_histogram: list[int]
    resolved_residue_plddt_sum: float
    unresolved_residue_plddt_sum: float


class CoordinateIdentity(TypedDict):
    """Identity of the native coordinate array used to construct an evaluation target."""

    sha256: str
    shape: list[int]
    dtype: str


class EvaluationRecord(TypedDict):
    """Quality and per-head predictions for one sample, grouped by target for resampling."""

    target_id: str
    stratum: str
    num_chains: int
    num_tokens: NotRequired[int]
    target_positions: NotRequired[CoordinateIdentity]
    sample: int
    true_lddt: float
    true_lddt_ca: NotRequired[float]
    true_true_ptm: NotRequired[float]
    true_true_iptm: NotRequired[float]
    tm_score: float
    dockq: float | None
    predictions: dict[str, PredictionSummary]


class HeadSummary(TypedDict):
    """Standard-target estimates and intervals, with separate per-stratum estimates."""

    overall: dict[str, float]
    interval_95: dict[str, list[float]]
    by_stratum: dict[str, dict[str, float]]
