"""CPU-only calculations for the v2 confidence protocol.

Test correlations use individual diffusion samples; bootstrap draws keep all samples of each
selected target together. Long targets are reported separately. The pilot protocol in
``metrics.py`` instead collapses samples by target and has different acceptance rules.
"""

from __future__ import annotations

import numpy as np

from collections import defaultdict
from collections.abc import Mapping, Sequence

from scipy.stats import rankdata

from .v2_records import EvaluationRecord, HeadSummary


PAIR_MARGIN_EVALUATION = 0.02
DOCKQ_MARGIN = 0.05
CALIBRATION_BINS = 10
DISORDER_BINS = 50
BOOTSTRAP_SAMPLES = 1000
LONG_STRATUM = "long"


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    """Average-rank correlation, undefined for fewer than three or non-finite observations."""
    if len(left) != len(right):
        raise ValueError("Spearman inputs must have equal lengths")
    if len(left) < 3 or not np.isfinite(left).all() or not np.isfinite(right).all():
        return float("nan")
    left_rank = rankdata(left, method="average")  # (observations,)
    right_rank = rankdata(right, method="average")  # (observations,)
    if np.ptp(left_rank) == 0 or np.ptp(right_rank) == 0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def pairwise_accuracy(
    predicted: Sequence[float], true: Sequence[float], margin: float
) -> tuple[int, int]:
    """Correctly ordered and total sample pairs whose true values differ by at least `margin`."""
    correct = total = 0
    for first in range(len(true)):
        for second in range(first + 1, len(true)):
            difference = true[first] - true[second]
            if abs(difference) < margin:
                continue
            total += 1
            correct += (predicted[first] - predicted[second]) * difference > 0
    return correct, total


def sample_metrics(records: Sequence[EvaluationRecord], head: str) -> dict[str, float]:
    """Aggregate sample statistics and within-target selection under the fixed v2 protocol."""
    by_target: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        by_target[str(record["target_id"])].append(record)
    predictions = [record["predictions"][head] for record in records]
    counts = np.sum([values["calibration_count"] for values in predictions], axis=0)  # (bins,)
    predicted_sums = np.sum(
        [values["calibration_predicted_sum"] for values in predictions], axis=0
    )  # (bins,)
    true_sums = np.sum(
        [values["calibration_true_sum"] for values in predictions], axis=0
    )  # (bins,)
    # Atom-weighted mean over bins of |mean predicted pLDDT - mean true lDDT|; empty bins add zero.
    calibration = np.abs(predicted_sums - true_sums).sum() / counts.sum()  # ()
    resolved = np.sum(
        [values["resolved_residue_histogram"] for values in predictions], axis=0
    )  # (bins,)
    unresolved = np.sum(
        [values["unresolved_residue_histogram"] for values in predictions], axis=0
    )  # (bins,)
    resolved_total, unresolved_total = float(resolved.sum()), float(unresolved.sum())
    both = resolved_total > 0 and unresolved_total > 0
    # Probability of lower pLDDT on an unresolved residue; shared bins count half.
    resolved_above = resolved_total - np.cumsum(
        resolved
    )  # (bins,) resolved residues in higher bins
    disorder_auroc = (
        float(
            (unresolved * (resolved_above + 0.5 * resolved)).sum()
            / (unresolved_total * resolved_total)
        )
        if both
        else float("nan")
    )
    below_half = DISORDER_BINS // 2  # bins under pLDDT 0.5
    complexes = [
        record
        for record in records
        if int(record["num_chains"]) > 1 and record["dockq"] is not None
    ]
    plddt_correct = plddt_total = dockq_correct = dockq_total = 0
    regret, random_regret = [], []
    complexes_without_dockq = 0
    for samples in by_target.values():
        scores = [sample["predictions"][head] for sample in samples]
        correct, total = pairwise_accuracy(
            [score["mean_plddt"] for score in scores],
            [sample["true_lddt"] for sample in samples],
            PAIR_MARGIN_EVALUATION,
        )
        plddt_correct, plddt_total = plddt_correct + correct, plddt_total + total
        if int(samples[0]["num_chains"]) == 1:
            quality = [float(sample["true_lddt"]) for sample in samples]
            ranking = [
                score["mean_plddt"] for score in scores
            ]  # ESMFold2 selects monomers by pLDDT
        elif all(sample["dockq"] is not None for sample in samples):
            quality = [float(sample["dockq"]) for sample in samples]
            ranking = [score["iptm"] for score in scores]  # and complexes by ipTM
            correct, total = pairwise_accuracy(ranking, quality, DOCKQ_MARGIN)
            dockq_correct, dockq_total = dockq_correct + correct, dockq_total + total
        else:
            # DockQ is undefined above 26 chains or without a native interface, so the complex has
            # no selection quality; every head and the reference exclude the same targets.
            complexes_without_dockq += 1
            continue
        regret.append(max(quality) - quality[int(np.argmax(ranking))])
        random_regret.append(max(quality) - float(np.mean(quality)))
    return {
        "targets": float(len(by_target)),
        "plddt_lddt_spearman": spearman(
            [record["predictions"][head]["mean_plddt"] for record in records],
            [record["true_lddt"] for record in records],
        ),
        "ptm_tm_spearman": spearman(
            [record["predictions"][head]["ptm"] for record in records],
            [record["tm_score"] for record in records],
        ),
        "iptm_dockq_spearman": spearman(
            [record["predictions"][head]["iptm"] for record in complexes],
            [record["dockq"] for record in complexes],
        )
        if complexes
        else float("nan"),
        "atom_plddt_mae": float(
            sum(values["absolute_error_sum"] for values in predictions) / counts.sum()
        ),
        "calibration_error_10bin": float(calibration),
        "plddt_ce": float(np.mean([values["plddt_ce"] for values in predictions])),
        "pae_ce": float(np.mean([values["pae_ce"] for values in predictions])),
        "within_target_plddt_accuracy": plddt_correct / plddt_total
        if plddt_total
        else float("nan"),
        "within_target_plddt_pairs": float(plddt_total),
        "within_target_iptm_dockq_accuracy": dockq_correct / dockq_total
        if dockq_total
        else float("nan"),
        "within_target_iptm_dockq_pairs": float(dockq_total),
        "top1_regret": float(np.mean(regret)),
        "random_selection_regret": float(np.mean(random_regret)),
        "complexes_without_dockq": float(complexes_without_dockq),
        "disorder_auroc": disorder_auroc,
        "resolved_residue_mean_plddt": sum(
            values["resolved_residue_plddt_sum"] for values in predictions
        )
        / resolved_total
        if resolved_total
        else float("nan"),
        "unresolved_residue_mean_plddt": sum(
            values["unresolved_residue_plddt_sum"] for values in predictions
        )
        / unresolved_total
        if unresolved_total
        else float("nan"),
        "resolved_fraction_below_50": float(resolved[:below_half].sum()) / resolved_total
        if resolved_total
        else float("nan"),
        "unresolved_fraction_below_50": float(unresolved[:below_half].sum()) / unresolved_total
        if unresolved_total
        else float("nan"),
        "unresolved_residues": unresolved_total,
    }


def bootstrap_records(
    by_target: Mapping[str, Sequence[EvaluationRecord]], rng: np.random.Generator
) -> list[EvaluationRecord]:
    """Resample whole targets; distinct draw IDs keep repeated targets as separate groups."""
    target_ids = sorted(by_target)
    return [
        {**record, "target_id": f"{target_ids[index]}#{draw}"}
        for draw, index in enumerate(rng.integers(len(target_ids), size=len(target_ids)))
        for record in by_target[target_ids[index]]
    ]


def summarize(records: Sequence[EvaluationRecord], heads: Sequence[str]) -> dict[str, HeadSummary]:
    """Standard-set estimates with target-bootstrap 95% intervals, plus per-stratum estimates.

    The long stratum lies beyond the training length, so it is reported only in `by_stratum`.
    """
    rng = np.random.default_rng(0)
    standard = [record for record in records if record["stratum"] != LONG_STRATUM]
    by_target: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for record in standard:
        by_target[str(record["target_id"])].append(record)
    strata = sorted({str(record["stratum"]) for record in records})
    summary: dict[str, HeadSummary] = {}
    for head in heads:
        draws: dict[str, list[float]] = defaultdict(list)
        for _ in range(BOOTSTRAP_SAMPLES):
            for name, value in sample_metrics(bootstrap_records(by_target, rng), head).items():
                draws[name].append(value)
        summary[head] = {
            "overall": sample_metrics(standard, head),
            "interval_95": {
                name: [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]
                for name, values in draws.items()
            },
            "by_stratum": {
                name: sample_metrics(
                    [record for record in records if record["stratum"] == name], head
                )
                for name in strata
            },
        }
    return summary
