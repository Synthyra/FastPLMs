"""Pre-registered acceptance gates and reported secondary metrics for v2 confidence heads.

Three gates decide whether a v2 head is accepted. They are fixed before any test result exists:

1. Against the pilot head: v2 is at least as good on target pLDDT/lDDT Spearman and calibration
   error, and no reported metric is significantly worse (the paired 95% interval of the difference
   lies entirely on the worse side).
2. Sample selection: the lower 95% bound of within-target pairwise accuracy exceeds 0.5 for pLDDT
   and for ipTM against DockQ, and top-1 regret is below the regret of a random choice.
3. Production parity: each Spearman correlation is at most 0.03 below production `esmfold2`,
   calibration error at most 0.01 above it, and each within-target accuracy at most 0.03 below it.

Intervals come from one bootstrap over standard test targets shared by every head. The production
reference folds its own samples of the same targets, so its records join each draw by target id.

`production_agreement` reports how closely each head tracks production's confidence on the same
targets. It is reported only and gates nothing, because the gates were fixed before any result.
"""

from __future__ import annotations

import math
import numpy as np

from collections import defaultdict
from collections.abc import Mapping, Sequence

from .online_training import spearman
from .test_evaluation import BOOTSTRAP_SAMPLES, LONG_STRATUM, _metrics


HIGHER_IS_BETTER = {
    "plddt_lddt_spearman": True,
    "ptm_tm_spearman": True,
    "iptm_dockq_spearman": True,
    "atom_plddt_mae": False,
    "calibration_error_10bin": False,
    "plddt_ce": False,
    "pae_ce": False,
    "within_target_plddt_accuracy": True,
    "within_target_iptm_dockq_accuracy": True,
    "top1_regret": False,
}
# Reported with paired intervals but never gated; added before any test evaluation ran.
SECONDARY_METRICS = (
    "disorder_auroc",
    "resolved_residue_mean_plddt",
    "unresolved_residue_mean_plddt",
    "resolved_fraction_below_50",
    "unresolved_fraction_below_50",
)
REPORTED_METRICS = (*HIGHER_IS_BETTER, *SECONDARY_METRICS)
SPEARMAN_METRICS = ("plddt_lddt_spearman", "ptm_tm_spearman", "iptm_dockq_spearman")
ACCURACY_METRICS = ("within_target_plddt_accuracy", "within_target_iptm_dockq_accuracy")
# Confidence scores compared against production directly; ipTM is meaningful only for complexes.
AGREEMENT_SCORES = ("mean_plddt", "ptm", "iptm")
SPEARMAN_TOLERANCE = 0.03
CALIBRATION_TOLERANCE = 0.01
ACCURACY_TOLERANCE = 0.03

Record = Mapping[str, object]


def _by_target(records: Sequence[Record]) -> dict[str, list[Record]]:
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        if record["stratum"] != LONG_STRATUM:
            groups[str(record["target_id"])].append(record)
    return groups


def _interval(values: Sequence[float]) -> list[float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return [float("nan"), float("nan")]
    return [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]


def paired_estimates(
    model_records: Sequence[Record], heads: Sequence[str], reference_records: Sequence[Record], seed: int = 0
) -> dict[str, dict[str, object]]:
    """Point estimates and paired intervals for every head, production, and each head minus the pilot.

    `model_records` carry predictions of all `heads` on shared samples; `reference_records` carry the
    production predictions under the head name `production`.
    """
    model_targets, reference_targets = _by_target(model_records), _by_target(reference_records)
    target_ids = sorted(set(model_targets) & set(reference_targets))
    # An evaluation may have skipped a target that did not fit in memory, so compare what both kept.
    unshared = sorted(set(model_targets) ^ set(reference_targets))
    if not target_ids:
        raise ValueError("the model and reference evaluations share no standard test targets")

    def metrics_for(draw: Sequence[str]) -> dict[str, dict[str, float]]:
        # Each drawn copy gets its own id, so a target drawn twice stays two groups.
        model_draw = [{**record, "target_id": f"{target}#{copy}"} for copy, target in enumerate(draw) for record in model_targets[target]]
        reference_draw = [{**record, "target_id": f"{target}#{copy}"} for copy, target in enumerate(draw) for record in reference_targets[target]]
        values = {head: _metrics(model_draw, head) for head in heads}
        values["production"] = _metrics(reference_draw, "production")
        return values

    point = metrics_for(target_ids)
    rng = np.random.default_rng(seed)
    draws: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for _ in range(BOOTSTRAP_SAMPLES):
        drawn = [target_ids[index] for index in rng.integers(len(target_ids), size=len(target_ids))]
        values = metrics_for(drawn)
        for source, metrics in values.items():
            for name in REPORTED_METRICS:
                draws[source][name].append(metrics[name])
                if source != "pilot" and "pilot" in values:
                    draws[f"{source}-minus-pilot"][name].append(metrics[name] - values["pilot"][name])
                if source != "production":
                    draws[f"{source}-minus-production"][name].append(metrics[name] - values["production"][name])
    estimates: dict[str, dict[str, object]] = {}
    for source, metrics in draws.items():
        if "-minus-" in source:
            first, second = source.split("-minus-")
            estimate = {name: point[first][name] - point[second][name] for name in REPORTED_METRICS}
        else:
            estimate = {name: point[source][name] for name in REPORTED_METRICS} | {
                "random_selection_regret": point[source]["random_selection_regret"],
                "targets": point[source]["targets"],
            }
        estimates[source] = {"estimate": estimate, "interval_95": {name: _interval(values) for name, values in metrics.items()}}
    estimates["shared_targets"] = {"compared": len(target_ids), "not_in_both_evaluations": unshared}
    return estimates


def significantly_worse(difference_interval: Sequence[float], higher_is_better: bool) -> bool:
    """True when the whole paired interval of `candidate - baseline` lies on the worse side of zero."""
    low, high = difference_interval
    return high < 0 if higher_is_better else low > 0


def acceptance_gates(estimates: Mapping[str, Mapping[str, object]], head: str = "v2") -> dict[str, object]:
    """Evaluate the three pre-registered gates for `head` from `paired_estimates` output."""
    candidate = estimates[head]["estimate"]  # type: ignore[index]
    candidate_interval = estimates[head]["interval_95"]  # type: ignore[index]
    pilot = estimates["pilot"]["estimate"]  # type: ignore[index]
    versus_pilot = estimates[f"{head}-minus-pilot"]["interval_95"]  # type: ignore[index]
    production = estimates["production"]["estimate"]  # type: ignore[index]

    regressions = [name for name, higher in HIGHER_IS_BETTER.items() if significantly_worse(versus_pilot[name], higher)]
    beats_pilot = {
        "plddt_lddt_spearman_not_lower": candidate["plddt_lddt_spearman"] >= pilot["plddt_lddt_spearman"],
        "calibration_error_not_higher": candidate["calibration_error_10bin"] <= pilot["calibration_error_10bin"],
        "significant_regressions": regressions,
    }
    selection = {
        **{f"{name}_lower_bound_above_half": bool(candidate_interval[name][0] > 0.5) for name in ACCURACY_METRICS},
        "top1_regret_below_random": candidate["top1_regret"] < candidate["random_selection_regret"],
    }
    parity = {
        **{f"{name}_within_{SPEARMAN_TOLERANCE}": candidate[name] >= production[name] - SPEARMAN_TOLERANCE for name in SPEARMAN_METRICS},
        f"calibration_error_within_{CALIBRATION_TOLERANCE}": candidate["calibration_error_10bin"] <= production["calibration_error_10bin"] + CALIBRATION_TOLERANCE,
        **{f"{name}_within_{ACCURACY_TOLERANCE}": candidate[name] >= production[name] - ACCURACY_TOLERANCE for name in ACCURACY_METRICS},
    }
    gates = {
        "beats_pilot": bool(beats_pilot["plddt_lddt_spearman_not_lower"] and beats_pilot["calibration_error_not_higher"] and not regressions),
        "sample_selection": all(bool(value) for value in selection.values()),
        "production_parity": all(bool(value) for value in parity.values()),
    }
    return {"head": head, "passed": all(gates.values()), "gates": gates, "beats_pilot": beats_pilot, "sample_selection": selection, "production_parity": parity}


def _target_scores(records: Sequence[Record], head: str) -> dict[str, dict[str, float]]:
    """Each standard target's confidence scores, averaged over that head's samples of the target."""
    return {
        target: {
            **{name: float(np.mean([float(record["predictions"][head][name]) for record in group])) for name in AGREEMENT_SCORES},  # type: ignore[index]
            "num_chains": float(group[0]["num_chains"]),  # type: ignore[arg-type]
        }
        for target, group in _by_target(records).items()
    }


def production_agreement(model_records: Sequence[Record], heads: Sequence[str], reference_records: Sequence[Record]) -> dict[str, dict[str, float]]:
    """How closely each head's confidence tracks production `esmfold2` on the same targets.

    Each model folds its own diffusion samples, so a head and production never score one structure.
    The comparison is therefore between per-target mean scores: a high rank correlation means the
    two heads call the same targets confident, and the mean difference gives the offset between
    their scales.
    """
    reference = _target_scores(reference_records, "production")
    agreement: dict[str, dict[str, float]] = {}
    for head in heads:
        candidate = _target_scores(model_records, head)
        shared = sorted(set(candidate) & set(reference))
        values: dict[str, float] = {}
        for name in AGREEMENT_SCORES:
            compared = [target for target in shared if name != "iptm" or candidate[target]["num_chains"] > 1]
            head_scores = [candidate[target][name] for target in compared]
            production_scores = [reference[target][name] for target in compared]
            values[f"{name}_spearman"] = spearman(head_scores, production_scores)
            values[f"{name}_mean_difference"] = float(np.mean(np.subtract(head_scores, production_scores))) if compared else float("nan")
        agreement[head] = {**values, "targets": float(len(shared)), "multi_chain_targets": float(sum(candidate[target]["num_chains"] > 1 for target in shared))}
    return agreement
