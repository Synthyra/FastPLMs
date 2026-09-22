"""Target-level metrics and acceptance gates for confidence-head pilots.

Records may contain multiple stochastic decoys for one target. All reported
statistics first collapse those records by target ID, so a second seed cannot
increase the effective sample count.
"""

from __future__ import annotations

import math
import random

from typing import Any


BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 17
REQUIRED_STRATA = {"monomer": 64, "dimer": 64}


def _number(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vector(record: dict[str, Any], name: str) -> list[float]:
    values = record.get(name)
    if not isinstance(values, list | tuple) or not values:
        raise ValueError(f"record is missing non-empty {name}")
    return [_number(value, name) for value in values]


def _mean(values: list[float], name: str) -> float:
    if not values:
        raise ValueError(f"undefined metric: {name}")
    return math.fsum(values) / len(values)


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        average_rank = (index + end - 1) / 2.0 + 1.0
        for position in order[index:end]:
            ranks[position] = average_rank
        index = end
    return ranks


def _spearman(left: list[float], right: list[float], name: str) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError(f"undefined metric: {name}")
    left_ranks = _rank(left)
    right_ranks = _rank(right)
    left_mean = _mean(left_ranks, name)
    right_mean = _mean(right_ranks, name)
    numerator = math.fsum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left_ranks, right_ranks, strict=False)
    )
    left_norm = math.fsum((a - left_mean) ** 2 for a in left_ranks)
    right_norm = math.fsum((b - right_mean) ** 2 for b in right_ranks)
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError(f"undefined metric: {name}")
    return numerator / math.sqrt(left_norm * right_norm)


def _group_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("each record must be a mapping")
        target_id = record.get("target_id", record.get("id"))
        kind = record.get("kind")
        if not target_id or kind not in REQUIRED_STRATA:
            raise ValueError("each record needs target_id/id and kind=monomer or dimer")
        target = groups.setdefault(str(target_id), {"kind": kind, "records": []})
        if target["kind"] != kind:
            raise ValueError(f"target {target_id} has inconsistent strata")
        target["records"].append(record)
    return groups


def _target_metrics(target: dict[str, Any]) -> dict[str, Any]:
    records = target["records"]
    atom_mae: list[float] = []
    ca_mae: list[float] = []
    predicted_seed_means: list[float] = []
    true_seed_means: list[float] = []
    plddt_ce: list[float] = []
    pae_ce: list[float] = []
    pae_overflow: list[float] = []
    ptm: list[float] = []
    tm_score: list[float] = []
    iptm: list[float] = []
    dockq: list[float] = []
    for record in records:
        predicted = _vector(record, "plddt_pred")
        truth = _vector(record, "plddt_true")
        if len(predicted) != len(truth):
            raise ValueError("plddt_pred and plddt_true must have equal length")
        predicted_seed_means.append(_mean(predicted, "predicted_plddt_seed"))
        true_seed_means.append(_mean(truth, "true_plddt_seed"))
        atom_mae.append(
            _mean([abs(a - b) for a, b in zip(predicted, truth, strict=False)], "atom_mae")
        )
        ca_pred = _vector(record, "plddt_ca_pred")
        ca_true = _vector(record, "plddt_ca_true")
        if len(ca_pred) != len(ca_true):
            raise ValueError("plddt_ca_pred and plddt_ca_true must have equal length")
        ca_mae.append(_mean([abs(a - b) for a, b in zip(ca_pred, ca_true, strict=False)], "ca_mae"))
        for destination, field in (
            (plddt_ce, "plddt_ce"),
            (pae_ce, "pae_ce"),
            (pae_overflow, "pae_overflow_fraction"),
            (ptm, "ptm"),
            (tm_score, "tm_score"),
        ):
            destination.append(_number(record.get(field), field))
        if record.get("iptm") is not None and record.get("dockq") is not None:
            iptm.append(_number(record["iptm"], "iptm"))
            dockq.append(_number(record["dockq"], "dockq"))
        elif target["kind"] == "dimer":
            raise ValueError("dimer records require iptm and dockq")
    return {
        "atom_mae": _mean(atom_mae, "atom_mae"),
        "ca_mae": _mean(ca_mae, "ca_mae"),
        "predicted_plddt": _mean(predicted_seed_means, "predicted_plddt"),
        "true_plddt": _mean(true_seed_means, "true_plddt"),
        "plddt_ce": _mean(plddt_ce, "plddt_ce"),
        "pae_ce": _mean(pae_ce, "pae_ce"),
        "pae_overflow_fraction": _mean(pae_overflow, "pae_overflow_fraction"),
        "ptm": _mean(ptm, "ptm"),
        "tm_score": _mean(tm_score, "tm_score"),
        "iptm": _mean(iptm, "iptm") if iptm else None,
        "dockq": _mean(dockq, "dockq") if dockq else None,
    }


def _calibration_error(targets: list[dict[str, Any]]) -> float:
    target_errors: list[float] = []
    for target in targets:
        seed_errors: list[float] = []
        for record in target["records"]:
            predicted = _vector(record, "plddt_pred")
            truth = _vector(record, "plddt_true")
            if len(predicted) != len(truth):
                raise ValueError("plddt_pred and plddt_true must have equal length")
            bins: list[list[tuple[float, float]]] = [[] for _ in range(10)]
            for estimate, observed in zip(predicted, truth, strict=False):
                if not 0.0 <= estimate <= 1.0 or not 0.0 <= observed <= 1.0:
                    raise ValueError("pLDDT values must be in [0, 1]")
                bins[min(9, int(estimate * 10))].append((estimate, observed))
            total = sum(len(values) for values in bins)
            if not total:
                raise ValueError("undefined metric: calibration_error")
            seed_errors.append(
                math.fsum(
                    abs(
                        _mean([a for a, _ in values], "calibration_bin")
                        - _mean([b for _, b in values], "calibration_bin")
                    )
                    * len(values)
                    / total
                    for values in bins
                    if values
                )
            )
        target_errors.append(_mean(seed_errors, "calibration_target"))
    return _mean(target_errors, "calibration_error")


def _bootstrap_interval(values: list[float]) -> list[float]:
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        raise ValueError("undefined bootstrap metric")
    rng = random.Random(BOOTSTRAP_SEED)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLES):
        samples.append(_mean([values[rng.randrange(len(values))] for _ in values], "bootstrap"))
    samples.sort()
    return [samples[int(0.025 * BOOTSTRAP_SAMPLES)], samples[int(0.975 * BOOTSTRAP_SAMPLES) - 1]]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return JSON-compatible target-level metrics for prediction records."""
    if not records:
        raise ValueError("records cannot be empty")
    grouped = _group_records(records)
    targets = [
        {
            "target_id": target_id,
            "kind": target["kind"],
            "metrics": _target_metrics(target),
            **target,
        }
        for target_id, target in grouped.items()
    ]
    target_metrics = [target["metrics"] for target in targets]
    predicted = [metric["predicted_plddt"] for metric in target_metrics]
    truth = [metric["true_plddt"] for metric in target_metrics]
    ptm = [metric["ptm"] for metric in target_metrics]
    tm_score = [metric["tm_score"] for metric in target_metrics]
    dimer_metrics = [target["metrics"] for target in targets if target["kind"] == "dimer"]
    strata = {kind: sum(target["kind"] == kind for target in targets) for kind in REQUIRED_STRATA}
    final_targets = [
        target
        for target in targets
        if any(
            record.get("split") in {"final_test", "final-test", "final"}
            for record in target["records"]
        )
    ]
    final_strata = {
        kind: sum(target["kind"] == kind for target in final_targets) for kind in REQUIRED_STRATA
    }
    return {
        "target_count": len(targets),
        "strata_counts": strata,
        "final_test_strata_counts": final_strata,
        "atom_mae": _mean([metric["atom_mae"] for metric in target_metrics], "atom_mae"),
        "ca_mae": _mean([metric["ca_mae"] for metric in target_metrics], "ca_mae"),
        "calibration_error_10bin": _calibration_error([target for target in grouped.values()]),
        "target_plddt_spearman": _optional_spearman(predicted, truth, "target_plddt_spearman"),
        "tm_spearman": _optional_spearman(ptm, tm_score, "tm_spearman"),
        "tm_spearman_by_stratum": {
            kind: _optional_spearman(
                [target["metrics"]["ptm"] for target in targets if target["kind"] == kind],
                [target["metrics"]["tm_score"] for target in targets if target["kind"] == kind],
                f"tm_spearman_{kind}",
            )
            for kind in REQUIRED_STRATA
        },
        "iptm_dockq_spearman": _optional_spearman(
            [item["iptm"] for item in dimer_metrics],
            [item["dockq"] for item in dimer_metrics],
            "iptm_dockq_spearman",
        ),
        "plddt_ce": _mean([metric["plddt_ce"] for metric in target_metrics], "plddt_ce"),
        "pae_ce": _mean([metric["pae_ce"] for metric in target_metrics], "pae_ce"),
        "pae_overflow_fraction": _mean(
            [metric["pae_overflow_fraction"] for metric in target_metrics], "pae_overflow_fraction"
        ),
        "bootstrap": {
            "atom_mae": _bootstrap_interval([metric["atom_mae"] for metric in target_metrics]),
            "target_plddt_spearman": _bootstrap_metric(targets, "target_plddt_spearman"),
            "iptm_dockq_spearman": _bootstrap_metric(
                [target for target in targets if target["kind"] == "dimer"], "iptm_dockq_spearman"
            ),
        },
    }


def _optional_spearman(left: list[float], right: list[float], name: str) -> float | None:
    try:
        return _spearman(left, right, name)
    except ValueError:
        return None


def _bootstrap_metric(targets: list[dict[str, Any]], metric_name: str) -> dict[str, Any]:
    if len(targets) < 2:
        return {"interval": None, "valid_resamples": 0}
    rng = random.Random(BOOTSTRAP_SEED)
    values: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [targets[rng.randrange(len(targets))] for _ in targets]
        sample_metrics = [target["metrics"] for target in sample]
        try:
            if metric_name == "target_plddt_spearman":
                estimate = _spearman(
                    [item["predicted_plddt"] for item in sample_metrics],
                    [item["true_plddt"] for item in sample_metrics],
                    metric_name,
                )
            else:
                estimate = _spearman(
                    [item["iptm"] for item in sample_metrics],
                    [item["dockq"] for item in sample_metrics],
                    metric_name,
                )
        except ValueError:
            continue
        values.append(estimate)
    if len(values) < 100:
        return {"interval": None, "valid_resamples": len(values)}
    values.sort()
    lower = values[int(0.025 * len(values))]
    upper = values[int(0.975 * len(values)) - 1]
    return {"interval": [lower, upper], "valid_resamples": len(values)}


def evaluate_acceptance(
    candidate_summary: dict[str, Any],
    donor_summary: dict[str, Any],
    frequency_summary: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the predeclared pilot gates and return a JSON report."""
    summaries = {
        "candidate": candidate_summary,
        "donor": donor_summary,
        "frequency": frequency_summary,
    }
    for label, summary in summaries.items():
        for key, value in summary.items():
            if isinstance(value, int | float) and not math.isfinite(float(value)):
                raise ValueError(f"{label} contains non-finite {key}")
        if any(
            summary.get("final_test_strata_counts", {}).get(kind, 0) < count
            for kind, count in REQUIRED_STRATA.items()
        ):
            raise ValueError(f"{label} is missing required final-test strata")

    def at_least(summary: dict[str, Any], key: str, threshold: float) -> bool:
        value = summary.get(key)
        return isinstance(value, int | float) and math.isfinite(float(value)) and value >= threshold

    def at_most(summary: dict[str, Any], key: str, threshold: float) -> bool:
        value = summary.get(key)
        return isinstance(value, int | float) and math.isfinite(float(value)) and value <= threshold

    def less_than(candidate_key: str, baseline: dict[str, Any]) -> bool:
        candidate = candidate_summary.get(candidate_key)
        baseline_value = baseline.get(candidate_key)
        return (
            all(
                isinstance(value, int | float) and math.isfinite(float(value))
                for value in (candidate, baseline_value)
            )
            and candidate < baseline_value
        )

    candidate_rank = candidate_summary.get("target_plddt_spearman")
    donor_rank = donor_summary.get("target_plddt_spearman")
    gates = {
        "plddt_mae": at_most(candidate_summary, "atom_mae", 0.10),
        "calibration_error": at_most(candidate_summary, "calibration_error_10bin", 0.05),
        "plddt_spearman": at_least(candidate_summary, "target_plddt_spearman", 0.50),
        "iptm_dockq_spearman": at_least(candidate_summary, "iptm_dockq_spearman", 0.30),
        "plddt_beats_donor": less_than("plddt_ce", donor_summary),
        "pae_beats_donor": less_than("pae_ce", donor_summary),
        "plddt_beats_frequency": less_than("plddt_ce", frequency_summary),
        "pae_beats_frequency": less_than("pae_ce", frequency_summary),
        "donor_rank_decrease": isinstance(candidate_rank, int | float)
        and isinstance(donor_rank, int | float)
        and candidate_rank >= donor_rank - 0.02,
    }
    return {
        "accepted": all(gates.values()),
        "gates": gates,
        "thresholds": {
            "atom_mae": 0.10,
            "calibration_error_10bin": 0.05,
            "target_plddt_spearman": 0.50,
            "iptm_dockq_spearman": 0.30,
            "max_donor_rank_decrease": 0.02,
        },
    }
