"""CPU tests for target-level confidence-head metrics."""

from __future__ import annotations

import math

import pytest

from tools.confidence.metrics import evaluate_acceptance, summarize


def _record(
    target_id: str, kind: str, seed: int, split: str = "final_test", offset: float = 0.0
) -> dict[str, object]:
    return {
        "id": target_id,
        "kind": kind,
        "seed": seed,
        "split": split,
        "plddt_pred": [0.2 + offset, 0.8 + offset],
        "plddt_true": [0.1 + offset, 0.9],
        "plddt_ca_pred": [0.2 + offset],
        "plddt_ca_true": [0.1 + offset],
        "plddt_ce": 0.2 + offset,
        "pae_ce": 0.3 + offset,
        "pae_overflow_fraction": 0.1,
        "ptm": 0.2 + offset,
        "tm_score": 0.3 + offset,
        "iptm": 0.2 + offset,
        "dockq": 0.3 + offset,
    }


def _panel(offset: float = 0.0, target_count: int = 64) -> list[dict[str, object]]:
    records = []
    for index in range(target_count):
        kind = "monomer" if index < target_count // 2 else "dimer"
        target_offset = offset + (index % 20) * 0.005
        records.extend(
            [
                _record(f"target-{index}", kind, 0, offset=target_offset),
                _record(f"target-{index}", kind, 1, offset=target_offset),
            ]
        )
    return records


def test_summarize_deduplicates_seeds_and_reports_metrics() -> None:
    summary = summarize(_panel())
    assert summary["target_count"] == 64
    assert summary["final_test_strata_counts"] == {"monomer": 32, "dimer": 32}
    assert summary["atom_mae"] >= 0.0
    assert len(summary["bootstrap"]["atom_mae"]) == 2


def test_acceptance_requires_64_targets_per_final_stratum() -> None:
    summary = summarize(_panel())
    with pytest.raises(ValueError, match="missing required final-test strata"):
        evaluate_acceptance(summary, summary, summary)


def test_acceptance_gates_compare_both_categorical_losses() -> None:
    records = []
    records = _panel(target_count=128)
    candidate = summarize(records)
    donor = dict(candidate, plddt_ce=0.3, pae_ce=0.4, target_plddt_spearman=0.53)
    frequency = dict(candidate, plddt_ce=0.25, pae_ce=0.35)
    report = evaluate_acceptance(candidate, donor, frequency)
    assert report["accepted"] is False
    assert report["gates"]["plddt_beats_frequency"] is True
    assert report["gates"]["pae_beats_frequency"] is True


def test_nonfinite_values_are_rejected() -> None:
    record = _record("x", "monomer", 0)
    record["plddt_ce"] = math.nan
    with pytest.raises(ValueError, match="finite"):
        summarize([record])


def test_monomers_do_not_require_or_contribute_interface_metrics() -> None:
    records = [record for record in _panel(target_count=8) if record["kind"] == "monomer"]
    for record in records:
        record.pop("iptm")
        record.pop("dockq")
    summary = summarize(records)
    assert summary["iptm_dockq_spearman"] is None
    assert summary["bootstrap"]["iptm_dockq_spearman"] == {"interval": None, "valid_resamples": 0}


def test_constant_prediction_reports_undefined_rank_without_crashing() -> None:
    records = _panel(target_count=8)
    for record in records:
        record["plddt_pred"] = [0.5, 0.5]
        record["plddt_ca_pred"] = [0.5]
    summary = summarize(records)
    assert summary["target_plddt_spearman"] is None
    assert summary["bootstrap"]["target_plddt_spearman"]["valid_resamples"] == 0


def test_interface_bootstrap_uses_dimer_targets_only() -> None:
    records = _panel(target_count=8)
    for record in records[:8]:
        record.pop("iptm")
        record.pop("dockq")
    summary = summarize(records)
    bootstrap = summary["bootstrap"]["iptm_dockq_spearman"]
    assert bootstrap["valid_resamples"] >= 100
    assert bootstrap["interval"] is not None
