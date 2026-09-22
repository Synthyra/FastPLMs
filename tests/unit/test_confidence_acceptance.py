"""Pre-registered v2 acceptance gates on synthetic evaluation records."""

import pytest

from tools.confidence import acceptance

from .test_confidence_test_evaluation import _calibration_fields


QUALITIES = (0.5, 0.7, 0.9)
TARGETS = (("m1", 1), ("m2", 1), ("d1", 2), ("d2", 2))


def _prediction(order: str, quality: float) -> dict[str, object]:
    """A head that ranks and calibrates perfectly ("correct") or inverts every sample ("reversed")."""
    score = quality if order == "correct" else 1.4 - quality
    return {"mean_plddt": score, "ptm": score, "iptm": score, "plddt_ce": 1.0 if order == "correct" else 3.0, "pae_ce": 2.0, **_calibration_fields(score, quality)}


def _records(orders: dict[str, str]) -> list[dict[str, object]]:
    records = []
    for target, chains in TARGETS:
        for sample, quality in enumerate(QUALITIES):
            records.append(
                {
                    "target_id": target,
                    "stratum": "monomer_short" if chains == 1 else "dimer_hetero",
                    "num_chains": chains,
                    "sample": sample,
                    "true_lddt": quality,
                    "tm_score": quality,
                    "dockq": quality if chains > 1 else None,
                    "predictions": {head: _prediction(order, quality) for head, order in orders.items()},
                }
            )
    return records


@pytest.fixture(autouse=True)
def few_bootstrap_draws(monkeypatch):
    monkeypatch.setattr(acceptance, "BOOTSTRAP_SAMPLES", 50)


def test_a_head_that_ranks_and_calibrates_like_production_passes_every_gate():
    model = _records({"v2": "correct", "pilot": "reversed", "donor": "reversed"})
    reference = _records({"production": "correct"})
    estimates = acceptance.paired_estimates(model, ["v2", "pilot", "donor"], reference)
    result = acceptance.acceptance_gates(estimates)
    assert result["gates"] == {"beats_pilot": True, "sample_selection": True, "production_parity": True}
    assert result["passed"]
    assert estimates["v2-minus-pilot"]["estimate"]["plddt_ce"] == pytest.approx(-2.0)


def test_a_head_that_inverts_samples_fails_against_the_pilot_and_production():
    model = _records({"v2": "reversed", "pilot": "correct", "donor": "reversed"})
    reference = _records({"production": "correct"})
    result = acceptance.acceptance_gates(acceptance.paired_estimates(model, ["v2", "pilot", "donor"], reference))
    assert result["gates"] == {"beats_pilot": False, "sample_selection": False, "production_parity": False}
    assert "within_target_plddt_accuracy" in result["beats_pilot"]["significant_regressions"]


def test_the_long_stratum_is_left_out_and_only_shared_targets_are_compared():
    model = _records({"v2": "correct", "pilot": "correct", "donor": "correct"})
    reference = _records({"production": "correct"})
    for record in model + reference:
        if record["target_id"] == "m2":
            record["stratum"] = "long"
    estimates = acceptance.paired_estimates(model, ["v2", "pilot", "donor"], reference)
    assert estimates["v2"]["estimate"]["targets"] == 3.0
    assert estimates["shared_targets"] == {"compared": 3, "not_in_both_evaluations": []}

    # A target one evaluation had to skip leaves the others comparable.
    partial = [record for record in reference if record["target_id"] != "d2"]
    estimates = acceptance.paired_estimates(model, ["v2", "pilot", "donor"], partial)
    assert estimates["shared_targets"] == {"compared": 2, "not_in_both_evaluations": ["d2"]}
    assert estimates["v2"]["estimate"]["targets"] == 2.0
    with pytest.raises(ValueError, match="share no standard test targets"):
        acceptance.paired_estimates(model, ["v2", "pilot", "donor"], [])


def test_significance_follows_the_direction_of_each_metric():
    assert acceptance.significantly_worse([-0.2, -0.1], higher_is_better=True)
    assert not acceptance.significantly_worse([-0.2, 0.1], higher_is_better=True)
    assert acceptance.significantly_worse([0.1, 0.2], higher_is_better=False)
    assert not acceptance.significantly_worse([float("nan"), float("nan")], higher_is_better=False)


AGREEMENT_TARGETS = (("m1", 1), ("m2", 1), ("d1", 2), ("d2", 2), ("d3", 2))


def _agreement_records() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """A head that tracks production on pLDDT but reverses its interface ranking of the complexes."""
    model, reference = [], []
    for index, (target, chains) in enumerate(AGREEMENT_TARGETS):
        interface = 0.0 if chains == 1 else 0.5 + 0.1 * index
        for sample in range(2):
            shared = {"target_id": target, "stratum": "monomer_short" if chains == 1 else "dimer_hetero", "num_chains": chains, "sample": sample}
            model.append({**shared, "predictions": {"v2": {"mean_plddt": 0.55 + 0.1 * index, "ptm": 0.55 + 0.1 * index, "iptm": 1.0 - interface}}})
            reference.append({**shared, "predictions": {"production": {"mean_plddt": 0.5 + 0.1 * index, "ptm": 0.5 + 0.1 * index, "iptm": interface}}})
    return model, reference


def test_agreement_with_production_follows_each_score_on_its_own_targets():
    model, reference = _agreement_records()
    agreement = acceptance.production_agreement(model, ["v2"], reference)["v2"]
    assert agreement["mean_plddt_spearman"] == pytest.approx(1.0)
    assert agreement["mean_plddt_mean_difference"] == pytest.approx(0.05)
    # ipTM uses the three complexes only, where this head ranks interfaces backwards.
    assert agreement["iptm_spearman"] == pytest.approx(-1.0)
    assert (agreement["targets"], agreement["multi_chain_targets"]) == (5.0, 3.0)


def test_agreement_uses_standard_targets_shared_by_both_evaluations():
    model, reference = _agreement_records()
    for record in model + reference:
        if record["target_id"] == "m2":
            record["stratum"] = "long"
    agreement = acceptance.production_agreement(model, ["v2"], [record for record in reference if record["target_id"] != "d3"])["v2"]
    assert agreement["targets"] == 3.0
    assert agreement["iptm_spearman"] != agreement["iptm_spearman"]  # two complexes are too few to rank
