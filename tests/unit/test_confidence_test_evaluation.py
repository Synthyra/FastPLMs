"""Confidence v2 test metrics: calibration sums, within-target ranking, regret, and bootstrap."""

import numpy as np
import pytest
import torch

from types import SimpleNamespace

from tools.confidence.test_evaluation import _metrics, _summaries, bootstrap_records, summarize


def _calibration_fields(predicted: float, true: float, atoms: int = 2) -> dict[str, object]:
    """Summary sums of `atoms` resolved residues, one atom each, sharing one predicted and one true value."""
    counts, predicted_sums, true_sums = [0] * 10, [0.0] * 10, [0.0] * 10
    index = min(int(predicted * 10), 9)
    counts[index], predicted_sums[index], true_sums[index] = atoms, predicted * atoms, true * atoms
    residues = [0] * 50
    residues[min(int(predicted * 50), 49)] = atoms
    return {
        "absolute_error_sum": abs(predicted - true) * atoms,
        "calibration_count": counts,
        "calibration_predicted_sum": predicted_sums,
        "calibration_true_sum": true_sums,
        "resolved_residue_histogram": residues,
        "unresolved_residue_histogram": [0] * 50,
        "resolved_residue_plddt_sum": predicted * atoms,
        "unresolved_residue_plddt_sum": 0.0,
    }


def _records(head_order: str) -> list[dict[str, object]]:
    """Two monomers and one dimer with three samples each; the head ranks samples as `head_order` says."""
    records = []
    for target, chains in (("m1", 1), ("m2", 1), ("d1", 2)):
        for sample, quality in enumerate((0.5, 0.7, 0.9)):
            score = quality if head_order == "correct" else 1.4 - quality
            records.append(
                {
                    "target_id": target,
                    "stratum": "monomer_short" if chains == 1 else "dimer_hetero",
                    "num_chains": chains,
                    "sample": sample,
                    "true_lddt": quality,
                    "tm_score": quality,
                    "dockq": quality if chains > 1 else None,
                    "predictions": {
                        "head": {
                            "mean_plddt": score,
                            "ptm": score,
                            "iptm": score,
                            "plddt_ce": 1.0,
                            "pae_ce": 2.0,
                            **_calibration_fields(score, quality),
                        }
                    },
                }
            )
    return records


def test_sample_sums_reproduce_atom_level_calibration_and_error():
    generator = torch.Generator().manual_seed(0)
    atoms, tokens = 60, 12
    plddt_logits = torch.randn(1, atoms, 50, generator=generator) * 3  # (1, a, bins)
    pae_logits = torch.randn(1, tokens, tokens, 64, generator=generator)  # (1, t, t, bins)
    plddt_score = torch.rand(atoms, generator=generator)  # (a,)
    labeled = torch.rand(atoms, generator=generator) > 0.2  # (a,)
    rollout = SimpleNamespace(
        targets=[
            {
                "plddt_mask": labeled,
                "plddt_score": plddt_score,
                "plddt_target": (plddt_score * 50).long().clamp(max=49),
                "pae_target": torch.zeros(tokens, tokens, dtype=torch.long),
                "pae_mask": torch.ones(tokens, tokens, dtype=torch.bool),
            }
        ],
        head_inputs={
            "atom_attention_mask": torch.ones(1, atoms),
            "asym_id": torch.zeros(1, tokens, dtype=torch.long),
            "token_attention_mask": torch.ones(1, tokens),
        },
        layout=SimpleNamespace(chain_ca=(torch.arange(1, atoms, 5),)),
        true_coords=torch.zeros(1, atoms, 3),
    )
    summary = _summaries(plddt_logits, pae_logits, rollout, 0)
    record = {"target_id": "t", "stratum": "monomer_short", "num_chains": 1, "sample": 0, "true_lddt": 0.5, "tm_score": 0.5, "dockq": None}
    metrics = _metrics([{**record, "predictions": {"head": summary}}], "head")

    predicted = (plddt_logits.softmax(-1) * ((torch.arange(50) + 0.5) / 50)).sum(-1)[0][labeled].numpy()  # (n,)
    true = plddt_score[labeled].numpy()  # (n,)
    bins = np.clip((predicted * 10).astype(int), 0, 9)
    expected = sum((bins == b).mean() * abs(predicted[bins == b].mean() - true[bins == b].mean()) for b in range(10) if (bins == b).any())
    assert metrics["calibration_error_10bin"] == pytest.approx(expected, abs=1e-5)
    assert metrics["atom_plddt_mae"] == pytest.approx(np.abs(predicted - true).mean(), abs=1e-5)


def test_disorder_metrics_separate_unresolved_from_resolved_residues():
    atoms, tokens = 16, 4
    ca = torch.tensor([1, 5, 9, 13])  # one C-alpha per residue; the last two are unresolved
    plddt_logits = torch.full((1, atoms, 50), -50.0)  # (1, a, bins)
    plddt_logits[0, :, 45] = 50.0  # expected pLDDT 0.91
    plddt_logits[0, ca[2:], 45] = -50.0
    plddt_logits[0, ca[2:], 15] = 50.0  # expected pLDDT 0.31
    true_coords = torch.zeros(1, atoms, 3)  # (samples, a, 3)
    true_coords[0, ca[2:]] = float("nan")
    rollout = SimpleNamespace(
        targets=[
            {
                "plddt_mask": torch.isfinite(true_coords[0]).all(-1),
                "plddt_score": torch.full((atoms,), 0.9),
                "plddt_target": torch.full((atoms,), 45),
                "pae_target": torch.zeros(tokens, tokens, dtype=torch.long),
                "pae_mask": torch.ones(tokens, tokens, dtype=torch.bool),
            }
        ],
        head_inputs={
            "atom_attention_mask": torch.ones(1, atoms),
            "asym_id": torch.zeros(1, tokens, dtype=torch.long),
            "token_attention_mask": torch.ones(1, tokens),
        },
        layout=SimpleNamespace(chain_ca=(ca,)),
        true_coords=true_coords,
    )
    summary = _summaries(plddt_logits, torch.zeros(1, tokens, tokens, 64), rollout, 0)
    record = {"target_id": "t", "stratum": "monomer_short", "num_chains": 1, "sample": 0, "true_lddt": 0.9, "tm_score": 0.9, "dockq": None}
    metrics = _metrics([{**record, "predictions": {"head": summary}}], "head")
    assert metrics["disorder_auroc"] == 1.0
    assert metrics["unresolved_residue_mean_plddt"] == pytest.approx(0.31, abs=1e-3)
    assert metrics["resolved_residue_mean_plddt"] == pytest.approx(0.91, abs=1e-3)
    assert metrics["unresolved_fraction_below_50"] == 1.0
    assert metrics["resolved_fraction_below_50"] == 0.0
    assert metrics["unresolved_residues"] == 2.0


def test_a_correctly_ranking_head_has_full_accuracy_and_no_regret():
    metrics = _metrics(_records("correct"), "head")
    assert metrics["within_target_plddt_accuracy"] == 1.0
    assert metrics["within_target_iptm_dockq_accuracy"] == 1.0
    assert metrics["top1_regret"] == 0.0
    assert metrics["random_selection_regret"] == pytest.approx(0.2)
    assert metrics["calibration_error_10bin"] == pytest.approx(0.0)


def test_a_reversed_head_picks_the_worst_sample():
    metrics = _metrics(_records("reversed"), "head")
    assert metrics["within_target_plddt_accuracy"] == 0.0
    assert metrics["top1_regret"] == pytest.approx(0.4)


def test_complexes_without_dockq_leave_selection_metrics_without_failing():
    records = _records("correct")
    for record in records:
        if record["target_id"] == "d1":
            record["dockq"] = None  # for example, a complex above 26 chains
    metrics = _metrics(records, "head")
    assert metrics["complexes_without_dockq"] == 1.0
    assert metrics["within_target_iptm_dockq_pairs"] == 0.0
    assert metrics["top1_regret"] == 0.0
    assert metrics["within_target_plddt_accuracy"] == 1.0


def test_bootstrap_keeps_repeated_targets_as_separate_groups():
    by_target = {"a": [{"target_id": "a", "sample": 0}], "b": [{"target_id": "b", "sample": 0}]}
    resampled = bootstrap_records(by_target, np.random.default_rng(3))
    assert len(resampled) == 2
    assert len({record["target_id"] for record in resampled}) == 2


def test_summary_reports_the_long_stratum_outside_the_overall_estimate():
    records = _records("correct")
    for record in records:
        if record["target_id"] == "m2":
            record["stratum"] = "long"
    summary = summarize(records, ["head"])["head"]
    assert summary["overall"]["targets"] == 2.0
    assert set(summary["by_stratum"]) == {"dimer_hetero", "long", "monomer_short"}
