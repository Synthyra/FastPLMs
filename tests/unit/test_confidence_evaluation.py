"""CPU tests for frequency-baseline evaluation helpers."""

from __future__ import annotations

import math
import pytest

from tools.confidence.evaluation import _sample_ranking, build_frequency_baseline


def _target(plddt: list[int], pae: list[int]) -> dict[str, object]:
    return {
        "plddt_target": plddt,
        "pae_target": pae,
        "plddt_mask": [True] * len(plddt),
        "pae_mask": [True] * len(pae),
    }


def test_frequency_baseline_fits_training_only_and_normalizes_targets() -> None:
    result = build_frequency_baseline(
        [_target([1, 1], [2]), _target([3], [4, 4])], [_target([1], [2])]
    )
    assert result["training_target_count"] == 2
    assert result["evaluation_target_count"] == 1
    assert len(result["plddt_probabilities"]) == 50
    assert math.isfinite(result["plddt_ce"])
    assert result["plddt_probabilities"][1] == result["plddt_probabilities"][3]


def test_frequency_baseline_rejects_empty_or_invalid_targets() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        build_frequency_baseline([])
    with pytest.raises(ValueError, match="outside"):
        build_frequency_baseline([_target([50], [1])])


def test_frequency_baseline_accepts_single_pass_target_generators() -> None:
    training = (_target([index], [index]) for index in (1, 2, 3))
    evaluation = (_target([index], [index]) for index in (1, 2))
    result = build_frequency_baseline(training, evaluation)
    assert result["training_target_count"] == 3
    assert result["evaluation_target_count"] == 2


def test_sample_ranking_is_paired_by_target_and_seed() -> None:
    predictions = [
        {"target_id": "a", "seed": 17, "plddt_pred": [0.2], "plddt_true": [0.1]},
        {"target_id": "a", "seed": 29, "plddt_pred": [0.3], "plddt_true": [0.2]},
        {"target_id": "b", "seed": 17, "plddt_pred": [0.8], "plddt_true": [0.7]},
        {"target_id": "b", "seed": 29, "plddt_pred": [0.7], "plddt_true": [0.9]},
        {"target_id": "unpaired", "seed": 17, "plddt_pred": [0.1]},
    ]
    result = _sample_ranking(predictions)
    assert result["paired_target_count"] == 2
    assert result["seed_rank_spearman"] == pytest.approx(1.0)
    assert result["plddt_selection"]["selection_accuracy"] == pytest.approx(0.5)
    assert result["plddt_selection"]["mean_regret"] == pytest.approx(0.1)


def test_interface_selection_uses_only_complete_dimer_pairs() -> None:
    predictions = [
        {
            "target_id": "dimer",
            "seed": 17,
            "plddt_pred": [0.2],
            "plddt_true": [0.2],
            "iptm": 0.8,
            "dockq": 0.7,
        },
        {
            "target_id": "dimer",
            "seed": 29,
            "plddt_pred": [0.3],
            "plddt_true": [0.3],
            "iptm": 0.6,
            "dockq": 0.5,
        },
        {
            "target_id": "monomer",
            "seed": 17,
            "plddt_pred": [0.2],
            "plddt_true": [0.2],
            "iptm": None,
            "dockq": None,
        },
        {
            "target_id": "monomer",
            "seed": 29,
            "plddt_pred": [0.3],
            "plddt_true": [0.3],
            "iptm": None,
            "dockq": None,
        },
    ]
    result = _sample_ranking(predictions)
    assert result["interface_selection"]["paired_target_count"] == 1
    assert result["interface_selection"]["selection_accuracy"] == pytest.approx(1.0)
