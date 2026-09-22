"""Online confidence training: schedule, EMA, checkpoint selection, ranking gradients, and sampling."""

import pytest
import torch

from tools.confidence.online_training import (
    ExponentialMovingAverage,
    OnlineTrainingConfig,
    TargetSampler,
    learning_rate,
    pairwise_accuracy,
    restore,
    selected_checkpoint,
    spearman,
)
from tools.confidence.ranking import ranking_pairs, sample_ranking_loss


def test_schedule_warms_up_then_decays_to_the_minimum_over_planned_updates():
    config = OnlineTrainingConfig(
        model_id="esmfold2_300",
        learning_rate=1e-4,
        minimum_learning_rate=1e-5,
        warmup_updates=10,
        planned_updates=110,
    )
    assert learning_rate(0, config) == pytest.approx(1e-5)
    assert learning_rate(9, config) == pytest.approx(1e-4)
    assert learning_rate(10, config) == pytest.approx(1e-4)
    assert learning_rate(60, config) == pytest.approx(5.5e-5)
    assert learning_rate(110, config) == pytest.approx(1e-5)
    assert learning_rate(500, config) == pytest.approx(1e-5)


def test_moving_average_tracks_weights_and_swaps_back():
    module = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        module.weight.fill_(1.0)
    ema = ExponentialMovingAverage(module, decay=0.9)
    with torch.no_grad():
        module.weight.fill_(2.0)
    ema.update(module)
    torch.testing.assert_close(ema.shadow["weight"], torch.full((1, 2), 1.1))

    replaced = ema.swapped_into(module)
    torch.testing.assert_close(module.weight.detach(), torch.full((1, 2), 1.1))
    restore(module, replaced)
    torch.testing.assert_close(module.weight.detach(), torch.full((1, 2), 2.0))


def test_selection_keeps_final_weights_unless_they_trail_the_best_by_over_two_percent():
    best = {"total_ce": 10.0}
    assert selected_checkpoint({"total_ce": 9.0}, best) == "final-ema"
    assert selected_checkpoint({"total_ce": 10.2}, best) == "final-ema"
    assert selected_checkpoint({"total_ce": 10.21}, best) == "best-ema"


def test_per_sample_ranking_terms_sum_to_the_joint_pairwise_gradient():
    generator = torch.Generator().manual_seed(0)
    weights = torch.randn(4, 3, generator=generator, requires_grad=True)
    features = torch.randn(4, 3, generator=generator)
    pairs = ranking_pairs([0.8, 0.5, 0.52, 0.9], margin=0.01)
    temperature = 0.5
    assert sorted(pairs) == [(0, 1), (0, 2), (2, 1), (3, 0), (3, 1), (3, 2)]

    scores = (weights * features).sum(-1)  # (samples,)
    joint = torch.stack(
        [torch.nn.functional.softplus(-(scores[better] - scores[worse]) / temperature) for better, worse in pairs]
    ).mean()
    (joint_gradient,) = torch.autograd.grad(joint, weights)

    detached = scores.detach()
    accumulated = torch.zeros_like(weights)
    summed = 0.0
    for sample in range(4):
        score = (weights[sample] * features[sample]).sum()
        loss = sample_ranking_loss(sample, score, detached, pairs, temperature)
        (gradient,) = torch.autograd.grad(loss, weights)
        accumulated += gradient
        summed += float(loss.detach())
    torch.testing.assert_close(accumulated, joint_gradient)
    assert summed == pytest.approx(2 * float(joint))


def test_ranking_loss_without_pairs_has_a_zero_gradient():
    weight = torch.tensor(2.0, requires_grad=True)
    loss = sample_ranking_loss(0, weight * 3.0, torch.tensor([6.0, 1.0]), [], temperature=0.05)
    loss.backward()
    assert float(loss) == 0.0
    assert weight.grad is not None and float(weight.grad) == 0.0


def test_pairwise_accuracy_skips_pairs_within_the_margin():
    assert pairwise_accuracy([0.1, 0.2, 0.3], [0.5, 0.51, 0.9], margin=0.02) == (2, 2)
    assert pairwise_accuracy([0.3, 0.2], [0.5, 0.9], margin=0.02) == (0, 1)


def test_spearman_uses_ranks_and_needs_three_values():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 400]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2], [1, 2]) != spearman([1, 2], [1, 2])  # NaN


def test_sampler_balances_chain_counts_then_follows_weights():
    targets = [
        {"target_id": "monomer-heavy", "num_chains": 1, "weight": 3.0},
        {"target_id": "monomer-light", "num_chains": 1, "weight": 1.0},
        {"target_id": "dimer", "num_chains": 2, "weight": 0.01},
    ]
    sampler = TargetSampler(targets, monomer_fraction=0.5, seed=0)
    draws = [sampler.draw()["target_id"] for _ in range(20_000)]
    assert draws.count("dimer") / len(draws) == pytest.approx(0.5, abs=0.02)
    assert draws.count("monomer-heavy") / len(draws) == pytest.approx(0.375, abs=0.02)
