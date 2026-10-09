"""The texture score and PAE losses of the cached confidence-head ablation."""

import pytest
import torch

from tools.confidence.cached_ablation import (
    PAE_CENTERS,
    ArmConfig,
    arm_config,
    expected_pae,
    high_frequency,
    learning_rate,
    pae_loss,
    roughness,
)


TOKENS = 24


def _labels(error: torch.Tensor) -> dict[str, torch.Tensor]:
    # error: (l, l) for l = TOKENS
    return {  # (...) pae_error, pae_target, pae_mask: each (l, l)
        "pae_error": error,
        "pae_target": (error * 2).long().clamp(0, 63),  # 0.5 Å bins
        "pae_mask": torch.ones(TOKENS, TOKENS, dtype=torch.bool),
    }


def _domains() -> torch.Tensor:
    """Two rigid domains: low error within each, high error between them."""
    error = torch.full((TOKENS, TOKENS), 12.0)
    error[:12, :12] = 1.0
    error[12:, 12:] = 1.5
    return error  # (l, l)


def test_high_frequency_vanishes_on_constant_map_and_ignores_masked_cells() -> None:
    values = torch.full((TOKENS, TOKENS), 7.0)
    mask = torch.ones(TOKENS, TOKENS, dtype=torch.bool)
    mask[3, 5] = False
    values[3, 5] = 1000.0  # a masked-out cell must not leak into its neighbours' blur
    assert high_frequency(values, mask)[mask].abs().max() < 1e-4


def test_roughness_separates_speckle_from_domain_blocks() -> None:
    mask = torch.ones(TOKENS, TOKENS, dtype=torch.bool)
    speckled = _domains() + 3.0 * torch.randn(TOKENS, TOKENS, generator=torch.Generator().manual_seed(0))
    assert roughness(_domains(), mask) < 0.2
    assert roughness(speckled, mask) > 2 * roughness(_domains(), mask)


def test_distillation_at_fraction_zero_is_the_hard_cross_entropy() -> None:
    logits = torch.randn(TOKENS, TOKENS, 64, generator=torch.Generator().manual_seed(1))
    teacher = torch.randn(TOKENS, TOKENS, 64, generator=torch.Generator().manual_seed(2))
    labels = _labels(_domains())
    loss, hard, texture = pae_loss(logits, labels, teacher, ArmConfig("esmfold2_300", "control"))
    assert torch.equal(loss, hard) and float(texture) == 0.0
    distilled, hard_again, _ = pae_loss(logits, labels, teacher, ArmConfig("esmfold2_300", "distill", distill_fraction=0.5))
    assert torch.equal(hard, hard_again) and not torch.equal(distilled, hard)


def test_texture_term_is_zero_when_prediction_matches_the_truth_texture() -> None:
    error = _domains()
    # Logits peaked on each cell's own bin reproduce the error map up to bin centering.
    bins = (error / 0.5).long().clamp(0, 63)
    logits = torch.full((TOKENS, TOKENS, 64), -30.0).scatter(-1, bins[..., None], 30.0)
    labels = _labels(PAE_CENTERS[bins])
    _, _, texture = pae_loss(logits, labels, logits, ArmConfig("esmfold2_300", "texture", texture_weight=1.0))
    assert float(texture) < 1e-4
    assert torch.allclose(expected_pae(logits), PAE_CENTERS[bins], atol=1e-4)


def test_arm_config_keeps_only_its_own_terms_and_rejects_unknown_arms() -> None:
    assert arm_config("esmfold2_300", "control", 3, 0.25, 0.5).texture_weight == 0.0
    assert arm_config("esmfold2_300", "distill", 3, 0.25, 0.5).texture_weight == 0.0
    assert arm_config("esmfold2_300", "distill_texture", 3, 0.25, 0.5).distill_fraction == 0.5
    with pytest.raises(ValueError):
        arm_config("esmfold2_300", "smoother", 3, 0.25, 0.5)


def test_run_name_marks_only_a_non_default_texture_weight() -> None:
    assert arm_config("esmfold2_300", "texture", 3, 0.25, 0.5).run_name == "texture"
    assert arm_config("esmfold2_300", "control", 3, 1.0, 0.5).run_name == "control"
    assert arm_config("esmfold2_300", "texture", 3, 1.0, 0.5).run_name == "texture-weight1"


def test_learning_rate_warms_up_then_decays_to_its_minimum() -> None:
    config = ArmConfig("esmfold2_300", "control")
    rates = [learning_rate(update, 100, config) for update in range(100)]
    assert rates[0] < rates[9] == pytest.approx(config.learning_rate)
    assert rates[-1] == pytest.approx(config.minimum_learning_rate, rel=0.05)
