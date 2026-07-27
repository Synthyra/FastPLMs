"""CPU contracts for the exhaustive ESMC weight-geometry deep dive."""

from __future__ import annotations

import numpy as np
import pytest

from tools.analysis import esmc_weight_geometry_deep_dive as deep

torch = pytest.importorskip("torch")


def test_gini_and_entropy_effective_cover_uniform_and_concentrated_weights() -> None:
    assert deep._gini(np.ones(8)) == pytest.approx(0)
    entropy, effective, hhi = deep._entropy_effective(np.ones(8))
    assert entropy == pytest.approx(np.log(8))
    assert effective == pytest.approx(8)
    assert hhi == pytest.approx(1 / 8)

    concentrated = np.zeros(8)
    concentrated[3] = 1
    _, effective, hhi = deep._entropy_effective(concentrated)
    assert effective == pytest.approx(1)
    assert hhi == pytest.approx(1)


def test_spectrum_shape_energy_and_gap_metrics_are_exact() -> None:
    metrics = deep._spectrum_shape(np.asarray([4.0, 2.0, 1.0, 0.5]))
    assert metrics["energy_at_1"] == pytest.approx(16 / 21.25)
    assert metrics["energy_at_2"] == pytest.approx(20 / 21.25)
    assert metrics["s1_s2_ratio"] == pytest.approx(2)
    assert metrics["gap_ratio_1"] == pytest.approx(2)
    assert 1 <= metrics["log_spectrum_knee_rank"] <= 4


def test_randomized_svd_recovers_known_low_rank_matrix() -> None:
    generator = torch.Generator().manual_seed(41)
    left, _ = torch.linalg.qr(torch.randn(40, 5, generator=generator))
    right, _ = torch.linalg.qr(torch.randn(30, 5, generator=generator))
    singular = torch.tensor([9.0, 5.0, 3.0, 2.0, 1.0])
    matrix = (left * singular) @ right.T
    u, s, v, diagnostics = deep.randomized_svd(
        matrix,
        5,
        device=torch.device("cpu"),
        seed=17,
    )
    assert torch.allclose(s, singular, atol=2e-5)
    reconstructed = (u * s) @ v.T
    assert torch.allclose(reconstructed, matrix, atol=2e-5)
    assert diagnostics["captured_frobenius_energy_fraction"] == pytest.approx(1, abs=1e-6)


def test_subspace_rows_distinguish_identical_and_orthogonal_spaces() -> None:
    identity = np.eye(8)
    same = deep._subspace_row(identity[:, :4], identity[:, :4], (2, 4))
    orthogonal = deep._subspace_row(identity[:, :4], identity[:, 4:], (4,))
    assert same[1]["normalized_overlap"] == pytest.approx(1)
    assert same[1]["maximum_angle_degrees"] == pytest.approx(0)
    assert orthogonal[0]["normalized_overlap"] == pytest.approx(0)
    assert orthogonal[0]["minimum_angle_degrees"] == pytest.approx(90)


def test_local_robust_z_marks_isolated_peak() -> None:
    values = np.zeros(20)
    values[10] = 5
    assert deep._local_robust_z(values, 10) == float("inf")
    assert deep._local_robust_z(values, 9) == 0


def test_polynomial_detrending_removes_cubic_depth_trend() -> None:
    x = np.linspace(-1, 1, 80)
    values = 2 + 3 * x - 4 * x**2 + 2 * x**3
    trend, residual = deep._poly_detrend(values)
    assert np.max(np.abs(trend - values)) < 1e-10
    assert np.max(np.abs(residual)) < 1e-10
