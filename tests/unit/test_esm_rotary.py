"""Properties of the shared ESM rotary table used by ESM2, DPLM, DPLM2, and ESMFold."""

from __future__ import annotations

import pytest
import torch

from fastplms.models._esm_rotary import RotaryEmbedding


# Lengths straddle common SIMD widths so vectorized tail handling is exercised.
LENGTH_PAIRS = ((1, 2), (7, 8), (9, 17), (31, 64), (63, 65), (127, 513), (1000, 1024))


def _fresh_tables(
    head_dim: int, seq_len: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary = RotaryEmbedding(head_dim).to(dtype)
    probe = torch.zeros(1, 1, seq_len, head_dim, dtype=dtype)  # (b=1, h=1, l, d)
    return rotary._update_cos_sin_tables(probe, seq_dimension=-2)  # (1, 1, l, d) each


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16), ids=("fp32", "bf16"))
@pytest.mark.parametrize("head_dim", (16, 64))
@pytest.mark.parametrize(("short", "long"), LENGTH_PAIRS)
def test_rotary_table_prefix_is_independent_of_table_length(
    short: int, long: int, head_dim: int, dtype: torch.dtype
) -> None:
    """Row t depends only on t, so a longer table's prefix equals a shorter table.

    Reusing a longer cached table by slicing relies on this holding bitwise, not
    just to tolerance, so it is asserted with exact equality.
    """
    short_cos, short_sin = _fresh_tables(head_dim, short, dtype)
    long_cos, long_sin = _fresh_tables(head_dim, long, dtype)

    assert torch.equal(long_cos[:, :, :short, :], short_cos)
    assert torch.equal(long_sin[:, :, :short, :], short_sin)


def test_rotary_forward_matches_for_a_table_cached_at_a_longer_length() -> None:
    head_dim = 16
    query = torch.randn(2, 4, 9, head_dim)  # (b=2, h=4, l=9, d=16)
    key = torch.randn(2, 4, 9, head_dim)  # (b=2, h=4, l=9, d=16)

    cold = RotaryEmbedding(head_dim)
    expected_query, expected_key = cold(query, key)  # each (b=2, h=4, l=9, d=16)

    warm = RotaryEmbedding(head_dim)
    warm(torch.randn(1, 4, 40, head_dim), torch.randn(1, 4, 40, head_dim))
    actual_query, actual_key = warm(query, key)  # each (b=2, h=4, l=9, d=16)

    assert torch.equal(actual_query, expected_query)
    assert torch.equal(actual_key, expected_key)


def test_shorter_request_reuses_the_longer_cached_table() -> None:
    head_dim = 16
    rotary = RotaryEmbedding(head_dim)
    rotary._update_cos_sin_tables(torch.zeros(1, 1, 40, head_dim), seq_dimension=-2)
    long_cos = rotary._cos_cached  # (1, 1, l=40, d=16)

    cos, sin = rotary._update_cos_sin_tables(torch.zeros(1, 1, 9, head_dim), seq_dimension=-2)

    assert rotary._cos_cached is long_cos
    assert rotary._seq_len_cached == 40
    # Callers still receive exactly the requested number of rows.
    assert cos.shape == sin.shape == (1, 1, 9, head_dim)


def test_longer_request_rebuilds_the_table() -> None:
    head_dim = 16
    rotary = RotaryEmbedding(head_dim)
    rotary._update_cos_sin_tables(torch.zeros(1, 1, 9, head_dim), seq_dimension=-2)

    cos, _ = rotary._update_cos_sin_tables(  # each (1, 1, l=40, d=16)
        torch.zeros(1, 1, 40, head_dim), seq_dimension=-2
    )

    assert rotary._seq_len_cached == 40
    assert torch.equal(cos, _fresh_tables(head_dim, 40, torch.float32)[0])


def test_dtype_conversion_invalidates_a_longer_cached_table() -> None:
    """A length change after ``.to(dtype)`` has always rebuilt from the new ``inv_freq``."""
    head_dim = 16
    rotary = RotaryEmbedding(head_dim)
    rotary._update_cos_sin_tables(torch.zeros(1, 1, 40, head_dim), seq_dimension=-2)
    rotary = rotary.to(torch.bfloat16)

    probe = torch.zeros(1, 1, 9, head_dim, dtype=torch.bfloat16)  # (b=1, h=1, l=9, d=16)
    cos, sin = rotary._update_cos_sin_tables(probe, seq_dimension=-2)  # each (1, 1, l=9, d=16)

    expected_cos, expected_sin = _fresh_tables(head_dim, 9, torch.bfloat16)
    assert cos.dtype == torch.bfloat16
    assert torch.equal(cos, expected_cos)
    assert torch.equal(sin, expected_sin)


@pytest.mark.parametrize("seq_len", (9, 40), ids=("shorter", "same"))
@pytest.mark.parametrize("training", (True, False), ids=("train", "eval"))
def test_gradient_forward_rebuilds_inference_tables(seq_len: int, training: bool) -> None:
    head_dim = 16
    warm = RotaryEmbedding(head_dim).train(training)
    with torch.inference_mode():
        probe = torch.randn(1, 2, 40, head_dim)  # (b=1, h=2, l=40, d=16)
        warm(probe, probe)

    query = torch.randn(1, 2, seq_len, head_dim, requires_grad=True)  # (b=1, h=2, l, d=16)
    key = torch.randn_like(query, requires_grad=True)
    cold_query = query.detach().clone().requires_grad_()
    cold_key = key.detach().clone().requires_grad_()
    expected = RotaryEmbedding(head_dim)(cold_query, cold_key)
    actual = warm(query, key)

    sum(output.square().sum() for output in expected).backward()
    sum(output.square().sum() for output in actual).backward()

    for actual_output, expected_output in zip(actual, expected, strict=True):
        assert torch.equal(actual_output, expected_output)
    assert torch.equal(query.grad, cold_query.grad)
    assert torch.equal(key.grad, cold_key.grad)
