"""ESM++ reads full-width rotary tables from its cache and changes no value."""

from __future__ import annotations

import pytest
import torch

from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
    RotaryEmbedding,
    apply_rotary_emb_torch,
)


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("rotary_dim", (8, 6))
def test_cached_full_tables_match_per_call_concatenation(
    dtype: torch.dtype, rotary_dim: int
) -> None:
    rotary = RotaryEmbedding(dim=rotary_dim)
    generator = torch.Generator().manual_seed(2)
    # A shorter batch after a longer one reads a prefix of the cached tables.
    for token_count in (9, 17, 5):
        queries = torch.randn(2, token_count, 3, 8, generator=generator).to(dtype)  # (b, l, h, d)
        keys = torch.randn(2, token_count, 3, 8, generator=generator).to(dtype)  # (b, l, h, d)

        rotated_queries, rotated_keys = rotary(queries, keys)  # each (b, l, h, d)

        assert rotary._cos_cached is not None and rotary._sin_cached is not None
        # The half-width tables keep their public shape: (l_cached, d_r / 2).
        assert rotary._cos_cached.shape == (rotary._seq_len_cached, rotary_dim // 2)
        for rotated, original in ((rotated_queries, queries), (rotated_keys, keys)):
            expected = apply_rotary_emb_torch(  # (b, l, h, d)
                original, rotary._cos_cached, rotary._sin_cached, rotary.interleaved
            )
            assert torch.equal(rotated, expected)
            assert rotated.dtype == dtype


def test_moving_the_module_clears_the_full_tables() -> None:
    rotary = RotaryEmbedding(dim=8)
    states = torch.randn(1, 4, 2, 8)  # (b, l, h, d)
    rotary(states, states)
    assert rotary._cos_full_cached is not None

    rotary.to(torch.float64)

    assert rotary._cos_full_cached is None
    assert rotary._sin_full_cached is None
