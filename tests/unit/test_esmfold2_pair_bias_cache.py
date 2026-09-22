"""The ESMFold2 token transformer projects its pair bias once per sampling run.

Every denoising step conditions on the same pair tensor, so reusing the
projection must not change a single value.
"""

from __future__ import annotations

import pytest
import torch

from typing import Any

from fastplms.models.esmfold2 import modeling_esmfold2_common as common


NUM_BLOCKS = 3
NUM_STEPS = 4


def _token_transformer() -> common.DiffusionTransformer:
    torch.manual_seed(2)
    transformer = common.DiffusionTransformer(
        d_model=32, d_pair=8, num_heads=4, num_blocks=NUM_BLOCKS, d_cond=16
    ).eval()
    # The conditioned gates initialize closed, which would hide the attention output.
    for parameter in transformer.parameters():
        torch.nn.init.normal_(parameter, std=0.3)
    return transformer


def _step_inputs(step: int, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(100 + step)
    tokens = torch.randn(2 * samples, 7, 32, generator=generator)  # (b * samples, n, d_model)
    conditioning = torch.randn(2 * samples, 7, 16, generator=generator)  # (b * samples, n, d_cond)
    return tokens, conditioning


@pytest.mark.parametrize("samples", (1, 3))
@pytest.mark.parametrize("autocast", (False, True))
def test_cached_pair_bias_matches_per_step_projection_bitwise(samples: int, autocast: bool) -> None:
    transformer = _token_transformer()
    pair = torch.randn(2, 7, 7, 8, generator=torch.Generator().manual_seed(5))  # (b, n, n, d_pair)
    token_mask = torch.ones(2, 7, dtype=torch.bool)  # (b, n)
    token_mask[1, 5:] = False
    inference_cache: dict[str, Any] = {}

    for step in range(NUM_STEPS):
        tokens, conditioning = _step_inputs(step, samples)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            per_step, _ = transformer(
                tokens, conditioning, pair, attention_mask=token_mask, num_diffusion_samples=samples
            )
            cached, _ = transformer(
                tokens,
                conditioning,
                pair,
                attention_mask=token_mask,
                num_diffusion_samples=samples,
                inference_cache=inference_cache,
            )
        assert torch.equal(per_step, cached), step

    block_caches = inference_cache["token_pair_bias"]
    assert sorted(block_caches) == list(range(NUM_BLOCKS))
    for block_cache in block_caches.values():
        assert block_cache["pair_bias"].shape == (2 * samples, 7, 7, 4)


def test_pair_bias_is_projected_once_per_block(monkeypatch: pytest.MonkeyPatch) -> None:
    transformer = _token_transformer()
    pair = torch.randn(2, 7, 7, 8)  # (b, n, n, d_pair)
    projections: list[int] = []
    for block_index, block in enumerate(transformer.attn_blocks):
        block.pair_bias_proj.register_forward_hook(
            lambda _module, _inputs, _output, index=block_index: projections.append(index)
        )

    inference_cache: dict[str, Any] = {}
    for step in range(NUM_STEPS):
        tokens, conditioning = _step_inputs(step, samples=1)
        with torch.no_grad():
            transformer(tokens, conditioning, pair, inference_cache=inference_cache)
    assert projections == list(range(NUM_BLOCKS))

    projections.clear()
    with torch.no_grad():
        transformer(tokens, conditioning, pair)
        transformer(tokens, conditioning, pair)
    assert projections == 2 * list(range(NUM_BLOCKS))


def test_precomputed_three_dimensional_bias_is_never_cached() -> None:
    transformer = _token_transformer()
    bias = torch.randn(2, 7, 7)  # (b, n, n), already a per-pair logit
    tokens, conditioning = _step_inputs(0, samples=1)
    inference_cache: dict[str, Any] = {}
    with torch.no_grad():
        cached, _ = transformer(tokens, conditioning, bias, inference_cache=inference_cache)
        uncached, _ = transformer(tokens, conditioning, bias)
    assert torch.equal(cached, uncached)
    assert all(not block_cache for block_cache in inference_cache["token_pair_bias"].values())
