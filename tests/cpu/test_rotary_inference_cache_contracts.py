"""A rotary table built under inference mode must not reach a gradient forward, in eval mode too.

The design loop of the protein design engine holds its models in ``eval()`` and differentiates through them, yet its
critic stage scores sequences under ``torch.inference_mode()`` first. The table that stage caches is an inference
tensor, and autograd refuses to save one: the gradient forward failed with "Inference tensors cannot be saved for
backward" because the cache was rebuilt only for a module in training mode.
"""

import pytest
import torch

from fastplms.models.esm3.modeling_esm3 import RotaryEmbedding as Esm3RotaryEmbedding
from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
    RotaryEmbedding as EsmPlusPlusRotaryEmbedding,
)


HEAD_DIM = 16
TOKEN_COUNT = 6
HEADS = 2

ROTARY_CLASSES = [
    pytest.param(EsmPlusPlusRotaryEmbedding, id="esm_plusplus"),
    pytest.param(Esm3RotaryEmbedding, id="esm3"),
]


def _queries_and_keys() -> tuple[torch.Tensor, torch.Tensor]:
    query = torch.randn(1, TOKEN_COUNT, HEADS, HEAD_DIM, requires_grad=True)  # (b, l, h, d)
    key = torch.randn(1, TOKEN_COUNT, HEADS, HEAD_DIM, requires_grad=True)  # (b, l, h, d)
    return query, key


@pytest.mark.parametrize("rotary_class", ROTARY_CLASSES)
def test_an_eval_module_differentiates_after_an_inference_forward(rotary_class: type[torch.nn.Module]) -> None:
    rotary = rotary_class(HEAD_DIM).eval()
    query, key = _queries_and_keys()

    with torch.inference_mode():
        rotary(query.detach(), key.detach())

    rotated_query, rotated_key = rotary(query, key)
    (rotated_query.sum() + rotated_key.sum()).backward()

    assert query.grad is not None and key.grad is not None


@pytest.mark.parametrize("rotary_class", ROTARY_CLASSES)
def test_the_inference_cache_serves_a_later_inference_forward(rotary_class: type[torch.nn.Module]) -> None:
    rotary = rotary_class(HEAD_DIM).eval()
    query, key = _queries_and_keys()

    with torch.inference_mode():
        first_query, first_key = rotary(query.detach(), key.detach())
        second_query, second_key = rotary(query.detach(), key.detach())

    assert torch.equal(first_query, second_query)
    assert torch.equal(first_key, second_key)
