"""Check confidence-only state changes and exact frozen tensor preservation."""

import pytest
import torch

from tools.confidence.packaging import merge_head, verify_folding_state


def test_packaging_replaces_only_confidence_state():
    base = {"fold.weight": torch.tensor([1.0, -0.0]), "confidence_head.old": torch.ones(2)}
    merged = merge_head(base, {"new": torch.zeros(2)})
    assert set(merged) == {"fold.weight", "confidence_head.new"}
    assert verify_folding_state(base, merged) == 1
    merged["fold.weight"] = torch.tensor([1.0, 0.0])  # (2,)
    with pytest.raises(ValueError, match="tensor bytes"):
        verify_folding_state(base, merged)


@pytest.mark.parametrize(
    "head", [{}, {"confidence_head.x": torch.ones(1)}, {"x": torch.tensor([float("nan")])}]
)
def test_packaging_rejects_invalid_head(head):
    with pytest.raises(ValueError):
        merge_head({"fold.weight": torch.ones(1)}, head)


def test_packaging_rejects_changed_folding_schema():
    base = {"fold.weight": torch.ones(1)}
    with pytest.raises(ValueError, match="keys"):
        verify_folding_state(base, {})
    with pytest.raises(ValueError, match="schema"):
        verify_folding_state(base, {"fold.weight": torch.ones(1, dtype=torch.bfloat16)})
