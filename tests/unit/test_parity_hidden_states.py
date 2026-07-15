"""Focused contracts for live parity output normalization."""

from types import SimpleNamespace

import torch

from tests.parity.test_model_parity import _hidden_state_tuple, _last_hidden, tensor_metrics


def test_live_parity_accepts_layer_stacked_hidden_states() -> None:
    """Official ESMC returns H stacked across the leading layer axis."""

    H = torch.arange(3 * 2 * 4 * 5).reshape(3, 2, 4, 5)
    output = SimpleNamespace(hidden_states=H, last_hidden_state=None)

    layers = _hidden_state_tuple(output)
    assert len(layers) == 3
    assert all(torch.equal(layer, H[index]) for index, layer in enumerate(layers))
    assert torch.equal(_last_hidden(output), H[-1])


def test_live_parity_pooling_ignores_nonfinite_padding() -> None:
    candidate = torch.tensor([[[1.0, 2.0], [float("nan"), float("nan")]]])
    official = torch.tensor([[[1.0, 2.0], [float("nan"), float("nan")]]])
    residue_mask = torch.tensor([[True, False]])

    metrics = tensor_metrics(candidate, official, residue_mask)

    assert metrics.relative_l2 == 0.0
    assert metrics.pooled_cosine_min > 0.999999
