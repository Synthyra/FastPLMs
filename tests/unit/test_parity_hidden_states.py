"""Focused contracts for live parity output normalization."""

from types import SimpleNamespace

import torch

from tests.parity.test_model_parity import _hidden_state_tuple, _last_hidden


def test_live_parity_accepts_layer_stacked_hidden_states() -> None:
    """Official ESMC returns H stacked across the leading layer axis."""

    H = torch.arange(3 * 2 * 4 * 5).reshape(3, 2, 4, 5)
    output = SimpleNamespace(hidden_states=H, last_hidden_state=None)

    layers = _hidden_state_tuple(output)
    assert len(layers) == 3
    assert all(torch.equal(layer, H[index]) for index, layer in enumerate(layers))
    assert torch.equal(_last_hidden(output), H[-1])
