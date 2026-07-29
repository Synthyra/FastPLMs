"""CPU compile regressions for ESMFold2 atom metadata preparation."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from fastplms.models.esmfold2.modeling_esmfold2_common import (
    CHAR_VOCAB_SIZE,
    MAX_ATOMIC_NUMBER,
    MAX_CHARS,
    ESMFold2AtomEncoder,
)


def _atom_encoder_inputs() -> dict[str, torch.Tensor | int]:
    batch_size = 1
    n_atoms = 4
    num_diffusion_samples = 2
    return {
        "ref_pos": torch.randn(batch_size, n_atoms, 3),
        "atom_attention_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.float32),
        "ref_space_uid": torch.tensor([[0, 0, 1, 1]]),
        "ref_charge": torch.zeros(batch_size, n_atoms),
        "ref_element": torch.zeros(batch_size, n_atoms, MAX_ATOMIC_NUMBER),
        "ref_atom_name_chars": torch.zeros(
            batch_size,
            n_atoms,
            MAX_CHARS,
            CHAR_VOCAB_SIZE,
        ),
        "atom_to_token": torch.tensor([[0, 0, 2, 2]]),
        "r_l": torch.randn(batch_size * num_diffusion_samples, n_atoms, 3),
        "num_diffusion_samples": num_diffusion_samples,
    }


def test_atom_encoder_mask_metadata_preserves_compiled_outputs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    torch.manual_seed(2026)
    encoder = ESMFold2AtomEncoder(
        d_atom=8,
        d_token=6,
        n_blocks=0,
        n_heads=1,
    ).eval()
    inputs = _atom_encoder_inputs()

    with torch.no_grad():
        expected = encoder(**inputs)

    compiled_graphs: list[torch.fx.GraphModule] = []

    def counting_backend(
        graph_module: torch.fx.GraphModule,
        example_inputs: list[torch.Tensor],
    ) -> Callable[..., object]:
        del example_inputs
        compiled_graphs.append(graph_module)
        return graph_module.forward

    compiled_encoder = torch.compile(encoder, backend=counting_backend, dynamic=False)
    try:
        with torch.no_grad():
            actual = compiled_encoder(**inputs)
    finally:
        torch.compiler.reset()

    assert compiled_graphs
    for actual_tensor, expected_tensor in zip(actual[:3], expected[:3], strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor)

    actual_params = actual[3]
    expected_params = expected[3]
    torch.testing.assert_close(actual_params[0], expected_params[0])
    torch.testing.assert_close(actual_params[1], expected_params[1])
    torch.testing.assert_close(actual_params[2], expected_params[2])
    torch.testing.assert_close(actual_params[3], expected_params[3])
    assert actual_params[4] == expected_params[4] == 3
    assert actual_params[2].tolist() == [0, 1, 2, 4, 5, 6]
    assert actual_params[3].tolist() == [0, 3, 6]
    assert actual[0].shape == (2, 3, 6)
    torch.testing.assert_close(actual[0][:, 1], torch.zeros_like(actual[0][:, 1]))

    captured = capsys.readouterr()
    assert "Graph break from `Tensor.item()`" not in captured.err
