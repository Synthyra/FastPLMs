"""CPU contract checks for the native experimental confidence head."""

from __future__ import annotations

import torch

from fastplms.models.esmfold2.configuration_esmfold2 import ESMFold2Config
from fastplms.models.esmfold2.modeling_esmfold2_experimental import ConfidenceHead
from tools.confidence.labels import confidence_loss, compute_targets


def _config() -> ESMFold2Config:
    return ESMFold2Config(
        type="experimental",
        d_single=8,
        d_pair=8,
        inputs={"d_inputs": 4},
        confidence_head={
            "enabled": True,
            "num_plddt_bins": 50,
            "num_pae_bins": 64,
            "distogram_bins": 8,
            "folding_trunk": {"n_layers": 1, "n_heads": 2, "dropout": 0.0},
        },
    )


def _head_inputs() -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, ...]]:
    torch.manual_seed(11)
    length, atoms, d_inputs, d_pair = 3, 9, 4, 8
    s_inputs = torch.randn(1, length, d_inputs)
    z = torch.randn(1, length, length, d_pair)
    x_pred = torch.randn(1, atoms, 3)
    atom_to_token = torch.arange(length).repeat_interleave(3).reshape(1, atoms)
    return (
        {
            "s_inputs": s_inputs,
            "z": z,
            "x_pred": x_pred,
            "distogram_atom_idx": torch.tensor([[1, 4, 7]]),
            "token_attention_mask": torch.ones(1, length, dtype=torch.bool),
            "atom_to_token": atom_to_token,
            "atom_attention_mask": torch.ones(1, atoms, dtype=torch.bool),
            "asym_id": torch.zeros(1, length, dtype=torch.long),
            "mol_type": torch.zeros(1, length, dtype=torch.long),
            "num_diffusion_samples": 1,
            "relative_position_encoding": torch.zeros(1, length, length, d_pair),
            "token_bonds_encoding": torch.zeros(1, length, length, d_pair),
        },
        (
            x_pred[0].detach(),
            x_pred[0].detach(),
            torch.ones(atoms, dtype=torch.bool),
            atom_to_token[0],
            torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
            torch.ones(length, dtype=torch.bool),
        ),
    )


def test_native_head_outputs_match_label_contract_and_backpropagates() -> None:
    head = ConfidenceHead(_config()).train()
    inputs, geometry = _head_inputs()
    outputs = head(**inputs)
    targets = compute_targets(*geometry)
    losses = confidence_loss(outputs, targets)
    assert outputs["plddt_logits"].shape == (1, 9, 50)
    assert outputs["pae_logits"].shape == (1, 3, 3, 64)
    assert losses["total"].dtype == torch.float32
    losses["total"].backward()
    assert head.plddt_weight.grad is not None
    assert head.pae_head.weight.grad is not None


def test_bfloat16_logits_are_scored_in_float32() -> None:
    inputs, geometry = _head_inputs()
    outputs = ConfidenceHead(_config())(**inputs)
    targets = compute_targets(*geometry)
    bf16_logits = {
        name: value.detach().to(torch.bfloat16).requires_grad_()
        for name, value in outputs.items()
        if name.endswith("logits")
    }
    losses = confidence_loss(bf16_logits, targets)
    losses["total"].backward()
    assert losses["total"].dtype == torch.float32
    assert bf16_logits["plddt_logits"].grad is not None
    assert bf16_logits["pae_logits"].grad is not None
