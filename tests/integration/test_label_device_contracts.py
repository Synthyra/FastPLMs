"""Accelerator contracts for direct loss calls with host-resident labels."""

from __future__ import annotations

import pytest
import torch

from fastplms.models.e1.modeling_e1 import E1Config, E1ForTokenClassification
from fastplms.models.esm3.modeling_esm3 import FastESM3Config, FastESM3Model
from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
    ESMplusplusConfig,
    ESMplusplusForMaskedLM,
    ESMplusplusForTokenClassification,
)


def _esmc_config() -> ESMplusplusConfig:
    return ESMplusplusConfig(
        vocab_size=40,
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=1,
        num_labels=3,
        dropout=0.0,
        attn_backend="sdpa",
    )


def _e1_config() -> E1Config:
    return E1Config(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_num_sequences=4,
        max_num_positions_within_seq=32,
        max_num_positions_global=64,
        num_labels=3,
        attn_backend="sdpa",
        dtype="float32",
        use_cache=False,
    )


def _esm3_config() -> FastESM3Config:
    return FastESM3Config(
        hidden_size=8,
        num_attention_heads=2,
        num_vector_heads=2,
        num_hidden_layers=1,
        attn_backend="sdpa",
    )


@pytest.mark.gpu
@pytest.mark.parametrize("family", ("esmc_mlm", "esmc_token", "e1_token", "esm3"))
def test_advertised_heads_accept_host_labels_with_cuda_logits(family: str) -> None:
    assert torch.cuda.is_available(), "label device contracts require CUDA"
    device = torch.device("cuda")

    if family == "esmc_mlm":
        model = ESMplusplusForMaskedLM(_esmc_config()).to(device).train()
        # input_ids: (2, 4)
        input_ids = torch.tensor(((0, 4, 5, 2), (0, 6, 2, 1)), device=device)
        # attention_mask: (b, l)
        attention_mask = input_ids.ne(1)
        # labels: (b, l)
        labels = input_ids.cpu().masked_fill(~attention_mask.cpu(), -100)
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    elif family == "esmc_token":
        model = ESMplusplusForTokenClassification(_esmc_config()).to(device).train()
        # input_ids: (2, 4)
        input_ids = torch.tensor(((0, 4, 5, 2), (0, 6, 2, 1)), device=device)
        # attention_mask: (b, l)
        attention_mask = input_ids.ne(1)
        # labels: (b, l)
        labels = input_ids.cpu().remainder(3).masked_fill(~attention_mask.cpu(), -100)
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    elif family == "e1_token":
        model = E1ForTokenClassification(_e1_config()).to(device).train()
        # input_ids: (1, 4)
        input_ids = torch.tensor(((1, 5, 6, 2),), device=device)
        # positions: (..., 3)
        positions = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        labels = input_ids.cpu().remainder(3)
        output = model(
            input_ids=input_ids,
            within_seq_position_ids=positions,
            global_position_ids=positions,
            sequence_ids=torch.zeros_like(input_ids),
            labels=labels,
        )
    else:
        model = FastESM3Model(_esm3_config()).to(device).train()
        # input_ids: (2, 4)
        input_ids = torch.tensor(((0, 4, 5, 2), (0, 6, 2, 1)), device=device)
        # attention_mask: (b, l)
        attention_mask = input_ids.ne(1)
        # labels: (b, l)
        labels = input_ids.cpu().masked_fill(~attention_mask.cpu(), -100)
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    assert output.loss is not None
    assert output.loss.device.type == "cuda"
    assert torch.isfinite(output.loss)
    output.loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
