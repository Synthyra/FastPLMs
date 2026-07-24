"""Fresh Transformers v5 loader regressions for non-persistent rotary buffers."""

from __future__ import annotations

import torch
from transformers import PreTrainedModel

from fastplms.models.e1.modeling_e1 import E1Config, E1ForMaskedLM
from fastplms.models.esm3.modeling_esm3 import FastESM3Config, FastESM3Model


def test_e1_fresh_pretrained_load_rebuilds_rotary_cache(tmp_path) -> None:
    """A fresh local v5 load produces finite E1 outputs on its first forward."""

    config = E1Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_num_sequences=4,
        max_num_positions_within_seq=16,
        max_num_positions_global=16,
        attn_backend="sdpa",
    )
    source = E1ForMaskedLM(config).eval()
    source.save_pretrained(tmp_path, safe_serialization=True)
    reloaded = E1ForMaskedLM.from_pretrained(
        tmp_path,
        local_files_only=True,
        attn_implementation="sdpa",
    ).eval()
    rotary = reloaded.model.layers[0].norm_attn_norm.self_attn.rotary_emb
    assert rotary.inv_freq.numel() == 0

    # input_ids: (1, 5)
    input_ids = torch.tensor([[1, 4, 5, 6, 2]])
    # position_ids: (b, l)
    position_ids = torch.arange(5).unsqueeze(0)
    output = reloaded(
        input_ids=input_ids,
        within_seq_position_ids=position_ids,
        global_position_ids=position_ids,
        sequence_ids=torch.zeros_like(input_ids),
    )

    assert torch.isfinite(output.last_hidden_state).all()
    assert torch.isfinite(output.logits).all()
    assert rotary.inv_freq.numel() == 4


def test_esm3_fresh_pretrained_load_rebuilds_rotary_frequency(tmp_path) -> None:
    """A fresh local v5 load cannot reuse a materialized garbage frequency."""

    config = FastESM3Config(
        hidden_size=16,
        num_attention_heads=2,
        num_vector_heads=4,
        num_hidden_layers=1,
        attn_backend="sdpa",
    )
    source = FastESM3Model(config).eval()
    PreTrainedModel.save_pretrained(source, tmp_path, safe_serialization=True)
    reloaded = FastESM3Model.from_pretrained(
        tmp_path,
        local_files_only=True,
        attn_implementation="sdpa",
    ).eval()
    rotary = reloaded.esm3.transformer.blocks[0].attn.rotary
    assert rotary._cos_cached is None
    assert rotary._sin_cached is None

    # input_ids: (1, 5)
    input_ids = torch.tensor([[0, 5, 6, 7, 2]])
    output = reloaded(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
    )

    expected_inv_freq = rotary._compute_inv_freq(torch.device("cpu"))
    assert torch.equal(rotary.inv_freq, expected_inv_freq)
    assert torch.isfinite(output.last_hidden_state).all()
    assert torch.isfinite(output.logits).all()
