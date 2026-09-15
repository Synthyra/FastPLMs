"""Unit tests for the native ESMC to ESM++ state conversion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from tools.conversion.esmc_native import (
    esmc_native_to_fastplms_v1,
    esmc_native_to_reference_v1,
    write_reference_snapshot,
)


def _native_state() -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {
        "esmc.embed_tokens.weight": torch.tensor([[1.0, 2.0]]),
        "esmc.norm.weight": torch.tensor([3.0, 4.0]),
        "lm_head.dense.weight": torch.tensor([[5.0, 6.0]]),
        "lm_head.dense.bias": torch.tensor([7.0]),
        "lm_head.layer_norm.weight": torch.tensor([8.0]),
        "lm_head.layer_norm.bias": torch.tensor([9.0]),
        "lm_head.decoder.weight": torch.tensor([[10.0, 11.0]]),
        "lm_head.decoder.bias": torch.tensor([12.0]),
    }
    prefix = "esmc.layers.0"
    state.update(
        {
            f"{prefix}.input_layernorm.weight": torch.tensor([13.0, 14.0]),
            f"{prefix}.input_layernorm.bias": torch.tensor([15.0, 16.0]),
            f"{prefix}.post_attention_layernorm.weight": torch.tensor([17.0, 18.0]),
            f"{prefix}.post_attention_layernorm.bias": torch.tensor([19.0, 20.0]),
            f"{prefix}.self_attn.q_norm.weight": torch.tensor([21.0, 22.0]),
            f"{prefix}.self_attn.k_norm.weight": torch.tensor([23.0, 24.0]),
            f"{prefix}.self_attn.o_proj.weight": torch.tensor([[25.0, 26.0]]),
            f"{prefix}.self_attn.q_proj.weight": torch.tensor([[27.0, 28.0]]),
            f"{prefix}.self_attn.k_proj.weight": torch.tensor([[29.0, 30.0]]),
            f"{prefix}.self_attn.v_proj.weight": torch.tensor([[31.0, 32.0]]),
            f"{prefix}.mlp.gate_proj.weight": torch.tensor([[33.0, 34.0]]),
            f"{prefix}.mlp.up_proj.weight": torch.tensor([[35.0, 36.0]]),
            f"{prefix}.mlp.down_proj.weight": torch.tensor([[37.0, 38.0]]),
        }
    )
    return state


def test_native_conversion_preserves_names_and_fuses_in_declared_order() -> None:
    converted = esmc_native_to_fastplms_v1(_native_state(), num_layers=1)

    assert torch.equal(
        converted["transformer.blocks.0.attn.layernorm_qkv.1.weight"],
        torch.tensor([[27.0, 28.0], [29.0, 30.0], [31.0, 32.0]]),
    )
    assert torch.equal(
        converted["transformer.blocks.0.ffn.1.weight"],
        torch.tensor([[33.0, 34.0], [35.0, 36.0]]),
    )
    assert torch.equal(converted["sequence_head.3.bias"], torch.tensor([12.0]))
    assert "transformer.blocks.0.attn.out_proj.weight" in converted


def test_native_conversion_rejects_unexpected_keys() -> None:
    state = _native_state()
    state["unexpected.weight"] = torch.ones(1)

    with pytest.raises(ValueError, match="Unrecognized native ESMC checkpoint keys"):
        esmc_native_to_fastplms_v1(state, num_layers=1)


def test_native_conversion_rejects_missing_keys() -> None:
    state = _native_state()
    del state["esmc.layers.0.self_attn.v_proj.weight"]

    with pytest.raises(KeyError):
        esmc_native_to_fastplms_v1(state, num_layers=1)


def test_native_reference_conversion_omits_sequence_head_and_uses_te_names() -> None:
    converted = esmc_native_to_reference_v1(_native_state(), num_layers=1)

    assert len(converted) == 12
    assert all(not name.startswith("sequence_head.") for name in converted)
    assert "transformer.blocks.0.attn.layernorm_qkv.layer_norm_weight" in converted
    assert "transformer.blocks.0.attn.layernorm_qkv.weight" in converted
    assert "transformer.blocks.0.ffn.fc1_weight" in converted
    assert "transformer.blocks.0.ffn.fc2_weight" in converted


def test_write_reference_snapshot_roundtrips_cpu_checkpoint(tmp_path: Path) -> None:
    native_snapshot = tmp_path / "native"
    output_snapshot = tmp_path / "reference"
    native_snapshot.mkdir()
    save_file(_native_state(), str(native_snapshot / "model.safetensors"))
    (native_snapshot / "config.json").write_text(
        '{"model_type":"esmc","d_model":2,"n_heads":1,"n_layers":1,"vocab_size":2}\n',
        encoding="utf-8",
    )

    report = write_reference_snapshot(native_snapshot, output_snapshot)

    assert report["status"] == "passed"
    assert report["tensor_count"] == 12
    config = json.loads((output_snapshot / "config.json").read_text(encoding="utf-8"))
    assert config == {
        "architectures": ["ESMCModel"],
        "attn_implementation": "sdpa",
        "d_model": 2,
        "mask_token_id": 32,
        "model_type": "esmc",
        "n_heads": 1,
        "n_layers": 1,
        "pad_token_id": 1,
        "vocab_size": 2,
    }
    state = load_file(str(output_snapshot / "model.safetensors"), device="cpu")
    assert all(not name.startswith("sequence_head.") for name in state)
