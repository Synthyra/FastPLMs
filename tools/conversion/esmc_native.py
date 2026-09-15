"""Convert native ESMC snapshots to FastPLMs and reference layouts."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch
from collections.abc import Mapping
from safetensors.torch import load_file, save_file


def esmc_native_to_fastplms_v1(
    state: Mapping[str, torch.Tensor], num_layers: int
) -> dict[str, torch.Tensor]:
    """Fuse native Q/K/V and gate/up weights without changing their values."""

    remaining = dict(state)  # Each tensor retains its checkpoint dimensions.
    converted: dict[str, torch.Tensor] = {}

    def move(source: str, target: str) -> None:
        converted[target] = remaining.pop(source)  # Shape is unchanged.

    move("esmc.embed_tokens.weight", "embed.weight")  # (vocabulary, channel)
    move("esmc.norm.weight", "transformer.norm.weight")  # (channel,)
    for index in range(num_layers):
        source = f"esmc.layers.{index}"
        target = f"transformer.blocks.{index}"
        for suffix in ("weight", "bias"):
            move(f"{source}.input_layernorm.{suffix}", f"{target}.attn.layernorm_qkv.0.{suffix}")
            move(f"{source}.post_attention_layernorm.{suffix}", f"{target}.ffn.0.{suffix}")
        for native, fast in (("q_norm", "q_ln"), ("k_norm", "k_ln"), ("o_proj", "out_proj")):
            move(f"{source}.self_attn.{native}.weight", f"{target}.attn.{fast}.weight")
        converted[f"{target}.attn.layernorm_qkv.1.weight"] = torch.cat(
            [remaining.pop(f"{source}.self_attn.{name}_proj.weight") for name in ("q", "k", "v")],
            dim=0,
        )  # (3 * channel, channel), in Q, K, V order.
        converted[f"{target}.ffn.1.weight"] = torch.cat(
            [remaining.pop(f"{source}.mlp.{name}_proj.weight") for name in ("gate", "up")],
            dim=0,
        )  # (2 * intermediate, channel), SiLU gate followed by values.
        move(f"{source}.mlp.down_proj.weight", f"{target}.ffn.3.weight")
    for native, fast in (("dense", "0"), ("layer_norm", "2"), ("decoder", "3")):
        for suffix in ("weight", "bias"):
            move(f"lm_head.{native}.{suffix}", f"sequence_head.{fast}.{suffix}")
    if remaining:
        raise ValueError(f"Unrecognized native ESMC checkpoint keys: {sorted(remaining)}")
    return converted


def esmc_native_to_reference_v1(
    state: Mapping[str, torch.Tensor], num_layers: int
) -> dict[str, torch.Tensor]:
    """Convert native ESMC weights to the pinned reference model layout.

    The reference model uses the canonical fused Transformer Engine names even
    when its runtime selects the pure-PyTorch implementation. Its backbone
    excludes the masked-language-model sequence head.
    """

    fast_state = esmc_native_to_fastplms_v1(state, num_layers)
    converted: dict[str, torch.Tensor] = {}
    suffixes = {
        ".attn.layernorm_qkv.0.weight": ".attn.layernorm_qkv.layer_norm_weight",
        ".attn.layernorm_qkv.0.bias": ".attn.layernorm_qkv.layer_norm_bias",
        ".attn.layernorm_qkv.1.weight": ".attn.layernorm_qkv.weight",
        ".ffn.0.weight": ".ffn.layer_norm_weight",
        ".ffn.0.bias": ".ffn.layer_norm_bias",
        ".ffn.1.weight": ".ffn.fc1_weight",
        ".ffn.3.weight": ".ffn.fc2_weight",
    }
    for name, tensor in fast_state.items():
        if name.startswith("sequence_head."):
            continue
        reference_name = name
        for source_suffix, target_suffix in suffixes.items():
            if name.endswith(source_suffix):
                reference_name = name[: -len(source_suffix)] + target_suffix
                break
        if reference_name in converted:
            raise ValueError(f"Duplicate reference ESMC key after conversion: {reference_name}")
        converted[reference_name] = tensor
    expected_count = 2 + 10 * num_layers
    if len(converted) != expected_count:
        raise ValueError(
            f"Reference ESMC conversion emitted {len(converted)} tensors; "
            f"expected {expected_count} for {num_layers} layers."
        )
    return converted


def _positive_int(config: Mapping[str, object], *names: str) -> int:
    for name in names:
        value = config.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    raise ValueError(f"Native ESMC config omits a positive integer among {names!r}.")


def _reference_config(native_config: Mapping[str, object]) -> dict[str, object]:
    """Build the minimal config consumed by pinned ``ESMCModel``."""

    return {
        "model_type": "esmc",
        "architectures": ["ESMCModel"],
        "d_model": _positive_int(native_config, "d_model", "hidden_size"),
        "n_heads": _positive_int(native_config, "n_heads", "num_attention_heads"),
        "n_layers": _positive_int(native_config, "n_layers", "num_hidden_layers"),
        "vocab_size": _positive_int(native_config, "vocab_size"),
        "pad_token_id": int(native_config.get("pad_token_id", 1)),
        "mask_token_id": int(native_config.get("mask_token_id", 32)),
        "attn_implementation": "sdpa",
    }


def write_reference_snapshot(native_snapshot: Path, output_snapshot: Path) -> dict[str, object]:
    """Convert one local native snapshot and write a reference snapshot."""

    native_snapshot = native_snapshot.resolve()
    output_snapshot = output_snapshot.resolve()
    config_path = native_snapshot / "config.json"
    weight_path = native_snapshot / "model.safetensors"
    if not config_path.is_file() or not weight_path.is_file():
        raise FileNotFoundError(
            f"Native ESMC snapshot requires config.json and model.safetensors: {native_snapshot}"
        )
    native_config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(native_config, dict):
        raise ValueError(f"Native ESMC config must be a JSON object: {config_path}")
    num_layers = _positive_int(native_config, "n_layers", "num_hidden_layers")
    native_state = load_file(str(weight_path), device="cpu")
    reference_state = esmc_native_to_reference_v1(native_state, num_layers)
    config = _reference_config(native_config)
    output_snapshot.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=output_snapshot,
        prefix=".model.",
        suffix=".safetensors.tmp",
    )
    os.close(handle)
    try:
        save_file(reference_state, temporary_name)
        os.replace(temporary_name, output_snapshot / "model.safetensors")
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    config_text = json.dumps(config, indent=2, sort_keys=True) + "\n"
    config_temporary = output_snapshot / ".config.json.tmp"
    config_temporary.write_text(config_text, encoding="utf-8", newline="\n")
    os.replace(config_temporary, output_snapshot / "config.json")
    return {
        "status": "passed",
        "native_snapshot": str(native_snapshot),
        "reference_snapshot": str(output_snapshot),
        "num_layers": num_layers,
        "tensor_count": len(reference_state),
        "sequence_head_omitted": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    report = write_reference_snapshot(arguments.native_snapshot, arguments.output)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
