"""Shared semantic configuration extraction for native and candidate parity.

Keep this module independent of FastPLMs so the same extractor runs unchanged
inside isolated official-reference containers and candidate containers.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

SEMANTIC_PATHS: dict[str, tuple[str, ...]] = {
    "vocab_size": (
        "config.vocab_size",
        "vocab_size",
        "alphabet_size",
        "embed.num_embeddings",
        "embeddings.word_embeddings.num_embeddings",
        "encoder.sequence_embed.num_embeddings",
    ),
    "d_model": (
        "config.hidden_size",
        "config.d_model",
        "hidden_size",
        "d_model",
        "embed_dim",
        "embed.embedding_dim",
        "embeddings.word_embeddings.embedding_dim",
        "encoder.sequence_embed.embedding_dim",
    ),
    "n_layers": (
        "config.num_hidden_layers",
        "config.num_layers",
        "config.n_layers",
        "num_layers",
        "transformer.blocks",
        "layers",
        "encoder.layer",
        "encoder.block",
    ),
    "n_heads": (
        "config.num_attention_heads",
        "config.num_heads",
        "config.n_heads",
        "attention_heads",
        "transformer.blocks.0.attn.n_heads",
        "layers.0.self_attn.num_heads",
        "encoder.layer.0.attention.self.num_attention_heads",
    ),
    "d_ff": ("config.intermediate_size", "config.d_ff"),
    "layer_norm_epsilon": ("config.layer_norm_eps", "config.layer_norm_epsilon"),
    "max_positions": ("config.max_position_embeddings",),
    "relative_buckets": ("config.relative_attention_num_buckets",),
    "relative_max_distance": ("config.relative_attention_max_distance",),
    "pad_token_id": ("config.pad_token_id", "padding_idx"),
    "bos_token_id": ("config.bos_token_id", "cls_idx"),
    "eos_token_id": ("config.eos_token_id", "eos_idx"),
    "mask_token_id": ("config.mask_token_id", "mask_idx"),
    "token_dropout": ("config.token_dropout", "token_dropout"),
    "initializer_range": ("config.initializer_range",),
    "classifier_dropout": ("config.classifier_dropout",),
    "tie_word_embeddings": ("config.tie_word_embeddings",),
}


def _attribute(root: object, path: str) -> Any:
    current = root
    for part in path.split("."):
        if part.isdigit() and hasattr(current, "__len__") and hasattr(current, "__getitem__"):
            index = int(part)
            if index >= len(current):
                return None
            current = current[index]
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return None
    return current


def semantic_config(model: nn.Module) -> dict[str, Any]:
    """Extract the common inference semantics from an official or mirror model."""

    roots: list[object] = [model]
    if hasattr(model, "esm3"):
        roots.insert(0, model.esm3)
    result: dict[str, Any] = {}
    for semantic_name, paths in SEMANTIC_PATHS.items():
        for root in roots:
            for path in paths:
                value = _attribute(root, path)
                if value is None:
                    continue
                if isinstance(value, (nn.ModuleList, list, tuple)):
                    value = len(value)
                if torch.is_tensor(value) and value.numel() == 1:
                    value = value.item()
                if isinstance(value, (str, int, float, bool)):
                    result[semantic_name] = value
                    break
            if semantic_name in result:
                break
    missing = sorted({"vocab_size", "d_model", "n_layers", "n_heads"}.difference(result))
    if missing:
        raise RuntimeError(f"Could not extract required semantic configuration fields: {missing}")
    return result


def transformed_semantic_config(model: nn.Module, transform_name: str) -> dict[str, Any]:
    """Extract semantics after applying a declared checkpoint conversion."""

    result = semantic_config(model)
    if transform_name == "dplm_to_fastplms_v1":
        result["tie_word_embeddings"] = False
    return result


__all__ = ["SEMANTIC_PATHS", "semantic_config", "transformed_semantic_config"]
