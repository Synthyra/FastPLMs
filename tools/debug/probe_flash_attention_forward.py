"""Validate precompiled FlashAttention kernels on shared and model paths."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import torch
from torch.nn import functional as F

from fastplms.attention import FASTPLMS_ATTENTION_FUNCTIONS
from fastplms.attention._core import kernels_flash_attention_func
from fastplms.models.esm2.modeling_fastesm import FastEsmConfig, FastEsmModel
from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
    ESMplusplusConfig,
    ESMplusplusModel,
)

BACKENDS = ("flash_attention_2", "flash_attention_3")


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_float = actual.float()
    expected_float = expected.float()
    relative_l2 = torch.linalg.vector_norm(
        actual_float - expected_float
    ) / torch.linalg.vector_norm(expected_float).clamp_min(1e-12)
    cosine = F.cosine_similarity(
        actual_float.reshape(-1, actual.shape[-1]),
        expected_float.reshape(-1, expected.shape[-1]),
        dim=-1,
    )
    result = {
        "relative_l2": relative_l2.item(),
        "minimum_cosine": cosine.min().item(),
    }
    if result["relative_l2"] > 1e-2 or result["minimum_cosine"] < 0.999:
        raise RuntimeError(f"FlashAttention parity failed: {result}")
    return result


def _sdpa_reference(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        query_states.transpose(1, 2),
        key_states.transpose(1, 2),
        value_states.transpose(1, 2),
    ).transpose(1, 2)


def _shared_results() -> dict[str, Any]:
    torch.manual_seed(17)
    query_states = torch.randn(2, 17, 4, 16, device="cuda", dtype=torch.bfloat16)
    key_states = torch.randn_like(query_states)
    value_states = torch.randn_like(query_states)
    attention_mask = torch.tensor(
        [[1] * 17, [1] * 9 + [0] * 8],
        device="cuda",
        dtype=torch.bool,
    )
    dense_reference = _sdpa_reference(query_states, key_states, value_states)
    mixed_reference = torch.zeros_like(dense_reference)
    for batch_index, length in enumerate((17, 9)):
        mixed_reference[batch_index, :length] = _sdpa_reference(
            query_states[batch_index : batch_index + 1, :length],
            key_states[batch_index : batch_index + 1, :length],
            value_states[batch_index : batch_index + 1, :length],
        )[0]

    results: dict[str, Any] = {}
    module = SimpleNamespace(training=False, is_causal=False)
    for backend in BACKENDS:
        dense = kernels_flash_attention_func(
            query_states,
            key_states,
            value_states,
            implementation=backend,
        )
        mixed = kernels_flash_attention_func(
            query_states,
            key_states,
            value_states,
            attention_mask_2d=attention_mask,
            implementation=backend,
        )
        interface_output, interface_weights = FASTPLMS_ATTENTION_FUNCTIONS[backend](
            module,
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            attention_mask,
        )
        if interface_weights is not None or not torch.equal(interface_output, mixed):
            raise RuntimeError(f"{backend} AttentionInterface dispatch disagrees with core.")
        results[backend] = {
            "dense": _metrics(dense, dense_reference),
            "mixed_padding": _metrics(
                mixed[attention_mask],
                mixed_reference[attention_mask],
            ),
            "padding_is_zero": bool(
                torch.equal(
                    mixed[~attention_mask],
                    torch.zeros_like(mixed[~attention_mask]),
                )
            ),
        }
        if not results[backend]["padding_is_zero"]:
            raise RuntimeError(f"{backend} returned nonzero values at padded positions.")
    return results


def _model_specs() -> tuple[tuple[str, type[torch.nn.Module], object], ...]:
    esm_kwargs = {
        "vocab_size": 33,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 128,
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "max_position_embeddings": 64,
        "pad_token_id": 1,
        "mask_token_id": 32,
        "position_embedding_type": "rotary",
        "attn_backend": "eager",
    }
    return (
        ("esm2", FastEsmModel, FastEsmConfig(**esm_kwargs, token_dropout=False)),
        (
            "esm_plusplus",
            ESMplusplusModel,
            ESMplusplusConfig(
                vocab_size=33,
                hidden_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                dropout=0.0,
                attn_backend="eager",
                pad_token_id=1,
            ),
        ),
    )


def _last_hidden_state(output: object) -> torch.Tensor:
    value = getattr(output, "last_hidden_state", None)
    if not torch.is_tensor(value):
        raise TypeError("Model output omitted last_hidden_state.")
    return value


def _model_results() -> dict[str, Any]:
    input_ids = torch.tensor(
        [
            [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 2, 1, 1, 1, 1],
            [0, 20, 19, 18, 17, 16, 15, 14, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        device="cuda",
    )
    attention_mask = input_ids.ne(1)
    results: dict[str, Any] = {}
    for family, model_class, config in _model_specs():
        torch.manual_seed(23)
        model = model_class(config).eval().to(device="cuda", dtype=torch.bfloat16)
        outputs: dict[str, torch.Tensor] = {}
        with torch.inference_mode():
            for backend in ("eager", "sdpa", *BACKENDS):
                model.attn_backend = backend
                outputs[backend] = _last_hidden_state(
                    model(input_ids=input_ids, attention_mask=attention_mask)
                ).detach()
        valid = attention_mask
        family_results: dict[str, Any] = {}
        for backend in BACKENDS:
            family_results[backend] = {
                "vs_eager": _metrics(outputs[backend][valid], outputs["eager"][valid]),
                "vs_sdpa": _metrics(outputs[backend][valid], outputs["sdpa"][valid]),
                "finite": bool(torch.isfinite(outputs[backend]).all()),
            }
            if not family_results[backend]["finite"]:
                raise RuntimeError(f"{family} {backend} produced non-finite values.")
        results[family] = family_results
        del model
    return results


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FlashAttention validation.")
    result = {
        "device": torch.cuda.get_device_name(0),
        "shared": _shared_results(),
        "models": _model_results(),
        "torch": torch.__version__,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
