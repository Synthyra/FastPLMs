"""Validate precompiled FlashAttention kernels on shared and model paths."""

from __future__ import annotations

import json
import torch
from types import SimpleNamespace
from typing import Any
from torch.nn import functional as F

from fastplms.attention import FASTPLMS_ATTENTION_FUNCTIONS
from fastplms.attention._core import kernels_flash_attention_func
from fastplms.models.esm2.modeling_fastesm import FastEsmConfig, FastEsmModel
from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
    ESMplusplusConfig,
    ESMplusplusModel,
)


BACKENDS = ("flash_attention_2", "flash_attention_3")
MODEL_BACKENDS = {
    "esm2": BACKENDS,
    "esm_plusplus": BACKENDS,
}


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    # actual, expected: (..., d)
    actual_float = actual.float()  # (..., d)
    expected_float = expected.float()  # (..., d)
    relative_l2 = torch.linalg.vector_norm(
        actual_float - expected_float
    ) / torch.linalg.vector_norm(expected_float).clamp_min(1e-12)  # ()
    cosine = F.cosine_similarity(
        actual_float.reshape(-1, actual.shape[-1]),  # (n, d)
        expected_float.reshape(-1, expected.shape[-1]),  # (n, d)
        dim=-1,
    )  # (n,)
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
    # query_states, key_states, value_states: (b, l, h, d_h)
    return F.scaled_dot_product_attention(
        query_states.transpose(1, 2),  # (b, h, l, d_h)
        key_states.transpose(1, 2),  # (b, h, l, d_h)
        value_states.transpose(1, 2),  # (b, h, l, d_h)
    ).transpose(1, 2)  # (b, l, h, d_h)


def _shared_results() -> dict[str, Any]:
    torch.manual_seed(17)
    query_states = torch.randn(  # (b=2, l=17, h=4, d_h=16)
        2, 17, 4, 16, device="cuda", dtype=torch.bfloat16
    )
    key_states = torch.randn_like(query_states)  # (b, l, h, d_h)
    value_states = torch.randn_like(query_states)  # (b, l, h, d_h)
    attention_mask = torch.tensor(
        [[1] * 17, [1] * 9 + [0] * 8],
        device="cuda",
        dtype=torch.bool,
    )  # (b, l)
    dense_reference = _sdpa_reference(  # (b, l, h, d_h)
        query_states, key_states, value_states
    )
    mixed_reference = torch.zeros_like(dense_reference)  # (b, l, h, d_h)
    for batch_index, length in enumerate((17, 9)):
        mixed_reference[batch_index, :length] = _sdpa_reference(  # (l_i, h, d_h)
            query_states[batch_index : batch_index + 1, :length],  # (1, l_i, h, d_h)
            key_states[batch_index : batch_index + 1, :length],  # (1, l_i, h, d_h)
            value_states[batch_index : batch_index + 1, :length],  # (1, l_i, h, d_h)
        )[0]  # (l_i, h, d_h)

    results: dict[str, Any] = {}
    module = SimpleNamespace(training=False, is_causal=False)
    for backend in BACKENDS:
        dense = kernels_flash_attention_func(  # (b, l, h, d_h)
            query_states,
            key_states,
            value_states,
            implementation=backend,
        )
        mixed = kernels_flash_attention_func(  # (b, l, h, d_h)
            query_states,
            key_states,
            value_states,
            attention_mask_2d=attention_mask,
            implementation=backend,
        )
        interface_output, interface_weights = FASTPLMS_ATTENTION_FUNCTIONS[backend](
            module,
            query_states.transpose(1, 2),  # (b, h, l, d_h)
            key_states.transpose(1, 2),  # (b, h, l, d_h)
            value_states.transpose(1, 2),  # (b, h, l, d_h)
            attention_mask,  # (b, l)
        )
        # interface_output: (b, l, h, d_h); interface_weights: None
        if interface_weights is not None or not torch.equal(interface_output, mixed):
            raise RuntimeError(f"{backend} AttentionInterface dispatch disagrees with core.")
        results[backend] = {
            "dense": _metrics(dense, dense_reference),
            "mixed_padding": _metrics(
                mixed[attention_mask],  # (n_valid, h, d_h)
                mixed_reference[attention_mask],  # (n_valid, h, d_h)
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
    return value  # (b, l, d)


def _model_results() -> dict[str, Any]:
    input_ids = torch.tensor(
        [
            [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 2, 1, 1, 1, 1],
            [0, 20, 19, 18, 17, 16, 15, 14, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        device="cuda",
    )  # (b=2, l=17)
    attention_mask = input_ids.ne(1)  # (b, l)
    results: dict[str, Any] = {}
    for family, model_class, config in _model_specs():
        torch.manual_seed(23)
        model = model_class(config).eval().to(device="cuda", dtype=torch.bfloat16)
        outputs: dict[str, torch.Tensor] = {}
        backends = MODEL_BACKENDS[family]
        with torch.inference_mode():
            for backend in ("eager", "sdpa", *backends):
                model.attn_backend = backend
                outputs[backend] = _last_hidden_state(  # (b, l, d)
                    model(input_ids=input_ids, attention_mask=attention_mask)
                ).detach()
        valid = attention_mask  # (b, l)
        family_results: dict[str, Any] = {}
        for backend in backends:
            family_results[backend] = {
                "vs_eager": _metrics(
                    outputs[backend][valid],  # (n_valid, d)
                    outputs["eager"][valid],  # (n_valid, d)
                ),
                "vs_sdpa": _metrics(
                    outputs[backend][valid],  # (n_valid, d)
                    outputs["sdpa"][valid],  # (n_valid, d)
                ),
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
