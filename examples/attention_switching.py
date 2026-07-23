#!/usr/bin/env python3
"""Run an optimized attention backend and inspect its per-call eager fallback."""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Any

FLASH_BACKENDS = frozenset({"flash_attention_2", "flash_attention_3"})
DTYPE_NAMES = ("float32", "bfloat16")


def configure_offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def attention_configuration_snapshot(model: Any) -> tuple[tuple[str, str, Any], ...]:
    """Capture model/config backend state so a per-call fallback cannot mutate it."""

    attributes = (
        "attn_backend",
        "attn_implementation",
        "_attn_implementation",
        "_attn_implementation_internal",
    )
    state: list[tuple[str, str, Any]] = []
    for owner, value in (("model", model), ("config", getattr(model, "config", None))):
        if value is None:
            continue
        for attribute in attributes:
            if hasattr(value, attribute):
                state.append((owner, attribute, getattr(value, attribute)))
    if not state:
        raise RuntimeError("The loaded model does not expose its configured attention backend")
    return tuple(state)


def run_optimized_attention_example(
    model: Any,
    tokenizer: Any,
    sequences: list[str],
) -> Any:
    """Execute the configured backend without requesting attention tensors."""

    import torch

    batch = tokenizer(sequences, padding=True, return_tensors="pt")
    batch = {name: tensor.to(model.device) for name, tensor in batch.items()}
    with torch.inference_mode():
        return model(**batch, output_attentions=False)


def run_attention_example(
    model: Any,
    tokenizer: Any,
    sequences: list[str],
) -> tuple[Any, list[str]]:
    import torch

    batch = tokenizer(sequences, padding=True, return_tensors="pt")
    batch = {name: tensor.to(model.device) for name, tensor in batch.items()}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        with torch.inference_mode():
            output = model(**batch, output_attentions=True)
    return output, [str(item.message) for item in caught]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--backend",
        choices=(
            "eager",
            "sdpa",
            "flex_attention",
            "flash_attention_2",
            "flash_attention_3",
        ),
        default="sdpa",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Execution device. Use cpu for the portable path or cuda[:index] for CUDA.",
    )
    parser.add_argument(
        "--dtype",
        choices=DTYPE_NAMES,
        default="float32",
        help="Model compute dtype. FlashAttention 2 and 3 require bfloat16.",
    )
    return parser


def resolve_execution(backend: str, device_name: str, dtype_name: str) -> tuple[Any, Any]:
    """Resolve and validate the documented backend, device, and dtype contract."""

    import torch

    try:
        device = torch.device(device_name)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"Invalid execution device {device_name!r}") from error
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(
            f"The attention example supports only CPU or CUDA devices, got {device.type!r}"
        )
    if dtype_name not in DTYPE_NAMES:
        raise ValueError(f"Unsupported compute dtype {dtype_name!r}")
    dtype = torch.float32 if dtype_name == "float32" else torch.bfloat16

    if backend in FLASH_BACKENDS:
        if device.type != "cuda":
            raise ValueError(f"{backend} requires a CUDA device; pass --device cuda[:index]")
        if dtype is not torch.bfloat16:
            raise ValueError(f"{backend} requires --dtype bfloat16")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"CUDA device {device} was requested but CUDA is unavailable")
        if backend in FLASH_BACKENDS and not torch.cuda.is_bf16_supported():
            raise ValueError(f"{backend} requires CUDA hardware with BF16 support")
    return device, dtype


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    artifact = arguments.artifact.expanduser().resolve()
    if not (artifact / "config.json").is_file():
        raise SystemExit(f"Not a local artifact: {artifact}")

    try:
        device, dtype = resolve_execution(
            arguments.backend,
            arguments.device,
            arguments.dtype,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    configure_offline()
    from transformers import AutoModel, AutoTokenizer

    from fastplms.attention import clear_flex_attention_caches

    model = (
        AutoModel.from_pretrained(
            artifact,
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation=arguments.backend,
            dtype=dtype,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        artifact,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.set_attn_implementation(arguments.backend)
    configured_state = attention_configuration_snapshot(model)
    optimized_output = run_optimized_attention_example(
        model,
        tokenizer,
        ["MSTNPKPQRKTKRNT", "MKTII"],
    )
    if attention_configuration_snapshot(model) != configured_state:
        raise RuntimeError("The optimized attention call mutated the configured backend")
    fallback_output, warning_messages = run_attention_example(
        model,
        tokenizer,
        ["MSTNPKPQRKTKRNT", "MKTII"],
    )
    if attention_configuration_snapshot(model) != configured_state:
        raise RuntimeError("The output_attentions eager fallback mutated the configured backend")
    print("optimized", tuple(optimized_output.last_hidden_state.shape))
    print("fallback", tuple(fallback_output.last_hidden_state.shape))
    print(
        "execution",
        f"backend={arguments.backend}",
        f"device={device}",
        f"dtype={arguments.dtype}",
    )
    for message in warning_messages:
        print("warning", message)

    clear_flex_attention_caches()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
