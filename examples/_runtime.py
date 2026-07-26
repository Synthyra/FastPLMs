"""Shared fail-closed execution arguments for runnable examples."""

from __future__ import annotations

import argparse
from typing import Any


def add_execution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cpu", help="cpu or cuda[:index]")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="Model parameter and compute dtype",
    )


def resolve_execution(device_name: str, dtype_name: str) -> tuple[Any, Any]:
    """Resolve a CPU/CUDA device and dtype before loading a checkpoint."""

    import torch

    try:
        device = torch.device(device_name)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"Invalid execution device {device_name!r}") from error
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"Only CPU and CUDA devices are supported, got {device.type!r}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device {device} was requested but CUDA is unavailable")
    dtype = torch.float32 if dtype_name == "float32" else torch.bfloat16
    return device, dtype
