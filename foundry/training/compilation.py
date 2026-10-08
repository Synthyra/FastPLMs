"""When a training run compiles its model.

A compiled graph needs a stable shape and a device the compiler can trace, so the trainer asks here
whether to compile. `auto` compiles on CUDA and nowhere else; `off` never compiles.
"""

from __future__ import annotations

import torch

from enum import StrEnum


class CompileMode(StrEnum):
    OFF = "off"
    AUTO = "auto"


def normalize_compile_mode(value: str) -> CompileMode:
    """The `CompileMode` named by `value`, ignoring case and surrounding space."""
    normalized: str = value.strip().lower()
    valid_values: tuple[str, ...] = tuple(mode.value for mode in CompileMode)
    assert normalized in valid_values, f"Unsupported compile_mode '{value}'. Expected one of {valid_values}."
    return CompileMode(normalized)


def compile_enabled_for_environment(device: torch.device | None, mode: CompileMode) -> tuple[bool, str]:
    """Whether to compile on `device`, and the reason when not."""
    if mode == CompileMode.OFF:
        return False, "compile_mode=off"
    if device is None:
        return True, ""
    if device.type != "cuda":
        return False, f"compile support is only enabled on CUDA devices, got '{device.type}'"
    return True, ""
