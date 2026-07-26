"""Torch-only parameter initialization for the Boltz2 runtime."""

from __future__ import annotations

import math
import torch
from typing import Literal
from torch import Tensor


FanMode = Literal["fan_in", "fan_out", "fan_avg"]


def _calculate_fan(shape: torch.Size | tuple[int, ...], fan: FanMode = "fan_in") -> float:
    """Resolve the selected fan for a two-dimensional linear weight tensor."""

    if len(shape) != 2:
        raise ValueError(f"linear weights must be two-dimensional, received {tuple(shape)}")
    fan_out, fan_in = shape
    if fan == "fan_in":
        return float(fan_in)
    if fan == "fan_out":
        return float(fan_out)
    if fan == "fan_avg":
        return (fan_in + fan_out) / 2
    raise ValueError(f"invalid fan mode: {fan!r}")


def trunc_normal_init_(
    weights: Tensor,
    scale: float = 1.0,
    fan: FanMode = "fan_in",
) -> None:
    """Fill W from a normal distribution truncated at two standard deviations."""

    # weights: (d_out, d_in); initialization preserves this shape in place.
    variance = scale / max(1.0, _calculate_fan(weights.shape, fan))
    std = math.sqrt(variance)
    with torch.no_grad():
        torch.nn.init.trunc_normal_(weights, mean=0.0, std=std, a=-2 * std, b=2 * std)


def lecun_normal_init_(weights: Tensor) -> None:
    """Initialize W using fan-in-scaled truncated normal values."""

    # weights: (d_out, d_in), mutated in place.
    trunc_normal_init_(weights)


def he_normal_init_(weights: Tensor) -> None:
    """Initialize W using twice the fan-in variance."""

    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights: Tensor) -> None:
    """Initialize W with Xavier uniform values."""

    torch.nn.init.xavier_uniform_(weights, gain=1.0)


def _fill_(tensor: Tensor, value: float) -> None:
    # tensor: (...), mutated in place without changing shape.
    with torch.no_grad():
        tensor.fill_(value)


def final_init_(weights: Tensor) -> None:
    """Zero the final projection W."""

    _fill_(weights, 0.0)


def gating_init_(weights: Tensor) -> None:
    """Zero the gating projection W."""

    _fill_(weights, 0.0)


def bias_init_zero_(bias: Tensor) -> None:
    """Set the bias tensor to zero."""

    _fill_(bias, 0.0)


def bias_init_one_(bias: Tensor) -> None:
    """Set the bias tensor to one."""

    _fill_(bias, 1.0)


def normal_init_(weights: Tensor) -> None:
    """Initialize W with linear Kaiming-normal values."""

    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


def ipa_point_weights_init_(weights: Tensor) -> None:
    """Initialize W so applying softplus yields one."""

    _fill_(weights, 0.541324854612918)
    # weights: (d_out, d_in), mutated in place.
    # weights: (d_out, d_in), mutated in place.
    # weights: (...), mutated in place.
    # weights: (...), mutated in place.
    # bias: (...), mutated in place.
    # bias: (...), mutated in place.
    # weights: (d_out, d_in), mutated in place.
    # weights: (...), mutated in place.
