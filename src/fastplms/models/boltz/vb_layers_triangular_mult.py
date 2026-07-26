"""Incoming and outgoing triangular multiplicative pair updates."""

from __future__ import annotations

import importlib
import torch
from importlib.util import find_spec
from typing import Literal
from torch import Tensor, nn

from . import vb_layers_initialize as init


TriangleDirection = Literal["incoming", "outgoing"]


@torch.compiler.disable
def kernel_triangular_mult(
    x: Tensor,
    direction: TriangleDirection,
    mask: Tensor,
    norm_in_weight: Tensor,
    norm_in_bias: Tensor,
    p_in_weight: Tensor,
    g_in_weight: Tensor,
    norm_out_weight: Tensor,
    norm_out_bias: Tensor,
    p_out_weight: Tensor,
    g_out_weight: Tensor,
    eps: float,
) -> Tensor:
    """Dispatch the optional cuEquivariance triangle primitive lazily."""

    # x: (b, l, l, d); mask: (b, l, l); returned tensor: (b, l, l, d).
    if (
        find_spec("cuequivariance_torch") is None
        or find_spec("cuequivariance_ops_torch") is None
    ):
        raise RuntimeError(
            "Boltz2 use_kernels=True requires cuequivariance_torch and the CUDA 13 "
            "cuequivariance_ops_torch runtime from the 'structure,cueq' extras."
        )
    cueq = importlib.import_module("cuequivariance_torch")
    return cueq.triangle_multiplicative_update(
        x,
        direction=direction,
        mask=mask,
        norm_in_weight=norm_in_weight,
        norm_in_bias=norm_in_bias,
        p_in_weight=p_in_weight,
        g_in_weight=g_in_weight,
        norm_out_weight=norm_out_weight,
        norm_out_bias=norm_out_bias,
        p_out_weight=p_out_weight,
        g_out_weight=g_out_weight,
        eps=eps,
    )


class _TriangleMultiplication(nn.Module):
    direction: TriangleDirection
    equation: str

    def __init__(self, dim: int, direction: TriangleDirection, equation: str) -> None:
        super().__init__()
        self.direction = direction
        self.equation = equation
        self.norm_in = nn.LayerNorm(dim, eps=1e-5)
        self.p_in = nn.Linear(dim, 2 * dim, bias=False)
        self.g_in = nn.Linear(dim, 2 * dim, bias=False)
        self.norm_out = nn.LayerNorm(dim)
        self.p_out = nn.Linear(dim, dim, bias=False)
        self.g_out = nn.Linear(dim, dim, bias=False)

        init.bias_init_one_(self.norm_in.weight)
        init.bias_init_zero_(self.norm_in.bias)
        init.lecun_normal_init_(self.p_in.weight)
        init.gating_init_(self.g_in.weight)
        init.bias_init_one_(self.norm_out.weight)
        init.bias_init_zero_(self.norm_out.bias)
        init.final_init_(self.p_out.weight)
        init.gating_init_(self.g_out.weight)

    def _kernel_forward(self, pair_states: Tensor, mask: Tensor) -> Tensor:
        # pair_states: (b, l, l, d); mask: (b, l, l).
        return kernel_triangular_mult(
            pair_states,
            direction=self.direction,
            mask=mask,
            norm_in_weight=self.norm_in.weight,
            norm_in_bias=self.norm_in.bias,
            p_in_weight=self.p_in.weight,
            g_in_weight=self.g_in.weight,
            norm_out_weight=self.norm_out.weight,
            norm_out_bias=self.norm_out.bias,
            p_out_weight=self.p_out.weight,
            g_out_weight=self.g_out.weight,
            eps=1e-5,
        )  # (b, l, l, d)

    def forward(self, x: Tensor, mask: Tensor, use_kernels: bool = False) -> Tensor:
        """Transform pair tensor X with shape ``(b, l, l, d)``."""

        if use_kernels:
            return self._kernel_forward(x, mask)

        # X_norm is the normalized pair tensor used by the output gate.
        normalized = self.norm_in(x)  # (b, l, l, d)
        projected = (
            self.p_in(normalized) * self.g_in(normalized).sigmoid()
        )  # (b, l, l, 2 * d)
        projected = projected * mask.unsqueeze(-1)  # (b, l, l, 2 * d)
        left, right = torch.chunk(projected.float(), 2, dim=-1)  # each: (b, l, l, d)
        combined = torch.einsum(self.equation, left, right)  # (b, l, l, d)
        return (
            self.p_out(self.norm_out(combined)) * self.g_out(normalized).sigmoid()
        )  # (b, l, l, d)


class TriangleMultiplicationOutgoing(_TriangleMultiplication):
    """Aggregate pair paths that share their destination index."""

    def __init__(self, dim: int = 128) -> None:
        super().__init__(dim, direction="outgoing", equation="bikd,bjkd->bijd")


class TriangleMultiplicationIncoming(_TriangleMultiplication):
    """Aggregate pair paths that share their source index."""

    def __init__(self, dim: int = 128) -> None:
        super().__init__(dim, direction="incoming", equation="bkid,bkjd->bijd")
