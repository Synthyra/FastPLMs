"""Attention primitives for the Boltz pair stack.

The module retains the converted checkpoint names while providing a compact
implementation around PyTorch operations.  Optional triangle kernels are
loaded only at the call site, so importing FastPLMs does not initialize CUDA.

Portions of the numerical contract follow OpenFold and DeepMind AlphaFold
under Apache-2.0.  See ``THIRD_PARTY_NOTICES.md`` for provenance.
"""

from __future__ import annotations

import importlib
import math
import torch
from collections.abc import Callable, Sequence
from importlib.util import find_spec
from torch import nn

from . import vb_layers_initialize as initialize
from .vb_tri_attn_utils import flatten_final_dims, permute_final_dims


def _initialize_weight(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    method: str,
) -> None:
    # weight: (d_out, d_in); bias: (d_out,) or None; both retain shape in place.
    initializers = {
        "default": initialize.lecun_normal_init_,
        "relu": initialize.he_normal_init_,
        "glorot": initialize.glorot_uniform_init_,
        "normal": initialize.normal_init_,
        "final": initialize.final_init_,
        "gating": initialize.gating_init_,
    }
    try:
        initializers[method](weight)
    except KeyError as error:
        raise ValueError(f"unknown linear initialization {method!r}") from error
    if method == "gating" and bias is not None:
        bias.fill_(1.0)


class Linear(nn.Linear):
    """Linear projection with the initializers used by converted checkpoints."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Callable[[torch.Tensor, torch.Tensor | None], None] | None = None,
        precision: torch.dtype | None = None,
    ) -> None:
        super().__init__(in_dim, out_dim, bias=bias)
        with torch.no_grad():
            if self.bias is not None:
                self.bias.zero_()
            if init_fn is None:
                _initialize_weight(self.weight, self.bias, init)
            else:
                init_fn(self.weight, self.bias)
        self.precision = precision

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Project ``input`` while preserving explicit precision policies."""

        # input: (..., d_in); every branch returns (..., d_out).
        input_dtype = input.dtype
        if self.precision is not None:
            with torch.autocast("cuda", enabled=False):
                bias = (
                    None if self.bias is None else self.bias.to(self.precision)
                )  # (d_out,) or None
                projected = nn.functional.linear(
                    input.to(self.precision),
                    self.weight.to(self.precision),
                    bias,
                )  # (..., d_out)
            return projected.to(input_dtype)  # (..., d_out)
        if input_dtype is torch.bfloat16:
            with torch.autocast("cuda", enabled=False):
                bias = (
                    None if self.bias is None else self.bias.to(input_dtype)
                )  # (d_out,) or None
                return nn.functional.linear(
                    input, self.weight.to(input_dtype), bias
                )  # (..., d_out)
        return nn.functional.linear(input, self.weight, self.bias)  # (..., d_out)


class LayerNorm(nn.Module):
    """Layer normalization that avoids an autocast upcast for BF16 inputs."""

    def __init__(self, c_in: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.c_in = (c_in,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(c_in))  # (c_in,)
        self.bias = nn.Parameter(torch.zeros(c_in))  # (c_in,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., c_in); normalization preserves shape.
        if x.dtype is torch.bfloat16:
            with torch.autocast("cuda", enabled=False):
                return nn.functional.layer_norm(
                    x,
                    self.c_in,
                    self.weight.to(x.dtype),
                    self.bias.to(x.dtype),
                    self.eps,
                )
        return nn.functional.layer_norm(x, self.c_in, self.weight, self.bias, self.eps)


@torch.jit.ignore
def softmax_no_cast(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Apply softmax without promoting BF16 inputs under CUDA autocast."""

    if tensor.dtype is torch.bfloat16:
        with torch.autocast("cuda", enabled=False):
            return nn.functional.softmax(tensor, dim=dim)
    return nn.functional.softmax(tensor, dim=dim)


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    biases: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Compute biased scaled attention after query scaling."""

    # query/key/value: (..., h, n_q_or_k, d_h).
    scores = torch.matmul(
        query, permute_final_dims(key, (1, 0))
    )  # (..., h, n_q, n_k)
    for bias in biases:
        scores += bias  # (..., h, n_q, n_k)
    probabilities = softmax_no_cast(scores, dim=-1)  # (..., h, n_q, n_k)
    return torch.matmul(probabilities, value)  # (..., h, n_q, d_h)


@torch.compiler.disable
def kernel_triangular_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tri_bias: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Call the optional cuEquivariance triangle-attention primitive."""

    # q/k/v: (..., h, n, d_h); returned tensor uses the kernel's matching layout.
    if (
        find_spec("cuequivariance_torch") is None
        or find_spec("cuequivariance_ops_torch") is None
    ):
        raise RuntimeError(
            "Boltz2 use_kernels=True requires cuequivariance_torch and the CUDA 13 "
            "cuequivariance_ops_torch runtime from the 'structure,cueq' extras."
        )
    cueq = importlib.import_module("cuequivariance_torch")
    return cueq.triangle_attention(q, k, v, tri_bias, mask=mask, scale=scale)


class Attention(nn.Module):
    """Multi-head pair attention with optional query-dependent output gates."""

    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
    ) -> None:
        super().__init__()
        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        projected_dim = c_hidden * no_heads
        self.linear_q = Linear(c_q, projected_dim, bias=False, init="glorot")
        self.linear_k = Linear(c_k, projected_dim, bias=False, init="glorot")
        self.linear_v = Linear(c_v, projected_dim, bias=False, init="glorot")
        self.linear_o = Linear(projected_dim, c_q, bias=False, init="final")
        self.linear_g = Linear(c_q, projected_dim, bias=False, init="gating") if gating else None
        self.sigmoid = nn.Sigmoid()

    def _prep_qkv(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        apply_scale: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project Q, K, and V to ``(..., h, n, d)`` tensors."""

        def split_heads(X: torch.Tensor) -> torch.Tensor:
            # X: (..., n, h * d_h)
            return X.view(*X.shape[:-1], self.no_heads, -1).transpose(
                -2, -3
            )  # (..., h, n, d_h)

        # q_x: (..., n_q, c_q); kv_x: (..., n_k, c_k).
        q = split_heads(self.linear_q(q_x))  # (..., h, n_q, d_h)
        k = split_heads(self.linear_k(kv_x))  # (..., h, n_k, d_h)
        v = split_heads(self.linear_v(kv_x))  # (..., h, n_k, d_h)
        if apply_scale:
            q /= math.sqrt(self.c_hidden)  # (..., h, n_q, d_h)
        return q, k, v  # (..., h, n_q, d_h), two (..., h, n_k, d_h)

    def _wrap_up(self, output: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        """Apply the optional gate, merge heads, and project the update."""

        # output: (..., n_q, h, d_h); q_x: (..., n_q, c_q).
        if self.linear_g is not None:
            gate = self.sigmoid(self.linear_g(q_x))  # (..., n_q, h * d_h)
            gate = gate.view(
                *gate.shape[:-1], self.no_heads, -1
            )  # (..., n_q, h, d_h)
            output = output * gate  # (..., n_q, h, d_h)
        return self.linear_o(
            flatten_final_dims(output, 2)
        )  # (..., n_q, c_q)

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        tri_bias: torch.Tensor,
        mask_bias: torch.Tensor,
        mask: torch.Tensor,
        use_kernels: bool = False,
    ) -> torch.Tensor:
        """Return a gated pair update with shape ``(..., n, c_q)``."""

        q, k, v = self._prep_qkv(
            q_x, kv_x, apply_scale=not use_kernels
        )  # q: (..., h, n_q, d_h); k/v: (..., h, n_k, d_h)
        if use_kernels:
            output = kernel_triangular_attn(
                q,
                k,
                v,
                tri_bias=tri_bias,
                mask=mask.bool(),
                scale=1.0 / math.sqrt(self.c_hidden),
            )  # (..., h, n_q, d_h)
        else:
            output = _attention(
                q, k, v, (mask_bias, tri_bias)
            )  # (..., h, n_q, d_h)
        return self._wrap_up(
            output.transpose(-2, -3), q_x
        )  # (..., n_q, c_q)
    # tensor: (...); softmax preserves shape.
