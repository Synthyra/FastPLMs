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
from collections.abc import Callable, Sequence

import torch
from einops import rearrange
from torch import nn

from . import vb_layers_initialize as initialize
from .vb_tri_attn_utils import flatten_final_dims, permute_final_dims


def _initialize_weight(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    method: str,
) -> None:
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

        input_dtype = input.dtype
        if self.precision is not None:
            with torch.autocast("cuda", enabled=False):
                bias = None if self.bias is None else self.bias.to(self.precision)
                projected = nn.functional.linear(
                    input.to(self.precision),
                    self.weight.to(self.precision),
                    bias,
                )
            return projected.to(input_dtype)
        if input_dtype is torch.bfloat16:
            with torch.autocast("cuda", enabled=False):
                bias = None if self.bias is None else self.bias.to(input_dtype)
                return nn.functional.linear(input, self.weight.to(input_dtype), bias)
        return nn.functional.linear(input, self.weight, self.bias)


class LayerNorm(nn.Module):
    """Layer normalization that avoids an autocast upcast for BF16 inputs."""

    def __init__(self, c_in: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.c_in = (c_in,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(c_in))
        self.bias = nn.Parameter(torch.zeros(c_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    scores = torch.matmul(query, permute_final_dims(key, (1, 0)))
    for bias in biases:
        scores += bias
    probabilities = softmax_no_cast(scores, dim=-1)
    return torch.matmul(probabilities, value)


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

    triangle = importlib.import_module("cuequivariance_torch.primitives.triangle")
    return triangle.triangle_attention(q, k, v, tri_bias, mask=mask, scale=scale)


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
            return X.view(*X.shape[:-1], self.no_heads, -1).transpose(-2, -3)

        q = split_heads(self.linear_q(q_x))
        k = split_heads(self.linear_k(kv_x))
        v = split_heads(self.linear_v(kv_x))
        if apply_scale:
            q /= math.sqrt(self.c_hidden)
        return q, k, v

    def _wrap_up(self, output: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        """Apply the optional gate, merge heads, and project the update."""

        if self.linear_g is not None:
            gate = self.sigmoid(self.linear_g(q_x))
            gate = gate.view(*gate.shape[:-1], self.no_heads, -1)
            output = output * gate
        return self.linear_o(flatten_final_dims(output, 2))

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

        q, k, v = self._prep_qkv(q_x, kv_x, apply_scale=not use_kernels)
        if use_kernels:
            output = kernel_triangular_attn(
                q,
                k,
                v,
                tri_bias=tri_bias,
                mask=mask.bool(),
                scale=1.0 / math.sqrt(self.c_hidden),
            )
        else:
            output = _attention(q, k, v, (mask_bias, tri_bias))
        return self._wrap_up(output.transpose(-2, -3), q_x)


def _trifast_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    biases: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Adapt local pair tensors to the optional TriFast kernel contract."""

    original_ndim = q.ndim
    if len(biases) != 2:
        raise ValueError(f"TriFast requires two bias tensors, got {len(biases)}")
    mask, triangle_bias = biases
    if triangle_bias.ndim == 5:
        triangle_bias = triangle_bias.squeeze(1)
    if original_ndim == 4:
        q, k, v = (tensor.unsqueeze(0) for tensor in (q, k, v))
        mask = mask.unsqueeze(0)
    if q.ndim != 5:
        raise ValueError(f"TriFast requires five-dimensional Q, K, and V, got {q.ndim}")

    q = rearrange(q, "b i h j d -> b h i j d")
    k = rearrange(k, "b i h j d -> b h i j d")
    v = rearrange(v, "b i h j d -> b h i j d")
    kernel_mask = rearrange(mask, "b i () () j -> b i j").bool()

    from trifast import triangle_attention

    output = triangle_attention(q, k, v, triangle_bias, kernel_mask)
    output = rearrange(output, "b h i j d -> b i j h d")
    return output.squeeze(0) if original_ndim == 4 else output
