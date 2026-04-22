"""Shared attention infrastructure for all FastPLMs models.

Contains: AttentionBackend enum, backend resolution, mask creation,
flex attention helpers, flash kernel detection/dispatch, and pad/unpad utilities.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention, BlockMask
except ImportError:
    create_block_mask = None
    flex_attention = None
    BlockMask = None

_compiled_flex_attention = None


def _get_flex_attention_fn():
    """Return flex_attention callable: compiled (fused kernel) by default, or eager when debug flag is set."""
    global _compiled_flex_attention
    if flex_attention is None:
        return None
    flex_mod = torch.nn.attention.flex_attention
    if getattr(flex_mod, "_FLEX_ATTENTION_DISABLE_COMPILE_DEBUG", False):
        return flex_attention
    if _compiled_flex_attention is None:
        _compiled_flex_attention = torch.compile(
            flex_attention,
            dynamic=False,
        )
    return _compiled_flex_attention


### Kernels Flash Attention Detection
def _infer_kernels_flash_variant(kernel) -> Optional[str]:
    if hasattr(kernel, "fwd") and hasattr(kernel, "varlen_fwd"):
        return "flash_attn2"
    if hasattr(kernel, "flash_attn_func") and hasattr(kernel, "flash_attn_varlen_func"):
        return "flash_attn3"
    return None


def _try_get_kernels_flash():
    try:
        from kernels import get_kernel
    except ImportError:
        return None, None

    flash_kernel = None
    flash_kernel_variant = None
    try:
        flash_kernel = get_kernel("kernels-community/flash-attn3")
        flash_kernel_variant = _infer_kernels_flash_variant(flash_kernel)
        assert flash_kernel_variant is not None, "Loaded flash-attn3 kernel does not expose a supported API."
    except Exception:
        try:
            flash_kernel = get_kernel("kernels-community/flash-attn2")
            flash_kernel_variant = _infer_kernels_flash_variant(flash_kernel)
            assert flash_kernel_variant is not None, "Loaded flash-attn2 kernel does not expose a supported API."
        except Exception:
            flash_kernel = None
            flash_kernel_variant = None
    return flash_kernel, flash_kernel_variant


_FLASH_KERNELS_LOADED = False
FLASH_KERNEL = None
FLASH_KERNEL_VARIANT = None


def _ensure_flash_kernels_loaded():
    global _FLASH_KERNELS_LOADED, FLASH_KERNEL, FLASH_KERNEL_VARIANT
    if _FLASH_KERNELS_LOADED:
        return
    _FLASH_KERNELS_LOADED = True
    FLASH_KERNEL, FLASH_KERNEL_VARIANT = _try_get_kernels_flash()


def _kernels_flash_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Flash-attention forward, optionally overriding the softmax scale.

    When `softmax_scale is None`, the flash kernel applies its default
    `1 / sqrt(head_dim)`. Pass `softmax_scale=1.0` if the caller has already
    pre-scaled Q (the convention used by ESM2, DPLM, DPLM2, E1, ESMFold).
    Failing to override when Q is pre-scaled produces DOUBLE scaling and
    catastrophic downstream drift -- on DPLM-150M (30 layers) this was observed
    as pooled-embedding cosine ~-0.12 and argmax agreement ~0.27 vs sdpa.
    """
    assert FLASH_KERNEL is not None, "Kernel Flash Attention is not available in this environment."
    if FLASH_KERNEL_VARIANT == "flash_attn2":
        return FLASH_KERNEL.fwd(
            q=query_states, k=key_states, v=value_states,
            softmax_scale=softmax_scale, is_causal=causal,
        )[0]
    if FLASH_KERNEL_VARIANT == "flash_attn3":
        try:
            output = FLASH_KERNEL.flash_attn_func(
                q=query_states, k=key_states, v=value_states,
                softmax_scale=softmax_scale, causal=causal,
            )
        except TypeError:
            output = FLASH_KERNEL.flash_attn_func(
                query_states, key_states, value_states,
                0.0, softmax_scale, causal,
            )
        if isinstance(output, tuple):
            return output[0]
        return output
    raise AssertionError(f"Unsupported kernels flash attention variant: {FLASH_KERNEL_VARIANT}")


def _kernels_flash_varlen_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_in_batch_q: int,
    max_seqlen_in_batch_k: int,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Varlen flash-attention forward, optionally overriding the softmax scale.

    See `_kernels_flash_forward` docstring for why `softmax_scale=1.0` must be
    passed when Q has been pre-scaled by the caller.
    """
    assert FLASH_KERNEL is not None, "Kernel Flash Attention is not available in this environment."
    if FLASH_KERNEL_VARIANT == "flash_attn2":
        return FLASH_KERNEL.varlen_fwd(
            q=query_states, k=key_states, v=value_states,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q, max_seqlen_k=max_seqlen_in_batch_k,
            softmax_scale=softmax_scale, is_causal=causal,
        )[0]
    if FLASH_KERNEL_VARIANT == "flash_attn3":
        try:
            output = FLASH_KERNEL.flash_attn_varlen_func(
                q=query_states, k=key_states, v=value_states,
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q, max_seqlen_k=max_seqlen_in_batch_k,
                softmax_scale=softmax_scale, causal=causal,
            )
        except TypeError:
            output = FLASH_KERNEL.flash_attn_varlen_func(
                query_states, key_states, value_states,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_in_batch_q, max_seqlen_in_batch_k,
                0.0, softmax_scale, causal,
            )
        if isinstance(output, tuple):
            return output[0]
        return output
    raise AssertionError(f"Unsupported kernels flash attention variant: {FLASH_KERNEL_VARIANT}")


### Unpad / Pad helpers for varlen flash attention
class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices) -> torch.Tensor:
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        return torch.gather(
            rearrange(input, "b ... -> b (...)"), 0, indices.unsqueeze(1).expand(-1, second_dim)
        ).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output) -> Tuple[torch.Tensor, None]:
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]], device=grad_output.device, dtype=grad_output.dtype
        )
        grad_input.scatter_(0, indices.unsqueeze(1).expand(-1, grad_output.shape[1]), grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indices, first_axis_dim) -> torch.Tensor:
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype)
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx, grad_output) -> Tuple[torch.Tensor, None, None]:
        (indices,) = ctx.saved_tensors
        return grad_output[indices], None, None


index_first_axis = IndexFirstAxis.apply
index_put_first_axis = IndexPutFirstAxis.apply


def pad_input(hidden_states: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int) -> torch.Tensor:
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


def _unpad_input(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    attention_mask_2d: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[int, int]]:
    batch_size, seq_len, num_heads, head_dim = query_layer.shape
    seqlens = attention_mask_2d.sum(dim=1).int()
    cu_seqlens = F.pad(seqlens.cumsum(0, dtype=torch.int32), (1, 0))
    max_seqlen = int(seqlens.max().item())
    indices = attention_mask_2d.flatten().nonzero(as_tuple=False).flatten()
    query_layer = index_first_axis(query_layer.reshape(batch_size * seq_len, num_heads, head_dim), indices)
    key_layer = index_first_axis(key_layer.reshape(batch_size * seq_len, num_heads, head_dim), indices)
    value_layer = index_first_axis(value_layer.reshape(batch_size * seq_len, num_heads, head_dim), indices)
    return query_layer, key_layer, value_layer, indices, (cu_seqlens, cu_seqlens), (max_seqlen, max_seqlen)


def kernels_flash_attention_func(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask_2d: Optional[torch.Tensor] = None,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Public flash-attention entry point with optional padding handling.

    `softmax_scale`:
        None -> kernel applies its default `1 / sqrt(head_dim)`.
        float -> kernel uses the given scale (pass 1.0 when Q is pre-scaled
        by the caller).

    IMPORTANT: if your family multiplies Q by `1/sqrt(head_dim)` before calling
    this function (as ESM2, DPLM, DPLM2, E1, and ESMFold do) you MUST pass
    `softmax_scale=1.0`. Otherwise the kernel applies its default scale ON TOP
    of the caller's, producing effective scale `1/head_dim` and catastrophic
    downstream drift that compounds across layers.
    """
    assert FLASH_KERNEL is not None, "Kernel Flash Attention is not available in this environment."
    if not causal and attention_mask_2d is not None:
        batch_size, q_len = query_states.shape[:2]
        (
            query_states, key_states, value_states,
            indices_q, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k),
        ) = _unpad_input(query_states, key_states, value_states, attention_mask_2d)
        attn_output_unpad = _kernels_flash_varlen_forward(
            query_states=query_states, key_states=key_states, value_states=value_states,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_in_batch_q=max_seqlen_q, max_seqlen_in_batch_k=max_seqlen_k,
            softmax_scale=softmax_scale,
        )
        return pad_input(attn_output_unpad, indices_q, batch_size, q_len)
    else:
        return _kernels_flash_forward(
            query_states=query_states, key_states=key_states, value_states=value_states,
            causal=causal, softmax_scale=softmax_scale,
        )


### Attention Backend Enum & Resolution
class AttentionBackend(Enum):
    AUTO = "auto"
    KERNELS_FLASH = "kernels_flash"
    FLEX = "flex"
    SDPA = "sdpa"


VALID_ATTENTION_BACKENDS = tuple(b.value for b in AttentionBackend)


_BACKEND_CONFIRMED = False


def resolve_attention_backend(requested_backend: str) -> AttentionBackend:
    global _BACKEND_CONFIRMED
    assert requested_backend in VALID_ATTENTION_BACKENDS, (
        f"Unsupported attention backend: {requested_backend}. Expected one of {VALID_ATTENTION_BACKENDS}."
    )
    if requested_backend in (AttentionBackend.AUTO.value, AttentionBackend.KERNELS_FLASH.value):
        _ensure_flash_kernels_loaded()
    if requested_backend == AttentionBackend.AUTO.value:
        if FLASH_KERNEL is not None:
            resolved = AttentionBackend.KERNELS_FLASH
        elif flex_attention is not None:
            resolved = AttentionBackend.FLEX
        else:
            resolved = AttentionBackend.SDPA
    elif requested_backend == AttentionBackend.KERNELS_FLASH.value:
        assert FLASH_KERNEL is not None, "Kernels Flash Attention is not available in this environment."
        resolved = AttentionBackend.KERNELS_FLASH
    elif requested_backend == AttentionBackend.FLEX.value:
        assert flex_attention is not None, "Flex Attention is not available in this environment."
        resolved = AttentionBackend.FLEX
    elif requested_backend == AttentionBackend.SDPA.value:
        resolved = AttentionBackend.SDPA
    else:
        raise AssertionError(f"Unsupported attention backend: {requested_backend}")
    if not _BACKEND_CONFIRMED:
        print(f"Attention backend: config='{requested_backend}' -> resolved='{resolved.value}'")
        _BACKEND_CONFIRMED = True
    return resolved


@torch.compiler.disable
def get_attention_mask(
    effective_backend: AttentionBackend,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[BlockMask]]:
    """Build padding masks once for all encoder layers.

    Returns (attention_mask_2d, attention_mask_4d, flex_block_mask).
    """
    if attention_mask is None:
        return None, None, None

    attention_mask_2d = attention_mask.bool()

    if effective_backend == AttentionBackend.KERNELS_FLASH:
        return attention_mask_2d, None, None

    if effective_backend == AttentionBackend.FLEX:
        assert create_block_mask is not None, "Flex attention backend requested but torch.create_block_mask is unavailable."
        valid_lens = attention_mask_2d.sum(dim=-1)

        def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
            return (q_idx < valid_lens[batch_idx]) & (kv_idx < valid_lens[batch_idx])

        flex_block_mask = create_block_mask(mask_mod, batch_size, 1, seq_len, seq_len, device=device)
        return attention_mask_2d, None, flex_block_mask

    # SDPA / manual -- only mask the key dimension so padding query positions attend to
    # real keys and produce valid (non-NaN) outputs instead of NaN from softmax(-inf,...,-inf).
    attention_mask_4d = attention_mask_2d[:, None, None, :]
    return attention_mask_2d, attention_mask_4d, None


def bool_to_additive_mask(
    bool_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert a bool mask (True = valid) to a float additive mask (0.0 valid, -inf invalid).

    Why this exists: calling `bool_mask.masked_fill(bool_mask.logical_not(), float('-inf'))`
    directly on a bool tensor returns a bool tensor -- because `-inf` casts to `True` -- and
    silently drops the mask entirely. Always allocate a float tensor first, then fill it.
    This helper is the sanctioned way to build an SDPA additive mask from a bool validity mask.
    """
    assert bool_mask.dtype == torch.bool, (
        f"bool_to_additive_mask requires a bool tensor, got dtype={bool_mask.dtype}"
    )
    additive = torch.zeros_like(bool_mask, dtype=dtype)
    additive.masked_fill_(bool_mask.logical_not(), float("-inf"))
    return additive
