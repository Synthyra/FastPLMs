"""Shared attention infrastructure for all FastPLMs models.

Contains: AttentionBackend enum, backend resolution, mask creation,
flex attention helpers, flash kernel detection/dispatch, and pad/unpad utilities.
"""
from __future__ import annotations

from enum import Enum
from functools import partial

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
    """Return flex_attention callable: compiled (fused kernel) by default, or eager when debug flag is set.

    Uses kernel_options={"BACKEND": "FLASH"} to prefer Flash Attention 4 (FA4)
    on Hopper/Blackwell GPUs (PyTorch 2.11+). Automatically falls back to Triton
    on older hardware.
    """
    global _compiled_flex_attention
    if flex_attention is None:
        return None
    flex_mod = torch.nn.attention.flex_attention
    if getattr(flex_mod, "_FLEX_ATTENTION_DISABLE_COMPILE_DEBUG", False):
        return flex_attention
    if _compiled_flex_attention is None:
        _compiled_flex_attention = torch.compile(
            partial(flex_attention, kernel_options={"BACKEND": "FLASH"}),
            dynamic=False,
        )
    return _compiled_flex_attention


### Kernels Flash Attention Detection
def _infer_kernels_flash_variant(kernel) -> str | None:
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
) -> torch.Tensor:
    assert FLASH_KERNEL is not None, "Kernel Flash Attention is not available in this environment."
    if FLASH_KERNEL_VARIANT == "flash_attn2":
        return FLASH_KERNEL.fwd(q=query_states, k=key_states, v=value_states, is_causal=causal)[0]
    if FLASH_KERNEL_VARIANT == "flash_attn3":
        try:
            output = FLASH_KERNEL.flash_attn_func(q=query_states, k=key_states, v=value_states, causal=causal)
        except TypeError:
            output = FLASH_KERNEL.flash_attn_func(query_states, key_states, value_states, 0.0, None, causal)
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
) -> torch.Tensor:
    assert FLASH_KERNEL is not None, "Kernel Flash Attention is not available in this environment."
    if FLASH_KERNEL_VARIANT == "flash_attn2":
        return FLASH_KERNEL.varlen_fwd(
            q=query_states, k=key_states, v=value_states,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q, max_seqlen_k=max_seqlen_in_batch_k,
            is_causal=causal,
        )[0]
    if FLASH_KERNEL_VARIANT == "flash_attn3":
        try:
            output = FLASH_KERNEL.flash_attn_varlen_func(
                q=query_states, k=key_states, v=value_states,
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q, max_seqlen_k=max_seqlen_in_batch_k,
                causal=causal,
            )
        except TypeError:
            output = FLASH_KERNEL.flash_attn_varlen_func(
                query_states, key_states, value_states,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_in_batch_q, max_seqlen_in_batch_k,
                0.0, None, causal,
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
    def backward(ctx, grad_output) -> tuple[torch.Tensor, None]:
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
    def backward(ctx, grad_output) -> tuple[torch.Tensor, None, None]:
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor], tuple[int, int]]:
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
    attention_mask_2d: torch.Tensor | None = None,
    causal: bool = False,
) -> torch.Tensor:
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
        )
        return pad_input(attn_output_unpad, indices_q, batch_size, q_len)
    else:
        return _kernels_flash_forward(
            query_states=query_states, key_states=key_states, value_states=value_states, causal=causal,
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
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, BlockMask | None]:
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
