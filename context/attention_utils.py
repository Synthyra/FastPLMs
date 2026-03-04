import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention,
)
from enum import Enum
from typing import Any
from einops import rearrange, repeat
from kernels import get_kernel


def _infer_kernels_flash_variant(kernel: Any) -> str | None:
    if hasattr(kernel, "fwd") and hasattr(kernel, "varlen_fwd"):
        return "flash_attn2"
    if hasattr(kernel, "flash_attn_func") and hasattr(kernel, "flash_attn_varlen_func"):
        return "flash_attn3"
    return None


def try_get_kernels():
    flash_kernel_variant = None
    try:
        flash_kernel = get_kernel("kernels-community/flash-attn3")
        flash_kernel_variant = _infer_kernels_flash_variant(flash_kernel)
        print(f"Using flash-attn3 kernel: {flash_kernel}")
        assert flash_kernel_variant is not None, "Loaded flash-attn3 kernel does not expose a supported API."
    except Exception as e1:
        print(f"Failed to load flash-attn3 kernel: {e1}")
        try:
            flash_kernel = get_kernel("kernels-community/flash-attn2")
            flash_kernel_variant = _infer_kernels_flash_variant(flash_kernel)
            print(f"Using flash-attn2 kernel: {flash_kernel}")
            assert flash_kernel_variant is not None, "Loaded flash-attn2 kernel does not expose a supported API."
        except Exception as e2:
            print(f"Failed to load flash-attn2 kernel: {e2}")
            flash_kernel = None
            flash_kernel_variant = None
    try:
        layer_norm = get_kernel("kernels-community/triton-layer-norm")
        print(f"Using triton-layer-norm kernel: {layer_norm}")
    except Exception as e3:
        print(f"Failed to load triton-layer-norm kernel: {e3}")
        layer_norm = None
        print("Will be using PyTorch RMSNorm instead")
    return flash_kernel, flash_kernel_variant, layer_norm


FLASH_KERNEL, FLASH_KERNEL_VARIANT, LAYER_NORM = try_get_kernels()


def _kernels_flash_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    assert FLASH_KERNEL is not None, "Kernel Flash Attention is not available in this environment."

    if FLASH_KERNEL_VARIANT == "flash_attn2":
        return FLASH_KERNEL.fwd(q=query_states, k=key_states, v=value_states, is_causal=causal)[0]  # type: ignore[union-attr]
    if FLASH_KERNEL_VARIANT == "flash_attn3":
        try:
            output = FLASH_KERNEL.flash_attn_func(  # type: ignore[union-attr]
                q=query_states,
                k=key_states,
                v=value_states,
                causal=causal,
            )
        except TypeError:
            output = FLASH_KERNEL.flash_attn_func(  # type: ignore[union-attr]
                query_states,
                key_states,
                value_states,
                0.0,
                None,
                causal,
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
) -> torch.Tensor:
    assert FLASH_KERNEL is not None, "Kernel Flash Attention is not available in this environment."

    if FLASH_KERNEL_VARIANT == "flash_attn2":
        return FLASH_KERNEL.varlen_fwd(  # type: ignore[union-attr]
            q=query_states,
            k=key_states,
            v=value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            is_causal=causal,
        )[0]
    if FLASH_KERNEL_VARIANT == "flash_attn3":
        try:
            output = FLASH_KERNEL.flash_attn_varlen_func(  # type: ignore[union-attr]
                q=query_states,
                k=key_states,
                v=value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                causal=causal,
            )
        except TypeError:
            output = FLASH_KERNEL.flash_attn_varlen_func(  # type: ignore[union-attr]
                query_states,
                key_states,
                value_states,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_in_batch_q,
                max_seqlen_in_batch_k,
                0.0,
                None,
                causal,
            )
        if isinstance(output, tuple):
            return output[0]
        return output
    raise AssertionError(f"Unsupported kernels flash attention variant: {FLASH_KERNEL_VARIANT}")


class AttentionBackend(Enum):
    AUTO = "auto"
    KERNELS_FLASH = "kernels_flash"
    FLASH_ATTN = "flash_attn"
    FLEX = "flex"
    SDPA = "sdpa"


def _validate_attention_backend(attn_backend: str) -> None:
    valid_backends = (
        AttentionBackend.AUTO.value,
        AttentionBackend.KERNELS_FLASH.value,
        AttentionBackend.FLASH_ATTN.value,
        AttentionBackend.FLEX.value,
        AttentionBackend.SDPA.value,
    )
    assert attn_backend in valid_backends, (
        f"Unsupported attention backend: {attn_backend}. "
        f"Expected one of {valid_backends}."
    )


def resolve_attention_backend(requested_backend: str) -> AttentionBackend:
    """Resolve a (possibly AUTO) backend string to a concrete AttentionBackend.

    This is exposed at module level so TransformerStack can call it once and build
    the right mask type before iterating over layers.
    """
    _validate_attention_backend(requested_backend)
    if requested_backend == AttentionBackend.AUTO.value:
        if FLASH_KERNEL is not None:
            return AttentionBackend.KERNELS_FLASH
        else:
            return AttentionBackend.FLEX
    elif requested_backend == AttentionBackend.KERNELS_FLASH.value:
        if FLASH_KERNEL is not None:
            return AttentionBackend.KERNELS_FLASH
        else:
            raise AssertionError("Kernels Flash Attention is not available in this environment.")
    elif requested_backend == AttentionBackend.FLEX.value:
        return AttentionBackend.FLEX
    elif requested_backend == AttentionBackend.SDPA.value:
        return AttentionBackend.SDPA
    else:
        raise AssertionError(f"Unsupported attention backend: {requested_backend}")


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads from (batch, num_kv_heads, slen, head_dim) to (batch, num_heads, slen, head_dim)."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def sdpa_attention_func(
    query_states: torch.Tensor,       # (bs, q_len, nh, hs)
    key_states: torch.Tensor,         # (bs, kv_len, nkv, hs)
    value_states: torch.Tensor,       # (bs, kv_len, nkv, hs)
    attention_mask_4d: torch.Tensor | None,  # (bs, 1, q_len, kv_len) bool
    num_key_value_groups: int,
) -> torch.Tensor:
    query_BHLD = query_states.transpose(1, 2).contiguous()
    key_BHLD = key_states.transpose(1, 2).contiguous()
    value_BHLD = value_states.transpose(1, 2).contiguous()
    key_BHLD = _repeat_kv(key_BHLD, num_key_value_groups)
    value_BHLD = _repeat_kv(value_BHLD, num_key_value_groups)
    out_BHLD = F.scaled_dot_product_attention(
        query_BHLD, key_BHLD, value_BHLD, attn_mask=attention_mask_4d, is_causal=False
    )
    return out_BHLD.transpose(1, 2).contiguous()


def flex_attention_func(
    query_states: torch.Tensor,  # (bs, seqlen, nh, hs)
    key_states: torch.Tensor,    # (bs, seqlen, nkv, hs)
    value_states: torch.Tensor,  # (bs, seqlen, nkv, hs)
    block_mask: BlockMask | None = None,
) -> torch.Tensor:
    assert flex_attention is not None, "Flex Attention is not available in this environment"
    query_states = query_states.transpose(1, 2).contiguous()  # (bs, nh, seqlen, hs)
    key_states = key_states.transpose(1, 2).contiguous()      # (bs, nkv, seqlen, hs)
    value_states = value_states.transpose(1, 2).contiguous()  # (bs, nkv, seqlen, hs)

    outputs = flex_attention(
        query_states,
        key_states,
        value_states,
        block_mask=block_mask,
        enable_gqa=query_states.shape[1] != key_states.shape[1],
    )
    return outputs.transpose(1, 2)  # (bs, seqlen, nh, hs)


class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices) -> torch.Tensor:  # type: ignore[no-untyped-def]
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        return torch.gather(rearrange(input, "b ... -> b (...)"), 0, repeat(indices, "z -> z d", d=second_dim)).reshape(
            -1, *other_shape
        )

    @staticmethod
    def backward(ctx, grad_output) -> tuple[torch.Tensor, None]:  # type: ignore[no-untyped-def]
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]], device=grad_output.device, dtype=grad_output.dtype
        )
        grad_input.scatter_(0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indices, first_axis_dim) -> torch.Tensor:  # type: ignore[no-untyped-def]
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype)
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx, grad_output) -> tuple[torch.Tensor, None, None]:  # type: ignore[no-untyped-def]
        (indices,) = ctx.saved_tensors
        return grad_output[indices], None, None


index_first_axis = IndexFirstAxis.apply
index_put_first_axis = IndexPutFirstAxis.apply


def pad_input(hidden_states: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int) -> torch.Tensor:
    """Re-pad unpadded token tensors back to (batch, seqlen, ...) shape."""
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


def _unpad_input(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    attention_mask_2d: torch.Tensor,  # (batch, seq_len) bool - 1 for valid, 0 for pad
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor], tuple[int, int]]:
    """Remove padding tokens from Q/K/V for efficient varlen flash attention.

    Assumes q_len == kv_len (padded batch, not packed/cross-attention).
    The 2D mask is derived from attention_mask_4d[:, 0, 0, :] at the call sites,
    which equals the original 2D padding mask for non-causal symmetric masks.
    """
    batch_size, seq_len, num_kv_heads, head_dim = key_layer.shape
    num_q_heads = query_layer.shape[2]
    assert query_layer.shape[1] == seq_len, (
        f"Varlen unpadding requires q_len == kv_len, got {query_layer.shape[1]} != {seq_len}"
    )

    seqlens = attention_mask_2d.sum(dim=1).int()
    cu_seqlens = F.pad(seqlens.cumsum(0, dtype=torch.int32), (1, 0))
    max_seqlen = int(seqlens.max().item())
    indices = attention_mask_2d.flatten().nonzero(as_tuple=False).flatten()

    key_layer = index_first_axis(key_layer.reshape(batch_size * seq_len, num_kv_heads, head_dim), indices)
    value_layer = index_first_axis(value_layer.reshape(batch_size * seq_len, num_kv_heads, head_dim), indices)
    query_layer = index_first_axis(query_layer.reshape(batch_size * seq_len, num_q_heads, head_dim), indices)

    return (
        query_layer,
        key_layer,
        value_layer,
        indices,
        (cu_seqlens, cu_seqlens),
        (max_seqlen, max_seqlen),
    )


def kernels_flash_attention_func(
    query_states: torch.Tensor,       # (bs, seqlen, nh, hs)
    key_states: torch.Tensor,         # (bs, seqlen, nkv, hs)
    value_states: torch.Tensor,       # (bs, seqlen, nkv, hs)
    attention_mask_2d: torch.Tensor | None = None,  # (bs, seqlen) bool
    causal: bool = False,
) -> torch.Tensor:
    assert FLASH_KERNEL is not None, "Kernel Flash Attention is not available in this environment."

    if not causal and attention_mask_2d is not None:
        batch_size, q_len = query_states.shape[:2]
        (
            query_states,
            key_states,
            value_states,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_q, max_seqlen_k),
        ) = _unpad_input(query_states, key_states, value_states, attention_mask_2d)

        attn_output_unpad = _kernels_flash_varlen_forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_in_batch_q=max_seqlen_q,
            max_seqlen_in_batch_k=max_seqlen_k,
        )
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, q_len)
    else:
        attn_output = _kernels_flash_forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            causal=causal,
        )

    return attn_output
