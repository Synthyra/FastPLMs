"""Low-level attention kernels and mask construction.

The public backend contract lives in :mod:`fastplms.attention`. Optional
kernels are resolved only after a caller explicitly requests them, so importing
FastPLMs never downloads or compiles code.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from enum import Enum

import torch
from einops import rearrange
from torch.nn import functional as F

from ._kernel_lock import load_locked_kernel

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
except ImportError:
    create_block_mask = None
    flex_attention = None
    BlockMask = None

_MAX_FLEX_CACHE_ENTRIES = 128
_compiled_flex_attention: OrderedDict[tuple, object] = OrderedDict()
_flex_block_masks: OrderedDict[tuple, BlockMask] = OrderedDict()


def _remember(cache: OrderedDict, key: tuple, value):
    """Insert an item into a bounded least-recently-used cache."""
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _MAX_FLEX_CACHE_ENTRIES:
        cache.popitem(last=False)
    return value


def _get_flex_attention_fn(
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    shape: tuple[int, ...] | None = None,
    sequence_lengths: tuple[int, ...] | None = None,
    mask_semantics: str = "padding",
):
    """Return a compiled Flex callable for an explicit execution signature.

    The signature contains the device, dtype, tensor shape, per-sequence
    lengths, and mask semantics. This keeps compiled graphs from being reused
    across incompatible padding or masking contracts.
    """
    if flex_attention is None:
        return None
    flex_mod = torch.nn.attention.flex_attention
    if getattr(flex_mod, "_FLEX_ATTENTION_DISABLE_COMPILE_DEBUG", False):
        return flex_attention
    key = (
        None if device is None else str(device),
        None if dtype is None else str(dtype),
        shape,
        sequence_lengths,
        mask_semantics,
    )
    compiled = _compiled_flex_attention.get(key)
    if compiled is None:
        compiled = torch.compile(flex_attention, dynamic=False)
        _remember(_compiled_flex_attention, key, compiled)
    else:
        _compiled_flex_attention.move_to_end(key)
    return compiled


# Hugging Face `kernels` exposes slightly different APIs for FlashAttention 2
# and 3. Detect the loaded variant once so every caller uses the same dispatch.
def _infer_kernels_flash_variant(kernel) -> str | None:
    if hasattr(kernel, "fwd") and hasattr(kernel, "varlen_fwd"):
        return "flash_attn2"
    if hasattr(kernel, "flash_attn_func") and hasattr(kernel, "flash_attn_varlen_func"):
        return "flash_attn3"
    return None


def _load_kernels_flash(implementation: str) -> tuple[object, str]:
    """Load exactly the requested FlashAttention kernel.

    Loading is deferred until backend selection. A FlashAttention-2 request
    never falls through to FlashAttention-3, or vice versa.
    """
    from fastplms.registry import get_model_registry

    kernel_spec = get_model_registry().attention_kernels[implementation]
    repository = kernel_spec.repository
    try:
        flash_kernel = load_locked_kernel(repository, kernel_spec.revision)
    except Exception as error:
        raise RuntimeError(
            f"Unable to load the manifest-pinned kernel "
            f"{repository}@{kernel_spec.revision} for {implementation!r}."
        ) from error
    flash_kernel_variant = _infer_kernels_flash_variant(flash_kernel)
    if flash_kernel_variant != kernel_spec.expected_variant:
        raise RuntimeError(
            f"{repository}@{kernel_spec.revision} exposed {flash_kernel_variant!r}; "
            f"expected {kernel_spec.expected_variant!r}."
        )
    return flash_kernel, flash_kernel_variant


_FLASH_KERNELS: dict[str, tuple[object, str]] = {}


def _validate_kernels_flash_dtype(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    implementation: str,
) -> torch.dtype:
    """Reject dtypes outside the immutable kernel manifest before dispatch."""

    tensor_dtypes = {query_states.dtype, key_states.dtype, value_states.dtype}
    if len(tensor_dtypes) != 1:
        observed = ", ".join(sorted(str(dtype) for dtype in tensor_dtypes))
        raise RuntimeError(
            f"{implementation!r} requires Q, K, and V to share one dtype; "
            f"received {observed}."
        )
    runtime_dtype = query_states.dtype
    if (
        runtime_dtype == torch.float32
        and query_states.is_cuda
        and torch.is_autocast_enabled("cuda")
    ):
        runtime_dtype = torch.get_autocast_dtype("cuda")
    dtype_names = {
        torch.float32: "float32",
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
    }
    runtime_dtype_name = dtype_names.get(runtime_dtype, str(runtime_dtype))
    from fastplms.registry import get_model_registry

    supported = get_model_registry().attention_kernels[implementation].dtypes
    if runtime_dtype_name not in supported:
        expected = ", ".join(supported)
        raise RuntimeError(
            f"{implementation!r} supports only manifest-declared dtype(s) {expected}; "
            f"received {runtime_dtype_name}. Use CUDA BF16 autocast for FP32-resident "
            "models."
        )
    return runtime_dtype


def _validate_kernels_flash_device(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    implementation: str,
) -> torch.device:
    """Require Q, K, and V on one CUDA device before loading a kernel."""

    devices = (query_states.device, key_states.device, value_states.device)
    if len(set(devices)) != 1:
        observed = ", ".join(str(device) for device in devices)
        raise RuntimeError(
            f"{implementation!r} requires Q, K, and V on one device; received {observed}."
        )
    device = devices[0]
    if device.type != "cuda" or not all(
        tensor.is_cuda for tensor in (query_states, key_states, value_states)
    ):
        raise RuntimeError(
            f"{implementation!r} requires CUDA Q, K, and V; received device {device}."
        )
    return device


def _ensure_flash_kernels_loaded(implementation: str) -> tuple[object, str]:
    cached = _FLASH_KERNELS.get(implementation)
    if cached is not None:
        return cached
    loaded = _load_kernels_flash(implementation)
    _FLASH_KERNELS[implementation] = loaded
    return loaded


def _kernels_flash_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    causal: bool = False,
    softmax_scale: float | None = None,
    implementation: str = "flash_attention_3",
) -> torch.Tensor:
    """Flash-attention forward, optionally overriding the softmax scale.

    When `softmax_scale is None`, the flash kernel applies its default
    `1 / sqrt(head_dim)`. Pass `softmax_scale=1.0` if the caller has already
    pre-scaled Q (the convention used by ESM2, DPLM, DPLM2, E1, ESMFold).
    Failing to override when Q is pre-scaled applies the scale twice and breaks
    parity with eager attention and SDPA.
    """
    flash_kernel, flash_kernel_variant = _ensure_flash_kernels_loaded(implementation)
    if flash_kernel_variant == "flash_attn2":
        return flash_kernel.fwd(
            q=query_states,
            k=key_states,
            v=value_states,
            softmax_scale=softmax_scale,
            is_causal=causal,
        )[0]
    if flash_kernel_variant == "flash_attn3":
        try:
            output = flash_kernel.flash_attn_func(
                q=query_states,
                k=key_states,
                v=value_states,
                softmax_scale=softmax_scale,
                causal=causal,
            )
        except TypeError:
            output = flash_kernel.flash_attn_func(
                query_states,
                key_states,
                value_states,
                0.0,
                softmax_scale,
                causal,
            )
        if isinstance(output, tuple):
            return output[0]
        return output
    raise RuntimeError(f"Unsupported FlashAttention kernel variant: {flash_kernel_variant}")


def _kernels_flash_varlen_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_in_batch_q: int,
    max_seqlen_in_batch_k: int,
    causal: bool = False,
    softmax_scale: float | None = None,
    implementation: str = "flash_attention_3",
) -> torch.Tensor:
    """Varlen flash-attention forward, optionally overriding the softmax scale.

    See `_kernels_flash_forward` docstring for why `softmax_scale=1.0` must be
    passed when Q has been pre-scaled by the caller.
    """
    flash_kernel, flash_kernel_variant = _ensure_flash_kernels_loaded(implementation)
    if flash_kernel_variant == "flash_attn2":
        return flash_kernel.varlen_fwd(
            q=query_states,
            k=key_states,
            v=value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            softmax_scale=softmax_scale,
            is_causal=causal,
        )[0]
    if flash_kernel_variant == "flash_attn3":
        try:
            output = flash_kernel.flash_attn_varlen_func(
                q=query_states,
                k=key_states,
                v=value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                softmax_scale=softmax_scale,
                causal=causal,
            )
        except TypeError:
            output = flash_kernel.flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_in_batch_q,
                max_seqlen_in_batch_k,
                0.0,
                softmax_scale,
                causal,
            )
        if isinstance(output, tuple):
            return output[0]
        return output
    raise RuntimeError(f"Unsupported FlashAttention kernel variant: {flash_kernel_variant}")


# Varlen flash attention runs only on real tokens. These helpers remove padding
# before the kernel call and restore the original padded batch shape afterward.
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
            [ctx.first_axis_dim, grad_output.shape[1]],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        grad_input.scatter_(0, indices.unsqueeze(1).expand(-1, grad_output.shape[1]), grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indices, first_axis_dim) -> torch.Tensor:
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(
            first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype
        )
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx, grad_output) -> tuple[torch.Tensor, None, None]:
        (indices,) = ctx.saved_tensors
        return grad_output[indices], None, None


index_first_axis = IndexFirstAxis.apply
index_put_first_axis = IndexPutFirstAxis.apply


def pad_input(
    hidden_states: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int
) -> torch.Tensor:
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


def _unpad_input(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    attention_mask_2d: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor],
    tuple[int, int],
]:
    batch_size, seq_len, num_heads, head_dim = query_layer.shape
    seqlens = attention_mask_2d.sum(dim=1).int()
    cu_seqlens = F.pad(seqlens.cumsum(0, dtype=torch.int32), (1, 0))
    max_seqlen = int(seqlens.max().item())
    indices = attention_mask_2d.flatten().nonzero(as_tuple=False).flatten()
    query_layer = index_first_axis(
        query_layer.reshape(batch_size * seq_len, num_heads, head_dim), indices
    )
    key_layer = index_first_axis(
        key_layer.reshape(batch_size * seq_len, num_heads, head_dim), indices
    )
    value_layer = index_first_axis(
        value_layer.reshape(batch_size * seq_len, num_heads, head_dim), indices
    )
    return (
        query_layer,
        key_layer,
        value_layer,
        indices,
        (cu_seqlens, cu_seqlens),
        (max_seqlen, max_seqlen),
    )


def _validate_flash_padding_mask(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask_2d: torch.Tensor,
) -> torch.Tensor:
    """Validate the self-attention padding mask used by the varlen kernels."""

    if attention_mask_2d.ndim != 2:
        raise ValueError(
            "FlashAttention padding masks must have shape (batch, sequence_length)."
        )
    expected_shape = query_states.shape[:2]
    if tuple(attention_mask_2d.shape) != tuple(expected_shape):
        raise ValueError(
            "FlashAttention padding mask shape must match the query batch and "
            f"sequence dimensions; expected {tuple(expected_shape)}, received "
            f"{tuple(attention_mask_2d.shape)}."
        )
    if key_states.shape[:2] != expected_shape or value_states.shape[:2] != expected_shape:
        raise ValueError(
            "Masked FlashAttention requires Q, K, and V to share batch and "
            "sequence dimensions."
        )
    if attention_mask_2d.device != query_states.device:
        raise ValueError(
            "FlashAttention padding mask and Q, K, and V must be on the same device."
        )
    return attention_mask_2d.to(dtype=torch.bool)


def kernels_flash_attention_func(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask_2d: torch.Tensor | None = None,
    causal: bool = False,
    softmax_scale: float | None = None,
    implementation: str = "flash_attention_3",
) -> torch.Tensor:
    """Public flash-attention entry point with optional padding handling.

    `softmax_scale`:
        None -> kernel applies its default `1 / sqrt(head_dim)`.
        float -> kernel uses the given scale (pass 1.0 when Q is pre-scaled
        by the caller).

    Caller contract: if a model family pre-scales Q by `1/sqrt(head_dim)`
    before calling this function (ESM2, DPLM, DPLM2, E1, and ESMFold do), pass
    `softmax_scale=1.0`. Otherwise the flash kernel applies its default scale
    again, yielding an effective `1/head_dim` scale that drifts across layers.
    """
    _validate_kernels_flash_device(
        query_states,
        key_states,
        value_states,
        implementation,
    )
    runtime_dtype = _validate_kernels_flash_dtype(
        query_states,
        key_states,
        value_states,
        implementation,
    )
    if query_states.dtype != runtime_dtype:
        query_states = query_states.to(dtype=runtime_dtype)
        key_states = key_states.to(dtype=runtime_dtype)
        value_states = value_states.to(dtype=runtime_dtype)
    if attention_mask_2d is not None:
        attention_mask_2d = _validate_flash_padding_mask(
            query_states,
            key_states,
            value_states,
            attention_mask_2d,
        )
    _ensure_flash_kernels_loaded(implementation)
    if attention_mask_2d is not None:
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
            causal=causal,
            softmax_scale=softmax_scale,
            implementation=implementation,
        )
        output = pad_input(attn_output_unpad, indices_q, batch_size, q_len)
        return output.masked_fill(~attention_mask_2d[:, :, None, None], 0)
    else:
        return _kernels_flash_forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            causal=causal,
            softmax_scale=softmax_scale,
            implementation=implementation,
        )


# User-facing backend strings follow the Transformers attention interface.
# Keep ``str`` plus ``Enum`` so stringification stays compatible with existing
# configuration serialization rather than adopting ``StrEnum.__str__``.
class AttentionBackend(str, Enum):  # noqa: UP042
    EAGER = "eager"
    SDPA = "sdpa"
    FLEX_ATTENTION = "flex_attention"
    FLASH_ATTENTION_2 = "flash_attention_2"
    FLASH_ATTENTION_3 = "flash_attention_3"

    # Internal spelling retained to keep attention modules concise. It is an
    # enum alias, not an accepted public backend string.
    FLEX = FLEX_ATTENTION

    @property
    def is_flash(self) -> bool:
        return self in {
            AttentionBackend.FLASH_ATTENTION_2,
            AttentionBackend.FLASH_ATTENTION_3,
        }


VALID_ATTENTION_BACKENDS = tuple(b.value for b in AttentionBackend)


def warn_attention_backend_fallback(
    requested_backend: str | AttentionBackend,
    *,
    effective_backend: str | AttentionBackend,
    reason: str,
) -> None:
    """Warn when one forward call cannot honor the configured backend."""

    requested = resolve_attention_backend(requested_backend).value
    effective = resolve_attention_backend(effective_backend).value
    if requested == effective:
        return
    warnings.warn(
        f"{reason} The requested {requested!r} attention implementation cannot "
        f"satisfy this call, so FastPLMs is using {effective!r} attention for this "
        "call only. This can change performance and memory use; the configured "
        "backend remains unchanged for subsequent calls.",
        RuntimeWarning,
        stacklevel=3,
    )


def resolve_attention_backend(
    requested_backend: str | AttentionBackend | None,
) -> AttentionBackend:
    """Validate a backend without silently substituting another implementation."""
    if requested_backend is None:
        requested_backend = AttentionBackend.SDPA.value
    if isinstance(requested_backend, AttentionBackend):
        resolved = requested_backend
    else:
        try:
            resolved = AttentionBackend(requested_backend)
        except ValueError as error:
            raise ValueError(
                f"Unsupported attention implementation {requested_backend!r}; "
                f"expected one of {VALID_ATTENTION_BACKENDS}."
            ) from error
    if resolved == AttentionBackend.FLEX_ATTENTION and flex_attention is None:
        raise RuntimeError(
            "'flex_attention' was requested, but this PyTorch build does not provide it."
        )
    return resolved


def get_attn_implementation(config) -> str:
    """Read the Transformers attention setting, defaulting to SDPA."""
    requested = getattr(config, "_attn_implementation", None)
    if requested is None:
        requested = getattr(config, "attn_backend", None)
    return resolve_attention_backend(requested).value


def set_config_attn_implementation(config, implementation: str) -> str:
    """Set both the Transformers field and the internal dispatch field."""
    resolved = resolve_attention_backend(implementation).value
    if hasattr(config, "_attn_implementation_internal"):
        config._attn_implementation_internal = resolved
    else:
        config._attn_implementation = resolved
    # Existing checkpoint configs contain this field. Keeping it synchronized
    # preserves their state schema while the public API uses attn_implementation.
    config.attn_backend = resolved
    return resolved


@torch.compiler.disable
def get_attention_mask(
    effective_backend: AttentionBackend,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    attention_mask: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
    mask_semantics: str = "padding",
) -> tuple[torch.Tensor | None, torch.Tensor | None, BlockMask | None]:
    """Build padding masks once for all encoder layers.

    Returns (attention_mask_2d, attention_mask_4d, flex_block_mask).
    """
    if attention_mask is None:
        return None, None, None

    effective_backend = resolve_attention_backend(effective_backend)
    attention_mask_2d = attention_mask.to(device=device, dtype=torch.bool)

    if effective_backend.is_flash:
        return attention_mask_2d, None, None

    if effective_backend == AttentionBackend.FLEX_ATTENTION:
        if create_block_mask is None:
            raise RuntimeError(
                "'flex_attention' was requested, but torch.create_block_mask is unavailable."
            )
        valid_lengths = attention_mask_2d.sum(dim=-1)
        sequence_lengths = tuple(int(length) for length in valid_lengths.tolist())
        # Packed multimodal batches can contain more than one valid span. Cache
        # the exact key-validity pattern, not just row lengths, so masks with
        # equal token counts can never reuse an incompatible BlockMask.
        valid_positions = tuple(
            tuple(int(index) for index in row.nonzero(as_tuple=False).flatten().tolist())
            for row in attention_mask_2d
        )
        cache_key = (
            str(device),
            None if dtype is None else str(dtype),
            (batch_size, seq_len, seq_len),
            sequence_lengths,
            valid_positions,
            mask_semantics,
        )
        flex_block_mask = _flex_block_masks.get(cache_key)
        if flex_block_mask is not None:
            _flex_block_masks.move_to_end(cache_key)
            return attention_mask_2d, None, flex_block_mask

        def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
            del head_idx, q_idx
            # Match eager and SDPA: padding masks suppress invalid keys only.
            # Invalid queries still attend to real keys and therefore remain
            # finite; downstream residue masks exclude their outputs.
            return attention_mask_2d[batch_idx, kv_idx]

        flex_block_mask = create_block_mask(
            mask_mod, batch_size, 1, seq_len, seq_len, device=device
        )
        _remember(_flex_block_masks, cache_key, flex_block_mask)
        return attention_mask_2d, None, flex_block_mask

    # SDPA/manual masks only keys. Padding queries still attend to real keys, so
    # their outputs stay finite instead of softmaxing over all -inf scores.
    attention_mask_4d = attention_mask_2d[:, None, None, :]
    return attention_mask_2d, attention_mask_4d, None


def bool_to_additive_mask(
    bool_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert a bool mask (True = valid) to a float additive mask (0.0 valid, -inf invalid).

    Why this exists: calling `bool_mask.masked_fill(bool_mask.logical_not(), float('-inf'))`
    directly on a bool tensor returns a bool tensor because `-inf` casts to `True`.
    That silently drops the mask. Always allocate a float tensor first, then fill it.
    This helper is the sanctioned way to build an SDPA additive mask from a bool validity mask.
    """
    assert bool_mask.dtype == torch.bool, (
        f"bool_to_additive_mask requires a bool tensor, got dtype={bool_mask.dtype}"
    )
    additive = torch.zeros_like(bool_mask, dtype=dtype)
    additive.masked_fill_(bool_mask.logical_not(), float("-inf"))
    return additive
