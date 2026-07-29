"""Hugging Face-compatible ESMC models implemented by FastPLMs."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from functools import partial
from typing import ClassVar
from einops import rearrange
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import (
    MaskedLMOutput,
    ModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)


try:
    from fastplms.attention import (
        AttentionBackend,
        BlockMask,
        FastPLMsAttentionMixin,
        _get_flex_attention_fn,
        _get_flex_block_mask,
        flex_attention,
        get_attention_mask,
        kernels_flash_attention_func,
        resolve_attention_backend,
        resolve_attention_backend_for_call,
    )
    from fastplms.embeddings import EmbeddingMixin, Pooler, select_hidden_state_embeddings
    from fastplms.models.ttt import FastPLMTestTimeTrainingMixin
except ModuleNotFoundError as error:
    _COMPOSITE_REQUIRED_NAMES = (
        "AttentionBackend",
        "BlockMask",
        "EmbeddingMixin",
        "FastPLMsAttentionMixin",
        "FastPLMTestTimeTrainingMixin",
        "Pooler",
        "_get_flex_attention_fn",
        "_get_flex_block_mask",
        "flex_attention",
        "get_attention_mask",
        "kernels_flash_attention_func",
        "resolve_attention_backend",
        "resolve_attention_backend_for_call",
        "select_hidden_state_embeddings",
    )
    if error.name != "fastplms" or any(
        name not in globals() for name in _COMPOSITE_REQUIRED_NAMES
    ):
        raise
    # Legacy flat Hub composites define every shared symbol above this block.


class ESMplusplusConfig(PretrainedConfig):
    """Configuration class for ESM++ model.

    Args:
        vocab_size: Size of the vocabulary
        hidden_size: Dimension of hidden layers
        num_attention_heads: Number of attention heads
        num_hidden_layers: Number of transformer layers
        num_labels: Number of output labels for classification
        problem_type: Type of problem - regression, single/multi label classification
    """

    model_type = "ESMplusplus"

    def __init__(
        self,
        vocab_size: int = 64,
        hidden_size: int = 960,
        num_attention_heads: int = 15,
        num_hidden_layers: int = 30,
        num_labels: int | None = None,
        problem_type: str | None = None,
        dropout: float = 0.0,
        initializer_range: float = 0.02,
        classifier_dropout: float = 0.1,
        classifier_pooling_types: list[str] | None = None,
        attn_backend: str | None = None,
        pad_token_id: int = 1,
        mask_token_id: int = 32,
        **kwargs,
    ):
        if num_labels is None:
            configured_labels = kwargs.get("id2label")
            num_labels = len(configured_labels) if configured_labels else 2
        super().__init__(
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
            num_labels=num_labels,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.problem_type = problem_type
        self.dropout = dropout
        self.initializer_range = initializer_range
        self.classifier_dropout = classifier_dropout
        self.classifier_pooling_types = (
            list(classifier_pooling_types) if classifier_pooling_types is not None else None
        )
        self.tie_word_embeddings = False
        self.attn_backend = attn_backend


### Rotary Embeddings
def rotate_half(x: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
    """Rotate the final axis of X by 90 degrees in each two-dimensional plane."""
    if interleaved:
        paired = x.unflatten(-1, (-1, 2))
        return torch.stack((-paired[..., 1], paired[..., 0]), dim=-1).flatten(-2)

    # torch.chunk assigns an odd remainder to the first half. Express the same
    # public behavior explicitly while keeping the ESMC path branch-free.
    midpoint = (x.shape[-1] + 1) // 2
    return torch.cat((-x[..., midpoint:], x[..., :midpoint]), dim=-1)


def apply_rotary_emb_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
    _inplace: bool = False,
) -> torch.Tensor:
    """Apply cached rotary angles to X while preserving any unrotated features."""
    del _inplace  # Kept in the signature for checkpoint remote-code compatibility.
    rotary_width = 2 * cos.shape[-1]
    if rotary_width > x.shape[-1]:
        raise AssertionError("rotary width exceeds the attention head dimension")

    token_count = x.shape[1]
    cos_full = torch.cat((cos[:token_count], cos[:token_count]), dim=-1).unsqueeze(1)
    sin_full = torch.cat((sin[:token_count], sin[:token_count]), dim=-1).unsqueeze(1)
    x_rotary = x[..., :rotary_width]
    y_rotary = x_rotary * cos_full + rotate_half(x_rotary, interleaved) * sin_full
    if rotary_width == x.shape[-1]:
        return y_rotary
    return torch.cat((y_rotary, x[..., rotary_width:]), dim=-1)


class RotaryEmbedding(torch.nn.Module):
    """Rotary position embeddings.

    Based on the paper "RoFormer: Enhanced Transformer with Rotary Position Embedding"

    Args:
        dim: Dimension of the embedding
        base: Base for computing angular frequencies
        interleaved: Whether to use interleaved rotations
        scale_base: Base for scaling
        scaling_factor: Factor for scaling positions
        pos_idx_in_fp32: Whether to compute position indices in fp32
        device: Computation device
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        interleaved: bool = False,
        scale_base: float | None = None,
        scaling_factor: float = 1.0,
        pos_idx_in_fp32: bool = True,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.dim, self.base = dim, float(base)
        self.interleaved, self.scale_base = interleaved, scale_base
        self.scaling_factor, self.pos_idx_in_fp32 = scaling_factor, pos_idx_in_fp32
        self.device = device
        self._clear_cache()
        self.reset_parameters()

    def _clear_cache(self) -> None:
        self._seq_len_cached = 0
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None
        self._cos_k_cached: torch.Tensor | None = None
        self._sin_k_cached: torch.Tensor | None = None

    def reset_parameters(self, device: torch.device | str | None = None) -> None:
        """Rebuild the non-persistent frequency buffers on ``device``."""
        if device is not None:
            buffer_device = torch.device(device)
        elif "inv_freq" in self._buffers and isinstance(self._buffers["inv_freq"], torch.Tensor):
            buffer_device = self._buffers["inv_freq"].device
        else:
            buffer_device = self.device
        inv_freq = self._compute_inv_freq(buffer_device)
        self._clear_cache()
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        arange = torch.arange(0, self.dim, 2, device=buffer_device, dtype=torch.float32)
        scale = (
            (arange + 0.4 * self.dim) / (1.4 * self.dim) if self.scale_base is not None else None
        )
        self.register_buffer("scale", scale)

    def _compute_inv_freq(self, device: torch.device | None = None) -> torch.Tensor:
        """Compute inverse frequency bands on their execution device."""
        return 1 / (
            self.base
            ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )

    def _apply(self, fn, recurse: bool = True):
        """Move the module, then regenerate device-specific RoPE frequencies."""
        if self.inv_freq.is_meta:
            self.reset_parameters(device="cpu")
        result = super()._apply(fn, recurse=recurse)
        self.register_buffer(
            "inv_freq",
            self._compute_inv_freq(self.inv_freq.device),
            persistent=False,
        )
        self._clear_cache()
        return result

    def _cache_is_current(
        self,
        token_count: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> bool:
        cached = self._cos_cached
        return (
            cached is not None
            and self._seq_len_cached >= token_count
            and cached.device == device
            and cached.dtype == dtype
            and not (self.training and cached.is_inference())
        )

    def _rotary_angles(
        self,
        token_count: int,
        device: torch.device | None,
    ) -> torch.Tensor:
        position_dtype = torch.float32 if self.pos_idx_in_fp32 else self.inv_freq.dtype
        positions = torch.arange(token_count, device=device, dtype=position_dtype)  # (l,)
        positions.div_(self.scaling_factor)
        frequencies = (
            self.inv_freq.to(torch.float32)
            if self.pos_idx_in_fp32 and self.inv_freq.dtype != torch.float32
            else self.inv_freq
        )
        return torch.outer(positions, frequencies)  # (l, d / 2)

    def _update_cos_sin_cache(
        self, seqlen: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        """Build angle tables when the requested cache identity has changed."""
        if self._cache_is_current(seqlen, device, dtype):
            return

        self._seq_len_cached = seqlen
        angles = self._rotary_angles(seqlen, device)  # (l, d / 2)
        cos_angles = torch.cos(angles)  # (l, d / 2)
        sin_angles = torch.sin(angles)  # (l, d / 2)
        if self.scale is None:
            self._cos_cached = cos_angles.to(dtype)
            self._sin_cached = sin_angles.to(dtype)
            return

        centered_positions = (
            torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device) - seqlen // 2
        ) / self.scale_base
        scale = self.scale ** centered_positions.unsqueeze(-1)
        self._cos_cached = (cos_angles * scale).to(dtype)
        self._sin_cached = (sin_angles * scale).to(dtype)
        self._cos_k_cached = (cos_angles / scale).to(dtype)
        self._sin_k_cached = (sin_angles / scale).to(dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to queries and keys.

        Args:
            q: Query tensor Q with shape (b, l, h, d).
            k: Key tensor K with shape (b, l, h, d).

        Returns:
            Tuple of rotated query and key tensors
        """
        # The pinned Biohub Transformers oracle recomputes inverse frequencies
        # on the execution device. CPU and CUDA differ by about one FP32 ULP in
        # some bands, which is immaterial in BF16 but accumulates measurably in
        # deep FP32 execution.
        self._update_cos_sin_cache(q.shape[1], device=q.device, dtype=q.dtype)
        if self._cos_cached is None or self._sin_cached is None:
            raise RuntimeError(
                "Rotary cache initialization did not produce cosine and sine values."
            )
        if self.scale is not None:
            raise AssertionError("Scaled rotary embeddings are unsupported for ESMC.")

        cos_angles = self._cos_cached
        sin_angles = self._sin_cached
        return (
            apply_rotary_emb_torch(q, cos_angles, sin_angles, self.interleaved, True),
            apply_rotary_emb_torch(k, cos_angles, sin_angles, self.interleaved, True),
        )


def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    """Compute corrected dimension for SwiGLU."""
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class SwiGLU(nn.Module):
    """SwiGLU activation function."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


def swiglu_ln_ffn(d_model: int, expansion_ratio: float) -> nn.Sequential:
    """Create SwiGLU feedforward network with layer normalization."""
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, swiglu_correction_fn(expansion_ratio, d_model) * 2, bias=False),
        SwiGLU(),
        nn.Linear(swiglu_correction_fn(expansion_ratio, d_model), d_model, bias=False),
    )


class MultiHeadAttention(nn.Module):
    """Multi-head attention with rotary embeddings and configurable backend.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        attn_backend: One of "eager", "sdpa", or "flex_attention".
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_backend: str = "sdpa",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = self.d_model // self.n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.attn_backend = resolve_attention_backend(attn_backend)
        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 3, bias=False)
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_ln = nn.LayerNorm(d_model, bias=False)
        self.k_ln = nn.LayerNorm(d_model, bias=False)
        self.reshaper = partial(rearrange, pattern="b s (h d) -> b h s d", h=n_heads)
        self.rotary = RotaryEmbedding(d_model // n_heads)

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k: (b, l, d)
        q = q.unflatten(-1, (self.n_heads, self.d_head))  # (b, l, h, d_h)
        k = k.unflatten(-1, (self.n_heads, self.d_head))  # (b, l, h, d_h)
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(
        self,
        x: torch.Tensor,
        attention_mask_2d: torch.Tensor | None = None,
        attention_mask_4d: torch.Tensor | None = None,
        flex_block_mask: BlockMask | None = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        # x: (b, l, d)
        qkv = self.layernorm_qkv(x)  # (b, l, 3 * d)
        query_sequence, key_sequence, value_sequence = torch.chunk(qkv, 3, dim=-1)
        query_sequence, key_sequence = (
            self.q_ln(query_sequence).to(query_sequence.dtype),
            self.k_ln(key_sequence).to(query_sequence.dtype),
        )
        query_sequence, key_sequence = self._apply_rotary(query_sequence, key_sequence)
        query_heads, key_heads, value_heads = map(
            self.reshaper, (query_sequence, key_sequence, value_sequence)
        )  # each (b, h, l, d_h)

        attn_output, attn_weights, s_max = self._attn(
            query_heads,
            key_heads,
            value_heads,
            attention_mask_2d=attention_mask_2d,
            attention_mask_4d=attention_mask_4d,
            flex_block_mask=flex_block_mask,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
        )

        output = self.out_proj(attn_output)
        return output, attn_weights, s_max

    def _attn(
        self,
        query_heads: torch.Tensor,
        key_heads: torch.Tensor,
        value_heads: torch.Tensor,
        attention_mask_2d: torch.Tensor | None = None,
        attention_mask_4d: torch.Tensor | None = None,
        flex_block_mask: BlockMask | None = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        if output_attentions:
            return self._manual_attn(
                query_heads, key_heads, value_heads, attention_mask_4d, output_s_max
            )

        if self.attn_backend == AttentionBackend.EAGER:
            attn_output, _, s_max = self._manual_attn(
                query_heads, key_heads, value_heads, attention_mask_4d, output_s_max
            )
            return attn_output, None, s_max
        if self.attn_backend.is_flash:
            attn_output, attn_weights = self._kernels_flash_attn(
                query_heads, key_heads, value_heads, attention_mask_2d
            )
        elif self.attn_backend == AttentionBackend.FLEX:
            attn_output, attn_weights = self._flex_attn(
                query_heads,
                key_heads,
                value_heads,
                flex_block_mask,
                attention_mask_2d,
            )
        elif self.attn_backend == AttentionBackend.SDPA:
            attn_output, attn_weights = self._sdpa_attn(
                query_heads, key_heads, value_heads, attention_mask_4d
            )
        else:
            raise AssertionError(f"Unsupported resolved backend: {self.attn_backend}")

        s_max = self._compute_s_max(query_heads, key_heads) if output_s_max else None
        return attn_output, attn_weights, s_max

    @torch.no_grad()
    def _compute_s_max(
        self, query_heads: torch.Tensor, key_heads: torch.Tensor
    ) -> list[torch.Tensor]:
        q_norm = torch.linalg.vector_norm(query_heads, dim=-1)  # (b, h, l)
        k_norm = torch.linalg.vector_norm(key_heads, dim=-1)  # (b, h, l)
        s_max_bound = (q_norm.max(dim=-1).values * k_norm.max(dim=-1).values).max(
            dim=0
        ).values * self.scale
        return [s_max_bound[h] for h in range(self.n_heads)]

    def _manual_attn(
        self,
        query_heads: torch.Tensor,
        key_heads: torch.Tensor,
        value_heads: torch.Tensor,
        attention_mask_4d: torch.Tensor | None = None,
        output_s_max: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor] | None]:
        # query_heads, key_heads, value_heads: (b, h, l, d_h)
        attn_weights = (
            torch.matmul(query_heads, key_heads.transpose(-2, -1)) * self.scale
        )  # (b, h, l, l)
        if attention_mask_4d is not None:
            attn_weights = attn_weights.masked_fill(attention_mask_4d.logical_not(), float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        context_heads = torch.matmul(attn_weights, value_heads)  # (b, h, l, d_h)
        attn_output = rearrange(context_heads, "b h s d -> b s (h d)")  # (b, l, d)
        s_max = self._compute_s_max(query_heads, key_heads) if output_s_max else None
        return attn_output, attn_weights, s_max

    def _kernels_flash_attn(
        self,
        query_heads: torch.Tensor,
        key_heads: torch.Tensor,
        value_heads: torch.Tensor,
        attention_mask_2d: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        query_tokens = query_heads.transpose(1, 2).contiguous()
        key_tokens = key_heads.transpose(1, 2).contiguous()
        value_tokens = value_heads.transpose(1, 2).contiguous()
        attn_output = kernels_flash_attention_func(
            query_states=query_tokens,
            key_states=key_tokens,
            value_states=value_tokens,
            attention_mask_2d=attention_mask_2d,
            causal=False,
            implementation=self.attn_backend.value,
        )
        return rearrange(attn_output, "b s h d -> b s (h d)"), None

    def _flex_attn(
        self,
        query_heads: torch.Tensor,
        key_heads: torch.Tensor,
        value_heads: torch.Tensor,
        flex_block_mask: BlockMask | None = None,
        attention_mask_2d: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        if flex_attention is None:
            raise RuntimeError("Flex attention is not available in this environment.")
        fn = _get_flex_attention_fn(
            device=query_heads.device,
            dtype=query_heads.dtype,
            shape=tuple(query_heads.shape),
            mask_semantics="padding",
        )
        context_heads = fn(
            query_heads,
            key_heads,
            value_heads,
            block_mask=flex_block_mask,
            scale=self.scale,
            kernel_options={"PRESCALE_QK": True, "BLOCK_N": 32},
        )
        return rearrange(context_heads, "b h s d -> b s (h d)"), None

    def _sdpa_attn(
        self,
        query_heads: torch.Tensor,
        key_heads: torch.Tensor,
        value_heads: torch.Tensor,
        attention_mask_4d: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        context_heads = F.scaled_dot_product_attention(
            query_heads,
            key_heads,
            value_heads,
            attn_mask=attention_mask_4d,
            scale=self.scale,
        )
        return rearrange(context_heads, "b h s d -> b s (h d)"), None


def RegressionHead(d_model: int, output_dim: int, hidden_dim: int | None = None) -> nn.Module:
    """Create a regression head with optional hidden dimension.

    Args:
        d_model: Input dimension
        output_dim: Output dimension
        hidden_dim: Optional hidden dimension (defaults to d_model)
    """
    hidden_dim = hidden_dim if hidden_dim is not None else d_model
    return nn.Sequential(
        nn.Linear(d_model, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, output_dim),
    )


class UnifiedTransformerBlock(nn.Module):
    """Transformer block with attention and feedforward layers."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        residue_scaling_factor: float = 1,
        expansion_ratio: float = 8 / 3,
        dropout: float = 0.0,
        attn_backend: str = "sdpa",
    ) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, attn_backend=attn_backend)
        self.ffn = swiglu_ln_ffn(d_model, expansion_ratio)
        self.scaling_factor = residue_scaling_factor
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask_2d: torch.Tensor | None = None,
        attention_mask_4d: torch.Tensor | None = None,
        flex_block_mask: BlockMask | None = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        attn_output, attn_weights, s_max = self.attn(
            x,
            attention_mask_2d=attention_mask_2d,
            attention_mask_4d=attention_mask_4d,
            flex_block_mask=flex_block_mask,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
        )
        x = x + self.dropout(attn_output) / self.scaling_factor
        x = x + self.dropout(self.ffn(x)) / self.scaling_factor
        return x, attn_weights, s_max


@dataclass
class TransformerOutput(ModelOutput):
    """Output type for transformer encoder."""

    last_hidden_state: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor] | None = None
    attentions: tuple[torch.Tensor] | None = None
    s_max: tuple[list[torch.Tensor], ...] | None = None


@dataclass
class ESMplusplusOutput(MaskedLMOutput):
    """Masked-LM output with FastPLMs fields after the HF contract."""

    s_max: tuple[list[torch.Tensor], ...] | None = None
    last_hidden_state: torch.Tensor | None = None


@dataclass
class ESMplusplusSequenceClassifierOutput(SequenceClassifierOutput):
    """Sequence-classification output with optional attention diagnostics."""

    s_max: tuple[list[torch.Tensor], ...] | None = None


@dataclass
class ESMplusplusTokenClassifierOutput(TokenClassifierOutput):
    """Token-classification output with optional attention diagnostics."""

    s_max: tuple[list[torch.Tensor], ...] | None = None


class TransformerStack(nn.Module):
    """Stack of transformer blocks."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.0,
        attn_backend: str = "sdpa",
    ) -> None:
        super().__init__()
        self.attention_backend = resolve_attention_backend(attn_backend)
        self.blocks = nn.ModuleList(
            [
                UnifiedTransformerBlock(
                    d_model,
                    n_heads,
                    residue_scaling_factor=math.sqrt(n_layers / 36),
                    dropout=dropout,
                    attn_backend=attn_backend,
                )
                for i in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model, bias=False)
        self.gradient_checkpointing = False

    @property
    def attn_backend(self) -> AttentionBackend:
        return self.attention_backend

    @attn_backend.setter
    def attn_backend(self, backend: str) -> None:
        resolved = resolve_attention_backend(backend)
        self.attention_backend = resolved
        for block in self.blocks:
            block.attn.attn_backend = resolved

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        output_hidden_states: bool | None = False,
        output_attentions: bool | None = False,
        output_s_max: bool | None = False,
        esmfold2_hidden_states: bool = False,
    ) -> TransformerOutput:
        # x: (b, l, d); attention_mask, sequence_id: (b, l)
        hidden_states = () if output_hidden_states else None
        attentions = () if output_attentions else None
        full_s_max = () if output_s_max else None
        # Match the pinned Biohub Transformers contract: a supplied sequence_id
        # is authoritative and must encode padding as -1.  attention_mask is
        # ignored in that mode rather than intersected with the chain mask.
        attention_mask_2d, attention_mask_4d, flex_block_mask = (
            self._prepare_attention_masks(
                attention_mask=attention_mask,
                sequence_id=sequence_id,
                batch_size=x.shape[0],
                seq_len=x.shape[1],
                device=x.device,
                dtype=x.dtype,
                output_attentions=bool(output_attentions),
            )
        )

        for block in self.blocks:
            if output_hidden_states:
                if hidden_states is None:
                    raise RuntimeError(
                        "Hidden-state collection was not initialized for an enabled request."
                    )
                # Biohub Transformers records the input to each block followed
                # by the final normalized state. This gives n_layers + 1 states
                # and, for ESMC-6B, the 81-state order consumed by ESMFold2.
                hidden_states += (x,)
            if self.gradient_checkpointing and self.training:
                x, attn_weights, s_max = self._gradient_checkpointing_func(
                    block.__call__,
                    x=x,
                    attention_mask_2d=attention_mask_2d,
                    attention_mask_4d=attention_mask_4d,
                    flex_block_mask=flex_block_mask,
                    output_attentions=output_attentions,
                    output_s_max=output_s_max,
                )
            else:
                x, attn_weights, s_max = block(
                    x=x,
                    attention_mask_2d=attention_mask_2d,
                    attention_mask_4d=attention_mask_4d,
                    flex_block_mask=flex_block_mask,
                    output_attentions=output_attentions,
                    output_s_max=output_s_max,
                )

            if attentions is not None:
                attentions += (attn_weights,)
            if full_s_max is not None:
                full_s_max += (s_max,)

        last_hidden_state = self.norm(x)
        if output_hidden_states:
            hidden_states += (last_hidden_state,)

        return TransformerOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=hidden_states,
            attentions=attentions,
            s_max=full_s_max,
        )

    @torch.compiler.disable
    def _prepare_attention_masks(
        self,
        attention_mask: torch.Tensor | None,
        sequence_id: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
        effective_backend: AttentionBackend | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, BlockMask | None]:
        mask_name = "sequence_id" if sequence_id is not None else "attention_mask"
        mask_pattern = sequence_id if sequence_id is not None else attention_mask
        if mask_pattern is None:
            backend = resolve_attention_backend_for_call(
                self.attention_backend,
                output_attentions=output_attentions,
            )
            return get_attention_mask(
                effective_backend=backend,
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
                attention_mask=None,
                dtype=dtype,
                mask_semantics="padding",
            )

        expected_shape = (batch_size, seq_len)
        if mask_pattern.ndim != 2 or tuple(mask_pattern.shape) != expected_shape:
            raise ValueError(
                f"{mask_name} must have shape {expected_shape}; "
                f"received {tuple(mask_pattern.shape)}."
            )
        if mask_pattern.device != device:
            mask_pattern = mask_pattern.to(device=device)
        if sequence_id is None:
            mask_pattern = mask_pattern.to(dtype=torch.bool)
        attention_mask_2d = (
            mask_pattern if mask_pattern.dtype == torch.bool else mask_pattern != -1
        )
        if not bool(attention_mask_2d.any(dim=1).all()):
            raise ValueError("attention_mask must keep at least one valid key per batch row.")

        if mask_pattern.dtype == torch.bool:
            # Biohub's boolean single-chain form groups biological positions
            # together and padding positions together. Padding queries remain
            # finite without allowing their states to enter residue attention.
            attention_mask_4d = (
                mask_pattern[:, None, :, None] == mask_pattern[:, None, None, :]
            )
        else:
            attention_mask_4d = (
                mask_pattern.unsqueeze(-1) == mask_pattern.unsqueeze(-2)
            ).unsqueeze(1)
        backend = (
            resolve_attention_backend_for_call(
                self.attention_backend,
                output_attentions=output_attentions,
            )
            if effective_backend is None
            else resolve_attention_backend(effective_backend)
        )

        if backend.is_flash:
            if mask_pattern.dtype != torch.bool:
                raise ValueError(
                    "ESM++ FlashAttention only supports boolean sequence_id padding masks. "
                    "Use eager, sdpa, or flex_attention for chain-aware integer sequence_id "
                    "masks."
                )
            return attention_mask_2d, attention_mask_4d, None

        if backend == AttentionBackend.FLEX:
            if mask_pattern.dtype == torch.bool:

                def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
                    del head_idx
                    return mask_pattern[batch_idx, q_idx] == mask_pattern[batch_idx, kv_idx]

            else:

                def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
                    del head_idx
                    q_id = mask_pattern[batch_idx, q_idx]
                    kv_id = mask_pattern[batch_idx, kv_idx]
                    return q_id == kv_id

            flex_block_mask = _get_flex_block_mask(
                mask_pattern=mask_pattern,
                batch_size=batch_size,
                query_length=seq_len,
                key_value_length=seq_len,
                device=device,
                dtype=dtype,
                mask_semantics=(
                    "boolean_sequence_id"
                    if mask_pattern.dtype == torch.bool
                    else "integer_sequence_id"
                ),
                mask_mod=mask_mod,
            )
            return attention_mask_2d, attention_mask_4d, flex_block_mask

        return attention_mask_2d, attention_mask_4d, None


class PreTrainedESMplusplusModel(FastPLMsAttentionMixin, PreTrainedModel):
    """
    init weights for ESM++ models
    """

    config_class = ESMplusplusConfig
    base_model_prefix = "esm++"
    supports_gradient_checkpointing = True
    all_tied_weights_keys: ClassVar[dict[str, str]] = {}
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_flash_attn_3 = True
    _fastplms_attention_implementations = (
        "eager",
        "sdpa",
        "flex_attention",
        "flash_attention_2",
        "flash_attention_3",
    )

    @property
    def tokenizer(self) -> EsmSequenceTokenizer:
        """Construct the sequence tokenizer only when a raw-sequence API needs it."""

        tokenizer = self.__dict__.get("_fastplms_tokenizer")
        if tokenizer is None:
            tokenizer = EsmSequenceTokenizer()
            self.__dict__["_fastplms_tokenizer"] = tokenizer
        return tokenizer

    @tokenizer.setter
    def tokenizer(self, value: EsmSequenceTokenizer | None) -> None:
        self.__dict__["_fastplms_tokenizer"] = value

    def _init_weights(self, module):
        """Initialize the weights"""
        # HF from_pretrained marks loaded parameters with `_is_hf_initialized`.
        # Skip this module if any local parameter is already marked as loaded.
        for parameter in module.parameters(recurse=False):
            if parameter.__dict__.get("_is_hf_initialized"):
                return

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    @property
    def attn_backend(self) -> str:
        return self.config.attn_backend

    @attn_backend.setter
    def attn_backend(self, backend: str) -> None:
        if backend not in self._fastplms_attention_implementations:
            raise ValueError(
                f"{type(self).__name__} does not support {backend!r}; expected one of "
                f"{self._fastplms_attention_implementations}."
            )
        self.set_attn_implementation(backend)

    def _reset_rotary_embeddings(self):
        """Refresh non-persistent rotary buffers after checkpoint loading."""
        for module in self.modules():
            if isinstance(module, RotaryEmbedding):
                module.reset_parameters()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        output_loading_info = (
            bool(kwargs["output_loading_info"]) if "output_loading_info" in kwargs else False
        )
        loaded = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        if output_loading_info:
            model, loading_info = loaded
            model._reset_rotary_embeddings()
            return model, loading_info
        loaded._reset_rotary_embeddings()
        return loaded


### ESM++ Models
class ESMplusplusModel(PreTrainedESMplusplusModel, EmbeddingMixin):
    """
    ESM++ transformer backbone.

    Official ESM++ checkpoints contain the sequence head even when loaded through
    ``AutoModel``.  Keep that module in the base class so the checkpoint has one
    exact state-dict contract across ``AutoModel`` and ``AutoModelForMaskedLM``;
    the base forward path intentionally does not compute or return logits.
    """

    config_class = ESMplusplusConfig

    def __init__(self, config: ESMplusplusConfig, **kwargs) -> None:
        PreTrainedESMplusplusModel.__init__(self, config, **kwargs)
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed = nn.Embedding(self.vocab_size, config.hidden_size)
        self.transformer = TransformerStack(
            d_model=config.hidden_size,
            n_heads=config.num_attention_heads,
            n_layers=config.num_hidden_layers,
            dropout=config.dropout,
            attn_backend=config.attn_backend,
        )
        self.sequence_head = RegressionHead(config.hidden_size, self.vocab_size)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def get_output_embeddings(self):
        return self.sequence_head[-1]

    def set_output_embeddings(self, new_embeddings):
        self.sequence_head[-1] = new_embeddings

    def _embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        hidden_state_index: int = -1,
        store_all_hidden_states: bool = False,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        x = self.embed(input_ids)
        output_hidden_states = store_all_hidden_states or hidden_state_index != -1
        output = self.transformer(
            x=x,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=False,
        )
        return select_hidden_state_embeddings(
            output.last_hidden_state,
            output.hidden_states,
            hidden_state_index=hidden_state_index,
            store_all_hidden_states=store_all_hidden_states,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_s_max: bool | None = False,
        esmfold2_hidden_states: bool = False,
        return_dict: bool | None = None,
    ) -> TransformerOutput | tuple[torch.Tensor, ...]:
        """Run ESMC inference with the pinned Biohub mask precedence.

        ``sequence_id`` is authoritative when supplied: non-negative integers
        identify chains and ``-1`` identifies padding.  In that mode
        ``attention_mask`` is ignored, matching the official implementation.
        Without ``sequence_id``, ``attention_mask`` is the ordinary padding
        mask and defaults to ``input_ids != pad_token_id``.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is None and sequence_id is None and input_ids is not None:
            attention_mask = input_ids.ne(self.config.pad_token_id)

        x = self.embed(input_ids) if inputs_embeds is None else inputs_embeds

        transformer_output = self.transformer(
            x=x,
            attention_mask=attention_mask,
            sequence_id=sequence_id,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
            esmfold2_hidden_states=esmfold2_hidden_states,
        )
        result = TransformerOutput(
            last_hidden_state=transformer_output.last_hidden_state,
            hidden_states=transformer_output.hidden_states,
            attentions=transformer_output.attentions,
            s_max=transformer_output.s_max,
        )
        return result if return_dict else result.to_tuple()


class ESMplusplusForMaskedLM(
    FastPLMTestTimeTrainingMixin, PreTrainedESMplusplusModel, EmbeddingMixin
):
    """
    ESM++ model for masked language modeling.
    Implements the base ESM++ architecture with a masked language modeling head.
    """

    config_class = ESMplusplusConfig

    def __init__(self, config: ESMplusplusConfig, **kwargs) -> None:
        PreTrainedESMplusplusModel.__init__(self, config, **kwargs)
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed = nn.Embedding(self.vocab_size, config.hidden_size)
        self.transformer = TransformerStack(
            d_model=config.hidden_size,
            n_heads=config.num_attention_heads,
            n_layers=config.num_hidden_layers,
            dropout=config.dropout,
            attn_backend=config.attn_backend,
        )
        self.sequence_head = RegressionHead(config.hidden_size, self.vocab_size)
        self.ce_loss = nn.CrossEntropyLoss()
        self.init_weights()
        self.init_ttt({"lora_target_replace_module": "MultiHeadAttention"})

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def get_output_embeddings(self):
        return self.sequence_head[-1]

    def set_output_embeddings(self, new_embeddings):
        self.sequence_head[-1] = new_embeddings

    def _embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        hidden_state_index: int = -1,
        store_all_hidden_states: bool = False,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        x = self.embed(input_ids)
        output_hidden_states = store_all_hidden_states or hidden_state_index != -1
        output = self.transformer(
            x=x,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=False,
        )
        return select_hidden_state_embeddings(
            output.last_hidden_state,
            output.hidden_states,
            hidden_state_index=hidden_state_index,
            store_all_hidden_states=store_all_hidden_states,
        )

    def _ttt_get_trainable_modules(self) -> list[nn.Module]:
        return [self.transformer]

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_s_max: bool | None = False,
        esmfold2_hidden_states: bool = False,
        return_dict: bool | None = None,
        compute_logits: bool = True,
    ) -> ESMplusplusOutput | tuple[torch.Tensor, ...]:
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if labels is not None and not compute_logits:
            raise ValueError("labels require compute_logits=True.")
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if attention_mask is None and sequence_id is None and input_ids is not None:
            attention_mask = input_ids.ne(self.config.pad_token_id)

        x = self.embed(input_ids) if inputs_embeds is None else inputs_embeds

        output = self.transformer(
            x=x,
            attention_mask=attention_mask,
            sequence_id=sequence_id,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
            esmfold2_hidden_states=esmfold2_hidden_states,
        )

        last_hidden_state = output.last_hidden_state
        logits = self.sequence_head(last_hidden_state) if compute_logits else None
        loss = None
        if labels is not None:
            if logits is None:
                raise ValueError("labels require compute_logits=True.")
            labels = labels.to(logits.device)
            loss = self.ce_loss(logits.view(-1, self.vocab_size), labels.view(-1))

        result = ESMplusplusOutput(
            loss=loss,
            logits=logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            s_max=output.s_max,
            last_hidden_state=last_hidden_state,
        )
        return result if return_dict else result.to_tuple()


class ESMplusplusForSequenceClassification(ESMplusplusForMaskedLM, EmbeddingMixin):
    """
    ESM++ model for sequence classification.
    Extends the base ESM++ model with a classification head.
    """

    def __init__(self, config: ESMplusplusConfig, **kwargs) -> None:
        pooling_types = kwargs.pop("pooling_types", None)
        if pooling_types is None:
            pooling_types = config.classifier_pooling_types or ["mean", "var"]
        elif not isinstance(pooling_types, list):
            raise TypeError("pooling_types must be a non-empty list of strings.")
        elif not pooling_types:
            raise ValueError("pooling_types must contain at least one pooling operation.")
        elif not all(isinstance(pooling_type, str) for pooling_type in pooling_types):
            raise TypeError("pooling_types must be a non-empty list of strings.")
        if "parti" in pooling_types:
            raise ValueError(
                "pooling_types cannot contain 'parti' for sequence classification "
                "because the classifier does not expose layer attentions to its pooler."
            )
        config.classifier_pooling_types = list(pooling_types)

        ESMplusplusForMaskedLM.__init__(self, config, **kwargs)
        self.config = config
        self.num_labels = config.num_labels
        self.classifier = RegressionHead(
            config.hidden_size * len(pooling_types),
            config.num_labels,
            config.hidden_size * 4,
        )
        # Large intermediate projections help with sequence classification tasks (*4)
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.pooler = Pooler(pooling_types)
        self.init_weights()

    def _embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        hidden_state_index: int = -1,
        store_all_hidden_states: bool = False,
    ) -> torch.Tensor:
        x = self.embed(input_ids)
        output_hidden_states = store_all_hidden_states or hidden_state_index != -1
        output = self.transformer(
            x=x,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=False,
        )
        return select_hidden_state_embeddings(
            output.last_hidden_state,
            output.hidden_states,
            hidden_state_index=hidden_state_index,
            store_all_hidden_states=store_all_hidden_states,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_s_max: bool | None = False,
        return_dict: bool | None = None,
    ) -> ESMplusplusSequenceClassifierOutput | tuple[torch.Tensor, ...]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        pooling_mask = attention_mask
        if pooling_mask is None:
            if sequence_id is not None:
                pooling_mask = (
                    sequence_id if sequence_id.dtype == torch.bool else sequence_id.ne(-1)
                )
            elif input_ids is not None:
                pooling_mask = input_ids.ne(self.config.pad_token_id)
            else:
                if inputs_embeds is None:
                    raise ValueError("You have to specify either input_ids or inputs_embeds")
                pooling_mask = torch.ones(
                    inputs_embeds.shape[:2],
                    dtype=torch.bool,
                    device=inputs_embeds.device,
                )

        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sequence_id=sequence_id,
            inputs_embeds=inputs_embeds,
            labels=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_s_max=output_s_max,
            return_dict=True,
            compute_logits=False,
        )

        last_hidden_state = output.last_hidden_state
        features = self.pooler(last_hidden_state, pooling_mask)
        logits = self.classifier(features)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                if self.num_labels == 1:
                    loss = self.mse(logits.flatten(), labels.flatten())
                else:
                    loss = self.mse(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = self.ce(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = self.bce(logits, labels)

        result = ESMplusplusSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            s_max=output.s_max,
        )
        return result if return_dict else result.to_tuple()


class ESMplusplusForTokenClassification(ESMplusplusForMaskedLM, EmbeddingMixin):
    """
    ESM++ model for token classification.
    Extends the base ESM++ model with a token classification head.
    """

    def __init__(self, config: ESMplusplusConfig, **kwargs) -> None:
        ESMplusplusForMaskedLM.__init__(self, config, **kwargs)
        self.config = config
        self.num_labels = config.num_labels
        self.classifier = RegressionHead(
            config.hidden_size, config.num_labels, config.hidden_size * 4
        )
        # Large intermediate projections help with sequence classification tasks (*4)
        self.loss_fct = nn.CrossEntropyLoss()
        self.init_weights()

    def _embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        hidden_state_index: int = -1,
        store_all_hidden_states: bool = False,
    ) -> torch.Tensor:
        x = self.embed(input_ids)
        output_hidden_states = store_all_hidden_states or hidden_state_index != -1
        output = self.transformer(
            x,
            attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=False,
        )
        return select_hidden_state_embeddings(
            output.last_hidden_state,
            output.hidden_states,
            hidden_state_index=hidden_state_index,
            store_all_hidden_states=store_all_hidden_states,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_s_max: bool | None = False,
        return_dict: bool | None = None,
    ) -> ESMplusplusTokenClassifierOutput | tuple[torch.Tensor, ...]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sequence_id=sequence_id,
            inputs_embeds=inputs_embeds,
            labels=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_s_max=output_s_max,
            return_dict=True,
            compute_logits=False,
        )

        last_hidden_state = output.last_hidden_state
        logits = self.classifier(last_hidden_state)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        result = ESMplusplusTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            s_max=output.s_max,
        )
        return result if return_dict else result.to_tuple()


### Tokenization
SEQUENCE_VOCAB = [
    "<cls>",
    "<pad>",
    "<eos>",
    "<unk>",
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
    "X",
    "B",
    "U",
    "Z",
    "O",
    ".",
    "-",
    "|",
    "<mask>",
]


def _build_sequence_tokenizer_backend(
    *,
    unk_token: str,
    cls_token: str,
    pad_token: str,
    mask_token: str,
    eos_token: str,
    chain_break_token: str,
) -> Tokenizer:
    """Build the fixed ESMC character vocabulary and boundary-token policy."""
    vocabulary = dict(zip(SEQUENCE_VOCAB, range(len(SEQUENCE_VOCAB)), strict=True))
    backend = Tokenizer(BPE(vocabulary, merges=[], unk_token=unk_token))
    backend.add_special_tokens([cls_token, pad_token, mask_token, eos_token, chain_break_token])
    backend.post_processor = TemplateProcessing(
        single="<cls> $A <eos>",
        pair="<cls>:0 $A:0 <eos>:0 $B:1 <eos>:1",
        special_tokens=[
            ("<cls>", backend.token_to_id("<cls>")),
            ("<eos>", backend.token_to_id("<eos>")),
        ],
    )
    return backend


class EsmSequenceTokenizer(PreTrainedTokenizerFast):
    model_input_names: ClassVar[list[str]] = ["input_ids", "attention_mask"]

    def __init__(
        self,
        unk_token="<unk>",
        cls_token="<cls>",
        pad_token="<pad>",
        mask_token="<mask>",
        eos_token="<eos>",
        chain_break_token="|",
        **kwargs,
    ):
        backend = _build_sequence_tokenizer_backend(
            unk_token=unk_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            eos_token=eos_token,
            chain_break_token=chain_break_token,
        )
        self.cb_token = chain_break_token
        super().__init__(
            tokenizer_object=backend,
            unk_token=unk_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            eos_token=eos_token,
            additional_special_tokens=[chain_break_token],
            **kwargs,
        )

    # ESMC does not use BOS, so expose the sequence-start token through the HF BOS fields.
    @property
    def bos_token(self):
        return self.cls_token

    @property
    def bos_token_id(self):
        return self.cls_token_id

    @property
    def chain_break_token(self):
        return self.cb_token

    @property
    def chain_break_token_id(self):
        return self.convert_tokens_to_ids(self.chain_break_token)

    @property
    def all_token_ids(self):
        return list(range(self.vocab_size))

    @property
    def special_token_ids(self):
        return self.all_special_ids
