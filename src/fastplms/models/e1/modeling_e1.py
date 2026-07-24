from __future__ import annotations

import hashlib
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, TypedDict
from tqdm.auto import tqdm
from transformers import PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput
from transformers.utils import logging


try:
    from fastplms.attention import (
        AttentionBackend,
        BlockMask,
        FastPLMsAttentionMixin,
        resolve_attention_backend,
        resolve_attention_backend_for_call,
    )
    from fastplms.embeddings import (
        EmbeddingBatch,
        EmbeddingMixin,
        EmbeddingResult,
        Pooler,
        embed_dataset,
        select_hidden_state_embeddings,
    )
    from fastplms.models.ttt import FastPLMTestTimeTrainingMixin
except ModuleNotFoundError as error:
    _COMPOSITE_REQUIRED_NAMES = (
        "AttentionBackend",
        "BlockMask",
        "EmbeddingBatch",
        "EmbeddingMixin",
        "EmbeddingResult",
        "FastPLMsAttentionMixin",
        "FastPLMTestTimeTrainingMixin",
        "Pooler",
        "embed_dataset",
        "resolve_attention_backend",
        "resolve_attention_backend_for_call",
        "select_hidden_state_embeddings",
    )
    if error.name != "fastplms" or any(
        name not in globals() for name in _COMPOSITE_REQUIRED_NAMES
    ):
        raise
    # Legacy flat Hub composites define every shared symbol above this block.

from .attention import (  # noqa: F401
    _document_ids,
    _get_unpad_data,
    _unpad_input,
    block_mask_creator,
    block_min_max_seq_ids,
    build_block_causal_mask_4d,
    build_within_seq_mask_4d,
    create_block_causal_mask_optimized,
    create_within_seq_block_mask,
    direct_block_mask,
    doc_id_mask,
    flex_attention_func,
    get_overlapping_blocks,
    kernels_flash_attention_func,
    varlen_flex_attention_func,
)
from .cache import DynamicCache, KVCache  # noqa: F401
from .preparation import (  # noqa: F401
    BOS_TOKEN_ID,
    E1_TOKENIZER_REPO_ID,
    E1_VOCAB_SIZE,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    DataPrepConfig,
    E1BatchPreparer,
    _load_tokenizer_file,
    get_context,
    get_tokenizer,
)
from .retrieval import (  # noqa: F401
    COLABFOLD_HOST,
    DEFAULT_EMBED_MAX_TOKENS,
    DEFAULT_EMBED_SIMILARITY,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_SIMILARITY_THRESHOLDS,
    DOCKER_IMAGE,
    E1_MSA_SAMPLING_SOURCE_REVISION,
    LOWERCASE_CHARS,
    ColabFoldSearcher,
    ContextCache,
    ContextSpecification,
    E1Prediction,
    HomologueSearcher,
    IdSequence,
    IndexedSequence,
    _ColabFoldResponse,
    _E1ContextPredictor,
    _forward_for_embedding,
    _make_homologue_searcher,
    _pool_hidden_states,
    _safe_extract_tar,
    _sequence_output_dir,
    _strip_a3m_insertions,
    build_context_specifications,
    compute_ppll,
    convert_to_tensor,
    get_context_id,
    get_msa_for_sequence,
    get_num_neighbors,
    get_query_from_a3m,
    get_similarity_to_query,
    load_msa_dir,
    load_msa_from_hf,
    parse_msa,
    read_fasta_sequences,
    sample_context,
    sample_contexts_for_msa,
    sample_multiple_contexts,
    write_fasta_sequences,
)


def _get_logger():
    """Resolve the Transformers logger only when a runtime path emits a message."""

    return logging.get_logger(__name__)


_TOKENIZER_LOAD_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "fastplms_e1_tokenizer_load_context",
    default=None,
)


class E1Config(PretrainedConfig):
    model_type = "E1"
    keys_to_ignore_at_inference: ClassVar[list[str]] = ["past_key_values"]

    def __init__(  # type: ignore
        self,
        # Model architecture/initialization
        vocab_size=None,
        hidden_size=4096,
        intermediate_size=16384,
        gated_mlp=False,
        num_hidden_layers=40,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        dtype="bfloat16",
        gradient_checkpointing=False,
        no_ffn_gradient_checkpointing=False,
        use_cache=False,
        # Tokenization
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        tie_word_embeddings=False,
        # Attention implementation & rotary positional embeddings
        global_attention_every_n_layers=0,
        max_num_sequences=512,
        max_num_positions_within_seq=8192,
        max_num_positions_global=1024 * 128,
        rope_theta_within_seq=10000.0,
        rope_theta_global=100000.0,
        clip_qkv=None,
        attn_backend=None,
        **kwargs,
    ) -> None:
        super().__init__(
            pad_token_id=PAD_TOKEN_ID,
            bos_token_id=BOS_TOKEN_ID,
            eos_token_id=EOS_TOKEN_ID,
            tie_word_embeddings=tie_word_embeddings,
            dtype=dtype,
            **kwargs,
        )

        self.hidden_size = hidden_size
        if intermediate_size is None:
            intermediate_size = 3 * hidden_size if gated_mlp else 4 * hidden_size
        self.intermediate_size = intermediate_size
        self.gated_mlp = gated_mlp
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_num_positions_within_seq = max_num_positions_within_seq
        self.max_num_positions_global = max_num_positions_global

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta_within_seq = rope_theta_within_seq
        self.rope_theta_global = rope_theta_global
        self.max_num_sequences = max_num_sequences
        if clip_qkv is not None and clip_qkv <= 0:
            raise ValueError(f"clip_qkv must be positive when provided, got {clip_qkv}.")
        self.clip_qkv = clip_qkv
        self.global_attention_every_n_layers = global_attention_every_n_layers

        self.vocab_size = E1_VOCAB_SIZE
        self.gradient_checkpointing = gradient_checkpointing
        self.no_ffn_gradient_checkpointing = no_ffn_gradient_checkpointing
        if not isinstance(use_cache, bool):
            raise TypeError("use_cache must be a boolean.")
        self.use_cache = use_cache
        self.attn_backend = attn_backend

        if vocab_size is not None:
            if vocab_size < self.vocab_size:
                _get_logger().warning(
                    f"Using vocab_size {vocab_size} smaller than {self.vocab_size} "
                    "from the tokenizer contract."
                )
                self.vocab_size = vocab_size
            elif vocab_size > self.vocab_size:
                _get_logger().warning(
                    f"Using vocab_size {vocab_size} instead of smaller {self.vocab_size} "
                    "from E1 tokenizer contract."
                )
                self.vocab_size = vocab_size
        if pad_token_id is not None and pad_token_id != self.pad_token_id:
            _get_logger().warning(
                f"Ignoring pad_token_id. Using {self.pad_token_id} from E1 tokenizer contract"
            )
        if bos_token_id is not None and bos_token_id != self.bos_token_id:
            _get_logger().warning(
                f"Ignoring bos_token_id. Using {self.bos_token_id} from E1 tokenizer contract"
            )
        if eos_token_id is not None and eos_token_id != self.eos_token_id:
            _get_logger().warning(
                f"Ignoring eos_token_id. Using {self.eos_token_id} from E1 tokenizer contract"
            )


class AttentionLayerType(Enum):
    WITHIN_SEQ = "within_seq"
    GLOBAL = "global"


class AttentionArgs(TypedDict, total=False):
    within_seq_block_mask: BlockMask | None
    block_causal_block_mask: BlockMask | None
    within_seq_mask_4d: torch.Tensor | None
    block_causal_mask_4d: torch.Tensor | None


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).

    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to (batch,
    num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        # Transformers may instantiate modules on the meta device while loading
        # a checkpoint. Precomputed non-persistent buffers would then be
        # materialized without values. Empty buffers make initialization lazy
        # and deterministic on the first real-device forward.
        empty = torch.empty(0, dtype=torch.float32, device=device)
        self.register_buffer("inv_freq", empty, persistent=False)
        self.register_buffer("cos_cached", empty.clone(), persistent=False)
        self.register_buffer("sin_cached", empty.clone(), persistent=False)
        self.max_seq_len_cached = 0

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _set_sin_cos_cache(self, seq_len: int, device: torch.device) -> None:
        # Compute angles in FP32, matching the official cache constructed before
        # the model is converted to its inference dtype.
        self.max_seq_len_cached = seq_len
        inv_freq = self.base ** -(
            torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim
        )  # (d / 2,)
        self.inv_freq = inv_freq
        t = torch.arange(seq_len, device=device, dtype=torch.float32)  # (l,)
        angles = torch.outer(t, inv_freq)  # (l, d / 2)
        angles = torch.cat((angles, angles), dim=1)  # (l, d)
        self.cos_cached = angles.cos()
        self.sin_cached = angles.sin()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.LongTensor,
        seq_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k: (b, l, h, d)
        device, dtype = q.device, q.dtype
        seq_len = position_ids.max().item() + 1 if seq_len is None else seq_len

        if seq_len > self.max_seq_len_cached:
            self._set_sin_cos_cache(seq_len=seq_len, device=device)

        # Selecting by position inserts a head axis for broadcasting.
        idxs = position_ids.to(device)
        cos = self.cos_cached.to(device=device, dtype=dtype).unsqueeze(-2)[idxs]  # (b, l, 1, d)
        sin = self.sin_cached.to(device=device, dtype=dtype).unsqueeze(-2)[idxs]  # (b, l, 1, d)

        # Apply the real and imaginary parts of the rotary transform to Q and K.
        # Both halves reuse C and S, so rotate_half supplies the cross terms.
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed


class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    def __init__(self, config: E1Config, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.max_num_seqs = config.max_num_sequences
        self.clip_qkv = config.clip_qkv

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        if self.config.global_attention_every_n_layers > 0:
            self.layer_type = (
                AttentionLayerType.GLOBAL
                if (self.layer_idx + 1) % self.config.global_attention_every_n_layers == 0
                else AttentionLayerType.WITHIN_SEQ
            )
        else:
            self.layer_type = AttentionLayerType.WITHIN_SEQ

        self.rope_theta = (
            config.rope_theta_within_seq
            if self.layer_type == AttentionLayerType.WITHIN_SEQ
            else config.rope_theta_global
        )
        self.max_position_embeddings = (
            config.max_num_positions_within_seq
            if self.layer_type == AttentionLayerType.WITHIN_SEQ
            else config.max_num_positions_global
        )

        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        self.attn_backend = resolve_attention_backend(config.attn_backend)

    def prepare_qkv(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_value: DynamicCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # hidden_states: (b, l, d); position_ids: (b, l)
        bsz, q_len, _ = hidden_states.size()
        query_states: torch.Tensor = self.q_proj(hidden_states)
        key_states: torch.Tensor = self.k_proj(hidden_states)
        val_states: torch.Tensor = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        )  # (b, l, h, d_h)
        key_states = key_states.view(
            bsz, q_len, self.num_kv_heads, self.head_dim
        )  # (b, l, h_kv, d_h)
        val_states = val_states.view(
            bsz, q_len, self.num_kv_heads, self.head_dim
        )  # (b, l, h_kv, d_h)

        if self.clip_qkv is not None:
            query_states = query_states.clamp(-self.clip_qkv, self.clip_qkv)
            key_states = key_states.clamp(-self.clip_qkv, self.clip_qkv)
            val_states = val_states.clamp(-self.clip_qkv, self.clip_qkv)

        query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)

        if use_cache and past_key_value is not None:
            key_states, val_states = past_key_value.update(key_states, val_states, self.layer_idx)

        input_dtype = query_states.dtype
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_dtype("cuda")
        else:
            target_dtype = self.q_proj.weight.dtype
        if input_dtype != target_dtype:
            _get_logger().warning_once(
                f"The input hidden states seems to be silently casted in {input_dtype}. "
                f"This might be because you have upcasted embedding or layer norm layers "
                f"in {input_dtype}. We will cast back the input in {target_dtype}."
            )
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            val_states = val_states.to(target_dtype)

        return query_states, key_states, val_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        within_seq_position_ids: torch.LongTensor,
        global_position_ids: torch.LongTensor,
        sequence_ids: torch.LongTensor,
        attention_args: AttentionArgs | None = None,
        past_key_value: DynamicCache | None = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
        use_cache: bool = False,
        effective_backend: AttentionBackend | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, DynamicCache | None, list[torch.Tensor] | None]:
        # hidden_states: (b, l, d); position and sequence IDs: (b, l)
        is_cache_prefilled = (
            use_cache
            and past_key_value is not None
            and past_key_value.get_seq_length(self.layer_idx) > 0
        )

        query_states, key_states, val_states = self.prepare_qkv(
            hidden_states=hidden_states,
            position_ids=within_seq_position_ids
            if self.layer_type == AttentionLayerType.WITHIN_SEQ
            else global_position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        attn_output, attn_weights, s_max = self._attn(
            query_states=query_states,
            key_states=key_states,
            val_states=val_states,
            sequence_ids=sequence_ids,
            attention_args=attention_args,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
            is_cache_prefilled=is_cache_prefilled,
            effective_backend=effective_backend,
        )

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, past_key_value, s_max

    def _attn(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        val_states: torch.Tensor,
        sequence_ids: torch.Tensor,
        attention_args: AttentionArgs | None = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
        is_cache_prefilled: bool = False,
        effective_backend: AttentionBackend | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        # A filled cache changes the implementation shape, not the layer's
        # biological attention contract. Global layers must retain the cached
        # context, while within-sequence layers consume only the newly appended
        # sequence. This matches the pinned E1 inference implementation.
        effective_layer_type = self.layer_type

        if effective_backend is None:
            effective_backend = resolve_attention_backend_for_call(
                self.attn_backend,
                output_attentions=output_attentions,
            )
        if output_attentions:
            return self._manual_attn(
                query_states,
                key_states,
                val_states,
                sequence_ids=sequence_ids,
                attention_args=attention_args,
                effective_layer_type=effective_layer_type,
                output_s_max=output_s_max,
                is_cache_prefilled=is_cache_prefilled,
            )

        if effective_backend == AttentionBackend.EAGER:
            attn_output, _, s_max = self._manual_attn(
                query_states,
                key_states,
                val_states,
                sequence_ids=sequence_ids,
                attention_args=attention_args,
                effective_layer_type=effective_layer_type,
                output_s_max=output_s_max,
                is_cache_prefilled=is_cache_prefilled,
            )
            return attn_output, None, s_max
        if effective_backend.is_flash:
            if effective_layer_type == AttentionLayerType.WITHIN_SEQ:
                attn_output, attn_weights = self._kernels_flash_attn(
                    query_states,
                    key_states,
                    val_states,
                    sequence_ids=sequence_ids,
                    is_cache_prefilled=is_cache_prefilled,
                )
            else:
                raise ValueError(
                    "E1 global attention does not support a kernels Flash backend; "
                    "use eager, sdpa, or flex_attention."
                )
        elif effective_backend == AttentionBackend.FLEX:
            attn_output, attn_weights = self._flex_attn(
                query_states,
                key_states,
                val_states,
                sequence_ids=sequence_ids,
                attention_args=attention_args,
                effective_layer_type=effective_layer_type,
                is_cache_prefilled=is_cache_prefilled,
            )
        elif effective_backend == AttentionBackend.SDPA:
            attn_output, attn_weights = self._sdpa_attn(
                query_states,
                key_states,
                val_states,
                sequence_ids=sequence_ids,
                attention_args=attention_args,
                effective_layer_type=effective_layer_type,
                is_cache_prefilled=is_cache_prefilled,
            )
        else:
            raise AssertionError(f"Unsupported resolved backend: {effective_backend}")

        s_max_key_states = key_states
        if (
            is_cache_prefilled
            and effective_layer_type == AttentionLayerType.WITHIN_SEQ
            and query_states.shape[1] < key_states.shape[1]
        ):
            s_max_key_states = key_states[:, -query_states.shape[1] :]
        s_max = self._compute_s_max(query_states, s_max_key_states) if output_s_max else None
        return attn_output, attn_weights, s_max

    @torch.no_grad()
    def _compute_s_max(
        self,
        query_states: torch.Tensor,  # Q has shape (b, l, h, d).
        key_states: torch.Tensor,  # K has shape (b, l, h_kv, d).
    ) -> list[torch.Tensor]:
        query_heads = query_states.transpose(1, 2).contiguous()  # (b, h, l, d_h)
        key_heads = key_states.transpose(1, 2).contiguous()  # (b, h_kv, l, d_h)
        key_heads = repeat_kv(key_heads, self.num_key_value_groups)
        scale = 1.0 / (self.head_dim**0.5)
        q_norm = torch.linalg.vector_norm(query_heads, dim=-1)  # (b, h, l)
        k_norm = torch.linalg.vector_norm(key_heads, dim=-1)  # (b, h, l)
        s_max_bound = (q_norm.max(dim=-1).values * k_norm.max(dim=-1).values).max(
            dim=0
        ).values * scale
        return [s_max_bound[h] for h in range(self.num_heads)]

    def _kernels_flash_attn(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        val_states: torch.Tensor,
        sequence_ids: torch.Tensor,
        is_cache_prefilled: bool = False,
    ) -> tuple[torch.Tensor, None]:
        bsz, q_len = query_states.shape[0], query_states.shape[1]
        _, kv_len = key_states.shape[0], key_states.shape[1]

        if self.layer_type == AttentionLayerType.GLOBAL:
            q_sequence_ids = sequence_ids
            if q_len < kv_len:
                first_token_id = sequence_ids[:, 0].unsqueeze(1)
                k_sequence_ids = torch.cat(
                    [first_token_id.expand(bsz, kv_len - q_len), sequence_ids], dim=-1
                )
            else:
                k_sequence_ids = sequence_ids
        else:
            if q_len < kv_len:
                key_states = key_states[:, -q_len:]
                val_states = val_states[:, -q_len:]
            q_sequence_ids = k_sequence_ids = sequence_ids

        attn_output = kernels_flash_attention_func(
            query_states,
            key_states,
            val_states,
            q_sequence_ids=q_sequence_ids,
            k_sequence_ids=k_sequence_ids,
            causal=False,
            implementation=self.attn_backend.value,
        )
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        return attn_output, None

    def _flex_attn(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        val_states: torch.Tensor,
        sequence_ids: torch.Tensor,
        attention_args: AttentionArgs | None = None,
        effective_layer_type: AttentionLayerType = AttentionLayerType.WITHIN_SEQ,
        is_cache_prefilled: bool = False,
    ) -> tuple[torch.Tensor, None]:
        bsz, q_len = query_states.shape[0], query_states.shape[1]
        kv_len = key_states.shape[1]
        if is_cache_prefilled and q_len < kv_len:
            if effective_layer_type == AttentionLayerType.WITHIN_SEQ:
                key_states = key_states[:, -q_len:]
                val_states = val_states[:, -q_len:]
                block_mask = create_within_seq_block_mask(sequence_ids)
                outputs = flex_attention_func(
                    query_states,
                    key_states,
                    val_states,
                    block_mask=block_mask,
                    mask_semantics=effective_layer_type.value,
                )
            else:
                q_sequence_ids, k_sequence_ids = self._cached_global_sequence_ids(
                    sequence_ids,
                    kv_len,
                )
                outputs = varlen_flex_attention_func(
                    query_states,
                    key_states,
                    val_states,
                    q_sequence_ids=q_sequence_ids,
                    k_sequence_ids=k_sequence_ids,
                )
            outputs = outputs.reshape(bsz, q_len, self.hidden_size).contiguous()
            return outputs, None

        if effective_layer_type == AttentionLayerType.WITHIN_SEQ:
            block_mask = (
                attention_args["within_seq_block_mask"] if attention_args is not None else None
            )
        else:
            block_mask = (
                attention_args["block_causal_block_mask"] if attention_args is not None else None
            )
        outputs = flex_attention_func(
            query_states,
            key_states,
            val_states,
            block_mask=block_mask,
            mask_semantics=effective_layer_type.value,
        )
        outputs = outputs.reshape(bsz, q_len, self.hidden_size).contiguous()
        return outputs, None

    @staticmethod
    def _cached_global_sequence_ids(
        query_sequence_ids: torch.Tensor,
        kv_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assign cached context to the incoming query sequence.

        E1 retrieval cache hits contain one incoming sequence. The pinned
        implementation relabels the cached prefix with that sequence ID so its
        valid query tokens attend the complete cached context, while padding is
        excluded by the equality mask or packed Flex path.
        """

        q_len = query_sequence_ids.shape[1]
        cached_len = kv_len - q_len
        if cached_len < 0:
            raise ValueError(f"E1 cached KV length {kv_len} is shorter than query length {q_len}.")
        first_sequence_id = query_sequence_ids[:, :1]
        if bool(first_sequence_id.eq(-1).any()):
            raise ValueError("E1 cached queries must start with a non-padding sequence token.")
        cached_sequence_ids = first_sequence_id.expand(-1, cached_len)
        key_sequence_ids = torch.cat((cached_sequence_ids, query_sequence_ids), dim=-1)
        return query_sequence_ids, key_sequence_ids

    def _cached_attention_mask_4d(
        self,
        sequence_ids: torch.Tensor,
        kv_len: int,
        effective_layer_type: AttentionLayerType,
    ) -> torch.Tensor:
        if effective_layer_type == AttentionLayerType.WITHIN_SEQ:
            return build_within_seq_mask_4d(sequence_ids)
        query_sequence_ids, key_sequence_ids = self._cached_global_sequence_ids(
            sequence_ids,
            kv_len,
        )
        query_valid = query_sequence_ids.ne(-1)
        key_valid = key_sequence_ids.ne(-1)
        same_sequence = query_sequence_ids.unsqueeze(-1).eq(key_sequence_ids.unsqueeze(-2))
        return (same_sequence & query_valid.unsqueeze(-1) & key_valid.unsqueeze(-2)).unsqueeze(1)

    def _sdpa_attn(
        self,
        query_states: torch.Tensor,  # Q has shape (b, l, h, d).
        key_states: torch.Tensor,  # K has shape (b, l, h_kv, d).
        val_states: torch.Tensor,  # V has shape (b, l, h_kv, d).
        sequence_ids: torch.Tensor,
        attention_args: AttentionArgs | None = None,
        effective_layer_type: AttentionLayerType = AttentionLayerType.WITHIN_SEQ,
        is_cache_prefilled: bool = False,
    ) -> tuple[torch.Tensor, None]:
        bsz, q_len = query_states.shape[:2]
        kv_len = key_states.shape[1]

        if is_cache_prefilled and q_len < kv_len:
            if effective_layer_type == AttentionLayerType.WITHIN_SEQ:
                key_states = key_states[:, -q_len:]
                val_states = val_states[:, -q_len:]
            attention_mask_4d = self._cached_attention_mask_4d(
                sequence_ids,
                kv_len,
                effective_layer_type,
            )
        elif attention_args is not None:
            if effective_layer_type == AttentionLayerType.WITHIN_SEQ:
                attention_mask_4d = attention_args["within_seq_mask_4d"]
            else:
                attention_mask_4d = attention_args["block_causal_mask_4d"]
        else:
            attention_mask_4d = None

        query_heads = query_states.transpose(1, 2).contiguous()  # (b, h, l, d_h)
        key_heads = key_states.transpose(1, 2).contiguous()  # (b, h_kv, l, d_h)
        value_heads = val_states.transpose(1, 2).contiguous()  # (b, h_kv, l, d_h)
        key_heads = repeat_kv(key_heads, self.num_key_value_groups)
        value_heads = repeat_kv(value_heads, self.num_key_value_groups)
        context_heads = F.scaled_dot_product_attention(
            query_heads, key_heads, value_heads, attn_mask=attention_mask_4d
        )  # (b, h, l, d_h)
        attn_output = (
            context_heads.transpose(1, 2).reshape(bsz, q_len, self.hidden_size).contiguous()
        )
        return attn_output, None

    def _manual_attn(
        self,
        query_states: torch.Tensor,  # Q has shape (b, l, h, d).
        key_states: torch.Tensor,  # K has shape (b, l, h_kv, d).
        val_states: torch.Tensor,  # V has shape (b, l, h_kv, d).
        sequence_ids: torch.Tensor,
        attention_args: AttentionArgs | None = None,
        effective_layer_type: AttentionLayerType = AttentionLayerType.WITHIN_SEQ,
        output_s_max: bool = False,
        is_cache_prefilled: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor] | None]:
        bsz, q_len = query_states.shape[:2]
        kv_len = key_states.shape[1]

        if is_cache_prefilled and q_len < kv_len:
            if effective_layer_type == AttentionLayerType.WITHIN_SEQ:
                key_states = key_states[:, -q_len:]
                val_states = val_states[:, -q_len:]
            attention_mask_4d = self._cached_attention_mask_4d(
                sequence_ids,
                kv_len,
                effective_layer_type,
            )
        elif attention_args is not None:
            if effective_layer_type == AttentionLayerType.WITHIN_SEQ:
                attention_mask_4d = attention_args["within_seq_mask_4d"]
            else:
                attention_mask_4d = attention_args["block_causal_mask_4d"]
        else:
            attention_mask_4d = None

        query_heads = query_states.transpose(1, 2).contiguous()  # (b, h, l, d_h)
        key_heads = key_states.transpose(1, 2).contiguous()  # (b, h_kv, l, d_h)
        value_heads = val_states.transpose(1, 2).contiguous()  # (b, h_kv, l, d_h)
        key_heads = repeat_kv(key_heads, self.num_key_value_groups)
        value_heads = repeat_kv(value_heads, self.num_key_value_groups)
        scale = 1.0 / (self.head_dim**0.5)
        attn_weights = (
            torch.matmul(query_heads, key_heads.transpose(-2, -1)) * scale
        )  # (b, h, l, l)
        if attention_mask_4d is not None:
            attention_mask_4d = attention_mask_4d.to(dtype=torch.bool)
            attn_weights = attn_weights.masked_fill(
                attention_mask_4d.logical_not(),
                torch.finfo(attn_weights.dtype).min,
            )
        attn_weights = F.softmax(attn_weights, dim=-1)
        if attention_mask_4d is not None:
            attn_weights = attn_weights.masked_fill(attention_mask_4d.logical_not(), 0.0)
        context_heads = torch.matmul(attn_weights, value_heads)  # (b, h, l, d_h)
        attn_output = (
            context_heads.transpose(1, 2).reshape(bsz, q_len, self.hidden_size).contiguous()
        )
        s_max = self._compute_s_max(query_states, key_states) if output_s_max else None
        return attn_output, attn_weights, s_max


class MLP(nn.Module):
    def __init__(self, config: E1Config) -> None:
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act_fn(self.w1(hidden_states)))


class GLUMLP(nn.Module):
    def __init__(self, config: E1Config) -> None:
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        hidden_states = self.w2(hidden_states)
        return hidden_states


class FFN(nn.Module):
    def __init__(self, config: E1Config) -> None:
        super().__init__()
        mlp_cls = GLUMLP if config.gated_mlp else MLP
        self.mlp = mlp_cls(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_states)


@dataclass
class E1ModelOutputWithPast(ModelOutput):
    """E1 encoder outputs.

    ``last_hidden_state`` is H with shape (b, l, d). Optional hidden states use
    the same shape per layer, while attention tensors have shape (b, h, l, l).
    ``past_key_values`` stores the reusable K and V tensors for cached decoding.
    """

    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: DynamicCache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    s_max: tuple[list[torch.Tensor], ...] | None = None


@dataclass
class E1MaskedLMOutputWithPast(ModelOutput):
    """Masked-LM output with the standard HF fields first, then E1 diagnostics."""

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    mlm_loss: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: DynamicCache | None = None
    s_max: tuple[list[torch.Tensor], ...] | None = None


@dataclass
class E1ClassificationOutputWithPast(ModelOutput):
    """Sequence-classifier output matching HF ``SequenceClassifierOutputWithPast``."""

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: DynamicCache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    last_hidden_state: torch.FloatTensor | None = None
    s_max: tuple[list[torch.Tensor], ...] | None = None


@dataclass
class E1TokenClassificationOutputWithPast(ModelOutput):
    """Token-classifier output with the standard HF fields before E1 extensions."""

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: DynamicCache | None = None
    s_max: tuple[list[torch.Tensor], ...] | None = None


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        return torch.nn.functional.rms_norm(
            hidden_states, (self.hidden_size,), self.weight, self.variance_epsilon
        ).to(input_dtype)


class NormAttentionNorm(nn.Module):
    def __init__(self, config: E1Config, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        within_seq_position_ids: torch.LongTensor,
        global_position_ids: torch.LongTensor,
        sequence_ids: torch.LongTensor,
        attention_args: AttentionArgs | None = None,
        past_key_value: DynamicCache | None = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
        use_cache: bool = False,
        effective_backend: AttentionBackend | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        DynamicCache | None,
        list[torch.Tensor] | None,
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value, s_max = self.self_attn(
            hidden_states=hidden_states,
            within_seq_position_ids=within_seq_position_ids,
            global_position_ids=global_position_ids,
            sequence_ids=sequence_ids,
            attention_args=attention_args,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
            use_cache=use_cache,
            effective_backend=effective_backend,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return hidden_states, residual, self_attn_weights, present_key_value, s_max


class DecoderLayer(nn.Module):
    def __init__(self, config: E1Config, layer_idx: int) -> None:
        super().__init__()
        self.initializer_range = config.initializer_range
        self.hidden_size = config.hidden_size
        self.norm_attn_norm = NormAttentionNorm(config, layer_idx)
        self.ffn = FFN(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        within_seq_position_ids: torch.LongTensor,
        global_position_ids: torch.LongTensor,
        sequence_ids: torch.LongTensor,
        attention_args: AttentionArgs | None = None,
        past_key_value: DynamicCache | None = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
        use_cache: bool = False,
        effective_backend: AttentionBackend | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, DynamicCache | None, list[torch.Tensor] | None]:
        hidden_states, residual, self_attn_weights, present_key_value, s_max = self.norm_attn_norm(
            hidden_states=hidden_states,
            within_seq_position_ids=within_seq_position_ids,
            global_position_ids=global_position_ids,
            sequence_ids=sequence_ids,
            attention_args=attention_args,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
            use_cache=use_cache,
            effective_backend=effective_backend,
        )

        # Fully Connected
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, self_attn_weights, present_key_value, s_max


class E1PreTrainedModel(FastPLMsAttentionMixin, PreTrainedModel):
    config_class = E1Config
    embedding_unsupported_pooling = ("cls", "parti")
    config: E1Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules: ClassVar[list[str]] = ["DecoderLayer"]
    _transformer_layer_cls: ClassVar[list[type[nn.Module]]] = [DecoderLayer]
    _skip_keys_device_placement = "past_key_values"
    all_tied_weights_keys: ClassVar[dict[str, str]] = {}
    _supports_flash_attn_2 = False
    _supports_flash_attn_3 = False
    _fastplms_attention_implementations = ("sdpa", "flex_attention")
    _is_internal_encoder = False

    def __init__(self, config: E1Config, *args: Any, **kwargs: Any) -> None:
        super().__init__(config, *args, **kwargs)
        # The E1 agreement requires this exact attribution when an E1 model is
        # launched. Internal encoder construction is excluded so each public
        # model launch displays the attribution exactly once.
        if not self._is_internal_encoder:
            print("Profluent-E1", file=sys.stderr, flush=True)

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        *model_args: Any,
        **kwargs: Any,
    ) -> E1PreTrainedModel:
        tokenizer_token = None
        if "token" in kwargs:
            tokenizer_token = kwargs["token"]
        elif "use_auth_token" in kwargs:
            tokenizer_token = kwargs["use_auth_token"]
        load_context_token = _TOKENIZER_LOAD_CONTEXT.set(
            {
                "tokenizer_source": pretrained_model_name_or_path,
                "local_files_only": bool(kwargs.get("local_files_only", False)),
                "cache_dir": kwargs.get("cache_dir"),
                "revision": kwargs.get("revision"),
                "token": tokenizer_token,
            }
        )
        try:
            return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        finally:
            _TOKENIZER_LOAD_CONTEXT.reset(load_context_token)

    @staticmethod
    def _tokenizer_kwargs_from_config(config: E1Config) -> dict[str, Any]:
        load_context = _TOKENIZER_LOAD_CONTEXT.get()
        resolved_revision = getattr(config, "_commit_hash", None)
        if not isinstance(resolved_revision, str) or not resolved_revision.strip():
            resolved_revision = None
        if load_context is not None:
            tokenizer_kwargs = dict(load_context)
            if resolved_revision is not None:
                tokenizer_kwargs["revision"] = resolved_revision
            return tokenizer_kwargs

        tokenizer_source = None
        if isinstance(config._name_or_path, str) and len(config._name_or_path) > 0:
            tokenizer_source = config._name_or_path
        return {
            "tokenizer_source": tokenizer_source,
            "local_files_only": False,
            "cache_dir": None,
            "revision": resolved_revision,
            "token": None,
        }

    @property
    def prep_tokens(self) -> E1BatchPreparer:
        """Create E1's raw-sequence preparer only when a sequence API uses it."""

        preparer = self.__dict__.get("_fastplms_prep_tokens")
        if preparer is not None:
            return preparer
        encoder = self._modules.get("model")
        if encoder is not None and encoder is not self:
            return encoder.prep_tokens
        tokenizer_kwargs = self.__dict__.get("_fastplms_tokenizer_kwargs")
        if tokenizer_kwargs is None:
            raise RuntimeError("E1 tokenizer settings were not initialized.")
        preparer = E1BatchPreparer(
            data_prep_config=DataPrepConfig(
                max_num_sequences=self.config.max_num_sequences,
                max_num_positions_within_seq=self.config.max_num_positions_within_seq,
            ),
            **tokenizer_kwargs,
        )
        self.__dict__["_fastplms_prep_tokens"] = preparer
        return preparer

    @prep_tokens.setter
    def prep_tokens(self, value: E1BatchPreparer | None) -> None:
        self.__dict__["_fastplms_prep_tokens"] = value

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
            return
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            return

        nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding) and module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()

    def _backward_compatibility_gradient_checkpointing(self) -> None:
        if self.supports_gradient_checkpointing and getattr(
            self.config, "gradient_checkpointing", False
        ):
            self.gradient_checkpointing_enable(dict(use_reentrant=False))

    def post_init(self) -> None:
        super().post_init()

    @property
    def _device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def attn_backend(self) -> str:
        return self.config.attn_backend

    @attn_backend.setter
    def attn_backend(self, backend: str) -> None:
        if backend not in self._fastplms_attention_implementations:
            raise ValueError(
                f"E1 does not support {backend!r}; expected one of "
                f"{self._fastplms_attention_implementations}."
            )
        self.config.attn_backend = backend
        resolved = resolve_attention_backend(backend)
        for module in self.modules():
            if isinstance(module, FAST_E1_ENCODER):
                module._attn_backend = resolved
            elif isinstance(module, Attention):
                module.attn_backend = resolved


class FAST_E1_ENCODER(E1PreTrainedModel, EmbeddingMixin):
    config: E1Config
    config_class = E1Config
    _is_internal_encoder = True

    def __init__(self, config: E1Config, **kwargs) -> None:
        E1PreTrainedModel.__init__(self, config, **kwargs)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_seq_id = nn.Embedding(config.max_num_sequences, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = config.gradient_checkpointing
        self.__dict__["_fastplms_tokenizer_kwargs"] = (
            E1PreTrainedModel._tokenizer_kwargs_from_config(config)
        )
        self.__dict__["_fastplms_prep_tokens"] = None
        self._attn_backend = resolve_attention_backend(config.attn_backend)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def _embed(
        self,
        sequences: list[str],
        return_attention_mask: bool = False,
        hidden_state_index: int = -1,
        store_all_hidden_states: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        batch = self.prep_tokens.get_batch_kwargs(sequences, device=self._device)
        # The native preparer also returns training labels plus retrieval
        # descriptors. The encoder accepts only its aligned model inputs.
        encoder_batch: dict[str, torch.Tensor] = {}
        for name in (
            "input_ids",
            "within_seq_position_ids",
            "global_position_ids",
            "sequence_ids",
        ):
            value = batch[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Prepared E1 field {name!r} must be a tensor.")
            encoder_batch[name] = value
        output_hidden_states = store_all_hidden_states or hidden_state_index != -1
        output = self.forward(
            **encoder_batch,
            output_hidden_states=output_hidden_states,
            output_attentions=False,
            return_dict=True,
        )
        embeddings = select_hidden_state_embeddings(
            output.last_hidden_state,
            output.hidden_states,
            hidden_state_index=hidden_state_index,
            store_all_hidden_states=store_all_hidden_states,
        )
        if return_attention_mask:
            attention_mask = (encoder_batch["sequence_ids"] != -1).long()
            return embeddings, attention_mask
        else:
            return embeddings

    def _prepare_hidden_states(
        self,
        input_ids: torch.LongTensor | None,
        inputs_embeds: torch.FloatTensor | None,
        within_seq_position_ids: torch.LongTensor | None,
        global_position_ids: torch.LongTensor | None,
        sequence_ids: torch.LongTensor | None,
    ) -> tuple[torch.Tensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        if (input_ids is None) == (inputs_embeds is None):
            message = (
                "Must specify either input_ids or inputs_embeds"
                if input_ids is None
                else "Cannot specify both input_ids and inputs_embeds"
            )
            raise ValueError(message)

        source = input_ids if input_ids is not None else inputs_embeds
        if source is None:
            raise RuntimeError("E1 input validation did not resolve an input tensor.")
        expected_rank = 2 if input_ids is not None else 3
        if source.ndim != expected_rank:
            source_name = "input_ids" if input_ids is not None else "inputs_embeds"
            raise ValueError(
                f"{source_name} must have rank {expected_rank}; got shape {tuple(source.shape)}."
            )
        batch_size, sequence_length = source.shape[:2]
        if sequence_length == 0:
            raise ValueError("E1 inputs must contain at least one token.")
        if inputs_embeds is not None and inputs_embeds.shape[-1] != self.config.hidden_size:
            raise ValueError(
                "inputs_embeds hidden dimension must match config.hidden_size; "
                f"got {inputs_embeds.shape[-1]} and {self.config.hidden_size}."
            )
        if inputs_embeds is not None:
            default_positions = torch.arange(sequence_length, device=source.device).expand(
                batch_size,
                -1,
            )
            if within_seq_position_ids is None:
                within_seq_position_ids = default_positions
            if global_position_ids is None:
                global_position_ids = default_positions
            if sequence_ids is None:
                sequence_ids = torch.zeros_like(default_positions)

        if within_seq_position_ids is None or global_position_ids is None or sequence_ids is None:
            raise ValueError("Position and sequence IDs are required when input_ids are provided.")
        expected_shape = (batch_size, sequence_length)
        aligned_inputs = {
            "within_seq_position_ids": within_seq_position_ids,
            "global_position_ids": global_position_ids,
            "sequence_ids": sequence_ids,
        }
        for name, value in aligned_inputs.items():
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}; got {tuple(value.shape)}."
                )
        within_positions = within_seq_position_ids.long()
        global_positions = global_position_ids.long()
        sequence_numbers = sequence_ids.long()
        lowest_position, highest_position = torch.aminmax(within_positions)
        if (
            lowest_position.item() < -1
            or highest_position.item() >= self.config.max_num_positions_within_seq
        ):
            raise ValueError(
                "Position ids must be in the range "
                f"[-1, {self.config.max_num_positions_within_seq}); got max "
                f"{highest_position.item()} and min {lowest_position.item()}"
            )
        lowest_global, highest_global = torch.aminmax(global_positions)
        if (
            lowest_global.item() < -1
            or highest_global.item() >= self.config.max_num_positions_global
        ):
            raise ValueError(
                "Global position ids must be in the range "
                f"[-1, {self.config.max_num_positions_global}); got max "
                f"{highest_global.item()} and min {lowest_global.item()}"
            )
        lowest_sequence, highest_sequence = torch.aminmax(sequence_numbers)
        if lowest_sequence.item() < -1 or highest_sequence.item() >= self.config.max_num_sequences:
            raise ValueError(
                "Sequence ids must be in the range "
                f"[-1, {self.config.max_num_sequences}); got max "
                f"{highest_sequence.item()} and min {lowest_sequence.item()}"
            )

        if inputs_embeds is None:
            if input_ids is None:
                raise RuntimeError("E1 input validation lost the token ID tensor.")
            token_embeddings = self.embed_tokens(input_ids)
            inputs_embeds = token_embeddings + self.embed_seq_id(sequence_numbers.clamp_min(0))
        layer_dtype = self.layers[0].norm_attn_norm.self_attn.q_proj.weight.dtype
        target_dtype = (
            torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else layer_dtype
        )
        return (
            inputs_embeds.to(target_dtype),
            within_positions,
            global_positions,
            sequence_numbers,
        )

    def _resolve_forward_cache(
        self,
        past_key_values: DynamicCache | None,
        use_cache: bool,
    ) -> tuple[DynamicCache | None, bool]:
        checkpointing = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        if checkpointing and use_cache:
            _get_logger().warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing; "
                "setting `use_cache=False`."
            )
            use_cache = False
        if not use_cache:
            return None, False
        return past_key_values if past_key_values is not None else DynamicCache(), True

    def _build_forward_attention_args(
        self,
        sequence_ids: torch.LongTensor,
        past_key_values: DynamicCache | None,
        effective_backend: AttentionBackend,
    ) -> AttentionArgs | None:
        if past_key_values is not None and past_key_values.get_seq_length() != 0:
            return None

        use_flex = effective_backend == AttentionBackend.FLEX
        use_dense_mask = effective_backend in {
            AttentionBackend.EAGER,
            AttentionBackend.SDPA,
        }
        return AttentionArgs(
            block_causal_block_mask=(
                create_block_causal_mask_optimized(sequence_ids)
                if use_flex and self.config.global_attention_every_n_layers > 0
                else None
            ),
            within_seq_block_mask=(
                create_within_seq_block_mask(sequence_ids) if use_flex else None
            ),
            within_seq_mask_4d=(build_within_seq_mask_4d(sequence_ids) if use_dense_mask else None),
            block_causal_mask_4d=(
                build_block_causal_mask_4d(sequence_ids) if use_dense_mask else None
            ),
        )

    def _run_decoder_layers(
        self,
        hidden_states: torch.Tensor,
        within_seq_position_ids: torch.LongTensor,
        global_position_ids: torch.LongTensor,
        sequence_ids: torch.LongTensor,
        attention_args: AttentionArgs | None,
        past_key_values: DynamicCache | None,
        use_cache: bool,
        output_attentions: bool,
        output_hidden_states: bool,
        output_s_max: bool,
        effective_backend: AttentionBackend,
    ) -> E1ModelOutputWithPast:
        hidden_history: list[torch.Tensor] | None = [] if output_hidden_states else None
        attention_history: list[torch.Tensor] | None = [] if output_attentions else None
        s_max_history: list[list[torch.Tensor]] | None = [] if output_s_max else None
        next_cache: DynamicCache | None = None

        for layer in self.layers:
            if hidden_history is not None:
                hidden_history.append(hidden_states)
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                layer_output = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    within_seq_position_ids,
                    global_position_ids,
                    sequence_ids,
                    attention_args,
                    past_key_values,
                    output_attentions,
                    output_s_max,
                    use_cache,
                    effective_backend,
                )
            else:
                layer_output = layer(
                    hidden_states,
                    within_seq_position_ids=within_seq_position_ids,
                    global_position_ids=global_position_ids,
                    sequence_ids=sequence_ids,
                    attention_args=attention_args,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_s_max=output_s_max,
                    use_cache=use_cache,
                    effective_backend=effective_backend,
                )
            hidden_states, attention, layer_cache, s_max = layer_output
            if use_cache:
                past_key_values = layer_cache
                next_cache = layer_cache
            if attention_history is not None:
                if attention is None:
                    raise RuntimeError(
                        "An E1 layer did not return attention tensors when requested."
                    )
                attention_history.append(attention)
            if s_max_history is not None:
                if s_max is None:
                    raise RuntimeError(
                        "An E1 layer did not return s_max diagnostics when requested."
                    )
                s_max_history.append(s_max)

        hidden_states = self.norm(hidden_states)
        if hidden_history is not None:
            hidden_history.append(hidden_states)
        return E1ModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=tuple(hidden_history) if hidden_history is not None else None,
            attentions=tuple(attention_history) if attention_history is not None else None,
            s_max=tuple(s_max_history) if s_max_history is not None else None,
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        within_seq_position_ids: torch.LongTensor | None = None,
        global_position_ids: torch.LongTensor | None = None,
        sequence_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: DynamicCache | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_s_max: bool = False,
        return_dict: bool | None = None,
    ) -> E1ModelOutputWithPast | tuple[Any, ...]:
        """Transform token or soft embeddings H with shape (b, l, d)."""

        use_cache = (
            use_cache if use_cache is not None else bool(getattr(self.config, "use_cache", False))
        )
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        hidden_states, within_positions, global_positions, sequence_numbers = (
            self._prepare_hidden_states(
                input_ids,
                inputs_embeds,
                within_seq_position_ids,
                global_position_ids,
                sequence_ids,
            )
        )
        cache, use_cache = self._resolve_forward_cache(past_key_values, use_cache)
        effective_backend = resolve_attention_backend_for_call(
            self._attn_backend,
            output_attentions=bool(output_attentions),
        )
        attention_args = self._build_forward_attention_args(
            sequence_numbers,
            cache,
            effective_backend,
        )
        result = self._run_decoder_layers(
            hidden_states,
            within_positions,
            global_positions,
            sequence_numbers,
            attention_args,
            cache,
            use_cache,
            output_attentions,
            output_hidden_states,
            output_s_max,
            effective_backend,
        )
        if not return_dict:
            return result.to_tuple()
        return result


class E1Model(E1PreTrainedModel, EmbeddingMixin):
    config: E1Config
    config_class = E1Config

    def __init__(self, config: E1Config, **kwargs) -> None:
        E1PreTrainedModel.__init__(self, config, **kwargs)
        self.model: FAST_E1_ENCODER = FAST_E1_ENCODER(config, **kwargs)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.set_input_embeddings(value)

    def _embed(
        self, sequences: list[str], return_attention_mask: bool = False, **kwargs
    ) -> torch.Tensor:
        return self.model._embed(sequences, return_attention_mask=return_attention_mask, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        within_seq_position_ids: torch.LongTensor | None = None,
        global_position_ids: torch.LongTensor | None = None,
        sequence_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: DynamicCache | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_s_max: bool = False,
        return_dict: bool | None = None,
    ) -> E1ModelOutputWithPast | tuple[Any, ...]:
        return self.model(
            input_ids=input_ids,
            within_seq_position_ids=within_seq_position_ids,
            global_position_ids=global_position_ids,
            sequence_ids=sequence_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_s_max=output_s_max,
            return_dict=return_dict,
        )


class E1ForMaskedLM(FastPLMTestTimeTrainingMixin, E1PreTrainedModel, EmbeddingMixin):
    config: E1Config
    config_class = E1Config

    def __init__(self, config: E1Config, **kwargs) -> None:
        E1PreTrainedModel.__init__(self, config, **kwargs)
        self.model: FAST_E1_ENCODER = FAST_E1_ENCODER(config, **kwargs)
        self.vocab_size = config.vocab_size
        self.mlm_head = torch.nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size, bias=True),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps),
            nn.Linear(config.hidden_size, config.vocab_size, bias=True),
        )
        self.gradient_checkpointing = config.gradient_checkpointing
        self.post_init()
        self.init_ttt({"lora_target_replace_module": "Attention"})

    @property
    def device_mesh(self) -> torch.distributed.device_mesh.DeviceMesh:
        return self.model.device_mesh

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.set_input_embeddings(value)

    def _embed(
        self, sequences: list[str], return_attention_mask: bool = False, **kwargs
    ) -> torch.Tensor:
        return self.model._embed(sequences, return_attention_mask=return_attention_mask, **kwargs)

    def get_output_embeddings(self) -> nn.Linear:
        return self.mlm_head[-1]

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.mlm_head[-1] = value

    def _ttt_get_trainable_modules(self) -> list[nn.Module]:
        return [self.model]

    def _ttt_tokenize(
        self,
        seq: str | list[str] | None = None,
        input_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        if input_ids is not None:
            return {
                "input_ids": input_ids,
                "within_seq_position_ids": kwargs["within_seq_position_ids"],
                "global_position_ids": kwargs["global_position_ids"],
                "sequence_ids": kwargs["sequence_ids"],
            }
        if seq is None:
            raise ValueError("Pass either seq or E1 token tensors for TTT.")
        sequences = [seq] if isinstance(seq, str) else seq
        batch = self.prep_tokens.get_batch_kwargs(sequences, device=torch.device("cpu"))
        return {
            "input_ids": batch["input_ids"],
            "within_seq_position_ids": batch["within_seq_position_ids"],
            "global_position_ids": batch["global_position_ids"],
            "sequence_ids": batch["sequence_ids"],
        }

    def _ttt_mask_token(self) -> int:
        return int(self.prep_tokens.mask_token_id)

    def _ttt_padding_token(self) -> int:
        return int(self.prep_tokens.pad_token_id)

    def _ttt_replacement_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        ids = [self.prep_tokens.vocab[aa] for aa in amino_acids]
        return torch.tensor(ids, device=input_ids.device, dtype=input_ids.dtype)

    def _ttt_non_special_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return ~self.prep_tokens.get_boundary_token_mask(input_ids)

    def _ttt_predict_logits(
        self,
        batch: torch.Tensor | dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if not isinstance(batch, dict):
            raise TypeError("E1 TTT expects a tensor dictionary.")
        output = self(
            input_ids=batch["input_ids"],
            within_seq_position_ids=batch["within_seq_position_ids"],
            global_position_ids=batch["global_position_ids"],
            sequence_ids=batch["sequence_ids"],
            return_dict=True,
        )
        return output.logits

    def search_homologues(
        self,
        sequence: str,
        output_dir: str,
        provider: str = "colabfold",
        target_db: str | None = None,
        seq_id: str | None = None,
        **kwargs,
    ) -> str:
        searcher = _make_homologue_searcher(provider=provider, target_db=target_db, **kwargs)
        return searcher.search(sequence=sequence, output_dir=output_dir, seq_id=seq_id)

    def batch_search_homologues(
        self,
        sequences: list[str],
        output_dir: str,
        provider: str = "colabfold",
        target_db: str | None = None,
        seq_ids: list[str] | None = None,
        continue_on_error: bool = True,
        **kwargs,
    ) -> dict[str, str]:
        searcher = _make_homologue_searcher(provider=provider, target_db=target_db, **kwargs)
        return searcher.batch_search(
            sequences=sequences,
            output_dir=output_dir,
            seq_ids=seq_ids,
            continue_on_error=continue_on_error,
        )

    def sample_msa_contexts(
        self,
        a3m_path: str,
        seed: int = 42,
        max_context_tokens: list[int] | None = None,
        similarity_thresholds: list[float] | None = None,
        min_query_similarity: float = 0.3,
        context_cache_dir: str | None = None,
    ) -> dict[str, str]:
        context_specs = build_context_specifications(
            max_context_tokens=max_context_tokens,
            similarity_thresholds=similarity_thresholds,
            min_query_similarity=min_query_similarity,
        )
        cache = None
        if context_cache_dir is not None:
            key = repr((max_context_tokens, similarity_thresholds, min_query_similarity))
            specs_hash = hashlib.md5(key.encode()).hexdigest()[:8]
            cache = ContextCache(context_cache_dir, specs_hash, seed)
            cached = cache.load(a3m_path)
            if cached is not None:
                return cached
        contexts = sample_contexts_for_msa(a3m_path, context_specs, seed=seed)
        if cache is not None:
            cache.store(a3m_path, contexts)
        return contexts

    @torch.inference_mode()
    def score_ppll(
        self,
        sequences: list[str],
        a3m_path: str,
        ensemble: bool = True,
        seed: int = 42,
        max_context_tokens: list[int] | None = None,
        similarity_thresholds: list[float] | None = None,
        min_query_similarity: float = 0.3,
        max_batch_tokens: int = 131072,
        cache_size: int = 1,
        context_cache_dir: str | None = None,
        progress: bool = True,
    ) -> list[float] | list[list[float]]:
        """Score sequences with FastPLMs PPLL reduction over sampled E1 MSA contexts.

        This intentionally differs from Profluent's official E1Scorer, which scores
        mutants against a parent sequence with wildtype or masked marginal log-prob
        deltas. Here each sequence is scored by mean correct-token probability and
        optionally averaged across sampled contexts.
        """
        contexts = self.sample_msa_contexts(
            a3m_path=a3m_path,
            seed=seed,
            max_context_tokens=max_context_tokens,
            similarity_thresholds=similarity_thresholds,
            min_query_similarity=min_query_similarity,
            context_cache_dir=context_cache_dir,
        )
        if not contexts:
            raise ValueError("At least one sampled MSA context is required for PPLL scoring.")

        predictor = _E1ContextPredictor(
            model=self,
            data_prep_config=DataPrepConfig(remove_X_tokens=True),
            max_batch_tokens=max_batch_tokens,
            fields_to_save=["logits"],
            save_masked_positions_only=False,
            keep_predictions_in_gpu=False,
            use_cache=True,
            cache_size=cache_size,
            progress=progress,
        )
        vocab = predictor.batch_preparer.vocab
        seq_token_ids = [
            torch.tensor([vocab[aa] for aa in seq if aa != "X"], device=self.device)
            for seq in sequences
        ]
        context_ids = list(contexts.keys())
        all_scores = torch.zeros(len(sequences), len(context_ids), device=self.device)

        iterator = tqdm(context_ids, desc="Scoring with contexts", disable=not progress)
        for ctx_idx, ctx_id in enumerate(iterator):
            predictions = list(
                predictor.predict(
                    sequences=sequences,
                    sequence_ids=list(range(len(sequences))),
                    context_seqs={ctx_id: contexts[ctx_id]},
                )
            )
            for prediction in predictions:
                seq_idx = prediction["id"]
                if not isinstance(seq_idx, int):
                    raise TypeError("Expected integer sequence ids for score aggregation.")
                all_scores[seq_idx, ctx_idx] = compute_ppll(
                    prediction["logits"], seq_token_ids[seq_idx]
                )
            if predictor.kv_cache is not None:
                predictor.kv_cache.reset()

        if ensemble:
            return all_scores.mean(dim=1).tolist()
        return all_scores.tolist()

    @torch.inference_mode()
    def embed_with_msa(
        self,
        sequences: list[str],
        a3m_path: str | None = None,
        context: str | None = None,
        pooling_types: list[str] | None = None,
        pooling: str = "mean",
        matrix_embed: bool = False,
        seed: int = 42,
        max_batch_tokens: int = 131072,
        embed_max_tokens: int = DEFAULT_EMBED_MAX_TOKENS,
        embed_similarity: float = DEFAULT_EMBED_SIMILARITY,
        min_query_similarity: float = 0.3,
        progress: bool = True,
    ) -> torch.Tensor | list[torch.Tensor]:
        if a3m_path is not None and context is None:
            spec = ContextSpecification(
                max_num_samples=511,
                max_token_length=embed_max_tokens,
                max_query_similarity=embed_similarity,
                min_query_similarity=min_query_similarity,
            )
            contexts, _ = sample_multiple_contexts(
                msa_path=a3m_path,
                context_specifications=[spec],
                seed=seed,
            )
            context = contexts[0] if contexts else None

        hidden_list = _forward_for_embedding(
            model=self,
            sequences=sequences,
            context=context,
            max_batch_tokens=max_batch_tokens,
            progress=progress,
        )
        if matrix_embed:
            return hidden_list
        if pooling_types is not None:
            return _pool_hidden_states(hidden_list, pooling_types, self.device)
        if pooling not in ("mean", "cls"):
            raise ValueError("pooling must be 'mean' or 'cls' when pooling_types is not provided")
        embeddings = [
            hidden.mean(dim=0) if pooling == "mean" else hidden[0] for hidden in hidden_list
        ]
        return torch.stack(embeddings)

    @torch.inference_mode()
    def embed_dataset_with_msa(
        self,
        sequences: list[str],
        msa_lookup: dict[str, str] | None = None,
        msa_dir: str | None = None,
        msa_hf_path: str | None = None,
        batch_size: int = 2,
        max_len: int = 2048,
        pooling_types: list[str] | None = None,
        pooling: str = "mean",
        matrix_embed: bool = False,
        embed_dtype: torch.dtype = torch.bfloat16,
        embed_max_tokens: int = DEFAULT_EMBED_MAX_TOKENS,
        embed_similarity: float = DEFAULT_EMBED_SIMILARITY,
        min_query_similarity: float = 0.3,
        seed: int = 42,
        progress: bool = True,
        max_batch_tokens: int = 131072,
        batch_window_size: int | None = None,
        max_tokens_per_batch: int | None = None,
        output: str | os.PathLike[str] | None = None,
        format: str = "safetensors",
        resume: bool = True,
        shard_size: int = 2 * 1024**3,
        model_state_fingerprint: str | None = None,
    ) -> EmbeddingResult:
        """Embed an ordered sequence dataset with optional sampled MSA context.

        Unlike the legacy dictionary return, the result preserves duplicate
        sequences and input order. ``output`` uses the same transactional,
        resumable SQLite or safetensors persistence as :meth:`embed_dataset`.
        ``max_len`` counts biological residues.
        """

        if not sequences:
            raise ValueError("sequences must contain at least one protein sequence.")
        if any(not isinstance(sequence, str) or not sequence for sequence in sequences):
            raise ValueError("sequences must contain non-empty strings.")
        if max_len <= 0:
            raise ValueError("max_len must be positive.")
        if msa_lookup is None:
            if msa_dir is not None:
                msa_lookup = load_msa_dir(msa_dir)
            elif msa_hf_path is not None:
                msa_lookup = load_msa_from_hf(msa_hf_path)
            else:
                msa_lookup = {}

        truncated_sequences = [sequence[:max_len] for sequence in sequences]
        unique_seqs = sorted(set(truncated_sequences), key=lambda value: (-len(value), value))
        context_map: dict[str, str | None] = {}
        spec = ContextSpecification(
            max_num_samples=511,
            max_token_length=embed_max_tokens,
            max_query_similarity=embed_similarity,
            min_query_similarity=min_query_similarity,
        )
        for seq in unique_seqs:
            a3m_path = get_msa_for_sequence(seq, msa_lookup)
            if a3m_path is None:
                context_map[seq] = None
                continue
            contexts, _ = sample_multiple_contexts(
                msa_path=a3m_path,
                context_specifications=[spec],
                seed=seed,
            )
            context_map[seq] = contexts[0] if contexts else None

        context_digest = hashlib.sha256()
        for sequence in unique_seqs:
            for value in (sequence, context_map[sequence] or ""):
                encoded = value.encode("utf-8")
                context_digest.update(len(encoded).to_bytes(8, "big"))
                context_digest.update(encoded)

        def embed_msa_batch(batch_sequences: list[str]) -> EmbeddingBatch:
            grouped_positions: dict[str | None, list[int]] = defaultdict(list)
            for position, sequence in enumerate(batch_sequences):
                grouped_positions[context_map[sequence]].append(position)
            hidden_by_position: list[torch.Tensor | None] = [None] * len(batch_sequences)
            for context, positions in grouped_positions.items():
                context_sequences = [batch_sequences[position] for position in positions]
                hidden_states = _forward_for_embedding(
                    model=self,
                    sequences=context_sequences,
                    context=context,
                    max_batch_tokens=max_batch_tokens,
                    progress=progress,
                )
                for position, hidden in zip(positions, hidden_states, strict=True):
                    hidden_by_position[position] = hidden
            resolved = [hidden for hidden in hidden_by_position if hidden is not None]
            if len(resolved) != len(batch_sequences):
                raise RuntimeError("E1 MSA embedding did not return every requested sequence.")
            max_residues = max(hidden.shape[0] for hidden in resolved)
            hidden_size = resolved[0].shape[-1]
            X = resolved[0].new_zeros((len(resolved), max_residues, hidden_size))
            residue_mask = torch.zeros(
                (len(resolved), max_residues),
                dtype=torch.bool,
                device=X.device,
            )
            for position, hidden in enumerate(resolved):
                residue_count = hidden.shape[0]
                X[position, :residue_count] = hidden
                residue_mask[position, :residue_count] = True
            return EmbeddingBatch(X=X, residue_mask=residue_mask)

        resolved_pooling: str | list[str] | None = (
            None if matrix_embed else pooling_types if pooling_types is not None else pooling
        )
        adapter_identity = {
            "kind": "e1-msa-v1",
            "sampling_source_revision": E1_MSA_SAMPLING_SOURCE_REVISION,
            "context_sha256": context_digest.hexdigest(),
            "context_count": sum(context is not None for context in context_map.values()),
            "seed": seed,
            "embed_max_tokens": embed_max_tokens,
            "embed_similarity": embed_similarity,
            "min_query_similarity": min_query_similarity,
            "max_batch_tokens": max_batch_tokens,
        }
        return embed_dataset(
            self,
            [(str(position), sequence) for position, sequence in enumerate(sequences)],
            batch_size=batch_size,
            pooling=resolved_pooling,
            full_embeddings=matrix_embed,
            output=output,
            format=format,
            resume=resume,
            max_length=max_len,
            truncate=True,
            dtype=embed_dtype,
            shard_size=shard_size,
            model_state_fingerprint=model_state_fingerprint,
            batch_window_size=batch_window_size,
            max_tokens_per_batch=max_tokens_per_batch,
            _embedding_batch_fn=embed_msa_batch,
            _embedding_batch_identity=adapter_identity,
            _allowed_unsupported_pooling=("cls",),
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        within_seq_position_ids: torch.LongTensor | None = None,
        global_position_ids: torch.LongTensor | None = None,
        sequence_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        past_key_values: DynamicCache | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_s_max: bool = False,
        return_dict: bool | None = None,
    ) -> E1MaskedLMOutputWithPast | tuple[Any, ...]:
        """Return hidden states and masked-token logits for E1 inputs.

        Token, position, sequence, and label tensors have shape (b, l).
        Callers may instead provide precomputed H with shape (b, l, d).
        """
        use_cache = (
            use_cache if use_cache is not None else bool(getattr(self.config, "use_cache", False))
        )
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs: E1ModelOutputWithPast = self.model(
            input_ids=input_ids,
            within_seq_position_ids=within_seq_position_ids,
            global_position_ids=global_position_ids,
            sequence_ids=sequence_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_s_max=output_s_max,
            return_dict=True,
        )

        last_hidden_state = outputs.last_hidden_state
        loss = None

        mlm_logits = self.mlm_head(last_hidden_state).float()
        mlm_loss = None
        if labels is not None:
            mlm_logits_flat = mlm_logits.contiguous().view(-1, self.config.vocab_size)
            mlm_labels_flat = labels.to(mlm_logits_flat.device).contiguous().view(-1)
            mlm_loss = F.cross_entropy(
                mlm_logits_flat,
                mlm_labels_flat,
                ignore_index=-100,
                reduction="none",
            )
            mask = mlm_labels_flat.ne(-100) & mlm_labels_flat.ne(self.model.padding_idx)
            n_mlm = mask.sum().clamp_min(1)
            mlm_loss = (mlm_loss * mask.to(mlm_loss)).sum() / n_mlm
            loss = 0.0
            loss += mlm_loss

        result = E1MaskedLMOutputWithPast(
            loss=loss,
            logits=mlm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mlm_loss=mlm_loss,
            last_hidden_state=last_hidden_state,
            past_key_values=outputs.past_key_values,
            s_max=outputs.s_max,
        )
        if not return_dict:
            return result.to_tuple()
        return result


class E1ForSequenceClassification(E1PreTrainedModel, EmbeddingMixin):
    config: E1Config
    config_class = E1Config

    def __init__(self, config: E1Config, **kwargs) -> None:
        pooling_types = kwargs.pop("pooling_types", None)
        if pooling_types is None:
            pooling_types = ["mean", "var"]
        elif not isinstance(pooling_types, list):
            raise TypeError("pooling_types must be a non-empty list of pooling names")
        elif not pooling_types or not all(isinstance(name, str) for name in pooling_types):
            raise ValueError("pooling_types must be a non-empty list of pooling names")

        E1PreTrainedModel.__init__(self, config, **kwargs)
        self.model: FAST_E1_ENCODER = FAST_E1_ENCODER(config, **kwargs)
        self.vocab_size = config.vocab_size
        self.num_labels = config.num_labels
        self.pooler = Pooler(pooling_types)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size * len(pooling_types), config.hidden_size * 4),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size * 4),
            nn.Linear(config.hidden_size * 4, config.num_labels),
        )
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.gradient_checkpointing = config.gradient_checkpointing
        self.post_init()

    @property
    def device_mesh(self) -> torch.distributed.device_mesh.DeviceMesh:
        return self.model.device_mesh

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.set_input_embeddings(value)

    def _embed(
        self, sequences: list[str], return_attention_mask: bool = False, **kwargs
    ) -> torch.Tensor:
        return self.model._embed(sequences, return_attention_mask=return_attention_mask, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        within_seq_position_ids: torch.LongTensor | None = None,
        global_position_ids: torch.LongTensor | None = None,
        sequence_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        past_key_values: DynamicCache | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_s_max: bool = False,
        return_dict: bool | None = None,
    ) -> E1ClassificationOutputWithPast | tuple[Any, ...]:
        use_cache = (
            use_cache if use_cache is not None else bool(getattr(self.config, "use_cache", False))
        )
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs: E1ModelOutputWithPast = self.model(
            input_ids=input_ids,
            within_seq_position_ids=within_seq_position_ids,
            global_position_ids=global_position_ids,
            sequence_ids=sequence_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_s_max=output_s_max,
            return_dict=True,
        )

        attention_mask = (
            (sequence_ids != -1).long()
            if sequence_ids is not None
            else torch.ones(
                outputs.last_hidden_state.shape[:2],
                device=outputs.last_hidden_state.device,
                dtype=torch.long,
            )
        )
        x = outputs.last_hidden_state
        features = self.pooler(x, attention_mask)
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

        result = E1ClassificationOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            last_hidden_state=x,
            s_max=outputs.s_max,
        )
        if not return_dict:
            return result.to_tuple()
        return result


class E1ForTokenClassification(E1PreTrainedModel, EmbeddingMixin):
    config: E1Config
    config_class = E1Config

    def __init__(self, config: E1Config, **kwargs) -> None:
        E1PreTrainedModel.__init__(self, config, **kwargs)
        self.model: FAST_E1_ENCODER = FAST_E1_ENCODER(config, **kwargs)
        self.vocab_size = config.vocab_size
        self.num_labels = config.num_labels
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size * 4),
            nn.Linear(config.hidden_size * 4, config.num_labels),
        )
        self.loss_fct = nn.CrossEntropyLoss()
        self.gradient_checkpointing = config.gradient_checkpointing
        self.post_init()

    @property
    def device_mesh(self) -> torch.distributed.device_mesh.DeviceMesh:
        return self.model.device_mesh

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.set_input_embeddings(value)

    def _embed(
        self, sequences: list[str], return_attention_mask: bool = False, **kwargs
    ) -> torch.Tensor:
        return self.model._embed(sequences, return_attention_mask=return_attention_mask, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        within_seq_position_ids: torch.LongTensor | None = None,
        global_position_ids: torch.LongTensor | None = None,
        sequence_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        past_key_values: DynamicCache | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_s_max: bool = False,
        return_dict: bool | None = None,
    ) -> E1TokenClassificationOutputWithPast | tuple[Any, ...]:
        use_cache = (
            use_cache if use_cache is not None else bool(getattr(self.config, "use_cache", False))
        )
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs: E1ModelOutputWithPast = self.model(
            input_ids=input_ids,
            within_seq_position_ids=within_seq_position_ids,
            global_position_ids=global_position_ids,
            sequence_ids=sequence_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_s_max=output_s_max,
            return_dict=True,
        )

        x = outputs.last_hidden_state
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        result = E1TokenClassificationOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            last_hidden_state=x,
            past_key_values=outputs.past_key_values,
            s_max=outputs.s_max,
        )
        if not return_dict:
            return result.to_tuple()
        return result
