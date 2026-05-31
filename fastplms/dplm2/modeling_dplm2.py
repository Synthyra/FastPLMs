from __future__ import annotations
"""
FastPLMs-compatible DPLM2 implementation.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from einops import rearrange
from enum import Enum
from typing import List, Optional, Tuple, Union

from transformers import EsmTokenizer
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    ModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.models.esm.configuration_esm import EsmConfig
from transformers.models.esm.modeling_esm import (
    EsmAttention,
    EsmClassificationHead,
    EsmEmbeddings,
    EsmEncoder,
    EsmIntermediate,
    EsmLayer,
    EsmLMHead,
    EsmOutput,
    EsmPooler,
    EsmPreTrainedModel,
    EsmSelfAttention,
    EsmSelfOutput,
    RotaryEmbedding,
    apply_rotary_pos_emb,
)

try:
    from fastplms.attention import (
        AttentionBackend, VALID_ATTENTION_BACKENDS,
        resolve_attention_backend, get_attention_mask,
        _get_flex_attention_fn,
        _ensure_flash_kernels_loaded, FLASH_KERNEL, FLASH_KERNEL_VARIANT,
        _kernels_flash_forward, _kernels_flash_varlen_forward,
        kernels_flash_attention_func,
        index_first_axis, index_put_first_axis, pad_input, _unpad_input,
        create_block_mask, flex_attention, BlockMask,
    )
    from fastplms.embedding_mixin import Pooler, EmbeddingMixin, ProteinDataset, parse_fasta, build_collator
except ImportError:
    pass  # Running as HF Hub composite; shared definitions are above


def _infer_modality_type(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    input_mask = attention_mask.bool()
    modality_type = ((input_ids < 33) & input_mask).int()
    modality_type[~input_mask] = 2
    return modality_type


def _normalize_dplm2_input_ids(input_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    if input_ids.numel() == 0:
        return input_ids

    normalized_input_ids = input_ids.clone()
    generic_to_aa_special_ids = {
        vocab_size: 2,
        vocab_size + 1: 3,
        vocab_size + 2: 0,
        vocab_size + 3: 32,
    }
    for generic_id, aa_id in generic_to_aa_special_ids.items():
        normalized_input_ids[input_ids == generic_id] = aa_id

    valid_token_mask = normalized_input_ids.ge(0)
    if valid_token_mask.any():
        max_token_id = int(normalized_input_ids[valid_token_mask].max().item())
        assert max_token_id < vocab_size, (
            f"Found token id {max_token_id} outside the DPLM2 embedding table (vocab_size={vocab_size}). "
            "Tokenizer special tokens must be normalized before embedding."
        )
    return normalized_input_ids


def _has_packed_multimodal_layout(
    type_ids: Optional[torch.Tensor],
    aa_type: int,
    struct_type: int,
    pad_type: int,
) -> bool:
    if type_ids is None:
        return False
    assert type_ids.ndim == 2, f"Expected type_ids to have shape (batch, seq_len), got {tuple(type_ids.shape)}"
    seq_len = type_ids.shape[-1]
    if seq_len % 2 != 0:
        return False

    half_len = seq_len // 2
    first_half = type_ids[:, :half_len]
    second_half = type_ids[:, half_len:]

    first_half_valid = ((first_half == aa_type) | (first_half == pad_type)).all(dim=-1)
    second_half_valid = ((second_half == struct_type) | (second_half == pad_type)).all(dim=-1)
    aa_count = (first_half == aa_type).sum(dim=-1)
    struct_count = (second_half == struct_type).sum(dim=-1)
    packed_rows = first_half_valid & second_half_valid & aa_count.gt(0) & aa_count.eq(struct_count)
    return bool(packed_rows.all())


@dataclass
class DPLM2MaskedLMOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    s_max: Optional[Tuple[List[torch.Tensor], ...]] = None


@dataclass
class DPLM2EncoderOutput(ModelOutput):
    last_hidden_state: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    s_max: Optional[Tuple[List[torch.Tensor], ...]] = None


class DPLM2Config(EsmConfig):
    model_type = "dplm2"

    def __init__(
        self,
        attn_backend: str = "sdpa",
        aa_type: int = 1,
        struct_type: int = 0,
        pad_type: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attn_backend = attn_backend
        self.aa_type = aa_type
        self.struct_type = struct_type
        self.pad_type = pad_type
        self.tie_word_embeddings = False


class DPLM2PreTrainedModel(EsmPreTrainedModel):
    config_class = DPLM2Config
    base_model_prefix = "dplm2"
    supports_gradient_checkpointing = True
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    all_tied_weights_keys = {}

    @classmethod
    def is_remote_code(cls) -> bool:
        # Prevent post-load reinitialization of tensors already loaded from checkpoints.
        return True

    @property
    def attn_backend(self) -> str:
        return self.config.attn_backend

    @attn_backend.setter
    def attn_backend(self, backend: str) -> None:
        assert backend in VALID_ATTENTION_BACKENDS, f"Unsupported attn_backend: {backend}. Expected one of {VALID_ATTENTION_BACKENDS}."
        self.config.attn_backend = backend
        resolved = resolve_attention_backend(backend)
        for module in self.modules():
            if isinstance(module, ModifiedEsmEncoder):
                module.attention_backend = resolved
            elif isinstance(module, ModifiedEsmSelfAttention):
                module.attn_backend = resolved



class ModifiedRotaryEmbedding(RotaryEmbedding):
    def __init__(self, dim: int, aa_type: int, struct_type: int, pad_type: int):
        super().__init__(dim)
        self.aa_type = aa_type
        self.struct_type = struct_type
        self.pad_type = pad_type

    def _has_multimodal_tokens(self, type_ids: Optional[torch.Tensor]) -> bool:
        # The split rotary path only works when the sequence tensor is already packed
        # as [AA half | structure half]. Plain protein batches can still contain
        # high-ID special tokens, so mere modality presence is not enough.
        return _has_packed_multimodal_layout(
            type_ids=type_ids,
            aa_type=self.aa_type,
            struct_type=self.struct_type,
            pad_type=self.pad_type,
        )

    def _update_cos_sin_tables(
        self,
        x: torch.Tensor,
        type_ids: Optional[torch.Tensor],
        seq_dimension: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[seq_dimension]
        if self._has_multimodal_tokens(type_ids):
            seq_len = seq_len // 2

        cache_is_stale = (
            self._cos_cached is None
            or self._sin_cached is None
            or seq_len != self._seq_len_cached
            or self._cos_cached.device != x.device
            or self._cos_cached.dtype != x.dtype
        )
        if cache_is_stale:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(device=x.device, dtype=x.dtype)
            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

        return self._cos_cached, self._sin_cached

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        type_ids: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(
            k,
            type_ids=type_ids,
            seq_dimension=-2,
        )

        if self._has_multimodal_tokens(type_ids):
            q_1, q_2 = q.chunk(2, dim=-2)
            k_1, k_2 = k.chunk(2, dim=-2)
            q_1 = apply_rotary_pos_emb(q_1, self._cos_cached, self._sin_cached)
            q_2 = apply_rotary_pos_emb(q_2, self._cos_cached, self._sin_cached)
            k_1 = apply_rotary_pos_emb(k_1, self._cos_cached, self._sin_cached)
            k_2 = apply_rotary_pos_emb(k_2, self._cos_cached, self._sin_cached)
            return torch.cat((q_1, q_2), dim=-2), torch.cat((k_1, k_2), dim=-2)

        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )


class ModifiedEsmSelfAttention(EsmSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type)
        self.config = config
        self.scale = self.attention_head_size**-0.5
        self.dropout_prob = config.attention_probs_dropout_prob
        self.attn_backend = resolve_attention_backend(config.attn_backend)
        self.rotary_embeddings = ModifiedRotaryEmbedding(
            dim=self.attention_head_size,
            aa_type=config.aa_type,
            struct_type=config.struct_type,
            pad_type=config.pad_type,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor] = None,
        attention_mask_4d: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[BlockMask] = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
        type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        batch_size, seq_length = hidden_states.shape[:-1]
        hidden_shape = (batch_size, seq_length, -1, self.attention_head_size)
        query_BHLD = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        key_BHLD = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        value_BHLD = self.value(hidden_states).view(hidden_shape).transpose(1, 2)

        query_BHLD = query_BHLD * self.scale

        if self.position_embedding_type == "rotary":
            query_BHLD, key_BHLD = self.rotary_embeddings(query_BHLD, key_BHLD, type_ids)

        attn_output, attn_weights, s_max = self._attn(
            query_BHLD, key_BHLD, value_BHLD,
            attention_mask_2d=attention_mask_2d,
            attention_mask_4d=attention_mask_4d,
            flex_block_mask=flex_block_mask,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
        )
        return attn_output, attn_weights, s_max

    def _attn(
        self,
        query_BHLD: torch.Tensor,
        key_BHLD: torch.Tensor,
        value_BHLD: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor] = None,
        attention_mask_4d: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[BlockMask] = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        if output_attentions:
            return self._manual_attn(query_BHLD, key_BHLD, value_BHLD, attention_mask_4d, output_s_max)

        if self.attn_backend == AttentionBackend.KERNELS_FLASH:
            attn_output, attn_weights = self._kernels_flash_attn(query_BHLD, key_BHLD, value_BHLD, attention_mask_2d)
        elif self.attn_backend == AttentionBackend.FLEX:
            attn_output, attn_weights = self._flex_attn(query_BHLD, key_BHLD, value_BHLD, flex_block_mask)
        elif self.attn_backend == AttentionBackend.SDPA:
            attn_output, attn_weights = self._sdpa_attn(query_BHLD, key_BHLD, value_BHLD, attention_mask_4d)
        else:
            raise AssertionError(f"Unsupported resolved backend: {self.attn_backend}")

        s_max = self._compute_s_max(query_BHLD, key_BHLD) if output_s_max else None
        return attn_output, attn_weights, s_max

    @torch.no_grad()
    def _compute_s_max(self, query_BHLD: torch.Tensor, key_BHLD: torch.Tensor) -> List[torch.Tensor]:
        q_norm = torch.linalg.vector_norm(query_BHLD, dim=-1)
        k_norm = torch.linalg.vector_norm(key_BHLD, dim=-1)
        s_max_bound = (q_norm.max(dim=-1).values * k_norm.max(dim=-1).values).max(dim=0).values
        return [s_max_bound[h] for h in range(self.num_attention_heads)]

    def _manual_attn(
        self,
        query_BHLD: torch.Tensor,
        key_BHLD: torch.Tensor,
        value_BHLD: torch.Tensor,
        attention_mask_4d: Optional[torch.Tensor] = None,
        output_s_max: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        attn_weights = torch.matmul(query_BHLD, key_BHLD.transpose(-1, -2))
        if attention_mask_4d is not None:
            attn_weights = attn_weights.masked_fill(attention_mask_4d.logical_not(), float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.dropout_prob > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout_prob, training=self.training)
        context_BHLD = torch.matmul(attn_weights, value_BHLD)
        attn_output = rearrange(context_BHLD, "b h s d -> b s (h d)")
        s_max = self._compute_s_max(query_BHLD, key_BHLD) if output_s_max else None
        return attn_output, attn_weights, s_max

    def _kernels_flash_attn(
        self,
        query_BHLD: torch.Tensor,
        key_BHLD: torch.Tensor,
        value_BHLD: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        query_BLHD = query_BHLD.transpose(1, 2).contiguous()
        key_BLHD = key_BHLD.transpose(1, 2).contiguous()
        value_BLHD = value_BHLD.transpose(1, 2).contiguous()
        # Q is pre-scaled by self.scale in forward() -- pass softmax_scale=1.0
        # to prevent the kernel from applying its default 1/sqrt(head_dim).
        attn_output = kernels_flash_attention_func(
            query_states=query_BLHD, key_states=key_BLHD, value_states=value_BLHD,
            attention_mask_2d=attention_mask_2d, causal=False,
            softmax_scale=1.0,
        )
        return rearrange(attn_output, "b s h d -> b s (h d)"), None

    def _flex_attn(
        self,
        query_BHLD: torch.Tensor,
        key_BHLD: torch.Tensor,
        value_BHLD: torch.Tensor,
        flex_block_mask: Optional[BlockMask] = None,
    ) -> Tuple[torch.Tensor, None]:
        assert flex_attention is not None, "Flex attention is not available in this environment."
        fn = _get_flex_attention_fn()
        context_BHLD = fn(query_BHLD, key_BHLD, value_BHLD, block_mask=flex_block_mask, scale=1.0)
        return rearrange(context_BHLD, "b h s d -> b s (h d)"), None

    def _sdpa_attn(
        self,
        query_BHLD: torch.Tensor,
        key_BHLD: torch.Tensor,
        value_BHLD: torch.Tensor,
        attention_mask_4d: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        context_BHLD = F.scaled_dot_product_attention(
            query_BHLD, key_BHLD, value_BHLD,
            attn_mask=attention_mask_4d,
            dropout_p=self.dropout_prob if self.training else 0.0,
            scale=1.0,
        )
        return rearrange(context_BHLD, "b h s d -> b s (h d)"), None


class ModifiedEsmAttention(EsmAttention):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.self = ModifiedEsmSelfAttention(config)
        self.output = EsmSelfOutput(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor] = None,
        attention_mask_4d: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[BlockMask] = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
        type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        hidden_states_ln = self.LayerNorm(hidden_states)
        attn_output, attn_weights, s_max = self.self(
            hidden_states_ln,
            attention_mask_2d=attention_mask_2d,
            attention_mask_4d=attention_mask_4d,
            flex_block_mask=flex_block_mask,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
            type_ids=type_ids,
        )
        attention_output = self.output(attn_output, hidden_states)
        return attention_output, attn_weights, s_max


class ModifiedEsmLayer(EsmLayer):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ModifiedEsmAttention(config)
        self.intermediate = EsmIntermediate(config)
        self.output = EsmOutput(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor] = None,
        attention_mask_4d: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[BlockMask] = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
        type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        attention_output, attn_weights, s_max = self.attention(
            hidden_states,
            attention_mask_2d=attention_mask_2d,
            attention_mask_4d=attention_mask_4d,
            flex_block_mask=flex_block_mask,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
            type_ids=type_ids,
        )
        layer_output = self.feed_forward_chunk(attention_output)
        return layer_output, attn_weights, s_max


class ModifiedEsmEncoder(EsmEncoder):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.attention_backend = resolve_attention_backend(config.attn_backend)
        self.layer = nn.ModuleList([ModifiedEsmLayer(config) for _ in range(config.num_hidden_layers)])
        self.emb_layer_norm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        output_s_max: bool = False,
        type_ids: Optional[torch.Tensor] = None,
    ) -> DPLM2EncoderOutput:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        full_s_max = () if output_s_max else None

        attention_mask_2d, attention_mask_4d, flex_block_mask = get_attention_mask(
            effective_backend=self.attention_backend,
            batch_size=hidden_states.shape[0],
            seq_len=hidden_states.shape[1],
            device=hidden_states.device,
            attention_mask=attention_mask,
        )

        for layer_module in self.layer:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                hidden_states, attn_weights, s_max = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask_2d,
                    attention_mask_4d,
                    flex_block_mask,
                    output_attentions,
                    output_s_max,
                    type_ids,
                )
            else:
                hidden_states, attn_weights, s_max = layer_module(
                    hidden_states,
                    attention_mask_2d=attention_mask_2d,
                    attention_mask_4d=attention_mask_4d,
                    flex_block_mask=flex_block_mask,
                    output_attentions=output_attentions,
                    output_s_max=output_s_max,
                    type_ids=type_ids,
                )

            if all_attentions is not None:
                all_attentions = all_attentions + (attn_weights,)
            if full_s_max is not None:
                full_s_max = full_s_max + (s_max,)

        if self.emb_layer_norm_after:
            hidden_states = self.emb_layer_norm_after(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return DPLM2EncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            s_max=full_s_max,
        )


class FAST_DPLM2_ENCODER(DPLM2PreTrainedModel, EmbeddingMixin):
    """Inner encoder class that holds the actual ESM-style weights (embeddings, encoder)
    so that the weight keys are prefixed with 'esm.' in the outer DPLM2Model,
    matching pretrained DPLM2 checkpoints."""

    def __init__(self, config, **kwargs):
        DPLM2PreTrainedModel.__init__(self, config, **kwargs)
        self.config = config
        self.embeddings = EsmEmbeddings(config)
        self.encoder = ModifiedEsmEncoder(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        input_ids = _normalize_dplm2_input_ids(input_ids, self.config.vocab_size)
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        type_ids = _infer_modality_type(input_ids, attention_mask)
        token_embedding_output = self.embeddings(input_ids, attention_mask=attention_mask)
        encoder_outputs = self.encoder(
            token_embedding_output,
            attention_mask=attention_mask,
            output_hidden_states=False,
            output_attentions=False,
            type_ids=type_ids,
        )
        return encoder_outputs.last_hidden_state

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_s_max: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        type_ids: Optional[torch.Tensor] = None,
    ) -> DPLM2EncoderOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_ids = _normalize_dplm2_input_ids(input_ids, self.config.vocab_size)
        elif inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        token_embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            token_embedding_output,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
            type_ids=type_ids,
        )

        return DPLM2EncoderOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            s_max=encoder_outputs.s_max,
        )


class DPLM2Model(DPLM2PreTrainedModel, EmbeddingMixin):
    config_class = DPLM2Config
    def __init__(self, config, add_pooling_layer=True):
        DPLM2PreTrainedModel.__init__(self, config)
        self.config = config
        self.esm = FAST_DPLM2_ENCODER(config)
        self.pooler = EsmPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.esm.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.esm.embeddings.word_embeddings = value

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.esm._embed(input_ids, attention_mask)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_s_max: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        type_ids: Optional[torch.Tensor] = None,
    ) -> DPLM2EncoderOutput:
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_s_max=output_s_max,
            type_ids=type_ids,
        )
        sequence_output = outputs.last_hidden_state
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return DPLM2EncoderOutput(
            last_hidden_state=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            s_max=outputs.s_max,
        )


class DPLM2ForMaskedLM(DPLM2PreTrainedModel, EmbeddingMixin):
    config_class = DPLM2Config
    def __init__(self, config, dropout: float = 0.1, vocab_size: Optional[int] = None):
        config.hidden_dropout_prob = dropout
        config.tie_word_embeddings = False
        if vocab_size is not None:
            config.vocab_size = vocab_size
        DPLM2PreTrainedModel.__init__(self, config)
        self.esm = FAST_DPLM2_ENCODER(config)
        self.lm_head = EsmLMHead(config)
        self.loss_fct = nn.CrossEntropyLoss()
        self.post_init()
        self.pad_id = config.pad_token_id
        self.tokenizer = self.__class__.tokenizer
        if isinstance(config._name_or_path, str) and len(config._name_or_path) > 0:
            self.tokenizer = EsmTokenizer.from_pretrained(config._name_or_path)

    def get_input_embeddings(self) -> nn.Module:
        return self.esm.get_input_embeddings()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def _get_modality_type(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_ids = _normalize_dplm2_input_ids(input_ids, self.config.vocab_size)
        return _infer_modality_type(input_ids, attention_mask)

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_id)
        type_ids = self._get_modality_type(input_ids, attention_mask)
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            type_ids=type_ids,
            output_attentions=False,
            output_hidden_states=False,
        )
        return outputs.last_hidden_state

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_s_max: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], DPLM2MaskedLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is None:
            assert input_ids is not None
            attention_mask = input_ids.ne(self.pad_id)

        if type_ids is None:
            assert input_ids is not None
            type_ids = self._get_modality_type(input_ids, attention_mask)

        outputs = self.esm(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_s_max=output_s_max,
            type_ids=type_ids,
        )

        sequence_output = outputs.last_hidden_state
        logits = self.lm_head(sequence_output)
        loss = None
        if labels is not None:
            labels = _normalize_dplm2_input_ids(labels, self.config.vocab_size)
            labels = labels.to(logits.device)
            loss = self.loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if return_dict is False:
            output = (logits, sequence_output, outputs.hidden_states, outputs.attentions)
            if loss is not None:
                return (loss,) + output
            return output

        return DPLM2MaskedLMOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            s_max=outputs.s_max,
        )


class DPLM2ForSequenceClassification(DPLM2PreTrainedModel, EmbeddingMixin):
    config_class = DPLM2Config

    def __init__(self, config):
        DPLM2PreTrainedModel.__init__(self, config)
        self.num_labels = config.num_labels
        self.esm = FAST_DPLM2_ENCODER(config)
        self.classifier = EsmClassificationHead(config)
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.esm.get_input_embeddings()

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.esm._embed(input_ids, attention_mask)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_s_max: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> DPLM2MaskedLMOutput:
        if type_ids is None and input_ids is not None:
            if attention_mask is None:
                attention_mask = input_ids.ne(self.config.pad_token_id)
            input_ids = _normalize_dplm2_input_ids(input_ids, self.config.vocab_size)
            type_ids = _infer_modality_type(input_ids, attention_mask)

        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            type_ids=type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_s_max=output_s_max,
        )
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                if self.num_labels == 1:
                    loss = self.mse(logits.squeeze(), labels.squeeze())
                else:
                    loss = self.mse(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = self.ce(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = self.bce(logits, labels)

        return DPLM2MaskedLMOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            s_max=outputs.s_max,
        )


class DPLM2ForTokenClassification(DPLM2PreTrainedModel, EmbeddingMixin):
    config_class = DPLM2Config

    def __init__(self, config):
        DPLM2PreTrainedModel.__init__(self, config)
        self.num_labels = config.num_labels
        self.esm = FAST_DPLM2_ENCODER(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.esm.get_input_embeddings()

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.esm._embed(input_ids, attention_mask)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_s_max: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> DPLM2MaskedLMOutput:
        if type_ids is None and input_ids is not None:
            if attention_mask is None:
                attention_mask = input_ids.ne(self.config.pad_token_id)
            input_ids = _normalize_dplm2_input_ids(input_ids, self.config.vocab_size)
            type_ids = _infer_modality_type(input_ids, attention_mask)

        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            type_ids=type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_s_max=output_s_max,
        )
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return DPLM2MaskedLMOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            s_max=outputs.s_max,
        )


if __name__ == "__main__":
    import random

    import torch

    from torch import Tensor
    from transformers import EsmTokenizer

    def print_tensor_shapes(prefix: str, obj):
        if isinstance(obj, Tensor):
            print(f"{prefix}{obj.shape}")
        elif isinstance(obj, dict):
            for name, value in obj.items():
                print_tensor_shapes(f"{prefix}{name}.", value)
        elif isinstance(obj, list):
            for idx, value in enumerate(obj):
                print_tensor_shapes(f"{prefix}[{idx}].", value)
        elif isinstance(obj, tuple):
            for idx, value in enumerate(obj):
                print_tensor_shapes(f"{prefix}[{idx}].", value)
        elif hasattr(obj, "__dict__"):
            for name, value in vars(obj).items():
                if name.startswith("_"):
                    continue
                print_tensor_shapes(f"{prefix}{name}.", value)
        else:
            print(f"{prefix}{type(obj)}")

    random.seed(0)
    torch.manual_seed(0)

    num_attention_heads = random.choice([2, 4])
    config = DPLM2Config(
        hidden_size=16 * num_attention_heads,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=random.choice([1, 2]),
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        attn_backend="sdpa",
    )
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    batch = tokenizer(["ACDEFGH", "MKTW"], return_tensors="pt", padding="longest")
    batch["labels"] = batch["input_ids"].clone()
    model = DPLM2ForMaskedLM(config=config).eval()

    with torch.no_grad():
        output = model(**batch, return_dict=True)

    print("Batch shape:")
    print_tensor_shapes("", batch)
    print("Output shape:")
    print_tensor_shapes("", output)
