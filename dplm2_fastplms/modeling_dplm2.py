"""
FastPLMs-compatible DPLM2 implementation.

This module is based on:
https://github.com/bytedance/dplm
"""

import entrypoint_setup
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

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
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except (ImportError, AttributeError):
    create_block_mask = None
    flex_attention = None

try:
    from .base_tokenizer import BaseSequenceTokenizer
except ImportError:
    from base_tokenizer import BaseSequenceTokenizer

try:
    from .embedding_mixin import EmbeddingMixin
except ImportError:
    try:
        from ..embedding_mixin import EmbeddingMixin
    except ImportError:
        from embedding_mixin import EmbeddingMixin


def _create_pad_block_mask(attention_mask_2d: torch.Tensor):
    assert create_block_mask is not None, "Flex attention block mask requires create_block_mask."
    token_valid = attention_mask_2d.bool()
    batch_size, seq_len = token_valid.shape

    def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
        return token_valid[batch_idx, q_idx] & token_valid[batch_idx, kv_idx]

    return create_block_mask(
        mask_mod,
        batch_size,
        1,
        seq_len,
        seq_len,
        device=attention_mask_2d.device,
    )


def _infer_modality_type(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    input_mask = attention_mask.bool()
    modality_type = ((input_ids < 33) & input_mask).int()
    modality_type[~input_mask] = 2
    return modality_type


@dataclass
class DPLM2MaskedLMOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


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



class ModifiedRotaryEmbedding(RotaryEmbedding):
    def __init__(self, dim: int, aa_type: int, struct_type: int):
        super().__init__(dim)
        self.aa_type = aa_type
        self.struct_type = struct_type

    def _has_multimodal_tokens(self, type_ids: Optional[torch.Tensor]) -> bool:
        if type_ids is None:
            return False
        aa_present = (type_ids == self.aa_type).any()
        struct_present = (type_ids == self.struct_type).any()
        return bool(aa_present and struct_present)

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
        )
        if cache_is_stale:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
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
        self.attn_backend = config.attn_backend
        self.rotary_embeddings = ModifiedRotaryEmbedding(
            dim=self.attention_head_size,
            aa_type=config.aa_type,
            struct_type=config.struct_type,
        )

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        type_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        flex_block_mask: Optional[object] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        if past_key_values is not None:
            past_key_value = past_key_values

        mixed_query_layer = self.query(hidden_states)
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer) * self.attention_head_size**-0.5

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        if self.position_embedding_type == "rotary":
            query_layer, key_layer = self.rotary_embeddings(query_layer, key_layer, type_ids)

        if self.position_embedding_type in ["relative_key", "relative_key_query"]:
            raise NotImplementedError

        query_layer = query_layer.contiguous()
        key_layer = key_layer.contiguous()
        value_layer = value_layer.contiguous()

        if output_attentions:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32).to(query_layer.dtype)
            context_layer = torch.matmul(attention_probs, value_layer)
        else:
            attention_probs = None
            if self.attn_backend == "flex":
                assert flex_attention is not None, "Flex attention backend requested but torch.flex_attention is unavailable."
                assert query_layer.dtype in (torch.float16, torch.bfloat16), (
                    f"Flex attention backend requires float16 or bfloat16, got {query_layer.dtype}."
                )
                assert is_cross_attention is False, "Flex attention backend currently does not support cross-attention."
                assert past_key_value is None, "Flex attention backend currently does not support KV caching."
                if attention_mask is not None:
                    assert flex_block_mask is not None, (
                        "Flex attention backend requires a block mask when attention_mask is provided."
                    )
                context_layer = flex_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    block_mask=flex_block_mask,
                    scale=1.0,
                )
            else:
                context_layer = F.scaled_dot_product_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    attn_mask=attention_mask,
                    scale=1.0,
                )

        if head_mask is not None and torch.is_tensor(head_mask):
            context_layer = context_layer * head_mask

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class ModifiedEsmAttention(EsmAttention):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.self = ModifiedEsmSelfAttention(config)
        self.output = EsmSelfOutput(config)
        self.pruned_heads = set()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        type_ids=None,
        flex_block_mask=None,
    ):
        hidden_states_ln = self.LayerNorm(hidden_states)
        self_outputs = self.self(
            hidden_states_ln,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            type_ids,
            flex_block_mask=flex_block_mask,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class ModifiedEsmLayer(EsmLayer):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ModifiedEsmAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if self.is_decoder is False:
                raise RuntimeError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = ModifiedEsmAttention(config)
        self.intermediate = EsmIntermediate(config)
        self.output = EsmOutput(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        type_ids=None,
        flex_block_mask=None,
    ):
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            type_ids=type_ids,
            flex_block_mask=flex_block_mask,
        )
        attention_output = self_attention_outputs[0]

        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]

        if self.is_decoder and encoder_hidden_states is not None:
            if self.add_cross_attention is False:
                raise AttributeError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention "
                    "layers by setting `config.add_cross_attention=True`"
                )

            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
                type_ids=None,
                flex_block_mask=None,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]
            present_key_value = present_key_value + cross_attention_outputs[-1]

        layer_output = self.feed_forward_chunk(attention_output)
        outputs = (layer_output,) + outputs

        if self.is_decoder:
            outputs = outputs + (present_key_value,)
        return outputs


class ModifiedEsmEncoder(EsmEncoder):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.layer = nn.ModuleList([ModifiedEsmLayer(config) for _ in range(config.num_hidden_layers)])
        self.emb_layer_norm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        type_ids=None,
        flex_block_mask=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        next_decoder_cache = () if use_cache else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    type_ids,
                    flex_block_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    type_ids,
                    flex_block_mask,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = next_decoder_cache + (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if self.emb_layer_norm_after:
            hidden_states = self.emb_layer_norm_after(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if return_dict is False:
            return tuple(
                value
                for value in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if value is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class DPLM2Model(DPLM2PreTrainedModel, EmbeddingMixin):
    config_class = DPLM2Config

    def __init__(self, config, add_pooling_layer=True):
        DPLM2PreTrainedModel.__init__(self, config)
        self.config = config
        self.embeddings = EsmEmbeddings(config)
        self.encoder = ModifiedEsmEncoder(config)
        self.pooler = EsmPooler(config) if add_pooling_layer else None
        self.post_init()

    def _convert_head_mask_to_5d(self, head_mask: torch.Tensor, num_hidden_layers: int) -> torch.Tensor:
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        assert head_mask.dim() == 5, f"head_mask.dim != 5, got {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)
        return head_mask

    def get_head_mask(
        self,
        head_mask: Optional[torch.Tensor],
        num_hidden_layers: int,
        is_attention_chunked: bool = False,
    ) -> Union[torch.Tensor, List[None]]:
        if head_mask is None:
            return [None] * num_hidden_layers
        head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
        if is_attention_chunked:
            head_mask = head_mask.unsqueeze(-1)
        return head_mask

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        type_ids = _infer_modality_type(input_ids, attention_mask)
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            type_ids=type_ids,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        return outputs.last_hidden_state

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        type_ids: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)

        token_attention_mask = None
        if attention_mask.dim() == 2:
            token_attention_mask = attention_mask.bool()
            if self.config.attn_backend == "flex" and output_attentions is False:
                extended_attention_mask = None
            else:
                extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
        elif attention_mask.dim() == 4:
            if self.config.attn_backend == "flex" and output_attentions is False:
                extended_attention_mask = None
            else:
                extended_attention_mask = attention_mask
            if input_ids is not None:
                token_attention_mask = input_ids.ne(self.config.pad_token_id)
        else:
            raise ValueError(f"Unsupported attention_mask shape: {attention_mask.shape}")

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = encoder_attention_mask

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_attention_mask = token_attention_mask
        if embedding_attention_mask is None and input_ids is not None:
            embedding_attention_mask = input_ids.ne(self.config.pad_token_id)

        flex_block_mask = None
        if (
            self.config.attn_backend == "flex"
            and token_attention_mask is not None
            and output_attentions is False
        ):
            assert create_block_mask is not None, (
                "Flex attention backend requested but torch.create_block_mask is unavailable."
            )
            flex_block_mask = _create_pad_block_mask(token_attention_mask)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=embedding_attention_mask,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            type_ids=type_ids,
            flex_block_mask=flex_block_mask,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if return_dict is False:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=None,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class DPLM2ForMaskedLM(DPLM2PreTrainedModel, EmbeddingMixin):
    config_class = DPLM2Config

    def __init__(self, config, dropout: float = 0.1, vocab_size: Optional[int] = None):
        config.hidden_dropout_prob = dropout
        config.tie_word_embeddings = False
        if vocab_size is not None:
            config.vocab_size = vocab_size
        DPLM2PreTrainedModel.__init__(self, config)
        self.esm = DPLM2Model(config, add_pooling_layer=False)
        self.lm_head = EsmLMHead(config)
        self.loss_fct = nn.CrossEntropyLoss()
        self.post_init()
        self.pad_id = config.pad_token_id

    def get_input_embeddings(self) -> nn.Module:
        return self.esm.embeddings.word_embeddings

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def _get_modality_type(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
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
            return_dict=True,
        )
        return outputs.last_hidden_state

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
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
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            type_ids=type_ids,
        )

        sequence_output = outputs.last_hidden_state
        logits = self.lm_head(sequence_output)
        loss = None
        if labels is not None:
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
        )


class DPLM2ForSequenceClassification(DPLM2PreTrainedModel, EmbeddingMixin):
    config_class = DPLM2Config

    def __init__(self, config):
        DPLM2PreTrainedModel.__init__(self, config)
        self.num_labels = config.num_labels
        self.esm = DPLM2Model(config, add_pooling_layer=False)
        self.classifier = EsmClassificationHead(config)
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.esm.embeddings.word_embeddings

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
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        if type_ids is None and input_ids is not None:
            if attention_mask is None:
                attention_mask = input_ids.ne(self.config.pad_token_id)
            type_ids = _infer_modality_type(input_ids, attention_mask)

        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            type_ids=type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
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

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DPLM2ForTokenClassification(DPLM2PreTrainedModel, EmbeddingMixin):
    config_class = DPLM2Config

    def __init__(self, config):
        DPLM2PreTrainedModel.__init__(self, config)
        self.num_labels = config.num_labels
        self.esm = DPLM2Model(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.esm.embeddings.word_embeddings

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
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        if type_ids is None and input_ids is not None:
            if attention_mask is None:
                attention_mask = input_ids.ne(self.config.pad_token_id)
            type_ids = _infer_modality_type(input_ids, attention_mask)

        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            type_ids=type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
