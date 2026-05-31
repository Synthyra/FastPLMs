from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Any, Dict, List, Optional, Tuple
from einops import rearrange
from dataclasses import dataclass
from transformers import PreTrainedModel, PretrainedConfig, EsmTokenizer
from transformers.modeling_outputs import ModelOutput
from transformers.models.esm.modeling_esm import (
    EsmIntermediate,
    EsmOutput,
    EsmPooler,
    EsmLMHead,
    EsmSelfOutput,
    EsmClassificationHead,
    EsmContactPredictionHead,
    EsmEmbeddings,
    RotaryEmbedding,
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


@dataclass
class FastEsmEncoderOutput(ModelOutput):
    last_hidden_state: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    s_max: Optional[Tuple[List[torch.Tensor], ...]] = None


@dataclass
class EsmMaskedLMOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    s_max: Optional[Tuple[List[torch.Tensor], ...]] = None


class FastEsmConfig(PretrainedConfig):
    model_type = "fast_esm"
    def __init__(
        self,
        vocab_size: int = None,
        mask_token_id: int = None,
        pad_token_id: int = None,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 1026,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        position_embedding_type: str = "rotary",
        emb_layer_norm_before: bool = None,
        token_dropout: bool = True,
        attn_backend: str = "sdpa",
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.emb_layer_norm_before = emb_layer_norm_before
        self.tie_word_embeddings = False
        self.token_dropout = token_dropout
        self.attn_backend = attn_backend

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionar y of all the attributes that make up this configuration instance,
        """
        output = super().to_dict()
        return output


class EsmSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type: Optional[str] = None):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0, (
            f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
            f"heads ({config.num_attention_heads})"
        )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.scale = self.attention_head_size**-0.5

        self.dropout_prob = config.attention_probs_dropout_prob
        self.config = config
        self.attn_backend = resolve_attention_backend(config.attn_backend)
        self.position_embedding_type = position_embedding_type or config.position_embedding_type
        self.rotary_embeddings = None
        if self.position_embedding_type == "rotary":
            self.rotary_embeddings = RotaryEmbedding(dim=self.attention_head_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor] = None,
        attention_mask_4d: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[BlockMask] = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        batch_size, seq_length = hidden_states.shape[:-1]
        hidden_shape = (batch_size, seq_length, -1, self.attention_head_size)
        query_BHLD = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        key_BHLD = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        value_BHLD = self.value(hidden_states).view(hidden_shape).transpose(1, 2)

        query_BHLD = query_BHLD * self.scale

        if self.position_embedding_type == "rotary":
            query_BHLD, key_BHLD = self.rotary_embeddings(query_BHLD, key_BHLD)

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
        # Q has been pre-scaled by self.scale = 1/sqrt(head_dim) in forward().
        # Pass softmax_scale=1.0 to prevent the kernel from applying its default
        # 1/sqrt(head_dim) scale on top (which would yield effective scale
        # 1/head_dim and break parity vs sdpa).
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
        

class EsmAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = EsmSelfAttention(config)
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        hidden_states_ln = self.LayerNorm(hidden_states)
        attn_output, attn_weights, s_max = self.self(
            hidden_states_ln,
            attention_mask_2d=attention_mask_2d,
            attention_mask_4d=attention_mask_4d,
            flex_block_mask=flex_block_mask,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
        )
        attention_output = self.output(attn_output, hidden_states)
        return attention_output, attn_weights, s_max


class EsmLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = EsmAttention(config)
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        attention_output, attn_weights, s_max = self.attention(
            hidden_states,
            attention_mask_2d=attention_mask_2d,
            attention_mask_4d=attention_mask_4d,
            flex_block_mask=flex_block_mask,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
        )
        layer_output = self.feed_forward_chunk(attention_output)
        return layer_output, attn_weights, s_max

    def feed_forward_chunk(self, attention_output):
        attention_output_ln = self.LayerNorm(attention_output)
        intermediate_output = self.intermediate(attention_output_ln)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class EsmEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention_backend = resolve_attention_backend(config.attn_backend)
        self.layer = nn.ModuleList([EsmLayer(config) for _ in range(config.num_hidden_layers)])
        self.emb_layer_norm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        output_s_max: bool = False,
    ) -> FastEsmEncoderOutput:
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
                )
            else:
                hidden_states, attn_weights, s_max = layer_module(
                    hidden_states,
                    attention_mask_2d=attention_mask_2d,
                    attention_mask_4d=attention_mask_4d,
                    flex_block_mask=flex_block_mask,
                    output_attentions=output_attentions,
                    output_s_max=output_s_max,
                )

            if all_attentions is not None:
                all_attentions = all_attentions + (attn_weights,)
            if full_s_max is not None:
                full_s_max = full_s_max + (s_max,)

        if self.emb_layer_norm_after:
            hidden_states = self.emb_layer_norm_after(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return FastEsmEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            s_max=full_s_max,
        )


class FastEsmPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = FastEsmConfig
    base_model_prefix = "fastesm"
    supports_gradient_checkpointing = True
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    all_tied_weights_keys = {}

    @classmethod
    def is_remote_code(cls) -> bool:
        # Prevent post-load reinitialization of tensors already loaded from checkpoints.
        return True

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def post_init(self) -> None:
        super().post_init()

    def get_output_embeddings(self):
        # NOTE: get_output_embeddings() must return None to prevent accidental weight tying.
        # See e.g. https://github.com/huggingface/transformers/pull/39339#discussion_r2219126400
        return None

    @property
    def attn_backend(self) -> str:
        return self.config.attn_backend

    @attn_backend.setter
    def attn_backend(self, backend: str) -> None:
        assert backend in VALID_ATTENTION_BACKENDS, f"Unsupported attn_backend: {backend}. Expected one of {VALID_ATTENTION_BACKENDS}."
        self.config.attn_backend = backend
        resolved = resolve_attention_backend(backend)
        for module in self.modules():
            if isinstance(module, EsmEncoder):
                module.attention_backend = resolved
            elif isinstance(module, EsmSelfAttention):
                module.attn_backend = resolved


class FAST_ESM_ENCODER(FastEsmPreTrainedModel, EmbeddingMixin):
    def __init__(self, config, add_pooling_layer: Optional[bool] = True, **kwargs):
        FastEsmPreTrainedModel.__init__(self, config, **kwargs)
        self.config = config
        self.embeddings = EsmEmbeddings(config)
        self.encoder = EsmEncoder(config)
        self.contact_head = EsmContactPredictionHead(
            in_features=config.num_hidden_layers * config.num_attention_heads, bias=True
        )
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        token_embedding_output = self.embeddings(input_ids, attention_mask=attention_mask)
        encoder_outputs = self.encoder(
            token_embedding_output,
            attention_mask=attention_mask,
            output_hidden_states=False,
            output_attentions=False,
        )
        return encoder_outputs.last_hidden_state

    def predict_contacts(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attns = self(input_ids, attention_mask=attention_mask, output_attentions=True).attentions
        attns = torch.stack(attns, dim=1)
        attns *= attention_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        attns *= attention_mask.unsqueeze(1).unsqueeze(2).unsqueeze(4)
        return self.contact_head(input_ids, attns)

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
    ) -> FastEsmEncoderOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
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
        )

        return FastEsmEncoderOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            s_max=encoder_outputs.s_max,
        )


class FastEsmModel(FastEsmPreTrainedModel, EmbeddingMixin):
    def __init__(self, config, add_pooling_layer: Optional[bool] = True, **kwargs):
        FastEsmPreTrainedModel.__init__(self, config, **kwargs)
        self.config = config
        self.esm = FAST_ESM_ENCODER(config)
        self.pooler = EsmPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self):
        return self.esm.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.esm.embeddings.word_embeddings = value

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.esm._embed(input_ids, attention_mask)

    def predict_contacts(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.esm.predict_contacts(input_ids, attention_mask=attention_mask)

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
        **kwargs,
    ) -> FastEsmEncoderOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
        )
        sequence_output = outputs.last_hidden_state
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return FastEsmEncoderOutput(
            last_hidden_state=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            s_max=outputs.s_max,
        )


class FastEsmForMaskedLM(FastEsmPreTrainedModel, EmbeddingMixin):
    def __init__(self, config, **kwargs):
        FastEsmPreTrainedModel.__init__(self, config, **kwargs)
        self.esm = FAST_ESM_ENCODER(config, add_pooling_layer=False)
        self.lm_head = EsmLMHead(config)
        self.loss_fct = nn.CrossEntropyLoss()
        self.post_init()

    def get_input_embeddings(self):
        return self.esm.embeddings.word_embeddings

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.esm._embed(input_ids, attention_mask)

    def predict_contacts(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.esm.predict_contacts(input_ids, attention_mask=attention_mask)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_s_max: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> EsmMaskedLMOutput:
        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
        )
        sequence_output = outputs.last_hidden_state
        prediction_scores = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(prediction_scores.device)
            loss = self.loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        return EsmMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            last_hidden_state=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            s_max=outputs.s_max,
        )


class FastEsmForSequenceClassification(FastEsmPreTrainedModel, EmbeddingMixin):
    def __init__(self, config, **kwargs):
        FastEsmPreTrainedModel.__init__(self, config, **kwargs)
        self.num_labels = config.num_labels
        self.config = config
        self.esm = FAST_ESM_ENCODER(config, add_pooling_layer=False)
        self.classifier = EsmClassificationHead(config)
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.post_init()

    def get_input_embeddings(self):
        return self.esm.embeddings.word_embeddings

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.esm._embed(input_ids, attention_mask)

    def predict_contacts(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.esm.predict_contacts(input_ids, attention_mask=attention_mask)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_s_max: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> EsmMaskedLMOutput:
        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
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

        return EsmMaskedLMOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            s_max=outputs.s_max,
        )


class FastEsmForTokenClassification(FastEsmPreTrainedModel, EmbeddingMixin):
    def __init__(self, config, **kwargs):
        FastEsmPreTrainedModel.__init__(self, config, **kwargs)
        self.num_labels = config.num_labels
        self.esm = FAST_ESM_ENCODER(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        self.post_init()

    def get_input_embeddings(self):
        return self.esm.embeddings.word_embeddings

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.esm._embed(input_ids, attention_mask)

    def predict_contacts(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.esm.predict_contacts(input_ids, attention_mask=attention_mask)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_s_max: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> EsmMaskedLMOutput:
        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_s_max=output_s_max,
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return EsmMaskedLMOutput(
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

    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    num_attention_heads = random.choice([2, 4])
    config = FastEsmConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=16 * num_attention_heads,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=random.choice([1, 2]),
        intermediate_size=64 * num_attention_heads,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_position_embeddings=256,
        emb_layer_norm_before=False,
        position_embedding_type="rotary",
        attn_backend="sdpa",
    )
    batch = tokenizer(["ACDEFG", "MKTW"], return_tensors="pt", padding="longest")
    batch["labels"] = batch["input_ids"].clone()
    model = FastEsmForMaskedLM(config=config).eval()

    with torch.no_grad():
        output = model(**batch, return_dict=True)

    print("Batch shape:")
    print_tensor_shapes("", batch)
    print("Output shape:")
    print_tensor_shapes("", output)
