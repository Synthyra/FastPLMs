import entrypoint_setup
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple, Union, Dict, Any
from einops import rearrange
from dataclasses import dataclass
from transformers import PreTrainedModel, PretrainedConfig, EsmTokenizer
from transformers import initialization as init
from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput,
    TokenClassifierOutput
)
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
    from torch.nn.attention.flex_attention import create_block_mask
    from torch.nn.attention.flex_attention import flex_attention
except ImportError:
    create_block_mask = None
    flex_attention = None

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


@dataclass
class EsmMaskedLMOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


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
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
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
        self.attn_backend = config.attn_backend
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        self.rotary_embeddings = None
        if self.position_embedding_type == "rotary":
            self.rotary_embeddings = RotaryEmbedding(dim=self.attention_head_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, 'b s (h d) -> b h s d', h=self.num_attention_heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[object] = None,
        output_attentions: Optional[bool] = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for self attention.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask
            output_attentions: Whether to return attention weights
            
        Returns:
            Output tensor and optionally attention weights
        """
        query_layer = self.transpose_for_scores(self.query(hidden_states)) * self.scale
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        if self.position_embedding_type == "rotary":
            query_layer, key_layer = self.rotary_embeddings(query_layer, key_layer)

        if output_attentions:
            # Manual attention computation - apply scaling here
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            if attention_mask is not None:
                attention_scores = attention_scores.masked_fill(attention_mask.logical_not(), float("-inf"))
            attention_probs = F.softmax(attention_scores, dim=-1)
            if self.dropout_prob > 0:
                attention_probs = F.dropout(attention_probs, p=self.dropout_prob, training=self.training)
            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = rearrange(context_layer, 'b h s d -> b s (h d)')
            return context_layer, attention_probs
        else:
            if self.attn_backend == "flex":
                assert flex_attention is not None, "Flex attention backend requested but torch.flex_attention is unavailable."
                assert query_layer.dtype in (torch.float16, torch.bfloat16), (
                    f"Flex attention backend requires float16 or bfloat16, got {query_layer.dtype}."
                )
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
                sdpa_mask = None
                if attention_mask is not None:
                    sdpa_mask = torch.zeros_like(attention_mask, dtype=query_layer.dtype)
                    sdpa_mask.masked_fill_(attention_mask.logical_not(), float("-inf"))
                context_layer = F.scaled_dot_product_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    attn_mask=sdpa_mask,
                    dropout_p=self.dropout_prob if self.training else 0.0,
                    scale=1.0
                )
            context_layer = rearrange(context_layer, 'b h s d -> b s (h d)')
            return context_layer
        

class EsmAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = EsmSelfAttention(config)
        self.output = EsmSelfOutput(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[object] = None,
        output_attentions: Optional[bool] = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for attention layer.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask
            output_attentions: Whether to return attention weights
            
        Returns:
            Output tensor and optionally attention weights
        """
        hidden_states_ln = self.LayerNorm(hidden_states)
        self_outputs = self.self(
            hidden_states_ln,
            attention_mask,
            flex_block_mask,
            output_attentions,
        )
        if output_attentions:
            attention_output, attention_weights = self_outputs
            attention_output = self.output(attention_output, hidden_states)
            return attention_output, attention_weights
        else:
            attention_output = self_outputs
            return self.output(attention_output, hidden_states)


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
        attention_mask: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[object] = None,
        output_attentions: Optional[bool] = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for transformer layer.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask
            output_attentions: Whether to return attention weights
            
        Returns:
            Output tensor and optionally attention weights
        """
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            flex_block_mask,
            output_attentions,
        )
        if output_attentions:
            attention_output, attention_weights = attention_outputs
        else:
            attention_output = attention_outputs
            attention_weights = None

        layer_output = self.feed_forward_chunk(attention_output)
        
        if output_attentions:
            return layer_output, attention_weights
        return layer_output

    def feed_forward_chunk(self, attention_output):
        attention_output_ln = self.LayerNorm(attention_output)
        intermediate_output = self.intermediate(attention_output_ln)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class EsmEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([EsmLayer(config) for _ in range(config.num_hidden_layers)])
        self.emb_layer_norm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[object] = None,
        output_hidden_states: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        """Forward pass for transformer encoder.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights
            
        Returns:
            BaseModelOutputWithPastAndCrossAttentions containing model outputs
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer_module in self.layer:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    flex_block_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    flex_block_mask,
                    output_attentions,
                )

            if output_attentions:
                hidden_states, attention_weights = layer_outputs
                all_attentions = all_attentions + (attention_weights,)
            else:
                hidden_states = layer_outputs

        if self.emb_layer_norm_after:
            hidden_states = self.emb_layer_norm_after(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
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

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        super()._init_weights(module)
        if isinstance(module, EsmLMHead):
            init.zeros_(module.bias)
        elif isinstance(module, EsmEmbeddings):
            init.copy_(module.position_ids, torch.arange(module.position_ids.shape[-1]).expand((1, -1)))
        elif isinstance(module, RotaryEmbedding):
            inv_freq = 1.0 / (10000 ** (torch.arange(0, module.dim, 2, dtype=torch.int64).float() / module.dim))
            init.copy_(module.inv_freq, inv_freq)

    def get_output_embeddings(self):
        # NOTE: get_output_embeddings() must return None to prevent accidental weight tying.
        # See e.g. https://github.com/huggingface/transformers/pull/39339#discussion_r2219126400
        return None



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
        batch_size, seq_length = input_ids.shape
        flex_block_mask = None
        if attention_mask is not None:
            token_attention_mask = attention_mask.bool()
            if self.config.attn_backend == "flex":
                assert create_block_mask is not None, (
                    "Flex attention backend requested but torch.create_block_mask is unavailable."
                )
                flex_block_mask = _create_pad_block_mask(token_attention_mask)
                extended_attention_mask = None
            else:
                extended_attention_mask = token_attention_mask[:, None, None, :].expand(
                    batch_size, 1, seq_length, seq_length
                )
        else:
            extended_attention_mask = None
        encoder_outputs = self.encoder(
            token_embedding_output,
            attention_mask=extended_attention_mask,
            flex_block_mask=flex_block_mask,
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
        return_dict: Optional[bool] = None, # to play nice with HF adjacent packages
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        """Forward pass for base model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            inputs_embeds: Optional input embeddings
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights
            
        Returns:
            Model outputs including hidden states and optionally attention weights
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        token_embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        flex_block_mask = None
        if attention_mask is not None:
            token_attention_mask = attention_mask.bool()
            if (
                self.config.attn_backend == "flex"
                and not output_attentions
            ):
                assert create_block_mask is not None, (
                    "Flex attention backend requested but torch.create_block_mask is unavailable."
                )
                flex_block_mask = _create_pad_block_mask(token_attention_mask)
                extended_attention_mask = None
            else:
                extended_attention_mask = token_attention_mask[:, None, None, :].expand(
                    batch_size, 1, seq_length, seq_length
                )
        else:
            extended_attention_mask = None

        encoder_outputs = self.encoder(
            token_embedding_output,
            attention_mask=extended_attention_mask,
            flex_block_mask=flex_block_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        sequence_output = encoder_outputs.last_hidden_state

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
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
        return_dict: Optional[bool] = None, # to play nice with HF adjacent packages
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        """Forward pass for base model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            inputs_embeds: Optional input embeddings
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights
            
        Returns:
            Model outputs including hidden states and optionally attention weights
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        sequence_output = outputs.last_hidden_state
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FastEsmForMaskedLM(FastEsmPreTrainedModel, EmbeddingMixin):
    def __init__(self, config, **kwargs):
        FastEsmPreTrainedModel.__init__(self, config, **kwargs)
        self.esm = FAST_ESM_ENCODER(config, add_pooling_layer=False)
        self.lm_head = EsmLMHead(config)
        self.loss_fct = nn.CrossEntropyLoss()
        self.init_weights()

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
        return_dict: Optional[bool] = None, # to play nice with HF adjacent packages
        **kwargs,
    ) -> Union[Tuple, EsmMaskedLMOutput]:
        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
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
        self.init_weights()

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
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
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


class FastEsmForTokenClassification(FastEsmPreTrainedModel, EmbeddingMixin):
    def __init__(self, config, **kwargs):
        FastEsmPreTrainedModel.__init__(self, config, **kwargs)
        self.num_labels = config.num_labels
        self.esm = FAST_ESM_ENCODER(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        self.init_weights()

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
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, TokenClassifierOutput]:
        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
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


