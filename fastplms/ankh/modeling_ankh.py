from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer
from transformers.modeling_outputs import ModelOutput

try:
    from fastplms.attention import (
        AttentionBackend, VALID_ATTENTION_BACKENDS,
        resolve_attention_backend, get_attention_mask, bool_to_additive_mask,
        _get_flex_attention_fn,
        create_block_mask, flex_attention, BlockMask,
    )
    from fastplms.embedding_mixin import Pooler, EmbeddingMixin, ProteinDataset, parse_fasta, build_collator
except ImportError:
    pass  # Running as HF Hub composite; shared definitions are above


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AnkhEncoderOutput(ModelOutput):
    last_hidden_state: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


@dataclass
class AnkhMaskedLMOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class FastAnkhConfig(PretrainedConfig):
    model_type = "fast_ankh"
    attribute_map = {"hidden_size": "d_model"}

    def __init__(
        self,
        vocab_size: int = 144,
        d_model: int = 768,
        d_kv: int = 64,
        d_ff: int = 3072,
        num_heads: int = 12,
        num_layers: int = 48,
        relative_attention_num_buckets: int = 64,
        relative_attention_max_distance: int = 128,
        dense_act_fn: str = "gelu_new",
        layer_norm_epsilon: float = 1e-6,
        initializer_factor: float = 1.0,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        attn_backend: str = "sdpa",
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dense_act_fn = dense_act_fn
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.tie_word_embeddings = False
        self.attn_backend = attn_backend

    def to_dict(self) -> Dict[str, Any]:
        output = super().to_dict()
        return output


# ---------------------------------------------------------------------------
# Submodules
# ---------------------------------------------------------------------------

class AnkhRMSNorm(nn.Module):
    """T5-style RMS layer norm: scales without mean subtraction or bias."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(self.weight.dtype)


def _gelu_new(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class AnkhGatedFFN(nn.Module):
    """T5-style gated feed-forward: activation(wi_0(x)) * wi_1(x) -> wo."""

    def __init__(self, config: FastAnkhConfig):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.act = F.silu if config.dense_act_fn == "silu" else _gelu_new

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.wo(self.act(self.wi_0(hidden_states)) * self.wi_1(hidden_states))


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class AnkhSelfAttention(nn.Module):
    """T5-style self-attention with relative position bias and multi-backend dispatch.

    Only layer 0 has ``has_relative_attention_bias=True`` and owns the
    ``nn.Embedding`` that produces the position bias.  All other layers
    receive the precomputed bias through the forward call.
    """

    def __init__(self, config: FastAnkhConfig, has_relative_attention_bias: bool = False):
        super().__init__()
        self.num_heads = config.num_heads
        self.d_kv = config.d_kv
        self.inner_dim = self.num_heads * self.d_kv
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance

        self.q = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, config.d_model, bias=False)
        # T5/ANKH attention is unscaled: scores = Q K^T (no 1/sqrt(d_kv)).
        # The learned relative position bias absorbs any temperature.
        self.scale = 1.0

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                config.relative_attention_num_buckets, config.num_heads
            )

        self.attn_backend: AttentionBackend = AttentionBackend.SDPA  # set by encoder

    # ---- T5 relative position bucketing ----

    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> torch.Tensor:
        """Bidirectional log-bucketed relative position mapping (T5 style)."""
        # Bidirectional: half buckets for negative, half for positive
        num_buckets //= 2
        relative_buckets = (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.clamp(relative_position_if_large, max=num_buckets - 1)

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length: int, key_length: int, device: torch.device) -> torch.Tensor:
        """Compute (1, H, Q, K) position bias tensor for SDPA / manual paths."""
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        buckets = self._relative_position_bucket(
            relative_position,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(buckets)  # (Q, K, H)
        return values.permute(2, 0, 1).unsqueeze(0)  # (1, H, Q, K)

    # ---- Forward ----

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor] = None,
        attention_mask_4d: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[BlockMask] = None,
        position_bias: Optional[torch.Tensor] = None,
        flex_score_mod=None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Returns (attn_output, attn_weights_or_none, position_bias)."""
        batch_size, seq_length = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_length, self.num_heads, self.d_kv)

        query_BHLD = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        key_BHLD = self.k(hidden_states).view(hidden_shape).transpose(1, 2)
        value_BHLD = self.v(hidden_states).view(hidden_shape).transpose(1, 2)

        # Compute position bias on first layer (SDPA/manual only; flex uses score_mod)
        if position_bias is None and self.has_relative_attention_bias and self.attn_backend != AttentionBackend.FLEX:
            position_bias = self.compute_bias(seq_length, seq_length, hidden_states.device)
            # Fold padding mask into position bias so layers don't need separate mask.
            if attention_mask_4d is not None:
                position_bias = position_bias + bool_to_additive_mask(attention_mask_4d, position_bias.dtype)

        if output_attentions:
            attn_output, attn_weights = self._manual_attn(query_BHLD, key_BHLD, value_BHLD, position_bias)
            return self.o(attn_output), attn_weights, position_bias

        if self.attn_backend == AttentionBackend.FLEX:
            attn_output = self._flex_attn(query_BHLD, key_BHLD, value_BHLD, flex_block_mask, flex_score_mod)
        elif self.attn_backend == AttentionBackend.SDPA:
            attn_output = self._sdpa_attn(query_BHLD, key_BHLD, value_BHLD, position_bias)
        else:
            raise AssertionError(f"Unsupported backend for ANKH: {self.attn_backend}")

        return self.o(attn_output), None, position_bias

    def _sdpa_attn(
        self,
        query_BHLD: torch.Tensor,
        key_BHLD: torch.Tensor,
        value_BHLD: torch.Tensor,
        position_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # SDPA: position_bias is (1, H, Q, K) additive bias (includes padding mask)
        context_BHLD = F.scaled_dot_product_attention(
            query_BHLD, key_BHLD, value_BHLD,
            attn_mask=position_bias,
            scale=self.scale,
        )
        return context_BHLD.transpose(1, 2).contiguous().view(
            query_BHLD.shape[0], -1, self.inner_dim
        )

    def _flex_attn(
        self,
        query_BHLD: torch.Tensor,
        key_BHLD: torch.Tensor,
        value_BHLD: torch.Tensor,
        flex_block_mask: Optional[BlockMask],
        flex_score_mod,
    ) -> torch.Tensor:
        assert flex_attention is not None, "Flex attention is not available."
        fn = _get_flex_attention_fn()
        context_BHLD = fn(
            query_BHLD, key_BHLD, value_BHLD,
            score_mod=flex_score_mod,
            block_mask=flex_block_mask,
            scale=self.scale,
        )
        return context_BHLD.transpose(1, 2).contiguous().view(
            query_BHLD.shape[0], -1, self.inner_dim
        )

    def _manual_attn(
        self,
        query_BHLD: torch.Tensor,
        key_BHLD: torch.Tensor,
        value_BHLD: torch.Tensor,
        position_bias: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_weights = torch.matmul(query_BHLD, key_BHLD.transpose(-1, -2)) * self.scale
        if position_bias is not None:
            attn_weights = attn_weights + position_bias
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        context_BHLD = torch.matmul(attn_weights, value_BHLD)
        attn_output = context_BHLD.transpose(1, 2).contiguous().view(
            query_BHLD.shape[0], -1, self.inner_dim
        )
        return attn_output, attn_weights


# ---------------------------------------------------------------------------
# Encoder block & stack (T5-compatible key naming)
# ---------------------------------------------------------------------------

class AnkhSelfAttentionLayer(nn.Module):
    """Wraps AnkhSelfAttention + layer_norm to match T5Block.layer[0] key naming."""

    def __init__(self, config: FastAnkhConfig, has_relative_attention_bias: bool = False):
        super().__init__()
        self.SelfAttention = AnkhSelfAttention(config, has_relative_attention_bias)
        self.layer_norm = AnkhRMSNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor] = None,
        attention_mask_4d: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[BlockMask] = None,
        position_bias: Optional[torch.Tensor] = None,
        flex_score_mod=None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        normed = self.layer_norm(hidden_states)
        attn_output, attn_weights, position_bias = self.SelfAttention(
            normed,
            attention_mask_2d=attention_mask_2d,
            attention_mask_4d=attention_mask_4d,
            flex_block_mask=flex_block_mask,
            position_bias=position_bias,
            flex_score_mod=flex_score_mod,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + attn_output
        return hidden_states, attn_weights, position_bias


class AnkhFFLayer(nn.Module):
    """Wraps AnkhGatedFFN + layer_norm to match T5Block.layer[1] key naming."""

    def __init__(self, config: FastAnkhConfig):
        super().__init__()
        self.DenseReluDense = AnkhGatedFFN(config)
        self.layer_norm = AnkhRMSNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.DenseReluDense(normed)
        return hidden_states


class AnkhBlock(nn.Module):
    """Single transformer block with T5-compatible .layer ModuleList naming."""

    def __init__(self, config: FastAnkhConfig, has_relative_attention_bias: bool = False):
        super().__init__()
        self.layer = nn.ModuleList([
            AnkhSelfAttentionLayer(config, has_relative_attention_bias),
            AnkhFFLayer(config),
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor] = None,
        attention_mask_4d: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[BlockMask] = None,
        position_bias: Optional[torch.Tensor] = None,
        flex_score_mod=None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        hidden_states, attn_weights, position_bias = self.layer[0](
            hidden_states,
            attention_mask_2d=attention_mask_2d,
            attention_mask_4d=attention_mask_4d,
            flex_block_mask=flex_block_mask,
            position_bias=position_bias,
            flex_score_mod=flex_score_mod,
            output_attentions=output_attentions,
        )
        hidden_states = self.layer[1](hidden_states)
        return hidden_states, attn_weights, position_bias


# ---------------------------------------------------------------------------
# PreTrainedModel base
# ---------------------------------------------------------------------------

class AnkhPreTrainedModel(PreTrainedModel):
    config_class = FastAnkhConfig
    base_model_prefix = "encoder"
    supports_gradient_checkpointing = True
    _no_split_modules = ["AnkhBlock"]

    @classmethod
    def is_remote_code(cls) -> bool:
        return True

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        factor = self.config.initializer_factor
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=factor * (self.config.d_model ** -0.5))
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, AnkhRMSNorm):
            module.weight.data.fill_(1.0)

    def post_init(self) -> None:
        super().post_init()

    def get_output_embeddings(self):
        return None

    @property
    def attn_backend(self) -> str:
        return self.config.attn_backend

    @attn_backend.setter
    def attn_backend(self, backend: str) -> None:
        assert backend in VALID_ATTENTION_BACKENDS, (
            f"Unsupported attn_backend: {backend}. Expected one of {VALID_ATTENTION_BACKENDS}."
        )
        self.config.attn_backend = backend
        resolved = resolve_attention_backend(backend)
        if resolved == AttentionBackend.KERNELS_FLASH:
            print("ANKH: kernels_flash -> flex/sdpa fallback")
            resolved = AttentionBackend.FLEX if flex_attention is not None else AttentionBackend.SDPA
        for module in self.modules():
            if isinstance(module, FAST_ANKH_ENCODER):
                module.attention_backend = resolved
            elif isinstance(module, AnkhSelfAttention):
                module.attn_backend = resolved


# ---------------------------------------------------------------------------
# FAST_ANKH_ENCODER (mirrors T5Stack key naming)
# ---------------------------------------------------------------------------

class FAST_ANKH_ENCODER(AnkhPreTrainedModel, EmbeddingMixin):
    """Inner encoder that mirrors T5Stack attribute naming for weight compliance.

    State dict keys: embed_tokens.*, block.{i}.layer.0.SelfAttention.*,
    block.{i}.layer.1.DenseReluDense.*, final_layer_norm.*.
    """

    def __init__(self, config: FastAnkhConfig, **kwargs):
        AnkhPreTrainedModel.__init__(self, config, **kwargs)
        self.config = config

        resolved = resolve_attention_backend(config.attn_backend)
        if resolved == AttentionBackend.KERNELS_FLASH:
            print("ANKH: kernels_flash not supported (relative position bias); falling back to flex/sdpa")
            resolved = AttentionBackend.FLEX if flex_attention is not None else AttentionBackend.SDPA
        self.attention_backend = resolved

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.block = nn.ModuleList([
            AnkhBlock(config, has_relative_attention_bias=(i == 0))
            for i in range(config.num_layers)
        ])
        for blk in self.block:
            blk.layer[0].SelfAttention.attn_backend = self.attention_backend

        self.final_layer_norm = AnkhRMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.gradient_checkpointing = False
        self.tokenizer = AutoTokenizer.from_pretrained("ElnaggarLab/ankh-base")
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @torch.compiler.disable
    def _compute_materialized_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Precompute full (Q, K, H) bias tensor for flex score_mod lookup."""
        bias_embedding = self.block[0].layer[0].SelfAttention.relative_attention_bias
        context_position = torch.arange(seq_len, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(seq_len, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        buckets = AnkhSelfAttention._relative_position_bucket(
            relative_position,
            num_buckets=self.config.relative_attention_num_buckets,
            max_distance=self.config.relative_attention_max_distance,
        )
        return bias_embedding(buckets)  # (Q, K, H)

    def _build_flex_score_mod(self, seq_len: int, device: torch.device):
        """Build score_mod closure that reads from materialized bias tensor."""
        bias = self._compute_materialized_bias(seq_len, device)

        def score_mod(score, b, h, q_idx, kv_idx):
            return score + bias[q_idx, kv_idx, h]

        return score_mod

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        encoder_output = self._run_encoder(hidden_states, attention_mask=attention_mask)
        return encoder_output.last_hidden_state

    def _run_encoder(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> AnkhEncoderOutput:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        batch_size, seq_len = hidden_states.shape[:2]
        attention_mask_2d, attention_mask_4d, flex_block_mask = get_attention_mask(
            effective_backend=self.attention_backend,
            batch_size=batch_size,
            seq_len=seq_len,
            device=hidden_states.device,
            attention_mask=attention_mask,
        )

        flex_score_mod = None
        position_bias = None
        if self.attention_backend == AttentionBackend.FLEX:
            flex_score_mod = self._build_flex_score_mod(seq_len, hidden_states.device)

        for layer_module in self.block:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                hidden_states, attn_weights, position_bias = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask_2d,
                    attention_mask_4d,
                    flex_block_mask,
                    position_bias,
                    flex_score_mod,
                    output_attentions,
                )
            else:
                hidden_states, attn_weights, position_bias = layer_module(
                    hidden_states,
                    attention_mask_2d=attention_mask_2d,
                    attention_mask_4d=attention_mask_4d,
                    flex_block_mask=flex_block_mask,
                    position_bias=position_bias,
                    flex_score_mod=flex_score_mod,
                    output_attentions=output_attentions,
                )

            if all_attentions is not None:
                all_attentions = all_attentions + (attn_weights,)

        hidden_states = self.final_layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return AnkhEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        **kwargs,
    ) -> AnkhEncoderOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            hidden_states = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        return self._run_encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states or False,
            output_attentions=output_attentions or False,
        )


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------

class FastAnkhModel(AnkhPreTrainedModel, EmbeddingMixin):
    """ANKH encoder model for embedding extraction."""

    def __init__(self, config: FastAnkhConfig, **kwargs):
        AnkhPreTrainedModel.__init__(self, config, **kwargs)
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = FAST_ANKH_ENCODER(config)
        self.post_init()

    @property
    def tokenizer(self):
        return self.encoder.tokenizer

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def set_input_embeddings(self, value):
        self.encoder.embed_tokens = value

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder._embed(input_ids, attention_mask)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        **kwargs,
    ) -> AnkhEncoderOutput:
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )


class FastAnkhForMaskedLM(AnkhPreTrainedModel, EmbeddingMixin):
    """ANKH encoder with LM head for masked language modeling.

    NOTE: The LM head is initialized from the shared embedding weights but is NOT
    tied. The original ANKH models were trained with T5's span corruption objective
    using an encoder-decoder architecture. This encoder-only MaskedLM variant is
    not pre-trained for standard MLM and requires additional fine-tuning.
    """

    def __init__(self, config: FastAnkhConfig, **kwargs):
        AnkhPreTrainedModel.__init__(self, config, **kwargs)
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = FAST_ANKH_ENCODER(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.loss_fct = nn.CrossEntropyLoss()
        self.post_init()

    @property
    def tokenizer(self):
        return self.encoder.tokenizer

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def set_input_embeddings(self, value):
        self.encoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder._embed(input_ids, attention_mask)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        **kwargs,
    ) -> AnkhMaskedLMOutput:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        sequence_output = outputs.last_hidden_state
        logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return AnkhMaskedLMOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FastAnkhForSequenceClassification(AnkhPreTrainedModel, EmbeddingMixin):
    def __init__(self, config: FastAnkhConfig, **kwargs):
        AnkhPreTrainedModel.__init__(self, config, **kwargs)
        self.num_labels = config.num_labels
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = FAST_ANKH_ENCODER(config)
        self.classifier = nn.Linear(config.d_model, config.num_labels)
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.post_init()

    @property
    def tokenizer(self):
        return self.encoder.tokenizer

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder._embed(input_ids, attention_mask)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        **kwargs,
    ) -> AnkhMaskedLMOutput:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        # Pool: mean over non-padding tokens
        sequence_output = outputs.last_hidden_state
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(sequence_output.dtype)
            pooled = (sequence_output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = sequence_output.mean(dim=1)
        logits = self.classifier(pooled)

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
                loss = self.mse(logits.squeeze(), labels.squeeze()) if self.num_labels == 1 else self.mse(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = self.ce(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = self.bce(logits, labels)

        return AnkhMaskedLMOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FastAnkhForTokenClassification(AnkhPreTrainedModel, EmbeddingMixin):
    def __init__(self, config: FastAnkhConfig, **kwargs):
        AnkhPreTrainedModel.__init__(self, config, **kwargs)
        self.num_labels = config.num_labels
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = FAST_ANKH_ENCODER(config)
        self.classifier = nn.Linear(config.d_model, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        self.post_init()

    @property
    def tokenizer(self):
        return self.encoder.tokenizer

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder._embed(input_ids, attention_mask)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        **kwargs,
    ) -> AnkhMaskedLMOutput:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return AnkhMaskedLMOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
