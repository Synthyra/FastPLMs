from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import (
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    T5ForConditionalGeneration,
)
from transformers.modeling_outputs import ModelOutput

try:
    from fastplms.attention import (
        AttentionBackend,
        FastPLMsAttentionMixin,
        bool_to_additive_mask,
        get_attention_mask,
        resolve_attention_backend,
    )
    from fastplms.embeddings import EmbeddingMixin, select_hidden_state_embeddings
    from fastplms.models.ttt import FastPLMTestTimeTrainingMixin
except ImportError:
    pass  # Running as HF Hub composite; shared definitions are above


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AnkhEncoderOutput(ModelOutput):
    last_hidden_state: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


@dataclass
class AnkhMaskedLMOutput(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    last_hidden_state: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class FastAnkhConfig(PretrainedConfig):
    model_type = "fast_ankh"
    attribute_map: ClassVar[dict[str, str]] = {
        "head_dim": "d_kv",
        "hidden_size": "d_model",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        vocab_size: int = 144,
        d_model: int = 768,
        d_kv: int = 64,
        d_ff: int = 3072,
        num_heads: int = 12,
        num_layers: int = 48,
        num_decoder_layers: int | None = None,
        relative_attention_num_buckets: int = 64,
        relative_attention_max_distance: int = 128,
        dense_act_fn: str = "gelu_new",
        feed_forward_proj: str | None = None,
        dropout_rate: float = 0.0,
        layer_norm_epsilon: float = 1e-6,
        initializer_factor: float = 1.0,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        decoder_start_token_id: int | None = None,
        use_cache: bool = True,
        tie_word_embeddings: bool = True,
        attn_backend: str | None = None,
        **kwargs,
    ):
        if feed_forward_proj is None:
            feed_forward_proj = (
                "gated-gelu" if dense_act_fn == "gelu_new" else f"gated-{dense_act_fn}"
            )
        if decoder_start_token_id is None:
            decoder_start_token_id = pad_token_id
        serialized_encoder_decoder = kwargs.pop("is_encoder_decoder", True)
        if serialized_encoder_decoder is not True:
            raise ValueError(
                "FastAnkhConfig requires is_encoder_decoder=true to match the official "
                "T5 configuration."
            )
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            is_encoder_decoder=True,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_decoder_layers = num_layers if num_decoder_layers is None else num_decoder_layers
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dense_act_fn = dense_act_fn
        self.feed_forward_proj = feed_forward_proj
        self.is_gated_act = feed_forward_proj.startswith("gated-")
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.use_cache = use_cache
        self.scale_decoder_outputs = tie_word_embeddings
        self.tie_word_embeddings = tie_word_embeddings
        self.attn_backend = attn_backend

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()
        return output


def _load_ankh_tokenizer(config: FastAnkhConfig):
    """Load the tokenizer from the same immutable checkpoint as the model."""
    name_or_path = str(getattr(config, "_name_or_path", "")).strip()
    if not name_or_path:
        raise RuntimeError(
            "ANKH tokenizer loading requires a model loaded with from_pretrained "
            "so checkpoint provenance is available."
        )
    revision = getattr(config, "_commit_hash", None)
    tokenizer_kwargs = {"revision": revision} if revision else {}
    return AutoTokenizer.from_pretrained(name_or_path, **tokenizer_kwargs)


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
    return (
        0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    )


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

    def compute_bias(
        self, query_length: int, key_length: int, device: torch.device
    ) -> torch.Tensor:
        """Compute the position-bias tensor A with shape (1, h, q, k)."""
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        buckets = self._relative_position_bucket(
            relative_position,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(buckets)  # A has shape (q, k, h).
        return values.permute(2, 0, 1).unsqueeze(0)  # A has shape (1, h, q, k).

    # ---- Forward ----

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_4d: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Returns (attn_output, attn_weights_or_none, position_bias)."""
        batch_size, seq_length = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_length, self.num_heads, self.d_kv)

        query_heads = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        key_heads = self.k(hidden_states).view(hidden_shape).transpose(1, 2)
        value_heads = self.v(hidden_states).view(hidden_shape).transpose(1, 2)

        # The first layer computes the bias once; later layers reuse it.
        if position_bias is None and self.has_relative_attention_bias:
            position_bias = self.compute_bias(seq_length, seq_length, hidden_states.device)
            # Fold padding mask into position bias so layers don't need separate mask.
            if attention_mask_4d is not None:
                position_bias = position_bias + bool_to_additive_mask(
                    attention_mask_4d, position_bias.dtype
                )

        if output_attentions:
            attn_output, attn_weights = self._manual_attn(
                query_heads, key_heads, value_heads, position_bias
            )
            return self.o(attn_output), attn_weights, position_bias

        if self.attn_backend == AttentionBackend.EAGER:
            attn_output, _ = self._manual_attn(query_heads, key_heads, value_heads, position_bias)
        elif self.attn_backend == AttentionBackend.SDPA:
            attn_output = self._sdpa_attn(query_heads, key_heads, value_heads, position_bias)
        else:
            raise AssertionError(f"Unsupported backend for ANKH: {self.attn_backend}")

        return self.o(attn_output), None, position_bias

    def _sdpa_attn(
        self,
        query_heads: torch.Tensor,
        key_heads: torch.Tensor,
        value_heads: torch.Tensor,
        position_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        # A is the additive position bias with shape (1, h, q, k), including padding.
        # Official ANKH computes the QK and probability-value reductions in the
        # input dtype. Math SDPA otherwise promotes these reductions, which
        # changes the BF16 residual stream after many encoder layers.
        previous_reduction_policy = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
        try:
            with sdpa_kernel(SDPBackend.MATH):
                context_heads = F.scaled_dot_product_attention(
                    query_heads,
                    key_heads,
                    value_heads,
                    attn_mask=position_bias,
                    scale=self.scale,
                )
        finally:
            torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(previous_reduction_policy)
        return (
            context_heads.transpose(1, 2)
            .contiguous()
            .view(query_heads.shape[0], -1, self.inner_dim)
        )

    def _manual_attn(
        self,
        query_heads: torch.Tensor,
        key_heads: torch.Tensor,
        value_heads: torch.Tensor,
        position_bias: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_weights = torch.matmul(query_heads, key_heads.transpose(-1, -2)) * self.scale
        if position_bias is not None:
            attn_weights = attn_weights + position_bias
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        context_heads = torch.matmul(attn_weights, value_heads)
        attn_output = (
            context_heads.transpose(1, 2)
            .contiguous()
            .view(query_heads.shape[0], -1, self.inner_dim)
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
        attention_mask_4d: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        normed = self.layer_norm(hidden_states)
        attn_output, attn_weights, position_bias = self.SelfAttention(
            normed,
            attention_mask_4d=attention_mask_4d,
            position_bias=position_bias,
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
        self.layer = nn.ModuleList(
            [
                AnkhSelfAttentionLayer(config, has_relative_attention_bias),
                AnkhFFLayer(config),
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_4d: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        hidden_states, attn_weights, position_bias = self.layer[0](
            hidden_states,
            attention_mask_4d=attention_mask_4d,
            position_bias=position_bias,
            output_attentions=output_attentions,
        )
        hidden_states = self.layer[1](hidden_states)
        return hidden_states, attn_weights, position_bias


# ---------------------------------------------------------------------------
# PreTrainedModel base
# ---------------------------------------------------------------------------


class AnkhPreTrainedModel(FastPLMsAttentionMixin, PreTrainedModel):
    config_class = FastAnkhConfig
    base_model_prefix = "encoder"
    supports_gradient_checkpointing = True
    _no_split_modules: ClassVar[list[str]] = ["AnkhBlock"]
    _supports_flash_attn_2 = False
    _supports_flash_attn_3 = False
    _supports_flex_attn = False
    _fastplms_attention_implementations = ("eager", "sdpa")
    embedding_unsupported_pooling = ("cls",)

    @classmethod
    def is_remote_code(cls) -> bool:
        return True

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        factor = self.config.initializer_factor
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=factor * (self.config.d_model**-0.5))
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
        if backend not in self._fastplms_attention_implementations:
            raise ValueError(
                f"{type(self).__name__} does not support {backend!r}; expected one of "
                f"{self._fastplms_attention_implementations}."
            )
        self.config.attn_backend = backend
        resolved = resolve_attention_backend(backend)
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
        if resolved.is_flash:
            raise ValueError(
                "ANKH does not support FlashAttention because it requires relative position bias."
            )
        self.attention_backend = resolved

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.block = nn.ModuleList(
            [
                AnkhBlock(config, has_relative_attention_bias=(i == 0))
                for i in range(config.num_layers)
            ]
        )
        for blk in self.block:
            blk.layer[0].SelfAttention.attn_backend = self.attention_backend

        self.final_layer_norm = AnkhRMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.gradient_checkpointing = False
        self._fastplms_tokenizer = None
        self.post_init()

    @property
    def tokenizer(self):
        """Load the checkpoint tokenizer only when a sequence API needs it."""
        if self._fastplms_tokenizer is None:
            self._fastplms_tokenizer = _load_ankh_tokenizer(self.config)
        return self._fastplms_tokenizer

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        hidden_state_index: int = -1,
        store_all_hidden_states: bool = False,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        output_hidden_states = store_all_hidden_states or hidden_state_index != -1
        encoder_output = self._run_encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )
        return select_hidden_state_embeddings(
            encoder_output.last_hidden_state,
            encoder_output.hidden_states,
            hidden_state_index=hidden_state_index,
            store_all_hidden_states=store_all_hidden_states,
        )

    def _run_encoder(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> AnkhEncoderOutput:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        batch_size, seq_len = hidden_states.shape[:2]
        _, attention_mask_4d, _ = get_attention_mask(
            effective_backend=self.attention_backend,
            batch_size=batch_size,
            seq_len=seq_len,
            device=hidden_states.device,
            attention_mask=attention_mask,
            dtype=hidden_states.dtype,
            mask_semantics="padding",
        )

        position_bias = None

        for layer_module in self.block:
            if output_hidden_states:
                all_hidden_states = (*all_hidden_states, hidden_states)

            if self.gradient_checkpointing and self.training:
                hidden_states, attn_weights, position_bias = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask_4d,
                    position_bias,
                    output_attentions,
                )
            else:
                hidden_states, attn_weights, position_bias = layer_module(
                    hidden_states,
                    attention_mask_4d=attention_mask_4d,
                    position_bias=position_bias,
                    output_attentions=output_attentions,
                )

            if all_attentions is not None:
                all_attentions = (*all_attentions, attn_weights)

        hidden_states = self.final_layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        return AnkhEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        **kwargs,
    ) -> AnkhEncoderOutput:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

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

    _tied_weights_keys: ClassVar[dict[str, str]] = {
        "encoder.embed_tokens.weight": "shared.weight"
    }

    def __init__(self, config: FastAnkhConfig, **kwargs):
        AnkhPreTrainedModel.__init__(self, config, **kwargs)
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = FAST_ANKH_ENCODER(config)
        self.encoder.embed_tokens = self.shared
        self.post_init()

    @property
    def tokenizer(self):
        return self.encoder.tokenizer

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = value

    def _embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        hidden_state_index: int = -1,
        store_all_hidden_states: bool = False,
    ) -> torch.Tensor:
        return self.encoder._embed(
            input_ids,
            attention_mask,
            hidden_state_index=hidden_state_index,
            store_all_hidden_states=store_all_hidden_states,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        **kwargs,
    ) -> AnkhEncoderOutput:
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )


class FastAnkhForMaskedLMExtension(
    FastPLMTestTimeTrainingMixin, AnkhPreTrainedModel, EmbeddingMixin
):
    """ANKH encoder with LM head for masked language modeling.

    NOTE: The LM head is initialized from the shared embedding weights but is NOT
    tied. The original ANKH models were trained with T5's span corruption objective
    using an encoder-decoder architecture. This encoder-only MaskedLM variant is
    not pre-trained for standard MLM and requires additional fine-tuning.
    """

    _tied_weights_keys: ClassVar[dict[str, str]] = {
        "encoder.embed_tokens.weight": "shared.weight"
    }

    def __init__(self, config: FastAnkhConfig, **kwargs):
        # The historical Synthyra extension stores an independent output head.
        config.tie_word_embeddings = False
        AnkhPreTrainedModel.__init__(self, config, **kwargs)
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = FAST_ANKH_ENCODER(config)
        self.encoder.embed_tokens = self.shared
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.loss_fct = nn.CrossEntropyLoss()
        self.post_init()
        self.init_ttt({"lora_target_replace_module": "AnkhSelfAttention"})

    @property
    def tokenizer(self):
        return self.encoder.tokenizer

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        hidden_state_index: int = -1,
        store_all_hidden_states: bool = False,
    ) -> torch.Tensor:
        return self.encoder._embed(
            input_ids,
            attention_mask,
            hidden_state_index=hidden_state_index,
            store_all_hidden_states=store_all_hidden_states,
        )

    def _ttt_get_trainable_modules(self) -> list[nn.Module]:
        return [self.encoder]

    def _ttt_tokenize(
        self,
        seq: str | list[str] | None = None,
        input_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if input_ids is not None:
            return input_ids
        assert seq is not None, "Pass either seq or input_ids for ANKH TTT."
        sequences = [seq] if isinstance(seq, str) else seq
        spaced_sequences = [" ".join(sequence) for sequence in sequences]
        tokenized = self.tokenizer(spaced_sequences, return_tensors="pt", padding=True)
        return tokenized["input_ids"]

    def _ttt_replacement_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        ids = [self.tokenizer.convert_tokens_to_ids(aa) for aa in amino_acids]
        return torch.tensor(ids, device=input_ids.device, dtype=input_ids.dtype)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
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


class FastAnkhForConditionalGeneration(T5ForConditionalGeneration, EmbeddingMixin):
    """Official ANKH sequence-to-sequence architecture with exact T5 state keys.

    ANKH generation checkpoints are ordinary T5 conditional-generation models.
    This class intentionally delegates their decoder, cross-attention, language
    model head, caching, generation, and tied-weight behavior to Transformers.
    The optimized encoder-only implementation remains available through
    :class:`FastAnkhModel`.
    """

    config_class = FastAnkhConfig
    embedding_unsupported_pooling = ("cls",)
    _fastplms_attention_implementations = ("eager",)

    def __init__(self, config: FastAnkhConfig, **kwargs):
        requested_backend = getattr(config, "_attn_implementation", None) or config.attn_backend
        if requested_backend not in (None, "eager"):
            raise ValueError(
                "ANKH sequence-to-sequence checkpoints support only eager attention; "
                f"received {requested_backend!r}. Use FastAnkhModel for optimized "
                "encoder embeddings."
            )
        super().__init__(config, **kwargs)
        self._fastplms_tokenizer = None

    @property
    def tokenizer(self):
        if self._fastplms_tokenizer is None:
            self._fastplms_tokenizer = _load_ankh_tokenizer(self.config)
        return self._fastplms_tokenizer

    def _embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        hidden_state_index: int = -1,
        store_all_hidden_states: bool = False,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=store_all_hidden_states or hidden_state_index != -1,
            return_dict=True,
        )
        return select_hidden_state_embeddings(
            outputs.last_hidden_state,
            outputs.hidden_states,
            hidden_state_index=hidden_state_index,
            store_all_hidden_states=store_all_hidden_states,
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

    def _embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        hidden_state_index: int = -1,
        store_all_hidden_states: bool = False,
    ) -> torch.Tensor:
        return self.encoder._embed(
            input_ids,
            attention_mask,
            hidden_state_index=hidden_state_index,
            store_all_hidden_states=store_all_hidden_states,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
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
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss = (
                    self.mse(logits.squeeze(), labels.squeeze())
                    if self.num_labels == 1
                    else self.mse(logits, labels)
                )
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

    def _embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        hidden_state_index: int = -1,
        store_all_hidden_states: bool = False,
    ) -> torch.Tensor:
        return self.encoder._embed(
            input_ids,
            attention_mask,
            hidden_state_index=hidden_state_index,
            store_all_hidden_states=store_all_hidden_states,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
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
