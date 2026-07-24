from __future__ import annotations

import math
import torch
import torch.nn as nn
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, ClassVar
from tokenizers import pre_tokenizers
from torch.nn import functional as F
from transformers import (
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    T5ForConditionalGeneration,
)
from transformers.modeling_outputs import (
    MaskedLMOutput,
    ModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)


try:
    from fastplms.attention import (
        AttentionBackend,
        FastPLMsAttentionMixin,
        bool_to_additive_mask,
        get_attention_mask,
        resolve_attention_backend,
        resolve_attention_backend_for_call,
        set_config_attn_implementation,
    )
    from fastplms.embeddings import (
        EmbeddingBatch,
        EmbeddingMixin,
        select_hidden_state_embeddings,
    )
    from fastplms.models.ttt import FastPLMTestTimeTrainingMixin
except ModuleNotFoundError as error:
    _COMPOSITE_REQUIRED_NAMES = (
        "AttentionBackend",
        "EmbeddingBatch",
        "EmbeddingMixin",
        "FastPLMsAttentionMixin",
        "FastPLMTestTimeTrainingMixin",
        "bool_to_additive_mask",
        "get_attention_mask",
        "resolve_attention_backend",
        "resolve_attention_backend_for_call",
        "select_hidden_state_embeddings",
        "set_config_attn_implementation",
    )
    if error.name != "fastplms" or any(
        name not in globals() for name in _COMPOSITE_REQUIRED_NAMES
    ):
        raise
    # Legacy flat Hub composites define every shared symbol above this block.


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
        if isinstance(dropout_rate, bool) or not isinstance(dropout_rate, Real):
            raise TypeError("dropout_rate must be a real number in [0, 1).")
        dropout_rate = float(dropout_rate)
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1).")
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


_TOKENIZER_LOAD_CONTEXT_KEYS = (
    "cache_dir",
    "force_download",
    "local_files_only",
    "proxies",
    "subfolder",
    "token",
    "trust_remote_code",
)


def configure_ankh_tokenizer(tokenizer: Any) -> Any:
    """Apply ANKH's residue-aware pre-tokenizer to a tokenizer instance.

    The tokenizer files published by the official checkpoints use a leading
    metaspace convention intended for natural-language text.  Protein inputs
    are already residue-delimited, so retaining that convention emits a
    leading ``<unk>`` token.  FastPLMs configures the fast tokenizer to split
    raw residue strings and tight sentinel prompts without manufacturing a
    whitespace token.
    """

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        if getattr(tokenizer, "is_fast", None) is False:
            raise TypeError(
                "ANKH requires a fast tokenizer so its residue-aware pre-tokenizer "
                "can be configured."
            )
        # Lightweight tokenizer doubles used by offline CPU contracts need not
        # expose a Rust tokenizer backend.
        return tokenizer
    backend.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement="\u2581",
        prepend_scheme="never",
        split=True,
    )
    return tokenizer


def normalize_ankh_sequence(sequence: str) -> str:
    """Return one ANKH protein sequence in canonical raw-residue form."""

    if not isinstance(sequence, str):
        raise TypeError("ANKH protein sequences must be strings.")
    normalized = "".join(sequence.split())
    if not normalized:
        raise ValueError("ANKH protein sequences must not be empty or whitespace-only.")
    return normalized


def normalize_ankh_decoder_prompt(prompt: str) -> str:
    """Return a decoder prompt with residues and sentinels directly adjacent."""

    if not isinstance(prompt, str):
        raise TypeError("ANKH decoder prompts must be strings.")
    normalized = "".join(prompt.split())
    if not normalized:
        raise ValueError("ANKH decoder prompts must not be empty or whitespace-only.")
    return normalized


def _normalize_ankh_text_batch(
    values: str | Sequence[str],
    *,
    field: str,
) -> str | list[str]:
    normalizer = normalize_ankh_sequence if field == "sequence" else normalize_ankh_decoder_prompt
    if isinstance(values, str):
        return normalizer(values)
    if isinstance(values, bytes) or not isinstance(values, Sequence):
        raise TypeError(f"ANKH {field} inputs must be a string or a sequence of strings.")
    return [normalizer(value) for value in values]


def tokenize_ankh_sequences(
    tokenizer: Any,
    sequences: str | Sequence[str],
    **tokenizer_kwargs: Any,
) -> Any:
    """Tokenize raw ANKH protein sequences with one model-wide contract."""

    configured = configure_ankh_tokenizer(tokenizer)
    normalized = _normalize_ankh_text_batch(sequences, field="sequence")
    return configured(normalized, **tokenizer_kwargs)


def tokenize_ankh_decoder_prompts(
    tokenizer: Any,
    prompts: str | Sequence[str],
    **tokenizer_kwargs: Any,
) -> Any:
    """Tokenize explicit ANKH decoder prompts without whitespace ``<unk>`` tokens."""

    configured = configure_ankh_tokenizer(tokenizer)
    normalized = _normalize_ankh_text_batch(prompts, field="decoder prompt")
    return configured(normalized, **tokenizer_kwargs)


def _load_ankh_tokenizer(
    config: FastAnkhConfig,
    load_context: Mapping[str, Any] | None = None,
):
    """Load the tokenizer from the same immutable checkpoint as the model."""
    name_or_path = str(getattr(config, "_name_or_path", "")).strip()
    if not name_or_path:
        raise RuntimeError(
            "ANKH tokenizer loading requires a model loaded with from_pretrained "
            "so checkpoint provenance is available."
        )
    tokenizer_kwargs = {
        key: value
        for key, value in dict(load_context or {}).items()
        if key in _TOKENIZER_LOAD_CONTEXT_KEYS and value is not None
    }
    # The resolved commit is authoritative. In particular, do not reload the
    # tokenizer from a moving branch when Transformers resolved model weights to
    # an immutable Hub commit.
    revision = getattr(config, "_commit_hash", None)
    if revision:
        tokenizer_kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **tokenizer_kwargs)
    return configure_ankh_tokenizer(tokenizer)


class _AnkhTokenizerLoadMixin:
    """Keep tokenizer loading scoped to the model instance and weight request."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        load_context = {key: kwargs[key] for key in _TOKENIZER_LOAD_CONTEXT_KEYS if key in kwargs}
        loaded = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        model.__dict__["_fastplms_tokenizer_load_context"] = load_context
        model.__dict__["_fastplms_tokenizer"] = None
        return loaded

    @property
    def tokenizer(self):
        tokenizer = self.__dict__.get("_fastplms_tokenizer")
        if tokenizer is None:
            tokenizer = _load_ankh_tokenizer(
                self.config,
                self.__dict__.get("_fastplms_tokenizer_load_context"),
            )
            self.__dict__["_fastplms_tokenizer"] = tokenizer
        return tokenizer

    @tokenizer.setter
    def tokenizer(self, value) -> None:
        self.__dict__["_fastplms_tokenizer"] = configure_ankh_tokenizer(value)

    def _tokenize_sequence_batch(
        self,
        sequences: Sequence[str],
        *,
        tokenizer: Any | None = None,
        **tokenizer_kwargs: Any,
    ) -> Any:
        resolved_tokenizer = tokenizer if tokenizer is not None else self.tokenizer
        return tokenize_ankh_sequences(
            resolved_tokenizer,
            sequences,
            **tokenizer_kwargs,
        )

    def embed_dataset(self, inputs: Any, **kwargs: Any) -> Any:
        explicit_tokenizer = kwargs.get("tokenizer")
        if explicit_tokenizer is not None:
            kwargs["tokenizer"] = configure_ankh_tokenizer(explicit_tokenizer)
        decoder_inputs = kwargs.get("decoder_inputs")
        if (
            decoder_inputs is not None
            and not isinstance(decoder_inputs, (str, bytes))
            and isinstance(decoder_inputs, Sequence)
        ):
            kwargs["decoder_inputs"] = [
                normalize_ankh_decoder_prompt(value) for value in decoder_inputs
            ]
        return EmbeddingMixin.embed_dataset(self, inputs, **kwargs)


def _validate_hidden_state_source(hidden_state_source: str) -> str:
    if hidden_state_source not in {"encoder", "decoder"}:
        raise ValueError(
            "hidden_state_source must be either 'encoder' or 'decoder'; "
            f"received {hidden_state_source!r}."
        )
    return hidden_state_source


def _require_encoder_embedding_source(
    hidden_state_source: str,
    *,
    decoder_inputs: Sequence[str] | None = None,
    decoder_input_ids: torch.Tensor | None = None,
    decoder_attention_mask: torch.Tensor | None = None,
) -> None:
    source = _validate_hidden_state_source(hidden_state_source)
    if source == "decoder":
        raise ValueError(
            "Decoder hidden states require FastAnkhForConditionalGeneration loaded "
            "through AutoModelForSeq2SeqLM; the encoder-only ANKH view does not "
            "allocate a decoder."
        )
    decoder_values = (decoder_inputs, decoder_input_ids, decoder_attention_mask)
    if any(value is not None for value in decoder_values):
        raise ValueError(
            "decoder_inputs, decoder_input_ids, and decoder_attention_mask are only "
            "valid when hidden_state_source='decoder'."
        )


def _biological_token_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: Any,
) -> torch.Tensor:
    mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
    special_ids = tuple(int(value) for value in getattr(tokenizer, "all_special_ids", ()))
    if special_ids:
        mask = mask & ~torch.isin(
            input_ids,
            torch.tensor(special_ids, device=input_ids.device, dtype=input_ids.dtype),
        )
    return mask


# ---------------------------------------------------------------------------
# Submodules
# ---------------------------------------------------------------------------


class AnkhRMSNorm(nn.Module):
    """T5-style RMS layer norm: scales without mean subtraction or bias."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (..., d)
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)  # (..., 1)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(self.weight.dtype)


def _gelu_new(x: torch.Tensor) -> torch.Tensor:
    return (
        0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    )


class AnkhGatedFFN(nn.Module):
    """T5-style gated feed-forward: activation(wi_0(x)) * wi_1(x) -> wo."""

    def __init__(self, config: FastAnkhConfig) -> None:
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.act = F.silu if config.dense_act_fn == "silu" else _gelu_new
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (b, l, d)
        hidden_states = self.act(self.wi_0(hidden_states)) * self.wi_1(hidden_states)
        return self.wo(self.dropout(hidden_states))


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class AnkhSelfAttention(nn.Module):
    """T5-style self-attention with relative position bias and multi-backend dispatch.

    Only layer 0 has ``has_relative_attention_bias=True`` and owns the
    ``nn.Embedding`` that produces the position bias.  All other layers
    receive the precomputed bias through the forward call.
    """

    def __init__(
        self,
        config: FastAnkhConfig,
        has_relative_attention_bias: bool = False,
    ) -> None:
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
        self.dropout_prob = float(config.dropout_rate)

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
        values = self.relative_attention_bias(buckets)  # (q, k, h)
        return values.permute(2, 0, 1).unsqueeze(0)  # (1, h, q, k)

    # ---- Forward ----

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_4d: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
        output_attentions: bool = False,
        effective_backend: AttentionBackend | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Returns (attn_output, attn_weights_or_none, position_bias)."""
        # hidden_states: (b, l, d)
        batch_size, seq_length = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_length, self.num_heads, self.d_kv)

        query_heads = self.q(hidden_states).view(hidden_shape).transpose(1, 2)  # (b, h, l, d_h)
        key_heads = self.k(hidden_states).view(hidden_shape).transpose(1, 2)  # (b, h, l, d_h)
        value_heads = self.v(hidden_states).view(hidden_shape).transpose(1, 2)  # (b, h, l, d_h)

        # The first layer computes the bias once; later layers reuse it.
        if position_bias is None and self.has_relative_attention_bias:
            position_bias = self.compute_bias(seq_length, seq_length, hidden_states.device)
            # Fold padding mask into position bias so layers don't need separate mask.
            if attention_mask_4d is not None:
                position_bias = position_bias + bool_to_additive_mask(
                    attention_mask_4d, position_bias.dtype
                )

        if effective_backend is None:
            effective_backend = resolve_attention_backend_for_call(
                self.attn_backend,
                output_attentions=output_attentions,
            )
        if output_attentions:
            attn_output, attn_weights = self._manual_attn(
                query_heads, key_heads, value_heads, position_bias
            )
            return self.o(attn_output), attn_weights, position_bias

        if effective_backend == AttentionBackend.EAGER:
            attn_output, _ = self._manual_attn(query_heads, key_heads, value_heads, position_bias)
        elif effective_backend == AttentionBackend.SDPA:
            attn_output = self._sdpa_attn(query_heads, key_heads, value_heads, position_bias)
        else:
            raise AssertionError(f"Unsupported backend for ANKH: {effective_backend}")

        return self.o(attn_output), None, position_bias

    def _sdpa_attn(
        self,
        query_heads: torch.Tensor,
        key_heads: torch.Tensor,
        value_heads: torch.Tensor,
        position_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        # position_bias: (1, h, l, l), including padding
        # Never mutate torch.backends.cuda process-global reduction policy from
        # a model forward. Concurrent model requests must not change each
        # other's numerical behavior or restore a stale process setting.
        context_heads = F.scaled_dot_product_attention(
            query_heads,
            key_heads,
            value_heads,
            attn_mask=position_bias,
            dropout_p=self.dropout_prob if self.training else 0.0,
            scale=self.scale,
        )  # (b, h, l, d_h)
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
        # query_heads, key_heads, value_heads: (b, h, l, d_h)
        attn_weights = (
            torch.matmul(query_heads, key_heads.transpose(-1, -2)) * self.scale
        )  # (b, h, l, l)
        if position_bias is not None:
            attn_weights = attn_weights + position_bias
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        if self.dropout_prob > 0 and self.training:
            attn_weights = F.dropout(
                attn_weights,
                p=self.dropout_prob,
                training=self.training,
            )
        context_heads = torch.matmul(attn_weights, value_heads)  # (b, h, l, d_h)
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

    def __init__(
        self,
        config: FastAnkhConfig,
        has_relative_attention_bias: bool = False,
    ) -> None:
        super().__init__()
        self.SelfAttention = AnkhSelfAttention(config, has_relative_attention_bias)
        self.layer_norm = AnkhRMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_4d: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
        output_attentions: bool = False,
        effective_backend: AttentionBackend | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        normed = self.layer_norm(hidden_states)
        attn_output, attn_weights, position_bias = self.SelfAttention(
            normed,
            attention_mask_4d=attention_mask_4d,
            position_bias=position_bias,
            output_attentions=output_attentions,
            effective_backend=effective_backend,
        )
        hidden_states = hidden_states + self.dropout(attn_output)
        return hidden_states, attn_weights, position_bias


class AnkhFFLayer(nn.Module):
    """Wraps AnkhGatedFFN + layer_norm to match T5Block.layer[1] key naming."""

    def __init__(self, config: FastAnkhConfig) -> None:
        super().__init__()
        self.DenseReluDense = AnkhGatedFFN(config)
        self.layer_norm = AnkhRMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.dropout(self.DenseReluDense(normed))
        return hidden_states


class AnkhBlock(nn.Module):
    """Single transformer block with T5-compatible .layer ModuleList naming."""

    def __init__(
        self,
        config: FastAnkhConfig,
        has_relative_attention_bias: bool = False,
    ) -> None:
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
        effective_backend: AttentionBackend | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        hidden_states, attn_weights, position_bias = self.layer[0](
            hidden_states,
            attention_mask_4d=attention_mask_4d,
            position_bias=position_bias,
            output_attentions=output_attentions,
            effective_backend=effective_backend,
        )
        hidden_states = self.layer[1](hidden_states)
        return hidden_states, attn_weights, position_bias


# ---------------------------------------------------------------------------
# PreTrainedModel base
# ---------------------------------------------------------------------------


class AnkhPreTrainedModel(
    _AnkhTokenizerLoadMixin,
    FastPLMsAttentionMixin,
    PreTrainedModel,
):
    config_class = FastAnkhConfig
    base_model_prefix = "encoder"
    supports_gradient_checkpointing = True
    _no_split_modules: ClassVar[list[str]] = ["AnkhBlock"]
    _supports_flash_attn_2 = False
    _supports_flash_attn_3 = False
    _supports_flex_attn = False
    _fastplms_attention_implementations = ("eager", "sdpa")
    embedding_unsupported_pooling = ("cls",)

    def __init__(self, config: FastAnkhConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.__dict__["_fastplms_tokenizer"] = None
        self.__dict__["_fastplms_tokenizer_load_context"] = {}

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

    def _embedding_metadata(self, **context: Any) -> Mapping[str, Any]:
        source = _validate_hidden_state_source(context.get("hidden_state_source", "encoder"))
        if source != "encoder":
            raise ValueError(
                "Decoder hidden states require FastAnkhForConditionalGeneration loaded "
                "through AutoModelForSeq2SeqLM."
            )
        return {
            "architecture": "ANKH-T5",
            "hidden_state_stack": "encoder",
            "layer_order": "embedding-plus-transformer-blocks",
        }

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

    def __init__(self, config: FastAnkhConfig, **kwargs) -> None:
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
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gradient_checkpointing = False
        self.post_init()

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
        hidden_state_source: str = "encoder",
        decoder_inputs: Sequence[str] | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _require_encoder_embedding_source(
            hidden_state_source,
            decoder_inputs=decoder_inputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )
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
        # T5Stack applies this module both to the input embeddings and after
        # final normalization. Keeping those as separate calls preserves the
        # official training-time stochastic path without affecting eval mode.
        hidden_states = self.dropout(hidden_states)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        batch_size, seq_len = hidden_states.shape[:2]
        effective_backend = resolve_attention_backend_for_call(
            self.attention_backend,
            output_attentions=output_attentions,
        )
        _, attention_mask_4d, _ = get_attention_mask(
            effective_backend=effective_backend,
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
                    effective_backend,
                )
            else:
                hidden_states, attn_weights, position_bias = layer_module(
                    hidden_states,
                    attention_mask_4d=attention_mask_4d,
                    position_bias=position_bias,
                    output_attentions=output_attentions,
                    effective_backend=effective_backend,
                )

            if all_attentions is not None:
                all_attentions = (*all_attentions, attn_weights)

        hidden_states = self.dropout(self.final_layer_norm(hidden_states))

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
        return_dict: bool | None = None,
    ) -> AnkhEncoderOutput | tuple[torch.Tensor, ...]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            hidden_states = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        outputs = self._run_encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states or False,
            output_attentions=output_attentions or False,
        )
        return outputs if return_dict else outputs.to_tuple()


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------


class FastAnkhModel(AnkhPreTrainedModel, EmbeddingMixin):
    """ANKH encoder model for embedding extraction."""

    _tied_weights_keys: ClassVar[dict[str, str]] = {"encoder.embed_tokens.weight": "shared.weight"}
    # The published ANKH checkpoint is the complete official T5 state. AutoModel
    # intentionally exposes only its encoder view without allocating a decoder.
    _keys_to_ignore_on_load_unexpected: ClassVar[list[str]] = [
        r"^decoder\.",
        r"^lm_head\.",
    ]

    def __init__(self, config: FastAnkhConfig, **kwargs) -> None:
        AnkhPreTrainedModel.__init__(self, config, **kwargs)
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = FAST_ANKH_ENCODER(config)
        self.encoder.embed_tokens = self.shared
        self.post_init()

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
        **embedding_kwargs,
    ) -> torch.Tensor:
        return self.encoder._embed(
            input_ids,
            attention_mask,
            hidden_state_index=hidden_state_index,
            store_all_hidden_states=store_all_hidden_states,
            **embedding_kwargs,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
    ) -> AnkhEncoderOutput | tuple[torch.Tensor, ...]:
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
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

    _tied_weights_keys: ClassVar[dict[str, str]] = {"encoder.embed_tokens.weight": "shared.weight"}
    _keys_to_ignore_on_load_unexpected: ClassVar[list[str]] = [r"^decoder\."]

    def __init__(self, config: FastAnkhConfig, **kwargs) -> None:
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
        **embedding_kwargs,
    ) -> torch.Tensor:
        return self.encoder._embed(
            input_ids,
            attention_mask,
            hidden_state_index=hidden_state_index,
            store_all_hidden_states=store_all_hidden_states,
            **embedding_kwargs,
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
        if seq is None:
            raise ValueError("Pass either seq or input_ids for ANKH TTT.")
        sequences = [seq] if isinstance(seq, str) else seq
        tokenized = tokenize_ankh_sequences(
            self.tokenizer,
            sequences,
            return_tensors="pt",
            padding=True,
        )
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
        return_dict: bool | None = None,
    ) -> MaskedLMOutput | tuple[torch.Tensor, ...]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )
        sequence_output = outputs.last_hidden_state
        logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits, *outputs.to_tuple()[1:])
            return (loss, *output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FastAnkhForConditionalGeneration(
    _AnkhTokenizerLoadMixin,
    T5ForConditionalGeneration,
    EmbeddingMixin,
):
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

    def __init__(self, config: FastAnkhConfig, **kwargs) -> None:
        requested_backend = getattr(config, "_attn_implementation", None) or config.attn_backend
        if requested_backend not in (None, "eager"):
            raise ValueError(
                "ANKH sequence-to-sequence checkpoints support only eager attention; "
                f"received {requested_backend!r}. Use FastAnkhModel for optimized "
                "encoder embeddings."
            )
        set_config_attn_implementation(config, "eager")
        super().__init__(config, **kwargs)
        self.__dict__["_fastplms_tokenizer"] = None
        self.__dict__["_fastplms_tokenizer_load_context"] = {}

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        encoder_outputs: Any | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        decoder_inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> Any:
        """Run official T5 seq2seq behavior with a fail-closed public signature.

        Transformers' T5 forward currently accepts and silently ignores arbitrary
        keyword arguments. FastPLMs keeps the supported T5 arguments explicit so a
        misspelled generation, cache, or conditioning argument cannot appear to
        have taken effect.
        """

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def _prepare_decoder_embedding_inputs(
        self,
        *,
        batch_size: int,
        decoder_inputs: Sequence[str] | None,
        decoder_input_ids: torch.Tensor | None,
        decoder_attention_mask: torch.Tensor | None,
        tokenizer: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        if (decoder_inputs is None) == (decoder_input_ids is None):
            raise ValueError(
                "hidden_state_source='decoder' requires exactly one of "
                "decoder_inputs or decoder_input_ids. Decoder inputs are task-specific "
                "and FastPLMs will not synthesize shifted encoder tokens."
            )
        resolved_tokenizer = configure_ankh_tokenizer(
            tokenizer if tokenizer is not None else self.tokenizer
        )
        if decoder_inputs is not None:
            if decoder_attention_mask is not None:
                raise ValueError(
                    "decoder_attention_mask may only accompany decoder_input_ids; "
                    "decoder_inputs are tokenized with their own attention mask."
                )
            values = [decoder_inputs] if isinstance(decoder_inputs, str) else list(decoder_inputs)
            if len(values) != batch_size:
                raise ValueError(
                    "decoder_inputs must align one-to-one with encoder inputs; "
                    f"expected {batch_size}, received {len(values)}."
                )
            encoded = tokenize_ankh_decoder_prompts(
                resolved_tokenizer,
                values,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            decoder_input_ids = encoded["input_ids"]
            decoder_attention_mask = encoded.get("attention_mask")
        if decoder_input_ids is None:
            raise RuntimeError(
                "Decoder input resolution completed without decoder_input_ids."
            )
        if decoder_input_ids.ndim != 2 or decoder_input_ids.shape[0] != batch_size:
            raise ValueError(
                "decoder_input_ids must have shape (batch, decoder_sequence_length); "
                f"expected batch {batch_size}, received {tuple(decoder_input_ids.shape)}."
            )
        if decoder_attention_mask is None:
            pad_token_id = self.config.pad_token_id
            decoder_attention_mask = (
                torch.ones_like(decoder_input_ids, dtype=torch.bool)
                if pad_token_id is None
                else decoder_input_ids.ne(pad_token_id)
            )
            decoder_start_token_id = self.config.decoder_start_token_id
            if (
                decoder_input_ids.shape[1] > 0
                and decoder_start_token_id is not None
                and decoder_start_token_id == pad_token_id
            ):
                decoder_attention_mask[:, 0] |= decoder_input_ids[:, 0].eq(decoder_start_token_id)
        if tuple(decoder_attention_mask.shape) != tuple(decoder_input_ids.shape):
            raise ValueError(
                "decoder_attention_mask must have the same shape as decoder_input_ids; "
                f"received {tuple(decoder_attention_mask.shape)} and "
                f"{tuple(decoder_input_ids.shape)}."
            )
        device = self.shared.weight.device
        return (
            decoder_input_ids.to(device=device),
            decoder_attention_mask.to(device=device),
            resolved_tokenizer,
        )

    def _extract_embedding_stack(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        hidden_state_source: str,
        hidden_state_index: int,
        store_all_hidden_states: bool,
        decoder_inputs: Sequence[str] | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        tokenizer: Any | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...] | None]:
        source = _validate_hidden_state_source(hidden_state_source)
        device = self.shared.weight.device
        input_ids = input_ids.to(device=device)
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        attention_mask = attention_mask.to(device=device)
        need_hidden_states = store_all_hidden_states or hidden_state_index != -1
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=need_hidden_states,
            output_attentions=output_attentions and source == "encoder",
            return_dict=True,
        )
        if source == "encoder":
            if any(
                value is not None
                for value in (decoder_inputs, decoder_input_ids, decoder_attention_mask)
            ):
                raise ValueError(
                    "Decoder inputs are only valid when hidden_state_source='decoder'."
                )
            X = select_hidden_state_embeddings(
                encoder_outputs.last_hidden_state,
                encoder_outputs.hidden_states,
                hidden_state_index=hidden_state_index,
                store_all_hidden_states=store_all_hidden_states,
            )
            return X, input_ids, attention_mask, encoder_outputs.attentions

        decoder_input_ids, decoder_attention_mask, _ = self._prepare_decoder_embedding_inputs(
            batch_size=input_ids.shape[0],
            decoder_inputs=decoder_inputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            tokenizer=tokenizer,
        )
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=need_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )
        X = select_hidden_state_embeddings(
            decoder_outputs.last_hidden_state,
            decoder_outputs.hidden_states,
            hidden_state_index=hidden_state_index,
            store_all_hidden_states=store_all_hidden_states,
        )
        return X, decoder_input_ids, decoder_attention_mask, decoder_outputs.attentions

    def _embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        hidden_state_index: int = -1,
        store_all_hidden_states: bool = False,
        hidden_state_source: str = "encoder",
        decoder_inputs: Sequence[str] | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        tokenizer: Any | None = None,
    ) -> torch.Tensor:
        X, _, _, _ = self._extract_embedding_stack(
            input_ids,
            attention_mask,
            hidden_state_source=hidden_state_source,
            hidden_state_index=hidden_state_index,
            store_all_hidden_states=store_all_hidden_states,
            decoder_inputs=decoder_inputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            tokenizer=tokenizer,
        )
        return X

    def _embedding_batch(
        self,
        sequences: Sequence[str],
        *,
        tokenizer: Any | None = None,
        max_length: int | None = None,
        truncate: bool = True,
        need_attentions: bool = False,
        hidden_state_source: str = "encoder",
        hidden_state_index: int = -1,
        store_all_hidden_states: bool = False,
        decoder_inputs: Sequence[str] | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
    ) -> EmbeddingBatch:
        del max_length, truncate  # The shared runner already crops biological residues.
        resolved_tokenizer = configure_ankh_tokenizer(
            tokenizer if tokenizer is not None else self.tokenizer
        )
        encoded = tokenize_ankh_sequences(
            resolved_tokenizer,
            list(sequences),
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        device = self.shared.weight.device
        input_ids = encoded["input_ids"].to(device=device)
        attention_mask = encoded.get("attention_mask", input_ids.new_ones(input_ids.shape)).to(
            device=device
        )
        X, selected_ids, selected_attention_mask, attentions = self._extract_embedding_stack(
            input_ids,
            attention_mask,
            hidden_state_source=hidden_state_source,
            hidden_state_index=hidden_state_index,
            store_all_hidden_states=store_all_hidden_states,
            decoder_inputs=decoder_inputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            tokenizer=resolved_tokenizer,
            output_attentions=need_attentions,
        )
        residue_mask = _biological_token_mask(
            selected_ids,
            selected_attention_mask,
            resolved_tokenizer,
        )
        return EmbeddingBatch(X=X, residue_mask=residue_mask, attentions=attentions)

    def _embedding_metadata(self, **context: Any) -> Mapping[str, Any]:
        source = _validate_hidden_state_source(context.get("hidden_state_source", "encoder"))
        return {
            "architecture": "ANKH-T5",
            "hidden_state_stack": source,
            "layer_order": "embedding-plus-transformer-blocks",
            "decoder_inputs": "explicit-task-inputs-required" if source == "decoder" else None,
            "decoder_residue_mask": (
                "attention-mask-minus-tokenizer-specials" if source == "decoder" else None
            ),
        }


class FastAnkhForSequenceClassification(AnkhPreTrainedModel, EmbeddingMixin):
    _tied_weights_keys: ClassVar[dict[str, str]] = {"encoder.embed_tokens.weight": "shared.weight"}
    _keys_to_ignore_on_load_unexpected: ClassVar[list[str]] = [
        r"^decoder\.",
        r"^lm_head\.",
    ]

    def __init__(self, config: FastAnkhConfig, **kwargs) -> None:
        AnkhPreTrainedModel.__init__(self, config, **kwargs)
        self.num_labels = config.num_labels
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = FAST_ANKH_ENCODER(config)
        self.encoder.embed_tokens = self.shared
        self.classifier = nn.Linear(config.d_model, config.num_labels)
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.post_init()

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
        **embedding_kwargs,
    ) -> torch.Tensor:
        return self.encoder._embed(
            input_ids,
            attention_mask,
            hidden_state_index=hidden_state_index,
            store_all_hidden_states=store_all_hidden_states,
            **embedding_kwargs,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
    ) -> SequenceClassifierOutput | tuple[torch.Tensor, ...]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
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

        if not return_dict:
            output = (logits, *outputs.to_tuple()[1:])
            return (loss, *output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FastAnkhForTokenClassification(AnkhPreTrainedModel, EmbeddingMixin):
    _tied_weights_keys: ClassVar[dict[str, str]] = {"encoder.embed_tokens.weight": "shared.weight"}
    _keys_to_ignore_on_load_unexpected: ClassVar[list[str]] = [
        r"^decoder\.",
        r"^lm_head\.",
    ]

    def __init__(self, config: FastAnkhConfig, **kwargs) -> None:
        AnkhPreTrainedModel.__init__(self, config, **kwargs)
        self.num_labels = config.num_labels
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = FAST_ANKH_ENCODER(config)
        self.encoder.embed_tokens = self.shared
        self.classifier = nn.Linear(config.d_model, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        self.post_init()

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
        **embedding_kwargs,
    ) -> torch.Tensor:
        return self.encoder._embed(
            input_ids,
            attention_mask,
            hidden_state_index=hidden_state_index,
            store_all_hidden_states=store_all_hidden_states,
            **embedding_kwargs,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
    ) -> TokenClassifierOutput | tuple[torch.Tensor, ...]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits, *outputs.to_tuple()[1:])
            return (loss, *output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
