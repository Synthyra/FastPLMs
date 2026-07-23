from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from transformers import EsmTokenizer, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import (
    MaskedLMOutput,
    ModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.models.esm.modeling_esm import (
    EsmClassificationHead,
    EsmContactPredictionHead,
    EsmEmbeddings,
    EsmIntermediate,
    EsmLMHead,
    EsmOutput,
    EsmPooler,
    EsmSelfOutput,
)

from fastplms.models._esm_rotary import RotaryEmbedding

try:
    from fastplms.attention import (
        AttentionBackend,
        BlockMask,
        FastPLMsAttentionMixin,
        _get_flex_attention_fn,
        flex_attention,
        get_attention_mask,
        kernels_flash_attention_func,
        resolve_attention_backend,
        resolve_attention_backend_for_call,
    )
    from fastplms.embeddings import EmbeddingMixin, select_hidden_state_embeddings
    from fastplms.models.ttt import FastPLMTestTimeTrainingMixin
except ModuleNotFoundError as error:
    _COMPOSITE_REQUIRED_NAMES = (
        "AttentionBackend",
        "BlockMask",
        "EmbeddingMixin",
        "FastPLMsAttentionMixin",
        "FastPLMTestTimeTrainingMixin",
        "_get_flex_attention_fn",
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


@dataclass
class FastEsmEncoderOutput(ModelOutput):
    last_hidden_state: torch.Tensor | None = None
    pooler_output: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None
    s_max: tuple[list[torch.Tensor], ...] | None = None


@dataclass
class EsmMaskedLMOutput(MaskedLMOutput):
    """Masked-LM output with FastPLMs diagnostics after the HF fields."""

    s_max: tuple[list[torch.Tensor], ...] | None = None
    last_hidden_state: torch.Tensor | None = None


@dataclass
class EsmSequenceClassifierOutput(SequenceClassifierOutput):
    """Sequence-classification output with optional attention diagnostics."""

    s_max: tuple[list[torch.Tensor], ...] | None = None


@dataclass
class EsmTokenClassifierOutput(TokenClassifierOutput):
    """Token-classification output with optional attention diagnostics."""

    s_max: tuple[list[torch.Tensor], ...] | None = None


class FastEsmConfig(PretrainedConfig):
    model_type = "fast_esm"

    def __init__(
        self,
        vocab_size: int | None = None,
        bos_token_id: int | None = 0,
        eos_token_id: int | None = 2,
        mask_token_id: int | None = None,
        pad_token_id: int | None = None,
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
        emb_layer_norm_before: bool | None = None,
        token_dropout: bool = True,
        add_pooling_layer: bool = False,
        attn_backend: str | None = None,
        **kwargs,
    ):
        bos_token_id = 0 if bos_token_id is None else bos_token_id
        eos_token_id = 2 if eos_token_id is None else eos_token_id
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
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
        self.add_pooling_layer = add_pooling_layer
        self.attn_backend = attn_backend

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete configuration to a Python dictionary."""
        return super().to_dict()


_TOKENIZER_LOAD_CONTEXT_KEYS = (
    "cache_dir",
    "force_download",
    "local_files_only",
    "proxies",
    "revision",
    "subfolder",
    "token",
    "trust_remote_code",
)


class FastEsmTokenizer(EsmTokenizer):
    """Retain fair-esm's strict handling of residues outside its alphabet."""

    def __call__(
        self,
        text: Any = None,
        *args: Any,
        truncation: Any = None,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> Any:
        if truncation and max_length is not None:
            residue_limit = max(1, max_length - 2)
            if isinstance(text, str):
                text = text[:residue_limit]
            elif isinstance(text, (list, tuple)) and all(
                isinstance(sequence, str) for sequence in text
            ):
                text = [sequence[:residue_limit] for sequence in text]
        return super().__call__(
            text,
            *args,
            truncation=truncation,
            max_length=max_length,
            **kwargs,
        )

    def _convert_token_to_id(self, token: str) -> int:
        try:
            return self._token_to_id[token]
        except KeyError:
            raise KeyError(token) from None


class EsmSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type: str | None = None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of "
                f"attention heads ({config.num_attention_heads})"
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
        attention_mask_2d: torch.Tensor | None = None,
        attention_mask_4d: torch.Tensor | None = None,
        flex_block_mask: BlockMask | None = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        hidden_shape = (batch_size, seq_length, -1, self.attention_head_size)
        query_heads = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        key_heads = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        value_heads = self.value(hidden_states).view(hidden_shape).transpose(1, 2)

        query_heads = query_heads * self.scale

        if self.position_embedding_type == "rotary":
            query_heads, key_heads = self.rotary_embeddings(query_heads, key_heads)

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
        return attn_output, attn_weights, s_max

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

        if (
            self.training
            and self.dropout_prob > 0
            and (self.attn_backend.is_flash or self.attn_backend == AttentionBackend.FLEX_ATTENTION)
        ):
            raise RuntimeError(
                f"ESM2 {self.attn_backend.value} attention is inference-only when attention "
                "dropout is nonzero. Use eager or SDPA for this training configuration."
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
        q_norm = torch.linalg.vector_norm(query_heads, dim=-1)
        k_norm = torch.linalg.vector_norm(key_heads, dim=-1)
        s_max_bound = (q_norm.max(dim=-1).values * k_norm.max(dim=-1).values).max(dim=0).values
        return [s_max_bound[h] for h in range(self.num_attention_heads)]

    def _manual_attn(
        self,
        query_heads: torch.Tensor,
        key_heads: torch.Tensor,
        value_heads: torch.Tensor,
        attention_mask_4d: torch.Tensor | None = None,
        output_s_max: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor] | None]:
        attn_weights = torch.matmul(query_heads, key_heads.transpose(-1, -2))
        if attention_mask_4d is not None:
            attn_weights = attn_weights.masked_fill(attention_mask_4d.logical_not(), float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.dropout_prob > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout_prob, training=self.training)
        context_heads = torch.matmul(attn_weights, value_heads)
        attn_output = rearrange(context_heads, "b h s d -> b s (h d)")
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
        # Q has been pre-scaled by self.scale = 1/sqrt(head_dim) in forward().
        # Pass softmax_scale=1.0 to prevent the kernel from applying its default
        # 1/sqrt(head_dim) scale on top (which would yield effective scale
        # 1/head_dim and break parity vs sdpa).
        attn_output = kernels_flash_attention_func(
            query_states=query_tokens,
            key_states=key_tokens,
            value_states=value_tokens,
            attention_mask_2d=attention_mask_2d,
            causal=False,
            softmax_scale=1.0,
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
            query_heads, key_heads, value_heads, block_mask=flex_block_mask, scale=1.0
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
            dropout_p=self.dropout_prob if self.training else 0.0,
            scale=1.0,
        )
        return rearrange(context_heads, "b h s d -> b s (h d)"), None


class EsmAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = EsmSelfAttention(config)
        self.output = EsmSelfOutput(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_2d: torch.Tensor | None = None,
        attention_mask_4d: torch.Tensor | None = None,
        flex_block_mask: BlockMask | None = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
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
        attention_mask_2d: torch.Tensor | None = None,
        attention_mask_4d: torch.Tensor | None = None,
        flex_block_mask: BlockMask | None = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
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
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        output_s_max: bool = False,
    ) -> FastEsmEncoderOutput:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        full_s_max = () if output_s_max else None

        effective_backend = resolve_attention_backend_for_call(
            self.attention_backend,
            output_attentions=output_attentions,
        )
        attention_mask_2d, attention_mask_4d, flex_block_mask = get_attention_mask(
            effective_backend=effective_backend,
            batch_size=hidden_states.shape[0],
            seq_len=hidden_states.shape[1],
            device=hidden_states.device,
            attention_mask=attention_mask,
            dtype=hidden_states.dtype,
            mask_semantics="padding",
        )

        for layer_module in self.layer:
            if output_hidden_states:
                all_hidden_states = (*all_hidden_states, hidden_states)

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
                all_attentions = (*all_attentions, attn_weights)
            if full_s_max is not None:
                full_s_max = (*full_s_max, s_max)

        if self.emb_layer_norm_after:
            hidden_states = self.emb_layer_norm_after(hidden_states)

        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        return FastEsmEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            s_max=full_s_max,
        )


class FastEsmPreTrainedModel(FastPLMsAttentionMixin, PreTrainedModel):
    """Initialize weights and provide the shared pretrained-model interface."""

    config_class = FastEsmConfig
    # Every advertised task wrapper stores the shared encoder at ``self.esm``.
    # Transformers uses this name for ``base_model`` and for loading an
    # unprefixed base checkpoint into a prefixed task wrapper.
    base_model_prefix = "esm"
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

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        load_context = {key: kwargs[key] for key in _TOKENIZER_LOAD_CONTEXT_KEYS if key in kwargs}
        if "token" not in load_context and "use_auth_token" in kwargs:
            load_context["token"] = kwargs["use_auth_token"]
        load_context["source"] = pretrained_model_name_or_path

        loaded = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        model.__dict__["_fastplms_tokenizer_load_context"] = load_context
        model.__dict__["_fastplms_tokenizer"] = None
        return loaded

    @property
    def tokenizer(self):
        tokenizer = self.__dict__.get("_fastplms_tokenizer")
        if tokenizer is None:
            load_context = dict(self.__dict__.get("_fastplms_tokenizer_load_context") or {})
            source = load_context.pop("source", None)
            if source is None:
                source = str(getattr(self.config, "_name_or_path", "")).strip()
            if not source:
                raise RuntimeError(
                    "ESM2 tokenizer loading requires a model loaded with from_pretrained "
                    "so checkpoint provenance is available."
                )
            tokenizer_kwargs = {
                key: value
                for key, value in load_context.items()
                if key in _TOKENIZER_LOAD_CONTEXT_KEYS and value is not None
            }
            resolved_revision = getattr(self.config, "_commit_hash", None)
            if resolved_revision:
                tokenizer_kwargs["revision"] = resolved_revision
            tokenizer = FastEsmTokenizer.from_pretrained(source, **tokenizer_kwargs)
            if getattr(tokenizer, "bos_token_id", None) is None and hasattr(tokenizer, "cls_token"):
                tokenizer.bos_token = tokenizer.cls_token
            self.__dict__["_fastplms_tokenizer"] = tokenizer
        return tokenizer

    @tokenizer.setter
    def tokenizer(self, value) -> None:
        self.__dict__["_fastplms_tokenizer"] = value

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
        if backend not in self._fastplms_attention_implementations:
            raise ValueError(
                f"{type(self).__name__} does not support {backend!r}; expected one of "
                f"{self._fastplms_attention_implementations}."
            )
        self.config.attn_backend = backend
        resolved = resolve_attention_backend(backend)
        for module in self.modules():
            if isinstance(module, EsmEncoder):
                module.attention_backend = resolved
            elif isinstance(module, EsmSelfAttention):
                module.attn_backend = resolved


class FAST_ESM_ENCODER(FastEsmPreTrainedModel, EmbeddingMixin):
    def __init__(self, config, add_pooling_layer: bool | None = True, **kwargs):
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

    def _embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        hidden_state_index: int = -1,
        store_all_hidden_states: bool = False,
    ) -> torch.Tensor:
        token_embedding_output = self.embeddings(input_ids, attention_mask=attention_mask)
        output_hidden_states = store_all_hidden_states or hidden_state_index != -1
        encoder_outputs = self.encoder(
            token_embedding_output,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=False,
        )
        return select_hidden_state_embeddings(
            encoder_outputs.last_hidden_state,
            encoder_outputs.hidden_states,
            hidden_state_index=hidden_state_index,
            store_all_hidden_states=store_all_hidden_states,
        )

    def predict_contacts(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        attns = self(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        ).attentions
        attns = torch.stack(attns, dim=1)
        attns *= attention_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        attns *= attention_mask.unsqueeze(1).unsqueeze(2).unsqueeze(4)
        return self.contact_head(input_ids, attns)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_s_max: bool | None = False,
        return_dict: bool | None = None,
    ) -> FastEsmEncoderOutput | tuple[torch.Tensor, ...]:
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

        result = FastEsmEncoderOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            s_max=encoder_outputs.s_max,
        )
        return result if return_dict else result.to_tuple()


class FastEsmModel(FastEsmPreTrainedModel, EmbeddingMixin):
    def __init__(self, config, add_pooling_layer: bool | None = None, **kwargs):
        FastEsmPreTrainedModel.__init__(self, config, **kwargs)
        self.config = config
        self.esm = FAST_ESM_ENCODER(config)
        if add_pooling_layer is None:
            add_pooling_layer = config.add_pooling_layer
        config.add_pooling_layer = bool(add_pooling_layer)
        self.pooler = EsmPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self):
        return self.esm.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.esm.embeddings.word_embeddings = value

    def _embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        hidden_state_index: int = -1,
        store_all_hidden_states: bool = False,
    ) -> torch.Tensor:
        return self.esm._embed(
            input_ids,
            attention_mask,
            hidden_state_index=hidden_state_index,
            store_all_hidden_states=store_all_hidden_states,
        )

    def predict_contacts(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.esm.predict_contacts(input_ids, attention_mask=attention_mask)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_s_max: bool | None = False,
        return_dict: bool | None = None,
    ) -> FastEsmEncoderOutput | tuple[torch.Tensor, ...]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
            return_dict=True,
        )
        sequence_output = outputs.last_hidden_state
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        result = FastEsmEncoderOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            s_max=outputs.s_max,
        )
        return result if return_dict else result.to_tuple()


class FastEsmForMaskedLM(FastPLMTestTimeTrainingMixin, FastEsmPreTrainedModel, EmbeddingMixin):
    def __init__(self, config, **kwargs):
        FastEsmPreTrainedModel.__init__(self, config, **kwargs)
        self.esm = FAST_ESM_ENCODER(config, add_pooling_layer=False)
        self.lm_head = EsmLMHead(config)
        self.loss_fct = nn.CrossEntropyLoss()
        self.post_init()
        self.init_ttt({"lora_target_replace_module": "EsmAttention"})

    def get_input_embeddings(self):
        return self.esm.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.esm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        old_bias = self.lm_head.bias
        new_vocab_size = int(new_embeddings.out_features)
        if old_bias.shape[0] != new_vocab_size:
            resized_bias = old_bias.new_zeros(new_vocab_size)
            copy_length = min(old_bias.shape[0], new_vocab_size)
            with torch.no_grad():
                resized_bias[:copy_length].copy_(old_bias[:copy_length])
            self.lm_head.bias = nn.Parameter(resized_bias)
        # EsmLMHead.forward adds this standalone bias after the decoder. HF's
        # generic LM-head resizer may create a biased Linear, which would apply
        # the bias twice and introduce an undeclared shared tensor on save.
        new_embeddings.bias = None
        self.lm_head.decoder = new_embeddings

    def _embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        hidden_state_index: int = -1,
        store_all_hidden_states: bool = False,
    ) -> torch.Tensor:
        return self.esm._embed(
            input_ids,
            attention_mask,
            hidden_state_index=hidden_state_index,
            store_all_hidden_states=store_all_hidden_states,
        )

    def predict_contacts(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.esm.predict_contacts(input_ids, attention_mask=attention_mask)

    def _ttt_get_trainable_modules(self) -> list[nn.Module]:
        return [self.esm]

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_s_max: bool | None = False,
        return_dict: bool | None = None,
    ) -> EsmMaskedLMOutput | tuple[torch.Tensor, ...]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
            return_dict=True,
        )
        sequence_output = outputs.last_hidden_state
        prediction_scores = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(prediction_scores.device)
            loss = self.loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        result = EsmMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            s_max=outputs.s_max,
            last_hidden_state=sequence_output,
        )
        return result if return_dict else result.to_tuple()


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

    def set_input_embeddings(self, value):
        self.esm.set_input_embeddings(value)

    def _embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        hidden_state_index: int = -1,
        store_all_hidden_states: bool = False,
    ) -> torch.Tensor:
        return self.esm._embed(
            input_ids,
            attention_mask,
            hidden_state_index=hidden_state_index,
            store_all_hidden_states=store_all_hidden_states,
        )

    def predict_contacts(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.esm.predict_contacts(input_ids, attention_mask=attention_mask)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_s_max: bool | None = False,
        return_dict: bool | None = None,
    ) -> EsmSequenceClassifierOutput | tuple[torch.Tensor, ...]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_s_max=output_s_max,
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
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
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

        result = EsmSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            s_max=outputs.s_max,
        )
        return result if return_dict else result.to_tuple()


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

    def set_input_embeddings(self, value):
        self.esm.set_input_embeddings(value)

    def _embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        hidden_state_index: int = -1,
        store_all_hidden_states: bool = False,
    ) -> torch.Tensor:
        return self.esm._embed(
            input_ids,
            attention_mask,
            hidden_state_index=hidden_state_index,
            store_all_hidden_states=store_all_hidden_states,
        )

    def predict_contacts(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.esm.predict_contacts(input_ids, attention_mask=attention_mask)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_s_max: bool | None = False,
        return_dict: bool | None = None,
    ) -> EsmTokenClassifierOutput | tuple[torch.Tensor, ...]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_s_max=output_s_max,
            return_dict=True,
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        result = EsmTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            s_max=outputs.s_max,
        )
        return result if return_dict else result.to_tuple()
