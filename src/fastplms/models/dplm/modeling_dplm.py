"""FastPLMs-compatible DPLM implementation."""

# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import EsmTokenizer
from transformers.modeling_outputs import (
    MaskedLMOutput,
    ModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.models.esm.configuration_esm import EsmConfig
from transformers.models.esm.modeling_esm import (
    EsmAttention,
    EsmClassificationHead,
    EsmContactPredictionHead,
    EsmEmbeddings,
    EsmEncoder,
    EsmLayer,
    EsmLMHead,
    EsmPooler,
    EsmPreTrainedModel,
    EsmSelfAttention,
)

from fastplms.models._diffusion_generation import generate_dplm
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
class DPLMMaskedLMOutput(MaskedLMOutput):
    """Masked-LM output with DPLM extensions after the HF fields."""

    s_max: tuple[list[torch.Tensor], ...] | None = None
    last_hidden_state: torch.Tensor | None = None


@dataclass
class DPLMSequenceClassifierOutput(SequenceClassifierOutput):
    """Sequence-classification output with optional attention diagnostics."""

    s_max: tuple[list[torch.Tensor], ...] | None = None


@dataclass
class DPLMTokenClassifierOutput(TokenClassifierOutput):
    """Token-classification output with optional attention diagnostics."""

    s_max: tuple[list[torch.Tensor], ...] | None = None


@dataclass
class DPLMEncoderOutput(ModelOutput):
    last_hidden_state: torch.Tensor | None = None
    pooler_output: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None
    s_max: tuple[list[torch.Tensor], ...] | None = None


def _reject_unsupported_dplm_arguments(**arguments: object) -> None:
    unsupported = [
        name
        for name, value in arguments.items()
        if value is not None and not (name == "use_cache" and value is False)
    ]
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(
            "DPLM is an encoder-only diffusion model and does not support "
            f"decoder, cross-attention, or KV-cache arguments: {names}."
        )


class DPLMConfig(EsmConfig):
    model_type = "dplm"

    def __init__(
        self,
        attn_backend: str | None = None,
        add_pooling_layer: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attn_backend = attn_backend
        self.add_pooling_layer = add_pooling_layer
        self.tie_word_embeddings = False


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


class DPLMPreTrainedModel(FastPLMsAttentionMixin, EsmPreTrainedModel):
    config_class = DPLMConfig
    # All advertised wrappers install the encoder at ``self.esm``.  Keep the
    # Hugging Face base-model and checkpoint-prefix contract aligned with that
    # actual module path.
    base_model_prefix = "esm"
    supports_gradient_checkpointing = True
    all_tied_weights_keys: ClassVar[dict[str, str]] = {}
    _supports_flash_attn = True
    _supports_flash_attn_2 = False
    _supports_flash_attn_3 = True
    _fastplms_attention_implementations = (
        "eager",
        "sdpa",
        "flex_attention",
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
                    "DPLM tokenizer loading requires a model loaded with from_pretrained "
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
            tokenizer = EsmTokenizer.from_pretrained(source, **tokenizer_kwargs)
            self.__dict__["_fastplms_tokenizer"] = tokenizer
        return tokenizer

    @tokenizer.setter
    def tokenizer(self, value) -> None:
        self.__dict__["_fastplms_tokenizer"] = value

    @property
    def attn_backend(self) -> str:
        return self.config.attn_backend

    @attn_backend.setter
    def attn_backend(self, backend: str) -> None:
        if backend not in self._fastplms_attention_implementations:
            raise ValueError(
                f"DPLM does not support {backend!r}; expected one of "
                f"{self._fastplms_attention_implementations}."
            )
        self.config.attn_backend = backend
        resolved = resolve_attention_backend(backend)
        for module in self.modules():
            if isinstance(module, ModifiedEsmEncoder):
                module.attention_backend = resolved
            elif isinstance(module, ModifiedEsmSelfAttention):
                module.attn_backend = resolved


class ModifiedEsmSelfAttention(EsmSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type)
        self.config = config
        self.scale = self.attention_head_size**-0.5
        self.dropout_prob = float(config.attention_probs_dropout_prob)
        self.attn_backend = resolve_attention_backend(config.attn_backend)
        if self.position_embedding_type == "rotary":
            self.rotary_embeddings = RotaryEmbedding(dim=self.attention_head_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = (*x.size()[:-1], self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_2d: torch.Tensor | None = None,
        attention_mask_4d: torch.Tensor | None = None,
        flex_block_mask: object | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_value: tuple[tuple[torch.FloatTensor]] | None = None,
        output_attentions: bool | None = False,
        output_s_max: bool | None = False,
        past_key_values: tuple[tuple[torch.FloatTensor]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        if past_key_values is not None:
            past_key_value = past_key_values

        mixed_query_layer = self.query(hidden_states)
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            cross_attn_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            cross_attn_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
            cross_attn_mask = None
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            cross_attn_mask = None

        query_layer = self.transpose_for_scores(mixed_query_layer) * self.scale

        if self.position_embedding_type == "rotary":
            query_layer, key_layer = self.rotary_embeddings(query_layer, key_layer)

        if self.position_embedding_type in ["relative_key", "relative_key_query"]:
            raise NotImplementedError

        query_layer = query_layer.contiguous()
        key_layer = key_layer.contiguous()
        value_layer = value_layer.contiguous()

        if is_cross_attention:
            if self.attn_backend not in {
                AttentionBackend.EAGER,
                AttentionBackend.SDPA,
            }:
                raise RuntimeError(
                    f"DPLM cross-attention does not implement {self.attn_backend.value!r}. "
                    "Use eager or SDPA for decoder cross-attention."
                )
            if output_attentions:
                attn_output, attn_weights, s_max = self._manual_attn(
                    query_layer,
                    key_layer,
                    value_layer,
                    cross_attn_mask,
                    output_s_max,
                )
            elif self.attn_backend == AttentionBackend.EAGER:
                attn_output, _, s_max = self._manual_attn(
                    query_layer,
                    key_layer,
                    value_layer,
                    cross_attn_mask,
                    output_s_max,
                )
                attn_weights = None
            elif self.attn_backend == AttentionBackend.SDPA:
                attn_output, attn_weights = self._sdpa_attn(
                    query_layer,
                    key_layer,
                    value_layer,
                    cross_attn_mask,
                )
                s_max = self._compute_s_max(query_layer, key_layer) if output_s_max else None
        else:
            attn_output, attn_weights, s_max = self._attn(
                query_layer,
                key_layer,
                value_layer,
                attention_mask_2d=attention_mask_2d,
                attention_mask_4d=attention_mask_4d,
                flex_block_mask=flex_block_mask,
                output_attentions=output_attentions,
                output_s_max=output_s_max,
            )

        if head_mask is not None and torch.is_tensor(head_mask):
            batch_size, seq_len, _ = attn_output.shape
            attn_output = attn_output.view(
                batch_size, seq_len, self.num_attention_heads, self.attention_head_size
            )
            attn_output = attn_output.permute(0, 2, 1, 3) * head_mask
            attn_output = rearrange(attn_output, "b h s d -> b s (h d)")

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
                f"DPLM {self.attn_backend.value} attention is inference-only when attention "
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
            if attention_mask_4d.dtype == torch.bool:
                attn_weights = attn_weights.masked_fill(
                    attention_mask_4d.logical_not(),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights + attention_mask_4d.to(
                    device=attn_weights.device,
                    dtype=attn_weights.dtype,
                )
        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.dropout_prob > 0 and self.training:
            attn_weights = F.dropout(
                attn_weights,
                p=self.dropout_prob,
                training=True,
            )
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
        # Pass softmax_scale=1.0 to prevent double-scaling by the kernel.
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
        # The pinned official DPLM path uses Torch's efficient SDPA kernel for
        # its non-null padding mask. Torch 2.13 otherwise selects cuDNN on H100,
        # changing every downstream hidden state. Requiring the same public
        # SDPA kernel makes the official FP32-storage/BF16-autocast path exact.
        kernel_context = (
            sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
            if query_heads.is_cuda
            else contextlib.nullcontext()
        )
        with kernel_context:
            context_heads = F.scaled_dot_product_attention(
                query_heads,
                key_heads,
                value_heads,
                attn_mask=attention_mask_4d,
                dropout_p=self.dropout_prob if self.training else 0.0,
                scale=1.0,
            )
        return rearrange(context_heads, "b h s d -> b s (h d)"), None


class ModifiedEsmAttention(EsmAttention):
    def __init__(self, config):
        # Reuse Transformers' maintained ESM container layout, replacing only
        # the self-attention engine that FastPLMs extends. This preserves the
        # checkpoint schema without duplicating an upstream DPLM constructor.
        EsmAttention.__init__(self, config)
        self.self = ModifiedEsmSelfAttention(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_2d: torch.Tensor | None = None,
        attention_mask_4d: torch.Tensor | None = None,
        flex_block_mask: object | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[tuple[torch.FloatTensor]] | None = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        hidden_states_ln = self.LayerNorm(hidden_states)
        attn_output, attn_weights, s_max = self.self(
            hidden_states_ln,
            attention_mask_2d=attention_mask_2d,
            attention_mask_4d=attention_mask_4d,
            flex_block_mask=flex_block_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
        )
        attention_output = self.output(attn_output, hidden_states)
        return attention_output, attn_weights, s_max


class ModifiedEsmLayer(EsmLayer):
    def __init__(self, config):
        # Transformers owns the feed-forward, normalization, and decoder
        # plumbing. Only attention dispatch differs for DPLM.
        EsmLayer.__init__(self, config)
        self.attention = ModifiedEsmAttention(config)
        if self.add_cross_attention:
            self.crossattention = ModifiedEsmAttention(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_2d: torch.Tensor | None = None,
        attention_mask_4d: torch.Tensor | None = None,
        flex_block_mask: object | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[tuple[torch.FloatTensor]] | None = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        attention_output, attn_weights, s_max = self.attention(
            hidden_states,
            attention_mask_2d=attention_mask_2d,
            attention_mask_4d=attention_mask_4d,
            flex_block_mask=flex_block_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
            past_key_value=past_key_value[:2] if past_key_value is not None else None,
        )

        if self.is_decoder and encoder_hidden_states is not None:
            if self.add_cross_attention is False:
                raise AttributeError(
                    f"If `encoder_hidden_states` are passed, {self} has to be "
                    "instantiated with cross-attention "
                    "layers by setting `config.add_cross_attention=True`"
                )
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_output, _, _ = self.crossattention(
                attention_output,
                attention_mask_2d=attention_mask_2d,
                attention_mask_4d=attention_mask_4d,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
                output_s_max=False,
            )
            attention_output = cross_attention_output

        layer_output = self.feed_forward_chunk(attention_output)
        return layer_output, attn_weights, s_max


class ModifiedEsmEncoder(EsmEncoder):
    def __init__(self, config):
        # Start from the public Transformers encoder contract, then substitute
        # backend-aware layers while retaining every canonical state key.
        EsmEncoder.__init__(self, config)
        self.attention_backend = resolve_attention_backend(config.attn_backend)
        self.layer = nn.ModuleList(
            ModifiedEsmLayer(config) for _ in range(config.num_hidden_layers)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_values: list[tuple[tuple[torch.FloatTensor]]] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_s_max: bool = False,
    ) -> DPLMEncoderOutput:
        first_parameter = next(self.parameters(), None)
        if (
            not self.training
            and first_parameter is not None
            and first_parameter.dtype == torch.bfloat16
        ):
            raise RuntimeError(
                "DPLM BF16 inference requires FP32-resident parameters under "
                "CUDA BF16 autocast; static BF16 parameters do not meet the "
                "declared parity contract."
            )
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
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

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = (*all_hidden_states, hidden_states)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                hidden_states, attn_weights, s_max = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask_2d,
                    attention_mask_4d,
                    flex_block_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    output_s_max,
                )
            else:
                hidden_states, attn_weights, s_max = layer_module(
                    hidden_states,
                    attention_mask_2d=attention_mask_2d,
                    attention_mask_4d=attention_mask_4d,
                    flex_block_mask=flex_block_mask,
                    head_mask=layer_head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    output_s_max=output_s_max,
                )

            if all_self_attentions is not None:
                all_self_attentions = (*all_self_attentions, attn_weights)
            if full_s_max is not None:
                full_s_max = (*full_s_max, s_max)

        if self.emb_layer_norm_after:
            hidden_states = self.emb_layer_norm_after(hidden_states)

        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        return DPLMEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            s_max=full_s_max,
        )


class FAST_DPLM_ENCODER(DPLMPreTrainedModel, EmbeddingMixin):
    """Inner encoder class that holds the actual ESM-style weights (embeddings, encoder,
    contact_head) so that the weight keys are prefixed with 'esm.' in the outer DPLMModel,
    matching pretrained DPLM checkpoints."""

    def __init__(self, config, **kwargs):
        DPLMPreTrainedModel.__init__(self, config, **kwargs)
        self.config = config
        self.embeddings = EsmEmbeddings(config)
        self.encoder = ModifiedEsmEncoder(config)
        self.contact_head = EsmContactPredictionHead(
            in_features=config.num_hidden_layers * config.num_attention_heads,
            bias=True,
        )
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
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
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        embedding_output = self.embeddings(input_ids, attention_mask=attention_mask)
        output_hidden_states = store_all_hidden_states or hidden_state_index != -1
        encoder_outputs = self.encoder(
            embedding_output,
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
        attns = self(input_ids, attention_mask=attention_mask, output_attentions=True).attentions
        attns = torch.stack(attns, dim=1)
        attns *= attention_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        attns *= attention_mask.unsqueeze(1).unsqueeze(2).unsqueeze(4)
        return self.contact_head(input_ids, attns)

    def _convert_head_mask_to_5d(
        self, head_mask: torch.Tensor, num_hidden_layers: int
    ) -> torch.Tensor:
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        if head_mask.dim() != 5:
            raise ValueError(f"head_mask.dim != 5, got {head_mask.dim()}")
        head_mask = head_mask.to(dtype=self.dtype)
        return head_mask

    def get_head_mask(
        self,
        head_mask: torch.Tensor | None,
        num_hidden_layers: int,
        is_attention_chunked: bool = False,
    ) -> torch.Tensor | list[None]:
        if head_mask is None:
            return [None] * num_hidden_layers
        head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
        if is_attention_chunked:
            head_mask = head_mask.unsqueeze(-1)
        return head_mask

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_s_max: bool | None = False,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor] | DPLMEncoderOutput:
        if self.config.is_decoder or self.config.add_cross_attention:
            raise ValueError(
                "DPLM is encoder-only; is_decoder and add_cross_attention must be false."
            )
        _reject_unsupported_dplm_arguments(
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
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

        expected_attention_mask_shape = (batch_size, seq_length)
        if attention_mask is None:
            attention_mask_2d = torch.ones((batch_size, seq_length), device=device).bool()
        elif attention_mask.dim() == 4:
            raise ValueError(
                "DPLM accepts a two-dimensional padding mask. Passing a four-dimensional "
                "custom attention mask is unsupported because it cannot be applied to both "
                "the embedding and optimized-attention paths without changing semantics."
            )
        elif (
            attention_mask.dim() != 2
            or tuple(attention_mask.shape) != expected_attention_mask_shape
        ):
            raise ValueError(
                f"attention_mask must have shape {expected_attention_mask_shape}; "
                f"received {tuple(attention_mask.shape)}."
            )
        else:
            attention_mask_2d = attention_mask.to(device=device, dtype=torch.bool)

        encoder_extended_attention_mask = encoder_attention_mask
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask_2d,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask_2d,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_s_max=output_s_max,
        )
        sequence_output = encoder_outputs.last_hidden_state

        if return_dict is False:
            return (sequence_output, *encoder_outputs[1:])

        result = DPLMEncoderOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            s_max=encoder_outputs.s_max,
        )
        return result


class DPLMModel(DPLMPreTrainedModel, EmbeddingMixin):
    config_class = DPLMConfig

    def __init__(self, config, add_pooling_layer: bool | None = None):
        DPLMPreTrainedModel.__init__(self, config)
        self.config = config
        self.esm = FAST_DPLM_ENCODER(config)
        if add_pooling_layer is None:
            add_pooling_layer = config.add_pooling_layer
        config.add_pooling_layer = bool(add_pooling_layer)
        self.pooler = EsmPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
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
        return self.esm.predict_contacts(input_ids, attention_mask)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_s_max: bool | None = False,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor] | DPLMEncoderOutput:
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_s_max=output_s_max,
            return_dict=True,
        )
        sequence_output = outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        result = DPLMEncoderOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            s_max=outputs.s_max,
        )
        return result if return_dict else result.to_tuple()


class DPLMForMaskedLM(FastPLMTestTimeTrainingMixin, DPLMPreTrainedModel, EmbeddingMixin):
    config_class = DPLMConfig

    def __init__(self, config, dropout: float | None = None):
        if dropout is not None:
            config.hidden_dropout_prob = dropout
        DPLMPreTrainedModel.__init__(self, config)
        self.esm = FAST_DPLM_ENCODER(config)
        self.lm_head = EsmLMHead(config)
        self.loss_fct = nn.CrossEntropyLoss()
        self.post_init()
        self.contact_head = None
        self.init_ttt({"lora_target_replace_module": "ModifiedEsmAttention"})

    def get_input_embeddings(self) -> nn.Module:
        return self.esm.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
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

    def generate(
        self,
        input_tokens: torch.Tensor,
        tokenizer: object | None = None,
        max_iter: int | None = None,
        temperature: float | None = None,
        partial_masks: torch.Tensor | None = None,
        sampling_strategy: str = "gumbel_argmax",
        disable_resample: bool = False,
        resample_ratio: float = 0.25,
        show_progress: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Generate protein tokens with the official DPLM diffusion schedule.

        ``input_tokens`` is X with shape (b, l). Positions marked ``True`` in
        ``partial_masks`` remain fixed. ``max_iter=None`` uses the official
        500-step schedule; shorter schedules are useful for rapid exploration.
        """

        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected DPLM generation arguments: {names}")
        return generate_dplm(
            self,
            input_tokens,
            tokenizer=tokenizer,
            max_iter=max_iter,
            temperature=temperature,
            partial_masks=partial_masks,
            sampling_strategy=sampling_strategy,
            disable_resample=disable_resample,
            resample_ratio=resample_ratio,
            show_progress=show_progress,
        )

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
        inputs_embeds: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        decoder_inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_s_max: bool | None = False,
        return_dict: bool | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor] | DPLMMaskedLMOutput:
        _reject_unsupported_dplm_arguments(
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_inputs_embeds=decoder_inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if attention_mask is None and input_ids is not None:
            attention_mask = input_ids.ne(self.config.pad_token_id)

        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_s_max=output_s_max,
            return_dict=True,
        )
        sequence_output = outputs.last_hidden_state
        logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        result = DPLMMaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            s_max=outputs.s_max,
            last_hidden_state=sequence_output,
        )
        return result if return_dict else result.to_tuple()


class DPLMForSequenceClassification(DPLMPreTrainedModel, EmbeddingMixin):
    config_class = DPLMConfig

    def get_input_embeddings(self) -> nn.Module:
        return self.esm.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.esm.set_input_embeddings(value)

    def __init__(self, config):
        DPLMPreTrainedModel.__init__(self, config)
        self.num_labels = config.num_labels
        self.esm = FAST_DPLM_ENCODER(config)
        self.classifier = EsmClassificationHead(config)
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.post_init()

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

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_s_max: bool | None = False,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor, ...] | DPLMSequenceClassifierOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
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

        result = DPLMSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            s_max=outputs.s_max,
        )
        return result if return_dict else result.to_tuple()


class DPLMForTokenClassification(DPLMPreTrainedModel, EmbeddingMixin):
    config_class = DPLMConfig

    def get_input_embeddings(self) -> nn.Module:
        return self.esm.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.esm.set_input_embeddings(value)

    def __init__(self, config):
        DPLMPreTrainedModel.__init__(self, config)
        self.num_labels = config.num_labels
        self.esm = FAST_DPLM_ENCODER(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        self.post_init()

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

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_s_max: bool | None = False,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor, ...] | DPLMTokenClassifierOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_s_max=output_s_max,
            return_dict=True,
        )
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        result = DPLMTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            s_max=outputs.s_max,
        )
        return result if return_dict else result.to_tuple()
