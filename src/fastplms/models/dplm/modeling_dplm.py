from __future__ import annotations

# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
FastPLMs-compatible DPLM implementation.
"""

import contextlib

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from einops import rearrange

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
    EsmContactPredictionHead,
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
)

from fastplms.models._esm_rotary import RotaryEmbedding
from fastplms.models._diffusion_generation import generate_dplm

try:
    from fastplms.attention import (
        AttentionBackend,
        resolve_attention_backend,
        get_attention_mask,
        _get_flex_attention_fn,
        FastPLMsAttentionMixin,
        kernels_flash_attention_func,
        create_block_mask,
        flex_attention,
        BlockMask,
    )
    from fastplms.embeddings import EmbeddingMixin, select_hidden_state_embeddings
    from fastplms.models.ttt import FastPLMTestTimeTrainingMixin
except ImportError:
    pass  # Running as HF Hub composite; shared definitions are above


@dataclass
class DPLMMaskedLMOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    s_max: Optional[Tuple[List[torch.Tensor], ...]] = None


@dataclass
class DPLMEncoderOutput(ModelOutput):
    last_hidden_state: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    s_max: Optional[Tuple[List[torch.Tensor], ...]] = None


class DPLMConfig(EsmConfig):
    model_type = "dplm"

    def __init__(
        self,
        attn_backend: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attn_backend = attn_backend
        self.tie_word_embeddings = False


class DPLMPreTrainedModel(FastPLMsAttentionMixin, EsmPreTrainedModel):
    config_class = DPLMConfig
    base_model_prefix = "dplm"
    supports_gradient_checkpointing = True
    all_tied_weights_keys = {}
    _supports_flash_attn = True
    _supports_flash_attn_2 = False
    _supports_flash_attn_3 = True
    _fastplms_attention_implementations = (
        "eager",
        "sdpa",
        "flex_attention",
        "flash_attention_3",
    )

    @property
    def tokenizer(self):
        tokenizer = self.__dict__.get("_fastplms_tokenizer")
        if tokenizer is None:
            source = str(getattr(self.config, "_name_or_path", "")).strip()
            if not source:
                raise RuntimeError(
                    "DPLM tokenizer loading requires a model loaded with from_pretrained "
                    "so checkpoint provenance is available."
                )
            revision = getattr(self.config, "_commit_hash", None)
            tokenizer_kwargs = {"revision": revision} if revision else {}
            tokenizer = EsmTokenizer.from_pretrained(source, **tokenizer_kwargs)
            self.__dict__["_fastplms_tokenizer"] = tokenizer
        return tokenizer

    @tokenizer.setter
    def tokenizer(self, value) -> None:
        self.__dict__["_fastplms_tokenizer"] = value

    @classmethod
    def is_remote_code(cls) -> bool:
        # Prevent post-load reinitialization of tensors already loaded from checkpoints.
        return True

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
        self.attn_backend = resolve_attention_backend(config.attn_backend)
        if self.position_embedding_type == "rotary":
            self.rotary_embeddings = RotaryEmbedding(dim=self.attention_head_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor] = None,
        attention_mask_4d: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[object] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        output_s_max: Optional[bool] = False,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
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
            if output_attentions:
                attn_output, attn_weights, s_max = self._manual_attn(
                    query_layer,
                    key_layer,
                    value_layer,
                    cross_attn_mask,
                    output_s_max,
                )
            else:
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
        attention_mask_2d: Optional[torch.Tensor] = None,
        attention_mask_4d: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[BlockMask] = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        if output_attentions:
            return self._manual_attn(
                query_heads, key_heads, value_heads, attention_mask_4d, output_s_max
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
    ) -> List[torch.Tensor]:
        q_norm = torch.linalg.vector_norm(query_heads, dim=-1)
        k_norm = torch.linalg.vector_norm(key_heads, dim=-1)
        s_max_bound = (q_norm.max(dim=-1).values * k_norm.max(dim=-1).values).max(dim=0).values
        return [s_max_bound[h] for h in range(self.num_attention_heads)]

    def _manual_attn(
        self,
        query_heads: torch.Tensor,
        key_heads: torch.Tensor,
        value_heads: torch.Tensor,
        attention_mask_4d: Optional[torch.Tensor] = None,
        output_s_max: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        attn_weights = torch.matmul(query_heads, key_heads.transpose(-1, -2))
        if attention_mask_4d is not None:
            attn_weights = attn_weights.masked_fill(attention_mask_4d.logical_not(), float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        context_heads = torch.matmul(attn_weights, value_heads)
        attn_output = rearrange(context_heads, "b h s d -> b s (h d)")
        s_max = self._compute_s_max(query_heads, key_heads) if output_s_max else None
        return attn_output, attn_weights, s_max

    def _kernels_flash_attn(
        self,
        query_heads: torch.Tensor,
        key_heads: torch.Tensor,
        value_heads: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
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
        flex_block_mask: Optional[BlockMask] = None,
        attention_mask_2d: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        assert flex_attention is not None, "Flex attention is not available in this environment."
        sequence_lengths = (
            tuple(int(length) for length in attention_mask_2d.sum(dim=-1).tolist())
            if attention_mask_2d is not None
            else (query_heads.shape[-2],) * query_heads.shape[0]
        )
        fn = _get_flex_attention_fn(
            device=query_heads.device,
            dtype=query_heads.dtype,
            shape=tuple(query_heads.shape),
            sequence_lengths=sequence_lengths,
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
        attention_mask_4d: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
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
        attention_mask_2d: Optional[torch.Tensor] = None,
        attention_mask_4d: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[object] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
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
        attention_mask_2d: Optional[torch.Tensor] = None,
        attention_mask_4d: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[object] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
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
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention "
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
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[Tuple[torch.FloatTensor]]]] = None,
        use_cache: Optional[bool] = None,
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

        attention_mask_2d, attention_mask_4d, flex_block_mask = get_attention_mask(
            effective_backend=self.attention_backend,
            batch_size=hidden_states.shape[0],
            seq_len=hidden_states.shape[1],
            device=hidden_states.device,
            attention_mask=attention_mask,
            dtype=hidden_states.dtype,
            mask_semantics="padding",
        )

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

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
                all_self_attentions = all_self_attentions + (attn_weights,)
            if full_s_max is not None:
                full_s_max = full_s_max + (s_max,)

        if self.emb_layer_norm_after:
            hidden_states = self.emb_layer_norm_after(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

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
        attention_mask: Optional[torch.Tensor] = None,
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
        output_s_max: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], DPLMEncoderOutput]:
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

        if attention_mask is None:
            attention_mask_2d = torch.ones((batch_size, seq_length), device=device).bool()
        elif attention_mask.dim() == 2:
            attention_mask_2d = attention_mask.bool()
        elif attention_mask.dim() == 4:
            assert input_ids is not None, (
                "4D attention_mask requires input_ids to infer token-level mask."
            )
            attention_mask_2d = input_ids.ne(self.config.pad_token_id)
        else:
            raise ValueError(f"Unsupported attention_mask shape: {attention_mask.shape}")

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
            return (sequence_output,) + encoder_outputs[1:]

        return DPLMEncoderOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            s_max=encoder_outputs.s_max,
        )


class DPLMModel(DPLMPreTrainedModel, EmbeddingMixin):
    config_class = DPLMConfig

    def __init__(self, config, add_pooling_layer=True):
        DPLMPreTrainedModel.__init__(self, config)
        self.config = config
        self.esm = FAST_DPLM_ENCODER(config)
        self.pooler = EsmPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.esm.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.esm.embeddings.word_embeddings = value

    def _embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
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
        output_s_max: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], DPLMEncoderOutput]:
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
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if return_dict is False:
            return (sequence_output, pooled_output) + outputs[1:]

        return DPLMEncoderOutput(
            last_hidden_state=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            s_max=outputs.s_max,
        )


class DPLMForMaskedLM(FastPLMTestTimeTrainingMixin, DPLMPreTrainedModel, EmbeddingMixin):
    config_class = DPLMConfig

    def __init__(self, config, dropout: float = 0.1):
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

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
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
        attention_mask: Optional[torch.Tensor] = None,
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
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_s_max: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], DPLMMaskedLMOutput]:
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

        if return_dict is False:
            output = (logits, sequence_output, outputs.hidden_states, outputs.attentions)
            if loss is not None:
                return (loss,) + output
            return output

        return DPLMMaskedLMOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            s_max=outputs.s_max,
        )


class DPLMForSequenceClassification(DPLMPreTrainedModel, EmbeddingMixin):
    config_class = DPLMConfig

    def get_input_embeddings(self) -> nn.Module:
        return self.esm.get_input_embeddings()

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
        attention_mask: Optional[torch.Tensor] = None,
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
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_s_max: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], DPLMMaskedLMOutput]:
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

        return DPLMMaskedLMOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            s_max=outputs.s_max,
        )


class DPLMForTokenClassification(DPLMPreTrainedModel, EmbeddingMixin):
    config_class = DPLMConfig

    def get_input_embeddings(self) -> nn.Module:
        return self.esm.get_input_embeddings()

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
        attention_mask: Optional[torch.Tensor] = None,
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
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_s_max: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], DPLMMaskedLMOutput]:
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

        return DPLMMaskedLMOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            s_max=outputs.s_max,
        )
