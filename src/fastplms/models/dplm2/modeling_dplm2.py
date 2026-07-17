"""
FastPLMs-compatible DPLM2 implementation.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from transformers import AutoTokenizer
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

from fastplms.models._esm_rotary import RotaryEmbedding, apply_rotary_pos_emb
from fastplms.models._diffusion_generation import generate_dplm2
from fastplms.models.dplm2.tokenization_dplm2 import DPLM2Tokenizer

try:
    from fastplms.attention import (
        AttentionBackend,
        FastPLMsAttentionMixin,
        get_attention_mask,
        resolve_attention_backend,
    )
    from fastplms.embeddings import EmbeddingMixin, select_hidden_state_embeddings
    from fastplms.models.ttt import FastPLMTestTimeTrainingMixin
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
    assert type_ids.ndim == 2, (
        f"Expected type_ids to have shape (batch, seq_len), got {tuple(type_ids.shape)}"
    )
    seq_len = type_ids.shape[-1]
    if seq_len % 2 != 0:
        return False

    half_len = seq_len // 2
    first_half = type_ids[:, :half_len]
    second_half = type_ids[:, half_len:]

    first_is_aa = ((first_half == aa_type) | (first_half == pad_type)).all(dim=-1)
    first_is_struct = ((first_half == struct_type) | (first_half == pad_type)).all(dim=-1)
    second_is_aa = ((second_half == aa_type) | (second_half == pad_type)).all(dim=-1)
    second_is_struct = ((second_half == struct_type) | (second_half == pad_type)).all(dim=-1)
    first_count = first_half.ne(pad_type).sum(dim=-1)
    second_count = second_half.ne(pad_type).sum(dim=-1)
    modalities_are_separate = (first_is_aa & second_is_struct) | (first_is_struct & second_is_aa)
    packed_rows = modalities_are_separate & first_count.gt(0) & first_count.eq(second_count)
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
        attn_backend: Optional[str] = None,
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


class DPLM2PreTrainedModel(FastPLMsAttentionMixin, EsmPreTrainedModel):
    config_class = DPLM2Config
    base_model_prefix = "dplm2"
    supports_gradient_checkpointing = True
    all_tied_weights_keys = {}
    _supports_flex_attn = False
    _supports_flash_attn = False
    _supports_flash_attn_2 = False
    _supports_flash_attn_3 = False
    _fastplms_attention_implementations = ("sdpa",)

    @property
    def tokenizer(self):
        tokenizer = self.__dict__.get("_fastplms_tokenizer")
        if tokenizer is None:
            source = str(getattr(self.config, "_name_or_path", "")).strip()
            if not source:
                raise RuntimeError(
                    "DPLM2 tokenizer loading requires a model loaded with from_pretrained "
                    "so checkpoint provenance is available."
                )
            revision = getattr(self.config, "_commit_hash", None)
            tokenizer_kwargs = {"revision": revision} if revision else {}
            tokenizer = DPLM2Tokenizer.from_pretrained(source, **tokenizer_kwargs)
            self.__dict__["_fastplms_tokenizer"] = tokenizer
        return tokenizer

    @tokenizer.setter
    def tokenizer(self, value) -> None:
        self.__dict__["_fastplms_tokenizer"] = value

    def _tokenize_sequence_batch(
        self,
        sequences: Sequence[str],
        *,
        tokenizer: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Tokenize raw amino-acid sequences with official DPLM2 boundaries."""

        resolved = tokenizer if tokenizer is not None else self.tokenizer
        sequence_list = [sequences] if isinstance(sequences, str) else sequences
        formatted = [
            f"{resolved.aa_cls_token}{sequence}{resolved.aa_eos_token}"
            for sequence in sequence_list
        ]
        return resolved(formatted, add_special_tokens=False, **kwargs)

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
                f"DPLM2 does not support {backend!r}; expected one of "
                f"{self._fastplms_attention_implementations}."
            )
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
        # as two equal-length, modality-specific halves. Either track may come first.
        # Plain protein batches can still contain high-ID special tokens, so mere
        # modality presence is not enough.
        return _has_packed_multimodal_layout(
            type_ids=type_ids,
            aa_type=self.aa_type,
            struct_type=self.struct_type,
            pad_type=self.pad_type,
        )

    def align_frequency_buffer(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Match the official model-wide ``to(device, dtype)`` conversion.

        Transformers' meta-device loader converts parameters to the requested
        dtype but can leave this persistent rotary buffer in FP32. The pinned
        official implementation moves the complete module, including
        ``inv_freq``. Aligning the buffer before building rotary factors keeps
        Q, K, and V in one dtype for every attention backend.
        """

        if self.inv_freq.device == device and self.inv_freq.dtype == dtype:
            return
        self.inv_freq = self.inv_freq.to(device=device, dtype=dtype)
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

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
            or self._cos_cached.dtype != self.inv_freq.dtype
        )
        if cache_is_stale:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            # Match the official DPLM2 operation order: rotary factors inherit
            # the frequency-buffer dtype. This keeps them in FP32 under BF16
            # autocast, while a model explicitly converted to BF16 still builds
            # BF16 factors and remains usable without autocast.
            emb = torch.cat((freqs, freqs), dim=-1).to(device=x.device)
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
        attention_mask_4d: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
        type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        batch_size, seq_length = hidden_states.shape[:-1]
        hidden_shape = (batch_size, seq_length, -1, self.attention_head_size)
        query_heads = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        key_heads = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        value_heads = self.value(hidden_states).view(hidden_shape).transpose(1, 2)

        query_heads = query_heads * self.scale

        if self.position_embedding_type == "rotary":
            self.rotary_embeddings.align_frequency_buffer(
                device=query_heads.device,
                dtype=self.query.weight.dtype,
            )
            query_heads, key_heads = self.rotary_embeddings(query_heads, key_heads, type_ids)

        attn_output, attn_weights, s_max = self._attn(
            query_heads,
            key_heads,
            value_heads,
            attention_mask_4d=attention_mask_4d,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
        )
        return attn_output, attn_weights, s_max

    def _attn(
        self,
        query_heads: torch.Tensor,
        key_heads: torch.Tensor,
        value_heads: torch.Tensor,
        attention_mask_4d: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        if output_attentions:
            return self._manual_attn(
                query_heads, key_heads, value_heads, attention_mask_4d, output_s_max
            )

        if self.attn_backend != AttentionBackend.SDPA:
            raise AssertionError(f"Unsupported resolved backend: {self.attn_backend}")
        attn_output, attn_weights = self._sdpa_attn(
            query_heads,
            key_heads,
            value_heads,
            attention_mask_4d,
        )

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
        if self.dropout_prob > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout_prob, training=self.training)
        context_heads = torch.matmul(attn_weights, value_heads)
        attn_output = rearrange(context_heads, "b h s d -> b s (h d)")
        s_max = self._compute_s_max(query_heads, key_heads) if output_s_max else None
        return attn_output, attn_weights, s_max

    def _sdpa_attn(
        self,
        query_heads: torch.Tensor,
        key_heads: torch.Tensor,
        value_heads: torch.Tensor,
        attention_mask_4d: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        # Pinned DPLM2 uses PyTorch's efficient SDPA kernel for its non-null
        # padding mask. Newer PyTorch releases otherwise select cuDNN on H100,
        # which exceeds the fixed deep-BF16 parity target. This is still the
        # public SDPA operation and raises if its required CUDA kernel is absent.
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
        nn.Module.__init__(self)
        self.self = ModifiedEsmSelfAttention(config)
        self.output = EsmSelfOutput(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_4d: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
        type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        hidden_states_ln = self.LayerNorm(hidden_states)
        attn_output, attn_weights, s_max = self.self(
            hidden_states_ln,
            attention_mask_4d=attention_mask_4d,
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
        attention_mask_4d: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
        type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        attention_output, attn_weights, s_max = self.attention(
            hidden_states,
            attention_mask_4d=attention_mask_4d,
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
        self.layer = nn.ModuleList(
            [ModifiedEsmLayer(config) for _ in range(config.num_hidden_layers)]
        )
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
        first_parameter = next(self.parameters(), None)
        if (
            not self.training
            and first_parameter is not None
            and first_parameter.dtype == torch.bfloat16
        ):
            raise RuntimeError(
                "DPLM2 BF16 inference requires FP32-resident parameters under "
                "CUDA BF16 autocast; static BF16 parameters do not meet the "
                "declared parity contract."
            )
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        full_s_max = () if output_s_max else None

        _, attention_mask_4d, _ = get_attention_mask(
            effective_backend=self.attention_backend,
            batch_size=hidden_states.shape[0],
            seq_len=hidden_states.shape[1],
            device=hidden_states.device,
            attention_mask=attention_mask,
            dtype=hidden_states.dtype,
            mask_semantics="padding",
        )

        for layer_module in self.layer:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                hidden_states, attn_weights, s_max = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask_4d,
                    output_attentions,
                    output_s_max,
                    type_ids,
                )
            else:
                hidden_states, attn_weights, s_max = layer_module(
                    hidden_states,
                    attention_mask_4d=attention_mask_4d,
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
        self.contact_head = EsmContactPredictionHead(
            in_features=config.num_hidden_layers * config.num_attention_heads,
            bias=True,
        )
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def predict_contacts(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict residue contacts with the checkpoint's tied contact head."""
        input_ids = _normalize_dplm2_input_ids(input_ids, self.config.vocab_size)
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        type_ids = self._get_modality_type(input_ids, attention_mask)
        attentions = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            type_ids=type_ids,
            output_attentions=True,
        ).attentions
        if attentions is None:
            raise RuntimeError("DPLM2 did not return attention maps for contact prediction.")
        # A is the layer/head attention tensor; M marks valid tokens.
        attention_tensor = torch.stack(attentions, dim=1)
        residue_mask = attention_mask.to(dtype=attention_tensor.dtype)
        attention_tensor = (
            attention_tensor
            * residue_mask[:, None, None, :, None]
            * residue_mask[:, None, None, None, :]
        )
        return self.contact_head(input_ids, attention_tensor)

    def _get_modality_type(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return _infer_modality_type(input_ids, attention_mask)

    def _embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        hidden_state_index: int = -1,
        store_all_hidden_states: bool = False,
    ) -> torch.Tensor:
        input_ids = _normalize_dplm2_input_ids(input_ids, self.config.vocab_size)
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        type_ids = _infer_modality_type(input_ids, attention_mask)
        token_embedding_output = self.embeddings(input_ids, attention_mask=attention_mask)
        output_hidden_states = store_all_hidden_states or hidden_state_index != -1
        encoder_outputs = self.encoder(
            token_embedding_output,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=False,
            type_ids=type_ids,
        )
        return select_hidden_state_embeddings(
            encoder_outputs.last_hidden_state,
            encoder_outputs.hidden_states,
            hidden_state_index=hidden_state_index,
            store_all_hidden_states=store_all_hidden_states,
        )

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
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if input_ids is not None:
            input_ids = _normalize_dplm2_input_ids(input_ids, self.config.vocab_size)
        if input_ids is None and inputs_embeds is None:
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

    def predict_contacts(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.esm.predict_contacts(input_ids, attention_mask)

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
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_s_max: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        type_ids: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPoolingAndCrossAttentions]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        direct_dplm_esm = getattr(self.config, "dplm_type", None) == "dplm_esm"
        if input_ids is not None:
            normalized_input_ids = _normalize_dplm2_input_ids(
                input_ids, self.config.vocab_size
            )
            if attention_mask is None:
                attention_mask = normalized_input_ids.ne(self.config.pad_token_id)
            if type_ids is None and not direct_dplm_esm:
                type_ids = _infer_modality_type(normalized_input_ids, attention_mask)
            input_ids = normalized_input_ids

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

        if not return_dict:
            return (sequence_output, pooled_output) + tuple(
                value
                for value in (outputs.hidden_states, outputs.attentions, outputs.s_max)
                if value is not None
            )

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DPLM2ForMaskedLM(FastPLMTestTimeTrainingMixin, DPLM2PreTrainedModel, EmbeddingMixin):
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
        max_iter: int | None = None,
        temperature: float = 1.0,
        partial_masks: torch.Tensor | None = None,
        unmasking_strategy: str = "stochastic1.0",
        sampling_strategy: str = "annealing@2.0:0.1",
        show_progress: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Generate packed sequence and structure tokens with DPLM2 diffusion.

        ``input_tokens`` is X with shape (b, l). Positions marked ``True`` in
        ``partial_masks`` remain fixed. The returned mapping contains
        ``output_tokens``, matching the official DPLM2 public API.
        """

        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected DPLM2 generation arguments: {names}")
        return generate_dplm2(
            self,
            input_tokens,
            max_iter=max_iter,
            temperature=temperature,
            partial_masks=partial_masks,
            unmasking_strategy=unmasking_strategy,
            sampling_strategy=sampling_strategy,
            show_progress=show_progress,
        )

    def predict_contacts(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the official ESM contact head output from the encoder."""
        return self.esm.predict_contacts(input_ids, attention_mask)

    def _get_modality_type(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        input_ids = _normalize_dplm2_input_ids(input_ids, self.config.vocab_size)
        return _infer_modality_type(input_ids, attention_mask)

    def _embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        hidden_state_index: int = -1,
        store_all_hidden_states: bool = False,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_id)
        type_ids = self._get_modality_type(input_ids, attention_mask)
        output_hidden_states = store_all_hidden_states or hidden_state_index != -1
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            type_ids=type_ids,
            output_attentions=False,
            output_hidden_states=output_hidden_states,
        )
        return select_hidden_state_embeddings(
            outputs.last_hidden_state,
            outputs.hidden_states,
            hidden_state_index=hidden_state_index,
            store_all_hidden_states=store_all_hidden_states,
        )

    def _ttt_get_trainable_modules(self) -> list[nn.Module]:
        return [self.esm]

    def _ttt_tokenize(
        self,
        seq: str | list[str] | None = None,
        input_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        if input_ids is not None:
            return input_ids
        assert seq is not None, "Pass either seq or input_ids for TTT."
        sequences = [seq] if isinstance(seq, str) else seq
        tokenized = self._tokenize_sequence_batch(
            sequences,
            return_tensors="pt",
            padding=True,
        )
        return tokenized["input_ids"]

    def _ttt_mask_token(self) -> int:
        return int(self.tokenizer._token_to_id[self.tokenizer.aa_mask_token])

    def _ttt_replacement_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        tokenizer = self.tokenizer
        special_ids = set(tokenizer.all_special_ids)
        struct_boundary = int(tokenizer._token_to_id[tokenizer.struct_cls_token])
        residue_ids = [
            token_id for token_id in range(struct_boundary) if token_id not in special_ids
        ]
        assert residue_ids, "DPLM2 TTT amino-acid replacement set is empty."
        return torch.tensor(residue_ids, device=input_ids.device, dtype=input_ids.dtype)

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
        direct_dplm_esm = getattr(self.config, "dplm_type", None) == "dplm_esm"

        if attention_mask is None:
            assert input_ids is not None
            attention_mask = input_ids.ne(self.pad_id)

        if type_ids is None and not direct_dplm_esm:
            assert input_ids is not None
            type_ids = self._get_modality_type(input_ids, attention_mask)

        if input_ids is not None:
            input_ids = _normalize_dplm2_input_ids(input_ids, self.config.vocab_size)
            if inputs_embeds is None and not direct_dplm_esm:
                # The official multimodal wrapper applies the embedding block
                # once before entering EsmForDPLM2. The inner ESM model then
                # applies it a second time using these intermediate embeddings.
                inputs_embeds = self.esm.embeddings(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

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


# Importing the DPLM2 model implementation makes its paired tokenizer visible
# to AutoTokenizer. This is registration only; it performs no I/O or downloads.
try:
    AutoTokenizer.register(
        DPLM2Config,
        tokenizer_class=DPLM2Tokenizer,
        exist_ok=True,
    )
except TypeError:
    # Transformers 4.x used this name; 5.x prefers tokenizer_class.
    AutoTokenizer.register(
        DPLM2Config,
        slow_tokenizer_class=DPLM2Tokenizer,
        exist_ok=True,
    )
