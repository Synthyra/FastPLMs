"""FastESMFold with FastESM2 attention.

Usage:
    from transformers import AutoModel
    model = AutoModel.from_pretrained("Synthyra/FastESMFold", trust_remote_code=True).cuda()

    # Basic folding, no TTT
    result = model.fold_protein("MKTLLILAVVA...")
    print(result["plddt"], result["pdb_string"][:100])

The runtime uses public Transformers folding components. It does not import or
depend on the pinned fair-esm or OpenFold parity repositories.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from einops import rearrange
from torch.nn import functional as F
from transformers.modeling_outputs import ModelOutput
from transformers.models.esm.configuration_esm import EsmConfig
from transformers.models.esm.modeling_esm import (
    EsmEmbeddings,
    EsmIntermediate,
    EsmOutput,
    EsmSelfOutput,
)
from transformers.models.esm.modeling_esmfold import (
    EsmForProteinFolding,
    collate_dense_tensors,
)
from transformers.models.esm.openfold_utils import residue_constants

from fastplms.models._esm_rotary import RotaryEmbedding


# Hub composite artifacts define these shared names earlier in the assembled file.
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
except ModuleNotFoundError as error:
    _COMPOSITE_REQUIRED_NAMES = (
        "AttentionBackend",
        "BlockMask",
        "FastPLMsAttentionMixin",
        "_get_flex_attention_fn",
        "flex_attention",
        "get_attention_mask",
        "kernels_flash_attention_func",
        "resolve_attention_backend",
        "resolve_attention_backend_for_call",
    )
    if error.name != "fastplms" or any(
        name not in globals() for name in _COMPOSITE_REQUIRED_NAMES
    ):
        raise
    # Legacy flat Hub composites define every shared symbol above this block.


@dataclass
class FastEsmEncoderOutput(ModelOutput):
    last_hidden_state: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


@dataclass
class FastEsmForProteinFoldingOutput(ModelOutput):
    """Folding output with a standard Transformers AutoModel prefix."""

    last_hidden_state: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None
    frames: torch.Tensor | None = None
    sidechain_frames: torch.Tensor | None = None
    unnormalized_angles: torch.Tensor | None = None
    angles: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    states: torch.Tensor | None = None
    s_s: torch.Tensor | None = None
    s_z: torch.Tensor | None = None
    distogram_logits: torch.Tensor | None = None
    lm_logits: torch.Tensor | None = None
    aatype: torch.Tensor | None = None
    atom14_atom_exists: torch.Tensor | None = None
    residx_atom14_to_atom37: torch.Tensor | None = None
    residx_atom37_to_atom14: torch.Tensor | None = None
    atom37_atom_exists: torch.Tensor | None = None
    residue_index: torch.Tensor | None = None
    lddt_head: torch.Tensor | None = None
    plddt: torch.Tensor | None = None
    ptm_logits: torch.Tensor | None = None
    ptm: torch.Tensor | None = None
    aligned_confidence_probs: torch.Tensor | None = None
    predicted_aligned_error: torch.Tensor | None = None
    max_predicted_aligned_error: torch.Tensor | None = None
    mlm_targets: torch.Tensor | None = None


# ``EsmForProteinFolding.forward`` calls
# ``compute_language_model_representations`` without forwarding output controls.
# Context variables bridge that private call boundary without mutating the model
# instance, so concurrent calls can independently request attention tensors.
_ESMFOLD_OUTPUT_ATTENTIONS: ContextVar[bool] = ContextVar(
    "fastplms_esmfold_output_attentions",
    default=False,
)
_ESMFOLD_CAPTURED_ATTENTIONS: ContextVar[
    tuple[torch.Tensor, ...] | None
] = ContextVar(
    "fastplms_esmfold_captured_attentions",
    default=None,
)


def _align_internal_esm_attentions(
    attentions: tuple[torch.Tensor, ...],
    residue_mask: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Remove internal BOS/EOS positions while preserving public padding slots."""

    # attention: (b, h, l + 2, l + 2); residue_mask: (b, l)
    batch_size, sequence_length = residue_mask.shape
    public_positions = torch.arange(sequence_length, device=residue_mask.device)  # (l,)
    valid_lengths = residue_mask.to(dtype=torch.int64).sum(dim=-1, keepdim=True)  # (b, 1)
    # Internal layout is BOS, compact biological residues, EOS, then padding.
    # Public padding positions therefore advance by two rather than one.
    internal_positions = public_positions.unsqueeze(0) + 1  # (1, l)
    internal_positions = internal_positions + (
        public_positions.unsqueeze(0) >= valid_lengths
    ).to(dtype=torch.int64)

    aligned: list[torch.Tensor] = []
    for attention in attentions:
        if attention.shape[0] != batch_size or attention.shape[-2:] != (
            sequence_length + 2,
            sequence_length + 2,
        ):
            raise RuntimeError(
                "FastESM attention shape does not match the folding input: "
                f"got {tuple(attention.shape)} for residue mask "
                f"{tuple(residue_mask.shape)}."
            )
        query_index = internal_positions[:, None, :, None].expand(
            batch_size,
            attention.shape[1],
            sequence_length,
            attention.shape[-1],
        )
        query_aligned = torch.gather(attention, dim=2, index=query_index)
        key_index = internal_positions[:, None, None, :].expand(
            batch_size,
            attention.shape[1],
            sequence_length,
            sequence_length,
        )
        aligned.append(torch.gather(query_aligned, dim=3, index=key_index))
    return tuple(aligned)


class EsmSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type: str | None = None) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})."
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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # hidden_states: (b, l, d)
        batch_size, seq_length = hidden_states.shape[:-1]
        hidden_shape = (batch_size, seq_length, -1, self.attention_head_size)
        query_heads = self.query(hidden_states).view(hidden_shape).transpose(1, 2)  # (b, h, l, d_h)
        key_heads = self.key(hidden_states).view(hidden_shape).transpose(1, 2)  # (b, h, l, d_h)
        value_heads = self.value(hidden_states).view(hidden_shape).transpose(1, 2)  # (b, h, l, d_h)

        query_heads = query_heads * self.scale

        if self.position_embedding_type == "rotary":
            query_heads, key_heads = self.rotary_embeddings(query_heads, key_heads)

        attn_output, attn_weights = self._attn(
            query_heads,
            key_heads,
            value_heads,
            attention_mask_2d=attention_mask_2d,
            attention_mask_4d=attention_mask_4d,
            flex_block_mask=flex_block_mask,
            output_attentions=output_attentions,
        )
        return attn_output, attn_weights

    def _attn(
        self,
        query_heads: torch.Tensor,
        key_heads: torch.Tensor,
        value_heads: torch.Tensor,
        attention_mask_2d: torch.Tensor | None = None,
        attention_mask_4d: torch.Tensor | None = None,
        flex_block_mask: BlockMask | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if output_attentions:
            return self._manual_attn(query_heads, key_heads, value_heads, attention_mask_4d)

        if (
            self.training
            and self.dropout_prob > 0
            and self.attn_backend == AttentionBackend.FLEX_ATTENTION
        ):
            raise RuntimeError(
                "ESMFold flex_attention is inference-only when attention dropout is "
                "nonzero. Use eager or SDPA for this training configuration."
            )

        if self.attn_backend == AttentionBackend.EAGER:
            attn_output, _ = self._manual_attn(
                query_heads, key_heads, value_heads, attention_mask_4d
            )
            return attn_output, None
        if self.attn_backend.is_flash:
            return self._kernels_flash_attn(query_heads, key_heads, value_heads, attention_mask_2d)
        elif self.attn_backend == AttentionBackend.FLEX:
            return self._flex_attn(
                query_heads,
                key_heads,
                value_heads,
                flex_block_mask,
                attention_mask_2d,
            )
        elif self.attn_backend == AttentionBackend.SDPA:
            return self._sdpa_attn(query_heads, key_heads, value_heads, attention_mask_4d)
        else:
            raise AssertionError(f"Unsupported resolved backend: {self.attn_backend}")

    def _manual_attn(
        self,
        query_heads: torch.Tensor,
        key_heads: torch.Tensor,
        value_heads: torch.Tensor,
        attention_mask_4d: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # query_heads, key_heads, value_heads: (b, h, l, d_h)
        attn_weights = torch.matmul(
            query_heads, key_heads.transpose(-1, -2)
        )  # (b, h, l, l)
        if attention_mask_4d is not None:
            attn_weights = attn_weights.masked_fill(attention_mask_4d.logical_not(), float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.dropout_prob > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout_prob, training=self.training)
        context_heads = torch.matmul(attn_weights, value_heads)  # (b, h, l, d_h)
        attn_output = rearrange(context_heads, "b h s d -> b s (h d)")  # (b, l, d)
        return attn_output, attn_weights

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
        # Q is pre-scaled by self.scale in forward() -- pass softmax_scale=1.0
        # to prevent the kernel from applying its default 1/sqrt(head_dim).
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
    def __init__(self, config) -> None:
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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden_states_ln = self.LayerNorm(hidden_states)
        attn_output, attn_weights = self.self(
            hidden_states_ln,
            attention_mask_2d=attention_mask_2d,
            attention_mask_4d=attention_mask_4d,
            flex_block_mask=flex_block_mask,
            output_attentions=output_attentions,
        )
        attention_output = self.output(attn_output, hidden_states)
        return attention_output, attn_weights


class EsmLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attention_output, attn_weights = self.attention(
            hidden_states,
            attention_mask_2d=attention_mask_2d,
            attention_mask_4d=attention_mask_4d,
            flex_block_mask=flex_block_mask,
            output_attentions=output_attentions,
        )
        layer_output = self._feed_forward(attention_output)
        return layer_output, attn_weights

    def _feed_forward(self, attention_output: torch.Tensor) -> torch.Tensor:
        attention_output_ln = self.LayerNorm(attention_output)
        intermediate_output = self.intermediate(attention_output_ln)
        return self.output(intermediate_output, attention_output)


class FastEsmEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.attention_backend = resolve_attention_backend(config.attn_backend)
        self.layer = nn.ModuleList([EsmLayer(config) for _ in range(config.num_hidden_layers)])
        self.emb_layer_norm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> FastEsmEncoderOutput:
        # hidden_states: (b, l, d); attention_mask: (b, l)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

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

            hidden_states, attn_weights = layer_module(
                hidden_states,
                attention_mask_2d=attention_mask_2d,
                attention_mask_4d=attention_mask_4d,
                flex_block_mask=flex_block_mask,
                output_attentions=output_attentions,
            )

            if all_attentions is not None:
                all_attentions = (*all_attentions, attn_weights)

        if self.emb_layer_norm_after:
            hidden_states = self.emb_layer_norm_after(hidden_states)

        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        return FastEsmEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class FastEsmBackbone(nn.Module):
    """FastESM2 backbone with multi-backend attention. Drop-in replacement for
    transformers.EsmModel inside EsmForProteinFolding.

    Folding uses hidden states only. The standalone ESM2 contact regressor and
    masked-LM head are therefore omitted from this structure-only backbone.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embeddings = EsmEmbeddings(config)
        self.encoder = FastEsmEncoder(config)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> FastEsmEncoderOutput | tuple[Any, ...]:
        output_attentions = (
            self.config.output_attentions
            if output_attentions is None
            else output_attentions
        )
        output_hidden_states = (
            self.config.output_hidden_states
            if output_hidden_states is None
            else output_hidden_states
        )
        return_dict = self.config.use_return_dict if return_dict is None else return_dict

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
        )
        output = FastEsmEncoderOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        return output if return_dict else output.to_tuple()


class FastEsmFoldConfig(EsmConfig):
    model_type = "fast_esmfold"

    def __init__(self, attn_backend: str | None = None, **kwargs: Any) -> None:
        # Earlier mirrors serialized an untrained ESMFold-specific TTT policy.
        # It is intentionally ignored because the official checkpoint has no
        # trained masked-language-model head.
        kwargs.pop("ttt_config", None)
        super().__init__(**kwargs)
        self.attn_backend = attn_backend


class FastEsmForProteinFolding(FastPLMsAttentionMixin, EsmForProteinFolding):
    """ESMFold with FastESM2 attention backends.

    Inherits all folding logic (trunk, structure module, output_to_pdb, infer)
    from transformers.EsmForProteinFolding. Replaces the ESM2 backbone with
    FastESM2 for selectable attention implementations.

    Key API:
        result = model.fold_protein("MKTL...")
        # result = {"plddt": float, "ptm": float, "pdb_string": str}
    """

    config_class = FastEsmFoldConfig
    _supports_flash_attn_2 = False
    _supports_flash_attn_3 = False
    _fastplms_attention_implementations = ("eager", "sdpa", "flex_attention")

    def __init__(self, config: FastEsmFoldConfig) -> None:
        super().__init__(config)

        # Replace the standard ESM2 backbone with the multi-backend FastESM2
        # implementation while retaining the canonical checkpoint key schema.
        self.esm = FastEsmBackbone(config)
        self.esm.requires_grad_(False)
        if config.esmfold_config.fp16_esm:
            self.esm.half()

    def compute_language_model_representations(
        self,
        esmaa: torch.Tensor,
    ) -> torch.Tensor:
        """Run the internal ESM stem with a structured output unconditionally.

        The outer folding model still honors ``config.return_dict``. This
        internal call must remain structured because the folding stem selects
        hidden states by name before constructing the public output.
        """

        device = next(self.parameters()).device
        batch_size, sequence_length = esmaa.shape
        output_attentions = _ESMFOLD_OUTPUT_ATTENTIONS.get()
        if self.config.esmfold_config.bypass_lm:
            if output_attentions:
                _ESMFOLD_CAPTURED_ATTENTIONS.set(())
            return torch.zeros(
                batch_size,
                sequence_length,
                self.esm_s_combine.size(0),
                self.esm_feats,
                device=device,
            )

        bos = esmaa.new_full((batch_size, 1), self.esm_dict_cls_idx)
        eos = esmaa.new_full((batch_size, 1), self.esm_dict_padding_idx)
        residue_mask = esmaa != self.esm_dict_padding_idx
        with_special_tokens = torch.cat([bos, esmaa, eos], dim=1)
        with_special_tokens[
            range(batch_size),
            (with_special_tokens != self.esm_dict_padding_idx).sum(1),
        ] = self.esm_dict_eos_idx
        esm_output = self.esm(
            with_special_tokens,
            attention_mask=with_special_tokens != self.esm_dict_padding_idx,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )
        if not isinstance(esm_output, FastEsmEncoderOutput):
            raise TypeError("FastESM internal backbone did not return FastEsmEncoderOutput.")
        if esm_output.hidden_states is None:
            raise RuntimeError("FastESM internal backbone omitted requested hidden states.")
        if output_attentions:
            if esm_output.attentions is None:
                raise RuntimeError("FastESM internal backbone omitted requested attentions.")
            _ESMFOLD_CAPTURED_ATTENTIONS.set(
                _align_internal_esm_attentions(esm_output.attentions, residue_mask)
            )
        return torch.stack(esm_output.hidden_states, dim=2)[:, 1:-1]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        masking_pattern: torch.Tensor | None = None,
        num_recycles: int | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> FastEsmForProteinFoldingOutput | tuple[Any, ...]:
        """Run folding with Meta ESMFold's 0-to-100 pLDDT convention."""

        config = getattr(self, "config", None)
        resolved_attentions = (
            bool(getattr(config, "output_attentions", False))
            if output_attentions is None
            else output_attentions
        )
        resolved_hidden_states = (
            bool(getattr(config, "output_hidden_states", False))
            if output_hidden_states is None
            else output_hidden_states
        )
        resolved_return_dict = (
            bool(getattr(config, "use_return_dict", True))
            if return_dict is None
            else return_dict
        )

        request_token = _ESMFOLD_OUTPUT_ATTENTIONS.set(bool(resolved_attentions))
        capture_token = _ESMFOLD_CAPTURED_ATTENTIONS.set(None)
        captured_attentions: tuple[torch.Tensor, ...] | None = None
        try:
            output = super().forward(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                masking_pattern=masking_pattern,
                num_recycles=num_recycles,
                output_hidden_states=resolved_hidden_states,
            )
            captured_attentions = _ESMFOLD_CAPTURED_ATTENTIONS.get()
        finally:
            _ESMFOLD_CAPTURED_ATTENTIONS.reset(capture_token)
            _ESMFOLD_OUTPUT_ATTENTIONS.reset(request_token)
        # Transformers 5.13 returns categorical lDDT probabilities on [0, 1],
        # while Meta ESMFold's public forward output reports pLDDT on [0, 100].
        output["plddt"] = output["plddt"] * 100
        payload = dict(output)
        payload.pop("last_hidden_state", None)
        payload.pop("hidden_states", None)
        parent_attentions = payload.pop("attentions", None)
        if captured_attentions is None:
            captured_attentions = parent_attentions
        if resolved_attentions and captured_attentions is None:
            # A bypassed or injected folding stem has no attention layers, but
            # still honors the output contract without fabricating tensors.
            captured_attentions = ()
        sequence_state = payload.get("s_s")
        structured = FastEsmForProteinFoldingOutput(
            last_hidden_state=sequence_state,
            hidden_states=(sequence_state,)
            if resolved_hidden_states and sequence_state is not None
            else None,
            attentions=captured_attentions if resolved_attentions else None,
            **payload,
        )
        return structured if resolved_return_dict else structured.to_tuple()

    @torch.no_grad()
    def infer(
        self,
        sequences: str | list[str],
        residx: torch.Tensor | list[torch.Tensor] | None = None,
        masking_pattern: torch.Tensor | None = None,
        num_recycles: int | None = None,
        residue_index_offset: int | None = 512,
        chain_linker: str | None = "G" * 25,
    ):
        """Fold raw sequences through Meta ESMFold's public input contract.

        Transformers v5 narrows ``infer`` even though ``forward`` retains the
        required controls. This adapter restores recycle selection, explicit
        residue indices, masking, and colon-delimited multimer preparation.
        """

        sequence_batch = [sequences] if isinstance(sequences, str) else sequences
        linker = "" if chain_linker is None else chain_linker
        index_offset = 0 if residue_index_offset is None else residue_index_offset
        unknown_index = residue_constants.restype_order_with_x["X"]
        aatype_batch: list[torch.Tensor] = []
        residx_batch: list[torch.Tensor] = []
        linker_mask_batch: list[torch.Tensor] = []
        chain_index_batch: list[torch.Tensor] = []

        for sequence in sequence_batch:
            chains = sequence.split(":")
            joined_sequence = linker.join(chains)
            encoded = torch.tensor(
                [
                    residue_constants.restype_order_with_x.get(residue, unknown_index)
                    for residue in joined_sequence
                ],
                dtype=torch.int64,
            )
            sequence_residx = torch.arange(len(encoded), dtype=torch.int64)
            cursor = 0
            for chain_number, chain in enumerate(chains):
                segment_length = len(chain) + len(linker)
                sequence_residx[cursor : cursor + segment_length] += chain_number * index_offset
                cursor += segment_length

            linker_mask = torch.ones_like(encoded, dtype=torch.float32)
            chain_indices: list[int] = []
            cursor = 0
            for chain_number, chain in enumerate(chains):
                if chain_number > 0:
                    chain_indices.extend([chain_number - 1] * len(linker))
                chain_indices.extend([chain_number] * len(chain))
                cursor += len(chain)
                linker_mask[cursor : cursor + len(linker)] = 0
                cursor += len(linker)

            aatype_batch.append(encoded)
            residx_batch.append(sequence_residx)
            linker_mask_batch.append(linker_mask)
            chain_index_batch.append(torch.tensor(chain_indices, dtype=torch.int64))

        aatype = collate_dense_tensors(aatype_batch)
        attention_mask = collate_dense_tensors(
            [aatype.new_ones(len(encoded)) for encoded in aatype_batch]
        )
        prepared_residx = collate_dense_tensors(residx_batch)
        linker_mask = collate_dense_tensors(linker_mask_batch)
        chain_index = collate_dense_tensors(chain_index_batch, pad_v=-1)
        if residx is None:
            residx = prepared_residx
        elif not isinstance(residx, torch.Tensor):
            residx = collate_dense_tensors(residx)

        device = next(self.parameters()).device
        aatype = aatype.to(device)
        attention_mask = attention_mask.to(device)
        residx = residx.to(device)
        linker_mask = linker_mask.to(device)
        output = self.forward(
            aatype,
            attention_mask,
            position_ids=residx,
            masking_pattern=masking_pattern,
            num_recycles=num_recycles,
        )
        output["atom37_atom_exists"] = output["atom37_atom_exists"] * linker_mask.unsqueeze(2)
        output["mean_plddt"] = (output["plddt"] * output["atom37_atom_exists"]).sum(
            dim=(1, 2)
        ) / output["atom37_atom_exists"].sum(dim=(1, 2))
        output["chain_index"] = chain_index
        return output

    @staticmethod
    def _ttt_unavailable() -> None:
        raise RuntimeError(
            "ESMFold TTT is unavailable: the pinned Meta ESMFold checkpoint does "
            "not contain a trained masked-language-model head. FastPLMs does not "
            "construct or serialize a random replacement head."
        )

    def ttt(self, seq: str, **kwargs: Any) -> None:
        """Reject ESMFold-specific TTT because no faithful MLM objective exists."""

        del seq, kwargs
        self._ttt_unavailable()

    def ttt_reset(self) -> None:
        """Reject reset because ESMFold does not expose a faithful TTT path."""

        self._ttt_unavailable()

    def _fold_single(self, sequence: str, return_pdb_string: bool = True) -> dict[str, Any]:
        """Fold a sequence once and return pLDDT, ptm, and optionally PDB string."""
        with torch.no_grad():
            output = self.infer(sequence)
        if "mean_plddt" in output:
            # ``infer`` masks multimer linker atoms before computing this mean.
            # Reusing it prevents synthetic linker confidence from affecting the
            # public fold_protein summary.
            mean_plddt = float(output["mean_plddt"].reshape(-1)[0].item())
        else:
            plddt = output["plddt"]
            # P has shape (b, l, 37), with confidence for each atom37 position.
            # Use CA atom (index 1) only, matching PDB B-factor output.
            if plddt.dim() == 3:
                mean_plddt = float(plddt[:, :, 1].mean().item())
            elif plddt.dim() == 2:
                mean_plddt = float(plddt[:, 1].mean().item())
            else:
                mean_plddt = float(plddt.mean().item())
        result = {
            "plddt": mean_plddt,
            "ptm": float(output["ptm"].item()) if "ptm" in output else None,
        }
        if return_pdb_string:
            pdb_strings = self.output_to_pdb(output)
            result["pdb_string"] = pdb_strings[0] if isinstance(pdb_strings, list) else pdb_strings
        return result

    def fold_protein(
        self,
        sequence: str,
        return_pdb_string: bool = True,
        ttt: bool = False,
    ) -> dict[str, Any]:
        """Fold a protein sequence.

        Passing ``ttt=True`` fails explicitly because the official Meta
        checkpoint contains no trained masked-language-model head.

        Args:
            sequence: Protein sequence (single-letter amino acid codes)
            return_pdb_string: If True, include PDB string in output
            ttt: Reserved rejection flag for unsupported ESMFold TTT

        Returns:
            Dict with keys:
                - plddt: float, mean pLDDT
                - ptm: float, predicted TM-score
                - pdb_string: str (if return_pdb_string=True), PDB from best step
                - step_plddts: list[float], baseline pLDDT when TTT is disabled
                - best_step: int, 0 when TTT is disabled
        """
        if ttt:
            return self.fold_protein_ttt(
                sequence=sequence,
                return_pdb_string=return_pdb_string,
            )
        result = self._fold_single(sequence, return_pdb_string=return_pdb_string)
        return {
            "plddt": result["plddt"],
            "ptm": result["ptm"],
            "pdb_string": result.get("pdb_string"),
            "step_plddts": [result["plddt"]],
            "best_step": 0,
        }

    def fold_protein_ttt(
        self,
        sequence: str,
        return_pdb_string: bool = True,
    ) -> None:
        """Reject ESMFold TTT because the official checkpoint has no MLM head."""

        del sequence, return_pdb_string
        self._ttt_unavailable()
