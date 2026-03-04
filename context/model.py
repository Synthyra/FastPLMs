import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput

from models.transformer.attention import (
    Attention,
    AttentionBackend,
    resolve_attention_backend,
)
from models.transformer.feedforward import FFN
from models.transformer.attention_utils import LAYER_NORM


@dataclass
class IntermediateOutput(ModelOutput):
    hidden_state: torch.FloatTensor | None = None
    attention_weights: torch.FloatTensor | None = None
    s_max: list[torch.Tensor] | None = None


@dataclass
class TransformerOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    s_max: tuple[list[torch.Tensor], ...] | None = None


def get_attention_mask(
    effective_backend: AttentionBackend,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, "BlockMask | None"]:
    """Build the padding mask once for all transformer layers.

    Args:
        effective_backend: Already-resolved backend (no AUTO). Determines mask format.
        attention_mask: (batch, seq_len) with 1 for valid tokens, 0 for padding.
                        If None, no masking is applied.

    Returns:
        (attention_mask_2d, attention_mask_4d, flex_block_mask).
        attention_mask_2d is always provided when attention_mask is not None.
        attention_mask_4d is provided for SDPA/manual backends.
        flex_block_mask is provided for the FLEX backend.
    """
    if attention_mask is None:
        return None, None, None

    attention_mask_2d = attention_mask.bool()

    if effective_backend == AttentionBackend.FLEX:
        assert create_block_mask is not None, (
            "Flex attention backend requested but torch.create_block_mask is unavailable."
        )
        valid_lens = attention_mask_2d.sum(dim=-1)

        def mask_mod(b, h, q_idx, kv_idx):  # type: ignore[no-untyped-def]
            return (q_idx < valid_lens[b]) & (kv_idx < valid_lens[b])

        flex_block_mask = create_block_mask(
            mask_mod,
            batch_size,
            1,
            seq_len,
            seq_len,
            device=device,
        )
        return attention_mask_2d, None, flex_block_mask
    else:
        # (batch, 1, q_len, kv_len): True where both query and key positions are valid.
        attention_mask_4d = attention_mask_2d[:, None, :, None] & attention_mask_2d[:, None, None, :]
        return attention_mask_2d, attention_mask_4d, None


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        if LAYER_NORM is None:
            return torch.nn.functional.rms_norm(
                hidden_states, (self.hidden_size,), self.weight, self.variance_epsilon
            ).to(input_dtype)
        else:
            return LAYER_NORM.rms_norm_fn(
                x=hidden_states,
                weight=self.weight,
                bias=None,  # no bias
                residual=None,
                eps=self.variance_epsilon,
                dropout_p=0.0,  # no dropout by default
                prenorm=False,
                residual_in_fp32=False,
            ).to(input_dtype)


class NormAttentionNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_2d: torch.Tensor | None = None,
        attention_mask_4d: torch.Tensor | None = None,
        flex_block_mask: "BlockMask | None" = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, s_max = self.self_attn(
            hidden_states=hidden_states,
            attention_mask_2d=attention_mask_2d,
            attention_mask_4d=attention_mask_4d,
            flex_block_mask=flex_block_mask,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return hidden_states, residual, self_attn_weights, s_max


class TransformerBlock(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.norm_attn_norm = NormAttentionNorm(config)
        self.ffn = FFN(config.hidden_size, config.expansion_ratio, config.dropout, config.bias, config.ffn_type)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_2d: torch.Tensor | None = None,
        attention_mask_4d: torch.Tensor | None = None,
        flex_block_mask: "BlockMask | None" = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_s_max: bool = False,
    ) -> TransformerOutput:
        hidden_states, residual, self_attn_weights, s_max = self.norm_attn_norm(
            hidden_states=hidden_states,
            attention_mask_2d=attention_mask_2d,
            attention_mask_4d=attention_mask_4d,
            flex_block_mask=flex_block_mask,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
        )
        hidden_states = residual + self.ffn(hidden_states)
        return IntermediateOutput(
            hidden_state=hidden_states,
            attention_weights=self_attn_weights,
            s_max=s_max,
        )


class TransformerStack(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(config, i) for i in range(config.num_layers)])
        # Resolve once so get_attention_mask always builds the right mask format.
        self.attention_backend = resolve_attention_backend(config.attn_backend)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_s_max: bool = False,
    ) -> TransformerOutput:
        full_hidden_states = () if output_hidden_states else None
        full_attentions = () if output_attentions else None
        full_s_max = () if output_s_max else None
        attention_mask_2d, attention_mask_4d, flex_block_mask = get_attention_mask(
            self.attention_backend,
            hidden_states.shape[0],
            hidden_states.shape[1],
            hidden_states.device,
            attention_mask=attention_mask,
        )
        for layer in self.layers:
            intermediate_output = layer(
                hidden_states=hidden_states,
                attention_mask_2d=attention_mask_2d,
                attention_mask_4d=attention_mask_4d,
                flex_block_mask=flex_block_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                output_s_max=output_s_max,
            )
            if output_hidden_states:
                full_hidden_states += (intermediate_output.hidden_state,)
            if output_attentions:
                full_attentions += (intermediate_output.attention_weights,)
            if output_s_max:
                full_s_max += (intermediate_output.s_max,)

        return TransformerOutput(
            last_hidden_state=hidden_states,
            hidden_states=full_hidden_states,
            attentions=full_attentions,
            s_max=full_s_max,
        )
