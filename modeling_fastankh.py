import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, T5Config
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from torch import Tensor
from einops import rearrange, repeat
from functools import partial
from typing import Optional, Tuple


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function as used in Google's BERT and OpenAI GPT.
    Reference: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * input.pow(3))))


class T5DenseGatedActDense(nn.Module):
    def __init__(self, d_model: int, intermediate_dim: int):
        super().__init__()
        self.wi_0 = nn.Linear(d_model, intermediate_dim, bias=False)
        self.wi_1 = nn.Linear(d_model, intermediate_dim, bias=False)
        self.wo = nn.Linear(intermediate_dim, d_model, bias=False)
        self.dropout = nn.Dropout(p=0.0)
        self.act = NewGELUActivation()

    def forward(self, hidden_states: Tensor) -> Tensor:
        x_gelu = self.act(self.wi_0(hidden_states))
        x_linear = self.wi_1(hidden_states)
        x = x_gelu * x_linear
        x = self.wo(x)
        return x


class RotaryEmbedding(nn.Module):
    """
    Rotary Embeddings for self-attention.
    Reference: https://arxiv.org/abs/2104.09864
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        interleaved: bool = False,
        scale_base: Optional[float] = None,
        scaling_factor: float = 1.0,
        pos_idx_in_fp32: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        self.interleaved = interleaved
        self.scale_base = scale_base
        self.scaling_factor = scaling_factor
        self.device = device

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None
        self.reset_parameters()

    def reset_parameters(self):
        inv_freq = self._compute_inv_freq(self.device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_inv_freq(self, device: Optional[torch.device] = None) -> torch.Tensor:
        return 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )

    def _update_cos_sin_cache(self, seqlen: int, device: Optional[torch.device], dtype: Optional[torch.dtype]):
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32) / self.scaling_factor
                inv_freq = self.inv_freq.float() if self.inv_freq.dtype != torch.float32 else self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype) / self.scaling_factor
                inv_freq = self.inv_freq

            freqs = torch.outer(t, inv_freq)
            self._cos_cached = torch.cos(freqs).to(dtype=dtype)
            self._sin_cached = torch.sin(freqs).to(dtype=dtype)

    def forward(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        self._update_cos_sin_cache(q.shape[1], device=q.device, dtype=q.dtype)
        return (
            apply_rotary_emb_torch(q, self._cos_cached, self._sin_cached, self.interleaved),
            apply_rotary_emb_torch(k, self._cos_cached, self._sin_cached, self.interleaved)
        )


def rotate_half(x: Tensor, interleaved: bool = False) -> Tensor:
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)


def apply_rotary_emb_torch(x: Tensor, cos: Tensor, sin: Tensor, interleaved: bool = False) -> Tensor:
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    seqlen = x.size(1)
    cos = cos[:seqlen]
    sin = sin[:seqlen]
    cos = repeat(cos, "s d -> s 1 (2 d)")
    sin = repeat(sin, "s d -> s 1 (2 d)")
    return torch.cat(
        [
            x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )


class T5Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_kv: Optional[int] = None):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_kv = d_kv if d_kv is not None else d_model // n_heads
        self.inner_dim = n_heads * self.d_kv

        # Linear projections
        self.q_ln = nn.LayerNorm(d_model, bias=False)
        self.k_ln = nn.LayerNorm(d_model, bias=False)
        self.q = nn.Linear(d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, d_model, bias=False)

        self.reshaper = partial(rearrange, pattern="b s (h d) -> b h s d", h=n_heads)
        self.rotary = RotaryEmbedding(self.d_kv)

    def _apply_rotary(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        q = q.unflatten(-1, (self.n_heads, self.d_kv))
        k = k.unflatten(-1, (self.n_heads, self.d_kv))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # Compute Q, K, V
        q = self.q(self.q_ln(hidden_states).to(hidden_states.dtype))
        k = self.k(self.k_ln(hidden_states).to(hidden_states.dtype))
        v = self.v(hidden_states)

        # Apply rotary embeddings
        q, k = self._apply_rotary(q, k)
        q, k, v = map(self.reshaper, (q, k, v))

        attn_weights = None
        if output_attentions:
            # Manual computation if we need attention weights
            attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_kv)
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)
        else:
            # Use PyTorch native scaled_dot_product_attention
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=0.0
            )

        attn_output = rearrange(attn_output, "b h s d -> b s (h d)")
        attn_output = self.o(attn_output)
        return attn_output, attn_weights


class T5LayerSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_kv: Optional[int] = None):
        super().__init__()
        self.SelfAttention = T5Attention(d_model, n_heads, d_kv=d_kv)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        attn_output, attn_weights = self.SelfAttention(hidden_states, attention_mask, output_attentions)
        hidden_states = hidden_states + attn_output
        return hidden_states, attn_weights


class T5LayerFF(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model, bias=False)
        self.DenseReluDense = T5DenseGatedActDense(d_model, d_ff)
        self.dropout = nn.Dropout(p=0.0)

    def forward(self, hidden_states: Tensor) -> Tensor:
        normed_states = self.layer_norm(hidden_states)
        ff_output = self.DenseReluDense(normed_states)
        hidden_states = hidden_states + ff_output
        return hidden_states


class T5Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, d_kv: Optional[int] = None):
        super().__init__()
        self.layer = nn.ModuleList([
            T5LayerSelfAttention(d_model, n_heads, d_kv=d_kv),
            T5LayerFF(d_model, d_ff)
        ])

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        hidden_states, attn_weights = self.layer[0](hidden_states, attention_mask, output_attentions)
        hidden_states = self.layer[1](hidden_states)
        return hidden_states, attn_weights


class T5Stack(nn.Module):
    def __init__(self, config, embed_tokens):
        super().__init__()
        self.config = config
        self.embed_tokens = embed_tokens
        self.block = nn.ModuleList([
            T5Block(config.hidden_size, config.num_attention_heads, config.d_ff, config.d_kv)
            for _ in range(config.num_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ):
        hidden_states = () if output_hidden_states else None
        attentions = () if output_attentions else None

        x = self.embed_tokens(input_ids)
        x = self.dropout(x)

        for block in self.block:
            if output_attentions:
                x, attn_weights = block(x, attention_mask, output_attentions)
                if attentions is not None:
                    attentions += (attn_weights,)
            else:
                x, _ = block(x, attention_mask, output_attentions)

            if output_hidden_states and hidden_states is not None:
                hidden_states += (x,)

        x = self.final_layer_norm(x)

        return FastankhOutput(
            last_hidden_state=x,
            hidden_states=hidden_states,
            attentions=attentions
        )


@dataclass
class FastankhOutput(ModelOutput):
    """
    Output type for Fastankh models.
    """
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None


class T5EncoderModel(PreTrainedModel):
    """
    T5-based encoder model for masked language modeling.
    """

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size)
        self.encoder = T5Stack(config, self.shared)
        self.sequence_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> FastankhOutput:
        """
        Forward pass for masked language modeling.

        Args:
            input_ids: Input token IDs [batch, seq_len].
            attention_mask: Attention mask [batch, seq_len].
            labels: Optional labels for masked tokens [batch, seq_len].
            output_hidden_states: Whether to return all hidden states.
            output_attentions: Whether to return attention weights.

        Returns:
            FastankhOutput: A dataclass containing the loss, logits, last hidden state, hidden states, and attentions.
        """
        encoder_outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )

        logits = self.sequence_head(encoder_outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss = self.ce_loss(logits.view(-1, self.vocab_size), labels.view(-1))

        return FastankhOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


if __name__ == "__main__":
    config = T5Config.from_pretrained("lhallee/ankh_large_encoder")
    model = T5EncoderModel.from_pretrained("lhallee/ankh_large_encoder", config=config)
    print(model)
    input_ids = torch.randint(0, 30, (1, 512))
    output = model(input_ids)
    print(output.last_hidden_state.shape)
