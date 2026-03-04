import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask

from .rotary import RotaryEmbedding
from .attention_utils import (
    resolve_attention_backend,
    AttentionBackend,
    kernels_flash_attention_func,
    flex_attention_func,
    sdpa_attention_func,
    _repeat_kv,
)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.clip_qkv = config.clip_qkv
        self.scale = 1.0 / (self.head_dim**0.5)

        assert (self.head_dim * self.num_heads) == self.hidden_size, (
            f"hidden_size must be divisible by num_heads "
            f"(got hidden_size={self.hidden_size}, num_heads={self.num_heads})."
        )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.bias)
        self.rotary_emb = RotaryEmbedding(dim=self.head_dim, base=config.rope_theta)
        self.attn_backend = resolve_attention_backend(config.attn_backend)

    def prepare_qkv(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, q_len, _ = hidden_states.size()
        query_states: torch.Tensor = self.q_proj(hidden_states)
        key_states: torch.Tensor = self.k_proj(hidden_states)
        val_states: torch.Tensor = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim)
        val_states = val_states.view(bsz, q_len, self.num_kv_heads, self.head_dim)

        if self.clip_qkv is not None:
            query_states = query_states.clamp(-self.clip_qkv, self.clip_qkv)
            key_states = key_states.clamp(-self.clip_qkv, self.clip_qkv)
            val_states = val_states.clamp(-self.clip_qkv, self.clip_qkv)

        query_states, key_states = self.rotary_emb(query_states, key_states)
        return query_states, key_states, val_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_2d: torch.Tensor | None = None,  # (bs, seq_len) bool
        attention_mask_4d: torch.Tensor | None = None,  # (bs, 1, q_len, kv_len) bool
        flex_block_mask: BlockMask | None = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        query_states, key_states, val_states = self.prepare_qkv(hidden_states=hidden_states)

        attn_output, attn_weights, s_max = self._attn(
            query_states=query_states,
            key_states=key_states,
            val_states=val_states,
            attention_mask_2d=attention_mask_2d,
            attention_mask_4d=attention_mask_4d,
            flex_block_mask=flex_block_mask,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
        )

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, s_max

    def _attn(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        val_states: torch.Tensor,
        attention_mask_2d: torch.Tensor | None = None,
        attention_mask_4d: torch.Tensor | None = None,
        flex_block_mask: BlockMask | None = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        if output_attentions:
            self.last_backend_used = "manual"
            return self._manual_attn(
                query_states=query_states,
                key_states=key_states,
                val_states=val_states,
                attention_mask_4d=attention_mask_4d,
                output_s_max=output_s_max,
            )

        if self.attn_backend == AttentionBackend.KERNELS_FLASH:
            attn_output, attn_weights = self._kernels_flash_attn(
                query_states=query_states,
                key_states=key_states,
                val_states=val_states,
                attention_mask_2d=attention_mask_2d,
            )
        elif self.attn_backend == AttentionBackend.FLEX:
            attn_output, attn_weights = self._flex_attn(
                query_states=query_states,
                key_states=key_states,
                val_states=val_states,
                flex_block_mask=flex_block_mask,
            )
        elif self.attn_backend == AttentionBackend.SDPA:
            attn_output, attn_weights = self._sdpa_attn(
                query_states=query_states,
                key_states=key_states,
                val_states=val_states,
                attention_mask_4d=attention_mask_4d,
            )
        else:
            raise AssertionError(f"Unsupported resolved backend: {self.attn_backend}")

        s_max = self._compute_s_max(query_states, key_states) if output_s_max else None
        return attn_output, attn_weights, s_max

    @torch.no_grad()
    def _compute_s_max(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
    ) -> list[torch.Tensor]:
        query_BHLD = query_states.transpose(1, 2).contiguous()
        key_BHLD = key_states.transpose(1, 2).contiguous()
        key_BHLD = _repeat_kv(key_BHLD, self.num_key_value_groups)
        q_norm = torch.linalg.vector_norm(query_BHLD, dim=-1)
        k_norm = torch.linalg.vector_norm(key_BHLD, dim=-1)
        s_max_bound = (q_norm.max(dim=-1).values * k_norm.max(dim=-1).values).max(dim=0).values * self.scale
        return [s_max_bound[h] for h in range(self.num_heads)]

    def _manual_attn(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        val_states: torch.Tensor,
        attention_mask_4d: torch.Tensor | None = None,
        output_s_max: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        query_BHLD = query_states.transpose(1, 2).contiguous()
        key_BHLD = key_states.transpose(1, 2).contiguous()
        val_BHLD = val_states.transpose(1, 2).contiguous()
        key_BHLD = _repeat_kv(key_BHLD, self.num_key_value_groups)
        val_BHLD = _repeat_kv(val_BHLD, self.num_key_value_groups)
        attn_weights = torch.matmul(query_BHLD, key_BHLD.transpose(-2, -1)) * self.scale
        if attention_mask_4d is not None:
            attn_weights = attn_weights.masked_fill(attention_mask_4d.logical_not(), float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        context_BHLD = torch.matmul(attn_weights, val_BHLD)
        attn_output = context_BHLD.transpose(1, 2).reshape(
            query_states.shape[0], query_states.shape[1], self.hidden_size
        ).contiguous()
        s_max = self._compute_s_max(query_states, key_states) if output_s_max else None
        return attn_output, attn_weights, s_max

    def _kernels_flash_attn(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        val_states: torch.Tensor,
        attention_mask_2d: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, q_len = query_states.shape[:2]
        attn_output = kernels_flash_attention_func(
            query_states=query_states,
            key_states=key_states,
            value_states=val_states,
            attention_mask_2d=attention_mask_2d,
            causal=False,
        )
        return attn_output.reshape(bsz, q_len, self.hidden_size).contiguous(), None

    def _flex_attn(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        val_states: torch.Tensor,
        flex_block_mask: BlockMask | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, q_len = query_states.shape[:2]
        outputs = flex_attention_func(query_states, key_states, val_states, block_mask=flex_block_mask)
        return outputs.reshape(bsz, q_len, self.hidden_size).contiguous(), None

    def _sdpa_attn(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        val_states: torch.Tensor,
        attention_mask_4d: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, q_len = query_states.shape[:2]
        attn_output = sdpa_attention_func(
            query_states=query_states,
            key_states=key_states,
            value_states=val_states,
            attention_mask_4d=attention_mask_4d,
            num_key_value_groups=self.num_key_value_groups,
        )
        return attn_output.reshape(bsz, q_len, self.hidden_size).contiguous(), None
