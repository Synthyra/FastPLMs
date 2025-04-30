import torch
from typing import Optional, Tuple
from torch.nn.attention.flex_attention import flex_attention


def flex_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    head_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    causal_mask = attention_mask
    if causal_mask is not None:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]

    def causal_mod(score, b, h, q_idx, kv_idx):
        if softcap is not None:
            score = softcap * torch.tanh(score / softcap)
        if causal_mask is not None:
            score = score + causal_mask[b][0][q_idx][kv_idx]
        if head_mask is not None:
            score = score + head_mask[b][h][0][0]
        return score

    attn_output, attention_weights = flex_attention(
        query,
        key,
        value,
        score_mod=causal_mod,
        enable_gqa=True,
        scale=scaling,
        # Last time checked on PyTorch == 2.5.1: Flex Attention always computes the lse regardless.
        # For simplification, we thus always return it as no additional computations are introduced.
        return_lse=True,
    )
    # lse is returned in float32
    attention_weights = attention_weights.to(value.dtype)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attention_weights


if __name__ == "__main__":
    # py -m wip.t5.transformers_flex_attn
    batch_size, seq_len, n_heads, head_dim = 1, 10, 12, 64

    query = torch.randn(batch_size, n_heads, seq_len, head_dim)
    key = torch.randn(batch_size, n_heads, seq_len, head_dim)
    value = torch.randn(batch_size, n_heads, seq_len, head_dim)
    attention_mask = torch.randint(0, 2, (batch_size, seq_len))
    scaling = 1.0
    softcap = 2.0
    head_mask = None

    attention_mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_len, seq_len).bool()

    attn_output, attention_weights = flex_attention_forward(
        query,
        key,
        value,
        attention_mask,
        scaling,
        softcap,
        head_mask,
    )
    print(attn_output.shape)
    print(attention_weights.shape)
