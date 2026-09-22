"""Design probe: what would running an encoder on packed tokens buy, and at what startup cost?

This measures execution strategies on a generic pre-norm rotary encoder with
ESM2-150M and ESM2-650M dimensions. It is not a FastPLMs model and supports no
claim about one; it sizes the opportunity before any runtime change.

Strategies, all BF16 autocast over FP32 weights, inference only:

- ``sdpa_padded``: padded batch, SDPA with a key-padding mask (today's default).
- ``varlen_per_layer``: padded hidden states, gather and scatter around PyTorch's
  variable-length attention in every layer (the shape of today's Flash path).
- ``varlen_packed``: unpad once, run every layer on packed tokens, pad once.
- ``flex_packed``: packed tokens in one row with a document block mask, through
  compiled FlexAttention. Reported with its compile time and per-batch mask cost.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import torch

from collections.abc import Callable
from typing import cast
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.nn.attention.varlen import varlen_attn


# Attention over rotated Q, K and V, each (..., tokens, h, d_h); returns the same shape.
Attend = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

ENCODERS = {"esm2_150m_shape": (30, 640, 20), "esm2_650m_shape": (33, 1280, 20)}
BATCH_SIZE = 16
MAX_LENGTH = 1024
# Packed totals are rounded up to this, so compiled FlexAttention sees few distinct shapes.
FLEX_TOTAL_MULTIPLE = 1024
ROUNDS = 3
BATCHES_PER_ROUND = 8


def natural_lengths(generator: random.Random) -> list[int]:
    """Protein-like lengths: log-normal with a median near 300 residues."""
    return [
        min(MAX_LENGTH, max(40, int(generator.lognormvariate(5.7, 0.6)))) for _ in range(BATCH_SIZE)
    ]


def rotate(states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # states: (..., tokens, h, d_h); cos/sin: (..., tokens, 1, d_h).
    first, second = states.chunk(2, dim=-1)  # each (..., tokens, h, d_h / 2)
    return states * cos + torch.cat((-second, first), dim=-1) * sin  # (..., tokens, h, d_h)


class Layer(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.attention_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.ffn_norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, 4 * width)
        self.down = nn.Linear(4 * width, width)

    def forward(
        self, hidden: torch.Tensor, attend: Attend, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        # hidden: (..., tokens, width); cos, sin broadcast over heads: (..., tokens, 1, d_h)
        qkv = self.qkv(self.attention_norm(hidden))  # (..., tokens, 3 * width)
        Q, K, V = qkv.view(*hidden.shape[:-1], 3, self.heads, -1).unbind(-3)  # each (..., tokens, h, d_h)
        attended = attend(rotate(Q, cos, sin), rotate(K, cos, sin), V)  # (..., tokens, h, d_h)
        hidden = hidden + self.out(attended.reshape(hidden.shape))  # (..., tokens, width)
        updated: torch.Tensor = hidden + self.down(F.gelu(self.up(self.ffn_norm(hidden))))  # (..., tokens, width)
        return updated  # (..., tokens, width)


class Encoder(nn.Module):
    cos: torch.Tensor  # (l_max, 1, d_h)
    sin: torch.Tensor  # (l_max, 1, d_h)

    def __init__(self, layers: int, width: int, heads: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(Layer(width, heads) for _ in range(layers))
        inverse = 1.0 / (10000 ** (torch.arange(0, width // heads, 2).float() / (width // heads)))  # (d_h / 2,)
        angles = torch.outer(torch.arange(MAX_LENGTH).float(), inverse)  # (l_max, d_h / 2)
        self.register_buffer("cos", torch.cat((angles, angles), -1).cos()[:, None])  # (l_max, 1, d_h)
        self.register_buffer("sin", torch.cat((angles, angles), -1).sin()[:, None])  # (l_max, 1, d_h)

    def run(self, hidden: torch.Tensor, attend: Attend, positions: torch.Tensor) -> torch.Tensor:
        # hidden: (..., tokens, width); positions: (..., tokens).
        # BF16 tables keep Q and K in the dtype the variable-length kernel accepts.
        cos = self.cos[positions].bfloat16()  # (..., tokens, 1, d_h)
        sin = self.sin[positions].bfloat16()  # (..., tokens, 1, d_h)
        for layer in self.layers:
            hidden = layer(hidden, attend, cos, sin)  # (..., tokens, width)
        return hidden  # (..., tokens, width)


class Batch:
    """One padded batch and the packing metadata derived from it once."""

    def __init__(self, lengths: list[int], width: int, device: torch.device) -> None:
        self.lengths = lengths
        self.longest = max(lengths)
        # b proteins; l longest length; t real tokens; d channels.
        length_column = torch.tensor(lengths, device=device)[:, None]  # (b, 1)
        self.mask = torch.arange(self.longest, device=device)[None] < length_column  # (b, l)
        self.hidden = torch.randn(len(lengths), self.longest, width, device=device)  # (b, l, d)
        self.indices = self.mask.flatten().nonzero().flatten()  # (t,)
        self.positions = self.indices % self.longest  # (t,) column of each real token
        self.cu_seqlens = F.pad(length_column.flatten().cumsum(0), (1, 0)).int()  # (b + 1,)
        self.total = sum(lengths)


def sdpa_padded(encoder: Encoder, batch: Batch) -> torch.Tensor:
    key_mask = batch.mask[:, None, None, :]  # (b, 1, 1, l)

    def attend(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # Q, K, V: (b, l, h, d_h).
        out = F.scaled_dot_product_attention(  # (b, h, l, d_h)
            Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2), attn_mask=key_mask
        )
        return out.transpose(1, 2)  # (b, l, h, d_h)

    positions = torch.arange(batch.longest, device=batch.hidden.device)  # (l,)
    return encoder.run(batch.hidden, attend, positions)  # (b, l, d)


def varlen_per_layer(encoder: Encoder, batch: Batch) -> torch.Tensor:
    def attend(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # Q, K, V: (b, l, h, d_h); packed real tokens: (t, h, d_h).
        flat = (-1, *Q.shape[2:])
        packed = cast(  # (t, h, d_h)
            torch.Tensor,
            varlen_attn(
                Q.reshape(flat)[batch.indices],
                K.reshape(flat)[batch.indices],
                V.reshape(flat)[batch.indices],
                batch.cu_seqlens,
                batch.cu_seqlens,
                batch.longest,
                batch.longest,
            ),
        )
        out = packed.new_zeros(Q.shape[0] * Q.shape[1], *Q.shape[2:])  # (b * l, h, d_h)
        out[batch.indices] = packed  # selected rows: (t, h, d_h)
        return out.view(Q.shape)  # (b, l, h, d_h)

    positions = torch.arange(batch.longest, device=batch.hidden.device)  # (l,)
    return encoder.run(batch.hidden, attend, positions)  # (b, l, d)


def varlen_packed(encoder: Encoder, batch: Batch) -> torch.Tensor:
    def attend(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # Q, K, V and return: (t, h, d_h).
        return cast(
            torch.Tensor,
            varlen_attn(Q, K, V, batch.cu_seqlens, batch.cu_seqlens, batch.longest, batch.longest),
        )

    packed = batch.hidden.flatten(0, 1)[batch.indices]  # (t, d)
    packed = encoder.run(packed, attend, batch.positions)  # (t, d)
    out = packed.new_zeros(batch.hidden.shape[0] * batch.longest, packed.shape[-1])  # (b * l, d)
    out[batch.indices] = packed  # selected rows: (t, d)
    return out.view(batch.hidden.shape)  # (b, l, d)


class FlexPacked:
    """Packed tokens in one row; a document mask keeps attention inside each protein."""

    def __init__(self) -> None:
        self.compiled = torch.compile(flex_attention, dynamic=False)
        self.mask_seconds: list[float] = []

    def __call__(self, encoder: Encoder, batch: Batch) -> torch.Tensor:
        device = batch.hidden.device
        padded_total = -(-batch.total // FLEX_TOTAL_MULTIPLE) * FLEX_TOTAL_MULTIPLE  # t_pad
        torch.cuda.synchronize()
        started = time.perf_counter()
        # Filler tokens get their own document, so no protein attends to them.
        document = torch.full((padded_total,), len(batch.lengths), device=device)  # (t_pad,)
        document[: batch.total] = torch.repeat_interleave(  # selected entries: (t,)
            torch.arange(len(batch.lengths), device=device),
            torch.tensor(batch.lengths, device=device),
        )
        block_mask = create_block_mask(
            lambda b, h, q_index, kv_index: document[q_index] == document[kv_index],
            None,
            None,
            padded_total,
            padded_total,
            device=device,
        )
        torch.cuda.synchronize()
        self.mask_seconds.append(time.perf_counter() - started)

        def attend(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
            # Q, K, V: (1, t_pad, h, d_h).
            out: torch.Tensor = self.compiled(  # (1, h, t_pad, d_h)
                Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2), block_mask=block_mask
            )
            return out.transpose(1, 2)  # (1, t_pad, h, d_h)

        packed = batch.hidden.new_zeros(1, padded_total, batch.hidden.shape[-1])  # (1, t_pad, d)
        packed[0, : batch.total] = batch.hidden.flatten(0, 1)[batch.indices]  # (t, d)
        positions = torch.zeros(padded_total, dtype=torch.long, device=device)  # (t_pad,)
        positions[: batch.total] = batch.positions  # selected positions: (t,)
        packed = encoder.run(packed, attend, positions[None])  # (1, t_pad, d)
        out = packed.new_zeros(batch.hidden.shape[0] * batch.longest, packed.shape[-1])  # (b * l, d)
        out[batch.indices] = packed[0, : batch.total]  # selected rows: (t, d)
        return out.view(batch.hidden.shape)  # (b, l, d)


Strategy = Callable[[Encoder, Batch], torch.Tensor]


def seconds_for(strategy: Strategy, encoder: Encoder, batches: list[Batch]) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in batches:
            strategy(encoder, batch)
    torch.cuda.synchronize()
    return time.perf_counter() - started


def probe(name: str, shape: tuple[int, int, int], device: torch.device) -> dict[str, object]:
    torch.manual_seed(0)
    encoder = Encoder(*shape).to(device).eval()
    generator = random.Random(0)
    batches = [
        Batch(natural_lengths(generator), shape[1], device) for _ in range(BATCHES_PER_ROUND)
    ]
    flex = FlexPacked()
    strategies: dict[str, Strategy] = {
        "sdpa_padded": sdpa_padded,
        "varlen_per_layer": varlen_per_layer,
        "varlen_packed": varlen_packed,
        "flex_packed": flex,
    }
    # The first pass is startup: kernel selection everywhere, compilation for FlexAttention.
    first_pass = {label: seconds_for(fn, encoder, batches) for label, fn in strategies.items()}
    flex.mask_seconds.clear()
    rounds: dict[str, list[float]] = {label: [] for label in strategies}
    for _ in range(ROUNDS):
        for label, strategy in strategies.items():
            rounds[label].append(seconds_for(strategy, encoder, batches))

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        reference = sdpa_padded(encoder, batches[0])  # (b, l, d)
        real = batches[0].mask  # (b, l)
        deviation = {
            label: float((fn(encoder, batches[0])[real] - reference[real]).abs().max())
            for label, fn in strategies.items()
        }
    real_tokens = sum(batch.total for batch in batches)
    padded_tokens = sum(len(batch.lengths) * batch.longest for batch in batches)
    baseline = statistics.median(rounds["sdpa_padded"])
    return {
        "encoder": name,
        "layers_width_heads": list(shape),
        "batch_size": BATCH_SIZE,
        "batches": BATCHES_PER_ROUND,
        "real_token_fraction": real_tokens / padded_tokens,
        "median_seconds": {label: statistics.median(times) for label, times in rounds.items()},
        "speedup_over_sdpa_padded": {
            label: baseline / statistics.median(times) for label, times in rounds.items()
        },
        "real_tokens_per_second": {
            label: real_tokens / statistics.median(times) for label, times in rounds.items()
        },
        "first_pass_seconds": first_pass,
        "flex_block_mask_seconds_per_batch": statistics.median(flex.mask_seconds),
        "max_abs_deviation_from_sdpa_on_real_tokens": deviation,
    }


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    device = torch.device("cuda")
    results = [probe(name, shape, device) for name, shape in ENCODERS.items()]
    print(json.dumps({"gpu": torch.cuda.get_device_name(0), "probes": results}, indent=2))


if __name__ == "__main__":
    main()
