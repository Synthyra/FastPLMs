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
    first, second = states.chunk(2, dim=-1)
    return states * cos + torch.cat((-second, first), dim=-1) * sin


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
        qkv = self.qkv(self.attention_norm(hidden))
        q, k, v = qkv.view(*hidden.shape[:-1], 3, self.heads, -1).unbind(-3)
        attended = attend(rotate(q, cos, sin), rotate(k, cos, sin), v)  # (..., tokens, h, d_h)
        hidden = hidden + self.out(attended.reshape(hidden.shape))
        updated: torch.Tensor = hidden + self.down(F.gelu(self.up(self.ffn_norm(hidden))))
        return updated


class Encoder(nn.Module):
    cos: torch.Tensor  # (l_max, 1, d_h)
    sin: torch.Tensor  # (l_max, 1, d_h)

    def __init__(self, layers: int, width: int, heads: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(Layer(width, heads) for _ in range(layers))
        inverse = 1.0 / (10000 ** (torch.arange(0, width // heads, 2).float() / (width // heads)))
        angles = torch.outer(torch.arange(MAX_LENGTH).float(), inverse)  # (l_max, d_h / 2)
        self.register_buffer("cos", torch.cat((angles, angles), -1).cos()[:, None])
        self.register_buffer("sin", torch.cat((angles, angles), -1).sin()[:, None])

    def run(self, hidden: torch.Tensor, attend: Attend, positions: torch.Tensor) -> torch.Tensor:
        # BF16 tables keep Q and K in the dtype the variable-length kernel accepts.
        cos = self.cos[positions].bfloat16()  # (..., tokens, 1, d_h)
        sin = self.sin[positions].bfloat16()
        for layer in self.layers:
            hidden = layer(hidden, attend, cos, sin)
        return hidden


class Batch:
    """One padded batch and the packing metadata derived from it once."""

    def __init__(self, lengths: list[int], width: int, device: torch.device) -> None:
        self.lengths = lengths
        self.longest = max(lengths)
        length_column = torch.tensor(lengths, device=device)[:, None]
        self.mask = torch.arange(self.longest, device=device)[None] < length_column  # (b, l)
        self.hidden = torch.randn(len(lengths), self.longest, width, device=device)  # (b, l, d)
        self.indices = self.mask.flatten().nonzero().flatten()  # (t,)
        self.positions = self.indices % self.longest  # (t,) column of each real token
        self.cu_seqlens = F.pad(length_column.flatten().cumsum(0), (1, 0)).int()  # (b + 1,)
        self.total = sum(lengths)


def sdpa_padded(encoder: Encoder, batch: Batch) -> torch.Tensor:
    key_mask = batch.mask[:, None, None, :]  # (b, 1, 1, l)

    def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=key_mask
        )
        return out.transpose(1, 2)

    positions = torch.arange(batch.longest, device=batch.hidden.device)
    return encoder.run(batch.hidden, attend, positions)


def varlen_per_layer(encoder: Encoder, batch: Batch) -> torch.Tensor:
    def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        flat = (-1, *q.shape[2:])
        packed = cast(
            torch.Tensor,
            varlen_attn(
                q.reshape(flat)[batch.indices],
                k.reshape(flat)[batch.indices],
                v.reshape(flat)[batch.indices],
                batch.cu_seqlens,
                batch.cu_seqlens,
                batch.longest,
                batch.longest,
            ),
        )
        out = packed.new_zeros(q.shape[0] * q.shape[1], *q.shape[2:])
        out[batch.indices] = packed
        return out.view(q.shape)

    positions = torch.arange(batch.longest, device=batch.hidden.device)
    return encoder.run(batch.hidden, attend, positions)


def varlen_packed(encoder: Encoder, batch: Batch) -> torch.Tensor:
    def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            varlen_attn(q, k, v, batch.cu_seqlens, batch.cu_seqlens, batch.longest, batch.longest),
        )

    packed = batch.hidden.flatten(0, 1)[batch.indices]  # (t, d)
    packed = encoder.run(packed, attend, batch.positions)
    out = packed.new_zeros(batch.hidden.shape[0] * batch.longest, packed.shape[-1])
    out[batch.indices] = packed
    return out.view(batch.hidden.shape)


class FlexPacked:
    """Packed tokens in one row; a document mask keeps attention inside each protein."""

    def __init__(self) -> None:
        self.compiled = torch.compile(flex_attention, dynamic=False)
        self.mask_seconds: list[float] = []

    def __call__(self, encoder: Encoder, batch: Batch) -> torch.Tensor:
        device = batch.hidden.device
        padded_total = -(-batch.total // FLEX_TOTAL_MULTIPLE) * FLEX_TOTAL_MULTIPLE
        torch.cuda.synchronize()
        started = time.perf_counter()
        # Filler tokens get their own document, so no protein attends to them.
        document = torch.full((padded_total,), len(batch.lengths), device=device)
        document[: batch.total] = torch.repeat_interleave(
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

        def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            out: torch.Tensor = self.compiled(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask
            )
            return out.transpose(1, 2)

        packed = batch.hidden.new_zeros(1, padded_total, batch.hidden.shape[-1])
        packed[0, : batch.total] = batch.hidden.flatten(0, 1)[batch.indices]
        positions = torch.zeros(padded_total, dtype=torch.long, device=device)
        positions[: batch.total] = batch.positions
        packed = encoder.run(packed, attend, positions[None])
        out = packed.new_zeros(batch.hidden.shape[0] * batch.longest, packed.shape[-1])
        out[batch.indices] = packed[0, : batch.total]
        return out.view(batch.hidden.shape)


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
        reference = sdpa_padded(encoder, batches[0])
        real = batches[0].mask
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
