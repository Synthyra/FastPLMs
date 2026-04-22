# Attention Backends

All FastPLMs sequence models share a common attention backend system controlled by `config.attn_backend`. This document covers how each backend works, when to use it, and how to configure it.

## Overview

| Backend | Key | Numerical Equivalence | Speed | Availability |
|---------|-----|----------------------|-------|-------------|
| PyTorch SDPA | `"sdpa"` | Exact | Fast | Any PyTorch >= 2.0 |
| Flash Attention | `"kernels_flash"` | Approximate | Fastest | `pip install kernels` |
| Flex Attention | `"flex"` | Near-exact | Very fast | PyTorch >= 2.5 |
| Auto | `"auto"` | Varies | Best available | Always |

## SDPA (Default)

PyTorch's `scaled_dot_product_attention` dispatches to a fused CUDA kernel (cuDNN or memory-efficient attention) that is faster and more memory-efficient than naive attention while being mathematically identical.

**When to use:** Reproducibility, numerical sensitivity, general-purpose inference.

**Implementation:** Each attention layer calls `F.scaled_dot_product_attention(query, key, value, attn_mask)` with a 4D mask of shape `(batch, 1, 1, seq_len)`.

**Attention weights:** SDPA does not natively return attention weights. When `output_attentions=True` is requested, all backends (including SDPA) compute attention weights via a separate naive matrix multiplication: `scores = Q @ K^T`, softmax, then `context = scores @ V`. This separate computation negates the memory savings of fused attention, so `output_attentions=True` should only be used for inspection or contact prediction, not during high-throughput inference.

## Flash Attention (`kernels_flash`)

Flash Attention 2/3 tiles the attention computation into blocks that fit in SRAM and applies an online softmax algorithm. This avoids materializing the full `(seq_len, seq_len)` attention matrix in HBM, achieving O(n) memory and typically 2-4x faster throughput than SDPA on Ampere (A100) and Hopper (H100) GPUs at long sequence lengths.

**When to use:** Maximum throughput on A100/H100, long sequences, large batch sizes.

**Numerical properties:** The online softmax and tiling introduce floating-point rounding differences compared to standard attention. These are typically small but not guaranteed to be inconsequential. They can compound across layers and interact with low-precision dtypes (bf16/fp16). If exact reproducibility matters, use `"sdpa"`.

**Installation:** FastPLMs uses the HuggingFace `kernels` package for pre-built Flash Attention binaries:

```bash
pip install kernels
```

No C++ compiler or CUDA toolkit version pinning required. The `kernels` package fetches a pre-compiled binary matched to your GPU architecture (SM80 for Ampere, SM90 for Hopper). If no compatible binary exists, the model gracefully falls back to `"flex"` or `"sdpa"`.

**Implementation details:**

1. Q, K, V are transposed from `(batch, heads, seq, dim)` to `(batch, seq, heads, dim)` for the kernels layout
2. For variable-length batches, padding tokens are removed via `_unpad_input()` which computes cumulative sequence lengths
3. The kernels flash function is called with the unpadded tensors
4. `pad_input()` reconstructs the full padded layout
5. Flash Attention 3 is tried first (Hopper GPUs), falling back to Flash Attention 2

## Flex Attention (`flex`)

PyTorch's `flex_attention` (PyTorch >= 2.5) generates a fused Triton kernel customized to the mask pattern. The primary advantage is **block masks** that skip padding tokens entirely at the CUDA block level, providing meaningful speedups on variable-length batches.

**When to use:** Variable-length batches with significant padding, best sustained throughput with `torch.compile`.

**Numerical properties:** Near-exact to SDPA. Differences are typically within floating-point rounding of naive computation.

**First-call compilation:** The first forward pass triggers JIT compilation via Triton, which takes 30-120 seconds. All subsequent calls with the same mask shape are fast. When combined with `torch.compile`, this yields the best sustained throughput.

**Implementation:**

1. A block mask is created from the 2D attention mask via `create_block_mask(mask_mod, batch, 1, seq_len, seq_len)`
2. The mask mod function returns True for positions that should attend to each other
3. `flex_attention(query, key, value, block_mask=block_mask)` generates and runs the fused kernel
4. E1 uses a block-causal variant where within-sequence attention is bidirectional but cross-sequence attention is causal

## Auto (`auto`)

Selects the best available backend in priority order: `kernels_flash` -> `flex` -> `sdpa`. Useful when you want maximum speed without manual configuration. The resolved backend may differ across machines depending on installed packages and GPU architecture.

## Per-Family Caveats

- **ANKH** supports only `sdpa` and `flex`. The flash-attention kernels can't accept the additive T5 relative position bias, so requesting `kernels_flash` (or `auto` resolving to it) silently falls back to `flex` (or `sdpa` if flex is unavailable). T5 attention is also unscaled (no `1/sqrt(d_kv)` factor) - the learned position bias absorbs the temperature.
- **E1** uses a block-causal flex variant: bidirectional within a sequence, causal across sequences in a packed multi-sequence batch.
- **DPLM2** packs amino-acid and structure tokens in the same sequence; the attention mask logic accounts for the multimodal layout but the backend choice is otherwise unchanged.

## Setting the Backend

Every FastPLMs sequence model (ESM2, ESM++, E1, DPLM, DPLM2, ANKH) supports **both** load-time and post-load backend switching. Pick whichever fits your workflow.

### At Load Time

Set `config.attn_backend` before calling `from_pretrained`:

```python
from transformers import AutoConfig, AutoModelForMaskedLM

config = AutoConfig.from_pretrained("Synthyra/ESM2-150M", trust_remote_code=True)
config.attn_backend = "flex"
model = AutoModelForMaskedLM.from_pretrained(
    "Synthyra/ESM2-150M", config=config, trust_remote_code=True
)
```

### After Load Time

Every family's `PreTrainedModel` subclass exposes a mutable `attn_backend` property whose setter propagates the change to every attention submodule in-place:

```python
model = AutoModelForMaskedLM.from_pretrained("Synthyra/ESM2-150M", trust_remote_code=True)
model.attn_backend = "flex"  # every attention layer now uses flex

model.attn_backend = "kernels_flash"  # flip to flash without reloading
```

This is useful for benchmarking multiple backends on the same weights, or for falling back at runtime if a backend turns out to be unavailable on the current GPU. The setter validates that the requested backend is installed and raises `AssertionError` otherwise -- except for ANKH's `kernels_flash` request, which silently falls back to `flex` (or `sdpa`) because the flash kernels cannot accept ANKH's additive relative position bias.

## Backend Resolution

Each model has a `resolve_attention_backend()` function that:

1. Validates the requested backend string
2. For `"auto"`, probes available backends in order: kernels_flash -> flex -> sdpa
3. Prints the resolved backend once (globally, to avoid log spam)
4. Returns an `AttentionBackend` enum value

The resolved enum is stored on each attention layer as `self.attn_backend` and on the encoder as `self.attention_backend`.

## Mask Transformations

`fastplms/attention.py::get_attention_mask()` builds a shared set of padding masks once per forward, and every family consumes the same output:

| Backend | Mask produced by `get_attention_mask` | Shape |
|---------|---------------------------------------|-------|
| SDPA | Boolean 4D mask (True = valid) | `(batch, 1, 1, seq_len)` |
| Flash | Boolean 2D mask (True = valid) | `(batch, seq_len)` |
| Flex | `BlockMask` via `create_block_mask` | Opaque block mask object |

Families that need an **additive** (float, `0.0`/`-inf`) mask -- ANKH is currently the only one, because T5 relative position bias is added directly to attention scores -- convert the shared bool mask with the `bool_to_additive_mask(bool_mask, dtype)` helper in `fastplms/attention.py`. Use the helper, don't hand-roll it:

> Never call `.masked_fill(bool_mask, float("-inf"))` on a bool tensor. `bool(float("-inf"))` is `True`, so the result is a bool tensor and the mask is silently dropped. `bool_to_additive_mask` allocates the float tensor correctly and is the only sanctioned way to produce an additive mask inside the codebase.

## Interaction with `torch.compile`

- **SDPA**: Works well with `torch.compile` out of the box
- **Flex**: Best performance when the entire model is compiled; the Triton kernel generation integrates with the compiler
- **Flash**: `torch.compile` wraps the kernels call; dynamic warmup detects when compilation has stabilized

The throughput benchmark (`testing/throughput.py`) applies `torch.compile` to all backends and uses dynamic warmup stabilization to ensure measurements reflect compiled performance.

## s_max Tracking

When `output_s_max=True` is passed (ESM2, E1), each attention layer computes the per-head maximum attention score bound: `max(||Q|| * ||K||)` per head. This is useful for numerical stability analysis and debugging but adds overhead and should not be enabled during production inference.
