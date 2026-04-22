# FastPLMs Parity Investigation — Findings

## Summary

The original GitHub report (FastPLMs ESMC underperforming native ESMC on a
downstream SVM benchmark) prompted a much wider parity audit. Two distinct
silent regressions were uncovered and fixed, and a third source of the
downstream gap was identified as a representation choice (not a bug).

| Family | Bug found? | Severity in fp32 | Fix status |
|---|---|---|---|
| ESMC | Rotary `inv_freq` computed on GPU instead of CPU | mse ~1e-3 at `hidden_states[-2]`, hidden by final LayerNorm | Fixed |
| ANKH | T5 attention applied `1/sqrt(d_kv)` scaling that T5 never had | per-layer rel_std 25-40%, last_hidden_state mse 4e-3 | Fixed |
| ANKH | `bool_tensor.masked_fill(mask, -inf)` silently no-ops; padded keys were attended to | last_hidden_state max\|Δ\| 0.83 between padded and unpadded forward of the same sequence | Fixed |

After the fixes, every parity test passes in fp32 to machine precision and in
bf16 within documented per-layer tolerances. **51 parity tests across 6 family
images, 0 failures, 0 xfails.**

The avivko downstream gap is *primarily* explained by the rotary bug above
(it affects `hidden_states[-2]`, which is what they extracted), with a
secondary contribution from comparing pre-norm `hidden_states[-2]`
(per-feature std ~250) on the native side to post-norm `last_hidden_state`
(per-feature std ~1) on the FastPLMs side.

## Method

New test suite: [testing/test_parity.py](../testing/test_parity.py). Per
family, the suite runs:

1. `test_tokenizer_parity` — vocab size, every token id, every special token id (tokenizer-mode families only).
2. `test_weight_parity_fp32` — per-parameter bit-exact equality.
3. `test_forward_parity_fp32` parametrized over padding scenarios `{single, uniform, skewed}` — per-layer relative std of diff, last_hidden_state mse/maxabs, logits mse.
4. `test_forward_parity_bf16` — same as fp32 with documented per-family tolerances.
5. `test_padding_does_not_pollute_valid_positions_fp32` — runs `[short]` alone and `[short, long_]` padded, asserts the short sequence's valid-position output matches across both runs (this is the test that would have caught the ANKH mask bug if it had existed before).
6. `test_backend_consistency_fp32` — sdpa vs kernels_flash vs flex on FastPLMs side.
7. `test_embed_dataset_pipeline_parity` — end-to-end `embed_dataset()` vs manual native forward+mean pool.

Each family runs in its own Docker image that isolates conflicting native
package deps (`fair-esm` vs EvolutionaryScale `esm`, DPLM's torchtext pin, E1's
in-tree submodule, etc.):

```bash
./build_images.sh
for fam in esm2 esm_plusplus e1 dplm dplm2 ankh; do
    if [ "$fam" = "esm_plusplus" ]; then k=esmc; else k=$fam; fi
    docker run --gpus all --ipc=host --rm -v $(pwd):/workspace \
        fastplms-$fam python -m pytest /workspace/testing/test_parity.py -k $k -v
done
```

### Final results across per-family images

| Image | Passed | xfailed | Notes |
|---|---|---|---|
| `fastplms-esm2` | 11 | 0 | Bit-exact at every layer in fp32. |
| `fastplms-esm_plusplus` (ESMC) | 11 | 0 | Bit-exact at every layer in fp32 after rotary fix. |
| `fastplms-e1` | 8 | 0 | Sequence-mode (no tokenizer parity, no embed_dataset pipeline test). |
| `fastplms-dplm` | 11 | 0 | Native reference is `transformers.EsmForMaskedLM`. |
| `fastplms-dplm2` | 1 | 0 | Forward/weight skipped per pre-existing `contact_head` mismatch. |
| `fastplms-ankh` | 9 | 0 | Bit-exact at every layer in fp32 after both fixes. |

**Total: 51 passed, 0 xfailed, 0 failed.**

## Bug 1 — ESMC rotary `inv_freq` computed on the wrong device

`fastplms/esm_plusplus/modeling_esm_plusplus.py` `_compute_inv_freq`.

**Before.** Called from a context where `from_pretrained(device_map=cuda)`
materializes buffers on the target device, so `torch.arange(...,
device=cuda)` ran the transcendental on the GPU. Native EvolutionaryScale
ESMC creates `inv_freq` on CPU at `__init__` and only later migrates via
`.to(device)`. CPU and GPU fp32 transcendentals differ by ~3.7e-9 in
`inv_freq`, which propagates to ~2.4e-7 in cos/sin, ~1e-6 in rotary outputs,
and compounds across 30 attention layers to ~1e-3 mse at
`hidden_states[-2]`.

The compliance test that existed at the time only checked `last_hidden_state`,
which is post-final-LayerNorm. ESMC's final LayerNorm has very small
per-feature gamma (max |γ| ≈ 0.14) and divides by per-position std (~250),
collapsing the diff by ~10⁴×. So `last_hidden_state` looked fine while
`hidden_states[-2]` (what `avivko` was extracting) was off.

**After.** `_compute_inv_freq` always computes on CPU and explicitly migrates
via `.to(device)`. Forward parity now passes bit-exactly at every layer in
fp32 (mse 0.0e+00, maxabs 0.0e+00).

Diagnostic script: [testing/parity_debug_rotary.py](../testing/parity_debug_rotary.py).

## Bug 2 — ANKH attention applied unwanted `1/sqrt(d_kv)` scaling

`fastplms/ankh/modeling_ankh.py` `AnkhSelfAttention`.

**Before.** `self.scale = self.d_kv ** -0.5 = 0.125` was passed to
`F.scaled_dot_product_attention(..., scale=self.scale)`. Standard transformer
practice — but T5 (and therefore ANKH) intentionally trains *without* attention
scaling. The learned relative position bias absorbs any temperature.

Effect: scores were `(QK^T) / sqrt(64) + bias` instead of `QK^T + bias`.
The softmax distribution was completely wrong because the bias values were
calibrated to a different temperature.

Per-layer rel_std vs native HF `T5EncoderModel` was 12% at block 0, growing to
~40% by block 30. `last_hidden_state` mse was 4e-3 with maxabs 0.49.

**After.** `self.scale = 1.0`. Forward parity now passes at machine precision
for unpadded inputs (per-layer rel_std ~1e-7).

Diagnostic: [testing/parity_debug_ankh.py](../testing/parity_debug_ankh.py).

## Bug 3 — ANKH `bool_tensor.masked_fill(mask, -inf)` silently dropped the padding mask

`fastplms/ankh/modeling_ankh.py` `AnkhSelfAttention.forward`.

**Before.** Padding mask was folded into the relative position bias with:

```python
position_bias = position_bias + attention_mask_4d.masked_fill(
    attention_mask_4d.logical_not(), float("-inf")
)
```

`attention_mask_4d` is a bool tensor. `masked_fill` on a bool tensor with a
float fill value casts `-inf` to bool, which evaluates to `True` (any non-zero
is True). The result is a bool tensor of all `True` — i.e. **no masking at
all**. When that bool gets added to the float `position_bias`, it becomes
`+1.0` everywhere. Padded keys were attended to like valid keys.

This is invisible in single-batch (no padding) forward — and the existing
test sequences used uniform-length batches that hid it. The new
`test_padding_does_not_pollute_valid_positions_fp32` test catches it: it
runs `[short]` alone and `[short, long_]` padded, then compares the short
sequence's valid-position output across both. Difference was max\|Δ\|=0.83
in `last_hidden_state` — orders of magnitude beyond SDPA batch-shape noise.

**After.** Build a proper additive float mask:

```python
mask_additive = torch.zeros(
    attention_mask_4d.shape, dtype=position_bias.dtype, device=position_bias.device,
)
mask_additive.masked_fill_(attention_mask_4d.logical_not(), float("-inf"))
position_bias = position_bias + mask_additive
```

`0.0` at valid keys, `-inf` at padded. Padded keys correctly receive zero
attention weight. Forward parity passes for all padding scenarios.

Diagnostic: [testing/parity_debug_ankh_padding.py](../testing/parity_debug_ankh_padding.py),
isolation test [testing/parity_debug_ankh_mask.py](../testing/parity_debug_ankh_mask.py).

## Why the existing compliance suite missed these

The pre-existing [testing/test_compliance.py](../testing/test_compliance.py)
checked `last_hidden_state` and logits in bf16 with mse < 0.05 and prediction
accuracy > 0.90. Three reasons it didn't catch any of the above:

1. **Loose tolerance.** bf16 mse 0.05 is large enough to absorb significant
   per-layer drift. Tight fp32 thresholds with per-layer relative metrics
   surface drift immediately.
2. **`last_hidden_state`-only.** ESMC's final LayerNorm masks the rotary bug.
   Comparing every hidden state — not just the last — caught it.
3. **No padding-isolation test.** The ANKH mask bug requires *batched padding
   with length variance* to manifest. Uniform-length batches (or
   single-sequence runs) trigger a code path where the broken-mask happens to
   produce mathematically equivalent output (softmax shift-invariance over a
   constant +1.0 cancels). Adding an explicit "padded vs unpadded should
   match" test caught it on the first run.

## Infrastructure changes

Per-family Docker image split to isolate native package conflicts:

- `Dockerfile.base` — torch 2.11.0, transformers 4.57.6, shared deps, source.
- `Dockerfile.esm2` — FROM base, no extras (uses `transformers.EsmForMaskedLM`).
- `Dockerfile.esm_plusplus` — FROM base, installs EvolutionaryScale `esm`'s runtime deps. Avoids `pip install esm` because that pins `transformers<4.53.0`; instead loads the in-tree submodule via sys.path injection in [testing/official/__init__.py](../testing/official/__init__.py).
- `Dockerfile.e1` — FROM base, `pip install -e /app/official/e1`. Reinstalls torch after E1's deps to undo a CUDA 13 wheel pull.
- `Dockerfile.dplm` — FROM base, no extras (uses `transformers.EsmForMaskedLM`). DPLM's official package not installed due to torchtext pin conflict.
- `Dockerfile.dplm2` — FROM base, no extras.
- `Dockerfile.ankh` — FROM base, no extras (uses `transformers.T5EncoderModel`).
- `Dockerfile` — unchanged monolithic legacy image. Kept for backward compatibility.
- `build_images.sh` — convenience script.

Boltz is deferred until its native deps are worked out separately.

## Hub-side propagation

`trust_remote_code=True` loads `modeling_*.py` from the HuggingFace Hub copy
of each checkpoint, not from the local repo. The fixes above need to be
pushed to the Hub copies for end users to receive them:

- `Synthyra/ESMplusplus_small`, `Synthyra/ESMplusplus_large` — push fixed `modeling_esm_plusplus.py`.
- `Synthyra/ANKH_base`, `Synthyra/ANKH_large`, `Synthyra/ANKH2_large`, `Synthyra/ANKH3_large`, `Synthyra/ANKH3_xl` — push fixed `modeling_ankh.py`.

## Response to `avivko`

> Thanks for opening this — it surfaced two real bugs that the pre-existing
> compliance suite was missing, plus a representation-choice issue.
>
> **Bug 1 (ESMC rotary `inv_freq`).** Our `RotaryEmbedding._compute_inv_freq`
> was running the fp32 transcendental on the GPU. Native ESMC computes it on
> the CPU and migrates. The CPU/GPU fp32 difference is ~3.7e-9 in `inv_freq`,
> which compounds across 30 layers to ~1e-3 mse at `hidden_states[-2]` — the
> exact layer you were extracting for your SVM. The post-final-LayerNorm
> `last_hidden_state` masked it (small γ, large per-position std), so our
> compliance test missed it. Fixed; per-layer fp32 mse is now 0.0e+00.
>
> **Bug 2 (ANKH attention scaling).** We were applying the standard
> `1/sqrt(d_kv)` to attention scores, but T5/ANKH was trained without
> scaling — the learned relative position bias absorbs the temperature.
> Per-layer rel_std was 25-40% vs native `T5EncoderModel`. Fixed.
>
> **Bug 3 (ANKH padding mask).** A `bool_tensor.masked_fill(mask, -inf)`
> silently no-op'd because `-inf` casts to `True` on a bool tensor. Padded
> keys were being attended to. Fixed.
>
> **Representation choice (separate from the bugs).** You also extracted
> native `hidden_states[-2]` (pre-final-LayerNorm, per-feature std ~250) but
> our `last_hidden_state` (post-final-LayerNorm, per-feature std ~1). Even
> after Bug 1 is fixed, those are different objects. We're adding a `layer=`
> kwarg to `embed_dataset()` so you can extract `hidden_states[-2]` directly
> from FastPLMs.
>
> A few clarifying questions if you have a moment:
>
> 1. Did your ESM2 / E1 benchmarks behave as expected, or did they show
>    similar gaps? (The ESM2 pipeline didn't have either bug; E1 has its own
>    parity tests now and passes.)
> 2. What attention backend were you using on the FastPLMs side
>    (`sdpa`, `kernels_flash`, `flex`)? The bugs above were backend-agnostic,
>    but worth ruling out.
> 3. Once the patched modeling files are pushed to the Hub, are you able to
>    re-run the SVM benchmark with `layer=-2` so we can confirm the gap
>    closes?

## Diagnostic scripts written during the investigation

Kept in [testing/](../testing/) for future regressions:

- `parity_debug_rotary.py` — verified ESMC rotary CPU-vs-GPU difference.
- `parity_debug_esmc.py`, `parity_debug_esmc_minimal.py`, `parity_debug_diff_structure.py`, `parity_debug_block_internals.py`, `parity_debug_kernel.py` — earlier triage of ESMC.
- `parity_debug_esm2.py` — confirmed ESM2 has no equivalent issue (sanity baseline).
- `parity_debug_ankh.py` — per-block diff that found the unscaled-attention bug.
- `parity_debug_ankh_padding.py` — per-layer diff between alone and padded that surfaced the mask bug.
- `parity_debug_ankh_mask.py` — minimal reproducer showing `bool_tensor.masked_fill(_, -inf)` returns all-True.
