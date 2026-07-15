# Migration to FastPLMs 1.0

FastPLMs 1.0 intentionally breaks the repository and Python API layout. There
are no compatibility import shims or aliases for old commands.

## Package layout

Source now follows the standard `src` layout. Imports use `fastplms`, while
repository paths use `src/fastplms`:

```text
old: fastplms/esm2/modeling_fastesm.py
new: src/fastplms/models/esm2/modeling_fastesm.py
```

Shared attention code moved to `fastplms.attention`. Shared embedding code moved
to `fastplms.embeddings`. Test-time training moved to `fastplms.models.ttt`.
Conversion, artifact, remote, and debugging commands live under `tools`.

## Attention

Replace family-specific `attn_backend`, `flex`, `kernels_flash`, or `auto`
configuration with the Transformers API:

```python
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    attn_implementation="flex_attention",
)
model.set_attn_implementation("sdpa")
```

The FastPLMs 1.0 names are `eager`, `sdpa`, `flex_attention`,
`flash_attention_2`, and `flash_attention_3`, limited by the model manifest.
ESM2 and ESM++ support both precompiled Hugging Face kernels; DPLM supports
FlashAttention 3 only. DPLM2 supports SDPA only. There is no source-build path
and no silent fallback. Leaving the value unspecified permits the normal
Transformers default.

## Embeddings

Replace dictionary-returning or family-specific helpers with ordered records:

```python
result = model.embed_dataset(
    inputs,
    batch_size=8,
    pooling=("mean",),
)
```

Duplicate identifiers are preserved. Call `result.as_dict()` only when keys are
unique or after selecting an explicit duplicate policy. New persisted outputs
use sharded safetensors or SQLite. `.pth` output was removed; legacy `.pth`
loading requires explicit unsafe-pickle opt-in.

`full_embeddings=True` now means ragged biological-residue tensors and cannot be
combined with pooling. Pooling no longer includes BOS, EOS, padding, chain
delimiters, or non-protein structure tokens.

## ANKH

Use `FastAnkhModel` or `AutoModel` for the official encoder and
`FastAnkhForConditionalGeneration` or `AutoModelForSeq2SeqLM` for the official
sequence-to-sequence model. The synthesized masked-LM head is now explicitly
named `FastAnkhForMaskedLMExtension` and must not be treated as official ANKH
parity.

## ESMFold2

Only the standard, fast, experimental cutoff 2025, and experimental fast cutoff
2025 variants remain supported. Dataset embedding accepts single protein chains
and exposes the learned width-256 representation.

Use `esmc_precision="auto"` and inspect `model.esmc_precision_status`; FastPLMs
1.0 selects the validated Transformer Engine FP8 path when ESMC is loaded
directly onto a supported CUDA device. It otherwise resolves to BF16 with the
reason recorded in status. Explicit `fp8` raises when unavailable. Use
`model.reload_esmc()` to change the policy or destination.

## Development infrastructure

Initialize official references with:

```bash
git submodule update --init --recursive
```

The old Dockerfile fleet and per-family shell builders were replaced by
`docker/Dockerfile`, Buildx Bake, Compose, and `tools/remote/run.py`. Throughput
measurements moved out of pytest into `benchmarks`.

## Checkpoint compatibility

Checkpoint parameter names are retained where possible. If a manifest entry
names a state transform, run the corresponding deterministic converter and
verify its exact conversion test. Do not rename keys manually or accept missing
or unexpected keys as a migration strategy.
