# Attention backends

FastPLMs uses the Transformers attention interface. Callers select a backend at
load time with `attn_implementation` or after loading with
`set_attn_implementation()`.

## Dependencies and platform requirements

FastPLMs runs on Python 3.11 through 3.14. The release validation environment
uses PyTorch 2.13 and Transformers 5.13. Install the core dependencies.
Transformers loads FastPLMs runtime source from the Hugging Face model:

```bash
python -m pip install \
  "torch>=2.13,<2.14" \
  "transformers>=5.13,<5.14"
```

FlashAttention 2 and 3 additionally require Hugging Face `kernels`, a
compatible Linux CUDA device, BF16 execution, and the manifest-pinned kernel
already in cache for offline use:

```bash
python -m pip install \
  "torch>=2.13,<2.14" \
  "transformers>=5.13,<5.14" \
  "kernels>=0.15,<0.16"
```

The Hub quick start below needs network access for the first model download.
Build a manifest-pinned local artifact and pass `local_files_only=True` for an
air-gapped run.

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "Synthyra/ESM2-150M",
    trust_remote_code=True,
    attn_implementation="flex_attention",
)
model.set_attn_implementation("sdpa")
```

Published Hub repositories are the normal user-facing model IDs. Pin
`revision` when the exact published model-code snapshot matters. Contributors
can use a manifest-built artifact under `dist/hub/<model>` for local,
offline validation before publishing an update.

If the caller does not choose a backend, FastPLMs leaves the value unspecified.
Transformers normally selects SDPA. FastPLMs does not implement an `auto`
backend. An unavailable requested implementation raises. For
`output_attentions=True`, FastPLMs uses eager attention for that call and emits
a warning. This does not change the configured backend.

## Implementations

| Name | Transformation | Mask | Main limitation |
| --- | --- | --- | --- |
| `eager` | Explicit score, softmax, and value products | Additive 4D mask | Materializes attention scores |
| `sdpa` | `scaled_dot_product_attention` | Boolean or additive 4D mask | Kernel dispatch is selected by Torch |
| `flex_attention` | Compiled Flex Attention score function | `BlockMask` | First shape and semantics require compilation |
| `flash_attention_2` | Precompiled `kernels-community/flash-attn2` handler at revision `db6b51744f0c` | Packed 2D mask | ESM2 and ESM++ only |
| `flash_attention_3` | Precompiled `kernels-community/flash-attn3` handler at revision `43f0bd269777` | Packed 2D mask | ESM2, ESM++, and DPLM only |

The manifest lists the backends for each family. A name not listed for that
family raises. A listed optional backend that cannot be imported also raises.
Missing dependencies are a configuration error. They do not show that another
kernel was tested.

## Choosing a backend

| Need | Start with | Why |
| --- | --- | --- |
| Official-parity or general inference | `sdpa` | It is the stable declared path for every sequence family |
| Attention maps or `parti` pooling | `eager` | It materializes the attention graph required by those outputs |
| Variable-length batches with compiled masks | `flex_attention` | Its `BlockMask` can avoid padded attention work |
| Precompiled BF16 CUDA kernel on a declared family | `flash_attention_2` or `flash_attention_3` | The immutable binary and compatible runtime are validated before import |
| Reproducible benchmark comparison | Set the exact backend explicitly | Leaving it unspecified delegates selection to Transformers and Torch |

Start from the family row in the
[generated support matrix](generated/support.md). Do not request a backend
because another model family exposes it. ESMC-6B, DPLM2, and ANKH have
family-specific numerical boundaries described below.

`output_attentions=True` requires the full materialized attention-probability
matrix. PyTorch SDPA and Flex Attention do not return that matrix, and the
pinned FlashAttention kernels do not expose it through the FastPLMs contract.
FastPLMs therefore uses eager attention for that forward call and emits a
single `RuntimeWarning` naming the configured backend, effective `eager`
backend, and full-attention-matrix reason. The eager 4-D mask is derived from
the original padding and causal semantics. The configured backend is retained
for later calls.

## FlashAttention compatibility policy

The Flash dependency is Hugging Face `kernels`, not the `flash-attn` Python
distribution. The adapters follow the
[Transformers kernel-loading contract](https://huggingface.co/docs/transformers/v5.13.0/kernel_doc/loading_kernels)
and resolve only the snapshot-pinned `kernels-community` repositories recorded
in the manifest. The immutable snapshot revisions are
`db6b51744f0cd7061386442c09df890fc6d9f47e` for FlashAttention 2 and
`43f0bd269777115d94ff826e0d113ce9c1c9087b` for FlashAttention 3. The tracked
`kernels.lock` records the exact hash of every published binary variant. The
loader asks `kernels` to download and hash-validate the compatible variant
before importing it. It never falls back to a branch, compiles source, imports
the `flash_attn` package, or substitutes one FlashAttention version for another.

After installing `kernels`, use `kernels download .` during image build or
cache preparation to fetch both locked binaries. This command downloads only
precompiled artifacts. It is not required when the runtime populates its
Hugging Face cache on first use.

An explicit kernel-load failure reports the manifest-pinned repository and
revision together with the underlying cause. The exception is not replaced by
a generic dependency error, and no alternate backend is selected.

Both pinned FlashAttention kernels are BF16-only. The Q, K, and V tensors must
share one dtype and one CUDA device. CPU tensors and mixed-device inputs raise
before binary download or import. Direct FP32 and FP16 calls raise before
kernel loading. An
FP32-resident model may use an advertised FlashAttention backend
only inside CUDA BF16 autocast, where the operation resolves to BF16 while the
stored parameters remain FP32. Parity, artifact, embedding, and benchmark
paths derive their backend and dtype combinations from this manifest contract;
they do not probe or fall back to an undeclared precision.

FlashAttention validation must cover dense and mixed-padding forward and
backward passes, including LoRA gradients, on the frozen release environment.
Both offline variables, `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, block
kernel downloads. A required uncached binary therefore raises with its pinned
repository, revision, and original loader error.

All five ESM++/ESMC backends can be selected. SDPA is the default and has the
highest numerical fidelity. Flex Attention and FlashAttention 3 are supported
and non-experimental, although their BF16 arithmetic differs from SDPA.
Accuracy metrics are diagnostics and warnings, not strict parity release gates.
Dispatch integrity, finite values, exact mask semantics, output shape, and
catastrophic biological disagreement are hard failures.

| ESMC backend | Status | Relative L2 | Q99.9 | Residue cosine | Pooled cosine | Top-1 | Jensen-Shannon |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `sdpa` | Recommended fidelity path | Exact-head report required | Exact-head report required | Exact-head report required | Exact-head report required | Exact-head report required | Exact-head report required |
| `eager` | Supported fallback semantics | Pending measured frozen-head GH200/aarch64 set | Pending | Pending | Pending | Pending | Pending |
| `flash_attention_2` | Supported; unavailable on current lock | Structured unavailable record required; prior execution evidence is historical and separate | Not measured | Not measured | Not measured | Not measured | Not measured |
| `flex_attention` | Supported, numerically divergent | Pending measured frozen-head GH200/aarch64 set | Pending | Pending | Pending | Pending | Pending |
| `flash_attention_3` | Supported, non-experimental; unavailable on current lock | Structured unavailable record required | Not measured | Not measured | Not measured | Not measured | Not measured |

The release candidate must replace eager, SDPA, and Flex pending cells with
distributions measured for each checkpoint, dtype, hardware, and locked
sequence panel. FlashAttention 2 and 3 instead require structured unavailable
records with no numerical fields. A threshold is not a measurement, and a
result from another head or revision is not carried forward. The current release-confirmation target is the exact
GH200/aarch64 workstation and repository container build. H100 and H200 remain
Hopper-class deployment examples, but they are not current release evidence.

Diagnostic jobs write immutable JSON reports under
`artifacts/diagnostics/esmc/`. Published accuracy bands produce warnings. The
separate corruption/catastrophe guardrails are relative L2 at most `0.25`,
relative Q99.9 at most `0.50`, first-percentile residue cosine at least `0.90`,
pooled cosine at least `0.95`, confident-position top-1 at least `0.80`, and
Jensen-Shannon divergence at most `0.05`. These broad limits catch broken
dispatch, masking, or output semantics; they are not parity or quality claims.

Default documentation generation remains pending even when that directory or
`FASTPLMS_DIAGNOSTIC_REPORTS` exists. On a frozen release head, explicitly
select the complete 30-record schema-v3 set and then check the generated cards:

```bash
PYTHONPATH=src python -m tools.artifacts.generate_docs \
  --source-root . \
  --esmc-report-root artifacts/diagnostics/esmc

PYTHONPATH=src python -m tools.artifacts.generate_docs \
  --source-root . \
  --esmc-report-root artifacts/diagnostics/esmc \
  --check
```

Use `--require-esmc-release-evidence` to select
`FASTPLMS_DIAGNOSTIC_REPORTS` or the default report directory without silently
falling back to pending output. Either evidence option fails closed on a
missing, extra, malformed, stale, self-digest-invalid, wrong-device, or
cross-device report, or on a missing/stale dependency lock, installed inventory,
container build/image identity, or official-reference source attestation. The
generated capability manifest and applicable model cards record the exact
candidate and official-source records, context, aggregate metric ranges, and
per-case minimum/median/maximum distributions.

The current locked GH200/aarch64 release image has no validated FlashAttention
2 kernel. Prior real execution was captured in separate workstation JUnit, but
the immutable report and environment attestation are not bundled in this
repository. It is not copied into the current ESMC release distribution or
used for a numerical claim. The manifest-pinned FlashAttention 3 revision contains x86-64
variants but no locked PyTorch 2.13, CUDA 13 aarch64 artifact. Older ARM
artifacts target different PyTorch/CUDA combinations and are not substituted.
Both backends remain supported and non-experimental, but current-platform
requests raise before dispatch and their schema-v3 records explicitly attest
that unavailability.

DPLM advertises eager, SDPA, Flex Attention, and FlashAttention 3. Its pinned
official BF16 contract keeps parameter storage in FP32 and uses CUDA BF16
autocast. Historical, non-release H100 diagnostics recorded eager and Flex
worst hidden-state relative L2 errors of `0.009212` and `0.006768`, respectively.
Static BF16 parameter storage is not the official DPLM precision path and is
not used to justify backend support, and those values are not current GH200
release evidence.

DPLM2 advertises SDPA only. Its pinned BF16 contract also keeps parameters in
FP32 and evaluates them under CUDA BF16 autocast; static BF16 parameter storage
raises before inference. A historical, non-release H100 diagnostic recorded
worst hidden-state relative L2 errors of `0.011772` for eager, `0.011231` for Flex
Attention, `0.013495` for FlashAttention 2, and `0.012656` for FlashAttention 3.
Each exceeds the fixed `0.01` engineering target. Explicit requests for any of
those backends therefore raise, and their dead kernel paths are not retained in
the DPLM2 implementation. These values are not current GH200 release evidence.

ANKH advertises eager attention and SDPA only. Selection is local to the model
instance and does not mutate process-global CUDA SDPA reduction policy. This
family support applies to the optimized ANKH encoder. The full
sequence-to-sequence checkpoint retains the decoder's declared implementation
boundary.

## Mask semantics

`fastplms.attention` centralizes mask conversion. The same biological validity
mask is normalized into:

- a packed 2D token mask for FlashAttention;
- a 4D mask for eager attention and SDPA;
- a Flex `BlockMask` for padding, causal, block-causal, or declared custom
  semantics.

The original attention mask must have exact shape `(batch, sequence)` before
backend dispatch. FlashAttention calls with a packed 2D padding mask always use the varlen kernel,
including causal self-attention. The causal flag is passed to the varlen kernel,
and padded query rows are restored as exact zeros after repadding. Masked calls
reject shapes or devices that do not match Q, K, and V before loading a kernel.

E1's block-causal pattern is a distinct semantic key. It is never represented
as ordinary padding attention. Mixed-length and skewed-padding parity cases
exercise every required representation.

Flex functions and masks are cached only after explicit execution. Compilation
is keyed by execution shape, device, dtype, and attention semantics rather than
the exact row-length tuple, so compatible batches reuse compiled work. Mask
content remains correct for each call. `clear_flex_attention_caches()` provides
bounded cleanup of FastPLMs compiled-function and `BlockMask` caches without
clearing process-global Torch compiler state. Importing FastPLMs does not
compile Flex or modify Dynamo or Inductor settings.

## Attention outputs and `parti`

The `parti` embedding pooler constructs an attention graph, so it requires:

```python
attn_implementation="eager"
```

It rejects sequences longer than 2,048 biological residues. Other backends do
not materialize the complete attention graph as a side effect. Models that do
not expose meaningful sequence attention, including ESMFold2, reject `parti`.

## Validation

Backend validation uses the same valid biological positions as official parity.
It measures relative L2 error, relative 99.9th-percentile error, first-percentile
residue cosine, per-sequence pooled cosine, confident-position top-1 agreement,
and Jensen-Shannon divergence for probability tensors. ESMC SDPA remains the
exact, recommended path; eager validates fallback and mask semantics; Flex is
the measured supported diagnostic backend. A published Flex-band miss warns
and records all six metric distributions, while dispatch, finiteness,
mask/shape integrity, and the separate corruption limits remain hard failures.
FlashAttention 2 and 3 remain supported, non-experimental interfaces, but the
current locked GH200/aarch64 image records them as unavailable and fails closed
before dispatch. Historical FlashAttention 2 execution evidence remains
separate from current release acceptance.

Performance is measured separately from correctness. See
[benchmarking](benchmarking.md) for compile-time, steady-state, padding, memory,
and regression methodology.
