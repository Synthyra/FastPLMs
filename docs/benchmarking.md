# Benchmarking

Performance measurements run outside pytest on the validated H100 workstation.
Pytest contains only a short CUDA-event smoke test for the harness. Correctness,
parity, and structure compliance remain separate release gates.

Every report records the exact Python, PyTorch, CUDA, Transformers, FastPLMs,
Hugging Face `kernels`, Transformer Engine, driver, and accelerator environment.
Performance claims apply only to a matching H100 environment fingerprint.

## Fixed release matrix

The manifest selects one deep representative for each sequence architecture:
ESM2, ESMC or ESM++, ESM3, E1, DPLM, DPLM2, and ANKH. For every attention
backend declared by that family, the suite measures:

- latency at `b=1`, `l=512`;
- throughput at `b=8`, `l=1024`;
- padding efficiency for lengths `(1024, 512, 256, 128, 64, 64, 32, 32)`.

The steady-state operation receives pre-tokenized, preallocated GPU tensors.
Loading, first forward, compilation, steady-state forward, and complete embedding
are separate records. Compilation and first-forward costs are never amortized
into steady-state throughput.

Within one checkpoint and precision, the suite loads model weights once. It
changes attention only through `set_attn_implementation()`. The first record
contains `load_ms`; later records set `model_reused=true` and leave `load_ms`
empty. This removes repeated checkpoint deserialization without moving
tokenization or host transfer into the measured forward.

Each case also records the manifest field `bf16_execution`. Families declaring
`static_parameters` load BF16 parameters. Families declaring
`fp32_parameters_autocast` retain FP32 parameters and time model computation
inside CUDA BF16 autocast. This includes structure-family entry points such as
folding and diffusion, not only token-model forwards. Results from the two
storage and execution policies are not mixed under one cache key. The
single-case CLI derives this field for registered checkpoint IDs. An
unregistered local path must provide `--bf16-execution` explicitly, and an
override that conflicts with a registered manifest entry raises.

ESMFold and Boltz2 are represented by explicit `structure_startup` records in
this harness. Their model loading is measurable, but a generic token forward is
not a folding-throughput contract. End-to-end folds, feature preparation,
sampling, and structure outputs run in the dedicated structure suite and are not
reported as tokens per second.

Run the fixed matrix with:

```bash
python -m benchmarks.suite \
  --output artifacts/benchmarks/h100.json
```

## FlashAttention source policy

FastPLMs accepts only immutable, precompiled artifacts loaded by the Hugging
Face `kernels` package. The benchmark workflow never compiles FlashAttention,
installs the source `flash-attn` distribution, or substitutes another
implementation. FlashAttention benchmark rows are included only for the
family-specific revision-pinned artifacts that pass dense, mixed-padding, and
checkpoint parity on the locked PyTorch 2.13 and CUDA 13 H100 stack.

## ESMFold2 representation modes

ESMFold2 keeps three representation measurements distinct:

- `projection` receives precomputed BF16 hidden states H with shape
  `(b, l, 81, 2560)` and measures only the learned map to Z with shape
  `(b, l, 256)`. It is labeled BF16 only. An FP8 label would be misleading
  because this operation does not run ESMC.
- `esmc_projection` receives preallocated residue tensors, runs ESMC with all 81
  hidden states, and applies the learned projection. This is the end-to-end
  representation path measured separately in BF16 and FP8 across every ESMFold2
  attention backend and each fixed shape. `esmc_reload_ms` records construction
  of runtime precision modules from canonical BF16 weights.
- `esmfold2_embed` measures one complete call through the shared embedding API,
  including residue encoding, ESMC inference, learned projection, residue-only
  pooling, and result construction. BF16 and FP8 are separate records.

None of these modes runs the folding trunk or diffusion sampler. Full ESMFold2
folding remains in the structure suite, where geometry and confidence metrics
are meaningful.

Run a single ESMC-plus-projection case with:

```bash
python -m benchmarks \
  --model dist/hub/ESMFold2 \
  --auto-class AutoModel \
  --backend sdpa \
  --precision bf16 \
  --mode esmc_projection \
  --batch-size 1 \
  --sequence-length 512 \
  --local-files-only \
  --output artifacts/benchmarks/esmfold2-esmc-projection.json
```

Build and validate `dist/hub/ESMFold2` from the manifest-pinned checkpoint
before running this command. For a published artifact, pass its exact
`Synthyra/ESMFold2` revision instead of benchmarking a mutable upstream
identifier.

Explicit FP8 fails when Transformer Engine or compatible hardware is
unavailable. It never falls back to BF16 under an FP8 label.

## Descriptive exhaustive sweep

The exhaustive entry point covers every manifest checkpoint, every declared
sequence backend, batch sizes `(1, 2, 4, 8)`, and lengths
`(128, 256, 512, 1024)`. It also covers both ESMFold2 representation operations.
Structure-only families retain startup records because folding belongs to the
structure suite.

```bash
python -m benchmarks.suite \
  --exhaustive \
  --output artifacts/benchmarks/h100-exhaustive.json
```

Exhaustive records use `matrix_kind="exhaustive"`,
`claim_scope="descriptive_only"`, and `claim_eligible=false`. The command rejects
a regression baseline. Its output can diagnose scaling behavior, but it cannot
establish a release gate or speed claim.

## Timing protocol

GPU work is timed with CUDA events. Warmup continues until the medians of two
consecutive ten-sample windows differ by less than 2 percent. A case that does
not stabilize fails so clocks, thermals, and competing workloads can be
investigated.

The measurement phase collects seven blocks. Each block lasts at least 250 ms
and contains at least five forwards. Reports retain every raw event sample,
logical and padded tokens per second, median and P95 latency, peak allocated and
reserved memory, compile time, first-forward time, load time, GPU temperature
and clocks before and after, and the complete environment fingerprint.

Logical throughput counts valid biological tokens. Padded throughput counts
allocated token positions. Both are reported so padding savings remain visible
without disguising the tensor shape executed by the backend.

## Regression gate

Let scalar throughput ratio `r` be:

```text
r = throughput_current / throughput_base
```

The gate compares matched measurement blocks with a deterministic paired
bootstrap interval:

- fail when the one-sided 95 percent upper confidence bound for `r` is below
  `0.95`;
- fail unconditionally when median `r` is below `0.90`;
- fail memory growth above the larger of 5 percent or 256 MiB;
- fail unconditionally when memory growth exceeds 10 percent;
- support a speed claim only when the one-sided lower confidence bound is at
  least `1.05`.

```bash
python -m benchmarks.regression \
  artifacts/benchmarks/current.json \
  benchmarks/baselines/h100.json \
  --output artifacts/benchmarks/gate.json
```

The command never updates a baseline. A baseline change is a separate,
reviewable file change supported by raw results and a matching environment.

## Interpreting results

Each dense, throughput, and mixed-padding case is evaluated independently. A
backend can improve padded batches while regressing dense batches. A throughput
improvement also does not relax parity: every advertised backend must pass its
correctness contract before its performance result can support a claim.
