# Hopper/SM90 benchmarks

Benchmarks run separately from pytest. The steady-state path measures only a
forward pass on pre-tokenized tensors that are already on the GPU. Startup,
compilation, full embedding, and the ESMFold2 projection use separate modes.

Run the complete manifest-derived release matrix on a compatible accelerator
host in its containerized native environment:

```bash
python -m benchmarks.suite \
  --backends eager sdpa flex_attention \
  --junit-output artifacts/junit/benchmark.xml \
  --output artifacts/benchmarks/h100.json
```

The output name is a legacy automation identifier. Each report records the
actual accelerator, architecture, and software fingerprint. Regression
comparison requires an exact match. Remote orchestration runs Bake on the
native host platform and verifies the architecture and content digest of each
loaded image. It does not use emulated images.

For the pre-publication baseline, build and consume the manifest-selected local
Hub artifacts in the same frozen source job:

```bash
python -m tools.artifacts.build_all \
  --benchmark-suite --source-root . --output-root dist/hub
python -m benchmarks.suite \
  --artifact-root dist/hub --local-files-only \
  --backends eager sdpa flex_attention \
  --junit-output artifacts/junit/benchmark-capture.xml \
  --output artifacts/benchmarks/h100-baseline-candidate.json
```

Artifact mode validates each selected artifact and the local ESMC dependency for
ESMFold2 before it loads the model. Reports retain registry repository and
revision case keys. They record weights, runtime, source, canonical-state, and
manifest identities without local paths. Local paths are never baseline identities.
The locked release runner records FA2 as prior focused evidence and FA3 as
unavailable on its declared platform. It does not download, build, or run either Flash
kernel. Capture reports contain the environment and artifact identities required
to promote a baseline. They keep cold compile time separate from warm throughput.

The experimental ESMFold2-300 and ESMFold2-600 checkpoints are included only
when their manifest artifacts are available. Their confidence heads are
disabled, so pLDDT, pTM, iPTM, and PAE are not benchmark outputs. The 300M
single-protein comparison passed, but the full structure benchmark remains
pending. The published mirrors are pinned to revisions
`a38a62ae930d157484b331c2bf4241684573adba` (300M) and
`71c67d0b2b73dc245ea7c3cc0d0476439a882d08` (600M). The 600M variant has no
inference validation result.

The matrix includes startup, compilation, full embedding, `b=1, l=512` latency,
`b=8, l=1024` throughput, the fixed skewed-padding case, and BF16 and FP8
ESMFold2 projection measurements. Use `--baseline <path>` to apply the
one-sided 95% regression gates. This option does not change the baseline.

```bash
python -m benchmarks \
  --model dist/hub/ESM2-8M \
  --backend sdpa \
  --mode steady \
  --batch-size 1 \
  --sequence-length 512 \
  --local-files-only \
  --output artifacts/benchmarks/esm2-sdpa.json
```

First, build the manifest-pinned local artifact. If you use a published Hub
artifact, pass its immutable revision and keep it in the report.

Use `--lengths 1024 512 256 128 64 64 32 32 --batch-size 8` for the
padding-efficiency case. Each report includes raw CUDA-event samples, logical
and padded token throughput, median and P95 latency, peak allocated and
reserved memory, startup time, temperatures and clocks before and after the
run, and the complete accelerator and software fingerprint.

Compare an immutable baseline with a new run using:

```bash
python -m benchmarks.regression current.json baseline.json \
  --output artifacts/benchmarks/gate.json
```

The command exits nonzero when it finds a regression. It does not rewrite the
baseline. A speed claim requires a one-sided lower confidence bound of at least
five percent improvement.
