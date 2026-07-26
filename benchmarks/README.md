# Hopper/SM90 benchmarks

Benchmarks run independently from pytest. The steady-state path times only a
forward pass over pre-tokenized tensors already resident on the GPU. Startup,
compilation, end-to-end embedding, and the ESMFold2 projection are distinct
modes.

Run the complete manifest-derived release matrix on the current NVIDIA GH200
validation workstation in its exact containerized Linux aarch64 environment:

```bash
python -m benchmarks.suite \
  --backends eager sdpa flex_attention \
  --junit-output artifacts/junit/benchmark.xml \
  --output artifacts/benchmarks/h100.json
```

The output name is a legacy automation identifier. Every report carries the
actual accelerator, architecture, and software fingerprint, and regression
comparisons require an exact match. H100 and H200 remain supported Hopper-class
devices, but are not interchangeable with or accepted as the current
GH200/aarch64 release evidence.
Remote orchestration binds Bake to native `linux/arm64` on the GH200 and verifies
the loaded image architecture and content digest rather than relying on
emulated `linux/amd64` images.

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

Artifact mode validates the complete selected artifacts and ESMFold2's local
ESMC dependency before loading. Reports retain registry repository/revision
case keys and path-free weights, runtime, source, canonical-state, and manifest
identities. Local filesystem paths are never baseline identities.
The GH200 release runner records FA2 as prior focused evidence and FA3 as
unavailable on linux/arm64; it never downloads, builds, or executes either
Flash kernel. Capture reports include the exact environment/artifact identities
needed for mechanical baseline promotion and keep cold compile time separate
from warm throughput blocks.

The matrix includes startup, compilation, full embedding, `b=1, l=512`
latency, `b=8, l=1024` throughput, the fixed skewed-padding case, and BF16/FP8
ESMFold2 projection measurements. Use `--baseline <path>` to apply the
one-sided 95% regression gates without modifying the baseline.

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

Build the manifest-pinned local artifact first. If a published Hub artifact is
used instead, pass its immutable revision and retain it in the report.

Use `--lengths 1024 512 256 128 64 64 32 32 --batch-size 8` for the
padding-efficiency case. Every report includes raw CUDA-event samples, logical
and padded token throughput, median and P95 latency, peak allocated and
reserved memory, startup time, before/after temperatures and clocks, and the
complete accelerator/software fingerprint.

Compare an immutable baseline with a new run using:

```bash
python -m benchmarks.regression current.json baseline.json \
  --output artifacts/benchmarks/gate.json
```

The command exits nonzero when a regression is established. It never rewrites
the baseline. A speed claim is supported only when the one-sided lower
confidence bound shows at least a five-percent improvement.
