# H100 benchmarks

Benchmarks run independently from pytest. The steady-state path times only a
forward pass over pre-tokenized tensors already resident on the GPU. Startup,
compilation, end-to-end embedding, and the ESMFold2 projection are distinct
modes.

Run the complete manifest-derived H100 matrix with:

```bash
python -m benchmarks.suite --output artifacts/benchmarks/h100.json
```

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
