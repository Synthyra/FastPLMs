# Testing & Benchmarking

FastPLMs uses pytest with Docker for all GPU testing. Tests cover model loading, attention backend consistency, weight/forward parity against official implementations, embedding stability, and throughput benchmarking.

**Requires PyTorch 2.11+**. Flex attention uses Flash Attention 4 (FA4) as its backend on Hopper/Blackwell GPUs. The Dockerfiles pin PyTorch 2.11.0 with CUDA 12.8.

## Docker Layout

Two Docker layouts are supported.

### Per-family images (recommended for parity / compliance work)

A shared base image plus one image per model family. Each family image installs only that family's native reference deps, so we can run e.g. ESM++ tests against EvolutionaryScale's `esm` package without breaking ESM2 tests that depend on `fair-esm` / `transformers.EsmForMaskedLM`.

| Image tag | Native reference package |
|-----------|---------------------------|
| `fastplms-base` | none (torch 2.11.0, transformers 4.57.6, FastPLMs source, shared deps) |
| `fastplms-esm2` | uses `transformers.EsmForMaskedLM` |
| `fastplms-esm_plusplus` | EvolutionaryScale `esm` runtime deps + `official/esm` submodule on `sys.path`. The `esm` package itself is **not** pip-installed (it pins `transformers<4.53.0`). |
| `fastplms-e1` | `pip install -e /app/official/e1` |
| `fastplms-dplm` | uses `transformers.EsmForMaskedLM` (DPLM's native package conflicts with our torchtext pin) |
| `fastplms-dplm2` | none beyond base |
| `fastplms-ankh` | uses `transformers.T5EncoderModel` |

Build:

```bash
git submodule update --init --recursive

# Build base + every family image
./build_images.sh

# Build a specific subset
./build_images.sh esm2 esm_plusplus
```

`build_images.sh` always builds `fastplms-base` first and then layers each requested family on top, with `--cache-from` chained so dep changes don't invalidate the base.

### Monolithic image (legacy)

The original `Dockerfile` (image tag `fastplms`) bundles everything compatible into a single image. Used by the broad test suites that don't need per-family isolation.

```bash
git submodule update --init --recursive
docker build -t fastplms .
```

## Running Tests

**Always pass `--ipc=host`** with PyTorch, otherwise multi-worker DataLoader and CUDA can deadlock.

### Per-family parity / compliance

```bash
# ESM2 -- model_key in conftest.py is "esm2"
docker run --rm --gpus all --ipc=host -v $(pwd):/workspace fastplms-esm2 \
    python -m pytest /workspace/testing/test_parity.py -k esm2 -v

# ESM++ -- model_key is "esmc"
docker run --rm --gpus all --ipc=host -v $(pwd):/workspace fastplms-esm_plusplus \
    python -m pytest /workspace/testing/test_parity.py -k esmc -v

# Everything else
for fam in e1 dplm dplm2 ankh; do
    docker run --rm --gpus all --ipc=host -v $(pwd):/workspace fastplms-$fam \
        python -m pytest /workspace/testing/test_parity.py -k $fam -v
done
```

### Broader suites in the monolithic image

```bash
# Fast tests (small models, no compliance, no structure)
docker run --gpus all --ipc=host fastplms python -m pytest /app/testing/ -m "gpu and not slow and not large and not structure" -v

# All sequence model tests except 3B
docker run --gpus all --ipc=host fastplms python -m pytest /app/testing/ -m "not large and not structure" -v

# Full suite including 3B models (requires 40+ GB VRAM)
docker run --gpus all --ipc=host fastplms python -m pytest /app/testing/ -m "not structure" -v

# Structure models only (Boltz2, ESMFold)
docker run --gpus all --ipc=host fastplms python -m pytest /app/testing/ -m "structure" -v

# Throughput benchmark (saves JSON/CSV/PNG)
docker run --gpus all --ipc=host -v ${PWD}:/workspace fastplms python -m pytest /app/testing/test_throughput.py -v -s

# Throughput benchmark, standalone, more configurable
docker run --gpus all --ipc=host -v ${PWD}:/workspace fastplms python -m testing.throughput \
    --model_paths Synthyra/ESM2-8M Synthyra/ESMplusplus_small \
    --backends sdpa flex kernels_flash \
    --batch_sizes 2 4 8 \
    --sequence_lengths 64 128 256 512 1024 2048

# Interactive shell
docker run --gpus all --ipc=host -v ${PWD}:/workspace -it fastplms bash
```

On Windows, replace `${PWD}` with `$(pwd)`.

## Pytest Markers

| Marker | Description | VRAM |
|--------|-------------|------|
| `gpu` | Requires CUDA GPU | Varies |
| `slow` | Loads two models simultaneously (compliance tests) | 2x model size |
| `large` | 3B parameter models | 24+ GB |
| `structure` | Structure prediction models (Boltz2, ESMFold) | 8+ GB |

Use `-m` to filter and `-k` to select by name:

```bash
# Only compliance tests
python -m pytest /workspace/testing/ -m slow -v

# Exclude large models
python -m pytest /workspace/testing/ -m "not large" -v

# Only a specific model family
python -m pytest /workspace/testing/ -k esm2 -v
```

## Test File Map

| File | What it tests | Markers |
|------|---------------|---------|
| `test_parity.py` | **Rigorous parity** vs official implementations: tokenizer parity, bit-exact weight parity, per-layer relative-std hidden-state diff (fp32 + bf16) across `single`/`uniform`/`skewed` padding scenarios, padding-isolation (`[short]` alone vs `[short, long_]` padded), backend consistency, end-to-end `embed_dataset` pipeline parity. Runs per family in its own Docker image. | `gpu` |
| `test_automodel.py` | Model loading + forward pass validity (no NaN/Inf) | `gpu` |
| `test_backend_consistency.py` | SDPA, Flex, Flash backends produce equivalent predictions (>= 95% agreement) | `gpu` |
| `test_compliance.py` | Original (looser, bf16-only) weight/forward compliance against official implementations. Kept as a smoke layer; `test_parity.py` is the source of truth. | `slow`, `gpu` |
| `test_embedding_mixin.py` | NaN stability, batch-vs-single match, FASTA parsing, DPLM2 utilities | `gpu` |
| `test_throughput.py` | Throughput benchmark across models/backends/batch sizes; saves JSON/CSV/PNG | `slow`, `gpu` |
| `test_structure_models.py` | Boltz2 and ESMFold loading + forward pass | `structure`, `slow`, `gpu` |

Each test file has both **default registry** tests (one small model per family for fast CI) and **full registry** tests (all 21+ checkpoints with size-based markers).

## Model Registries

### Default Registry (`MODEL_REGISTRY`)

Used by the base parametrized tests. One small model per family:

| Key | Model | Family |
|-----|-------|--------|
| `esm2` | ESM2-8M | ESM2 |
| `esmc` | ESMplusplus_small | ESM++ |
| `e1` | Profluent-E1-150M | E1 |
| `dplm` | DPLM-150M | DPLM |
| `dplm2` | DPLM2-150M | DPLM2 |
| `ankh` | ANKH_base | ANKH |

### Full Registry (`FULL_MODEL_REGISTRY`)

Used by the `test_full_*` parametrized tests. All checkpoints with `size_category`:

| Category | Models | Marker |
|----------|--------|--------|
| `small` | ESM2-8M, ESM2-35M, E1-150M, DPLM-150M, DPLM2-150M | (none) |
| `medium` | ESM2-150M, ESMC-small, E1-300M, ANKH-base | `slow` |
| `large` | ESM2-650M, ESMC-large, E1-600M, DPLM-650M, DPLM2-650M, ANKH-large, ANKH2-large, ANKH3-large | `slow` |
| `xlarge` | ESM2-3B, DPLM-3B, DPLM2-3B, ANKH3-xl | `large` |

### Structure Registry (`STRUCTURE_MODEL_REGISTRY`)

| Key | Model |
|-----|-------|
| `boltz2` | Synthyra/Boltz2 |
| `esmfold` | Synthyra/FastESMFold |

## Parity Testing (`test_parity.py`)

The parity suite is the source of truth for "FastPLMs matches the official implementation." It is intentionally strict: when it passes, FastPLMs and the native model agree at every layer to documented numerical tolerance, including under variable-length padded batches.

### What each test checks

| Test | Checks |
|------|--------|
| `test_tokenizer_parity` | Vocab size, every token id, every special token id (`pad`, `cls`, `eos`, `mask`, `unk`) match exactly. |
| `test_weight_parity_fp32` | Per-parameter bit-exact equality. Expected extras (e.g. ANKH's `lm_head.weight`) are allowlisted. |
| `test_forward_parity_fp32` | Per-layer relative-std-of-diff (`std(fast - native) / std(native)`), `last_hidden_state` MSE/maxabs, logits MSE. Parametrized over `single`/`uniform`/`skewed` padding scenarios. |
| `test_forward_parity_bf16` | Same as fp32 with documented per-family tolerances. |
| `test_padding_does_not_pollute_valid_positions_fp32` | Runs `[short]` alone and `[short, long_]` padded; asserts the short sequence's valid-position output matches across both. Catches mask-handling bugs that uniform-length tests miss. |
| `test_backend_consistency_fp32` | SDPA vs `kernels_flash` vs `flex` on FastPLMs side, against SDPA as ground truth. |
| `test_embed_dataset_pipeline_parity` | End-to-end `embed_dataset()` vs manual native forward + mean-pool. This is what downstream users actually call. |

### Tolerances

Per-family tolerances live in `FAMILY_TOLERANCES` at the top of `test_parity.py`. fp32 tolerances are tight (machine precision); bf16 tolerances are looser per-family because accumulated rounding scales with depth (ESMC has 30 layers, ANKH-base has 48, ESM2-8M has 6).

### Adding a new family

1. Add a registry entry in `testing/conftest.py` (`MODEL_REGISTRY` and `FULL_MODEL_REGISTRY`).
2. Implement `testing/official/<family>.py` exporting `load_official_model(reference_repo_id, device, dtype)` that returns `(wrapped_model, tokenizer)` with `.forward()` returning `.logits`, `.hidden_states`.
3. Add a `Dockerfile.<family>` that installs the family's native deps on top of `fastplms-base`, and add it to `build_images.sh`.
4. Add a `ParityTolerances(...)` entry in `FAMILY_TOLERANCES` with reasonable starting values, then tighten as you investigate failures.

## Compliance Testing (`test_compliance.py`)

Older, looser test layer kept for backward compatibility. Compares FastPLM and official outputs in bf16 with MSE < 0.05 and prediction accuracy > 0.90. Use `test_parity.py` instead for new work.

DPLM2 is excluded from weight compliance because the official model has an extra `contact_head` not present in the FastPLM version.

## Throughput Benchmarking

### Pytest Test (`test_throughput.py`)

Benchmarks multiple model families across all 3 backends, batch sizes, and sequence lengths. Saves structured results:

- `throughput_results.json`: machine-readable
- `throughput_results.csv`: spreadsheet-friendly
- `throughput_comparison.png`: visualization plot

The pytest test uses fewer timed batches (25 vs 100) for faster execution.

### Standalone Script (`testing/throughput.py`)

More configurable, with CLI arguments:

```bash
python -m testing.throughput \
    --model_paths Synthyra/ESM2-8M Synthyra/ESMplusplus_small \
    --backends sdpa flex kernels_flash \
    --batch_sizes 2 4 8 \
    --sequence_lengths 64 128 256 512 1024 2048 \
    --warmup_batches 10 \
    --timed_batches 100 \
    --output_path /workspace/throughput_comparison.png
```

Both pytest and standalone output JSON and CSV in addition to the plot.

### How throughput is measured

1. Model is compiled via `torch.compile()`
2. Dynamic warmup: 10-100 batches until latency stabilizes (relative change <= 10% over a 3-batch window)
3. Timed phase: N batches with `torch.cuda.synchronize()` around the timing loop
4. Reports non-padding tokens per second

### Boltz2 Compliance

Boltz2 has its own compliance script (`testing/run_boltz2_compliance.py`) that compares:
- Coordinate MAE/RMSE (raw and Kabsch-aligned)
- Pairwise distance MAE
- TM-score comparison

```bash
python -m testing.run_boltz2_compliance \
    --device cuda \
    --dtype float32 \
    --seed 42 \
    --num-sequences 3 \
    --recycling-steps 3 \
    --num-sampling-steps 200
```
