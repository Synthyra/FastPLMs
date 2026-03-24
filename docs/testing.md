# Testing & Benchmarking

FastPLMs uses pytest with Docker for all GPU testing. Tests cover model loading, attention backend consistency, weight/forward compliance against official implementations, embedding stability, and throughput benchmarking.

## Docker Setup

### Build

```bash
docker build -t fastplms .
```

The Dockerfile includes:
- CUDA 12.8, cuDNN, Python 3.12
- PyTorch 2.11.0 with CUDA 12.8 wheels
- Official reference repos installed via `pip install -e` from `official/` submodules
- All `requirements.txt` dependencies

### Run Tests

```bash
# Fast tests (small models, no compliance, no structure)
docker run --gpus all fastplms python -m pytest /app/testing/ -m "gpu and not slow and not large and not structure" -v

# All sequence model tests except 3B
docker run --gpus all fastplms python -m pytest /app/testing/ -m "not large and not structure" -v

# Full suite including 3B models (requires 40+ GB VRAM)
docker run --gpus all fastplms python -m pytest /app/testing/ -m "not structure" -v

# Structure models only (Boltz2, ESMFold)
docker run --gpus all fastplms python -m pytest /app/testing/ -m "structure" -v

# Everything
docker run --gpus all fastplms python -m pytest /app/testing/ -v

# Throughput benchmark (pytest, saves JSON/CSV/PNG)
docker run --gpus all -v ${PWD}:/workspace fastplms python -m pytest /app/testing/test_throughput.py -v -s

# Throughput benchmark (standalone, more configurable)
docker run --gpus all -v ${PWD}:/workspace fastplms python -m testing.throughput \
    --model_paths Synthyra/ESM2-8M Synthyra/ESMplusplus_small \
    --backends sdpa flex kernels_flash \
    --batch_sizes 2 4 8 \
    --sequence_lengths 64 128 256 512 1024 2048

# Interactive shell
docker run --gpus all -v ${PWD}:/workspace -it fastplms bash
```

On Windows, replace `${PWD}` with `$(pwd)`.

## Pytest Markers

| Marker | Description | VRAM |
|--------|-------------|------|
| `gpu` | Requires CUDA GPU | Varies |
| `slow` | Loads two models simultaneously (compliance tests) | 2x model size |
| `large` | 3B parameter models | 24+ GB |
| `structure` | Structure prediction models (Boltz2, ESMFold) | 8+ GB |

Use `-m` to filter:
```bash
# Only compliance tests
python -m pytest /app/testing/ -m slow -v

# Exclude large models
python -m pytest /app/testing/ -m "not large" -v

# Only a specific model family
python -m pytest /app/testing/ -k esm2 -v
```

## Test File Map

| File | What it Tests | Markers |
|------|---------------|---------|
| `test_automodel.py` | Model loading + forward pass validity (no NaN/Inf) | `gpu` |
| `test_backend_consistency.py` | SDPA, Flex, Flash backends produce equivalent predictions (>= 95% agreement) | `gpu` |
| `test_compliance.py` | Weight parity (bit-exact) and forward pass compliance against official implementations | `slow`, `gpu` |
| `test_embedding_mixin.py` | NaN stability, batch-vs-single match, FASTA parsing, DPLM2 utilities | `gpu` |
| `test_throughput.py` | Throughput benchmark across models/backends/batch sizes; saves JSON/CSV/PNG | `slow`, `gpu` |
| `test_structure_models.py` | Boltz2 and ESMFold loading + forward pass | `structure`, `slow`, `gpu` |

Each test file has both **default registry** tests (5 small models for fast CI) and **full registry** tests (all 16+ checkpoints with size-based markers).

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

### Full Registry (`FULL_MODEL_REGISTRY`)

Used by the `test_full_*` parametrized tests. All checkpoints with `size_category`:

| Category | Models | Marker |
|----------|--------|--------|
| `small` | ESM2-8M, ESM2-35M, E1-150M, DPLM-150M, DPLM2-150M | (none) |
| `medium` | ESM2-150M, ESMC-small, E1-300M | `slow` |
| `large` | ESM2-650M, ESMC-large, E1-600M, DPLM-650M, DPLM2-650M | `slow` |
| `xlarge` | ESM2-3B, DPLM-3B, DPLM2-3B | `large` |

### Structure Registry (`STRUCTURE_MODEL_REGISTRY`)

| Key | Model |
|-----|-------|
| `boltz2` | Synthyra/Boltz2 |
| `esmfold` | Synthyra/FastESMFold |

## Compliance Testing

Compliance tests verify that FastPLM implementations are faithful to the originals.

### Architecture

Each model family has a corresponding module in `testing/official/` (e.g., `testing/official/esm2.py`) that provides:

```python
def load_official_model(reference_repo_id, device, dtype):
    # Returns (wrapped_official_model, tokenizer)
    ...
```

The returned model is wrapped so its forward pass matches FastPLM's interface (returns `.logits`, `.hidden_states`).

### Weight Compliance

`test_weight_compliance` loads both models in float32 and compares state dicts via `fastplms.weight_parity_utils.assert_state_dict_equal()`. This checks that every parameter is bit-exact (MSE == 0.0).

**DPLM2 is excluded** because the official model has an extra `contact_head` not present in the FastPLM version.

### Forward Compliance

`test_forward_compliance` runs 25 random batches (batch_size=8, seq_len 16-128) through both models in bfloat16 and compares:

- **Logits MSE** threshold: 0.05 (bfloat16 accumulates rounding across layers)
- **Prediction accuracy** threshold: 0.90 (argmax agreement on non-pad tokens)

Per-layer hidden state MSE is tracked for debugging when thresholds are exceeded.

### Compliance Dependencies

| Dependency | Models | Install |
|------------|--------|---------|
| `esm` | ESMC | `pip install -e official/boltz` (includes esm) |
| `E1` | E1 | `pip install -e official/e1` |
| `transformers` (official) | ESM2, DPLM | `pip install -e official/transformers` |

Official repos live in the `official/` directory as git submodules and are installed via `pip install -e` in the Dockerfile. If a dependency is missing, those tests are skipped.

## Throughput Benchmarking

### Pytest Test (`test_throughput.py`)

Benchmarks multiple model families across all 3 backends, batch sizes, and sequence lengths. Saves structured results:

- `throughput_results.json`: Machine-readable results
- `throughput_results.csv`: Spreadsheet-friendly format
- `throughput_comparison.png`: Visualization plot

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

### How Throughput is Measured

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
