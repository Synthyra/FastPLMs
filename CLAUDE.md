# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

All tests run in Docker. Never run tests natively on Windows (missing Triton, flash-attention, CUDA kernels).

```bash
# Initialize submodules (required before Docker build)
git submodule update --init --recursive

# Build the Docker image
docker build -t fastplms .

# Fast tests (small models, no compliance)
docker run --gpus all fastplms python -m pytest /app/testing/ -m "gpu and not slow and not large and not structure" -v

# All tests except 3B models and structure prediction
docker run --gpus all fastplms python -m pytest /app/testing/ -m "not large and not structure" -v

# Single model family
docker run --gpus all fastplms python -m pytest /app/testing/ -k esm2 -v

# Single test file
docker run --gpus all fastplms python -m pytest /app/testing/test_automodel.py -v

# Throughput benchmark (saves JSON/CSV/PNG)
docker run --gpus all -v $(pwd):/workspace fastplms python -m pytest /app/testing/test_throughput.py -v -s

# Everything (requires 40+ GB VRAM for 3B models)
docker run --gpus all fastplms python -m pytest /app/testing/ -v
```

Pytest markers: `gpu`, `slow` (loads two models), `large` (3B models, 24+ GB VRAM), `structure` (Boltz2/ESMFold).

## Architecture

### Model Families

Each family lives under `fastplms/`: `fastplms/esm2/`, `fastplms/esm_plusplus/`, `fastplms/e1/`, `fastplms/dplm/`, `fastplms/dplm2/`, `fastplms/boltz/`, `fastplms/esmfold/`. Each contains:
- `modeling_*.py` -- HuggingFace `PreTrainedModel` + `PretrainedConfig` subclasses (also pushed to HF Hub for `trust_remote_code`)
- `get_weights.py` -- converts official checkpoints to FastPLM format

Official model loaders for compliance testing live in `testing/official/{family}.py`.

All sequence models load via `AutoModelForMaskedLM.from_pretrained("Synthyra/...", trust_remote_code=True)`.

### Two Input Modes

**Tokenizer mode** (ESM2, ESM++, DPLM, DPLM2): Standard HF tokenizer. Access via `model.tokenizer`. Forward: `_embed(input_ids, attention_mask)`.

**Sequence mode** (E1): No tokenizer. Uses `model.model.prep_tokens.get_batch_kwargs(sequences, device=device)` returning `input_ids`, `within_seq_position_ids`, `global_position_ids`, `sequence_ids`. Forward: `_embed(sequences, return_attention_mask=True)` returns `(embeddings, mask)`.

ESMC (ESM++) additionally requires `sequence_id = attention_mask.to(dtype=torch.bool)` in forward inputs.

### Attention Backends

All models share `config.attn_backend` with four options: `"sdpa"` (default, exact), `"kernels_flash"` (fastest, approximate), `"flex"` (near-exact, block masks), `"auto"` (best available). ESM2/ESM++/E1 set backend on config before `from_pretrained`; DPLM/DPLM2 expose a mutable `model.attn_backend` property.

### EmbeddingMixin

`fastplms/embedding_mixin.py` is inherited by every sequence model. Provides `embed_dataset()` for batch embedding with 8 pooling strategies, SQLite/pth storage, FASTA parsing, and deduplication. `Pooler` class handles mean/max/cls/norm/median/std/var/parti pooling.

### Test Registries

`testing/conftest.py` defines:
- `MODEL_REGISTRY`: 5 small models (one per family) for fast CI
- `FULL_MODEL_REGISTRY`: 16 checkpoints with `size_category` (small/medium/large/xlarge)
- `STRUCTURE_MODEL_REGISTRY`: Boltz2, ESMFold (different API, not MaskedLM)
- Shared helpers: `tokenize_batch()`, `add_model_specific_inputs()`, `mark_by_size()`

### Compliance Testing

Each family's `testing/official/{family}.py` returns `(wrapped_model, tokenizer)` with a standardized interface. Tests compare weight parity (bit-exact MSE) and forward pass (logits MSE < 0.05, pred accuracy > 0.90 in bfloat16). DPLM2 is excluded from compliance (official has extra `contact_head`).

### Docker Layout

Source at `/app` (PYTHONPATH=/app), runtime at `/workspace`. Caches under `/workspace/.cache/`. Base image: CUDA 12.8, Python 3.12, PyTorch 2.11.

### entrypoint_setup.py

Configures TF32 matmul, cuDNN benchmark, inductor autotune, and dynamo settings. Imported by standalone scripts (throughput, compliance) but not by model files.
