# Architecture Overview

FastPLMs provides optimized, HuggingFace-compatible implementations of protein language models (PLMs) with pluggable attention backends.

## Repository Layout

```
FastPLMs/
  esm2/                    # ESM2 (Meta AI)
  esm_plusplus/             # ESM++ / ESMC (EvolutionaryScale)
  e1_fastplms/             # E1 (Profluent Bio)
  dplm_fastplms/           # DPLM (ByteDance)
  dplm2_fastplms/          # DPLM2 (ByteDance)
  boltz_fastplms/          # Boltz2 (structure prediction)
  esmfold/                 # ESMFold (structure prediction)
  embedding_mixin.py       # Shared pooling & embedding utilities
  entrypoint_setup.py      # PyTorch runtime config
  testing/                 # Test suite + benchmarks
  docs/                    # Documentation
  readmes/                 # Per-model HuggingFace card READMEs
```

Each model family lives in its own package directory containing:

| File | Purpose |
|------|---------|
| `modeling_*.py` | HuggingFace-compatible `PreTrainedModel` + `PretrainedConfig` subclasses |
| `get_*_weights.py` | Script to convert official checkpoints to FastPLM format |
| `load_official.py` | Loads the original reference model for compliance testing |
| `__init__.py` | Package init (often minimal; models load via `trust_remote_code`) |

## How Model Loading Works

All FastPLMs models are distributed on the [HuggingFace Hub](https://huggingface.co/Synthyra) and loaded with `trust_remote_code=True`:

```python
from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained(
    "Synthyra/ESM2-150M",
    trust_remote_code=True,
)
```

When `trust_remote_code=True` is passed, HuggingFace downloads the `modeling_*.py` file from the Hub repo and executes it locally. The Hub copy is kept in sync with the canonical copy in this repository via `update_HF.py`.

The model's `config.json` on the Hub contains an `auto_map` entry that tells `AutoModel` which class to instantiate:

```json
{
  "auto_map": {
    "AutoConfig": "modeling_fastesm.FastEsmConfig",
    "AutoModelForMaskedLM": "modeling_fastesm.FastEsmForMaskedLM"
  }
}
```

## EmbeddingMixin

Every sequence model (ESM2, ESM++, E1, DPLM, DPLM2) inherits from `EmbeddingMixin` (`embedding_mixin.py`), which provides:

- `embed_dataset()`: Batch embedding pipeline with pooling, SQLite/pth storage, FASTA parsing, and deduplication
- `_embed()`: Abstract method implemented by each model to return last hidden states
- `load_embeddings_from_pth()` / `load_embeddings_from_db()`: Load previously saved embeddings

The mixin supports two modes:

1. **Tokenizer mode** (ESM2, ESM++, DPLM, DPLM2): The caller provides a tokenizer; `_embed(input_ids, attention_mask)` is called
2. **Sequence mode** (E1): The caller passes `tokenizer=None`; `_embed(sequences, return_attention_mask=True)` is called, which returns `(embeddings, mask)`

See [Embedding & Pooling API](embedding_api.md) for full details.

## Attention Backend System

All models share a common attention backend abstraction controlled by `config.attn_backend`. Four backends are available:

| Backend | Key | Numerics | Speed |
|---------|-----|----------|-------|
| PyTorch SDPA | `"sdpa"` | Exact | Fast |
| Flash Attention | `"kernels_flash"` | Approximate | Fastest |
| Flex Attention | `"flex"` | Near-exact | Very fast |
| Auto | `"auto"` | Varies | Best available |

Each model's attention layer stores an `AttentionBackend` enum and dispatches accordingly. See [Attention Backends](attention_backends.md) for implementation details.

**Backend setting differs by model family:**

- ESM2, ESM++, E1: Set on the config before calling `from_pretrained`
- DPLM, DPLM2: Expose a mutable `model.attn_backend` property that propagates to all layers

## Entrypoint Setup

`entrypoint_setup.py` configures PyTorch runtime defaults for optimal GPU performance:

- TensorFloat32 matmul precision (`torch.set_float32_matmul_precision('high')`)
- TF32 enabled for matmul and cuDNN
- cuDNN autotuner (`benchmark=True`)
- Deterministic mode off for speed
- Inductor max autotune GEMM backends (ATEN, CUTLASS, FBGEMM)
- Dynamo scalar output capture and recompile limit

This module is imported at the top of standalone scripts (`throughput.py`, `compliance.py`) but is not imported by the model files themselves.

## Docker Layout

The Dockerfile uses:

- **Base image**: `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04` with Python 3.12
- **Source code**: Copied to `/app` (PYTHONPATH=/app)
- **Runtime workdir**: `/workspace` for outputs, caches, and volume mounts
- **Caches**: `HF_HOME=/workspace/.cache/huggingface`, `TORCH_HOME=/workspace/.cache/torch`
- **Compliance deps**: E1 (from git) and `esm` package pre-installed

## Weight Conversion

Each model family has a `get_*_weights.py` script that:

1. Loads the official checkpoint (from HuggingFace or a local file)
2. Remaps parameter names and shapes to match the FastPLM architecture
3. Exports `config.json`, `pytorch_model.bin`, and the modeling source files
4. The exported directory can be pushed to HuggingFace via `update_HF.py`

## Compliance Testing

Each family also has a `load_official.py` that wraps the original model in a standardized interface returning `(model, tokenizer)`. This allows the compliance test suite to load both implementations side-by-side and compare:

- **Weight parity**: Bit-exact MSE comparison of state dicts
- **Forward compliance**: Logits MSE and prediction accuracy across random batches

See [Testing & Benchmarking](testing.md) for details on running compliance tests.
