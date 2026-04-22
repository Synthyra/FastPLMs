# Architecture Overview

FastPLMs provides optimized, HuggingFace-compatible implementations of protein language models (PLMs) with pluggable attention backends.

## Repository Layout

```
FastPLMs/
  fastplms/                  # Main package
    ankh/                    # ANKH (Elnaggar Lab)
    boltz/                   # Boltz2 (structure prediction)
    dplm/                    # DPLM (ByteDance)
    dplm2/                   # DPLM2 (ByteDance)
    e1/                      # E1 (Profluent Bio)
    esm2/                    # ESM2 (Meta AI)
    esm_plusplus/            # ESM++ / ESMC (EvolutionaryScale)
    esmfold/                 # ESMFold (structure prediction)
    attention.py             # Shared attention backend code
    embedding_mixin.py       # Shared pooling & embedding utilities
    weight_parity_utils.py   # Weight comparison utilities
    fine_tuning_example.py   # LoRA fine-tuning example
  official/                  # Official reference repos (git submodules)
    boltz/                   # Official Boltz
    e1/                      # Official E1
    dplm/                    # Official DPLM
    esm/                     # Official EvolutionaryScale ESM (sys.path-injected, not pip-installed)
    transformers/            # Official HF transformers
  entrypoint_setup.py        # PyTorch runtime config
  testing/                   # Test suite + benchmarks
    official/                # Official model loaders for compliance / parity
    test_parity.py           # Rigorous per-family parity suite
  Dockerfile                 # Monolithic image (legacy)
  Dockerfile.base            # Shared base image (per-family layout)
  Dockerfile.<family>        # One per family: esm2, esm_plusplus, e1, dplm, dplm2, ankh
  build_images.sh            # Builds base + selected family images
  update_HF.py               # Pushes composite modeling files + weights to HF Hub
  docs/                      # Documentation
```

Each model family lives in its own package directory containing:

| File | Purpose |
|------|---------|
| `modeling_*.py` | HuggingFace-compatible `PreTrainedModel` + `PretrainedConfig` subclasses |
| `get_*_weights.py` | Script to convert official checkpoints to FastPLM format |
| `README.md` | Per-model HuggingFace card README |
| `LICENSE` | Per-model license file |
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

Every sequence model (ESM2, ESM++, E1, DPLM, DPLM2) inherits from `EmbeddingMixin` (`fastplms/embedding_mixin.py`), which provides:

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

**Backend setting is uniform across families:**

- At load time, every family accepts `config.attn_backend = "..."` before `from_pretrained`.
- At runtime, every family exposes a mutable `model.attn_backend` property whose setter propagates to every attention submodule. Use this to benchmark backends on the same weights without reloading.
- Exception: ANKH silently resolves `kernels_flash` to `flex` (or `sdpa`), because T5 relative position bias is incompatible with the flash kernels.

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

There are two coexisting layouts.

### Per-family layout (recommended)

A shared base image plus one image per model family. This isolates conflicting native deps (notably EvolutionaryScale `esm` vs `fair-esm`, and DPLM's torchtext pin) so each family can be tested against its own native reference without breaking the others.

- `Dockerfile.base` produces `fastplms-base`: CUDA 12.8, Python 3.12, PyTorch 2.11.0, transformers, FastPLMs source at `/app`. No native reference packages.
- `Dockerfile.<family>` (esm2, esm_plusplus, e1, dplm, dplm2, ankh) layers on top of `fastplms-base` and installs only that family's native reference deps.
- `build_images.sh` is a convenience script that builds the base then any subset of family images.

`testing/official/<family>.py` provides the `load_official_model(...)` wrapper that the parity tests call. For ESM++, the EvolutionaryScale `esm` package itself is **not** pip-installed (it pins `transformers<4.53.0`); instead `testing/official/__init__.py` injects the in-tree `official/esm` submodule onto `sys.path` at import time.

### Monolithic layout (legacy)

The original top-level `Dockerfile` (image tag `fastplms`) bundles every dep that can coexist into a single image. Used by the broad test suites and throughput benchmarks where per-family isolation isn't needed.

### Common environment

Both layouts use:

- **Base image**: `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04` with Python 3.12
- **Source code**: Copied to `/app` (`PYTHONPATH=/app`)
- **Runtime workdir**: `/workspace` for outputs, caches, and volume mounts
- **Caches**: `HF_HOME=/workspace/.cache/huggingface`, `TORCH_HOME=/workspace/.cache/torch`

## Weight Conversion

Each model family has a `get_*_weights.py` script that:

1. Loads the official checkpoint (from HuggingFace or a local file)
2. Remaps parameter names and shapes to match the FastPLM architecture
3. Exports `config.json`, `pytorch_model.bin`, and the modeling source files
4. The exported directory can be pushed to HuggingFace via `update_HF.py`

## Parity & Compliance Testing

Each family has a corresponding module in `testing/official/` (e.g., `testing/official/esm2.py`) that wraps the original model in a standardized interface returning `(model, tokenizer)`. The parity / compliance suites load both implementations side-by-side and compare:

- **Tokenizer parity** (`test_parity.py`): vocab, every token id, every special token id
- **Weight parity**: bit-exact equality of every parameter, with family-specific allowlisted extras (e.g. ANKH's `lm_head.weight`, since native is a T5 encoder without a head)
- **Forward parity (fp32 + bf16)**: per-layer relative-std AND relative-maxabs hidden-state diff (two complementary metrics so a localized regression can't hide inside a collapsed scalar), `last_hidden_state` absolute + relative maxabs, logits MSE, padding-isolation across SDPA and Flex, end-to-end `embed_dataset` pipeline for every family
- **Backend consistency**: every family's supported backends (typically SDPA vs Flex vs `kernels_flash`) agree on the fast side to per-backend tolerance; ANKH compares SDPA vs Flex only because kernels_flash silently falls back
- **Backend setter propagation**: `model.attn_backend = X` actually updates every attention submodule. If this regresses, every backend-parametrized test becomes a no-op.

The parity suite runs per family in its own Docker image (per the per-family layout above). See [Testing & Benchmarking](testing.md) for full details.
