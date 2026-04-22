# FastPLMs

<img width="2816" height="1536" alt="FastPLMs Hero Image" src="https://github.com/user-attachments/assets/ffaf84b6-9970-40fd-aa31-1b314d6ca146" />

FastPLMs is an open-source initiative dedicated to making protein language models (pLMs) efficient and easy to use. By replacing native, often suboptimal attention implementations with **Flash Attention** or **Flex Attention**, we provide high-performance alternatives that are fully compatible with the HuggingFace `transformers` ecosystem and can easily be loaded with no extra code with `AutoModel`.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Documentation](#documentation)
3. [Supported Models](#supported-models)
4. [Attention Backends](#attention-backends)
5. [Embedding & Pooling](#embedding--pooling)
6. [Concrete Examples](#concrete-examples)
7. [Testing & Benchmarking](#testing--benchmarking)
8. [Installation & Docker](#installation--docker)

---

## Documentation

Detailed documentation is available in the [`docs/`](docs/) folder:

- [Architecture Overview](docs/architecture.md) - How FastPLMs wraps official models, the attention backend system, Docker layout
- [Per-Model Guides](docs/models.md) - Loading, configuration, and special handling for each model family
- [Attention Backends](docs/attention_backends.md) - SDPA, Flash, Flex, Auto: how they work, when to use each, numerical properties
- [Embedding & Pooling API](docs/embedding_api.md) - Pooler strategies, `embed_dataset()` parameters, SQLite/pth storage
- [Fine-Tuning Guide](docs/finetuning.md) - LoRA, Trainer patterns, dataset classes, metrics
- [Testing & Benchmarking](docs/testing.md) - Docker commands, pytest markers, compliance architecture, throughput benchmarks
- [Contributing](docs/contributing.md) - Code style, adding new models, required tests

---

## Introduction

### What are Protein Language Models (pLMs)?
Protein Language Models are transformer-based architectures trained on massive datasets of protein sequences (such as UniProt). These models learn the "grammar" of proteins, capturing evolutionary information, structural constraints, and functional motifs. They are used for:
- **Representation Learning**: Generating high-dimensional embeddings for downstream tasks (e.g., stability, function prediction).
- **Protein Generation**: Designing novel sequences with specific properties.
- **Structure Prediction**: Mapping sequences to their 3D folds (e.g., Boltz2).

### What is this repository?
FastPLMs provides optimized versions of these models. Our focus is on:
- **Speed**: Drastically faster inference through optimized attention kernels.
- **Memory Efficiency**: Lower VRAM usage, enabling larger batch sizes or longer sequences.
- **Seamless Integration**: Use `AutoModel.from_pretrained(..., trust_remote_code=True)` to load our optimized weights directly from HuggingFace.

---

## Supported Models

We maintain a comprehensive [HuggingFace Collection](https://huggingface.co/collections/Synthyra/pretrained-plms-675351ecc050f63baedd77de) of optimized models. Below is a summary of the supported families and their origins.

### Model Registry Summary

| Model Family | Organization | Official Implementation | FastPLMs Optimization | Checkpoints |
| :--- | :--- | :--- | :--- | :--- |
| **E1** | Profluent Bio | [Profluent-Bio/E1](https://github.com/Profluent-Bio/E1) | Flex Attention, Block-Causal | 150M, 300M, 600M |
| **ESM2** | Meta AI | [facebookresearch/esm](https://github.com/facebookresearch/esm) | Flash (SDPA) / Flex Attention | 8M, 35M, 150M, 650M, 3B |
| **ESM++** | EvolutionaryScale | [EvolutionaryScale/esm](https://github.com/evolutionaryscale/esm) | Optimized SDPA / Flex | Small (300M), Large (600M) |
| **DPLM** | ByteDance | [bytedance/dplm](https://github.com/bytedance/dplm) | Diffusion Optimized Attention | 150M, 650M, 3B |
| **DPLM2** | ByteDance | [bytedance/dplm](https://github.com/bytedance/dplm) | Multimodal Diffusion | 150M, 650M, 3B |
| **ANKH** | Elnaggar Lab | [ElnaggarLab/ankh](https://huggingface.co/ElnaggarLab/ankh-base) | T5 RPE via Flex score_mod | Base, Large, ANKH2-L, ANKH3-L, ANKH3-XL |
| **ESMFold** | Meta AI | [facebookresearch/esm](https://github.com/facebookresearch/esm) | ProteinTTT + Fast ESM2 backbone | Standard |
| **Boltz2** | MIT / Various | [jwohlwend/boltz](https://github.com/jwohlwend/boltz) | Optimized Structure Prediction | Standard |

### Full Model List

| Model Key | Family | Parameters | Organization | FastPLMs Repo ID | Official Reference |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `esm2_8m` | ESM2 | 7.5M | Meta AI | [Synthyra/ESM2-8M](https://huggingface.co/Synthyra/ESM2-8M) | [facebook/esm2_t6_8M_UR50D](https://huggingface.co/facebook/esm2_t6_8M_UR50D) |
| `esm2_35m` | ESM2 | 33.5M | Meta AI | [Synthyra/ESM2-35M](https://huggingface.co/Synthyra/ESM2-35M) | [facebook/esm2_t12_35M_UR50D](https://huggingface.co/facebook/esm2_t12_35M_UR50D) |
| `esm2_150m` | ESM2 | 148.2M | Meta AI | [Synthyra/ESM2-150M](https://huggingface.co/Synthyra/ESM2-150M) | [facebook/esm2_t30_150M_UR50D](https://huggingface.co/facebook/esm2_t30_150M_UR50D) |
| `esm2_650m` | ESM2 | 651.1M | Meta AI | [Synthyra/ESM2-650M](https://huggingface.co/Synthyra/ESM2-650M) | [facebook/esm2_t33_650M_UR50D](https://huggingface.co/facebook/esm2_t33_650M_UR50D) |
| `esm2_3b` | ESM2 | 2.84B | Meta AI | [Synthyra/ESM2-3B](https://huggingface.co/Synthyra/ESM2-3B) | [facebook/esm2_t36_3B_UR50D](https://huggingface.co/facebook/esm2_t36_3B_UR50D) |
| `esmplusplus_small` | ESM++ | 333.0M | EvolutionaryScale | [Synthyra/ESMplusplus_small](https://huggingface.co/Synthyra/ESMplusplus_small) | [EvolutionaryScale/esmc-300m](https://huggingface.co/EvolutionaryScale/esmc-300m-2024-12) |
| `esmplusplus_large` | ESM++ | 575.0M | EvolutionaryScale | [Synthyra/ESMplusplus_large](https://huggingface.co/Synthyra/ESMplusplus_large) | [EvolutionaryScale/esmc-600m](https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12) |
| `e1_150m` | E1 | 154.4M | Profluent Bio | [Synthyra/Profluent-E1-150M](https://huggingface.co/Synthyra/Profluent-E1-150M) | [Profluent-Bio/E1-150m](https://huggingface.co/Profluent-Bio/E1-150m) |
| `e1_300m` | E1 | 274.3M | Profluent Bio | [Synthyra/Profluent-E1-300M](https://huggingface.co/Synthyra/Profluent-E1-300M) | [Profluent-Bio/E1-300m](https://huggingface.co/Profluent-Bio/E1-300m) |
| `e1_600m` | E1 | 641.4M | Profluent Bio | [Synthyra/Profluent-E1-600M](https://huggingface.co/Synthyra/Profluent-E1-600M) | [Profluent-Bio/E1-600m](https://huggingface.co/Profluent-Bio/E1-600m) |
| `dplm_150m` | DPLM | 148.2M | ByteDance | [Synthyra/DPLM-150M](https://huggingface.co/Synthyra/DPLM-150M) | [airkingbd/dplm_150m](https://huggingface.co/airkingbd/dplm_150m) |
| `dplm_650m` | DPLM | 651.1M | ByteDance | [Synthyra/DPLM-650M](https://huggingface.co/Synthyra/DPLM-650M) | [airkingbd/dplm_650m](https://huggingface.co/airkingbd/dplm_650m) |
| `dplm_3b` | DPLM | 2.84B | ByteDance | [Synthyra/DPLM-3B](https://huggingface.co/Synthyra/DPLM-3B) | [airkingbd/dplm_3b](https://huggingface.co/airkingbd/dplm_3b) |
| `dplm2_150m` | DPLM2 | 158.7M | ByteDance | [Synthyra/DPLM2-150M](https://huggingface.co/Synthyra/DPLM2-150M) | [airkingbd/dplm2_150m](https://huggingface.co/airkingbd/dplm2_150m) |
| `dplm2_650m` | DPLM2 | 672.1M | ByteDance | [Synthyra/DPLM2-650M](https://huggingface.co/Synthyra/DPLM2-650M) | [airkingbd/dplm2_650m](https://huggingface.co/airkingbd/dplm2_650m) |
| `dplm2_3b` | DPLM2 | 2.88B | ByteDance | [Synthyra/DPLM2-3B](https://huggingface.co/Synthyra/DPLM2-3B) | [airkingbd/dplm2_3b](https://huggingface.co/airkingbd/dplm2_3b) |
| `ankh_base` | ANKH | 453.3M | Elnaggar Lab | [Synthyra/ANKH_base](https://huggingface.co/Synthyra/ANKH_base) | [ElnaggarLab/ankh-base](https://huggingface.co/ElnaggarLab/ankh-base) |
| `ankh_large` | ANKH | 1.15B | Elnaggar Lab | [Synthyra/ANKH_large](https://huggingface.co/Synthyra/ANKH_large) | [ElnaggarLab/ankh-large](https://huggingface.co/ElnaggarLab/ankh-large) |
| `ankh2_large` | ANKH | 1.15B | Elnaggar Lab | [Synthyra/ANKH2_large](https://huggingface.co/Synthyra/ANKH2_large) | [ElnaggarLab/ankh2-ext2](https://huggingface.co/ElnaggarLab/ankh2-ext2) |
| `ankh3_large` | ANKH | 1.15B | Elnaggar Lab | [Synthyra/ANKH3_large](https://huggingface.co/Synthyra/ANKH3_large) | [ElnaggarLab/ankh3-large](https://huggingface.co/ElnaggarLab/ankh3-large) |
| `ankh3_xl` | ANKH | 3.49B | Elnaggar Lab | [Synthyra/ANKH3_xl](https://huggingface.co/Synthyra/ANKH3_xl) | [ElnaggarLab/ankh3-xl](https://huggingface.co/ElnaggarLab/ankh3-xl) |
| `esmfold` | ESMFold | 3.53B | Meta AI | [Synthyra/FastESMFold](https://huggingface.co/Synthyra/FastESMFold) | [facebookresearch/esm](https://github.com/facebookresearch/esm) |
| `boltz2` | Boltz2 | 506.3M | MIT / Various | [Synthyra/Boltz2](https://huggingface.co/Synthyra/Boltz2) | [jwohlwend/boltz](https://github.com/jwohlwend/boltz) |

---

## Attention Backends

All FastPLMs models share a common set of attention backends, controlled via `config.attn_backend`. The default is `"sdpa"`, which is safe on all hardware and numerically equivalent to standard attention.

### Backend Comparison

| Backend | Key | Speed | Numerical Equivalence | Availability |
| :--- | :--- | :--- | :--- | :--- |
| PyTorch SDPA | `"sdpa"` | Fast | Exact | Any PyTorch ≥ 2.0 |
| Flash Attention | `"kernels_flash"` | Fastest | Approximate | Requires `pip install kernels` (pre-built) |
| Flex Attention | `"flex"` | Very fast | ~Exact | Requires PyTorch ≥ 2.11 (FA4 backend on Hopper/Blackwell) |
| Auto | `"auto"` | — | — | Always (selects best available) |

### SDPA (default)

PyTorch's [`scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) dispatches to a fused CUDA kernel (cuDNN or efficient attention) that is faster and more memory-efficient than naive attention, while being mathematically identical to it. This is the recommended default for reproducibility and general use. It is also the only backend where `output_attentions=True` is handled natively; with other backends, attentions are computed via a separate naive matrix multiplication when requested.

### Flash Attention (`kernels_flash`)

Flash Attention 2 and 3 are typically the fastest options on Ampere (A100) and Hopper (H100) GPUs, often 2–4× faster than SDPA at long sequence lengths. Flash Attention achieves this by tiling the computation and applying an online softmax, which means the results are **not bitwise identical** to SDPA or naive attention. Differences are on the order of floating-point rounding and are often inconsequential for standard inference — but they are not guaranteed to be so. They can compound across layers, interact with low-precision dtypes (fp16/bf16), or affect sensitive downstream tasks. Flash Attention is standard practice in large model training and the trade-off is well understood, but it should not be treated as a drop-in numerical equivalent of SDPA. If exact reproducibility or numerical sensitivity is a concern, use `"sdpa"` instead.

**No compilation required.** FastPLMs uses the HuggingFace [`kernels`](https://github.com/huggingface/kernels) package to load pre-built Flash Attention 2/3 binaries at runtime — no C++ compiler, no CUDA toolkit version pinning, no waiting:

```
pip install kernels
```

Building `flash-attn` from source is notoriously painful. The Ninja build system parallelizes aggressively across all available CPU cores, and each NVCC/CICC compiler process it spawns can consume **5–8 GB of RAM on its own**. On a 64-core machine this can push peak RAM usage to **~300 GB**, and even on a throttled single-threaded build (`MAX_JOBS=1 NVCC_THREADS=1`) the compile still takes **many hours** while grinding through paging. Pre-built community wheels cover 384+ version/GPU/CUDA/platform combinations and still routinely fall short of matching a user's exact environment. This is the point where most people give up and go without Flash Attention entirely. The `kernels` package sidesteps all of this by fetching a pre-compiled binary matched to your GPU architecture (SM80 for Ampere, SM90 for Hopper). If no compatible binary exists for your hardware, it gracefully falls back to `flex` or `sdpa` rather than erroring.

### Flex Attention (`flex`)

PyTorch's [`flex_attention`](https://pytorch.org/docs/stable/nn.attention.flex_attention.html) (PyTorch ≥ 2.5) generates a fused Triton kernel customized to the mask pattern at hand. It is numerically very close to SDPA — typically within floating-point rounding of naive computation. The primary advantage is that it can apply a **block mask** that skips padding tokens entirely, providing a meaningful speedup on batches with variable-length sequences (no compute wasted on padding). E1 uses a block-causal variant of this mask.

The **first forward pass** triggers JIT compilation via Triton, which can take 30–120 seconds. All subsequent calls are fast. Combining with `torch.compile` yields the best sustained throughput.

### Auto (`auto`)

Automatically selects the best available backend in order of preference: `kernels_flash` → `flex` → `sdpa`. Useful when you want maximum speed without configuring the environment manually, and you accept that the resolved backend may differ across machines.

### Setting the Backend

**At load time (every family):**
```python
from transformers import AutoConfig, AutoModel

config = AutoConfig.from_pretrained("Synthyra/ESM2-150M", trust_remote_code=True)
config.attn_backend = "flex"  # "sdpa", "kernels_flash", "flex", or "auto"
model = AutoModel.from_pretrained("Synthyra/ESM2-150M", config=config, trust_remote_code=True)
```

**After load time (every family):**

Every family's `PreTrainedModel` subclass exposes a mutable `attn_backend` property whose setter propagates the change to every attention submodule in-place, so you can swap backends on a loaded model without reloading the weights:

```python
model = AutoModel.from_pretrained("Synthyra/ESM2-150M", trust_remote_code=True)
model.attn_backend = "flex"           # every attention layer now uses flex
model.attn_backend = "kernels_flash"  # flip again, no reload
```

This is handy for benchmarking backends on the same weights or for falling back at runtime if a backend is unavailable. The setter asserts if the requested backend isn't installed on the current GPU (e.g. `kernels_flash` without the `kernels` package).

### Returning Attention Maps

All backends support `output_attentions=True`. For the optimized backends (SDPA, Flash Attention, Flex), attention weights are computed via a separate naive matrix multiplication and appended to the output — so enabling this negates the memory savings of those backends. Use it only for inspection or contact prediction, not during high-throughput inference.

---

## Embedding & Pooling

The `EmbeddingMixin` (shared across all models) provides a standardized way to extract representations from proteins.

### The Pooler
The `Pooler` class aggregates sequence-level residue representations into a single fixed-size vector. Supported strategies include:
- `mean`: Mask-aware average of all residues.
- `cls`: The first token's representation (Standard for classification).
- `max`: Element-wise maximum across the sequence.
- `var` / `std`: Variance or Standard Deviation of representations.
- `norm`: L2 normalization.
- `median`: Element-wise median.
- `parti`: Experimental PageRank-based attention pooling.

---

## Concrete Examples

### 1. Batch Embedding with SQLite (Scalable)
Ideal for embedding millions of sequences where you need to stream data or avoid OOM on RAM.

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("Synthyra/ESM2-150M", trust_remote_code=True).cuda()

sequences = ["MALWMRLLPLLALLALWGPDPAAA", "MKTIIALSYIFCLVFA", ...]

# Embed and store in SQLite
model.embed_dataset(
    sequences=sequences,
    batch_size=64,
    pooling_types=['mean', 'cls'], # Concatenates both
    sql=True,
    sql_db_path='large_protein_db.db',
    embed_dtype=torch.float32
)
```

### 2. Embedding from a FASTA File
Pass a FASTA file path directly — no manual parsing required. Multi-line sequences are handled automatically. You can combine `fasta_path` with an explicit `sequences` list and the two sources are merged before embedding.

```python
# Embed all sequences in a FASTA file and save to SQLite
model.embed_dataset(
    fasta_path='my_proteins.fasta',
    batch_size=64,
    pooling_types=['mean'],
    sql=True,
    sql_db_path='my_proteins.db',
)

# Mix a FASTA file with an explicit list
model.embed_dataset(
    sequences=["MKTIIALSYIFCLVFA"],
    fasta_path='additional_proteins.fasta',
    batch_size=32,
    save=True,
    save_path='combined_embeddings.pth',
)
```

### 3. High-Throughput In-Memory Embedding
Perfect for medium-sized datasets that fit in memory.

```python
# Embed and return as a dictionary
embeddings = model.embed_dataset(
    sequences=sequences,
    batch_size=128,
    pooling_types=['mean'],
    save=True,
    save_path='my_embeddings.pth'
)

# Access embedding
seq_vector = embeddings["MALWMRLLPLLALLALWGPDPAAA"] # torch.Tensor
```

### 4. Custom Pooling & Multi-Strategy
Concatenate multiple mathematical representations for richer downstream features.

```python
# Use a variety of pooling types
embeddings = model.embed_dataset(
    sequences=sequences,
    pooling_types=['mean', 'max', 'std', 'var'], # All 4 concatenated
    batch_size=32,
    full_embeddings=False
)

# Resulting vector size: 4 * hidden_size
print(embeddings[sequences[0]].shape)
```

---

## Testing & Benchmarking

FastPLMs includes a pytest-based test suite under `testing/` covering correctness, compliance, and performance. All GPU tests run inside Docker. See [docs/testing.md](docs/testing.md) for the full guide.

### Test Categories

| Test | What it checks | Marker |
| :--- | :--- | :--- |
| **AutoModel loading** | Every model loads via `AutoModelForMaskedLM.from_pretrained(..., trust_remote_code=True)` and produces valid outputs | `gpu` |
| **Backend consistency** | SDPA, Flex, and Flash backends produce equivalent predictions (>= 95% agreement) | `gpu` |
| **Weight compliance** | FastPLM weights are bit-exact with the original implementations (ESM2, ESMC, E1, DPLM) | `slow`, `gpu` |
| **Forward compliance** | Forward pass logits/predictions match the originals within tolerance | `slow`, `gpu` |
| **Rigorous parity** | Per-layer fp32 + bf16 hidden-state and last_hidden_state parity, padding-isolation, tokenizer parity, embed_dataset pipeline parity. Run per family in its own Docker image. | `gpu` |
| **NaN stability** | Batched inference with padding produces no NaN in real-token embeddings | `gpu` |
| **Batch-single match** | Batch and single-item embedding produce identical results | `gpu` |
| **Full model suite** | All of the above across every checkpoint (8M through 3B) | `gpu`, `large` |
| **Throughput benchmark** | Tokens/sec across models, backends, batch sizes, and sequence lengths | `slow`, `gpu` |
| **Structure models** | Boltz2 and ESMFold loading + forward pass | `structure`, `slow`, `gpu` |

### Running Tests with Docker

FastPLMs uses a per-family Docker setup. A single shared base image (`fastplms-base`) holds torch + transformers + the FastPLMs source, and one image per model family (`fastplms-esm2`, `fastplms-esm_plusplus`, `fastplms-e1`, `fastplms-dplm`, `fastplms-dplm2`, `fastplms-ankh`) layers on top with that family's native reference package. This isolates conflicting dependencies (e.g. EvolutionaryScale `esm` vs `fair-esm`, DPLM's torchtext pin) and keeps each image small.

```bash
# Initialize submodules (required before building Docker)
git submodule update --init --recursive

# Build base + every family image
./build_images.sh

# Build a single family
./build_images.sh esm2
./build_images.sh esm_plusplus
```

Run the parity / compliance tests for one family inside its image:

```bash
# ESM2
docker run --rm --gpus all --ipc=host -v $(pwd):/workspace fastplms-esm2 \
    python -m pytest /workspace/testing/test_parity.py -k esm2 -v

# ESM++ (model_key is "esmc")
docker run --rm --gpus all --ipc=host -v $(pwd):/workspace fastplms-esm_plusplus \
    python -m pytest /workspace/testing/test_parity.py -k esmc -v

# E1, DPLM, DPLM2, ANKH
for fam in e1 dplm dplm2 ankh; do
    docker run --rm --gpus all --ipc=host -v $(pwd):/workspace fastplms-$fam \
        python -m pytest /workspace/testing/test_parity.py -k $fam -v
done
```

The legacy monolithic `Dockerfile` (image tag `fastplms`) is still supported for the broader test suites that don't need native package isolation:

```bash
docker build -t fastplms .

# Fast tests (small models, no compliance, no structure)
docker run --gpus all --ipc=host fastplms python -m pytest /app/testing/ -m "gpu and not slow and not large and not structure" -v

# All sequence model tests except 3B
docker run --gpus all --ipc=host fastplms python -m pytest /app/testing/ -m "not large and not structure" -v

# Full suite including 3B models (requires 40+ GB VRAM)
docker run --gpus all --ipc=host fastplms python -m pytest /app/testing/ -m "not structure" -v

# Structure models only (Boltz2, ESMFold)
docker run --gpus all --ipc=host fastplms python -m pytest /app/testing/ -m "structure" -v
```

On Windows, replace `$(pwd)` with `${PWD}`. **Always pass `--ipc=host`** with PyTorch.

### Compliance / Native Reference Dependencies

The parity and compliance tests compare FastPLM outputs against the original model implementations. Each per-family Docker image installs only the deps it needs; outside Docker you can install them piecewise:

| Dependency | Used by | Install |
| :--- | :--- | :--- |
| `cloudpathlib`, `zstd`, `biotite` (+ `official/esm` submodule on `sys.path`) | ESM++ / ESMC | provided by `Dockerfile.esm_plusplus`; the EvolutionaryScale `esm` package itself is **not** pip-installed because it pins `transformers<4.53.0`. |
| `E1` | E1 | `pip install -e official/e1` (or use `Dockerfile.e1`) |
| `transformers` (`EsmForMaskedLM`, `T5EncoderModel`) | ESM2, DPLM, ANKH | already in `requirements.txt` |

If a native dep is missing in your environment, the corresponding parity tests are skipped rather than failing.

### Throughput Benchmarks

Throughput can be measured via the pytest test (saves structured JSON/CSV/PNG results) or the standalone script (more configurable).

```bash
# Pytest (benchmarks ESM2-8M, ESMplusplus_small, DPLM-150M, DPLM2-150M across all backends)
docker run --gpus all -v $(pwd):/workspace fastplms python -m pytest /app/testing/test_throughput.py -v -s
# Output: throughput_results.json, throughput_results.csv, throughput_comparison.png

# Standalone (fully configurable)
docker run --gpus all -v $(pwd):/workspace fastplms \
    python -m testing.throughput \
    --model_paths Synthyra/ESM2-8M Synthyra/ESMplusplus_small \
    --backends sdpa flex kernels_flash \
    --batch_sizes 2 4 8 \
    --sequence_lengths 64 128 256 512 1024 2048 \
    --output_path /workspace/throughput_comparison.png
```

---

## Installation & Docker

### Local Installation
```bash
git clone --recurse-submodules https://github.com/Synthyra/FastPLMs.git
cd FastPLMs
pip install -r requirements.txt
```

If you already cloned without `--recurse-submodules`, initialize submodules separately:
```bash
git submodule update --init --recursive
```

### Docker (Recommended for GPU Testing)

There are two Docker layouts; pick whichever matches your task.

**Per-family layout (recommended for parity / compliance work).** A shared base image plus one image per model family, each with that family's native reference package isolated from the others. Build all of them once with the helper script:

```bash
git submodule update --init --recursive
./build_images.sh                       # base + every family
./build_images.sh esm2 esm_plusplus     # subset
```

This produces `fastplms-base` and `fastplms-{esm2,esm_plusplus,e1,dplm,dplm2,ankh}`. Run a family's tests in its image:

```bash
docker run --rm --gpus all --ipc=host -v $(pwd):/workspace fastplms-esm2 \
    python -m pytest /workspace/testing/test_parity.py -k esm2 -v

docker run --rm --gpus all --ipc=host -v $(pwd):/workspace -it fastplms-esm2 bash
```

**Monolithic layout (legacy, single image).** The original `Dockerfile` bundles every dependency that can coexist in one image. Convenient for the broad test suites and throughput benchmarks; not suitable when two families' native deps conflict (notably ESM++ vs `fair-esm`).

```bash
git submodule update --init --recursive
docker build -t fastplms .

docker run --gpus all --ipc=host fastplms python -m pytest /app/testing/ -v
docker run --gpus all --ipc=host -v $(pwd):/workspace -it fastplms bash
```

On Windows, replace `$(pwd)` with `${PWD}`. Always pass `--ipc=host` with PyTorch.

---

## Suggestions & Contributions
Found a bug or have a feature request? Please open a [GitHub Issue](https://github.com/Synthyra/FastPLMs/issues). We are actively looking for contributions to optimize more pLM architectures!

---

## Citations

If you use FastPLMs, please cite the following along with the relevant model paper(s).

### FastPLMs

```bibtex
@misc{FastPLMs,
  author={Hallee, Logan and Bichara, David and Gleghorn, Jason P.},
  title={FastPLMs: Fast, efficient, protein language model inference from Huggingface AutoModel.},
  year={2024},
  url={https://huggingface.co/Synthyra/ESMplusplus_small},
  DOI={10.57967/hf/3726},
  publisher={Hugging Face}
}
```

### Flex Attention

```bibtex
@article{dong2024flexattention,
  title={Flex Attention: A Programming Model for Generating Optimized Attention Kernels},
  author={Dong, Juechu and Feng, Boyuan and Guessous, Driss and Liang, Yanbo and He, Horace},
  journal={arXiv preprint arXiv:2412.05496},
  year={2024}
}
```

### PyTorch

```bibtex
@inproceedings{paszke2019pytorch,
  title={PyTorch: An Imperative Style, High-Performance Deep Learning Library},
  author={Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and Desmaison, Alban and K{\"o}pf, Andreas and Yang, Edward and DeVito, Zach and Raison, Martin and Tejani, Alykhan and Chilamkurthy, Sasank and Steiner, Benoit and Fang, Lu and Bai, Junjie and Chintala, Soumith},
  booktitle={Advances in Neural Information Processing Systems 32},
  year={2019}
}
```

### ESM2

```bibtex
@article{lin2023esm2,
  title={Evolutionary-scale prediction of atomic-level protein structure with a language model},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and Zhu, Zhongkai and Lu, Wenting and Smestad, Nikita and Verkuil, Robert and Kabeli, Ori and Shmueli, Yaniv and dos Santos Costa, Allan and Fazel-Zarandi, Maryam and Sercu, Tom and Candido, Salvatore and Rives, Alexander},
  journal={Science},
  volume={379},
  number={6637},
  pages={1123--1130},
  year={2023},
  DOI={10.1126/science.ade2574}
}
```

### ESM++ (ESMC)

```bibtex
@article{hayes2024simulating,
  title={Simulating 500 million years of evolution with a language model},
  author={Hayes, Thomas and Rao, Roshan and Akin, Halil and Sofber, Nicholas J and Achour, Divya and Moez, Irfan and Garg, Rhitu and Angelova, Rami and Babu, Manan and Alcaide, Eric and others},
  journal={bioRxiv},
  year={2024}
}
```

### E1

```bibtex
@article{jain2025e1,
  title={E1: Retrieval-Augmented Protein Encoder Models},
  author={Jain, Sarthak and Beazer, Joel and Ruffolo, Jeffrey A and Bhatnagar, Aadyot and Madani, Ali},
  journal={bioRxiv},
  DOI={10.1101/2025.11.12.688125},
  year={2025}
}
```

### DPLM

```bibtex
@article{wang2024dplm,
  title={Diffusion Language Models Are Versatile Protein Learners},
  author={Wang, Xinyou and Ye, Zaixiang and Huang, Fei and Cao, Dongyan and Liang, Shujian and Huang, Liang},
  journal={Proceedings of the 41st International Conference on Machine Learning},
  year={2024}
}
```

### DPLM2

```bibtex
@article{wang2024dplm2,
  title={DPLM-2: A Multimodal Diffusion Protein Language Model},
  author={Wang, Xinyou and Ye, Zaixiang and Huang, Fei and Cao, Dongyan and Liang, Shujian and Huang, Liang},
  journal={arXiv preprint arXiv:2410.13782},
  year={2024}
}
```

### ANKH

```bibtex
@article{elnaggar2023ankh,
  title={Ankh: Optimized Protein Language Model Unlocks General-Purpose Modelling},
  author={Elnaggar, Ahmed and Essam, Hazem and Salah-Eldin, Wafaa and Moustafa, Walid and Elkerdawy, Mohamed and Rochereau, Charlotte and Rost, Burkhard},
  journal={arXiv preprint arXiv:2301.06568},
  year={2023}
}
```

```bibtex
@article{alsamkary2025ankh3,
  title={Ankh3: Multi-Task Pretraining with Sequence Denoising and Completion Enhances Protein Representations},
  author={Alsamkary, Hazem and Elshaffei, Mohamed and Elkerdawy, Mohamed and Elnaggar, Ahmed},
  journal={arXiv preprint arXiv:2505.20052},
  year={2025}
}
```

### Boltz

```bibtex
@article{passaro2025boltz2,
  title={Boltz-2: Exploring the Frontiers of Biomolecular Prediction},
  author={Passaro, Saro and Corso, Gabriele and Wohlwend, Jeremy and Reveiz, Mateo and Bordes, Florian and Wicky, Basile and Dayan, Peter and Jing, Bowen},
  journal={bioRxiv},
  year={2025}
}
```

```bibtex
@article{wohlwend2024boltz1,
  title={Boltz-1: Democratizing Biomolecular Interaction Modeling},
  author={Wohlwend, Jeremy and Corso, Gabriele and Passaro, Saro and Reveiz, Mateo and Leidal, Ken and Swanson, Wojtek and Kher, Gilmer and Lember, Tommi and Jaakkola, Tommi},
  journal={bioRxiv},
  year={2024}
}
```

### ESMFold / ProteinTTT

```bibtex
@misc{bushuiev2026proteinneed,
  title={One protein is all you need},
  author={Anton Bushuiev and Roman Bushuiev and Olga Pimenova and Nikola Zadorozhny and Raman Samusevich and Elisabet Manaskova and Rachel Seongeun Kim and Hannes St\"ark and Jiri Sedlar and Martin Steinegger and Tom\'a\v{s} Pluskal and Josef Sivic},
  year={2026},
  eprint={2411.02109},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2411.02109}
}
```
---