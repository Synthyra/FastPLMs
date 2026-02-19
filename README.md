# FastPLMs

<img width="2816" height="1536" alt="FastPLMs Hero Image" src="https://github.com/user-attachments/assets/ffaf84b6-9970-40fd-aa31-1b314d6ca146" />

FastPLMs is an open-source initiative dedicated to accelerating pretrained protein language models (pLMs). By replacing native, often suboptimal attention implementations with **Flash Attention** or **Flex Attention**, we provide high-performance alternatives that are fully compatible with the HuggingFace `transformers` ecosystem.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Supported Models](#supported-models)
3. [Efficiency: Flex & Flash Attention](#efficiency-flex--flash-attention)
4. [Embedding & Pooling](#embedding--pooling)
5. [Concrete Examples](#concrete-examples)
6. [Testing & Benchmarking](#testing--benchmarking)
7. [Installation & Docker](#installation--docker)

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
| **Boltz2** | MIT / Various | [jwohlwend/boltz](https://github.com/jwohlwend/boltz) | Optimized Structure Prediction | Standard |

### Full Model List

| Model Key | Family | Parameters | Organization | FastPLMs Repo ID | Official Reference |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `e1_150m` | E1 | 150M | Profluent Bio | [Synthyra/Profluent-E1-150M](https://huggingface.co/Synthyra/Profluent-E1-150M) | [Profluent-Bio/E1-150m](https://huggingface.co/Profluent-Bio/E1-150m) |
| `e1_300m` | E1 | 300M | Profluent Bio | [Synthyra/Profluent-E1-300M](https://huggingface.co/Synthyra/Profluent-E1-300M) | [Profluent-Bio/E1-300m](https://huggingface.co/Profluent-Bio/E1-300m) |
| `e1_600m` | E1 | 600M | Profluent Bio | [Synthyra/Profluent-E1-600M](https://huggingface.co/Synthyra/Profluent-E1-600M) | [Profluent-Bio/E1-600m](https://huggingface.co/Profluent-Bio/E1-600m) |
| `esm2_8m` | ESM2 | 8M | Meta AI | [Synthyra/ESM2-8M](https://huggingface.co/Synthyra/ESM2-8M) | [facebook/esm2_t6_8M_UR50D](https://huggingface.co/facebook/esm2_t6_8M_UR50D) |
| `esm2_35m` | ESM2 | 35M | Meta AI | [Synthyra/ESM2-35M](https://huggingface.co/Synthyra/ESM2-35M) | [facebook/esm2_t12_35M_UR50D](https://huggingface.co/facebook/esm2_t12_35M_UR50D) |
| `esm2_150m` | ESM2 | 150M | Meta AI | [Synthyra/ESM2-150M](https://huggingface.co/Synthyra/ESM2-150M) | [facebook/esm2_t30_150M_UR50D](https://huggingface.co/facebook/esm2_t30_150M_UR50D) |
| `esm2_650m` | ESM2 | 650M | Meta AI | [Synthyra/ESM2-650M](https://huggingface.co/Synthyra/ESM2-650M) | [facebook/esm2_t33_650M_UR50D](https://huggingface.co/facebook/esm2_t33_650M_UR50D) |
| `esm2_3b` | ESM2 | 3B | Meta AI | [Synthyra/ESM2-3B](https://huggingface.co/Synthyra/ESM2-3B) | [facebook/esm2_t36_3B_UR50D](https://huggingface.co/facebook/esm2_t36_3B_UR50D) |
| `esmplusplus_small` | ESM++ | 300M | EvolutionaryScale | [Synthyra/ESMplusplus_small](https://huggingface.co/Synthyra/ESMplusplus_small) | [EvolutionaryScale/esmc-300m](https://huggingface.co/EvolutionaryScale/esmc-300m-2024-12) |
| `esmplusplus_large` | ESM++ | 600M | EvolutionaryScale | [Synthyra/ESMplusplus_large](https://huggingface.co/Synthyra/ESMplusplus_large) | [EvolutionaryScale/esmc-600m](https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12) |
| `dplm_150m` | DPLM | 150M | ByteDance | [Synthyra/DPLM-150M](https://huggingface.co/Synthyra/DPLM-150M) | [airkingbd/dplm_150m](https://huggingface.co/airkingbd/dplm_150m) |
| `dplm_650m` | DPLM | 650M | ByteDance | [Synthyra/DPLM-650M](https://huggingface.co/Synthyra/DPLM-650M) | [airkingbd/dplm_650m](https://huggingface.co/airkingbd/dplm_650m) |
| `dplm_3b` | DPLM | 3B | ByteDance | [Synthyra/DPLM-3B](https://huggingface.co/Synthyra/DPLM-3B) | [airkingbd/dplm_3b](https://huggingface.co/airkingbd/dplm_3b) |
| `dplm2_150m` | DPLM2 | 150M | ByteDance | [Synthyra/DPLM2-150M](https://huggingface.co/Synthyra/DPLM2-150M) | [airkingbd/dplm2_150m](https://huggingface.co/airkingbd/dplm2_150m) |
| `dplm2_650m` | DPLM2 | 650M | ByteDance | [Synthyra/DPLM2-650M](https://huggingface.co/Synthyra/DPLM2-650M) | [airkingbd/dplm2_650m](https://huggingface.co/airkingbd/dplm2_650m) |
| `dplm2_3b` | DPLM2 | 3B | ByteDance | [Synthyra/DPLM2-3B](https://huggingface.co/Synthyra/DPLM2-3B) | [airkingbd/dplm2_3b](https://huggingface.co/airkingbd/dplm2_3b) |
| `boltz2` | Boltz2 | - | MIT / Various | [Synthyra/Boltz2](https://huggingface.co/Synthyra/Boltz2) | [jwohlwend/boltz](https://github.com/jwohlwend/boltz) |

---

## Efficiency: Flex & Flash Attention

### Flash Attention (SDPA)
We use PyTorch's `Scaled Dot Product Attention` (SDPA) as the default backend for most models. It provides significant speedups over native implementations while maintaining stability across different GPU architectures.

### Flex Attention
**Flex Attention** is a cutting-edge mechanism in PyTorch that allows for:
- **Dynamic Masking**: Efficiently ignoring padding without redundant compute.
- **Custom Patterns**: Supporting specialized masks (like E1's block-causal) with native performance.
- **Extreme Speed**: When combined with `torch.compile`, Flex Attention often provides the best possible throughput.

**To enable Flex Attention:**
```python
from transformers import AutoConfig, AutoModel

config = AutoConfig.from_pretrained("Synthyra/ESMplusplus_small", trust_remote_code=True)
config.attn_backend = "flex"
model = AutoModel.from_pretrained("Synthyra/ESMplusplus_small", config=config, trust_remote_code=True)
```

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

### 2. High-Throughput In-Memory Embedding
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

### 3. Custom Pooling & Multi-Strategy
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

FastPLMs includes a robust CLI-based testing suite under `testing/`.

### Running the Suite
- **Compliance Checks**: Verify that optimized models match reference outputs.
  ```bash
  py -m testing.run_compliance --families esm2
  ```
- **Throughput Benchmarks**: Measure tokens/sec and peak memory.
  ```bash
  py -m testing.run_throughput --device cuda --lengths 512,1024
  ```
- **Run Everything**: Execute the full suite across all families.
  ```bash
  py -m testing.run_all --full-models
  ```

Results are saved to `testing/results/<timestamp>/` as `metrics.json`, `metrics.csv`, and high-resolution plots.

---

## Installation & Docker

### Local Installation
```bash
git clone https://github.com/Synthyra/FastPLMs.git
cd FastPLMs
pip install -r requirements.txt
```

### Docker (Recommended for Testing)
```bash
# Build the image
docker build -t fastplms-test -f Dockerfile .

# Run benchmarks inside container
docker run --rm --gpus all -it -v ${PWD}:/workspace fastplms-test \
    python -m testing.run_throughput --device cuda
```

---

## Suggestions & Contributions
Found a bug or have a feature request? Please open a [GitHub Issue](https://github.com/Synthyra/FastPLMs/issues). We are actively looking for contributions to optimize more pLM architectures!
