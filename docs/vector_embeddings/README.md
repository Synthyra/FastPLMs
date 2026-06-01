---
pretty_name: Protify Vector Benchmark Embeddings
task_categories:
- feature-extraction
tags:
- protein-language-models
- protein-embeddings
- protify
- vector-benchmark
- pytorch
- xet
---

# Protify Vector Benchmark Embeddings

Precomputed pooled protein embeddings for the Protify vector benchmark.

This dataset stores ready-to-use `.pth.gz` embedding artifacts for a broad panel of protein language models and controls. It is intended for fast downstream benchmarking in Protify without repeatedly embedding the same benchmark sequences on local GPUs.

## What Is Included

Each file in `embeddings/` is a gzip-compressed PyTorch artifact:

```text
embeddings/<MODEL>_False_mean_var.pth.gz
```

The filename convention mirrors the Protify embedding cache settings:

| Field | Meaning |
| --- | --- |
| `<MODEL>` | Model or control used to generate the embeddings |
| `False` | Pooled sequence-level embeddings, not full residue-level embeddings |
| `mean_var` | Mean and variance pooling used for the vector benchmark |
| `.pth.gz` | PyTorch serialization compressed with gzip |

Use these files when you want fixed-size protein vectors for classical ML, vector search, nearest-neighbor analysis, low-shot benchmarking, or model comparison.

## Quick Start

Install the minimal loading dependencies:

```bash
pip install torch huggingface_hub
```

Download one embedding file:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="Synthyra/vector_embeddings",
    repo_type="dataset",
    filename="embeddings/ESM2-650_False_mean_var.pth.gz",
)
print(path)
```

Load the PyTorch payload:

```python
import gzip

import torch

with gzip.open(path, "rb") as handle:
    embeddings = torch.load(handle, map_location="cpu")

first_key = next(iter(embeddings))
first_vector = embeddings[first_key]

print(first_key)
print(first_vector.shape, first_vector.dtype)
```

Download with the Hugging Face CLI:

```bash
hf download Synthyra/vector_embeddings \
  --type dataset \
  --include "embeddings/ESM2-650_False_mean_var.pth.gz" \
  --local-dir vector_embeddings
```

Download the full repository only if you have enough disk space:

```bash
hf download Synthyra/vector_embeddings \
  --type dataset \
  --include "embeddings/*_False_mean_var.pth.gz" \
  --local-dir vector_embeddings
```

The full `embeddings/` directory is about 189.22 GiB.

## Protify Usage

Protify can use these precomputed embeddings as model-ready vector caches for benchmark runs. This avoids recomputing embeddings for every model and dataset combination, which is especially useful for large PLMs and laptop-scale downstream analysis.

Typical workflow:

1. Choose one or more embedding artifacts from this repository.
2. Point Protify at the downloaded cache files.
3. Run the vector benchmark across the selected tasks.
4. Compare model families against negative controls such as `Random` and `OneHot-Protein`.

## Model Inventory

| Model | File | Size |
| --- | --- | ---: |
| AMPLIFY-120 | `AMPLIFY-120_False_mean_var.pth.gz` | 3.63 GiB |
| AMPLIFY-350 | `AMPLIFY-350_False_mean_var.pth.gz` | 5.26 GiB |
| ANKH-Base | `ANKH-Base_False_mean_var.pth.gz` | 5.13 GiB |
| ANKH-Large | `ANKH-Large_False_mean_var.pth.gz` | 9.72 GiB |
| ANKH2-Large | `ANKH2-Large_False_mean_var.pth.gz` | 9.68 GiB |
| DPLM-150 | `DPLM-150_False_mean_var.pth.gz` | 4.26 GiB |
| DPLM-3B | `DPLM-3B_False_mean_var.pth.gz` | 15.60 GiB |
| DPLM-650 | `DPLM-650_False_mean_var.pth.gz` | 8.03 GiB |
| DSM-150 | `DSM-150_False_mean_var.pth.gz` | 4.28 GiB |
| DSM-650 | `DSM-650_False_mean_var.pth.gz` | 8.04 GiB |
| DSM-PPI | `DSM-PPI_False_mean_var.pth.gz` | 8.05 GiB |
| E1-150 | `E1-150_False_mean_var.pth.gz` | 4.22 GiB |
| E1-300 | `E1-300_False_mean_var.pth.gz` | 5.50 GiB |
| E1-600 | `E1-600_False_mean_var.pth.gz` | 6.76 GiB |
| ESM2-150 | `ESM2-150_False_mean_var.pth.gz` | 4.27 GiB |
| ESM2-35 | `ESM2-35_False_mean_var.pth.gz` | 3.33 GiB |
| ESM2-3B | `ESM2-3B_False_mean_var.pth.gz` | 15.55 GiB |
| ESM2-650 | `ESM2-650_False_mean_var.pth.gz` | 8.04 GiB |
| ESM2-8 | `ESM2-8_False_mean_var.pth.gz` | 2.39 GiB |
| ESMC-300 | `ESMC-300_False_mean_var.pth.gz` | 6.26 GiB |
| ESMC-600 | `ESMC-600_False_mean_var.pth.gz` | 7.39 GiB |
| GLM2-150 | `GLM2-150_False_mean_var.pth.gz` | 3.63 GiB |
| GLM2-650 | `GLM2-650_False_mean_var.pth.gz` | 6.85 GiB |
| GLM2-GAIA | `GLM2-GAIA_False_mean_var.pth.gz` | 6.73 GiB |
| OneHot-Protein | `OneHot-Protein_False_mean_var.pth.gz` | 0.53 GiB |
| ProtBert-BFD | `ProtBert-BFD_False_mean_var.pth.gz` | 6.68 GiB |
| ProtBert | `ProtBert_False_mean_var.pth.gz` | 6.33 GiB |
| ProtT5 | `ProtT5_False_mean_var.pth.gz` | 6.53 GiB |
| Random-Transformer | `Random-Transformer_False_mean_var.pth.gz` | 3.35 GiB |
| Random | `Random_False_mean_var.pth.gz` | 3.24 GiB |

## Notes

These files are large. Prefer single-file downloads with `hf_hub_download()` or `hf download --include` unless you explicitly need every model.

This repository is an artifact store, not a tabular Hugging Face Datasets dataset. Use `huggingface_hub` or the `hf` CLI rather than `datasets.load_dataset()`.

Embedding artifacts are intended to support reproducible Protify benchmarking. For redistribution or derived work, check the licenses and usage terms of the original benchmark datasets and the upstream model checkpoints used to generate each embedding set.

## Citation

If these embeddings help your work, please cite the relevant upstream model papers or model cards, Protify, and Synthyra resources used in your analysis.
