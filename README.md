# FastPLMs

<img width="2816" height="1536" alt="Gemini_Generated_Image_5bmmdc5bmmdc5bmm" src="https://github.com/user-attachments/assets/ffaf84b6-9970-40fd-aa31-1b314d6ca146" />

FastPLMs is an open-source effort to increase the efficiency of pretrained protein language models (pLMs), switching out native attention implementations for **Flash Attention** or **Flex Attention**.

All models can be loaded from HuggingFace 🤗 transformers via `AutoModel`, and this repository does not need to be cloned for most use cases.

---

<details>
<summary><b>What are Protein Language Models (pLMs)?</b></summary>

Protein Language Models (pLMs) are transformer-based models trained on large datasets of protein sequences (e.g., UniProt). By treating amino acids like words in a language, these models learn to:
- **Learn Representations**: Extract high-dimensional features (embeddings) that capture evolutionary information, structural constraints, and functional properties.
- **Protein Generation**: Design new-to-nature sequences for therapeutic or industrial applications.
- **Predict Properties**: Enable downstream tasks like thermostability prediction, protein-protein interactions, and enzyme classification.
- **Structure & Inverse Folding**: Map sequences to 3D structures or vice versa.

FastPLMs focuses on making these powerful models easier to use.
</details>

<details>
<summary><b>What is this repo?</b></summary>

FastPLMs provides optimized implementations of popular pLMs. We get rid of unnecessary requirements, integrate with Huggingface `AutoModel`, utilize efficient attention implementations, and add nice to have functions to easy embeddings of entire datasets:
- **Efficiency**: Faster inference and significantly lower memory footprint.
- **Compatibility**: Direct integration with HuggingFace `transformers`.
- **Flexibility**: Support for both `sdpa` (which will automatically call Flash Attention when possible) and PyTorch's new `Flex Attention`.
- **Dataset Scale**: Built-in tools for embedding millions of sequences efficiently.
</details>

<details>
<summary><b>Supported Models & Families</b></summary>

The currently supported models can be found in our [HuggingFace Collection](https://huggingface.co/collections/Synthyra/pretrained-plms-675351ecc050f63baedd77de).

Major families included:
- **E1 (Profluent)**: 150M, 300M, 600M parameters.
- **ESM2 (Meta)**: 8M, 35M, 150M, 650M, 3B parameters.
- **ESMC (called ESM++) (EvolutionaryScale)**: Small (300M), Large (600M).
- **DPLM / DPLM2 (ByteDance)**: Diffusion Protein Language Models, 150M to 3B.
- **Boltz2 (Boltz)**: High-performance structure prediction.
</details>

<details>
<summary><b>What is Flex Attention?</b></summary>

**Flex Attention** is a flexible and efficient attention mechanism introduced in PyTorch. In FastPLMs, we use it to:
- **Dynamic Masking**: Efficiently handle padding via block masks, avoiding unnecessary computation on pad tokens.
- **Block-Causal Patterns**: Support specialized attention patterns, such as the ones used in E1.
- **Performance**: High throughput when used with `torch.compile`.

**How to enable it:**
Set `attn_backend="flex"` in the model config before loading:
```python
from transformers import AutoConfig, AutoModel

config = AutoConfig.from_pretrained("Synthyra/ESMplusplus_small", trust_remote_code=True)
config.attn_backend = "flex"
model = AutoModel.from_pretrained("Synthyra/ESMplusplus_small", config=config, trust_remote_code=True)
```
*Note: Flex Attention requires float16 or bfloat16 and works best with `torch.compile`.*
</details>

<details>
<summary><b>How does the Embedding Mixin work?</b></summary>

The `EmbeddingMixin` (defined in `embedding_mixin.py`) provides a unified interface for extracting representations from any supported model.

**Key Feature: `embed_dataset()`**
This method allows for high-throughput embedding of sequence lists:
- **Automatic Sorting**: Sequences are sorted by length to minimize padding overhead.
- **Batching**: Optimized batch processing with progress tracking.
- **Caching**: Automatically skips sequences already found in an existing SQLite database or `.pth` file.
- **Storage**: Supports saving to standard PyTorch files or SQLite databases for streaming large datasets.

Example usage:
```python
embedding_dict = model.embed_dataset(
    sequences=['MALWMRLLPLLALLALWGPDPAAA', 'MKTIIALSYIFCLVFA'],
    batch_size=32,
    pooling_type=['mean', 'cls'], # Multi-pooling concatenated
    save=True,
    save_path='embeddings.pth'
)
```
</details>

<details>
<summary><b>How does the Pooler work?</b></summary>

The `Pooler` class provides a flexible way to aggregate sequence-level information into a single vector.

**Supported Pooling Strategies:**
- `mean`: Average of all non-pad tokens (mask-aware).
- `max`: Element-wise maximum across tokens.
- `cls`: The representation of the first token (usually the CLS token).
- `var` / `std`: Variance or Standard Deviation of the representations.
- `norm`: L2 normalization of the representation.
- `median`: Element-wise median across the sequence.
- `parti`: PageRank-based attention pooling (experimental).

You can specify multiple types (e.g., `['mean', 'var']`), which will be concatenated into a single embedding vector of size `num_pools * hidden_size`.
</details>

<details>
<summary><b>Which models can you specify pooling?</b></summary>

- **Custom Pooler Support**: E1 and ESM++ natively support the custom `Pooler` strategies via the `pooling_types` argument in their initialization or via `embed_dataset`.
- **HuggingFace Standard**: ESM2, DPLM, and DPLM2 use the standard HuggingFace pooling (usually mean or CLS), but they all support full custom pooling when using the `embed_dataset` method provided by `EmbeddingMixin`.
</details>

<details>
<summary><b>How can I run the tests?</b></summary>

The testing suite is CLI-first and located under `testing/`.

**Main Entrypoints:**
- **Full Suite**: `py -m testing.run_all`
- **Compliance**: `py -m testing.run_compliance` (Ensures outputs match reference models)
- **Benchmarks**: `py -m testing.run_throughput` (Memory and speed metrics)
- **Embeddings**: `py -m testing.run_embedding` (Mixin validation)

**Common CLI Options:**
- `--full-models`: Run across all supported checkpoints (not just representative ones).
- `--device cuda`: Specify hardware (auto-detects by default).
- `--dtype bfloat16`: Set calculation precision.

**Docker Execution:**
```bash
docker build -t fastplms-test -f Dockerfile .
docker run --rm --gpus all -it -v ${PWD}:/workspace fastplms-test python -m testing.run_all
```
Results are saved as structured JSON, CSV, and high-resolution PNG plots in `testing/results/`.
</details>

---

## Quick Start

### Loading a Model
```python
import torch
from transformers import AutoModel

# Load optimized ESM++ with standard SDPA attention
model = AutoModel.from_pretrained("Synthyra/ESMplusplus_small", trust_remote_code=True)

# Recommended for speed:
model = torch.compile(model)
```

### Switching Attention Backend
`sdpa` (Flash Attention) is the default for stability. For `Flex Attention`:
```python
from transformers import AutoConfig, AutoModel

config = AutoConfig.from_pretrained("Synthyra/ESMplusplus_small", trust_remote_code=True)
config.attn_backend = "flex"
model = AutoModel.from_pretrained("Synthyra/ESMplusplus_small", config=config, trust_remote_code=True)
```

## Suggestions
Have suggestions, comments, or requests? Found a bug? Open a GitHub issue and we'll respond soon.
