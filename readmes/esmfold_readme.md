---
library_name: transformers
tags:
  - protein
  - structure-prediction
  - esmfold
  - test-time-training
---

# NOTE
The GitHub with the implementation and requirements.txt can be found [here](https://github.com/Synthyra/FastPLMs.git)

# FastESMFold

FastESMFold is a self-contained, HuggingFace-compatible reimplementation of ESMFold with **built-in Test-Time Training (TTT)** and multi-backend attention (SDPA, Flash, Flex).

No dependency on `fair-esm`, `proteinttt`, or `openfold`. Just `transformers`, `torch`, and `einops`.

## Key Features

- **Always-on TTT**: Runs 10 steps of masked language model adaptation via LoRA before folding. Returns the structure with the highest pLDDT across all steps.
- **Best structure selection**: Folds after each TTT step, tracks per-step pLDDT, returns the best.
- **FastESM2 attention**: SDPA/Flash/Flex backends for the 3B ESM2 backbone.
- **Self-contained LoRA**: lora_diffusion-compatible implementation (no peft dependency). `Normal(0, 1/r)` initialization, `scale=alpha`.
- **3.5B parameters**: Full ESMFold v1 architecture (ESM2-3B backbone + folding trunk).

## Benchmark

Tested on 10 difficult sequences on A10G GPU:

| Metric | Value |
|--------|-------|
| Mean baseline pLDDT | 0.549 |
| Mean best TTT pLDDT | 0.637 |
| Mean improvement | +0.088 |
| Sequences improved >5pt | 5/10 |
| Time per sequence | ~20-45s |
| GPU memory peak | 18.3 GB |

On the hardest sequence (baseline pLDDT 0.38), TTT achieves 0.72 (+34 points).

## Use with transformers

### Basic usage
```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "Synthyra/FastESMFold",
    trust_remote_code=True,
    torch_dtype=torch.float32,
).cuda().eval()

result = model.fold_protein("MKTLLILAVVAAALA...")
print(f"pLDDT: {result['plddt']:.3f}")
print(f"Best step: {result['best_step']}")
print(f"Step pLDDTs: {result['step_plddts']}")
print(f"PDB length: {len(result['pdb_string'])} chars")
```

### Return values

`fold_protein(sequence)` returns a dict with:

| Key | Type | Description |
|-----|------|-------------|
| `plddt` | float | Best mean pLDDT across all TTT steps |
| `ptm` | float | Predicted TM-score from best step |
| `pdb_string` | str | PDB format structure from best step |
| `step_plddts` | list[float] | pLDDT at each step [baseline, s1, ..., s10] |
| `best_step` | int | Which step produced the best structure (0=baseline) |

### Loading from Synthyra/ESMFold-v1 with custom config
```python
from esmfold.modeling_fast_esmfold import FastEsmFoldConfig, FastEsmForProteinFolding

config = FastEsmFoldConfig.from_pretrained("Synthyra/ESMFold-v1")
config.attn_backend = "sdpa"
config.ttt_config = {
    "lr": 4e-4,
    "steps": 10,
    "ags": 4,
    "batch_size": 4,
    "lora_rank": 8,
    "lora_alpha": 32.0,
    "seed": 0,
}
model = FastEsmForProteinFolding.from_pretrained(
    "Synthyra/ESMFold-v1",
    config=config,
    torch_dtype=torch.float32,
).cuda().eval()
```

## Attention backends

The ESM2 backbone supports multiple attention backends via `config.attn_backend`:

| Backend | Key | Notes |
| :--- | :--- | :--- |
| PyTorch SDPA | `"sdpa"` | Default. Exact numerics, stable on all hardware. |
| Flash Attention | `"kernels_flash"` | Fastest. Requires `pip install kernels`. |
| Flex Attention | `"flex"` | Skips padding tokens via block mask. First use compiles a Triton kernel. |
| Auto | `"auto"` | Picks best available: `kernels_flash` > `flex` > `sdpa`. |

## TTT Configuration

TTT parameters can be customized via `config.ttt_config`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 4e-4 | Learning rate for SGD optimizer |
| `steps` | 10 | Number of optimizer steps |
| `ags` | 4 | Gradient accumulation steps per optimizer step |
| `batch_size` | 4 | Batch size for masked language model training |
| `mask_ratio` | 0.15 | Fraction of tokens to mask |
| `lora_rank` | 8 | LoRA rank (0 for full fine-tuning) |
| `lora_alpha` | 32.0 | LoRA scaling factor |
| `seed` | 0 | Random seed for reproducibility |

## How TTT Works

1. **Baseline fold** (step 0): Standard ESMFold prediction
2. **LoRA injection**: Rank-8 LoRA adapters on ESM2 attention Q/K/V projections
3. **Masked LM training**: 10 optimizer steps of BERT-style masked language modeling on the input sequence
4. **Per-step folding**: After each optimizer step, fold the sequence and record pLDDT
5. **Best selection**: Return the structure with the highest pLDDT
6. **Reset**: Restore LoRA weights to initial state for the next sequence

This is based on the ProteinTTT paper (test-time compute for protein structure prediction).

### Citation
If you use this implementation please cite it:
```
@misc {FastPLMs,
    author       = { Hallee, Logan and Bichara, David and Gleghorn, Jason P.},
    title        = { FastPLMs: Fast, efficient, protein language model inference from Huggingface AutoModel.},
    year         = {2024},
    url          = { https://huggingface.co/Synthyra/ESMplusplus_small },
    DOI          = { 10.57967/hf/3726 },
    publisher    = { Hugging Face }
}
```
