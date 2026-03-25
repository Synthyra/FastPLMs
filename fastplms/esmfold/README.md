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

FastESMFold is a self-contained, HuggingFace-compatible reimplementation of ESMFold with optional **Test-Time Training (TTT)** and multi-backend attention (SDPA, Flash, Flex).

No dependency on `fair-esm`, `proteinttt`, or `openfold`. Just `transformers`, `torch`, and `einops`.

## Why Test-Time Training?

Protein language models like ESM2 are trained on millions of sequences, but at inference time they process each new protein in a single forward pass with no adaptation. This is a missed opportunity: the input sequence itself contains structural signal that the model could learn from.

**Test-Time Training (TTT)** adapts the model to each individual protein before predicting its structure. The idea is simple: before folding, we briefly train the ESM2 backbone on the input sequence using masked language modeling (the same objective it was pretrained with). This forces the model to "study" the specific sequence, strengthening its internal representation of that protein's structural features.

The adaptation uses **LoRA** (Low-Rank Adaptation) for efficiency: only small adapter weights are trained (~4.4M parameters out of 3.5B), and the base model is restored after each prediction. This takes 20-45 seconds per sequence on an A10G GPU but can dramatically improve structure prediction quality, especially on difficult targets where standard ESMFold produces low-confidence predictions.

**When is TTT most useful?**
- Sequences with low baseline pLDDT (< 0.5): TTT can improve pLDDT by 10-30+ points
- Novel proteins with limited homology in training data
- Disordered or multi-domain proteins where ESMFold struggles

**When is TTT unnecessary?**
- Sequences that already fold well (baseline pLDDT > 0.7): TTT rarely helps and may slightly degrade predictions
- High-throughput screening where speed matters more than accuracy

## Key Features

- **Standard ESMFold**: Full ESMFold v1 structure prediction, loadable via `AutoModel`
- **Optional TTT**: Enable test-time training for improved structure prediction on difficult sequences
- **Best structure selection**: When TTT is enabled, folds after each step and returns the structure with the highest pLDDT
- **FastESM2 attention**: SDPA/Flash/Flex backends for the 3B ESM2 backbone
- **Self-contained LoRA**: lora_diffusion-compatible implementation (no peft dependency)
- **3.5B parameters**: Full ESMFold v1 architecture (ESM2-3B backbone + folding trunk)

## Use with transformers

### Standard structure prediction (no TTT)

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "Synthyra/FastESMFold",
    trust_remote_code=True,
    dtype=torch.float32,
).cuda().eval()

# Standard fold (no TTT)
with torch.no_grad():
    output = model.infer("MKTLLILAVVAAALA...")
pdb_strings = model.output_to_pdb(output)
plddt = output["plddt"].mean().item()
print(f"pLDDT: {plddt:.3f}")
```

### Structure prediction with TTT

TTT adapts the ESM2 backbone to a specific input sequence via masked language modeling before folding. This can dramatically improve pLDDT on difficult sequences (e.g., 0.38 to 0.72).

```python
# Configure TTT
model._ttt_cfg.steps = 10      # 10 optimizer steps (default)
model._ttt_cfg.lora_rank = 8   # LoRA rank (default)
model._ttt_cfg.lora_alpha = 32 # LoRA scale (default)

# fold_protein() runs TTT, folds after each step, returns best structure
result = model.fold_protein("MKTLLILAVVAAALA...")
print(f"pLDDT: {result['plddt']:.3f}")
print(f"Best step: {result['best_step']} (0=baseline, 1-10=TTT steps)")
print(f"Step pLDDTs: {[f'{p:.2f}' for p in result['step_plddts']]}")

# Save PDB
with open("structure.pdb", "w") as f:
    f.write(result["pdb_string"])
```

### Return values

`fold_protein(sequence)` returns a dict:

| Key | Type | Description |
|-----|------|-------------|
| `plddt` | float | Best mean pLDDT across all TTT steps |
| `ptm` | float | Predicted TM-score from best step |
| `pdb_string` | str | PDB format structure from best step |
| `step_plddts` | list[float] | pLDDT at each step [baseline, s1, ..., s10] |
| `best_step` | int | Which step produced the best structure (0=baseline) |

### Disabling TTT

To use FastESMFold as a standard ESMFold (no TTT), set `steps=0` or call `infer()` directly:

```python
# Option 1: Set TTT steps to 0
config = AutoConfig.from_pretrained("Synthyra/FastESMFold", trust_remote_code=True)
config.ttt_config = {"steps": 0}
model = AutoModel.from_pretrained("Synthyra/FastESMFold", config=config, trust_remote_code=True)
result = model.fold_protein("MKTLLILAVVAAALA...")  # No TTT, just baseline fold

# Option 2: Call infer() directly (inherited from EsmForProteinFolding)
with torch.no_grad():
    output = model.infer("MKTLLILAVVAAALA...")
pdb_strings = model.output_to_pdb(output)
```

## TTT Benchmark

Tested on 10 difficult sequences on A10G GPU:

| Metric | Value |
|--------|-------|
| Mean baseline pLDDT | 0.549 |
| Mean best TTT pLDDT | 0.637 |
| Mean improvement | +0.088 |
| Sequences improved >5pt | 5/10 |
| Time per sequence | ~20-45s |
| GPU memory peak | 18.3 GB |

On the hardest sequence (baseline pLDDT 0.38), TTT improves to 0.72 (+34 points).

## Attention backends

The ESM2 backbone supports multiple attention backends via `config.attn_backend`:

| Backend | Key | Notes |
| :--- | :--- | :--- |
| PyTorch SDPA | `"sdpa"` | Default. Exact numerics, stable on all hardware. |
| Flash Attention | `"kernels_flash"` | Fastest. Requires `pip install kernels`. |
| Flex Attention | `"flex"` | Skips padding tokens via block mask. First use compiles a Triton kernel. |
| Auto | `"auto"` | Picks best available: `kernels_flash` > `flex` > `sdpa`. |

```python
from transformers import AutoConfig, AutoModel

config = AutoConfig.from_pretrained("Synthyra/FastESMFold", trust_remote_code=True)
config.attn_backend = "kernels_flash"
model = AutoModel.from_pretrained("Synthyra/FastESMFold", config=config, trust_remote_code=True)
```

## TTT Configuration

TTT parameters are set via `config.ttt_config` (a dict) or by modifying `model._ttt_cfg` after loading:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 4e-4 | Learning rate for SGD optimizer |
| `steps` | 10 | Number of optimizer steps (0 to disable TTT) |
| `ags` | 4 | Gradient accumulation steps per optimizer step |
| `batch_size` | 4 | Batch size for masked language model training |
| `mask_ratio` | 0.15 | Fraction of tokens to mask |
| `lora_rank` | 8 | LoRA rank (0 for full backbone fine-tuning) |
| `lora_alpha` | 32.0 | LoRA scaling factor (applied as `scale=alpha`, matching lora_diffusion) |
| `seed` | 0 | Random seed for reproducible LoRA initialization and masking |
| `lora_target_class` | `"EsmSelfAttention"` | Which module class to inject LoRA into |

## How TTT Works

1. **Baseline fold** (step 0): Standard ESMFold prediction
2. **LoRA injection**: Rank-8 LoRA adapters on all `nn.Linear` layers inside ESM2 attention modules
3. **Masked LM training**: 10 optimizer steps (each with 4 gradient accumulation sub-steps) of BERT-style masked language modeling on the input sequence
4. **Per-step folding**: After each optimizer step, fold the sequence and record pLDDT
5. **Best selection**: Return the structure with the highest pLDDT
6. **Reset**: Restore LoRA weights to initial state for the next sequence

## Citations

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

```bibtex
@article{dong2024flexattention,
  title={Flex Attention: A Programming Model for Generating Optimized Attention Kernels},
  author={Dong, Juechu and Feng, Boyuan and Guessous, Driss and Liang, Yanbo and He, Horace},
  journal={arXiv preprint arXiv:2412.05496},
  year={2024}
}
```

```bibtex
@inproceedings{paszke2019pytorch,
  title={PyTorch: An Imperative Style, High-Performance Deep Learning Library},
  author={Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and Desmaison, Alban and K{\"o}pf, Andreas and Yang, Edward and DeVito, Zach and Raison, Martin and Tejani, Alykhan and Chilamkurthy, Sasank and Steiner, Benoit and Fang, Lu and Bai, Junjie and Chintala, Soumith},
  booktitle={Advances in Neural Information Processing Systems 32},
  year={2019}
}
```
