# FastANKH

Fast, optimized implementations of ANKH protein language models (T5-based) with multi-backend attention support.

**Requires PyTorch 2.11+** for Flash Attention 4 (FA4) backend support via flex attention.

## Models

| Model | Params | Layers | Hidden | Heads | Activation | Source |
|-------|--------|--------|--------|-------|------------|--------|
| ANKH_base | ~240M | 48 | 768 | 12 | gelu_new | ElnaggarLab/ankh-base |
| ANKH_large | ~1.1B | 48 | 1536 | 16 | gelu_new | ElnaggarLab/ankh-large |
| ANKH2_large | ~550M | 24 | 1536 | 16 | silu | ElnaggarLab/ankh2-ext2 |
| ANKH3_large | ~1.1B | 48 | 1536 | 16 | silu | ElnaggarLab/ankh3-large |
| ANKH3_xl | ~3B | 48 | 2560 | 32 | silu | ElnaggarLab/ankh3-xl |

## Usage

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("Synthyra/ANKH_base", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("Synthyra/ANKH_base")

# Set attention backend before loading for best performance
# Options: "sdpa" (default, exact), "flex" (FA4 on Hopper/Blackwell)
model.config.attn_backend = "flex"

sequences = ["MKTLLILAVL", "ACDEFGHIKLMNPQRSTVWY"]
inputs = tokenizer(sequences, return_tensors="pt", padding=True)
outputs = model(**inputs)

# Per-residue embeddings
embeddings = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
```

## Batch Embedding

```python
model = AutoModel.from_pretrained("Synthyra/ANKH_base", trust_remote_code=True).to("cuda")
embeddings = model.embed_dataset(
    sequences=["MKTLLILAVL", "ACDEFGHIKLMNPQRSTVWY"],
    batch_size=8,
    max_len=512,
    full_embeddings=True,
)
```

## Attention Backends

| Backend | Key | Notes |
|---------|-----|-------|
| SDPA | `"sdpa"` | Default. Exact attention with position bias as additive mask. |
| Flex | `"flex"` | Uses FA4 on Hopper/Blackwell GPUs (PyTorch 2.11+). Position bias computed via `score_mod`. Triton fallback on older hardware. |
| Flash | `"kernels_flash"` | Not supported for ANKH (no arbitrary bias support). Falls back to flex/sdpa. |

## Architecture

ANKH models are T5 encoder-only architectures:
- **No absolute position embeddings**: Uses T5-style relative position bias (log-bucketed, bidirectional)
- **RMS LayerNorm**: No mean subtraction, no bias term
- **Gated FFN**: `activation(wi_0(x)) * wi_1(x) -> wo(x)` with gelu_new (v1) or silu (v2/v3)
- **Pre-layer normalization**: Norm before attention and FFN, residual after
- **No bias in projections**: All q/k/v/o and FFN linear layers are bias=False

The relative position bias is computed once (materialized as a full tensor) and shared across all encoder layers. For the flex backend, the bias is passed as a `score_mod` closure for optimal performance.

## Notes

- The `FastAnkhForMaskedLM` variant includes an LM head initialized from the shared embedding weights. The original ANKH models were trained with T5's span corruption objective using an encoder-decoder architecture. This encoder-only MaskedLM head is **not pre-trained for standard MLM** and requires additional fine-tuning.
- ANKH3 models use a vocabulary of 256 tokens (vs 144 for v1/v2) and were trained with dual objectives ([NLU] for embeddings, [S2S] for generation).

## Citations

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
