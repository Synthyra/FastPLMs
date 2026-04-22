# Per-Model Guides

This document covers each model family supported by FastPLMs: loading, configuration, special handling, and available checkpoints.

All sequence models (ESM2, ESM++, E1, DPLM, DPLM2, ANKH) share the same embedding pipeline via `EmbeddingMixin`. They support most attention backends, with one exception: ANKH supports only `sdpa` and `flex` (T5 relative position bias is incompatible with the flash-attention kernels). Structure prediction models (Boltz2, ESMFold) have their own APIs.

---

## ESM2

**Organization:** Meta AI
**Architecture:** Transformer encoder with rotary position embeddings (RoPE)
**Checkpoints:** 8M, 35M, 150M, 650M, 3B

### Loading

```python
from transformers import AutoModelForMaskedLM, AutoConfig

# Default (SDPA backend)
model = AutoModelForMaskedLM.from_pretrained("Synthyra/ESM2-150M", trust_remote_code=True)

# With a specific backend
config = AutoConfig.from_pretrained("Synthyra/ESM2-150M", trust_remote_code=True)
config.attn_backend = "flex"
model = AutoModelForMaskedLM.from_pretrained("Synthyra/ESM2-150M", config=config, trust_remote_code=True)
```

### Key Details

- Uses the standard ESM tokenizer (`EsmTokenizer` from `transformers`)
- Tokenizer accessible via `model.tokenizer`
- Backend can be set on the config before `from_pretrained` OR via the mutable `model.attn_backend` property after load (same mechanism as every other family).
- Pre-LayerNorm architecture with a final `emb_layer_norm_after`
- Supports `output_attentions=True` and `output_hidden_states=True`

### Available Checkpoints

| Checkpoint | HuggingFace ID | Official Reference |
|------------|----------------|-------------------|
| ESM2-8M | `Synthyra/ESM2-8M` | `facebook/esm2_t6_8M_UR50D` |
| ESM2-35M | `Synthyra/ESM2-35M` | `facebook/esm2_t12_35M_UR50D` |
| ESM2-150M | `Synthyra/ESM2-150M` | `facebook/esm2_t30_150M_UR50D` |
| ESM2-650M | `Synthyra/ESM2-650M` | `facebook/esm2_t33_650M_UR50D` |
| ESM2-3B | `Synthyra/ESM2-3B` | `facebook/esm2_t36_3B_UR50D` |

---

## ESM++ (ESMC)

**Organization:** EvolutionaryScale
**Architecture:** Transformer encoder with configurable rotary embeddings (scaling, interleaving)
**Checkpoints:** Small (300M), Large (600M)

### Loading

```python
from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("Synthyra/ESMplusplus_small", trust_remote_code=True)
```

### Key Details

- Uses the ESM tokenizer (same as ESM2)
- **Requires `sequence_id`** parameter for batched inference: `sequence_id = attention_mask.to(dtype=torch.bool)`
- Uses `einops` for tensor reshaping operations
- Rotary embeddings support `interleaved` mode and `scale_base`/`scaling_factor` for dynamic scaling
- Backend can be set on the config before `from_pretrained` OR via the mutable `model.attn_backend` property after load.

### Batched Forward Pass

```python
tokenizer = model.tokenizer
tokenized = tokenizer(sequences, return_tensors="pt", padding=True)
tokenized = {k: v.to(device) for k, v in tokenized.items()}
tokenized["sequence_id"] = tokenized["attention_mask"].to(dtype=torch.bool)

with torch.inference_mode():
    output = model(**tokenized)
```

### Available Checkpoints

| Checkpoint | HuggingFace ID | Official Reference |
|------------|----------------|-------------------|
| ESM++ Small (300M) | `Synthyra/ESMplusplus_small` | `EvolutionaryScale/esmc-300m-2024-12` |
| ESM++ Large (600M) | `Synthyra/ESMplusplus_large` | `EvolutionaryScale/esmc-600m-2024-12` |

---

## E1

**Organization:** Profluent Bio
**Architecture:** Transformer with within-sequence and global (block-causal) attention layers
**Checkpoints:** 150M, 300M, 600M

### Loading

```python
from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("Synthyra/Profluent-E1-150M", trust_remote_code=True)
```

### Key Details

- **Sequence mode**: E1 does not use a standard HuggingFace tokenizer. Tokenization is built into the model via `E1BatchPreparer`
- Uses RMSNorm (not LayerNorm)
- Grouped-query attention (num_key_value_heads < num_heads)
- Two attention layer types that alternate:
  - `WITHIN_SEQ`: Attention within individual sequences only
  - `GLOBAL`: Cross-sequence block-causal attention
- Separate RoPE configurations for within-sequence and global attention (different `rope_theta`)
- KV caching via `DynamicCache` for efficient generation
- Backend can be set on the config before `from_pretrained` OR via the mutable `model.attn_backend` property after load.

### Tokenization (Sequence Mode)

E1's tokenization happens via `model.model.prep_tokens`:

```python
batch_kwargs = model.model.prep_tokens.get_batch_kwargs(sequences, device=device)
# Returns dict with:
#   input_ids, within_seq_position_ids, global_position_ids,
#   sequence_ids, labels, context, context_len
```

For the `embed_dataset()` pipeline, pass `tokenizer=None` and the mixin handles E1's sequence mode automatically.

### Embedding Extraction

```python
embeddings = model.embed_dataset(
    sequences=["MKTLLILAVVAAALA", "MALWMRLLPLLALL"],
    batch_size=2,
    tokenizer=None,  # E1 uses sequence mode
    pooling_types=["mean"],
    save=False,
)
```

### Available Checkpoints

| Checkpoint | HuggingFace ID | Official Reference |
|------------|----------------|-------------------|
| E1-150M | `Synthyra/Profluent-E1-150M` | `Profluent-Bio/E1-150m` |
| E1-300M | `Synthyra/Profluent-E1-300M` | `Profluent-Bio/E1-300m` |
| E1-600M | `Synthyra/Profluent-E1-600M` | `Profluent-Bio/E1-600m` |

### Compliance Dependencies

E1 compliance tests require the official E1 package:

```bash
pip install E1 @ git+https://github.com/Profluent-AI/E1.git
```

This is pre-installed in the Docker image.

---

## DPLM

**Organization:** ByteDance
**Architecture:** Diffusion-optimized transformer based on ESM2 architecture
**Checkpoints:** 150M, 650M, 3B

### Loading

```python
from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("Synthyra/DPLM-150M", trust_remote_code=True)
```

### Key Details

- Uses the ESM tokenizer (same as ESM2)
- **Mutable `attn_backend` property**: Unlike ESM2/ESM++/E1, DPLM exposes `model.attn_backend` as a property that propagates backend changes to all attention layers in-place
- Architecture extends `EsmConfig` and `EsmPreTrainedModel` from HuggingFace
- Supports cross-attention and KV caching for generation
- `ModifiedEsmSelfAttention` extends the official `EsmSelfAttention` with multi-backend support

### Post-Load Backend Switching

```python
model = AutoModelForMaskedLM.from_pretrained("Synthyra/DPLM-150M", trust_remote_code=True)
model.attn_backend = "flex"  # Updates every attention layer in-place
```

### Available Checkpoints

| Checkpoint | HuggingFace ID | Official Reference |
|------------|----------------|-------------------|
| DPLM-150M | `Synthyra/DPLM-150M` | `airkingbd/dplm_150m` |
| DPLM-650M | `Synthyra/DPLM-650M` | `airkingbd/dplm_650m` |
| DPLM-3B | `Synthyra/DPLM-3B` | `airkingbd/dplm_3b` |

---

## DPLM2

**Organization:** ByteDance
**Architecture:** Multimodal diffusion transformer handling both sequence and structure tokens
**Checkpoints:** 150M, 650M, 3B

### Loading

```python
from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("Synthyra/DPLM2-150M", trust_remote_code=True)
```

### Key Details

- Uses the ESM tokenizer
- **Multimodal input**: Handles both amino acid tokens and structure tokens in packed sequences
- Mutable `model.attn_backend` property (same as DPLM)
- Special token normalization: `_normalize_dplm2_input_ids()` maps tokens above vocab_size back into range
- Packed multimodal layout detection: `_has_packed_multimodal_layout()` checks if input_ids contain interleaved AA and structure tokens
- The official DPLM2 has an extra `contact_head` not present in the FastPLM version, so weight compliance testing is skipped for this family

### Available Checkpoints

| Checkpoint | HuggingFace ID | Official Reference |
|------------|----------------|-------------------|
| DPLM2-150M | `Synthyra/DPLM2-150M` | `airkingbd/dplm2_150m` |
| DPLM2-650M | `Synthyra/DPLM2-650M` | `airkingbd/dplm2_650m` |
| DPLM2-3B | `Synthyra/DPLM2-3B` | `airkingbd/dplm2_3b` |

---

## ANKH

**Organization:** Elnaggar Lab
**Architecture:** T5-style encoder with bidirectional gated GELU FFN and learned relative position bias (bucketed)
**Checkpoints:** Base, Large, ANKH2-Large, ANKH3-Large, ANKH3-XL

### Loading

```python
from transformers import AutoModelForMaskedLM, AutoConfig

# Default (SDPA)
model = AutoModelForMaskedLM.from_pretrained("Synthyra/ANKH_base", trust_remote_code=True)

# Flex backend (block-mask aware)
config = AutoConfig.from_pretrained("Synthyra/ANKH_base", trust_remote_code=True)
config.attn_backend = "flex"
model = AutoModelForMaskedLM.from_pretrained("Synthyra/ANKH_base", config=config, trust_remote_code=True)
```

### Key Details

- Uses the ANKH T5 tokenizer (`AutoTokenizer.from_pretrained("ElnaggarLab/ankh-base")`)
- Tokenizer accessible via `model.tokenizer`
- Backend can be set on the config before `from_pretrained` OR via the mutable `model.attn_backend` property after load (same mechanism as every other family).
- **Attention is unscaled** (no `1/sqrt(d_kv)` factor). T5 trains without scaling; the learned relative position bias absorbs the temperature.
- Only `sdpa` and `flex` are supported. Requesting `kernels_flash` silently falls back to `flex` (or `sdpa` if flex is unavailable) because flash kernels can't accept additive position bias.
- Layer 0 owns the relative-position-bias `nn.Embedding`; subsequent layers receive the precomputed bias through the forward call.
- The native ANKH checkpoint is a T5 encoder-decoder; FastPLMs uses the encoder only and bolts on a separate `lm_head` for the `ForMaskedLM` variant. For weight-parity comparisons against `transformers.T5EncoderModel`, the FastPLMs `lm_head.weight` is allowlisted as an expected extra parameter.

### Available Checkpoints

| Checkpoint | HuggingFace ID | Official Reference |
|------------|----------------|-------------------|
| ANKH-base | `Synthyra/ANKH_base` | `ElnaggarLab/ankh-base` |
| ANKH-large | `Synthyra/ANKH_large` | `ElnaggarLab/ankh-large` |
| ANKH2-large | `Synthyra/ANKH2_large` | `ElnaggarLab/ankh2-ext2` |
| ANKH3-large | `Synthyra/ANKH3_large` | `ElnaggarLab/ankh3-large` |
| ANKH3-XL | `Synthyra/ANKH3_xl` | `ElnaggarLab/ankh3-xl` |

---

## Boltz2

**Organization:** MIT / Various
**Architecture:** Diffusion-based structure prediction model
**Checkpoints:** Standard

### Loading

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("Synthyra/Boltz2", trust_remote_code=True, dtype=torch.float32)
```

### Key Details

- **Structure prediction model**, not a sequence encoder. Does not inherit from `EmbeddingMixin`
- Uses `AutoModel` (not `AutoModelForMaskedLM`)
- Custom featurization pipeline via `minimal_featurizer.build_boltz2_features()`
- Outputs atomic coordinates, pLDDT, pTM, ipTM confidence scores
- Can export predictions as CIF files via `model.save_as_cif(output, "pred.cif")`

### Predict Structure

```python
output = model.predict_structure(
    amino_acid_sequence="MSTNPKPQRKTKRNTNRRPQDVKFPGG",
    recycling_steps=3,
    num_sampling_steps=200,
    diffusion_samples=1,
)
print(output.sample_atom_coords.shape)  # (N_atoms, 3)
print(output.plddt.shape)               # (N_residues,)
```

### Compliance Testing

Boltz2 compliance is tested via a standalone script (`testing/run_boltz2_compliance.py`) that compares coordinates, pairwise distances, and TM-scores against the official implementation.

---

## ESMFold

**Organization:** Meta AI (wrapped by Synthyra)
**Architecture:** ESM2 backbone + structure module with optional Test-Time Training (TTT)
**Checkpoints:** Standard

### Loading

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("Synthyra/FastESMFold", trust_remote_code=True, dtype=torch.float32)
```

### Key Details

- Inherits from `transformers.EsmForProteinFolding` with the ESM2 backbone replaced by `FastEsmBackbone`
- Supports all attention backends via `config.attn_backend`
- Optional Test-Time Training (TTT): Adapts the ESM2 backbone via LoRA + masked LM before folding
- TTT typically improves low-confidence sequences (baseline pLDDT < 0.5) by 10-30+ points

### Structure Prediction

```python
# Without TTT
with torch.no_grad():
    output = model.infer("MKTLLILAVVAAALA")
pdb_string = model.output_to_pdb(output)

# With TTT (default: 10 optimizer steps)
result = model.fold_protein("MKTLLILAVVAAALA")
# result = {"plddt": float, "ptm": float, "pdb_string": str, ...}
```

### Disabling TTT

```python
from transformers import AutoConfig, AutoModel

config = AutoConfig.from_pretrained("Synthyra/FastESMFold", trust_remote_code=True)
config.ttt_config = {"steps": 0}
model = AutoModel.from_pretrained("Synthyra/FastESMFold", config=config, trust_remote_code=True)
```
