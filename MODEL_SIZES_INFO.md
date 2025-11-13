# BoltzGen Model Sizes and Checkpoints

## Available Model Checkpoints

### Standard Inference Checkpoint (Recommended)

**`boltz2_conf_final.ckpt`** - Full BoltzGen model with confidence prediction
- This is the **default and recommended checkpoint** for inference
- Downloaded automatically from HuggingFace: `boltzgen/boltzgen-1`
- Size: ~2GB (compressed checkpoint)
- Contains both structure prediction and confidence modules

**Architecture:**
```
Token embedding dim: 384
Token pair dim: 128  
Atom embedding dim: 128
Atom pair dim: 16
Pairformer blocks: 24
Total parameters: ~1.2B (estimated)
```

### Training Checkpoints

BoltzGen provides training configurations for different model sizes:

#### Small Model (`boltzgen_small.yaml`)
```yaml
token_s: 384
token_z: 128
atom_s: 128
atom_z: 16
pairformer_blocks: 12
```
- Recommended for development/testing
- Requires 8 GPUs with gradient accumulation
- Checkpoint: `boltzgen1_structuretrained_small.ckpt` (for resuming training)

#### Large Model (`boltzgen.yaml`)
```yaml
token_s: 384
token_z: 128
atom_s: 128
atom_z: 16
pairformer_blocks: 24
max_tokens: 512 (vs 256 for small)
max_atoms: 5120 (vs 2048 for small)
```
- Standard production model
- More capacity for complex structures

## Using Different Model Sizes

### 1. Use Default Checkpoint (Recommended)

```python
from minimal_fold_inference import load_model, fold_protein

# Uses boltz2_conf_final.ckpt automatically
model = load_model(device="cuda")
output = fold_protein(sequence="MKTAYIAK", model=model)
```

### 2. Load Custom Checkpoint (If You've Trained Your Own)

```python
from minimal_fold_inference import load_model

# Load your custom-trained small model
model = load_model(
    checkpoint_path="path/to/your_small_model.ckpt",
    device="cuda"
)

# Or load your custom-trained large model
model = load_model(
    checkpoint_path="path/to/your_large_model.ckpt",
    device="cuda"
)
```

### 3. Download Specific Checkpoint from HuggingFace

```python
from minimal_fold_inference import load_model

# Download by name (if available)
model = load_model(
    checkpoint_name="boltz2_conf_final.ckpt",  # Default
    device="cuda"
)

# For structure-only pretrained checkpoint (training use)
# Note: This is for resuming training, not for inference
model = load_model(
    checkpoint_name="boltzgen1_structuretrained_small.ckpt",
    device="cuda"
)
```

## Model Size Information Display

When you load a model, the scripts automatically display architecture information:

```
============================================================
MODEL ARCHITECTURE
============================================================
  Token embedding dim (token_s): 384
  Token pair dim (token_z): 128
  Atom embedding dim (atom_s): 128
  Atom pair dim (atom_z): 16
  Total parameters: 1,234,567,890
  Model size: ~4.58 GB (fp32)
  Pairformer blocks: 24
  Pairformer heads: 16
============================================================
```

## Important Notes

### For Inference:
- **Use `boltz2_conf_final.ckpt`** - This is the standard inference checkpoint
- There is currently **only one released inference checkpoint** (not separate small/medium/large)
- The model size is determined by the hyperparameters baked into the checkpoint

### For Training:
- BoltzGen provides **two training configurations**: small and large
- Small model: Faster training, less memory, good for development
- Large model: Better performance, more expensive to train
- Training checkpoints are separate from inference checkpoints

### Custom Checkpoints:
If you train your own model, you can use any checkpoint with our scripts:

```python
# Train your own small model
# Follow instructions in boltzgen/README.md
# Then use it:
model = load_model(
    checkpoint_path="./workdir/your_trained_model.ckpt",
    device="cuda"
)
```

## Comparison with AlphaFold/ESMFold

BoltzGen uses a different approach:

| Model | Parameters | Architecture |
|-------|-----------|--------------|
| AlphaFold2 | ~93M | Evoformer + Structure module |
| ESMFold | ~15M | Language model + Structure module |
| BoltzGen | ~1.2B | Pairformer + Atom diffusion |

BoltzGen is larger because it:
- Performs both design AND folding (generative model)
- Uses atom-level diffusion (not just backbone)
- Includes confidence prediction
- Supports proteins, peptides, nucleic acids, and small molecules

## Memory Requirements

**GPU Memory Needed (Inference):**

| Task | Sequence Length | GPU Memory (bf16) |
|------|----------------|-------------------|
| Single protein | 100 residues | ~8 GB |
| Single protein | 300 residues | ~16 GB |
| Complex (2 chains) | 200 total residues | ~20 GB |
| Design (50 residues) | + target (100 res) | ~24 GB |

**Tips to Reduce Memory:**
1. Reduce `diffusion_samples` (use 1 instead of 5)
2. Reduce `sampling_steps` (use 50-100 instead of 200)
3. Use smaller batch sizes
4. Use CPU (slower but no memory limit)

## Training Your Own Model

To train a custom-sized model:

1. **Modify the config file** (`boltzgen/config/train/boltzgen_small.yaml`)
   ```yaml
   model:
     token_s: 256  # Make smaller
     token_z: 64   # Make smaller
     pairformer_args:
       num_blocks: 8  # Reduce layers
   ```

2. **Train the model** (see BoltzGen training docs)

3. **Use your checkpoint** in inference:
   ```python
   model = load_model(
       checkpoint_path="./workdir/my_custom_model.ckpt",
       device="cuda"
   )
   ```

## Future Releases

The BoltzGen team may release additional checkpoints in the future. Check:
- HuggingFace: https://huggingface.co/boltzgen/boltzgen-1
- GitHub: https://github.com/HannesStark/boltzgen

## Summary

✅ **For most users**: Use the default `boltz2_conf_final.ckpt` (loaded automatically)
✅ **For custom training**: Train your own small/large model, then load it with `checkpoint_path`
❌ **Not available**: Pre-trained small/medium/large inference checkpoints (only one standard checkpoint)

The model size is **fixed in the checkpoint** - you can't change the size without retraining.

