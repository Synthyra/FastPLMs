# BoltzGen Checkpoint Usage Examples

## Quick Reference

### Default Usage (Recommended)
```python
from minimal_fold_inference import load_model, fold_protein

# Automatically downloads and loads boltz2_conf_final.ckpt
model = load_model(device="cuda")
output = fold_protein(sequence="MKTAYIAK", model=model)
```

### Custom Checkpoint from Local Path
```python
from minimal_fold_inference import load_model

# Load your own trained model
model = load_model(
    checkpoint_path="/path/to/my_trained_model.ckpt",
    device="cuda"
)
```

### Specify Checkpoint Name from HuggingFace
```python
from minimal_fold_inference import load_model

# Download specific checkpoint by name
model = load_model(
    checkpoint_name="boltz2_conf_final.ckpt",  # Default
    device="cuda"
)
```

## Complete Examples

### Example 1: Default Inference
```python
"""Standard folding with default checkpoint"""
from minimal_fold_inference import fold_protein

# No need to load model manually - handled automatically
output = fold_protein(
    sequence="MKTAYIAKQRQISFVKSHFSRQLE",
    recycling_steps=3,
    sampling_steps=200,
    diffusion_samples=5,
    device="cuda"
)

print(f"Coordinates: {output['sample_atom_coords'].shape}")
print(f"PTM score: {output['ptm'].item():.3f}")
```

### Example 2: Load Model Once, Use Multiple Times
```python
"""Efficient batch processing"""
from minimal_fold_inference import load_model, fold_protein

# Load model once
print("Loading model...")
model = load_model(device="cuda")

# Process multiple sequences
sequences = [
    "MKTAYIAKQRQISFVK",
    "SHFSRQLEERLGLIEV",
    "APILSRVGDGTQDNLS",
]

results = []
for i, seq in enumerate(sequences):
    print(f"\nFolding sequence {i+1}/{len(sequences)}...")
    output = fold_protein(
        sequence=seq,
        model=model,  # Reuse loaded model
        recycling_steps=3,
        sampling_steps=100,  # Reduced for speed
        diffusion_samples=1,
        device="cuda"
    )
    results.append({
        'sequence': seq,
        'coords': output['sample_atom_coords'],
        'ptm': output['ptm'].item()
    })

# Find best prediction
best = max(results, key=lambda x: x['ptm'])
print(f"\nBest prediction: PTM = {best['ptm']:.3f}")
```

### Example 3: Using Your Own Trained Checkpoint
```python
"""Load a custom-trained model"""
from minimal_fold_inference import load_model, fold_protein

# Suppose you trained a small model following BoltzGen training docs
model = load_model(
    checkpoint_path="./workdir/my_small_model_epoch_50.ckpt",
    device="cuda"
)

# Model info is automatically displayed:
# ============================================================
# MODEL ARCHITECTURE
# ============================================================
#   Token embedding dim (token_s): 256
#   Token pair dim (token_z): 64
#   Atom embedding dim (atom_s): 128
#   Atom pair dim (atom_z): 16
#   Total parameters: 500,000,000
#   Model size: ~1.86 GB (fp32)
#   Pairformer blocks: 8
#   Pairformer heads: 8
# ============================================================

# Use your custom model for inference
output = fold_protein(
    sequence="MKTAYIAK",
    model=model,
    device="cuda"
)
```

### Example 4: Design with Custom Checkpoint
```python
"""Protein design with custom model"""
from minimal_design_inference import load_model, design_protein

# Load your custom design checkpoint
model = load_model(
    checkpoint_path="./my_checkpoints/design_model.ckpt",
    device="cuda"
)

# Design a protein binder
output = design_protein(
    design_length=30,
    target_cif_path="target.cif",
    model=model,
    recycling_steps=3,
    sampling_steps=500,
    diffusion_samples=1,
    device="cuda"
)

print(f"Design coordinates: {output['sample_atom_coords'].shape}")
print(f"iPTM: {output['iptm'].item():.3f}")
```

### Example 5: Comparing Multiple Model Versions
```python
"""Compare predictions from different model checkpoints"""
from minimal_fold_inference import load_model, fold_protein

sequence = "MKTAYIAKQRQISFVKSHFSRQLE"

# Load different model versions
models = {
    'standard': load_model(device="cuda"),
    'custom_v1': load_model(checkpoint_path="./models/v1.ckpt", device="cuda"),
    'custom_v2': load_model(checkpoint_path="./models/v2.ckpt", device="cuda"),
}

# Run predictions with each model
results = {}
for name, model in models.items():
    print(f"\nRunning inference with {name}...")
    output = fold_protein(
        sequence=sequence,
        model=model,
        recycling_steps=3,
        sampling_steps=100,
        diffusion_samples=1,
        device="cuda"
    )
    results[name] = {
        'coords': output['sample_atom_coords'],
        'ptm': output['ptm'].item()
    }
    print(f"  PTM: {results[name]['ptm']:.3f}")

# Compare results
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
for name, result in results.items():
    print(f"{name:15s}: PTM = {result['ptm']:.3f}")
```

### Example 6: Model Architecture Inspection
```python
"""Inspect model architecture without running inference"""
import torch
from minimal_fold_inference import load_model

# Load model (downloads if needed)
model = load_model(device="cpu")  # Use CPU for inspection

# Architecture info is printed automatically when loading
# You can also access config details:
print("\nModel details:")
print(f"  Has confidence module: {hasattr(model, 'confidence_module')}")
print(f"  Has inverse folding: {model.inverse_fold if hasattr(model, 'inverse_fold') else False}")
print(f"  Number of pairformer blocks: {model.pairformer_module}")

# Count parameters by module
for name, module in model.named_children():
    params = sum(p.numel() for p in module.parameters())
    print(f"  {name}: {params:,} parameters")
```

### Example 7: Handling Different Checkpoint Formats
```python
"""Handle various checkpoint formats"""
import torch
from minimal_fold_inference import Boltz

def load_any_checkpoint(path: str, device: str = "cuda"):
    """Load checkpoint regardless of format"""
    print(f"Loading checkpoint from {path}...")
    
    # Load checkpoint
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    
    # Extract config and state dict (format may vary)
    if "hyper_parameters" in ckpt:
        # BoltzGen format
        config = ckpt["hyper_parameters"]
        state_dict = ckpt["state_dict"]
    elif "config" in ckpt and "model" in ckpt:
        # Alternative format
        config = ckpt["config"]
        state_dict = ckpt["model"]
    else:
        raise ValueError("Unknown checkpoint format")
    
    # Set inference defaults
    config["validators"] = None
    config["inference_logging"] = False
    
    # Create and load model
    model = Boltz(**config)
    
    # Handle EMA weights
    if any(k.startswith("ema.") for k in state_dict.keys()):
        state_dict = {
            k.replace("ema.", ""): v
            for k, v in state_dict.items()
            if k.startswith("ema.")
        }
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    return model

# Use the custom loader
model = load_any_checkpoint("./my_checkpoint.ckpt", device="cuda")
```

## Checkpoint Management Tips

### 1. Cache Checkpoints Locally
```python
import os
import huggingface_hub

# Download once, reuse many times
cache_dir = "./model_cache"
os.makedirs(cache_dir, exist_ok=True)

checkpoint_path = huggingface_hub.hf_hub_download(
    repo_id="boltzgen/boltzgen-1",
    filename="boltz2_conf_final.ckpt",
    cache_dir=cache_dir,
    repo_type="model",
)

# Use cached checkpoint
from minimal_fold_inference import load_model
model = load_model(checkpoint_path=checkpoint_path, device="cuda")
```

### 2. Version Control for Custom Checkpoints
```bash
# Organize your trained checkpoints
models/
├── small/
│   ├── epoch_10.ckpt
│   ├── epoch_20.ckpt
│   └── best.ckpt
├── large/
│   ├── epoch_50.ckpt
│   └── best.ckpt
└── experiments/
    ├── exp1_custom_arch.ckpt
    └── exp2_modified_loss.ckpt
```

### 3. Model Selection Based on Task
```python
"""Choose checkpoint based on task requirements"""

def get_model_for_task(task: str, device: str = "cuda"):
    """Load appropriate model for specific task"""
    
    if task == "folding":
        # Standard folding: use default
        return load_model(device=device)
    
    elif task == "design":
        # Design: use default with design masking
        model = load_model(device=device)
        # Design mode is handled by masker
        return model
    
    elif task == "fast_screening":
        # Fast screening: use small model if available
        return load_model(
            checkpoint_path="./models/small/best.ckpt",
            device=device
        )
    
    elif task == "high_accuracy":
        # High accuracy: use large model if available
        return load_model(
            checkpoint_path="./models/large/best.ckpt",
            device=device
        )
    
    else:
        raise ValueError(f"Unknown task: {task}")

# Usage
model = get_model_for_task("folding")
```

## Troubleshooting

### Issue: Checkpoint Download Fails
```python
# Solution: Specify cache directory
import os
os.environ['HF_HOME'] = '/path/to/cache'

from minimal_fold_inference import load_model
model = load_model(device="cuda")
```

### Issue: Checkpoint Format Mismatch
```python
# Solution: Check checkpoint contents
import torch
ckpt = torch.load("checkpoint.ckpt", map_location="cpu")
print("Keys in checkpoint:", ckpt.keys())

# Expected keys: "hyper_parameters", "state_dict", "epoch", etc.
```

### Issue: Out of Memory When Loading Large Model
```python
# Solution 1: Load on CPU first
model = load_model(device="cpu")
# Then move specific modules to GPU as needed

# Solution 2: Use smaller checkpoint
model = load_model(
    checkpoint_path="./models/small_model.ckpt",
    device="cuda"
)
```

## Summary

✅ **Default usage**: Just call `load_model()` - handles everything automatically
✅ **Custom checkpoints**: Pass `checkpoint_path="your_path.ckpt"`
✅ **Reuse models**: Load once, use for multiple predictions
✅ **Inspect architecture**: Model info displayed automatically when loading

For more details, see **`MODEL_SIZES_INFO.md`**.

