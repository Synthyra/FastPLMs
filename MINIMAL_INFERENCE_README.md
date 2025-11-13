# BoltzGen Minimal Inference Scripts

This directory contains minimal, function-based inference scripts for BoltzGen that bypass the file-based configuration system and allow direct programmatic access.

## Files

- **`minimal_fold_inference.py`**: Fold proteins from sequence
- **`minimal_design_inference.py`**: Design proteins against targets

## Installation

```bash
# Install dependencies
pip install torch numpy huggingface_hub

# The scripts will automatically use the boltzgen code from the boltzgen/ directory
```

## Model Sizes and Checkpoints

**Default Checkpoint**: `boltz2_conf_final.ckpt` (~2GB, downloaded automatically)
- This is the **only released inference checkpoint** from BoltzGen
- Model size is fixed at ~1.2B parameters
- For custom model sizes, you need to train your own checkpoint

See **`MODEL_SIZES_INFO.md`** for detailed information about:
- Model architecture details
- Training your own small/large models
- Memory requirements
- Custom checkpoint usage

## Usage

### 1. Folding a Single Protein

```python
from minimal_fold_inference import fold_protein

# Fold a protein from sequence
output = fold_protein(
    sequence="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEV",
    recycling_steps=3,
    sampling_steps=200,
    diffusion_samples=5,
    device="cuda"  # or "cpu"
)

# Access results
coords = output['sample_atom_coords']  # Shape: [num_samples, num_atoms, 3]
ptm = output['ptm']  # Predicted TM-score
```

### 2. Folding a Protein Complex

```python
from minimal_fold_inference import fold_complex

# Fold a multi-chain complex
output = fold_complex(
    sequences=[
        "MKTAYIAKQRQISFVKSHFSRQLE",  # Chain A
        "AAAAAAVTTTTPPPAAAAAA",       # Chain B
    ],
    recycling_steps=3,
    sampling_steps=200,
    diffusion_samples=5,
    device="cuda"
)

# Access results
coords = output['sample_atom_coords']
iptm = output['iptm']  # Interface predicted TM-score
```

### 3. Designing a Protein De Novo

```python
from minimal_design_inference import design_protein

# Design a protein with specified secondary structure
output = design_protein(
    design_length=50,
    secondary_structure="HHHHHHHHHEEEEEEELLLLLHHHHHHHHHEEEEEEELLLLL",
    recycling_steps=3,
    sampling_steps=500,
    diffusion_samples=1,
    step_scale=1.8,
    noise_scale=0.95,
    device="cuda"
)

# Access results
coords = output['sample_atom_coords']
sequence_logits = output['res_type_logits']  # Predicted sequences
```

### 4. Designing a Protein Against a Target

```python
from minimal_design_inference import design_protein

# Design a binder against a target structure
output = design_protein(
    design_length=30,
    target_cif_path="path/to/target.cif",
    recycling_steps=3,
    sampling_steps=500,
    diffusion_samples=1,
    device="cuda"
)

# Access results
coords = output['sample_atom_coords']
iptm = output['iptm']  # Interface quality
design_ptm = output['design_ptm']  # Design quality
```

## Key Parameters

### Folding Parameters

- **`sequence`**: Amino acid sequence (single letter codes)
- **`sequences`**: List of sequences for complex prediction
- **`recycling_steps`**: Number of recycling iterations (default: 3)
  - More recycling = better quality but slower
- **`sampling_steps`**: Number of diffusion steps (default: 200 for folding)
  - More steps = better quality but slower
- **`diffusion_samples`**: Number of samples to generate
  - Generate multiple samples and pick the best

### Design Parameters

- **`design_length`**: Number of residues to design
- **`target_cif_path`**: Path to target structure (optional, for binder design)
- **`secondary_structure`**: Secondary structure specification (optional)
  - `'H'` = helix, `'E'` = sheet, `'L'` = loop
  - Example: `"HHHHHEEELLL"` for a helix-sheet-loop motif
- **`sampling_steps`**: Number of diffusion steps (default: 500 for design)
- **`step_scale`**: Diffusion step scale (default: 1.8)
  - Higher values = more diverse designs
- **`noise_scale`**: Diffusion noise scale (default: 0.95)
  - Lower values = more diverse designs

## Output Format

All functions return a dictionary with:

### Common Outputs

- **`sample_atom_coords`**: `torch.Tensor [num_samples, num_atoms, 3]`
  - Predicted atomic coordinates in Ångströms
- **`ptm`**: Predicted TM-score (folding quality)
- **`pae`**: Predicted aligned error matrix

### Folding-Specific Outputs

- **`pdistogram`**: Predicted distance distribution

### Complex-Specific Outputs

- **`iptm`**: Interface predicted TM-score
- **`interaction_pae`**: Interface PAE

### Design-Specific Outputs

- **`res_type_logits`**: `torch.Tensor [num_samples, num_tokens, num_aa_types]`
  - Predicted amino acid probabilities for each position
- **`design_ptm`**: Design quality score
- **`target_ptm`**: Target structure quality score
- **`design_to_target_iptm`**: Design-to-target interface score

## Example Scripts

Run the included examples:

```bash
# Test folding
py minimal_fold_inference.py

# Test design
py minimal_design_inference.py
```

## Advanced Usage

### Loading Model Once for Multiple Predictions

```python
from minimal_fold_inference import load_model, fold_protein

# Load model once (uses default checkpoint)
model = load_model(device="cuda")

# Use for multiple predictions
for seq in sequences:
    output = fold_protein(
        sequence=seq,
        model=model,  # Reuse loaded model
        device="cuda"
    )
```

### Using Custom Checkpoints

```python
from minimal_fold_inference import load_model

# Load your own trained checkpoint
model = load_model(
    checkpoint_path="path/to/your_custom_model.ckpt",
    device="cuda"
)

# Or specify a different checkpoint name from HuggingFace
model = load_model(
    checkpoint_name="boltz2_conf_final.ckpt",  # Default
    device="cuda"
)

# The model will automatically display its architecture info:
# - Token/atom embedding dimensions
# - Total parameters
# - Model size in GB
# - Pairformer configuration
```

### Extracting Designed Sequences

```python
from minimal_design_inference import design_protein, extract_designed_sequence

output = design_protein(design_length=50, ...)

# Get the design mask from features
design_mask = output['design_mask']  # If available

# Extract sequence
sequence = extract_designed_sequence(output, design_mask)
print(f"Designed sequence: {sequence}")
```

## Working with Raw Output

The scripts return raw PyTorch tensors. To save results:

```python
import torch
import numpy as np

# Save coordinates to numpy
coords = output['sample_atom_coords'].cpu().numpy()
np.save('predicted_coords.npy', coords)

# Save as PDB/CIF (implement your own writer or use BoltzGen's)
# See boltzgen/src/boltzgen/data/write/ for reference
```

## Performance Tips

1. **Use GPU**: Much faster than CPU
2. **Batch predictions**: Load model once, reuse for multiple predictions
3. **Reduce sampling steps**: For testing, use 50-100 steps
4. **Reduce diffusion samples**: Use 1 sample for testing
5. **Mixed precision**: Model uses bf16-mixed by default (faster on modern GPUs)

## Differences from Official BoltzGen CLI

### Advantages of Minimal Scripts

✅ Direct function calls - no YAML files
✅ Easy integration into pipelines
✅ Direct access to raw outputs
✅ Simple Python API

### Limitations

❌ No automatic file I/O (CIF/PDB writing)
❌ No MSA support
❌ No template support
❌ Simplified feature creation
❌ No built-in metrics computation

For production use with MSAs, templates, and full features, use the official BoltzGen CLI:

```bash
boltzgen run design_spec.yaml --output results/
```

## Troubleshooting

### Out of Memory

- Reduce `diffusion_samples`
- Reduce sequence length
- Use CPU instead of GPU
- Close other GPU processes

### Slow Inference

- Reduce `sampling_steps` (e.g., 50 for testing)
- Reduce `recycling_steps` (e.g., 1 for testing)
- Use GPU instead of CPU

### Import Errors

Make sure the `boltzgen/` directory is in the same folder as these scripts.

## Next Steps

After running inference, you might want to:

1. **Save structures**: Implement CIF/PDB writers
2. **Visualize**: Use PyMOL, ChimeraX, or molecular viewers
3. **Analyze**: Compute RMSD, TM-score, interface metrics
4. **Filter designs**: Rank by confidence metrics (ptm, iptm, pae)
5. **Refinement**: Use Rosetta or other tools for final refinement

## Credits

Based on BoltzGen by the MIT CSAIL group.
Minimal inference scripts created for easy integration and programmatic access.

