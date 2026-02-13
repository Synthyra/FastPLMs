---
library_name: transformers
tags: []
---

# NOTE
The GitHub with the implementation and requirements can be found [here](https://github.com/Synthyra/FastPLMs.git).

# Boltz2 AutoModel (Inference-only)
This is a barebones Huggingface `AutoModel` compatible implementation of Boltz2 focused on fast inference workflows.

The implementation is located in `boltz_automodel/` and exposes:
- `Boltz2Config`
- `Boltz2Model`
- `predict_structure(amino_acid_sequence, ...)`
- `save_as_cif(structure_output, output_path, ...)`

## Design goals
- Inference-only (no training hooks, no Lightning trainer usage).
- Lightweight runtime around `torch` + `transformers` (plus `numpy`).
- AutoModel remote-code compatibility via `trust_remote_code=True`.
- Confidence outputs included in prediction outputs (`plddt`, `ptm`, `iptm`, and derived confidence score when available).

## Runtime note
This implementation is self-contained inside `boltz_automodel/` and does not require
the original cloned `boltz` package at runtime.

## Use with transformers

### Load from an exported directory
```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "Synthyra/Boltz2",
    trust_remote_code=True,
    dtype=torch.float32,
).eval()
```

### Predict structure from sequence
```python
out = model.predict_structure(
    amino_acid_sequence="MSTNPKPQRKTKRNTNRRPQDVKFPGG",
    recycling_steps=3,
    num_sampling_steps=200,
    diffusion_samples=1,
)

print(out.sample_atom_coords.shape)
print(None if out.plddt is None else out.plddt.shape)
```

### Save CIF
```python
model.save_as_cif(out, "prediction.cif")
```

## Convert Boltz checkpoint to HF export
Use:

```bash
py -m boltz_automodel.get_boltz2_weights --checkpoint_path boltz_automodel/weights/boltz2_conf.ckpt --output_dir boltz2_automodel_export
```

The export directory contains:
- `config.json`
- `pytorch_model.bin`
- `modeling_boltz2.py`
- `minimal_featurizer.py`
- `minimal_structures.py`
- `cif_writer.py`
- `vb_*.py` (self-contained vendored Boltz2 inference modules/constants)

## Output object fields
`predict_structure(...)` returns `Boltz2StructureOutput` with:
- `sample_atom_coords`
- `atom_pad_mask`
- `plddt`
- `complex_plddt`
- `ptm`
- `iptm`
- `confidence_score` (derived when available)
- `raw_output`

## Limitations
- Current featurization path is protein-only and minimal.
- This implementation is meant for practical inference and export workflows, not full Boltz training parity.

## Docker-first compliance testing

Build the container at repo root:

```bash
docker build -t fastplms-test -f Dockerfile .
```

Launch a test shell:

```bash
docker run --rm --gpus all -it -v ${PWD}:/workspace fastplms-test bash
```

Inside the container, run Boltz2 compliance against pip `boltz`:

```bash
python -m test_scripts.run_boltz2_compliance --device cuda --dtype float32 --seed 42 --num-sequences 3 --recycling-steps 3 --num-sampling-steps 200 --diffusion-samples 1 --pass-coord-metric aligned --write-cif-artifacts
```

Artifacts are written to `test_scripts/results/<timestamp>/boltz2_compliance/` by default:
- `metrics.json`
- `metrics.csv`
- `summary.txt`
- `structures/seq_<idx>/ours_seq<idx>.cif`
- `structures/seq_<idx>/ref_seq<idx>.cif`

Coordinate metrics now include both raw and rigid-aligned variants:
- `coord_mae`, `coord_rmse`, `coord_max_abs` (raw frame-dependent deltas)
- `coord_mae_aligned`, `coord_rmse_aligned`, `coord_max_abs_aligned` (Kabsch aligned)
- `pairwise_dist_mae` (frame-invariant pairwise-distance delta)

Pass/fail uses `--pass-coord-metric aligned` by default. Set `--pass-coord-metric raw` to use the raw coordinate thresholds.

