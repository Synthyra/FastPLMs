---
library_name: transformers
tags:
  - biology
  - protein-structure
  - esmfold2
  - multimodal-protein-model
---

# FastPLMs ESMFold2

FastPLMs ESMFold2 is a self-contained Hugging Face `AutoModel` wrapper for Biohub's ESMFold2 and ESMFold2-Fast structure predictors. It vendors the released Biohub ESMFold2 model code, ESMC backbone code, input builder, MSA helpers, and structure export utilities needed for remote-code loading.

## Load With AutoModel

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "Synthyra/ESMFold2-Fast",
    trust_remote_code=True,
    dtype=torch.float32,
).eval().cuda()
```

Use `Synthyra/ESMFold2` for the full model and `Synthyra/ESMFold2-Fast` for the faster release variant.
The folding trunk runs in fp32; the 6B ESMC backbone is loaded in bf16 by default via `esmc_precision="bf16"`.

## Fold One Protein

```python
sequence = "MKTLLILAVVAAALA"

result = model.fold_protein(
    sequence,
    num_loops=3,
    num_sampling_steps=50,
    num_diffusion_samples=1,
    seed=0,
)

print(float(result.plddt.mean()))
print(float(result.ptm))
```

## Save mmCIF or PDB

```python
model.save_as_cif(result, "prediction.cif")
model.save_as_pdb(result, "prediction.pdb")

cif_text = model.result_to_cif(result)
pdb_text = model.result_to_pdb(result)
```

`result_to_cif` preserves the full `MolecularComplex`. `result_to_pdb` converts through Biohub's protein-only `ProteinComplex` representation, so use mmCIF for complexes with ligands or nucleic acids.

## Fold Complexes

```python
types = model.input_types

complex_input = types.StructurePredictionInput(
    sequences=[
        types.ProteinInput(id="A", sequence="MKTLLILAVVAAALA"),
        types.DNAInput(id="B", sequence="GATAGC"),
        types.LigandInput(id="L", ccd=["SAH"]),
    ]
)

result = model.fold(
    complex_input,
    num_loops=3,
    num_sampling_steps=50,
    num_diffusion_samples=1,
    seed=0,
)

model.save_as_cif(result, "complex_prediction.cif")
```

## Use MSAs

```python
types = model.input_types

msa = types.MSA.from_a3m("query.a3m", max_sequences=128)
input_with_msa = types.StructurePredictionInput(
    sequences=[
        types.ProteinInput(id="A", sequence=msa.query, msa=msa),
    ]
)

result = model.fold(input_with_msa, num_sampling_steps=50, seed=0)
```

## Raw Tensor Inference

```python
features, chain_infos = model.prepare_structure_input(complex_input, seed=0)

with torch.inference_mode():
    output = model(
        **features,
        num_loops=3,
        num_sampling_steps=50,
        num_diffusion_samples=1,
    )

decoded = model.input_builder.decode(output, features, chain_infos)
```

Set `load_esmc=False` when loading if you want to provide precomputed `lm_hidden_states` manually or run folding-trunk tests without loading the 6B ESMC backbone:

```python
model = AutoModel.from_pretrained(
    "Synthyra/ESMFold2-Fast",
    trust_remote_code=True,
    load_esmc=False,
).cuda().eval()
```
