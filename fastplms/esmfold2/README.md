---
library_name: transformers
tags:
  - biology
  - protein-structure
  - esmfold2
  - multimodal-protein-model
---

# FastPLMs ESMFold2

FastPLMs ESMFold2 is a self-contained Hugging Face `AutoModel` wrapper for
Biohub's ESMFold2, ESMFold2-Fast, and experimental ESMFold2 structure
predictors. It vendors the released Biohub ESMFold2 model code, input builder,
MSA helpers, and structure export utilities, while loading the PLM backbone
through FastPLMs ESM++.

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

Use `Synthyra/ESMFold2` for the full model, `Synthyra/ESMFold2-Fast` for the
faster release variant, and the `Synthyra/ESMFold2-Experimental*` checkpoints
for differentiable binder design and experimental critic ensembles.
The folding trunk runs in fp32; the 6B FastPLMs ESM++ backbone is loaded in
bf16 by default via `esmc_precision="bf16"` and uses the flex attention backend
by default inside ESMFold2.

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

## Experimental Test-Time Training

TTT is disabled by default. Standard `fold_protein(...)`, `fold(...)`, raw tensor
inference, and `state_dict()` keys are unchanged unless you explicitly pass
`ttt=True` or call `fold_protein_ttt(...)`.

The ESMFold2 TTT path is experimental and protein-only in v1. It trains local
LoRA adapters only on `_esmc` with a masked language modeling objective. The
folding trunk, confidence head, diffusion head, and structure input pipeline are
frozen. TTT can improve difficult low-confidence folds, but it adds substantial
test-time compute and can degrade already confident predictions.

```python
result = model.fold_protein(
    "MSTNPKPQRKTKRNT",
    num_loops=1,
    num_sampling_steps=10,
    num_diffusion_samples=1,
    seed=0,
    ttt=True,
    ttt_config={
        "steps": 1,
        "ags": 1,
        "batch_size": 1,
        "lora_rank": 8,
        "lora_alpha": 32.0,
    },
)

print(result.ttt_metrics["losses"])
print(result.ttt_metrics["step_plddts"])
print(result.ttt_metrics["best_step"])
```

`load_esmc=True` is required for TTT because the ESM++ MLM head is loaded lazily
from `config.esmc_id`. If that pretrained MLM head cannot be loaded, TTT raises
an assertion instead of silently using a random head.

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

## Binder Design With FastPLMs ESMFold2

FastPLMs includes a FastPLMs-only port of the Biohub ESMFold2 binder design
tutorial at `cookbook/tutorials/binder_design_fastplms.py`. The workflow uses
ESMFold2 experimental checkpoints for differentiable folding losses, ESM++ for
sequence regularization, and ESMFold2 hero critics for final confidence scoring.

![FastPLMs EGFR minibinder design](https://raw.githubusercontent.com/Synthyra/FastPLMs/main/docs/assets/egfr_fastplms_binder_design.png)

The optimizer follows the official strategy:

1. Optimize mutable `#` residues as continuous amino acid logits.
2. Suppress cysteine design by masking cysteine logits and gradients.
3. Backpropagate through ESMFold2 `res_type_soft` using intra-contact,
   inter-contact, and globularity losses from the distogram.
4. Add an ESM++ masked-LM pseudoperplexity regularizer on mutable binder
   residues.
5. Keep the late-trajectory sequence with the best iPTM.
6. Fold the selected sequence with the final critic ensemble and write
   `results.parquet`, `selection.parquet`, `trajectory.jsonl`,
   `best_sequences.fasta`, and per-critic PDB/CIF/logit files.

Run the verified EGFR 128 amino acid de novo minibinder example:

```bash
cd /home/ubuntu/FastPLMs

sudo -n docker run --gpus all --rm \
  -v /home/ubuntu/FastPLMs:/app \
  -v /home/ubuntu/FastPLMs:/workspace \
  -v /home/ubuntu/.cache/huggingface:/workspace/.cache/huggingface \
  -w /workspace fastplms-esmfold2 \
  python /app/cookbook/tutorials/binder_design_fastplms.py \
    --backend local \
    --target-name egfr \
    --binder-sequence '################################################################################################################################' \
    --not-antibody \
    --steps 150 \
    --batch-size 1 \
    --seed 103 \
    --output-dir /workspace/campaign_egfr_len128_b1_s150_seed103_consensus_cli
```

Verified result:

| Metric | Value |
| :--- | :--- |
| Binder length | `128` |
| Seed | `103` |
| Steps | `150` |
| Hero mean iPTM | `0.913870` |
| Hero min iPTM | `0.904600` |
| All four hero critics above 0.9 | `True` |

Binder sequence:

```text
SAVKHLLEIVKYLEEAIEKALEVDPVFLVPPAAEELLIAAKVIKELAKENPELIEVYELLMKAVKGLKKLVRSNDKEILREVIRLLRKAAKVIREILKNNPDLDPELRKALEELAKVLEEIAEVLEQQ
```

See the full guide in [`docs/binder_design.md`](https://github.com/Synthyra/FastPLMs/blob/main/docs/binder_design.md)
for Modal execution, official pI and selection scoring, per-critic metrics, and
the tested cheaper step-count boundary.

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

Set `load_esmc=False` when loading if you want to provide precomputed `lm_hidden_states` manually or run folding-trunk tests without loading the 6B ESM++ backbone:

```python
model = AutoModel.from_pretrained(
    "Synthyra/ESMFold2-Fast",
    trust_remote_code=True,
    load_esmc=False,
).cuda().eval()
```

For FP8 LM inference, install `transformer_engine.pytorch` in a CUDA
environment with FP8-capable hardware and load the shared FastPLMs ESM++
backbone with:

```python
model = AutoModel.from_pretrained(
    "Synthyra/ESMFold2-Fast",
    trust_remote_code=True,
    esmc_precision="fp8",
).cuda().eval()
```

FP8 is inference-only for the ESMFold2 LM backbone. TTT remains a bf16/fp32
path.
