---
library_name: transformers
license: "mit"
tags:
  - protein-language-model
  - fastplms
---

<!-- Generated from src/fastplms/models.toml. Do not edit. -->

# Synthyra/ESMFold2-Experimental-Fast-Cutoff2025

This checkpoint packages the FastPLMs `ESMFold2` implementation.

Accepted inputs are raw amino-acid sequences or typed molecular-complex
specifications; low-level forward accepts prepared feature tensors.
Supported Transformers entry points are `AutoConfig`, `AutoModel`.

## Quick start

```python
from transformers import AutoModel

model_id = "Synthyra/ESMFold2-Experimental-Fast-Cutoff2025"
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
).eval()
```

This example uses the published Hub repository. For offline validation, build
the manifest-pinned artifact and replace `model_id` with its local
`dist/hub/ESMFold2-Experimental-Fast-Cutoff2025` path, then pass `local_files_only=True`.

Leave attention unspecified for the Transformers default. Supported explicit
choices are `eager`, `sdpa`, `flex_attention`.
Pass the selected name through `attn_implementation`.
For BF16 execution, this family uses FP32 parameters with CUDA BF16 autocast.

## Protein folding

The single-protein helper returns typed structure and confidence outputs:

```python
result = model.fold_protein(
    "MSTNPKPQRKTKRNT",
    num_loops=1,
    num_sampling_steps=200,
    num_diffusion_samples=1,
    seed=7,
)
pdb_text = model.result_to_pdb(result)
cif_text = model.result_to_cif(result)
print(result.ptm, result.plddt.mean().item())
```

No target structure is required. For complexes, construct the input from the
types exposed by the loaded artifact:

```python
types = model.input_types
complex_input = types.StructurePredictionInput(
    sequences=[
        types.ProteinInput(id="A", sequence="MSTNPKPQRKTKRNT"),
        types.ProteinInput(id="B", sequence="MKTIIALSYIFCLVFA"),
        types.DNAInput(id="C", sequence="ATGC"),
        types.LigandInput(id="L", smiles="O"),
    ]
)
complex_result = model.fold(
    complex_input,
    num_loops=1,
    num_sampling_steps=200,
    seed=7,
)
print(complex_result.ptm, complex_result.plddt.mean().item())
```

The typed interface also supports RNA, protein MSAs, modifications, covalent
bonds, pockets, and distogram conditioning. Prepared `ref_pos` values are
component reference geometries created during featurization, not target
coordinates. Predicted coordinates and confidence scores are outputs and do
not establish biochemical activity.

## Learned representation and ESMC precision

ESMFold2 combines the ordered 81 ESMC-6B states `H: (b, l, 81, 2560)` with the
checkpoint's learned projection. Retrieve the resulting residue representation
through the public embedding API:

```python
representations = model.embed_dataset(
    ["MSTNPKPQRKTKRNT", "MKTIIALSYIFCLVFA"],
    batch_size=2,
    full_embeddings=True,
)
print(representations[0].tensor.shape)  # (sequence_length, 256)
```

`model.embed_dataset(..., full_embeddings=True)` returns one `(l, 256)` residue
tensor per single-chain input. It rejects complexes, ligands, MSAs,
chain-separated inputs, `cls`, and `parti` in the embedding path.

Set `esmc_precision` to `auto`, `bf16`, `fp32`, or `fp8` when loading.
`auto` always resolves to BF16. Explicit FP8 is experimental, inference-only,
and strict:

```python
model.reload_esmc(precision="fp8", device="cuda:0")
print(model.esmc_precision_status)
```

FP8 raises when the validated CUDA and Transformer Engine path is unavailable.
Canonical BF16 weights are retained, and transient quantization state is never
serialized.

## Binder-design research example

The FastPLMs binder-design workflow uses the experimental Fast Cutoff2025
checkpoint for differentiable inversion, both experimental Cutoff2025
checkpoints as critics, and ESM++ as the sequence prior:

![FastPLMs EGFR minibinder design](https://raw.githubusercontent.com/Synthyra/FastPLMs/main/docs/assets/egfr_fastplms_binder_design.png)

```bash
python examples/binder_design_fastplms.py \
  --target-name pd-l1 \
  --binder-name minibinder \
  --batch-size 4 \
  --steps 150 \
  --output-dir artifacts/binder-design
```

The workflow ranks candidates by mean iPTM across the approved critics after
the minibinder isoelectric-point filter. These are model-based prioritization
signals, not experimental evidence of affinity or specificity. See the
[complete workflow](https://github.com/Synthyra/FastPLMs/blob/main/docs/binder_design.md).

## Runtime contract

- Public input: Raw amino-acid sequences or typed molecular-complex specifications; low-level forward accepts prepared feature tensors
- Advertised AutoClasses: `AutoConfig`, `AutoModel`
- Attention implementations: `eager`, `sdpa`, `flex_attention`
- Precision policies: `auto`, `fp32`, `bf16`, `fp8` (experimental)
- BF16 execution: `fp32_parameters_autocast`
- Generation contract: `not_applicable`
- Optional dependency group: `structure`

## Provenance

- FastPLMs checkpoint: `Synthyra/ESMFold2-Experimental-Fast-Cutoff2025@8f022c2514a6c32692aaca078a8391d6bc6c4bac`
- Official checkpoint: `biohub/ESMFold2-Experimental-Fast-Cutoff2025@74b88548bf19688b8727432db0d698cb2e1d8783`
- Artifact source: `fast`
- State transform: `identity`
- BF16 execution: `fp32_parameters_autocast`
- Pinned upstreams: `biohub-esm`, `biohub-transformers`, `protein-ttt`
- Reference container: `reference-esmfold2`
- Release tiers: `check`, `compliance`, `structure`, `feature`, `artifact`, `benchmark`
- Unresolved required file identities: `0`

The local artifact records exact file identities, conversion provenance, source
revisions, and legal texts in `provenance.json`. A nonzero unresolved count is a
release blocker.

## Validation boundary

For tiers declared by the manifest, the release contract compares applicable
semantic configuration, tokenizer behavior, state keys, shapes, dtypes,
values, aliases, and representative inference with the pinned official
implementation. This metadata does not by itself claim that a particular build
passed, that one backend is faster, or that an output has biological or
therapeutic validity.

## License

Checkpoint terms: MIT. The Hub model-card identifier is
`mit`. Applicable source licenses, notices, attribution,
and conversion records are distributed with the local artifact. Review them
before use.
