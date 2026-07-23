---
library_name: transformers
license: "mit"
tags:
  - protein-language-model
  - fastplms
---

<!-- Generated from src/fastplms/models.toml. Do not edit. -->

# Synthyra/ESMFold2-Experimental-Cutoff2025

This checkpoint packages the FastPLMs `ESMFold2` implementation.

Accepted inputs are raw amino-acid sequences or typed molecular-complex
specifications; low-level forward accepts prepared feature tensors.
Supported Transformers entry points are `AutoConfig`, `AutoModel`.

## Install and platform requirements

Install FastPLMs from the exact source revision paired with this model card:

```bash
python -m pip install \
  "fastplms[structure] @ git+https://github.com/Synthyra/FastPLMs.git@<runtime-revision>"
```

Python 3.11-3.14, PyTorch 2.13, and Transformers 5.13 are required. Structure inference requires the `structure` extra and a CUDA device for the published execution contract. The current validated release target is the exact NVIDIA GH200 on Linux aarch64; Linux x86-64, CPU-only, Windows, and macOS structure runs are not current release evidence. The Hub quick start below requires network
access on first download. For an air-gapped run, first build the manifest-pinned
local artifact and use the offline form shown in the example.

## Quick start

```python
from transformers import AutoModel

model_id = "Synthyra/ESMFold2-Experimental-Cutoff2025"
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
).eval()
```

This example uses the published Hub repository. For offline validation, build
the manifest-pinned artifact and replace `model_id` with its local
`dist/hub/ESMFold2-Experimental-Cutoff2025` path, then pass `local_files_only=True`.

Leave attention unspecified for the Transformers default. Supported explicit
choices are `eager`, `sdpa`, `flex_attention`.
Pass the selected name through `attn_implementation`.
When an optimized backend cannot return full attention tensors,
`output_attentions=True` emits one explicit runtime warning and uses a correctly
masked eager implementation for that call only. The warning identifies the
configured backend, effective backend, and reason. Configuration and later
calls are unchanged.
For BF16 execution, this family uses FP32 parameters with CUDA BF16 autocast.

## Alignment-conditioning contract

This is a full 48-block ESMFold2 checkpoint. It supports both
single-sequence inference and optional MSA-conditioned inference. Typed
multichain and multimolecule inputs may attach an MSA to each applicable
protein chain.


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
bonds, and distogram conditioning. The public schema recognizes
`PocketConditioning`, but the pinned official runtime discards it and hard-codes
a zero pocket feature. FastPLMs therefore rejects non-null pocket conditioning
instead of silently ignoring it. Prepared `ref_pos` values are component
reference geometries created during featurization, not target coordinates.
Predicted coordinates and confidence scores are outputs and do not establish
biochemical activity.

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

The ESMC backbone uses SDPA as the recommended highest-fidelity path. Flex
Attention is supported and non-experimental but can be numerically divergent;
ESMFold2 does not advertise FlashAttention for the folding interface.

| Backend | Support | Measurement status |
| --- | --- | --- |
| `sdpa` | Recommended fidelity path | Pending complete validated 30-record frozen-head GH200/aarch64 set |
| `eager` | Supported | Pending complete validated 30-record frozen-head GH200/aarch64 set |
| `flex_attention` | Supported, numerically divergent | Pending complete validated 30-record frozen-head GH200/aarch64 set |

No threshold, report from another checkpoint, or result from another
accelerator is substituted for a measurement. A release set contains all
30 model/backend/panel records from one exact GH200 device and aarch64 runtime:
18 eager/SDPA/Flex measurements include relative L2, Q99.9, residue
cosine, pooled cosine, top-1, and Jensen-Shannon distributions; 12
FlashAttention 2/3 records explicitly attest locked-platform unavailability.

Metrics must be tied to the exact ESMFold2 and ESMC revisions, dtype, current
GH200/aarch64 device and container images, dependency lock, source attestations,
and sequence panel. Pending cells are not performance or parity claims.

## Hash-pinned CCD runtime asset

Structure preparation requires `ccd.pkl` from
`biohub/ESMFold2@1ebf0e3481a5184eb6171d40615c79e384b48796`. The manifest pins
its 417,306,584-byte size and SHA-256
`9ff44b1927c6b9198e38ffe0928706827a09a350c15530beeeabebfa88038fc5`
under MIT terms. This is a trusted-deserialization boundary: FastPLMs only
allows the exact manifest repository/revision snapshot link to resolve within
that repository's contained blob directory; user-supplied asset and `cache_dir`
symlinks are rejected. The loader creates a private temporary snapshot, verifies
its size and SHA-256, and unpickles only that loader-owned snapshot, closing
path-replacement and in-place source-write races. Offline execution requires the
exact cache object and never downloads a replacement.

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
- AutoClass weight status: `AutoConfig` = `FastPLMs extension`, `AutoModel` = `pretrained`
- Attention implementations: `eager`, `sdpa`, `flex_attention`
- Precision policies: `auto`, `fp32`, `bf16`, `fp8` (experimental)
- BF16 execution: `fp32_parameters_autocast`
- Generation contract: `not_applicable`
- Optional dependency group: `structure`
- Weight publication allowed: `true`
- Weight license status: `resolved`
- Redistributable: `true`
- Complete weight publication required: `false`

## Provenance

- FastPLMs weights: `Synthyra/ESMFold2-Experimental-Cutoff2025@632ff4a9e68f1de78ee956a613267bdcdb5b354d`
- Runtime revision: recorded separately in the built artifact and published commit
- Source-tree and runtime-bundle SHA-256: recorded in `provenance.json`
- Generator/schema version and complete/runtime-only attestations: recorded in `provenance.json`
- Official checkpoint: `biohub/ESMFold2-Experimental-Cutoff2025@56f94f5c1069ecde17512c96928850518340d287`
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
