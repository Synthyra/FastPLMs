---
library_name: transformers
license: "mit"
tags:
  - protein-language-model
  - fastplms
---

<!-- Generated from src/fastplms/models.toml. Do not edit. -->

# Synthyra/ESMFold2-Fast

This checkpoint uses the FastPLMs `ESMFold2` implementation.
Its input mode is `structure` and its advertised AutoClasses
are `AutoConfig`, `AutoModel`.

## Quick start

```python
from transformers import AutoModel

model_id = "Synthyra/ESMFold2-Fast"
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
).eval()
```

This example uses the published Hub repository. For offline validation, build
the manifest-pinned artifact and replace `model_id` with its local
`dist/hub/<model>` path, then pass `local_files_only=True`.

Leave attention unspecified for the Transformers default or request one of
`eager`, `sdpa`, `flex_attention` with `attn_implementation`.
The BF16 execution policy is `fp32_parameters_autocast`:
FP32 parameters with CUDA BF16 autocast.

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

For complexes, construct the model's `StructurePredictionInput` with explicit
protein, DNA, RNA, ligand, and MSA objects. Confidence scores are model outputs
and do not establish biochemical activity.

## Learned representation and ESMC precision

ESMFold2 combines the ordered 81 ESMC-6B states `H: (b, l, 81, 2560)` with the
checkpoint's learned projection:

```python
Z = model.project_esmc_hidden_states(H)  # Z: (b, l, 256)
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

## Optional folding TTT

The standard and Fast checkpoints expose opt-in folding TTT on their ESMC
backbone:

```python
adapted = model.fold_protein_ttt(
    "MSTNPKPQRKTKRNT",
    num_loops=1,
    num_sampling_steps=50,
    seed=7,
    ttt_config={"steps": 3, "batch_size": 1, "seed": 7},
)
print(adapted.ttt_metrics)
```

Entering a gradient-enabled path reloads canonical BF16 ESMC weights. TTT adds
latency and memory, can worsen a prediction, and does not calibrate confidence
or establish biological validity.

## Runtime contract

- Input mode: `structure`
- Advertised AutoClasses: `AutoConfig`, `AutoModel`
- Attention implementations: `eager`, `sdpa`, `flex_attention`
- Precision policies: `auto`, `fp32`, `bf16`, `fp8` (experimental)
- BF16 execution: `fp32_parameters_autocast`
- Generation contract: `not_applicable`
- Optional dependency group: `structure`

## Provenance

- FastPLMs checkpoint: `Synthyra/ESMFold2-Fast@407875bfcaa42552bfcb25acd67ee1888b790170`
- Official checkpoint: `biohub/ESMFold2-Fast@b28d8ace5e05e61e5bec1e6820cfd3e221819d12`
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
