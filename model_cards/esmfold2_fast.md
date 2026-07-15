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

## Load

```python
from transformers import AutoModel

artifact_path = "dist/hub/ESMFold2-Fast"
model = AutoModel.from_pretrained(
    artifact_path,
    local_files_only=True,
    trust_remote_code=True,
)
```


After publication, replace `artifact_path` with the Hub repository ID and pass
the immutable revision of the published FastPLMs 1.0 artifact. The checkpoint
revision below identifies the source weights; it is not a claim that the
generated artifact already exists at that Hub revision.

Leave attention unspecified for the Transformers default or request one of
`eager`, `sdpa`, `flex_attention` with `attn_implementation`.
The BF16 execution policy is `fp32_parameters_autocast`:
FP32 parameters with CUDA BF16 autocast.

## Learned representation and FP8

ESMFold2 combines the ordered 81 ESMC-6B states `H: (b, l, 81, 2560)`
with the checkpoint's learned projection:

```python
Z = model.project_esmc_hidden_states(H)  # Z: (b, l, 256)
```

`model.embed_dataset(..., full_embeddings=True)` returns one residue tensor
with shape `(l, 256)` per single-chain input. Residue-statistic poolers are
supported; `cls`, `parti`, complexes, ligands, MSAs, and chain-separated
embedding inputs are rejected.

Set `esmc_precision` to `auto`, `bf16`, `fp32`, or `fp8` when loading. The
runtime can be rebuilt explicitly with
`model.reload_esmc(precision=..., device=...)`; `model.esmc_precision_status`
records the requested and resolved precision, reason, device, and Transformer
Engine version. `auto` always resolves to BF16. Explicit `fp8` is an
experimental, inference-only opt-in and raises when the path is unavailable.
Canonical BF16 weights are retained, and transient Transformer Engine
quantization state is never serialized.

## Provenance

- FastPLMs checkpoint: `Synthyra/ESMFold2-Fast@407875bfcaa42552bfcb25acd67ee1888b790170`
- Official checkpoint: `biohub/ESMFold2-Fast@b28d8ace5e05e61e5bec1e6820cfd3e221819d12`
- Artifact source: `fast`
- State transform: `identity`
- Generation contract: `not_applicable`
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
