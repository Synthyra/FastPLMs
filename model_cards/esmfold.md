---
library_name: transformers
license: "mit"
tags:
  - protein-language-model
  - fastplms
---

<!-- Generated from src/fastplms/models.toml. Do not edit. -->

# Synthyra/FastESMFold

This checkpoint uses the FastPLMs `ESMFold` implementation.
Its input mode is `structure` and its advertised AutoClasses
are `AutoConfig`, `AutoModel`.

## Load

```python
from transformers import AutoModel

artifact_path = "dist/hub/FastESMFold"
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

## Provenance

- FastPLMs checkpoint: `Synthyra/FastESMFold@b88c8cb50d19b2cf7ab4fee4b0a61f5e02da7823`
- Official checkpoint: `facebook/esmfold_v1@75a3841ee059df2bf4d56688166c8fb459ddd97a`
- Artifact source: `fast`
- State transform: `esmfold_meta_to_fastplms_v1`
- Generation contract: `not_applicable`
- BF16 execution: `fp32_parameters_autocast`
- Pinned upstreams: `fair-esm`, `openfold`
- Reference container: `reference-esmfold`
- Release tiers: `check`, `compliance`, `structure`, `feature`, `artifact`, `benchmark`
- Unresolved required file identities: `0`

The local artifact records exact file identities, conversion provenance, source
revisions, and legal texts in `provenance.json`. A nonzero unresolved count is a
release blocker.

## Validation boundary

The release contract compares semantic configuration, tokenizer behavior, state
keys, shapes, dtypes, values, aliases, and representative inference with the
pinned official implementation. This metadata does not by itself claim that a
particular build passed, that one backend is faster, or that an output has
biological or therapeutic validity.

## License

Checkpoint terms: MIT. The Hub model-card identifier is
`mit`. Applicable source licenses, notices, attribution,
and conversion records are distributed with the local artifact. Review them
before use.
