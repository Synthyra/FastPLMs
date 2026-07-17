---
library_name: transformers
license: "mit"
tags:
  - protein-language-model
  - fastplms
---

<!-- Generated from src/fastplms/models.toml. Do not edit. -->

# Synthyra/ESM3_small

This checkpoint uses the FastPLMs `ESM3` implementation.
Its input mode is `tokenizer` and its advertised AutoClasses
are `AutoConfig`, `AutoModel`.

## Load

```python
from transformers import AutoModel

artifact_path = "dist/hub/ESM3_small"
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

- FastPLMs checkpoint: `Synthyra/ESM3_small@7ddb5a740f9e5f93933eb6410c0ee8684bc63ec1`
- Official checkpoint: `biohub/esm3-sm-open-v1@47f0545b2b6daf26a93439a3cd610f4f7f3d5478`
- Artifact source: `fast`
- State transform: `esm3_to_fastplms_v1`
- Generation contract: `not_applicable`
- BF16 execution: `fp32_parameters_autocast`
- Pinned upstreams: `biohub-esm`, `biohub-transformers`
- Reference container: `reference-biohub-esm`
- Release tiers: `check`, `compliance`, `feature`, `artifact`, `benchmark`
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
