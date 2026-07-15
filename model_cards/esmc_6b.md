---
library_name: transformers
license: "mit"
tags:
  - protein-language-model
  - fastplms
---

<!-- Generated from src/fastplms/models.toml. Do not edit. -->

# Synthyra/ESMplusplus_6B

This checkpoint uses the FastPLMs `ESMC` implementation.
Its input mode is `tokenizer` and its advertised AutoClasses
are `AutoConfig`, `AutoModel`, `AutoModelForMaskedLM`.

## Load

```python
from transformers import AutoModel

artifact_path = "dist/hub/ESMplusplus_6B"
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
`eager`, `sdpa`, `flex_attention`, `flash_attention_2`, `flash_attention_3` with `attn_implementation`.
The BF16 execution policy is `static_parameters`:
parameters loaded directly in BF16.

## Notes and limitations

Reproducibility note: in the locked H100 environment, SDPA matches the pinned Biohub implementation exactly across complete hidden states, including special and padding positions. Alternative BF16 kernels use different reduction and tiling arithmetic. On one seeded batch of randomly generated compliance sequences, alternate-backend deep-state relative L2 errors were approximately 0.01 to 0.012. These values are not MSE, exclude padding from the biological-residue metric, and remain below the 0.03 hard limit. They describe that representative test batch rather than an expected error distribution for biological sequences. A strict test configured for the 0.01 engineering target may therefore report a small alternate-kernel miss; this does not indicate an architecture or checkpoint difference.

## Provenance

- FastPLMs checkpoint: `Synthyra/ESMplusplus_6B@0d579cce3b0f09efa6b3baddf6cc3fd8c9b616c8`
- Official checkpoint: `biohub/ESMC-6B@45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a`
- Artifact source: `fast`
- State transform: `esmc_to_fastplms_v1`
- Generation contract: `not_applicable`
- BF16 execution: `static_parameters`
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
