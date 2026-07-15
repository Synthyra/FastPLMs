---
library_name: transformers
license: "cc-by-nc-sa-4.0"
tags:
  - protein-language-model
  - fastplms
---

<!-- Generated from src/fastplms/models.toml. Do not edit. -->

# Synthyra/ANKH2_large

This checkpoint uses the FastPLMs `ANKH` implementation.
Its input mode is `tokenizer` and its advertised AutoClasses
are `AutoConfig`, `AutoModel`, `AutoModelForMaskedLM`, `AutoModelForSeq2SeqLM`, `AutoModelForSequenceClassification`, `AutoModelForTokenClassification`.

## Load

```python
from transformers import AutoModel

artifact_path = "dist/hub/ANKH2_large"
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
`eager`, `sdpa` with `attn_implementation`.
The BF16 execution policy is `static_parameters`:
parameters loaded directly in BF16.

## Notes and limitations

ANKH parity covers the official encoder and sequence-to-sequence heads. AutoModelForMaskedLM exposes the separately named FastPLMs synthesized masked-LM extension and is not an official ANKH head.

## Provenance

- FastPLMs checkpoint: `Synthyra/ANKH2_large@392de5ed52bbfd73b45f545e378aaebcff096d0e`
- Official checkpoint: `ElnaggarLab/ankh2-ext2@aa9b9fa72288c47d9f618ce80c011e24b54e17a8`
- Artifact source: `official`
- State transform: `ankh_t5_to_fastplms_v1`
- Generation contract: `not_applicable`
- BF16 execution: `static_parameters`
- Pinned upstreams: `ankh`
- Reference container: `reference-ankh`
- Release tiers: `check`, `compliance`, `feature`, `artifact`, `benchmark`
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

Checkpoint terms: CC-BY-NC-SA-4.0. The Hub model-card identifier is
`cc-by-nc-sa-4.0`. Applicable source licenses, notices, attribution,
and conversion records are distributed with the local artifact. Review them
before use.
