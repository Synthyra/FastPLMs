---
library_name: transformers
license: "mit"
tags:
  - protein-language-model
  - fastplms
---

<!-- Generated from src/fastplms/models.toml. Do not edit. -->

# Synthyra/ESMplusplus_small

This checkpoint uses the FastPLMs `ESMC` implementation.
Its input mode is `tokenizer` and its advertised AutoClasses
are `AutoConfig`, `AutoModel`, `AutoModelForMaskedLM`.

## Load

```python
from transformers import AutoModel

artifact_path = "dist/hub/ESMplusplus_small"
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
`eager`, `sdpa`, `flash_attention_2` with `attn_implementation`.
The BF16 execution policy is `static_parameters`:
parameters loaded directly in BF16.

## Notes and limitations

Release contract: SDPA must match the pinned Biohub implementation bit-for-bit across every hidden state, last hidden state, logits, special token, and padding position. Eager and FlashAttention 2 are release-gated in BF16 against the pinned boundary-length and biological panels with a relative-L2 engineering target of 0.029, hard limit of 0.03, relative-Q99.9 target of 0.049, first-percentile residue-cosine target of 0.997, and Jensen-Shannon target of 0.0004. The global pooled-cosine and top-1 thresholds remain unchanged. Flex Attention is not advertised because ESMC-6B exceeds the relative-L2 hard limit on the pinned generated boundary panel; FlashAttention 3 is not advertised because ESMC-6B falls below the residue-cosine hard limit on that panel. Any target miss blocks release.

## Provenance

- FastPLMs checkpoint: `Synthyra/ESMplusplus_small@46c5f7d562e47d4c14165b424c71ab7db008e6fb`
- Official checkpoint: `biohub/ESMC-300M@a59b831785f907e96e6a246b1d142bfb76df31ee`
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
