---
library_name: transformers
license: "apache-2.0"
tags:
  - protein-language-model
  - fastplms
---

<!-- Generated from src/fastplms/models.toml. Do not edit. -->

# Synthyra/DPLM2-150M

This checkpoint uses the FastPLMs `DPLM2` implementation.
Its input mode is `tokenizer` and its advertised AutoClasses
are `AutoConfig`, `AutoModel`, `AutoModelForMaskedLM`, `AutoModelForSequenceClassification`, `AutoModelForTokenClassification`.

## Load

```python
from transformers import AutoModel

artifact_path = "dist/hub/DPLM2-150M"
model = AutoModel.from_pretrained(
    artifact_path,
    local_files_only=True,
    trust_remote_code=True,
)
```

The paired custom tokenizer is loaded through the same pinned artifact:

```python
from transformers import AutoTokenizer

artifact_path = "dist/hub/DPLM2-150M"
tokenizer = AutoTokenizer.from_pretrained(
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
`sdpa` with `attn_implementation`.
The BF16 execution policy is `fp32_parameters_autocast`:
FP32 parameters with CUDA BF16 autocast.

## Provenance

- FastPLMs checkpoint: `Synthyra/DPLM2-150M@182745b8dc5661f898481a4fa60a7af9d53385c4`
- Official checkpoint: `airkingbd/dplm2_150m@3451d984d06497f835ed49634bd68c9dfb54d730`
- Artifact source: `official`
- State transform: `dplm2_to_fastplms_v1`
- Generation contract: `required`
- BF16 execution: `fp32_parameters_autocast`
- Tokenizer class: `fastplms.models.dplm2.tokenization_dplm2.DPLM2Tokenizer`
- Pinned upstreams: `dplm`
- Reference container: `reference-dplm`
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

Checkpoint terms: Apache-2.0. The Hub model-card identifier is
`apache-2.0`. Applicable source licenses, notices, attribution,
and conversion records are distributed with the local artifact. Review them
before use.
