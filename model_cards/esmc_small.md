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

## Quick start

```python
from transformers import AutoModel

model_id = "Synthyra/ESMplusplus_small"
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
).eval()
```

This example uses the published Hub repository. For offline validation, build
the manifest-pinned artifact and replace `model_id` with its local
`dist/hub/<model>` path, then pass `local_files_only=True`.

Leave attention unspecified for the Transformers default or request one of
`eager`, `sdpa`, `flex_attention`, `flash_attention_2`, `flash_attention_3` with `attn_implementation`.
The BF16 execution policy is `static_parameters`:
parameters loaded directly in BF16.

## Tokenization and forward inference

Load the tokenizer from the same artifact as the model. Padding is represented
explicitly by the attention mask:

```python
import torch
from transformers import AutoTokenizer

model_id = "Synthyra/ESMplusplus_small"
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
)
batch = tokenizer(
    ["MSTNPKPQRKTKRNT", "MKTIIALSYIFCLVFA"],
    padding=True,
    return_tensors="pt",
)

with torch.inference_mode():
    output = model(**batch)

print(output.last_hidden_state.shape)
```

## Dataset embeddings

The shared embedding API accepts sequences, `(id, sequence)` pairs,
`EmbeddingInput` records, or a FASTA path. Results preserve order and duplicate
identifiers:

```python
result = model.embed_dataset(
    ["MSTNPKPQRKTKRNT", "MKTIIALSYIFCLVFA"],
    batch_size=2,
    pooling=("mean", "std"),
)

for record in result:
    print(record.id, record.sequence, record.tensor.shape)
```

Set `full_embeddings=True` for one residue tensor with shape `(l, d)` per
sequence. Set `output` to a directory for transactional safetensors or choose
`format="sqlite"` for batch-level commits and exact resume. Pooling excludes
boundary, padding, and other non-biological positions.

For a long FASTA run, stream completed batches into SQLite:

```python
persisted = model.embed_dataset(
    "proteins.fasta",
    batch_size=64,
    pooling=("mean",),
    output="protein-embeddings.sqlite",
    format="sqlite",
    resume=True,
)
```

Resume verifies the input order, model state, tokenizer policy, backend, dtype,
and pooling configuration. It never appends incompatible records to an
existing run.

## ESMC behavior

This artifact exposes the Biohub ESMC sequence encoder and masked-language-model
head through Transformers. It is also the language-model family used by
ESMFold2. Request SDPA when exact pinned Biohub inference parity is required;
the provenance section records backend-specific validation boundaries for this
checkpoint.

## Notes and limitations

Release contract: SDPA must match the pinned Biohub implementation bit-for-bit across every hidden state, last hidden state, logits, special token, and padding position. Eager and FlashAttention 2 are release-gated in BF16 against the pinned boundary-length and biological panels with a relative-L2 engineering target of 0.029, hard limit of 0.03, relative-Q99.9 target of 0.049, first-percentile residue-cosine target of 0.997, and Jensen-Shannon target of 0.0004. The global pooled-cosine and top-1 thresholds remain unchanged. Flex Attention and FlashAttention 3 remain selectable as opt-in alternatives, but they are not strict-parity choices: on the locked H100 BF16 generated-boundary panel, ESMC-6B Flex Attention exceeds the 0.03 relative-L2 hard limit and FlashAttention 3 falls below the 0.995 residue-cosine hard limit. The deviation is consistent with backend-specific BF16 kernel arithmetic; it is not a weight-conversion difference or silent fallback. Use SDPA for exact Biohub parity or FlashAttention 2 for release-gated acceleration.

## Runtime contract

- Input mode: `tokenizer`
- Advertised AutoClasses: `AutoConfig`, `AutoModel`, `AutoModelForMaskedLM`
- Attention implementations: `eager`, `sdpa`, `flex_attention`, `flash_attention_2`, `flash_attention_3`
- Precision policies: `default`
- BF16 execution: `static_parameters`
- Generation contract: `not_applicable`
- Optional dependency group: `core`

## Provenance

- FastPLMs checkpoint: `Synthyra/ESMplusplus_small@46c5f7d562e47d4c14165b424c71ab7db008e6fb`
- Official checkpoint: `biohub/ESMC-300M@a59b831785f907e96e6a246b1d142bfb76df31ee`
- Artifact source: `fast`
- State transform: `esmc_to_fastplms_v1`
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
