---
library_name: transformers
license: "cc-by-nc-sa-4.0"
tags:
  - protein-language-model
  - fastplms
---

<!-- Generated from src/fastplms/models.toml. Do not edit. -->

# Synthyra/ANKH2_large

This checkpoint packages the FastPLMs `ANKH` implementation.

Accepted inputs are amino-acid sequences tokenized for encoder or sequence-to-
sequence use.
Supported Transformers entry points are `AutoConfig`, `AutoModel`,
`AutoModelForMaskedLM`, `AutoModelForSeq2SeqLM`,
`AutoModelForSequenceClassification`, `AutoModelForTokenClassification`.

## Quick start

```python
from transformers import AutoModel

model_id = "Synthyra/ANKH2_large"
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
).eval()
```

This example uses the published Hub repository. For offline validation, build
the manifest-pinned artifact and replace `model_id` with its local
`dist/hub/ANKH2_large` path, then pass `local_files_only=True`.

Leave attention unspecified for the Transformers default. Supported explicit
choices are `eager`, `sdpa`.
Pass the selected name through `attn_implementation`.
For BF16 execution, this family uses parameters loaded directly in BF16.

## Tokenization and forward inference

Load the tokenizer from the same artifact as the model. Padding is represented
explicitly by the attention mask:

```python
import torch
from transformers import AutoTokenizer

model_id = "Synthyra/ANKH2_large"
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
`EmbeddingInput` records, insertion-ordered `{id: sequence}` mappings, or a
FASTA path. Results preserve order and duplicate identifiers:

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
sequence. Set `output` to a directory for bounded-memory, transactional
safetensors with ordered-prefix resume, or choose `format="sqlite"` for
batch-level database commits and exact resume. Pooling excludes boundary,
padding, and other non-biological positions.

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

## Encoder and sequence-to-sequence use

`AutoModel` loads the optimized ANKH encoder. The official-compatible decoder
and language-model head are available through `AutoModelForSeq2SeqLM`:

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_id = "Synthyra/ANKH2_large"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
seq2seq = AutoModelForSeq2SeqLM.from_pretrained(
    model_id,
    trust_remote_code=True,
).eval()
batch = tokenizer("MSTNPKPQRKTKRNT", return_tensors="pt")

with torch.inference_mode():
    generated_ids = seq2seq.generate(**batch, max_new_tokens=16)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
```

ANKH artifacts retain CC BY-NC-SA 4.0 terms. The notes below distinguish the
official heads from FastPLMs extensions.

## Notes and limitations

ANKH parity covers the official encoder and sequence-to-sequence heads.
AutoModelForMaskedLM exposes the separately named FastPLMs synthesized
masked-LM extension and is not an official ANKH head.

## Runtime contract

- Public input: Amino-acid sequences tokenized for encoder or sequence-to-sequence use
- Advertised AutoClasses: `AutoConfig`, `AutoModel`, `AutoModelForMaskedLM`, `AutoModelForSeq2SeqLM`, `AutoModelForSequenceClassification`, `AutoModelForTokenClassification`
- Attention implementations: `eager`, `sdpa`
- Precision policies: `default`
- BF16 execution: `static_parameters`
- Generation contract: `not_applicable`
- Optional dependency group: `core`

## Provenance

- FastPLMs checkpoint: `Synthyra/ANKH2_large@392de5ed52bbfd73b45f545e378aaebcff096d0e`
- Official checkpoint: `ElnaggarLab/ankh2-ext2@aa9b9fa72288c47d9f618ce80c011e24b54e17a8`
- Artifact source: `official`
- State transform: `ankh_t5_to_fastplms_v1`
- BF16 execution: `static_parameters`
- Pinned upstreams: `ankh`
- Reference container: `reference-ankh`
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

Checkpoint terms: CC-BY-NC-SA-4.0. The Hub model-card identifier is
`cc-by-nc-sa-4.0`. Applicable source licenses, notices, attribution,
and conversion records are distributed with the local artifact. Review them
before use.
