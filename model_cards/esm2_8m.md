---
library_name: transformers
license: "mit"
tags:
  - protein-language-model
  - fastplms
---

<!-- Generated from src/fastplms/models.toml. Do not edit. -->

# Synthyra/ESM2-8M

This checkpoint packages the FastPLMs `ESM2` implementation.

Accepted inputs are amino-acid sequences tokenized to residue IDs.
Supported Transformers entry points are `AutoConfig`, `AutoModel`,
`AutoModelForMaskedLM`, `AutoModelForSequenceClassification`,
`AutoModelForTokenClassification`.

## Quick start

```python
from transformers import AutoModel

model_id = "Synthyra/ESM2-8M"
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
).eval()
```

This example uses the published Hub repository. For offline validation, build
the manifest-pinned artifact and replace `model_id` with its local
`dist/hub/ESM2-8M` path, then pass `local_files_only=True`.

Leave attention unspecified for the Transformers default. Supported explicit
choices are `eager`, `sdpa`, `flex_attention`, `flash_attention_2`,
`flash_attention_3`.
Pass the selected name through `attn_implementation`.
For BF16 execution, this family uses FP32 parameters with CUDA BF16 autocast.

## Tokenization and forward inference

Load the tokenizer from the same artifact as the model. Padding is represented
explicitly by the attention mask:

```python
import torch
from transformers import AutoTokenizer

model_id = "Synthyra/ESM2-8M"
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

## Masked language modeling and contacts

Use the masked-language-model AutoClass when logits are required:

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_id = "Synthyra/ESM2-8M"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
masked_model = AutoModelForMaskedLM.from_pretrained(
    model_id,
    trust_remote_code=True,
).eval()
batch = tokenizer("MSTNPKPQRKTKRNT", return_tensors="pt")

with torch.inference_mode():
    logits = masked_model(**batch).logits
    contacts = masked_model.predict_contacts(
        batch["input_ids"],
        batch["attention_mask"],
    )

print(logits.shape, contacts.shape)
```

Contact prediction materializes attention maps and should not be enabled in a
high-throughput embedding path unless those maps are required.

Plain `AutoModel` omits the optional ESM pooler because this masked-language-
model checkpoint contains no trained pooler weights. Pass
`add_pooling_layer=True` only when intentionally initializing and training that
head.

## Runtime contract

- Public input: Amino-acid sequences tokenized to residue IDs
- Advertised AutoClasses: `AutoConfig`, `AutoModel`, `AutoModelForMaskedLM`, `AutoModelForSequenceClassification`, `AutoModelForTokenClassification`
- Attention implementations: `eager`, `sdpa`, `flex_attention`, `flash_attention_2`, `flash_attention_3`
- Precision policies: `default`
- BF16 execution: `fp32_parameters_autocast`
- Generation contract: `not_applicable`
- Optional dependency group: `core`

## Provenance

- FastPLMs checkpoint: `Synthyra/ESM2-8M@185ecbd45665d050a8dae326d91886d330c5f9d0`
- Official checkpoint: `facebook/esm2_t6_8M_UR50D@c731040fcd8d73dceaa04b0a8e6329b345b0f5df`
- Artifact source: `fast`
- State transform: `esm2_hf_to_fastplms_v1`
- BF16 execution: `fp32_parameters_autocast`
- Pinned upstreams: `fair-esm`
- Reference container: `reference-esm2`
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
