---
library_name: transformers
license: "cc-by-nc-sa-4.0"
tags:
  - protein-language-model
  - fastplms
---

<!-- Generated from src/fastplms/models.toml. Do not edit. -->

# Synthyra/ANKH3_large

This checkpoint packages the FastPLMs `ANKH` implementation.

Accepted inputs are amino-acid sequences tokenized for encoder or sequence-to-
sequence use.
Supported Transformers entry points are `AutoConfig`, `AutoModel`,
`AutoModelForMaskedLM`, `AutoModelForSeq2SeqLM`,
`AutoModelForSequenceClassification`, `AutoModelForTokenClassification`.

## Install and platform requirements

Install the current FastPLMs package:

```bash
python -m pip install \
  "fastplms @ git+https://github.com/Synthyra/FastPLMs.git"
```

Python 3.11-3.14, PyTorch 2.13, and Transformers 5.13 are required. The declared CPU gate covers tiny offline contracts; published checkpoint throughput and parity require the documented device tier. The Hub quick start below requires network
access on first download. For an air-gapped run, first build the manifest-pinned
local artifact and use the offline form shown in the example.

## Quick start

```python
from transformers import AutoModel

model_id = "Synthyra/ANKH3_large"
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
).eval()
```

This example uses the published Hub repository. For offline validation, build
the manifest-pinned artifact and replace `model_id` with its local
`dist/hub/ANKH3_large` path, then pass `local_files_only=True`.

Leave attention unspecified for the Transformers default. Supported explicit
choices are `eager`, `sdpa`.
Pass the selected name through `attn_implementation`.
When an optimized backend cannot return full attention tensors,
`output_attentions=True` emits one explicit runtime warning and uses a correctly
masked eager implementation for that call only. The warning identifies the
configured backend, effective backend, and reason. Configuration and later
calls are unchanged.
For BF16 execution, this family uses parameters loaded directly in BF16.

## Tokenization and forward inference

`Synthyra/ANKH3_large` contains the complete encoder-decoder checkpoint.
`AutoModel` loads the encoder view without allocating the decoder, while
`AutoModelForSeq2SeqLM` loads the encoder, decoder, cross-attention, and
language-model head.

Use the tokenizer owned by the loaded model so tokenizer files, revision,
offline/cache policy, and ANKH's residue-aware pre-tokenizer stay aligned.
Pass raw protein strings without inserted residue spaces:

```python
import torch

tokenizer = model.tokenizer
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

Dataset embeddings default to the encoder's final hidden state. Layer indices
use the selected stack's native hidden-state order:

```python
encoder_result = model.embed_dataset(
    ["MSTNPKPQRKTKRNT"],
    hidden_state_source="encoder",
    hidden_state_index=-1,
    full_embeddings=True,
)
all_encoder_layers = model.embed_dataset(
    ["MSTNPKPQRKTKRNT"],
    hidden_state_source="encoder",
    store_all_hidden_states=True,
    full_embeddings=True,
)
```

Decoder representations require `AutoModelForSeq2SeqLM` and exactly one
explicit, aligned `decoder_inputs` sequence or `decoder_input_ids` tensor. ANKH
does not infer a shifted source sequence because official tasks use prompts,
sentinel tokens, or generated tokens that depend on the task. Protein inputs
remain raw residue strings and sentinel prompts remain tight, as in
`M<extra_id_0>`:

```python
from transformers import AutoModelForSeq2SeqLM

seq2seq = AutoModelForSeq2SeqLM.from_pretrained(
    "Synthyra/ANKH3_large",
    trust_remote_code=True,
).eval()
decoder_result = seq2seq.embed_dataset(
    ["MSTNPKPQRKTKRNT"],
    hidden_state_source="decoder",
    hidden_state_index=-1,
    decoder_inputs=["M<extra_id_0>"],
    full_embeddings=True,
)
```

`decoder_attention_mask` is accepted only with `decoder_input_ids`. Decoder
pooling excludes start, EOS, padding, sentinel, and other tokenizer-special
positions. Persisted results record the selected stack and layer, decoder input
and mask fingerprints, input-position alignment, and biological-mask policy.

## Encoder and sequence-to-sequence use

`Synthyra/ANKH3_large` contains the complete ANKH encoder-decoder checkpoint.
Use `AutoModel` for encoder embeddings and `AutoModelForSeq2SeqLM` for
task-specific decoding:

```python
import torch
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

repo_id = "Synthyra/ANKH3_large"
tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
encoder = AutoModel.from_pretrained(repo_id, trust_remote_code=True).eval()
seq2seq = AutoModelForSeq2SeqLM.from_pretrained(
    repo_id,
    trust_remote_code=True,
).eval()
batch = tokenizer("MSTNPKPQRKTKRNT", return_tensors="pt")

with torch.inference_mode():
    encoder_hidden = encoder(**batch).last_hidden_state
    generated_ids = seq2seq.generate(**batch, max_new_tokens=16)
print(encoder_hidden.shape)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
```

ANKH artifacts retain CC BY-NC-SA 4.0 terms. The notes below distinguish the
official heads from FastPLMs extensions. The complete checkpoint is larger than
the former encoder-only mirror while preserving encoder-output parity.

## Notes and limitations

ANKH parity covers the official encoder and sequence-to-sequence heads.
AutoModelForMaskedLM exposes the separately named FastPLMs synthesized
masked-LM extension and is not an official ANKH head.

## Runtime contract

- Public input: Amino-acid sequences tokenized for encoder or sequence-to-sequence use
- Advertised AutoClasses: `AutoConfig`, `AutoModel`, `AutoModelForMaskedLM`, `AutoModelForSeq2SeqLM`, `AutoModelForSequenceClassification`, `AutoModelForTokenClassification`
- AutoClass weight status: `AutoConfig` = `FastPLMs extension`, `AutoModel` = `pretrained`, `AutoModelForMaskedLM` = `FastPLMs extension`, `AutoModelForSeq2SeqLM` = `pretrained`, `AutoModelForSequenceClassification` = `base weights + untrained task head`, `AutoModelForTokenClassification` = `base weights + untrained task head`
- Attention implementations: `eager`, `sdpa`
- Precision policies: `default`
- BF16 execution: `static_parameters`
- Generation contract: `required`
- Optional dependency group: `core`
- Weight publication allowed: `true`
- Weight license status: `resolved`
- Redistributable: `true`
- Complete weight publication required: `false`

## Release record

- FastPLMs weights: `Synthyra/ANKH3_large`
- Runtime revision: recorded separately in the built artifact and published commit
- Source-tree and runtime-bundle SHA-256: recorded in `provenance.json`
- Generator/schema version and complete/runtime-only attestations: recorded in `provenance.json`
- Canonical transformed state SHA-256: `60acb7ef86e85dc0c51fc1edf4c8e69a0480049723b6b2c95e6e9faa720c112a`
- Conversion equality attestation: recorded in `provenance.json`
- Official checkpoint: `ElnaggarLab/ankh3-large`
- Artifact source: `official`
- State transform: `ankh_t5_to_fastplms_v1`
- BF16 execution: `static_parameters`
- Pinned upstreams: `ankh`
- Reference container: `reference-ankh`
- Release tiers: `check`, `compliance`, `feature`, `artifact`, `benchmark`
- Unresolved required file identities: `0`

The local artifact records exact file identities, conversion details, source
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
