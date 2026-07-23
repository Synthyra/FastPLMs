---
library_name: transformers
license: "cc-by-nc-sa-4.0"
tags:
  - protein-language-model
  - fastplms
---

<!-- Generated from src/fastplms/models.toml. Do not edit. -->

# Synthyra/ANKH_base

This checkpoint packages the FastPLMs `ANKH` implementation.

Accepted inputs are amino-acid sequences tokenized for encoder or sequence-to-
sequence use.
Supported Transformers entry points are `AutoConfig`, `AutoModel`,
`AutoModelForMaskedLM`, `AutoModelForSeq2SeqLM`,
`AutoModelForSequenceClassification`, `AutoModelForTokenClassification`.

## Install and platform requirements

Install FastPLMs from the exact source revision paired with this model card:

```bash
python -m pip install \
  "fastplms @ git+https://github.com/Synthyra/FastPLMs.git@<runtime-revision>"
```

Python 3.11-3.14, PyTorch 2.13, and Transformers 5.13 are required. The declared CPU gate covers tiny offline contracts; published checkpoint throughput and parity require the documented device tier. The Hub quick start below requires network
access on first download. For an air-gapped run, first build the manifest-pinned
local artifact and use the offline form shown in the example.

## Quick start

```python
from transformers import AutoModel

model_id = "Synthyra/ANKH_base"
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
).eval()
```

This example uses the published Hub repository. For offline validation, build
the manifest-pinned artifact and replace `model_id` with its local
`dist/hub/ANKH_base` path, then pass `local_files_only=True`.

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

The live `Synthyra/ANKH_base` revision `7ec329aae8e3e174bf22a1eb9e0e9fcc12b53092` is legacy
encoder-only. The Hub quick start above is therefore an encoder-only
`AutoModel` path, not evidence that decoder or language-model-head weights are
already published.

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

The current live Hub revision is legacy encoder-only. It supports encoder
dataset embeddings, which default to the encoder's final hidden state. Layer
indices use the selected stack's native hidden-state order:

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
`M<extra_id_0>`. Until the atomic Hub replacement is published, load the
validated complete local artifact and fail closed on its registry-bound
attestation:

```python
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM
from fastplms.registry import get_model_registry
from tools.artifacts.build import validate_artifact

artifact = Path("dist/hub/ANKH_base").resolve()
registry = get_model_registry()
validate_artifact(artifact, spec=registry["ankh_base"], registry=registry)
seq2seq = AutoModelForSeq2SeqLM.from_pretrained(
    artifact,
    trust_remote_code=True,
    local_files_only=True,
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

The current manifest Hub revision `7ec329aae8e3e174bf22a1eb9e0e9fcc12b53092` for
`Synthyra/ANKH_base` is legacy encoder-only. It supports the `AutoModel`
encoder path, but it is not the full FastPLMs 1.0 sequence-to-sequence
artifact. Do not load `AutoModelForSeq2SeqLM` from that live revision.

The full encoder-decoder replacement is still pending atomic publication. Use
sequence-to-sequence behavior only from a locally built artifact whose complete
weight, runtime, provenance, and registry validation has passed. The following
snippet fails closed if that artifact is missing or invalid:

```python
import torch
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM
from fastplms.registry import get_model_registry
from tools.artifacts.build import validate_artifact

artifact = Path("dist/hub/ANKH_base").resolve()
registry = get_model_registry()
validate_artifact(artifact, spec=registry["ankh_base"], registry=registry)
seq2seq = AutoModelForSeq2SeqLM.from_pretrained(
    artifact,
    trust_remote_code=True,
    local_files_only=True,
).eval()
tokenizer = seq2seq.tokenizer
batch = tokenizer("MSTNPKPQRKTKRNT", return_tensors="pt")

with torch.inference_mode():
    generated_ids = seq2seq.generate(**batch, max_new_tokens=16)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
```

ANKH artifacts retain CC BY-NC-SA 4.0 terms. The notes below distinguish the
official heads from FastPLMs extensions. Once validated and published, the 1.0
replacement will increase the default repository size while preserving
encoder-output parity. Runtime code,
configuration, tokenizer, card, provenance, and every weight shard must be
published atomically. Files-only publication is forbidden for this migration.

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
- Complete weight publication required: `true`

## Provenance

- FastPLMs weights: `Synthyra/ANKH_base@7ec329aae8e3e174bf22a1eb9e0e9fcc12b53092`
- Runtime revision: recorded separately in the built artifact and published commit
- Source-tree and runtime-bundle SHA-256: recorded in `provenance.json`
- Generator/schema version and complete/runtime-only attestations: recorded in `provenance.json`
- Canonical transformed state SHA-256: `cdd8d30d88e5bf41f44e1eef4470d8e46607aba5f7c7c805b06c035b89c8c16f`
- Conversion equality attestation: recorded in `provenance.json`
- Official checkpoint: `ElnaggarLab/ankh-base@d99cb6b966530dfc2ae96bc69d9255c2a07308b0`
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
