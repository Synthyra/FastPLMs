---
library_name: transformers
license: "apache-2.0"
tags:
  - protein-language-model
  - fastplms
---

<!-- Generated from src/fastplms/models.toml. Do not edit. -->

# Synthyra/DPLM2-650M

This checkpoint packages the FastPLMs `DPLM2` implementation.

Accepted inputs are tokenized amino-acid and structure tracks with explicit
modality boundaries.
Supported Transformers entry points are `AutoConfig`, `AutoModel`,
`AutoModelForMaskedLM`, `AutoModelForSequenceClassification`,
`AutoModelForTokenClassification`.

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

model_id = "Synthyra/DPLM2-650M"
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
).eval()
```

This example uses the published Hub repository. For offline validation, build
the manifest-pinned artifact and replace `model_id` with its local
`dist/hub/DPLM2-650M` path, then pass `local_files_only=True`.

Leave attention unspecified for the Transformers default. Supported explicit
choices are `sdpa`.
Pass the selected name through `attn_implementation`.
When an optimized backend cannot return full attention tensors,
`output_attentions=True` emits one explicit runtime warning and uses a correctly
masked eager implementation for that call only. The warning identifies the
configured backend, effective backend, and reason. Configuration and later
calls are unchanged.
For BF16 execution, this family uses FP32 parameters with CUDA BF16 autocast.

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

## Amino-acid and structure co-generation

DPLM2 uses separate structure and amino-acid tracks with modality-specific
boundary and mask tokens:

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_id = "Synthyra/DPLM2-650M"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
generator = AutoModelForMaskedLM.from_pretrained(
    model_id,
    trust_remote_code=True,
).cuda().eval()
vocab = tokenizer.get_vocab()
l = 64
structure = [
    vocab["<cls_struct>"],
    *([vocab["<mask_struct>"]] * l),
    vocab["<eos_struct>"],
]
amino_acids = [
    vocab["<cls_aa>"],
    *([vocab["<mask_aa>"]] * l),
    vocab["<eos_aa>"],
]
input_ids = torch.tensor([structure + amino_acids], device="cuda")

with torch.inference_mode():
    generated = generator.generate(input_ids, max_iter=100)["output_tokens"]
print(generated.shape)
```

Generic `cls_token`, `eos_token`, `mask_token`, and `unk_token` aliases are
intentionally unset. Callers constructing multimodal tensors must choose the
amino-acid or structure token explicitly. Raw amino-acid sequences remain
supported by `model.embed_dataset(...)`.

Plain `AutoModel` omits the optional ESM pooler because this co-generation
checkpoint contains no trained pooler weights. Pass `add_pooling_layer=True`
only when intentionally initializing and training that head.

The checkpoint weights are Apache-2.0. The maintained ByteDance
[LICENSE](https://github.com/bytedance/dplm/blob/main/LICENSE) and [README](https://github.com/bytedance/dplm/blob/main/README.md#overview) document the license
basis for the pretrained DPLM1 and DPLM2 weights. Complete publication remains
subject to all artifact, legal, parity, and atomic-publication preflights.

## Runtime contract

- Public input: Tokenized amino-acid and structure tracks with explicit modality boundaries
- Advertised AutoClasses: `AutoConfig`, `AutoModel`, `AutoModelForMaskedLM`, `AutoModelForSequenceClassification`, `AutoModelForTokenClassification`
- AutoClass weight status: `AutoConfig` = `FastPLMs extension`, `AutoModel` = `pretrained`, `AutoModelForMaskedLM` = `pretrained`, `AutoModelForSequenceClassification` = `base weights + untrained task head`, `AutoModelForTokenClassification` = `base weights + untrained task head`
- Attention implementations: `sdpa`
- Precision policies: `default`
- BF16 execution: `fp32_parameters_autocast`
- Generation contract: `required`
- Optional dependency group: `core`
- Weight publication allowed: `true`
- Weight license status: `resolved`
- Redistributable: `true`
- Complete weight publication required: `false`

## Release record

- FastPLMs weights: `Synthyra/DPLM2-650M`
- Runtime revision: recorded separately in the built artifact and published commit
- Source-tree and runtime-bundle SHA-256: recorded in `provenance.json`
- Generator/schema version and complete/runtime-only attestations: recorded in `provenance.json`
- Canonical transformed state SHA-256: `cba76b6602d2258de9fffff953b608d93cb8ef4a9e89b0bbd27e160c81e78bb4`
- Conversion equality attestation: recorded in `provenance.json`
- Official checkpoint: `airkingbd/dplm2_650m`
- Artifact source: `official`
- State transform: `dplm2_to_fastplms_v1`
- BF16 execution: `fp32_parameters_autocast`
- Tokenizer class: `fastplms.models.dplm2.tokenization_dplm2.DPLM2Tokenizer`
- Pinned upstreams: `dplm`
- Reference container: `reference-dplm`
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

Checkpoint terms: Apache-2.0. The Hub model-card identifier is
`apache-2.0`. Applicable source licenses, notices, attribution,
and conversion records are distributed with the local artifact. Review them
before use.
