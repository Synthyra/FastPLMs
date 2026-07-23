---
library_name: transformers
license: "mit"
tags:
  - protein-language-model
  - fastplms
---

<!-- Generated from src/fastplms/models.toml. Do not edit. -->

# Synthyra/ESM3_small

This checkpoint packages the FastPLMs `ESM3` implementation.

Accepted inputs are sequence, structure, and function tracks prepared through
the multimodal helpers.
Supported Transformers entry points are `AutoConfig`, `AutoModel`.

## Install and platform requirements

Install the current FastPLMs package:

```bash
python -m pip install "fastplms @ git+https://github.com/Synthyra/FastPLMs.git"
```

Python 3.11-3.14, PyTorch 2.13, and Transformers 5.13 are required. The declared CPU gate covers tiny offline contracts; published checkpoint throughput and parity require the documented device tier. The Hub quick start below requires network
access on first download. For an air-gapped run, first build the manifest-pinned
local artifact and use the offline form shown in the example.

## Quick start

```python
from transformers import AutoModel

model_id = "Synthyra/ESM3_small"
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
).eval()
```

This example uses the published Hub repository. For offline validation, build
the manifest-pinned artifact and replace `model_id` with its local
`dist/hub/ESM3_small` path, then pass `local_files_only=True`.

Leave attention unspecified for the Transformers default. Supported explicit
choices are `eager`, `sdpa`, `flex_attention`.
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

## Sequence inference and masked-sequence generation

ESM3 owns its sequence preparation. This example exercises the sequence track;
the public input contract also supports structure and function tracks through
the multimodal helpers:

```python
import torch

batch = model.tokenize_sequences(
    ["MKTAYIAKQ", "GGGG"],
    device=model.device,
)
with torch.inference_mode():
    output = model(**batch)

print(output.last_hidden_state.shape)
print(output.logits.shape)
print(output.structure_logits.shape)
print(output.function_logits.shape)
```

When `return_dict=False`, ESM3 follows the standard base-model tuple prefix:
`last_hidden_state`, then requested `hidden_states` and `attentions`. Multimodal
logits and extensions follow that prefix. Prefer named fields for individual
tracks.

Generate masked sequence positions with an explicit seed:

```python
from fastplms.models.esm3.modeling_esm3 import FastESM3GenerationConfig

config = FastESM3GenerationConfig(
    num_steps=8,
    temperature=1.0,
    seed=7,
)
generated = model.generate("MK____A", config)
print(generated)
```

Underscores mark positions to generate. Model outputs are predictions over
tracks, not experimental measurements of structure or function.

## Runtime contract

- Public input: Sequence, structure, and function tracks prepared through the multimodal helpers
- Advertised AutoClasses: `AutoConfig`, `AutoModel`
- AutoClass weight status: `AutoConfig` = `FastPLMs extension`, `AutoModel` = `pretrained`
- Attention implementations: `eager`, `sdpa`, `flex_attention`
- Precision policies: `default`
- BF16 execution: `fp32_parameters_autocast`
- Generation contract: `not_applicable`
- Optional dependency group: `core`
- Weight publication allowed: `true`
- Weight license status: `resolved`
- Redistributable: `true`
- Complete weight publication required: `false`

## Release record

- FastPLMs weights: `Synthyra/ESM3_small`
- Runtime revision: recorded separately in the built artifact and published commit
- Source-tree and runtime-bundle SHA-256: recorded in `provenance.json`
- Generator/schema version and complete/runtime-only attestations: recorded in `provenance.json`
- Official checkpoint: `biohub/esm3-sm-open-v1`
- Artifact source: `fast`
- State transform: `esm3_to_fastplms_v1`
- BF16 execution: `fp32_parameters_autocast`
- Pinned upstreams: `biohub-esm`, `biohub-transformers`
- Reference container: `reference-biohub-esm`
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

Checkpoint terms: MIT. The Hub model-card identifier is
`mit`. Applicable source licenses, notices, attribution,
and conversion records are distributed with the local artifact. Review them
before use.
