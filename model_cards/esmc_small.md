---
library_name: transformers
license: "mit"
tags:
  - protein-language-model
  - fastplms
---

<!-- Generated from src/fastplms/models.toml. Do not edit. -->

# Synthyra/ESMplusplus_small

This checkpoint packages the FastPLMs `ESMC` implementation.

Accepted inputs are amino-acid sequences tokenized to residue IDs.
Supported Transformers entry points are `AutoConfig`, `AutoModel`,
`AutoModelForMaskedLM`.

## Install and platform requirements

Install FastPLMs from the exact source revision paired with this model card:

```bash
python -m pip install \
  "fastplms @ git+https://github.com/Synthyra/FastPLMs.git@<runtime-revision>"
```

Python 3.11-3.14, PyTorch 2.13, and Transformers 5.13 are required. Eager, SDPA, and Flex use the core install. FlashAttention requires the `flash` extra, compatible CUDA hardware, and BF16 execution. The Hub quick start below requires network
access on first download. For an air-gapped run, first build the manifest-pinned
local artifact and use the offline form shown in the example.

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
`dist/hub/ESMplusplus_small` path, then pass `local_files_only=True`.

Leave attention unspecified for the Transformers default. Supported explicit
choices are `eager`, `sdpa`, `flex_attention`, `flash_attention_2`,
`flash_attention_3`.
Pass the selected name through `attn_implementation`.
When an optimized backend cannot return full attention tensors,
`output_attentions=True` emits one explicit runtime warning and uses a correctly
masked eager implementation for that call only. The warning identifies the
configured backend, effective backend, and reason. Configuration and later
calls are unchanged.
For BF16 execution, this family uses parameters loaded directly in BF16.

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

## ESMC behavior

This artifact exposes the Biohub ESMC sequence encoder and masked-language-model
head through Transformers. It is also the language-model family used by
ESMFold2. SDPA is the default and the recommended choice for highest numerical
fidelity. Flex Attention and FlashAttention 3 are supported, non-experimental
backends, but their BF16 arithmetic may be numerically divergent from SDPA.
Those deviations produce diagnostic warnings rather than strict parity
failures; dispatch integrity, masks, finite outputs, shapes, and catastrophic
biological disagreement remain hard gates.

The current locked GH200/aarch64 release image executes eager, SDPA, and Flex.
It has no validated FlashAttention 2 kernel and no locked aarch64 artifact for
the manifest-pinned FlashAttention 3 kernel. Both Flash backends remain
supported, non-experimental interfaces, but requests fail closed without
dispatch on this image. Prior real FlashAttention 2 execution was captured in
separate workstation JUnit, but that immutable report and environment
attestation are not bundled in this repository. It is not reused as a current
ESMC release distribution or numerical claim.

When `sequence_id` is supplied, it is authoritative for ESMC attention grouping
and padding, and `attention_mask` is ignored. Values greater than or equal to
zero are valid sequence-group IDs; `-1` denotes padding. Omit `sequence_id` to
use `attention_mask` as the padding contract.

| Backend | Support | Measurement status |
| --- | --- | --- |
| `sdpa` | Recommended fidelity path | Pending complete validated 30-record frozen-head GH200/aarch64 set |
| `eager` | Supported | Pending complete validated 30-record frozen-head GH200/aarch64 set |
| `flash_attention_2` | Supported | Current GH200/aarch64 lock: unavailable; pending structured schema-v3 availability record |
| `flex_attention` | Supported, numerically divergent | Pending complete validated 30-record frozen-head GH200/aarch64 set |
| `flash_attention_3` | Supported, numerically divergent | Current GH200/aarch64 lock: unavailable; pending structured schema-v3 availability record |

No threshold, report from another checkpoint, or result from another
accelerator is substituted for a measurement. A release set contains all
30 model/backend/panel records from one exact GH200 device and aarch64 runtime:
18 eager/SDPA/Flex measurements include relative L2, Q99.9, residue
cosine, pooled cosine, top-1, and Jensen-Shannon distributions; 12
FlashAttention 2/3 records explicitly attest locked-platform unavailability.

No number is inferred from a threshold or copied from a different checkpoint.
Before release, replace each pending cell only through the strict 30-record
ingestion gate tied to this exact Hub revision, runtime revision, BF16 dtype,
the current exact GH200/aarch64 accelerator and container identity, and both
locked sequence panels. The set contains 18 eager/SDPA/Flex measurement records
and 12 structured Flash locked-platform unavailable records. H100 and H200
remain Hopper-class deployment examples, but they are not current ESMC release-confirmation evidence.
Results are never
carried across device, platform, source, lock, or image identities.
Diagnostic JSON is written under `artifacts/diagnostics/esmc/`. Catastrophe
guardrails (relative L2 `0.25`, Q99.9 `0.50`, residue cosine `0.90`, pooled
cosine `0.95`, top-1 `0.80`, Jensen-Shannon `0.05`) detect corruption and do
not constitute parity or quality claims.

## Runtime contract

- Public input: Amino-acid sequences tokenized to residue IDs
- Advertised AutoClasses: `AutoConfig`, `AutoModel`, `AutoModelForMaskedLM`
- AutoClass weight status: `AutoConfig` = `FastPLMs extension`, `AutoModel` = `pretrained`, `AutoModelForMaskedLM` = `pretrained`
- Attention implementations: `eager`, `sdpa`, `flex_attention`, `flash_attention_2`, `flash_attention_3`
- Precision policies: `default`
- BF16 execution: `static_parameters`
- Generation contract: `not_applicable`
- Optional dependency group: `core`
- Weight publication allowed: `true`
- Weight license status: `resolved`
- Redistributable: `true`
- Complete weight publication required: `false`

## Provenance

- FastPLMs weights: `Synthyra/ESMplusplus_small@46c5f7d562e47d4c14165b424c71ab7db008e6fb`
- Runtime revision: recorded separately in the built artifact and published commit
- Source-tree and runtime-bundle SHA-256: recorded in `provenance.json`
- Generator/schema version and complete/runtime-only attestations: recorded in `provenance.json`
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
