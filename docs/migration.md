# Migration to FastPLMs 1.0

FastPLMs 1.0 intentionally changes the repository layout, embedding return
types and storage, attention selection, artifact publication, and several model
contracts. There are no compatibility imports or silent keyword aliases. Run
the snippets in this guide in the offline CPU documentation job whenever this
contract changes.

## Dependencies and source layout

FastPLMs 1.0 requires Python 3.11-3.14, PyTorch 2.13, and Transformers 5.13.
The project is no longer installed as a Python distribution. Published models
carry their runtime source in the Hugging Face repository and load it with
`trust_remote_code=True`. Install dependencies directly:

```bash
python -m pip install \
  "torch>=2.13,<2.14" \
  "transformers>=5.13,<5.14"
```

Repository tools and source-level APIs run with `PYTHONPATH=src`. Their
dependencies are composed under `requirements/` rather than distribution
metadata:

```bash
uv pip install \
  -r requirements/profiles/cpu-validation.in \
  -c requirements/constraints/validation.txt \
  --torch-backend cpu
PYTHONPATH=src python -m pytest tests/cpu -m cpu_contract
```

Source remains under `src`:

| Pre-1.0 path/import | FastPLMs 1.0 |
| --- | --- |
| `fastplms/esm2/...` | `src/fastplms/models/esm2/...` |
| family-local attention helpers | `fastplms.attention` |
| `fastplms.embedding_mixin` | `fastplms.embeddings` and `fastplms.embed_dataset` |
| family-local TTT helpers | `fastplms.models.ttt` |
| `testing/...` operational scripts | `tools.remote`, `tools.artifacts`, `tools.goldens`, or `benchmarks` |

Model implementations remain loadable through the AutoClasses declared in the
[generated support matrix](generated/support.md). Runtime source must not import
`vendor/upstream`; those repositories are isolated compliance oracles.

## Attention selection and outputs

Replace `config.attn_backend`, `model.attn_backend`, `flex`,
`kernels_flash`, and `auto` with the Transformers attention interface:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    attn_implementation="flex_attention",
)
model.set_attn_implementation("sdpa")
```

The 1.0 names are `eager`, `sdpa`, `flex_attention`,
`flash_attention_2`, and `flash_attention_3`, restricted by the manifest.
There is no `auto` backend, source compilation, or unavailable-kernel fallback.

`output_attentions=True` is a documented exception: when an optimized backend
cannot return the full matrix, FastPLMs emits one warning containing the
configured backend, effective `eager` backend, and reason. It derives the eager
4-D mask from the original padding and causal semantics for that call only.
Configuration and later calls are unchanged.

ESMC Flex Attention and FlashAttention 3 remain supported and
non-experimental. SDPA is recommended for highest numerical fidelity. Their
published deviations are diagnostic warnings, while dispatch, masks, finite
outputs, shapes, and catastrophic biological disagreement remain hard gates.
Do not convert a threshold into a measured number.

## Embedding arguments

The historical method deduplicated and length-sorted sequences, returned a
sequence-keyed dictionary, and defaulted to writing pickle. The 1.0 operation
preserves order and duplicates and returns `EmbeddingResult`.

| Pre-1.0 argument | FastPLMs 1.0 replacement |
| --- | --- |
| `sequences=values` | positional `inputs`, accepting sequences, pairs, records, mapping, generator, or FASTA path |
| `fasta_path=path` | positional `inputs=path`; combine sources explicitly with a generator when needed |
| `max_len=n` | `max_length=n`, always measured in biological residues |
| `pooling_types=[...]` | `pooling=(...)` |
| `embed_dtype=dtype` | `dtype=dtype` |
| `save=True, save_path=...` | `output=..., format="safetensors"` |
| `sql=True, sql_db_path=...` | `output=..., format="sqlite"` |
| `padding="max_length"` or `"longest"` | bounded `batch_window_size` and optional `max_tokens_per_batch` |
| `num_workers` | removed; FASTA parsing and immutable spooling are bounded and deterministic |
| `hidden_state_index`, `store_all_hidden_states` | retained, applied to the model-selected hidden-state stack |

```python
result = model.embed_dataset(
    inputs,
    batch_size=8,
    batch_window_size=64,
    max_tokens_per_batch=8192,
    max_length=1024,
    pooling=("mean",),
    output="embeddings.sqlite",
    format="sqlite",
    resume=True,
)
```

`pooling=None` selects mean pooling unless `full_embeddings=True`. An explicit
pooler with `full_embeddings=True` raises. Full embeddings contain only
biological residues; BOS, EOS, padding, chain delimiters, and structure-only
tokens are removed.

E1's pre-1.0 `embed_dataset_with_msa` dictionary return is also replaced by an
ordered `EmbeddingResult`. Duplicate queries are preserved, and record IDs are
their zero-based input positions. Its native names `max_len`, `pooling_types`,
and `matrix_embed` remain because E1 has no tokenizer, while `output`, `format`,
`resume`, `shard_size`, `model_state_fingerprint`, `batch_window_size`, and
`max_tokens_per_batch` use the shared persistence and bounded-batching
contracts. See [`examples/e1_rag.py`](../examples/e1_rag.py).

## Return values and storage readers

Replace dictionary indexing with ordered records:

```python
for record in result:
    tensor = record.load_tensor()
    print(record.id, record.sequence, tensor.shape)
```

`result.as_dict(key="id")` raises for duplicate keys unless an explicit
duplicate policy is chosen. Sharded safetensors and SQLite are the writable
formats. The run manifest is the authoritative safetensors commit and can
recover a valid committed run when the standalone index is interrupted.

Replace old readers as follows:

```python
from fastplms.embeddings import (
    convert_legacy_sqlite,
    load_legacy_pth,
    load_result,
    load_sqlite_result,
)

current = load_result("embeddings")
selected = load_sqlite_result(
    "embeddings.sqlite",
    record_ids=["b", "a", "b"],
)
convert_legacy_sqlite("legacy.db", "embeddings-v1.sqlite")
trusted_pickle = load_legacy_pth(
    "legacy.pth",
    allow_unsafe_pickle=True,
)
```

SQLite retrieval opens read-only and preserves selector order and duplicates.
Legacy pickle remains executable input and therefore requires explicit opt-in.

## ANKH full-checkpoint replacement

FastPLMs 1.0 replaced each Synthyra encoder-only ANKH repository with the full
official-compatible T5 state. This increases the default checkpoint size while
preserving encoder output parity.

`AutoModel` loads the encoder and shared state without decoder allocation.
`AutoModelForSeq2SeqLM` loads encoder, decoder, cross-attention, and LM head from
the same repository. Set `ankh_id` to a Synthyra ANKH repository or a validated
local artifact:

```python
from transformers import AutoModel, AutoModelForSeq2SeqLM

encoder = AutoModel.from_pretrained(
    ankh_id,
    revision=ankh_revision,
    trust_remote_code=True,
)
seq2seq = AutoModelForSeq2SeqLM.from_pretrained(
    ankh_id,
    revision=ankh_revision,
    trust_remote_code=True,
)
```

ANKH embeddings default to `hidden_state_source="encoder"` and
`hidden_state_index=-1`. Decoder extraction requires exactly one explicit
aligned decoder text list or ID tensor:

Remove the pre-1.0 residue-spacing workaround. Pass raw sequences such as
`MSTNPK`, not `M S T N P K`, and write sentinel prompts as
`M<extra_id_0>`, not `M <extra_id_0>`. FastPLMs now applies the same safe
pre-tokenizer and normalization to model-owned and explicitly supplied ANKH
tokenizers.

```python
encoder_layers = encoder.embed_dataset(
    inputs,
    hidden_state_source="encoder",
    store_all_hidden_states=True,
    full_embeddings=True,
)
decoder_layer = seq2seq.embed_dataset(
    inputs,
    hidden_state_source="decoder",
    hidden_state_index=-1,
    decoder_inputs=["M<extra_id_0>" for _ in inputs],
    full_embeddings=True,
)
```

No shifted-source decoder input is invented. The official ANKH workflows use
task prompts, sentinels, or generated tokens. Decoder pooling excludes special
tokens and persisted metadata fingerprints the decoder input and alignment.

Files-only publication is forbidden for this migration. Every weight shard,
weight index, tokenizer asset, configuration, runtime source, model card, and
release record must land in one immutable Hub commit. Both AutoClass views
must pass artifact and live parity from that same commit.

Complete publication may remove a superseded monolithic weight path in that
same commit only when the path is pinned in the current registry, absent from
the validated sharded inventory, and its remote digest and parent still match
preflight. Files-only publication remains strictly add-only.

The separately named `FastAnkhForMaskedLMExtension` remains a FastPLMs
extension, not an official ANKH head. Sequence and token classification views
load pretrained base weights with newly initialized task heads.

## AutoClass weight meaning

Every family and entry point is classified in the
[capability-to-evidence manifest](generated/capability_evidence.md):

- `pretrained`: the advertised base or head exists in checkpoint state;
- `base weights + untrained task head`: train the classification head before
  interpreting logits;
- `FastPLMs extension`: integration code or a head that is not an official
  pretrained capability.

All advertised classes must honor `return_dict`, output flags, tuple order,
embedding resize and setters, initialization, forward/loss/backward, and
save/reload.

ESM2, ESMC, DPLM, and DPLM2 task outputs now keep the Transformers task
prefix: `loss` when labels are present, then `logits`, `hidden_states` when
requested, and `attentions` when requested. FastPLMs diagnostics such as
`s_max` follow those standard fields. Masked-LM outputs that expose
`last_hidden_state` place it after the diagnostic extension. Tuple output is
exactly `output.to_tuple()`, so disabled or unavailable fields, including an
unconfigured pooler, are omitted rather than represented by `None`. These
forwards also reject misspelled or unsupported keyword arguments instead of
silently discarding them.

ESM3 `AutoModel` tuple output now begins with the standard base-model fields in
this order: `last_hidden_state`, `hidden_states` when requested, and
`attentions` when requested. Sequence, structure, function, residue, and other
multimodal outputs follow that prefix. Prefer named fields when consuming the
additional tracks. This is a deliberate v1 tuple-order correction for callers
that previously treated element zero as sequence logits.

## ESMFold2 and structure dependencies

Only the standard, fast, experimental cutoff 2025, and experimental fast
cutoff 2025 ESMFold2 variants remain supported. Dataset embeddings accept a
single protein chain and expose the learned width-256 representation.

The full `ESMFold2` and `ESMFold2-Experimental-Cutoff2025` checkpoints have 48
folding blocks and retain optional MSA conditioning. The two Fast checkpoints
have 24 folding blocks and were trained without MSA conditioning, so they
reject MSA-derived inputs rather than silently ignoring them. This includes
`ProteinInput.msa` and low-level MSA-derived features. Fast still supports the
declared multichain and multimolecule inputs when every protein chain uses
`msa=None`. This distinction follows the official model description in
[Appendix A.2.1](https://biohub.ai/papers/esm_protein.pdf).

`esmc_precision="auto"` resolves to BF16. FP8 is an explicit, experimental,
inference-only request and raises when the validated Transformer Engine path is
unavailable. General structure dependencies live in
`requirements/features/structure.in`; reporting and binder dependencies remain
separate.

## Commands and validation tiers

Initialize official references only when running compliance:

```bash
git submodule update --init --recursive
python -m tools.remote --host user@gpu-host --identity /path/to/key --suite compliance
```

Routine pull requests use the offline CPU contract and static/source checks:

```bash
python -m pytest tests/cpu -m cpu_contract -n auto --dist=loadscope \
  --durations=25 --junitxml=artifacts/junit/cpu-contract.xml
PYTHONPATH=src python -m tools.artifacts.generate_docs --check
```

The routine `check` tier consumes immutable goldens and does not build live
official references. Exact-device Hopper/SM90 golden smoke, nightly
kernel/throughput work, and the
frozen-head compliance release candidate remain separate cost tiers.
