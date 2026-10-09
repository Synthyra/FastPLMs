# Embedding API

## Dependencies and platform requirements

The shared sequence embedding API requires Python 3.12-3.14, PyTorch 2.14, and
Transformers 5.17. Install these dependencies. Transformers loads the runtime
source from the pinned Hugging Face model repository:

```bash
python -m pip install \
  "torch>=2.14" \
  "transformers>=5.17"
```

Core tokenizer-mode embeddings run on CPU or CUDA. E1 uses its raw-sequence
adapter rather than a tokenizer. Structure models and optional FlashAttention
backends require the additional dependencies and CUDA platforms declared in
the support matrix.

## Quick start

Published models expose `model.embed_dataset(...)`. A source checkout also
exposes the same implementation as `fastplms.embed_dataset(model, ...)` when
run with `PYTHONPATH=src`. This minimal Hugging Face example returns one
mean-pooled vector per sequence:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "Synthyra/ESM2-150M",
    trust_remote_code=True,
).eval()
result = model.embed_dataset(
    [
        ("protein-a", "MSTNPKPQRKTKRNT"),
        ("protein-b", "MKTIIALSYIFCLVFA"),
    ],
    batch_size=2,
    pooling=("mean",),
)
print(result[0].id, result[0].tensor.shape)
```

The operation accepts sequences, `(id, sequence)` pairs, `EmbeddingInput`
values, an insertion-ordered `{id: sequence}` mapping, or a FASTA path. It
preserves input order, mapping and FASTA identifiers, and duplicate records.

## Argument reference

| Argument | Meaning |
| --- | --- |
| `inputs` | Sequence iterable, `(id, sequence)` pairs, `EmbeddingInput` values, `{id: sequence}` mapping, or FASTA path |
| `batch_size` | Number of records prepared together; must be positive |
| `pooling` | One pooler or an ordered pooler sequence; `None` selects mean unless `full_embeddings=True` |
| `full_embeddings` | Return residue-level tensors instead of pooled vectors |
| `taps` | Several hidden-state outputs from one forward pass per batch; see [One-pass taps](#one-pass-taps) |
| `output` | Safetensors directory or SQLite file; omit for in-memory results |
| `format` | `safetensors` or `sqlite` when `output` is set |
| `resume` | Reuse an exact compatible ordered prefix when persistent output exists |
| `tokenizer` | Explicit tokenizer override for a compatible tokenizer-mode family |
| `max_length` | Optional maximum number of biological residues, excluding tokenizer-added special tokens |
| `keep_special_tokens` | Tap runs only: keep CLS and EOS in every row output and pool over all `l + 2` rows; default `False`. A contract that keeps them selects it by itself |
| `truncate` | Truncate biological residues to `max_length`; when false, an over-length record raises |
| `batch_window_size` | Bounded number of records eligible for stable length bucketing; defaults to `16 * batch_size` |
| `max_tokens_per_batch` | Optional padded biological-residue budget for one inference batch |
| `dtype` | Output tensor dtype; `None` retains the model output dtype |
| `shard_size` | Target safetensors shard size in bytes |
| `model_state_fingerprint` | Caller-supplied state identity for offloaded or externally managed models |
| `**model_kwargs` | Family-specific embedding controls such as hidden-state selection |

`store_all_hidden_states=True` is a model keyword and requires
`full_embeddings=True`. `full_embeddings=True` cannot be combined with an
explicit pooler. The output format and every invalid argument combination are
validated before input hashing, model inference, or output creation.

## Bounded streaming and length policy

The runner reads FASTA input line by line into an immutable incrementally
fingerprinted spool. It never reads the complete FASTA file into memory. It
keeps one bounded `batch_window_size` group for length bucketing, applies
`max_tokens_per_batch` to the padded biological-residue count, and restores
source order. If omitted, the window is sixteen times the batch size. An
explicit value takes precedence. Result metadata and the resume fingerprint
record the resolved window. Persistent outputs store result descriptors once and
keep tensor payloads lazy.

SQLite prefixes commit at completed batch-window boundaries. Safetensors packs
windows into bounded shards and publishes a resumable prefix whenever a shard
flushes; an interruption replays the unflushed in-memory shard. Set
`batch_window_size=batch_size` when per-batch inference boundaries matter more
than the default padding-efficiency lookahead.

`result.metadata["batching"]["resume_commit_granularity"]` records
`"batch-window"` for persistent SQLite, `"shard-flush"` for persistent
safetensors, and `"not-applicable"` for an in-memory result. A new or replacement
SQLite run remains staged or deferred and does not replace the default readable
run until its first batch window commits.

`max_length` always counts amino-acid residues. Tokenizer-mode families add
their required BOS, EOS, or modality boundary width when constructing the model
token budget. `truncate=False` does not silently exceed the model contract: an
input longer than `max_length` raises with the record position and identifier.

## Result types

The following source-level imports are for contributor workflows run with
`PYTHONPATH=src`; Hugging Face users can pass `(id, sequence)` pairs directly
to the model method.

```python
from fastplms import EmbeddingInput

inputs = [
    EmbeddingInput("a", "MSTNPKPQRKTKRNT"),
    EmbeddingInput("a", "MKTIIALSYIFCLVFA"),
]
result = model.embed_dataset(inputs, batch_size=2)

for record in result.records:
    print(record.id, record.sequence, record.tensor)
```

`EmbeddingRecord(id, sequence, tensor)` is ordered and retains the original
sequence. `EmbeddingResult(records, metadata)` is sequence-like. Persisted
records may hold a `LazyTensorReference`; call `record.load_tensor()` to load
that tensor.

`result.as_dict(key="id")` raises when keys repeat. Callers must explicitly
choose a duplicate policy if they want `first` or `last`. This
prevents silent loss of repeated FASTA identifiers.

## Biological-residue policy

Models return a representation `X` and a biological residue mask `M`:

```text
X: (b, l, d)
M: (b, l)
```

Pooling includes positions where `M` is true. BOS, EOS, padding, chain
delimiters, and non-protein structure tokens are excluded. E1 derives `M` from
its native raw-sequence preparation because it has no tokenizer. DPLM2 accepts
raw amino-acid sequences through a model adapter that adds its modality-specific
boundaries and invokes the exact tokenizer with `add_special_tokens=False`.
Each persisted run records the token policy and tokenizer metadata.

A tap run with `keep_special_tokens=True` (the canonical policy, [feature store](feature_store.md#canonical-token-stores))
keeps CLS and EOS: `M` is the attention mask, so a sequence of `l` residues gives `l + 2` rows,
`X: (b, l+2, d)` before padding, and padding is the only thing masked. Taps, pooling and the SAE
reducer then cover all `l + 2` rows, with pooling semantics version 3. Without it, the residue-only
policy above stays in force for legacy stores.

## Pooling

The supported operations are:

| Name | Transformation | Limitation |
| --- | --- | --- |
| `mean` | Arithmetic mean over valid residues | None |
| `max` | Elementwise maximum over valid residues | None |
| `norm` | Elementwise L2 norm across valid residues | None |
| `median` | Elementwise median over valid residues | More expensive than mean |
| `std` | Elementwise population standard deviation | Requires at least one residue |
| `var` | Elementwise population variance | Requires at least one residue |
| `cls` | Model-defined classification position | Rejected without meaningful CLS semantics |
| `parti` | Attention-graph weighted residue summary | Eager only and at most 2,048 residues |

Multiple poolers are concatenated in request order. Metadata records the output
slice for each operation. `parti` uses Torch power iteration with damping 0.85,
tolerance `1e-6`, and at most 100 iterations. It requires an explicit
`attn_implementation="eager"` because it materializes the attention graph.

Pooling semantics version 2 uses the biological-residue mask, excluding padding and every
tokenizer-declared special token. `cls` remains the explicit position-zero exception.
Empty residue sets are rejected; population variance and standard deviation of one residue
are zero. Reductions accumulate in FP32 for FP16, BF16, and FP32 inputs, and in FP64 for FP64
inputs. The result is cast to the input dtype after reduction; nonfinite biological values
or output-cast overflow are errors. The runner's `dtype` conversion happens before pooling,
so the default FP32 output uses FP32 reductions. `pooling_semantics` in metadata and the run
fingerprint records this policy.

`max_length` counts biological residues. With `truncate=True`, the input prefix is selected
before the forward pass; with `truncate=False`, overlength inputs fail before that pass.
There is no sliding-window aggregation. Residue tensors retain input order after special-token
removal. `retained_positions` records this policy; the record's `sequence` remains the original
input, while the tensor's residue axis gives the retained length.

```python
result = model.embed_dataset(
    inputs,
    batch_size=8,
    pooling=("mean", "max", "std"),
)
print(result.metadata["pool_slices"])
```

Choose poolers for the downstream object. `mean` gives a sequence summary,
`max` shows large per-feature responses, and `std` or `var` describes
within-sequence dispersion. Concatenating poolers increases output width. It is
a feature-design decision, not a free accuracy improvement.

## Full residue embeddings

`full_embeddings=True` returns one ragged residue tensor per input and cannot be
combined with pooling:

```python
result = model.embed_dataset(
    inputs,
    batch_size=4,
    full_embeddings=True,
)
```

Each tensor has shape `(l_i, d)`, where `l_i` is the number of retained
biological residues for record `i`. Padding is never persisted as a residue
embedding.

Passing `store_all_hidden_states=True` requires `full_embeddings=True` and
returns one tensor with shape `(n, l_i, d)` per input, where `n` follows the
model's hidden-state output order. The biological residue mask is applied only
to the token axis. Safetensors and SQLite preserve this rank without flattening
the state axis.

ESMFold2 returns the learned projection with shape `(l_i, 256)`. Its dataset
path accepts only single-chain sequences and FASTA records and supports the
residue-statistic poolers. It rejects `cls` and `parti`.

### ANKH encoder and decoder layers

The Synthyra ANKH repositories contain the complete encoder-decoder
checkpoints. `AutoModel` exposes the encoder view and
`AutoModelForSeq2SeqLM` exposes the full sequence-to-sequence view.

ANKH defaults to the encoder final state:

```python
encoder = model.embed_dataset(
    inputs,
    hidden_state_source="encoder",
    hidden_state_index=-1,
    full_embeddings=True,
)
```

`hidden_state_index` is applied to the selected stack, and
`store_all_hidden_states=True` stores every state from that stack. Decoder
extraction requires the full `AutoModelForSeq2SeqLM` view and exactly one
explicit aligned `decoder_inputs` sequence or `decoder_input_ids` tensor:

Use raw protein strings such as `MSTNPK`, not space-separated residues.
Decoder sentinels must be adjacent to their residues, as in
`M<extra_id_0>`. FastPLMs applies this normalization consistently to the
model-owned tokenizer and an explicitly supplied tokenizer object.

```python
decoder = seq2seq.embed_dataset(
    inputs,
    hidden_state_source="decoder",
    decoder_inputs=["M<extra_id_0>" for _ in inputs],
    hidden_state_index=-1,
    full_embeddings=True,
)
```

There is no implicit shifted-source decoder input. Official ANKH tasks use
task-dependent prompts, sentinels, or generated tokens. A
`decoder_attention_mask` is valid only with `decoder_input_ids`. Decoder pooling
uses the decoder biological mask and excludes start, EOS, padding, sentinel,
and other tokenizer-special positions. Metadata records stack, layer, decoder
input and mask fingerprints, input-position alignment, and mask policy.

### E1 MSA-aware embeddings

E1 keeps its native raw-sequence and retrieval preparation, but returns the
same ordered, duplicate-preserving `EmbeddingResult` as the shared embedding
API. Record IDs are the zero-based input positions, so repeated query sequences
remain independently addressable as `"0"`, `"1"`, and so on.

```python
result = model.embed_dataset_with_msa(
    [query, query],
    msa_lookup={query: "/data/query.a3m"},
    batch_size=2,
    max_len=len(query),
    pooling_types=["mean"],
    seed=7,
    batch_window_size=2,
    max_tokens_per_batch=2 * len(query),
    output="e1-msa.sqlite",
    format="sqlite",
    resume=True,
)
assert [record.id for record in result] == ["0", "1"]
```

`max_len` is measured in biological residues. `matrix_embed=True` selects full
residue output. `output`, `format`, `resume`, `shard_size`, and
`model_state_fingerprint` have the same persistence and compatibility meaning
as ordinary dataset embedding. Local A3M input is offline; homology search and
Hub MSA acquisition are separate, explicit networked workflows.

## One-pass taps

`taps=` returns several outputs of one model from a single forward pass per
batch. Each tap names one hidden state and what to keep from it. The pass stops
once the deepest tapped state exists, so a plan whose deepest tap is state 27 of
a 36-block model runs 27 blocks. ESM++ (ESMC) models support taps. Every other
family raises rather than running one pass per tap.

```python
import torch

from fastplms.embeddings import HiddenTap

features = model.embed_dataset(
    inputs,
    batch_size=8,
    taps=[
        HiddenTap("last", layer=-1, dtype=torch.bfloat16),
        HiddenTap("last_meanvar", layer=-1, pooling=("mean", "var"), dtype=torch.float32),
        HiddenTap("mid_max", layer=12, pooling="max"),
    ],
)
record = features[0]
print(record.id, record.tensors["last"].shape, record.tensors["last_meanvar"].shape)
```

The tap types come from `fastplms.embeddings`, which a loaded FastPLMs artifact
installs as the `fastplms` package. Layer indices follow the order of
`output_hidden_states`: index `i` is the input to block `i`, index `n` (the
block count) is the final normalized state, and negative indices count back from
it, so `-1` is the final state. The final norm runs only when a tap names the
final state.

- `HiddenTap(name, layer, pooling=None, dtype=None)` keeps one `(l_i, d)` tensor of
  biological residue rows per record. With pooler names it applies `Pooler` as
  `pooling=` does, `cls` included, and equals the single-output run with the
  same `hidden_state_index`. It rejects `parti`, which needs the attention graph
  of a full pass. `dtype=None` inherits the run dtype; an explicit floating dtype overrides
  it for that output. Each conversion starts from the original captured state, so storing
  residue rows as BF16 does not quantize an FP32 pooled sibling.
- `ReducedTap(name, layer, reduce, identity)` calls `reduce` with a `TapBatch`:
  the state `X` with shape `(b, l, d)`, `token_mask` with every attended token,
  BOS and EOS included, and `residue_mask` with the biological residues. `reduce`
  returns one row per record, shape `(b, ...)`, such as a sparse-autoencoder
  encoding pooled per sequence. `identity` describes the reducer in plain data:
  strings, numbers, booleans, `None`, lists, and string-keyed mappings.
- `StreamingTap(name, layers, begin, identity, required_state_count=None)` reduces selected
  states during the same pass. Layers are distinct ascending nonnegative indices. `begin()`
  creates a fresh `LayerAccumulator` for each batch; `update(layer, TapBatch)` borrows each
  original state, and `finish()` returns token-aligned `(b, l, c)` features. The engine applies
  its biological residue mask and restores input order. Reducers must not mutate or retain
  borrowed hidden states. They own their arithmetic, independently of the run's output dtype,
  and record it in `identity`. An optional required state count rejects an incompatible depth.
  Only sibling `HiddenTap`/`ReducedTap` states are saved; no callbacks remain on the model.

The result is a `TapResult` of `TapRecord(id, sequence, tensors)` values in input
order, with `tensors` keyed by tap name. `result.metadata["taps"]` records the
plan with each layer resolved, the stop layer, and each pooled tap's output
slices. The run fingerprint binds the plan, per-tap dtype overrides, and every reducer identity.
The run `dtype` converts states before pooling or a `ReducedTap` unless a hidden tap overrides it;
use `dtype=None` to pass original model states to a reducer. A reducer's output
keeps the dtype it returns. On an FP8-enabled ESMC model, taps use the padding
and Transformer Engine context of `forward`.

`taps=` cannot be combined with `pooling`, `full_embeddings`,
`hidden_state_index`, or `store_all_hidden_states`. `embed_dataset` returns tap
records in memory, and `output=` raises: persist them with `embed_into_features`
(see [the feature store](feature_store.md)), which writes each tap into the
store of its key and embeds only the sequences that store lacks.

For bounded delivery, supply `tap_sink(records, identity)` to `embed_dataset`. It receives at
most `batch_window_size` ordered `TapRecord` objects at a time, plus the run and input
fingerprints. Retaining these records in the callback retains their tensors; consume them and
release them before returning. A successful call returns `TapRunReceipt(record_count, metadata)`
without tensors. Exceptions stop delivery and restore the model's original training mode.
This uses the same batching and tap executor as the in-memory path. The feature writer supplies
this sink itself, so ordinary `embed_into_features` no longer collects a full-corpus `TapResult`.

Run-fingerprint schema 5 excludes the physical input-storage choice. Identical ordered inputs
and computation have the same identity whether their descriptors live in memory or a disk
spool. Metadata still records that choice. Schema-4 output remains readable through its existing
loader, but cannot silently resume as a schema-5 run; keep it for its historical experiment.

### Fixed batch shapes

A GEMM's reduction order follows its shape, so in BF16 a sequence's rows can change with the
companions that set its batch's padded width and row count. `embed_token_features(geometry=...)`
and `TokenTapExecutor(geometry=...)` remove that dependence: with a `BatchGeometry`, a sequence of
`l` residues after the crop runs in the bucket `T = bucket_tokens * ceil((l + 2) / bucket_tokens)`,
and every batch of that bucket holds exactly `rows(T)` sequences padded to `T` columns. `rows(T)` is
`min(max_rows, token_budget // T)`. A bucket's last batch repeats its last sequence to fill and
discards the copies' outputs. `plan_geometry_batches` groups the sequences by bucket, longest
bucket first, in row-key order, so the plan does not depend on the input order either.

```python
from fastplms.embeddings import BatchGeometry, embed_token_features

geometry = BatchGeometry(bucket_tokens=64, token_budget=32768, max_rows=256)  # 32 buckets up to 2,048 tokens
embed_token_features(model, sequences, root, features, taps=taps, geometry=geometry)
```

A geometry run takes none of `max_sequences`, `max_tokens`, `window` or `fixed_batch_size`, and its
run record's batch policy is `geometry.describe()`. Without a geometry, batches follow the token
budget and pad to their longest member, as before.

## Safetensors storage

With `format="safetensors"`, `output` names an output directory. FastPLMs writes
generation-scoped shards and then transactionally publishes:

```text
output/
  embeddings-run-<generation>-00001.safetensors
  embeddings-records-run-<generation>-00001.jsonl
  embeddings-index-run-<generation>-00001.json
  index.json
  run.json
```

The default maximum shard size is 2 GiB. Tensors are packed across inference
batches and written one shard at a time, so the complete tensor dataset is
never materialized in host memory. Each flushed shard publishes an incomplete
ordered prefix that a matching `resume=True` call can continue. An interrupted,
unflushed shard is recomputed. Generation descriptors preserve record position,
identifier, sequence, shape, dtype, tensor hash, and shard key. Loading the
result creates lazy references rather than reading every shard into memory.
`run.json` is the transactional commit marker. It points to one immutable
generation index by filename and SHA-256 digest and is atomically replaced only
after that index, its descriptor shards, and every tensor shard are durable.
`index.json` is a non-authoritative convenience pointer; reopening follows
`run.json` even when the convenience pointer is missing or interrupted.

Successful overwrites retain earlier immutable generation indexes, descriptors,
and tensor shards. This is required because an `EmbeddingResult` opened before
the overwrite resolves lazy tensors through the earlier paths. FastPLMs does not
guess when those readers are released. Preview stale generations. Then collect
them only after you confirm that no reader or writer for the output is active:

```python
from fastplms.embeddings import garbage_collect_safetensors_generations

stale = garbage_collect_safetensors_generations("output")  # dry run
garbage_collect_safetensors_generations(
    "output",
    dry_run=False,
    confirm_no_active_readers_or_writers=True,
)
```

Destructive collection invalidates any older `EmbeddingResult`,
`EmbeddingRecord`, or `LazyTensorReference` that still names a collected shard.
It also removes abandoned generation files from interrupted writers. Never run
it concurrently with embedding, overwrite, resume, or result retrieval.

## SQLite streaming, retrieval, and resume

Use `format="sqlite"` when a long run should commit each batch:

```python
result = model.embed_dataset(
    inputs,
    batch_size=16,
    output="embeddings.sqlite",
    format="sqlite",
    resume=True,
)
```

Tensor payloads store raw bytes and an explicit dtype, so BF16 is lossless.
Each completed batch window is committed transactionally. Resume is allowed
only when the full run fingerprint matches and existing records form the exact
ordered prefix of the request.

SQLite keeps runs under their full fingerprint. With `resume=False`, a new or
restarted run becomes the default result as soon as its first batch commits;
other fingerprints remain available through `run_id`. An interrupted overwrite
therefore exposes a resumable incomplete prefix while retaining the previous
complete run. This is batch-transactional behavior, not the full-run atomic
replacement provided by safetensors generations.

Reopening uses SQLite read-only mode. Filtered retrieval accepts exactly one
ordered selector and preserves request order and duplicates:

```python
from fastplms.embeddings import load_sqlite_result

selected = load_sqlite_result(
    "embeddings.sqlite",
    record_ids=["protein-b", "protein-a", "protein-b"],
)
print([record.id for record in selected])
```

Selectors are `positions`, `record_ids`, or `sequences`; `run_id` may select a
specific compatible run. A writable connection is never opened by the result
reader.

Convert an older FastPLMs SQLite database once, then use the current read-only
reader:

```python
from fastplms.embeddings import convert_legacy_sqlite

convert_legacy_sqlite("legacy.sqlite", "embeddings-v1.sqlite")
```

Compact and weights-only tensor blobs convert without pickle. An unsupported
pickle payload is rejected unless `allow_unsafe_pickle=True` is explicitly set
for a trusted source.

## Run metadata

Persisted results include:

- model ID, immutable revision, checkpoint hash, and package versions;
- Torch and Transformers versions, backend/device policy, checkpoint identity,
  and adapter identity;
- tensor dtype and resolved attention backend. A model loaded with
  `attn_implementation="auto"` settles that request before the run is
  fingerprinted, so the record names the implementation that executed and
  never `auto`;
- selected layer or projection;
- tokenizer and biological-residue policy;
- pooling names and output slices;
- truncation settings;
- input and complete-run fingerprints;
- fingerprint schema version and exact model-state fingerprint;
- generation-indexed output tensor shapes and SHA-256 hashes.

When a model is loaded from `dist/hub/<model>`, Transformers does not assign a
Hub commit to `config._commit_hash`. The artifact therefore carries
packaging-only model ID, checkpoint repository, immutable revision, and
checkpoint-identity hash fields. Embedding metadata and resume fingerprints use
those fields as the fallback, so local offline runs retain complete traceability.
The packaging fields are excluded from semantic configuration parity.

Run-fingerprint schema v4 binds pooling semantics and the current bytes, names, dtypes, and shapes of
each model parameter and persistent buffer. State tensors are copied to CPU in
bounded chunks. The digest is recomputed from authoritative bytes for each
persisted run. FastPLMs does not trust object identity, autograd version
counters, or cached state digests. A mutation through `Parameter.data` or
another storage alias changes the model-state digest and resume identity.
Changing any material input, model state, or setting prevents resume into an
incompatible output. Results from older fingerprint schemas cannot resume.

The tokenizer identity hashes the vocabulary, special tokens, and backend
configuration, but not the padding and truncation that the previous encode call
left on the backend. Transformers sets both from each call's own arguments, and
the run records its own truncation settings. Identical runs in one process
therefore share one fingerprint, and a tokenizer that was never called hashes
exactly as it did before this rule.

Models with meta-device tensors, custom offloading, or an externally managed
state identity may pass the keyword-only `model_state_fingerprint` override.
The caller is responsible for changing this value whenever the effective model
state changes; metadata records whether the identity was computed or supplied
by the caller.

## Legacy `.pth` files

FastPLMs never writes pickle-based `.pth` embeddings. A read-only importer is
available for existing files only when the caller explicitly enables unsafe
pickle loading. Treat such files as executable input and use the opt-in only for
trusted data. Convert imported records to safetensors or SQLite immediately.
