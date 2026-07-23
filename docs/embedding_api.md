# Embedding API

## Install and platform requirements

The shared sequence embedding API requires Python 3.11-3.14, PyTorch 2.13, and
Transformers 5.13. Install an immutable FastPLMs runtime revision before loading
a published checkpoint:

```bash
python -m pip install \
  "fastplms @ git+https://github.com/Synthyra/FastPLMs.git@<runtime-revision>"
```

Core tokenizer-mode embeddings run on CPU or CUDA. E1 uses its raw-sequence
adapter rather than a tokenizer. Structure models and optional FlashAttention
backends require the extras and CUDA platforms declared in the support matrix.

## Quick start

FastPLMs exposes the same dataset operation as `fastplms.embed_dataset(model,
...)` and `model.embed_dataset(...)`. This minimal example loads a tokenizer-
based model and returns one mean-pooled vector per sequence:

```python
from transformers import AutoModel

from fastplms import EmbeddingInput, embed_dataset

model = AutoModel.from_pretrained(
    "Synthyra/ESM2-150M",
    trust_remote_code=True,
).eval()
result = embed_dataset(
    model,
    [
        EmbeddingInput("protein-a", "MSTNPKPQRKTKRNT"),
        EmbeddingInput("protein-b", "MKTIIALSYIFCLVFA"),
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
| `output` | Safetensors directory or SQLite file; omit for in-memory results |
| `format` | `safetensors` or `sqlite` when `output` is set |
| `resume` | Reuse an exact compatible ordered prefix when persistent output exists |
| `tokenizer` | Explicit tokenizer override for a compatible tokenizer-mode family |
| `max_length` | Optional maximum number of biological residues, excluding tokenizer-added special tokens |
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

FASTA input is read line by line into an immutable, incrementally fingerprinted
spool. The runner never reads the complete FASTA file into memory. It keeps only
one bounded `batch_window_size` group eligible for length bucketing, applies
`max_tokens_per_batch` to the padded biological-residue count, and then restores
exact source order. Omitting the window size resolves it to sixteen times the
batch size; an explicit value always wins. The resolved window is included in
the result metadata and resume fingerprint. Result descriptors are stored once;
tensor payloads remain lazy for persistent outputs.

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

```python
result = model.embed_dataset(
    inputs,
    batch_size=8,
    pooling=("mean", "max", "std"),
)
print(result.metadata["pool_slices"])
```

Choose poolers based on the downstream object. `mean` is a stable sequence
summary, `max` highlights large per-feature responses, and `std` or `var`
captures within-sequence dispersion. Concatenating poolers increases output
width and should be treated as a feature-design decision rather than a free
accuracy improvement.

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

The currently published immutable Synthyra ANKH revisions are legacy
encoder-only checkpoints. Decoder examples require either a validated local
full 1.0 artifact or a new immutable Hub revision published after the atomic
replacement; the existing Hub revisions must not be used for this path.

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
and tensor shards. This is required for correctness: an `EmbeddingResult` opened
before the overwrite still resolves its lazy tensors through the earlier paths.
FastPLMs never guesses when those readers have been released. Preview and then
explicitly collect stale generations only after guaranteeing that no reader or
writer for the output remains active:

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
- tensor dtype and resolved attention backend;
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
those fields as the fallback, so local offline runs retain complete provenance.
The packaging fields are excluded from semantic configuration parity.

Run-fingerprint schema v3 binds the exact current bytes, names, dtypes, and
shapes of every model parameter and persistent buffer. State tensors are copied
to CPU in bounded chunks rather than duplicating the complete model. The digest
is recomputed from authoritative bytes for every persisted run; object identity,
autograd version counters, and cached state digests are never trusted. Mutations
through `Parameter.data` or another storage alias therefore change both the
model-state digest and resume identity. Changing any material input, model
state, or setting prevents resume into an incompatible output. Results written
by older fingerprint schemas cannot be resumed.

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
