# Embedding API

FastPLMs exposes the same dataset operation as `fastplms.embed_dataset(model,
...)` and `model.embed_dataset(...)`.

```python
result = embed_dataset(
    model,
    inputs,
    batch_size=2,
    pooling=("mean",),
    full_embeddings=False,
    output=None,
    format="safetensors",
    resume=True,
    model_state_fingerprint=None,
)
```

The operation accepts sequences, `(id, sequence)` pairs, `EmbeddingInput`
values, or a FASTA path. It preserves input order, FASTA identifiers, and
duplicate records.

## Argument reference

| Argument | Meaning |
| --- | --- |
| `inputs` | Sequence iterable, `(id, sequence)` pairs, `EmbeddingInput` values, or FASTA path |
| `batch_size` | Number of records prepared together; must be positive |
| `pooling` | One pooler or an ordered pooler sequence; required unless `full_embeddings=True` |
| `full_embeddings` | Return residue-level tensors instead of pooled vectors |
| `output` | Safetensors directory or SQLite file; omit for in-memory results |
| `format` | `safetensors` or `sqlite` when `output` is set |
| `resume` | Reuse an exact compatible ordered prefix when persistent output exists |
| `tokenizer` | Explicit tokenizer override for a compatible tokenizer-mode family |
| `max_length` | Optional maximum prepared sequence length |
| `truncate` | Truncate to `max_length`; disabling it retains the complete sequence |
| `dtype` | Output tensor dtype; `None` retains the model output dtype |
| `shard_size` | Target safetensors shard size in bytes |
| `model_state_fingerprint` | Caller-supplied state identity for offloaded or externally managed models |
| `**model_kwargs` | Family-specific embedding controls such as hidden-state selection |

`store_all_hidden_states=True` is a model keyword and requires
`full_embeddings=True`. `full_embeddings=True` cannot be combined with an
explicit pooler. Invalid combinations raise before persistence begins.

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

## Safetensors storage

With `format="safetensors"`, `output` names an output directory. FastPLMs writes
transactional temporary files and then publishes:

```text
output/
  embeddings-00001-of-000NN.safetensors
  index.json
  run.json
```

The default shard target is 2 GiB. The JSON index preserves record position,
identifier, sequence, shape, dtype, tensor hash, and shard key. Loading the
result creates lazy references rather than reading every shard into memory.
`run.json` is the transactional commit marker: it stores the complete run
metadata and the SHA-256 digest of `index.json`. Loading and resume reject a
missing or mismatched manifest instead of accepting a partially published run.

## SQLite streaming and resume

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
Each batch is committed transactionally. Resume is allowed only when the full
run fingerprint matches and existing records form the exact ordered prefix of
the request.

## Run metadata

Persisted results include:

- model ID, immutable revision, checkpoint hash, and package versions;
- tensor dtype and resolved attention backend;
- selected layer or projection;
- tokenizer and biological-residue policy;
- pooling names and output slices;
- truncation settings;
- input and complete-run fingerprints;
- fingerprint schema version and exact model-state fingerprint;
- output tensor shapes and SHA-256 hashes.

When a model is loaded from `dist/hub/<model>`, Transformers does not assign a
Hub commit to `config._commit_hash`. The artifact therefore carries
packaging-only model ID, checkpoint repository, immutable revision, and
checkpoint-identity hash fields. Embedding metadata and resume fingerprints use
those fields as the fallback, so local offline runs retain complete provenance.
The packaging fields are excluded from semantic configuration parity.

Run-fingerprint schema v2 binds the exact bytes, names, dtypes, and shapes of
every model parameter and persistent buffer. State tensors are copied to CPU in
bounded chunks rather than duplicating the complete model. Changing any
material input, model state, or setting changes the run fingerprint and
prevents resume into an incompatible output. Results written by older
fingerprint schemas cannot be resumed.

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
