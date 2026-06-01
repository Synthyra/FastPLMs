# Embedding & Pooling API

The `EmbeddingMixin` class (`fastplms/embedding_mixin.py`) provides a standardized interface for extracting protein representations from tokenizer-based FastPLMs sequence models and E1. ESM3 exposes the same `embed_dataset()` user API through its wrapper.

## Pooler

The `Pooler` class aggregates per-residue representations `(batch, seq_len, hidden_size)` into fixed-size vectors `(batch, hidden_size)`.

### Construction

```python
from fastplms.embedding_mixin import Pooler

pooler = Pooler(pooling_types=["mean", "max"])
```

### Strategies

| Strategy | Key | Description |
|----------|-----|-------------|
| Mean | `"mean"` | Mask-aware average over all residues |
| Max | `"max"` | Element-wise maximum (masked positions zeroed) |
| CLS | `"cls"` | First token's representation |
| L2 Norm | `"norm"` | L2 norm over the sequence dimension |
| Median | `"median"` | Element-wise median (masked positions zeroed) |
| Variance | `"var"` | Variance over non-masked positions, computed correctly via mean-centered squared diffs |
| Std Dev | `"std"` | Square root of variance pooling |
| PageRank | `"parti"` | Experimental: uses `networkx.pagerank` over attention matrices to weight token importance |

### Calling Convention

```python
# emb: (batch, seq_len, hidden_size)
# attention_mask: (batch, seq_len) - 1 for real tokens, 0 for padding
# attentions: (batch, num_layers, seq_len, seq_len) - required only for "parti"
pooled = pooler(emb, attention_mask=attention_mask, attentions=attentions)
# pooled: (batch, num_pooling_types * hidden_size)
```

When multiple strategies are specified, their outputs are concatenated along the last dimension.

### PageRank Pooling (`parti`)

The `parti` strategy:
1. Max-pools attention matrices across all layers to get `(batch, seq_len, seq_len)`
2. Converts each attention matrix to a directed graph via `networkx`
3. Runs PageRank (alpha=0.85, tol=1e-6, max_iter=100) to get per-token importance scores
4. Computes a weighted average of embeddings using importance scores as weights

This requires `output_attentions=True` when calling the model.

---

## EmbeddingMixin

### `embed_dataset()`

The primary entry point for batch embedding.

```python
embeddings = model.embed_dataset(
    sequences=["MALWMRLLPLLALL", "MKTLLILAVVAAALA"],
    batch_size=32,
    pooling_types=["mean"],
    save=True,
    save_path="embeddings.pth",
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sequences` | `List[str]` | `None` | Protein sequences to embed |
| `fasta_path` | `str` | `None` | Path to a FASTA file; sequences are parsed and combined with `sequences` |
| `tokenizer` | `PreTrainedTokenizerBase` | `None` | Tokenizer for tokenizer-mode models. Defaults to `model.tokenizer` when available. Pass `None` for E1 sequence mode |
| `batch_size` | `int` | `2` | Batch size for inference |
| `max_len` | `int` | `512` | Maximum sequence length (longer sequences are truncated if `truncate=True`) |
| `truncate` | `bool` | `True` | Whether to truncate sequences exceeding `max_len` |
| `full_embeddings` | `bool` | `False` | If True, return per-residue embeddings instead of pooled vectors |
| `embed_dtype` | `torch.dtype` | `torch.float32` | Dtype for stored embeddings |
| `pooling_types` | `List[str]` | `["mean"]` | Pooling strategies to apply (concatenated) |
| `num_workers` | `int` | `0` | DataLoader workers (tokenizer mode only) |
| `sql` | `bool` | `False` | Use SQLite storage instead of in-memory dict |
| `sql_db_path` | `str` | `"embeddings.db"` | Path to SQLite database |
| `save` | `bool` | `True` | Save embeddings to `.pth` file |
| `save_path` | `str` | `"embeddings.pth"` | Path to `.pth` output file |
| `hidden_state_index` | `int` | `-1` | Hidden-state tuple index to embed from. `-1` preserves current last-hidden-state behavior |
| `store_all_hidden_states` | `bool` | `False` | Store every hidden state as layer-first token-wise embeddings. Requires `full_embeddings=True` |

At least one of `sequences` or `fasta_path` must be provided. If both are given, the two sources are merged.

#### Return Value

- **In-memory mode** (`sql=False`): Returns `Dict[str, torch.Tensor]` mapping each sequence to its embedding
- **SQLite mode** (`sql=True`): Returns `None`; embeddings are written to the database

#### Deduplication & Resumability

- Sequences are deduplicated before embedding
- Sorted by length (longest first) for efficient padding
- If `save_path` already exists, previously embedded sequences are loaded and only new sequences are processed
- SQLite mode similarly checks which sequences are already in the database

### Two Modes

**Tokenizer mode** (ESM2, ESM++, DPLM, DPLM2):
```python
# The model tokenizer is used by default
embeddings = model.embed_dataset(
    sequences=sequences,
    batch_size=32,
)
```

The mixin builds a `DataLoader` with `build_collator(tokenizer)` and calls `_embed(input_ids, attention_mask)`. Pass `tokenizer=...` only when you need a custom tokenizer wrapper.

**Sequence mode** (E1):
```python
# Pass tokenizer=None
embeddings = model.embed_dataset(
    sequences=sequences,
    tokenizer=None,
    batch_size=32,
)
```

The mixin iterates over chunks and calls `_embed(sequences, return_attention_mask=True)`, which returns `(embeddings, attention_mask)`.

**ESM3 wrapper mode**:
```python
embeddings = model.embed_dataset(
    sequences=sequences,
    batch_size=4,
    pooling_types=["mean", "cls"],
)
```

ESM3 supports pooled `mean`, `cls`, and `max` embeddings, plus residue-wise embeddings with `full_embeddings=True`. SQLite streaming is not enabled for ESM3.

---

## Hidden-State Selection

By default, `embed_dataset()` embeds from the same final representation as before. Pass `hidden_state_index` when you want pooling from a specific hidden-state tuple entry:

```python
embeddings = model.embed_dataset(
    sequences=sequences,
    batch_size=32,
    pooling_types=["mean"],
    hidden_state_index=12,
)
```

To save token-wise embeddings for every hidden state, use `store_all_hidden_states=True` with `full_embeddings=True`:

```python
model.embed_dataset(
    sequences=sequences,
    batch_size=4,
    full_embeddings=True,
    store_all_hidden_states=True,
    sql=True,
    sql_db_path="all_layers.db",
)
```

Saved all-layer tensors are layer-first per sequence:

```python
embeddings["MALWMRLL..."].shape == (num_hidden_states, num_real_tokens, hidden_size)
```

You can later pool a selected layer without rerunning the model:

```python
pooled = model.load_pooled_embeddings_from_db(
    "all_layers.db",
    pooling_types=["mean", "cls"],
    hidden_state_index=12,
)

pooled_pth = model.load_pooled_embeddings_from_pth(
    "all_layers.pth",
    pooling_types=["mean"],
    hidden_state_index=-1,
)
```

`pool_embeddings()` provides the same conversion for an in-memory dictionary returned by `load_embeddings_from_pth()` or `load_embeddings_from_db()`.

---

## Storage Formats

### `.pth` (PyTorch)

A dictionary serialized via `torch.save`:
```python
{
    "MALWMRLLPLLALL": tensor(...),  # (hidden_size,), (seq_len, hidden_size), or (num_hidden_states, seq_len, hidden_size)
    "MKTLLILAVVAAALA": tensor(...),
}
```

Load with:
```python
embeddings = model.load_embeddings_from_pth("embeddings.pth")
```

### SQLite

Schema:
```sql
CREATE TABLE embeddings (
    sequence TEXT PRIMARY KEY,
    embedding BLOB NOT NULL
);
```

- `embedding`: Compact tensor blob containing dtype, rank, shape, and raw bytes

Load with:
```python
# All embeddings
embeddings = model.load_embeddings_from_db("embeddings.db")

# Specific sequences
embeddings = model.load_embeddings_from_db("embeddings.db", sequences=["MALWMRLLPLLALL"])
```

SQLite mode writes asynchronously and commits when the writer queue drains.

---

## FASTA Parsing

The `parse_fasta()` utility reads a FASTA file and returns a list of sequences:

```python
from fastplms.embedding_mixin import parse_fasta

sequences = parse_fasta("proteins.fasta")
```

Multi-line sequences are concatenated. Header lines (starting with `>`) are discarded. Empty lines are skipped.

You can pass a FASTA file directly to `embed_dataset`:

```python
model.embed_dataset(
    fasta_path="proteins.fasta",
    batch_size=64,
    pooling_types=["mean"],
    sql=True,
    sql_db_path="proteins.db",
)
```

---

## Full Embeddings (Per-Residue)

When `full_embeddings=True`, the pooler is bypassed and per-residue embeddings are returned. Padding tokens are stripped using the attention mask:

```python
embeddings = model.embed_dataset(
    sequences=sequences,
    batch_size=32,
    full_embeddings=True,
    save=False,
)
# embeddings["MALWMRLL..."].shape == (num_real_tokens, hidden_size)
```

Each sequence's embedding has shape `(num_real_tokens, hidden_size)` for one hidden state, or `(num_hidden_states, num_real_tokens, hidden_size)` when `store_all_hidden_states=True`. `num_real_tokens` excludes padding according to the model attention mask.
