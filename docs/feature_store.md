# Feature store

One storage format for every embedding a project keeps, so a sequence is embedded
once per feature and every head trained afterwards reads the same numbers.

A feature is one value per sequence: a pooled vector, a per-residue block, or a
sparse-autoencoder code vector. A store answers two questions, and only these:
what is this feature of this sequence, and which of these sequences is it
missing?

```python
import torch

from fastplms.embeddings import HiddenTap, embed_into_features
from fastplms.features import DENSE, RAGGED, StoredFeature, open_feature

pooled = StoredFeature(key="esmc600__a1b2c3d4__l30__mean__float32__maxlen-none__0f1e2d3c4b5a6978",
                       layout=DENSE, width=1152, dtype=torch.float32)
residues = StoredFeature(key="esmc600__a1b2c3d4__l30__residues__bfloat16__maxlen-none__89abcdef01234567",
                         layout=RAGGED, width=1152, dtype=torch.bfloat16)

receipts = embed_into_features(
    model,
    sequences,
    "/mnt/features",                      # the store root: NVMe or a Modal volume, never Dropbox
    {"mean": pooled, "residues": residues},
    taps=[HiddenTap("mean", layer=-1, pooling="mean"), HiddenTap("residues", layer=-1)],
    batch_size=8,
)

store = open_feature("/mnt/features", pooled)
vectors = store.read(sequences)           # one (1152,) tensor per sequence, in the order asked
```

Calling `embed_into_features` again with the same sequences runs no model and
returns `{}`. Calling it with more sequences embeds only the new ones. A store
that lags the others, because it was added later, catches up on the next call
without re-embedding what the others already hold.

Output tensors are consumed one `batch_window_size` window at a time through the same embedding
engine, then released before the next window. `max_part_bytes` defaults to 256 MiB and caps the
encoded tensor payload of each part. Safetensors headers and compressed identity sidecars are
additional; this is not a cap on whole-process memory. A single row cannot span parts and fails
if it exceeds the explicit budget. Use a finite residue limit and bounded window size as well.
The canonical sequence inventory, membership sets and segment commit metadata still scale with
the number of inputs. No part becomes visible until the complete segment passes its contract
checks and commits. All features share the same extraction windows and encoder calls.

## The key

`foundry.embedding.feature_key` composes the model, its pinned revision, the
sparse autoencoder and its revision, the layer, the pooling, the dtype, and the
residue limit into one filename-safe name, and `feature_key(...).name` is the
`key` above. FastPLMs never invents a key: it stores the name and the descriptor
it is given, and refuses a second, different feature under a name it already
holds, because that would serve one project another project's numbers.

Only the caller knows the pinned revisions, which is why the key belongs to the
caller. FastPLMs ships without `foundry`, so the store takes the name as a
string.

Canonical suite work uses the full `foundry.embedding.FeatureSpec` contract instead of the
smaller historical `feature_key`. Pass `FeatureExtractionContract` as `contract=` to
`embed_into_features`, with its `stored_features` and exact `embedding_options`. It validates
the original sequence inventory, all measured dependencies and physical descriptors, and
recaptures them before each segment commits. FastPLMs accepts this through the caller-owned
`FeatureRunContract` protocol without importing foundry. A `feature_spec_v1` descriptor requires
a contract; it cannot silently use the legacy path. The caller's `foundry.embedding` package,
which is not part of this repository, documents the enforced contract under "Enforced persistence".

## Layout on disk

```text
<root>/<key>/
    feature.json                      the descriptor, layout, width, and dtype
    index.sqlite                      sequence sha256 -> segment, part, row, residue count
    .locks/                           stable process-lock files, never removed by sweep; a segment's
                                      lock is named by 128 bits of the fingerprint's SHA-256, which
                                      keeps the path short enough for Windows' 260-character limit
    segments/<run fingerprint>/
        part-00000.safetensors        immutable, memory-mappable, one per append
        part-00000.rows.json.gz        optional opaque identity records; required for canonical rows
        part-00001.safetensors
        run.json                      the commit marker, written last
```

**A segment is immutable and committed once.** Parts appear as a run streams, and
nothing reads them until `run.json` names them. A run that dies leaves a
directory the index ignores and `store.sweep()` removes, so a reader never sees a
half-written feature. Live writers own kernel locks, so sweep skips them. A new attempt with
the same fingerprint discards an inactive, uncommitted directory while holding its exclusive
lock. A committed directory is never overwritten. The segment is named
by the run fingerprint `embed_dataset` computes, which binds the model state, the
tap plan, every reducer identity, and the execution context.

**The index is a cache of the commit markers, not the record.** Every indexed row
can be rebuilt from the committed segments, which `store.reindex()` does and
which opening a store does for missing rows. Opening, reindexing and committing verify all
committed parts and sidecars before recovering index entries. A missing or structurally corrupt
`index.sqlite` is rebuilt from verified segments. A valid database containing changed row
addresses fails on open: inspect the discrepancy and explicitly call `reindex()` to repair it.
A malformed marker, missing part, changed checksum or duplicate committed row fails the whole
recovery, leaving the current index unchanged. No segment wins by filename or modification time.

**A sequence is identified by the SHA-256 of its exact UTF-8 bytes.** Two callers
agree without coordinating, and a sequence that differs by one residue is a
different row. The store keeps the digest and the residue count, never the
sequence text.

## Row layouts

| Layout | Tensors | For |
| --- | --- | --- |
| `dense` | `values (n, w)` | pooled vectors |
| `csr` | `indptr (n+1)`, `indices (nnz)`, `values (nnz)`, optional `positions (nnz)` | max-pooled sparse-autoencoder codes |
| `ragged` | `offsets (n+1)`, `values (sum r_i, d)` | per-residue hidden states |
| `ragged_topk` | `offsets (n+1)`, `indices (sum r_i, k)`, `values (sum r_i, k)` | per-residue sparse-autoencoder codes |

Rows are addressed individually and never sliced by column, so
compressed-sparse-row is the right sparse layout: a batch materializes with one
gather and no search. Index dtypes follow the measured store this layout came
from: `int32` code indices, `int16` argmax residues, `int64` row offsets, so an
entry costs eight bytes against four bytes per column dense.

`store.read` gives one tensor per sequence: `(w,)` for dense, `(r_i, d)` for
ragged, and a densified `(w,)` for csr. `store.read_sparse` keeps a csr row
compressed. Reading a sequence the store lacks raises, because a silent zero row
is indistinguishable from a real one; call `store.missing` first.

`ragged_topk` retains each residue's k integer code indices and floating values, including
zero-valued entries and the encoder's order. `width` is the codebook size; `sparse_count` is k.
Only this layout declares `sparse_count`, leaving existing descriptors unchanged. The index
records biological residue counts, and offsets address sequences, not individual codes.
Indices are int32, offsets int64, and values keep the declared floating dtype. A part's
tensor payload is `8 * (n + 1) + sum(r_i) * k * (4 + value_bytes)`. Stored indices must be
unique within each residue and within the codebook; both writes and reads check these rules,
shapes, offsets, finiteness, checksums and independently supplied content pins when present.

Use `store.read_topk(sequences)` for this layout. It returns `TopKRow(indices, values)` objects,
each holding `(r_i, k)` tensors. The ordinary `read` refuses implicit residue-by-codebook
densification. A zero-length physical row is representable, but a canonical biological
extraction must retain at least one residue and match its independently bound row identity.
This is a separate layout from pooled CSR `positions`, which only name argmax residues and
cannot reconstruct the codes of a crop.

`foundry.embedding.sae_residue_tap` fills this layout through `SparseResidueTap` in the existing
one-pass engine. Pass a `FeatureExtractionContract` as for the other canonical features.
The reducer selects only biological residues, uses the existing chunked SAE encoder, and
preserves its value dtype. Sparse and pooled SAE taps may share an encoder pass, though they
each perform their requested SAE reduction. No second feature store or backbone is introduced.

`positions` carries the argmax residue of each code, which is what makes a code
interpretable as "here". No tap produces them, so `embed_into_features` refuses a
feature that keeps them; a pipeline that computes them itself writes through
`store.segment(...)` directly.

## Writing without a model

A pipeline that produces features another way opens a segment itself:

```python
with store.segment("my-run-fingerprint", {"machine": "gh200"}) as writer:
    writer.append_bounded(batch_sequences, batch_rows, max_tensor_bytes=256 * 1024**2)
```

A pipeline that embeds a window of sequences at a time, and wants a run that dies to lose at most
that window, commits each window with `write_rows`:

```python
from fastplms.features import write_rows

absent = set(store.missing(window))                   # a restart repeats a window: skip what is committed
write_rows(store, {sequence: row for sequence, row in zip(window, rows) if sequence in absent},
           metadata={"chunk": [start, stop]})
```

`write_rows(store, rows, metadata=)` takes a mapping from sequence to row (a tensor for dense and
ragged, a `SparseRow` for csr, a `TopKRow` for ragged top-k), writes it as one bounded segment, and
names the segment by the digest of the sequences it holds. It returns how many rows it wrote, and
writes nothing for an empty mapping. A sequence the store already holds raises, so call
`store.missing` first.

The segment commits on a clean exit, abandons itself if it wrote nothing, and
stays uncommitted if the body raises. Writing a sequence the store already holds
is an error: a feature has one row.
`append` remains available for a caller that already bounds a single part. `append_bounded`
splits a supplied window without changing row order or values; each part records its measured
`tensor_bytes`. Neither method accepts further writes after the segment is committed. Sparse
indices must be unique integers, and indices/argmax positions must fit the stored integer range.
Oversized or fractional indices and positions fail instead of silently wrapping or truncating.

For a complete canonical descriptor, `store.segment(..., before_commit=callback)` also requires
`writer.append(..., row_metadata=records)`. The callback must validate the live scientific
contract; the normal extraction API supplies it. Each part records SHA-256 digests of the
safetensors file and any row-identity sidecar. The writer rechecks staged file digests and the
canonical descriptor before calling the callback and publishing its commit marker.

`store.row_metadata(sequences, verify_data=True)` returns the opaque identity dictionaries in
request order, including repeated queries. It checks sequence/index/marker associations and
metadata/data digests, and fails on missing or corrupt evidence. Foundry validates the record's
scientific schema and original inventory. `store.read` alone does not perform that validation.
Legacy rows without sidecars remain legacy rows; recompute them for canonical use.

`FeatureStore.read_only(directory)` now uses an escaped SQLite URI with `mode=ro` and refuses
segment creation, reindexing, sweeping and index initialization. It neither creates a missing
index nor repairs a corrupt one. Handles store absolute paths and descriptors, hold no open
SQLite connection between calls, and can be sent to spawned workers. Worker reads require only
read permissions. This supplies the physical handle; each training adapter still owns scientific
contract validation, collation, prefetch limits and corpus-memory qualification.

## Reading a batch at a time

`store.read` is the verified one-shot read. Each call opens a fresh index connection, hashes every
part it touches, and loads those parts whole, which is right for a caller that reads once and
wrong for a training loop that reads a batch per step. `FeatureReader` gives the same rows, in the
same types, at random-access cost:

```python
from fastplms.features import FeatureReader

with FeatureReader.open(store.directory) as reader:
    reader.verify()                        # optional: pay for verification before forking workers
    batch = reader.read(sequences[:64])    # (w,) rows for dense, (r_i, d) for ragged, as the store
    sparse = reader.read_sparse(...)       # csr rows, compressed
    block = reader.read_csr(...)           # csr rows as one CsrRows block: indptr, indices, values
    codes = reader.read_topk(...)          # ragged_topk rows
```

`read_csr` is the read for a caller that wants a matrix, such as the design matrix of a
gradient-boosted model. `CsrRows` holds `indptr (n + 1,)` int64 and `indices`, `values`, and
optional `positions`, each `(nnz,)`, with a sequence asked twice appearing twice. Wrap the arrays
in whatever sparse type the caller uses; the reader imports no array library.

A part is verified the first time a row from it is read, by the store's own checks (checksum
against the commit marker, physical layout, identity sidecar, content pins), and then served by
memory-mapped slices. There is one read-only index connection per thread and process, so threads
may share a reader, and a reader pickles as the store it reads, so a spawned worker verifies only
the parts it touches. A sequence the feature lacks raises `KeyError`, as it does for the store.
This is the reader for plain features. A canonical `feature_spec_v1` feature is still read
through `foundry.embedding.FeatureView`, which adds the contract and the identity checks.

## Canonical token stores

Since 2026-10-01, a canonical sequence of `l` residues, cropped from the N-terminus at 2,046, has `l + 2` rows in every
per-token stream. Row 0 is CLS, rows 1 to `l` are the residues, row `l + 1` is EOS. The special tokens
act as attention sinks and carry information of their own, so a model trained on a store sees them.
Padding is the only thing masked. A legacy store (`feature_spec_v1`) holds `l` residue rows; the two
never share a key, because a v2 descriptor carries `"special_tokens": "kept"` and hashes differently.

```text
(b, l+2, d) hidden state   row 0 CLS | rows 1..l residues | row l+1 EOS      legacy v1: (b, l, d)
(b, d)      pooled         mean or variance over all l+2 rows                 legacy v1: over l rows
(b, c)      SAE maximum    max over all l+2 rows of the Biohub top-k codes    c = 16384 codebook
```

Per real base (`esmc_small` d=960 layer 23, `esmc_large` d=1152 layer 27, `esmc_6b` d=2560 layer 60),
each stream is its own feature store, named by its tap:

| Stream | Layout | Per sequence |
|---|---|---|
| `layer_hidden` | `ragged`, bf16 | `(l+2, d)` hidden state at the SAE layer |
| `final_hidden` | `ragged`, bf16 | `(l+2, d)` final normalized state |
| `structural` | `ragged`, bf16 | `(l+2, 256)` ESMFold2 projection, defined on CLS and EOS because it is per token |
| `sae_codes` | `ragged_topk`, fp32 | `(l+2, 64)` indices (int32) and values after the Biohub encode |
| `sae_max` | `csr`, fp32 | `(16384,)` max over all `l+2` rows, stored sparse |
| `final_mean_var` | `dense`, fp32 | `(2d,)` mean then population variance of `final_hidden` over `l+2` rows |

`esmc_small_random` (a random-init ESMC-300 architecture with a recorded seed) holds `layer_hidden`,
`final_hidden` and `final_mean_var`, and no SAE or structural stream.

A v2 row identity is `feature_row_v2`: the feature digest, the sequence key, `context_length` (`l`)
and `retained_positions: {start: 0, stop: l + 2}`, never a list of positions.

**The row key is one rule, made in one place.** A row is keyed by the SHA-256 of the protein's
*normalized* text: ASCII whitespace removed and letters uppercased, nothing else rewritten
(`foundry.embedding.contracts.normalize_sequence`, the `protein_ascii_v1` rule and the one the dataset
census keys follow). `sequence_inventory`, the reader and the job all call it, and the engine refuses any
other spelling before a forward pass (`check_canonical_text`), so "acd", "ACD " and "A C D" are one row, never
two. A store only appends by that key. The legacy feature-store functions (`sequence_digest`,
`FeatureStore.missing`, every `feature_spec_v1` store) still hash the *exact* UTF-8 text they are given:
legacy callers pass normalized text already, and changing that function would re-key the legacy stores.

**Reading.** `foundry.embedding.open_canonical_view(root, inventory)` opens a base's streams as a
`FeatureView` (`inventory` from `sequence_inventory`, which also normalizes). `lengths` counts residues `l`,
never CLS or EOS. `residues` returns `l + 2` rows per sequence, padded to the longest, with a mask that
is false only on padding. A crop is a half-open residue interval `[start, stop)` and keeps the CLS and EOS
rows around the cropped residues, so a crop of `n` residues returns `n + 2` rows. A request may use a raw
spelling; the view answers it with the normalized row. Random-vector and one-hot controls are not stored:
`foundry.embedding.SyntheticControlView` generates them with the same layout, seeded by the row's
SHA-256, and pools them like a base.

**Writing.** `embed_token_features` (and `embed_into_features(..., keep_special_tokens=True)`, which routes
to it) is the canonical path. A contract captured with the special tokens kept carries that choice
(`contract.keep_special_tokens`), so `embed_into_features(..., contract=contract, **contract.embedding_options)`
takes this path without the argument, and a residue-only contract refuses `keep_special_tokens=True`. It checks every text, hashes each once, and plans batches by padded-token
budget, longest first inside windows. `TokenTapExecutor` builds each batch from host-known lengths (the
attention mask, row selection and offsets are host arrays moved in one pinned copy), so it never reads a
device value; one device flag per batch carries the finite check and is read after the outputs land. Outputs
copy to pinned host buffers without blocking and join a bounded queue (`queue_bytes`). `AsyncFeatureWriter`
packs rows into parts of `part_bytes` (default 1 GiB), writes and hashes each part in one pass on a pool
thread (`append_packed`), commits every stream together after `segment_bytes`, and opens stores with
`deep_verify=False`, so an open or a commit never re-reads a part already stored. A killed run commits
whole segments only; a rerun embeds the rows a stream still lacks. Costs per part: one flush for the part
file, one for its row-identity sidecar, and one directory flush per segment (the legacy writer did these
per window, and re-read every committed part at each open and commit). `partition_sequences` shards a
corpus; the `foundry.embedding.job` command wraps all of this, with `--shard i/n`.


## Converting a cache another format holds

`convert_rows` keeps an existing cache by moving it into a store, and proves the move:

```python
from fastplms.features import convert_rows, describe_file

receipt = convert_rows(
    store,
    lambda: decode_old_cache(path),        # yields (sequence, row); called twice, so no arguments
    origin={"format": "my_sqlite_v1", "decoder": "my_project.decode_old_cache",
            "files": [describe_file(path)]},
)
```

The decoder yields tensors for `dense` and `ragged` features, `SparseRow` for `csr`, and `TopKRow`
for `ragged_topk`. Rows go through the ordinary segment writer, then the source is decoded a
second time and every row is compared, bit for bit on the stored representation, with a fresh
`FeatureReader` read of the committed store. Each of these raises `ConversionMismatch`: a value
the store's dtype cannot hold exactly, a sequence absent after the commit, a row that differs, a
sequence repeated with different rows, and a source that decodes differently the second time.
Rows the store held beforehand are compared too, so a store that mixes converted rows with rows
embedded afresh fails, since the old numbers are not the store's numbers.

`origin` is plain data, recorded in every segment's commit marker, and its digest names the
conversion: calling again with the same origin resumes an interrupted one. Segments commit as they
fill (`rows_per_segment`), committed segments are kept, and rows already in the store are not
written twice. The old cache is never modified or deleted. Undo a conversion by removing the
segments the receipt names and calling `store.reindex()`.

## Transactions and restart behavior

Keep the segment context open for the entire write. A writer cannot be used after context exit,
after an I/O failure, or after commit. An inactive attempt's parts are discarded before retry;
there is no partial-part resume. Duplicate sequences are refused within a part, across a writer's
parts, and across committed segments. Two writers may stage disjoint segments concurrently.
Writers belong to the process that opens them; a spawned or forked copy cannot mutate one.
Only one process can own a fingerprint; a competing owner fails immediately. Sweep acquires
that same lock before deleting uncommitted files and leaves live writers alone.

A short store-wide lock serializes descriptor creation, commit and index recovery. Commit
checks existing durable segments, begins a SQLite transaction, and inserts with a uniqueness
constraint, never replacement. Conflicting staged writers fail before publishing a marker.
Index changes remain invisible until after the marker is durable. Death before the marker
leaves no accepted rows; death after it allows a fresh writable open to reconstruct the index.
The marker records a descriptor digest and its own canonical JSON digest (transaction schema 2).
Older markers with part digests still receive physical validation; markers without part digests
must be recomputed. These checks detect corruption, not a malicious party rewriting every digest.

Files are flushed before atomic rename, and each part and sidecar then leaves the page cache
(`flush_and_evict`: `fsync`, then `posix_fadvise` with `DONTNEED` where the platform has it).
On a GH200 written pages would otherwise fill the HBM node and make `nvidia-smi` read the GPU as
full. POSIX directory entries are flushed afterward. The
protocol follows [SQLite's transaction and flush assumptions](https://www.sqlite.org/atomiccommit.html)
and uses [kernel file locks](https://docs.python.org/3/library/fcntl.html). Lock files are never
unlinked, because that could create two independent locks for the same logical store.
Independent processes on the GH200's local filesystem are the qualification target. Power-loss
hardware behavior, Windows directory durability, distributed filesystem locking, and multiple
Modal containers writing the same volume are not established by process-death tests. Use one
writer host/container per store; separate roots plus a single merge host for distributed shards.

`partition_sequences(inputs, shard=i, shards=n)` assigns exact unique sequences by SHA-256
modulo `n`, preserving their input order. Supply the same inventory and shard count everywhere.
Copy only completed, verified segments between stores with the identical descriptor, then open
or reindex at the merge destination. Disjoint segments recover; overlapping rows are refused,
even if their tensor values agree. Never copy a shard's index over another store's index.

All features commit independently. If a multi-feature extraction stops after one feature commits,
the next extraction validates and reuses it, then fills the remaining features. A consumer must
require every feature it needs. Physical `read`, `read_sparse` and `residue_counts` validate the
requested committed parts and their index associations; the caller still validates scientific
row identities. This verification reads file contents and costs I/O. Input/part metadata remains
linear in corpus size, and training-reader performance remains separately qualified.

## Where stores live

On workstation NVMe for SSH runs and on a Modal volume for Modal runs, never in
Dropbox. Reusable sets, such as reference proteomes and benchmark sets, go to
private Hub datasets.

A partial copy holds only some committed parts of each segment, as a fetch of a few proteins from a
published store leaves it (`foundry.embedding.hub.fetch_rows`). `FeatureStore(..., partial=True,
deep_verify=False)` indexes the parts present: a row of an absent part reads as missing, a part fetched
later is indexed on the next open, and the copy takes no new segment. Absent parts cannot be verified,
so a partial copy never opens with deep verification.

## Beside a Hub artifact

A process that loads a published model with `trust_remote_code=True` cannot also `import
fastplms.features`. The artifact installs its embedded runtime as `fastplms`, refuses to install once a
different `fastplms` is imported, and ships no `features` package, so after it has installed,
`fastplms.features` is gone, whichever came first. The store imports nothing outside its own package,
so such a process loads it under another module name with
`foundry.embedding.load_private_feature_store()` (a path search that imports nothing, then the
package's own `__init__.py`). Atlas, synth's embedding engine, and confounders do. A process that
imports FastPLMs' real model classes has no artifact to protect and imports `fastplms.features` as
usual, so that `embed_into_features` and the store agree on their classes. Shipping the store inside
the artifact runtime would remove the difference; that is a change to every published repository.
