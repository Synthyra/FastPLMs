---
id: foundry.serialization
kind: library_module
status: active
aliases: [foundry.serialization]
---
# foundry.serialization

Dictionaries on disk as JSON, pickle, or PyTorch files, gzipped when the path ends in `.gz`
(through [`foundry.compression`](../compression/README.md)). Every reader and writer asserts that
the path carries its format's suffix and that the payload is a dictionary, and a writer creates
the parent directory.

| Name | Does |
|---|---|
| `read_json_dict`, `write_json_dict(path, payload, allow_nan=True)` | `.json` or `.json.gz`, indented by 2 |
| `write_json(path, payload)` | Strict JSON: `normalize_json_payload` first, then `allow_nan=False` |
| `normalize_json_payload(payload)` | NumPy arrays and scalars as Python values, tuples as lists, NaN and infinity as None |
| `read_pickle_dict`, `write_pickle_dict(path, payload, compresslevel=None)` | `.pkl` or `.pkl.gz`, at the highest pickle protocol |
| `read_torch_dict(path, map_location=None, weights_only=None)`, `write_torch_dict` | `.pt` or `.pth`, through `torch.load` and `torch.save` |
| `write_json_dict_atomic`, `write_pickle_dict_atomic` | Write a temporary sibling, `<name>.tmp` or `<stem>.tmp.gz`, then `os.replace` it onto the target, so a reader never sees a half-written file |

## Atomic writes of bytes, text and JSON

In `foundry.serialization.atomic`, exported by the package, standard library only. A reader sees
the old file or the whole new one, never part of one.

| Name | Does |
|---|---|
| `atomic_replace(path, *, durable=True)` | A context manager that yields a hidden temporary sibling, `.<name>.<pid>.<uuid>.tmp`, in the target's directory for any writer that takes a path (Parquet, `np.save`, `torch.save`, safetensors, `csv`). On a clean exit it flushes the file to disk and renames it over `path`; if the block raises, or the rename fails, it removes the temporary file and leaves `path` as it was. The parent directory is created |
| `write_bytes_atomic(path, content, *, durable=True)` | `atomic_replace` around `write_bytes` |
| `write_text_atomic(path, text, *, encoding="utf-8", durable=True)` | The text encoded and written as given, so `"\n"` stays one byte on Windows, where `Path.write_text` and a text-mode `open` write `"\r\n"`. Text that does not encode raises before any file exists |
| `write_json_atomic(path, payload, *, sort_keys=True, allow_nan=True, ensure_ascii=True, default=None, durable=True)` | `json.dumps` indented by 2, keys sorted, ending in one newline. The payload is serialized first, so one that does not serialize raises and leaves `path` as it was |

`durable=False` skips the `fsync` for a file rewritten often enough that its durability is not worth the
wait. The directory entry is not flushed. On Windows a rename that Dropbox or an antivirus scan refuses
with `PermissionError` is retried up to 20 times, 0.25 seconds apart, and the last error propagates.

`write_json_atomic` is not `write_json_dict_atomic`, which also writes `.gz`, takes only a dictionary,
keeps insertion order, ends no line, and writes `<name>.tmp`. Both stay: the older pair serves the
caches that read through `read_json_dict`.

Local helpers of this shape lived in ten entities (embedding_translation, sequence_gradient, phenomics,
dataset_builders, serving_api, production_suite, projepa, annotation_vocabulary, atlas, fastplms), which
differed in the guarantees above: a fixed `.tmp` name or a unique one, an `fsync` or none, cleanup on failure or none,
and a text-mode write that turns `"\n"` into `"\r\n"` on Windows. This is the strongest of them.
Parquet is not a function here: `atomic_replace` covers it, and the two copies that wrote Parquet took
rows in different shapes. Tests: `tests/tier1_unit/test_serialization_atomic.py`, which pins each
guarantee above and needs NumPy only for the package import.

## History and tests

Promoted on 2026-09-24 from the `artifacts.py` that DatasetDev, Atlas, and synth each held,
which differed only in comments and synth's extraction of the temporary path. Each is now a
module that imports these names from here. The private `_normalize_json_payload` is public here
as `normalize_json_payload`; only Atlas's tests used it. The atomic byte, text and JSON writers
were added on 2026-10-05.

Tests: `tests/tier1_unit/test_serialization.py`, with Atlas's `tests/test_utils_artifacts.py` as
the guard on the names it imports. Both need NumPy and PyTorch, so the tooling environment skips
them.
