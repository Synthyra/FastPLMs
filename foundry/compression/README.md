---
id: foundry.compression
kind: library_module
status: active
aliases: [foundry.compression]
---
# foundry.compression

Gzip-transparent file openers, and a zip archive of a directory. A path ending in `.gz`, in any
letter case, opens through `gzip` and any other path opens directly, so a cache reads the same way
however it was stored.

| Name | Does |
|---|---|
| `open_text_maybe_gzip(path, mode="rt", encoding=None, newline=None)` | Opens in a text mode, through `gzip` for a `.gz` path. `newline` is `open`'s: `"\n"` writes line feeds on Windows too. A mode without `t`, or with `b`, fails an assertion |
| `open_binary_maybe_gzip(path, mode="rb", compresslevel=None)` | Opens in a binary mode. A gzip write uses `compresslevel`, or `GZIP_COMPRESSLEVEL` (1) when it is None |
| `is_gzip_path(path)` | Whether the path ends in `.gz` |
| `zip_directory(output_dir, archive_suffix="_archive")` | Writes `<output_dir><archive_suffix>.zip` beside the directory and returns its path |

Promoted on 2026-09-24 from the `zip.py` that DatasetDev, Atlas, and synth each held. The
three copies differed only in imports and comments, and Atlas's lacked `zip_directory`. Each is
now a module that imports these names from here, so `dataset_dev_utils.zip`, `utils.zip`, and
`synth.utilities.zip` keep working.

Tests: `tests/tier1_unit/test_compression.py`, with Atlas's `tests/test_utils_zip.py` as the
guard on the names it imports.
