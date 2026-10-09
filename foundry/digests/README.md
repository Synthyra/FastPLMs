---
id: foundry.digests
kind: library_module
status: active
aliases: [foundry.digests]
---
# foundry.digests

The SHA-256 of a file's bytes, read in chunks so a large file never sits in memory whole.

| Name | Does |
|---|---|
| `sha256_file(path, chunk_size=1 << 20)` | The lowercase hexadecimal SHA-256 of the file at `path`, opened as `Path(path)` and read `chunk_size` bytes at a time. A `chunk_size` that is not positive raises `ValueError` before the file is opened |

Promoted on 2026-09-25 from thirty copies that each module or script wrote for itself, and that
each now imports under its own name:

- foundry: `hub_files.digest`.
- embedding_translation: `data.split_provenance.sha256_file`, `helpers.artifact_provenance.file_sha256`,
  `helpers.checkpointing.checkpoint_sha256`, `helpers.control_generation._sha256_file`,
  `helpers.dataset_snapshot._sha256_file`, `helpers.embedding_store._sha256_file`,
  `helpers.evaluation_schema._file_sha256`, `helpers.preprocessing_study._file_sha256`,
  `helpers.run_manifest._file_sha256`, `scripts.prepare_protify_stage2_embeddings._sha256_file`,
  and `scripts.refresh_control_manifests._sha256`.
- annotation_vocabulary: `hub._sha256`, `hub_release._sha256`, and `mmseqs.sha256_file`.
- DatasetDev: latent_sep's `artifacts.sha256_file` and `mmseqs._sha256_file`;
  `archive_manifest._sha256`; `atlasfold_data.archive.file_sha256` and `atlasfold_data.publish._sha256`;
  `de_novo.build_sequence_origin_dataset.sha256_file` and
  `de_novo.rigorous.proteingym_adapter._sha256_file`; the benignity dataset builder's source hash,
  the C3 dataset builder's SHA-256 helper, and the candidate-control builder's source hash.
- base_model_distillation: `esmc_runtime.io.sha256_file`.
- folding_service: `esmfold2_workstation_cli._sha256_file`. The runner stages itself on a
  workstation, so it ships this module there with the rest of the foundry it imports.
- The workspace's tools: `migrate.hash_file` and `research_build.file_digest`.
- synth: the script `scripts.core.build_cross_organism_demo._sha256_path`.

Ten more moved later the same day, once no session held their projects:

- phenomics: `atlas_virtual_cell.provenance.sha256_file`, `_sha256_file` in six `data` modules
  (`chembl`, `download`, `expression`, `jump`, `jump_target`, `proteome`), and `_sha256` in
  `reporting.holdout_results` and `reporting.render`. Each called `path.open`; eight read 8 MiB at a
  time and `render` 1 MiB, and only `provenance` took a size, keyword-only and unchecked. No caller
  passes one. phenomics' README lists them among the files changed since its release manifest.
- sae_xgboost_ppi: `sae_xgb.utils.sha256_file`, which read 1 MiB at a time and did not check its size.
  The stage modules still import it from `utils`.

Called as their callers call them, the copies and this function return the same digest for every
file. Where the copies differed from each other, and so from this one:

- **The path.** For a `Path`, `path.open("rb")`, `Path(path).open("rb")`, and `open(path, "rb")` are
  the same call. Twenty-six copies called `path.open` and refused anything else, a `str` included,
  with `AttributeError`; a `str` or other `os.PathLike` now works. Two, `split_provenance` and
  `esmc_runtime.io`, called `open(path)`, which reads a `str` exactly as written and takes a `bytes`
  path or a file descriptor. Now a `bytes` path or a descriptor raises `TypeError`, and a `str` is
  spelled as `Path` spells it before it is opened: an error names it that way, with backslashes on
  Windows, a trailing separator is dropped, so `"file/"` is the file, and `""` is the current
  directory. None of their callers pass anything but a `Path`.
- **The chunk size.** Each copy with a default read 1 MiB at a time, except
  `control_generation`, `embedding_store`, and the script `prepare_protify_stage2_embeddings`, which
  read 16 MiB, and the tools' `migrate.hash_file`, which read 64 KiB. No caller passes a size, so
  those four now read 1 MiB at a time and return the same digest. The size was keyword-only in
  `embedding_store` and in latent_sep's `artifacts`; it is now positional or keyword. Eighteen copies
  took no size and now accept one.
- **How the file is read.** Three copies, atlasfold's `archive.file_sha256` and `publish._sha256` and
  the tools' `research_build.file_digest`, handed the open file to `hashlib.file_digest`, which reads
  it into a 256 KiB buffer with `readinto`. They now call `read` for 1 MiB at a time, as the others
  did, and return the same digest.
- **A size that is not positive.** `checkpoint_sha256` refused one exactly as this does.
  annotation_vocabulary's `mmseqs.sha256_file` refused a size below 1 with "Hash chunk size must be
  positive"; the message is now "chunk_size must be positive", and a fraction between 0 and 1 now
  reaches `read`, which raises `TypeError`. The other ten copies with a size did not check it: 0
  returned the digest of nothing whatever the file held, a negative size or `None` read the whole
  file at once, and a size that is not a number failed in `read`. Now 0 or a negative size raises
  `ValueError`, and `None` or a size that is not a number raises `TypeError` from the check, each
  before the file is opened, so a missing file with such a size reports the size.
- **The function object.** Each module's name is an alias, so its `__name__`, `__qualname__`,
  `__module__`, and `__doc__` are this module's. `hub_files.CHUNK_BYTES`, which only `digest` and its
  tier 1 test used, is gone; the test now names the 1 MiB default itself.

Three scripts moved, embedding_translation's two and synth's one above. Each already imports
foundry, through its project's modules, wherever it runs, and the tree holds no record of its
bytes. Of the 127 scripts that still hold a copy, the tree records the bytes of 31 in source
snapshots, provenance notes, and build manifests; 50 are in phenomics, contact_esmc6b, and
sequence_gradient, where another session held the project, its scripts, or its README when they
were counted; four sit in
embedding_translation's `results/`, as the source and provenance records of runs; five write their
own digest into what they produce; synth's `ppp_screening_execute.py` hashes its own file, and is
the one copy since folding_service's was retired on 2026-09-26; two
import foundry on one path only, `generate_latent_mapping_figure.py` when it extracts and
`recover_analysis_archive.py` with `--secrets-env`; and 34 import no foundry at all, so moving
theirs would add foundry to wherever they run.

Left as they are: annotation_vocabulary's `study` modules, which hash their own source into the
receipts they write and refuse to go on once those bytes change; DatasetDev's
`de_novo.rigorous.sequence_search._sha256` and the benignity release compiler's file hash, whose
own bytes four DeNovo-Origin-Rigorous search receipts and the benignity pilot's manifest record;
speedrunning_plms' `research.benchmark._sha256`, since the research runner ships a source snapshot
that holds no foundry, and each result names the benchmark by that file's digest; contact_esmc6b's
`contact_fit.py`, an adapter pinned to a frozen source archive; functions that return more than the
digest, such as latent_sep's `sources.compute_checksums`; and FastPLMs' tools, since FastPLMs
vendors no foundry. The unification plan, `docs/decisions/workspace/2026-09-24_foundry_unification.md`,
records each.

## Known issues

Copies that wait to import `sha256_file`, each returning the same digest meanwhile:

- TODO(digests_held_copies): import `sha256_file` in place of the seven copies left in
  contact_esmc6b and sequence_gradient, each of which changes bytes a record holds.
  contact_esmc6b's `references/build.json` holds three of its four as they are, though
  `scripts/deploy.py` rewrites it on each deploy, and `atlasv2.runtime.identity` hashes every file
  under `src/` into each run's identity, so the next run would not match the runs before it.
  sequence_gradient's bundle would vendor `digests` too, and its campaign manifest hashes every file
  under `runner/`. protein_screening's `artifacts.file_digest`
  and `foundry.datasets.mmseqs._sha256_file` wait until protein_screening's build is done, since
  its Modal pipeline hashes every Python file under `foundry/datasets` into the identity of each
  stage it runs. The scripts in those projects wait with them, and each then needs the check the
  three scripts above had.
- Four copies run in DatasetDev's Modal containers: de_novo's
  `rigorous.release_compiler._file_sha256`, `rigorous.remote_build._sha256_file`, and
  `rigorous.stage_proteinbase.sha256_file`, and the benignity builder's remote-build file hash. Those
  images ship the builder directories without foundry, so the four wait for item 31 of
  [the approvals](../../docs/decisions/workspace/approvals_waiting_on_logan.md), which adds
  foundry to both images and builds them.

abca4_explore's `pipeline._sha256` now imports this function. Its projection vendors
`foundry.digests`, and its wheel packages foundry beside `abca4_avi`.

Tests: `tests/tier1_unit/test_digests.py`, with each consumer's suite as the guard on the names it
imports.
