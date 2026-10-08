---
id: artifact_archive
kind: library_module
status: active
aliases:
- artifact_archive
- foundry.artifact_archive
---

# Artifact archives

Selective restoration of regular files from immutable Hugging Face artifact manifests.
Promoted on its second runtime consumer: the SAE GP app and the production probe grid.
The Modal archival worker imports the same implementation.

`restore_files(repository, manifest_path, revision, manifest_sha256, destination,
prefixes=("",), cache_dir=None)` verifies the pinned manifest, downloads only shards
containing selected paths, checks shard and fragment hashes, reconstructs files in
offset order, and verifies each original SHA-256. Existing matching files are reused;
different bytes fail without overwrite. Paths stay inside the destination. Directory
and symbolic-link metadata remain in the manifest; this runtime API restores regular
files only.

`restore_manifest_files` accepts an already verified manifest and implements the same
restoration. Its caller supplies trust in that manifest. Cache location and HF auth
belong to the caller and `huggingface_hub`.

The archival worker is [volume_hf_archive.py](../../modal/volume_hf_archive.py), and the
focused restoration tests are [test_volume_hf_archive.py](../../tests/tier1_unit/test_volume_hf_archive.py).
