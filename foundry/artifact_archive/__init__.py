"""Restore selected artifact files from immutable lossless Hugging Face archives."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tarfile
import time
import uuid

from pathlib import Path
from typing import NotRequired, TypedDict
from huggingface_hub import hf_hub_download

from foundry.digests import sha256_file


CHUNK_BYTES = 8 * 1024**2


# The manifest schema. modal/volume_hf_archive.py writes manifests and this module reads them;
# both use these records.
class ArchiveEntry(TypedDict):
    """One inventoried path under a source's root; a regular file gains `sha256` once hashed."""

    path: str
    size: int
    mtime_ns: int
    mode: int  # permission bits
    type: str  # file, symlink, or directory
    target: NotRequired[str]  # a symbolic link's target
    sha256: NotRequired[str]


class ArchiveFragment(TypedDict):
    """The bytes of one file from `offset`, stored as tar member `member` of a shard.

    `sha256` is set once the shard is written, so every fragment in a manifest has it.
    """

    path: str
    offset: int
    size: int
    member: str
    sha256: NotRequired[str]


class ArchiveShard(TypedDict):
    """One committed tar file in the archive repository, at the revision that committed it."""

    path: str
    sha256: str
    size: int
    revision: str
    fragments: list[ArchiveFragment]


class ArchiveManifest(TypedDict):
    """What a source's archive holds, as `sources/<volume>/<inventory>/manifest.json` records it."""

    source_volume: str
    source_inventory_sha256: str
    excluded_prefixes: list[str]
    files: list[ArchiveEntry]
    shards: list[ArchiveShard]


def restore_files(repository: str, manifest_path: str, revision: str, manifest_sha256: str,
                  destination: Path, prefixes: tuple[str, ...] = ("",), cache_dir: str | None = None) -> list[str]:
    """Restore selected regular files from verified immutable fragments, without overwrites."""
    downloaded = Path(hf_hub_download(repository, manifest_path, repo_type="dataset", revision=revision,
                                      cache_dir=cache_dir))
    if sha256_file(downloaded) != manifest_sha256:
        raise ValueError("Archive manifest digest differs")
    return restore_manifest_files(repository, json.loads(downloaded.read_text()), destination, prefixes, cache_dir)


def restore_manifest_files(repository: str, manifest: ArchiveManifest, destination: Path,
                           prefixes: tuple[str, ...], cache_dir: str | None = None) -> list[str]:
    """Restore the manifest's regular files under `prefixes`, and return every selected path, sorted.

    The caller vouches for the manifest, as `restore_files` does by checking its digest.
    """
    destination.mkdir(parents=True, exist_ok=True)
    files = {entry["path"]: entry for entry in manifest["files"] if entry["type"] == "file"
             and any(entry["path"].startswith(prefix) for prefix in prefixes)}

    # TODO(artifact_archive_partial_cleanup): a check that fails after the partial files exist leaves them behind
    # Each file still to restore is written to a partial sibling, and a matching file is reused.
    partials: dict[str, Path] = {}
    offsets = {path: 0 for path in files}
    existing: set[str] = set()
    for path, entry in files.items():
        target = destination / path
        if ".." in Path(path).parts or Path(path).is_absolute():
            raise ValueError(f"Unsafe archive path: {path}")
        # Raises ValueError when a symbolic link in the destination leads outside it.
        target.resolve().relative_to(destination.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256_file(target) != entry["sha256"]:
                raise ValueError(f"Refusing to overwrite restored file: {path}")
            existing.add(path)
            continue
        partial = target.with_name(target.name + f".restore-{uuid.uuid4().hex}.partial")
        partial.write_bytes(b"")
        partials[path] = partial

    print(f"artifact_archive: restoring {len(partials)} files from {repository}, {len(existing)} already present",
          file=sys.stderr, flush=True)
    started = time.monotonic()

    # A file is rebuilt by appending its fragments in manifest order, which must be offset order.
    for shard in manifest["shards"]:
        needed = [fragment for fragment in shard["fragments"]
                  if fragment["path"] in files and fragment["path"] not in existing]
        if not needed:
            continue
        payload = Path(hf_hub_download(repository, shard["path"], repo_type="dataset", revision=shard["revision"],
                                      cache_dir=cache_dir))
        if sha256_file(payload) != shard["sha256"] or payload.stat().st_size != shard["size"]:
            raise ValueError("Archive tar digest differs")
        with tarfile.open(payload) as archive:
            for fragment in needed:
                path = fragment["path"]
                if offsets[path] != fragment["offset"]:
                    raise ValueError(f"Restore fragment gap: {path}")
                incoming = archive.extractfile(fragment["member"])
                if incoming is None:
                    raise ValueError(f"Missing archive fragment: {path}")
                digest = hashlib.sha256()
                length = 0
                with partials[path].open("ab") as outgoing:
                    for chunk in iter(lambda: incoming.read(CHUNK_BYTES), b""):
                        digest.update(chunk)
                        length += len(chunk)
                        outgoing.write(chunk)
                if digest.hexdigest() != fragment["sha256"] or length != fragment["size"]:
                    raise ValueError(f"Restored fragment differs: {path}")
                offsets[path] += length

    # A file another writer restored meanwhile is kept when its bytes match.
    for path, partial in partials.items():
        entry = files[path]
        if offsets[path] != entry["size"] or sha256_file(partial) != entry["sha256"]:
            raise ValueError(f"Restored original file differs: {path}")
        target = destination / path
        if target.exists():
            if sha256_file(target) != entry["sha256"]:
                raise ValueError(f"Another writer changed restored file: {path}")
            partial.unlink()
        else:
            os.replace(partial, target)

    print(f"artifact_archive: restored {len(partials)} files in {time.monotonic() - started:.1f} s",
          file=sys.stderr, flush=True)
    return sorted(files)
