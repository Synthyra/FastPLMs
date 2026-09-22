"""Explicit, digest-checked hydration of public research evidence.

No runtime or test import downloads files. Run ``fetch`` before offline checks;
``stage`` prepares only the manifest allowlist for a separate reviewed upload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import tomllib

from dataclasses import dataclass
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "Synthyra/FastPLMs-artifacts"
MANIFEST = ROOT / "evidence.toml"


@dataclass(frozen=True)
class EvidenceFile:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class EvidenceStore:
    repository: str
    revision: str
    files: tuple[EvidenceFile, ...]


def _allowed_path(value: str) -> bool:
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or "\\" in value:
        return False
    if any(part in {".", ".."} or ":" in part for part in path.parts):
        return False
    return len(path.parts) >= 3 and (
        (
            path.parts[:2] in {("docs", "evidence"), ("docs", "validation")}
            and path.suffix == ".json"
        )
        or (path.parts[:2] == ("tests", "goldens") and path.suffix in {".json", ".safetensors"})
    )


def load_manifest(path: Path = MANIFEST) -> EvidenceStore:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    if set(payload) != {"schema_version", "repository", "revision", "files"}:
        raise ValueError("Unsupported evidence manifest schema")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("Unsupported evidence manifest schema")
    if payload["repository"] != REPOSITORY:
        raise ValueError(f"Evidence repository must be {REPOSITORY}")
    revision = payload["revision"]
    if not isinstance(revision, str) or (
        revision != "pending" and re.fullmatch(r"[0-9a-f]{40}", revision) is None
    ):
        raise ValueError("Evidence revision must be an immutable commit or pending")
    entries = payload["files"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("Evidence manifest must contain files")
    files = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256"}:
            raise ValueError("Invalid evidence file declaration")
        relative, size, digest = entry["path"], entry["size"], entry["sha256"]
        if not isinstance(relative, str) or not _allowed_path(relative):
            raise ValueError(f"Evidence path is outside the allowlist: {relative!r}")
        if relative.casefold() in seen:
            raise ValueError(f"Duplicate evidence path: {relative}")
        seen.add(relative.casefold())
        if type(size) is not int or size < 0:
            raise ValueError(f"Invalid evidence size: {relative}")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"Invalid SHA-256: {relative}")
        files.append(EvidenceFile(relative, size, digest))
    return EvidenceStore(REPOSITORY, revision, tuple(files))


def _local_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Evidence path escapes destination: {relative}")
    return path


def verify_file(path: Path, entry: EvidenceFile) -> None:
    if not path.is_file() or path.stat().st_size != entry.size:
        raise ValueError(f"Missing evidence or size mismatch: {entry.path}")
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != entry.sha256:
        raise ValueError(f"Evidence SHA-256 mismatch: {entry.path}")


def verify(store: EvidenceStore, root: Path) -> None:
    for entry in store.files:
        verify_file(_local_path(root, entry.path), entry)


def _copy_verified(source: Path, destination: Path, entry: EvidenceFile) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".evidence-", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as original:
            shutil.copyfileobj(original, output)
        verify_file(temporary, entry)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def stage(store: EvidenceStore, root: Path, destination: Path) -> None:
    """Create a fresh upload directory containing exactly the validated files."""
    verify(store, root)
    if destination.exists():
        raise ValueError("Upload staging directory already exists; choose a fresh directory")
    destination.mkdir(parents=True)
    for entry in store.files:
        _copy_verified(_local_path(root, entry.path), _local_path(destination, entry.path), entry)


def fetch(store: EvidenceStore, root: Path) -> None:
    """Download pinned files explicitly, retaining any differing local evidence."""
    if re.fullmatch(r"[0-9a-f]{40}", store.revision) is None:
        raise ValueError("Fetch requires a published immutable dataset revision")
    pending = []
    for entry in store.files:
        destination = _local_path(root, entry.path)
        if destination.exists():
            verify_file(destination, entry)
        else:
            pending.append((entry, destination))
    if not pending:
        return
    from huggingface_hub import hf_hub_download

    for entry, destination in pending:
        source = Path(
            hf_hub_download(
                repo_id=store.repository,
                repo_type="dataset",
                revision=store.revision,
                filename=entry.path,
            )
        )
        _copy_verified(source, destination, entry)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "verify", "fetch", "stage"))
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    store = load_manifest(args.manifest)
    if args.command == "verify":
        verify(store, args.root)
    elif args.command == "fetch":
        fetch(store, args.root)
    elif args.command == "stage":
        if args.output is None:
            parser.error("stage requires --output")
        stage(store, args.root, args.output)
    print(
        json.dumps(
            {
                "command": args.command,
                "repository": store.repository,
                "revision": store.revision,
                "files": [entry.path for entry in store.files],
                "bytes": sum(entry.size for entry in store.files),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
