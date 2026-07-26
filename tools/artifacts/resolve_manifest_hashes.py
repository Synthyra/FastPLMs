"""Resolve manifest file identities at immutable Hugging Face revisions.

This tool is intentionally read-only. It prints a JSON mapping that can be
reviewed before ``models.toml`` is changed. Small Git-managed files are
downloaded and hashed as Git blobs. LFS files use the SHA-256 identity exposed
by Hub metadata, so multi-gigabyte weight shards are never downloaded merely
to resolve provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any
from huggingface_hub import HfApi, hf_hub_download


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()


def _lfs_sha256(sibling: Any) -> str | None:
    lfs = sibling.lfs
    if lfs is None:
        return None
    value = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    return str(value) if value else None


def resolve_manifest(manifest: Path) -> dict[str, dict[str, str]]:
    """Return identities for every unresolved manifest path."""
    with manifest.open("rb") as handle:
        document = tomllib.load(handle)

    api = HfApi()
    resolved: dict[str, dict[str, str]] = {}
    failures: list[str] = []
    for model in document["models"]:
        model_id = model["id"]
        model_result: dict[str, str] = {}
        for label in ("fast", "official"):
            paths = model.get(f"{label}_unresolved_files", ())
            if not paths:
                continue
            repo_id = model[f"{label}_repo"]
            revision = model[f"{label}_revision"]
            try:
                info = api.model_info(
                    repo_id=repo_id,
                    revision=revision,
                    files_metadata=True,
                )
            except Exception as error:  # pragma: no cover - network diagnostic
                failures.append(f"{repo_id}@{revision}: {type(error).__name__}: {error}")
                continue
            siblings = {sibling.rfilename: sibling for sibling in info.siblings}
            for path in paths:
                key = f"{label}:{path}"
                sibling = siblings.get(path)
                if sibling is None:
                    failures.append(f"{repo_id}@{revision}:{path}: missing from Hub metadata")
                    continue
                sha256 = _lfs_sha256(sibling)
                if sha256 is not None:
                    model_result[key] = f"sha256:{sha256}"
                    continue
                try:
                    downloaded = hf_hub_download(
                        repo_id=repo_id,
                        filename=path,
                        revision=revision,
                    )
                    payload = Path(downloaded).read_bytes()
                except Exception as error:  # pragma: no cover - network diagnostic
                    failures.append(f"{repo_id}@{revision}:{path}: {type(error).__name__}: {error}")
                    continue
                model_result[key] = f"git-sha1:{_git_blob_sha1(payload)}"
        if model_result:
            resolved[model_id] = model_result

    if failures:
        detail = "\n".join(failures)
        raise RuntimeError(f"Manifest identities could not be resolved:\n{detail}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = resolve_manifest(args.manifest)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
