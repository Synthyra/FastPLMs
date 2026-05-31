"""Overwrite Hub-cached composite modeling files with a freshly-built local composite.

Purpose: validate in-flight changes to fastplms/ against the parity suite WITHOUT
having to push to HF Hub first. The parity tests load via trust_remote_code, which
caches the Hub `modeling_*.py` at
`~/.cache/huggingface/modules/transformers_modules/<repo>/<rev>/modeling_*.py`.
This script rebuilds those files from the current checkout so the next pytest run
exercises local code.

Usage:
    python testing/debug_scripts/patch_hub_cache.py             # patch every Synthyra family
    python testing/debug_scripts/patch_hub_cache.py dplm esm2   # patch only listed families
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from update_HF import MODEL_REGISTRY, build_composite  # noqa: E402


def repo_slug(repo_id: str) -> str:
    """HF transformers_modules slug is the repo id with '-' replaced by '_hyphen_'."""
    return repo_id.replace("-", "_hyphen_")


def cache_dir_for(repo_id: str) -> Path:
    base = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return base / "modules" / "transformers_modules" / repo_slug(repo_id)


def patch_entry(entry: dict) -> int:
    if not entry.get("composite", False):
        return 0
    composite_body = build_composite(
        entry["modeling_src"],
        include_embedding_mixin=entry.get("include_embedding_mixin", True),
    )
    dest_filename = entry["modeling_dest"]
    patched = 0
    for repo_id in entry["repo_ids"]:
        cache_root = cache_dir_for(repo_id)
        if not cache_root.exists():
            print(f"  [skip] {repo_id}: no cache at {cache_root}")
            continue
        # Every revision subdir gets patched.
        revs = [p for p in cache_root.iterdir() if p.is_dir()]
        if not revs:
            print(f"  [skip] {repo_id}: no revisions under {cache_root}")
            continue
        for rev in revs:
            dest = rev / dest_filename
            if not dest.exists():
                print(f"  [skip] {repo_id}@{rev.name}: no {dest_filename}")
                continue
            dest.write_text(composite_body, encoding="utf-8")
            print(f"  [patch] {repo_id}@{rev.name}: {dest}")
            patched += 1
    return patched


def main() -> int:
    families_filter = set(sys.argv[1:])
    total = 0
    for entry in MODEL_REGISTRY:
        if families_filter and entry["family"] not in families_filter:
            continue
        print(f"== {entry['family']} ==")
        total += patch_entry(entry)
    print(f"\nPatched {total} composite file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
