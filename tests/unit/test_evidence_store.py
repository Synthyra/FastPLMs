"""Evidence hydration is explicit, pinned, and preserves unverified local files."""

from __future__ import annotations

import hashlib
import sys
import types
import pytest

from pathlib import Path

from tools.artifacts.evidence_store import (
    REPOSITORY,
    EvidenceFile,
    EvidenceStore,
    fetch,
    load_manifest,
    stage,
    verify,
)


def _store(revision: str = "a" * 40) -> EvidenceStore:
    content = b"result\n"
    entry = EvidenceFile(
        "docs/evidence/test.json", len(content), hashlib.sha256(content).hexdigest()
    )
    return EvidenceStore(REPOSITORY, revision, (entry,))


def _write_manifest(path: Path, relative: str, revision: str = "pending") -> None:
    path.write_text(
        f'schema_version = 1\nrepository = "{REPOSITORY}"\nrevision = "{revision}"\n'
        f'[[files]]\npath = "{relative}"\nsize = 7\nsha256 = "' + "a" * 64 + '"\n',
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "relative",
    [
        "../docs/evidence/test.json",
        "/docs/evidence/test.json",
        "docs/evidence/../test.json",
        "docs/evidence//test.json",
        "docs/evidence/C:secret.json",
        "src/config.json",
        "docs/evidence/head.safetensors",
    ],
)
def test_manifest_rejects_paths_outside_allowlist(tmp_path: Path, relative: str) -> None:
    manifest = tmp_path / "evidence.toml"
    _write_manifest(manifest, relative)
    with pytest.raises(ValueError, match="allowlist"):
        load_manifest(manifest)


def test_pending_manifest_can_be_planned_but_not_fetched(tmp_path: Path) -> None:
    manifest = tmp_path / "evidence.toml"
    _write_manifest(manifest, "tests/goldens/toy.safetensors")
    store = load_manifest(manifest)
    with pytest.raises(ValueError, match="immutable"):
        fetch(store, tmp_path)


def test_stage_is_exact_and_refuses_existing_destination(tmp_path: Path) -> None:
    root = tmp_path / "source"
    source = root / "docs/evidence/test.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"result\n")
    (source.parent / "unlisted.json").write_bytes(b"exclude")
    destination = tmp_path / "upload"
    stage(_store(), root, destination)
    installed = [
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    ]
    assert installed == ["docs/evidence/test.json"]
    verify(_store(), destination)
    with pytest.raises(ValueError, match="already exists"):
        stage(_store(), root, destination)


def test_fetch_checks_download_and_uses_immutable_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cached = tmp_path / "cached"
    cached.write_bytes(b"result\n")
    calls = []

    def download(**kwargs: str) -> str:
        calls.append(kwargs)
        return str(cached)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=download)
    )
    root = tmp_path / "checkout"
    fetch(_store(), root)
    verify(_store(), root)
    fetch(_store(), root)
    assert calls == [
        {
            "repo_id": REPOSITORY,
            "repo_type": "dataset",
            "revision": "a" * 40,
            "filename": "docs/evidence/test.json",
        }
    ]


def test_corrupt_download_is_not_installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cached = tmp_path / "cached"
    cached.write_bytes(b"wrong!\n")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=lambda **kwargs: str(cached)),
    )
    root = tmp_path / "checkout"
    with pytest.raises(ValueError, match="SHA-256"):
        fetch(_store(), root)
    assert not (root / "docs/evidence/test.json").exists()
    assert not list(root.rglob(".evidence-*"))


def test_differing_local_evidence_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "docs/evidence/test.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"newer evaluation")
    with pytest.raises(ValueError, match="size mismatch"):
        fetch(_store(), tmp_path)
    assert path.read_bytes() == b"newer evaluation"


def test_manifest_rejects_duplicate_paths(tmp_path: Path) -> None:
    manifest = tmp_path / "evidence.toml"
    _write_manifest(manifest, "docs/evidence/test.json")
    text = manifest.read_text(encoding="utf-8")
    manifest.write_text(text + "[[files]]" + text.split("[[files]]", 1)[1], encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_manifest(manifest)


def test_symlink_destination_cannot_escape_root(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    try:
        (root / "docs").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks requires platform permission")
    with pytest.raises(ValueError, match="escapes"):
        fetch(_store(), root)
