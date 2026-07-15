"""End-to-end contracts for Git-free remote source archives."""

from __future__ import annotations

import json
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.artifacts.build import ArtifactError, _validate_vendor_revisions
from tools.remote.run import create_source_archive
from tools.source_provenance import ARCHIVE_PROVENANCE_NAME


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        [
            "git",
            "-c",
            "user.name=FastPLMs Tests",
            "-c",
            "user.email=fastplms-tests@example.invalid",
            "-c",
            "protocol.file.allow=always",
            *arguments,
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _revision_contract(revision: str) -> tuple[SimpleNamespace, SimpleNamespace]:
    source = SimpleNamespace(path="vendor/upstream/toy", revision=revision)
    registry = SimpleNamespace(upstreams={"toy": source})
    spec = SimpleNamespace(family=SimpleNamespace(upstreams=("toy",)))
    return registry, spec


def _create_repository(tmp_path: Path) -> tuple[Path, str]:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(upstream, "init", "--initial-branch=main")
    (upstream / "LICENSE").write_text("Synthetic license\n", encoding="utf-8")
    (upstream / "weights.py").write_text("scale = 2\n", encoding="utf-8")
    _git(upstream, "add", "LICENSE", "weights.py")
    _git(upstream, "commit", "-m", "Create pinned upstream")
    revision = _git(upstream, "rev-parse", "HEAD")

    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--initial-branch=main")
    (repository / "README.md").write_text("Synthetic FastPLMs tree\n", encoding="utf-8")
    (repository / ".secrets.env").write_text("TOKEN=not-archived\n", encoding="utf-8")
    _git(repository, "add", "README.md")
    _git(
        repository,
        "submodule",
        "add",
        str(upstream),
        "vendor/upstream/toy",
    )
    _git(repository, "commit", "-m", "Pin synthetic upstream")
    return repository, revision


def test_source_archive_preserves_revision_proof_without_git_metadata(
    tmp_path: Path,
) -> None:
    repository, revision = _create_repository(tmp_path)
    archive_path = tmp_path / "source.tar.gz"
    create_source_archive(repository, archive_path)

    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(archive_path, "r:gz") as archive:
        member_names = [member.name for member in archive.getmembers()]
        archive.extractall(extracted, filter="data")

    assert ARCHIVE_PROVENANCE_NAME in member_names
    assert ".secrets.env" not in member_names
    assert not any(".git" in Path(name).parts for name in member_names)
    assert not any(path.name == ".git" for path in extracted.rglob(".git"))

    provenance = json.loads((extracted / ARCHIVE_PROVENANCE_NAME).read_text(encoding="utf-8"))
    record = provenance["submodules"]["vendor/upstream/toy"]
    assert record["gitlink_revision"] == revision
    assert record["head_revision"] == revision
    assert record["tracked_files"] == ["LICENSE", "weights.py"]

    registry, spec = _revision_contract(revision)
    _validate_vendor_revisions(extracted, registry, spec)

    (extracted / "vendor" / "upstream" / "toy" / "weights.py").write_text(
        "scale = 3\n",
        encoding="utf-8",
    )
    with pytest.raises(ArtifactError, match="tracked-tree digest differs"):
        _validate_vendor_revisions(extracted, registry, spec)

    marker = extracted / ARCHIVE_PROVENANCE_NAME
    marker.unlink()
    marker.mkdir()
    with pytest.raises(ArtifactError, match="not a regular file"):
        _validate_vendor_revisions(extracted, registry, spec)


def test_archive_provenance_is_not_used_when_git_metadata_exists(tmp_path: Path) -> None:
    repository, revision = _create_repository(tmp_path)
    archive_path = tmp_path / "source.tar.gz"
    create_source_archive(repository, archive_path)
    with tarfile.open(archive_path, "r:gz") as archive:
        marker = archive.extractfile(ARCHIVE_PROVENANCE_NAME)
        assert marker is not None
        (repository / ARCHIVE_PROVENANCE_NAME).write_bytes(marker.read())

    (repository / "vendor" / "upstream" / "toy" / ".git").unlink()
    wrong_revision = "0" * 40 if revision != "0" * 40 else "1" * 40
    registry, spec = _revision_contract(wrong_revision)
    with pytest.raises(ArtifactError, match="not initialized"):
        _validate_vendor_revisions(repository, registry, spec)


def test_source_archive_rejects_modified_tracked_submodule_files(tmp_path: Path) -> None:
    repository, _ = _create_repository(tmp_path)
    (repository / "vendor" / "upstream" / "toy" / "weights.py").write_text(
        "scale = 3\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="modified tracked files"):
        create_source_archive(repository, tmp_path / "source.tar.gz")
