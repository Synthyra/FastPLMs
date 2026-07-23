"""End-to-end contracts for Git-free remote source archives."""

from __future__ import annotations

import json
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.artifacts.build as artifact_build
from tools.artifacts.build import (
    ArtifactError,
    _validate_vendor_revisions,
    _validated_runtime_snapshot,
)
from tools.remote.run import create_source_archive
from tools.source_provenance import (
    ARCHIVE_PROVENANCE_NAME,
    SourceProvenanceError,
    validate_archived_root,
)


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
    runtime = repository / "src" / "fastplms" / "toy_runtime"
    runtime.mkdir(parents=True)
    (runtime / "core.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repository / ".secrets.env").write_text("TOKEN=not-archived\n", encoding="utf-8")
    _git(repository, "add", "README.md", "src/fastplms/toy_runtime/core.py")
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
    root_record = provenance["root"]
    assert root_record["head_revision"] == _git(repository, "rev-parse", "HEAD")
    assert ".secrets.env" not in root_record["files"]
    runtime_record = root_record["files"]["src/fastplms/toy_runtime/core.py"]
    assert runtime_record["mode"] == "100644"
    assert runtime_record["size"] == len("VALUE = 1\n")
    assert len(runtime_record["sha256"]) == 64
    validate_archived_root(extracted)
    archived_runtime = extracted / "src" / "fastplms" / "toy_runtime" / "core.py"
    archived_runtime.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(SourceProvenanceError, match="tracked bytes or modes differ"):
        validate_archived_root(extracted)
    archived_runtime.write_text("VALUE = 1\n", encoding="utf-8")
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


def _runtime_contract() -> tuple[SimpleNamespace, SimpleNamespace]:
    registry = SimpleNamespace()
    family = SimpleNamespace(runtime_paths=("toy_runtime",), attention=())
    return registry, SimpleNamespace(family=family)


def test_git_free_runtime_snapshot_is_content_addressed_and_rejects_unknown_files(
    tmp_path: Path,
) -> None:
    repository, _ = _create_repository(tmp_path)
    archive_path = tmp_path / "source.tar.gz"
    create_source_archive(repository, archive_path)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(extracted, filter="data")
    registry, spec = _runtime_contract()

    runtime_revision, payloads, source_tree_sha256 = _validated_runtime_snapshot(
        extracted,
        registry,
        spec,
    )

    assert runtime_revision == f"source-tree-sha256:{source_tree_sha256}"
    assert payloads == {"toy_runtime/core.py": b"VALUE = 1\n"}
    assert len(source_tree_sha256) == 64

    extra = extracted / "src" / "fastplms" / "toy_runtime" / "extra.py"
    extra.write_text("EXTRA = True\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="inventory differs"):
        _validated_runtime_snapshot(extracted, registry, spec)


def test_git_free_runtime_revision_does_not_trust_diagnostic_head(
    tmp_path: Path,
) -> None:
    repository, _ = _create_repository(tmp_path)
    archive_path = tmp_path / "source.tar.gz"
    create_source_archive(repository, archive_path)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(extracted, filter="data")
    marker = extracted / ARCHIVE_PROVENANCE_NAME
    provenance = json.loads(marker.read_text(encoding="utf-8"))
    provenance["root"]["head_revision"] = "f" * 40
    marker.write_text(json.dumps(provenance), encoding="utf-8")

    diagnostic_head, _inventory = validate_archived_root(extracted)
    registry, spec = _runtime_contract()
    runtime_revision, _payloads, source_tree_sha256 = _validated_runtime_snapshot(
        extracted,
        registry,
        spec,
    )

    assert diagnostic_head == "f" * 40
    assert runtime_revision == f"source-tree-sha256:{source_tree_sha256}"
    assert runtime_revision != diagnostic_head


def test_git_free_runtime_snapshot_rejects_mutation_between_validation_and_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _ = _create_repository(tmp_path)
    archive_path = tmp_path / "source.tar.gz"
    create_source_archive(repository, archive_path)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(extracted, filter="data")
    registry, spec = _runtime_contract()
    original_snapshot = artifact_build._snapshot_runtime_sources

    def tampered_snapshot(*args, **kwargs):
        payloads = original_snapshot(*args, **kwargs)
        payloads["toy_runtime/core.py"] = b"VALUE = 2\n"
        return payloads

    monkeypatch.setattr(artifact_build, "_snapshot_runtime_sources", tampered_snapshot)

    with pytest.raises(ArtifactError, match="mutated during snapshot"):
        _validated_runtime_snapshot(extracted, registry, spec)


@pytest.mark.parametrize(
    "mutation",
    ("schema", "revision", "digest", "size", "path"),
)
def test_archived_root_rejects_malformed_or_forged_metadata(
    tmp_path: Path,
    mutation: str,
) -> None:
    repository, _ = _create_repository(tmp_path)
    archive_path = tmp_path / "source.tar.gz"
    create_source_archive(repository, archive_path)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(extracted, filter="data")
    marker = extracted / ARCHIVE_PROVENANCE_NAME
    provenance = json.loads(marker.read_text(encoding="utf-8"))
    root_record = provenance["root"]
    runtime_name = "src/fastplms/toy_runtime/core.py"
    if mutation == "schema":
        provenance["unknown"] = True
    elif mutation == "revision":
        root_record["head_revision"] = "not-a-commit"
    elif mutation == "digest":
        root_record["files"][runtime_name]["sha256"] = "0" * 64
    elif mutation == "size":
        root_record["files"][runtime_name]["size"] += 1
    else:
        root_record["files"]["../escape.py"] = root_record["files"].pop(runtime_name)
    marker.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(SourceProvenanceError):
        validate_archived_root(extracted)


@pytest.mark.parametrize(
    "unsafe_name",
    ("C:\\escape.py", "src\\fastplms\\escape.py", "payload:stream.py", "."),
)
def test_archived_root_rejects_nonportable_paths(
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    repository, _ = _create_repository(tmp_path)
    archive_path = tmp_path / "source.tar.gz"
    create_source_archive(repository, archive_path)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(extracted, filter="data")
    marker = extracted / ARCHIVE_PROVENANCE_NAME
    provenance = json.loads(marker.read_text(encoding="utf-8"))
    files = provenance["root"]["files"]
    files[unsafe_name] = files.pop("src/fastplms/toy_runtime/core.py")
    marker.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(SourceProvenanceError, match="Non-portable tracked path"):
        validate_archived_root(extracted)


def test_archived_root_rejects_parent_symlink_traversal(tmp_path: Path) -> None:
    repository, _ = _create_repository(tmp_path)
    archive_path = tmp_path / "source.tar.gz"
    create_source_archive(repository, archive_path)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(extracted, filter="data")
    runtime = extracted / "src" / "fastplms" / "toy_runtime"
    preserved = extracted / "src" / "fastplms" / "toy_runtime-preserved"
    runtime.rename(preserved)
    runtime.symlink_to(preserved, target_is_directory=True)

    with pytest.raises(SourceProvenanceError, match="traverses a symlink"):
        validate_archived_root(extracted)


def test_git_free_runtime_snapshot_rejects_symlinks_and_unknown_extensions(
    tmp_path: Path,
) -> None:
    repository, _ = _create_repository(tmp_path)
    archive_path = tmp_path / "source.tar.gz"
    create_source_archive(repository, archive_path)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(extracted, filter="data")
    registry, spec = _runtime_contract()
    runtime = extracted / "src" / "fastplms" / "toy_runtime"
    core = runtime / "core.py"
    core.unlink()
    core.symlink_to(extracted / "README.md")
    with pytest.raises(ArtifactError, match="Symlinks are not allowed"):
        _validated_runtime_snapshot(extracted, registry, spec)

    core.unlink()
    core.write_text("VALUE = 1\n", encoding="utf-8")
    (runtime / "payload.bin").write_bytes(b"unknown")
    with pytest.raises(ArtifactError, match="unapproved extension"):
        _validated_runtime_snapshot(extracted, registry, spec)


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


def test_source_archive_rejects_a_tracked_sensitive_path(tmp_path: Path) -> None:
    repository, _ = _create_repository(tmp_path)
    _git(repository, "add", ".secrets.env")
    _git(repository, "commit", "-m", "Track a forbidden credential-shaped file")

    with pytest.raises(RuntimeError, match="tracks forbidden source path"):
        create_source_archive(repository, tmp_path / "source.tar.gz")
