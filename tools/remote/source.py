"""Clean Git source archives with pinned submodules and verified evidence."""

from __future__ import annotations

import hashlib
import io
import re
import subprocess
import tarfile

from pathlib import Path, PurePosixPath

from tools.artifacts.evidence_store import load_manifest
from tools.execution.source import (
    excluded_from_upload as _is_sensitive,
    require_regular_source,
)
from tools.source_record import (
    ARCHIVE_PROVENANCE_NAME,
    archive_root_record,
    render_archive_provenance,
    tracked_tree_digest,
)


GIT_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _require_matching_archive_digest(output: str, expected_sha256: str) -> None:
    fields = output.split()
    if (
        re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        or not fields
        or fields[0] != expected_sha256
    ):
        raise RuntimeError("Uploaded source archive SHA-256 differs from local bytes")


def _git_files(repository: Path) -> list[Path]:
    command = [
        "git",
        "-c",
        f"safe.directory={repository.resolve().as_posix()}",
        "ls-files",
        "-z",
        "--cached",
    ]
    completed = subprocess.run(command, cwd=repository, check=True, capture_output=True)
    return [Path(raw.decode()) for raw in completed.stdout.split(b"\0") if raw]


def _require_clean_repository(repository: Path) -> None:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository.resolve().as_posix()}",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip():
        raise RuntimeError(
            "Remote runs require a clean Git worktree so the reported revision "
            "identifies the exact source."
        )


def _require_clean_tracked_repository(repository: Path) -> None:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository.resolve().as_posix()}",
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
            "--ignore-submodules=all",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip():
        raise RuntimeError(
            "Source archives require clean tracked root files so the content "
            "attestation identifies exact bytes."
        )


def _git_head_revision(repository: Path) -> str:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository.resolve().as_posix()}",
            "rev-parse",
            "HEAD",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    if GIT_REVISION_PATTERN.fullmatch(revision) is None:
        raise RuntimeError(f"Git returned an invalid HEAD revision: {revision!r}")
    return revision


def _is_tracked_file(repository: Path, relative_name: str) -> bool:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository.resolve().as_posix()}",
            "ls-files",
            "--error-unmatch",
            "--",
            relative_name,
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0 and completed.stdout.strip() == relative_name


def _gitlink_revision(repository: Path, relative_root: Path) -> str:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository.resolve().as_posix()}",
            "ls-files",
            "--stage",
            "-z",
            "--",
            relative_root.as_posix(),
        ],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    records = [record for record in completed.stdout.split(b"\0") if record]
    if len(records) != 1:
        raise RuntimeError(f"Expected one Git-link record for {relative_root.as_posix()!r}")
    try:
        metadata, encoded_path = records[0].split(b"\t", 1)
        mode, revision, stage = metadata.decode("ascii").split()
        recorded_path = encoded_path.decode()
    except (UnicodeDecodeError, ValueError) as error:
        raise RuntimeError(
            f"Could not parse Git-link record for {relative_root.as_posix()!r}"
        ) from error
    if (
        mode != "160000"
        or stage != "0"
        or recorded_path != relative_root.as_posix()
        or GIT_REVISION_PATTERN.fullmatch(revision) is None
    ):
        raise RuntimeError(f"Invalid Git-link record for {relative_root.as_posix()!r}")
    return revision


def _submodule_files(
    repository: Path,
    submodule: Path,
    relative_root: Path,
) -> tuple[list[tuple[Path, Path]], dict[str, object]]:
    git_metadata = submodule / ".git"
    if not (git_metadata.exists() or git_metadata.is_symlink()):
        raise RuntimeError(
            f"Submodule {relative_root.as_posix()!r} is not initialized. Run "
            "'git submodule update --init --recursive'."
        )
    safe_directory = f"safe.directory={submodule.resolve().as_posix()}"
    head = subprocess.run(
        ["git", "-c", safe_directory, "rev-parse", "HEAD"],
        cwd=submodule,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    gitlink = _gitlink_revision(repository, relative_root)
    if GIT_REVISION_PATTERN.fullmatch(head) is None or head != gitlink:
        raise RuntimeError(
            f"Submodule {relative_root.as_posix()!r} is at {head!r}, "
            f"but its Git link records {gitlink!r}."
        )
    status = subprocess.run(
        ["git", "-c", safe_directory, "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=submodule,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(
            f"Submodule {relative_root.as_posix()!r} has modified tracked files; "
            "source archives require the exact pinned tree."
        )
    completed = subprocess.run(
        ["git", "-c", safe_directory, "ls-files", "-z"],
        cwd=submodule,
        check=True,
        capture_output=True,
    )
    output: list[tuple[Path, Path]] = []
    tracked_files: list[str] = []
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        child = Path(raw.decode())
        source = submodule / child
        if not (source.is_file() or source.is_symlink()):
            raise RuntimeError(
                f"Tracked submodule path is unavailable or unsupported: "
                f"{(relative_root / child).as_posix()}"
            )
        child_name = child.as_posix()
        if _is_sensitive(PurePosixPath(child_name)):
            raise RuntimeError(
                f"Submodule tracks forbidden source path: {(relative_root / child).as_posix()}"
            )
        require_regular_source(repository, relative_root / child)
        output.append((source, relative_root / child))
        tracked_files.append(child_name)
    tracked_files.sort()
    record: dict[str, object] = {
        "file_count": len(tracked_files),
        "gitlink_revision": gitlink,
        "head_revision": head,
        "tracked_files": tracked_files,
        "tree_sha256": tracked_tree_digest(submodule, tracked_files),
    }
    return output, record


def _archive_evidence(repository: Path) -> list[tuple[str, bytes]]:
    """Read only manifest-pinned evidence, keeping verified bytes for archiving."""
    store = load_manifest(repository / "evidence.toml")
    if store.revision == "pending":
        tracked = {path.as_posix() for path in _git_files(repository)}
        untracked = [entry.path for entry in store.files if entry.path not in tracked]
        if untracked:
            raise RuntimeError(
                "Pending evidence must remain tracked until its dataset revision is published: "
                + ", ".join(untracked)
            )
    payloads = []
    for entry in store.files:
        if _is_sensitive(PurePosixPath(entry.path)):
            raise RuntimeError(f"Forbidden evidence path: {entry.path}")
        try:
            source = require_regular_source(repository, Path(entry.path))
            payload = source.read_bytes()
        except OSError as error:
            raise RuntimeError(
                "Evidence must be hydrated before creating a source archive; run "
                "python -m tools.artifacts.evidence_store fetch"
            ) from error
        if len(payload) != entry.size or hashlib.sha256(payload).hexdigest() != entry.sha256:
            raise RuntimeError(f"Evidence identity mismatch: {entry.path}")
        payloads.append((entry.path, payload))
    return payloads


def create_source_archive(
    repository: Path,
    destination: Path,
) -> dict[str, dict[str, object]]:
    """Archive tracked source, pinned submodules, and verified local evidence."""

    repository = repository.resolve()
    _require_clean_tracked_repository(repository)
    head_revision = _git_head_revision(repository)
    files: list[tuple[Path, Path]] = []
    root_tracked_files: list[str] = []
    provenance: dict[str, dict[str, object]] = {}
    for relative in _git_files(repository):
        source = repository / relative
        posix = PurePosixPath(relative.as_posix())
        if _is_sensitive(posix):
            raise RuntimeError(f"Repository tracks forbidden source path: {posix.as_posix()!r}")
        if posix.as_posix() == ARCHIVE_PROVENANCE_NAME:
            raise RuntimeError("Repository may not track the generated source provenance marker")
        if not source.exists() and not source.is_symlink():
            if posix.parts[:2] == ("vendor", "upstream"):
                continue
            raise RuntimeError(f"Tracked source path is unavailable: {posix.as_posix()!r}")
        if source.is_file() or source.is_symlink():
            require_regular_source(repository, relative)
            files.append((source, relative))
            root_tracked_files.append(posix.as_posix())
        elif posix.parts[:2] == ("vendor", "upstream"):
            git_metadata = source / ".git"
            if not (git_metadata.exists() or git_metadata.is_symlink()):
                continue
            submodule_files, record = _submodule_files(repository, source, relative)
            files.extend(submodule_files)
            provenance[posix.as_posix()] = record
        else:
            raise RuntimeError(f"Tracked source path has an unsupported type: {posix.as_posix()!r}")

    root_record = archive_root_record(
        repository,
        root_tracked_files,
        head_revision=head_revision,
    )
    evidence = _archive_evidence(repository) if "evidence.toml" in root_tracked_files else []
    evidence_paths = {name for name, _ in evidence}

    seen: set[str] = set()
    with tarfile.open(
        destination,
        "w:gz",
        format=tarfile.PAX_FORMAT,
        dereference=False,
    ) as archive:
        for source, relative in sorted(files, key=lambda item: item[1].as_posix()):
            archive_name = relative.as_posix()
            if (
                archive_name in seen
                or archive_name in evidence_paths
                or _is_sensitive(PurePosixPath(archive_name))
            ):
                continue
            seen.add(archive_name)
            archive.add(source, arcname=archive_name, recursive=False)
        for name, payload in evidence:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(payload))
        provenance_bytes = render_archive_provenance(provenance, root=root_record)
        provenance_info = tarfile.TarInfo(ARCHIVE_PROVENANCE_NAME)
        provenance_info.size = len(provenance_bytes)
        provenance_info.mode = 0o644
        provenance_info.mtime = 0
        provenance_info.uid = 0
        provenance_info.gid = 0
        provenance_info.uname = ""
        provenance_info.gname = ""
        archive.addfile(provenance_info, io.BytesIO(provenance_bytes))
    _require_clean_tracked_repository(repository)
    if _git_head_revision(repository) != head_revision:
        destination.unlink(missing_ok=True)
        raise RuntimeError("Repository revision changed while creating the source archive")
    return provenance
