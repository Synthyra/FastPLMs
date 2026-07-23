"""Fail-closed inspection for FastPLMs wheel and source distributions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import subprocess
import tarfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast, overload

_SENSITIVE_NAMES = {
    ".env",
    ".git",
    ".secrets.env",
    "credentials",
    "credentials.json",
    "id_ed25519",
    "id_rsa",
}
_SENSITIVE_SUFFIXES = {".key", ".p12", ".pem", ".pfx"}
_FORBIDDEN_PARTS = {"__pycache__", "official", "tests", "vendor"}
_NATIVE_BINARY_SUFFIXES = {".dll", ".dylib", ".exe", ".pyd", ".so"}
_MODEL_BINARY_SUFFIXES = {
    ".bin",
    ".onnx",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
}
_MAX_MEMBER_BYTES = 32 * 1024**2
_MAX_TOTAL_BYTES = 128 * 1024**2
_TREE_DOMAIN = b"fastplms-distribution-inventory-v1\0"
_SOURCE_TREE_DOMAIN = b"fastplms-tracked-distribution-source-v1\0"
_DISTRIBUTION_SOURCE_SCOPES = (
    "src/fastplms",
    "pyproject.toml",
    "README.md",
    "kernels.lock",
    "LICENSE",
    "LICENSES",
    "THIRD_PARTY_NOTICES.md",
)
SourceSnapshot = tuple[str, dict[str, bytes], str]


class DistributionInspectionError(RuntimeError):
    """A built distribution contains unsafe, stale, or unexpected content."""


@overload
def _git(
    project_root: Path,
    *arguments: str,
    text: Literal[True],
) -> subprocess.CompletedProcess[str]: ...


@overload
def _git(
    project_root: Path,
    *arguments: str,
    text: Literal[False] = False,
) -> subprocess.CompletedProcess[bytes]: ...


def _git(
    project_root: Path,
    *arguments: str,
    text: bool = False,
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    command = [
        "git",
        "-c",
        f"safe.directory={project_root.as_posix()}",
        *arguments,
    ]
    try:
        result = subprocess.run(
            command,
            cwd=project_root,
            check=True,
            capture_output=True,
            text=text,
        )
        return cast(
            subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes],
            result,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = ""
        if isinstance(error, subprocess.CalledProcessError):
            stderr = error.stderr
            detail = (
                stderr.strip()
                if isinstance(stderr, str)
                else stderr.decode(errors="replace").strip()
            )
        raise DistributionInspectionError(
            "Distribution inspection requires a verifiable Git worktree"
            + (f": {detail}" if detail else ".")
        ) from error


def _tracked_source_snapshot(project_root: Path) -> SourceSnapshot:
    """Return immutable tracked bytes for every declared distribution source scope."""

    project_root = project_root.resolve()
    top_level = _git(project_root, "rev-parse", "--show-toplevel", text=True).stdout.strip()
    if Path(top_level).resolve() != project_root:
        raise DistributionInspectionError(
            f"Project root {project_root} is not the Git worktree root {top_level}."
        )
    revision = _git(project_root, "rev-parse", "--verify", "HEAD", text=True).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise DistributionInspectionError(f"Invalid tracked source revision: {revision!r}")
    status = _git(
        project_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *_DISTRIBUTION_SOURCE_SCOPES,
        text=True,
    ).stdout
    if status.strip():
        changed = sorted(line[3:] for line in status.splitlines() if len(line) >= 4)
        raise DistributionInspectionError(
            "Distribution source scopes must be clean and fully tracked: "
            f"{changed[:20]}"
        )

    result = _git(
        project_root,
        "ls-files",
        "--stage",
        "-z",
        "--",
        *_DISTRIBUTION_SOURCE_SCOPES,
    )
    files: dict[str, bytes] = {}
    for raw_record in result.stdout.split(b"\0"):
        if not raw_record:
            continue
        try:
            header, raw_path = raw_record.split(b"\t", maxsplit=1)
            mode, object_id, stage = header.decode("ascii").split()
            relative_name = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise DistributionInspectionError(
                "Git returned an invalid tracked-file record."
            ) from error
        if stage != "0" or mode not in {"100644", "100755"}:
            raise DistributionInspectionError(
                f"Distribution source is not a regular stage-0 file: {relative_name!r}"
            )
        normalized = PurePosixPath(relative_name).as_posix()
        if normalized != relative_name or relative_name in files:
            raise DistributionInspectionError(
                f"Distribution source has a non-portable or duplicate path: {relative_name!r}"
            )
        files[relative_name] = _git(project_root, "cat-file", "blob", object_id).stdout
    if not files or not any(name.startswith("src/fastplms/") for name in files):
        raise DistributionInspectionError("Tracked distribution source inventory is empty.")

    digest = hashlib.sha256()
    digest.update(_SOURCE_TREE_DOMAIN)
    for relative_name, contents in sorted(files.items()):
        for value in (
            relative_name.encode("utf-8"),
            str(len(contents)).encode("ascii"),
            hashlib.sha256(contents).hexdigest().encode("ascii"),
        ):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    return revision, files, digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_member(raw_name: str, *, root: str | None = None) -> str:
    if "\\" in raw_name:
        raise DistributionInspectionError(
            f"Distribution member uses a Windows path separator: {raw_name!r}"
        )
    name = raw_name.removesuffix("/")
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or path.as_posix() != name:
        raise DistributionInspectionError(f"Non-portable distribution member: {raw_name!r}")
    if root is not None:
        if not path.parts or path.parts[0] != root or len(path.parts) == 1:
            raise DistributionInspectionError(
                f"Source distribution member escapes {root!r}: {raw_name!r}"
            )
        path = PurePosixPath(*path.parts[1:])
    lowered = tuple(part.lower() for part in path.parts)
    if any(part in _SENSITIVE_NAMES for part in lowered):
        raise DistributionInspectionError(f"Sensitive distribution path: {raw_name!r}")
    if any(part in _FORBIDDEN_PARTS for part in lowered):
        raise DistributionInspectionError(f"Forbidden distribution path: {raw_name!r}")
    if path.suffix.lower() in _SENSITIVE_SUFFIXES:
        raise DistributionInspectionError(f"Sensitive distribution suffix: {raw_name!r}")
    if path.suffix.lower() in _NATIVE_BINARY_SUFFIXES:
        raise DistributionInspectionError(
            f"Native binary is forbidden in the pure-Python distribution: {raw_name!r}"
        )
    if path.suffix.lower() in _MODEL_BINARY_SUFFIXES:
        raise DistributionInspectionError(
            f"Model or serialized binary is forbidden in the distribution: {raw_name!r}"
        )
    if path.suffix.lower() in {".pyc", ".pyo"}:
        raise DistributionInspectionError(f"Bytecode is forbidden in distributions: {raw_name!r}")
    return path.as_posix()


def _inventory_digest(records: Sequence[tuple[str, int, str]]) -> str:
    digest = hashlib.sha256()
    digest.update(_TREE_DOMAIN)
    for name, size, content_sha256 in sorted(records):
        for value in (name.encode(), str(size).encode(), content_sha256.encode()):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    return digest.hexdigest()


def _require_members(members: set[str], required: set[str], *, kind: str) -> None:
    missing = sorted(required.difference(members))
    if missing:
        raise DistributionInspectionError(f"{kind} is missing required members: {missing}")


def _tracked_legal_files(source_files: Mapping[str, bytes]) -> dict[str, bytes]:
    legal_files = {
        relative_name: payload
        for relative_name, payload in source_files.items()
        if relative_name in {"LICENSE", "THIRD_PARTY_NOTICES.md"}
        or relative_name.startswith("LICENSES/")
    }
    required_roots = {"LICENSE", "THIRD_PARTY_NOTICES.md"}
    if not required_roots.issubset(legal_files) or not any(
        relative_name.startswith("LICENSES/") for relative_name in legal_files
    ):
        raise DistributionInspectionError(
            "Tracked source is missing the required LICENSE, LICENSES/, or "
            "THIRD_PARTY_NOTICES.md legal inventory."
        )
    return legal_files


def _assert_exact_content_inventory(
    actual: Mapping[str, bytes],
    expected: Mapping[str, bytes],
    *,
    label: str,
) -> None:
    missing = sorted(set(expected).difference(actual))
    unexpected = sorted(set(actual).difference(expected))
    if missing or unexpected:
        raise DistributionInspectionError(
            f"{label} inventory differs from tracked Git source: "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    mismatched = sorted(name for name, payload in expected.items() if actual[name] != payload)
    if mismatched:
        raise DistributionInspectionError(
            f"{label} members differ from tracked Git blobs: {mismatched[:20]}"
        )


def _assert_wheel_legal_identity(
    contents: Mapping[str, bytes],
    source_files: Mapping[str, bytes],
    *,
    dist_info: str,
) -> None:
    source_legal = _tracked_legal_files(source_files)
    expected = {
        f"{dist_info}/licenses/{relative_name}": payload
        for relative_name, payload in source_legal.items()
    }
    prefix = f"{dist_info}/licenses/"
    actual = {name: payload for name, payload in contents.items() if name.startswith(prefix)}
    _assert_exact_content_inventory(actual, expected, label="Wheel legal")

    metadata_name = f"{dist_info}/METADATA"
    try:
        metadata_text = contents[metadata_name].decode("utf-8")
    except (KeyError, UnicodeDecodeError) as error:
        raise DistributionInspectionError("Wheel METADATA is missing or is not UTF-8.") from error
    declared = [
        line.removeprefix("License-File: ")
        for line in metadata_text.splitlines()
        if line.startswith("License-File: ")
    ]
    expected_declarations = sorted(source_legal)
    if sorted(declared) != expected_declarations or len(declared) != len(set(declared)):
        raise DistributionInspectionError(
            "Wheel METADATA License-File inventory differs from tracked Git source: "
            f"expected={expected_declarations[:20]}, received={sorted(declared)[:20]}"
        )


def _assert_sdist_legal_identity(
    contents: Mapping[str, bytes],
    source_files: Mapping[str, bytes],
) -> None:
    expected = _tracked_legal_files(source_files)
    actual = {
        relative_name: payload
        for relative_name, payload in contents.items()
        if relative_name in {"LICENSE", "THIRD_PARTY_NOTICES.md"}
        or relative_name.startswith("LICENSES/")
    }
    _assert_exact_content_inventory(actual, expected, label="Source-distribution legal")


def _assert_wheel_source_identity(
    contents: Mapping[str, bytes],
    source_files: Mapping[str, bytes],
) -> None:
    expected = {
        relative_name.removeprefix("src/"): payload
        for relative_name, payload in source_files.items()
        if relative_name.startswith("src/fastplms/")
    }
    actual = {
        relative_name: payload
        for relative_name, payload in contents.items()
        if relative_name.startswith("fastplms/")
    }
    if set(actual) != set(expected):
        raise DistributionInspectionError(
            "Wheel runtime inventory differs from tracked Git source: "
            f"missing={sorted(set(expected).difference(actual))[:20]}, "
            f"unexpected={sorted(set(actual).difference(expected))[:20]}"
        )
    mismatched = sorted(
        relative_name
        for relative_name, expected_payload in expected.items()
        if actual[relative_name] != expected_payload
    )
    if mismatched:
        raise DistributionInspectionError(
            f"Wheel runtime members differ from tracked Git blobs: {mismatched[:20]}"
        )


def _is_tracked_sdist_scope(relative_name: str) -> bool:
    return (
        relative_name in {
            "pyproject.toml",
            "README.md",
            "kernels.lock",
            "LICENSE",
            "THIRD_PARTY_NOTICES.md",
        }
        or relative_name.startswith("src/fastplms/")
        or relative_name.startswith("LICENSES/")
    )


def _assert_sdist_source_identity(
    contents: Mapping[str, bytes],
    source_files: Mapping[str, bytes],
) -> None:
    actual = {
        relative_name: payload
        for relative_name, payload in contents.items()
        if _is_tracked_sdist_scope(relative_name)
    }
    if set(actual) != set(source_files):
        raise DistributionInspectionError(
            "Source-distribution inventory differs from tracked Git source: "
            f"missing={sorted(set(source_files).difference(actual))[:20]}, "
            f"unexpected={sorted(set(actual).difference(source_files))[:20]}"
        )
    mismatched = sorted(
        relative_name
        for relative_name, expected_payload in source_files.items()
        if actual[relative_name] != expected_payload
    )
    if mismatched:
        raise DistributionInspectionError(
            "Source-distribution members differ from tracked Git blobs: "
            f"{mismatched[:20]}"
        )


def inspect_wheel(
    path: Path,
    *,
    project_root: Path,
    _source_snapshot: SourceSnapshot | None = None,
) -> dict[str, Any]:
    """Inspect one wheel without extracting it."""

    if path.name != "fastplms-1.0.0-py3-none-any.whl":
        raise DistributionInspectionError(
            f"FastPLMs must remain a py3-none-any wheel, found {path.name!r}"
        )
    dist_info = "fastplms-1.0.0.dist-info"
    records: list[tuple[str, int, str]] = []
    contents: dict[str, bytes] = {}
    with zipfile.ZipFile(path) as archive:
        seen: set[str] = set()
        for info in archive.infolist():
            name = _normalized_member(info.filename)
            if not (
                name in {"fastplms", dist_info}
                or name.startswith("fastplms/")
                or name.startswith(f"{dist_info}/")
            ):
                raise DistributionInspectionError(
                    f"Wheel member is outside the package allowlist: {name!r}"
                )
            if name in seen:
                raise DistributionInspectionError(f"Wheel contains duplicate member: {name!r}")
            seen.add(name)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise DistributionInspectionError(f"Wheel contains a symlink: {name!r}")
            if info.is_dir():
                continue
            if info.file_size > _MAX_MEMBER_BYTES:
                raise DistributionInspectionError(
                    f"Wheel member exceeds the 32 MiB limit: {name!r}"
                )
            content = archive.read(info)
            contents[name] = content
            records.append((name, len(content), hashlib.sha256(content).hexdigest()))

        if sum(size for _, size, _ in records) > _MAX_TOTAL_BYTES:
            raise DistributionInspectionError("Wheel exceeds the 128 MiB unpacked limit")

        _require_members(
            seen,
            {
                "fastplms/__init__.py",
                "fastplms/models.toml",
                f"{dist_info}/METADATA",
                f"{dist_info}/RECORD",
                f"{dist_info}/WHEEL",
                f"{dist_info}/kernels.lock",
                f"{dist_info}/licenses/LICENSE",
                f"{dist_info}/licenses/THIRD_PARTY_NOTICES.md",
            },
            kind="wheel",
        )
        wheel_metadata = archive.read(f"{dist_info}/WHEEL").decode("utf-8")
        if "Root-Is-Purelib: true" not in wheel_metadata.splitlines():
            raise DistributionInspectionError("Wheel does not declare a purelib root")
        if "Tag: py3-none-any" not in wheel_metadata.splitlines():
            raise DistributionInspectionError("Wheel does not declare the py3-none-any tag")
    source_revision, source_files, source_tree_sha256 = (
        _source_snapshot or _tracked_source_snapshot(project_root)
    )
    expected_kernel_lock = source_files.get("kernels.lock")
    if expected_kernel_lock is None:
        raise DistributionInspectionError("Tracked source is missing kernels.lock.")
    kernel_member = f"{dist_info}/kernels.lock"
    if contents[kernel_member] != expected_kernel_lock:
        raise DistributionInspectionError(
            f"Wheel member differs from the tracked source: {kernel_member!r}"
        )
    _assert_wheel_source_identity(contents, source_files)
    _assert_wheel_legal_identity(contents, source_files, dist_info=dist_info)

    return {
        "kind": "wheel",
        "filename": path.name,
        "sha256": _sha256(path),
        "size": path.stat().st_size,
        "member_count": len(records),
        "member_bytes": sum(size for _, size, _ in records),
        "inventory_sha256": _inventory_digest(records),
        "source_revision": source_revision,
        "source_tree_sha256": source_tree_sha256,
    }


def inspect_sdist(
    path: Path,
    *,
    project_root: Path,
    _source_snapshot: SourceSnapshot | None = None,
) -> dict[str, Any]:
    """Inspect one gzip-compressed source distribution without extracting it."""

    records: list[tuple[str, int, str]] = []
    with tarfile.open(path, mode="r:gz") as archive:
        members = archive.getmembers()
        roots = {PurePosixPath(member.name).parts[0] for member in members if member.name}
        if roots != {"fastplms-1.0.0"}:
            raise DistributionInspectionError(
                f"Source distribution has an unexpected top-level layout: {sorted(roots)}"
            )
        seen: set[str] = set()
        contents: dict[str, bytes] = {}
        for member in members:
            if member.name.removesuffix("/") == "fastplms-1.0.0" and member.isdir():
                continue
            name = _normalized_member(member.name, root="fastplms-1.0.0")
            if name in seen:
                raise DistributionInspectionError(
                    f"Source distribution contains duplicate member: {name!r}"
                )
            seen.add(name)
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise DistributionInspectionError(
                    f"Source distribution contains a non-regular member: {name!r}"
                )
            if member.isdir():
                continue
            if not member.isfile():
                raise DistributionInspectionError(
                    f"Source distribution contains an unknown member type: {name!r}"
                )
            if member.size > _MAX_MEMBER_BYTES:
                raise DistributionInspectionError(
                    f"Source distribution member exceeds the 32 MiB limit: {name!r}"
                )
            stream = archive.extractfile(member)
            if stream is None:
                raise DistributionInspectionError(
                    f"Source distribution member cannot be read: {name!r}"
                )
            content = stream.read()
            contents[name] = content
            records.append((name, len(content), hashlib.sha256(content).hexdigest()))

        if sum(size for _, size, _ in records) > _MAX_TOTAL_BYTES:
            raise DistributionInspectionError(
                "Source distribution exceeds the 128 MiB unpacked limit"
            )

        _require_members(
            seen,
            {
                "LICENSE",
                "README.md",
                "THIRD_PARTY_NOTICES.md",
                "kernels.lock",
                "pyproject.toml",
                "src/fastplms/__init__.py",
                "src/fastplms/models.toml",
            },
            kind="source distribution",
        )
    source_revision, source_files, source_tree_sha256 = (
        _source_snapshot or _tracked_source_snapshot(project_root)
    )
    _assert_sdist_legal_identity(contents, source_files)
    _assert_sdist_source_identity(contents, source_files)

    return {
        "kind": "sdist",
        "filename": path.name,
        "sha256": _sha256(path),
        "size": path.stat().st_size,
        "member_count": len(records),
        "member_bytes": sum(size for _, size, _ in records),
        "inventory_sha256": _inventory_digest(records),
        "source_revision": source_revision,
        "source_tree_sha256": source_tree_sha256,
    }


def inspect_distributions(dist: Path, *, project_root: Path) -> dict[str, Any]:
    """Return a deterministic attestation for exactly one wheel and one sdist."""

    wheels = sorted(dist.glob("fastplms-1.0.0-*.whl"))
    sdists = sorted(dist.glob("fastplms-1.0.0.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise DistributionInspectionError(
            "Expected exactly one FastPLMs 1.0.0 wheel and one source distribution; "
            f"found wheels={len(wheels)}, sdists={len(sdists)}"
        )
    unexpected = sorted(
        path.name
        for path in dist.iterdir()
        if path.is_file() and path not in {wheels[0], sdists[0]}
    )
    if unexpected:
        raise DistributionInspectionError(
            f"Distribution directory contains extra files: {unexpected}"
        )
    source_snapshot = _tracked_source_snapshot(project_root)
    source_revision, source_files, source_tree_sha256 = source_snapshot
    return {
        "schema_version": 2,
        "project": "fastplms",
        "version": "1.0.0",
        "source": {
            "revision": source_revision,
            "tree_sha256": source_tree_sha256,
            "tracked_file_count": len(source_files),
        },
        "artifacts": [
            inspect_wheel(
                wheels[0],
                project_root=project_root,
                _source_snapshot=source_snapshot,
            ),
            inspect_sdist(
                sdists[0],
                project_root=project_root,
                _source_snapshot=source_snapshot,
            ),
        ],
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/distribution-inspection.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    payload = inspect_distributions(
        arguments.dist.resolve(),
        project_root=arguments.project_root.resolve(),
    )
    _atomic_json(arguments.output, payload)
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
