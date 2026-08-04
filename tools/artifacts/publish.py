"""Publish validated artifacts through parent-guarded Hub commits.

Runtime-only updates preserve the exact bytes validated during preflight. A
complete update includes weights and both attestations in one atomic commit and
may remove only obsolete registry-pinned paths. Neither mode creates
repositories, removes unpinned remote files, or changes settings.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

from fastplms.registry import FileDigest, ModelRegistry, ModelSpec, get_model_registry
from tools.artifacts.build import (
    _RUNTIME_ATTESTATION_NAME,
    _RUNTIME_ATTESTATION_SCHEMA_VERSION,
    _WEIGHT_INDEX,
    _WEIGHT_SUFFIXES,
    ArtifactError,
    _artifact_auto_map,
    _checkpoint_identity_hash,
    _is_runtime_update_path,
    _materialize_model_card,
    _render_artifact_requirements,
    _resolve_artifact_manifest_path,
    _tree_sha256,
    _validate_bootstrap,
    _validate_registry_provenance,
    _validate_runtime_bundle,
    _validated_release_tool_snapshot,
    _validated_runtime_snapshot,
    hash_file,
    render_model_card,
    validate_artifact,
)


_COMPLETE_ATTESTATION_FILES = frozenset({"artifact-manifest.json", "source-record.json"})
_REQUIRED_FILES_ONLY_PATHS = frozenset(
    {
        "README.md",
        "config.json",
        "fastplms_bundle.py",
        "modeling_fastplms.py",
        "requirements.txt",
        "THIRD_PARTY_NOTICES.md",
        "LICENSES/FastPLMs-Apache-2.0.txt",
        _RUNTIME_ATTESTATION_NAME,
    }
)
_GENERATED_RUNTIME_PATHS = frozenset(
    {
        "README.md",
        "config.json",
        "fastplms_bundle.py",
        "modeling_fastplms.py",
        "requirements.txt",
        "THIRD_PARTY_NOTICES.md",
        _RUNTIME_ATTESTATION_NAME,
    }
)
_RUNTIME_SUFFIXES = frozenset({".json", ".lock", ".py", ".toml"})
_SENSITIVE_NAMES = frozenset(
    {
        ".env",
        ".netrc",
        "credentials",
        "credentials.json",
        "id_ed25519",
        "id_rsa",
        "secrets.json",
        "token",
        "token.txt",
    }
)
_SENSITIVE_SUFFIXES = frozenset({".key", ".p12", ".pfx", ".pem"})
_SENSITIVE_STEMS = frozenset(
    {"credential", "credentials", "secret", "secrets", "token"}
)
_MAX_RUNTIME_FILE_BYTES = 128 * 1024**2
_MAX_RETAINED_COMPLETE_BYTES = 128 * 1024**2
_MAX_DECLARED_ASSET_BYTES = 2 * 1024**3
_MAX_RELEASE_TEXT_BYTES = 8 * 1024**2
_REQUIRED_COMPLETE_AUTOMODEL_VIEWS = ("AutoModel", "AutoModelForSeq2SeqLM")


@dataclass(frozen=True)
class FilesOnlyPublishPlan:
    """One preflighted add-only commit for a manifest-declared model."""

    model_id: str
    repo_id: str
    revision: str
    parent_commit: str
    artifact_path: Path
    files: tuple[str, ...]
    payloads: tuple[tuple[str, bytes], ...]
    runtime_revision: str
    source_tree_sha256: str
    runtime_bundle_sha256: str
    release_tool_revision: str
    release_tool_sha256: str
    release_revision: str
    release_source_sha256: str


@dataclass(frozen=True)
class CompletePublishPlan:
    """One complete, atomic weights-plus-runtime commit."""

    model_id: str
    repo_id: str
    revision: str
    parent_commit: str
    artifact_path: Path
    files: tuple[str, ...]
    digests: tuple[tuple[str, str], ...]
    deletes: tuple[str, ...] = ()
    replacement_weight_paths: tuple[str, ...] = ()
    runtime_revision: str | None = None
    source_tree_sha256: str | None = None
    runtime_bundle_sha256: str | None = None
    release_tool_revision: str | None = None
    release_tool_sha256: str | None = None
    release_revision: str | None = None
    release_source_sha256: str | None = None
    validation_manifest_sha256: str | None = None
    validated_auto_classes: tuple[str, ...] = ()


@dataclass(frozen=True)
class FilesOnlyPublishResult:
    """Identity of one completed Hub commit."""

    model_id: str
    repo_id: str
    commit_oid: str
    commit_url: str


def _is_weight_path(path: str) -> bool:
    relative = PurePosixPath(path)
    name = relative.name.lower()
    return (
        relative.suffix.lower() in _WEIGHT_SUFFIXES
        or name == _WEIGHT_INDEX
        or name.endswith(".safetensors.index.json")
        or name.endswith(".bin.index.json")
    )


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"Unable to read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ArtifactError(f"{label} must contain a JSON object: {path}")
    return value


def _encoded_digest(path: Path, encoded: str) -> str:
    try:
        algorithm, expected = encoded.split(":", maxsplit=1)
    except ValueError as error:
        raise ArtifactError(f"Invalid artifact digest for {path}: {encoded!r}") from error
    actual = hash_file(path, algorithm)
    if actual != expected:
        raise ArtifactError(
            f"Artifact file digest differs for {path}: expected {expected}, received {actual}."
        )
    return actual


def _read_validated_bytes(
    path: Path,
    encoded: str,
    *,
    max_bytes: int | None = None,
) -> bytes:
    """Read once and validate the exact immutable payload retained by a plan."""

    try:
        algorithm, expected = encoded.split(":", maxsplit=1)
    except ValueError as error:
        raise ArtifactError(f"Invalid artifact digest for {path}: {encoded!r}") from error
    with path.open("rb") as handle:
        payload = handle.read(max_bytes + 1 if max_bytes is not None else -1)
    if max_bytes is not None and len(payload) > max_bytes:
        raise ArtifactError(
            f"Artifact payload exceeds its {max_bytes}-byte retained publication limit: {path}"
        )
    if algorithm == "sha256":
        actual = hashlib.sha256(payload).hexdigest()
    elif algorithm == "git-sha1":
        digest = hashlib.sha1(usedforsecurity=False)
        digest.update(f"blob {len(payload)}\0".encode("ascii"))
        digest.update(payload)
        actual = digest.hexdigest()
    else:
        raise ArtifactError(f"Unsupported artifact digest algorithm: {algorithm!r}")
    if actual != expected:
        raise ArtifactError(
            f"Artifact file digest differs for {path}: expected {expected}, received {actual}."
        )
    return payload


def _open_validated_payload(
    handle: BinaryIO,
    path: Path,
    encoded: str,
) -> tuple[int, int, int, int]:
    """Hash one retained open file and return its stable filesystem identity."""

    try:
        algorithm, expected = encoded.split(":", maxsplit=1)
    except ValueError as error:
        raise ArtifactError(f"Invalid artifact digest for {path}: {encoded!r}") from error
    before = os.fstat(handle.fileno())
    if algorithm == "sha256":
        digest = hashlib.sha256()
    elif algorithm == "git-sha1":
        digest = hashlib.sha1(usedforsecurity=False)
        digest.update(f"blob {before.st_size}\0".encode("ascii"))
    else:
        raise ArtifactError(f"Unsupported artifact digest algorithm: {algorithm!r}")
    handle.seek(0)
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    after = os.fstat(handle.fileno())
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ArtifactError(f"Artifact file changed while it was hashed: {path}")
    actual = digest.hexdigest()
    if actual != expected:
        raise ArtifactError(
            f"Artifact file digest differs for {path}: expected {expected}, received {actual}."
        )
    handle.seek(0)
    return identity


def _snapshot_validated_payload(
    source: Path,
    destination: Path,
    encoded: str,
) -> None:
    """Copy one large payload into publisher-owned storage while hashing it."""

    try:
        algorithm, expected = encoded.split(":", maxsplit=1)
    except ValueError as error:
        raise ArtifactError(f"Invalid artifact digest for {source}: {encoded!r}") from error
    try:
        with source.open("rb") as reader, destination.open("xb") as writer:
            before = os.fstat(reader.fileno())
            if algorithm == "sha256":
                digest = hashlib.sha256()
            elif algorithm == "git-sha1":
                digest = hashlib.sha1(usedforsecurity=False)
                digest.update(f"blob {before.st_size}\0".encode("ascii"))
            else:
                raise ArtifactError(
                    f"Unsupported artifact digest algorithm: {algorithm!r}"
                )
            for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                digest.update(chunk)
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
            after = os.fstat(reader.fileno())
    except ArtifactError:
        raise
    except OSError as error:
        raise ArtifactError(f"Unable to snapshot complete artifact payload: {source}") from error
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ArtifactError(f"Artifact file changed while it was snapshotted: {source}")
    actual = digest.hexdigest()
    if actual != expected:
        raise ArtifactError(
            f"Artifact file digest differs for {source}: expected {expected}, received {actual}."
        )


def _canonical_weight_paths(provenance: Mapping[str, Any]) -> frozenset[str]:
    canonical = provenance.get("canonical_weights")
    if not isinstance(canonical, Mapping):
        raise ArtifactError("Artifact provenance is missing canonical_weights.")
    index = canonical.get("index")
    shards = canonical.get("shards")
    if not isinstance(index, str) or not isinstance(shards, Mapping):
        raise ArtifactError("Artifact provenance has invalid canonical weight metadata.")
    if any(not isinstance(path, str) for path in shards):
        raise ArtifactError("Artifact provenance contains an invalid canonical shard path.")
    return frozenset({index, *shards})


def _assert_current_registry_spec(spec: ModelSpec) -> ModelRegistry:
    registry = get_model_registry()
    try:
        current = registry[spec.id]
    except KeyError as error:
        raise ArtifactError(f"Model {spec.id!r} is absent from the current registry.") from error
    if current != spec:
        raise ArtifactError(
            f"Model {spec.id!r} differs from the current registry; rebuild the plan."
        )
    return registry


def _sensitive_path(relative_name: str) -> bool:
    relative = PurePosixPath(relative_name)
    lowered = tuple(part.lower() for part in relative.parts)
    return (
        any(part in _SENSITIVE_NAMES for part in lowered)
        or any(PurePosixPath(part).stem in _SENSITIVE_STEMS for part in lowered)
        or relative.suffix.lower() in _SENSITIVE_SUFFIXES
        or any(part in {".git", "__pycache__"} for part in lowered)
    )


def _declared_non_weight_assets(
    spec: ModelSpec,
    registry: ModelRegistry,
) -> frozenset[str]:
    paths = {
        item.path for item in spec.artifact_checkpoint.files if not _is_weight_path(item.path)
    }
    tokenizer_source = (
        registry[spec.tokenizer_source_id].official
        if spec.tokenizer_source_id is not None
        else spec.official
    )
    paths.update(item.path for item in tokenizer_source.files if not _is_weight_path(item.path))
    return frozenset(paths)


def _declared_legal_paths(
    spec: ModelSpec,
    registry: ModelRegistry,
) -> frozenset[str]:
    paths = {"LICENSES/FastPLMs-Apache-2.0.txt"}
    for source_id in spec.family.upstreams:
        source = registry.upstreams[source_id]
        paths.update(
            f"LICENSES/{source_id}/{item.path}" for item in source.distribution_files
        )
    return frozenset(paths)


def _canonical_release_bytes(raw: bytes, path: Path) -> bytes:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ArtifactError(f"Release text must be valid UTF-8: {path}") from error
    if "\x00" in text:
        raise ArtifactError(f"Release text contains a NUL byte: {path}")
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def _validated_release_text_snapshot(
    spec: ModelSpec,
    registry: ModelRegistry,
    *,
    runtime_revision: str,
    source_tree_sha256: str,
    runtime_bundle_sha256: str,
) -> tuple[str, str, dict[str, bytes]]:
    """Return release texts from one clean, tracked immutable Git revision."""

    source_root = Path(__file__).resolve().parents[2]
    _validated_release_tool_snapshot(source_root)
    git_metadata = source_root / ".git"
    if not (git_metadata.exists() or git_metadata.is_symlink()):
        raise ArtifactError("Publication release texts require a verifiable Git worktree.")
    command_prefix = [
        "git",
        "-c",
        f"safe.directory={source_root.as_posix()}",
    ]
    card_source = source_root / "model_cards" / f"{spec.id}.md"
    card_relative = card_source.relative_to(source_root).as_posix()
    try:
        revision = subprocess.run(
            [*command_prefix, "rev-parse", "HEAD"],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        card_tree = subprocess.run(
            [*command_prefix, "ls-tree", "-z", revision, "--", card_relative],
            cwd=source_root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ArtifactError("Unable to inspect the tracked publication model card.") from error
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ArtifactError(f"Git returned an invalid release revision: {revision!r}")
    card_entries = [entry for entry in card_tree.split(b"\0") if entry]
    if len(card_entries) > 1:
        raise ArtifactError("Git returned ambiguous publication model-card entries.")
    generated_card = not card_entries
    if card_entries:
        try:
            metadata, archived_name = card_entries[0].decode("utf-8").split("\t", maxsplit=1)
            mode, kind, _ = metadata.split(maxsplit=2)
        except (UnicodeDecodeError, ValueError) as error:
            raise ArtifactError("Git returned invalid publication model-card metadata.") from error
        if mode != "100644" or kind != "blob" or archived_name != card_relative:
            raise ArtifactError("The tracked publication model card is not a regular file.")
    elif card_source.exists() or card_source.is_symlink():
        raise ArtifactError(
            f"Publication model card is untracked at the validated revision: {card_source}"
        )
    sources: dict[str, Path] = {
        "LICENSES/FastPLMs-Apache-2.0.txt": source_root / "LICENSE",
        "THIRD_PARTY_NOTICES.md": source_root / "THIRD_PARTY_NOTICES.md",
    }
    if generated_card:
        sources[".source/tools/artifacts/build.py"] = source_root.joinpath(
            "tools", "artifacts", "build.py"
        )
        sources[".source/tools/artifacts/generate_docs.py"] = source_root.joinpath(
            "tools", "artifacts", "generate_docs.py"
        )
    else:
        sources["README.md"] = card_source
    for source_id in spec.family.upstreams:
        source = registry.upstreams[source_id]
        for item in source.distribution_files:
            sources[f"LICENSES/{source_id}/{item.path}"] = source_root.joinpath(
                "LICENSES",
                source_id,
                *PurePosixPath(item.path).parts,
            )
    relative_sources: dict[str, str] = {}
    for artifact_name, path in sources.items():
        if path.is_symlink() or not path.is_file():
            raise ArtifactError(
                f"Publication release source must be a regular non-symlink file: {path}"
            )
        try:
            size = path.stat().st_size
        except OSError as error:
            raise ArtifactError(f"Unable to inspect publication release source: {path}") from error
        if size > _MAX_RELEASE_TEXT_BYTES:
            raise ArtifactError(
                f"Publication release source exceeds {_MAX_RELEASE_TEXT_BYTES} bytes: {path}"
            )
        try:
            relative_sources[path.relative_to(source_root).as_posix()] = artifact_name
        except ValueError as error:
            raise ArtifactError(f"Release source escapes the repository: {path}") from error
    if len(relative_sources) != len(sources):
        raise ArtifactError("Release source paths contain an ambiguous duplicate.")
    source_names = tuple(sorted(relative_sources))
    try:
        status = subprocess.run(
            [
                *command_prefix,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                *source_names,
            ],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if status:
            raise ArtifactError(
                "Publication release texts must be tracked and clean; scoped Git status: "
                + status.replace("\n", "; ")
            )
        tracked = subprocess.run(
            [*command_prefix, "ls-files", "-z", "--", *source_names],
            cwd=source_root,
            check=True,
            capture_output=True,
        ).stdout
        tracked_names = {raw.decode("utf-8") for raw in tracked.split(b"\0") if raw}
        if tracked_names != set(source_names):
            raise ArtifactError("Publication release texts contain untracked or missing files.")
        archive = subprocess.run(
            [
                *command_prefix,
                "archive",
                "--format=tar",
                revision,
                "--",
                *source_names,
            ],
            cwd=source_root,
            check=True,
            capture_output=True,
        ).stdout
        archived: dict[str, bytes] = {}
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as handle:
            for member in handle.getmembers():
                if member.isdir():
                    continue
                if not member.isfile() or member.name in archived:
                    raise ArtifactError(
                        f"Tracked release source is not a unique regular file: {member.name}"
                    )
                extracted = handle.extractfile(member)
                if extracted is None:
                    raise ArtifactError(f"Unable to read tracked release source: {member.name}")
                payload = extracted.read(_MAX_RELEASE_TEXT_BYTES + 1)
                if len(payload) > _MAX_RELEASE_TEXT_BYTES:
                    raise ArtifactError(
                        f"Tracked release source exceeds {_MAX_RELEASE_TEXT_BYTES} bytes: "
                        f"{member.name}"
                    )
                archived[PurePosixPath(member.name).as_posix()] = payload
    except ArtifactError:
        raise
    except (
        OSError,
        subprocess.CalledProcessError,
        UnicodeDecodeError,
        tarfile.TarError,
    ) as error:
        raise ArtifactError("Unable to validate tracked publication release texts.") from error
    if set(archived) != set(source_names):
        raise ArtifactError("Tracked release archive differs from the validated source allowlist.")
    payloads: dict[str, bytes] = {}
    identity_payloads: dict[str, bytes] = {}
    for source_name, artifact_name in relative_sources.items():
        raw = archived[source_name]
        identity_payloads[artifact_name] = raw
        if not artifact_name.startswith(".source/"):
            payloads[artifact_name] = (
                raw
                if artifact_name == "README.md"
                else _canonical_release_bytes(raw, source_root / source_name)
            )
    if generated_card:
        payloads["README.md"] = render_model_card(spec).encode("utf-8")
    try:
        card_template = payloads["README.md"].decode("utf-8")
    except UnicodeDecodeError as error:
        raise ArtifactError("Publication model-card template is not valid UTF-8.") from error
    payloads["README.md"] = _materialize_model_card(
        card_template,
        runtime_revision=runtime_revision,
        source_tree_sha256=source_tree_sha256,
        runtime_bundle_sha256=runtime_bundle_sha256,
    ).encode("utf-8")
    identity_payloads["README.md"] = payloads["README.md"]
    identity = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in identity_payloads.items()
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return revision, digest, payloads


def _assert_current_release_texts(
    artifact_path: Path,
    spec: ModelSpec,
    registry: ModelRegistry,
) -> tuple[str, str]:
    """Bind cards and legal texts to the current validated source tree."""

    provenance = _load_json_object(artifact_path / "source-record.json", "source-record.json")
    runtime_revision = provenance.get("runtime_revision")
    source_tree_sha256 = provenance.get("source_tree_sha256")
    runtime_bundle_sha256 = provenance.get("runtime_bundle_sha256")
    if (
        not isinstance(runtime_revision, str)
        or not isinstance(source_tree_sha256, str)
        or not isinstance(runtime_bundle_sha256, str)
    ):
        raise ArtifactError("Artifact provenance lacks a complete model-card runtime identity.")
    release_revision, release_digest, expected = _validated_release_text_snapshot(
        spec,
        registry,
        runtime_revision=runtime_revision,
        source_tree_sha256=source_tree_sha256,
        runtime_bundle_sha256=runtime_bundle_sha256,
    )
    try:
        actual_card = (artifact_path / "README.md").read_bytes()
    except OSError as error:
        raise ArtifactError("Artifact model card is missing.") from error
    if b"<runtime-revision>" in actual_card:
        raise ArtifactError(
            "Artifact model card retains an unresolved runtime-revision placeholder."
        )
    if actual_card != expected["README.md"]:
        raise ArtifactError("Artifact model card differs from the current source tree.")
    for relative_name, expected_payload in expected.items():
        if relative_name == "README.md":
            continue
        path = artifact_path.joinpath(*PurePosixPath(relative_name).parts)
        try:
            actual = path.read_bytes()
        except OSError as error:
            raise ArtifactError(
                f"Artifact is missing current legal text {relative_name!r}."
            ) from error
        if actual != expected_payload:
            raise ArtifactError(
                f"Artifact legal text differs from the current source: {relative_name!r}."
            )
    return release_revision, release_digest


def _validate_publishable_non_weight_path(
    relative_name: str,
    path: Path,
    *,
    declared_assets: frozenset[str],
    declared_legal_paths: frozenset[str],
) -> None:
    if _sensitive_path(relative_name):
        raise ArtifactError(f"Artifact contains a sensitive publication path: {relative_name!r}")
    relative = PurePosixPath(relative_name)
    size = path.stat().st_size
    allowed = False
    size_limit = _MAX_RUNTIME_FILE_BYTES
    if relative_name in _GENERATED_RUNTIME_PATHS:
        allowed = True
    elif relative.parts and relative.parts[0] == "fastplms":
        allowed = relative.suffix.lower() in _RUNTIME_SUFFIXES
    elif relative_name in declared_legal_paths:
        allowed = True
    elif relative_name in declared_assets:
        allowed = True
        size_limit = _MAX_DECLARED_ASSET_BYTES
    if not allowed:
        raise ArtifactError(
            f"Artifact path is outside the publication allowlist: {relative_name!r}"
        )
    if size > size_limit:
        raise ArtifactError(
            f"Artifact path exceeds its {size_limit}-byte publication limit: {relative_name!r}"
        )


def _artifact_inventory(root: Path) -> frozenset[str]:
    result: set[str] = set()
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ArtifactError(f"Symlinks are forbidden in publication artifacts: {path}")
        if not path.is_file():
            continue
        relative_name = path.relative_to(root).as_posix()
        if _sensitive_path(relative_name):
            raise ArtifactError(
                f"Artifact contains a sensitive publication path: {relative_name!r}"
            )
        result.add(relative_name)
    return frozenset(result)


def _expected_runtime_assets(spec: ModelSpec, registry: ModelRegistry) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for asset in registry.runtime_assets.values():
        if asset.consumer_family != spec.family.id:
            continue
        material = (
            f"{asset.repository}@{asset.revision}:{asset.path}:{asset.sha256}:{asset.size}"
        ).encode()
        records.append(
            {
                "id": asset.id,
                "repository": asset.repository,
                "revision": asset.revision,
                "path": asset.path,
                "sha256": asset.sha256,
                "size": asset.size,
                "license": asset.license_expression,
                "consumer_family": asset.consumer_family,
                "trust_kind": asset.trust_kind,
                "offline_behavior": asset.offline_behavior,
                "cache_identity": hashlib.sha256(material).hexdigest(),
            }
        )
    return records


def _assert_current_runtime_source(
    spec: ModelSpec,
    registry: ModelRegistry,
    provenance: Mapping[str, Any],
) -> tuple[str, str]:
    """Bind one artifact to the current clean tracked runtime source scope."""

    source_root = Path(__file__).resolve().parents[2]
    runtime_revision, _, source_tree_sha256 = _validated_runtime_snapshot(
        source_root,
        registry,
        spec,
    )
    if provenance.get("runtime_revision") != runtime_revision:
        raise ArtifactError(
            "Artifact runtime revision differs from the current clean source revision; "
            "rebuild it."
        )
    if provenance.get("source_tree_sha256") != source_tree_sha256:
        raise ArtifactError(
            "Artifact runtime source-tree digest differs from the current tracked sources; "
            "rebuild it."
        )
    return runtime_revision, source_tree_sha256


def _assert_current_release_tool_source(
    provenance: Mapping[str, Any],
) -> tuple[str, str, dict[str, bytes]]:
    """Bind one artifact to the current immutable release-tool scope."""

    source_root = Path(__file__).resolve().parents[2]
    tool_revision, tool_sha256, payloads = _validated_release_tool_snapshot(source_root)
    if provenance.get("release_tool_revision") != tool_revision:
        raise ArtifactError(
            "Artifact release-tool revision differs from the current clean tool scope; "
            "rebuild it."
        )
    if provenance.get("release_tool_sha256") != tool_sha256:
        raise ArtifactError(
            "Artifact release-tool digest differs from the current clean tool scope; "
            "rebuild it."
        )
    return tool_revision, tool_sha256, payloads


def _assert_artifact_requirements(
    artifact_path: Path,
    spec: ModelSpec,
    release_tool_payloads: Mapping[str, bytes],
) -> None:
    expected = _render_artifact_requirements(spec, release_tool_payloads).encode("utf-8")
    path = artifact_path / "requirements.txt"
    try:
        actual = path.read_bytes()
    except OSError as error:
        raise ArtifactError(f"Artifact requirements are missing: {path}") from error
    if len(actual) > _MAX_RELEASE_TEXT_BYTES:
        raise ArtifactError("Artifact requirements exceed the release-text size limit.")
    if actual != expected:
        raise ArtifactError(
            "Artifact requirements differ from the current direct dependency contract."
        )


def _revalidate_plan_runtime_source(
    *,
    model_id: str,
    repo_id: str,
    runtime_revision: str | None,
    source_tree_sha256: str | None,
    runtime_bundle_sha256: str | None,
    release_tool_revision: str | None,
    release_tool_sha256: str | None,
    release_revision: str | None,
    release_source_sha256: str | None,
) -> None:
    """Reject a plan if scoped runtime or release bytes changed after preflight."""

    if (
        runtime_revision is None
        and source_tree_sha256 is None
        and runtime_bundle_sha256 is None
        and release_tool_revision is None
        and release_tool_sha256 is None
        and release_revision is None
        and release_source_sha256 is None
    ):
        return
    if (
        runtime_revision is None
        or source_tree_sha256 is None
        or runtime_bundle_sha256 is None
        or release_tool_revision is None
        or release_tool_sha256 is None
        or release_revision is None
        or release_source_sha256 is None
    ):
        raise ArtifactError(f"Publication plan for {repo_id} has partial source identity.")
    registry = get_model_registry()
    spec = registry.get(model_id)
    if spec is None or spec.fast.repo_id != repo_id:
        raise ArtifactError(f"Publication plan identity is stale for {repo_id}.")
    current_revision, _, current_source_tree_sha256 = _validated_runtime_snapshot(
        Path(__file__).resolve().parents[2],
        registry,
        spec,
    )
    if (
        current_revision != runtime_revision
        or current_source_tree_sha256 != source_tree_sha256
    ):
        raise ArtifactError(
            f"Scoped runtime sources changed after publication preflight for {repo_id}."
        )
    current_tool_revision, current_tool_sha256, _tool_payloads = (
        _validated_release_tool_snapshot(Path(__file__).resolve().parents[2])
    )
    if (
        current_tool_revision != release_tool_revision
        or current_tool_sha256 != release_tool_sha256
    ):
        raise ArtifactError(
            f"Release tools changed after publication preflight for {repo_id}."
        )
    current_release_revision, current_release_source_sha256, _ = (
        _validated_release_text_snapshot(
            spec,
            registry,
            runtime_revision=runtime_revision,
            source_tree_sha256=source_tree_sha256,
            runtime_bundle_sha256=runtime_bundle_sha256,
        )
    )
    if (
        current_release_revision != release_revision
        or current_release_source_sha256 != release_source_sha256
    ):
        raise ArtifactError(
            f"Scoped release texts changed after publication preflight for {repo_id}."
        )


def _run_required_complete_autoclass_probe(
    spec: ModelSpec,
    artifact_path: Path,
) -> tuple[str, ...]:
    """Validate ANKH encoder and seq2seq views from the same local artifact."""

    required = tuple(
        name for name in _REQUIRED_COMPLETE_AUTOMODEL_VIEWS if name in spec.auto_map
    )
    if required != _REQUIRED_COMPLETE_AUTOMODEL_VIEWS:
        raise ArtifactError(
            f"{spec.id} does not advertise both required complete-publication AutoClasses."
        )
    cases = [
        {
            "auto_class": auto_class,
            "class_path": spec.auto_map[auto_class],
            "expected_missing_key_prefixes": [],
            "expected_unexpected_key_prefixes": [],
        }
        for auto_class in required
    ]
    with tempfile.TemporaryDirectory(
        prefix=".fastplms-complete-probe-",
        dir=artifact_path.parent,
    ) as directory:
        probe_root = Path(directory)
        cases_path = probe_root / "cases.json"
        output_path = probe_root / "result.json"
        cases_path.write_text(
            json.dumps(cases, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        command = [
            sys.executable,
            "-I",
            "-S",
            str(Path(__file__).with_name("offline_probe.py")),
            "--artifact",
            str(artifact_path),
            "--family",
            spec.family.id,
            "--bf16-execution",
            spec.family.bf16_execution,
            "--cases-file",
            str(cases_path),
            "--implementation",
            "artifact",
            "--output",
            str(output_path),
        ]
        site_packages = sorted(
            {
                Path(entry).resolve()
                for entry in sys.path
                if entry
                and Path(entry).name in {"site-packages", "dist-packages"}
                and Path(entry).is_dir()
            },
            key=lambda path: path.as_posix(),
        )
        for path in site_packages:
            command.extend(("--runtime-site-package", str(path)))
        environment = os.environ.copy()
        environment.pop("PYTHONHOME", None)
        environment.pop("PYTHONPATH", None)
        environment["HF_HOME"] = str(probe_root / "hf-home")
        environment["HF_MODULES_CACHE"] = str(probe_root / "modules")
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        environment["PYTHONNOUSERSITE"] = "1"
        try:
            completed = subprocess.run(
                command,
                cwd=probe_root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=60 * 60,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ArtifactError(
                f"Required complete-publication AutoClass probe failed for {spec.id}."
            ) from error
        if completed.returncode != 0:
            details = (completed.stdout + completed.stderr).strip()[-4000:]
            raise ArtifactError(
                f"Required complete-publication AutoClass probe failed for {spec.id}:\n"
                f"{details}"
            )
        try:
            results = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactError(
                f"Complete-publication AutoClass probe returned invalid output for {spec.id}."
            ) from error
        if not isinstance(results, dict) or set(results) != set(required) or any(
            not isinstance(results[name], dict) for name in required
        ):
            raise ArtifactError(
                f"Complete-publication AutoClass probe omitted a required view for {spec.id}."
            )
    return required


def _validate_files_only_artifact(
    artifact_path: Path,
    spec: ModelSpec,
) -> tuple[
    tuple[str, ...],
    tuple[tuple[str, bytes], ...],
    str,
    str,
    str,
    str,
    str,
    str,
    str,
]:
    """Validate only the files eligible for an add-only Hub commit.

    Weight files are identified from both provenance and filename rules, but
    they are deliberately neither opened nor hashed.
    """

    registry = _assert_current_registry_spec(spec)
    if spec.family.requires_complete_weight_publication:
        raise ArtifactError(
            f"{spec.id} requires one complete weights-plus-runtime publication; "
            "files-only publication is forbidden."
        )
    artifact_path = artifact_path.resolve()
    if not artifact_path.is_dir():
        raise ArtifactError(f"Built artifact does not exist: {artifact_path}")
    expected_name = spec.fast.repo_id.split("/", maxsplit=1)[1]
    if artifact_path.name != expected_name:
        raise ArtifactError(
            f"Artifact directory {artifact_path.name!r} does not match "
            f"manifest repository {expected_name!r}."
        )

    manifest = _load_json_object(
        artifact_path / "artifact-manifest.json",
        "artifact-manifest.json",
    )
    provenance = _load_json_object(
        artifact_path / "source-record.json",
        "source-record.json",
    )
    config = _load_json_object(artifact_path / "config.json", "config.json")
    if provenance.get("model_id") != spec.id:
        raise ArtifactError(
            f"Artifact provenance model_id does not match selected model {spec.id!r}."
        )
    if config.get("fastplms_model_id") != spec.id:
        raise ArtifactError(
            f"Artifact config fastplms_model_id does not match selected model {spec.id!r}."
        )
    _validate_registry_provenance(provenance, registry, spec)
    selected_checkpoint = spec.artifact_checkpoint
    expected_config_identity = {
        "auto_map": _artifact_auto_map(spec),
        "fastplms_model_id": spec.id,
        "fastplms_checkpoint_repo_id": selected_checkpoint.repo_id,
        "fastplms_checkpoint_revision": selected_checkpoint.revision,
        "fastplms_checkpoint_hash": _checkpoint_identity_hash(selected_checkpoint),
        "fastplms_weights_revision": selected_checkpoint.revision,
        "fastplms_runtime_revision": provenance.get("runtime_revision"),
        "fastplms_source_tree_sha256": provenance.get("source_tree_sha256"),
        "fastplms_runtime_bundle_sha256": provenance.get("runtime_bundle_sha256"),
        "fastplms_release_tool_revision": provenance.get("release_tool_revision"),
        "fastplms_release_tool_sha256": provenance.get("release_tool_sha256"),
    }
    if any(config.get(name) != value for name, value in expected_config_identity.items()):
        raise ArtifactError(
            "Artifact config packaging identity differs from the current registry or source."
        )
    release_revision, release_source_sha256 = _assert_current_release_texts(
        artifact_path,
        spec,
        registry,
    )

    expected_fast_checkpoint = {
        "repo_id": spec.fast.repo_id,
        "revision": spec.fast.revision,
        "files": {item.path: item.encoded for item in spec.fast.files},
        "unresolved_files": list(spec.fast.unresolved_files),
    }
    if provenance.get("fast_checkpoint") != expected_fast_checkpoint:
        raise ArtifactError(
            f"Artifact Fast checkpoint provenance differs from the current registry for {spec.id}."
        )
    if provenance.get("runtime_assets") != _expected_runtime_assets(spec, registry):
        raise ArtifactError(
            f"Artifact runtime-asset provenance differs from the current registry for {spec.id}."
        )

    inventory = _artifact_inventory(artifact_path)
    expected_inventory = frozenset({"artifact-manifest.json", *manifest})
    if inventory != expected_inventory:
        missing = sorted(expected_inventory.difference(inventory))
        extra = sorted(inventory.difference(expected_inventory))
        raise ArtifactError(
            "Artifact file inventory differs from artifact-manifest.json; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    canonical_weights = _canonical_weight_paths(provenance)
    declared_assets = _declared_non_weight_assets(spec, registry)
    declared_legal_paths = _declared_legal_paths(spec, registry)
    selected: list[str] = []
    selected_digests: dict[str, str] = {}
    payloads: list[tuple[str, bytes]] = []
    for relative_name, encoded_digest in sorted(manifest.items()):
        if not isinstance(relative_name, str) or not isinstance(encoded_digest, str):
            raise ArtifactError("artifact-manifest.json keys and values must be strings.")
        is_weight = relative_name in canonical_weights or _is_weight_path(relative_name)
        path = _resolve_artifact_manifest_path(artifact_path, relative_name)
        if path.is_symlink() or not path.is_file():
            raise ArtifactError(f"Files-only artifact path is missing or unsafe: {path}")
        if not is_weight and relative_name not in _COMPLETE_ATTESTATION_FILES:
            _validate_publishable_non_weight_path(
                relative_name,
                path,
                declared_assets=declared_assets,
                declared_legal_paths=declared_legal_paths,
            )
        if (
            is_weight
            or relative_name in _COMPLETE_ATTESTATION_FILES
            or not _is_runtime_update_path(relative_name)
        ):
            continue
        payload = _read_validated_bytes(
            path,
            encoded_digest,
            max_bytes=_MAX_RUNTIME_FILE_BYTES,
        )
        selected.append(relative_name)
        selected_digests[relative_name] = encoded_digest
        payloads.append((relative_name, payload))

    missing = sorted(_REQUIRED_FILES_ONLY_PATHS.difference(selected))
    if missing:
        raise ArtifactError(f"Files-only artifact is missing required paths: {missing}")
    if not any(path.startswith("fastplms/") for path in selected):
        raise ArtifactError("Files-only artifact contains no packaged FastPLMs runtime sources.")
    if any(_is_weight_path(path) for path in selected):
        raise ArtifactError("Files-only upload plan unexpectedly contains a weight path.")

    runtime_attestation = _load_json_object(
        artifact_path / _RUNTIME_ATTESTATION_NAME,
        _RUNTIME_ATTESTATION_NAME,
    )
    attested_files = runtime_attestation.get("files")
    expected_attested_files = {
        name: digest
        for name, digest in selected_digests.items()
        if name != _RUNTIME_ATTESTATION_NAME
    }
    expected_runtime_identity = {
        "schema_version": _RUNTIME_ATTESTATION_SCHEMA_VERSION,
        "scope": "runtime-only",
        "model_id": spec.id,
        "weights": {"repo_id": spec.fast.repo_id, "revision": spec.fast.revision},
        "runtime_revision": provenance.get("runtime_revision"),
        "source_tree_sha256": provenance.get("source_tree_sha256"),
        "runtime_bundle_sha256": provenance.get("runtime_bundle_sha256"),
        "release_tool_revision": provenance.get("release_tool_revision"),
        "release_tool_sha256": provenance.get("release_tool_sha256"),
        "weights_license_status": provenance.get("weights_license_status"),
        "redistributable": provenance.get("redistributable"),
        "files": expected_attested_files,
    }
    if (
        runtime_attestation != expected_runtime_identity
        or attested_files != expected_attested_files
    ):
        raise ArtifactError(
            "Runtime-only attestation differs from the selected files or current weight identity."
        )
    if _tree_sha256(artifact_path / "fastplms") != runtime_attestation.get(
        "source_tree_sha256"
    ):
        raise ArtifactError("Runtime source-tree digest differs from packaged sources.")
    runtime_bundle_sha256 = runtime_attestation.get("runtime_bundle_sha256")
    if not isinstance(runtime_bundle_sha256, str):
        raise ArtifactError("Runtime bundle digest is missing from the runtime attestation.")
    _validate_runtime_bundle(
        artifact_path / "fastplms_bundle.py",
        artifact_path / "fastplms",
        runtime_bundle_sha256,
    )
    _validate_bootstrap(
        artifact_path / "modeling_fastplms.py",
        spec,
        runtime_bundle_sha256,
    )
    runtime_revision, source_tree_sha256 = _assert_current_runtime_source(
        spec,
        registry,
        provenance,
    )
    (
        release_tool_revision,
        release_tool_sha256,
        release_tool_payloads,
    ) = _assert_current_release_tool_source(provenance)
    _assert_artifact_requirements(artifact_path, spec, release_tool_payloads)
    return (
        tuple(selected),
        tuple(payloads),
        runtime_revision,
        source_tree_sha256,
        runtime_bundle_sha256,
        release_tool_revision,
        release_tool_sha256,
        release_revision,
        release_source_sha256,
    )


def _attribute(value: object, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _remote_file_digest(sibling: object, expected: FileDigest) -> str | None:
    if expected.algorithm == "sha256":
        lfs = _attribute(sibling, "lfs")
        digest = _attribute(lfs, "sha256") if lfs is not None else None
        return digest if isinstance(digest, str) else None
    if expected.algorithm == "git-sha1":
        digest = _attribute(sibling, "blob_id")
        return digest if isinstance(digest, str) else None
    return None


def _verify_remote_weights(spec: ModelSpec, model_info: object) -> None:
    siblings = _attribute(model_info, "siblings")
    if not isinstance(siblings, Iterable):
        raise ArtifactError(f"Hub metadata for {spec.fast.repo_id} contains no file listing.")
    remote_files = {
        path: sibling
        for sibling in siblings
        if isinstance(path := _attribute(sibling, "rfilename"), str)
    }
    expected_weights = tuple(item for item in spec.fast.files if _is_weight_path(item.path))
    if not expected_weights:
        raise ArtifactError(f"{spec.id} declares no Fast checkpoint weight files.")
    for expected in expected_weights:
        sibling = remote_files.get(expected.path)
        if sibling is None:
            raise ArtifactError(
                f"Hub repository {spec.fast.repo_id} is missing pinned weight {expected.path}."
            )
        actual = _remote_file_digest(sibling, expected)
        if actual is None:
            raise ArtifactError(
                f"Hub did not return {expected.algorithm} metadata for "
                f"{spec.fast.repo_id}/{expected.path}."
            )
        if actual != expected.digest:
            raise ArtifactError(
                f"Hub weight identity differs for {spec.fast.repo_id}/{expected.path}: "
                f"expected {expected.digest}, received {actual}."
            )


def _obsolete_registry_pinned_paths(
    spec: ModelSpec,
    model_info: object,
    new_inventory: Iterable[str],
) -> tuple[str, ...]:
    """Verify current weights and select only superseded declared paths."""

    siblings = _attribute(model_info, "siblings")
    if not isinstance(siblings, Iterable):
        raise ArtifactError(f"Hub metadata for {spec.fast.repo_id} contains no file listing.")
    remote_files = {
        path: sibling
        for sibling in siblings
        if isinstance(path := _attribute(sibling, "rfilename"), str)
    }
    pinned = {
        item.path: item for item in spec.fast.files if _is_weight_path(item.path)
    }
    replacement_weights = {
        relative_name for relative_name in new_inventory if _is_weight_path(relative_name)
    }
    remote_weights = {
        relative_name for relative_name in remote_files if _is_weight_path(relative_name)
    }
    ambiguous = sorted(remote_weights.difference(pinned, replacement_weights))
    if ambiguous:
        raise ArtifactError(
            f"Hub repository {spec.fast.repo_id} contains unpinned competing weight files: "
            f"{ambiguous}. Resolve their identity before complete replacement."
        )
    for relative_name, expected in pinned.items():
        sibling = remote_files.get(relative_name)
        if sibling is None:
            raise ArtifactError(
                f"Hub repository {spec.fast.repo_id} is missing pinned file {relative_name}."
            )
        actual = _remote_file_digest(sibling, expected)
        if actual is None:
            raise ArtifactError(
                f"Hub did not return {expected.algorithm} metadata for "
                f"{spec.fast.repo_id}/{relative_name}."
            )
        if actual != expected.digest:
            raise ArtifactError(
                f"Hub pinned-file identity differs for {spec.fast.repo_id}/{relative_name}: "
                f"expected {expected.digest}, received {actual}."
            )
    return tuple(sorted(set(pinned).difference(replacement_weights)))


def prepare_files_only_plan(
    spec: ModelSpec,
    *,
    artifact_root: Path,
    revision: str,
    api: HfApi,
) -> FilesOnlyPublishPlan:
    """Validate local non-weight files and the pinned remote weight identity."""

    repository_name = spec.fast.repo_id.split("/", maxsplit=1)[1]
    artifact_path = artifact_root.resolve() / repository_name
    (
        files,
        payloads,
        runtime_revision,
        source_tree_sha256,
        runtime_bundle_sha256,
        release_tool_revision,
        release_tool_sha256,
        release_revision,
        release_source_sha256,
    ) = _validate_files_only_artifact(artifact_path, spec)
    info = api.model_info(
        spec.fast.repo_id,
        revision=revision,
        files_metadata=True,
    )
    parent_commit = _attribute(info, "sha")
    if not isinstance(parent_commit, str) or not parent_commit:
        raise ArtifactError(f"Hub repository {spec.fast.repo_id} has no commit identity.")
    _verify_remote_weights(spec, info)
    return FilesOnlyPublishPlan(
        model_id=spec.id,
        repo_id=spec.fast.repo_id,
        revision=revision,
        parent_commit=parent_commit,
        artifact_path=artifact_path,
        files=files,
        payloads=payloads,
        runtime_revision=runtime_revision,
        source_tree_sha256=source_tree_sha256,
        runtime_bundle_sha256=runtime_bundle_sha256,
        release_tool_revision=release_tool_revision,
        release_tool_sha256=release_tool_sha256,
        release_revision=release_revision,
        release_source_sha256=release_source_sha256,
    )


def prepare_files_only_plans(
    specs: Iterable[ModelSpec],
    *,
    artifact_root: Path,
    revision: str,
    api: HfApi,
) -> tuple[FilesOnlyPublishPlan, ...]:
    """Preflight every repository before the first remote mutation."""

    selected = tuple(specs)
    blocked = [
        spec.id for spec in selected if spec.family.requires_complete_weight_publication
    ]
    if blocked:
        raise ArtifactError(
            "Files-only publication is forbidden for models requiring complete weights: "
            + ", ".join(blocked)
        )
    return tuple(
        prepare_files_only_plan(
            spec,
            artifact_root=artifact_root,
            revision=revision,
            api=api,
        )
        for spec in selected
    )


def prepare_complete_plan(
    spec: ModelSpec,
    *,
    artifact_root: Path,
    revision: str,
    api: HfApi,
) -> CompletePublishPlan:
    """Preflight one complete artifact for a single atomic Hub commit."""

    registry = _assert_current_registry_spec(spec)
    if not spec.family.weights_publication_allowed:
        raise ArtifactError(
            f"{spec.id} checkpoint publication is blocked by unresolved weight-license terms."
        )
    repository_name = spec.fast.repo_id.split("/", maxsplit=1)[1]
    artifact_path = artifact_root.resolve() / repository_name
    validate_artifact(artifact_path, spec=spec, registry=registry)
    release_revision, release_source_sha256 = _assert_current_release_texts(
        artifact_path,
        spec,
        registry,
    )
    manifest = _load_json_object(
        artifact_path / "artifact-manifest.json",
        "artifact-manifest.json",
    )
    inventory = _artifact_inventory(artifact_path)
    if inventory != frozenset({"artifact-manifest.json", *manifest}):
        raise ArtifactError(
            "Complete artifact inventory differs from artifact-manifest.json."
        )
    provenance = _load_json_object(artifact_path / "source-record.json", "source-record.json")
    if (
        provenance.get("weights_license_status") != "resolved"
        or provenance.get("redistributable") is not True
    ):
        raise ArtifactError(
            f"Complete publication is forbidden for non-redistributable artifact {spec.id}."
        )
    for label, checkpoint in (("fast", spec.fast), ("official", spec.official)):
        expected_checkpoint = {
            "repo_id": checkpoint.repo_id,
            "revision": checkpoint.revision,
            "files": {item.path: item.encoded for item in checkpoint.files},
            "unresolved_files": list(checkpoint.unresolved_files),
        }
        if provenance.get(f"{label}_checkpoint") != expected_checkpoint:
            raise ArtifactError(
                f"Complete artifact {label} checkpoint differs from the current registry."
            )
    if provenance.get("runtime_assets") != _expected_runtime_assets(spec, registry):
        raise ArtifactError(
            "Complete artifact runtime-asset provenance differs from the current registry."
        )
    runtime_revision, source_tree_sha256 = _assert_current_runtime_source(
        spec,
        registry,
        provenance,
    )
    (
        release_tool_revision,
        release_tool_sha256,
        release_tool_payloads,
    ) = _assert_current_release_tool_source(provenance)
    _assert_artifact_requirements(artifact_path, spec, release_tool_payloads)
    runtime_bundle_sha256 = provenance.get("runtime_bundle_sha256")
    if not isinstance(runtime_bundle_sha256, str):
        raise ArtifactError("Complete artifact lacks a runtime-bundle identity.")
    canonical_weights = _canonical_weight_paths(provenance)
    declared_assets = _declared_non_weight_assets(spec, registry)
    declared_legal_paths = _declared_legal_paths(spec, registry)
    digests: list[tuple[str, str]] = []
    for relative_name in sorted(inventory):
        path = _resolve_artifact_manifest_path(artifact_path, relative_name)
        if relative_name not in {"artifact-manifest.json", "source-record.json"} and (
            relative_name not in canonical_weights and not _is_weight_path(relative_name)
        ):
            _validate_publishable_non_weight_path(
                relative_name,
                path,
                declared_assets=declared_assets,
                declared_legal_paths=declared_legal_paths,
            )
        if relative_name == "artifact-manifest.json":
            digest = f"sha256:{hash_file(path)}"
        else:
            encoded = manifest.get(relative_name)
            if not isinstance(encoded, str):
                raise ArtifactError(
                    f"Complete artifact manifest is missing {relative_name!r}."
                )
            _encoded_digest(path, encoded)
            digest = encoded
        digests.append((relative_name, digest))
    required = {
        "README.md",
        "config.json",
        "source-record.json",
        "artifact-manifest.json",
        _RUNTIME_ATTESTATION_NAME,
        *canonical_weights,
    }
    missing = sorted(required.difference(inventory))
    if missing:
        raise ArtifactError(f"Complete artifact is missing required atomic paths: {missing}")
    validated_auto_classes: tuple[str, ...] = ()
    validation_manifest_sha256: str | None = None
    if spec.family.requires_complete_weight_publication:
        validated_auto_classes = _run_required_complete_autoclass_probe(
            spec,
            artifact_path,
        )
        validation_manifest_sha256 = hash_file(
            artifact_path / "artifact-manifest.json"
        )
    info = api.model_info(spec.fast.repo_id, revision=revision, files_metadata=True)
    parent_commit = _attribute(info, "sha")
    if not isinstance(parent_commit, str) or not parent_commit:
        raise ArtifactError(f"Hub repository {spec.fast.repo_id} has no commit identity.")
    files = tuple(name for name, _ in digests)
    deletes = _obsolete_registry_pinned_paths(spec, info, files)
    return CompletePublishPlan(
        model_id=spec.id,
        repo_id=spec.fast.repo_id,
        revision=revision,
        parent_commit=parent_commit,
        artifact_path=artifact_path,
        files=files,
        digests=tuple(digests),
        deletes=deletes,
        replacement_weight_paths=tuple(sorted(canonical_weights)),
        runtime_revision=runtime_revision,
        source_tree_sha256=source_tree_sha256,
        runtime_bundle_sha256=runtime_bundle_sha256,
        release_tool_revision=release_tool_revision,
        release_tool_sha256=release_tool_sha256,
        release_revision=release_revision,
        release_source_sha256=release_source_sha256,
        validation_manifest_sha256=validation_manifest_sha256,
        validated_auto_classes=validated_auto_classes,
    )


def prepare_complete_plans(
    specs: Iterable[ModelSpec],
    *,
    artifact_root: Path,
    revision: str,
    api: HfApi,
) -> tuple[CompletePublishPlan, ...]:
    """Preflight all complete commits before the first remote mutation."""

    selected = tuple(specs)
    blocked = [spec.id for spec in selected if not spec.family.weights_publication_allowed]
    if blocked:
        raise ArtifactError(
            "Complete checkpoint publication is blocked by unresolved weight licenses: "
            + ", ".join(blocked)
        )
    return tuple(
        prepare_complete_plan(
            spec,
            artifact_root=artifact_root,
            revision=revision,
            api=api,
        )
        for spec in selected
    )


def publish_files_only(
    plans: Iterable[FilesOnlyPublishPlan],
    *,
    api: HfApi,
    commit_message: str,
    dry_run: bool = False,
) -> tuple[FilesOnlyPublishResult, ...]:
    """Execute preflighted add-only commits, or print their dry-run plans."""

    results: list[FilesOnlyPublishResult] = []
    for plan in plans:
        _revalidate_plan_runtime_source(
            model_id=plan.model_id,
            repo_id=plan.repo_id,
            runtime_revision=plan.runtime_revision,
            source_tree_sha256=plan.source_tree_sha256,
            runtime_bundle_sha256=plan.runtime_bundle_sha256,
            release_tool_revision=plan.release_tool_revision,
            release_tool_sha256=plan.release_tool_sha256,
            release_revision=plan.release_revision,
            release_source_sha256=plan.release_source_sha256,
        )
        print(
            f"{'[dry-run] ' if dry_run else ''}{plan.repo_id}: "
            f"{len(plan.files)} non-weight files at {plan.parent_commit}"
        )
        for relative_name in plan.files:
            print(f"  {relative_name}")
        if dry_run:
            continue
        payload_by_name = dict(plan.payloads)
        if set(payload_by_name) != set(plan.files):
            raise ArtifactError(
                f"Preflighted payload inventory differs for {plan.repo_id}; rebuild the plan."
            )
        operations = [
            CommitOperationAdd(
                path_in_repo=relative_name,
                path_or_fileobj=io.BytesIO(payload_by_name[relative_name]),
            )
            for relative_name in plan.files
        ]
        commit = api.create_commit(
            repo_id=plan.repo_id,
            repo_type="model",
            revision=plan.revision,
            parent_commit=plan.parent_commit,
            operations=operations,
            commit_message=commit_message,
            commit_description=(
                "Add-only FastPLMs files-only publication. "
                "Checkpoint weights and complete-artifact attestations are unchanged."
            ),
        )
        oid = _attribute(commit, "oid")
        url = _attribute(commit, "commit_url")
        if not isinstance(oid, str) or not isinstance(url, str):
            raise ArtifactError(f"Hub returned incomplete commit metadata for {plan.repo_id}.")
        results.append(
            FilesOnlyPublishResult(
                model_id=plan.model_id,
                repo_id=plan.repo_id,
                commit_oid=oid,
                commit_url=url,
            )
        )
    return tuple(results)


def _revalidate_complete_plan(
    plan: CompletePublishPlan,
    api: HfApi,
) -> ModelSpec:
    """Rebuild one complete preflight immediately before any remote mutation."""

    registry = get_model_registry()
    selected_spec = registry.get(plan.model_id)
    if selected_spec is None:
        raise ArtifactError(
            f"Complete publication plan model {plan.model_id!r} is absent from the "
            "current registry."
        )
    if plan.repo_id != selected_spec.fast.repo_id:
        raise ArtifactError(
            f"Complete plan repository differs from the registry for {plan.model_id}."
        )
    current_plan = prepare_complete_plan(
        selected_spec,
        artifact_root=plan.artifact_path.resolve().parent,
        revision=plan.revision,
        api=api,
    )
    if current_plan != plan:
        raise ArtifactError(
            f"Complete plan for {plan.repo_id} differs from a current full preflight; "
            "rebuild the plan."
        )
    return selected_spec


def publish_complete(
    plans: Iterable[CompletePublishPlan],
    *,
    api: HfApi,
    commit_message: str,
    dry_run: bool = False,
) -> tuple[FilesOnlyPublishResult, ...]:
    """Publish each complete artifact in one parent-guarded atomic commit."""

    results: list[FilesOnlyPublishResult] = []
    for plan in plans:
        selected_spec = _revalidate_complete_plan(plan, api)
        if selected_spec is not None:
            if not selected_spec.family.weights_publication_allowed:
                raise ArtifactError(
                    f"{plan.model_id} checkpoint publication is blocked by unresolved "
                    "weight-license terms."
                )
            replacement_weights = set(plan.replacement_weight_paths)
            if (
                replacement_weights != {
                    path for path in plan.files if _is_weight_path(path)
                }
                or _WEIGHT_INDEX not in replacement_weights
                or not any(
                    path != _WEIGHT_INDEX and path.endswith(".safetensors")
                    for path in replacement_weights
                )
            ):
                raise ArtifactError(
                    f"Complete plan for {plan.repo_id} lacks a canonical replacement "
                    "weight set."
                )
        if (
            selected_spec is not None
            and selected_spec.family.requires_complete_weight_publication
        ):
            manifest_path = plan.artifact_path / "artifact-manifest.json"
            if (
                plan.repo_id != selected_spec.fast.repo_id
                or plan.validated_auto_classes != _REQUIRED_COMPLETE_AUTOMODEL_VIEWS
                or plan.validation_manifest_sha256 is None
                or not manifest_path.is_file()
                or hash_file(manifest_path) != plan.validation_manifest_sha256
            ):
                raise ArtifactError(
                    f"Complete plan for {plan.repo_id} lacks a current required AutoClass probe."
                )
        if len(set(plan.deletes)) != len(plan.deletes):
            raise ArtifactError(f"Complete delete paths are repeated for {plan.repo_id}.")
        if set(plan.deletes).intersection(plan.files):
            raise ArtifactError(
                f"Complete add and delete paths overlap for {plan.repo_id}."
            )
        if plan.deletes:
            if selected_spec is None:
                raise ArtifactError(
                    f"Complete plan for {plan.repo_id} cannot authorize deletes without "
                    "a current registry model."
                )
            current_info = api.model_info(
                plan.repo_id,
                revision=plan.revision,
                files_metadata=True,
            )
            if _attribute(current_info, "sha") != plan.parent_commit:
                raise ArtifactError(
                    f"Remote parent changed after complete preflight for {plan.repo_id}."
                )
            permitted_deletes = _obsolete_registry_pinned_paths(
                selected_spec,
                current_info,
                plan.files,
            )
            if plan.deletes != permitted_deletes:
                raise ArtifactError(
                    f"Complete plan for {plan.repo_id} contains an unproven obsolete "
                    "weight delete."
                )
        print(
            f"{'[dry-run] ' if dry_run else ''}{plan.repo_id}: "
            f"{len(plan.files)} complete files, {len(plan.deletes)} guarded deletes "
            f"at {plan.parent_commit}"
        )
        for relative_name in plan.deletes:
            print(f"  delete {relative_name}")
        if dry_run:
            continue
        expected = dict(plan.digests)
        if set(expected) != set(plan.files):
            raise ArtifactError(
                f"Preflighted complete inventory differs for {plan.repo_id}; rebuild the plan."
            )
        # Bounded files are retained as exact bytes. Large shards are copied to
        # a publisher-owned temporary snapshot, then rehashed and held open
        # through the synchronous Hub commit.
        with contextlib.ExitStack() as open_payloads:
            snapshot_root = Path(
                open_payloads.enter_context(
                    tempfile.TemporaryDirectory(
                        prefix=".fastplms-complete-publish-",
                        dir=plan.artifact_path.parent,
                    )
                )
            )
            operations: list[CommitOperationAdd | CommitOperationDelete] = []
            retained_handles: list[
                tuple[BinaryIO, tuple[int, int, int, int], Path]
            ] = []
            for index, relative_name in enumerate(plan.files):
                path = _resolve_artifact_manifest_path(plan.artifact_path, relative_name)
                if path.is_symlink() or not path.is_file():
                    raise ArtifactError(
                        f"Preflighted complete artifact path changed: {relative_name!r}"
                    )
                if path.stat().st_size <= _MAX_RETAINED_COMPLETE_BYTES:
                    payload: BinaryIO = io.BytesIO(
                        _read_validated_bytes(
                            path,
                            expected[relative_name],
                            max_bytes=_MAX_RETAINED_COMPLETE_BYTES,
                        )
                    )
                else:
                    snapshot = snapshot_root / f"{index:05d}.payload"
                    _snapshot_validated_payload(
                        path,
                        snapshot,
                        expected[relative_name],
                    )
                    payload = open_payloads.enter_context(snapshot.open("rb"))
                    identity = _open_validated_payload(
                        payload,
                        snapshot,
                        expected[relative_name],
                    )
                    retained_handles.append((payload, identity, snapshot))
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=relative_name,
                        path_or_fileobj=payload,
                    )
                )
            operations.extend(
                CommitOperationDelete(path_in_repo=relative_name)
                for relative_name in plan.deletes
            )
            for handle, expected_identity, path in retained_handles:
                current = os.fstat(handle.fileno())
                if expected_identity != (
                    current.st_dev,
                    current.st_ino,
                    current.st_size,
                    current.st_mtime_ns,
                ):
                    raise ArtifactError(
                        f"Preflighted complete artifact changed before upload: {path}"
                    )
                handle.seek(0)
            commit = api.create_commit(
                repo_id=plan.repo_id,
                repo_type="model",
                revision=plan.revision,
                parent_commit=plan.parent_commit,
                operations=operations,
                commit_message=commit_message,
                commit_description=(
                    "Atomic FastPLMs complete publication. Checkpoint weights, tokenizer "
                    "assets, runtime sources, model card, legal texts, and scoped "
                    "attestations are updated together. Deletes are restricted to obsolete "
                    "current-registry-pinned paths."
                ),
            )
        oid = _attribute(commit, "oid")
        url = _attribute(commit, "commit_url")
        if not isinstance(oid, str) or not isinstance(url, str):
            raise ArtifactError(f"Hub returned incomplete commit metadata for {plan.repo_id}.")
        results.append(
            FilesOnlyPublishResult(
                model_id=plan.model_id,
                repo_id=plan.repo_id,
                commit_oid=oid,
                commit_url=url,
            )
        )
    return tuple(results)


def _selected_specs(
    registry: ModelRegistry,
    model_ids: Iterable[str],
    *,
    all_models: bool,
) -> tuple[ModelSpec, ...]:
    selected = tuple(model_ids)
    if all_models and selected:
        raise ArtifactError("Pass model IDs or --all, not both.")
    if all_models or not selected:
        selected = tuple(registry)
    unknown = sorted(set(selected).difference(registry))
    if unknown:
        raise ArtifactError(f"Unknown model IDs: {unknown}")
    if len(set(selected)) != len(selected):
        raise ArtifactError("Model IDs must not be repeated.")
    return tuple(registry[model_id] for model_id in selected)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_ids", nargs="*", help="Stable IDs from src/fastplms/models.toml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--files-only",
        action="store_true",
        help="Required safety mode: upload no checkpoint weights and delete no remote files",
    )
    mode.add_argument(
        "--complete",
        action="store_true",
        help="Atomically upload a complete validated artifact, including checkpoint weights",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Explicit alias for the default behavior of publishing every manifest model",
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("dist/hub"))
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--commit-message",
        default="Update FastPLMs runtime files",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    if not arguments.files_only and not arguments.complete:
        raise SystemExit(
            "Refusing to publish without an explicit --files-only or --complete mode."
        )
    if arguments.complete and (arguments.all or not arguments.model_ids):
        raise SystemExit("Complete publication requires explicit model IDs and forbids --all.")
    registry = get_model_registry()
    try:
        specs = _selected_specs(
            registry,
            arguments.model_ids,
            all_models=arguments.all,
        )
        api = HfApi()
        if arguments.files_only:
            files_only_plans = prepare_files_only_plans(
                specs,
                artifact_root=arguments.artifact_root,
                revision=arguments.revision,
                api=api,
            )
            results = publish_files_only(
                files_only_plans,
                api=api,
                commit_message=arguments.commit_message,
                dry_run=arguments.dry_run,
            )
        else:
            complete_plans = prepare_complete_plans(
                specs,
                artifact_root=arguments.artifact_root,
                revision=arguments.revision,
                api=api,
            )
            results = publish_complete(
                complete_plans,
                api=api,
                commit_message=arguments.commit_message,
                dry_run=arguments.dry_run,
            )
    except ArtifactError as error:
        raise SystemExit(str(error)) from error
    for result in results:
        print(f"{result.repo_id}: {result.commit_oid} {result.commit_url}")


if __name__ == "__main__":
    main()


__all__ = [
    "CompletePublishPlan",
    "FilesOnlyPublishPlan",
    "FilesOnlyPublishResult",
    "prepare_complete_plan",
    "prepare_complete_plans",
    "prepare_files_only_plan",
    "prepare_files_only_plans",
    "publish_complete",
    "publish_files_only",
]
