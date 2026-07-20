"""Publish validated non-weight files from local Hub artifacts.

This command is intentionally limited to add-only, files-only commits. It never
creates repositories, uploads checkpoint weights, deletes remote files, or
changes repository settings.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi

from fastplms.registry import FileDigest, ModelRegistry, ModelSpec, get_model_registry
from tools.artifacts.build import (
    ArtifactError,
    _WEIGHT_INDEX,
    _WEIGHT_SUFFIXES,
    _resolve_artifact_manifest_path,
    hash_file,
)

_ATTESTATION_FILES = frozenset({"artifact-manifest.json", "provenance.json"})
_REQUIRED_FILES_ONLY_PATHS = frozenset(
    {
        "README.md",
        "config.json",
        "fastplms_bundle.py",
        "modeling_fastplms.py",
        "THIRD_PARTY_NOTICES.md",
        "LICENSES/FastPLMs-Apache-2.0.txt",
    }
)


@dataclass(frozen=True)
class FilesOnlyPublishPlan:
    """One preflighted add-only commit for a manifest-declared model."""

    model_id: str
    repo_id: str
    revision: str
    parent_commit: str
    artifact_path: Path
    files: tuple[str, ...]


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


def _validate_files_only_artifact(
    artifact_path: Path,
    spec: ModelSpec,
) -> tuple[str, ...]:
    """Validate only the files eligible for an add-only Hub commit.

    Weight files are identified from both provenance and filename rules, but
    they are deliberately neither opened nor hashed.
    """

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
        artifact_path / "provenance.json",
        "provenance.json",
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

    canonical_weights = _canonical_weight_paths(provenance)
    selected: list[str] = []
    for relative_name, encoded_digest in sorted(manifest.items()):
        if not isinstance(relative_name, str) or not isinstance(encoded_digest, str):
            raise ArtifactError("artifact-manifest.json keys and values must be strings.")
        is_weight = relative_name in canonical_weights or _is_weight_path(relative_name)
        if is_weight or relative_name in _ATTESTATION_FILES:
            continue
        path = _resolve_artifact_manifest_path(artifact_path, relative_name)
        if path.is_symlink() or not path.is_file():
            raise ArtifactError(f"Files-only artifact path is missing or unsafe: {path}")
        _encoded_digest(path, encoded_digest)
        selected.append(relative_name)

    missing = sorted(_REQUIRED_FILES_ONLY_PATHS.difference(selected))
    if missing:
        raise ArtifactError(f"Files-only artifact is missing required paths: {missing}")
    if not any(path.startswith("fastplms/") for path in selected):
        raise ArtifactError("Files-only artifact contains no packaged FastPLMs runtime sources.")
    if any(_is_weight_path(path) for path in selected):
        raise ArtifactError("Files-only upload plan unexpectedly contains a weight path.")
    return tuple(selected)


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
    files = _validate_files_only_artifact(artifact_path, spec)
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
    )


def prepare_files_only_plans(
    specs: Iterable[ModelSpec],
    *,
    artifact_root: Path,
    revision: str,
    api: HfApi,
) -> tuple[FilesOnlyPublishPlan, ...]:
    """Preflight every repository before the first remote mutation."""

    return tuple(
        prepare_files_only_plan(
            spec,
            artifact_root=artifact_root,
            revision=revision,
            api=api,
        )
        for spec in specs
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
        print(
            f"{'[dry-run] ' if dry_run else ''}{plan.repo_id}: "
            f"{len(plan.files)} non-weight files at {plan.parent_commit}"
        )
        for relative_name in plan.files:
            print(f"  {relative_name}")
        if dry_run:
            continue
        operations = [
            CommitOperationAdd(
                path_in_repo=relative_name,
                path_or_fileobj=plan.artifact_path.joinpath(*PurePosixPath(relative_name).parts),
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
    parser.add_argument(
        "--files-only",
        action="store_true",
        help="Required safety mode: upload no checkpoint weights and delete no remote files",
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
    if not arguments.files_only:
        raise SystemExit("Refusing to publish without the explicit --files-only safety mode.")
    registry = get_model_registry()
    try:
        specs = _selected_specs(
            registry,
            arguments.model_ids,
            all_models=arguments.all,
        )
        api = HfApi()
        plans = prepare_files_only_plans(
            specs,
            artifact_root=arguments.artifact_root,
            revision=arguments.revision,
            api=api,
        )
        results = publish_files_only(
            plans,
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
    "FilesOnlyPublishPlan",
    "FilesOnlyPublishResult",
    "prepare_files_only_plan",
    "prepare_files_only_plans",
    "publish_files_only",
]
