"""Compile FastPLMs Hub files and upload them to their model repositories.

``--files-only`` publishes source-backed files directly from this checkout.
Without it, the prepared files in ``dist/hub/<repository>`` are included too,
including checkpoint weights.
"""

from __future__ import annotations

import argparse
import io
import sys
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi


_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PACKAGE = _SOURCE_ROOT / "src"
if str(_SOURCE_PACKAGE) not in sys.path:
    sys.path.insert(0, str(_SOURCE_PACKAGE))


from fastplms.registry import ModelRegistry, ModelSpec, get_model_registry  # noqa: E402
from tools.artifacts.build import (  # noqa: E402
    ArtifactError,
    _render_artifact_requirements,
    _render_bootstrap,
    _render_runtime_bundle,
    _runtime_source_entries,
)


_REQUIREMENT_FILES = (
    "requirements/core.in",
    "requirements/features/flash.in",
    "requirements/features/structure.in",
)
_BUILD_ONLY_FILES = {
    "artifact-manifest.json",
    "runtime-attestation.json",
    "source-record.json",
}


def compile_model_files(spec: ModelSpec, source_root: Path) -> dict[str, bytes]:
    """Compile the current checkout into files for one Hub model repository."""

    source_root = source_root.resolve()
    runtime_files = {
        target.as_posix(): source.read_bytes()
        for source, target in _runtime_source_entries(source_root, spec)
    }
    with tempfile.TemporaryDirectory(prefix=".fastplms-publish-", dir=source_root) as directory:
        package_root = Path(directory) / "fastplms"
        for relative_name, payload in runtime_files.items():
            destination = package_root.joinpath(*PurePosixPath(relative_name).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        runtime_hash, runtime_bundle = _render_runtime_bundle(package_root)

    requirement_payloads = {
        relative_name: source_root.joinpath(*PurePosixPath(relative_name).parts).read_bytes()
        for relative_name in _REQUIREMENT_FILES
    }
    files = {
        f"fastplms/{relative_name}": payload
        for relative_name, payload in runtime_files.items()
    }
    files.update(
        {
            "README.md": (source_root / "model_cards" / f"{spec.id}.md").read_bytes(),
            "fastplms_bundle.py": runtime_bundle,
            "modeling_fastplms.py": _render_bootstrap(spec, runtime_hash).encode("utf-8"),
            "requirements.txt": _render_artifact_requirements(
                spec,
                requirement_payloads,
            ).encode("utf-8"),
            "THIRD_PARTY_NOTICES.md": (source_root / "THIRD_PARTY_NOTICES.md").read_bytes(),
            "LICENSES/FastPLMs-Apache-2.0.txt": (source_root / "LICENSE").read_bytes(),
        }
    )
    registry = get_model_registry()
    for source_id in spec.family.upstreams:
        upstream = registry.upstreams[source_id]
        for item in upstream.distribution_files:
            source = source_root / "LICENSES" / source_id / item.path
            files[f"LICENSES/{source_id}/{item.path}"] = source.read_bytes()
    return dict(sorted(files.items()))


def _artifact_files(spec: ModelSpec, artifact_root: Path) -> dict[str, Path]:
    repository_name = spec.fast.repo_id.split("/", maxsplit=1)[-1]
    artifact = artifact_root.resolve() / repository_name
    if not artifact.is_dir():
        raise ArtifactError(
            f"Prepared artifact does not exist: {artifact}. Build it with "
            f"'PYTHONPATH=src python -m tools.artifacts.build_all {spec.id} --replace'."
        )
    return {
        path.relative_to(artifact).as_posix(): path
        for path in sorted(artifact.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file() and path.relative_to(artifact).as_posix() not in _BUILD_ONLY_FILES
    }


def _selected_specs(
    registry: ModelRegistry,
    model_ids: Iterable[str],
) -> tuple[ModelSpec, ...]:
    selected = tuple(model_ids) or tuple(registry)
    unknown = sorted(set(selected).difference(registry))
    if unknown:
        raise ArtifactError(f"Unknown model IDs: {unknown}")
    if len(selected) != len(set(selected)):
        raise ArtifactError("Model IDs must not be repeated.")
    return tuple(registry[model_id] for model_id in selected)


def _operation(path_in_repo: str, source: bytes | Path) -> CommitOperationAdd:
    payload: io.BytesIO | Path
    payload = io.BytesIO(source) if isinstance(source, bytes) else source
    return CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=payload)


def publish_models(
    specs: Iterable[ModelSpec],
    *,
    source_root: Path,
    artifact_root: Path,
    revision: str,
    api: HfApi,
    commit_message: str,
    files_only: bool,
    dry_run: bool = False,
) -> tuple[object, ...]:
    """Compile and publish every selected model."""

    commits: list[object] = []
    for spec in specs:
        files: dict[str, bytes | Path] = {}
        if not files_only:
            files.update(_artifact_files(spec, artifact_root))
        files.update(compile_model_files(spec, source_root))

        prefix = "[dry-run] " if dry_run else ""
        print(f"{prefix}{spec.fast.repo_id}: {len(files)} files")
        for relative_name in files:
            print(f"  {relative_name}")
        if dry_run:
            continue

        commit = api.create_commit(
            repo_id=spec.fast.repo_id,
            repo_type="model",
            revision=revision,
            operations=[
                _operation(relative_name, source)
                for relative_name, source in files.items()
            ],
            commit_message=commit_message,
        )
        commits.append(commit)
    return tuple(commits)


def _attribute(value: object, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_ids", nargs="*", help="Stable IDs from src/fastplms/models.toml")
    parser.add_argument(
        "--files-only",
        action="store_true",
        help="Compile and upload source-backed files without prepared artifact files or weights",
    )
    parser.add_argument("--source-root", type=Path, default=_SOURCE_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=Path("dist/hub"))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--commit-message", default="Update FastPLMs files")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    try:
        specs = _selected_specs(get_model_registry(), arguments.model_ids)
        commits = publish_models(
            specs,
            source_root=arguments.source_root,
            artifact_root=arguments.artifact_root,
            revision=arguments.revision,
            api=HfApi(),
            commit_message=arguments.commit_message,
            files_only=arguments.files_only,
            dry_run=arguments.dry_run,
        )
    except (ArtifactError, OSError) as error:
        raise SystemExit(str(error)) from error
    for commit in commits:
        oid = _attribute(commit, "oid")
        url = _attribute(commit, "commit_url")
        print(" ".join(str(value) for value in (oid, url) if value is not None))


if __name__ == "__main__":
    main()


__all__ = ["compile_model_files", "main", "publish_models"]
