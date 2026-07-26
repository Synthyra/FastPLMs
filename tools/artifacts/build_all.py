"""Materialize every manifest artifact from immutable Hub snapshots.

The low-level builder remains network-free. This orchestration command resolves
only the repository IDs and revisions pinned by ``models.toml``, then hands the
verified local snapshots to that builder. It never uploads or deletes a Hub
repository.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from huggingface_hub import snapshot_download

from benchmarks.suite import benchmark_artifact_model_ids
from fastplms.registry import get_model_registry
from tools.artifacts.build import (
    _TOKENIZER_FILE_NAMES,
    _tokenizer_checkpoint,
    build_local_artifact,
    validate_artifact,
)


def build_all_artifacts(
    *,
    output_root: Path,
    source_root: Path,
    model_ids: Iterable[str] | None = None,
    benchmark_suite: bool = False,
    replace: bool = False,
) -> tuple[Path, ...]:
    """Build selected artifacts after resolving their immutable snapshots."""

    registry = get_model_registry()
    if model_ids is not None and benchmark_suite:
        raise ValueError("model_ids and benchmark_suite are mutually exclusive")
    if model_ids is not None:
        selected = tuple(model_ids)
    elif benchmark_suite:
        selected = benchmark_artifact_model_ids()
    else:
        selected = tuple(registry)
    unknown = sorted(set(selected).difference(registry))
    if unknown:
        raise ValueError(f"Unknown model IDs: {unknown}")

    destinations: list[Path] = []
    for model_id in selected:
        spec = registry[model_id]
        checkpoint = spec.artifact_checkpoint
        snapshot = Path(
            snapshot_download(
                repo_id=checkpoint.repo_id,
                revision=checkpoint.revision,
                allow_patterns=[item.path for item in checkpoint.files],
            )
        )
        tokenizer_snapshot: Path | None = None
        if spec.family.tokenizer_mode == "tokenizer":
            tokenizer_checkpoint = _tokenizer_checkpoint(registry, spec)
            tokenizer_files = [
                item.path
                for item in tokenizer_checkpoint.files
                if Path(item.path).name in _TOKENIZER_FILE_NAMES
            ]
            if not tokenizer_files:
                raise RuntimeError(f"{model_id}: official tokenizer files are not declared")
            if checkpoint == tokenizer_checkpoint:
                tokenizer_snapshot = snapshot
            else:
                tokenizer_snapshot = Path(
                    snapshot_download(
                        repo_id=tokenizer_checkpoint.repo_id,
                        revision=tokenizer_checkpoint.revision,
                        allow_patterns=tokenizer_files,
                    )
                )
        destination = build_local_artifact(
            model_id=model_id,
            checkpoint_dir=snapshot,
            output_root=output_root,
            source_root=source_root,
            tokenizer_dir=tokenizer_snapshot,
            replace=replace,
        )
        # Keep the completed artifact bound to the same current registry entry
        # used for construction.  This is required for official-source
        # artifacts, whose canonical-state commitment cannot be authenticated
        # from their self-authored provenance alone.
        validate_artifact(destination, spec=spec, registry=registry)
        destinations.append(destination)
    return tuple(destinations)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_ids", nargs="*")
    parser.add_argument("--output-root", type=Path, default=Path("dist/hub"))
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--benchmark-suite",
        action="store_true",
        help="Build only the fixed benchmark matrix artifacts and nested backbones.",
    )
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    paths = build_all_artifacts(
        output_root=arguments.output_root,
        source_root=arguments.source_root,
        model_ids=arguments.model_ids or None,
        benchmark_suite=arguments.benchmark_suite,
        replace=arguments.replace,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
