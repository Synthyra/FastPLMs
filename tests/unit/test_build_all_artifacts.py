"""Contracts for manifest-wide local artifact orchestration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.suite import benchmark_artifact_model_ids
from fastplms.registry import ModelRegistry, ModelSpec, get_model_registry
from tools.artifacts import build as build_module
from tools.artifacts import build_all as build_all_module


def test_build_all_registry_binds_official_source_artifact_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """ANKH and DPLM2 must not be revalidated from self-attestation alone."""

    registry = get_model_registry()
    model_ids = ("ankh_base", "dplm2_150m")
    assert all(registry[model_id].artifact_source == "official" for model_id in model_ids)

    downloaded: dict[Path, tuple[str, str]] = {}

    def fake_snapshot_download(
        *,
        repo_id: str,
        revision: str,
        allow_patterns: list[str],
    ) -> str:
        assert allow_patterns
        destination = tmp_path / "snapshots" / str(len(downloaded))
        destination.mkdir(parents=True)
        downloaded[destination] = (repo_id, revision)
        return str(destination)

    built: dict[str, Path] = {}

    def fake_build_local_artifact(
        *,
        model_id: str,
        checkpoint_dir: Path,
        output_root: Path,
        source_root: Path,
        tokenizer_dir: Path | None,
        replace: bool,
    ) -> Path:
        del source_root
        assert not replace
        spec = registry[model_id]
        selected = spec.artifact_checkpoint
        assert downloaded[checkpoint_dir] == (selected.repo_id, selected.revision)
        if tokenizer_dir is not None:
            assert tokenizer_dir in downloaded
        destination = output_root / spec.fast.repo_id.split("/", maxsplit=1)[1]
        destination.mkdir(parents=True)
        built[model_id] = destination
        return destination

    validated: list[tuple[Path, ModelSpec, ModelRegistry]] = []

    def fake_validate_artifact(
        path: Path,
        *,
        spec: ModelSpec,
        registry: ModelRegistry,
    ) -> None:
        validated.append((path, spec, registry))

    monkeypatch.setattr(build_all_module, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(
        build_all_module,
        "build_local_artifact",
        fake_build_local_artifact,
    )
    monkeypatch.setattr(build_all_module, "validate_artifact", fake_validate_artifact)

    output_root = tmp_path / "artifacts"
    destinations = build_all_module.build_all_artifacts(
        output_root=output_root,
        source_root=tmp_path / "source",
        model_ids=model_ids,
    )

    assert destinations == tuple(built[model_id] for model_id in model_ids)
    assert validated == [
        (built[model_id], registry[model_id], registry) for model_id in model_ids
    ]


def test_benchmark_artifact_selection_is_manifest_derived_and_includes_nested_backbone() -> None:
    registry = get_model_registry()
    selected = benchmark_artifact_model_ids()
    expected = {
        spec.id
        for spec in registry.values()
        if (spec.is_deep_reference or spec.family.id == "esmfold2")
        and "benchmark" in spec.family.test_tiers
    }
    backbone_id = registry.families["esmfold2"].backbone_model
    assert backbone_id is not None
    expected.add(backbone_id)

    assert set(selected) == expected
    assert "esmc_6b" in selected
    assert selected == tuple(model_id for model_id in registry if model_id in expected)


def test_build_all_rejects_mixed_explicit_and_benchmark_selection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_all_module.build_all_artifacts(
            output_root=tmp_path / "artifacts",
            source_root=tmp_path / "source",
            model_ids=("esm2_8m",),
            benchmark_suite=True,
        )


def test_single_artifact_cli_revalidates_against_current_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The normal CLI must not fall back to self-attested artifact validation."""

    registry = get_model_registry()
    spec = registry["ankh_base"]
    destination = tmp_path / "ANKH-Base"
    destination.mkdir()
    monkeypatch.setattr(
        build_module,
        "_parse_args",
        lambda: SimpleNamespace(
            model_id=spec.id,
            checkpoint_dir=tmp_path / "checkpoint",
            output_root=tmp_path / "dist",
            source_root=tmp_path,
            tokenizer_dir=None,
            replace=False,
        ),
    )
    monkeypatch.setattr(build_module, "get_model_registry", lambda: registry)
    monkeypatch.setattr(
        build_module,
        "build_local_artifact",
        lambda **_kwargs: destination,
    )
    validated: list[tuple[Path, ModelSpec, ModelRegistry]] = []

    def validate(
        path: Path,
        *,
        spec: ModelSpec,
        registry: ModelRegistry,
    ) -> None:
        validated.append((path, spec, registry))

    monkeypatch.setattr(build_module, "validate_artifact", validate)

    build_module.main()

    assert validated == [(destination, spec, registry)]
