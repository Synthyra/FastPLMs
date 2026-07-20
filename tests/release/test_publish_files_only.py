from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from huggingface_hub import CommitOperationAdd

from fastplms.registry import ModelSpec, get_model_registry
from tools.artifacts import ArtifactError, hash_file
from tools.artifacts.publish import (
    _is_weight_path,
    _selected_specs,
    main,
    prepare_files_only_plan,
    publish_files_only,
)


class FakeApi:
    def __init__(self, spec: ModelSpec, *, corrupt_weight: bool = False) -> None:
        siblings: list[SimpleNamespace] = []
        for expected in spec.fast.files:
            if not _is_weight_path(expected.path):
                continue
            digest = "0" * len(expected.digest) if corrupt_weight else expected.digest
            siblings.append(
                SimpleNamespace(
                    rfilename=expected.path,
                    blob_id=digest if expected.algorithm == "git-sha1" else None,
                    lfs={"sha256": digest} if expected.algorithm == "sha256" else None,
                )
            )
        self.info = SimpleNamespace(sha="a" * 40, siblings=siblings)
        self.model_info_calls: list[dict[str, Any]] = []
        self.create_commit_calls: list[dict[str, Any]] = []

    def model_info(self, repo_id: str, **kwargs: Any) -> SimpleNamespace:
        self.model_info_calls.append({"repo_id": repo_id, **kwargs})
        return self.info

    def create_commit(self, **kwargs: Any) -> SimpleNamespace:
        self.create_commit_calls.append(kwargs)
        return SimpleNamespace(
            oid="b" * 40,
            commit_url=f"https://huggingface.co/{kwargs['repo_id']}/commit/{'b' * 40}",
        )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _files_only_artifact(root: Path, spec: ModelSpec) -> Path:
    artifact = root / spec.fast.repo_id.split("/", maxsplit=1)[1]
    safe_files = {
        "README.md": "---\nlicense: mit\n---\n\n# Test card\n",
        "config.json": json.dumps({"fastplms_model_id": spec.id}) + "\n",
        "fastplms_bundle.py": "RUNTIME_HASH = 'test'\nRUNTIME_DATA = ()\n",
        "modeling_fastplms.py": "from .fastplms_bundle import RUNTIME_DATA\n",
        "THIRD_PARTY_NOTICES.md": "# Notices\n",
        "LICENSES/FastPLMs-Apache-2.0.txt": "Apache-2.0\n",
        "fastplms/__init__.py": "__version__ = '1.0.0'\n",
    }
    for relative_name, contents in safe_files.items():
        path = artifact.joinpath(*relative_name.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")

    provenance = {
        "model_id": spec.id,
        "canonical_weights": {
            "index": "model.safetensors.index.json",
            "shards": {
                "model-00001-of-00001.safetensors": "sha256:" + "1" * 64,
            },
        },
    }
    _write_json(artifact / "provenance.json", provenance)
    manifest = {
        relative_name: f"sha256:{hash_file(artifact.joinpath(*relative_name.split('/')))}"
        for relative_name in safe_files
    }
    manifest.update(
        {
            "provenance.json": f"sha256:{hash_file(artifact / 'provenance.json')}",
            "model.safetensors.index.json": "sha256:" + "2" * 64,
            "model-00001-of-00001.safetensors": "sha256:" + "1" * 64,
        }
    )
    _write_json(artifact / "artifact-manifest.json", manifest)
    return artifact


def test_files_only_plan_excludes_weights_and_complete_artifact_attestations(
    tmp_path: Path,
) -> None:
    spec = get_model_registry()["esm2_8m"]
    artifact = _files_only_artifact(tmp_path, spec)
    api = FakeApi(spec)

    plan = prepare_files_only_plan(
        spec,
        artifact_root=tmp_path,
        revision="main",
        api=api,  # type: ignore[arg-type]
    )

    assert plan.artifact_path == artifact
    assert plan.repo_id == spec.fast.repo_id
    assert "README.md" in plan.files
    assert "config.json" in plan.files
    assert "fastplms/__init__.py" in plan.files
    assert "artifact-manifest.json" not in plan.files
    assert "provenance.json" not in plan.files
    assert not any(_is_weight_path(path) for path in plan.files)
    assert api.model_info_calls == [
        {
            "repo_id": spec.fast.repo_id,
            "revision": "main",
            "files_metadata": True,
        }
    ]


@pytest.mark.parametrize("model_id", tuple(get_model_registry()))
def test_files_only_plan_supports_every_manifest_model(
    tmp_path: Path,
    model_id: str,
) -> None:
    spec = get_model_registry()[model_id]
    _files_only_artifact(tmp_path, spec)
    api = FakeApi(spec)

    plan = prepare_files_only_plan(
        spec,
        artifact_root=tmp_path,
        revision="main",
        api=api,  # type: ignore[arg-type]
    )

    assert plan.model_id == model_id
    assert plan.repo_id == spec.fast.repo_id
    assert not any(_is_weight_path(path) for path in plan.files)


def test_files_only_publish_uses_additions_and_parent_commit_only(
    tmp_path: Path,
) -> None:
    spec = get_model_registry()["esm2_8m"]
    _files_only_artifact(tmp_path, spec)
    api = FakeApi(spec)
    plan = prepare_files_only_plan(
        spec,
        artifact_root=tmp_path,
        revision="main",
        api=api,  # type: ignore[arg-type]
    )

    results = publish_files_only(
        (plan,),
        api=api,  # type: ignore[arg-type]
        commit_message="Update runtime files",
    )

    assert results[0].commit_oid == "b" * 40
    assert len(api.create_commit_calls) == 1
    call = api.create_commit_calls[0]
    assert call["repo_id"] == spec.fast.repo_id
    assert call["repo_type"] == "model"
    assert call["revision"] == "main"
    assert call["parent_commit"] == "a" * 40
    assert all(isinstance(operation, CommitOperationAdd) for operation in call["operations"])
    assert {operation.path_in_repo for operation in call["operations"]} == set(plan.files)
    assert not any(_is_weight_path(operation.path_in_repo) for operation in call["operations"])


def test_files_only_dry_run_performs_no_commit(tmp_path: Path) -> None:
    spec = get_model_registry()["esm2_8m"]
    _files_only_artifact(tmp_path, spec)
    api = FakeApi(spec)
    plan = prepare_files_only_plan(
        spec,
        artifact_root=tmp_path,
        revision="main",
        api=api,  # type: ignore[arg-type]
    )

    assert (
        publish_files_only(
            (plan,),
            api=api,  # type: ignore[arg-type]
            commit_message="Unused",
            dry_run=True,
        )
        == ()
    )
    assert api.create_commit_calls == []


def test_files_only_rejects_local_model_identity_mismatch(tmp_path: Path) -> None:
    spec = get_model_registry()["esm2_8m"]
    artifact = _files_only_artifact(tmp_path, spec)
    config = {"fastplms_model_id": "esm2_35m"}
    _write_json(artifact / "config.json", config)
    manifest = json.loads((artifact / "artifact-manifest.json").read_text(encoding="utf-8"))
    manifest["config.json"] = f"sha256:{hash_file(artifact / 'config.json')}"
    _write_json(artifact / "artifact-manifest.json", manifest)

    with pytest.raises(ArtifactError, match="fastplms_model_id"):
        prepare_files_only_plan(
            spec,
            artifact_root=tmp_path,
            revision="main",
            api=FakeApi(spec),  # type: ignore[arg-type]
        )


def test_files_only_rejects_remote_weight_identity_mismatch(tmp_path: Path) -> None:
    spec = get_model_registry()["esm2_8m"]
    _files_only_artifact(tmp_path, spec)

    with pytest.raises(ArtifactError, match="Hub weight identity differs"):
        prepare_files_only_plan(
            spec,
            artifact_root=tmp_path,
            revision="main",
            api=FakeApi(spec, corrupt_weight=True),  # type: ignore[arg-type]
        )


def test_files_only_selection_defaults_to_every_manifest_model() -> None:
    registry = get_model_registry()
    assert [spec.id for spec in _selected_specs(registry, ("esm2_8m",), all_models=False)] == [
        "esm2_8m"
    ]
    expected = list(registry)
    assert [spec.id for spec in _selected_specs(registry, (), all_models=False)] == expected
    assert [spec.id for spec in _selected_specs(registry, (), all_models=True)] == expected
    with pytest.raises(ArtifactError, match="not both"):
        _selected_specs(registry, ("esm2_8m",), all_models=True)
    with pytest.raises(ArtifactError, match="Unknown model IDs"):
        _selected_specs(registry, ("not-a-model",), all_models=False)


def test_publish_cli_requires_explicit_files_only_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["publish", "esm2_8m"])
    with pytest.raises(SystemExit, match="explicit --files-only"):
        main()
