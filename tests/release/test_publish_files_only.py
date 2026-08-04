"""Focused contracts for the FastPLMs Hub publisher."""

from __future__ import annotations

import sys
import pytest
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from huggingface_hub import CommitOperationAdd

from fastplms.registry import ModelSpec, get_model_registry
from tools.artifacts import publish as publish_module
from tools.artifacts.build import ArtifactError
from tools.artifacts.publish import (
    _selected_specs,
    compile_model_files,
    publish_models,
)


ROOT = Path(__file__).resolve().parents[2]


class FakeApi:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create_commit(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            oid="a" * 40,
            commit_url=f"https://huggingface.co/{kwargs['repo_id']}/commit/{'a' * 40}",
        )


def _operation_payload(operation: CommitOperationAdd) -> bytes:
    source = operation.path_or_fileobj
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    source.seek(0)
    return source.read()


def test_compile_model_files_builds_current_runtime_bundle() -> None:
    spec = get_model_registry()["esm2_8m"]

    files = compile_model_files(spec, ROOT)

    assert "README.md" in files
    assert "requirements.txt" in files
    assert "fastplms_bundle.py" in files
    assert "modeling_fastplms.py" in files
    assert "fastplms/models/esm2/modeling_fastesm.py" in files
    assert "fastplms/models/ankh/modeling_ankh.py" not in files
    compile(files["fastplms_bundle.py"], "fastplms_bundle.py", "exec")
    compile(files["modeling_fastplms.py"], "modeling_fastplms.py", "exec")


def test_selection_defaults_to_every_model_and_rejects_bad_ids() -> None:
    registry = get_model_registry()

    assert tuple(spec.id for spec in _selected_specs(registry, ())) == tuple(registry)
    with pytest.raises(ArtifactError, match="Unknown model IDs"):
        _selected_specs(registry, ("missing",))
    with pytest.raises(ArtifactError, match="must not be repeated"):
        _selected_specs(registry, ("esm2_8m", "esm2_8m"))


def test_files_only_compiles_and_uploads_without_an_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = get_model_registry()["ankh_base"]
    api = FakeApi()
    monkeypatch.setattr(
        publish_module,
        "compile_model_files",
        lambda _spec, _root: {
            "modeling_fastplms.py": b"runtime",
            "README.md": b"card",
        },
    )

    commits = publish_models(
        (spec,),
        source_root=ROOT,
        artifact_root=tmp_path / "does-not-exist",
        revision="main",
        api=api,  # type: ignore[arg-type]
        commit_message="Update runtime",
        files_only=True,
    )

    assert len(commits) == 1
    assert len(api.calls) == 1
    call = api.calls[0]
    assert call["repo_id"] == spec.fast.repo_id
    assert call["revision"] == "main"
    assert {operation.path_in_repo for operation in call["operations"]} == {
        "README.md",
        "modeling_fastplms.py",
    }


def test_default_mode_adds_prepared_artifact_files_and_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = get_model_registry()["esm2_8m"]
    artifact = tmp_path / spec.fast.repo_id.rsplit("/", maxsplit=1)[1]
    artifact.mkdir()
    (artifact / "config.json").write_bytes(b"config")
    (artifact / "model.safetensors").write_bytes(b"weights")
    (artifact / "modeling_fastplms.py").write_bytes(b"stale")
    (artifact / "source-record.json").write_bytes(b"build-only")
    monkeypatch.setattr(
        publish_module,
        "compile_model_files",
        lambda _spec, _root: {"modeling_fastplms.py": b"current"},
    )
    api = FakeApi()

    publish_models(
        (spec,),
        source_root=ROOT,
        artifact_root=tmp_path,
        revision="main",
        api=api,  # type: ignore[arg-type]
        commit_message="Update model",
        files_only=False,
    )

    operations = {
        operation.path_in_repo: operation for operation in api.calls[0]["operations"]
    }
    assert set(operations) == {
        "config.json",
        "model.safetensors",
        "modeling_fastplms.py",
    }
    assert _operation_payload(operations["model.safetensors"]) == b"weights"
    assert _operation_payload(operations["modeling_fastplms.py"]) == b"current"


def test_default_mode_explains_how_to_build_a_missing_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = get_model_registry()["esm2_8m"]
    monkeypatch.setattr(publish_module, "compile_model_files", lambda *_: {})

    with pytest.raises(ArtifactError, match=r"tools\.artifacts\.build_all esm2_8m --replace"):
        publish_models(
            (spec,),
            source_root=ROOT,
            artifact_root=tmp_path,
            revision="main",
            api=FakeApi(),  # type: ignore[arg-type]
            commit_message="Update model",
            files_only=False,
        )


def test_dry_run_compiles_but_does_not_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = get_model_registry()["esm2_8m"]
    monkeypatch.setattr(
        publish_module,
        "compile_model_files",
        lambda _spec, _root: {"modeling_fastplms.py": b"runtime"},
    )
    api = FakeApi()

    commits = publish_models(
        (spec,),
        source_root=ROOT,
        artifact_root=tmp_path,
        revision="main",
        api=api,  # type: ignore[arg-type]
        commit_message="Dry run",
        files_only=True,
        dry_run=True,
    )

    assert commits == ()
    assert api.calls == []


@pytest.mark.parametrize("files_only", [False, True])
def test_cli_accepts_default_and_files_only_modes(
    files_only: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_publish(
        specs: Iterable[ModelSpec],
        **kwargs: object,
    ) -> tuple[object, ...]:
        captured["specs"] = tuple(spec.id for spec in specs)
        captured.update(kwargs)
        return ()

    arguments = ["publish", "esm2_8m"]
    if files_only:
        arguments.append("--files-only")
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(publish_module, "publish_models", fake_publish)
    monkeypatch.setattr(publish_module, "HfApi", FakeApi)

    publish_module.main()

    assert captured["specs"] == ("esm2_8m",)
    assert captured["files_only"] is files_only
