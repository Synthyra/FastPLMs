"""Live confidence binding changes only configs after checking public base and head identities."""

from __future__ import annotations

import json

import pytest

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

from tools.confidence import live_binding as binding
from tools.confidence.live_checkpoints import DATASET_REPO, LiveCheckpoint


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[LiveCheckpoint, Mock, dict[str, Any]]:
    latest = {
        "model_id": "esmfold2_300",
        "campaign": "campaign",
        "run_id": "run123",
        "update": 4,
        "base_weight_sha256": "b" * 64,
        "head_sha256": "c" * 64,
        "checkpoint_kind": "ema",
        "head_state_format": "native_confidence_head",
    }
    checkpoint = LiveCheckpoint(tmp_path, "confidence/v2/campaign/esmfold2_300/latest.json", latest)
    spec = SimpleNamespace(fast=SimpleNamespace(
        repo_id="Synthyra/ESMFold2-300",
        file_map={"model.safetensors": SimpleNamespace(digest="b" * 64)},
    ))
    spec.confidence_training_base = spec.fast
    monkeypatch.setattr(binding, "get_model_spec", lambda model_id: spec)
    api = Mock()
    api.dataset_info.return_value = SimpleNamespace(sha="d" * 40)
    api.model_info.return_value = SimpleNamespace(
        sha="a" * 40,
        siblings=[SimpleNamespace(rfilename="model.safetensors", lfs=SimpleNamespace(sha256="b" * 64))],
    )
    api.file_exists.return_value = True
    api.create_commit.return_value = SimpleNamespace(oid="e" * 40)
    config = {"confidence_head": {"enabled": False, "channels": 128}, "other": {"unchanged": [1, 2]}}

    def download(repo_id: str, filename: str, *, repo_type: str, revision: str) -> str:
        path = tmp_path / f"{repo_type}.json"
        payload = latest if repo_type == "dataset" else config
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    api.hf_hub_download.side_effect = download
    return checkpoint, api, config


def test_binding_writes_only_parent_protected_config(hub: tuple[LiveCheckpoint, Mock, dict[str, Any]]) -> None:
    checkpoint, api, original = hub
    assert binding.bind_live_head(checkpoint, api) == "e" * 40
    commit = api.create_commit.call_args.kwargs
    assert commit["repo_id"] == "Synthyra/ESMFold2-300"
    assert commit["repo_type"] == "model"
    assert commit["parent_commit"] == "a" * 40
    assert len(commit["operations"]) == 1
    operation = commit["operations"][0]
    assert type(operation).__name__ == "CommitOperationAdd"
    assert operation.path_in_repo == "config.json"
    config = json.loads(operation.path_or_fileobj)
    assert config["other"] == original["other"]
    assert config["confidence_head"] == original["confidence_head"]
    assert config["confidence_head_source"] == {
        "repo_id": DATASET_REPO,
        "repo_type": "dataset",
        "latest_path": checkpoint.latest_path,
        "revision": "main",
        "model_id": "esmfold2_300",
        "base_weight_sha256": "b" * 64,
    }
    assert api.hf_hub_download.call_args.kwargs["revision"] == "a" * 40
    assert "token" not in commit
    api.create_repo.assert_not_called()
    api.delete_file.assert_not_called()


def test_matching_binding_does_not_commit(hub: tuple[LiveCheckpoint, Mock, dict[str, Any]]) -> None:
    checkpoint, api, config = hub
    binding.bind_live_head(checkpoint, api)
    config.update(json.loads(api.create_commit.call_args.kwargs["operations"][0].path_or_fileobj))
    api.create_commit.reset_mock()
    assert binding.bind_live_head(checkpoint, api) == "a" * 40
    api.create_commit.assert_not_called()


@pytest.mark.parametrize("problem", ["weights", "helper", "config", "source", "resolved", "untrained", "published"])
def test_binding_fails_closed(hub: tuple[LiveCheckpoint, Mock, dict[str, Any]], problem: str) -> None:
    checkpoint, api, config = hub
    if problem == "weights":
        api.model_info.return_value.siblings[0].lfs.sha256 = "f" * 64
    elif problem == "helper":
        api.file_exists.return_value = False
    elif problem == "config":
        config.pop("confidence_head")
    elif problem == "source":
        config["confidence_head_source"] = {"latest_path": "another-campaign/latest.json"}
    elif problem == "resolved":
        config["confidence_head_resolved"] = {"update": 1}
    elif problem == "untrained":
        checkpoint.latest["update"] = 0
    else:
        checkpoint.latest["checkpoint_kind"] = "raw_parameters"
    with pytest.raises(ValueError):
        binding.bind_live_head(checkpoint, api)
    api.create_commit.assert_not_called()


def test_model_parent_race_is_not_retried(hub: tuple[LiveCheckpoint, Mock, dict[str, Any]]) -> None:
    checkpoint, api, _ = hub
    api.create_commit.side_effect = RuntimeError("parent changed")
    with pytest.raises(RuntimeError, match="parent changed"):
        binding.bind_live_head(checkpoint, api)
    assert api.create_commit.call_count == 1
