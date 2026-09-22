"""Current EMA publication retains buffers, identities, and atomic dataset updates."""

from __future__ import annotations

import json
import shutil

import pytest
import torch

from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO
from unittest.mock import Mock

from safetensors.torch import load_file

from tools.confidence import live_checkpoints as live


def checkpoint_state(update: int = 4) -> dict[str, Any]:
    return {
        "update": update,
        "elapsed": 30.0,
        "config": {"model_id": "esmfold2_300", "seed": 17},
        "provenance": {
            "model_id": "esmfold2_300",
            "base_repo": "Synthyra/ESMFold2-300",
            "base_revision": "a" * 40,
            "base_weight_sha256": "b" * 64,
            "source_files": {"tools/confidence/online_training.py": {"sha256": "c" * 64, "size": 12}},
            "donor_revision": "d" * 40,
        },
        "head": {"weight": torch.tensor([1.0, 2.0]), "buffer": torch.tensor([3])},  # each (2,) / (1,)
        "ema": {"weight": torch.tensor([4.0, 5.0])},  # (2,), trainable parameters only
    }


@pytest.fixture
def staged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> live.LiveCheckpoint:
    campaign = tmp_path / "campaign"
    run = campaign / "runs" / "esmfold2_300" / "v2"
    run.mkdir(parents=True)
    (run / "last.pt").write_bytes(b"trusted training checkpoint")
    (run / "wandb-id.txt").write_text("run123", encoding="utf-8")
    loader = Mock(return_value=checkpoint_state())
    monkeypatch.setattr(torch, "load", loader)
    result = live.stage_live_checkpoint(campaign, "esmfold2_300", tmp_path / "staged")
    assert result is not None
    assert loader.call_args.kwargs == {"map_location": "cpu", "weights_only": False}
    assert loader.call_args.args[0] != run / "last.pt"
    return result


def test_ema_overlay_preserves_nontrainable_state(staged: live.LiveCheckpoint) -> None:
    state = load_file(staged.directory / "head.safetensors")  # saved weights: (2,); buffer: (1,)
    assert torch.equal(state["weight"], torch.tensor([4.0, 5.0]))
    assert torch.equal(state["buffer"], torch.tensor([3]))
    assert staged.latest["evaluation_status"] == "pending"
    assert staged.latest["head_state_format"] == "native_confidence_head"
    assert staged.latest["checkpoint_kind"] == "ema"
    metadata = json.loads((staged.directory / "provenance.json").read_text())
    assert metadata["config"] == checkpoint_state()["config"]
    assert metadata["provenance"]["source_files"]
    assert staged.latest["head_sha256"] == live.file_identity(staged.directory / "head.safetensors").sha256


@pytest.mark.parametrize("problem", ["empty", "unknown", "shape", "dtype", "nonfinite"])
def test_invalid_ema_fails_closed(problem: str) -> None:
    checkpoint = checkpoint_state()
    if problem == "empty":
        checkpoint["ema"] = {}
    elif problem == "unknown":
        checkpoint["ema"]["unknown"] = torch.tensor([1.0])  # (1,)
    elif problem == "shape":
        checkpoint["ema"]["weight"] = torch.ones(3)  # (3,), incompatible with (2,)
    elif problem == "dtype":
        checkpoint["ema"]["weight"] = torch.ones(2, dtype=torch.float64)  # (2,)
    else:
        checkpoint["ema"]["weight"] = torch.tensor([float("nan"), 1.0])  # (2,)
    with pytest.raises(ValueError):
        live.ema_head_state(checkpoint)


def test_snapshot_keeps_old_inode_during_atomic_replacement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, snapshot = tmp_path / "last.pt", tmp_path / "snapshot.pt"
    source.write_bytes(b"old complete checkpoint")
    original_copy = shutil.copyfileobj

    def replace_then_copy(incoming: BinaryIO, outgoing: BinaryIO) -> None:
        replacement = tmp_path / "next.pt"
        replacement.write_bytes(b"new complete checkpoint")
        replacement.replace(source)
        original_copy(incoming, outgoing)

    monkeypatch.setattr(live.shutil, "copyfileobj", replace_then_copy)
    live._copy_checkpoint(source, snapshot)
    assert snapshot.read_bytes() == b"old complete checkpoint"
    assert source.read_bytes() == b"new complete checkpoint"


def test_missing_and_untrained_checkpoints_are_not_promoted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    campaign = tmp_path / "campaign"
    assert live.stage_live_checkpoint(campaign, "esmfold2_300", tmp_path / "output") is None
    run = campaign / "runs" / "esmfold2_300" / "v2"
    run.mkdir(parents=True)
    (run / "last.pt").write_bytes(b"trusted")
    (run / "wandb-id.txt").write_text("run123")
    monkeypatch.setattr(torch, "load", Mock(return_value=checkpoint_state(0)))
    assert live.stage_live_checkpoint(campaign, "esmfold2_300", tmp_path / "output") is None


def fake_api() -> Mock:
    api = Mock()
    api.dataset_info.return_value = SimpleNamespace(sha="f" * 40)
    api.file_exists.return_value = False
    api.create_commit.return_value = SimpleNamespace(oid="e" * 40)
    return api


def test_publication_is_scoped_add_only_and_parent_protected(staged: live.LiveCheckpoint) -> None:
    api = fake_api()
    assert live.publish_live_checkpoint(staged, api) == "e" * 40
    commit = api.create_commit.call_args.kwargs
    assert commit["repo_id"] == live.DATASET_REPO
    assert commit["repo_type"] == "dataset"
    assert commit["parent_commit"] == "f" * 40
    assert commit["revision"] == "main"
    operations = commit["operations"]
    assert len(operations) == 3
    assert {type(operation).__name__ for operation in operations} == {"CommitOperationAdd"}
    assert {operation.path_in_repo for operation in operations} == {
        staged.latest_path, staged.latest["head_path"], staged.latest["provenance_path"]
    }
    assert "token" not in commit
    api.create_repo.assert_not_called()
    api.delete_file.assert_not_called()


@pytest.mark.parametrize("previous_update", [4, 8])
def test_latest_never_repeats_or_moves_backwards(staged: live.LiveCheckpoint, previous_update: int) -> None:
    api = fake_api()
    previous = {**staged.latest, "update": previous_update}
    pointer = staged.directory / "previous.json"
    pointer.write_text(json.dumps(previous))
    api.file_exists.side_effect = lambda repo, name, **kwargs: name == staged.latest_path
    api.hf_hub_download.return_value = str(pointer)
    assert live.publish_live_checkpoint(staged, api) is None
    assert api.hf_hub_download.call_args.kwargs["revision"] == "f" * 40
    api.create_commit.assert_not_called()


def test_changed_staged_head_is_rejected_before_network(staged: live.LiveCheckpoint) -> None:
    (staged.directory / "head.safetensors").write_bytes(b"changed")
    api = fake_api()
    with pytest.raises(ValueError, match="changed"):
        live.publish_live_checkpoint(staged, api)
    api.dataset_info.assert_not_called()


def test_conflicting_immutable_path_fails(staged: live.LiveCheckpoint) -> None:
    api = fake_api()
    conflict = staged.directory / "conflict"
    conflict.write_bytes(b"other checkpoint")
    api.file_exists.side_effect = lambda repo, name, **kwargs: name == staged.latest["head_path"]
    api.hf_hub_download.return_value = str(conflict)
    with pytest.raises(ValueError, match="Immutable"):
        live.publish_live_checkpoint(staged, api)
    api.create_commit.assert_not_called()


def test_parent_conflict_is_propagated_without_retry(staged: live.LiveCheckpoint) -> None:
    api = fake_api()
    api.create_commit.side_effect = RuntimeError("parent changed")
    with pytest.raises(RuntimeError, match="parent changed"):
        live.publish_live_checkpoint(staged, api)
    assert api.create_commit.call_count == 1
