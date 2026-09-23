"""Bind released confidence weights to evaluated training and base identities."""

import hashlib
import json

import pytest
import torch

from types import SimpleNamespace

from safetensors.torch import save_file

from tools.confidence import evaluated_package
from tools.confidence.artifact_validation import _validate_embedded_head


@pytest.fixture
def evaluated_directory(tmp_path, monkeypatch):
    checkpoint = {"size": 100, "sha256": "a" * 64}
    model = {
        "repo_id": "Synthyra/ESMFold2-300",
        "revision": "b" * 40,
        "files": [
            {"path": "model.safetensors", "algorithm": "sha256", "digest": "c" * 64}
        ],
    }
    request = {
        "checkpoint_inputs": {
            "v2": {**checkpoint, "snapshot": "checkpoints/v2.safetensors"}
        },
        "metadata": {"model": model},
    }
    training = {
        "status": "complete",
        "model_id": "esmfold2_300",
        "stopped_by": "planned_updates",
        "updates": 780,
        "config": {"planned_updates": 780},
        "selected_checkpoint": "final-ema.safetensors",
        "checkpoint_files": {"final-ema.safetensors": checkpoint},
        "provenance": {
            "base_repo": model["repo_id"],
            "base_revision": model["revision"],
            "base_weight_sha256": "c" * 64,
        },
        "wandb_url": "https://wandb.ai/team/project/runs/test",
    }
    (tmp_path / "inputs").mkdir()
    (tmp_path / "request.json").write_text(json.dumps(request))
    (tmp_path / "inputs/training-report.json").write_text(json.dumps(training))
    monkeypatch.setattr(
        evaluated_package,
        "verify_evaluation",
        lambda directory: {
            "model_id": "esmfold2_300",
            "split": "test",
            "evaluation_id": "campaign",
        },
    )
    return tmp_path


def test_identity_uses_exact_evaluated_head_and_original_base(evaluated_directory):
    identity = evaluated_package.evaluated_head_identity(evaluated_directory)
    assert identity["head_path"] == str(
        evaluated_directory / "checkpoints/v2.safetensors"
    )
    assert identity["head_sha256"] == "a" * 64
    assert identity["base_weight_sha256"] == "c" * 64
    assert identity["frozen_base"]["revision"] == "b" * 40


@pytest.mark.parametrize(
    "mutation",
    ["different_head", "different_base", "partial_training", "best_instead_of_final"],
)
def test_release_rejects_unmatched_training(evaluated_directory, mutation):
    path = evaluated_directory / "inputs/training-report.json"
    training = json.loads(path.read_text())
    if mutation == "different_head":
        training["checkpoint_files"]["final-ema.safetensors"]["sha256"] = "d" * 64
    elif mutation == "different_base":
        training["provenance"]["base_revision"] = "e" * 40
    elif mutation == "partial_training":
        training["updates"] = 400
    else:
        training["selected_checkpoint"] = "best-ema.safetensors"
    path.write_text(json.dumps(training))
    with pytest.raises(ValueError):
        evaluated_package.evaluated_head_identity(evaluated_directory)


def test_embedded_release_removes_external_loading_and_stale_provenance(
    evaluated_directory,
):
    identity = evaluated_package.evaluated_head_identity(evaluated_directory)
    original = {
        "confidence_head": {"enabled": False, "d_pair": 128},
        "confidence_head_source": {"revision": "main"},
        "confidence_head_resolved": {"update": 100},
        "fastplms_checkpoint_revision": "old",
        "fastplms_weights_revision": "old",
        "fastplms_runtime_revision": "old",
    }
    released = evaluated_package.embedded_config(original, identity, "f" * 64)
    assert released["confidence_head"] == {"enabled": True, "d_pair": 128}
    assert released["fastplms_checkpoint_hash"] == "f" * 64
    assert released["confidence_head_release"]["name"] == "v1"
    assert released["confidence_head_release"]["head_sha256"] == "a" * 64
    for name in original.keys() - {"confidence_head"}:
        assert name not in released
    assert original["confidence_head"]["enabled"] is False


def test_packaging_requires_new_directory_before_download(tmp_path, monkeypatch):
    def unexpected_download(*args, **kwargs):
        pytest.fail("Existing package must fail before downloading")

    monkeypatch.setattr(evaluated_package, "hf_hub_download", unexpected_download)
    with pytest.raises(FileExistsError):
        evaluated_package.prepare_evaluated_package(
            tmp_path / "evaluation", tmp_path, tmp_path
        )


def test_reload_checks_embedded_tensor_bytes_and_disallows_external_source(tmp_path):
    head = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1.0, -0.0]]))  # (output=1, input=2)
    path = tmp_path / "evaluated.safetensors"
    save_file(head.state_dict(), str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    model = SimpleNamespace(
        confidence_head=head,
        config=SimpleNamespace(
            confidence_head_source=None,
            confidence_head_release={"head_sha256": digest},
        ),
    )
    assert _validate_embedded_head(model, path) == digest
    with torch.no_grad():
        head.weight[0, 1] = (
            0.0  # scalar signed-zero change, equal numerically but different bytes
        )
    with pytest.raises(ValueError, match="tensor differs"):
        _validate_embedded_head(model, path)
    model.config.confidence_head_source = {"revision": "main"}
    with pytest.raises(ValueError, match="external source"):
        _validate_embedded_head(model, path)
