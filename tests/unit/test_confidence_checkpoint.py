"""Offline contracts for explicitly bound, independently published confidence heads."""

from __future__ import annotations

import hashlib
import json
import pytest
import torch

from pathlib import Path
from safetensors.torch import save_file

from fastplms.models.esmfold2 import confidence_checkpoint
from fastplms.models.esmfold2.configuration_esmfold2 import ESMFold2Config
from fastplms.models.esmfold2.modeling_esmfold2_classification import (
    ESMFold2ExperimentalForSequenceClassification,
)
from fastplms.models.esmfold2.modeling_esmfold2_experimental import (
    ConfidenceHead,
    ESMFold2ExperimentalModel,
)
from tests.unit.test_esmfold2_small import _tiny_experimental_config


REVISION = "a" * 40
SOURCE = {
    "repo_id": "Synthyra/FastPLMs-artifacts",
    "repo_type": "dataset",
    "latest_path": "confidence/v2/test/esmfold2_300/latest.json",
    "revision": "main",
    "model_id": "esmfold2_300",
    "base_weight_sha256": "b" * 64,
}


@pytest.fixture
def published_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = _tiny_experimental_config()
    config.confidence_head_source = dict(SOURCE)
    model = ESMFold2ExperimentalModel(config).eval()
    state = ConfidenceHead(config).state_dict()
    snapshot = (
        tmp_path / "datasets--Synthyra--FastPLMs-artifacts" / "snapshots" / REVISION
    )
    latest = snapshot / SOURCE["latest_path"]
    head_path = "confidence/v2/test/esmfold2_300/run/updates/00000001/head.safetensors"
    checkpoint = snapshot / head_path
    checkpoint.parent.mkdir(parents=True)
    save_file(state, str(checkpoint))
    metadata = {
        **SOURCE,
        "schema_version": 1,
        "update": 1,
        "evaluation_status": "pending",
        "head_state_format": "native_confidence_head",
        "checkpoint_kind": "ema",
        "head_path": head_path,
        "head_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "head_size": checkpoint.stat().st_size,
    }
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(json.dumps(metadata), encoding="utf-8")
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(snapshot / kwargs["filename"])

    monkeypatch.setattr(confidence_checkpoint, "hf_hub_download", download)
    return model, state, latest, checkpoint, metadata, calls


def test_checkpoint_loading_pins_snapshot_preserves_base_and_rng(
    published_head,
) -> None:
    model, state, _latest, _checkpoint, metadata, calls = published_head
    original = {key: value.clone() for key, value in model.state_dict().items()}
    random_state = torch.random.get_rng_state().clone()

    confidence_checkpoint.install_confidence_checkpoint(
        model, download_options={"local_files_only": True, "cache_dir": "/test-cache"}
    )

    assert calls[0]["revision"] == "main"
    assert calls[1]["revision"] == REVISION
    assert all(
        call["local_files_only"] and call["cache_dir"] == "/test-cache"
        for call in calls
    )
    assert torch.equal(random_state, torch.random.get_rng_state())
    assert model.config.confidence_head.enabled
    assert model.config.confidence_head_source is None
    assert model.config.confidence_head_resolved == {
        **metadata,
        "dataset_revision": REVISION,
    }
    assert model.confidence_head is not None and not model.confidence_head.training
    assert all(
        torch.equal(value, model.state_dict()[key]) for key, value in original.items()
    )
    assert all(
        torch.equal(value, model.confidence_head.state_dict()[key])
        for key, value in state.items()
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("update", 0),
        ("update", True),
        ("head_sha256", "0" * 64),
        ("head_size", 1),
        ("base_weight_sha256", "c" * 64),
        ("model_id", "esmfold2_600"),
        ("evaluation_status", "passed"),
        ("head_path", "../head.safetensors"),
        ("head_state_format", "whole_model"),
    ],
)
def test_checkpoint_rejects_invalid_publication(
    published_head, field: str, value: object
) -> None:
    model, _state, latest, _checkpoint, metadata, _calls = published_head
    metadata[field] = value
    latest.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        confidence_checkpoint.install_confidence_checkpoint(model, download_options={})
    assert model.confidence_head is None
    assert not model.config.confidence_head.enabled
    assert model.config.confidence_head_source == SOURCE


@pytest.mark.parametrize("damage", ["keys", "shape", "nonfinite"])
def test_checkpoint_rejects_invalid_native_state(published_head, damage: str) -> None:
    model, state, latest, checkpoint, metadata, _calls = published_head
    key = next(iter(state))
    if damage == "keys":
        del state[key]
    elif damage == "shape":
        state[key] = torch.zeros(1)
    else:
        state[key] = torch.full_like(state[key], float("nan"))
    save_file(state, str(checkpoint))
    metadata["head_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    metadata["head_size"] = checkpoint.stat().st_size
    latest.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        confidence_checkpoint.install_confidence_checkpoint(model, download_options={})
    assert model.confidence_head is None


def test_checkpoint_rejects_offload_before_download(published_head) -> None:
    model, _state, _latest, _checkpoint, _metadata, calls = published_head
    model.hf_device_map = {"": "cpu", "folding_trunk": "disk"}
    with pytest.raises(ValueError, match="offload"):
        confidence_checkpoint.install_confidence_checkpoint(model, download_options={})
    assert not calls


def test_checkpoint_inherits_resident_model_dtype(published_head) -> None:
    model, _state, _latest, _checkpoint, _metadata, _calls = published_head
    model.to(dtype=torch.bfloat16)
    confidence_checkpoint.install_confidence_checkpoint(model, download_options={})
    assert all(
        parameter.dtype == torch.bfloat16
        for parameter in model.confidence_head.parameters()
    )
    assert model.confidence_head.boundaries.dtype == torch.bfloat16


def test_checkpoint_from_pretrained_opt_out_and_embedded_reload(
    published_head, tmp_path: Path
) -> None:
    model, state, _latest, _checkpoint, _metadata, calls = published_head
    base = tmp_path / "base"
    model.save_pretrained(base)
    disabled = ESMFold2ExperimentalModel.from_pretrained(
        base,
        load_esmc=False,
        load_confidence_head=False,
        local_files_only=True,
    )
    assert disabled.confidence_head is None and not calls
    loaded, information = ESMFold2ExperimentalModel.from_pretrained(
        base,
        load_esmc=False,
        local_files_only=True,
        output_loading_info=True,
    )
    assert not information["missing_keys"] and not information["unexpected_keys"]
    assert loaded.confidence_head is not None
    assert len(calls) == 2
    embedded = tmp_path / "embedded"
    loaded.save_pretrained(embedded)
    reloaded = ESMFold2ExperimentalModel.from_pretrained(
        embedded,
        load_esmc=False,
        local_files_only=True,
    )
    assert len(calls) == 2
    assert (
        reloaded.config.confidence_head_resolved
        == loaded.config.confidence_head_resolved
    )
    assert reloaded.config.confidence_head_source is None
    assert all(
        torch.equal(value, reloaded.confidence_head.state_dict()[key])
        for key, value in state.items()
    )


def test_checkpoint_classifier_keeps_confidence_frozen(
    published_head, tmp_path: Path
) -> None:
    model, _state, _latest, _checkpoint, _metadata, _calls = published_head
    model.config.classifier_hidden_size = 8
    base = tmp_path / "base-classifier"
    model.save_pretrained(base)
    classifier = ESMFold2ExperimentalForSequenceClassification.from_pretrained(
        base,
        load_esmc=False,
        local_files_only=True,
    )
    assert classifier.confidence_head is not None
    assert not any(
        parameter.requires_grad for parameter in classifier.confidence_head.parameters()
    )
    assert all(
        parameter.requires_grad for parameter in classifier.classifier.parameters()
    )


def test_checkpoint_config_rejects_external_and_embedded_heads() -> None:
    config = _tiny_experimental_config().to_dict()
    config["confidence_head_source"] = dict(SOURCE)
    config["confidence_head"]["enabled"] = True
    with pytest.raises(ValueError, match="disabled experimental"):
        ESMFold2Config(**config)


def test_unbound_checkpoint_constructor_does_not_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(**_kwargs):
        raise AssertionError("A constructor must not download a confidence head.")

    monkeypatch.setattr(confidence_checkpoint, "hf_hub_download", forbidden)
    config = _tiny_experimental_config()
    config.confidence_head_source = dict(SOURCE)
    model = ESMFold2ExperimentalModel(config)
    assert model.confidence_head is None
