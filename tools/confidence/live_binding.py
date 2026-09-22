"""Bind disabled base confidence heads to a verified live dataset pointer."""

from __future__ import annotations

import json

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastplms.registry import get_model_spec
from .experiment_artifacts import validate_evaluation_id
from .live_checkpoints import DATASET_REPO, MODEL_IDS, LiveCheckpoint


if TYPE_CHECKING:
    from huggingface_hub import HfApi


RUNTIME_HELPER_PATH = "fastplms/models/esmfold2/confidence_checkpoint.py"


def _parent_sha(value: object) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Hub repository has no valid parent commit")
    return value


def _download_json(api: HfApi, repo_id: str, path: str, repo_type: str, revision: str) -> dict[str, Any]:
    downloaded = api.hf_hub_download(repo_id, path, repo_type=repo_type, revision=revision)
    payload = json.loads(Path(downloaded).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Hub {path} must contain a JSON object")
    return payload


def bind_live_head(checkpoint: LiveCheckpoint, api: HfApi) -> str:
    """Update only config.json after verifying the public trained pointer and base weights."""
    from huggingface_hub import CommitOperationAdd

    latest = checkpoint.latest
    model_id = latest.get("model_id")
    if model_id not in MODEL_IDS or type(latest.get("update")) is not int or latest["update"] <= 0:
        raise ValueError("Live binding requires a supported model and a positive trained update")
    campaign = validate_evaluation_id(latest["campaign"])
    expected_latest = f"confidence/v2/{campaign}/{model_id}/latest.json"
    if checkpoint.latest_path != expected_latest:
        raise ValueError("Live pointer does not belong to the campaign model")
    spec = get_model_spec(model_id)
    base_hash = spec.fast.file_map["model.safetensors"].digest
    if latest.get("base_weight_sha256") != base_hash:
        raise ValueError("Live head base weights differ from the manifest")

    dataset_parent = _parent_sha(api.dataset_info(DATASET_REPO, revision="main").sha)
    published = _download_json(api, DATASET_REPO, checkpoint.latest_path, "dataset", dataset_parent)
    if (
        published.get("model_id") != model_id
        or published.get("base_weight_sha256") != base_hash
        or published.get("run_id") != latest["run_id"]
        or published.get("campaign") != latest["campaign"]
        or published.get("head_state_format") != "native_confidence_head"
        or published.get("checkpoint_kind") != "ema"
        or type(published.get("update")) is not int
        or published["update"] < checkpoint.update
    ):
        raise ValueError("Public latest pointer does not verify this trained head")
    if published["update"] == checkpoint.update and published.get("head_sha256") != latest["head_sha256"]:
        raise ValueError("Public head differs from the staged trained update")

    repo_id = spec.fast.repo_id
    info = api.model_info(repo_id, revision="main", files_metadata=True)
    parent = _parent_sha(info.sha)
    weights = [entry for entry in info.siblings if entry.rfilename == "model.safetensors"]
    if len(weights) != 1:
        raise ValueError("Published model must contain its pinned base model.safetensors")
    lfs = weights[0].lfs
    remote_hash = lfs.get("sha256") if isinstance(lfs, Mapping) else getattr(lfs, "sha256", None)
    if remote_hash != base_hash:
        raise ValueError("Published model.safetensors does not match the manifest base weights")
    if not api.file_exists(repo_id, RUNTIME_HELPER_PATH, repo_type="model", revision=parent):
        raise ValueError("Publish the confidence-loading runtime before binding live heads")
    config = _download_json(api, repo_id, "config.json", "model", parent)
    source = {
        "repo_id": DATASET_REPO,
        "repo_type": "dataset",
        "latest_path": checkpoint.latest_path,
        "revision": "main",
        "model_id": model_id,
        "base_weight_sha256": base_hash,
    }
    existing = config.get("confidence_head_source")
    if existing is not None and existing != source:
        raise ValueError("Model config already refers to a different confidence source")
    head_config = config.get("confidence_head")
    if not isinstance(head_config, dict):
        raise ValueError("Published base config omits its confidence head settings")
    if config.get("confidence_head_resolved") is not None:
        raise ValueError("Base config unexpectedly contains a resolved confidence head")
    if existing == source and head_config.get("enabled") is False:
        return parent
    config["confidence_head"] = {**head_config, "enabled": False}
    config["confidence_head_source"] = source
    payload = (json.dumps(config, indent=2, allow_nan=False) + "\n").encode("utf-8")
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        revision="main",
        parent_commit=parent,
        operations=[CommitOperationAdd("config.json", payload)],
        commit_message=f"Bind {model_id} to its verified current EMA confidence head",
    )
    return commit.oid
