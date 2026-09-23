"""Embed an exactly evaluated confidence head in its frozen model checkpoint."""

from __future__ import annotations

import hashlib
import json

import torch

from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

from fastplms.models.esmfold2.configuration_esmfold2 import ESMFold2Config
from fastplms.models.esmfold2.modeling_esmfold2_experimental import ConfidenceHead
from fastplms.registry import get_model_spec
from tools.artifacts.build import _artifact_auto_map, _decode_runtime_bundle
from tools.artifacts.publish import compile_model_files
from .config import MODEL_IDS
from .experiment_artifacts import verify_evaluation
from .packaging import merge_head, verify_folding_state


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evaluated_head_identity(directory: Path) -> dict[str, Any]:
    """Bind completed training, the evaluated snapshot, and its frozen base."""
    completion = verify_evaluation(directory)
    model_id = completion["model_id"]
    if model_id not in MODEL_IDS or completion["split"] != "test":
        raise ValueError(
            "Packaging requires a supported model's completed test evaluation"
        )
    request = _read(directory / "request.json")
    training = _read(directory / "inputs/training-report.json")
    selected = training.get("selected_checkpoint")
    if (
        training.get("status") != "complete"
        or training.get("model_id") != model_id
        or training.get("stopped_by") != "planned_updates"
        or training.get("updates") != training.get("config", {}).get("planned_updates")
        or selected != "final-ema.safetensors"
    ):
        raise ValueError(
            "Packaging requires the completed final EMA training checkpoint"
        )
    # Historical raw records retain their original label; the public release calls it v1.
    checkpoint = request["checkpoint_inputs"]["v2"]
    identity = {name: checkpoint[name] for name in ("size", "sha256")}
    if training["checkpoint_files"].get(selected) != identity:
        raise ValueError(
            "The evaluated head differs from the final training checkpoint"
        )
    model = request["metadata"]["model"]
    base_hash = next(
        item["digest"] for item in model["files"] if item["path"] == "model.safetensors"
    )
    provenance = training["provenance"]
    if (
        provenance.get("base_repo") != model["repo_id"]
        or provenance.get("base_revision") != model["revision"]
        or provenance.get("base_weight_sha256") != base_hash
    ):
        raise ValueError(
            "Training and evaluation identify different frozen base checkpoints"
        )
    return {
        "model_id": model_id,
        "head_path": str(directory / checkpoint["snapshot"]),
        "head_sha256": identity["sha256"],
        "frozen_base": model,
        "base_weight_sha256": base_hash,
        "evaluation_id": completion["evaluation_id"],
        "evaluation_request_sha256": _sha256(directory / "request.json"),
        "training_url": training["wandb_url"],
        "updates": training["updates"],
    }


def embedded_config(
    config: dict[str, Any], identity: dict[str, Any], weight_sha256: str
) -> dict[str, Any]:
    """Remove rolling-head state when publishing a self-contained checkpoint."""
    payload = dict(config)
    settings = payload.get("confidence_head")
    if not isinstance(settings, dict):
        raise ValueError("Base config omits native confidence head settings")
    payload["confidence_head"] = {**settings, "enabled": True}
    for name in (
        "confidence_head_source",
        "confidence_head_resolved",
        "fastplms_checkpoint_revision",
        "fastplms_weights_revision",
        "fastplms_runtime_revision",
        "fastplms_source_tree_sha256",
        "fastplms_release_tool_revision",
        "fastplms_release_tool_sha256",
    ):
        payload.pop(name, None)
    payload["fastplms_model_id"] = identity["model_id"]
    payload["fastplms_checkpoint_repo_id"] = identity["frozen_base"]["repo_id"]
    payload["fastplms_checkpoint_hash"] = weight_sha256
    payload["confidence_head_release"] = {
        "name": "v1",
        "head_sha256": identity["head_sha256"],
        "base_weight_sha256": identity["base_weight_sha256"],
        "training_url": identity["training_url"],
        "updates": identity["updates"],
        "evaluation_request_sha256": identity["evaluation_request_sha256"],
    }
    return payload


def prepare_evaluated_package(
    directory: Path, output: Path, source_root: Path
) -> dict[str, Any]:
    """Stage weights and runtime files; publication and GPU validation are separate steps."""
    if output.exists():
        raise FileExistsError(f"Package output already exists: {output}")
    identity = evaluated_head_identity(directory)
    source = identity["frozen_base"]
    weights_path = Path(
        hf_hub_download(
            source["repo_id"], "model.safetensors", revision=source["revision"]
        )
    )
    config_path = Path(
        hf_hub_download(source["repo_id"], "config.json", revision=source["revision"])
    )
    if _sha256(weights_path) != identity["base_weight_sha256"]:
        raise ValueError("Downloaded base weights differ from the evaluated checkpoint")
    encoded = config_path.read_bytes()
    config_hash = hashlib.sha1(f"blob {len(encoded)}\0".encode() + encoded).hexdigest()
    expected_config = next(
        item for item in source["files"] if item["path"] == "config.json"
    )
    if expected_config != {
        "path": "config.json",
        "algorithm": "git-sha1",
        "digest": config_hash,
    }:
        raise ValueError("Downloaded config differs from the evaluated checkpoint")
    head = load_file(
        identity["head_path"]
    )  # parameter-specific native head tensor shapes
    with torch.random.fork_rng(devices=[]):
        native_head = ConfidenceHead(ESMFold2Config(**json.loads(encoded)))
    native_head.load_state_dict(head, strict=True)
    del native_head
    base = load_file(str(weights_path))  # parameter-specific frozen model tensor shapes
    merged = merge_head(
        base, head
    )  # preserves every non-confidence tensor shape and byte
    preserved = verify_folding_state(base, merged)
    output.mkdir(parents=True)
    save_file(merged, str(output / "model.safetensors"), metadata={"format": "pt"})
    reloaded = load_file(
        str(output / "model.safetensors")
    )  # same named parameter shapes
    verify_folding_state(base, reloaded)
    for name, value in head.items():
        # Verify serialized head dtype, shape and bytes, including signed zeros.
        saved = reloaded[f"confidence_head.{name}"]  # same parameter shape as value
        if (
            saved.dtype != value.dtype
            or saved.shape != value.shape
            or not torch.equal(
                saved.contiguous().reshape(-1).view(torch.uint8),
                value.contiguous().reshape(-1).view(torch.uint8),
            )
        ):
            raise ValueError(
                f"Serialized confidence tensor differs from evaluated head: {name}"
            )
    del base, merged, reloaded, head
    spec = get_model_spec(identity["model_id"])
    for relative, payload in compile_model_files(spec, source_root).items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    weight_hash = _sha256(output / "model.safetensors")
    config = embedded_config(json.loads(encoded), identity, weight_hash)
    config["auto_map"] = _artifact_auto_map(spec)
    config["fastplms_runtime_bundle_sha256"], _ = _decode_runtime_bundle(
        output / "fastplms_bundle.py"
    )
    (output / "config.json").write_text(
        json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return {
        **identity,
        "status": "prepared",
        "artifact": str(output),
        "weight_sha256": weight_hash,
        "config_sha256": _sha256(output / "config.json"),
        "preserved_folding_tensors": preserved,
        "embedded_head_identity": True,
        "complete_artifact_compliance": False,
    }
