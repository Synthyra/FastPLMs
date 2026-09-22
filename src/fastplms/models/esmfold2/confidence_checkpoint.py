"""Load explicit head-only confidence checkpoints without changing base weights."""

from __future__ import annotations

import hashlib
import json
import re
import torch

from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers.utils.hub import extract_commit_hash

if TYPE_CHECKING:
    from .modeling_esmfold2_experimental import ESMFold2ExperimentalModel


_DOWNLOAD_OPTIONS = (
    "cache_dir",
    "token",
    "local_files_only",
    "force_download",
)


def _checkpoint_metadata(path: Path, source: dict[str, str]) -> dict[str, Any]:
    if path.stat().st_size > 1_000_000:
        raise ValueError("Confidence checkpoint metadata exceeds one megabyte.")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError("Unsupported confidence checkpoint metadata schema.")
    for key in ("repo_id", "repo_type", "model_id", "base_weight_sha256"):
        if metadata.get(key) != source[key]:
            raise ValueError(
                f"Confidence checkpoint {key} does not match its configured source."
            )
    update = metadata.get("update")
    if type(update) is not int or update <= 0:
        raise ValueError("A published training head must have a positive update count.")
    if metadata.get("evaluation_status") != "pending":
        raise ValueError(
            "Rolling confidence heads must explicitly declare pending evaluation."
        )
    if (
        metadata.get("head_state_format") != "native_confidence_head"
        or metadata.get("checkpoint_kind") != "ema"
    ):
        raise ValueError("Expected a native EMA confidence-head state dictionary.")
    head_path = metadata.get("head_path")
    if not isinstance(head_path, str) or not head_path:
        raise ValueError("Confidence checkpoint metadata requires head_path.")
    relative_path = PurePosixPath(head_path)
    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
        or "\\" in head_path
        or relative_path.suffix != ".safetensors"
    ):
        raise ValueError(
            "Confidence checkpoint head_path must be a relative safetensors path."
        )
    digest = metadata.get("head_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(
            "Confidence checkpoint metadata requires a SHA256 head digest."
        )
    if type(metadata.get("head_size")) is not int or metadata["head_size"] <= 0:
        raise ValueError(
            "Confidence checkpoint metadata requires a positive head_size."
        )
    return metadata


def install_confidence_checkpoint(
    model: ESMFold2ExperimentalModel, *, download_options: dict[str, Any]
) -> None:
    """Resolve a rolling pointer once, verify its immutable head, and embed it."""

    from .modeling_esmfold2_experimental import ConfidenceHead

    source = model.config.confidence_head_source
    if source is None:
        return
    if model.confidence_head is not None:
        raise ValueError(
            "An external confidence checkpoint cannot replace an embedded head."
        )
    devices = {parameter.device for parameter in model.parameters()}
    device_map = getattr(model, "hf_device_map", {})
    if (
        len(devices) != 1
        or any(device.type == "meta" for device in devices)
        or "disk" in device_map.values()
    ):
        raise ValueError(
            "External confidence heads require a single resident model device; offload is unsupported."
        )
    if any(
        getattr(getattr(module, "_hf_hook", None), "offload", False)
        for module in model.modules()
    ):
        raise ValueError("External confidence heads do not support an offloaded model.")
    options = {
        key: download_options[key]
        for key in _DOWNLOAD_OPTIONS
        if key in download_options
    }
    latest = Path(
        hf_hub_download(
            repo_id=source["repo_id"],
            repo_type="dataset",
            filename=source["latest_path"],
            revision=source["revision"],
            **options,
        )
    )
    revision = extract_commit_hash(str(latest), None)
    if revision is None or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Could not resolve an immutable confidence dataset revision.")
    metadata = _checkpoint_metadata(latest, source)
    checkpoint = Path(
        hf_hub_download(
            repo_id=source["repo_id"],
            repo_type="dataset",
            filename=metadata["head_path"],
            revision=revision,
            **options,
        )
    )
    if checkpoint.stat().st_size != metadata["head_size"]:
        raise ValueError(
            "Confidence checkpoint size does not match its publication metadata."
        )
    with checkpoint.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != metadata["head_sha256"]:
        raise ValueError(
            "Confidence checkpoint SHA256 does not match its publication metadata."
        )
    state = load_file(str(checkpoint), device="cpu")
    # Head construction must not consume the caller's folding random-number stream.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        head = ConfidenceHead(model.config)
    expected = head.state_dict()
    if set(state) != set(expected):
        raise ValueError("Confidence checkpoint keys do not match the native head.")
    for key, tensor in state.items():
        # Every parameter/buffer must retain the native architecture's exact shape.
        if (
            tensor.shape != expected[key].shape
            or tensor.is_floating_point() != expected[key].is_floating_point()
        ):
            raise ValueError(
                f"Confidence checkpoint tensor {key!r} has an incompatible shape or dtype."
            )
        if not torch.isfinite(tensor).all().item():
            raise ValueError(
                f"Confidence checkpoint tensor {key!r} contains nonfinite values."
            )
    head.load_state_dict(state, strict=True)
    parameter = next(model.parameters())
    head.to(device=parameter.device, dtype=parameter.dtype)
    head.train(model.training)
    head.set_kernel_backend(model._kernel_backend)
    model.confidence_head = head
    model.config.confidence_head.enabled = True
    model.config.confidence_head_resolved = {**metadata, "dataset_revision": revision}
    # save_pretrained now persists a self-contained head and its exact provenance.
    model.config.confidence_head_source = None
