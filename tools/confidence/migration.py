"""Verify durable v2 resume inputs and record an explicitly stopped worker handoff."""

from __future__ import annotations

import hashlib
import json
import re
import struct

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path


MODEL_IDS = frozenset({"esmfold2_300", "esmfold2_600"})
TERMINAL_CALL_STATUSES = frozenset({"TERMINATED", "CANCELLED", "FAILED", "SUCCESS"})
MAX_HEADER_BYTES = 16 * 1024 * 1024


def _model_directory(root: Path, model_id: str) -> Path:
    if model_id not in MODEL_IDS:
        raise ValueError("Migration supports only the two v2 confidence models")
    directory = root / "runs" / model_id
    if not directory.resolve().is_relative_to(root.resolve()):
        raise ValueError("Model directory escapes the campaign")
    return directory


def _file_identity(path: Path) -> dict[str, int | str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Resume input must be a regular file: {path.name}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"size": path.stat().st_size, "sha256": digest}


def _cache_header(path: Path) -> tuple[dict[str, object], str]:
    # NumPy safe_open can materialize large cache storage; inspect bounded header bytes only.
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated validation cache header: {path.name}")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size > MAX_HEADER_BYTES:
            raise ValueError(f"Validation cache header is too large: {path.name}")
        payload = stream.read(header_size)
    if len(payload) != header_size:
        raise ValueError(f"Truncated validation cache header: {path.name}")
    header = json.loads(payload)
    if not isinstance(header, dict):
        raise ValueError(f"Invalid validation cache header: {path.name}")
    tensor = header.get("x_pred", {})
    if not isinstance(tensor, dict) or not isinstance(header.get("__metadata__", {}), dict):
        raise ValueError(f"Invalid validation cache tensor or metadata: {path.name}")
    offsets = tensor.get("data_offsets", [])
    if not isinstance(offsets, list) or len(offsets) != 2 or any(type(value) is not int for value in offsets) or not 0 <= offsets[0] <= offsets[1] <= path.stat().st_size - 8 - header_size:
        raise ValueError(f"Cached coordinate offsets exceed file bounds: {path.name}")
    return header, hashlib.sha256(payload).hexdigest()


def verify_validation_cache(
    root: Path,
    model_id: str,
    *,
    num_loops: int = 3,
    num_sampling_steps: int = 50,
    samples_per_target: int = 4,
) -> dict[str, object]:
    """Validate partial cache headers without loading tensor payloads or initializing Torch."""
    from .target_splits import load_split

    directory = _model_directory(root, model_id) / (
        f"validation-cache-{num_loops}-loops-{num_sampling_steps}-steps-"
        f"{samples_per_target}-samples"
    )
    if directory.is_symlink():
        raise ValueError("Validation cache cannot be a symlink")
    # Match host.split_targets exactly; cache filenames address this ordered list.
    targets = sorted(
        (target for target in load_split(root / "splits") if target["split"] == "validation"),
        key=lambda target: hashlib.sha256(str(target["target_id"]).encode()).hexdigest(),
    )
    if not targets:
        raise ValueError("Migration requires a nonempty verified validation split")
    entries = {}
    for path in sorted(directory.glob("*.safetensors")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Cache entry must be a regular file: {path.name}")
        if not re.fullmatch(r"[0-9]{4}\.safetensors", path.name):
            raise ValueError(f"Unexpected validation cache filename: {path.name}")
        index = int(path.stem)
        if index >= len(targets):
            raise ValueError(f"Cache index exceeds the validation split: {path.name}")
        target = targets[index]
        header, header_digest = _cache_header(path)
        metadata = header.get("__metadata__", {})
        if metadata.get("target_id") != str(target["target_id"]):
            raise ValueError(f"Cached target differs from the validation split: {path.name}")
        if metadata.get("num_chains") != str(target["num_chains"]):
            raise ValueError(f"Cached chain count differs: {path.name}")
        shape = header["x_pred"].get("shape", [])  # (samples, atoms, xyz)
        if not isinstance(shape, list) or len(shape) != 3 or any(type(value) is not int or value <= 0 for value in shape) or shape[0] != samples_per_target or shape[-1] != 3:
            raise ValueError(f"Cached sample shape differs: {path.name}")
        entries[path.name] = {
            "target_id": str(target["target_id"]),
            "size": path.stat().st_size,
            "header_sha256": header_digest,
            "sample_shape": shape,
        }
    return {
        "model_id": model_id,
        "directory": directory.relative_to(root).as_posix(),
        "expected_targets": len(targets),
        "cached_targets": len(entries),
        "missing_targets": len(targets) - len(entries),
        "ignored_temporary_files": sorted(path.name for path in directory.glob("*.tmp")),
        "payload_hashes_verified": False,
        "files": entries,
    }


def checkpoint_identity(
    root: Path, model_id: str, *, trusted: bool = False
) -> dict[str, object]:
    """Read a trusted local resume pickle on CPU; an absent checkpoint means no saved update."""
    directory = _model_directory(root, model_id) / "v2"
    path = directory / "last.pt"
    if not path.exists():
        return {"model_id": model_id, "checkpoint_present": False, "update": 0}
    if not trusted:
        raise ValueError("Reading last.pt requires explicit trust in the local training checkpoint")
    identity = _file_identity(path)
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["config"]["model_id"] != model_id:
        raise ValueError("Resume checkpoint identifies another model")
    update = checkpoint["update"]
    if isinstance(update, bool) or not isinstance(update, int) or update < 0:
        raise ValueError("Resume checkpoint has an invalid update count")
    for name in ("head", "optimizer", "ema", "history", "best", "rng", "elapsed"):
        if name not in checkpoint:
            raise ValueError(f"Resume checkpoint is missing {name}")
    companions = {
        name: _file_identity(directory / name)
        for name in ("best-ema.safetensors", "wandb-id.txt")
    }
    if _file_identity(path) != identity:
        raise ValueError("Resume checkpoint changed during inspection; stop its writer first")
    return {
        "model_id": model_id,
        "checkpoint_present": True,
        "update": update,
        "elapsed_seconds": checkpoint["elapsed"],
        "config": checkpoint["config"],
        "provenance": checkpoint.get("provenance"),
        "file": identity,
        "companions": companions,
    }


def write_migration_receipt(
    root: Path,
    model_id: str,
    *,
    old_gpu: str,
    new_gpu: str,
    old_call_id: str,
    new_call_id: str,
    old_call_status: str,
    checkpoint: Mapping[str, object],
    validation_cache: Mapping[str, object],
    source_files: Mapping[str, object],
) -> Path:
    """Record a confirmed terminal writer and its replacement once; never dispatch workers."""
    _model_directory(root, model_id)
    if old_call_status.upper() not in TERMINAL_CALL_STATUSES:
        raise ValueError("The old training call must be confirmed terminal before migration")
    for call_id in (old_call_id, new_call_id):
        if not re.fullmatch(r"fc-[A-Za-z0-9]+", call_id):
            raise ValueError("Migration requires exact Modal function call IDs")
    if old_call_id == new_call_id:
        raise ValueError("Migration requires a different replacement call")
    if checkpoint.get("model_id") != model_id or validation_cache.get("model_id") != model_id:
        raise ValueError("Migration evidence identifies another model")
    if not old_gpu or not new_gpu or not source_files:
        raise ValueError("Migration requires GPU names and source identities")
    directory = root / "migrations"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{model_id}-{old_call_id}.json"
    receipt = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "campaign": root.name,
        "model_id": model_id,
        "old_gpu": old_gpu,
        "new_gpu": new_gpu,
        "old_call_id": old_call_id,
        "new_call_id": new_call_id,
        "old_call_status": old_call_status.upper(),
        "checkpoint": dict(checkpoint),
        "validation_cache": dict(validation_cache),
        "source_files": dict(source_files),
    }
    with path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return path
