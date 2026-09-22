"""Publish current EMA heads from trusted training checkpoints without loading models."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .experiment_artifacts import file_identity, validate_evaluation_id
from tools.execution.source import require_regular_source


if TYPE_CHECKING:
    from huggingface_hub import HfApi
    from torch import Tensor


DATASET_REPO = "Synthyra/FastPLMs-artifacts"
MODEL_IDS = frozenset({"esmfold2_300", "esmfold2_600"})
SCHEMA_VERSION = 1


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"Invalid {name} SHA256")
    return value


def ema_head_state(checkpoint: Mapping[str, Any]) -> dict[str, Tensor]:
    """Overlay trainable EMA parameters onto the complete saved head, retaining buffers."""
    import torch

    head, ema = checkpoint.get("head"), checkpoint.get("ema")
    if not isinstance(head, Mapping) or not head or not isinstance(ema, Mapping) or not ema:
        raise ValueError("Checkpoint requires nonempty head and EMA state dictionaries")
    if not set(ema).issubset(head):
        raise ValueError("EMA keys must belong to the complete saved head")
    state: dict[str, Tensor] = {}
    for name, original in head.items():
        # State shapes depend on the named parameter or buffer and remain unchanged.
        if not isinstance(name, str) or not name or not torch.is_tensor(original):
            raise ValueError("Head entries must be named tensors")
        value = ema.get(name, original)  # same checkpoint-defined shape (...)
        if not torch.is_tensor(value) or value.shape != original.shape or value.dtype != original.dtype:
            raise ValueError(f"EMA shape or dtype differs from the saved head: {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite head tensor: {name}")
        state[name] = value.detach().cpu().contiguous().clone()  # (...), independent storage
    return state  # named tensors retain their checkpoint-defined shapes


@dataclass(frozen=True)
class LiveCheckpoint:
    directory: Path
    latest_path: str
    latest: dict[str, Any]

    @property
    def update(self) -> int:
        return self.latest["update"]


def _copy_checkpoint(source: Path, destination: Path) -> None:
    """Keep one open inode while the trainer atomically replaces last.pt."""
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        before = os.fstat(incoming.fileno())
        shutil.copyfileobj(incoming, outgoing)
        after = os.fstat(incoming.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Training checkpoint changed in place while being copied")


def stage_live_checkpoint(
    campaign_root: Path, model_id: str, output_root: Path
) -> LiveCheckpoint | None:
    """Snapshot a trusted campaign's last.pt and stage its latest trained EMA head.

    Only use this with the caller-owned training volume. The resumable checkpoint
    contains Python/NumPy RNG state, so loading its immutable copy requires pickle.
    This function must never receive a checkpoint from an untrusted download.
    """
    import torch

    from safetensors.torch import save_file

    if model_id not in MODEL_IDS:
        raise ValueError(f"Unsupported live confidence model: {model_id}")
    campaign = validate_evaluation_id(campaign_root.name)
    relative = Path("runs") / model_id / "v2"
    try:
        source = require_regular_source(campaign_root, relative / "last.pt")
    except FileNotFoundError:
        return None
    run_file = require_regular_source(campaign_root, relative / "wandb-id.txt")
    run_id = validate_evaluation_id(run_file.read_text(encoding="utf-8").strip())
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="live-head-", dir=output_root) as temporary:
        snapshot = Path(temporary) / "last.pt"
        _copy_checkpoint(source, snapshot)
        identity = file_identity(snapshot)
        checkpoint = torch.load(snapshot, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise ValueError("Training checkpoint must be a mapping")
        update = checkpoint.get("update")
        if type(update) is not int or update < 0:
            raise ValueError("Training checkpoint update must be a nonnegative integer")
        if update == 0:
            return None
        config, provenance = checkpoint.get("config"), checkpoint.get("provenance")
        if not isinstance(config, Mapping) or config.get("model_id") != model_id:
            raise ValueError("Training configuration model does not match its directory")
        if not isinstance(provenance, Mapping) or provenance.get("model_id") != model_id:
            raise ValueError("Training provenance model does not match its directory")
        source_files = provenance.get("source_files")
        if not isinstance(source_files, Mapping) or not source_files:
            raise ValueError("Training checkpoint omits recorded source identities")
        base_hash = _digest(provenance.get("base_weight_sha256"), "base weight")
        elapsed = checkpoint.get("elapsed")
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Training checkpoint elapsed seconds are invalid")
        prefix = f"confidence/v2/{campaign}/{model_id}"
        update_path = f"{prefix}/{run_id}/updates/{update:08d}-{identity.sha256}"
        state = ema_head_state(checkpoint)  # complete named head tensors, shapes unchanged
        staged = Path(temporary) / "head.safetensors"
        save_file(state, str(staged))
        head_identity = file_identity(staged)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "campaign": campaign,
            "model_id": model_id,
            "run_id": run_id,
            "wandb_url": f"https://wandb.ai/lhallee/fastplms-confidence/runs/{run_id}",
            "update": update,
            "elapsed_seconds": elapsed,
            "head_kind": "current_ema",
            "head_state_format": "native_confidence_head",
            "checkpoint_kind": "ema",
            "evaluation_status": "pending",
            "checkpoint_sha256": identity.sha256,
            "checkpoint_size": identity.size,
            "head_sha256": head_identity.sha256,
            "head_size": head_identity.size,
            "base_repo": provenance.get("base_repo"),
            "base_revision": provenance.get("base_revision"),
            "base_weight_sha256": base_hash,
            "config": dict(config),
            "config_sha256": _sha256(_json_bytes(config)),
            "provenance": dict(provenance),
            "source_sha256": _sha256(_json_bytes(source_files)),
        }
        metadata_bytes = _json_bytes(metadata)
        latest = {
            key: value for key, value in metadata.items() if key not in {"config", "provenance"}
        }
        latest.update({
            "repo_id": DATASET_REPO,
            "repo_type": "dataset",
            "head_path": f"{update_path}/head.safetensors",
            "provenance_path": f"{update_path}/provenance.json",
            "provenance_sha256": _sha256(metadata_bytes),
        })
        # Each staging call owns its directory, including repeated observations of one update.
        directory = Path(tempfile.mkdtemp(prefix=f"{model_id}-{update}-", dir=output_root))
        shutil.copyfile(staged, directory / "head.safetensors")
        (directory / "provenance.json").write_bytes(metadata_bytes)
        (directory / "latest.json").write_bytes(_json_bytes(latest))
        return LiveCheckpoint(directory, f"{prefix}/latest.json", latest)


def publish_live_checkpoint(checkpoint: LiveCheckpoint, api: HfApi) -> str | None:
    """Add immutable EMA files and advance latest in one parent-protected dataset commit."""
    from huggingface_hub import CommitOperationAdd

    latest = checkpoint.latest
    if latest["repo_id"] != DATASET_REPO or latest["repo_type"] != "dataset":
        raise ValueError("Live heads belong to the established artifact dataset")
    campaign = validate_evaluation_id(latest["campaign"])
    run_id = validate_evaluation_id(latest["run_id"])
    if latest["model_id"] not in MODEL_IDS or type(checkpoint.update) is not int or checkpoint.update <= 0:
        raise ValueError("Live publication requires a supported model and a trained update")
    digest = _digest(latest["checkpoint_sha256"], "checkpoint")
    prefix = f"confidence/v2/{campaign}/{latest['model_id']}"
    immutable = f"{prefix}/{run_id}/updates/{checkpoint.update:08d}-{digest}"
    if (
        checkpoint.latest_path != f"{prefix}/latest.json"
        or latest["head_path"] != f"{immutable}/head.safetensors"
        or latest["provenance_path"] != f"{immutable}/provenance.json"
    ):
        raise ValueError("Live checkpoint paths leave the campaign model namespace")
    files = {
        latest["head_path"]: ("head.safetensors", latest["head_sha256"]),
        latest["provenance_path"]: ("provenance.json", latest["provenance_sha256"]),
    }
    for name, expected in files.values():
        if file_identity(checkpoint.directory / name).sha256 != expected:
            raise ValueError(f"Staged live checkpoint changed: {name}")
    if (checkpoint.directory / "latest.json").read_bytes() != _json_bytes(latest):
        raise ValueError("Staged latest pointer changed")
    parent = api.dataset_info(DATASET_REPO, revision="main").sha
    if not isinstance(parent, str) or len(parent) != 40 or any(c not in "0123456789abcdef" for c in parent):
        raise ValueError("Artifact dataset has no valid remote parent commit")
    if api.file_exists(DATASET_REPO, checkpoint.latest_path, repo_type="dataset", revision=parent):
        previous_file = api.hf_hub_download(
            DATASET_REPO, checkpoint.latest_path, repo_type="dataset", revision=parent
        )
        previous = json.loads(Path(previous_file).read_text(encoding="utf-8"))
        if previous.get("run_id") != latest["run_id"] or previous.get("model_id") != latest["model_id"]:
            raise ValueError("Latest pointer belongs to a different training run")
        if type(previous.get("update")) is not int:
            raise ValueError("Existing latest pointer has an invalid update")
        if previous["update"] > checkpoint.update:
            return None
        if previous["update"] == checkpoint.update:
            if previous.get("head_sha256") != latest["head_sha256"]:
                raise ValueError("One training update has conflicting EMA heads")
            return None
    operations = []
    for remote_path, (name, expected) in files.items():
        if api.file_exists(DATASET_REPO, remote_path, repo_type="dataset", revision=parent):
            existing = api.hf_hub_download(DATASET_REPO, remote_path, repo_type="dataset", revision=parent)
            # Hub cache files may be SDK-managed links to immutable blobs.
            with Path(existing).open("rb") as stream:
                observed = hashlib.file_digest(stream, "sha256").hexdigest()
            if observed != expected:
                raise ValueError(f"Immutable live checkpoint path already differs: {remote_path}")
        else:
            operations.append(CommitOperationAdd(remote_path, checkpoint.directory / name))
    operations.append(CommitOperationAdd(checkpoint.latest_path, checkpoint.directory / "latest.json"))
    commit = api.create_commit(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        revision="main",
        parent_commit=parent,
        operations=operations,
        commit_message=f"Publish {latest['model_id']} current EMA update {checkpoint.update}",
    )
    return commit.oid
