"""Package accepted confidence heads and verify the native inference artifact."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys

import torch
import wandb

from pathlib import Path
from collections.abc import Mapping
from typing import Any

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

from fastplms.registry import get_model_spec
from .config import DONOR_REPO, DONOR_REVISION, DONOR_WEIGHT_SHA256, MODEL_IDS
from .training import HeadContext, _wandb_run


def _hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2) + "\n")


def merge_head(
    base: Mapping[str, torch.Tensor], head: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Replace only the confidence subtree, preserving all folding tensors."""
    # State tensors have parameter-specific shapes; replacement preserves each supplied shape.
    if not head or any(key.startswith("confidence_head.") for key in head):
        raise ValueError("Expected a nonempty native head state without a module prefix")
    if any(not torch.isfinite(value).all() for value in head.values()):
        raise ValueError("Confidence head contains nonfinite values")
    merged = {key: value for key, value in base.items() if not key.startswith("confidence_head.")}
    merged.update({f"confidence_head.{key}": value for key, value in head.items()})
    return merged


def verify_folding_state(
    base: Mapping[str, torch.Tensor], packaged: Mapping[str, torch.Tensor]
) -> int:
    keys = {key for key in base if not key.startswith("confidence_head.")}
    if keys != {key for key in packaged if not key.startswith("confidence_head.")}:
        raise ValueError("Packaging changed folding tensor keys")
    for key in keys:
        left, right = base[key], packaged[key]  # parameter-specific shapes, checked below
        if left.dtype != right.dtype or left.shape != right.shape:
            raise ValueError(f"Packaging changed folding tensor schema: {key}")
        if not torch.equal(
            left.contiguous().reshape(-1).view(torch.uint8),
            right.contiguous().reshape(-1).view(torch.uint8),
        ):
            raise ValueError(f"Packaging changed folding tensor bytes: {key}")
    return len(keys)


def _validate_and_record(directory: Path, report: dict[str, Any], run: Any) -> dict[str, Any]:
    artifact = Path(report["artifact"])
    process = subprocess.run(
        [
            sys.executable,
            "-I",
            str(Path(__file__).with_name("artifact_validation.py")),
            str(artifact),
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    (directory / "package/reload.log").write_text(process.stdout + process.stderr)
    if process.returncode:
        raise RuntimeError("Isolated artifact validation failed: " + process.stderr[-2500:])
    report["inference"] = _read(directory / "package/reload-validation.json")
    report["status"] = "passed"
    _write(directory / "package/result.json", report)
    run.summary.update(
        {
            "status": "passed",
            "head_sha256": report["head_sha256"],
            "weight_sha256": report["weight_sha256"],
        }
    )
    record = wandb.Artifact(f"{report['model_id']}-confidence-package", type="validation")
    record.add_file(str(directory / "package/result.json"))
    record.add_file(str(directory / "package/validation-complex.cif"))
    run.log_artifact(record).wait()
    return report


def package_head(root: Path, model_id: str, validate_only: bool = False) -> dict[str, Any]:
    if model_id not in MODEL_IDS:
        raise ValueError("Only the two approved ESMFold2 models may be packaged")
    directory = root / model_id
    evaluation = _read(directory / "evaluate/result.json")
    if evaluation.get("model_id") != model_id or not evaluation.get("acceptance", {}).get(
        "accepted"
    ):
        raise ValueError("Packaging requires an accepted model-specific final evaluation")
    training = _read(directory / "train/result.json")
    if training.get("status") != "complete":
        raise ValueError("Training is not complete")
    head_path = directory / "train/best.safetensors"
    head_hash = _hash(head_path)
    checkpoint = torch.load(directory / "train/last.pt", map_location="cpu", weights_only=True)
    if checkpoint["best_sha256"] != head_hash or checkpoint["model_id"] != model_id:
        raise ValueError("Selected confidence head differs from the training checkpoint")
    spec = get_model_spec(model_id)
    config = {"model_id": model_id, "head_sha256": head_hash, "base_revision": spec.fast.revision}
    run = _wandb_run(root, model_id, "package", config)
    failed = False
    try:
        if validate_only:
            report = _read(directory / "package/preparation.json")
            artifact = Path(report["artifact"])
            if report["model_id"] != model_id or report["head_sha256"] != head_hash:
                raise ValueError("Prepared artifact identifies a different confidence head")
            if report["weight_sha256"] != _hash(artifact / "model.safetensors") or report[
                "config_sha256"
            ] != _hash(artifact / "config.json"):
                raise ValueError("Prepared artifact changed before validation")
            return _validate_and_record(directory, report, run)
        snapshot = Path(snapshot_download(spec.fast.repo_id, revision=spec.fast.revision))
        expected_base = spec.fast.file_map["model.safetensors"].digest
        if (
            _hash(snapshot / "model.safetensors") != expected_base
            or training["base_weight_sha256"] != expected_base
        ):
            raise ValueError("Base checkpoint differs from the frozen training source")
        base = load_file(str(snapshot / "model.safetensors"))
        head = load_file(str(head_path))
        context = HeadContext(model_id, device="cpu")
        context.head.load_state_dict(head, strict=True)
        settings = checkpoint["settings"]
        del context, checkpoint
        artifact = directory / "package/artifact"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        if artifact.exists():
            raise ValueError("Package artifact already exists; inspect its report before replacing")
        shutil.copytree(
            snapshot, artifact, ignore=shutil.ignore_patterns(".cache", ".git", "model.safetensors")
        )
        save_file(
            merge_head(base, head), str(artifact / "model.safetensors"), metadata={"format": "pt"}
        )
        packaged = load_file(str(artifact / "model.safetensors"))
        preserved = verify_folding_state(base, packaged)
        del base, head, packaged
        payload = _read(artifact / "config.json")
        payload["confidence_head"]["enabled"] = True
        payload["fastplms_checkpoint_repo_id"] = spec.fast.repo_id
        payload["fastplms_checkpoint_hash"] = _hash(artifact / "model.safetensors")
        payload.pop("fastplms_checkpoint_revision", None)
        payload.pop("fastplms_weights_revision", None)
        _write(artifact / "config.json", payload)
        report = {
            "status": "prepared",
            "model_id": model_id,
            "artifact": str(artifact),
            "base_repo": spec.fast.repo_id,
            "base_revision": spec.fast.revision,
            "base_weight_sha256": expected_base,
            "head_sha256": head_hash,
            "weight_sha256": _hash(artifact / "model.safetensors"),
            "config_sha256": _hash(artifact / "config.json"),
            "preserved_folding_tensors": preserved,
            "donor_repo": DONOR_REPO,
            "donor_revision": DONOR_REVISION,
            "donor_weight_sha256": DONOR_WEIGHT_SHA256,
            "training_run": training["wandb_url"],
            "evaluation_run": evaluation["wandb_url"],
            "wandb_url": run.url,
            "evaluation_sha256": _hash(directory / "evaluate/result.json"),
            "training_settings": settings,
            "complete_artifact_compliance": False,
        }
        _write(directory / "package/preparation.json", report)
        return _validate_and_record(directory, report, run)
    except BaseException as error:
        failed = True
        run.summary.update({"status": "failed", "error": str(error)})
        raise
    finally:
        run.finish(exit_code=1 if failed else 0)
