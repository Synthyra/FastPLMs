"""Prepare accepted confidence-head artifacts for a later Hub publication."""

from __future__ import annotations

import hashlib
import json
import shutil

from dataclasses import replace
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _inventory(root: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return files


def prepare_release(root: Path, model_id: str) -> dict[str, Any]:
    """Prepare, validate, and inventory one accepted model artifact.

    This function never calls ``create_commit`` or uploads files. The returned
    inventory is suitable for a separately reviewed add-only weight update.
    """
    from huggingface_hub import HfApi

    from fastplms.registry import ConfidenceAdaptation, get_model_spec
    from tools.confidence.config import MODEL_IDS
    from tools.artifacts.generate_docs import render_model_card
    from tools.artifacts.publish import compile_model_files

    if model_id not in MODEL_IDS:
        raise ValueError("release preparation supports only the approved ESMFold2 model IDs")
    model_root = root / model_id
    package_report = _read(model_root / "package" / "result.json")
    evaluation = _read(model_root / "evaluate" / "result.json")
    training = _read(model_root / "train" / "result.json")
    for name, report in (
        ("package", package_report),
        ("evaluation", evaluation),
    ):
        if report.get("model_id") != model_id:
            raise ValueError(f"{name} report identifies a different model")
    if package_report.get("status") != "passed":
        raise ValueError("release requires a passed package validation")
    if not evaluation.get("acceptance", {}).get("accepted"):
        raise ValueError("release requires an accepted final evaluation")
    if training.get("status") != "complete":
        raise ValueError("release requires completed training")
    if (
        training["base_weight_sha256"] != package_report["base_weight_sha256"]
        or training["wandb_url"] != package_report["training_run"]
    ):
        raise ValueError("Training identity differs from the validated package")
    artifact = Path(package_report["artifact"])
    if not artifact.is_dir() or not (artifact / "model.safetensors").is_file():
        raise ValueError("package report does not identify a complete artifact")
    source_root = Path(__file__).resolve().parents[2]
    release_root = model_root / "release"
    if release_root.exists():
        raise ValueError("release directory already exists; inspect it before replacing")
    release_artifact = release_root / "package" / "artifact"
    spec = get_model_spec(model_id)
    if package_report.get("base_weight_sha256") != spec.confidence_training_base.file_map["model.safetensors"].digest:
        raise ValueError("package base weight does not match the manifest-pinned source")
    if (
        package_report.get("base_repo") != spec.confidence_training_base.repo_id
        or package_report.get("base_revision") != spec.confidence_training_base.revision
    ):
        raise ValueError("package source identity does not match the manifest")
    if _sha256(artifact / "model.safetensors") != package_report.get("weight_sha256") or _sha256(
        artifact / "config.json"
    ) != package_report.get("config_sha256"):
        raise ValueError("package files changed after validation")
    package_config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    if package_config.get("fastplms_model_id") != model_id or not package_config.get(
        "confidence_head", {}
    ).get("enabled"):
        raise ValueError("package config does not identify an enabled approved confidence head")
    evidence_path = f"docs/evidence/confidence/{model_id}.json"
    source_evidence = source_root / evidence_path
    if not source_evidence.is_file():
        raise ValueError(f"measured evidence is missing: {source_evidence}")
    # Launcher receipts add controller fields; compare the actual evaluation fields.
    recorded_evidence = _read(source_evidence)
    if any(recorded_evidence.get(key) != value for key, value in evaluation.items()):
        raise ValueError("Checked-in evaluation differs from the accepted remote report")
    shutil.copytree(artifact, release_artifact)
    adaptation = ConfidenceAdaptation(
        head_sha256=package_report["head_sha256"],
        base_weight_sha256=package_report["base_weight_sha256"],
        donor_repo=package_report["donor_repo"],
        donor_revision=package_report["donor_revision"],
        donor_weight_sha256=package_report["donor_weight_sha256"],
        training_url=training["wandb_url"],
        evaluation_url=evaluation["wandb_url"],
        evidence_path=evidence_path,
    )
    adapted_spec = replace(
        spec,
        confidence_adaptation=adaptation,
        notes=(
            "Experimental Fast base with frozen backbone and folding tensors and a "
            "Synthyra-adapted native confidence head. Held-out short monomer/dimer "
            "evaluation and focused artifact inference checks passed. FP32 folding "
            "parameters use BF16 autocast and SDPA; FP8 is unsupported."
        ),
    )
    evidence = {
        "model_id": model_id,
        "training": training,
        "evaluation": evaluation,
        "package": package_report,
        "settings": package_report.get(
            "training_settings", training.get("settings", training.get("training_settings", {}))
        ),
    }
    (release_root / "confidence-training.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    compiled = compile_model_files(adapted_spec, source_root)
    compiled["README.md"] = render_model_card(adapted_spec, evidence_root=source_root).encode(
        "utf-8"
    )
    compiled[evidence_path] = source_evidence.read_bytes()
    compiled["confidence-training.json"] = (release_root / "confidence-training.json").read_bytes()
    for relative, payload in compiled.items():
        destination = release_artifact / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    from tools.artifacts.build import _decode_runtime_bundle

    runtime_hash, _ = _decode_runtime_bundle(release_artifact / "fastplms_bundle.py")
    config_path = release_artifact / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["fastplms_runtime_bundle_sha256"] = runtime_hash
    config["fastplms_model_id"] = model_id
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    for stale_name in (
        "artifact-manifest.json",
        "source-record.json",
        "runtime-attestation.json",
        "preparation.json",
        "preparation-record.json",
    ):
        stale_path = release_artifact / stale_name
        if stale_path.exists():
            stale_path.unlink()
    validation_report = {
        "status": "prepared",
        "model_id": model_id,
        "artifact": str(release_artifact),
        "head_sha256": package_report["head_sha256"],
        "weight_sha256": _sha256(release_artifact / "model.safetensors"),
    }
    (release_root / "package").mkdir(parents=True, exist_ok=True)
    from .packaging import _validate_and_record
    from .training import _wandb_run

    run = _wandb_run(
        root, model_id, "release", {"model_id": model_id, "stage": "release_validation"}
    )
    failed = False
    try:
        _validate_and_record(release_root, validation_report, run)
    except BaseException:
        failed = True
        raise
    finally:
        run.finish(exit_code=1 if failed else 0)
    report = {
        "status": "prepared",
        "model_id": model_id,
        "artifact": str(release_artifact),
        "complete_artifact_compliance": False,
        "weight_update_required": True,
        "evidence": evidence_path,
        "validation": validation_report,
    }
    api = HfApi()
    report["remote_parent_commit"] = api.model_info(spec.fast.repo_id, revision="main").sha
    report["inventory"] = _inventory(release_artifact)
    (release_root / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


__all__ = ["prepare_release"]
