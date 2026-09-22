"""Launch the v2 confidence reproduction on persistent Modal storage."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import modal

from pathlib import Path

from .modal_app import base_image, credentials, volume as pilot_volume, with_source_files
from .v2_campaign import MODEL_IDS, PLANNED_UPDATES, prepare_data, write_json


ROOT = Path(__file__).resolve().parents[2]
VOLUME_NAME = "fastplms-confidence-v2"
REMOTE_ROOT = Path("/experiment")
app = modal.App("fastplms-confidence-v2")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = with_source_files(
    base_image
    .uv_pip_install("datasets>=4,<5", "cuequivariance==0.10.0", "cuequivariance-torch==0.10.0", "cuequivariance-ops-torch-cu13==0.10.0")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)
MOUNTS = {"/vol": pilot_volume, str(REMOTE_ROOT): volume}


def campaign_root(campaign: str) -> Path:
    from .experiment_artifacts import validate_evaluation_id

    return REMOTE_ROOT / validate_evaluation_id(campaign)


@app.function(image=image, cpu=8, memory=131072, timeout=21600, volumes=MOUNTS, secrets=[credentials], max_containers=1)
def prepare(campaign: str) -> dict[str, object]:
    root = campaign_root(campaign)
    try:
        return prepare_data(root, Path("/vol/confidence"))
    finally:
        volume.commit()
        pilot_volume.commit()


@app.function(image=image, cpu=4, memory=32768, timeout=1800, volumes=MOUNTS, max_containers=1)
def verify(campaign: str) -> dict[str, object]:
    paths = [
        "tests/unit/test_confidence_online_training.py",
        "tests/unit/test_confidence_rollouts.py",
        "tests/unit/test_confidence_test_evaluation.py",
        "tests/unit/test_confidence_evaluation_progress.py",
        "tests/unit/test_confidence_target_pool.py",
        "tests/unit/test_confidence_target_splits.py",
        "tests/unit/test_confidence_acceptance.py",
        "tests/unit/test_confidence_experiment_artifacts.py",
        "tests/unit/test_confidence_v2_analysis.py",
        "tests/unit/test_confidence_v2_campaign.py",
    ]
    completed = subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", "not gpu", *paths], capture_output=True, text=True)
    from .experiment_artifacts import source_identity

    result = {"status": "passed" if completed.returncode == 0 else "failed", "returncode": completed.returncode, "output": completed.stdout + completed.stderr, "source_files": source_identity(ROOT)}
    write_json(campaign_root(campaign) / "verification.json", result)
    volume.commit()
    if completed.returncode:
        raise RuntimeError(result["output"])
    return result


@app.function(image=image, cpu=4, memory=65536, gpu="H200", timeout=79200, startup_timeout=900, volumes=MOUNTS, secrets=[credentials], max_containers=2)
def train(campaign: str, model_id: str) -> dict[str, object]:
    from .v2_campaign import train_model

    root = campaign_root(campaign)
    status_path = root / "status" / f"train-{model_id}.json"
    write_json(status_path, {"status": "running", "model_id": model_id})
    try:
        result = train_model(root, model_id)
        volume.commit()
        archive.remote(campaign, f"train-{model_id}")
        if result["updates"] != PLANNED_UPDATES:
            raise RuntimeError(f"Training stopped after {result['updates']}/{PLANNED_UPDATES} updates; checkpoint saved, reproduction incomplete")
        evaluation = evaluate.spawn(campaign, model_id)
        write_json(status_path, {"status": "complete", "report": result, "evaluation_call_id": evaluation.object_id})
        return result
    except BaseException as error:
        write_json(status_path, {"status": "failed", "error": str(error), "model_id": model_id})
        raise
    finally:
        volume.commit()
        pilot_volume.commit()


def run_evaluation(campaign: str, model_id: str) -> dict[str, object]:
    from .v2_campaign import evaluate_model

    root = campaign_root(campaign)
    status_path = root / "status" / f"evaluate-{model_id}.json"
    write_json(status_path, {"status": "running", "model_id": model_id})
    try:
        evaluate_model(root, model_id)
        volume.commit()
        archive.remote(campaign, f"evaluate-{model_id}")
        result = {"status": "complete", "model_id": model_id}
        write_json(status_path, result)
        return result
    except BaseException as error:
        write_json(status_path, {"status": "failed", "error": str(error), "model_id": model_id})
        raise
    finally:
        volume.commit()
        pilot_volume.commit()


@app.function(image=image, cpu=4, memory=65536, gpu="H200", timeout=14400, startup_timeout=900, volumes=MOUNTS, secrets=[credentials], max_containers=2)
def evaluate(campaign: str, model_id: str) -> dict[str, object]:
    return run_evaluation(campaign, model_id)


@app.function(image=image, cpu=4, memory=65536, gpu="H200", timeout=18000, startup_timeout=900, volumes=MOUNTS, secrets=[credentials], max_containers=1)
def reference(campaign: str) -> dict[str, object]:
    return run_evaluation(campaign, "esmfold2")


@app.function(image=image, cpu=2, memory=16384, timeout=3600, volumes=MOUNTS, secrets=[credentials], max_containers=1)
def archive(campaign: str, group: str) -> dict[str, object]:
    from . import host
    from .experiment_artifacts import source_identity
    from .v2_campaign import configure_host, evaluations_archived, export_evaluation, file_hash, publish_files

    volume.reload()
    root = campaign_root(campaign)
    if group == "inputs":
        write_json(root / "source-files.json", source_identity(ROOT))
        names = ["prepared.json", "dataset.json", "source-files.json", "verification.json", "splits/targets.parquet", "splits/split-report.json", "pilot/records.json"]
    elif group.startswith("train-") and group[6:] in MODEL_IDS:
        model_id = group[6:]
        relative = f"runs/{model_id}/v2"
        names = [f"{relative}/{name}" for name in ("final-ema.safetensors", "best-ema.safetensors", "report.json", "wandb-id.txt")]
        write_json(root / relative / "file-identities.json", {name: file_hash(root / name) for name in names})
        names.append(f"{relative}/file-identities.json")
    elif group.startswith("evaluate-") and group[9:] in (*MODEL_IDS, "esmfold2"):
        names = export_evaluation(root, group[9:])
    else:
        raise ValueError(f"Unknown archive group: {group}")
    try:
        revision = publish_files(root, names, group)
        if not (root / "complete.json").exists() and evaluations_archived(root):
            configure_host(root, "analysis")
            acceptance_names = []
            for model_id in MODEL_IDS:
                name = f"evaluation/{campaign}/acceptance-{model_id}.json"
                if not (root / name).exists():
                    host.stage_acceptance(model_id, campaign)
                acceptance_names.append(name)
            acceptance_revision = publish_files(root, acceptance_names, "acceptance")
            write_json(root / "complete.json", {"status": "complete", "acceptance_revision": acceptance_revision, "model_publication": "pending_review", "test_set_status": "spent"})
        return {"status": "archived", "group": group, "revision": revision}
    finally:
        volume.commit()


@app.function(image=image, cpu=2, memory=8192, timeout=21600, volumes=MOUNTS, secrets=[credentials], max_containers=1)
def start(campaign: str) -> dict[str, object]:
    from .v2_campaign import validate_prepared

    verify.remote(campaign)
    root = campaign_root(campaign)
    deadline = time.monotonic() + 5 * 3600
    while not (root / "prepared.json").exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Data preparation did not finish within five hours")
        time.sleep(30)
        volume.reload()
    validate_prepared(root)
    archive.remote(campaign, "inputs")
    reservation = root / "started.json"
    with reservation.open("x", encoding="utf-8") as stream:
        json.dump({"status": "dispatching", "campaign": campaign}, stream)
    volume.commit()
    calls = {}
    for model_id in MODEL_IDS:
        calls[model_id] = train.spawn(campaign, model_id).object_id
        write_json(root / "dispatch.json", {"status": "dispatching", "calls": calls})
        volume.commit()
    calls["esmfold2"] = reference.spawn(campaign).object_id
    result = {"status": "dispatched", "calls": calls, "campaign": campaign}
    write_json(root / "dispatch.json", result)
    volume.commit()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "verify", "start"))
    parser.add_argument("--campaign", required=True)
    args = parser.parse_args()
    from .experiment_artifacts import validate_evaluation_id

    validate_evaluation_id(args.campaign)
    worker = {"prepare": prepare, "verify": verify, "start": start}[args.stage]
    with modal.enable_output(), app.run(detach=True):
        call = worker.spawn(args.campaign)
        receipt = {"app_id": app.app_id, "call_id": call.object_id, "stage": args.stage, "campaign": args.campaign, "volume": VOLUME_NAME}
        write_json(ROOT / "artifacts/confidence-v2" / args.campaign / f"{args.stage}-dispatch.json", receipt)
        print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
