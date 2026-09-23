"""Launch the v2 confidence reproduction on persistent Modal storage."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import modal

from pathlib import Path

from .modal_app import (
    base_image,
    credentials,
    volume as pilot_volume,
    with_source_files,
)
from .v2_campaign import MODEL_IDS, PLANNED_UPDATES, prepare_data, write_json


ROOT = Path(__file__).resolve().parents[2]
VOLUME_NAME = "fastplms-confidence-v2"
TRAINING_GPU = "RTX-PRO-6000"
REMOTE_ROOT = Path("/experiment")
app = modal.App("fastplms-confidence-v2")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = with_source_files(
    base_image.uv_pip_install(
        "datasets>=4,<5",
        "cuequivariance==0.10.0",
        "cuequivariance-torch==0.10.0",
        "cuequivariance-ops-torch-cu13==0.10.0",
    ).env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)
training_image = with_source_files(
    base_image.uv_pip_install(
        "torch==2.13.0+cu132", index_url="https://download.pytorch.org/whl/cu132"
    )
    .uv_pip_install(
        "datasets>=4,<5",
        "cuequivariance==0.10.0",
        "cuequivariance-torch==0.10.0",
        "cuequivariance-ops-torch-cu13==0.10.0",
    )
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


@app.function(image=training_image, cpu=4, memory=65536, gpu=TRAINING_GPU, timeout=79200, startup_timeout=900, volumes=MOUNTS, secrets=[credentials], max_containers=2)
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


def run_evaluation(
    campaign: str, model_id: str, *, resume: bool = False
) -> dict[str, object]:
    from .v2_campaign import evaluate_model

    root = campaign_root(campaign)
    status_path = root / "status" / f"evaluate-{model_id}.json"
    if resume:
        recovery = json.loads(
            (root / "evaluation" / campaign / model_id / "resume.json").read_text(
                encoding="utf-8"
            )
        )
        remaining = recovery["metadata"]["deadline_unix"] - time.time()
        if remaining <= 0:
            raise TimeoutError("Recovery deadline expired; refusing more GPU work")

    write_json(status_path, {"status": "running", "model_id": model_id})
    try:
        if resume:
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import sys; from pathlib import Path; from tools.confidence.v2_campaign import evaluate_model; evaluate_model(Path(sys.argv[1]), sys.argv[2], resume=True)",
                    str(root),
                    model_id,
                ],
                check=True,
                timeout=remaining,
            )
        else:
            evaluate_model(root, model_id)
        volume.commit()
        archive.remote(campaign, f"evaluate-{model_id}")
        result = {"status": "complete", "model_id": model_id}
        write_json(status_path, result)
        return result
    except BaseException as error:
        write_json(
            status_path, {"status": "failed", "error": str(error), "model_id": model_id}
        )
        raise
    finally:
        volume.commit()
        pilot_volume.commit()


@app.function(
    image=training_image,
    cpu=4,
    memory=65536,
    gpu=TRAINING_GPU,
    timeout=14400,
    startup_timeout=900,
    volumes=MOUNTS,
    secrets=[credentials],
    max_containers=2,
)
def evaluate(campaign: str, model_id: str, resume: bool = False) -> dict[str, object]:
    return run_evaluation(campaign, model_id, resume=resume)


@app.function(image=training_image, cpu=4, memory=65536, gpu=TRAINING_GPU, timeout=10800, startup_timeout=900, volumes=MOUNTS, secrets=[credentials], max_containers=1)
def continue_evaluation(campaign: str, model_id: str) -> dict[str, object]:
    return run_evaluation(campaign, model_id, resume=True)


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


@app.function(image=image, cpu=2, memory=8192, timeout=4200, volumes=MOUNTS, secrets=[credentials], max_containers=1)
def start_evaluation(campaign: str, model_id: str) -> dict[str, object]:
    """Recover a completed training run whose evaluation was never dispatched."""
    from .v2_campaign import validate_evaluation_recovery

    volume.reload()
    root = campaign_root(campaign)
    validate_evaluation_recovery(root, model_id)
    checked = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/unit/test_confidence_v2_campaign.py", "tests/unit/test_confidence_experiment_artifacts.py"],
        capture_output=True, text=True, timeout=120,
    )
    write_json(root / "status" / f"recovery-checks-{model_id}.json", {"returncode": checked.returncode, "output": checked.stdout + checked.stderr})
    volume.commit()
    if checked.returncode:
        raise RuntimeError(checked.stdout + checked.stderr)
    archive.remote(campaign, f"train-{model_id}")
    volume.reload()
    validate_evaluation_recovery(root, model_id)
    receipt_path = root / "status" / f"dispatch-evaluate-{model_id}.json"
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump({"status": "dispatching", "model_id": model_id}, stream)
    volume.commit()
    call = evaluate.spawn(campaign, model_id)
    receipt = {"status": "dispatched", "model_id": model_id, "call_id": call.object_id, "gpu": TRAINING_GPU, "timeout_seconds": 14400}
    write_json(receipt_path, receipt)
    volume.commit()
    return receipt


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    timeout=1200,
    volumes=MOUNTS,
    secrets=[credentials],
    max_containers=1,
)
def resume_evaluation(
    campaign: str, model_id: str, previous_call_id: str, prior_elapsed_seconds: int
) -> dict[str, object]:
    """Resume only a terminal evaluation, within its original four-hour allocation."""
    from .call_status import training_call_status
    from .experiment_artifacts import (
        EvaluationArtifacts,
        verify_evaluation,
        write_new_json,
    )
    from .test_evaluation import validated_partial_targets
    from .v2_campaign import configure_host

    if model_id not in MODEL_IDS or not 0 < prior_elapsed_seconds < 14400:
        raise ValueError(
            "Recovery needs a trained model and prior elapsed seconds below four hours"
        )
    volume.reload()
    root = campaign_root(campaign)
    original_dispatch = json.loads(
        (root / "status" / f"dispatch-evaluate-{model_id}.json").read_text(
            encoding="utf-8"
        )
    )
    if original_dispatch.get("call_id") != previous_call_id:
        raise ValueError("Previous call does not match the reserved evaluation")
    terminal_status = training_call_status(previous_call_id)
    if terminal_status is None:
        raise RuntimeError(
            "Previous evaluation is still active; refusing duplicate GPU work"
        )
    checked = subprocess.run(
        [sys.executable, "-m", "pytest", "-q",
         "tests/unit/test_confidence_evaluation_progress.py",
         "tests/unit/test_confidence_experiment_artifacts.py",
         "tests/unit/test_confidence_v2_campaign.py"],
        capture_output=True, text=True, timeout=180,
    )
    write_json(root / "status" / f"resume-checks-{model_id}.json", {"returncode": checked.returncode, "output": checked.stdout + checked.stderr})
    volume.commit()
    if checked.returncode:
        raise RuntimeError(checked.stdout + checked.stderr)
    directory = root / "evaluation" / campaign / model_id
    if (directory / "completion.json").exists():
        verify_evaluation(directory)
        return archive.remote(campaign, f"evaluate-{model_id}")
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    if request["model_id"] != model_id or request["evaluation_id"] != campaign:
        raise ValueError(
            "Recovery directory does not match the requested model and campaign"
        )
    targets = json.loads((directory / "targets.json").read_text(encoding="utf-8"))
    completed = validated_partial_targets(
        directory / "partial-records", targets, set(request["checkpoint_inputs"])
    )
    remaining = min(10800, 14400 - prior_elapsed_seconds)
    metadata = {
        "previous_call_id": previous_call_id,
        "previous_call_status": terminal_status,
        "prior_elapsed_seconds": prior_elapsed_seconds,
        "remaining_seconds": remaining,
        "deadline_unix": time.time() + remaining,
        "gpu": TRAINING_GPU,
        "retained_targets": len(completed),
        "requested_targets": len(targets),
    }
    # The exclusive receipt prevents two controllers from dispatching the same recovery.
    receipt_path = root / "status" / f"resume-evaluate-{model_id}.json"
    write_new_json(receipt_path, {"status": "reserved", **metadata})
    EvaluationArtifacts(directory).resume(metadata)
    configure_host(root, model_id)
    ledger_path = root / "ledgers" / f"{model_id}.json"
    entries = (
        json.loads(ledger_path.read_text(encoding="utf-8"))
        if ledger_path.exists()
        else []
    )
    charged = sum(
        entry["seconds"]
        for entry in entries
        if str(entry["stage"]).startswith("evaluate-")
    )
    if prior_elapsed_seconds > charged:
        entries.append(
            {
                "category": f"model-{model_id}",
                "stage": "evaluate-preemption-recovery",
                "seconds": prior_elapsed_seconds - charged,
                "started": time.time() - prior_elapsed_seconds,
            }
        )
        write_json(ledger_path, entries)
    volume.commit()
    call = continue_evaluation.spawn(campaign, model_id)
    receipt = {"status": "dispatched", "call_id": call.object_id, **metadata}
    write_json(receipt_path, receipt)
    volume.commit()
    return receipt


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
    parser.add_argument("stage", choices=("prepare", "verify", "start", "evaluate"))
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--model-id", choices=MODEL_IDS)
    parser.add_argument("--resume-evaluation", action="store_true")
    parser.add_argument("--previous-call-id")
    parser.add_argument("--prior-elapsed-seconds", type=int)
    args = parser.parse_args()
    from .experiment_artifacts import validate_evaluation_id

    validate_evaluation_id(args.campaign)
    if (args.stage == "evaluate") != (args.model_id is not None):
        parser.error("--model-id is required only for the evaluate stage")
    if args.resume_evaluation and (
        args.stage != "evaluate"
        or not args.previous_call_id
        or not args.prior_elapsed_seconds
    ):
        parser.error(
            "--resume-evaluation requires evaluate, --previous-call-id, and --prior-elapsed-seconds"
        )
    if not args.resume_evaluation and (
        args.previous_call_id is not None or args.prior_elapsed_seconds is not None
    ):
        parser.error("Recovery arguments require --resume-evaluation")
    worker = {
        "prepare": prepare,
        "verify": verify,
        "start": start,
        "evaluate": start_evaluation,
    }[args.stage]
    if args.resume_evaluation:
        worker = resume_evaluation
    with modal.enable_output(), app.run(detach=True):
        arguments = (
            (args.campaign, args.model_id)
            if args.stage == "evaluate"
            else (args.campaign,)
        )
        if args.resume_evaluation:
            arguments = (*arguments, args.previous_call_id, args.prior_elapsed_seconds)
        call = worker.spawn(*arguments)
        receipt = {
            "app_id": app.app_id,
            "call_id": call.object_id,
            "stage": args.stage,
            "campaign": args.campaign,
            "volume": VOLUME_NAME,
        }
        stage = f"evaluate-{args.model_id}" if args.stage == "evaluate" else args.stage
        if args.resume_evaluation:
            stage = f"resume-{stage}"
        write_json(
            ROOT / "artifacts/confidence-v2" / args.campaign / f"{stage}-dispatch.json",
            receipt,
        )
        print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
