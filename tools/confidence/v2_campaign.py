"""Durable inputs and stage execution for the Modal v2 confidence rerun."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time

from pathlib import Path


DATASET_REPO = "Synthyra/AtlasFold-Data"
DATASET_REVISION = "98b5212fd04cc34e3cdcb43c9bc6a66639ef4041"
ARTIFACT_REPO = "Synthyra/FastPLMs-artifacts"
HISTORICAL_SPLIT_SHA256 = (
    "da801db539deabdc36d6060e5ef8570ca6cff15374a04dca5280bdcc31166bdd"
)
MODEL_IDS = ("esmfold2_300", "esmfold2_600")
PLANNED_UPDATES = 780
TRAINING_SECONDS = 73_800


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def configure_host(root: Path, worker: str) -> None:
    from . import host

    host.DATA_ROOT = root
    host.POOL_DIR = root / "pool"
    host.SPLITS_DIR = root / "splits"
    host.PILOT_DIR = root / "pilot"
    host.RUNS_DIR = root / "runs"
    host.LEDGER_PATH = root / "ledgers" / f"{worker}.json"
    host.LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    host.MMSEQS = "mmseqs"
    # Modal also bounds each invocation independently with a hard timeout.
    host.MODEL_BUDGET_HOURS = 26.0


def validate_prepared(root: Path) -> dict[str, object]:
    from .target_splits import load_split

    receipt = json.loads((root / "prepared.json").read_text(encoding="utf-8"))
    if receipt["dataset_revision"] != DATASET_REVISION or receipt["status"] != "prepared":
        raise ValueError("Prepared data does not match the campaign dataset pin")
    for name, digest in receipt["pilot_files"].items():
        if file_hash(root / "pilot" / name) != digest:
            raise ValueError(f"Pilot input changed: {name}")
    targets = load_split(root / "splits")
    for name in {str(target["positions_file"]) for target in targets}:
        if not (root / "pool" / "positions" / name).is_file():
            raise FileNotFoundError(f"Prepared coordinates are missing: {name}")
    return receipt


def prepare_data(root: Path, pilot_root: Path, workers: int = 8) -> dict[str, object]:
    from huggingface_hub import snapshot_download

    from .target_pool import build_pool
    from .target_splits import build_splits, load_split

    receipt = root / "prepared.json"
    if receipt.exists():
        return validate_prepared(root)
    pilot = root / "pilot"
    pilot.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pilot_root / "data/records.json", pilot / "records.json")
    for model_id in MODEL_IDS:
        shutil.copyfile(pilot_root / model_id / "train/best.safetensors", pilot / f"{model_id}-head.safetensors")
    dataset_root = root.parent / "atlasfold" / DATASET_REVISION
    snapshot_download(
        DATASET_REPO,
        repo_type="dataset",
        revision=DATASET_REVISION,
        allow_patterns=["data/rcsb/train-*.parquet", "data/rcsb_multimer/train-*.parquet"],
        local_dir=dataset_root,
        max_workers=8,
    )
    source_files = sorted(dataset_root.glob("data/*/*.parquet"))
    if len(source_files) != 80:
        raise ValueError(f"Expected 80 pinned AtlasFold Parquets, found {len(source_files)}")
    inventory = {path.relative_to(dataset_root).as_posix(): {"sha256": file_hash(path), "size": path.stat().st_size} for path in source_files}
    write_json(root / "dataset.json", {"repo_id": DATASET_REPO, "revision": DATASET_REVISION, "files": inventory})
    pool = build_pool(dataset_root, root / "pool", workers)
    splits = build_splits(root / "pool", pilot / "records.json", root / "splits", "mmseqs", workers)
    load_split(root / "splits")
    result = {
        "status": "prepared",
        "dataset_revision": DATASET_REVISION,
        "pool": pool,
        "splits": splits,
        "historical_split_sha256": HISTORICAL_SPLIT_SHA256,
        "matches_historical_split_bytes": splits["targets_sha256"] == HISTORICAL_SPLIT_SHA256,
        "test_set_status": "spent",
        "mmseqs_version": subprocess.check_output(["mmseqs", "version"], text=True).strip(),
        "pilot_files": {path.name: file_hash(path) for path in sorted(pilot.iterdir())},
    }
    write_json(receipt, result)
    return result


def publish_files(root: Path, names: list[str], group: str) -> str:
    """Archive only explicitly selected campaign files in the public dataset."""
    from huggingface_hub import CommitOperationAdd, HfApi
    from huggingface_hub.errors import HfHubHTTPError

    api = HfApi()
    operations = []
    for name in names:
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Invalid campaign artifact path: {name}")
        operations.append(CommitOperationAdd(path_in_repo=f"confidence-v2/{root.name}/{name}", path_or_fileobj=path))
    for attempt in range(3):
        parent = api.dataset_info(ARTIFACT_REPO).sha
        try:
            commit = api.create_commit(
                repo_id=ARTIFACT_REPO,
                repo_type="dataset",
                operations=operations,
                parent_commit=parent,
                commit_message=f"Archive {root.name} {group}",
            )
            break
        except HfHubHTTPError as error:
            if error.response is None or error.response.status_code != 412 or attempt == 2:
                raise
            # Other campaign workers publish disjoint files to the same dataset branch.
            time.sleep(attempt + 1)
    revision = str(commit.oid)
    write_json(root / "uploads" / f"{group}.json", {"repo_id": ARTIFACT_REPO, "revision": revision, "files": names})
    return revision


def train_model(root: Path, model_id: str) -> dict[str, object]:
    from . import host
    from .cache import load_folding_model
    from .online_training import OnlineTrainingConfig, ema_decay_for, train_online
    from .rollouts import use_fast_folding_kernels

    if model_id not in MODEL_IDS:
        raise ValueError("The v2 campaign supports only the 300M and 600M confidence heads")
    configure_host(root, model_id)
    config = OnlineTrainingConfig(
        model_id=model_id,
        planned_updates=PLANNED_UPDATES,
        warmup_updates=78,
        ema_decay=ema_decay_for(PLANNED_UPDATES),
        maximum_seconds=TRAINING_SECONDS,
        validation_interval_seconds=5400,
        checkpoint_interval_seconds=1800,
    )
    with host.gpu_budget(f"model-{model_id}", "train-v2", host.MODEL_BUDGET_HOURS) as remaining:
        started = time.monotonic()
        model = load_folding_model(model_id)
        use_fast_folding_kernels(model)
        return train_online(
            model, host.POOL_DIR, host.split_targets("train"), host.split_targets("validation"),
            host.RUNS_DIR / model_id / "v2", config, host.log,
            min(TRAINING_SECONDS + 1800, remaining - 4 * 3600) - (time.monotonic() - started),
        )


def validate_evaluation_recovery(root: Path, model_id: str) -> None:
    """Permit recovery only when completed training has no dispatched evaluation."""
    from .experiment_artifacts import verify_evaluation

    if model_id not in MODEL_IDS:
        raise ValueError("Select a trained 300M or 600M head")
    report = json.loads((root / "runs" / model_id / "v2/report.json").read_text(encoding="utf-8"))
    if report.get("status") != "complete" or report.get("model_id") != model_id or report.get("updates") != PLANNED_UPDATES:
        raise ValueError("Evaluation requires the completed training run")
    status_root = root / "status"
    if any(path.exists() for path in (
        root / "evaluation" / root.name / model_id,
        status_root / f"dispatch-evaluate-{model_id}.json",
        status_root / f"evaluate-{model_id}.json",
    )):
        raise FileExistsError("Evaluation already reserved; inspect its existing worker and records")
    training_status = status_root / f"train-{model_id}.json"
    if training_status.exists():
        status = json.loads(training_status.read_text(encoding="utf-8"))
        if status.get("evaluation_call_id") or status.get("status") == "running":
            raise FileExistsError("Training may have an active evaluation dispatch; inspect its call ID")
    verify_evaluation(root / "public/evaluation/esmfold2", require_checkpoints=False)


def evaluate_model(root: Path, model_id: str, *, resume: bool = False) -> None:
    from . import host

    configure_host(root, model_id)
    if model_id == "esmfold2":
        host.stage_reference("test", None, root.name)
    else:
        host.stage_evaluate(model_id, "v2", "test", None, root.name, resume=resume)


def export_evaluation(root: Path, model_id: str) -> list[str]:
    from .experiment_artifacts import export_evaluation as export
    from .experiment_artifacts import verify_evaluation

    destination = root / "public" / "evaluation" / model_id
    if destination.exists():
        completion = verify_evaluation(destination, require_checkpoints=False)
        if completion["model_id"] != model_id or completion["evaluation_id"] != root.name:
            raise ValueError("Existing public evaluation belongs to a different campaign or model")
    else:
        export(root / "evaluation" / root.name / model_id, destination)
    return [path.relative_to(root).as_posix() for path in sorted(destination.rglob("*")) if path.is_file()]


def evaluations_archived(root: Path) -> bool:
    return all(
        (root / "evaluation" / root.name / model_id / "completion.json").is_file()
        and (root / "uploads" / f"evaluate-{model_id}.json").is_file()
        for model_id in (*MODEL_IDS, "esmfold2")
    )
