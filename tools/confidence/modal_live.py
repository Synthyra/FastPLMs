"""Mirror ongoing v2 EMA checkpoints to Hugging Face without restarting training."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time

import modal

from pathlib import Path

from .modal_v2 import MOUNTS, ROOT, campaign_root, credentials, image as runtime_image, volume
from .v2_campaign import MODEL_IDS, write_json


app = modal.App("fastplms-confidence-live-publication")
image = runtime_image.add_local_dir(str(ROOT / "tests/release"), "/workspace/tests/release")
POLL_SECONDS = 120
MAXIMUM_SECONDS = 23 * 3600 + 1800
MAX_CONSECUTIVE_ERRORS = 3


@app.function(image=image, cpu=4, memory=32768, timeout=1800, startup_timeout=900, volumes=MOUNTS, max_containers=1)
def verify() -> dict[str, object]:
    paths = [
        "tests/unit/test_confidence_live_checkpoints.py",
        "tests/unit/test_confidence_live_binding.py",
        "tests/unit/test_confidence_checkpoint.py",
        "tests/unit/test_confidence_migration.py",
        "tests/unit/test_confidence_call_status.py",
        "tests/unit/test_confidence_online_training.py",
        "tests/unit/test_confidence_adaptation.py",
        "tests/release/test_model_card_licenses.py",
    ]
    completed = subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", "not gpu", *paths], capture_output=True, text=True)
    result = {"returncode": completed.returncode, "output": completed.stdout + completed.stderr}
    if completed.returncode:
        raise RuntimeError(result["output"])
    return result


@app.function(image=image, cpu=1, memory=8192, timeout=86400, startup_timeout=900, volumes=MOUNTS, secrets=[credentials], max_containers=1)
def publish(campaign: str, calls: dict[str, str]) -> dict[str, object]:
    from huggingface_hub import HfApi

    from .call_status import training_call_status
    from .live_binding import bind_live_head
    from .live_checkpoints import publish_live_checkpoint, stage_live_checkpoint

    if set(calls) != set(MODEL_IDS):
        raise ValueError("Live publication requires both exact training call IDs")
    root = campaign_root(campaign)
    status_path = root / "status/live-publication.json"
    api = HfApi()
    signatures: dict[str, tuple[int, int]] = {}
    terminal: dict[str, str] = {}
    publications: dict[str, dict[str, object]] = {}
    errors = 0
    deadline = time.monotonic() + MAXIMUM_SECONDS
    try:
        while time.monotonic() < deadline:
            volume.reload()
            try:
                for model_id in MODEL_IDS:
                    path = root / "runs" / model_id / "v2/last.pt"
                    if path.is_file():
                        stat = path.stat()
                        signature = (stat.st_size, stat.st_mtime_ns)
                        if signatures.get(model_id) != signature:
                            with tempfile.TemporaryDirectory(prefix="publish-confidence-") as temporary:
                                checkpoint = stage_live_checkpoint(root, model_id, Path(temporary))
                                if checkpoint is not None:
                                    dataset_revision = publish_live_checkpoint(checkpoint, api) or api.dataset_info(checkpoint.latest["repo_id"]).sha
                                    model_revision = bind_live_head(checkpoint, api)
                                    publications[model_id] = {"update": checkpoint.update, "dataset_revision": dataset_revision, "model_revision": model_revision, "latest_path": checkpoint.latest_path, "head_sha256": checkpoint.latest["head_sha256"]}
                                    print(json.dumps({"model_id": model_id, **publications[model_id]}), flush=True)
                            signatures[model_id] = signature
                    if model_id not in terminal:
                        training_status = training_call_status(calls[model_id])
                        if training_status is not None:
                            terminal[model_id] = training_status
                errors = 0
            except Exception as error:
                errors += 1
                write_json(status_path, {"status": "retrying", "error_type": type(error).__name__, "error": str(error), "consecutive_errors": errors, "publications": publications, "training": terminal})
                volume.commit()
                if errors >= MAX_CONSECUTIVE_ERRORS:
                    raise
                time.sleep(POLL_SECONDS)
                continue
            status = {"status": "running", "publications": publications, "training": terminal}
            if len(terminal) == len(MODEL_IDS):
                # Refresh once after terminal results so the final committed checkpoint is visible.
                volume.reload()
                final_signatures = {}
                for model_id in MODEL_IDS:
                    path = root / "runs" / model_id / "v2/last.pt"
                    if path.is_file():
                        stat = path.stat()
                        final_signatures[model_id] = (stat.st_size, stat.st_mtime_ns)
                if all(signatures.get(model_id) == signature for model_id, signature in final_signatures.items()):
                    status["status"] = "complete" if all(value == "complete" for value in terminal.values()) else "training_failed"
                    write_json(status_path, status)
                    volume.commit()
                    return status
            write_json(status_path, status)
            volume.commit()
            time.sleep(POLL_SECONDS)
        raise TimeoutError("Live checkpoint publication exceeded its 23.5-hour limit")
    except BaseException as error:
        write_json(status_path, {"status": "failed", "error_type": type(error).__name__, "error": str(error), "publications": publications, "training": terminal})
        volume.commit()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    from .experiment_artifacts import validate_evaluation_id

    campaign = validate_evaluation_id(args.campaign)
    local_root = ROOT / "artifacts/confidence-v2" / campaign
    receipt_path = local_root / "live-publication-dispatch.json"
    if not args.verify_only and receipt_path.exists():
        raise FileExistsError("Live publication is already dispatched; inspect its receipt before retrying")
    with modal.enable_output(), app.run(detach=True):
        result = verify.remote()
        write_json(local_root / "live-publication-verification.json", result)
        print(result["output"])
        if not args.verify_only:
            calls = json.loads((local_root / "resume-dispatch.json").read_text(encoding="utf-8"))["calls"]
            call = publish.spawn(campaign, calls)
            receipt = {"app_id": app.app_id, "call_id": call.object_id, "campaign": campaign, "training_calls": calls}
            write_json(receipt_path, receipt)
            print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
