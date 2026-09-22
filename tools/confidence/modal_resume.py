"""Resume explicitly stopped v2 workers on a benchmarked GPU."""

from __future__ import annotations

import argparse
import json

import modal

from .gpu_benchmark import GPU_RATES
from .modal_gpu_benchmark import environment
from .modal_v2 import MOUNTS, ROOT, app, campaign_root, credentials, train, volume
from .v2_campaign import MODEL_IDS, write_json


def resume_training(campaign: str, model_id: str) -> dict[str, object]:
    return train.get_raw_f()(campaign, model_id)


resume_workers = {
    gpu: app.function(name="resume_" + gpu.replace("!", "").replace("-", "_"), image=environment, gpu=gpu, cpu=4, memory=65536, timeout=79200, startup_timeout=900, volumes=MOUNTS, secrets=[credentials], max_containers=2)(resume_training)
    for gpu in GPU_RATES
}


@app.function(image=environment, cpu=4, memory=32768, timeout=900, volumes=MOUNTS, max_containers=1)
def inspect_resume(campaign: str, old_calls: dict[str, str]) -> dict[str, object]:
    from .experiment_artifacts import source_identity
    from .migration import checkpoint_identity, verify_validation_cache

    volume.reload()
    root = campaign_root(campaign)
    results = {}
    for model_id in MODEL_IDS:
        try:
            modal.FunctionCall.from_id(old_calls[model_id]).get(timeout=0)
        except TimeoutError as error:
            raise RuntimeError(f"Old worker is still active: {model_id}") from error
        except modal.exception.RemoteError:
            pass  # Terminal cancellation or remote failure; neither still writes the run.
        results[model_id] = {
            "checkpoint": checkpoint_identity(root, model_id, trusted=True),
            "validation_cache": verify_validation_cache(root, model_id),
            "source_files": source_identity(ROOT),
        }
    return results


@app.function(image=environment, cpu=1, memory=2048, timeout=300, volumes=MOUNTS, max_containers=1)
def record_resume(campaign: str, gpu: str, old_calls: dict[str, str], new_calls: dict[str, str], evidence: dict[str, object]) -> None:
    from .migration import write_migration_receipt

    volume.reload()
    root = campaign_root(campaign)
    for model_id in MODEL_IDS:
        item = evidence[model_id]
        write_migration_receipt(root, model_id, old_gpu="H200", new_gpu=gpu, old_call_id=old_calls[model_id], new_call_id=new_calls[model_id], old_call_status="TERMINATED", checkpoint=item["checkpoint"], validation_cache=item["validation_cache"], source_files=item["source_files"])
    write_json(root / "resume-dispatch.json", {"gpu": gpu, "old_calls": old_calls, "calls": new_calls})
    volume.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="v2-reproduction-20260922")
    parser.add_argument("--gpu", required=True, choices=tuple(GPU_RATES))
    args = parser.parse_args()
    local_root = ROOT / "artifacts/confidence-v2" / args.campaign
    old_calls = json.loads((local_root / "dispatch.json").read_text(encoding="utf-8"))["calls"]
    receipt_path = local_root / "resume-dispatch.json"
    if receipt_path.exists():
        raise FileExistsError("A resume dispatch already exists; inspect it before another migration")
    with modal.enable_output(), app.run(detach=True):
        evidence = inspect_resume.remote(args.campaign, old_calls)
        calls = {}
        for model_id in MODEL_IDS:
            calls[model_id] = resume_workers[args.gpu].spawn(args.campaign, model_id).object_id
            write_json(receipt_path, {"app_id": app.app_id, "gpu": args.gpu, "calls": calls, "status": "dispatching"})
        record_resume.remote(args.campaign, args.gpu, old_calls, calls, evidence)
        receipt = {"app_id": app.app_id, "gpu": args.gpu, "calls": calls, "status": "dispatched", "cache_targets": {model_id: item["validation_cache"]["cached_targets"] for model_id, item in evidence.items()}}
        write_json(receipt_path, receipt)
        print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
