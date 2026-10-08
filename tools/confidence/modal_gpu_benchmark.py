"""Run a bounded seven-GPU v2 cost-throughput comparison on Modal."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import modal

from pathlib import Path

from .gpu_benchmark import GPU_RATES, PRICE_DATE, PRICE_SOURCE, benchmark_model, hourly_rate
from .modal_app import base_image, credentials, volume, hf_volume, MOUNTS, with_source_files
from .volume_artifacts import V2_ROOT, restore_v2


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ID = "gpu-cost-20260922"
app = modal.App("fastplms-confidence-gpu-benchmark")
environment = with_source_files(
    base_image.uv_pip_install("torch==2.13.0+cu132", index_url="https://download.pytorch.org/whl/cu132")
    .uv_pip_install("datasets>=5.0", "cuequivariance==0.10.0", "cuequivariance-torch==0.10.0", "cuequivariance-ops-torch-cu13==0.10.0")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)


def run_benchmark(campaign: str, gpu: str, benchmark_id: str = BENCHMARK_ID, rounds: int = 1, long_probe: bool = True) -> dict[str, object]:
    import torch

    from .experiment_artifacts import environment_identity, source_identity, validate_evaluation_id
    from .v2_campaign import MODEL_IDS
    from tools.stored_files import write_stored_json

    started = time.monotonic()
    root = V2_ROOT / validate_evaluation_id(campaign)
    directory = root / "benchmarks" / validate_evaluation_id(benchmark_id) / gpu.replace("!", "")
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()
    benchmark_record = {"gpu_requested": gpu, "gpu_actual": torch.cuda.get_device_name(), "compute_capability": torch.cuda.get_device_capability(), "gpu_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30, "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip(), "torch_cuda": torch.version.cuda, "environment": environment_identity(), "source_files": source_identity(ROOT), "price_source": PRICE_SOURCE, "price_date": PRICE_DATE, "allocated_hourly_rate": hourly_rate(gpu), "models": {}}
    try:
        for model_id in MODEL_IDS:
            benchmark_record["models"][model_id] = benchmark_model(root, model_id, gpu, directory / f"{model_id}.json", rounds, long_probe)
        return benchmark_record
    finally:
        benchmark_record["elapsed_seconds"] = time.monotonic() - started
        write_stored_json(directory / "result.json", benchmark_record, sort_keys=False)
        volume.commit()
        hf_volume.commit()


workers = {
    gpu: app.function(name="benchmark_" + gpu.replace("!", "").replace("-", "_"), image=environment, gpu=gpu, cpu=4, memory=65536, timeout=1500, startup_timeout=900, volumes=MOUNTS, secrets=[credentials], max_containers=1)(run_benchmark)
    for gpu in GPU_RATES
}


def main() -> None:
    from tools.stored_files import write_stored_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="v2-reproduction-20260922")
    parser.add_argument("--gpus", nargs="+", choices=tuple(GPU_RATES), default=list(GPU_RATES))
    parser.add_argument("--benchmark-id", default=BENCHMARK_ID)
    parser.add_argument("--rounds", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--skip-long-probe", action="store_true")
    args = parser.parse_args()
    with modal.enable_output(), app.run(detach=True):
        prepare_retained_inputs.remote(args.campaign)
        calls = {gpu: workers[gpu].spawn(args.campaign, gpu, args.benchmark_id, args.rounds, not args.skip_long_probe).object_id for gpu in args.gpus}
        receipt = {"app_id": app.app_id, "campaign": args.campaign, "benchmark_id": args.benchmark_id, "calls": calls}
        filename = "benchmark-dispatch.json" if args.benchmark_id == BENCHMARK_ID else f"{args.benchmark_id}-dispatch.json"
        write_stored_json(ROOT / "artifacts/confidence-v2" / args.campaign / filename, receipt, sort_keys=False)
        print(json.dumps(receipt, indent=2))


@app.function(image=environment, cpu=(2, 2), memory=(8192, 8192), timeout=7200, volumes=MOUNTS,
              secrets=[credentials], max_containers=1)
def prepare_retained_inputs(campaign: str) -> None:
    restore_v2(campaign, "benchmark")
    volume.commit()


if __name__ == "__main__":
    main()
