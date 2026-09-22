"""Compare the unchanged v2 training workload across GPU hardware."""

from __future__ import annotations

import gc
import math
import time

from pathlib import Path


GPU_RATES = {
    "A100-80GB": 0.000694,
    "RTX-PRO-6000": 0.000842,
    "L40S": 0.000542,
    "H100!": 0.001097,
    "H200": 0.001261,
    "B200": 0.001736,
    "B300": 0.001972,
}
CPU_CORES = 4
MEMORY_GIB = 64
PRICE_SOURCE = "https://modal.com/pricing"
PRICE_DATE = "2026-09-22"
MEASURED_TARGETS = 16
SAMPLER_SEED = 101


def hourly_rate(gpu: str) -> float:
    return 3600 * (GPU_RATES[gpu] + CPU_CORES * 0.0000131 + MEMORY_GIB * 0.00000222)


def benchmark_model(root: Path, model_id: str, gpu: str, output: Path, rounds: int = 1, long_probe: bool = True) -> dict[str, object]:
    import torch

    from . import host
    from .cache import load_folding_model
    from .online_training import ExponentialMovingAverage, OnlineTrainingConfig, TargetSampler, structure, target_step
    from .rollouts import fold, use_fast_folding_kernels
    from .training import HeadContext
    from .v2_campaign import configure_host, write_json

    configure_host(root, "benchmark")
    targets = host.split_targets("train")
    sampler = TargetSampler(targets, 0.5, SAMPLER_SEED)
    panel = [sampler.draw() for _ in range(MEASURED_TARGETS)]
    warmup = min(targets, key=lambda target: abs(int(target["num_tokens"]) - 256))
    guard = min(
        (target for target in targets if int(target["num_chains"]) > 1),
        key=lambda target: abs(int(target["num_tokens"]) - 1024),
    )
    long_guard = min(
        (target for target in host.split_targets("unused") if target["variant"] == "long" and int(target["num_chains"]) > 1),
        key=lambda target: abs(int(target["num_tokens"]) - 2048),
    )
    config = OnlineTrainingConfig(model_id=model_id, targets_per_update=MEASURED_TARGETS)
    result: dict[str, object] = {"status": "running", "model_id": model_id, "gpu": gpu, "samples": 4, "loops": 3, "diffusion_steps": 50, "parameter_dtype": "float32", "autocast_dtype": "bfloat16", "rows": []}
    write_json(output, result)
    torch.manual_seed(17)
    model = load_folding_model(model_id)
    use_fast_folding_kernels(model)
    context = HeadContext(model_id)
    context.head.train()
    optimizer = torch.optim.AdamW(context.head.parameters(), lr=1e-4, weight_decay=0.01)
    ema = ExponentialMovingAverage(context.head, 1 - 1 / 78)

    def measure(target: dict[str, object], index: int, training: bool) -> dict[str, object]:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        rollout = fold(model, structure(root / "pool", target), 4 if training else 5, 5000 + index, 3, 50)
        torch.cuda.synchronize()
        folding_seconds = time.monotonic() - started
        if training:
            losses = target_step(context, rollout, config)
            if not all(math.isfinite(value) for value in losses.values()):
                raise ValueError("Training losses are not finite")
        else:
            from .online_training import head_output

            with torch.no_grad():
                for sample in range(5):
                    head_output(context, rollout.head_inputs, rollout.x_pred, sample)
            losses = {}
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        memory = torch.cuda.max_memory_allocated() / 2**30
        del rollout
        return {"target_id": target["target_id"], "num_tokens": int(target["num_tokens"]), "num_chains": int(target["num_chains"]), "seconds": elapsed, "fold_seconds": folding_seconds, "peak_memory_gib": memory, "losses": losses}

    try:
        result["warmup"] = measure(warmup, -1, True)
        optimizer.zero_grad(set_to_none=True)
        result["rounds"] = []
        for round_index in range(rounds):
            round_rows = []
            for index, target in enumerate(panel):
                row = {"round": round_index, **measure(target, index, True)}
                round_rows.append(row)
                result["rows"].append(row)
                write_json(output, result)
            torch.cuda.synchronize()
            started = time.monotonic()
            torch.nn.utils.clip_grad_norm_(context.head.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(context.head)
            torch.cuda.synchronize()
            optimizer_seconds = time.monotonic() - started
            seconds = sum(row["seconds"] for row in round_rows) + optimizer_seconds
            result["rounds"].append({"round": round_index, "seconds_per_update": seconds, "optimizer_seconds": optimizer_seconds})
        result["seconds_per_update"] = seconds
        result["dollars_per_update"] = seconds * hourly_rate(gpu) / 3600
        result["training_guard"] = measure(guard, 100, True)
        optimizer.zero_grad(set_to_none=True)
        result["status"] = "passed"
        write_json(output, result)
        if long_probe:
            context.head.eval()
            try:
                result["long_evaluation_guard"] = measure(long_guard, 101, False)
            except torch.OutOfMemoryError:
                result["long_evaluation_guard"] = {"status": "out_of_memory", "num_tokens": int(long_guard["num_tokens"])}
                torch.cuda.empty_cache()
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
    finally:
        write_json(output, result)
        model = None
        context = None
        del optimizer, ema
        gc.collect()
        torch.cuda.empty_cache()
    return result
