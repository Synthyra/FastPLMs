"""Launch bounded Modal work using local credentials without uploading them."""

from __future__ import annotations

import argparse
import json
import time
import uuid

from pathlib import Path
from typing import Any

from .budget import BudgetLedger
from .config import (
    BENCHMARK_TIMEOUT_SECONDS,
    CAMPAIGN_TIMEOUT_SECONDS,
    CPU_STAGE_TIMEOUT_SECONDS,
    GPU_STARTUP_TIMEOUT_SECONDS,
    MAX_PARALLEL_WORKERS,
    MODEL_IDS,
    TRAIN_TIMEOUT_SECONDS,
    resource_rate,
)


ROOT = Path(__file__).resolve().parents[2]
CPU_STAGES = {"tests", "lint", "format", "prepare", "docs"}
GPU_CHOICES = ("L4", "L40S", "H100")
PARALLEL_HELP = (
    f"Dispatch two GPU workers under one ledger controller (maximum {MAX_PARALLEL_WORKERS})"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "tests",
            "lint",
            "format",
            "prepare",
            "docs",
            "benchmark",
            "cache",
            "train",
            "evaluate",
            "campaign",
            "package",
            "release",
        ),
    )
    parser.add_argument("--model", choices=MODEL_IDS, default=MODEL_IDS[0])
    parser.add_argument("--gpu", choices=GPU_CHOICES, default="L4")
    parser.add_argument("--options", default="{}", help="Non-secret JSON stage options")
    parser.add_argument(
        "--parallel",
        action="store_true",
        help=PARALLEL_HELP,
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Submit Modal calls with spawn and return without waiting for receipts",
    )
    return parser


def _timeout(stage: str, cpu: bool) -> int:
    if cpu:
        return CPU_STAGE_TIMEOUT_SECONDS
    if stage == "benchmark":
        return BENCHMARK_TIMEOUT_SECONDS
    if stage == "train":
        return TRAIN_TIMEOUT_SECONDS
    if stage == "campaign":
        return CAMPAIGN_TIMEOUT_SECONDS
    return 7_200


def _stage_name(stage: str) -> str:
    return "prepare" if stage == "benchmark" else stage


def _call_with_mode(worker: Any, arguments: tuple[Any, ...], *, detached: bool) -> Any:
    if detached:
        return worker.spawn(*arguments)
    return worker.remote(*arguments)


def _call_identifier(call: Any) -> str | None:
    value = getattr(call, "object_id", None)
    return str(value) if value else None


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _apply_formatted_files(result: dict[str, Any]) -> None:
    formatted = result.pop("formatted_files", {})
    generated = result.pop("generated_files", {})
    if formatted and generated:
        raise ValueError("Remote worker returned both formatted and generated files")
    files = formatted or generated
    for relative, content in files.items():
        path = (ROOT / relative).resolve()
        confidence_source = path.parent == ROOT / "tools/confidence" or (
            path.parent == ROOT / "tests/unit" and path.name.startswith("test_confidence_")
        )
        generated_doc = path.suffix == ".md" and (
            path.parent == ROOT / "model_cards" or path.parent == ROOT / "docs/generated"
        )
        allowed = generated_doc if generated else confidence_source and path.suffix == ".py"
        if not allowed:
            raise ValueError("Remote formatter returned a path outside the confidence workflow")
        path.write_text(content, encoding="utf-8")
    if files:
        result["updated_paths"] = list(files)


def _reserve_workers(
    ledger: BudgetLedger,
    *,
    app_id: str,
    stage: str,
    models: list[str],
    gpu: str,
    timeout: int,
    cpu: bool,
    count: int,
) -> list[dict[str, str]]:
    rate = resource_rate(None if cpu else gpu)
    reservation_cost = (timeout + (0 if cpu else GPU_STARTUP_TIMEOUT_SECONDS)) * rate
    reservations = []
    if len(models) != count:
        raise ValueError("The worker model list must match the requested worker count")
    for worker_index, model in enumerate(models):
        call_id = uuid.uuid4().hex
        reservation = ledger.reserve(
            _stage_name(stage),
            f"{stage}/{model}/{gpu}/{worker_index}",
            reservation_cost,
            app_id=app_id,
            call_id=call_id,
            worker_index=worker_index,
            timeout_seconds=timeout,
            gpu=None if cpu else gpu,
        )
        reservations.append({"reservation": reservation, "call_id": call_id, "model": model})
    return reservations


def main() -> None:
    args = _parser().parse_args()
    try:
        options = json.loads(args.options)
    except json.JSONDecodeError as error:
        raise SystemExit(f"--options must be JSON: {error}") from error
    if not isinstance(options, dict):
        raise SystemExit("--options must contain a JSON object")
    cpu = args.stage in CPU_STAGES
    if args.parallel and cpu:
        raise SystemExit("--parallel is only supported for GPU stages")
    worker_count = MAX_PARALLEL_WORKERS if args.parallel else 1
    models = list(MODEL_IDS) if args.parallel else [args.model]
    timeout = _timeout(args.stage, cpu)
    app_id = uuid.uuid4().hex
    local_root = ROOT / "artifacts/confidence"
    ledger = BudgetLedger(local_root / "budget.json")
    reservations = _reserve_workers(
        ledger,
        app_id=app_id,
        stage=args.stage,
        models=models,
        gpu=args.gpu,
        timeout=timeout,
        cpu=cpu,
        count=worker_count,
    )
    dispatch_manifest = {
        "app_id": app_id,
        "stage": args.stage,
        "models": models,
        "gpu": args.gpu,
        "timeout_seconds": timeout,
        "parallel_workers": worker_count,
        "reservations": reservations,
    }
    _write_json(local_root / f"dispatch-{app_id}.json", dispatch_manifest)

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".secrets.env", override=False)
    import modal

    from .modal_app import app, cpu_stage, gpu_workers

    started = time.monotonic()
    calls: list[tuple[dict[str, str], Any]] = []
    modal_app_id = app_id
    with modal.enable_output(), app.run(detach=args.detach):
        modal_app_id = str(getattr(app, "app_id", None) or app_id)
        dispatch_manifest["modal_app_id"] = modal_app_id
        _write_json(local_root / f"dispatch-{app_id}.json", dispatch_manifest)
        for reservation in reservations:
            call_options = dict(options)
            call_options.update(
                controller_app_id=modal_app_id,
                controller_call_id=reservation["call_id"],
            )
            if args.stage == "train":
                call_options.setdefault("maximum_seconds", 36_000)
            if cpu:
                worker = cpu_stage
                arguments = (args.stage, call_options)
            else:
                worker = gpu_workers[args.gpu, timeout]
                arguments = (args.stage, reservation["model"], call_options)
            call = _call_with_mode(worker, arguments, detached=args.detach or args.parallel)
            calls.append((reservation, call))
            ledger.attach_call(
                reservation["reservation"],
                _call_identifier(call),
                app_id=modal_app_id,
            )
        if args.parallel and not args.detach:
            results = [call.get() for _, call in calls]

    if args.detach:
        receipt = {
            "status": "dispatched",
            "app_id": app_id,
            "modal_app_id": modal_app_id,
            "calls": [],
        }
        for reservation, call in calls:
            receipt["calls"].append(
                {
                    "reservation": reservation["reservation"],
                    "model": reservation["model"],
                    "call_id": reservation["call_id"],
                    "remote_call_id": _call_identifier(call),
                }
            )
        _write_json(local_root / f"result-{app_id}.json", receipt)
        print(json.dumps(receipt, indent=2))
        return
    elif not args.parallel:
        results = [call for _, call in calls]

    elapsed = time.monotonic() - started
    output_results = []
    for (reservation, _), result in zip(calls, results, strict=True):
        if not isinstance(result, dict):
            result = {"status": "passed", "result": result}
        _apply_formatted_files(result)
        observed = float(result.get("elapsed_seconds", elapsed)) * resource_rate(
            None if cpu else args.gpu
        )
        ledger.complete(reservation["reservation"], observed)
        output_results.append(result)
    receipt = {
        "status": "passed",
        "app_id": app_id,
        "modal_app_id": modal_app_id,
        "results": output_results,
    }
    if any(result.get("status") == "failed" for result in output_results):
        receipt["status"] = "failed"
    _write_json(local_root / f"result-{app_id}.json", receipt)
    budget = {key: value for key, value in ledger.summary().items() if key != "reservations"}
    print(json.dumps({"result": receipt, "budget": budget}, indent=2))
    if receipt["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
