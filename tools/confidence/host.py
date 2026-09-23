"""Confidence-head v2 stages for a single GPU workstation.

Run from the repository root with `PYTHONPATH=src:. python -m tools.confidence.host <stage>`.
Data and runs live under `~/data/confidence-v2`. GPU stages record their wall time in
`ledger.json` and refuse to start once a budget is spent: 2 hours of smoke experiments in total
and 24 hours of training and evaluation per model.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import uuid
import warnings

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from .experiment_artifacts import (
    EvaluationArtifacts,
    environment_identity,
    file_identity,
    new_evaluation_id,
    source_identity,
    validate_evaluation_id,
    verify_evaluation_group,
    write_new_json,
)


DATA_ROOT = Path.home() / "data" / "confidence-v2"
ATLASFOLD_HUB = Path.home() / "data" / "atlasfold" / "hub"
POOL_DIR = DATA_ROOT / "pool"
SPLITS_DIR = DATA_ROOT / "splits"
PILOT_DIR = DATA_ROOT / "pilot"
RUNS_DIR = DATA_ROOT / "runs"
LEDGER_PATH = DATA_ROOT / "ledger.json"
MODAL_CLI = Path.home() / "venvs" / "modal" / "bin" / "modal"
MODAL_CREDENTIALS = Path.home() / ".config" / "synthyra" / "modal.env"
MMSEQS = str(Path.home() / "bin" / "mmseqs")
SMOKE_BUDGET_HOURS = 2.0
MODEL_BUDGET_HOURS = 24.0
REFERENCE_BUDGET_HOURS = 5.0
EVALUATION_RESERVE_HOURS = (
    2.0  # kept free of training so a model's test evaluation fits its budget
)
RATE_TARGETS = 64  # training draws timed to set planned updates
RATE_SAMPLER_SEED = 101
PILOT_FILES = {
    "confidence/data/records.json": "records.json",
    "confidence/esmfold2_300/train/best.safetensors": "esmfold2_300-head.safetensors",
    "confidence/esmfold2_600/train/best.safetensors": "esmfold2_600-head.safetensors",
    "confidence/esmfold2_300/evaluate-report.json": "esmfold2_300-evaluate-report.json",
    "confidence/esmfold2_600/evaluate-report.json": "esmfold2_600-evaluate-report.json",
}


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def ledger_hours(category: str) -> float:
    if not LEDGER_PATH.exists():
        return 0.0
    entries = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    return sum(entry["seconds"] for entry in entries if entry["category"] == category) / 3600


@contextmanager
def gpu_budget(category: str, stage: str, budget_hours: float) -> Iterator[float]:
    """Refuse to start past the budget; yield the remaining seconds and record the time used."""
    spent = ledger_hours(category)
    if spent >= budget_hours:
        raise RuntimeError(f"{category} has used {spent:.2f} of {budget_hours} GPU hours")
    started = time.time()
    try:
        yield (budget_hours - spent) * 3600
    finally:
        entries = (
            json.loads(LEDGER_PATH.read_text(encoding="utf-8")) if LEDGER_PATH.exists() else []
        )
        entries.append(
            {
                "category": category,
                "stage": stage,
                "started": started,
                "seconds": time.time() - started,
            }
        )
        write_json(LEDGER_PATH, entries)


def fetch_pilot_artifacts() -> None:
    """Copy the pilot's split records, selected heads, and reports from its Modal volume."""
    from dotenv import dotenv_values

    environment = {
        **os.environ,
        **{name: value for name, value in dotenv_values(MODAL_CREDENTIALS).items() if value},
    }
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    for remote, local in PILOT_FILES.items():
        subprocess.run(
            [
                str(MODAL_CLI),
                "volume",
                "get",
                "--force",
                "fastplms-confidence-pilot",
                remote,
                str(PILOT_DIR / local),
            ],
            check=True,
            env=environment,
        )
        log(f"copied {remote}")


def stage_pool(workers: int) -> None:
    from .target_pool import build_pool

    report = build_pool(ATLASFOLD_HUB, POOL_DIR, workers)
    log(json.dumps(report))


def stage_splits(threads: int) -> None:
    from .target_splits import build_splits

    report = build_splits(POOL_DIR, PILOT_DIR / "records.json", SPLITS_DIR, MMSEQS, threads)
    log(json.dumps(report))


def split_targets(split: str) -> list[dict[str, object]]:
    """Targets of one split in a fixed, stratum-mixed order."""
    import hashlib

    from .target_splits import load_split

    chosen = [target for target in load_split(SPLITS_DIR) if target["split"] == split]
    return sorted(
        chosen, key=lambda target: hashlib.sha256(str(target["target_id"]).encode()).hexdigest()
    )


def stage_smoke(check: str, model_id: str) -> None:
    from .online_smoke import closest_targets, kernels, overfit, parity, throughput, training_rate
    from .online_training import OnlineTrainingConfig, TargetSampler
    from .target_pool import TOKEN_BUDGET

    train = split_targets("train")
    short = [target for target in train if int(target["num_tokens"]) <= 384]
    with gpu_budget("smoke", f"smoke-{check}-{model_id}", SMOKE_BUDGET_HOURS):
        if check == "throughput":
            # Size long-target evaluations on the unused split without touching the test set.
            long_targets = [
                target for target in split_targets("unused") if target["variant"] == "long"
            ]
            chosen = closest_targets(train, (128, 384, 768, 1024), 768) + closest_targets(
                long_targets, (1536, 2048), 0
            )
            result: object = throughput(model_id, POOL_DIR, chosen, 4, TOKEN_BUDGET, log)
        elif check == "kernels":
            result = kernels(model_id, POOL_DIR, closest_targets(train, (512, 1024), 768), 4, log)
        elif check == "training-rate":
            sampler = TargetSampler(
                train,
                OnlineTrainingConfig(model_id=model_id).monomer_fraction,
                seed=RATE_SAMPLER_SEED,
            )
            result = training_rate(
                model_id, POOL_DIR, [sampler.draw() for _ in range(RATE_TARGETS + 1)], log
            )
        elif check == "parity":
            chosen = [
                next(target for target in short if int(target["num_chains"]) == 1),
                next(
                    target
                    for target in short
                    if int(target["num_chains"]) == 2
                    and target["sequences"][0] == target["sequences"][1]
                ),
                next(
                    target
                    for target in short
                    if int(target["num_chains"]) == 2
                    and target["sequences"][0] != target["sequences"][1]
                ),
            ]
            result = parity(model_id, POOL_DIR, chosen, log)
        else:
            # Small targets keep 150 full passes over the fixed rollouts within the smoke budget.
            small = [target for target in short if int(target["num_tokens"]) <= 256]
            monomers = [target for target in small if int(target["num_chains"]) == 1][:6]
            complexes = [target for target in small if int(target["num_chains"]) > 1][:6]
            result = overfit(model_id, POOL_DIR, monomers + complexes, updates=150, log=log)
    write_json(DATA_ROOT / "smoke" / f"{check}-{model_id}.json", result)


def stage_train(args: argparse.Namespace) -> None:
    from .cache import load_folding_model
    from .online_training import OnlineTrainingConfig, ema_decay_for, train_online
    from .rollouts import use_fast_folding_kernels

    category, budget = (
        ("smoke", SMOKE_BUDGET_HOURS) if args.probe else (f"model-{args.model}", MODEL_BUDGET_HOURS)
    )
    with gpu_budget(category, f"train-{args.run}", budget) as remaining_seconds:
        config = OnlineTrainingConfig(
            model_id=args.model,
            planned_updates=args.planned_updates,
            warmup_updates=min(300, max(1, args.planned_updates // 10)),
            ema_decay=ema_decay_for(args.planned_updates),
            ranking_weight=args.ranking_weight,
            maximum_seconds=int(args.hours * 3600),
            validation_interval_seconds=args.validation_minutes * 60,
            checkpoint_interval_seconds=1800,
            validation_limit=args.validation_limit,
        )
        model = load_folding_model(args.model)
        use_fast_folding_kernels(model)
        deadline_seconds = remaining_seconds - (
            0.0 if args.probe else EVALUATION_RESERVE_HOURS * 3600
        )
        report = train_online(
            model,
            POOL_DIR,
            split_targets("train"),
            split_targets("validation"),
            RUNS_DIR / args.model / args.run,
            config,
            log,
            deadline_seconds,
        )
    log(json.dumps(report))


def reference_model() -> object:
    """Load registry-pinned production `esmfold2` and its release confidence head on the CPU."""
    import torch

    from fastplms.models.esmfold2.configuration_esmfold2 import ESMFold2Config
    from fastplms.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    from fastplms.registry import get_model_spec

    spec = get_model_spec("esmfold2")
    config = ESMFold2Config.from_pretrained(
        spec.fast.repo_id,
        revision=spec.fast.revision,
        attn_implementation="sdpa",
        esmc_precision="bf16",
    )
    model = ESMFold2Model.from_pretrained(
        spec.fast.repo_id,
        revision=spec.fast.revision,
        config=config,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
        esmc_precision="bf16",
        load_esmc=True,
    )
    return model.eval().requires_grad_(False)


def stage_download_reference() -> None:
    """Load the production checkpoint once on the CPU so its download is not charged as GPU time."""
    reference_model()
    log("production esmfold2 and its ESMC backbone are cached")


def evaluation_scope(
    split: str, limit: int | None, category: str, budget_hours: float
) -> tuple[list[dict[str, object]], str, float, Path]:
    """Targets, budget category, budget hours, and output root of an evaluation.

    The test split is spent. Subsequent evaluations retain that designation. Validation is a
    dry run of the same code, charged to the smoke budget and written apart from test results.
    """
    if limit is not None and limit <= 0:
        raise ValueError("Evaluation limit must be positive")
    targets = split_targets(split)[:limit]
    if split == "test":
        return targets, category, budget_hours, DATA_ROOT / "evaluation"
    return targets, "smoke", SMOKE_BUDGET_HOURS, DATA_ROOT / f"evaluation-dry-run-{split}"


def evaluation_output(split: str, model_id: str, evaluation_id: str | None) -> tuple[str, Path]:
    """Choose a new result group and refuse a reused model directory before model loading."""
    if split not in {"test", "validation"}:
        raise ValueError("Evaluations require the test or validation split")
    validate_evaluation_id(model_id)
    identifier = validate_evaluation_id(evaluation_id or new_evaluation_id())
    root = DATA_ROOT / ("evaluation" if split == "test" else f"evaluation-dry-run-{split}")
    output = root / identifier / model_id
    if output.exists():
        raise FileExistsError(f"Evaluation output must be a new directory: {output}")
    return identifier, output


def prepare_evaluation(
    output_dir: Path,
    evaluation_id: str,
    model_id: str,
    split: str,
    targets: list[dict[str, object]],
    head_files: dict[str, Path | None],
    training_report: Path | None = None,
) -> EvaluationArtifacts:
    from fastplms.registry import get_model_spec

    from .config import DONOR_REPO, DONOR_REVISION, DONOR_WEIGHT_SHA256
    from .rollouts import INFERENCE_LOOPS, INFERENCE_SAMPLING_STEPS
    from .test_evaluation import EVALUATION_SAMPLES, EVALUATION_SEED_OFFSET

    spec = get_model_spec(model_id)
    input_files = {"split-report.json": SPLITS_DIR / "split-report.json"}
    if training_report is not None:
        input_files["training-report.json"] = training_report
    metadata = {
        "model": {
            "repo_id": spec.fast.repo_id,
            "revision": spec.fast.revision,
            "files": [asdict(item) for item in spec.fast.files],
        },
        "donor": (
            {
                "repo_id": DONOR_REPO,
                "revision": DONOR_REVISION,
                "weight_sha256": DONOR_WEIGHT_SHA256,
            }
            if head_files
            else None
        ),
        "inference": {
            "samples": EVALUATION_SAMPLES,
            "seed_offset": EVALUATION_SEED_OFFSET,
            "seed_rule": "seed_offset + index_in_targets_json",
            "recycling_loops": INFERENCE_LOOPS,
            "diffusion_steps": INFERENCE_SAMPLING_STEPS,
            "attention_backend": "sdpa",
            "fast_folding_kernels": True,
            "parameter_dtype": "float32",
            "fold_autocast_dtype": "bfloat16",
            "head_autocast_dtype": "bfloat16",
            "esmc_precision": "bf16",
        },
        "dataset": {
            "repo_id": "Synthyra/AtlasFold-Data",
            "split_identity": "inputs/split-report.json",
        },
        "source_files": source_identity(Path(__file__).resolve().parents[2]),
    }
    return EvaluationArtifacts.create(
        output_dir,
        evaluation_id=evaluation_id,
        model_id=model_id,
        split=split,
        targets=targets,
        head_files=head_files,
        metadata=metadata,
        input_files=input_files,
    )


def validate_resume_protocol(request: dict[str, object]) -> None:
    from .config import DONOR_REPO, DONOR_REVISION, DONOR_WEIGHT_SHA256
    from .experiment_artifacts import verify_resume_runtime
    from .rollouts import INFERENCE_LOOPS, INFERENCE_SAMPLING_STEPS
    from .test_evaluation import EVALUATION_SAMPLES, EVALUATION_SEED_OFFSET

    expected_inference = {
        "samples": EVALUATION_SAMPLES,
        "seed_offset": EVALUATION_SEED_OFFSET,
        "seed_rule": "seed_offset + index_in_targets_json",
        "recycling_loops": INFERENCE_LOOPS,
        "diffusion_steps": INFERENCE_SAMPLING_STEPS,
        "attention_backend": "sdpa",
        "fast_folding_kernels": True,
        "parameter_dtype": "float32",
        "fold_autocast_dtype": "bfloat16",
        "head_autocast_dtype": "bfloat16",
        "esmc_precision": "bf16",
    }
    expected_donor = {
        "repo_id": DONOR_REPO,
        "revision": DONOR_REVISION,
        "weight_sha256": DONOR_WEIGHT_SHA256,
    }
    if (
        request["metadata"]["inference"] != expected_inference
        or request["metadata"]["donor"] != expected_donor
    ):
        raise ValueError("Inference policy or donor checkpoint changed before recovery")
    verify_resume_runtime(request)


def stage_evaluate(
    model_id: str,
    run: str,
    split: str,
    limit: int | None,
    evaluation_id: str | None = None,
    *,
    resume: bool = False,
) -> None:
    if resume:
        identifier = validate_evaluation_id(evaluation_id or "")
        output_dir = (
            DATA_ROOT / "evaluation" / identifier / validate_evaluation_id(model_id)
        )
        if split != "test" or limit is not None:
            raise ValueError("Recovery requires the complete original test inventory")
    else:
        identifier, output_dir = evaluation_output(split, model_id, evaluation_id)
    validate_evaluation_id(run)

    from .cache import load_folding_model
    from .rollouts import use_fast_folding_kernels
    from .test_evaluation import (
        fold_and_score,
        load_heads,
        summarize,
        validated_partial_targets,
    )

    run_dir = RUNS_DIR / model_id / run
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "complete" or report.get("model_id") != model_id:
        raise ValueError(
            "Evaluation requires a completed training report for the requested model"
        )
    selected = Path(report["selected_checkpoint"])
    if selected.name != str(selected) or selected.suffix != ".safetensors":
        raise ValueError(
            "Selected checkpoint must be a safetensors filename within the training run"
        )
    head_files = {
        "v2": run_dir / selected,
        "pilot": PILOT_DIR / f"{model_id}-head.safetensors",
        "donor": None,
    }
    targets, category, budget, _ = evaluation_scope(
        split, limit, f"model-{model_id}", MODEL_BUDGET_HOURS
    )
    deadline = None
    all_targets_saved = False
    if resume:
        from fastplms.registry import get_model_spec

        from .experiment_artifacts import verify_evaluation

        artifacts = EvaluationArtifacts(output_dir)
        if (output_dir / "completion.json").exists():
            verify_evaluation(output_dir)
            return
        recovery = json.loads((output_dir / "resume.json").read_text(encoding="utf-8"))
        if (
            asdict(file_identity(output_dir / "request.json"))
            != recovery["original_request"]
        ):
            raise ValueError("Original evaluation request changed before recovery")
        request = json.loads((output_dir / "request.json").read_text(encoding="utf-8"))
        validate_resume_protocol(request)
        if (
            request["model_id"] != model_id
            or request["evaluation_id"] != identifier
            or request["split"] != split
        ):
            raise ValueError("Recovery request belongs to a different evaluation")
        if (
            json.loads((output_dir / "targets.json").read_text(encoding="utf-8"))
            != targets
        ):
            raise ValueError("Evaluation targets changed before recovery")
        spec = get_model_spec(model_id)
        expected_model = {
            "repo_id": spec.fast.repo_id,
            "revision": spec.fast.revision,
            "files": [asdict(item) for item in spec.fast.files],
        }
        if request["metadata"]["model"] != expected_model:
            raise ValueError("Pinned folding model changed before recovery")
        for name, path in {
            "split-report.json": SPLITS_DIR / "split-report.json",
            "training-report.json": run_dir / "report.json",
        }.items():
            if (
                asdict(file_identity(path))
                != request["public_inputs"][f"inputs/{name}"]
            ):
                raise ValueError(f"Evaluation input changed before recovery: {name}")
        remaining = recovery["metadata"]["deadline_unix"] - time.time()
        if remaining <= 0:
            raise TimeoutError("The recovery GPU budget has expired")
        deadline = time.monotonic() + remaining
        retained = validated_partial_targets(
            output_dir / "partial-records", targets, set(head_files)
        )
        all_targets_saved = len(retained) == len(targets)
        history = output_dir / "resume-history"
        history.mkdir(exist_ok=True)
        attempt = uuid.uuid4().hex
        write_new_json(
            history / f"attempt-{attempt}.json",
            {
                "started_unix": time.time(),
                "remaining_seconds": remaining,
                "retained_targets": len(retained),
                "source_files": source_identity(Path(__file__).resolve().parents[2]),
                "environment": environment_identity(),
                "gpu": recovery["metadata"]["gpu"],
            },
        )
        for name in ("records.json", "skipped.json", "summary.json"):
            path = output_dir / name
            if path.exists():
                path.rename(history / f"interrupted-{attempt}-{name}")
    else:
        artifacts = prepare_evaluation(
            output_dir,
            identifier,
            model_id,
            split,
            targets,
            head_files,
            run_dir / "report.json",
        )
    log(f"evaluation {identifier}: {output_dir}")
    try:
        with gpu_budget(category, f"evaluate-{run}-{split}", budget):
            model = None if all_targets_saved else load_folding_model(model_id)
            if model is not None:
                use_fast_folding_kernels(model)
            records = fold_and_score(
                model,
                POOL_DIR,
                targets,
                {name: None for name in artifacts.head_files()}
                if all_targets_saved
                else load_heads(model_id, artifacts.head_files()),
                output_dir / "records.json",
                log,
                **({"resume": True, "deadline": deadline} if resume else {}),
            )
        summary = summarize(records, list(head_files))
        write_new_json(
            output_dir / "summary.json",
            {
                "run": run,
                "head_files": {name: str(path) for name, path in head_files.items()},
                **summary,
            },
        )
        artifacts.complete()
    except BaseException as error:
        artifacts.fail(error)
        raise
    log(json.dumps(summary, indent=2)[:4000])


def stage_reference(split: str, limit: int | None, evaluation_id: str | None = None) -> None:
    identifier, output_dir = evaluation_output(split, "esmfold2", evaluation_id)

    from .rollouts import use_fast_folding_kernels
    from .test_evaluation import fold_and_score, summarize

    targets, category, budget, _ = evaluation_scope(
        split, limit, "reference", REFERENCE_BUDGET_HOURS
    )
    artifacts = prepare_evaluation(output_dir, identifier, "esmfold2", split, targets, {})
    log(f"evaluation {identifier}: {output_dir}")
    try:
        with gpu_budget(category, f"reference-esmfold2-{split}", budget):
            model = reference_model().to("cuda")  # type: ignore[attr-defined]
            use_fast_folding_kernels(model)
            records = fold_and_score(
                model, POOL_DIR, targets, None, output_dir / "records.json", log
            )
        summary = summarize(records, ["production"])
        write_new_json(output_dir / "summary.json", summary)
        artifacts.complete()
    except BaseException as error:
        artifacts.fail(error)
        raise
    log(json.dumps(summary, indent=2)[:4000])


def stage_acceptance(model_id: str, evaluation_id: str | None = None) -> None:
    """Apply the pre-registered gates to a model's test records and the production reference."""
    from .acceptance import acceptance_gates, paired_estimates, production_agreement

    evaluation = DATA_ROOT / "evaluation"
    if evaluation_id is not None:
        evaluation /= validate_evaluation_id(evaluation_id)
        verified = verify_evaluation_group(
            {
                model_id: evaluation / model_id,
                "esmfold2": evaluation / "esmfold2",
            }
        )
        if verified[model_id]["evaluation_id"] != evaluation_id:
            raise ValueError("Evaluation manifest group differs from the requested group")
    output = evaluation / f"acceptance-{model_id}.json"
    if output.exists():
        raise FileExistsError(f"Acceptance output already exists: {output}")
    model_records = json.loads((evaluation / model_id / "records.json").read_text(encoding="utf-8"))
    reference_records = json.loads(
        (evaluation / "esmfold2" / "records.json").read_text(encoding="utf-8")
    )
    heads = ["v2", "pilot", "donor"]
    estimates = paired_estimates(model_records, heads, reference_records)
    result = acceptance_gates(estimates)
    agreement = production_agreement(model_records, heads, reference_records)
    write_new_json(
        output,
        {
            **result,
            "production_agreement": agreement,
            "estimates": estimates,
            "evaluation_id": evaluation_id,
            "test_set_status": "spent",
            "input_files": {
                f"{name}/records.json": asdict(file_identity(evaluation / name / "records.json"))
                for name in (model_id, "esmfold2")
            },
            "analysis_source_files": {
                name: asdict(file_identity(Path(__file__).with_name(name)))
                for name in ("acceptance.py", "v2_analysis.py")
            },
        },
    )
    log(json.dumps({**result, "production_agreement": agreement}, indent=2))


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".secrets.env", override=False)
    os.environ.setdefault("HF_HOME", str(Path.home() / "data" / "hf-home"))
    # Inputs range from under 100 to 2,048 tokens, which fragments fixed-size allocator blocks.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # These checkpoints use single sequences; per-chain notices would fill day-long logs.
    warnings.filterwarnings("ignore", message="No MSA provided for")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    stages = parser.add_subparsers(dest="stage", required=True)
    stages.add_parser("pilot-artifacts", help="copy pilot records and heads from Modal")
    pool_parser = stages.add_parser("pool", help="build the target pool from AtlasFold-Data")
    pool_parser.add_argument("--workers", type=int, default=48)
    splits_parser = stages.add_parser(
        "splits", help="cluster chains and assign train, validation, and test"
    )
    splits_parser.add_argument("--threads", type=int, default=48)
    smoke_parser = stages.add_parser("smoke", help="GPU throughput, parity, or overfit check")
    smoke_parser.add_argument(
        "check", choices=("throughput", "kernels", "parity", "overfit", "training-rate")
    )
    smoke_parser.add_argument("--model", required=True, choices=("esmfold2_300", "esmfold2_600"))
    train_parser = stages.add_parser("train", help="train a head on online rollouts")
    train_parser.add_argument("--model", required=True, choices=("esmfold2_300", "esmfold2_600"))
    train_parser.add_argument("--run", required=True, help="run directory name under runs/<model>")
    train_parser.add_argument("--planned-updates", type=int, required=True)
    train_parser.add_argument("--hours", type=float, required=True)
    train_parser.add_argument("--ranking-weight", type=float, default=0.5)
    train_parser.add_argument("--validation-minutes", type=int, default=90)
    train_parser.add_argument("--validation-limit", type=int)
    train_parser.add_argument(
        "--probe", action="store_true", help="charge the smoke budget instead of the model budget"
    )
    evaluate_parser = stages.add_parser(
        "evaluate", help="score the v2, pilot, and donor heads on the test set"
    )
    evaluate_parser.add_argument("--model", required=True, choices=("esmfold2_300", "esmfold2_600"))
    evaluate_parser.add_argument("--run", required=True)
    evaluate_parser.add_argument(
        "--split", default="test", choices=("test", "validation"), help="validation runs a dry run"
    )
    evaluate_parser.add_argument("--limit", type=int, help="evaluate only the first targets")
    evaluate_parser.add_argument(
        "--evaluation-id", help="New result group; reuse its ID across models and reference"
    )
    reference_parser = stages.add_parser(
        "reference", help="score production esmfold2 on the test set"
    )
    reference_parser.add_argument(
        "--split", default="test", choices=("test", "validation"), help="validation runs a dry run"
    )
    reference_parser.add_argument("--limit", type=int, help="evaluate only the first targets")
    reference_parser.add_argument(
        "--evaluation-id", help="Result group shared with the evaluated heads"
    )
    stages.add_parser(
        "download-reference", help="cache the production checkpoint without using the GPU"
    )
    acceptance_parser = stages.add_parser(
        "acceptance", help="apply the pre-registered gates to test records"
    )
    acceptance_parser.add_argument(
        "--model", required=True, choices=("esmfold2_300", "esmfold2_600")
    )
    acceptance_parser.add_argument(
        "--evaluation-id", help="Result group; omission reads the historical layout"
    )
    args = parser.parse_args()

    stages_by_name: dict[str, Callable[[], None]] = {
        "pilot-artifacts": fetch_pilot_artifacts,
        "pool": lambda: stage_pool(args.workers),
        "splits": lambda: stage_splits(args.threads),
        "smoke": lambda: stage_smoke(args.check, args.model),
        "train": lambda: stage_train(args),
        "evaluate": lambda: stage_evaluate(
            args.model, args.run, args.split, args.limit, args.evaluation_id
        ),
        "reference": lambda: stage_reference(args.split, args.limit, args.evaluation_id),
        "download-reference": stage_download_reference,
        "acceptance": lambda: stage_acceptance(args.model, args.evaluation_id),
    }
    stages_by_name[args.stage]()


if __name__ == "__main__":
    main()
