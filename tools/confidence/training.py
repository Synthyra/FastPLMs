"""Train native confidence heads from frozen, model-specific folding caches."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import time
import uuid

import torch
import transformers
import wandb

from pathlib import Path
from dataclasses import replace

from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor, nn

from fastplms.models.esmfold2.configuration_esmfold2 import ESMFold2Config
from fastplms.models.esmfold2.modeling_esmfold2_common import ResIdxAsymIdSymIdEntityIdEncoding
from fastplms.models.esmfold2.modeling_esmfold2_experimental import ConfidenceHead
from fastplms.registry import get_model_spec

from .cache import _structure_input, cache_target, confidence_inputs, load_cache, load_folding_model
from .config import (
    DONOR_REPO,
    DONOR_REVISION,
    DONOR_TENSOR_BYTES,
    DONOR_TENSOR_COUNT,
    DONOR_WEIGHT_SHA256,
    TrainingConfig,
    WANDB_PROJECT,
    resource_rate,
)
from .labels import compute_targets, confidence_loss
from .data import ATLASFOLD_REVISION
from .metrics import summarize
from .selection import improves_checkpoint
from .structure_metrics import compute_structure_metrics


def _file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class HeadContext(nn.Module):
    """Only the head and two small frozen encoders are resident during training."""

    def __init__(self, model_id: str, device: str = "cuda") -> None:
        super().__init__()
        spec = get_model_spec(model_id)
        config = ESMFold2Config.from_pretrained(spec.fast.repo_id, revision=spec.fast.revision)
        self.rel_pos = ResIdxAsymIdSymIdEntityIdEncoding(
            n_relative_residx_bins=config.n_relative_residx_bins,
            n_relative_chain_bins=config.n_relative_chain_bins,
            d_pair=config.d_pair,
        )
        self.token_bonds = nn.Linear(1, config.d_pair, bias=False)
        self.head = ConfidenceHead(config)
        self.head.set_chunk_size(32)
        base = Path(
            hf_hub_download(spec.fast.repo_id, "model.safetensors", revision=spec.fast.revision)
        )
        expected = spec.fast.file_map["model.safetensors"].digest
        if _file_hash(base) != expected:
            raise ValueError("Frozen checkpoint hash differs from the registry")
        donor = Path(hf_hub_download(DONOR_REPO, "model.safetensors", revision=DONOR_REVISION))
        if _file_hash(donor) != DONOR_WEIGHT_SHA256:
            raise ValueError("Donor checkpoint hash differs from its pinned identity")
        with safe_open(str(base), framework="pt") as handle:
            for name in ("rel_pos", "token_bonds"):
                state = {
                    key[len(name) + 1 :]: handle.get_tensor(key)
                    for key in handle.keys()  # noqa: SIM118 - safetensors reader is not iterable
                    if key.startswith(name + ".")
                }
                getattr(self, name).load_state_dict(state, strict=True)
        with safe_open(str(donor), framework="pt") as handle:
            state = {
                key.removeprefix("confidence_head."): handle.get_tensor(key)
                for key in handle.keys()  # noqa: SIM118 - safetensors reader is not iterable
                if key.startswith("confidence_head.")
            }
        if (
            len(state) != DONOR_TENSOR_COUNT
            or sum(t.numel() * t.element_size() for t in state.values()) != DONOR_TENSOR_BYTES
        ):
            raise ValueError("Donor confidence subtree has an unexpected schema")
        self.head.load_state_dict(state, strict=True)
        self.rel_pos.requires_grad_(False)
        self.token_bonds.requires_grad_(False)
        self.to(device=device, dtype=torch.float32)
        self.eval()
        self.base_weight_sha256 = expected


def _cache_path(root: Path, model_id: str, record: dict, seed: int = 17) -> Path:
    identifier = hashlib.sha256(str(record["id"]).encode()).hexdigest()[:24]
    return root / model_id / "cache" / f"{identifier}-{seed}.safetensors"


def _records(root: Path) -> list[dict]:
    receipt = json.loads((root / "data/split-report.json").read_text())
    if receipt["status"] != "verified":
        raise ValueError("Sequence-cluster split verification has not passed")
    path = root / "data/records.json"
    if _file_hash(path) != receipt["records_sha256"]:
        raise ValueError("Dataset manifest changed after split verification")
    return json.loads(path.read_text())


def _targets(cache: dict[str, Tensor]) -> dict[str, Tensor]:
    stored = {
        name.removeprefix("target/"): value
        for name, value in cache.items()
        if name.startswith("target/")
    }
    if stored:
        return stored
    return compute_targets(
        cache["x_pred"].reshape(-1, 3),
        cache["true_coords"],
        cache["resolved_mask"],
        cache["atom_to_token"].reshape(-1),
        cache["backbone_indices"],
        cache["token_attention_mask"].reshape(-1).bool(),
    )


def _cache_one(model, record: dict, root: Path, model_id: str, seed: int) -> dict:
    path = _cache_path(root, model_id, record, seed)
    if path.exists():
        try:
            load_cache(
                path,
                model_id=model_id,
                model_revision=get_model_spec(model_id).fast.revision,
                seed=seed,
            )
        except (OSError, ValueError):
            path.rename(path.with_name(f"{path.name}.invalid-{uuid.uuid4().hex[:8]}"))
    if not path.exists():
        cache_target(model, record, root / "data", path, seed)
    cache, metadata = load_cache(
        path, model_id=model_id, model_revision=get_model_spec(model_id).fast.revision, seed=seed
    )
    record_hash = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if metadata["record_sha256"] != record_hash:
        raise ValueError("Cached record differs from the selected dataset")
    if not any(name.startswith("target/") for name in cache):
        cache.update({f"target/{name}": value for name, value in _targets(cache).items()})
        temporary = path.with_suffix(".tmp")
        save_file(
            {key: value.contiguous() for key, value in cache.items()},
            str(temporary),
            metadata=metadata,
        )
        temporary.replace(path)
    return {"id": record["id"], "path": str(path), "seed": seed, "bytes": path.stat().st_size}


def generate_caches(
    root: Path, model_id: str, maximum_targets: int = 32, maximum_seconds: int = 6300
) -> dict:
    run = _wandb_run(root, model_id, "cache", _training_settings(root, model_id))
    failed = False
    try:
        report = _generate_caches(root, model_id, maximum_targets, maximum_seconds, run)
        report["wandb_url"] = run.url
        run.summary["status"] = report["status"]
        run.summary["remaining"] = report["remaining"]
        return report
    except BaseException as error:
        failed = True
        run.summary["status"] = "failed"
        run.summary["error_type"] = type(error).__name__
        run.summary["error"] = str(error)
        raise
    finally:
        run.finish(exit_code=1 if failed else 0)


def _generate_caches(
    root: Path, model_id: str, maximum_targets: int, maximum_seconds: int, run
) -> dict:
    started = time.monotonic()
    records = _records(root)
    tasks = [
        (record, seed)
        for record in records
        for seed in ((17, 29) if record["split"] == "final_test" else (17,))
    ]
    pending = []
    invalid = []
    revision = get_model_spec(model_id).fast.revision
    for record, seed in tasks:
        path = _cache_path(root, model_id, record, seed)
        if not path.exists():
            pending.append((record, seed))
            continue
        try:
            load_cache(path, model_id=model_id, model_revision=revision, seed=seed)
        except (OSError, ValueError):
            invalid.append(str(path))
            pending.append((record, seed))
    run.log({"cache/completed": len(tasks) - len(pending), "cache/total": len(tasks)})
    if not pending:
        return {"status": "complete", "cached": len(tasks), "remaining": 0, "invalid": invalid}
    model = load_folding_model(model_id)
    completed = []
    for record, seed in pending[:maximum_targets]:
        if time.monotonic() - started > maximum_seconds:
            break
        completed.append(_cache_one(model, record, root, model_id, seed))
        elapsed = time.monotonic() - started
        run.log(
            {
                "cache/completed": len(tasks) - len(pending) + len(completed),
                "cache/total": len(tasks),
                "cache/remaining": len(pending) - len(completed),
                "cache/elapsed_seconds": elapsed,
                "cache/targets_per_second": len(completed) / elapsed,
                "cache/peak_memory_bytes": torch.cuda.max_memory_allocated(),
            }
        )
    return {
        "status": "partial" if len(pending) > len(completed) else "complete",
        "completed": completed,
        "remaining": len(pending) - len(completed),
        "invalid": invalid,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
    }


def _forward(context: HeadContext, cache: dict[str, Tensor]) -> dict[str, Tensor]:
    inputs = confidence_inputs(context, cache)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return context.head(**inputs)


def _load_example(root: Path, model_id: str, record: dict, seed: int = 17) -> tuple[dict, dict]:
    cache, metadata = load_cache(
        _cache_path(root, model_id, record, seed),
        model_id=model_id,
        model_revision=get_model_spec(model_id).fast.revision,
        seed=seed,
    )
    expected_record = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if metadata.get("record_sha256") != expected_record:
        raise ValueError("Cached labels belong to a different selected record")
    return cache, {key: value.cuda() for key, value in _targets(cache).items()}


def _prediction_record(
    record: dict, cache: dict, targets: dict, output: dict, quality: dict
) -> dict:
    losses = confidence_loss(output, targets)
    atom_mask = targets["plddt_mask"]
    ca_mask = targets["lddt_ca_mask"]
    ca_indices = cache["backbone_indices"][:, 1].long().cuda()
    atom_prediction = output["plddt_per_atom"].reshape(-1).float()
    pae_mask = targets["pae_mask"]
    return {
        "target_id": record["id"],
        "kind": record["kind"],
        "split": record["split"],
        "plddt_pred": atom_prediction[atom_mask].tolist(),
        "plddt_true": targets["plddt_score"][atom_mask].tolist(),
        "plddt_ca_pred": atom_prediction[ca_indices.clamp_min(0)][ca_mask].tolist(),
        "plddt_ca_true": targets["lddt_ca"][ca_mask].tolist(),
        "plddt_ce": float(losses["plddt"]),
        "pae_ce": float(losses["pae"]),
        "pae_overflow_fraction": float((targets["pae_error"][pae_mask] >= 32).float().mean()),
        "ptm": float(output["ptm"].reshape(-1)[0]),
        "iptm": float(output["iptm"].reshape(-1)[0]) if record["kind"] == "dimer" else None,
        **quality,
    }


@torch.no_grad()
def predict_records(
    context: HeadContext, root: Path, model_id: str, records: list[dict], two_seeds: bool = False
) -> list[dict]:
    context.head.eval()
    predictions = []
    for record in records:
        for seed in (17, 29) if two_seeds else (17,):
            cache, targets = _load_example(root, model_id, record, seed)
            quality_path = _cache_path(root, model_id, record, seed).with_suffix(".quality.json")
            if quality_path.exists():
                quality = json.loads(quality_path.read_text())
            else:
                quality = compute_structure_metrics(
                    cache, record, quality_path.parent / (quality_path.stem + "-structures")
                )
                quality_path.write_text(json.dumps(quality) + "\n")
            output = _forward(context, cache)
            prediction = _prediction_record(record, cache, targets, output, quality)
            prediction["seed"] = seed
            predictions.append(prediction)
    return predictions


def _wandb_run(root: Path, model_id: str, kind: str, config: dict):
    directory = root / model_id / kind
    directory.mkdir(parents=True, exist_ok=True)
    identifier_file = directory / "wandb-id.txt"
    identifier = (
        identifier_file.read_text().strip()
        if identifier_file.exists() and kind not in {"smoke", "benchmark"}
        else uuid.uuid4().hex[:12]
    )
    identifier_file.write_text(identifier)
    run = wandb.init(
        project=WANDB_PROJECT,
        id=identifier,
        resume="allow",
        mode="online",
        group="esmfold2-confidence-pilot",
        job_type=kind,
        name=f"{model_id}-{kind}",
        config=config,
        dir=str(directory),
        save_code=False,
        settings=wandb.Settings(init_timeout=90, disable_git=True, console="off"),
    )
    mode = getattr(getattr(run, "settings", None), "mode", None)
    if mode is not None and mode != "online":
        run.finish(exit_code=1)
        raise RuntimeError("confidence training requires an online W&B run")
    return run


def _training_settings(root: Path, model_id: str) -> dict:
    config = TrainingConfig().to_dict()
    archive_root = root.parent / "archives" if root.name == "smoke" else root / "archives"
    archive_receipts = {
        path.stem: json.loads(path.read_text()) for path in sorted(archive_root.glob("*.json"))
    }
    gpu_name = torch.cuda.get_device_name()
    gpu_type = next((name for name in ("H100", "L40S", "L4") if name in gpu_name), None)
    if "H200" in gpu_name:
        # Modal may fulfill an H100 request with H200 at the requested rate.
        gpu_type = "H100"
    if gpu_type is None:
        raise ValueError(f"No recorded resource rate for GPU {gpu_name}")
    spec = get_model_spec(model_id)
    config.update(
        model_id=model_id,
        model_repo=spec.fast.repo_id,
        model_revision=spec.fast.revision,
        model_weight_sha256=spec.fast.file_map["model.safetensors"].digest,
        donor_repo=DONOR_REPO,
        donor_revision=DONOR_REVISION,
        donor_weight_sha256=DONOR_WEIGHT_SHA256,
        atlasfold_revision=ATLASFOLD_REVISION,
        dataset_archives=archive_receipts,
        dataset_sha256=_file_hash(root / "data/records.json"),
        training_code_sha256=_file_hash(Path(__file__)),
        workflow_files={
            path.name: _file_hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        torch_version=str(torch.__version__),
        transformers_version=transformers.__version__,
        python_version=platform.python_version(),
        gpu=torch.cuda.get_device_name(),
        backbone_precision="bf16",
        head_parameters="float32",
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        estimated_dollars_per_second=resource_rate(gpu_type),
    )
    return config


def benchmark(root: Path, model_id: str, maximum_seconds: int = 480, smoke: bool = False) -> dict:
    if smoke:
        root = root / "smoke"
    run = _wandb_run(
        root, model_id, "smoke" if smoke else "benchmark", _training_settings(root, model_id)
    )
    failed = False
    try:
        report = _benchmark(root, model_id, maximum_seconds)
        report["wandb_url"] = run.url
        report["optimizer_updates"] = 0
        run.log({key: value for key, value in report.items() if isinstance(value, int | float)})
        run.summary["status"] = report["status"]
        run.summary["target_seconds"] = report["target_seconds"]
        return report
    except BaseException:
        failed = True
        raise
    finally:
        run.finish(exit_code=1 if failed else 0)


def _benchmark(root: Path, model_id: str, maximum_seconds: int) -> dict:
    records = sorted(
        (r for r in _records(root) if r["split"] == "train"),
        key=lambda r: len(r["sequence"])
        if "sequence" in r
        else sum(len(c["sequence"]) for c in r["chains"]),
    )
    panel = [records[index] for index in sorted({0, len(records) // 2, len(records) - 1})]
    model = load_folding_model(model_id)
    benchmark_root = root / model_id / "benchmark" / uuid.uuid4().hex
    (benchmark_root / "cache").mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    times = []
    for record in panel:
        item_start = time.monotonic()
        fresh_path = benchmark_root / "cache" / f"{len(times)}.safetensors"
        cache_target(model, record, root / "data", fresh_path, 17)
        torch.cuda.synchronize()
        times.append(time.monotonic() - item_start)
        if time.monotonic() - started > maximum_seconds:
            break
    context = HeadContext(model_id)
    initial_context_hash = _state_hash(context)
    cache, _ = load_cache(
        benchmark_root / "cache" / "0.safetensors",
        model_id=model_id,
        model_revision=get_model_spec(model_id).fast.revision,
        seed=17,
    )
    with torch.no_grad():
        cached_output = _forward(context, cache)
    prepared, _ = model.prepare_structure_input(_structure_input(panel[0]), seed=17)
    prepared = {name: value.to(model.device) for name, value in prepared.items()}
    model.confidence_head = context.head
    model.config.confidence_head.enabled = True
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        enabled = model(
            **prepared,
            calculate_confidence=True,
            output_hidden_states=True,
            return_dict=True,
            num_loops=3,
            num_sampling_steps=15,
            num_diffusion_samples=1,
            seed=17,
        )
        disabled = model(
            **prepared,
            calculate_confidence=False,
            output_hidden_states=True,
            return_dict=True,
            num_loops=3,
            num_sampling_steps=15,
            num_diffusion_samples=1,
            seed=17,
        )
    torch.testing.assert_close(enabled.hidden_states[0], disabled.hidden_states[0], rtol=0, atol=0)
    torch.testing.assert_close(enabled.hidden_states[1], disabled.hidden_states[1], rtol=0, atol=0)
    torch.testing.assert_close(
        enabled["sample_atom_coords"], disabled["sample_atom_coords"], rtol=0, atol=0
    )
    torch.testing.assert_close(cached_output["plddt_logits"], enabled["plddt_logits"])
    torch.testing.assert_close(cached_output["pae_logits"], enabled["pae_logits"])
    del model, enabled, disabled, cached_output
    torch.cuda.empty_cache()
    targets = {key: value.cuda() for key, value in _targets(cache).items()}
    started_head = time.monotonic()
    output = _forward(context, cache)
    loss = confidence_loss(output, targets)["total"]
    loss.backward()
    gradients = [
        parameter.grad for parameter in context.head.parameters() if parameter.grad is not None
    ]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError("Confidence smoke check produced missing or nonfinite gradients")
    if initial_context_hash != _state_hash(context):
        raise RuntimeError("Smoke check changed model parameters without an optimizer update")
    if any(
        parameter.grad is not None
        for module in (context.rel_pos, context.token_bonds)
        for parameter in module.parameters()
    ):
        raise RuntimeError("Frozen positional encoders received gradients")
    torch.cuda.synchronize()
    return {
        "status": "benchmarked",
        "gpu": torch.cuda.get_device_name(),
        "target_seconds": times,
        "head_microbatch_seconds": time.monotonic() - started_head,
        "head_loss": float(loss.detach()),
        "parameters_with_gradients": len(gradients),
        "parameters_unchanged": True,
        "frozen_encoders_have_no_gradients": True,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
    }


def _learning_rate(update: int, config: TrainingConfig) -> float:
    if update < config.warmup_updates:
        return config.learning_rate * (update + 1) / config.warmup_updates
    fraction = (update - config.warmup_updates) / (config.maximum_updates - config.warmup_updates)
    return config.minimum_learning_rate + 0.5 * (
        config.learning_rate - config.minimum_learning_rate
    ) * (1 + math.cos(math.pi * fraction))


def _checkpoint_settings_match(actual: dict, expected: dict) -> bool:
    # Modal can fulfill the same GPU request with a different compatible GPU.
    # Preserve numerical/software/data settings while recording actual hardware.
    return {key: value for key, value in actual.items() if key != "gpu"} == {
        key: value for key, value in expected.items() if key != "gpu"
    }


def train_head(
    root: Path, model_id: str, maximum_seconds: int = 36000, overfit: bool = False
) -> dict:
    config = TrainingConfig()
    if not overfit:
        # Wall time controls the long run; this update bound protects against a
        # broken clock without ending a fast H100 run at the old pilot limit.
        config = replace(config, maximum_updates=1_000_000, early_stopping_patience=12)
    records = _records(root)
    train_records = [record for record in records if record["split"] == "train"]
    validation = [record for record in records if record["split"] == "validation"]
    kind = "overfit" if overfit else "train"
    directory = root / model_id / kind
    settings = _training_settings(root, model_id)
    settings.update(config.to_dict())
    settings.update(maximum_seconds=maximum_seconds, selection="both_heads_improve_with_rank_guard")
    run = _wandb_run(root, model_id, kind, settings)
    started = time.monotonic()
    failed = False
    try:
        torch.manual_seed(config.seed)
        context = HeadContext(model_id)
        if overfit:
            train_records = [
                record
                for chain_count in (1, 2)
                for record in [
                    item for item in train_records if len(item["chains"]) == chain_count
                ][:4]
            ]
            if len(train_records) != 8:
                raise ValueError("Overfit check requires four monomers and four dimers")
            validation = train_records
        else:
            overfit_report = json.loads((root / model_id / "overfit/result.json").read_text())
            if overfit_report["status"] != "passed":
                raise ValueError("The eight-target overfit check must pass before training")
        from .evaluation import build_frequency_baseline

        def cached_targets(selected_records):
            for record in selected_records:
                cache, _ = load_cache(
                    _cache_path(root, model_id, record, 17),
                    model_id=model_id,
                    model_revision=get_model_spec(model_id).fast.revision,
                    seed=17,
                )
                yield {name: value.cpu() for name, value in _targets(cache).items()}

        frequency_baseline = build_frequency_baseline(
            cached_targets(train_records), cached_targets(validation)
        )
        run.log(
            {
                f"baseline/{key}": value
                for key, value in frequency_baseline.items()
                if isinstance(value, int | float)
            }
        )
        optimizer = torch.optim.AdamW(
            context.head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        checkpoint = directory / "last.pt"
        update, best, patience = 0, None, 0
        previous_seconds = 0.0
        frozen_hashes = (_state_hash(context.rel_pos), _state_hash(context.token_bonds))
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location="cuda", weights_only=True)
            expected_checkpoint = {
                "model_id": model_id,
                "model_revision": get_model_spec(model_id).fast.revision,
                "model_pins": context.base_weight_sha256,
                "base_weight_sha256": context.base_weight_sha256,
                "frozen_hashes": frozen_hashes,
            }
            if not _checkpoint_settings_match(state["settings"], settings) or any(
                state.get(key) != value for key, value in expected_checkpoint.items()
            ):
                raise ValueError("Checkpoint settings differ from the current run")
            run.summary["current_gpu"] = settings["gpu"]
            for filename, field in (
                ("best.safetensors", "best_sha256"),
                ("donor-validation.json", "donor_validation_sha256"),
            ):
                saved_path = directory / filename
                observed_hash = _file_hash(saved_path) if saved_path.exists() else None
                if observed_hash != state[field]:
                    raise ValueError(f"Checkpoint companion file differs: {filename}")
            context.head.load_state_dict(state["head"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["rng"].cpu())
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
            update, best, patience = state["update"], state["best"], state["patience"]
            previous_seconds = state["training_seconds"]
        donor_path = directory / "donor-validation.json"
        if checkpoint.exists() and not donor_path.exists():
            raise ValueError("Resuming training requires the original donor validation report")
        if donor_path.exists():
            donor = json.loads(donor_path.read_text())
        else:
            donor_predictions = predict_records(context, root, model_id, validation)
            donor = summarize(donor_predictions)
            donor_path.write_text(json.dumps(donor, indent=2) + "\n")
            run.log(
                {
                    f"donor/{key}": value
                    for key, value in donor.items()
                    if isinstance(value, int | float)
                }
            )
        maximum_updates = 100 if overfit else config.maximum_updates
        if best is None:
            best = donor
        accumulated = 1 if overfit else config.accumulation_steps
        orders = {}
        latest = (
            summarize(predict_records(context, root, model_id, validation))
            if checkpoint.exists()
            else donor
        )
        while update < maximum_updates and patience < config.early_stopping_patience:
            elapsed = previous_seconds + time.monotonic() - started
            if elapsed > maximum_seconds:
                break
            context.head.train()
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                if overfit:
                    group["lr"] = config.learning_rate
                elif update < config.warmup_updates:
                    group["lr"] = _learning_rate(update, config)
                else:
                    fraction = min(1.0, elapsed / maximum_seconds)
                    group["lr"] = config.minimum_learning_rate + 0.5 * (
                        config.learning_rate - config.minimum_learning_rate
                    ) * (1 + math.cos(math.pi * fraction))
            loss_values = {"total": 0.0, "plddt": 0.0, "pae": 0.0}
            for microbatch in range(accumulated):
                epoch, offset = divmod(update * accumulated + microbatch, len(train_records))
                if epoch not in orders:
                    order = list(range(len(train_records)))
                    random.Random(config.seed + epoch).shuffle(order)
                    orders[epoch] = order
                record = train_records[orders[epoch][offset]]
                cache, targets = _load_example(root, model_id, record)
                output = _forward(context, cache)
                losses = confidence_loss(output, targets)
                if not torch.isfinite(losses["total"]):
                    raise FloatingPointError("Nonfinite confidence loss")
                (losses["total"] / accumulated).backward()
                for key in loss_values:
                    loss_values[key] += float(losses[key].detach()) / accumulated
            gradient_norm = nn.utils.clip_grad_norm_(
                context.head.parameters(), config.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            update += 1
            run.log(
                {
                    **{f"train/{key}": value for key, value in loss_values.items()},
                    "update": update,
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "elapsed_seconds": time.monotonic() - started,
                    "training_seconds": previous_seconds + time.monotonic() - started,
                    "estimated_session_dollars": (time.monotonic() - started)
                    * settings["estimated_dollars_per_second"],
                    "estimated_total_dollars": (previous_seconds + time.monotonic() - started)
                    * settings["estimated_dollars_per_second"],
                    "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                }
            )
            if update % config.validation_interval == 0 or update == maximum_updates:
                latest = summarize(predict_records(context, root, model_id, validation))
                if improves_checkpoint(latest, best, donor, overfit=overfit):
                    best, patience = latest, 0
                    save_file(
                        {
                            key: value.detach().cpu().contiguous()
                            for key, value in context.head.state_dict().items()
                        },
                        str(directory / "best.tmp"),
                    )
                    (directory / "best.tmp").replace(directory / "best.safetensors")
                else:
                    patience += 1
                run.log(
                    {
                        f"validation/{key}": value
                        for key, value in latest.items()
                        if isinstance(value, int | float)
                    }
                )
            if (
                update % config.validation_interval == 0
                or previous_seconds + time.monotonic() - started > maximum_seconds
            ):
                _save_checkpoint(
                    checkpoint,
                    context,
                    optimizer,
                    settings,
                    update,
                    best,
                    patience,
                    model_id,
                    frozen_hashes,
                    previous_seconds + time.monotonic() - started,
                )
        _save_checkpoint(
            checkpoint,
            context,
            optimizer,
            settings,
            update,
            best,
            patience,
            model_id,
            frozen_hashes,
            previous_seconds + time.monotonic() - started,
        )
        if frozen_hashes != (_state_hash(context.rel_pos), _state_hash(context.token_bonds)):
            raise RuntimeError("Frozen positional parameters changed during training")
        total_seconds = previous_seconds + time.monotonic() - started
        complete = (
            update == maximum_updates
            or patience >= config.early_stopping_patience
            or (not overfit and total_seconds >= maximum_seconds)
        )
        status = "complete" if complete else "paused"
        best_available = (directory / "best.safetensors").exists()
        if complete and not overfit and not best_available:
            status = "no_improvement"
        if overfit and complete:
            status = (
                "passed"
                if latest["plddt_ce"] + 0.1 * latest["pae_ce"]
                < 0.8 * (donor["plddt_ce"] + 0.1 * donor["pae_ce"])
                else "failed"
            )
        report = {
            "status": status,
            "updates": update,
            "training_seconds": total_seconds,
            "best_validation": best,
            "best_checkpoint_available": best_available,
            "wandb_url": run.url,
            "donor": donor,
            "validation": latest,
            "base_weight_sha256": context.base_weight_sha256,
            "estimated_session_dollars": (time.monotonic() - started)
            * settings["estimated_dollars_per_second"],
            "estimated_total_dollars": total_seconds * settings["estimated_dollars_per_second"],
        }
        (directory / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        artifact = wandb.Artifact(f"{model_id}-{kind}", type="model", metadata=settings)
        artifact.add_file(str(directory / "result.json"))
        artifact.add_file(str(checkpoint))
        artifact.add_file(str(root / "data/split-report.json"))
        artifact.add_file(str(root / "data/records.json"))
        artifact.add_file(str(donor_path))
        if (directory / "best.safetensors").exists():
            artifact.add_file(str(directory / "best.safetensors"))
        run.log_artifact(artifact).wait()
        return report
    except BaseException:
        failed = True
        raise
    finally:
        run.finish(exit_code=1 if failed else 0)


def _save_checkpoint(
    path,
    context,
    optimizer,
    settings,
    update,
    best,
    patience,
    model_id,
    frozen_hashes,
    training_seconds,
) -> None:
    temporary = path.with_suffix(".tmp")
    best_path = path.parent / "best.safetensors"
    donor_path = path.parent / "donor-validation.json"
    torch.save(
        {
            "head": context.head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "settings": settings,
            "update": update,
            "best": best,
            "patience": patience,
            "training_seconds": training_seconds,
            "best_sha256": _file_hash(best_path) if best_path.exists() else None,
            "donor_validation_sha256": _file_hash(donor_path) if donor_path.exists() else None,
            "model_id": model_id,
            "model_revision": get_model_spec(model_id).fast.revision,
            "model_pins": context.base_weight_sha256,
            "base_weight_sha256": context.base_weight_sha256,
            "frozen_hashes": frozen_hashes,
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        },
        temporary,
    )
    temporary.replace(path)
    # Persist resumable optimizer/RNG state during long Modal calls.
    import modal

    from .config import VOLUME_NAME

    modal.Volume.from_name(VOLUME_NAME).commit()


def evaluate_head(root: Path, model_id: str) -> dict:
    # Final evaluation is a separate stage so its labels never select a checkpoint.
    from .evaluation import evaluate_final

    return evaluate_final(root, model_id)
