"""Train an ESMFold2 confidence head on fresh rollouts of the frozen folding model.

Every update folds new targets with several diffusion samples, so the head sees how the quality of
samples of one target differs and never revisits a cached sample. The loss adds a within-target
ranking term to the native pLDDT and PAE cross-entropy. The schedule is a fixed warmup and cosine
over a planned number of updates, with an exponential moving average of the head weights and no
early stopping.
"""

from __future__ import annotations

import json
import math
import random
import time
import uuid
import numpy as np
import torch
import wandb

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from .labels import _masked_cross_entropy
from .ranking import expected_mean_plddt, expected_tm_scores, ranking_pairs, sample_ranking_loss
from .rollouts import INFERENCE_LOOPS, INFERENCE_SAMPLING_STEPS, Rollout, TargetStructure, fold, head_chunk_size
from .target_pool import load_positions
from .training import HeadContext


WANDB_ENTITY = "lhallee"
WANDB_PROJECT = "fastplms-confidence"
WANDB_GROUP = "esmfold2-confidence-v2"
VALIDATION_SEED = 29
PAIR_MARGIN_EVALUATION = 0.02
SELECTION_TOLERANCE = 0.02
EMA_HORIZON_FRACTION = 0.1
MAX_SKIPPED_TARGETS = 20
MAX_SKIPPED_FRACTION = 0.01
PROGRESS_LOG_UPDATES = 10


@dataclass(frozen=True)
class OnlineTrainingConfig:
    model_id: str
    samples_per_target: int = 4
    targets_per_update: int = 16
    learning_rate: float = 1e-4
    minimum_learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_updates: int = 300
    planned_updates: int = 2000
    ema_decay: float = 0.999
    gradient_clip: float = 1.0
    pae_weight: float = 1.0
    ranking_weight: float = 0.5
    ranking_margin: float = 0.01
    ranking_temperature: float = 0.05
    monomer_fraction: float = 0.5
    validation_interval_seconds: int = 2700
    checkpoint_interval_seconds: int = 1800
    maximum_seconds: int = 21 * 3600
    validation_limit: int | None = None  # use only the first validation targets, for probes
    num_loops: int = INFERENCE_LOOPS
    num_sampling_steps: int = INFERENCE_SAMPLING_STEPS
    seed: int = 17


def ema_decay_for(planned_updates: int) -> float:
    """Decay whose averaging horizon, 1 / (1 - decay), spans a tenth of the planned updates (at least 10)."""
    return 1.0 - 1.0 / max(10, round(EMA_HORIZON_FRACTION * planned_updates))


def learning_rate(update: int, config: OnlineTrainingConfig) -> float:
    """Linear warmup, then cosine decay to the minimum over the planned updates."""
    if update < config.warmup_updates:
        return config.learning_rate * (update + 1) / config.warmup_updates
    progress = min(1.0, (update - config.warmup_updates) / max(1, config.planned_updates - config.warmup_updates))
    return config.minimum_learning_rate + 0.5 * (config.learning_rate - config.minimum_learning_rate) * (
        1 + math.cos(math.pi * progress)
    )


class ExponentialMovingAverage:
    """Shadow copy of trainable parameters, updated after every optimizer step."""

    def __init__(self, module: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {name: parameter.detach().clone() for name, parameter in module.named_parameters() if parameter.requires_grad}

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        for name, parameter in module.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def swapped_into(self, module: nn.Module) -> dict[str, Tensor]:
        """Load the shadow weights into `module` and return the weights they replaced."""
        replaced = {name: parameter.detach().clone() for name, parameter in module.named_parameters() if name in self.shadow}
        for name, parameter in module.named_parameters():
            if name in self.shadow:
                parameter.copy_(self.shadow[name])
        return replaced


@torch.no_grad()
def restore(module: nn.Module, weights: Mapping[str, Tensor]) -> None:
    for name, parameter in module.named_parameters():
        if name in weights:
            parameter.copy_(weights[name])


class TargetSampler:
    """Draw training targets: first monomer or multi-chain, then a target in proportion to its weight."""

    def __init__(self, targets: Sequence[Mapping[str, object]], monomer_fraction: float, seed: int) -> None:
        self.groups = []
        for is_monomer in (True, False):
            members = [target for target in targets if (target["num_chains"] == 1) == is_monomer]
            weights = np.array([float(target["weight"]) for target in members])  # (n_group,)
            self.groups.append((members, weights / weights.sum()))
        self.monomer_fraction = monomer_fraction
        self.rng = np.random.default_rng(seed)

    def draw(self) -> Mapping[str, object]:
        members, probabilities = self.groups[0 if self.rng.random() < self.monomer_fraction else 1]
        return members[int(self.rng.choice(len(members), p=probabilities))]


def structure(pool_dir: Path, target: Mapping[str, object]) -> TargetStructure:
    return TargetStructure(
        target_id=str(target["target_id"]),
        sequences=tuple(target["sequences"]),  # type: ignore[arg-type]
        positions=load_positions(pool_dir, dict(target)),
    )


def head_output(context: HeadContext, inputs: Mapping[str, Tensor], x_pred: Tensor, sample: int) -> dict[str, Tensor]:
    """Run the head on one diffusion sample; `x_pred` holds all samples, shape (k, a, 3)."""
    context.head.set_chunk_size(head_chunk_size(inputs["token_attention_mask"].shape[-1]))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return context.head(**inputs, x_pred=x_pred[sample : sample + 1], num_diffusion_samples=1)


def sample_scores(output: Mapping[str, Tensor], targets: Mapping[str, Tensor], inputs: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
    """Differentiable mean pLDDT over labeled atoms and ipTM of one sample; each has shape (1,)."""
    plddt = expected_mean_plddt(output["plddt_logits"], targets["plddt_mask"][None])
    _, iptm = expected_tm_scores(output["pae_logits"], inputs["asym_id"], inputs["token_attention_mask"])
    return plddt, iptm


def cross_entropy(output: Mapping[str, Tensor], targets: Mapping[str, Tensor], pae_weight: float) -> tuple[Tensor, Tensor]:
    plddt = _masked_cross_entropy(output["plddt_logits"][0], targets["plddt_target"], targets["plddt_mask"])
    pae = _masked_cross_entropy(output["pae_logits"][0], targets["pae_target"], targets["pae_mask"])
    return plddt, pae * pae_weight


def target_step(context: HeadContext, rollout: Rollout, config: OnlineTrainingConfig) -> dict[str, float]:
    """Accumulate gradients for one target; samples pass through the head one at a time."""
    samples = rollout.x_pred.shape[0]
    plddt_pairs = ranking_pairs([quality["lddt"] for quality in rollout.quality], config.ranking_margin)
    iptm_pairs = (
        ranking_pairs([quality["true_iptm"] for quality in rollout.quality], config.ranking_margin)
        if rollout.num_chains > 1
        else []
    )
    # Samples of one target often differ by less than the margin, and then the ranking term has no
    # gradient, so the scoring pass for partner scores runs only when some pair is ranked.
    ranked = config.ranking_weight > 0 and bool(plddt_pairs or iptm_pairs)
    if ranked:
        with torch.no_grad():
            detached = [
                sample_scores(head_output(context, rollout.head_inputs, rollout.x_pred, k), rollout.targets[k], rollout.head_inputs)
                for k in range(samples)
            ]
        detached_plddt = torch.cat([scores[0] for scores in detached]).float()  # (k,)
        detached_iptm = torch.cat([scores[1] for scores in detached]).float()  # (k,)
    totals = {"plddt_ce": 0.0, "pae_ce": 0.0, "ranking": 0.0}
    for sample in range(samples):
        output = head_output(context, rollout.head_inputs, rollout.x_pred, sample)
        plddt_ce, pae_ce = cross_entropy(output, rollout.targets[sample], config.pae_weight)
        loss = (plddt_ce + pae_ce) / samples
        if ranked:
            plddt_score, iptm_score = sample_scores(output, rollout.targets[sample], rollout.head_inputs)
            ranking = sample_ranking_loss(sample, plddt_score[0], detached_plddt, plddt_pairs, config.ranking_temperature)
            if iptm_pairs:
                ranking = 0.5 * (ranking + sample_ranking_loss(sample, iptm_score[0], detached_iptm, iptm_pairs, config.ranking_temperature))
            loss = loss + config.ranking_weight * ranking
            totals["ranking"] += float(ranking.detach()) / 2  # each pair appears in two per-sample terms
        (loss / config.targets_per_update).backward()
        totals["plddt_ce"] += float(plddt_ce.detach()) / samples
        totals["pae_ce"] += float(pae_ce.detach()) / samples / config.pae_weight
    totals["ranking_pairs"] = float(len(plddt_pairs) + len(iptm_pairs))
    return totals


def selected_checkpoint(final: Mapping[str, float], best: Mapping[str, float]) -> str:
    """Pre-registered rule: keep the final EMA weights unless their validation total CE trails the best by over 2%."""
    return "final-ema" if final["total_ce"] <= (1 + SELECTION_TOLERANCE) * best["total_ce"] else "best-ema"


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 3:
        return float("nan")
    left_rank = np.argsort(np.argsort(left, kind="stable"), kind="stable").astype(float)
    right_rank = np.argsort(np.argsort(right, kind="stable"), kind="stable").astype(float)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def pairwise_accuracy(predicted: Sequence[float], true: Sequence[float], margin: float) -> tuple[int, int]:
    """Correctly ordered and total sample pairs whose true values differ by at least `margin`."""
    correct = total = 0
    for first in range(len(true)):
        for second in range(first + 1, len(true)):
            difference = true[first] - true[second]
            if abs(difference) < margin:
                continue
            total += 1
            correct += (predicted[first] - predicted[second]) * difference > 0
    return correct, total


@torch.no_grad()
def build_validation_cache(model: nn.Module, pool_dir: Path, targets: Sequence[Mapping[str, object]], cache_dir: Path, config: OnlineTrainingConfig, log: Callable[[str], None]) -> None:
    """Fold each validation target once and store head inputs, samples, and labels."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for index, target in enumerate(targets):
        path = cache_dir / f"{index:04d}.safetensors"
        if path.exists():
            continue
        rollout = fold(model, structure(pool_dir, target), config.samples_per_target, VALIDATION_SEED, config.num_loops, config.num_sampling_steps)
        tensors = {name: value.detach().contiguous().cpu() for name, value in rollout.head_inputs.items()}
        for name in ("z", "relative_position_encoding", "token_bonds_encoding"):
            # Pair tensors come from bf16 autocast, so bf16 storage is lossless; verify before halving.
            if torch.equal(tensors[name], tensors[name].bfloat16().float()):
                tensors[name] = tensors[name].bfloat16()
        tensors["x_pred"] = rollout.x_pred.contiguous().cpu()
        for sample, sample_targets in enumerate(rollout.targets):
            for name in ("plddt_target", "plddt_mask", "plddt_score", "pae_target", "pae_mask"):
                tensors[f"target/{sample}/{name}"] = sample_targets[name].contiguous().cpu()
        metadata = {"target_id": str(target["target_id"]), "num_chains": str(rollout.num_chains), "quality": json.dumps(rollout.quality)}
        save_file(tensors, str(path.with_suffix(".tmp")), metadata=metadata)
        path.with_suffix(".tmp").replace(path)
        if (index + 1) % 16 == 0:
            log(f"validation cache: {index + 1}/{len(targets)} targets")


@torch.no_grad()
def validate(context: HeadContext, cache_dir: Path, limit: int | None, pae_weight: float) -> dict[str, float]:
    """Score cached validation rollouts with the head's current weights."""
    from safetensors import safe_open

    context.head.eval()
    plddt_ce, pae_ce, target_predicted, target_true, correct, total, interface_correct, interface_total = [], [], [], [], 0, 0, 0, 0
    calibration_predicted, calibration_true = [], []
    for path in sorted(cache_dir.glob("*.safetensors"))[:limit]:
        with safe_open(str(path), framework="pt") as handle:
            metadata = handle.metadata()
        tensors = {name: value.cuda() for name, value in load_file(str(path)).items()}
        quality = json.loads(metadata["quality"])
        inputs = {name: (value.float() if value.dtype == torch.bfloat16 else value) for name, value in tensors.items() if "/" not in name and name != "x_pred"}
        predicted_plddt, predicted_iptm = [], []
        for sample in range(tensors["x_pred"].shape[0]):
            targets = {name.split("/")[-1]: value for name, value in tensors.items() if name.startswith(f"target/{sample}/")}
            output = head_output(context, inputs, tensors["x_pred"], sample)
            losses = cross_entropy(output, targets, 1.0)
            plddt_ce.append(float(losses[0]))
            pae_ce.append(float(losses[1]))
            plddt_score, iptm_score = sample_scores(output, targets, inputs)
            predicted_plddt.append(float(plddt_score[0]))
            predicted_iptm.append(float(iptm_score[0]))
            per_atom = (output["plddt_logits"][0].float().softmax(-1) * ((torch.arange(50, device="cuda") + 0.5) / 50)).sum(-1)
            mask = targets["plddt_mask"].bool()
            calibration_predicted.append(per_atom[mask].cpu())
            calibration_true.append(targets["plddt_score"][mask].float().cpu())
        true_lddt = [item["lddt"] for item in quality]
        target_predicted.append(float(np.mean(predicted_plddt)))
        target_true.append(float(np.mean(true_lddt)))
        pair_correct, pair_total = pairwise_accuracy(predicted_plddt, true_lddt, PAIR_MARGIN_EVALUATION)
        correct, total = correct + pair_correct, total + pair_total
        if int(metadata["num_chains"]) > 1:
            pair_correct, pair_total = pairwise_accuracy(predicted_iptm, [item["true_iptm"] for item in quality], PAIR_MARGIN_EVALUATION)
            interface_correct, interface_total = interface_correct + pair_correct, interface_total + pair_total
    predicted_atoms, true_atoms = torch.cat(calibration_predicted).numpy(), torch.cat(calibration_true).numpy()
    bins = np.clip((predicted_atoms * 10).astype(int), 0, 9)
    calibration = sum(
        (bins == index).mean() * abs(predicted_atoms[bins == index].mean() - true_atoms[bins == index].mean())
        for index in range(10)
        if (bins == index).any()
    )
    context.head.train()
    return {
        "plddt_ce": float(np.mean(plddt_ce)),
        "pae_ce": float(np.mean(pae_ce)),
        "total_ce": float(np.mean(plddt_ce) + pae_weight * np.mean(pae_ce)),
        "target_plddt_spearman": spearman(target_predicted, target_true),
        "within_target_plddt_accuracy": correct / total if total else float("nan"),
        "within_target_plddt_pairs": float(total),
        "within_target_iptm_accuracy": interface_correct / interface_total if interface_total else float("nan"),
        "within_target_iptm_pairs": float(interface_total),
        "calibration_error_10bin": float(calibration),
    }


def _rng_state(sampler: TargetSampler, rollout_rng: np.random.Generator) -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "sampler": sampler.rng.bit_generator.state,
        "rollout": rollout_rng.bit_generator.state,
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def train_online(
    model: nn.Module,
    pool_dir: Path,
    train_targets: Sequence[Mapping[str, object]],
    validation_targets: Sequence[Mapping[str, object]],
    run_dir: Path,
    config: OnlineTrainingConfig,
    log: Callable[[str], None],
    deadline_seconds: float,
) -> dict[str, object]:
    """Run or resume one training run in `run_dir` and return its final report.

    `config.maximum_seconds` bounds the training time of the whole run across resumes, while
    `deadline_seconds` bounds this invocation, so a resumed run still respects the GPU budget.
    """
    invocation_started = time.monotonic()
    if deadline_seconds <= 0:
        raise RuntimeError("no GPU time is left for this training invocation")
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    context = HeadContext(config.model_id)
    context.head.train()
    parameters = [parameter for parameter in context.head.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    ema = ExponentialMovingAverage(context.head, config.ema_decay)
    sampler = TargetSampler(train_targets, config.monomer_fraction, config.seed)
    rollout_rng = np.random.default_rng(config.seed + 1)
    update, elapsed, history, best = 0, 0.0, [], None

    checkpoint_path = run_dir / "last.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
        if checkpoint["config"] != asdict(config):
            raise ValueError("checkpoint settings differ from this run")
        context.head.load_state_dict(checkpoint["head"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        ema.shadow = checkpoint["ema"]
        update, elapsed, history, best = checkpoint["update"], checkpoint["elapsed"], checkpoint["history"], checkpoint["best"]
        states = checkpoint["rng"]
        random.setstate(states["python"])
        np.random.set_state(states["numpy_global"])
        sampler.rng.bit_generator.state = states["sampler"]
        rollout_rng.bit_generator.state = states["rollout"]
        torch.set_rng_state(states["torch"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in states["cuda"]])
        log(f"resumed at update {update} after {elapsed / 3600:.2f} h")

    identifier_path = run_dir / "wandb-id.txt"
    if not identifier_path.exists():
        identifier_path.write_text(uuid.uuid4().hex[:12])
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        id=identifier_path.read_text().strip(),
        name=run_dir.name,
        job_type="train",
        resume="allow",
        mode="online",
        config={**asdict(config), "train_targets": len(train_targets), "validation_targets": len(validation_targets)},
        dir=str(run_dir),
        settings=wandb.Settings(init_timeout=120, disable_git=True, console="off"),
    )
    if getattr(run.settings, "mode", "online") != "online":
        run.finish(exit_code=1)
        raise RuntimeError("confidence training requires an online W&B run")

    # Cached rollouts depend on the sampling settings, so the directory name records them.
    validation_dir = run_dir.parent / f"validation-cache-{config.num_loops}-loops-{config.num_sampling_steps}-steps-{config.samples_per_target}-samples"
    build_validation_cache(model, pool_dir, validation_targets[: config.validation_limit], validation_dir, config, log)

    started = time.monotonic() - elapsed
    last_validation = last_checkpoint = time.monotonic()

    def save_checkpoint() -> None:
        state = {
            "config": asdict(config),
            "head": context.head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema": ema.shadow,
            "update": update,
            "elapsed": time.monotonic() - started,
            "history": history,
            "best": best,
            "rng": _rng_state(sampler, rollout_rng),
        }
        torch.save(state, run_dir / "last.tmp")
        (run_dir / "last.tmp").replace(checkpoint_path)

    def run_validation() -> dict[str, float]:
        nonlocal best
        replaced = ema.swapped_into(context.head)
        metrics = validate(context, validation_dir, config.validation_limit, config.pae_weight)
        if best is None or metrics["total_ce"] < best["total_ce"]:
            best = {**metrics, "update": update}
            save_file({name: value.contiguous() for name, value in context.head.state_dict().items()}, str(run_dir / "best-ema.safetensors"))
        restore(context.head, replaced)
        history.append({"update": update, **metrics})
        run.log({f"validation/{name}": value for name, value in metrics.items()}, step=update)
        log(f"validation at update {update}: " + ", ".join(f"{name}={value:.4f}" for name, value in metrics.items()))
        return metrics

    if update == 0 and not history:
        run_validation()  # the donor head's starting point of the learning curve
        last_validation = time.monotonic()
    drawn = skipped = 0
    while (
        update < config.planned_updates
        and time.monotonic() - started < config.maximum_seconds
        and time.monotonic() - invocation_started < deadline_seconds
    ):
        update_started = time.monotonic()
        totals = {"plddt_ce": 0.0, "pae_ce": 0.0, "ranking": 0.0, "ranking_pairs": 0.0}
        tokens = fold_seconds = 0.0
        completed = 0
        while completed < config.targets_per_update:
            target = sampler.draw()
            drawn += 1
            fold_started = time.monotonic()
            try:
                rollout = fold(model, structure(pool_dir, target), config.samples_per_target, int(rollout_rng.integers(2**31)), config.num_loops, config.num_sampling_steps)
                fold_seconds += time.monotonic() - fold_started
                losses = target_step(context, rollout, config)
            except (torch.OutOfMemoryError, ValueError) as error:
                # One target that cannot be folded or labeled is replaced by the next draw, so a
                # rare failure does not end a day-long run; frequent failures still stop it. A
                # failure inside the head step keeps the samples' gradients accumulated before it.
                skipped += 1
                log(f"skipped {target['target_id']} ({target['num_tokens']} tokens): {type(error).__name__}: {error}")
                rollout = None
                torch.cuda.empty_cache()
                if skipped > max(MAX_SKIPPED_TARGETS, MAX_SKIPPED_FRACTION * drawn):
                    raise RuntimeError(f"skipped {skipped} of {drawn} drawn targets") from error
                continue
            completed += 1
            tokens += float(target["num_tokens"])
            for name, value in losses.items():
                totals[name] += value / config.targets_per_update
            del rollout
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(update, config)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        ema.update(context.head)
        update += 1
        seconds = time.monotonic() - update_started
        run.log(
            {
                **{f"train/{name}": value for name, value in totals.items()},
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/gradient_norm": gradient_norm,
                "train/update_seconds": seconds,
                "train/fold_fraction": fold_seconds / seconds,
                "train/tokens_per_target": tokens / config.targets_per_update,
                "train/elapsed_hours": (time.monotonic() - started) / 3600,
                "train/peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                "train/skipped_targets": skipped,
            },
            step=update,
        )
        if update % PROGRESS_LOG_UPDATES == 0:
            log(
                f"update {update}/{config.planned_updates}: {seconds:.1f} s, fold {fold_seconds / seconds:.0%}, "
                f"{tokens / config.targets_per_update:.0f} tokens per target, plddt_ce {totals['plddt_ce']:.3f}, "
                f"pae_ce {totals['pae_ce']:.3f}, ranking {totals['ranking']:.3f}, skipped {skipped}"
            )
        if time.monotonic() - last_validation >= config.validation_interval_seconds:
            run_validation()
            last_validation = time.monotonic()
        if time.monotonic() - last_checkpoint >= config.checkpoint_interval_seconds:
            save_checkpoint()
            last_checkpoint = time.monotonic()

    final = run_validation()
    replaced = ema.swapped_into(context.head)
    save_file({name: value.contiguous() for name, value in context.head.state_dict().items()}, str(run_dir / "final-ema.safetensors"))
    restore(context.head, replaced)
    save_checkpoint()
    selected = selected_checkpoint(final, best)
    if update >= config.planned_updates:
        stopped_by = "planned_updates"
    elif time.monotonic() - started >= config.maximum_seconds:
        stopped_by = "maximum_seconds"
    else:
        stopped_by = "gpu_budget_deadline"
    report = {
        "status": "complete",
        "model_id": config.model_id,
        "updates": update,
        "stopped_by": stopped_by,
        "elapsed_hours": (time.monotonic() - started) / 3600,
        "final_validation": final,
        "best_validation": best,
        "selected_checkpoint": f"{selected}.safetensors",
        "wandb_url": run.url,
        "config": asdict(config),
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    run.summary.update({"status": "complete", "selected_checkpoint": selected})
    run.finish()
    return report
