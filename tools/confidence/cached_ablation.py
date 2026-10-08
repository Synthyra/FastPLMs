"""Fine-tune ESMFold2 confidence heads on cached rollouts to test fixes for PAE texture.

v1 heads trained online: each update folded fresh targets, and folding took most of the time.
This ablation folds a fixed set of targets once per model, stores the head's inputs, the four
diffusion samples, their labels, and a teacher's PAE logits for each sample, then trains head
variants for a few epochs over that cache. The teacher is production `esmfold2`: its trunk
folds the same target, and its own confidence head scores the small model's samples.

Arms differ only in the PAE term of v1's objective (pLDDT CE + PAE CE + 0.5 ranking):

- ``control``: unchanged, so the other arms are compared at equal training.
- ``texture``: adds the mean absolute difference between the high-frequency parts of the
  expected PAE and of the true aligned error, so the head keeps real steps at domain and chain
  boundaries but loses cell-level noise the truth does not have.
- ``distill``: the PAE target mixes the one-hot truth with the teacher's probabilities.
- ``distill_texture``: both.

Layout under ``--root``: ``splits/`` and ``heads/`` (archived v1 inputs), ``atlasfold/`` and
``pool/`` (coordinates), ``cache/<model>/<split>/NNNN.safetensors``, ``runs/<model>/<arm>/``,
``evaluation/<model>/<name>.json``, and ``gpu-ledger.json``, which bounds GPU time in total.

Shape symbols: k samples, a atoms, t tokens, d_pair pair channels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import numpy as np
import torch
import torch.nn.functional as functional

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from scipy.stats import spearmanr
from torch import Tensor

from fastplms.digests import file_sha256
from fastplms.models.esmfold2.esmfold2_input_builder import ProteinInput, StructurePredictionInput
from fastplms.models.esmfold2.reproducibility import seed_context
from tools.stored_files import write_stored_json
from .labels import PAE_BINS, PAE_MAX_ANGSTROM, _masked_cross_entropy
from .online_training import (
    VALIDATION_SEED,
    ExponentialMovingAverage,
    TargetSampler,
    ema_decay_for,
    head_output,
    restore,
    sample_scores,
    structure,
)
from .ranking import ranking_pairs, sample_ranking_loss
from .rollouts import INFERENCE_LOOPS, INFERENCE_SAMPLING_STEPS, chain_label, fold, head_chunk_size
from .training import HeadContext
from .v2_analysis import PAIR_MARGIN_EVALUATION, pairwise_accuracy, spearman
from .v2_campaign import ARTIFACT_REPO, DATASET_REPO, DATASET_REVISION


EXPERIMENT_ID = "2026-09-28_esmfold2_confidence_smoothing"
WANDB_ENTITY = "lhallee"
WANDB_PROJECT = "fastplms-confidence"
CAMPAIGN = "confidence-v2/v2-reproduction-20260922"
# The v1 campaign's final commit on the artifact dataset; its split and heads are read from here.
CAMPAIGN_REVISION = "d7039a56e732e2de7685abefdbea57dfc6a9da13"
MODEL_IDS = ("esmfold2_300", "esmfold2_600")
ARMS = ("control", "texture", "distill", "distill_texture")
BUDGET_HOURS = 24.0
SAMPLES_PER_TARGET = 4
TARGETS_PER_UPDATE = 16
# The teacher's own diffusion sample is discarded, so it takes the fewest steps that run.
TEACHER_SAMPLING_STEPS = 2
SELECTION_TARGETS = 128  # the first half of validation, in v1's hash order, picks epochs
BLUR_SIGMA = 1.0  # residues; the scale of the roughness score in the texture investigation
BLUR_RADIUS = 4
TEXTURE_WEIGHT = 0.25  # the ablation's λ_t; a run with another weight is named for it
BOOTSTRAP_DRAWS = 1000
PAIR_SPEARMAN_LIMIT = 20_000
PAIR_FEATURES = ("residue_index", "asym_id", "sym_id", "entity_id", "token_index", "token_bonds")
HEAD_FEATURES = (
    "distogram_atom_idx",
    "token_attention_mask",
    "atom_to_token",
    "atom_attention_mask",
    "asym_id",
    "mol_type",
)
PAE_CENTERS = (torch.arange(PAE_BINS) + 0.5) * PAE_MAX_ANGSTROM / PAE_BINS  # (64,) Å


@dataclass(frozen=True)
class ArmConfig:
    model_id: str
    arm: str
    epochs: int = 3
    learning_rate: float = 5e-5  # half v1's peak: the head starts converged
    minimum_learning_rate: float = 5e-6
    warmup_fraction: float = 0.1
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    ranking_weight: float = 0.5
    ranking_margin: float = 0.01
    ranking_temperature: float = 0.05
    texture_weight: float = 0.0
    distill_fraction: float = 0.0
    seed: int = 17

    @property
    def run_name(self) -> str:
        """The arm, suffixed with its texture weight when that is not the ablation's."""
        if self.texture_weight in (0.0, TEXTURE_WEIGHT):
            return self.arm
        return f"{self.arm}-weight{self.texture_weight:g}"


def arm_config(model_id: str, arm: str, epochs: int, texture_weight: float, distill_fraction: float) -> ArmConfig:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {ARMS}")
    return ArmConfig(
        model_id=model_id,
        arm=arm,
        epochs=epochs,
        texture_weight=texture_weight if "texture" in arm else 0.0,
        distill_fraction=distill_fraction if "distill" in arm else 0.0,
    )


# GPU budget --------------------------------------------------------------------------------------


def ledger_hours(root: Path) -> float:
    path = root / "gpu-ledger.json"
    entries = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    return sum(float(entry["hours"]) for entry in entries)


@contextmanager
def gpu_stage(root: Path, name: str, max_hours: float) -> Iterator[float]:
    """Charge a GPU stage's wall time to the ledger, refusing one that could pass the budget."""
    spent = ledger_hours(root)
    if spent + max_hours > BUDGET_HOURS:
        raise RuntimeError(f"{name}: {spent:.2f} h spent, {max_hours} h more would pass {BUDGET_HOURS} h")
    started = time.monotonic()
    try:
        yield time.monotonic() + max_hours * 3600
    finally:
        path = root / "gpu-ledger.json"
        entries = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        entries.append({"stage": name, "hours": (time.monotonic() - started) / 3600, "ended": time.strftime("%Y-%m-%dT%H:%M:%S")})
        write_stored_json(path, entries, sort_keys=False)


def log(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


# Inputs ------------------------------------------------------------------------------------------


def prepare(root: Path, workers: int) -> None:
    """Fetch v1's split and heads, and rebuild the coordinate pool from pinned AtlasFold-Data."""
    from huggingface_hub import hf_hub_download, snapshot_download

    from .target_pool import build_pool
    from .target_splits import load_split

    for name in ("splits/targets.parquet", "splits/split-report.json"):
        source = hf_hub_download(ARTIFACT_REPO, f"{CAMPAIGN}/{name}", repo_type="dataset", revision=CAMPAIGN_REVISION)
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(Path(source).read_bytes())
    split = load_split(root / "splits")  # checks the table against its verified digest
    for model_id in MODEL_IDS:
        source = hf_hub_download(
            ARTIFACT_REPO, f"{CAMPAIGN}/runs/{model_id}/v2/final-ema.safetensors", repo_type="dataset", revision=CAMPAIGN_REVISION
        )
        (root / "heads").mkdir(parents=True, exist_ok=True)
        (root / "heads" / f"{model_id}-v1.safetensors").write_bytes(Path(source).read_bytes())
    if not (root / "pool" / "targets.parquet").exists():
        dataset_root = root / "atlasfold" / DATASET_REVISION
        snapshot_download(
            DATASET_REPO,
            repo_type="dataset",
            revision=DATASET_REVISION,
            allow_patterns=["data/rcsb/train-*.parquet", "data/rcsb_multimer/train-*.parquet"],
            local_dir=dataset_root,
            max_workers=16,
        )
        log(json.dumps(build_pool(dataset_root, root / "pool", workers)))
    import pyarrow.parquet as pq

    pool = {row["target_id"]: row for row in pq.read_table(root / "pool" / "targets.parquet").to_pylist()}
    mismatched = [
        target["target_id"]
        for target in split
        if target["split"] in ("train", "validation")
        and (
            target["target_id"] not in pool
            or any(pool[target["target_id"]][key] != target[key] for key in ("positions_file", "residue_offset", "num_tokens", "sequences"))
        )
    ]
    if mismatched:
        raise ValueError(f"{len(mismatched)} split targets differ from the rebuilt pool, e.g. {mismatched[:3]}")
    log(f"prepared: {len(split)} split targets match the rebuilt pool")


def split_targets(root: Path, split: str) -> list[dict[str, object]]:
    """One split's targets in v1's order, a hash of each target id."""
    from .target_splits import load_split

    chosen = [target for target in load_split(root / "splits") if target["split"] == split]
    return sorted(chosen, key=lambda target: hashlib.sha256(str(target["target_id"]).encode()).hexdigest())


def training_targets(root: Path, count: int, seed: int) -> list[dict[str, object]]:
    """Distinct training targets drawn with v1's sampler: half monomers, inverse cluster weights."""
    sampler = TargetSampler(split_targets(root, "train"), monomer_fraction=0.5, seed=seed)
    chosen: dict[str, dict[str, object]] = {}
    while len(chosen) < count:
        target = sampler.draw()
        chosen.setdefault(str(target["target_id"]), dict(target))
    return list(chosen.values())


# Caching -----------------------------------------------------------------------------------------


def load_teacher() -> torch.nn.Module:
    from .host import reference_model
    from .rollouts import use_fast_folding_kernels

    teacher = reference_model().to("cuda")  # type: ignore[attr-defined]
    use_fast_folding_kernels(teacher)  # type: ignore[arg-type]
    return teacher  # type: ignore[return-value]


@torch.no_grad()
def teacher_pae_logits(teacher: torch.nn.Module, sequences: Sequence[str], seed: int, student_features: Mapping[str, Tensor], x_pred: Tensor) -> Tensor:
    """Production's head on its own trunk states, scoring each student sample; returns (k, t, t, 64)."""
    # student_features: (...) one tensor per prepared feature name; x_pred: (k, a, 3)
    device = x_pred.device
    request = StructurePredictionInput(
        sequences=[ProteinInput(id=chain_label(index), sequence=sequence) for index, sequence in enumerate(sequences)]
    )
    features, _ = teacher.prepare_structure_input(request, seed=seed)  # type: ignore[operator]
    features = {name: value.to(device) for name, value in features.items()}
    for name in ("atom_to_token", "distogram_atom_idx", "token_attention_mask", "atom_attention_mask"):
        if not torch.equal(features[name], student_features[name]):
            raise ValueError(f"teacher and student tokenize {name} differently")
    tokens = features["token_attention_mask"].shape[-1]
    teacher.confidence_head.set_chunk_size(head_chunk_size(tokens))  # type: ignore[union-attr, operator]
    with torch.autocast("cuda", dtype=torch.bfloat16), seed_context(seed):
        output = teacher(
            **features,
            num_loops=INFERENCE_LOOPS,
            num_sampling_steps=TEACHER_SAMPLING_STEPS,
            num_diffusion_samples=1,
            output_hidden_states=True,
            return_dict=True,
        )
        inputs = {
            "s_inputs": output.hidden_states[0].detach().float(),  # (1, t, d_inputs)
            "z": output.hidden_states[1].detach().float(),  # (1, t, t, d_pair)
            **{name: features[name] for name in HEAD_FEATURES},
            "relative_position_encoding": teacher.rel_pos(  # type: ignore[operator]
                **{name: features[name] for name in PAIR_FEATURES if name != "token_bonds"}
            ).float(),  # (1, t, t, d_pair)
            "token_bonds_encoding": teacher.token_bonds(features["token_bonds"].float()).float(),  # type: ignore[operator]
        }
        logits = [
            teacher.confidence_head(**inputs, x_pred=x_pred[sample : sample + 1], num_diffusion_samples=1)["pae_logits"][0]  # type: ignore[operator]
            for sample in range(x_pred.shape[0])
        ]  # each (t, t, 64)
    return torch.stack(logits).bfloat16()  # (k, t, t, 64)


def cache_record(student: torch.nn.Module, teacher: torch.nn.Module, pool_dir: Path, target: Mapping[str, object], seed: int) -> tuple[dict[str, Tensor], dict[str, str]]:
    """Fold one target, label its samples, and add the teacher's logits; returns tensors and metadata."""
    rollout = fold(student, structure(pool_dir, target), SAMPLES_PER_TARGET, seed, INFERENCE_LOOPS, INFERENCE_SAMPLING_STEPS)
    request = StructurePredictionInput(
        sequences=[ProteinInput(id=chain_label(index), sequence=sequence) for index, sequence in enumerate(target["sequences"])]  # type: ignore[arg-type]
    )
    features, _ = student.prepare_structure_input(request, seed=seed)  # type: ignore[operator]
    features = {name: value.to(rollout.x_pred.device) for name, value in features.items()}
    teacher_logits = teacher_pae_logits(teacher, target["sequences"], seed, features, rollout.x_pred)  # type: ignore[arg-type]
    tensors: dict[str, Tensor] = {
        "s_inputs": rollout.head_inputs["s_inputs"],  # (1, t, d_inputs)
        "z": rollout.head_inputs["z"].bfloat16(),  # (1, t, t, d_pair); the head reads it under bf16 autocast
        "x_pred": rollout.x_pred,  # (k, a, 3)
        **{f"feature/{name}": features[name] for name in {*PAIR_FEATURES, *HEAD_FEATURES}},
    }
    tensors["feature/token_bonds"] = features["token_bonds"].to(torch.uint8)  # 0/1 bond map
    if not torch.equal(tensors["feature/token_bonds"].float(), features["token_bonds"].float()):
        raise ValueError("token_bonds is not a 0/1 map")
    for sample, labels in enumerate(rollout.targets):
        tensors[f"target/{sample}/plddt_target"] = labels["plddt_target"].to(torch.uint8)  # 50 bins
        tensors[f"target/{sample}/plddt_mask"] = labels["plddt_mask"]
        tensors[f"target/{sample}/plddt_score"] = labels["plddt_score"].float()
        tensors[f"target/{sample}/pae_target"] = labels["pae_target"].to(torch.uint8)  # 64 bins
        tensors[f"target/{sample}/pae_mask"] = labels["pae_mask"]
        tensors[f"target/{sample}/pae_error"] = labels["pae_error"].clamp(max=PAE_MAX_ANGSTROM).half()  # Å
        tensors[f"teacher/{sample}/pae_logits"] = teacher_logits[sample]  # (t, t, 64)
    metadata = {
        "target_id": str(target["target_id"]),
        "num_chains": str(rollout.num_chains),
        "num_tokens": str(target["num_tokens"]),
        "seed": str(seed),
        "quality": json.dumps(rollout.quality),
    }
    return {name: value.detach().contiguous().cpu() for name, value in tensors.items()}, metadata  # (...) one tensor per cache name, then string metadata


def build_cache(root: Path, model_id: str, split: str, count: int, max_hours: float, seed: int) -> None:
    """Fold and store `count` targets of a split (validation: all 256 in v1's order and seed)."""
    from .cache import load_folding_model
    from .rollouts import use_fast_folding_kernels

    targets = split_targets(root, "validation")[:count] if split == "validation" else training_targets(root, count, seed)
    cache_dir = root / "cache" / model_id / split
    write_stored_json(cache_dir / "targets.json", [str(target["target_id"]) for target in targets], sort_keys=False)
    seeds = np.random.default_rng(seed + 1).integers(2**31, size=len(targets))  # (targets,)
    with gpu_stage(root, f"cache-{model_id}-{split}", max_hours) as deadline:
        student = load_folding_model(model_id)
        use_fast_folding_kernels(student)
        teacher = load_teacher()
        skipped: list[str] = []
        started = time.monotonic()
        for index, target in enumerate(targets):
            path = cache_dir / f"{index:05d}.safetensors"
            if path.exists():
                continue
            if time.monotonic() > deadline:
                log(f"stopped at the stage limit after {index} targets")
                break
            target_seed = VALIDATION_SEED if split == "validation" else int(seeds[index])
            try:
                tensors, metadata = cache_record(student, teacher, root / "pool", target, target_seed)
            except (torch.OutOfMemoryError, ValueError) as error:
                skipped.append(f"{target['target_id']}: {type(error).__name__}: {error}")
                log(f"skipped {target['target_id']} ({target['num_tokens']} tokens): {error}")
                torch.cuda.empty_cache()
                continue
            save_file(tensors, str(path.with_suffix(".tmp")), metadata=metadata)
            path.with_suffix(".tmp").replace(path)
            if (index + 1) % 25 == 0:
                rate = (time.monotonic() - started) / (index + 1)
                log(f"{model_id} {split}: {index + 1}/{len(targets)} targets, {rate:.1f} s each")
        write_stored_json(cache_dir / "skipped.json", skipped, sort_keys=False)


# Training ----------------------------------------------------------------------------------------


def cached_files(root: Path, model_id: str, split: str) -> list[Path]:
    return sorted((root / "cache" / model_id / split).glob("*.safetensors"))


# Head inputs, samples, per-sample labels, per-sample teacher logits, and metadata, as `load_record` returns them.
CachedRecord = tuple[dict[str, Tensor], Tensor, list[dict[str, Tensor]], list[Tensor], dict[str, str]]


def load_record(context: HeadContext, path: Path) -> CachedRecord:
    """Head inputs, samples (k, a, 3), per-sample labels, per-sample teacher logits, and metadata."""
    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata()
    tensors = {name: value.cuda(non_blocking=True) for name, value in load_file(str(path)).items()}
    features = {name.removeprefix("feature/"): value for name, value in tensors.items() if name.startswith("feature/")}
    # `fold` computes both encodings under bf16 autocast; rebuilding them the same way reproduces
    # the head inputs of v1's online training exactly.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        relative_position = context.rel_pos(**{name: features[name] for name in PAIR_FEATURES if name != "token_bonds"})  # (1, t, t, d_pair)
        token_bonds = context.token_bonds(features["token_bonds"].float())  # (1, t, t, d_pair)
    inputs = {
        "s_inputs": tensors["s_inputs"].float(),
        "z": tensors["z"].float(),
        **{name: features[name] for name in HEAD_FEATURES},
        "relative_position_encoding": relative_position.float(),
        "token_bonds_encoding": token_bonds.float(),
    }
    samples = tensors["x_pred"].shape[0]
    labels = []
    for sample in range(samples):
        prefix = f"target/{sample}/"
        labels.append(
            {
                "plddt_target": tensors[prefix + "plddt_target"].long(),
                "plddt_mask": tensors[prefix + "plddt_mask"].bool(),
                "plddt_score": tensors[prefix + "plddt_score"],
                "pae_target": tensors[prefix + "pae_target"].long(),
                "pae_mask": tensors[prefix + "pae_mask"].bool(),
                "pae_error": tensors[prefix + "pae_error"].float(),
            }
        )
    teacher = [tensors[f"teacher/{sample}/pae_logits"] for sample in range(samples)]  # each (t, t, 64)
    return inputs, tensors["x_pred"], labels, teacher, metadata


def blur_kernel(device: torch.device) -> Tensor:
    offsets = torch.arange(-BLUR_RADIUS, BLUR_RADIUS + 1, device=device, dtype=torch.float32)  # (9,)
    kernel = torch.exp(-0.5 * (offsets / BLUR_SIGMA) ** 2)  # (9,)
    return kernel / kernel.sum()  # (9,)


def high_frequency(values: Tensor, mask: Tensor) -> Tensor:
    """A (t, t) map minus its Gaussian blur, averaging only over masked-in cells."""
    # values, mask: (t, t)
    kernel = blur_kernel(values.device)  # (9,)
    weights = mask.float()[None, None]  # (1, 1, t, t)
    stacked = torch.cat([values[None, None] * weights, weights])  # (2, 1, t, t): weighted values, weights
    rows = functional.conv2d(stacked, kernel.view(1, 1, -1, 1), padding=(BLUR_RADIUS, 0))  # (2, 1, t, t)
    both = functional.conv2d(rows, kernel.view(1, 1, 1, -1), padding=(0, BLUR_RADIUS))  # (2, 1, t, t)
    smooth = both[0, 0] / both[1, 0].clamp(min=1e-6)  # (t, t)
    return values - smooth  # (t, t)


def roughness(values: Tensor, mask: Tensor) -> float:
    """Fraction of a map's variance, over masked-in cells, left after the Gaussian blur."""
    # values, mask: (t, t)
    if int(mask.sum()) < 2:
        return float("nan")
    high = high_frequency(values.float(), mask)[mask]  # (valid cells,)
    centered = values.float()[mask] - values.float()[mask].mean()  # (valid cells,)
    total = float(centered.square().sum())
    return float(high.square().sum()) / total if total > 0 else float("nan")


def expected_pae(pae_logits: Tensor) -> Tensor:
    """Expected aligned error in Å from (..., 64) logits."""
    # pae_logits: (..., 64)
    return (pae_logits.float().softmax(-1) * PAE_CENTERS.to(pae_logits.device)).sum(-1)  # (...)


def pae_loss(logits: Tensor, labels: Mapping[str, Tensor], teacher_logits: Tensor, config: ArmConfig) -> tuple[Tensor, Tensor, Tensor]:
    """The arm's PAE loss, the plain PAE CE for logging, and the texture term; each ()."""
    # logits, teacher_logits: (t, t, 64); labels: (...) one tensor per label name; pae_target, pae_mask, pae_error (t, t)
    mask = labels["pae_mask"]  # (t, t)
    hard = _masked_cross_entropy(logits, labels["pae_target"], mask)  # ()
    if config.distill_fraction > 0 and bool(mask.any()):
        log_probs = logits.float()[mask].log_softmax(-1)  # (valid pairs, 64)
        target = functional.one_hot(labels["pae_target"][mask], PAE_BINS).float()  # (valid pairs, 64)
        target = (1 - config.distill_fraction) * target + config.distill_fraction * teacher_logits.float()[mask].softmax(-1)
        loss = -(target * log_probs).sum(-1).mean()  # ()
    else:
        loss = hard
    texture = logits.sum() * 0.0  # ()
    if config.texture_weight > 0 and int(mask.sum()) > 1:
        predicted = high_frequency(expected_pae(logits), mask)  # (t, t)
        true = high_frequency(labels["pae_error"], mask)  # (t, t)
        texture = (predicted - true)[mask].abs().mean()  # ()
        loss = loss + config.texture_weight * texture
    return loss, hard, texture  # (), (), ()


def target_loss(context: HeadContext, record: CachedRecord, config: ArmConfig) -> dict[str, float]:
    """Accumulate one target's gradients, one sample at a time, as v1's `target_step` does."""
    inputs, x_pred, labels, teacher, metadata = record
    samples = x_pred.shape[0]
    quality = json.loads(metadata["quality"])
    plddt_pairs = ranking_pairs([item["lddt"] for item in quality], config.ranking_margin)
    iptm_pairs = ranking_pairs([item["true_iptm"] for item in quality], config.ranking_margin) if int(metadata["num_chains"]) > 1 else []
    ranked = config.ranking_weight > 0 and bool(plddt_pairs or iptm_pairs)
    if ranked:
        with torch.no_grad():
            detached = [sample_scores(head_output(context, inputs, x_pred, k), labels[k], inputs) for k in range(samples)]
        detached_plddt = torch.cat([scores[0] for scores in detached]).float()  # (k,)
        detached_iptm = torch.cat([scores[1] for scores in detached]).float()  # (k,)
    totals = {"plddt_ce": 0.0, "pae_ce": 0.0, "pae_loss": 0.0, "texture": 0.0, "ranking": 0.0}
    for sample in range(samples):
        output = head_output(context, inputs, x_pred, sample)
        plddt_ce = _masked_cross_entropy(output["plddt_logits"][0], labels[sample]["plddt_target"], labels[sample]["plddt_mask"])  # ()
        pae, hard, texture = pae_loss(output["pae_logits"][0], labels[sample], teacher[sample], config)
        loss = (plddt_ce + pae) / samples  # ()
        if ranked:
            plddt_score, iptm_score = sample_scores(output, labels[sample], inputs)  # each (1,)
            ranking = sample_ranking_loss(sample, plddt_score[0], detached_plddt, plddt_pairs, config.ranking_temperature)
            if iptm_pairs:
                ranking = 0.5 * (ranking + sample_ranking_loss(sample, iptm_score[0], detached_iptm, iptm_pairs, config.ranking_temperature))
            loss = loss + config.ranking_weight * ranking
            totals["ranking"] += float(ranking.detach()) / 2
        (loss / TARGETS_PER_UPDATE).backward()
        totals["plddt_ce"] += float(plddt_ce.detach()) / samples
        totals["pae_ce"] += float(hard.detach()) / samples
        totals["pae_loss"] += float(pae.detach()) / samples
        totals["texture"] += float(texture.detach()) / samples
    return totals


def learning_rate(update: int, total: int, config: ArmConfig) -> float:
    warmup = max(1, round(config.warmup_fraction * total))
    if update < warmup:
        return config.learning_rate * (update + 1) / warmup
    progress = min(1.0, (update - warmup) / max(1, total - warmup))
    return config.minimum_learning_rate + 0.5 * (config.learning_rate - config.minimum_learning_rate) * (1 + math.cos(math.pi * progress))


def load_head(context: HeadContext, path: Path) -> None:
    context.head.load_state_dict(load_file(str(path), device="cuda"), strict=True)


def train_arm(root: Path, config: ArmConfig, max_hours: float) -> dict[str, object]:
    """Fine-tune the v1 head on the training cache; keep each epoch's EMA weights and scores.

    After each epoch the optimizer, EMA, and sampler state go to `resume.pt`, so a run killed by a machine
    restart continues from its last finished epoch, in the same W&B run, instead of starting over.
    """
    import wandb

    run_dir = root / "runs" / config.model_id / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    files = cached_files(root, config.model_id, "train")
    selection = cached_files(root, config.model_id, "validation")[:SELECTION_TARGETS]
    updates_per_epoch = len(files) // TARGETS_PER_UPDATE
    total = updates_per_epoch * config.epochs
    torch.manual_seed(config.seed)
    context = HeadContext(config.model_id)
    load_head(context, root / "heads" / f"{config.model_id}-v1.safetensors")
    context.head.train()
    parameters = [parameter for parameter in context.head.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    ema = ExponentialMovingAverage(context.head, ema_decay_for(total))
    history = []
    rng = np.random.default_rng(config.seed)
    update = 0
    resume_path = run_dir / "resume.pt"
    saved = torch.load(resume_path, map_location="cuda", weights_only=False) if resume_path.exists() else None
    if saved is not None:
        context.head.load_state_dict(saved["head"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        ema.shadow = saved["ema"]
        rng.bit_generator.state = saved["rng"]
        history, update = saved["history"], saved["update"]
        log(f"{config.run_name}: resuming after epoch {len(history)}, update {update}")
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=EXPERIMENT_ID,
        name=f"{config.model_id}-{config.run_name}",
        id=saved["wandb_id"] if saved is not None else None,
        resume="allow" if saved is not None else None,
        job_type="ablation",
        config={**asdict(config), "train_targets": len(files), "updates": total},
        dir=str(run_dir),
        settings=wandb.Settings(init_timeout=120, disable_git=True, console="off"),
    )
    with gpu_stage(root, f"train-{config.model_id}-{config.run_name}", max_hours) as deadline:
        for epoch in range(len(history), config.epochs):
            order = rng.permutation(len(files))[: updates_per_epoch * TARGETS_PER_UPDATE]  # (targets used,)
            epoch_rng_state = rng.bit_generator.state
            for start in range(0, len(order), TARGETS_PER_UPDATE):
                if time.monotonic() > deadline:
                    raise RuntimeError(f"{config.run_name}: stage limit reached at update {update}")
                started = time.monotonic()
                totals: dict[str, float] = {}
                for index in order[start : start + TARGETS_PER_UPDATE]:
                    for name, value in target_loss(context, load_record(context, files[int(index)]), config).items():
                        totals[name] = totals.get(name, 0.0) + value / TARGETS_PER_UPDATE
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate(update, total, config)
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(context.head)
                update += 1
                run.log(
                    {**{f"train/{name}": value for name, value in totals.items()},
                     "train/learning_rate": optimizer.param_groups[0]["lr"], "train/gradient_norm": gradient_norm,
                     "train/update_seconds": time.monotonic() - started, "train/epoch": epoch},
                    step=update,
                )
                if update % 10 == 0:
                    log(f"{config.run_name} update {update}/{total}: " + ", ".join(f"{name} {value:.3f}" for name, value in totals.items()))
            replaced = ema.swapped_into(context.head)
            weights = run_dir / f"epoch-{epoch + 1}-ema.safetensors"
            save_file({name: value.contiguous() for name, value in context.head.state_dict().items()}, str(weights))
            scores = summarize(score_records(context, selection))
            restore(context.head, replaced)
            history.append({"epoch": epoch + 1, "update": update, **scores})
            run.log({f"selection/{name}": value for name, value in scores.items()}, step=update)
            log(f"{config.run_name} epoch {epoch + 1}: " + ", ".join(f"{name} {value:.4f}" for name, value in scores.items()))
            write_stored_json(run_dir / "history.json", history, sort_keys=False)
            torch.save(
                {"head": context.head.state_dict(), "optimizer": optimizer.state_dict(), "ema": ema.shadow, "rng": epoch_rng_state,
                 "history": history, "update": update, "wandb_id": run.id},
                resume_path.with_suffix(".tmp"),
            )
            resume_path.with_suffix(".tmp").replace(resume_path)
    best = min(history, key=lambda entry: entry["total_ce"])
    report = {"config": asdict(config), "updates": update, "train_targets": len(files), "selected_epoch": best["epoch"], "history": history, "wandb_url": run.url}
    write_stored_json(run_dir / "report.json", report, sort_keys=False)
    run.summary.update({"selected_epoch": best["epoch"]})
    run.finish()
    return report


# Evaluation --------------------------------------------------------------------------------------


@torch.no_grad()
def score_records(context: HeadContext, files: Sequence[Path]) -> list[dict[str, object]]:
    """Per-target scores of the head's current weights on cached rollouts."""
    context.head.eval()
    records = []
    for path in files:
        inputs, x_pred, labels, teacher, metadata = load_record(context, path)
        quality = json.loads(metadata["quality"])
        per_sample: dict[str, list[float]] = {}
        plddt_scores, iptm_scores, atoms_predicted, atoms_true = [], [], [], []
        for sample in range(x_pred.shape[0]):
            output = head_output(context, inputs, x_pred, sample)
            mask = labels[sample]["pae_mask"]  # (t, t)
            predicted = expected_pae(output["pae_logits"][0])  # (t, t)
            teacher_expected = expected_pae(teacher[sample])  # (t, t)
            true = labels[sample]["pae_error"]  # (t, t)
            values = {
                "plddt_ce": float(_masked_cross_entropy(output["plddt_logits"][0], labels[sample]["plddt_target"], labels[sample]["plddt_mask"])),
                "pae_ce": float(_masked_cross_entropy(output["pae_logits"][0], labels[sample]["pae_target"], mask)),
                "teacher_pae_ce": float(_masked_cross_entropy(teacher[sample].float(), labels[sample]["pae_target"], mask)),
                "roughness": roughness(predicted, mask),
                "teacher_roughness": roughness(teacher_expected, mask),
                "true_roughness": roughness(true, mask),
                "pae_bias": float((predicted - true)[mask].mean()) if bool(mask.any()) else float("nan"),
            }
            if int(mask.sum()) >= 10:
                chosen = torch.randperm(int(mask.sum()), generator=torch.Generator().manual_seed(0))[:PAIR_SPEARMAN_LIMIT]  # (pairs,)
                values["pair_spearman"] = float(spearmanr(predicted[mask].cpu()[chosen], true[mask].cpu()[chosen]).statistic)
                values["teacher_pair_spearman"] = float(spearmanr(teacher_expected[mask].cpu()[chosen], true[mask].cpu()[chosen]).statistic)
            for name, value in values.items():
                per_sample.setdefault(name, []).append(value)
            plddt_score, iptm_score = sample_scores(output, labels[sample], inputs)  # each (1,)
            plddt_scores.append(float(plddt_score[0]))
            iptm_scores.append(float(iptm_score[0]))
            per_atom = (output["plddt_logits"][0].float().softmax(-1) * ((torch.arange(50, device="cuda") + 0.5) / 50)).sum(-1)  # (a,)
            atom_mask = labels[sample]["plddt_mask"]  # (a,)
            atoms_predicted.extend(per_atom[atom_mask].cpu().tolist())
            atoms_true.extend(labels[sample]["plddt_score"][atom_mask].cpu().tolist())
        true_lddt = [item["lddt"] for item in quality]
        correct, pairs = pairwise_accuracy(plddt_scores, true_lddt, PAIR_MARGIN_EVALUATION)
        record = {
            "target_id": metadata["target_id"],
            "num_chains": int(metadata["num_chains"]),
            **{name: float(np.nanmean(values)) for name, values in per_sample.items()},
            "predicted_plddt": float(np.mean(plddt_scores)),
            "true_lddt": float(np.mean(true_lddt)),
            "ranking_correct": correct,
            "ranking_pairs": pairs,
            "atoms_predicted": atoms_predicted,
            "atoms_true": atoms_true,
        }
        if record["num_chains"] > 1:
            record["iptm_correct"], record["iptm_pairs"] = pairwise_accuracy(iptm_scores, [item["true_iptm"] for item in quality], PAIR_MARGIN_EVALUATION)
            # ipTM is a maximum over residue rows, so a noisy PAE map is expected to inflate it.
            record["predicted_iptm"] = float(np.mean(iptm_scores))
            record["true_iptm"] = float(np.mean([item["true_iptm"] for item in quality]))
            record["iptm_bias"] = record["predicted_iptm"] - record["true_iptm"]
        records.append(record)
    context.head.train()
    return records


def calibration_error(predicted: np.ndarray, true: np.ndarray) -> float:
    # predicted, true: (atoms,)
    bins = np.clip((predicted * 10).astype(int), 0, 9)  # (atoms,)
    return float(sum((bins == index).mean() * abs(predicted[bins == index].mean() - true[bins == index].mean()) for index in range(10) if (bins == index).any()))


MEAN_METRICS = ("plddt_ce", "pae_ce", "roughness", "pair_spearman", "pae_bias", "teacher_pae_ce", "teacher_roughness", "true_roughness", "teacher_pair_spearman", "iptm_bias")


def summarize(records: Sequence[Mapping[str, object]]) -> dict[str, float]:
    """v1's validation block plus the texture scores, over a set of per-target records."""
    summary = {name: float(np.nanmean([record[name] for record in records if name in record])) for name in MEAN_METRICS}
    summary["total_ce"] = summary["plddt_ce"] + summary["pae_ce"]
    pairs = sum(int(record["ranking_pairs"]) for record in records)
    summary["within_target_plddt_accuracy"] = sum(int(record["ranking_correct"]) for record in records) / pairs if pairs else float("nan")
    interface = sum(int(record.get("iptm_pairs", 0)) for record in records)
    summary["within_target_iptm_accuracy"] = sum(int(record.get("iptm_correct", 0)) for record in records) / interface if interface else float("nan")
    summary["target_plddt_spearman"] = spearman([record["predicted_plddt"] for record in records], [record["true_lddt"] for record in records])
    complexes = [record for record in records if "predicted_iptm" in record]
    summary["target_iptm_spearman"] = spearman([record["predicted_iptm"] for record in complexes], [record["true_iptm"] for record in complexes]) if len(complexes) > 2 else float("nan")
    summary["calibration_error_10bin"] = calibration_error(
        np.concatenate([np.asarray(record["atoms_predicted"]) for record in records]),
        np.concatenate([np.asarray(record["atoms_true"]) for record in records]),
    )
    return summary


def evaluate(root: Path, model_id: str, name: str, weights: Path, max_hours: float) -> None:
    """Score one head on all validation targets and store per-target records for the analysis."""
    files = cached_files(root, model_id, "validation")
    with gpu_stage(root, f"evaluate-{model_id}-{name}", max_hours):
        context = HeadContext(model_id)
        load_head(context, weights)
        records = score_records(context, files)
    output = {
        "model_id": model_id,
        "name": name,
        "weights": str(weights),
        "weights_sha256": file_sha256(weights),
        "selection": summarize(records[:SELECTION_TARGETS]),
        "report": summarize(records[SELECTION_TARGETS:]),
        "records": [{key: value for key, value in record.items() if not key.startswith("atoms_")} for record in records],
        "atoms": [{"predicted": record["atoms_predicted"], "true": record["atoms_true"]} for record in records],
    }
    write_stored_json(root / "evaluation" / model_id / f"{name}.json", output, sort_keys=False)
    log(json.dumps({"name": name, "report": output["report"]}, indent=1))


def bootstrap(root: Path, model_id: str, baseline: str, names: Sequence[str]) -> dict[str, object]:
    """Paired target-bootstrap intervals of each head minus a baseline, on the reporting half."""
    loaded = {name: json.loads((root / "evaluation" / model_id / f"{name}.json").read_text(encoding="utf-8")) for name in (baseline, *names)}

    def records(name: str) -> list[dict[str, object]]:
        entry = loaded[name]
        combined = [{**record, "atoms_predicted": atoms["predicted"], "atoms_true": atoms["true"]} for record, atoms in zip(entry["records"], entry["atoms"], strict=True)]
        return combined[SELECTION_TARGETS:]

    base = records(baseline)
    rng = np.random.default_rng(0)
    draws = [rng.integers(len(base), size=len(base)) for _ in range(BOOTSTRAP_DRAWS)]  # each (targets,)
    summary: dict[str, object] = {"baseline": baseline, "targets": len(base), "estimates": {}}
    for name in names:
        other = records(name)
        point = {metric: summarize(other)[metric] - summarize(base)[metric] for metric in summarize(base)}
        samples: dict[str, list[float]] = {metric: [] for metric in point}
        for draw in draws:
            first, second = summarize([base[i] for i in draw]), summarize([other[i] for i in draw])
            for metric in point:
                samples[metric].append(second[metric] - first[metric])
        summary["estimates"][name] = {  # type: ignore[index]
            metric: {"difference": point[metric], "low": float(np.nanpercentile(samples[metric], 2.5)), "high": float(np.nanpercentile(samples[metric], 97.5))}
            for metric in point
        }
    write_stored_json(root / "evaluation" / model_id / f"bootstrap-vs-{baseline}.json", summary, sort_keys=False)
    return summary


def download() -> None:
    """Fetch every checkpoint on the CPU, so downloads are not charged as GPU time."""
    from .cache import load_folding_model
    from .host import reference_model

    for model_id in MODEL_IDS:
        load_folding_model(model_id, device="cpu")
        HeadContext(model_id, device="cpu")
    reference_model()
    log("student bases, donor head, and production esmfold2 are cached")


def pipeline(root: Path, model_id: str, train_count: int, arms: Sequence[str], epochs: int, texture_weight: float, distill_fraction: float, seed: int, train_hours: float = 3.0) -> None:
    """One model's whole ablation, resumable: caches, v1's scores, each arm, and the bootstrap."""
    validation_count = len(split_targets(root, "validation"))
    if len(cached_files(root, model_id, "validation")) < validation_count:
        build_cache(root, model_id, "validation", validation_count, 1.5, seed)
    if len(cached_files(root, model_id, "train")) < train_count:
        build_cache(root, model_id, "train", train_count, train_count * 12 / 3600 + 0.5, seed)
    evaluation = root / "evaluation" / model_id
    if not (evaluation / "v1.json").exists():
        evaluate(root, model_id, "v1", root / "heads" / f"{model_id}-v1.safetensors", 0.5)
    for arm in arms:
        config = arm_config(model_id, arm, epochs, texture_weight, distill_fraction)
        report_path = root / "runs" / model_id / config.run_name / "report.json"
        if not report_path.exists():
            train_arm(root, config, train_hours)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not (evaluation / f"{config.run_name}.json").exists():
            weights = root / "runs" / model_id / config.run_name / f"epoch-{report['selected_epoch']}-ema.safetensors"
            evaluate(root, model_id, config.run_name, weights, 0.5)
    # The bootstrap runs on the CPU for tens of minutes; hand the GPU back first.
    torch.cuda.empty_cache()
    # Every head evaluated so far, so a later pipeline call extends the comparison, not replaces it.
    heads = sorted(path.stem for path in evaluation.glob("*.json") if path.stem != "v1" and not path.stem.startswith(("bootstrap-", "maps")))
    bootstrap(root, model_id, "v1", heads)
    if "control" in heads:
        bootstrap(root, model_id, "control", [head for head in heads if head != "control"])
    log(f"{model_id} pipeline complete; {ledger_hours(root):.2f} GPU hours spent")


MAP_SEQUENCES = {
    "ci2": "MKTEWPELVGKSVEEAKKVILQDKPEAQIIVLPVGTIVTMEYRIDRVRLFVDKLDNIAEVPRVG",
    "ubiquitin": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
}


@torch.no_grad()
def head_maps(root: Path, model_id: str, heads: Mapping[str, Path], max_hours: float) -> None:
    """Expected PAE of several heads on the same samples of the texture investigation's sequences."""
    import random

    from .cache import load_folding_model
    from .rollouts import use_fast_folding_kernels

    # CI2 shuffled with random.Random(0), the unfoldable control of the texture investigation.
    shuffled = list(MAP_SEQUENCES["ci2"])
    random.Random(0).shuffle(shuffled)
    sequences = {**MAP_SEQUENCES, "ci2_shuffled": "".join(shuffled)}
    maps: dict[str, np.ndarray] = {}
    scores: dict[str, dict[str, float]] = {}
    with gpu_stage(root, f"maps-{model_id}", max_hours):
        student = load_folding_model(model_id)
        use_fast_folding_kernels(student)
        context = HeadContext(model_id)
        for name, sequence in sequences.items():
            request = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=sequence)])
            features, _ = student.prepare_structure_input(request, seed=0)  # type: ignore[operator]
            features = {key: value.to("cuda") for key, value in features.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16), seed_context(0):
                output = student(**features, num_loops=INFERENCE_LOOPS, num_sampling_steps=INFERENCE_SAMPLING_STEPS,
                                 num_diffusion_samples=SAMPLES_PER_TARGET, output_hidden_states=True, return_dict=True)
                inputs = {
                    "s_inputs": output.hidden_states[0].float(),
                    "z": output.hidden_states[1].float(),
                    **{key: features[key] for key in HEAD_FEATURES},
                    "relative_position_encoding": student.rel_pos(**{key: features[key] for key in PAIR_FEATURES if key != "token_bonds"}).float(),
                    "token_bonds_encoding": student.token_bonds(features["token_bonds"].float()).float(),
                }
            x_pred = output["sample_atom_coords"].reshape(SAMPLES_PER_TARGET, -1, 3).float()  # (k, a, 3)
            mask = torch.ones(len(sequence), len(sequence), dtype=torch.bool, device="cuda")  # (t, t)
            for head, weights in heads.items():
                load_head(context, weights)
                context.head.eval()
                expected = torch.stack([expected_pae(head_output(context, inputs, x_pred, k)["pae_logits"][0]) for k in range(SAMPLES_PER_TARGET)])  # (k, t, t)
                maps[f"{head}|{name}"] = expected.cpu().numpy()
                scores[f"{head}|{name}"] = {
                    "roughness": float(np.mean([roughness(expected[k], mask) for k in range(SAMPLES_PER_TARGET)])),
                    "mean_pae": float(expected.mean()),
                }
                log(f"{head:24s} {name:13s} roughness {scores[f'{head}|{name}']['roughness']:.3f} mean PAE {scores[f'{head}|{name}']['mean_pae']:.2f}")
    np.savez_compressed(root / "evaluation" / model_id / "maps.npz", **maps)
    write_stored_json(root / "evaluation" / model_id / "maps.json", scores, sort_keys=False)


def probe(root: Path, model_id: str) -> None:
    """Time one fold and teacher pass, check pair encodings rebuild exactly, and size a record."""
    from .cache import load_folding_model
    from .rollouts import use_fast_folding_kernels

    with gpu_stage(root, f"probe-{model_id}", 0.5):
        student = load_folding_model(model_id)
        use_fast_folding_kernels(student)
        teacher = load_teacher()
        context = HeadContext(model_id)
        for target in split_targets(root, "validation")[:2]:
            started = time.monotonic()
            rollout = fold(student, structure(root / "pool", target), SAMPLES_PER_TARGET, VALIDATION_SEED)
            folded = time.monotonic() - started
            tensors, metadata = cache_record(student, teacher, root / "pool", target, VALIDATION_SEED)
            total = time.monotonic() - started
            path = root / "probe.safetensors"
            save_file(tensors, str(path), metadata=metadata)
            inputs = load_record(context, path)[0]
            rebuilt, bonds = inputs["relative_position_encoding"], inputs["token_bonds_encoding"]
            size = path.stat().st_size / 2**20
            path.unlink()
            log(
                f"{metadata['target_id']} {target['num_tokens']} tokens: fold {folded:.1f} s, fold+record {total:.1f} s, {size:.0f} MiB; "
                f"rel_pos equal {torch.equal(rebuilt, rollout.head_inputs['relative_position_encoding'])} "
                f"max diff {float((rebuilt - rollout.head_inputs['relative_position_encoding']).abs().max()):.2e}, "
                f"token_bonds equal {torch.equal(bonds, rollout.head_inputs['token_bonds_encoding'])}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("prepare", "download", "probe", "cache", "train", "evaluate", "bootstrap", "pipeline", "maps", "ledger"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_IDS)
    parser.add_argument("--split", choices=("train", "validation"))
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--arms", nargs="*", choices=ARMS, default=["control", "texture", "distill"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--texture-weight", type=float, default=TEXTURE_WEIGHT)
    parser.add_argument("--distill-fraction", type=float, default=0.5)
    parser.add_argument("--name")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--baseline", default="v1")
    parser.add_argument("--names", nargs="*", default=[])
    parser.add_argument("--max-hours", type=float, default=1.0)
    parser.add_argument("--train-hours", type=float, default=3.0, help="Per-arm training limit of the pipeline stage.")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    arguments = parser.parse_args()
    root = arguments.root
    root.mkdir(parents=True, exist_ok=True)
    stages: dict[str, Callable[[], object]] = {
        "prepare": lambda: prepare(root, arguments.workers),
        "download": download,
        # --names takes name=path pairs, a path relative to --root.
        "maps": lambda: head_maps(root, arguments.model, {item.split("=", 1)[0]: root / item.split("=", 1)[1] for item in arguments.names}, arguments.max_hours),
        "pipeline": lambda: pipeline(
            root, arguments.model, arguments.count, arguments.arms, arguments.epochs,
            arguments.texture_weight, arguments.distill_fraction, arguments.seed, arguments.train_hours,
        ),
        "probe": lambda: probe(root, arguments.model),
        "cache": lambda: build_cache(root, arguments.model, arguments.split, arguments.count, arguments.max_hours, arguments.seed),
        "train": lambda: train_arm(root, arm_config(arguments.model, arguments.arm, arguments.epochs, arguments.texture_weight, arguments.distill_fraction), arguments.max_hours),
        "evaluate": lambda: evaluate(root, arguments.model, arguments.name, arguments.weights, arguments.max_hours),
        "bootstrap": lambda: log(json.dumps(bootstrap(root, arguments.model, arguments.baseline, arguments.names), indent=1)),
        "ledger": lambda: log(f"{ledger_hours(root):.2f} GPU hours spent of {BUDGET_HOURS}"),
    }
    stages[arguments.stage]()


if __name__ == "__main__":
    main()
