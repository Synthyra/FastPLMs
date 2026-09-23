"""Final confidence-head evaluation and training-label frequency baselines."""

from __future__ import annotations

import json
import math

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .config import WANDB_PROJECT
from .metrics import _optional_spearman, evaluate_acceptance, summarize


def _values(value: Any) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()
    elif hasattr(value, "reshape"):
        value = value.reshape(-1).tolist()
    return [int(item) for item in value]


def _mask_values(value: Any, size: int) -> list[bool]:
    if value is None:
        return [True] * size
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()
    elif hasattr(value, "reshape"):
        value = value.reshape(-1).tolist()
    mask = [bool(item) for item in value]
    if len(mask) != size:
        raise ValueError("target mask and target bins have different lengths")
    return mask


def _target_field(target: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in target:
            return target[name]
    raise ValueError(f"target is missing one of {names}")


def build_frequency_baseline(
    training_targets: Iterable[Mapping[str, Any]],
    evaluation_targets: Iterable[Mapping[str, Any]] | None = None,
    smoothing: float = 1.0,
) -> dict[str, Any]:
    """Build target-normalized pLDDT and PAE categorical baselines.

    Histograms are fitted only from ``training_targets``. If evaluation targets
    are provided, returned cross-entropies are measured on them; otherwise they
    are measured on the training targets. Each target contributes one mean loss
    after masking, independent of its atom or pair count.
    """
    if smoothing <= 0 or not math.isfinite(smoothing):
        raise ValueError("smoothing must be positive and finite")
    training_count = 0
    counts = {"plddt": [smoothing] * 50, "pae": [smoothing] * 64}
    for target in training_targets:
        training_count += 1
        for name, aliases, bins in (
            ("plddt", ("plddt_target", "plddt_bins"), 50),
            ("pae", ("pae_target", "pae_bins"), 64),
        ):
            values = _values(_target_field(target, *aliases))
            mask = _mask_values(target.get(f"{name}_mask"), len(values))
            valid_values = [value for value, valid in zip(values, mask, strict=False) if valid]
            if not valid_values:
                raise ValueError(f"undefined {name} training histogram for target")
            target_weight = 1.0 / len(valid_values)
            for value in valid_values:
                if not 0 <= value < bins:
                    raise ValueError(f"{name} target bin is outside [0, {bins})")
                counts[name][value] += target_weight
    if training_count == 0:
        raise ValueError("training_targets cannot be empty")
    probabilities = {
        name: [count / math.fsum(values) for count in values] for name, values in counts.items()
    }
    if evaluation_targets is None:
        # Target-normalized sufficient statistics also give training CE without
        # retaining coordinates or consuming a single-pass iterator twice.
        return {
            "smoothing": smoothing,
            "plddt_probabilities": probabilities["plddt"],
            "pae_probabilities": probabilities["pae"],
            **{
                f"{name}_ce": -math.fsum(
                    (count - smoothing) * math.log(probability)
                    for count, probability in zip(counts[name], probabilities[name], strict=False)
                )
                / training_count
                for name in counts
            },
            "training_target_count": training_count,
            "evaluation_target_count": training_count,
        }
    evaluation = evaluation_targets
    evaluation_count = 0
    losses: dict[str, list[float]] = {"plddt": [], "pae": []}
    for target in evaluation:
        evaluation_count += 1
        for name, aliases in (
            ("plddt", ("plddt_target", "plddt_bins")),
            ("pae", ("pae_target", "pae_bins")),
        ):
            values = _values(_target_field(target, *aliases))
            mask = _mask_values(target.get(f"{name}_mask"), len(values))
            valid_values = [value for value, valid in zip(values, mask, strict=False) if valid]
            if any(value < 0 or value >= len(probabilities[name]) for value in valid_values):
                raise ValueError(f"{name} evaluation bin is outside the categorical range")
            valid_losses = [-math.log(probabilities[name][value]) for value in valid_values]
            if not valid_losses:
                raise ValueError(f"undefined {name} baseline loss for target")
            losses[name].append(math.fsum(valid_losses) / len(valid_losses))
    if evaluation_count == 0:
        raise ValueError("evaluation_targets cannot be empty")
    return {
        "smoothing": smoothing,
        "plddt_probabilities": probabilities["plddt"],
        "pae_probabilities": probabilities["pae"],
        "plddt_ce": math.fsum(losses["plddt"]) / len(losses["plddt"]),
        "pae_ce": math.fsum(losses["pae"]) / len(losses["pae"]),
        "training_target_count": training_count,
        "evaluation_target_count": evaluation_count,
    }


def _require_completed_training(root: Path, model_id: str) -> Path:
    directory = root / model_id / "train"
    result_path = directory / "result.json"
    checkpoint = directory / "best.safetensors"
    if not result_path.exists() or not checkpoint.exists():
        raise ValueError(
            "final evaluation requires a completed training result and best.safetensors"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "complete":
        raise ValueError("final evaluation requires training status complete")
    return checkpoint


def _sample_ranking(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure paired-seed confidence selection and target-level regret."""
    by_target: dict[str, dict[int, dict[str, float | None]]] = {}
    for prediction in predictions:
        seed = int(prediction.get("seed", 17))
        values = prediction.get("plddt_pred")
        truths = prediction.get("plddt_true")
        if not isinstance(values, list) or not values or not isinstance(truths, list) or not truths:
            continue
        by_target.setdefault(str(prediction["target_id"]), {})[seed] = {
            "predicted_plddt": math.fsum(float(value) for value in values) / len(values),
            "true_plddt": math.fsum(float(value) for value in truths) / len(truths),
            "iptm": float(prediction["iptm"]) if prediction.get("iptm") is not None else None,
            "dockq": float(prediction["dockq"]) if prediction.get("dockq") is not None else None,
        }
    paired = [(seeds[17], seeds[29]) for seeds in by_target.values() if 17 in seeds and 29 in seeds]
    if not paired:
        return {
            "paired_target_count": 0,
            "seed_rank_spearman": None,
            "mean_abs_plddt_delta": None,
            "plddt_selection": _selection_metrics([]),
            "interface_selection": _selection_metrics([]),
        }
    seed_17 = [pair[0]["predicted_plddt"] for pair in paired]
    seed_29 = [pair[1]["predicted_plddt"] for pair in paired]
    plddt_pairs = [
        (
            pair[0]["predicted_plddt"],
            pair[1]["predicted_plddt"],
            pair[0]["true_plddt"],
            pair[1]["true_plddt"],
        )
        for pair in paired
    ]
    interface_pairs = [
        (pair[0]["iptm"], pair[1]["iptm"], pair[0]["dockq"], pair[1]["dockq"])
        for pair in paired
        if pair[0]["iptm"] is not None
        and pair[1]["iptm"] is not None
        and pair[0]["dockq"] is not None
        and pair[1]["dockq"] is not None
    ]
    return {
        "paired_target_count": len(paired),
        "seed_rank_spearman": _optional_spearman(seed_17, seed_29, "seed_rank_spearman"),
        "mean_abs_plddt_delta": math.fsum(
            abs(left - right) for left, right in zip(seed_17, seed_29, strict=False)
        )
        / len(paired),
        "plddt_selection": _selection_metrics(plddt_pairs),
        "interface_selection": _selection_metrics(interface_pairs),
    }


def _selection_metrics(
    pairs: list[tuple[float | None, float | None, float | None, float | None]],
) -> dict[str, Any]:
    confidence_ties = 0
    quality_ties = 0
    correct = 0
    regrets: list[float] = []
    for seed_17_confidence, seed_29_confidence, seed_17_quality, seed_29_quality in pairs:
        if any(
            value is None
            for value in (seed_17_confidence, seed_29_confidence, seed_17_quality, seed_29_quality)
        ):
            continue
        confidence_delta = seed_17_confidence - seed_29_confidence
        quality_delta = seed_17_quality - seed_29_quality
        if confidence_delta == 0.0:
            confidence_ties += 1
            continue
        if quality_delta == 0.0:
            quality_ties += 1
            continue
        selected_quality = seed_17_quality if confidence_delta > 0 else seed_29_quality
        best_quality = max(seed_17_quality, seed_29_quality)
        correct += int(selected_quality == best_quality)
        regrets.append(best_quality - selected_quality)
    comparable = len(regrets)
    return {
        "paired_target_count": len(pairs),
        "confidence_tie_count": confidence_ties,
        "quality_tie_count": quality_ties,
        "comparable_count": comparable,
        "selection_accuracy": correct / comparable if comparable else None,
        "mean_regret": math.fsum(regrets) / comparable if comparable else None,
    }


def evaluate_final(root: Path, model_id: str) -> dict[str, Any]:
    """Evaluate donor and fixed best head on final-test targets and log W&B."""
    checkpoint = _require_completed_training(root, model_id)
    import wandb
    from safetensors.torch import load_file

    from fastplms.registry import get_model_spec

    from .training import HeadContext, _cache_path, _records, _targets, _wandb_run, predict_records
    from .cache import load_cache

    records = _records(root)
    final_records = [record for record in records if record.get("split") == "final_test"]
    validation_records = [record for record in records if record.get("split") == "validation"]
    training_records = [record for record in records if record.get("split") == "train"]
    if not final_records:
        raise ValueError("final evaluation requires final_test records")
    missing_seeds = [
        record["id"]
        for record in final_records
        if any(not _cache_path(root, model_id, record, seed).exists() for seed in (17, 29))
    ]
    if missing_seeds:
        raise ValueError(
            f"final evaluation requires cache seeds 17 and 29 for every target: {missing_seeds[:5]}"
        )
    run = _wandb_run(
        root,
        model_id,
        "evaluate",
        {
            "model_id": model_id,
            "stage": "final_test",
            "two_seeds": True,
            "wandb_project": WANDB_PROJECT,
        },
    )
    directory = root / model_id / "evaluate"
    directory.mkdir(parents=True, exist_ok=True)
    try:
        donor_context = HeadContext(model_id)
        candidate_context = HeadContext(model_id)
        candidate_context.head.load_state_dict(load_file(str(checkpoint)), strict=True)
        donor_predictions = predict_records(
            donor_context, root, model_id, final_records, two_seeds=True
        )
        candidate_predictions = predict_records(
            candidate_context, root, model_id, final_records, two_seeds=True
        )
        donor_validation = (
            summarize(predict_records(donor_context, root, model_id, validation_records))
            if validation_records
            else None
        )
        candidate_validation = (
            summarize(predict_records(candidate_context, root, model_id, validation_records))
            if validation_records
            else None
        )

        def target_stream(selected_records: list[dict[str, Any]], seeds: tuple[int, ...]):
            for selected_record in selected_records:
                for seed in seeds:
                    cache, _ = load_cache(
                        _cache_path(root, model_id, selected_record, seed),
                        model_id=model_id,
                        model_revision=get_model_spec(model_id).confidence_training_base.revision,
                        seed=seed,
                    )
                    yield _targets(cache)

        baseline = build_frequency_baseline(
            target_stream(training_records, (17,)), target_stream(final_records, (17, 29))
        )
        donor_summary = summarize(donor_predictions)
        candidate_summary = summarize(candidate_predictions)
        frequency_summary = {
            "plddt_ce": baseline["plddt_ce"],
            "pae_ce": baseline["pae_ce"],
            "final_test_strata_counts": candidate_summary["final_test_strata_counts"],
            "training_target_count": baseline["training_target_count"],
            "evaluation_target_count": baseline["evaluation_target_count"],
            "evaluation_seed_count": 2,
            "evaluation_strata_counts_per_seed": {
                kind: sum(record.get("kind") == kind for record in final_records)
                for kind in ("monomer", "dimer")
            },
        }
        report = {
            "status": "complete",
            "model_id": model_id,
            "wandb_url": run.url,
            "donor": donor_summary,
            "candidate": candidate_summary,
            "donor_validation": donor_validation,
            "candidate_validation": candidate_validation,
            "donor_sample_ranking": _sample_ranking(donor_predictions),
            "candidate_sample_ranking": _sample_ranking(candidate_predictions),
            "frequency": frequency_summary,
            "acceptance": evaluate_acceptance(candidate_summary, donor_summary, frequency_summary),
            "frequency_baseline": baseline,
        }
        (directory / "final-records.json").write_text(
            json.dumps({"donor": donor_predictions, "candidate": candidate_predictions}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        (directory / "result.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        run.log(
            {
                f"final/{key}": value
                for key, value in candidate_summary.items()
                if isinstance(value, int | float)
            }
        )
        artifact = wandb.Artifact(f"{model_id}-final-evaluation", type="evaluation")
        artifact.add_file(str(directory / "final-records.json"))
        artifact.add_file(str(directory / "result.json"))
        run.log_artifact(artifact).wait()
        run.finish(exit_code=0)
        return report
    except Exception:
        run.finish(exit_code=1)
        raise


frequency_baseline = build_frequency_baseline


__all__ = ["build_frequency_baseline", "evaluate_final", "frequency_baseline"]
