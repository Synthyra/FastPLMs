"""Recompute v2 metrics from saved records without refolding or changing source evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform

from importlib.metadata import version
from pathlib import Path
from typing import Any


MODEL_IDS = ("esmfold2_300", "esmfold2_600")
HEADS = ("v2", "pilot", "donor")
SOURCE_FILES = ("recompute.py", "test_evaluation.py", "acceptance.py", "online_training.py")


def _read_hashed(path: Path) -> tuple[Any, str]:
    content = path.read_bytes()
    return json.loads(content), hashlib.sha256(content).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def corrected_evidence(
    original: dict[str, Any],
    summary: dict[str, Any],
    production_summary: dict[str, Any],
    acceptance: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Preserve training metadata while replacing test metrics in the existing evidence schema."""
    evidence = copy.deepcopy(original)
    combined = {**summary, **production_summary}
    estimates = acceptance["estimates"]
    evidence["test"]["heads"] = {
        head: {
            **estimates[head]["estimate"],
            "interval_95": estimates[head]["interval_95"],
            "by_stratum": values["by_stratum"],
        }
        for head, values in combined.items()
    }
    counts = provenance["target_counts"][original["model_id"]]
    evidence["test"]["targets"] = counts["total"]
    evidence["test"]["standard_targets"] = counts["standard"]
    evidence["test"]["long_targets"] = counts["long"]
    evidence["test"]["shared_standard_targets"] = estimates["shared_targets"]["compared"]
    evidence["test"]["not_in_both_evaluations"] = estimates["shared_targets"][
        "not_in_both_evaluations"
    ]
    evidence["test"]["strata"] = (
        f"{counts['total']} saved targets ({counts['standard']} standard, {counts['long']} long). "
        f"Head estimates and paired intervals use {estimates['shared_targets']['compared']} "
        "shared standard targets. "
        "Per-stratum metrics use each source's own saved targets, including long targets."
    )
    evidence["production_agreement"] = acceptance["production_agreement"]
    evidence["gates"] = {"passed": acceptance["passed"], **acceptance["gates"]}
    evidence["gate_detail"] = {
        name: acceptance[name] for name in ("beats_pilot", "sample_selection", "production_parity")
    }
    validation = evidence.get("training", {}).get("final_validation", {})
    if "target_plddt_spearman" in validation:
        previous = validation.pop("target_plddt_spearman")
        evidence["training"]["final_validation_correlation_review"] = {
            "status": "unverified_pending_validation_cache_rescore",
            "original_target_plddt_spearman": previous,
            "reason": "Saved test records cannot verify the training validation correlation.",
        }
        validation["target_plddt_spearman"] = None
    evidence["recomputation"] = copy.deepcopy(provenance)
    prior_review = evidence.get("metrics_review", {})
    evidence["metrics_review"] = {
        "status": "corrected_test_metrics",
        "reason": "Test metrics and gates were recomputed from saved predictions.",
        "validation_status": "unverified_pending_validation_cache_rescore",
        "historical_values_preserved": "in_original_input_files",
        "original_review": prior_review,
    }
    if "training_history_audit" in prior_review:
        evidence["metrics_review"]["training_history_audit"] = prior_review[
            "training_history_audit"
        ]
    return evidence


def recompute(
    evaluation_dir: Path, output_dir: Path, evidence_dir: Path | None = None
) -> dict[str, Any]:
    """Write corrected summaries, paired gates, and optional evidence copies to a new directory.

    All calculations consume saved predictions on the same spent test set. This command never
    loads weights, folds structures, or treats the correction as a new held-out evaluation.
    """
    if output_dir.exists():
        raise FileExistsError(f"Recomputation output must be a new directory: {output_dir}")
    records: dict[str, Any] = {}
    inputs: dict[str, str] = {}
    evidence_sources: dict[str, dict[str, Any]] = {}
    for model_id in (*MODEL_IDS, "esmfold2"):
        path = evaluation_dir / model_id / "records.json"
        records[model_id], inputs[str(path.resolve())] = _read_hashed(path)
        if not isinstance(records[model_id], list) or not records[model_id]:
            raise ValueError(f"Expected nonempty saved records: {path}")
    if evidence_dir is not None:
        for model_id in MODEL_IDS:
            path = evidence_dir / f"{model_id}-v2.json"
            evidence_sources[model_id], inputs[str(path.resolve())] = _read_hashed(path)
            if evidence_sources[model_id].get("model_id") != model_id:
                raise ValueError(f"Evidence model identity differs: {path}")

    # --help needs only the standard library; calculation needs the confidence environment.
    from .acceptance import acceptance_gates, paired_estimates, production_agreement
    from .test_evaluation import BOOTSTRAP_SAMPLES, summarize

    provenance = {
        "status": "complete",
        "operation": "corrected_metrics_from_saved_predictions",
        "test_set_status": "spent",
        "new_heldout_evaluation": False,
        "refolding": False,
        "correction": "Spearman uses average ranks for ties and rejects undefined correlations.",
        "input_sha256": inputs,
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in SOURCE_FILES
        },
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": 0,
        "target_counts": {
            model_id: {
                "total": len({record["target_id"] for record in saved}),
                "standard": len(
                    {record["target_id"] for record in saved if record["stratum"] != "long"}
                ),
                "long": len(
                    {record["target_id"] for record in saved if record["stratum"] == "long"}
                ),
            }
            for model_id, saved in records.items()
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
        },
        "training_weights_changed": False,
        "training_partial_gradient_exposure": "not_audited_by_metric_recomputation",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    reference = records["esmfold2"]
    production_summary = summarize(reference, ["production"])
    _write_json(output_dir / "esmfold2" / "summary.json", production_summary)
    for model_id in MODEL_IDS:
        summary = summarize(records[model_id], list(HEADS))
        estimates = paired_estimates(records[model_id], list(HEADS), reference, seed=0)
        acceptance = {
            **acceptance_gates(estimates),
            "production_agreement": production_agreement(records[model_id], list(HEADS), reference),
            "estimates": estimates,
        }
        _write_json(output_dir / model_id / "summary.json", summary)
        _write_json(output_dir / f"acceptance-{model_id}.json", acceptance)
        if evidence_dir is not None:
            evidence = corrected_evidence(
                evidence_sources[model_id], summary, production_summary, acceptance, provenance
            )
            _write_json(output_dir / "evidence" / f"{model_id}-v2.json", evidence)
    _write_json(output_dir / "recomputation.json", provenance)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New directory; existing paths are refused."
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        help="Read original *-v2.json files and write corrected copies under output/evidence.",
    )
    args = parser.parse_args()
    recompute(args.evaluation_dir, args.output_dir, args.evidence_dir)
    print(
        f"Corrected saved-record metrics written to {args.output_dir}; the test set remains spent."
    )


if __name__ == "__main__":
    main()
