"""Project verified evaluation records into the current confidence release evidence."""

from __future__ import annotations

import json
import math

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .acceptance import acceptance_gates
from .experiment_artifacts import file_identity, verify_evaluation_group
from .v2_analysis import (
    BOOTSTRAP_SAMPLES,
    DOCKQ_MARGIN,
    LONG_STRATUM,
    PAIR_MARGIN_EVALUATION,
    sample_metrics,
)


VALIDATION_CHECKS = (
    "strict_reload",
    "embedded_head_identity",
    "confidence_default_enabled",
    "confidence_ranges",
    "seeded_coordinate_equality",
    "cif_confidence",
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _public_values(value: Any) -> Any:
    """Represent undefined measurements as null without altering the original records."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _public_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_public_values(item) for item in value]
    return value


def _coverage(
    targets: list[dict[str, Any]], records: list[dict[str, Any]]
) -> dict[str, dict[str, int]]:
    requested = {target["target_id"]: target["stratum"] for target in targets}
    retained = {record["target_id"] for record in records}
    return {
        group: {
            "requested": sum(
                (stratum == LONG_STRATUM) == long for stratum in requested.values()
            ),
            "retained": sum(
                (requested[target] == LONG_STRATUM) == long for target in retained
            ),
            "skipped": sum(
                (stratum == LONG_STRATUM) == long and target not in retained
                for target, stratum in requested.items()
            ),
        }
        for group, long in (("standard", False), ("long", True))
    }


def validate_artifact_evidence(report: Mapping[str, Any], head_sha256: str) -> None:
    if report.get("status") not in {"pending", "passed", "failed"}:
        raise ValueError("Artifact validation must declare pending, passed, or failed")
    if report["status"] == "passed" and (
        report.get("head_sha256") != head_sha256
        or not isinstance(report.get("weight_sha256"), str)
        or len(report["weight_sha256"]) != 64
        or any(
            report.get("checks", {}).get(name) is not True for name in VALIDATION_CHECKS
        )
    ):
        raise ValueError(
            "Passed artifact validation requires the exact head and all checks"
        )


def build_release_evidence(
    model_id: str,
    evaluation_dirs: Mapping[str, Path],
    acceptance_path: Path,
    *,
    expected_head_sha256: str,
    artifact_validation: Mapping[str, Any] | None = None,
    evidence_sources: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Create a v1 view while preserving the historical names in immutable raw inputs."""
    if model_id not in {"esmfold2_300", "esmfold2_600"}:
        raise ValueError(
            "Release evidence supports only the two trained confidence heads"
        )
    directories = {name: evaluation_dirs[name] for name in (model_id, "esmfold2")}
    completions = verify_evaluation_group(directories)
    candidate_dir = directories[model_id]
    request = _read(candidate_dir / "request.json")
    checkpoint = request["checkpoint_inputs"]["v2"]
    if checkpoint["sha256"] != expected_head_sha256:
        raise ValueError("Release head differs from the evaluated checkpoint")
    training = _read(candidate_dir / "inputs/training-report.json")
    identity = {key: checkpoint[key] for key in ("size", "sha256")}
    if (
        training.get("status") != "complete"
        or training.get("model_id") != model_id
        or training.get("selected_checkpoint") != "final-ema.safetensors"
        or training["checkpoint_files"].get("final-ema.safetensors") != identity
        or training.get("stopped_by") != "planned_updates"
        or training.get("updates") != training["config"].get("planned_updates")
    ):
        raise ValueError(
            "Current training report does not identify the evaluated final EMA"
        )
    base = request["metadata"]["model"]
    weight = next(item for item in base["files"] if item["path"] == "model.safetensors")
    provenance = training["provenance"]
    if (
        provenance.get("base_repo") != base["repo_id"]
        or provenance.get("base_revision") != base["revision"]
        or provenance.get("base_weight_sha256") != weight["digest"]
    ):
        raise ValueError("Training and evaluation disagree on the frozen base identity")
    acceptance = _read(acceptance_path)
    if (
        acceptance.get("evaluation_id") != completions[model_id]["evaluation_id"]
        or acceptance.get("head") != "v2"
    ):
        raise ValueError("Acceptance belongs to a different evaluation or head")
    for name, directory in directories.items():
        if acceptance["input_files"].get(f"{name}/records.json") != asdict(
            file_identity(directory / "records.json")
        ):
            raise ValueError("Acceptance records differ from the completed evaluation")
    for name in ("acceptance.py", "v2_analysis.py"):
        if acceptance["analysis_source_files"].get(name) != asdict(
            file_identity(Path(__file__).with_name(name))
        ):
            raise ValueError(
                "Acceptance analysis source differs from the verified release analysis"
            )
    records = {
        name: _read(directory / "records.json")
        for name, directory in directories.items()
    }
    shared = set.intersection(
        *(
            {
                record["target_id"]
                for record in values
                if record["stratum"] != LONG_STRATUM
            }
            for values in records.values()
        )
    )
    estimates = acceptance["estimates"]
    expected_gates = acceptance_gates(estimates)
    if any(acceptance.get(name) != value for name, value in expected_gates.items()):
        raise ValueError("Saved acceptance gates differ from their paired estimates")
    if estimates["shared_targets"]["compared"] != len(shared):
        raise ValueError("Paired estimates have a different shared-target count")
    heads = {}
    candidate_summary = _read(candidate_dir / "summary.json")
    production_summary = _read(directories["esmfold2"] / "summary.json")
    for raw_name, public_name in (
        ("v2", "v1"),
        ("pilot", "pilot"),
        ("donor", "donor"),
        ("production", "production"),
    ):
        source = "esmfold2" if raw_name == "production" else model_id
        compared = [
            record for record in records[source] if record["target_id"] in shared
        ]
        observed = sample_metrics(compared, raw_name)
        for metric, value in estimates[raw_name]["estimate"].items():
            actual = observed[metric]
            if not (
                (math.isnan(value) and math.isnan(actual))
                or math.isclose(value, actual, rel_tol=1e-10, abs_tol=1e-12)
            ):
                raise ValueError(
                    f"Paired point estimate differs from saved predictions: {raw_name}/{metric}"
                )
        summary = production_summary if raw_name == "production" else candidate_summary
        heads[public_name] = {
            **observed,
            "interval_95": estimates[raw_name]["interval_95"],
            "by_stratum": summary[raw_name]["by_stratum"],
        }
    validation = dict(artifact_validation or {"status": "pending"})
    validate_artifact_evidence(validation, expected_head_sha256)
    coverage = {
        name: _coverage(_read(directory / "targets.json"), records[name])
        for name, directory in directories.items()
    }
    return _public_values(
        {
            "release": "v1",
            "model_id": model_id,
            "head_sha256": expected_head_sha256,
            "frozen_base": request["metadata"]["model"],
            "donor": request["metadata"]["donor"],
            "training": {
                key: training[key]
                for key in (
                    "updates",
                    "elapsed_hours",
                    "wandb_url",
                    "config",
                    "final_validation",
                    "provenance",
                )
            },
            "data": _read(candidate_dir / "inputs/split-report.json"),
            "test": {
                "set_status": "spent",
                "new_heldout_evaluation": False,
                "inference": request["metadata"]["inference"],
                "coverage": coverage,
                "shared_standard_targets": len(shared),
                "heads": heads,
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "pair_margins": {"lddt": PAIR_MARGIN_EVALUATION, "dockq": DOCKQ_MARGIN},
            },
            "production_agreement": {
                ("v1" if key == "v2" else key): value
                for key, value in acceptance["production_agreement"].items()
            },
            "gates": {"passed": acceptance["passed"], **acceptance["gates"]},
            "gate_detail": {name: acceptance[name] for name in acceptance["gates"]},
            "artifact_validation": validation,
            "metrics_review": {
                "status": "validated",
                "reason": "Completed prediction manifests, paired analysis inputs and source identities verified.",
            },
            "source_mapping": {"v1": "v2", "pilot": "pilot"},
            "evidence_sources": dict(evidence_sources or {}),
            "input_files": {
                "acceptance": asdict(file_identity(acceptance_path)),
                **{
                    name: asdict(file_identity(directory / "completion.json"))
                    for name, directory in directories.items()
                },
            },
        }
    )
