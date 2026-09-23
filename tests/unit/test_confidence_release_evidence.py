"""Release evidence preserves failed gates, raw identities, and the spent-test boundary."""

import json

import pytest

from dataclasses import asdict, replace
from pathlib import Path

from fastplms.registry import ConfidenceAdaptation, get_model_spec
from tools.artifacts.doc_generation.confidence_evidence import load_confidence_evidence
from tools.artifacts.doc_generation.model_cards import render_model_card
from tools.confidence import acceptance, release_evidence
from tools.confidence.experiment_artifacts import file_identity
from .test_confidence_acceptance import _records


@pytest.fixture
def release_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(acceptance, "BOOTSTRAP_SAMPLES", 20)
    monkeypatch.setattr(release_evidence, "BOOTSTRAP_SAMPLES", 20)
    model_id = "esmfold2_600"
    records = {
        model_id: _records({"v2": "reversed", "pilot": "correct", "donor": "reversed"}),
        "esmfold2": _records({"production": "correct"}),
    }
    directories = {name: tmp_path / name for name in records}
    for name, directory in directories.items():
        directory.mkdir()
        (directory / "records.json").write_text(json.dumps(records[name]))
        targets = [
            {"target_id": row["target_id"], "stratum": row["stratum"]}
            for row in records[name]
            if row["sample"] == 0
        ]
        (directory / "targets.json").write_text(json.dumps(targets))
        heads = ("production",) if name == "esmfold2" else ("v2", "pilot", "donor")
        (directory / "summary.json").write_text(
            json.dumps({head: {"by_stratum": {}} for head in heads})
        )
        (directory / "completion.json").write_text("{}")
    candidate = directories[model_id]
    checkpoint = {"sha256": "a" * 64, "size": 100}
    source = {
        "repo_id": "Synthyra/ESMFold2-600",
        "revision": "b" * 40,
        "files": [{"path": "model.safetensors", "digest": "c" * 64}],
    }
    request = {
        "checkpoint_inputs": {"v2": checkpoint},
        "metadata": {
            "model": source,
            "donor": {},
            "inference": {
                "samples": 3,
                "recycling_loops": 3,
                "diffusion_steps": 50,
                "parameter_dtype": "float32",
                "fold_autocast_dtype": "bfloat16",
                "attention_backend": "sdpa",
            },
        },
    }
    (candidate / "request.json").write_text(json.dumps(request))
    (candidate / "inputs").mkdir()
    training = {
        "status": "complete",
        "model_id": model_id,
        "selected_checkpoint": "final-ema.safetensors",
        "checkpoint_files": {"final-ema.safetensors": checkpoint},
        "stopped_by": "planned_updates",
        "updates": 780,
        "elapsed_hours": 18.0,
        "wandb_url": "https://wandb.ai/team/project/runs/train",
        "config": {
            "planned_updates": 780,
            "targets_per_update": 16,
            "samples_per_target": 4,
            "num_loops": 3,
            "num_sampling_steps": 50,
            "pae_weight": 1.0,
            "ranking_weight": 0.5,
        },
        "final_validation": {},
        "provenance": {
            "base_repo": source["repo_id"],
            "base_revision": source["revision"],
            "base_weight_sha256": "c" * 64,
        },
    }
    (candidate / "inputs/training-report.json").write_text(json.dumps(training))
    (candidate / "inputs/split-report.json").write_text(
        json.dumps({"train_targets": 475969})
    )
    monkeypatch.setattr(
        release_evidence,
        "verify_evaluation_group",
        lambda paths: {name: {"evaluation_id": "original-campaign"} for name in paths},
    )
    estimates = acceptance.paired_estimates(
        records[model_id], ["v2", "pilot", "donor"], records["esmfold2"]
    )
    report = {
        **acceptance.acceptance_gates(estimates),
        "evaluation_id": "original-campaign",
        "estimates": estimates,
        "production_agreement": acceptance.production_agreement(
            records[model_id], ["v2", "pilot", "donor"], records["esmfold2"]
        ),
        "input_files": {
            f"{name}/records.json": asdict(file_identity(directory / "records.json"))
            for name, directory in directories.items()
        },
        "analysis_source_files": {
            name: asdict(file_identity(Path(acceptance.__file__).with_name(name)))
            for name in ("acceptance.py", "v2_analysis.py")
        },
    }
    path = tmp_path / "acceptance.json"
    path.write_text(json.dumps(report))
    return model_id, directories, path


def test_verified_metrics_do_not_turn_failed_gates_into_passes(release_inputs):
    evidence = release_evidence.build_release_evidence(
        *release_inputs, expected_head_sha256="a" * 64
    )
    assert evidence["metrics_review"]["status"] == "validated"
    assert evidence["gates"]["passed"] is False
    assert evidence["gates"]["sample_selection"] is False
    assert evidence["test"]["new_heldout_evaluation"] is False
    assert evidence["test"]["shared_standard_targets"] == 4
    assert evidence["test"]["heads"]["v1"]["within_target_plddt_pairs"] == 12
    assert evidence["source_mapping"]["v1"] == "v2"
    assert evidence["artifact_validation"]["status"] == "pending"
    json.dumps(evidence, allow_nan=False)


@pytest.mark.parametrize("problem", ("head", "records", "analysis", "partial_training"))
def test_release_refuses_unmatched_evidence(release_inputs, problem):
    model_id, directories, path = release_inputs
    expected = "b" * 64 if problem == "head" else "a" * 64
    if problem in {"records", "analysis"}:
        report = json.loads(path.read_text())
        key = "input_files" if problem == "records" else "analysis_source_files"
        report[key] = {}
        path.write_text(json.dumps(report))
    elif problem == "partial_training":
        training_path = directories[model_id] / "inputs/training-report.json"
        training = json.loads(training_path.read_text())
        training["updates"] = 300
        training_path.write_text(json.dumps(training))
    with pytest.raises(ValueError):
        release_evidence.build_release_evidence(
            *release_inputs, expected_head_sha256=expected
        )


def test_v1_card_reports_failed_quality_and_pending_artifact_checks(
    release_inputs, tmp_path
):
    model_id, _, _ = release_inputs
    evidence = release_evidence.build_release_evidence(
        *release_inputs, expected_head_sha256="a" * 64
    )
    relative = f"docs/evidence/confidence/{model_id}-v1.json"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(evidence, allow_nan=False))
    spec = get_model_spec(model_id)
    adaptation = ConfidenceAdaptation(
        head_sha256="a" * 64,
        base_weight_sha256=spec.fast.file_map["model.safetensors"].digest,
        donor_repo="Synthyra/donor",
        donor_revision="c" * 40,
        donor_weight_sha256="d" * 64,
        training_url="https://wandb.ai/team/project/runs/train",
        evaluation_url="https://huggingface.co/datasets/Synthyra/FastPLMs-artifacts/tree/"
        + "e" * 40,
        evidence_path=relative,
        release="v1",
        frozen_base=spec.fast,
    )
    card = render_model_card(
        replace(spec, confidence_adaptation=adaptation), evidence_root=tmp_path
    )
    assert "## Confidence head v1" in card
    assert "| Within-target sample selection | Failed |" in card
    assert "| Strict Transformers reload | Pending |" in card
    assert "already-used test split" in card
    assert "passed held-out quality checks" not in card
    assert "Current v2 confidence head" not in card
    assert "1,024 experimental AtlasFold targets" not in card


def test_v1_evidence_without_review_is_withheld(tmp_path):
    path = tmp_path / "v1.json"
    path.write_text(json.dumps({"release": "v1"}))
    report = load_confidence_evidence(path, protocol="v1")
    assert report.review_status == "unreviewed"
    assert not report.can_report_metrics


def test_artifact_pass_requires_every_check_and_evaluated_head():
    report = {
        "status": "passed",
        "head_sha256": "a" * 64,
        "weight_sha256": "b" * 64,
        "checks": {name: True for name in release_evidence.VALIDATION_CHECKS},
    }
    release_evidence.validate_artifact_evidence(report, "a" * 64)
    report["checks"]["embedded_head_identity"] = False
    with pytest.raises(ValueError, match="all checks"):
        release_evidence.validate_artifact_evidence(report, "a" * 64)
