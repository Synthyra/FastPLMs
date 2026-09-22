"""Saved-record correction preserves original evidence and records its input identities."""

import hashlib
import json
import sys
import copy

import pytest

from pathlib import Path
from types import ModuleType

from tools.confidence.recompute import MODEL_IDS, SOURCE_FILES, corrected_evidence, recompute


def _summary(heads):
    return {
        head: {
            "overall": {"targets": 3.0, "plddt_lddt_spearman": 0.75},
            "interval_95": {},
            "by_stratum": {},
        }
        for head in heads
    }


def _acceptance():
    return {
        "passed": False,
        "gates": {"beats_pilot": True, "sample_selection": True, "production_parity": False},
        "beats_pilot": {},
        "sample_selection": {},
        "production_parity": {},
        "production_agreement": {"v2": {"mean_plddt_spearman": 0.5}},
        "estimates": _estimates(),
    }


def _estimates():
    return {
        **{
            head: {
                "estimate": {"targets": 3.0, "plddt_lddt_spearman": 0.8},
                "interval_95": {"plddt_lddt_spearman": [0.4, 0.9]},
            }
            for head in ("v2", "pilot", "donor", "production")
        },
        "shared_targets": {"compared": 3, "not_in_both_evaluations": []},
    }


def _evidence(model_id):
    return {
        "model_id": model_id,
        "status": "complete",
        "metrics_review": {"status": "requires_recomputation"},
        "training": {
            "updates": 100,
            "final_validation": {"total_ce": 3.0, "target_plddt_spearman": 0.9},
        },
        "test": {"targets": 512, "samples_per_target": 5, "heads": {"obsolete": {}}},
    }


def test_corrected_evidence_replaces_metrics_without_claiming_validation_was_rescored():
    original = _evidence("esmfold2_300")
    provenance = {
        "refolding": False,
        "target_counts": {"esmfold2_300": {"total": 4, "standard": 3, "long": 1}},
    }
    corrected = corrected_evidence(
        original,
        _summary(["v2", "pilot", "donor"]),
        _summary(["production"]),
        _acceptance(),
        provenance,
    )
    assert original["training"]["final_validation"]["target_plddt_spearman"] == 0.9
    assert corrected["training"]["final_validation"]["target_plddt_spearman"] is None
    assert corrected["training"]["final_validation"]["total_ce"] == 3.0
    assert (
        corrected["training"]["final_validation_correlation_review"]["status"]
        == "unverified_pending_validation_cache_rescore"
    )
    assert set(corrected["test"]["heads"]) == {"v2", "pilot", "donor", "production"}
    assert corrected["test"]["targets"] == 4
    assert corrected["test"]["standard_targets"] == 3
    assert corrected["test"]["heads"]["v2"]["plddt_lddt_spearman"] == 0.8
    assert corrected["test"]["heads"]["v2"]["interval_95"] == _estimates()["v2"]["interval_95"]
    assert corrected["metrics_review"]["status"] == "corrected_test_metrics"
    assert corrected["gates"]["passed"] is False


def test_recompute_hashes_inputs_and_writes_only_new_outputs(monkeypatch, tmp_path):
    evaluation = tmp_path / "evaluation"
    originals = {}
    for model_id in (*MODEL_IDS, "esmfold2"):
        path = evaluation / model_id / "records.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps([{"target_id": "saved-record", "stratum": "monomer_short"}]),
            encoding="utf-8",
        )
        originals[path] = path.read_bytes()
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    for model_id in MODEL_IDS:
        path = evidence_dir / f"{model_id}-v2.json"
        path.write_text(json.dumps(_evidence(model_id)), encoding="utf-8")
        originals[path] = path.read_bytes()

    # Stub only the expensive metric execution; exercise real file IO and evidence conversion.
    evaluation_module = ModuleType("tools.confidence.v2_analysis")
    evaluation_module.BOOTSTRAP_SAMPLES = 1000
    evaluation_module.summarize = lambda records, heads: _summary(heads)
    acceptance_module = ModuleType("tools.confidence.acceptance")
    acceptance_module.paired_estimates = lambda *args, **kwargs: _estimates()
    acceptance_module.acceptance_gates = lambda estimates: _acceptance()
    acceptance_module.production_agreement = lambda *args: {"v2": {"mean_plddt_spearman": 0.5}}
    monkeypatch.setitem(sys.modules, evaluation_module.__name__, evaluation_module)
    monkeypatch.setitem(sys.modules, acceptance_module.__name__, acceptance_module)

    output = tmp_path / "corrected"
    provenance = recompute(evaluation, output, evidence_dir)
    assert provenance["test_set_status"] == "spent"
    assert provenance["new_heldout_evaluation"] is False
    assert provenance["refolding"] is False
    assert provenance["evaluation_lineage"] == {
        "status": "legacy_records_without_manifests",
        "manifests_verified": False,
        "shared_target_identity": "unverified",
    }
    assert set(provenance["source_sha256"]) == set(SOURCE_FILES)
    for path, content in originals.items():
        assert path.read_bytes() == content
        assert (
            provenance["input_sha256"][str(path.resolve())] == hashlib.sha256(content).hexdigest()
        )
    assert (output / "recomputation.json").exists()
    for model_id in MODEL_IDS:
        assert (output / model_id / "summary.json").exists()
        assert (output / f"acceptance-{model_id}.json").exists()
        assert (output / "evidence" / f"{model_id}-v2.json").exists()
    with pytest.raises(FileExistsError, match="new directory"):
        recompute(evaluation, output, evidence_dir)


def test_recompute_requires_all_saved_records_before_creating_output(tmp_path):
    with pytest.raises(FileNotFoundError):
        recompute(tmp_path / "missing", tmp_path / "corrected")
    assert not (tmp_path / "corrected").exists()


def test_recompute_real_metrics_uses_paired_cohort_and_intervals(monkeypatch, tmp_path):
    from tools.confidence import acceptance, v2_analysis

    monkeypatch.setattr(acceptance, "BOOTSTRAP_SAMPLES", 8)
    monkeypatch.setattr(v2_analysis, "BOOTSTRAP_SAMPLES", 8)
    records = []
    for target in range(5):
        for sample in range(3):
            quality = 0.2 + 0.2 * sample + 0.02 * target
            score = 0.3 + 0.2 * sample  # Deliberate ties across targets.
            counts = [0] * 10
            predicted_sums, true_sums = [0.0] * 10, [0.0] * 10
            index = int(score * 10)
            counts[index], predicted_sums[index], true_sums[index] = 1, score, quality
            resolved, unresolved = [0] * 50, [0] * 50
            resolved[int(score * 50)], unresolved[5] = 1, 1
            prediction = {
                "mean_plddt": score,
                "ptm": score,
                "iptm": score,
                "plddt_ce": 1.0,
                "pae_ce": 2.0,
                "absolute_error_sum": abs(score - quality),
                "calibration_count": counts,
                "calibration_predicted_sum": predicted_sums,
                "calibration_true_sum": true_sums,
                "resolved_residue_histogram": resolved,
                "unresolved_residue_histogram": unresolved,
                "resolved_residue_plddt_sum": score,
                "unresolved_residue_plddt_sum": 0.1,
            }
            records.append(
                {
                    "target_id": f"target-{target}",
                    "sample": sample,
                    "stratum": "long" if target == 4 else "dimer_hetero",
                    "num_chains": 2,
                    "true_lddt": quality,
                    "tm_score": quality,
                    "dockq": quality,
                    "predictions": {
                        head: copy.deepcopy(prediction) for head in ("v2", "pilot", "donor")
                    },
                }
            )
    reference = [
        {**record, "predictions": {"production": record["predictions"]["v2"]}}
        for record in records
        if record["target_id"] != "target-3"
    ]
    evaluation = tmp_path / "evaluation"
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    for model_id in (*MODEL_IDS, "esmfold2"):
        path = evaluation / model_id / "records.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(reference if model_id == "esmfold2" else records), encoding="utf-8"
        )
        if model_id in MODEL_IDS:
            (evidence_dir / f"{model_id}-v2.json").write_text(
                json.dumps(_evidence(model_id)), encoding="utf-8"
            )

    output = tmp_path / "corrected"
    recompute(evaluation, output, evidence_dir)
    expected = acceptance.paired_estimates(records, ["v2", "pilot", "donor"], reference, seed=0)
    evidence = json.loads((output / "evidence" / "esmfold2_300-v2.json").read_text())
    assert evidence["test"]["targets"] == 5
    assert evidence["test"]["standard_targets"] == 4
    assert evidence["test"]["long_targets"] == 1
    assert evidence["test"]["shared_standard_targets"] == 3
    assert evidence["test"]["not_in_both_evaluations"] == ["target-3"]
    for head in ("v2", "pilot", "donor", "production"):
        corrected = evidence["test"]["heads"][head]
        for metric, point in expected[head]["estimate"].items():
            assert corrected[metric] == pytest.approx(point, nan_ok=True)
        for metric, interval in expected[head]["interval_95"].items():
            assert corrected["interval_95"][metric] == pytest.approx(interval, nan_ok=True)
        assert "long" in corrected["by_stratum"]
    assert evidence["metrics_review"]["status"] == "corrected_test_metrics"


def _manifest_group(
    root: Path,
    *,
    changed_model: str | None = None,
    change: str | None = None,
) -> Path:
    from tools.confidence.experiment_artifacts import EvaluationArtifacts, write_new_json

    from .test_confidence_v2_analysis import _records

    root.mkdir()
    checkpoint = root / "selected.safetensors"
    checkpoint.write_bytes(b"immutable test checkpoint")
    split_report = root / "split-report.json"
    split_report.write_text('{"status": "verified"}', encoding="utf-8")
    for model_id in (*MODEL_IDS, "esmfold2"):
        records = _records()
        targets = {}
        for record in records:
            sequences = ["AC"] if record["num_chains"] == 1 else ["AC", "DE"]
            record["num_tokens"] = 2 * record["num_chains"]
            target_id = record["target_id"]
            record["target_positions"] = {
                "sha256": hashlib.sha256(target_id.encode()).hexdigest(),
                "shape": [record["num_tokens"], 14, 3],
                "dtype": "float32",
            }
            if model_id == changed_model and target_id == "m1":
                if change == "sequence":
                    sequences = ["AG"]
                if change == "coordinates":
                    record["target_positions"]["sha256"] = "f" * 64
            targets[target_id] = {
                "target_id": target_id,
                "sequences": sequences,
                "num_chains": record["num_chains"],
                "num_tokens": record["num_tokens"],
                "stratum": record["stratum"],
            }
            if model_id == "esmfold2":
                record["predictions"] = {"production": record["predictions"]["v2"]}
        run = EvaluationArtifacts.create(
            root / model_id,
            evaluation_id="different"
            if model_id == changed_model and change == "group"
            else "group",
            model_id="wrong_model" if model_id == changed_model and change == "model" else model_id,
            split="validation" if model_id == changed_model and change == "split" else "test",
            targets=list(targets.values()),
            head_files=(
                {"v2": checkpoint, "pilot": checkpoint, "donor": None}
                if model_id in MODEL_IDS
                else {}
            ),
            metadata={"inference": {"samples": 3, "seed_offset": 1000}},
            input_files={"split-report.json": split_report},
        )
        write_new_json(run.directory / "records.json", records)
        write_new_json(run.directory / "skipped.json", [])
        write_new_json(run.directory / "summary.json", _summary(records[0]["predictions"]))
        run.complete()
    return root


def test_recompute_verifies_public_export_lineage_without_checkpoint_weights(monkeypatch, tmp_path):
    from tools.confidence import acceptance, v2_analysis
    from tools.confidence.experiment_artifacts import export_evaluation

    monkeypatch.setattr(acceptance, "BOOTSTRAP_SAMPLES", 5)
    monkeypatch.setattr(v2_analysis, "BOOTSTRAP_SAMPLES", 5)
    original = _manifest_group(tmp_path / "original")
    exported = tmp_path / "exported"
    for model_id in (*MODEL_IDS, "esmfold2"):
        export_evaluation(original / model_id, exported / model_id)
    assert not list(exported.rglob("*.safetensors"))
    provenance = recompute(exported, tmp_path / "corrected")
    assert provenance["evaluation_lineage"] == {
        "status": "verified_manifest_group",
        "manifests_verified": True,
        "shared_target_identity": "verified",
        "evaluation_id": "group",
        "checkpoint_weights_required": False,
    }
    for model_id in (*MODEL_IDS, "esmfold2"):
        path = exported / model_id / "completion.json"
        assert (
            provenance["input_sha256"][str(path.resolve())]
            == hashlib.sha256(path.read_bytes()).hexdigest()
        )


@pytest.mark.parametrize("change", ["model", "group", "split", "sequence", "coordinates"])
def test_recompute_rejects_incoherent_manifest_groups_before_writing(tmp_path, change):
    evaluation = _manifest_group(
        tmp_path / "evaluation", changed_model="esmfold2_600", change=change
    )
    with pytest.raises(ValueError):
        recompute(evaluation, tmp_path / "corrected")
    assert not (tmp_path / "corrected").exists()


@pytest.mark.parametrize("marker", ["request.json", "completion.json", "failure.json"])
def test_any_manifest_marker_prevents_fallback_to_legacy_records(tmp_path, marker):
    evaluation = tmp_path / "evaluation"
    for model_id in (*MODEL_IDS, "esmfold2"):
        directory = evaluation / model_id
        directory.mkdir(parents=True)
        (directory / "records.json").write_text(
            json.dumps([{"target_id": "saved-record", "stratum": "monomer_short"}]),
            encoding="utf-8",
        )
    (evaluation / "esmfold2_300" / marker).write_text("{}", encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        recompute(evaluation, tmp_path / "corrected")
    assert not (tmp_path / "corrected").exists()


def test_recompute_rejects_modified_records_in_an_otherwise_complete_group(tmp_path):
    evaluation = _manifest_group(tmp_path / "evaluation")
    path = evaluation / "esmfold2_600" / "records.json"
    records = json.loads(path.read_text())
    records[0]["true_lddt"] = 0.99
    path.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        recompute(evaluation, tmp_path / "corrected")
    assert not (tmp_path / "corrected").exists()
