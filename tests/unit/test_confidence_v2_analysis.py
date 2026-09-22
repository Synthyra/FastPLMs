"""V2 calculations retain their sample weighting and run without training dependencies."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import pytest

from pathlib import Path

from tools.confidence.v2_analysis import sample_metrics
from tools.confidence.v2_records import EvaluationRecord, PredictionSummary


def _records() -> list[EvaluationRecord]:
    records: list[EvaluationRecord] = []
    for target_id, num_chains in (("m1", 1), ("m2", 1), ("d1", 2)):
        for sample, quality in enumerate((0.25, 0.5, 0.75)):
            counts = [0] * 10
            sums = [0.0] * 10
            counts[int(quality * 10)] = 2
            sums[int(quality * 10)] = 2 * quality
            resolved, unresolved = [0] * 50, [0] * 50
            resolved[int(quality * 50)], unresolved[5] = 2, 1
            prediction: PredictionSummary = {
                "mean_plddt": quality,
                "ptm": quality,
                "iptm": quality,
                "plddt_ce": 1.0,
                "pae_ce": 2.0,
                "absolute_error_sum": 0.0,
                "calibration_count": counts,
                "calibration_predicted_sum": sums,
                "calibration_true_sum": sums,
                "resolved_residue_histogram": resolved,
                "unresolved_residue_histogram": unresolved,
                "resolved_residue_plddt_sum": 2 * quality,
                "unresolved_residue_plddt_sum": 0.1,
            }
            records.append(
                {
                    "target_id": target_id,
                    "stratum": "monomer_short" if num_chains == 1 else "dimer_hetero",
                    "num_chains": num_chains,
                    "sample": sample,
                    "true_lddt": quality,
                    "tm_score": quality,
                    "dockq": quality if num_chains > 1 else None,
                    "predictions": {
                        head: copy.deepcopy(prediction) for head in ("v2", "pilot", "donor")
                    },
                }
            )
    return records


def test_complete_sample_metrics_match_the_declared_protocol() -> None:
    expected = {
        "targets": 3.0,
        "plddt_lddt_spearman": 1.0,
        "ptm_tm_spearman": 1.0,
        "iptm_dockq_spearman": 1.0,
        "atom_plddt_mae": 0.0,
        "calibration_error_10bin": 0.0,
        "plddt_ce": 1.0,
        "pae_ce": 2.0,
        "within_target_plddt_accuracy": 1.0,
        "within_target_plddt_pairs": 9.0,
        "within_target_iptm_dockq_accuracy": 1.0,
        "within_target_iptm_dockq_pairs": 3.0,
        "top1_regret": 0.0,
        "random_selection_regret": 0.25,
        "complexes_without_dockq": 0.0,
        "disorder_auroc": 1.0,
        "resolved_residue_mean_plddt": 0.5,
        "unresolved_residue_mean_plddt": 0.1,
        "resolved_fraction_below_50": 1 / 3,
        "unresolved_fraction_below_50": 1.0,
        "unresolved_residues": 9.0,
    }
    assert sample_metrics(_records(), "v2") == pytest.approx(expected)


def test_v2_test_correlations_keep_samples_instead_of_collapsing_targets() -> None:
    # Two targets are insufficient for the validation target-level correlation, but six samples
    # remain valid observations under the separate v2 test protocol.
    records = [record for record in _records() if record["num_chains"] == 1]
    metrics = sample_metrics(records, "v2")
    assert metrics["targets"] == 2.0
    assert metrics["plddt_lddt_spearman"] == pytest.approx(1.0)


def test_existing_training_and_evaluation_imports_reexport_the_same_calculations() -> None:
    from tools.confidence import online_training, test_evaluation, v2_analysis

    assert online_training.spearman is v2_analysis.spearman
    assert online_training.pairwise_accuracy is v2_analysis.pairwise_accuracy
    assert test_evaluation._metrics is v2_analysis.sample_metrics
    assert test_evaluation.bootstrap_records is v2_analysis.bootstrap_records
    assert test_evaluation.summarize is v2_analysis.summarize


@pytest.mark.parametrize("manifest_backed", [False, True])
def test_saved_record_recomputation_needs_no_training_or_structure_imports(
    tmp_path: Path, manifest_backed: bool
) -> None:
    from .test_confidence_recompute import _manifest_group

    evaluation_dir = tmp_path / "evaluation"
    if manifest_backed:
        _manifest_group(evaluation_dir)
    else:
        records = _records()
        for model_id in ("esmfold2_300", "esmfold2_600", "esmfold2"):
            source = records
            if model_id == "esmfold2":
                source = [
                    {**record, "predictions": {"production": record["predictions"]["v2"]}}
                    for record in records
                ]
            path = evaluation_dir / model_id / "records.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(source), encoding="utf-8")

    script = """
import importlib.abc
import sys
from pathlib import Path

blocked = {"torch", "transformers", "safetensors", "wandb", "pyarrow", "gemmi", "DockQ", "tmtools"}

class RejectTrainingDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise AssertionError(f"Saved-record calculation imported {fullname}")

sys.meta_path.insert(0, RejectTrainingDependencies())
sys.path.insert(0, sys.argv[1])
from tools.confidence import acceptance, v2_analysis
from tools.confidence.recompute import recompute

acceptance.BOOTSTRAP_SAMPLES = v2_analysis.BOOTSTRAP_SAMPLES = 5
report = recompute(Path(sys.argv[2]), Path(sys.argv[3]))
assert report["refolding"] is False
assert report["bootstrap_samples"] == 5
assert not blocked.intersection(sys.modules)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(Path(__file__).resolve().parents[2]),
            str(evaluation_dir),
            str(tmp_path / "corrected"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    corrected = json.loads(
        (tmp_path / "corrected" / "esmfold2_300" / "summary.json").read_text()
    )
    assert corrected["v2"]["overall"]["within_target_plddt_accuracy"] == 1.0
