"""Evaluation evidence remains reproducible without publishing private checkpoint snapshots."""

import hashlib
import json

from dataclasses import asdict
from pathlib import Path

import pytest

from tools.confidence import host
from tools.confidence.experiment_artifacts import (
    EvaluationArtifacts,
    export_evaluation,
    file_identity,
    validate_evaluation_id,
    verify_evaluation,
    verify_evaluation_group,
    write_new_json,
)


def create_evaluation(tmp_path: Path) -> EvaluationArtifacts:
    checkpoint = tmp_path / "selected.safetensors"
    checkpoint.write_bytes(b"selected checkpoint bytes")
    split_report = tmp_path / "split-report.json"
    split_report.write_text('{"seed": 17}', encoding="utf-8")
    return EvaluationArtifacts.create(
        tmp_path / "evaluation",
        evaluation_id="review-1",
        model_id="esmfold2_300",
        split="test",
        targets=[{"target_id": "first"}, {"target_id": "oom"}],
        head_files={"v2": checkpoint, "donor": None},
        metadata={"inference": {"samples": 2, "seed_offset": 1000}},
        input_files={"split-report.json": split_report},
    )


def write_results(run: EvaluationArtifacts) -> None:
    write_new_json(
        run.directory / "records.json",
        [
            {"target_id": "first", "sample": sample, "predictions": {"v2": {}, "donor": {}}}
            for sample in range(2)
        ],
    )
    write_new_json(run.directory / "skipped.json", ["oom"])
    write_new_json(run.directory / "summary.json", {"v2": {}, "donor": {}})


def test_checkpoint_snapshot_is_exact_even_if_training_checkpoint_changes(tmp_path):
    run = create_evaluation(tmp_path)
    original_bytes = (tmp_path / "selected.safetensors").read_bytes()
    (tmp_path / "selected.safetensors").write_bytes(b"later checkpoint")
    assert run.head_files()["v2"].read_bytes() == original_bytes
    request = json.loads((run.directory / "request.json").read_text())
    assert (
        request["checkpoint_inputs"]["v2"]["sha256"] == hashlib.sha256(original_bytes).hexdigest()
    )
    assert request["test_set_status"] == "spent"
    assert request["new_heldout_evaluation"] is False


def test_export_contains_verified_raw_records_and_excludes_weights(tmp_path):
    run = create_evaluation(tmp_path)
    write_results(run)
    completion = run.complete()
    assert verify_evaluation(run.directory) == completion
    output = tmp_path / "public"
    exported = export_evaluation(run.directory, output)
    assert exported["checkpoint_weights_included"] is False
    assert (output / "records.json").read_bytes() == (run.directory / "records.json").read_bytes()
    assert not list(output.rglob("*.safetensors"))
    assert verify_evaluation(output, require_checkpoints=False) == completion
    with pytest.raises(FileExistsError):
        export_evaluation(run.directory, output)
    with pytest.raises(FileExistsError):
        run.complete()


@pytest.mark.parametrize("filename", ["records.json", "targets.json", "checkpoints/v2.safetensors"])
def test_corrupted_inputs_or_outputs_cannot_be_exported(tmp_path, filename):
    run = create_evaluation(tmp_path)
    write_results(run)
    run.complete()
    (run.directory / filename).write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        export_evaluation(run.directory, tmp_path / "public")
    assert not (tmp_path / "public").exists()


def test_partial_and_failed_runs_never_export_as_complete(tmp_path):
    run = create_evaluation(tmp_path)
    with pytest.raises(FileNotFoundError):
        export_evaluation(run.directory, tmp_path / "partial")
    run.fail(RuntimeError("folding failed"))
    write_results(run)
    with pytest.raises(ValueError, match="failed"):
        run.complete()
    with pytest.raises(ValueError, match="Failed"):
        export_evaluation(run.directory, tmp_path / "failed")
    assert not (run.directory / "completion.json").exists()


def test_completion_rejects_missing_or_duplicate_diffusion_samples(tmp_path):
    run = create_evaluation(tmp_path)
    write_results(run)
    records = run.directory / "records.json"
    saved = json.loads(records.read_text())
    saved[1]["sample"] = 0
    records.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="target/sample"):
        run.complete()


@pytest.mark.parametrize("identifier", ["../escape", ".", "one/two", "a\\b", "", "a:b"])
def test_evaluation_ids_cannot_escape_output_root(identifier):
    with pytest.raises(ValueError, match="Evaluation IDs"):
        validate_evaluation_id(identifier)


def test_existing_output_is_refused_before_model_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "DATA_ROOT", tmp_path)
    output = tmp_path / "evaluation" / "review-1" / "esmfold2_300"
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="new directory"):
        host.stage_evaluate("esmfold2_300", "train", "test", None, "review-1")
    assert list(output.iterdir()) == []


def test_generated_evaluation_ids_are_unique_and_group_outputs_by_model(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "DATA_ROOT", tmp_path)
    first, first_path = host.evaluation_output("test", "esmfold2_300", None)
    second, second_path = host.evaluation_output("test", "esmfold2_300", None)
    assert first != second and first_path != second_path
    shared, reference_path = host.evaluation_output("test", "esmfold2", first)
    assert shared == first and reference_path.parent == first_path.parent


@pytest.mark.parametrize("limit", [0, -1])
def test_evaluation_limit_rejects_empty_or_negative_requests(limit):
    with pytest.raises(ValueError, match="positive"):
        host.evaluation_scope("test", limit, "reference", 5)


@pytest.mark.parametrize("fail_loading", [False, True])
def test_host_reserves_inputs_before_loading_and_records_final_state(
    tmp_path, monkeypatch, fail_loading
):
    from tools.confidence import cache, rollouts, test_evaluation

    monkeypatch.setattr(host, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(host, "LEDGER_PATH", tmp_path / "ledger.json")
    monkeypatch.setattr(host, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(host, "PILOT_DIR", tmp_path / "pilot")
    monkeypatch.setattr(host, "SPLITS_DIR", tmp_path / "splits")
    run_dir = host.RUNS_DIR / "esmfold2_300" / "train"
    run_dir.mkdir(parents=True)
    host.PILOT_DIR.mkdir()
    host.SPLITS_DIR.mkdir()
    write_new_json(
        run_dir / "report.json",
        {
            "status": "complete",
            "model_id": "esmfold2_300",
            "selected_checkpoint": "best-ema.safetensors",
        },
    )
    write_new_json(host.SPLITS_DIR / "split-report.json", {"seed": 17})
    (run_dir / "best-ema.safetensors").write_bytes(b"selected")
    (host.PILOT_DIR / "esmfold2_300-head.safetensors").write_bytes(b"pilot")
    monkeypatch.setattr(host, "split_targets", lambda split: [{"target_id": "one"}])
    output = tmp_path / "evaluation" / "group" / "esmfold2_300"

    def load_model(model_id):
        assert (output / "request.json").is_file()
        if fail_loading:
            raise RuntimeError("model failed to load")
        return object()

    def load_heads(model_id, files):
        assert files["v2"].read_bytes() == b"selected"
        assert files["v2"].parent == output / "checkpoints"
        return files

    def score(model, pool, targets, heads, records_path, log):
        records = [
            {"target_id": "one", "sample": sample, "predictions": {head: {} for head in heads}}
            for sample in range(test_evaluation.EVALUATION_SAMPLES)
        ]
        write_new_json(records_path, records)
        write_new_json(records_path.with_name("skipped.json"), [])
        return records

    monkeypatch.setattr(cache, "load_folding_model", load_model)
    monkeypatch.setattr(rollouts, "use_fast_folding_kernels", lambda model: None)
    monkeypatch.setattr(test_evaluation, "load_heads", load_heads)
    monkeypatch.setattr(test_evaluation, "fold_and_score", score)
    monkeypatch.setattr(
        test_evaluation, "summarize", lambda records, heads: {head: {} for head in heads}
    )
    if fail_loading:
        with pytest.raises(RuntimeError, match="failed to load"):
            host.stage_evaluate("esmfold2_300", "train", "test", None, "group")
        assert (output / "failure.json").is_file()
        assert not (output / "completion.json").exists()
    else:
        host.stage_evaluate("esmfold2_300", "train", "test", None, "group")
        assert verify_evaluation(output)["status"] == "complete"
        request = json.loads((output / "request.json").read_text())
        inference = request["metadata"]["inference"]
        assert inference["parameter_dtype"] == "float32"
        assert inference["fold_autocast_dtype"] == "bfloat16"
        assert inference["head_autocast_dtype"] == "bfloat16"


@pytest.mark.parametrize(
    "omitted", ["request.json", "records.json", "summary.json", "targets.json"]
)
def test_completion_inventory_must_include_all_required_outputs(tmp_path, omitted):
    run = create_evaluation(tmp_path)
    write_results(run)
    completion = run.complete()
    completion["public_files"].pop(omitted)
    (run.directory / "completion.json").write_text(json.dumps(completion), encoding="utf-8")
    with pytest.raises(ValueError, match="omits required"):
        verify_evaluation(run.directory, require_checkpoints=False)


@pytest.mark.parametrize(
    "field,value",
    [("model_id", "esmfold2_600"), ("evaluation_id", "other"), ("split", "validation")],
)
def test_request_and_completion_identity_must_agree(tmp_path, field, value):
    run = create_evaluation(tmp_path)
    write_results(run)
    completion = run.complete()
    completion[field] = value
    (run.directory / "completion.json").write_text(json.dumps(completion), encoding="utf-8")
    with pytest.raises(ValueError, match="Request and completion"):
        verify_evaluation(run.directory, require_checkpoints=False)


def test_completion_cannot_drop_or_replace_an_original_input_identity(tmp_path):
    run = create_evaluation(tmp_path)
    write_results(run)
    completion = run.complete()
    completion["public_files"].pop("inputs/split-report.json")
    (run.directory / "completion.json").write_text(json.dumps(completion), encoding="utf-8")
    with pytest.raises(ValueError, match="public inventory"):
        verify_evaluation(run.directory, require_checkpoints=False)


def create_group_member(
    tmp_path: Path,
    model_id: str,
    *,
    sequence: str = "ACDE",
    coordinate_digest: str = "a" * 64,
    evaluation_id: str = "group",
    split: str = "test",
) -> EvaluationArtifacts:
    target = {
        "target_id": "shared",
        "sequences": [sequence],
        "num_tokens": len(sequence),
        "num_chains": 1,
        "stratum": "monomer_short",
    }
    targets = [target]
    skipped = []
    if model_id == "esmfold2_300":
        # The reference need not have the same requested targets or OOM skips.
        targets.append({**target, "target_id": "candidate_only"})
        skipped.append("candidate_only")
    run = EvaluationArtifacts.create(
        tmp_path / model_id,
        evaluation_id=evaluation_id,
        model_id=model_id,
        split=split,
        targets=targets,
        head_files={},
        metadata={"inference": {"samples": 2}},
        input_files={},
    )
    write_new_json(
        run.directory / "records.json",
        [
            {
                "target_id": "shared",
                "sample": sample,
                "stratum": "monomer_short",
                "num_chains": 1,
                "num_tokens": len(sequence),
                "predictions": {"production": {}},
                "target_positions": {
                    "sha256": coordinate_digest,
                    "shape": [len(sequence), 14, 3],
                    "dtype": "float32",
                },
            }
            for sample in range(2)
        ],
    )
    write_new_json(run.directory / "skipped.json", skipped)
    write_new_json(run.directory / "summary.json", {})
    run.complete()
    return run


def test_paired_group_allows_distinct_skips_and_requested_subsets(tmp_path):
    candidate = create_group_member(tmp_path, "esmfold2_300")
    reference = create_group_member(tmp_path, "esmfold2")
    result = verify_evaluation_group(
        {
            "esmfold2_300": candidate.directory,
            "esmfold2": reference.directory,
        }
    )
    assert set(result) == {"esmfold2_300", "esmfold2"}


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"sequence": "AAAA"}, "Biological identity"),
        ({"coordinate_digest": "b" * 64}, "Native coordinates"),
        ({"evaluation_id": "other"}, "different result groups"),
        ({"split": "validation"}, "requested split"),
    ],
)
def test_paired_group_rejects_incoherent_experiments(tmp_path, changes, message):
    candidate = create_group_member(tmp_path, "esmfold2_300")
    reference = create_group_member(tmp_path, "esmfold2", **changes)
    with pytest.raises(ValueError, match=message):
        verify_evaluation_group(
            {
                "esmfold2_300": candidate.directory,
                "esmfold2": reference.directory,
            }
        )


def test_paired_group_rejects_model_swap(tmp_path):
    candidate = create_group_member(tmp_path, "esmfold2_300")
    reference = create_group_member(tmp_path, "esmfold2")
    with pytest.raises(ValueError, match="model identity"):
        verify_evaluation_group(
            {
                "esmfold2_300": reference.directory,
                "esmfold2": candidate.directory,
            }
        )


def test_paired_group_requires_one_native_coordinate_identity_per_target(tmp_path):
    candidate = create_group_member(tmp_path, "esmfold2_300")
    reference = create_group_member(tmp_path, "esmfold2")
    records_path = candidate.directory / "records.json"
    records = json.loads(records_path.read_text())
    records[1]["target_positions"]["sha256"] = "b" * 64
    records_path.write_text(json.dumps(records), encoding="utf-8")
    completion_path = candidate.directory / "completion.json"
    completion = json.loads(completion_path.read_text())
    completion["public_files"]["records.json"] = asdict(file_identity(records_path))
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    with pytest.raises(ValueError, match="changes between samples"):
        verify_evaluation_group(
            {
                "esmfold2_300": candidate.directory,
                "esmfold2": reference.directory,
            }
        )


@pytest.mark.parametrize(
    "field,value", [("stratum", "long"), ("num_chains", 2), ("num_tokens", 123)]
)
def test_paired_group_rejects_record_biology_disagreeing_with_target(tmp_path, field, value):
    candidate = create_group_member(tmp_path, "esmfold2_300")
    reference = create_group_member(tmp_path, "esmfold2")
    records_path = candidate.directory / "records.json"
    records = json.loads(records_path.read_text())
    records[0][field] = value
    records_path.write_text(json.dumps(records), encoding="utf-8")
    completion_path = candidate.directory / "completion.json"
    completion = json.loads(completion_path.read_text())
    completion["public_files"]["records.json"] = asdict(file_identity(records_path))
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    with pytest.raises(ValueError, match="Record biological fields"):
        verify_evaluation_group(
            {
                "esmfold2_300": candidate.directory,
                "esmfold2": reference.directory,
            }
        )
