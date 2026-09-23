"""Fail closed when reusing preparation or publishing campaign artifacts."""

import json

import pytest

from pathlib import Path
from types import SimpleNamespace

from tools.confidence import target_splits, v2_campaign


def prepared_campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "campaign"
    pilot = root / "pilot"
    pilot.mkdir(parents=True)
    (pilot / "records.json").write_text("[]")
    positions = root / "pool/positions"
    positions.mkdir(parents=True)
    (positions / "one.npy").write_bytes(b"coordinates")
    v2_campaign.write_json(root / "prepared.json", {
        "status": "prepared",
        "dataset_revision": v2_campaign.DATASET_REVISION,
        "pilot_files": {"records.json": v2_campaign.file_hash(pilot / "records.json")},
    })
    monkeypatch.setattr(target_splits, "load_split", lambda _: [{"positions_file": "one.npy"}])
    return root


def test_prepared_data_rejects_changed_pilot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = prepared_campaign(tmp_path, monkeypatch)
    assert v2_campaign.validate_prepared(root)["status"] == "prepared"
    (root / "pilot/records.json").write_text("[1]")
    with pytest.raises(ValueError, match="Pilot input changed"):
        v2_campaign.validate_prepared(root)


def test_prepared_data_rejects_missing_coordinates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = prepared_campaign(tmp_path, monkeypatch)
    (root / "pool/positions/one.npy").unlink()
    with pytest.raises(FileNotFoundError, match="coordinates are missing"):
        v2_campaign.validate_prepared(root)


def test_prepared_data_rejects_different_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = prepared_campaign(tmp_path, monkeypatch)
    receipt = json.loads((root / "prepared.json").read_text())
    receipt["dataset_revision"] = "different"
    v2_campaign.write_json(root / "prepared.json", receipt)
    with pytest.raises(ValueError, match="dataset pin"):
        v2_campaign.validate_prepared(root)


def test_public_archive_is_explicit_and_parent_protected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    root = tmp_path / "campaign"
    root.mkdir()
    (root / "report.json").write_text("{}")
    calls = []
    api = SimpleNamespace(
        dataset_info=lambda _: SimpleNamespace(sha="parent"),
        create_commit=lambda **kwargs: calls.append(kwargs) or SimpleNamespace(oid="commit"),
    )
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    assert v2_campaign.publish_files(root, ["report.json"], "report") == "commit"
    assert calls[0]["parent_commit"] == "parent"
    assert calls[0]["repo_type"] == "dataset"
    assert [operation.path_in_repo for operation in calls[0]["operations"]] == ["confidence-v2/campaign/report.json"]
    (tmp_path / "outside.json").write_text("{}")
    with pytest.raises(ValueError, match="Invalid campaign artifact path"):
        v2_campaign.publish_files(root, ["../outside.json"], "report")
    assert len(calls) == 1


def test_completion_requires_every_evaluation_upload(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    for model_id in (*v2_campaign.MODEL_IDS, "esmfold2"):
        v2_campaign.write_json(root / "evaluation" / root.name / model_id / "completion.json", {})
    assert not v2_campaign.evaluations_archived(root)
    for model_id in v2_campaign.MODEL_IDS:
        v2_campaign.write_json(root / "uploads" / f"evaluate-{model_id}.json", {})
    assert not v2_campaign.evaluations_archived(root)
    v2_campaign.write_json(root / "uploads/evaluate-esmfold2.json", {})
    assert v2_campaign.evaluations_archived(root)


@pytest.mark.parametrize(("status_code", "failures", "succeeds", "attempts"), [(412, 1, True, 2), (412, 3, False, 3), (500, 1, False, 1)])
def test_archive_retry_preserves_parent_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status_code: int, failures: int, succeeds: bool, attempts: int) -> None:
    import httpx
    import huggingface_hub

    from huggingface_hub.errors import HfHubHTTPError

    (tmp_path / "report.json").write_text("{}")
    parents = []

    def commit(**kwargs: object) -> SimpleNamespace:
        parents.append(kwargs["parent_commit"])
        if len(parents) <= failures:
            raise HfHubHTTPError("Concurrent writer", response=httpx.Response(status_code, request=httpx.Request("POST", "https://huggingface.co")))
        return SimpleNamespace(oid="archived")

    api = SimpleNamespace(dataset_info=lambda _: SimpleNamespace(sha=f"parent-{len(parents)}"), create_commit=commit)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    monkeypatch.setattr(v2_campaign.time, "sleep", lambda _: None)
    if succeeds:
        assert v2_campaign.publish_files(tmp_path, ["report.json"], "report") == "archived"
    else:
        with pytest.raises(HfHubHTTPError):
            v2_campaign.publish_files(tmp_path, ["report.json"], "report")
        assert not (tmp_path / "uploads/report.json").exists()
    assert parents == [f"parent-{index}" for index in range(attempts)]


def test_export_retry_verifies_existing_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tools.confidence import experiment_artifacts

    destination = tmp_path / "public/evaluation/esmfold2_300"
    destination.mkdir(parents=True)
    (destination / "completion.json").write_text("{}")
    calls = []
    monkeypatch.setattr(experiment_artifacts, "verify_evaluation", lambda path, **kwargs: calls.append((path, kwargs)) or {"model_id": "esmfold2_300", "evaluation_id": tmp_path.name})
    assert v2_campaign.export_evaluation(tmp_path, "esmfold2_300") == ["public/evaluation/esmfold2_300/completion.json"]
    assert calls == [(destination, {"require_checkpoints": False})]


@pytest.mark.parametrize("existing", [None, "queued", "training", "directory", "receipt", "status", "incomplete"])
def test_evaluation_recovery_requires_undispatched_completed_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: str | None) -> None:
    from tools.confidence import experiment_artifacts

    model_id = "esmfold2_300"
    v2_campaign.write_json(tmp_path / "runs" / model_id / "v2/report.json", {
        "status": "complete", "model_id": model_id, "updates": 779 if existing == "incomplete" else 780,
    })
    if existing in {"queued", "training"}:
        status = {"evaluation_call_id": "fc-queued"} if existing == "queued" else {"status": "running"}
        v2_campaign.write_json(tmp_path / "status" / f"train-{model_id}.json", status)
    elif existing == "directory":
        (tmp_path / "evaluation" / tmp_path.name / model_id).mkdir(parents=True)
    elif existing in {"receipt", "status"}:
        name = f"dispatch-evaluate-{model_id}.json" if existing == "receipt" else f"evaluate-{model_id}.json"
        v2_campaign.write_json(tmp_path / "status" / name, {"status": "running"})
    verified = []
    monkeypatch.setattr(experiment_artifacts, "verify_evaluation", lambda *args, **kwargs: verified.append((args, kwargs)))
    if existing is None:
        v2_campaign.validate_evaluation_recovery(tmp_path, model_id)
        assert verified == [((tmp_path / "public/evaluation/esmfold2",), {"require_checkpoints": False})]
    else:
        with pytest.raises(ValueError if existing == "incomplete" else FileExistsError):
            v2_campaign.validate_evaluation_recovery(tmp_path, model_id)
        assert not verified
