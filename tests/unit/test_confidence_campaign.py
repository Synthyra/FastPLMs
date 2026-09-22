"""CPU-only tests for campaign sequencing and resumable progress."""

import pytest

from tools.confidence import campaign


def _install_fakes(monkeypatch, cache_status="complete", overfit_status="passed"):
    calls = []

    monkeypatch.setattr(
        campaign,
        "_records",
        lambda root: [{"id": "a", "split": "train"}, {"id": "b", "split": "final_test"}],
    )

    def fake_cache(root, model_id, **kwargs):
        calls.append(("cache", kwargs))
        return {"status": cache_status, "remaining": 0 if cache_status == "complete" else 1}

    def fake_train(root, model_id, **kwargs):
        calls.append(("train", kwargs))
        status = overfit_status if kwargs["overfit"] else "complete"
        return {"status": status}

    monkeypatch.setattr(campaign, "generate_caches", fake_cache)
    monkeypatch.setattr(campaign, "train_head", fake_train)
    return calls


def test_campaign_runs_cache_overfit_then_training(monkeypatch, tmp_path):
    calls = _install_fakes(monkeypatch)
    commits = []

    report = campaign.run_campaign(
        tmp_path,
        "esmfold2_300",
        volume_commit=lambda: commits.append(True),
    )

    assert report["status"] == "complete"
    assert [name for name, _ in calls] == ["cache", "train", "train"]
    assert calls[0][1]["maximum_targets"] == 3
    assert calls[1][1]["overfit"] is True
    assert calls[2][1]["overfit"] is False
    assert len(commits) >= 5
    assert (tmp_path / "esmfold2_300/campaign-progress.json").exists()


def test_campaign_stops_after_partial_cache(monkeypatch, tmp_path):
    calls = _install_fakes(monkeypatch, cache_status="partial")

    report = campaign.run_campaign(tmp_path, "esmfold2_300")

    assert report["status"] == "partial"
    assert report["reason"] == "cache_incomplete"
    assert len(calls) == 1


def test_campaign_stops_after_failed_overfit(monkeypatch, tmp_path):
    calls = _install_fakes(monkeypatch, overfit_status="failed")

    report = campaign.run_campaign(tmp_path, "esmfold2_300")

    assert report["status"] == "failed"
    assert report["reason"] == "overfit_failed"
    assert len(calls) == 2


def test_campaign_rejects_training_time_above_bound(tmp_path):
    with pytest.raises(ValueError, match="training_seconds must be between"):
        campaign.run_campaign(tmp_path, "esmfold2_300", training_seconds=36_001)
