"""W&B lifecycle contracts for prediction caching before optimization."""

import pytest

from types import SimpleNamespace

from tools.confidence import training


@pytest.mark.parametrize("fail", [False, True])
def test_cache_run_starts_before_work_and_records_outcome(monkeypatch, tmp_path, fail):
    events = []
    run = SimpleNamespace(
        url="https://wandb.ai/test/cache",
        summary={},
        finish=lambda **kwargs: events.append(("finish", kwargs["exit_code"])),
    )

    def start(*args):
        events.append(("start", args[2]))
        return run

    def cache(*args):
        events.append(("cache", args[-1] is run))
        if fail:
            raise RuntimeError("cache device mismatch")
        return {"status": "partial", "remaining": 3}

    monkeypatch.setattr(training, "_training_settings", lambda *args: {})
    monkeypatch.setattr(training, "_wandb_run", start)
    monkeypatch.setattr(training, "_generate_caches", cache)
    if fail:
        with pytest.raises(RuntimeError, match="cache device mismatch"):
            training.generate_caches(tmp_path, "esmfold2_300")
        assert run.summary["status"] == "failed"
        assert run.summary["error_type"] == "RuntimeError"
    else:
        report = training.generate_caches(tmp_path, "esmfold2_300")
        assert report["wandb_url"] == run.url
        assert run.summary == {"status": "partial", "remaining": 3}
    assert events == [("start", "cache"), ("cache", True), ("finish", int(fail))]
