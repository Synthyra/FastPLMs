"""GH200 stage bookkeeping: the GPU-hour ledger records every stage and enforces its budget."""

import json
import pytest

from tools.confidence import host


def test_budget_records_time_even_when_a_stage_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "LEDGER_PATH", tmp_path / "ledger.json")
    with host.gpu_budget("smoke", "first", budget_hours=1.0) as remaining_seconds:
        assert remaining_seconds == pytest.approx(3600)
    with pytest.raises(ZeroDivisionError), host.gpu_budget("smoke", "failing", budget_hours=1.0):
        1 / 0  # noqa: B018 - the stage body raises
    entries = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert [entry["stage"] for entry in entries] == ["first", "failing"]
    assert all(entry["seconds"] >= 0 for entry in entries)


def test_budget_refuses_to_start_once_spent_and_is_tracked_per_category(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.json"
    monkeypatch.setattr(host, "LEDGER_PATH", ledger)
    ledger.write_text(json.dumps([{"category": "smoke", "stage": "earlier", "started": 0.0, "seconds": 5400.0}]), encoding="utf-8")
    with host.gpu_budget("smoke", "next", budget_hours=2.0) as remaining_seconds:
        assert remaining_seconds == pytest.approx(1800)
    ledger.write_text(json.dumps([{"category": "smoke", "stage": "earlier", "started": 0.0, "seconds": 7200.0}]), encoding="utf-8")
    with pytest.raises(RuntimeError, match="smoke has used"), host.gpu_budget("smoke", "refused", budget_hours=2.0):
        pass
    with host.gpu_budget("model-esmfold2_300", "train", budget_hours=24.0) as remaining_seconds:
        assert remaining_seconds == pytest.approx(24 * 3600)
