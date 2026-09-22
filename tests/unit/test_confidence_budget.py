"""Cost accounting survives process failure without releasing uncertain spend."""

import pytest

from tools.confidence.budget import BudgetLedger


def test_reservations_survive_reload_and_complete(tmp_path):
    path = tmp_path / "budget.json"
    ledger = BudgetLedger(path)
    identifier = ledger.reserve("prepare", "download", 4.0)
    resumed = BudgetLedger(path)
    assert resumed.committed() == 4.0
    resumed.complete(identifier, 1.5)
    assert BudgetLedger(path).committed() == 1.5


def test_parallel_reservations_have_durable_controller_metadata(tmp_path):
    path = tmp_path / "budget.json"
    ledger = BudgetLedger(path)
    first = ledger.reserve(
        "train",
        "bounded training/0",
        500.0,
        app_id="app-1",
        call_id="call-1",
        worker_index=0,
        timeout_seconds=36_600,
        gpu="H100",
    )
    second = ledger.reserve(
        "train",
        "bounded training/1",
        500.0,
        app_id="app-1",
        call_id="call-2",
        worker_index=1,
        timeout_seconds=36_600,
        gpu="H100",
    )
    assert first != second
    resumed = BudgetLedger(path)
    resumed.attach_call(first, "modal-call-1")
    entry = next(item for item in resumed.entries if item["id"] == first)
    assert entry["app_id"] == "app-1"
    assert entry["call_id"] == "call-1"
    assert entry["remote_call_id"] == "modal-call-1"
    assert entry["status"] == "dispatched"
    assert resumed.summary()["limit_dollars"] is None


def test_invalid_and_duplicate_receipts(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json")
    with pytest.raises(ValueError):
        ledger.reserve("prepare", "invalid", float("nan"))
    identifier = ledger.reserve("prepare", "first", 1.0)
    ledger.complete(identifier, 0.5)
    with pytest.raises(ValueError):
        ledger.complete(identifier, 0.1)
