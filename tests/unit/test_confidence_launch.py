"""CPU-only tests for bounded confidence dispatch bookkeeping."""

from tools.confidence.budget import BudgetLedger
from tools.confidence.launch import _reserve_workers, _stage_name, _timeout


def test_parallel_reservations_use_distinct_models(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json")
    reservations = _reserve_workers(
        ledger,
        app_id="controller",
        stage="train",
        models=["esmfold2_300", "esmfold2_600"],
        gpu="H100",
        timeout=36_600,
        cpu=False,
        count=2,
    )

    assert [item["model"] for item in reservations] == ["esmfold2_300", "esmfold2_600"]
    assert [item["worker_index"] for item in ledger.entries] == [0, 1]
    assert all(item["gpu"] == "H100" for item in ledger.entries)


def test_stage_timeout_contract_is_bounded():
    assert _timeout("train", cpu=False) == 36_600
    assert _timeout("benchmark", cpu=False) == 600
    assert _timeout("tests", cpu=True) == 1_800
    assert _stage_name("benchmark") == "prepare"
