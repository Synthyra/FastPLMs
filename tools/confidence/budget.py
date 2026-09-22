"""Confidence-pilot accounting with its historical stage summary."""

from __future__ import annotations

from pathlib import Path

from tools.execution.budget import BudgetExceeded as BudgetExceeded
from tools.execution.budget import BudgetLedger as ExecutionBudgetLedger
from .config import STAGES


class BudgetLedger(ExecutionBudgetLedger):
    """Preserve the confidence summary, including stages with no reservations."""

    def __init__(self, path: Path) -> None:
        super().__init__(path, stages=STAGES)
