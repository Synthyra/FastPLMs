"""Persist cost reservations and remote identifiers for bounded remote work."""

from __future__ import annotations

import json
import math
import uuid

from datetime import datetime, UTC
from pathlib import Path



class BudgetExceeded(RuntimeError):
    """A dispatch bound prevented a requested reservation."""


class BudgetLedger:
    """Durable single-controller accounting for uncertain remote work.

    Reservations intentionally have no dollar ceiling. A reservation remains
    committed until a receipt is written, so a controller restart cannot
    accidentally release spend for a call whose result is unknown.
    """

    def __init__(self, path: Path, *, stages: tuple[str, ...] = ()) -> None:
        self.stages = stages
        self.path = path
        self.entries = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        if not isinstance(self.entries, list):
            raise ValueError("Budget ledger must contain a JSON list")

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.entries, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    def committed(self, stage: str | None = None) -> float:
        """Return reserved cost, using observed cost after completion."""
        costs = [
            float(entry.get("actual_dollars", entry["reserved_dollars"]))
            for entry in self.entries
            if stage is None or entry["stage"] == stage
        ]
        if any(not math.isfinite(cost) or cost < 0 for cost in costs):
            raise ValueError("Budget ledger costs must be nonnegative and finite")
        total = sum(costs)
        if not math.isfinite(total):
            raise ValueError("Committed budget must be finite")
        return total

    def reserve(
        self,
        stage: str,
        description: str,
        dollars: float,
        *,
        app_id: str | None = None,
        call_id: str | None = None,
        worker_index: int | None = None,
        timeout_seconds: int | None = None,
        gpu: str | None = None,
    ) -> str:
        """Record spend before dispatching a call and return its reservation ID."""
        if not stage or not description:
            raise ValueError("A reservation needs a stage and description")
        if not math.isfinite(dollars) or dollars <= 0:
            raise ValueError("A reservation needs a positive finite cost")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("A timeout must be positive")
        identifier = uuid.uuid4().hex
        entry: dict[str, object] = {
            "id": identifier,
            "stage": stage,
            "description": description,
            "reserved_dollars": dollars,
            "status": "reserved",
            "created_at": datetime.now(UTC).isoformat(),
        }
        optional = {
            "app_id": app_id,
            "call_id": call_id,
            "worker_index": worker_index,
            "timeout_seconds": timeout_seconds,
            "gpu": gpu,
        }
        entry.update({key: value for key, value in optional.items() if value is not None})
        self.entries.append(entry)
        self._save()
        return identifier

    def attach_call(
        self,
        identifier: str,
        remote_call_id: str | None,
        *,
        app_id: str | None = None,
    ) -> None:
        """Persist the provider call ID after Modal accepts a dispatched call."""
        entry = self._find(identifier)
        if entry["status"] not in {"reserved", "dispatched"}:
            raise ValueError("Reservation was already completed")
        if remote_call_id:
            entry.update(remote_call_id=remote_call_id, status="dispatched")
        if app_id:
            entry["app_id"] = app_id
        self._save()

    def complete(self, identifier: str, observed_dollars: float) -> None:
        """Write an observed receipt without releasing an unknown reservation."""
        if not math.isfinite(observed_dollars) or observed_dollars < 0:
            raise ValueError("Observed cost must be nonnegative and finite")
        entry = self._find(identifier)
        if entry["status"] not in {"reserved", "dispatched"}:
            raise ValueError("Reservation was already completed")
        entry.update(actual_dollars=observed_dollars, status="completed")
        self._save()

    def _find(self, identifier: str) -> dict[str, object]:
        for entry in self.entries:
            if entry.get("id") == identifier:
                return entry
        raise ValueError(f"Unknown reservation: {identifier}")

    def summary(self) -> dict[str, object]:
        stages = sorted({*self.stages, *(str(entry["stage"]) for entry in self.entries)})
        return {
            "estimated_committed_dollars": self.committed(),
            "limit_dollars": None,
            "contingency_dollars": None,
            "stages": {stage: self.committed(stage) for stage in stages},
            "reservations": self.entries,
        }
