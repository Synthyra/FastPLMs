"""Run the resumable confidence cache, overfit, and training campaign."""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any
from collections.abc import Callable

from .config import TRAINING_MAXIMUM_SECONDS
from .training import _records, generate_caches, train_head


DEFAULT_CACHE_SECONDS = 7_200
DEFAULT_OVERFIT_SECONDS = 1_800
DEFAULT_TRAINING_SECONDS = 36_000


def _write_progress(
    path: Path,
    progress: dict[str, Any],
    volume_commit: Callable[[], None] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
    if volume_commit is not None:
        volume_commit()


def _finish(
    path: Path,
    progress: dict[str, Any],
    status: str,
    volume_commit: Callable[[], None] | None = None,
    **details: Any,
) -> dict[str, Any]:
    progress.update(status=status, **details)
    _write_progress(path, progress, volume_commit)
    return progress


def _run_stage(
    name: str,
    function: Callable[..., dict[str, Any]],
    *args: Any,
    **kwargs: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    print(f"[campaign] starting {name}", flush=True)
    try:
        report = function(*args, **kwargs)
    except Exception as error:
        return None, {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    print(f"[campaign] {name}: {report.get('status', 'unknown')}", flush=True)
    return report, None


def run_campaign(
    root: Path,
    model_id: str,
    cache_seconds: int = DEFAULT_CACHE_SECONDS,
    overfit_seconds: int = DEFAULT_OVERFIT_SECONDS,
    training_seconds: int = DEFAULT_TRAINING_SECONDS,
    volume_commit: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Run all approved stages, retaining progress for a later invocation."""
    if cache_seconds <= 0 or cache_seconds > DEFAULT_CACHE_SECONDS:
        raise ValueError(f"cache_seconds must be between 1 and {DEFAULT_CACHE_SECONDS}")
    if overfit_seconds <= 0 or overfit_seconds > DEFAULT_OVERFIT_SECONDS:
        raise ValueError(f"overfit_seconds must be between 1 and {DEFAULT_OVERFIT_SECONDS}")
    if training_seconds <= 0 or training_seconds > TRAINING_MAXIMUM_SECONDS:
        raise ValueError(f"training_seconds must be between 1 and {TRAINING_MAXIMUM_SECONDS}")
    if cache_seconds + overfit_seconds + training_seconds > 45_000:
        raise ValueError("Campaign stage time bounds exceed 45,000 seconds")
    progress_path = root / model_id / "campaign-progress.json"
    progress: dict[str, Any] = {
        "status": "running",
        "model_id": model_id,
        "cache_seconds": cache_seconds,
        "overfit_seconds": overfit_seconds,
        "training_seconds": training_seconds,
        "stages": {},
    }
    _write_progress(progress_path, progress, volume_commit)

    try:
        records = _records(root)
        maximum_targets = sum(2 if record["split"] == "final_test" else 1 for record in records)
        progress["cache_targets"] = maximum_targets
        _write_progress(progress_path, progress, volume_commit)

        cache_report, error = _run_stage(
            "cache",
            generate_caches,
            root,
            model_id,
            maximum_targets=maximum_targets,
            maximum_seconds=cache_seconds,
        )
        progress["stages"]["cache"] = cache_report or error
        _write_progress(progress_path, progress, volume_commit)
        if error:
            return _finish(
                path=progress_path,
                progress=progress,
                status="failed",
                reason="cache_error",
                volume_commit=volume_commit,
            )
        if cache_report["status"] == "partial":
            return _finish(
                path=progress_path,
                progress=progress,
                status="partial",
                reason="cache_incomplete",
                volume_commit=volume_commit,
            )
        if cache_report["status"] != "complete":
            return _finish(
                path=progress_path,
                progress=progress,
                status="failed",
                reason="cache_failed",
                volume_commit=volume_commit,
            )

        overfit_report, error = _run_stage(
            "overfit",
            train_head,
            root,
            model_id,
            maximum_seconds=overfit_seconds,
            overfit=True,
        )
        progress["stages"]["overfit"] = overfit_report or error
        _write_progress(progress_path, progress, volume_commit)
        if error:
            return _finish(
                path=progress_path,
                progress=progress,
                status="failed",
                reason="overfit_error",
                volume_commit=volume_commit,
            )
        if overfit_report["status"] != "passed":
            return _finish(
                path=progress_path,
                progress=progress,
                status="failed",
                reason="overfit_failed",
                volume_commit=volume_commit,
            )

        training_report, error = _run_stage(
            "training",
            train_head,
            root,
            model_id,
            maximum_seconds=training_seconds,
            overfit=False,
        )
        progress["stages"]["training"] = training_report or error
        if error:
            return _finish(
                path=progress_path,
                progress=progress,
                status="failed",
                reason="training_error",
                volume_commit=volume_commit,
            )
        return _finish(
            path=progress_path,
            progress=progress,
            status=training_report["status"],
            volume_commit=volume_commit,
        )
    except Exception as error:
        return _finish(
            path=progress_path,
            progress=progress,
            status="failed",
            reason="campaign_error",
            error_type=type(error).__name__,
            error=str(error),
            volume_commit=volume_commit,
        )
