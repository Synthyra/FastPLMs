"""Shared run-role tags and explicit experiment-group defaults for W&B."""

from __future__ import annotations

import re

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


# An experiment id, `YYYY-MM-DD_slug`, the form `ws capture` reads a run's group back by.
EXPERIMENT_ID = re.compile(r"\d{4}-\d{2}-\d{2}_[a-z0-9_]+")


def role_for_job(job_type: str | None) -> str:
    """Classify a recorded job type without rewriting its scientific meaning."""
    job = (job_type or "").lower()
    if job == "fit_tune" or "sweep" in job or "hpo" in job:
        return "tuning"
    if any(term in job for term in ("diagnostic", "profile", "smoke", "overfit")):
        return "diagnostic"
    if any(term in job for term in ("eval", "benchmark", "reference")):
        return "evaluation"
    if job in {"fit_final", "train", "training", "release", "package"}:
        return "final-fit" if job == "fit_final" else job
    return "unclassified"


def organization_tags(tags: Sequence[str], job_type: str | None) -> list[str]:
    """Keep caller tags and supply missing lifecycle and role classifications."""
    organized = list(dict.fromkeys(tags))
    if not any(tag.startswith("lifecycle:") for tag in organized):
        organized.append("lifecycle:active")
    if not any(tag.startswith("role:") for tag in organized):
        organized.append(f"role:{role_for_job(job_type)}")
    return organized


def experiment_group(directory: Path, config: Mapping[str, Any]) -> str | None:
    """Use an explicit experiment id or an enclosing dated experiment directory."""
    supplied = config.get("experiment_id")
    if isinstance(supplied, str) and EXPERIMENT_ID.fullmatch(supplied):
        return supplied
    for candidate in (directory.name, directory.parent.name):
        if EXPERIMENT_ID.fullmatch(candidate):
            return candidate
    return None
