"""Apply one review-status policy before confidence metrics reach documentation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tools.artifacts.evidence_store import load_manifest


CONFIDENCE_RESEARCH_EVIDENCE = {
    "esmfold2_300": "docs/evidence/confidence/esmfold2_300-v2.json",
    "esmfold2_600": "docs/evidence/confidence/esmfold2_600-v2.json",
}
REPORTABLE_STATUSES = frozenset({"legacy_pilot", "validated", "corrected_test_metrics"})
WITHHELD_STATUSES = frozenset({"unreviewed", "pending", "invalidated", "requires_recomputation"})


@dataclass(frozen=True)
class ConfidenceEvidence:
    """Parsed evidence with an explicit decision about publishing its metrics."""

    payload: Mapping[str, object]
    review: Mapping[str, object]
    review_status: str

    @property
    def can_report_metrics(self) -> bool:
        return self.review_status in REPORTABLE_STATUSES


def load_confidence_evidence(
    path: Path,
    *,
    protocol: Literal["pilot", "v2"],
    evidence_root: Path | None = None,
) -> ConfidenceEvidence:
    """Keep legacy pilot evidence usable while requiring review of v2 metrics."""

    encoded = path.read_bytes()
    if evidence_root is not None and (evidence_root / "evidence.toml").is_file():
        store = load_manifest(evidence_root / "evidence.toml")
        relative = path.absolute().relative_to(evidence_root.absolute()).as_posix()
        entry = next((entry for entry in store.files if entry.path == relative), None)
        if entry is not None:
            if len(encoded) != entry.size:
                raise ValueError(f"Confidence evidence size mismatch: {relative}")
            if hashlib.sha256(encoded).hexdigest() != entry.sha256:
                raise ValueError(f"Confidence evidence SHA-256 mismatch: {relative}")
    # Decode the verified bytes so a later file replacement cannot change the reviewed payload.
    payload = json.loads(encoded.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Confidence evidence must be a JSON object: {path}")
    if "metrics_review" not in payload:
        status = "legacy_pilot" if protocol == "pilot" else "unreviewed"
        return ConfidenceEvidence(payload, {}, status)
    review = payload["metrics_review"]
    if not isinstance(review, dict):
        raise ValueError(f"Confidence metrics_review must be a JSON object: {path}")
    status = review.get("status")
    if not isinstance(status, str) or status not in REPORTABLE_STATUSES | WITHHELD_STATUSES:
        raise ValueError(f"Unknown confidence metrics review status {status!r}: {path}")
    if protocol == "v2" and status == "legacy_pilot":
        raise ValueError(f"V2 confidence evidence cannot use legacy pilot review status: {path}")
    return ConfidenceEvidence(payload, review, status)
