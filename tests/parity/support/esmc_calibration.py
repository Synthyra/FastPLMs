"""Compact, immutable biological sequences for ESMC backend calibration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "esmc_biological_holdout.json"
CASE_IDS = ("1crn-a", "1pga-a", "5pti-a", "1ubq-a", "1ema-a")
ESMC_BOUNDARY_LENGTHS = (13, 15, 16, 17, 29, 31, 32, 33, 61, 127, 128, 129)
CANONICAL_AAS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def load_esmc_biological_holdout(
    path: Path = FIXTURE_PATH,
) -> tuple[dict[str, str], ...]:
    """Load the five pinned sequence/source pairs and fail closed on drift."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported ESMC biological-holdout schema")
    cases = payload.get("cases")
    if not isinstance(cases, list) or tuple(case.get("case_id") for case in cases) != CASE_IDS:
        raise ValueError("ESMC biological-holdout inventory differs from the release contract")
    for case in cases:
        _validate_case(case)
    return tuple(dict(case) for case in cases)


def _validate_case(case: Mapping[str, Any]) -> None:
    case_id = str(case.get("case_id"))
    expected_fields = {
        "case_id",
        "sequence",
        "sequence_sha256",
        "source",
        "source_sha256",
    }
    if set(case) != expected_fields:
        raise ValueError(f"{case_id}: biological-holdout fields differ from the contract")
    sequence = case.get("sequence")
    if (
        not isinstance(sequence, str)
        or not sequence
        or not sequence.isupper()
        or not set(sequence).issubset(CANONICAL_AAS)
    ):
        raise ValueError(f"{case_id}: sequence must contain canonical uppercase amino acids")
    sequence_sha256 = hashlib.sha256(sequence.encode("ascii")).hexdigest()
    if case.get("sequence_sha256") != sequence_sha256:
        raise ValueError(f"{case_id}: sequence digest mismatch")
    source = case.get("source")
    if not isinstance(source, str) or not source.startswith("RCSB "):
        raise ValueError(f"{case_id}: source must identify an RCSB chain")
    source_sha256 = case.get("source_sha256")
    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        raise ValueError(f"{case_id}: source digest is not pinned")
    try:
        bytes.fromhex(source_sha256)
    except ValueError as error:
        raise ValueError(f"{case_id}: source digest is not hexadecimal") from error


__all__ = [
    "CASE_IDS",
    "ESMC_BOUNDARY_LENGTHS",
    "FIXTURE_PATH",
    "load_esmc_biological_holdout",
]
