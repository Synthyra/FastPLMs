"""Compact, immutable biological sequences for ESMC backend calibration."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "esmc_biological_holdout.json"
CASE_IDS = ("1crn-a", "1pga-a", "5pti-a", "1ubq-a", "1ema-a")
ESMC_BOUNDARY_LENGTHS = (13, 15, 16, 17, 29, 31, 32, 33, 61, 127, 128, 129)
ESMC_CALIBRATION_SEED = 42
ESMC_PANEL_DEFINITION_SCHEMA_VERSION = 1
CANONICAL_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
CANONICAL_AAS = frozenset(CANONICAL_AA_ALPHABET)
PANEL_KINDS = ("generated_kernel_boundary", "real_biological_holdout")


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


def _sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def generated_esmc_boundary_cases() -> tuple[dict[str, object], ...]:
    """Return the exact seed-locked kernel-boundary sequence panel."""

    generator = random.Random(ESMC_CALIBRATION_SEED)
    cases: list[dict[str, object]] = []
    for length in ESMC_BOUNDARY_LENGTHS:
        sequence = "M" + "".join(generator.choices(CANONICAL_AA_ALPHABET, k=length - 1))
        cases.append(
            {
                "case_id": f"generated-boundary-{length}",
                "sequence": sequence,
                "sequence_length": length,
                "sequence_sha256": _sequence_sha256(sequence),
            }
        )
    return tuple(cases)


def biological_esmc_holdout_cases() -> tuple[dict[str, object], ...]:
    """Return the exact source- and sequence-pinned biological panel."""

    return tuple(
        {
            **case,
            "sequence_length": len(case["sequence"]),
        }
        for case in load_esmc_biological_holdout()
    )


def esmc_calibration_batches() -> tuple[dict[str, object], ...]:
    """Build the two native-reference batches without embedding their digest."""

    return (
        {
            "kind": "generated_kernel_boundary",
            "seed": ESMC_CALIBRATION_SEED,
            "cases": [dict(case) for case in generated_esmc_boundary_cases()],
        },
        {
            "kind": "real_biological_holdout",
            "seed": ESMC_CALIBRATION_SEED,
            "cases": [dict(case) for case in biological_esmc_holdout_cases()],
        },
    )


def validate_esmc_calibration_batch(batch: Mapping[str, Any]) -> dict[str, object]:
    """Validate one immutable panel and return its canonical identity and digest."""

    if set(batch) != {"kind", "seed", "cases"}:
        raise ValueError("ESMC calibration batch fields differ from the release contract")
    kind = batch.get("kind")
    if kind not in PANEL_KINDS:
        raise ValueError(f"Unsupported ESMC calibration panel: {kind!r}")
    if batch.get("seed") != ESMC_CALIBRATION_SEED:
        raise ValueError("ESMC calibration seed differs from the release contract")
    expected = next(item for item in esmc_calibration_batches() if item["kind"] == kind)
    if batch != expected:
        raise ValueError(f"ESMC calibration panel {kind!r} differs from the release contract")
    expected_cases = expected["cases"]
    if not isinstance(expected_cases, list):
        raise ValueError(f"ESMC calibration panel {kind!r} cases are not an ordered list")

    definition = {
        "schema_version": ESMC_PANEL_DEFINITION_SCHEMA_VERSION,
        "kind": kind,
        "seed": ESMC_CALIBRATION_SEED,
        "cases": [dict(case) for case in expected_cases],
    }
    encoded = json.dumps(
        definition,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        **definition,
        "definition_sha256": hashlib.sha256(encoded).hexdigest(),
    }


__all__ = [
    "CANONICAL_AA_ALPHABET",
    "CASE_IDS",
    "ESMC_BOUNDARY_LENGTHS",
    "ESMC_CALIBRATION_SEED",
    "ESMC_PANEL_DEFINITION_SCHEMA_VERSION",
    "FIXTURE_PATH",
    "PANEL_KINDS",
    "biological_esmc_holdout_cases",
    "esmc_calibration_batches",
    "generated_esmc_boundary_cases",
    "load_esmc_biological_holdout",
    "validate_esmc_calibration_batch",
]
