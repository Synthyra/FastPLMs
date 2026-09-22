"""Assemble backend-bench receipts into the evidence file an automatic order cites.

The manifest may prefer FlashAttention for ``attn_implementation="auto"`` only when
``attention_auto_evidence`` names a measurement. This module copies the measured
numbers out of passed ``backend-bench`` runs, so they are never retyped by hand.
"""

from __future__ import annotations

import argparse
import json

from pathlib import Path
from typing import Any

from .lever_bench import BATCH_ROW_LENGTHS, MASKED_BACKENDS
from .source import ROOT


EVIDENCE_PATH = ROOT / "docs/evidence/attention/backend_latency.json"
_METHOD = (
    "Randomly initialized models at published dimensions, FP32 parameters under CUDA BF16 "
    "autocast, inference mode. Each backend runs in its own process, three interleaved rounds "
    "per case; a round reports the median forward latency after warm-up. One batch is 8 rows "
    "of 512 positions."
)


def _report(output: str) -> dict[str, Any]:
    """Return the final JSON object that the bench driver printed."""
    report: dict[str, Any] = json.loads(output[output.rindex("\n{\n") :])
    return report


def backend_evidence(run_directories: list[Path]) -> dict[str, Any]:
    runs = []
    for directory in run_directories:
        receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
        if receipt["stage"] != "backend-bench" or receipt["status"] != "passed":
            raise ValueError(f"{directory.name} is not a passed backend-bench run.")
        report = _report((directory / "output.txt").read_text(encoding="utf-8"))
        expected_cases = {
            f"{family}-{kind}" for family in MASKED_BACKENDS for kind in BATCH_ROW_LENGTHS
        }
        if set(report) != expected_cases:
            raise ValueError(f"{directory.name} does not cover the current bench cases.")
        runs.append(
            {
                "run": directory.name,
                "source": receipt["source"],
                "environment": receipt["result"]["environment"],
                "cases": {
                    case: {
                        "median_ms": measured["median_ms"],
                        "speedup_over_sdpa": measured["speedup_over_sdpa"],
                        "failures": measured["failures"],
                    }
                    for case, measured in sorted(report.items())
                },
            }
        )
    return {
        "schema_version": 1,
        "method": _METHOD,
        "residues_per_batch": {kind: sum(rows) for kind, rows in BATCH_ROW_LENGTHS.items()},
        "limitations": (
            "Descriptive latency of an uncommitted working tree on cloud workers. It orders "
            "backends for automatic selection and is not a release benchmark or a parity claim."
        ),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directories", nargs="+", type=Path)
    args = parser.parse_args()
    evidence = backend_evidence(args.run_directories)
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # The repository stores LF; without ``newline`` Windows would write CRLF.
    EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(EVIDENCE_PATH.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
