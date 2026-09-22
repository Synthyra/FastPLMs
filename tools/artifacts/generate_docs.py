"""Generate model support data and model cards from the typed manifest."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from tools.artifacts.doc_generation.esmc_evidence import (
    EsmcReportError,
)
from tools.artifacts.doc_generation.model_cards import (
    render_model_card,
)
from tools.artifacts.doc_generation.outputs import (
    expected_outputs,
    synchronize,
)
from tools.artifacts.doc_generation.support import (
    render_capability_evidence,
    render_support,
)


# Retained entry points used by callers predating the package split.
__all__ = [
    "expected_outputs",
    "main",
    "render_capability_evidence",
    "render_model_card",
    "render_support",
    "synchronize",
]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--esmc-report-root",
        type=Path,
        help=(
            "strictly validate and render one explicit complete 30-record schema-v3 "
            "ESMC release-evidence set"
        ),
    )
    parser.add_argument(
        "--require-esmc-release-evidence",
        action="store_true",
        help=(
            "require release evidence from --esmc-report-root, "
            "FASTPLMS_DIAGNOSTIC_REPORTS, or artifacts/diagnostics/esmc"
        ),
    )
    arguments = parser.parse_args(argv)
    try:
        failures = synchronize(
            arguments.source_root.resolve(),
            check=arguments.check,
            esmc_report_root=arguments.esmc_report_root,
            require_esmc_release_evidence=arguments.require_esmc_release_evidence,
        )
    except EsmcReportError as error:
        print(f"invalid ESMC release evidence: {error}")
        return 1
    if failures:
        for failure in failures:
            print(failure)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
