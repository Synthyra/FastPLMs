"""Launch one bounded Modal evidence stage and record its receipt and cost."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
import uuid
import xml.etree.ElementTree as ElementTree

from typing import Any

from tools.execution.budget import BudgetExceeded, BudgetLedger
from tools.stored_files import write_stored_json
from .config import (
    DEFAULT_GPU,
    DEFAULT_MAX_DOLLARS,
    ENVIRONMENT_PROBE_TIMEOUT_SECONDS,
    GPU_CHOICES,
    STARTUP_TIMEOUT_SECONDS,
    worker_rate,
)
from .source import ROOT, export_baseline_source, is_git_checkout, stage_upload_source
from .stages import STAGES, StageSpec, stage_arguments


ARTIFACT_ROOT = ROOT / "artifacts/gpu_evidence"
_JUNIT_COUNTS = ("tests", "failures", "errors", "skipped")
# The message tests/cpu/conftest.py reports when a test outlives its wall-clock budget.
_TIME_BUDGET_MARKER = "CPU contract exceeded its"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=tuple(STAGES))
    parser.add_argument("--gpu", choices=GPU_CHOICES, default=DEFAULT_GPU)
    parser.add_argument("--select", default=None, help="pytest -k expression for a focused run")
    parser.add_argument("--max-dollars", type=float, default=DEFAULT_MAX_DOLLARS)
    return parser


def reservation_dollars(spec: StageSpec, gpu: str | None) -> float:
    """Upper bound for one dispatch: startup, the environment probe, then the stage."""
    bounded_seconds = (
        STARTUP_TIMEOUT_SECONDS + ENVIRONMENT_PROBE_TIMEOUT_SECONDS + spec.timeout_seconds
    )
    return bounded_seconds * worker_rate(gpu)


def reserve_within_cap(
    ledger: BudgetLedger,
    spec: StageSpec,
    gpu: str | None,
    max_dollars: float,
) -> str:
    """Reserve a stage's worst-case cost, refusing before dispatch if it breaks the cap."""
    if not math.isfinite(max_dollars) or max_dollars <= 0:
        raise ValueError("max_dollars must be positive and finite")
    dollars = reservation_dollars(spec, gpu)
    committed = ledger.committed()
    if committed + dollars > max_dollars:
        raise BudgetExceeded(
            f"Stage {spec.name!r} could cost up to ${dollars:.2f}; with ${committed:.2f} already "
            f"committed that exceeds the ${max_dollars:.2f} cap."
        )
    return ledger.reserve(
        spec.name,
        f"{spec.name}/{gpu or 'cpu'}",
        dollars,
        timeout_seconds=spec.timeout_seconds,
        gpu=gpu,
    )


def settle_worker_receipt(
    ledger: BudgetLedger,
    reservation: str,
    *,
    wall_seconds: float,
    worker_seconds: float | None,
    gpu: str | None,
) -> float:
    """Settle a known result; retain the reservation when remote completion is unknown."""
    observed_dollars = max(wall_seconds, worker_seconds or 0.0) * worker_rate(gpu)
    if worker_seconds is not None:
        ledger.complete(reservation, observed_dollars)
    return observed_dollars


def junit_counts(junit_xml: str | None) -> dict[str, int] | None:
    """Total the pytest outcome counters across every suite in a JUnit report."""
    if junit_xml is None:
        return None
    root = ElementTree.fromstring(junit_xml)
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    return {name: sum(int(suite.get(name, "0")) for suite in suites) for name in _JUNIT_COUNTS}


def failing_tests_by_cause(junit_xml: str | None) -> dict[str, list[str]]:
    """Separate assertion failures from the CPU tier's wall-clock budget.

    The CPU contract budgets are calibrated on the validation workstation. A
    sandboxed cloud worker can exceed the per-test budget while every assertion
    holds, and that outcome must not be read as a code regression, or hidden.
    """
    causes: dict[str, set[str]] = {"assertion": set(), "time_budget": set()}
    if junit_xml is None:
        return {cause: [] for cause in causes}
    for case in ElementTree.fromstring(junit_xml).iter("testcase"):
        reports = [*case.findall("failure"), *case.findall("error")]
        if not reports:
            continue
        node_id = f"{case.get('classname')}::{case.get('name')}"
        only_budget = all(
            _TIME_BUDGET_MARKER in (report.get("message") or "") for report in reports
        )
        causes["time_budget" if only_budget else "assertion"].add(node_id)
    return {cause: sorted(node_ids) for cause, node_ids in causes.items()}


def _git_state(git_checkout: bool) -> dict[str, object]:
    """Identify the uploaded source; a dirty tree makes the run descriptive only.

    A tree without Git metadata has no revision, so the snapshot digest recorded beside
    this state is its only identity.
    """
    if not git_checkout:
        return {"revision": None, "dirty": None, "changed_paths": None}
    revision = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    changed = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "status", "--porcelain"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    return {"revision": revision, "dirty": bool(changed), "changed_paths": len(changed)}


def main() -> None:
    args = _parser().parse_args()
    spec = STAGES[args.stage]
    gpu = args.gpu if spec.device == "gpu" else None
    # Reject a malformed selection before any money is reserved.
    stage_arguments(spec, args.select, "junit.xml")
    git_checkout = is_git_checkout()
    if spec.compares_baseline and not git_checkout:
        raise SystemExit(
            f"Stage {spec.name!r} compares the working tree with its Git baseline, "
            "and this tree has no Git metadata."
        )

    if not math.isfinite(args.max_dollars) or args.max_dollars <= 0:
        raise SystemExit("max_dollars must be positive and finite")
    run_directory = (
        ARTIFACT_ROOT / f"{time.strftime('%Y%m%dT%H%M%S')}-{spec.name}-{uuid.uuid4().hex[:8]}"
    )
    snapshot = stage_upload_source(run_directory / "source")
    # Without Git nothing is exported there, so the worker image mounts no baseline.
    baseline_directory = run_directory / "baseline"
    baseline_revision = (
        export_baseline_source(destination=baseline_directory) if git_checkout else None
    )
    os.environ["FASTPLMS_EVIDENCE_SOURCE_ROOT"] = str(snapshot.root)
    os.environ["FASTPLMS_EVIDENCE_BASELINE_ROOT"] = str(baseline_directory)
    source_state = _git_state(git_checkout)
    source_state["snapshot"] = snapshot.to_dict()

    ledger = BudgetLedger(ARTIFACT_ROOT / "budget.json")
    try:
        reservation = reserve_within_cap(ledger, spec, gpu, args.max_dollars)
    except (BudgetExceeded, ValueError) as error:
        raise SystemExit(str(error)) from error

    receipt: dict[str, Any] = {
        "stage": spec.name,
        "selection": args.select,
        "gpu": gpu,
        "source": source_state,
        "baseline_revision": baseline_revision,
        "reserved_dollars": reservation_dollars(spec, gpu),
        "status": "dispatching",
    }
    write_stored_json(run_directory / "receipt.json", receipt, sort_keys=False)

    import modal

    from dotenv import load_dotenv

    # The Modal app reads credential names from the environment when it is imported.
    load_dotenv(ROOT / ".secrets.env", override=False)
    from .modal_app import app, cpu_worker, fold_workers, gpu_workers

    if gpu is None:
        worker = cpu_worker
    else:
        worker = (fold_workers if spec.image == "fold" else gpu_workers)[gpu]
    started = time.monotonic()
    worker_receipt: dict[str, Any] | None = None
    try:
        with modal.enable_output(), app.run():
            receipt["modal_app_id"] = str(getattr(app, "app_id", None))
            worker_receipt = worker.remote(spec.name, args.select)
    finally:
        # Charge the longer of the two clocks; wall time also covers startup.
        wall_seconds = time.monotonic() - started
        worker_seconds = float(worker_receipt["elapsed_seconds"]) if worker_receipt else None
        observed_dollars = settle_worker_receipt(
            ledger, reservation,
            wall_seconds=wall_seconds, worker_seconds=worker_seconds, gpu=gpu,
        )
        receipt.update(
            wall_seconds=wall_seconds,
            observed_dollars=observed_dollars,
            committed_dollars=ledger.committed(),
            status="failed" if worker_receipt is None else worker_receipt["status"],
            budget_settlement="awaiting_worker_receipt" if worker_receipt is None else "completed",
        )
        write_stored_json(run_directory / "receipt.json", receipt, sort_keys=False)

    # A failed dispatch re-raises out of the ``finally`` block above.
    assert worker_receipt is not None
    junit_xml = worker_receipt.pop("junit_xml")
    output_tail = worker_receipt.pop("output_tail")
    if junit_xml is not None:
        (run_directory / "junit.xml").write_text(junit_xml, encoding="utf-8")
    (run_directory / "output.txt").write_text(output_tail, encoding="utf-8")
    failing = failing_tests_by_cause(junit_xml)
    receipt.update(result=worker_receipt, junit=junit_counts(junit_xml), failing_tests=failing)
    write_stored_json(run_directory / "receipt.json", receipt, sort_keys=False)

    print(output_tail[-6_000:])
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "stage",
                    "status",
                    "junit",
                    "failing_tests",
                    "wall_seconds",
                    "observed_dollars",
                    "committed_dollars",
                )
            },
            indent=2,
        )
    )
    print(f"Artifacts: {run_directory.relative_to(ROOT).as_posix()}")
    if receipt["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
