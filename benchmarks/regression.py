"""Statistical regression gates for benchmark reports.

The gate compares paired measurement blocks from the same benchmark case. It
keeps raw measurements in the report so a baseline change is always an
explicit, reviewable file update.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GateThresholds:
    """Scalar thresholds used by the H100 regression gate."""

    confidence: float = 0.95
    soft_throughput_ratio: float = 0.95
    hard_throughput_ratio: float = 0.90
    claimed_improvement_ratio: float = 1.05
    memory_growth_fraction: float = 0.05
    memory_growth_bytes: int = 256 * 1024**2
    hard_memory_growth_fraction: float = 0.10
    bootstrap_samples: int = 10_000
    seed: int = 42


DEFAULT_GATE_THRESHOLDS = GateThresholds()


@dataclass(frozen=True)
class CaseGateResult:
    """Result for one matched model, backend, mode, and input shape."""

    case: str
    passed: bool
    median_ratio: float
    lower_confidence_bound: float
    upper_confidence_bound: float
    memory_growth_bytes: int
    memory_growth_fraction: float
    improvement_supported: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class GateResult:
    """Aggregate benchmark comparison."""

    passed: bool
    cases: tuple[CaseGateResult, ...]
    unmatched_current: tuple[str, ...]
    unmatched_baseline: tuple[str, ...]
    environment_mismatches: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "cases": [asdict(case) for case in self.cases],
            "unmatched_current": list(self.unmatched_current),
            "unmatched_baseline": list(self.unmatched_baseline),
            "environment_mismatches": list(self.environment_mismatches),
        }


ENVIRONMENT_FIELDS = (
    "python",
    "platform",
    "torch",
    "cuda_runtime",
    "cudnn",
    "transformers",
    "fastplms",
    "transformer_engine",
    "kernels",
    "kernels_data",
    "gpu",
    "gpu_capability",
)
NVIDIA_SMI_IDENTITY_FIELDS = ("name", "driver_version", "memory.total")


def percentile(values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated percentile without NumPy."""

    if not values:
        raise ValueError("At least one value is required")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_ratio_interval(
    current: Sequence[float],
    baseline: Sequence[float],
    *,
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Estimate a paired bootstrap interval for scalar throughput ratio ``r``."""

    if len(current) != len(baseline):
        raise ValueError("Current and baseline reports must have equally many blocks")
    if not current:
        raise ValueError("At least one paired block is required")
    if any(value <= 0.0 for value in baseline):
        raise ValueError("Baseline throughput must be positive")
    if samples < 1:
        raise ValueError("samples must be positive")

    paired_ratios = [new / old for new, old in zip(current, baseline, strict=True)]
    median_ratio = statistics.median(paired_ratios)
    generator = random.Random(seed)
    n = len(paired_ratios)
    bootstrapped = [
        statistics.median(paired_ratios[generator.randrange(n)] for _ in range(n))
        for _ in range(samples)
    ]
    tail = 1.0 - confidence
    # The release policy uses one-sided bounds: the upper bound detects a
    # regression and the lower bound supports a speed claim.
    return median_ratio, percentile(bootstrapped, tail), percentile(bootstrapped, confidence)


def _case_key(record: Mapping[str, Any]) -> str:
    fields = (
        "model",
        "revision",
        "auto_class",
        "backend",
        "precision",
        "mode",
        "batch_size",
        "sequence_length",
        "lengths",
    )
    return "|".join(f"{field}={record.get(field)}" for field in fields)


def _throughputs(record: Mapping[str, Any]) -> list[float]:
    blocks = record.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError(f"Benchmark case {_case_key(record)} has no measurement blocks")
    return [float(block["logical_tokens_per_second"]) for block in blocks]


def _throughput_records(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Index only cases containing steady-state measurement blocks.

    Startup, first-forward, compilation-only, and full-embedding records remain
    descriptive. They are retained in reports but are not throughput ratios.
    """

    result: dict[str, Mapping[str, Any]] = {}
    for record in report.get("results", []):
        blocks = record.get("blocks")
        if isinstance(blocks, list) and blocks:
            result[_case_key(record)] = record
    return result


def _peak_memory(record: Mapping[str, Any]) -> int:
    memory = record.get("memory", {})
    return int(memory.get("peak_allocated_bytes", 0))


def _environment_mismatches(
    current: Mapping[str, Any], baseline: Mapping[str, Any]
) -> tuple[str, ...]:
    """Return release-environment differences that invalidate a comparison."""

    current_environment = current.get("environment")
    baseline_environment = baseline.get("environment")
    if not isinstance(current_environment, Mapping):
        return ("current report has no environment fingerprint",)
    if not isinstance(baseline_environment, Mapping):
        return ("baseline report has no environment fingerprint",)

    mismatches: list[str] = []
    for field in ENVIRONMENT_FIELDS:
        current_value = current_environment.get(field)
        baseline_value = baseline_environment.get(field)
        if current_value != baseline_value:
            mismatches.append(
                f"environment.{field}: current={current_value!r}, baseline={baseline_value!r}"
            )

    current_smi = current_environment.get("nvidia_smi")
    baseline_smi = baseline_environment.get("nvidia_smi")
    if not isinstance(current_smi, Mapping):
        mismatches.append("environment.nvidia_smi: current report has no mapping")
    if not isinstance(baseline_smi, Mapping):
        mismatches.append("environment.nvidia_smi: baseline report has no mapping")
    if isinstance(current_smi, Mapping) and isinstance(baseline_smi, Mapping):
        for field in NVIDIA_SMI_IDENTITY_FIELDS:
            current_value = current_smi.get(field)
            baseline_value = baseline_smi.get(field)
            if current_value != baseline_value:
                mismatches.append(
                    f"environment.nvidia_smi.{field}: current={current_value!r}, "
                    f"baseline={baseline_value!r}"
                )
    return tuple(mismatches)


def compare_reports(
    current: Mapping[str, Any],
    baseline: Mapping[str, Any],
    thresholds: GateThresholds = DEFAULT_GATE_THRESHOLDS,
) -> GateResult:
    """Compare two reports and return a deterministic regression decision."""

    current_cases = _throughput_records(current)
    baseline_cases = _throughput_records(baseline)
    environment_mismatches = _environment_mismatches(current, baseline)
    shared = sorted(current_cases.keys() & baseline_cases.keys())
    case_results: list[CaseGateResult] = []

    for index, key in enumerate(shared):
        new = current_cases[key]
        old = baseline_cases[key]
        median_ratio, lower, upper = bootstrap_ratio_interval(
            _throughputs(new),
            _throughputs(old),
            confidence=thresholds.confidence,
            samples=thresholds.bootstrap_samples,
            seed=thresholds.seed + index,
        )
        baseline_memory = _peak_memory(old)
        current_memory = _peak_memory(new)
        memory_delta = current_memory - baseline_memory
        memory_fraction = memory_delta / baseline_memory if baseline_memory else 0.0
        soft_memory_limit = max(
            thresholds.memory_growth_bytes,
            int(baseline_memory * thresholds.memory_growth_fraction),
        )
        reasons: list[str] = []
        if upper < thresholds.soft_throughput_ratio:
            reasons.append(
                f"upper confidence bound {upper:.4f} is below "
                f"{thresholds.soft_throughput_ratio:.4f}"
            )
        if median_ratio < thresholds.hard_throughput_ratio:
            reasons.append(
                f"median throughput ratio {median_ratio:.4f} is below hard limit "
                f"{thresholds.hard_throughput_ratio:.4f}"
            )
        if memory_delta > soft_memory_limit:
            reasons.append(
                f"peak allocated memory grew by {memory_delta} bytes; limit is {soft_memory_limit}"
            )
        if baseline_memory and memory_fraction > thresholds.hard_memory_growth_fraction:
            reasons.append(
                f"peak allocated memory grew by {memory_fraction:.2%}; hard limit is "
                f"{thresholds.hard_memory_growth_fraction:.2%}"
            )
        case_results.append(
            CaseGateResult(
                case=key,
                passed=not reasons,
                median_ratio=median_ratio,
                lower_confidence_bound=lower,
                upper_confidence_bound=upper,
                memory_growth_bytes=memory_delta,
                memory_growth_fraction=memory_fraction,
                improvement_supported=lower >= thresholds.claimed_improvement_ratio,
                reasons=tuple(reasons),
            )
        )

    unmatched_current = tuple(sorted(current_cases.keys() - baseline_cases.keys()))
    unmatched_baseline = tuple(sorted(baseline_cases.keys() - current_cases.keys()))
    passed = (
        bool(shared)
        and not unmatched_current
        and not unmatched_baseline
        and not environment_mismatches
        and all(case.passed for case in case_results)
    )
    return GateResult(
        passed=passed,
        cases=tuple(case_results),
        unmatched_current=unmatched_current,
        unmatched_baseline=unmatched_baseline,
        environment_mismatches=environment_mismatches,
    )


def _load(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("current", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    result = compare_reports(_load(arguments.current), _load(arguments.baseline))
    rendered = json.dumps(result.to_dict(), indent=2, sort_keys=True)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
