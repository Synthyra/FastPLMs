from __future__ import annotations

from benchmarks.regression import (
    GateThresholds,
    bootstrap_ratio_interval,
    compare_reports,
)

ENVIRONMENT = {
    "python": "3.12.3",
    "platform": "Linux-test",
    "torch": "2.13.0+cu130",
    "cuda_runtime": "13.0",
    "cudnn": 92000,
    "transformers": "5.13.0",
    "fastplms": "1.0.0",
    "transformer_engine": "2.12.0",
    "kernels": "0.15.2",
    "kernels_data": "0.16.0",
    "gpu": "NVIDIA H100 80GB HBM3",
    "gpu_capability": [9, 0],
    "nvidia_smi": {
        "name": "NVIDIA H100 80GB HBM3",
        "driver_version": "999.0",
        "memory.total": "81559",
        "temperature.gpu": "31",
        "clocks.sm": "345",
    },
}


def _report(values: list[float], *, memory: int = 1_000_000_000) -> dict:
    return {
        "environment": ENVIRONMENT,
        "results": [
            {
                "model": "example/model",
                "revision": "abc123",
                "backend": "sdpa",
                "mode": "steady",
                "batch_size": 1,
                "sequence_length": 512,
                "lengths": [512],
                "blocks": [{"logical_tokens_per_second": value} for value in values],
                "memory": {"peak_allocated_bytes": memory},
            }
        ],
    }


def test_bootstrap_interval_is_deterministic() -> None:
    first = bootstrap_ratio_interval([105, 106, 104], [100, 100, 100], samples=500)
    second = bootstrap_ratio_interval([105, 106, 104], [100, 100, 100], samples=500)
    assert first == second
    assert first[0] == 1.05


def test_gate_accepts_equivalent_results() -> None:
    thresholds = GateThresholds(bootstrap_samples=500)
    result = compare_reports(
        _report([101, 100, 99, 101, 100, 99, 100]),
        _report([100] * 7),
        thresholds,
    )
    assert result.passed
    assert result.cases[0].passed


def test_gate_rejects_hard_throughput_regression() -> None:
    thresholds = GateThresholds(bootstrap_samples=500)
    result = compare_reports(_report([85] * 7), _report([100] * 7), thresholds)
    assert not result.passed
    assert any("hard limit" in reason for reason in result.cases[0].reasons)


def test_gate_rejects_large_memory_growth() -> None:
    thresholds = GateThresholds(bootstrap_samples=100)
    result = compare_reports(
        _report([100] * 7, memory=1_400_000_000),
        _report([100] * 7, memory=1_000_000_000),
        thresholds,
    )
    assert not result.passed
    assert any("memory" in reason for reason in result.cases[0].reasons)


def test_gate_requires_every_baseline_case() -> None:
    result = compare_reports({"environment": ENVIRONMENT, "results": []}, _report([100] * 7))
    assert not result.passed
    assert result.unmatched_baseline


def test_gate_rejects_current_case_without_a_baseline() -> None:
    baseline = _report([100] * 7)
    current = _report([100] * 7)
    extra = dict(current["results"][0])
    extra["backend"] = "flex_attention"
    current["results"].append(extra)

    result = compare_reports(current, baseline, GateThresholds(bootstrap_samples=100))

    assert not result.passed
    assert result.unmatched_current


def test_gate_rejects_environment_drift() -> None:
    current = _report([100] * 7)
    current["environment"] = {**ENVIRONMENT, "torch": "2.13.1+cu130"}

    result = compare_reports(
        current,
        _report([100] * 7),
        GateThresholds(bootstrap_samples=100),
    )

    assert not result.passed
    assert result.environment_mismatches == (
        "environment.torch: current='2.13.1+cu130', baseline='2.13.0+cu130'",
    )


def test_gate_ignores_telemetry_that_is_not_environment_identity() -> None:
    current = _report([100] * 7)
    current_smi = dict(ENVIRONMENT["nvidia_smi"])
    current_smi["temperature.gpu"] = "79"
    current_smi["clocks.sm"] = "1980"
    current["environment"] = {**ENVIRONMENT, "nvidia_smi": current_smi}

    result = compare_reports(
        current,
        _report([100] * 7),
        GateThresholds(bootstrap_samples=100),
    )

    assert result.passed
    assert not result.environment_mismatches


def test_descriptive_records_do_not_enter_throughput_gate() -> None:
    current = _report([100] * 7)
    baseline = _report([100] * 7)
    descriptive = {
        "model": "example/model",
        "revision": "abc123",
        "backend": "sdpa",
        "mode": "embed",
        "batch_size": 1,
        "sequence_length": 512,
        "lengths": [512],
        "blocks": [],
        "embedding_ms": 12.0,
        "memory": {"peak_allocated_bytes": 1_000_000_000},
    }
    current["results"].append({**descriptive, "embedding_ms": 11.0})
    baseline["results"].append(descriptive)

    result = compare_reports(
        current,
        baseline,
        GateThresholds(bootstrap_samples=100),
    )

    assert result.passed
    assert len(result.cases) == 1
