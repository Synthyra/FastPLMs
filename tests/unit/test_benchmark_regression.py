from __future__ import annotations

import pytest

from benchmarks.regression import (
    GateThresholds,
    bootstrap_ratio_interval,
    compare_reports,
)


ENVIRONMENT = {
    "python": "3.12.3",
    "platform": "Linux-test",
    "machine": "aarch64",
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
        "schema_version": 3,
        "status": "complete",
        "environment": ENVIRONMENT,
        "matrix_kind": "fixed",
        "claim_scope": "validated_hopper_sm90_exact_device",
        "backend_policy": {
            "requested": ["eager", "sdpa", "flex_attention"],
            "selection": "explicit_subset",
            "external_kernel_downloads": False,
            "external_kernel_builds": False,
        },
        "timing_contract": {
            "cold_compile_field": "results[].compile_ms",
            "warm_throughput_field": "results[].blocks",
        },
        "baseline_promotion_contract": {
            "requires_exact_environment_match": True,
            "requires_exact_artifact_inventory_match": True,
        },
        "expected_case_count": 1,
        "completed_case_count": 1,
        "results": [
            {
                "model": "example/model",
                "revision": "abc123",
                "auto_class": "AutoModel",
                "backend": "sdpa",
                "precision": "bf16",
                "mode": "steady",
                "batch_size": 1,
                "sequence_length": 512,
                "lengths": [512],
                "blocks": [{"logical_tokens_per_second": value} for value in values],
                "memory": {"peak_allocated_bytes": memory},
            }
        ],
    }


def _with_artifact_inventory(report: dict, *, runtime_revision: str = "a" * 40) -> dict:
    report["artifact_load_mode"] = "validated_local_build"
    report["artifacts"] = {
        "esm2_8m": {
            "model_id": "esm2_8m",
            "registry_repo_id": "Synthyra/ESM2-8M",
            "registry_revision": "b" * 40,
            "weights_revision": "b" * 40,
            "runtime_revision": runtime_revision,
            "source_tree_sha256": "c" * 64,
            "runtime_bundle_sha256": "d" * 64,
            "canonical_state_sha256": "e" * 64,
            "artifact_manifest_sha256": "f" * 64,
        }
    }
    return report


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
    current = _report([100] * 7)
    current["results"] = []
    current["expected_case_count"] = 0
    current["completed_case_count"] = 0
    result = compare_reports(current, _report([100] * 7))
    assert not result.passed
    assert result.unmatched_baseline


def test_gate_rejects_current_case_without_a_baseline() -> None:
    baseline = _report([100] * 7)
    current = _report([100] * 7)
    extra = dict(current["results"][0])
    extra["backend"] = "flex_attention"
    current["results"].append(extra)
    current["expected_case_count"] = 2
    current["completed_case_count"] = 2

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


def test_gate_rejects_cross_device_hopper_comparison() -> None:
    current = _report([100] * 7)
    current["environment"] = {
        **ENVIRONMENT,
        "gpu": "NVIDIA GH200 480GB",
        "nvidia_smi": {
            **ENVIRONMENT["nvidia_smi"],
            "name": "NVIDIA GH200 480GB",
            "memory.total": "97871",
        },
    }

    result = compare_reports(
        current,
        _report([100] * 7),
        GateThresholds(bootstrap_samples=100),
    )

    assert not result.passed
    assert any(
        mismatch.startswith("environment.gpu:") for mismatch in result.environment_mismatches
    )
    assert any(
        mismatch.startswith("environment.nvidia_smi.name:")
        for mismatch in result.environment_mismatches
    )
    assert any(
        mismatch.startswith("environment.nvidia_smi.memory.total:")
        for mismatch in result.environment_mismatches
    )


def test_gate_rejects_architecture_and_driver_drift() -> None:
    current = _report([100] * 7)
    current["environment"] = {
        **ENVIRONMENT,
        "machine": "x86_64",
        "nvidia_smi": {
            **ENVIRONMENT["nvidia_smi"],
            "driver_version": "580.105.08",
        },
    }

    result = compare_reports(
        current,
        _report([100] * 7),
        GateThresholds(bootstrap_samples=100),
    )

    assert not result.passed
    assert any(
        mismatch.startswith("environment.machine:")
        for mismatch in result.environment_mismatches
    )
    assert any(
        mismatch.startswith("environment.nvidia_smi.driver_version:")
        for mismatch in result.environment_mismatches
    )


def test_gate_rejects_missing_required_environment_identity() -> None:
    baseline = _report([100] * 7)
    baseline["environment"] = {
        key: value for key, value in ENVIRONMENT.items() if key != "machine"
    }

    result = compare_reports(
        _report([100] * 7),
        baseline,
        GateThresholds(bootstrap_samples=100),
    )

    assert not result.passed
    assert (
        "environment.machine: baseline report is missing the field"
        in result.environment_mismatches
    )


def test_gate_rejects_incomplete_or_different_promotion_contracts() -> None:
    current = _report([100] * 7)
    current["status"] = "running"
    baseline = _report([100] * 7)
    baseline["backend_policy"] = {
        **baseline["backend_policy"],
        "requested": ["sdpa"],
    }

    result = compare_reports(
        current,
        baseline,
        GateThresholds(bootstrap_samples=100),
    )

    assert not result.passed
    assert "current report status is not complete" in result.report_mismatches
    assert any(
        mismatch.startswith("backend_policy:") for mismatch in result.report_mismatches
    )


def test_gate_rejects_duplicate_and_nonfinite_measurements() -> None:
    duplicate = _report([100] * 7)
    duplicate["results"].append(dict(duplicate["results"][0]))
    duplicate["expected_case_count"] = 2
    duplicate["completed_case_count"] = 2
    with pytest.raises(ValueError, match="duplicate case"):
        compare_reports(duplicate, _report([100] * 7))

    nonfinite = _report([100] * 7)
    nonfinite["results"][0]["blocks"][0]["logical_tokens_per_second"] = float("nan")
    with pytest.raises(ValueError, match="non-positive/non-finite"):
        compare_reports(nonfinite, _report([100] * 7))


def test_gate_rejects_artifact_identity_drift() -> None:
    current = _with_artifact_inventory(_report([100] * 7), runtime_revision="1" * 40)
    baseline = _with_artifact_inventory(_report([100] * 7), runtime_revision="2" * 40)

    result = compare_reports(
        current,
        baseline,
        GateThresholds(bootstrap_samples=100),
    )

    assert not result.passed
    assert any(
        mismatch.startswith("artifacts.esm2_8m:") for mismatch in result.artifact_mismatches
    )


def test_gate_rejects_missing_artifact_inventory() -> None:
    current = _with_artifact_inventory(_report([100] * 7))

    result = compare_reports(
        current,
        _report([100] * 7),
        GateThresholds(bootstrap_samples=100),
    )

    assert not result.passed
    assert "baseline report has no artifact inventory mapping" in result.artifact_mismatches


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
    current["expected_case_count"] = baseline["expected_case_count"] = 2
    current["completed_case_count"] = baseline["completed_case_count"] = 2

    result = compare_reports(
        current,
        baseline,
        GateThresholds(bootstrap_samples=100),
    )

    assert result.passed
    assert len(result.cases) == 1
