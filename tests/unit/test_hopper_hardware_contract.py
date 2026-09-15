"""CPU contracts for generic CUDA device identity checks."""

from __future__ import annotations

import pytest

from tests.structure.support.hardware import (
    assert_recorded_device_matches,
    assert_same_device,
    device_fingerprint,
)


def _environment(
    name: str,
    *,
    capability: tuple[int, int] = (8, 9),
    total_memory: int = 96 * 1024**3,
) -> dict[str, object]:
    return {
        "cuda_device": name,
        "cuda_device_capability": list(capability),
        "cuda_total_memory": total_memory,
    }


@pytest.mark.parametrize("name", ("NVIDIA RTX 4070 Laptop GPU", "NVIDIA H100 PCIe"))
def test_release_hardware_accepts_any_named_cuda_device(name: str) -> None:
    fingerprint = device_fingerprint(_environment(name))

    assert fingerprint.name == name
    assert fingerprint.capability == (8, 9)


@pytest.mark.parametrize(
    "environment",
    (
        _environment("", capability=(8, 0)),
        _environment("NVIDIA A100-SXM4-80GB", capability=(-1, 0)),
        _environment("NVIDIA B200", capability=(10, "0")),
        {
            "cuda_device": "NVIDIA RTX 4070 Laptop GPU",
            "cuda_device_capability": [8, 9],
            "cuda_total_memory": 0,
        },
    ),
)
def test_release_hardware_rejects_malformed_identity(
    environment: dict[str, object],
) -> None:
    with pytest.raises(AssertionError):
        device_fingerprint(environment)


def test_comparisons_require_the_exact_same_device_fingerprint() -> None:
    rtx_4070 = _environment("NVIDIA RTX 4070 Laptop GPU", total_memory=8 * 1024**3)
    assert_same_device(rtx_4070, dict(rtx_4070))

    with pytest.raises(AssertionError, match="Cross-device comparison is forbidden"):
        assert_same_device(
            _environment("NVIDIA H100 PCIe", total_memory=80 * 1024**3),
            rtx_4070,
        )
    with pytest.raises(AssertionError, match="Cross-device comparison is forbidden"):
        assert_same_device(
            _environment("NVIDIA RTX 4070 Laptop GPU", total_memory=7 * 1024**3),
            rtx_4070,
        )


def test_recorded_device_comparison_rejects_cross_device_and_honors_identity_fields() -> None:
    current = _environment("NVIDIA RTX 4070 Laptop GPU", total_memory=8 * 1024**3)
    legacy_record = {"cuda_device": "NVIDIA RTX 4070 Laptop GPU"}
    assert_recorded_device_matches(current, legacy_record)
    assert_recorded_device_matches(current, dict(current))

    with pytest.raises(AssertionError, match="Cross-device recorded comparison is forbidden"):
        assert_recorded_device_matches(
            current,
            {"cuda_device": "NVIDIA H100 PCIe"},
        )
    with pytest.raises(AssertionError, match="cuda_total_memory"):
        assert_recorded_device_matches(
            current,
            {**current, "cuda_total_memory": 7 * 1024**3},
        )
