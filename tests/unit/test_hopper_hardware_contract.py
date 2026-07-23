"""CPU contracts for Hopper/SM90 release-hardware identification."""

from __future__ import annotations

import pytest

from tests.structure.support.hardware import (
    HOPPER_SM90_CAPABILITY,
    assert_recorded_hopper_device_matches,
    assert_same_hopper_sm90_device,
    hopper_sm90_fingerprint,
)


def _environment(
    name: str,
    *,
    capability: tuple[int, int] = HOPPER_SM90_CAPABILITY,
    total_memory: int = 96 * 1024**3,
) -> dict[str, object]:
    return {
        "cuda_device": name,
        "cuda_device_capability": list(capability),
        "cuda_total_memory": total_memory,
    }


@pytest.mark.parametrize(
    ("name", "product"),
    (
        ("NVIDIA H100 PCIe", "H100"),
        ("NVIDIA H200 NVL", "H200"),
        ("NVIDIA GH200 480GB", "GH200"),
    ),
)
def test_release_hardware_accepts_named_hopper_sm90_products(
    name: str,
    product: str,
) -> None:
    fingerprint = hopper_sm90_fingerprint(_environment(name))

    assert fingerprint.product == product
    assert fingerprint.capability == (9, 0)


@pytest.mark.parametrize(
    "environment",
    (
        _environment("NVIDIA A100-SXM4-80GB", capability=(8, 0)),
        _environment("NVIDIA B200", capability=(10, 0)),
        _environment("NVIDIA H100 PCIe", capability=(8, 0)),
        {
            "cuda_device": "NVIDIA H200",
            "cuda_device_capability": [9, 0],
            "cuda_total_memory": 0,
        },
    ),
)
def test_release_hardware_rejects_non_hopper_or_incomplete_identity(
    environment: dict[str, object],
) -> None:
    with pytest.raises(AssertionError):
        hopper_sm90_fingerprint(environment)


def test_comparisons_require_the_exact_same_hopper_device_fingerprint() -> None:
    h100 = _environment("NVIDIA H100 PCIe", total_memory=80 * 1024**3)
    assert_same_hopper_sm90_device(h100, dict(h100))

    with pytest.raises(AssertionError, match="Cross-device comparison is forbidden"):
        assert_same_hopper_sm90_device(
            _environment("NVIDIA GH200 480GB", total_memory=96 * 1024**3),
            h100,
        )
    with pytest.raises(AssertionError, match="Cross-device comparison is forbidden"):
        assert_same_hopper_sm90_device(
            _environment("NVIDIA H100 PCIe", total_memory=94 * 1024**3),
            h100,
        )


def test_golden_comparison_rejects_cross_device_and_honors_new_identity_fields() -> None:
    current = _environment("NVIDIA GH200 480GB", total_memory=96 * 1024**3)
    legacy_record = {"cuda_device": "NVIDIA GH200 480GB"}
    assert_recorded_hopper_device_matches(current, legacy_record)
    assert_recorded_hopper_device_matches(current, dict(current))

    with pytest.raises(AssertionError, match="Cross-device golden comparison is forbidden"):
        assert_recorded_hopper_device_matches(
            current,
            {"cuda_device": "NVIDIA H100 PCIe"},
        )
    with pytest.raises(AssertionError, match="cuda_total_memory"):
        assert_recorded_hopper_device_matches(
            current,
            {**current, "cuda_total_memory": 80 * 1024**3},
        )
