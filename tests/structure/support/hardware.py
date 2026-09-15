"""CUDA device identity checks used by numerical comparison tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceFingerprint:
    """CUDA fields needed to compare two runs on the same device."""

    name: str
    capability: tuple[int, int]
    total_memory: int


def device_fingerprint(environment: Mapping[str, object]) -> DeviceFingerprint:
    """Validate and return a generic CUDA device fingerprint."""

    name = environment.get("cuda_device")
    if not isinstance(name, str) or not name.strip():
        raise AssertionError("CUDA validation requires a non-empty device name.")

    raw_capability = environment.get("cuda_device_capability")
    if (
        not isinstance(raw_capability, Sequence)
        or isinstance(raw_capability, (str, bytes))
        or len(raw_capability) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in raw_capability
        )
    ):
        raise AssertionError(
            "CUDA validation requires cuda_device_capability as two non-negative integers."
        )
    capability = (raw_capability[0], raw_capability[1])

    total_memory = environment.get("cuda_total_memory")
    if not isinstance(total_memory, int) or isinstance(total_memory, bool) or total_memory <= 0:
        raise AssertionError("CUDA validation requires positive cuda_total_memory bytes.")

    return DeviceFingerprint(
        name=name.strip(),
        capability=capability,
        total_memory=total_memory,
    )


def assert_same_device(
    current: Mapping[str, object],
    baseline: Mapping[str, object],
) -> None:
    """Reject cross-device comparisons for paired numerical measurements."""

    current_fingerprint = device_fingerprint(current)
    baseline_fingerprint = device_fingerprint(baseline)
    if current_fingerprint != baseline_fingerprint:
        raise AssertionError(
            "Cross-device comparison is forbidden: "
            f"current={current_fingerprint!r}, baseline={baseline_fingerprint!r}."
        )


def assert_recorded_device_matches(
    current: Mapping[str, object],
    recorded: Mapping[str, object],
) -> None:
    """Match a live device to a recorded identity when a paired run requires it."""

    current_fingerprint = device_fingerprint(current)
    recorded_name = recorded.get("cuda_device")
    if recorded_name != current_fingerprint.name:
        raise AssertionError(
            "Cross-device recorded comparison is forbidden: "
            f"current device={current_fingerprint.name!r}, recorded device={recorded_name!r}."
        )
    optional_fields = ("cuda_device_capability", "cuda_total_memory")
    mismatches = [
        field
        for field in optional_fields
        if field in recorded and recorded[field] != current.get(field)
    ]
    if mismatches:
        details = ", ".join(
            f"{field}: current={current.get(field)!r}, recorded={recorded.get(field)!r}"
            for field in mismatches
        )
        raise AssertionError(f"Cross-device recorded comparison is forbidden: {details}.")
