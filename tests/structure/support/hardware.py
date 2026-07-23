"""Hardware identity contracts for Hopper/SM90 release validation."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

HOPPER_SM90_CAPABILITY = (9, 0)
HOPPER_PRODUCT_NAMES = ("H100", "H200", "GH200")
_HOPPER_PRODUCT_PATTERN = re.compile(r"(?<![A-Z0-9])(GH200|H200|H100)(?![A-Z0-9])")


@dataclass(frozen=True)
class HopperDeviceFingerprint:
    """Exact accelerator fields that must match across compared results."""

    product: str
    name: str
    capability: tuple[int, int]
    total_memory: int


def hopper_sm90_fingerprint(environment: Mapping[str, object]) -> HopperDeviceFingerprint:
    """Validate and return one allowed NVIDIA Hopper/SM90 device fingerprint."""

    name = environment.get("cuda_device")
    if not isinstance(name, str) or not name.strip():
        raise AssertionError("Hopper validation requires a non-empty CUDA device name.")
    match = _HOPPER_PRODUCT_PATTERN.search(name.upper())
    if match is None:
        allowed = ", ".join(HOPPER_PRODUCT_NAMES)
        raise AssertionError(
            f"Release validation requires an NVIDIA Hopper product ({allowed}); got {name!r}."
        )

    raw_capability = environment.get("cuda_device_capability")
    if (
        not isinstance(raw_capability, Sequence)
        or isinstance(raw_capability, (str, bytes))
        or len(raw_capability) != 2
        or any(not isinstance(value, int) or isinstance(value, bool) for value in raw_capability)
    ):
        raise AssertionError(
            "Hopper validation requires cuda_device_capability as two integer components."
        )
    capability = (raw_capability[0], raw_capability[1])
    if capability != HOPPER_SM90_CAPABILITY:
        raise AssertionError(
            f"Release validation requires compute capability 9.0; got {capability}."
        )

    total_memory = environment.get("cuda_total_memory")
    if not isinstance(total_memory, int) or isinstance(total_memory, bool) or total_memory <= 0:
        raise AssertionError("Hopper validation requires positive cuda_total_memory bytes.")

    return HopperDeviceFingerprint(
        product=match.group(1),
        name=name.strip(),
        capability=capability,
        total_memory=total_memory,
    )


def assert_same_hopper_sm90_device(
    current: Mapping[str, object],
    baseline: Mapping[str, object],
) -> None:
    """Reject cross-device comparisons even when both devices are Hopper/SM90."""

    current_fingerprint = hopper_sm90_fingerprint(current)
    baseline_fingerprint = hopper_sm90_fingerprint(baseline)
    if current_fingerprint != baseline_fingerprint:
        raise AssertionError(
            "Cross-device comparison is forbidden: "
            f"current={current_fingerprint!r}, baseline={baseline_fingerprint!r}."
        )


def assert_recorded_hopper_device_matches(
    current: Mapping[str, object],
    recorded: Mapping[str, object],
) -> None:
    """Match a live Hopper device to a golden's recorded hardware fields.

    Legacy goldens predate capability and memory fields, so their exact device
    name remains the strongest available identity. New captures must retain the
    additional fields and therefore receive the full comparison.
    """

    current_fingerprint = hopper_sm90_fingerprint(current)
    recorded_name = recorded.get("cuda_device")
    if recorded_name != current_fingerprint.name:
        raise AssertionError(
            "Cross-device golden comparison is forbidden: "
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
        raise AssertionError(f"Cross-device golden comparison is forbidden: {details}.")
