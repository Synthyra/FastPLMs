"""Positive CPU allowlist aliases for Hopper/SM90 hardware contracts."""

from tests.unit.test_hopper_hardware_contract import (
    test_comparisons_require_the_exact_same_hopper_device_fingerprint,
    test_golden_comparison_rejects_cross_device_and_honors_new_identity_fields,
    test_release_hardware_accepts_named_hopper_sm90_products,
    test_release_hardware_rejects_non_hopper_or_incomplete_identity,
)

__all__ = [
    "test_comparisons_require_the_exact_same_hopper_device_fingerprint",
    "test_golden_comparison_rejects_cross_device_and_honors_new_identity_fields",
    "test_release_hardware_accepts_named_hopper_sm90_products",
    "test_release_hardware_rejects_non_hopper_or_incomplete_identity",
]
