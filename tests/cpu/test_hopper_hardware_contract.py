"""CPU aliases for generic CUDA device identity contracts."""

from tests.unit.test_hopper_hardware_contract import (
    test_comparisons_require_the_exact_same_device_fingerprint,
    test_recorded_device_comparison_rejects_cross_device_and_honors_identity_fields,
    test_release_hardware_accepts_any_named_cuda_device,
    test_release_hardware_rejects_malformed_identity,
)


__all__ = [
    "test_comparisons_require_the_exact_same_device_fingerprint",
    "test_recorded_device_comparison_rejects_cross_device_and_honors_identity_fields",
    "test_release_hardware_accepts_any_named_cuda_device",
    "test_release_hardware_rejects_malformed_identity",
]
