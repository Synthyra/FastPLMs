"""Deterministic producer and read-only validator contracts for official goldens."""

from .bundle import (
    GoldenBundleRecord,
    GoldenError,
    require_declared_goldens,
    validate_golden_bundle,
    write_golden_bundle,
)
from .from_native import (
    GoldenMatrixEntry,
    NativeGoldenRecord,
    check_tier_specs,
    convert_native_result,
    detect_native_result_kind,
    golden_generation_matrix,
    missing_check_golden_ids,
    require_complete_check_goldens,
)


__all__ = [
    "GoldenBundleRecord",
    "GoldenError",
    "GoldenMatrixEntry",
    "NativeGoldenRecord",
    "check_tier_specs",
    "convert_native_result",
    "detect_native_result_kind",
    "golden_generation_matrix",
    "missing_check_golden_ids",
    "require_complete_check_goldens",
    "require_declared_goldens",
    "validate_golden_bundle",
    "write_golden_bundle",
]
