"""Reproducible, exact-device Hopper/SM90 benchmarks for FastPLMs."""

from .regression import GateResult, GateThresholds, compare_reports

__all__ = ["GateResult", "GateThresholds", "compare_reports"]
