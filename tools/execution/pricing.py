"""Recorded Modal resource rates used to reserve bounded execution costs."""

from __future__ import annotations


GPU_DOLLARS_PER_SECOND = {"L4": 0.000222, "L40S": 0.000542, "H100": 0.001097}
CPU_DOLLARS_PER_CORE_SECOND = 0.0000131
MEMORY_DOLLARS_PER_GIB_SECOND = 0.00000222


def resource_rate(gpu: str | None, cpu: float = 4.0, memory_gib: float = 32.0) -> float:
    """Per-second rate using requested resource allocations and recorded prices."""
    gpu_rate = GPU_DOLLARS_PER_SECOND[gpu] if gpu else 0.0
    return gpu_rate + cpu * CPU_DOLLARS_PER_CORE_SECOND + memory_gib * MEMORY_DOLLARS_PER_GIB_SECOND
