"""Explicit, reversible Torch runtime configuration.

Importing FastPLMs does not change global Torch settings. Callers that want a
runtime profile opt in with :func:`runtime_profile` and receive their previous
settings back when the context exits.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal


if TYPE_CHECKING:
    from collections.abc import Iterator


MatmulPrecision = Literal["highest", "high", "medium"]


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    """Requested Torch settings for a bounded inference or training block."""

    float32_matmul_precision: MatmulPrecision = "highest"
    allow_tf32: bool | None = None


@contextmanager
def runtime_profile(profile: RuntimeProfile | None = None) -> Iterator[None]:
    """Apply a Torch runtime profile and restore the previous global settings.

    The default profile requests the highest float32 matrix-multiplication
    precision and leaves TF32 policy unchanged. Torch is imported only when the
    context is entered.
    """

    import torch

    selected = profile or RuntimeProfile()
    previous_matmul_precision = torch.get_float32_matmul_precision()
    matmul_backend = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    previous_matmul_tf32 = (
        getattr(matmul_backend, "allow_tf32", None) if matmul_backend is not None else None
    )
    previous_cudnn_tf32 = (
        getattr(cudnn_backend, "allow_tf32", None) if cudnn_backend is not None else None
    )

    torch.set_float32_matmul_precision(selected.float32_matmul_precision)
    if selected.allow_tf32 is not None:
        if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
            matmul_backend.allow_tf32 = selected.allow_tf32
        if cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32"):
            cudnn_backend.allow_tf32 = selected.allow_tf32
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous_matmul_precision)
        if selected.allow_tf32 is not None:
            if matmul_backend is not None and previous_matmul_tf32 is not None:
                matmul_backend.allow_tf32 = previous_matmul_tf32
            if cudnn_backend is not None and previous_cudnn_tf32 is not None:
                cudnn_backend.allow_tf32 = previous_cudnn_tf32


__all__ = ["MatmulPrecision", "RuntimeProfile", "runtime_profile"]
