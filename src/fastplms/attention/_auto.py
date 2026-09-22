"""Opt-in automatic selection of an attention implementation.

``attn_implementation="auto"`` is a request, not a backend. Each family lists
its implementations in a preference order backed by measured evidence, and
FastPLMs configures the first one that this machine can execute. After that the
model holds a named implementation. Configuration files and embedding
fingerprints record that name and never the word ``auto``.

Eager attention, SDPA, and Flex attention can be judged when the model is built.
FlashAttention depends on the device and on the dtype that Q, K, and V will
have, so a preference order that contains it is resolved at the first forward or
by an explicit ``resolve_attn_implementation`` call.
"""

from __future__ import annotations

import torch

from dataclasses import dataclass

from ._core import _ensure_flash_kernels_loaded, resolve_attention_backend
from ._kernel_lock import require_kernels_package


AUTO_ATTENTION = "auto"
_FLASH_IMPLEMENTATIONS = frozenset({"flash_attention_2", "flash_attention_3"})
_DTYPE_NAMES = {
    torch.float32: "float32",
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
}


@dataclass(frozen=True)
class AttentionExecutionContext:
    """The device and dtype that the attention inputs of a forward will have."""

    device: torch.device
    dtype: torch.dtype


@dataclass(frozen=True)
class AttentionCandidate:
    """One implementation from the preference order and why it was or was not usable."""

    implementation: str
    usable: bool
    reason: str


@dataclass(frozen=True)
class AttentionResolution:
    """The outcome of one ``auto`` request.

    While ``deferred`` is true the model runs ``resolved`` provisionally, and the
    first forward or ``resolve_attn_implementation`` replaces this record.
    """

    requested: str
    resolved: str
    candidates: tuple[AttentionCandidate, ...]
    context: AttentionExecutionContext | None
    deferred: bool


def needs_execution_context(order: tuple[str, ...]) -> bool:
    return any(implementation in _FLASH_IMPLEMENTATIONS for implementation in order)


def provisional_implementation(order: tuple[str, ...]) -> str:
    """Return the implementation a model runs until its context is known."""
    for implementation in order:
        if implementation not in _FLASH_IMPLEMENTATIONS:
            return implementation
    raise ValueError(f"The automatic attention order {order} has no context-free implementation.")


def attention_execution_context(
    module: torch.nn.Module,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> AttentionExecutionContext:
    """Describe where the next forward of ``module`` will run its attention.

    Under CUDA autocast FP32 parameters produce autocast-dtype Q, K, and V, so
    the autocast dtype is the one that decides kernel eligibility.
    """
    parameter = next(module.parameters(), None)
    if parameter is None:
        raise RuntimeError("Automatic attention selection requires a model with parameters.")
    resolved_device = parameter.device if device is None else torch.device(device)
    if dtype is not None:
        return AttentionExecutionContext(resolved_device, dtype)
    if resolved_device.type == "cuda" and torch.is_autocast_enabled("cuda"):
        return AttentionExecutionContext(resolved_device, torch.get_autocast_dtype("cuda"))
    return AttentionExecutionContext(resolved_device, parameter.dtype)


def _unusable(implementation: str, reason: str) -> AttentionCandidate:
    return AttentionCandidate(implementation, False, reason)


def _flash_candidate(implementation: str, context: AttentionExecutionContext) -> AttentionCandidate:
    """Judge a FlashAttention kernel, leaving the possible download for the last gate."""
    from fastplms.registry import get_model_registry

    kernel_spec = get_model_registry().attention_kernels[implementation]
    if context.device.type != "cuda":
        return _unusable(
            implementation, f"It requires a CUDA device; the model is on {context.device}."
        )
    dtype_name = _DTYPE_NAMES.get(context.dtype, str(context.dtype))
    if dtype_name not in kernel_spec.dtypes:
        supported = ", ".join(kernel_spec.dtypes)
        return _unusable(
            implementation,
            f"It supports only {supported}; the attention inputs would be {dtype_name}. "
            "Use CUDA BF16 autocast or BF16 weights.",
        )
    capability = torch.cuda.get_device_capability(context.device)
    if capability < kernel_spec.min_cuda_capability:
        required = ".".join(str(part) for part in kernel_spec.min_cuda_capability)
        observed = ".".join(str(part) for part in capability)
        return _unusable(
            implementation,
            f"It requires CUDA compute capability {required} or newer; this GPU has {observed}.",
        )
    try:
        require_kernels_package()
        _ensure_flash_kernels_loaded(implementation)
    except RuntimeError as error:
        return _unusable(implementation, str(error))
    return AttentionCandidate(implementation, True, "The manifest-locked kernel loaded.")


def _candidate(
    implementation: str, context: AttentionExecutionContext | None
) -> AttentionCandidate:
    if implementation in _FLASH_IMPLEMENTATIONS:
        if context is None:
            return _unusable(implementation, "It needs a device and dtype, which are not known.")
        return _flash_candidate(implementation, context)
    try:
        # Flex attention is the one device-independent backend a PyTorch build can lack.
        resolve_attention_backend(implementation)
    except RuntimeError as error:
        return _unusable(implementation, str(error))
    return AttentionCandidate(implementation, True, "It runs on every supported device and dtype.")


def resolve_auto_attention(
    order: tuple[str, ...],
    context: AttentionExecutionContext | None,
) -> AttentionResolution:
    """Select the first implementation in ``order`` that can execute in ``context``."""
    candidates: list[AttentionCandidate] = []
    for implementation in order:
        candidate = _candidate(implementation, context)
        candidates.append(candidate)
        if candidate.usable:
            return AttentionResolution(
                requested=AUTO_ATTENTION,
                resolved=implementation,
                candidates=tuple(candidates),
                context=context,
                deferred=False,
            )
    reasons = "; ".join(
        f"{candidate.implementation}: {candidate.reason}" for candidate in candidates
    )
    raise RuntimeError(f"No implementation in the automatic attention order is usable. {reasons}")


def deferred_resolution(order: tuple[str, ...]) -> AttentionResolution:
    """Record an ``auto`` request that waits for its execution context."""
    return AttentionResolution(
        requested=AUTO_ATTENTION,
        resolved=provisional_implementation(order),
        candidates=(),
        context=None,
        deferred=True,
    )
