"""Training failures that a sweep should treat differently from a crash.

A diverged trial is a result, not a bug: a sweep records the offending configuration and prunes the
trials it dominates. Keeping the type separate from `torch.cuda.OutOfMemoryError` lets the sweep tell
the two apart.
"""

from __future__ import annotations


DIVERGENCE_KINDS = frozenset({
    "nan_loss",
    "inf_loss",
    "nan_logits",
    "nan_s_max",
    "loss_spike",
    "grad_nan",
})
"""The kinds of divergence a trainer reports, which drive a sweep's bookkeeping."""


class TrainingDivergedError(RuntimeError):
    """Raised when training numerics are no longer sane.

    `reason` is the human-readable cause, `step` the global step where it was detected (-1 when
    unknown), `value` the offending scalar (loss, gradient norm), `baseline` the reference scalar of
    a spike detection, and `kind` one of `DIVERGENCE_KINDS`.
    """

    def __init__(
        self,
        reason: str,
        *,
        step: int = -1,
        value: float | None = None,
        baseline: float | None = None,
        kind: str = "nan_loss",
    ) -> None:
        assert kind in DIVERGENCE_KINDS, f"Unknown divergence kind: {kind}"
        super().__init__(f"[{kind}] step={step} reason={reason} value={value} baseline={baseline}")
        self.reason = reason
        self.step = step
        self.value = value
        self.baseline = baseline
        self.kind = kind
