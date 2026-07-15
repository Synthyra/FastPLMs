"""Scalar schedules for structure-steering potentials."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence


class ParameterSchedule(ABC):
    """Map normalized diffusion time ``t`` to a potential parameter."""

    @abstractmethod
    def compute(self, t: float) -> float:
        """Evaluate the schedule at ``t``."""


class ExponentialInterpolation(ParameterSchedule):
    """Interpolate from ``start`` to ``end`` with exponential curvature."""

    def __init__(self, start: float, end: float, alpha: float) -> None:
        self.start = start
        self.end = end
        self.alpha = alpha

    def compute(self, t: float) -> float:
        span = self.end - self.start
        if self.alpha == 0:
            return self.start + span * t
        numerator = math.exp(self.alpha * t) - 1
        denominator = math.exp(self.alpha) - 1
        return self.start + span * numerator / denominator


class PiecewiseStepFunction(ParameterSchedule):
    """Select values separated by strict upper thresholds.

    A time exactly equal to a threshold remains in the lower interval.  This
    boundary convention is part of the steering-input contract.
    """

    def __init__(
        self,
        thresholds: Sequence[float],
        values: Sequence[float],
    ) -> None:
        self.thresholds = thresholds
        self.values = values

    def compute(self, t: float) -> float:
        assert self.thresholds
        assert len(self.values) == len(self.thresholds) + 1
        interval = next(
            (index for index, threshold in enumerate(self.thresholds) if t <= threshold),
            len(self.thresholds),
        )
        return self.values[interval]
