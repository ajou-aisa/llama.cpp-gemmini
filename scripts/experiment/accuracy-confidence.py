#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# How to run: imported by watch-accuracy-summary.py; test with test-accuracy-confidence.py.
"""Two-sided 95% Wilson intervals for completed binary ACC and ACC_NORM results.

Reference: https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm
Intervals describe item-sampling uncertainty, assuming independent binary items.
"""
from dataclasses import dataclass
import math
from statistics import NormalDist
from typing import Final

LEVEL: Final = 0.95
Z: Final = NormalDist().inv_cdf((1 + LEVEL) / 2)


class ConfidenceError(ValueError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    samples: int
    successes: int


def wilson_interval(proportion: float, samples: int) -> ConfidenceInterval:
    """Recover the correct count from an unrounded binary mean and full item count."""
    if type(samples) is not int or samples <= 0:
        raise ConfidenceError("CI requires a positive integer evaluation item count")
    if not math.isfinite(proportion) or not 0 <= proportion <= 1:
        raise ConfidenceError("CI requires a finite accuracy between zero and one")
    count = proportion * samples
    successes = round(count)
    if not math.isclose(count, successes, rel_tol=0, abs_tol=1e-6):
        raise ConfidenceError("CI requires the unrounded binary accuracy, not a rounded percentage")
    estimate = successes / samples
    squared = Z * Z
    denominator = 1 + squared / samples
    center = (estimate + squared / (2 * samples)) / denominator
    radius = Z * math.sqrt(estimate * (1 - estimate) / samples + squared / (4 * samples * samples)) / denominator
    lower = 0.0 if successes == 0 else max(0.0, center - radius)
    upper = 1.0 if successes == samples else min(1.0, center + radius)
    return ConfidenceInterval(estimate, lower, upper, samples, successes)


def format_accuracy(value: float | None, samples: int) -> str:
    """Render percentage point estimate and bounds, without changing stored scores."""
    if value is None:
        return "--"
    interval = wilson_interval(value, samples)
    return f"{100 * interval.estimate:.2f} [{100 * interval.lower:.2f},{100 * interval.upper:.2f}]"
