"""Streaming full-population statistics; only quantiles use histogram interpolation."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Final, TypedDict

import numpy as np
from numpy.typing import NDArray
from weight_io import InputError

EDGES: Final = np.linspace(-32.0, 8.0, 161)
FINE_EDGES: Final = np.linspace(-32.0, 8.0, 1281)


class Summary(TypedDict):
    count: int
    mean: float
    mean_abs: float
    rms: float
    std: float
    min: float
    max: float
    max_abs: float
    zero_fraction: float
    zeroed_nonzero_count: int
    zeroed_nonzero_fraction: float
    p50_abs_approx: float
    p99_abs_approx: float
    p999_abs_approx: float
    rmse: float
    max_abs_error: float
    relative_l2_error: float | None
    cosine: float | None
    sqnr_db: float | None
    outside_histogram: int


@dataclass(slots=True)
class Moments:
    """Mutable accumulator for one tensor, block, or the complete model."""
    count: int = 0
    zeros: int = 0
    total: float = 0.0
    absolute: float = 0.0
    square: float = 0.0
    error_square: float = 0.0
    reference_square: float = 0.0
    dot: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    error_max: float = 0.0
    histogram: NDArray[np.int64] = field(default_factory=lambda: np.zeros(160, dtype=np.int64))
    fine_histogram: NDArray[np.int64] = field(default_factory=lambda: np.zeros(1280, dtype=np.int64))
    zeroed_reference_histogram: NDArray[np.int64] = field(default_factory=lambda: np.zeros(1280, dtype=np.int64))
    code_histogram: NDArray[np.int64] = field(default_factory=lambda: np.zeros(256, dtype=np.int64))
    outside_histogram: int = 0

    def add(self, values: NDArray[np.float32], reference: NDArray[np.float32], codes: NDArray[np.int16] | None) -> None:
        x, ref = values.astype(np.float64), reference.astype(np.float64)
        if x.shape != ref.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(ref)):
            raise InputError("Non-finite or mismatched weight chunk")
        absolute = np.abs(x)
        error = x - ref
        self.count += x.size
        self.zeros += int(np.count_nonzero(x == 0))
        self.total += float(x.sum())
        self.absolute += float(absolute.sum())
        self.square += float(np.square(x).sum())
        self.error_square += float(np.square(error).sum())
        self.reference_square += float(np.square(ref).sum())
        self.dot += float((x * ref).sum())
        self.minimum = min(self.minimum, float(x.min()))
        self.maximum = max(self.maximum, float(x.max()))
        self.error_max = max(self.error_max, float(np.abs(error).max()))
        positive = absolute[absolute > 0]
        bins = np.floor((np.log2(positive) - FINE_EDGES[0]) * 32).astype(np.int32)
        self.outside_histogram += int(np.count_nonzero((bins < 0) | (bins >= 1280)))
        self.fine_histogram += np.bincount(np.clip(bins, 0, 1279), minlength=1280)
        self.histogram += np.bincount(np.clip(bins // 8, 0, 159), minlength=160)
        lost = np.abs(ref[(ref != 0) & (x == 0)])
        lost_bins = np.floor((np.log2(lost) - FINE_EDGES[0]) * 32).astype(np.int32)
        self.zeroed_reference_histogram += np.bincount(np.clip(lost_bins, 0, 1279), minlength=1280)
        if codes is not None:
            self.code_histogram += np.bincount(codes + 128, minlength=256)

    def merge(self, other: Moments) -> None:
        for key in ("count", "zeros", "total", "absolute", "square", "error_square", "reference_square", "dot", "outside_histogram"):
            setattr(self, key, getattr(self, key) + getattr(other, key))
        self.minimum = min(self.minimum, other.minimum)
        self.maximum = max(self.maximum, other.maximum)
        self.error_max = max(self.error_max, other.error_max)
        self.histogram += other.histogram
        self.fine_histogram += other.fine_histogram
        self.zeroed_reference_histogram += other.zeroed_reference_histogram
        self.code_histogram += other.code_histogram

    def quantile(self, probability: float) -> float:
        rank = self.count * probability - self.zeros
        if rank <= 0:
            return 0.0
        cumulative = np.cumsum(self.histogram)
        index = min(int(np.searchsorted(cumulative, rank)), 159)
        previous = int(cumulative[index - 1]) if index else 0
        fraction = (rank - previous) / max(1, int(self.histogram[index]))
        return float(2 ** (EDGES[index] + fraction * 0.25))

    def summary(self) -> Summary:
        rms = math.sqrt(self.square / self.count)
        error_rms = math.sqrt(self.error_square / self.count)
        reference_rms = math.sqrt(self.reference_square / self.count)
        zeroed_nonzero = int(self.zeroed_reference_histogram.sum())
        return Summary(count=self.count, mean=self.total / self.count, mean_abs=self.absolute / self.count,
                    rms=rms, std=math.sqrt(max(0.0, rms*rms - (self.total/self.count)**2)),
                    min=self.minimum, max=self.maximum, max_abs=max(abs(self.minimum), abs(self.maximum)),
                    zero_fraction=self.zeros / self.count,
                    zeroed_nonzero_count=zeroed_nonzero, zeroed_nonzero_fraction=zeroed_nonzero / self.count,
                    p50_abs_approx=self.quantile(0.5),
                    p99_abs_approx=self.quantile(0.99), p999_abs_approx=self.quantile(0.999),
                    rmse=error_rms, max_abs_error=self.error_max,
                    relative_l2_error=error_rms/reference_rms if reference_rms else None,
                    cosine=self.dot/math.sqrt(self.square*self.reference_square) if self.square*self.reference_square else None,
                    sqnr_db=20*math.log10(reference_rms/error_rms) if reference_rms and error_rms else None,
                    outside_histogram=self.outside_histogram)
