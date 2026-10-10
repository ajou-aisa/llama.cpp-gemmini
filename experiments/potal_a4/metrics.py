from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .types import F32, Quantized


@dataclass(frozen=True, slots=True)
class Totals:
    calls: int = 0
    values: int = 0
    main_bytes: int = 0
    digit_bytes: int = 0
    metadata_bytes: int = 0
    useful_digits: int = 0
    main_fragments: int = 0
    upper_fragments: int = 0
    saturated: int = 0
    error_squared: float = 0
    source_squared: float = 0
    zero_rows: int = 0
    row_sum_error: float = 0
    max_operand_bytes: int = 0

    def observe(self, source: F32, quantized: Quantized) -> Totals:
        packets = quantized.packets
        return replace(self,
            calls=self.calls + 1, values=self.values + source.size,
            saturated=self.saturated + quantized.saturated_values,
            error_squared=self.error_squared + float(np.square(quantized.values.astype(np.float64) - source).sum()),
            source_squared=self.source_squared + float(np.square(source.astype(np.float64)).sum()),
            main_bytes=self.main_bytes + sum(p.main.nbytes for p in packets),
            digit_bytes=self.digit_bytes + sum(p.digits.nbytes for p in packets),
            metadata_bytes=self.metadata_bytes + sum(p.metadata_bytes for p in packets),
            useful_digits=self.useful_digits + sum(p.useful_digits for p in packets),
            main_fragments=self.main_fragments + sum(p.main_fragments for p in packets),
            upper_fragments=self.upper_fragments + sum(p.upper_fragments for p in packets),
            max_operand_bytes=max(self.max_operand_bytes, sum(p.payload_bytes + p.metadata_bytes for p in packets)),
        )

    def probability(self, p: F32) -> Totals:
        sums = p.sum(axis=1, dtype=np.float64)
        return replace(self, zero_rows=self.zero_rows + int(np.count_nonzero(sums == 0)),
                       row_sum_error=max(self.row_sum_error, float(np.max(np.abs(sums - 1)))))


@dataclass(frozen=True, slots=True)
class Measurements:
    linear: Totals
    q: Totals
    p: Totals
    stationary_bytes: int = 0
    k_main_fragments: int = 0
    k_upper_fragments: int = 0

    @classmethod
    def empty(cls) -> Measurements:
        return cls(Totals(), Totals(), Totals())
