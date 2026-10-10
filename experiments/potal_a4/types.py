from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

import numpy as np
from numpy.typing import NDArray

F32 = NDArray[np.float32]
I64 = NDArray[np.int64]
BK: Final = 32
DIM: Final = 32
NEG: Final = -32768


class MainRule(str, Enum):
    SELECTIVE = "selective"
    DIRECT = "direct"
    CLIPPED = "clipped"


@dataclass(frozen=True, slots=True)
class QuantError(Exception):
    detail: str


@dataclass(frozen=True, slots=True)
class StreamPolicy:
    outputs: int
    rho: int = 2
    rule: MainRule = MainRule.DIRECT
    upper_limbs: int = 8


@dataclass(frozen=True, slots=True)
class PPolicy:
    outputs: int
    rows: int = 0
    bits: int = 8


@dataclass(frozen=True, slots=True)
class Packet:
    main: NDArray[np.int8]
    digits: NDArray[np.int8]
    row_lane: NDArray[np.uint16]
    columns: NDArray[np.uint32]
    scales: NDArray[np.int16]
    useful_digits: int
    main_fragments: int
    upper_fragments: int

    @property
    def payload_bytes(self) -> int:
        return self.main.nbytes + self.digits.nbytes

    @property
    def metadata_bytes(self) -> int:
        return self.row_lane.nbytes + self.columns.nbytes + self.scales.nbytes


@dataclass(frozen=True, slots=True)
class Quantized:
    values: F32
    packets: tuple[Packet, ...]
    saturated_values: int = 0


@dataclass(frozen=True, slots=True)
class Mode:
    name: str
    linear: MainRule | None
    attention: bool = False
    p_rows: int = 0
    p_bits: int = 8
    upper_limbs: int = 8


MODES: Final = (
    Mode("weights_only", None),
    Mode("linear_selective", MainRule.SELECTIVE),
    Mode("linear_direct", MainRule.DIRECT),
    Mode("original_a4nks", MainRule.SELECTIVE, True),
    Mode("direct_stripe", MainRule.DIRECT, True),
    Mode("direct_p32", MainRule.DIRECT, True, 32),
    Mode("direct_p8", MainRule.DIRECT, True, 8),
    Mode("direct_prow", MainRule.DIRECT, True, 1),
    Mode("direct_p6", MainRule.DIRECT, True, 8, 6),
    Mode("direct_p4", MainRule.DIRECT, True, 8, 4),
    Mode("direct_limb1", MainRule.DIRECT, True, 8, 8, 1),
    Mode("direct_limb0", MainRule.DIRECT, True, 8, 8, 0),
)
