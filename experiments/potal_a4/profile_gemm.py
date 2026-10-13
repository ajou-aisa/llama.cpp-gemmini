from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .profile_ops import Stationary, mark
from .types import BK, DIM, F32, I64, QuantError, Quantized


@dataclass(frozen=True, slots=True)
class QKTileMask:
    blocked: NDArray[np.bool_]

    @classmethod
    def from_values(cls, mask: F32) -> QKTileMask:
        if mask.ndim != 2 or not np.all(np.isfinite(mask) | np.isneginf(mask)):
            raise QuantError("QK mask must be a finite/-inf matrix")
        return cls(np.isneginf(mask))

    def tiles(self, original_rows: I64) -> tuple[tuple[slice, slice], ...]:
        retained = []
        width = self.blocked.shape[1]
        for first in range(0, len(original_rows), DIM):
            rows = slice(first, min(first + DIM, len(original_rows)))
            live = np.any(~self.blocked[original_rows[rows]], axis=0)
            begin = None
            for column in range(0, width, DIM):
                if live[column:column + DIM].any():
                    if begin is None:
                        begin = column
                elif begin is not None:
                    retained.append((rows, slice(begin, column)))
                    begin = None
            if begin is not None:
                retained.append((rows, slice(begin, width)))
        return tuple(retained)


@dataclass(frozen=True, slots=True)
class Stripe:
    main: I64
    scale: int
    lanes: tuple[tuple[int, I64, I64], ...]
    columns: tuple[I64, ...]


class BatchDot(Protocol):
    def add(self, a: I64, b: I64, rows: I64, shifts: I64,
            tiles: tuple[tuple[slice, slice], ...] | None) -> None: ...

    def flush(self, acc: I64) -> None: ...


def unpack(q: Quantized) -> tuple[Stripe, ...]:
    stripes = []
    for packet in q.packets:
        columns = tuple(packet.columns[(packet.columns >= b) & (packet.columns < b + BK)].astype(np.int64)
                        for b in range(0, packet.main.shape[1], BK))
        lanes = []
        for lane in np.unique(packet.row_lane[:, 0]):
            selected = np.flatnonzero(packet.row_lane[:, 0] == lane)
            active = packet.row_lane[selected, 1].astype(np.int64)
            plane = np.zeros((len(active), packet.main.shape[1]), dtype=np.int64)
            offset = 0
            for cols in columns:
                plane[:, cols] = packet.digits[selected, offset:offset + len(cols)]
                offset += ((len(cols) + 31) // 32) * 32
            lanes.append((int(lane), active, plane))
        stripes.append(Stripe(packet.main.astype(np.int64), int(packet.scales[0]), tuple(lanes), columns))
    return tuple(stripes)


def checked_bound(stripes: tuple[Stripe, ...], stationary: Stationary, zero_point: int) -> None:
    minimum = int(stationary.theta.min())
    for stripe in stripes:
        bound = 0
        for codes, extra in stationary.passes:
            for b in range(len(stationary.theta)):
                block = codes[b * BK:(b + 1) * BK]
                magnitude = int(np.abs(block).max(initial=0))
                coefficient = (int(np.abs(stripe.main[:, b * BK:(b + 1) * BK]).max(initial=0)) + zero_point) * magnitude * len(block)
                for lane, _, plane in stripe.lanes:
                    cols = stripe.columns[b]
                    coefficient += (int(np.abs(plane[:, cols]).max(initial=0)) * magnitude * len(cols)) << (4 * lane)
                bound += coefficient << (int(stationary.theta[b].max()) + extra - minimum)
        if bound > np.iinfo(np.int64).max:
            raise QuantError("Integer profiling accumulator bound exceeds int64; no approximate fallback")


def gemm(stripes: tuple[Stripe, ...], stationary: Stationary, role: str,
         times: dict[str, int], zero_point: int = 0, dot=None,
         qk_mask: QKTileMask | None = None, fragments: Counter[str] | None = None,
         batch: BatchDot | None = None) -> np.ndarray:
    if batch is not None and dot is not None:
        raise QuantError("Use either immediate or batched raw dots")
    minimum = int(stationary.theta.min())
    width = stationary.values.shape[1]
    output = np.empty((sum(len(st.main) for st in stripes), width), dtype=np.float64)
    if qk_mask is not None:
        if role != "qk" or zero_point or qk_mask.blocked.shape != output.shape:
            raise QuantError("Tile mask requires QK with matching original row/key dimensions")

    def partial(a: I64, b: I64, lane: int, label: str,
                tiles: tuple[tuple[slice, slice], ...] | None, rows: I64, shifts: I64) -> I64 | None:
        full = ((len(a) + DIM - 1) // DIM) * ((b.shape[1] + DIM - 1) // DIM) * ((a.shape[1] + DIM - 1) // DIM)
        executed = full
        if batch is not None:
            batch.add(a, b, rows, shifts, tiles)
            part = None
            calls = 1 if tiles is None else len(tiles)
            if tiles is not None:
                executed = sum(((columns.stop - columns.start + DIM - 1) // DIM) *
                               ((a.shape[1] + DIM - 1) // DIM) for _, columns in tiles)
        elif tiles is None:
            part = a @ b if dot is None else dot(a, b, lane)
            calls = 1
        else:
            part = np.zeros((len(a), b.shape[1]), dtype=np.int64)
            executed = 0
            for rows, columns in tiles:
                part[rows, columns] = a[rows] @ b[:, columns] if dot is None else dot(a[rows], b[:, columns], lane)
                executed += ((columns.stop - columns.start + DIM - 1) // DIM) * ((a.shape[1] + DIM - 1) // DIM)
            calls = len(tiles)
        if fragments is not None:
            fragments[label + "_full"] += full
            fragments[label + "_executed"] += executed
            fragments[label + "_skipped"] += full - executed
            fragments[label + "_dot_calls"] += calls
        return part

    colsum = np.zeros(width, dtype=np.int64)
    if zero_point:
        start = perf_counter_ns()
        for codes, extra in stationary.passes:
            for b in range(len(stationary.theta)):
                colsum += codes[b * BK:(b + 1) * BK].sum(axis=0) << (stationary.theta[b] + extra - minimum)
        mark(times, "pv_colsum", start)
    first = 0
    for stripe in stripes:
        start = perf_counter_ns()
        acc = np.zeros((len(stripe.main), width), dtype=np.int64)
        mark(times, role + "_allocate", start)
        main_tiles = None
        upper_tiles = {}
        main_rows = np.arange(len(stripe.main), dtype=np.int64)
        if qk_mask is not None:
            start = perf_counter_ns()
            main_tiles = qk_mask.tiles(first + main_rows)
            upper_tiles = {lane: qk_mask.tiles(first + rows) for lane, rows, _ in stripe.lanes}
            mark(times, "qk_mask_plan", start)
        for codes, extra in stationary.passes:
            prefix = "qk_hi" if role == "qk" and extra else role
            for b in range(len(stationary.theta)):
                start = perf_counter_ns()
                block = slice(b * BK, (b + 1) * BK)
                shifts = stationary.theta[b] + extra - minimum
                part = partial(stripe.main[:, block], codes[block], 0, prefix + "_main", main_tiles, main_rows, shifts)
                start = mark(times, prefix + "_main_dot", start)
                if part is not None:
                    acc += part << shifts
                mark(times, prefix + "_main_merge", start)
                for lane, rows, plane in stripe.lanes:
                    cols = stripe.columns[b]
                    if len(cols):
                        start = perf_counter_ns()
                        part = partial(plane[:, cols], codes[cols], lane, prefix + "_upper", upper_tiles.get(lane), rows, shifts + 4*lane)
                        start = mark(times, prefix + "_upper_dot", start)
                        if part is not None:
                            acc[rows] += part << (shifts + 4 * lane)
                        mark(times, prefix + "_upper_merge", start)
        if batch is not None:
            start = perf_counter_ns()
            batch.flush(acc)
            mark(times, role + "_batch_execute", start)
        if zero_point:
            start = perf_counter_ns()
            acc += zero_point * colsum
            mark(times, "pv_zero_point", start)
        start = perf_counter_ns()
        output[first:first + len(acc)] = np.ldexp(acc.astype(np.float64), stripe.scale + minimum)
        mark(times, role + "_publish", start)
        first += len(acc)
    if qk_mask is not None:
        start = perf_counter_ns()
        output[qk_mask.blocked] = 0
        mark(times, "qk_mask_zero_fill", start)
    return output
