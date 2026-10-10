from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np

from .profile_ops import Stationary, mark
from .types import BK, I64, QuantError, Quantized


@dataclass(frozen=True, slots=True)
class Stripe:
    main: I64
    scale: int
    lanes: tuple[tuple[int, I64, I64], ...]
    columns: tuple[I64, ...]


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
         times: dict[str, int], zero_point: int = 0) -> np.ndarray:
    minimum = int(stationary.theta.min())
    width = stationary.values.shape[1]
    output = np.empty((sum(len(st.main) for st in stripes), width), dtype=np.float64)
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
        for codes, extra in stationary.passes:
            prefix = "qk_hi" if role == "qk" and extra else role
            for b in range(len(stationary.theta)):
                start = perf_counter_ns()
                block = slice(b * BK, (b + 1) * BK)
                part = stripe.main[:, block] @ codes[block]
                start = mark(times, prefix + "_main_dot", start)
                shifts = stationary.theta[b] + extra - minimum
                acc += part << shifts
                mark(times, prefix + "_main_merge", start)
                for lane, rows, plane in stripe.lanes:
                    cols = stripe.columns[b]
                    if len(cols):
                        start = perf_counter_ns()
                        part = plane[:, cols] @ codes[cols]
                        start = mark(times, prefix + "_upper_dot", start)
                        acc[rows] += part << (shifts + 4 * lane)
                        mark(times, prefix + "_upper_merge", start)
        if zero_point:
            start = perf_counter_ns()
            acc += zero_point * colsum
            mark(times, "pv_zero_point", start)
        start = perf_counter_ns()
        output[first:first + len(acc)] = np.ldexp(acc.astype(np.float64), stripe.scale + minimum)
        mark(times, role + "_publish", start)
        first += len(acc)
    return output
