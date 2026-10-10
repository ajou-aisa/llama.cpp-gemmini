from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import TypedDict

import numpy as np

from .packet import lowdigit, make_packet, stripe_rows
from .stream import eps, qround, shift
from .types import BK, F32, I64, NEG, Quantized


def mark(times: dict[str, int], name: str, start: int) -> int:
    now = perf_counter_ns()
    times[name] = times.get(name, 0) + now - start
    return now


def query(x: F32, outputs: int, times: dict[str, int]) -> tuple[Quantized, dict[str, int]]:
    start = perf_counter_ns()
    rows, k = x.shape
    padded = np.pad(x, ((0, 0), (0, (-k) % BK))).reshape(rows, -1, BK)
    e = eps(padded)
    e1 = e.max(-1, keepdims=True)
    e2 = np.where(e < e1, e, NEG).max(-1, keepdims=True)
    top = (e == e1) & (e != NEG) & (e2 != NEG)
    pre = np.where(e2 != NEG, e2, e1)
    start = mark(times, "q_exponent_top", start)
    q = qround(padded, np.where(pre == NEG, 0, pre - 6))
    start = mark(times, "q_preliminary_round", start)
    population = ~top & (e != NEG)
    magnitude = np.where(population, np.abs(q), 0)
    count = population.sum(-1, keepdims=True)
    total = magnitude.sum(-1, keepdims=True)
    variance = count * (magnitude * magnitude).sum(-1, keepdims=True) - total * total
    centered = count * magnitude - total
    significant = population & (variance > 0) & (centered > 0) & (centered * centered > 4 * variance)
    final = np.where(significant.any(-1, keepdims=True),
                     np.where(population & ~significant, e, NEG).max(-1, keepdims=True), pre)
    start = mark(times, "q_sigma_select", start)
    q = qround(padded, np.where(final == NEG, 0, final - 6)).reshape(rows, -1)[:, :k]
    final = final[..., 0]
    mark(times, "q_final_round", start)
    selection = {"values": x.size, "top": int(top.sum()), "sigma": int(significant.sum())}
    height = stripe_rows((rows, outputs, k))
    result = np.empty_like(x)
    packets = []
    for first in range(0, rows, height):
        start = perf_counter_ns()
        sl = slice(first, first + height)
        ex = np.unique(final[sl][final[sl] != NEG])[::-1]
        anchor = int(ex[min(1, len(ex) - 1)]) if len(ex) else 0
        delta = np.repeat(np.where(final[sl] != NEG, final[sl] - anchor, 0), BK, axis=1)[:, :k]
        u = shift(q[sl], delta)
        main = lowdigit(u)
        residual = u - main
        start = mark(times, "q_fold_main", start)
        packets.append(make_packet(main, residual, (outputs, anchor - 6)))
        start = mark(times, "q_packet", start)
        result[sl] = np.ldexp((main + residual).astype(np.float32), anchor - 6)
        mark(times, "q_reference_reconstruct", start)
    return Quantized(result, tuple(packets)), selection


def probability(p: F32, outputs: int, times: dict[str, int]) -> Quantized:
    height = stripe_rows((len(p), outputs, p.shape[1]))
    result = np.empty_like(p)
    packets = []
    for first in range(0, len(p), height):
        start = perf_counter_ns()
        sl = slice(first, first + height)
        maximum = p[sl].max()
        scale = (int(eps(np.asarray(maximum))) if maximum > 0 else 0) - 7
        q = np.clip(qround(p[sl], np.asarray(scale)), 0, 255)
        lo = q & 15
        start = mark(times, "p_range_round_main", start)
        packets.append(make_packet(lo - 8, q - lo, (outputs, scale)))
        start = mark(times, "p_packet", start)
        result[sl] = np.ldexp(q.astype(np.float32), scale)
        mark(times, "p_reference_reconstruct", start)
    return Quantized(result, tuple(packets))


@dataclass(frozen=True, slots=True)
class Stationary:
    passes: tuple[tuple[I64, int], ...]
    theta: I64
    values: F32


def stationary_codes(x: F32, bits: int, times: dict[str, int], role: str) -> Stationary:
    start = perf_counter_ns()
    rows, columns = x.shape
    padded = np.pad(x, ((0, (-rows) % BK), (0, 0))).T.reshape(columns, -1, BK)
    exponent = eps(padded).max(-1, keepdims=True)
    scale = np.where(exponent == NEG, 0, exponent - (bits - 2))
    codes = np.clip(qround(padded, scale), -(1 << (bits - 1)), 119 if bits == 8 else 7)
    code_matrix = codes.reshape(columns, -1).T[:rows].copy()
    if bits == 8:
        lo = lowdigit(code_matrix)
        passes = ((lo, 0), ((code_matrix - lo) // 16, 4))
    else:
        passes = ((code_matrix, 0),)
    start = mark(times, role + "_range_round_split", start)
    values = np.ldexp(codes.astype(np.float32), scale.astype(np.int32)).reshape(columns, -1).T[:rows]
    mark(times, role + "_reference_reconstruct", start)
    return Stationary(passes, scale[..., 0].T.copy(), values)


class LimbStats(TypedDict):
    values: int
    stripes: int
    lane_nonzeros: dict[int, int]
    lane_active_rows: dict[int, int]
    lane_stripes: dict[int, int]
    nonzero_upper_digits_per_value: dict[int, int]
    main_fragments: int
    upper_fragments: int
    main_bytes: int
    upper_bytes: int
    metadata_bytes: int


def limb_stats(q: Quantized) -> LimbStats:
    counts: dict[int, int] = {}
    active_rows: dict[int, int] = {}
    stripe_presence: dict[int, int] = {}
    histogram: dict[int, int] = {}
    for packet in q.packets:
        upper_count = np.zeros(packet.main.shape, dtype=np.uint8)
        lanes = packet.row_lane[:, 0]
        for lane in np.unique(lanes):
            indices = np.flatnonzero(lanes == lane)
            counts[int(lane)] = counts.get(int(lane), 0) + int(np.count_nonzero(packet.digits[indices]))
            active_rows[int(lane)] = active_rows.get(int(lane), 0) + len(indices)
            stripe_presence[int(lane)] = stripe_presence.get(int(lane), 0) + 1
        offset = 0
        for first in range(0, packet.main.shape[1], BK):
            cols = packet.columns[(packet.columns >= first) & (packet.columns < first + BK)]
            for index, (_, row) in enumerate(packet.row_lane):
                upper_count[row, cols] += packet.digits[index, offset:offset + len(cols)] != 0
            offset += ((len(cols) + 31) // 32) * 32
        values, frequencies = np.unique(upper_count, return_counts=True)
        for value, frequency in zip(values, frequencies, strict=True):
            histogram[int(value)] = histogram.get(int(value), 0) + int(frequency)
    return {"values": q.values.size, "stripes": len(q.packets), "lane_nonzeros": counts,
            "lane_active_rows": active_rows, "lane_stripes": stripe_presence,
            "nonzero_upper_digits_per_value": histogram,
            "main_fragments": sum(p.main_fragments for p in q.packets),
            "upper_fragments": sum(p.upper_fragments for p in q.packets),
            "main_bytes": sum(p.main.nbytes for p in q.packets),
            "upper_bytes": sum(p.digits.nbytes for p in q.packets),
            "metadata_bytes": sum(p.metadata_bytes for p in q.packets)}
