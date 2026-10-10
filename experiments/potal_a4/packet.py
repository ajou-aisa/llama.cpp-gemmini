from __future__ import annotations

import math

import numpy as np

from .types import BK, DIM, I64, Packet


def stripe_rows(shape: tuple[int, int, int]) -> int:
    i, j, k = (math.ceil(v / DIM) for v in shape)
    ti, tj, tk = min(i, 2), min(j, 2), min(k, 64)
    while True:
        before = ti, tj, tk
        if tj < j and (ti + tj + 1) * tk * DIM <= 8192 and ti * (tj + 1) * DIM <= 256:
            tj += 1
        if ti < i and (ti + 1 + tj) * tk * DIM <= 8192 and (ti + 1) * tj * DIM <= 256:
            ti += 1
        if tk < k and (ti + tj) * (tk + 1) * DIM <= 8192:
            tk += 1
        if before == (ti, tj, tk):
            return ti * DIM


def lowdigit(u: I64) -> I64:
    return ((u + 8) & 15) - 8


def make_packet(main: I64, residual: I64, geometry: tuple[int, int]) -> Packet:
    outputs, scale = geometry
    rows, k = main.shape
    blocks = math.ceil(k / BK)
    active: list[tuple[int, int]] = []
    planes: list[I64] = []
    quotient = residual.copy()
    useful = 0
    lane = 0
    while np.any(quotient):
        digit = lowdigit(quotient)
        quotient = (quotient - digit) // 16
        selected = np.flatnonzero(np.any(digit, axis=1))
        active.extend((lane, int(row)) for row in selected)
        planes.extend(digit[row] for row in selected)
        useful += int(np.count_nonzero(digit))
        lane += 1
    columns = [np.flatnonzero(np.any(residual[:, b:b + BK], axis=0)) + b
               for b in range(0, k, BK)]
    padded_k = sum(math.ceil(len(c) / DIM) * DIM for c in columns)
    padded_m = math.ceil(len(active) / DIM) * DIM
    payload = np.zeros((padded_m, padded_k), dtype=np.int8)
    offset = 0
    for cols in columns:
        if len(cols):
            for index, plane in enumerate(planes):
                payload[index, offset:offset + len(cols)] = plane[cols]
        offset += math.ceil(len(cols) / DIM) * DIM
    return Packet(
        main.astype(np.int8), payload,
        np.asarray(active, dtype=np.uint16).reshape(-1, 2),
        np.concatenate(columns).astype(np.uint32),
        np.asarray([scale], dtype=np.int16), useful,
        math.ceil(rows / DIM) * math.ceil(outputs / DIM) * blocks,
        (padded_m // DIM) * math.ceil(outputs / DIM) * (padded_k // DIM),
    )


def reconstruct_packet(packet: Packet, zero_point: int = 0) -> I64:
    out = packet.main.astype(np.int64) + zero_point
    offset = 0
    for b in range(0, out.shape[1], BK):
        cols = packet.columns[(packet.columns >= b) & (packet.columns < b + BK)]
        for index, (lane, row) in enumerate(packet.row_lane):
            out[row, cols] += packet.digits[index, offset:offset + len(cols)].astype(np.int64) * (16 ** int(lane))
        offset += math.ceil(len(cols) / DIM) * DIM
    return out
