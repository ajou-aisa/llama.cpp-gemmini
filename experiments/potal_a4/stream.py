from __future__ import annotations

from typing import assert_never

import numpy as np

from .packet import lowdigit, make_packet, stripe_rows
from .types import BK, F32, I64, NEG, MainRule, QuantError, Quantized, StreamPolicy


def eps(x: F32) -> I64:
    return np.where(x != 0, np.frexp(np.abs(x))[1] - 1, NEG).astype(np.int64)


def qround(x: F32, exponent: I64) -> I64:
    with np.errstate(over="ignore", under="ignore"):
        rounded = np.rint(np.ldexp(x, -exponent.astype(np.int32))).astype(np.float64)
    return np.clip(rounded, -(2 ** 31), 2 ** 31 - 1).astype(np.int64)


def shift(q: I64, d: I64) -> I64:
    mag = np.abs(q)
    right = np.clip(-d, 1, 32)
    down = (mag + (1 << (right - 1))) >> right
    down = np.where(d < -32, 0, down) * np.sign(q)
    up = np.clip(q * (1 << np.clip(d, 0, 31)), -(2 ** 31), 2 ** 31 - 1)
    return np.where(d > 0, up, np.where(d < 0, down, q))


def adapt(x: F32, rho: int) -> tuple[I64, I64, NDMask]:
    rows, k = x.shape
    padded = np.pad(x, ((0, 0), (0, (-k) % BK))).reshape(rows, -1, BK)
    e = eps(padded)
    e1 = e.max(-1, keepdims=True)
    e2 = np.where(e < e1, e, NEG).max(-1, keepdims=True)
    top = (e == e1) & (e != NEG) & (e2 != NEG)
    pre = np.where(e2 != NEG, e2, e1)
    q = qround(padded, np.where(pre == NEG, 0, pre - rho))
    population = ~top & (e != NEG)
    magnitude = np.where(population, np.abs(q), 0)
    count = population.sum(-1, keepdims=True)
    total = magnitude.sum(-1, keepdims=True)
    variance = count * (magnitude * magnitude).sum(-1, keepdims=True) - total * total
    centered = count * magnitude - total
    significant = population & (variance > 0) & (centered > 0) & (centered * centered > 4 * variance)
    final = np.where(significant.any(-1, keepdims=True),
                     np.where(population & ~significant, e, NEG).max(-1, keepdims=True), pre)
    q = qround(padded, np.where(final == NEG, 0, final - rho))
    return final[..., 0], q.reshape(rows, -1)[:, :k], (top | significant).reshape(rows, -1)[:, :k]


NDMask = np.ndarray[tuple[int, ...], np.dtype[np.bool_]]


def stream_quant(x: F32, policy: StreamPolicy) -> Quantized:
    if x.ndim != 2 or not x.size or not np.isfinite(x).all() or not 0 <= policy.upper_limbs <= 8:
        raise QuantError("Expected a nonempty finite matrix and 0..8 upper limbs")
    e, q, outlier = adapt(x, policy.rho)
    height = stripe_rows((x.shape[0], policy.outputs, x.shape[1]))
    result = np.empty_like(x)
    packets = []
    saturated = 0
    for start in range(0, len(x), height):
        sl = slice(start, start + height)
        exponents = np.unique(e[sl][e[sl] != NEG])[::-1]
        anchor = int(exponents[min(1, len(exponents) - 1)]) if len(exponents) else 0
        delta = np.repeat(np.where(e[sl] != NEG, e[sl] - anchor, 0), BK, axis=1)[:, :x.shape[1]]
        u = shift(q[sl], delta)
        if policy.upper_limbs < 8:
            places = (16 ** (policy.upper_limbs + 1) - 1) // 15
            limited = np.clip(u, -8 * places, 7 * places)
            saturated += int(np.count_nonzero(u != limited))
            u = limited
        match policy.rule:
            case MainRule.DIRECT:
                main = lowdigit(u)
                residual = u - main
            case MainRule.SELECTIVE:
                correct = outlier[sl] | (delta > 0)
                main = np.where(correct, lowdigit(u), np.clip(u, -8, 7))
                residual = np.where(correct, u - main, 0)
            case MainRule.CLIPPED:
                main = np.clip(u, -8, 7)
                residual = u - main
            case unreachable:
                assert_never(unreachable)
        scale = anchor - policy.rho
        packets.append(make_packet(main, residual, (policy.outputs, scale)))
        result[sl] = np.ldexp((main + residual).astype(np.float32), scale)
    return Quantized(result, tuple(packets), saturated)
