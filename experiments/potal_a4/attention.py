from __future__ import annotations

import numpy as np

from .packet import make_packet, stripe_rows
from .stream import eps, qround
from .types import BK, F32, NEG, PPolicy, QuantError, Quantized


def p_quant(p: F32, policy: PPolicy) -> Quantized:
    if not np.isfinite(p).all() or np.any(p < 0) or policy.bits not in (4, 6, 8) or policy.rows < 0:
        raise QuantError("Expected finite nonnegative probabilities, 4/6/8 bits and nonnegative rows")
    geometry = stripe_rows((len(p), policy.outputs, p.shape[1]))
    height = min(geometry, policy.rows) if policy.rows else geometry
    result = np.empty_like(p)
    packets = []
    for start in range(0, len(p), height):
        sl = slice(start, start + height)
        maximum = p[sl].max()
        scale = (int(eps(np.asarray(maximum))) if maximum > 0 else 0) - (policy.bits - 1)
        q = np.clip(qround(p[sl], np.asarray(scale)), 0, (1 << policy.bits) - 1)
        low = q & 15
        packets.append(make_packet(low - 8, q - low, (policy.outputs, scale)))
        result[sl] = np.ldexp(q.astype(np.float32), scale)
    return Quantized(result, tuple(packets))


def stationary(x: F32, bits: int) -> F32:
    rows, columns = x.shape
    padded = np.pad(x, ((0, (-rows) % BK), (0, 0))).T.reshape(columns, -1, BK)
    exponent = eps(padded).max(-1, keepdims=True)
    scale = np.where(exponent == NEG, 0, exponent - (bits - 2))
    codes = np.clip(qround(padded, scale), -(1 << (bits - 1)), 119 if bits == 8 else 7)
    return np.ldexp(codes.astype(np.float32), scale.astype(np.int32)).reshape(columns, -1).T[:rows]


def softmax(scores: F32) -> F32:
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True, dtype=np.float64).astype(np.float32)
