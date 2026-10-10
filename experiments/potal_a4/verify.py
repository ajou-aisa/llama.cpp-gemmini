#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==1.26.4"]
# ///
# How to run: python -m experiments.potal_a4.verify
from __future__ import annotations

import numpy as np

from .attention import p_quant, stationary
from .packet import lowdigit, make_packet, reconstruct_packet, stripe_rows
from .stream import shift, stream_quant
from .types import MainRule, PPolicy, QuantError, StreamPolicy


def check_packets() -> None:
    rng = np.random.default_rng(0)
    values = np.concatenate((np.arange(-65536, 65537), [-(2 ** 31), 2 ** 31 - 1, 0x77777778],
                             rng.integers(-(2 ** 31), 2 ** 31, 10000))).astype(np.int64)
    for start in range(0, len(values), 1000):
        original = values[start:start + 1000].reshape(1, -1)
        main = lowdigit(original)
        packet = make_packet(main, original - main, (32, 0))
        assert np.array_equal(reconstruct_packet(packet), original)
        assert packet.digits.min(initial=0) >= -8 and packet.digits.max(initial=0) <= 7
    print(f"integer packet roundtrips: {len(values)}")


def check_rounding() -> None:
    assert shift(np.asarray([2 ** 30, -(2 ** 31), -(2 ** 31), 1]), np.asarray([-32, -32, -33, 99])).tolist() == [0, -1, 0, 2 ** 31 - 1]
    for k in (1, 31, 32, 33, 63):
        original = np.zeros((3, k), dtype=np.float32)
        result = stream_quant(original, StreamPolicy(35))
        assert np.array_equal(result.values, original)
    try:
        stream_quant(np.asarray([[np.inf]], dtype=np.float32), StreamPolicy(1))
    except QuantError as error:
        assert "finite" in error.detail
    else:
        raise AssertionError("nonfinite input accepted")


def check_linear_boundary() -> None:
    x = np.full((32, 32), 1.99, dtype=np.float32)
    selective = stream_quant(x, StreamPolicy(32, rule=MainRule.SELECTIVE))
    direct = stream_quant(x, StreamPolicy(32))
    assert np.all(selective.values == 1.75) and np.all(direct.values == 2)
    assert sum(p.upper_fragments for p in selective.packets) == 0
    assert sum(p.upper_fragments for p in direct.packets) == 1
    assert sum(p.payload_bytes for p in direct.packets) == 2048


def check_probability() -> None:
    assert stripe_rows((256, 32, 256)) == 256
    p = np.tril(np.ones((256, 256), dtype=np.float32)) / np.arange(1, 257, dtype=np.float32)[:, None]
    original = p_quant(p, PPolicy(32))
    row = p_quant(p, PPolicy(32, rows=1))
    assert original.values[-1].sum() == 0
    assert row.values[-1].sum() == 1
    assert sum(p.upper_fragments for p in original.packets) == 1
    assert np.array_equal(original.values == 0, np.where(p <= 1 / 256, True, False))
    for packet in row.packets:
        reconstructed = np.ldexp(reconstruct_packet(packet, 8).astype(np.float32), int(packet.scales[0]))
        assert np.isfinite(reconstructed).all()
    assert np.all(stationary(np.ones((256, 32), dtype=np.float32), 4) == 1)


def main() -> None:
    check_packets()
    check_rounding()
    check_linear_boundary()
    check_probability()
    print("PASS: int32 digits, shift boundaries, tails, zero, +8 linear and causal P failure/recovery")


if __name__ == "__main__":
    main()
