#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==1.26.4"]
# ///
# How to run: python -m experiments.potal_a4.verify
from __future__ import annotations

from collections import Counter
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import numpy as np

from .attention import p_quant, stationary
from .packet import lowdigit, make_packet, reconstruct_packet, stripe_rows
from .profile_gemm import QKTileMask, Stripe, gemm
from .native_measure import CpuBatch, prepare_head
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


def check_qk_tiles(batch: CpuBatch | None = None) -> None:
    rng = np.random.default_rng(72)
    for rows, keys in ((512, 512), (65, 97), (2, 128), (1, 64)):
        q = rng.normal(size=(rows, 32)).astype(np.float32)
        k = rng.normal(size=(keys, 32)).astype(np.float32)
        operand, plan, weights = prepare_head(0, q, k, {})
        plan = tuple(Stripe(st.main, st.scale,
                            tuple((lane, active[::-1], plane[::-1]) for lane, active, plane in st.lanes),
                            st.columns) for st in plan)
        positions = np.arange(rows) if rows > 2 else np.asarray([0, 100])[-rows:]
        blocked = np.arange(keys)[None, :] > positions[:, None]
        mask = np.where(blocked, -np.inf, -0.25).astype(np.float32)
        expected = operand.values.astype(np.float64) @ weights.values.astype(np.float64)
        counts = Counter()
        result = gemm(plan, weights, "qk", {}, qk_mask=QKTileMask.from_values(mask), fragments=counts, batch=batch)
        assert np.array_equal(result[~blocked], expected[~blocked])
        assert np.all(result[blocked] == 0) and np.isfinite(result).all()
        if rows == 512:
            assert counts["qk_main_full"] == 256 and counts["qk_main_executed"] == 136
            assert counts["qk_hi_main_full"] == 256 and counts["qk_hi_main_skipped"] == 120
        if rows == 1:
            assert counts["qk_main_skipped"] == 0
        if rows == 2:
            assert result[1, 32:64].any()
    for mask in (np.full((65, 97), -0.5, np.float32),
                 np.where(np.arange(65)[:, None] % 2 == np.arange(97)[None, :] % 2, 0, -np.inf).astype(np.float32),
                 np.full((65, 97), -np.inf, np.float32)):
        operand, plan, weights = prepare_head(0, rng.normal(size=(65, 32)).astype(np.float32),
                                              rng.normal(size=(97, 32)).astype(np.float32), {})
        expected = operand.values.astype(np.float64) @ weights.values.astype(np.float64)
        result = gemm(plan, weights, "qk", {}, qk_mask=QKTileMask.from_values(mask), batch=batch)
        live = ~np.isneginf(mask)
        assert np.array_equal(result[live], expected[live]) and np.all(result[~live] == 0)
    for invalid in (np.asarray([[np.nan]], np.float32), np.asarray([[np.inf]], np.float32)):
        try:
            QKTileMask.from_values(invalid)
        except QuantError as error:
            assert "mask" in error.detail
        else:
            raise AssertionError("Invalid mask accepted")
    print("QK tiles: causal/tails, original residual rows, finite bias, sequence masks and decode PASS")


def check_native_batch(library: str) -> None:
    from .im2p_measure import CompletedBatch, PreparedBatch

    lib = ct.CDLL(library)
    for direct in (False, True):
        batch = CpuBatch(lib, direct)
        a = np.array([[3], [-5]], np.int64)
        b = np.array([[1, -2, 7]], np.int64)
        rows = np.array([2, 0], np.int64)
        shifts = np.array([0, 3, 21], np.int64)
        out = np.full((3, 3), 123, np.int64)
        batch.add(a, b, rows, shifts, None)
        batch.add(-a, b, rows, shifts + 4, None)
        batch.flush(out)
        expected = np.array([[75, -1200, 1101004800], [0, 0, 0], [-45, 720, -660602880]], np.int64)
        assert np.array_equal(out, expected)

        wide = np.full((1, 32), -8, np.int64)
        batch.add(wide, wide.T, np.array([0]), np.array([21]), None)
        out = np.zeros((1, 1), np.int64)
        batch.flush(out)
        assert out[0, 0] == 4294967296

        for invalid in ("overflow", "row", "region"):
            out.fill(123)
            batch.add(np.ones((1, 1), np.int64), np.ones((1, 1), np.int64), np.array([0]), np.array([0]), None)
            batch.add(np.ones((1, 1), np.int64), np.ones((1, 1), np.int64),
                      np.array([1 if invalid == "row" else 0]), np.array([63 if invalid == "overflow" else 0]),
                      ((slice(0, 2), slice(0, 1)),) if invalid == "region" else None)
            try:
                batch.flush(out)
            except QuantError:
                assert out[0, 0] == 123
                assert not batch.work and not batch.buffers
            else:
                raise AssertionError(f"Native batch accepted {invalid}")
        try:
            batch.add(np.array([[256]]), np.array([[1]]), np.array([0]), np.array([0]), None)
        except QuantError:
            assert not batch.work
        else:
            raise AssertionError("Batch silently narrowed a non-nibble input")
        check_qk_tiles(batch)
        rng = np.random.default_rng(132)
        for length in (1, 33, 65):
            p = rng.random((35, length), dtype=np.float32)
            p[:2] = 0
            v = rng.normal(size=(7, length)).astype(np.float32)
            operand, plan, weights = prepare_head(1, p, v, {})
            expected = operand.values.astype(np.float64) @ weights.values.astype(np.float64)
            result = gemm(plan, weights, "pv", {}, 8, batch=batch)
            assert np.array_equal(result, expected)
            assert np.all(result[:2] == 0)
    for role in (0, 1):
        streamed = rng.normal(size=(35, 33)).astype(np.float32) if role == 0 else rng.random((35, 33), dtype=np.float32)
        streamed[:2] = 0
        stationary_values = rng.normal(size=(65, 33)).astype(np.float32)
        operand, plan, weights = prepare_head(role, streamed, stationary_values, {})
        name, zero_point = ("qk", 0) if role == 0 else ("pv", 8)
        reference = operand.values.astype(np.float64) @ weights.values.astype(np.float64)
        masks = [None]
        if role == 0:
            masks += [QKTileMask(np.arange(65)[None, :] > np.arange(35)[:, None]),
                      QKTileMask(np.ones(reference.shape, bool))]
        for mask in masks:
            prepared = PreparedBatch(lib)
            planned = gemm(plan, weights, name, {}, zero_point, qk_mask=mask, batch=prepared)
            expected = np.where(mask.blocked, 0, reference) if mask is not None else reference
            assert np.array_equal(planned, expected)
            counts = []
            for direct in (False, True):
                stats = prepared.execute(direct)
                counts.append(stats["raw_dot_calls"])
                prepared.verify()
                result = gemm(plan, weights, name, {}, zero_point, qk_mask=mask, batch=CompletedBatch(prepared))
                assert np.array_equal(result, expected)
            assert counts[0] == counts[1]
    print("Native batch: block/column/lane shifts, >int32 sum, failure publication, int8 range, QK masks and PV zero point PASS")
    print("Prepared replay: reused int8 jobs, row/K/column tails, all-masked QK and zero P PASS (CPU provider)")


def check_backend(executable: str, repeats: int = 1, report: Path | None = None, require_rtl: bool = False) -> None:
    subprocess.run([executable, "--case=attention"], check=True)
    rng = np.random.default_rng(782)
    cases = []
    for role in (0, 1):
        for k, rows, n, half, interleaved in ((33, 35, 65, 0, 0), (65, 1, 7, 1, 1), (32, 67, 33, 0, 1)):
            a = rng.normal(size=(1, 2, n, k)).astype(np.float32)
            a *= np.exp2(rng.integers(-3, 4, (1, 2, n, 1))).astype(np.float32)
            if half:
                a = a.astype(np.float16).astype(np.float32)
            b = (rng.normal(size=(2, 4, rows, k)).astype(np.float32) if role == 0 else
                 rng.random((2, 4, rows, k), dtype=np.float32))
            if rows > 1:
                b[..., 0, :] = 0
            result = np.empty((2, 4, rows, n), np.float32)
            for batch in range(2):
                for head in range(4):
                    operand, _, weights = prepare_head(role, b[batch, head], a[0, head//2], {})
                    result[batch, head] = operand.values.astype(np.float64) @ weights.values.astype(np.float64)
            cases.append((f"random-{'qk' if role == 0 else 'pv'}-{k}-{rows}",
                          (role, k, n, rows, 2, 4, 1, 2, half, interleaved), a, b, result))
    a = np.ones((3, 33), np.float32)
    b = np.ones((2, 33), np.float32)
    b[:, 0] = [np.float32(2.0**120), -np.float32(2.0**120)]
    operand, _, weights = prepare_head(0, b, a, {})
    expected = (operand.values.astype(np.float64) @ weights.values.astype(np.float64)).astype(np.float32)
    cases.append(("qk-int32-saturation-carry", (0, 33, 3, 2, 1, 1, 1, 1, 0, 0), a, b, expected))
    base = Path(__file__).parent
    for directory in ("results-upper-compare-gpt2-10chunk", "results-qk-batch-10chunk"):
        for role, name in enumerate(("qk-l0-h0.npz", "pv-l0-h0.npz")):
            with np.load(base/directory/name, allow_pickle=False) as saved:
                a, b = saved["stationary"], saved["streamed"]
                operand, _, weights = prepare_head(role, b, a, {})
                expected = (operand.values.astype(np.float64) @ weights.values.astype(np.float64)).astype(np.float32)
                if not np.array_equal(expected, saved["result"]):
                    raise AssertionError(f"Saved reference changed: {directory}/{name}")
                shape = (role, b.shape[1], a.shape[0], b.shape[0], 1, 1, 1, 1, 0, 0)
                cases.append((f"{directory}/{name}", shape, a.copy(), b.copy(), expected))
    records = []
    with tempfile.TemporaryDirectory(prefix="potal-backend-") as directory:
        fixture = Path(directory)/"attention.bin"
        for name, shape, a, b, expected in cases:
            with fixture.open("wb") as stream:
                np.asarray(shape, dtype="<i8").tofile(stream)
                for values in (a, b, expected):
                    np.asarray(values, dtype="<f4").tofile(stream)
            for repeat in range(repeats):
                result = subprocess.run([executable, "--attention-fixture", str(fixture)], check=True,
                                        capture_output=True, text=True)
                line = next(line for line in result.stdout.splitlines() if line.startswith("ATTENTION "))
                metrics = dict(field.split("=", 1) for field in line.split()[1:])
                record = {"case": name, "shape": shape, "repeat": repeat, "equal": True,
                          "role": metrics.pop("role"), **{key: int(value) for key, value in metrics.items()}}
                if require_rtl and record["raw_cycles"] <= 0:
                    raise AssertionError("RTL measurement requested but no actual Gemmini cycles were recorded")
                records.append(record)
                print(json.dumps(record), flush=True)
    if report:
        with report.open("x") as target:
            json.dump({"scope": "Native Gemmini backend QK/PV nodes; vanilla, no masked tile skipping",
                       "platform": platform.platform(), "executable": str(Path(executable).resolve()),
                       "executable_sha256": hashlib.sha256(Path(executable).read_bytes()).hexdigest(),
                       "require_rtl": require_rtl, "repeats": repeats,
                       "host_ns": "Graph compute including native quantization, packing, provider creation/dots, wide merge and publication; fixture loading and oracle comparison excluded",
                       "raw_cycles": "Real IM2P provider cycles; zero denotes CPU functional execution",
                       "limitations": ["Host simulator wall time is not NPU latency", "Single layer/head captures do not measure full-model speed", "Each repeat creates a fresh backend"],
                       "records": records}, target, indent=2)
    print("PASS: native backend QK/PV vs independent reference, GQA/batches, F16/views, tails, zero P and captured GPT-2/Llama")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--backend":
        parser = argparse.ArgumentParser()
        parser.add_argument("--backend", required=True)
        parser.add_argument("--repeats", type=int, default=1)
        parser.add_argument("--report", type=Path)
        parser.add_argument("--require-rtl", action="store_true")
        args = parser.parse_args()
        if args.repeats < 1:
            parser.error("repeats must be positive")
        check_backend(args.backend, args.repeats, args.report, args.require_rtl)
        return
    if len(sys.argv) == 3 and sys.argv[1] == "--native-batch":
        check_native_batch(sys.argv[2])
        return
    check_packets()
    check_rounding()
    check_linear_boundary()
    check_probability()
    check_qk_tiles()
    print("PASS: int32 digits, shift boundaries, tails, zero, +8 linear and causal P failure/recovery")


if __name__ == "__main__":
    main()
