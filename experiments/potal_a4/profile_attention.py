from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter_ns
from typing import TypedDict

import numpy as np

from .attention import p_quant, softmax, stationary
from .model import Forward, Weights
from .profile_gemm import checked_bound, gemm, unpack
from .profile_ops import Stationary, limb_stats, mark, probability, query, stationary_codes
from .stream import stream_quant
from .types import F32, MODES, MainRule, PPolicy, QuantError, Quantized, StreamPolicy


class CapturedForward(Forward):
    def __init__(self, weights: Weights) -> None:
        super().__init__(weights, next(m for m in MODES if m.name == "original_a4nks"))
        self.captured: list[tuple[F32, F32, F32]] = []

    def attention(self, operands: tuple[F32, F32, F32]) -> F32:
        self.captured.append(tuple(x.copy() for x in operands))
        return super().attention(operands)


def prepare(operands: tuple[F32, F32, F32], times: dict[str, int]) -> tuple[Quantized, Stationary, Quantized, Stationary, dict[str, int]]:
    q, k, v = operands
    sq, selection = query(q, len(k), times)
    sk = stationary_codes(k.T, 8, times, "k")
    sv = stationary_codes(v, 4, times, "v")
    start = perf_counter_ns()
    scores = (sq.values @ sk.values) * np.float32(0.125)
    scores += np.triu(np.full(scores.shape, -np.inf, dtype=np.float32), 1)
    start = mark(times, "reference_float_qk_mask", start)
    p = softmax(scores)
    mark(times, "softmax", start)
    sp = probability(p, v.shape[1], times)
    return sq, sk, sp, sv, selection


def check_components(operands: tuple[F32, F32, F32], parts: tuple[Quantized, Stationary, Quantized, Stationary, dict[str, int]]) -> None:
    q, k, v = operands
    sq, sk, sp, sv, _ = parts
    reference_q = stream_quant(q, StreamPolicy(len(k), 6, MainRule.DIRECT))
    assert np.array_equal(sq.values, reference_q.values)
    assert np.array_equal(sk.values, stationary(k.T, 8))
    assert np.array_equal(sv.values, stationary(v, 4))
    mask = np.triu(np.full((len(q), len(k)), -np.inf, dtype=np.float32), 1)
    p = softmax((reference_q.values @ sk.values) * np.float32(0.125) + mask)
    reference_p = p_quant(p, PPolicy(v.shape[1]))
    assert np.array_equal(sp.values, reference_p.values)
    for actual, expected in ((sq, reference_q), (sp, reference_p)):
        for left, right in zip(actual.packets, expected.packets, strict=True):
            for name in ("main", "digits", "row_lane", "columns", "scales"):
                assert np.array_equal(getattr(left, name), getattr(right, name))
            assert left.main_fragments == right.main_fragments and left.upper_fragments == right.upper_fragments


class ComponentCheck(TypedDict):
    relative_error: float
    bitwise_float64: bool


class TimedCase(TypedDict):
    warmup_runs: int
    timed_runs: list[dict[str, int]]
    checks: dict[str, ComponentCheck]


def timed_case(operands: tuple[F32, F32, F32]) -> TimedCase:
    records = []
    checks: dict[str, ComponentCheck] = {}
    for repeat in range(6):
        times: dict[str, int] = {}
        sq, sk, sp, sv, _ = prepare(operands, times)
        start = perf_counter_ns()
        q_plan, p_plan = unpack(sq), unpack(sp)
        mark(times, "reference_unpack", start)
        checked_bound(q_plan, sk, 0)
        checked_bound(p_plan, sv, 8)
        scores = gemm(q_plan, sk, "qk", times)
        output = gemm(p_plan, sv, "pv", times, 8)
        if repeat == 0:
            for role, actual, expected in (
                    ("qk", scores, sq.values.astype(np.float64) @ sk.values.astype(np.float64)),
                    ("pv", output, sp.values.astype(np.float64) @ sv.values.astype(np.float64))):
                error = float(np.max(np.abs(actual - expected)) / max(float(np.max(np.abs(expected))), 1e-300))
                assert error < 1e-12
                checks[role] = {"relative_error": error, "bitwise_float64": bool(np.array_equal(actual, expected))}
            expected_scores, expected_output = scores, output
        else:
            assert np.array_equal(scores, expected_scores) and np.array_equal(output, expected_output)
            records.append(times)
    return {"warmup_runs": 1, "timed_runs": records, "checks": checks}


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main() -> None:
    if len(sys.argv) != 3:
        raise QuantError("Expected BASELINE.json NEW_OUTPUT.json")
    baseline_path, output = (Path(p) for p in sys.argv[1:])
    if output.exists() or output.with_suffix(".captures.npz").exists():
        raise QuantError("Profiling outputs must be new")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "2":
        raise QuantError("Use OPENBLAS_NUM_THREADS=2 for comparable FP capture")
    baseline = json.loads(baseline_path.read_text())
    if baseline["mode"]["name"] != "original_a4nks":
        raise QuantError("Only original_a4nks is an accepted profiling policy")
    tokens_path = baseline_path.parent / "test.i32"
    assert digest(tokens_path) == baseline["tokens_sha256"]
    model_path = Path(baseline["model"])
    assert digest(model_path) == baseline["model_sha256"]
    tokens = np.fromfile(tokens_path, dtype="<i4")[:baseline["context"]]
    model = CapturedForward(Weights(model_path))
    hidden = model.run(tokens)
    with np.load(baseline_path.with_suffix(".chunk0.npz")) as saved:
        assert np.array_equal(hidden, saved["hidden"])
    assert len(model.captured) == 12
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix(".captures.npz"),
                        **{f"l{layer}_{role}": operand for layer, operands in enumerate(model.captured)
                           for role, operand in zip(("q", "k", "v"), operands, strict=True)})
    records = []
    for layer, matrices in enumerate(model.captured):
        heads = [x.reshape(len(tokens), 12, 64).transpose(1, 0, 2) for x in matrices]
        for head in range(12):
            operands = tuple(x[head] for x in heads)
            assert all(np.isfinite(x).all() for x in operands)
            parts = prepare(operands, {})
            check_components(operands, parts)
            sq, sk, sp, sv, selection = parts
            record = {"layer": layer, "head": head, "q": limb_stats(sq), "p": limb_stats(sp),
                      "selection": selection, "k_values": sk.values.size, "v_values": sv.values.size,
                      "k_pass_nonzeros": [int(np.count_nonzero(codes)) for codes, _ in sk.passes],
                      "v_nonzeros": int(np.count_nonzero(sv.passes[0][0]))}
            if layer in (0, 5, 11) and head in (0, 6, 11):
                record["timing"] = timed_case(operands)
            records.append(record)
        print(f"context={len(tokens)} layer={layer}: 12 heads checked", flush=True)
    report = {"scope": "Original-policy attention component replay; NumPy int64 dot and checked integer SCU, not Gemmini timing",
              "context": len(tokens), "heads": len(records), "timed_heads": 9,
              "baseline": str(baseline_path), "baseline_sha256": digest(baseline_path),
              "model_sha256": baseline["model_sha256"], "tokens_sha256": baseline["tokens_sha256"],
              "capture_sha256": digest(output.with_suffix(".captures.npz")),
              "hidden_bitwise_matches_baseline_chunk0": True,
              "numpy": np.__version__, "platform": platform.platform(), "machine": platform.machine(),
              "blas_threads": 2, "integer_dot_threads": 1,
              "excluded_timing_keys": ["reference_float_qk_mask", "reference_unpack", "q_reference_reconstruct",
                                       "p_reference_reconstruct", "k_reference_reconstruct", "v_reference_reconstruct"],
              "source_sha256": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
              "records": records}
    with output.open("x") as target:
        json.dump(report, target, indent=2)
    print(output, flush=True)


if __name__ == "__main__":
    main()
