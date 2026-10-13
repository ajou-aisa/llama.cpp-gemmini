#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
# How to run: from the experimental checkout, with its existing NumPy environment:
# PYTHONPATH=gguf-py:. OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=6 python PATH/compare.py
# Requires the completed GPT-2 replay and the existing Llama capture directories.

from __future__ import annotations

from collections import Counter
import ctypes as ct
import json
import os
from pathlib import Path
import platform
from statistics import median
from typing import Final, TypedDict

import numpy as np

from experiments.potal_a4.native_measure import measure_components, prepare_head
from experiments.potal_a4.packet import lowdigit
from experiments.potal_a4.profile_gemm import gemm
from experiments.potal_a4.run import digest
from experiments.potal_a4.types import I64

BASE: Final = Path(__file__).resolve().parents[1]
LIBRARY: Final = BASE.parents[1] / "build-potal-replay/libpotal-native-replay.dylib"


class DigitSummary(TypedDict):
    nonzeros: int
    unit_nonzeros: int
    unit_fraction_of_nonzeros: float
    histogram: dict[int, int]


def summarize(histogram: Counter[int]) -> DigitSummary:
    nonzeros = sum(count for value, count in histogram.items() if value)
    units = histogram[-1] + histogram[1]
    return {"nonzeros": nonzeros, "unit_nonzeros": units,
            "unit_fraction_of_nonzeros": units / nonzeros if nonzeros else 0.0,
            "histogram": dict(sorted(histogram.items()))}


def sign_only_upper(a: I64, b: I64, lane: int) -> I64:
    """Deliberately lossy negative control; never used in the candidate or PPL run."""
    return (np.sign(a) if lane >= 2 else a) @ b


def main() -> None:
    output = Path(__file__).with_name("comparison.json")
    if output.exists():
        raise FileExistsError(output)
    lib = ct.CDLL(str(LIBRARY))
    report = {"scope": "Unchanged real digits; CPU exact upper-lane specialization, not NPU timing",
              "platform": platform.platform(), "numpy": np.__version__,
              "environment": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
              "library_sha256": digest(LIBRARY), "models": {},
              "timing": "Sum of per-capture medians, one warmup and five rotated-order runs; excludes common quantization",
              "histogram_denominator": "Nonzero digits only; zeros in absent rows/stripes are not materialized",
              "limitations": ["Q/P full histograms cover each model's 10-chunk PPL run; K/V cover only the saved heads",
                              "Llama PPL is preserved Nano evidence; local Llama timing uses the newer local captures",
                              "CPU timings and scalar operation specialization do not establish Gemmini speedup"],
              "source_sha256": {p.name: digest(p) for p in (
                  Path(__file__), BASE / "native_measure.py", BASE / "native_model.cpp",
                  BASE / "profile_ops.py", BASE / "profile_gemm.py", BASE / "stream.py", BASE / "packet.py")}}
    for name, captures_name, accuracy_name in (
            ("gpt2", "results-upper-compare-gpt2-10chunk", "results-upper-compare-gpt2-10chunk"),
            ("llama3.2-1B", "results-qk-batch-10chunk", "results-unit-add-llama512")):
        captures, accuracy_dir = BASE / captures_name, BASE / accuracy_name
        accuracy = json.loads((accuracy_dir / "report.json").read_text())
        records = [json.loads(line) for line in (accuracy_dir / "chunks.jsonl").read_text().splitlines()]
        assert accuracy["chunks"] == 10 and len(records) == 20
        for chunk in range(10):
            pair = [record for record in records if record["chunk"] == chunk]
            assert {record["mode"] for record in pair} == {"dense", "sparse_unit_add"}
            for key in ("nll", "logits_sha256", "attention_trace_sha256"):
                assert pair[0][key] == pair[1][key], (name, chunk, key)
        full_histograms = {role: {lane: summarize(Counter({int(v): c for v, c in counts.items()}))
                                  for lane, counts in lanes.items()}
                           for role, lanes in accuracy["digit_histograms"].items()}
        histograms: dict[str, Counter[int]] = {}
        controls = []
        paths = []
        for prefix, role, streamed_role, stationary_role, zp in (("qk", 0, "q", "k", 0), ("pv", 1, "p", "v", 8)):
            for path in sorted(captures.glob(prefix + "-*.npz")):
                paths.append(path)
                with np.load(path) as saved:
                    operand, plan, weights = prepare_head(role, saved["streamed"], saved["stationary"], {})
                    reference = gemm(plan, weights, prefix, {}, zp)
                    reconstructed = operand.values.astype(np.float64) @ weights.values.astype(np.float64)
                    assert np.array_equal(reference, reconstructed), path.name
                    assert np.array_equal(reference.astype(np.float32), saved["result"]), path.name
                    live = ~np.isneginf(saved["mask"]) if prefix.startswith("qk") else np.ones(reference.shape, bool)
                    lossy = gemm(plan, weights, prefix, {}, zp, sign_only_upper)
                    controls.append({"capture": path.name, "sha256": digest(path),
                                     "bitwise_reconstructed_float64": True,
                                     "lossy_sign_changed_valid_outputs": int(np.count_nonzero(lossy[live] != reference[live])),
                                     "lossy_sign_max_abs_error_valid": float(np.abs(lossy[live] - reference[live]).max())})
                for stripe in plan:
                    for lane, _, plane in stripe.lanes:
                        values, counts = np.unique(plane, return_counts=True)
                        histograms.setdefault(f"{streamed_role}{lane}", Counter()).update(
                            {int(v): int(c) for v, c in zip(values, counts, strict=True)})
                for codes, extra in weights.passes:
                    values, counts = np.unique(codes, return_counts=True)
                    histograms.setdefault(f"{stationary_role}{extra // 4}", Counter()).update(
                        {int(v): int(c) for v, c in zip(values, counts, strict=True)})
        assert paths
        timings = accuracy["component_timings"] if captures == accuracy_dir else measure_components(lib, paths, False)
        totals = {role: {mode: sum(median(run["gemm_ns"] for run in item["runs"] if run["mode"] == mode)
                                  for item in timings if item["capture"].startswith(role)) / 1e6
                         for mode in ("dense", "sparse_multiply", "sparse_unit_add")}
                  for role in ("qk", "pv")}
        assert all(item["bitwise_integer_gemm"] for item in timings)
        report["models"][name] = {
            "accuracy_report": str(accuracy_dir / "report.json"), "accuracy_report_sha256": digest(accuracy_dir / "report.json"),
            "ppl": accuracy["ppl"], "bitwise_logits_attention_token_nll_10_chunks": True,
            "full_10chunk_stream_histograms": full_histograms,
            "capture_directory": str(captures), "sample_histograms": {key: summarize(hist) for key, hist in histograms.items()},
            "captures": controls, "cpu_sum_median_ms": totals, "component_timings": timings}
        print(json.dumps({"model": name, "ppl": accuracy["ppl"], "cpu_sum_median_ms": totals}), flush=True)
    q = np.arange(256, dtype=np.int64)
    lo = q & 15
    d1 = lowdigit((q - lo) // 16)
    d2 = ((q - lo) // 16 - d1) // 16
    assert np.array_equal(q, (lo - 8) + 8 + 16 * d1 + 256 * d2)
    assert np.array_equal(d2, (q >= 128).astype(np.int64))
    report["p_exhaustive_uint8_check"] = {"codes": 256, "exact_reconstruction": True, "lane2_equals_q_ge_128": True}
    with output.open("x") as target:
        json.dump(report, target, indent=2)
    print(output, flush=True)


if __name__ == "__main__":
    main()
