#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==1.26.4"]
# ///
from __future__ import annotations

import argparse
from collections import Counter
import ctypes as ct
from dataclasses import dataclass
import json
from pathlib import Path
import platform
from time import perf_counter_ns

import numpy as np
from numpy.typing import NDArray

from .native_measure import BatchWork, CpuBatch, prepare_head
from .profile_gemm import QKTileMask, gemm
from .run import digest
from .types import I64, QuantError


@dataclass(frozen=True, slots=True)
class PreparedStripe:
    work: ct.Array[BatchWork]
    buffers: list[tuple[np.ndarray, ...]]
    expected: I64
    actual: I64
    stats: NDArray[np.uint64]


class PreparedBatch(CpuBatch):
    def __init__(self, lib: ct.CDLL) -> None:
        super().__init__(lib)
        self.stripes: list[PreparedStripe] = []

    def flush(self, acc: I64) -> None:
        for a, b, rows, shifts, regions in self.buffers:
            for r0, r1, c0, c1 in regions:
                part = a[r0:r1].astype(np.int64) @ b[:, c0:c1].astype(np.int64)
                acc[rows[r0:r1], c0:c1] += part << shifts[c0:c1]
        self.stripes.append(PreparedStripe((BatchWork * len(self.work))(*self.work), self.buffers,
                                           acc.copy(), np.empty_like(acc), np.zeros(8, np.uint64)))
        self.work, self.buffers = [], []

    def execute(self, direct: bool) -> dict[str, int]:
        started = perf_counter_ns()
        for stripe in self.stripes:
            status = self.lib.potal_replay_batch(stripe.work, len(stripe.work), *stripe.actual.shape,
                                                direct, stripe.actual, stripe.stats)
            if status:
                raise QuantError(self.lib.potal_replay_error().decode())
        elapsed = perf_counter_ns() - started
        keys = ("validate_allocate_ns", "provider_host_ns", "wide_merge_ns", "max_native_workspace_bytes",
                "raw_dot_calls", "full_partial_bytes", "region_partial_bytes", "work_items")
        return {"native_call_wall_ns": elapsed, "ffi_calls": len(self.stripes), **{
            key: int(max(s.stats[i] for s in self.stripes) if i == 3 else sum(int(s.stats[i]) for s in self.stripes))
            for i, key in enumerate(keys)}}

    def verify(self) -> None:
        for stripe in self.stripes:
            if not np.array_equal(stripe.actual, stripe.expected):
                raise QuantError("Raw PE partials plus wide shifts differ from the NumPy int64 oracle")


class CompletedBatch:
    def __init__(self, prepared: PreparedBatch) -> None:
        self.remaining = iter(stripe.actual for stripe in prepared.stripes)

    def add(self, a: I64, b: I64, rows: I64, shifts: I64,
            tiles: tuple[tuple[slice, slice], ...] | None) -> None:
        pass

    def flush(self, acc: I64) -> None:
        acc[:] = next(self.remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real Gemmini QK/PV replay; vanilla without masks by default")
    parser.add_argument("library", type=Path)
    parser.add_argument("output", type=Path, help="New JSONL report; completed runs survive a later failure")
    parser.add_argument("captures", type=Path, nargs="+")
    parser.add_argument("--experiments", action="store_true", help="Also run direct accumulation and QK mask skipping")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    for path in args.captures:
        if not path.is_file() or not path.name.startswith(("qk-", "pv-")):
            parser.error(f"Expected an existing qk-*.npz or pv-*.npz capture: {path}")
    lib = ct.CDLL(str(args.library.resolve()))
    lib.potal_replay_error.restype = ct.c_char_p
    lib.potal_im2p_reset.argtypes = []
    lib.potal_im2p_reset.restype = ct.c_int
    lib.potal_im2p_stats.argtypes = []
    lib.potal_im2p_stats.restype = ct.c_char_p
    if lib.potal_im2p_reset():
        raise QuantError(lib.potal_replay_error().decode())
    sources = ("im2p_measure.py", "im2p_raw_probe.cpp", "raw_batch.hpp", "native_measure.py",
               "profile_gemm.py", "profile_ops.py", "stream.py", "packet.py", "attention.py", "types.py")
    with args.output.open("x") as report:
        header = {"event": "start", "scope": "Captured QK/PV raw Gemmini dots with exact host int64 merge",
                  "library": str(args.library.resolve()), "library_sha256": digest(args.library),
                  "runtime": json.loads(lib.potal_im2p_stats()), "platform": platform.platform(),
                  "numpy": np.__version__, "experiments": args.experiments, "repeats": args.repeats,
                  "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in sources},
                  "timing": "RTL cycles sum real raw requests, including runtime loads/stores; host times are simulator wall time",
                  "preparation": "Quantization, mask planning, int8 packing and independent oracles run before execution",
                  "limitations": ["No whole-model PPL or latency claim; no production attention dispatch",
                                  "No new cross-request NPU weight reuse; runtime read/preload counters are reported",
                                  "Existing profiling schedule submits per-BK/per-lane raw requests; product stacked RMD is unchanged",
                                  "Host wide merge, PV zero point and float publication are not RTL cycles",
                                  "Request counters are not transferred byte counts"]}
        report.write(json.dumps(header) + "\n")
        report.flush()
        completed = 0
        for path in args.captures:
            started = perf_counter_ns()
            capture_hash = digest(path)
            role, zp = ("qk", 0) if path.name.startswith("qk-") else ("pv", 8)
            quant_times: dict[str, int] = {}
            with np.load(path, allow_pickle=False) as saved:
                operand, plan, weights = prepare_head(0 if role == "qk" else 1, saved["streamed"], saved["stationary"], quant_times)
                reference = gemm(plan, weights, role, {}, zp)
                reconstructed = operand.values.astype(np.float64) @ weights.values.astype(np.float64)
                if not np.array_equal(reference, reconstructed) or not np.array_equal(reference.astype(np.float32), saved["result"]):
                    raise QuantError(f"Original-policy reconstruction or saved output differs: {path}")
                mask = QKTileMask.from_values(saved["mask"]) if args.experiments and role == "qk" else None
            dense = PreparedBatch(lib)
            fragments: Counter[str] = Counter()
            expected = gemm(plan, weights, role, {}, zp, fragments=fragments, batch=dense)
            if not np.array_equal(expected, reference):
                raise QuantError("Prepared native schedule differs from the original integer GEMM")
            variants = [("vanilla", dense, False, None, fragments)]
            if args.experiments:
                variants.append(("direct", dense, True, None, fragments))
                if mask is not None:
                    masked = PreparedBatch(lib)
                    masked_fragments: Counter[str] = Counter()
                    expected = gemm(plan, weights, role, {}, zp, qk_mask=mask, fragments=masked_fragments, batch=masked)
                    if not np.array_equal(expected[~mask.blocked], reference[~mask.blocked]) or np.any(expected[mask.blocked]):
                        raise QuantError("Masked schedule changed valid QK values")
                    variants.append(("qk-mask", masked, True, mask, masked_fragments))
            preparation_ns = perf_counter_ns() - started
            for repeat in range(args.repeats):
                order = variants[repeat % len(variants):] + variants[:repeat % len(variants)]
                for name, batch, direct, selected_mask, counts in order:
                    print(f"Gemmini {path.name}: {name}, repeat {repeat + 1}/{args.repeats}", flush=True)
                    if lib.potal_im2p_reset():
                        raise QuantError(lib.potal_replay_error().decode())
                    host = batch.execute(direct)
                    raw_metrics = lib.potal_im2p_stats()
                    if raw_metrics is None:
                        raise QuantError(lib.potal_replay_error().decode())
                    hardware = json.loads(raw_metrics)
                    if (hardware["raw_dot_calls"] != host["raw_dot_calls"] or
                            (hardware["raw_dot_calls"] > 0 and hardware["work_total_cycles"] <= 0)):
                        raise QuantError("Missing Gemmini execution or RTL cycles")
                    started = perf_counter_ns()
                    batch.verify()
                    publication_times: dict[str, int] = {}
                    result = gemm(plan, weights, role, publication_times, zp, qk_mask=selected_mask, batch=CompletedBatch(batch))
                    live = ~selected_mask.blocked if selected_mask is not None else np.ones(result.shape, bool)
                    if not np.array_equal(result[live], reference[live]) or (selected_mask is not None and np.any(result[selected_mask.blocked])):
                        raise QuantError("Published QK/PV differs from the original-policy reference")
                    record = {"event": "result", "capture": str(path.resolve()), "capture_sha256": capture_hash,
                              "role": role, "shape": list(reference.shape), "mode": name, "repeat": repeat + 1,
                              "reduction": plan[0].main.shape[1], "zero_point": zp,
                              "stationary_pass_shifts": [shift for _, shift in weights.passes],
                              "stream_lanes": sorted({lane for stripe in plan for lane, _, _ in stripe.lanes}),
                              "zero_point_add_ops": (weights.values.size + reference.size) if zp else 0,
                              "integer_accumulators_equal": True, "published_equal": True,
                              "equality_domain": "valid QK entries; masked entries zero" if selected_mask is not None else "all entries",
                              "gemmini": hardware, "host": host, "fragments": dict(counts),
                              "preparation_and_oracle_ns": preparation_ns, "quantization_ns": quant_times,
                              "verification_and_publication_ns": perf_counter_ns() - started,
                              "publication_stages_ns": publication_times}
                    report.write(json.dumps(record) + "\n")
                    report.flush()
                    completed += 1
                    print(json.dumps({"mode": name, "rtl_cycles": hardware["work_total_cycles"],
                                      "raw_calls": hardware["raw_dot_calls"], "integer_equal": True}), flush=True)
        report.write(json.dumps({"event": "complete", "runs": completed, "captures": len(args.captures)}) + "\n")


if __name__ == "__main__":
    main()
