#!/usr/bin/env python3
# How to run: python3 -B scripts/eval/check_e2e_collection.py --im2p ../IM2P.sim NATIVE_CHUNK_DIR
"""Validation of one PoTal collection (npu-cycle-trace.jsonl and cycle-log.jsonl of native/chunk-0), run by the full E2E
collector (e2e_run.py) after the collection and before any replay. It rejects:
  host stages     tests/check-cycle-sim-host-stages.py (streaming): missing, duplicate, undeclared or mismatched host-stage
                  declarations and measurements, canonical CPU overlap on one thread, residual host work outside its
                  canonical stages, missing or extra completion probes, rows the E2E timeline reader cannot read;
  NPU trace       IM2P.sim sim.cycle.npu_trace.validate_trace (structural, streaming): missing, duplicate or gapped
                  records, works and calls (work identity and ownership), host-stage dependencies, invalid geometry and
                  tiles, malformed stripe publication (PUBLISH order and continuation), paired facts inconsistent with
                  admission or fallback;
  lifecycle       over every phase (prefill and each decode step): one stripe work per stripe; an admitted PAIR_LOOP_TAIL
                  stripe owns no residual child work (one NPU service); a stripe that fell back (COMPACT_FALLBACK:
                  CAPACITY_NONFIT or RUN_AWARE_UNSUPPORTED) has exactly one residual child work; a compact stripe has
                  at most one; no residual child work without its stripe.
The checks run in this order and the first failure is reported."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

from eval_common import EvaluationError, Record, require

CHECKER = Path(__file__).resolve().parents[2] / "tests/check-cycle-sim-host-stages.py"
PARENT_SCOPES = ("stripe", "full")


def host_stages(native: Path) -> str:
    done = subprocess.run([sys.executable, "-B", str(CHECKER), str(native / "npu-cycle-trace.jsonl"),
                           str(native / "cycle-log.jsonl")], capture_output=True, text=True, check=False)
    lines = [line for line in done.stderr.splitlines() if line.strip()]
    require(done.returncode == 0 and done.stdout.startswith("POTAL_CANONICAL_HOST_INTERVALS_PASS"),
            "host-stage check failed: " + (lines[-1] if lines else done.stdout.strip()))
    return done.stdout.strip()


def npu_trace(native: Path, im2p: Path) -> Record:
    if str(im2p) not in sys.path:
        sys.path.insert(0, str(im2p))
    from sim.cycle.npu_trace import validate_trace  # noqa: PLC0415

    try:
        summary = validate_trace(native / "npu-cycle-trace.jsonl")
    except ValueError as error:
        raise EvaluationError(f"NPU trace validation failed: {error}") from error
    return {key: value for key, value in summary.items() if not isinstance(value, (list, dict))}


def stripe_key(record: Record) -> tuple:
    return (record.get("phase_id"), record.get("semantic_phase_kind"), record.get("semantic_decode_index"),
            record.get("operation_id"), record.get("stripe_id"))


def lifecycle(native: Path) -> dict[str, int]:
    parents: dict[tuple, Record | None] = {}
    children: Counter = Counter()
    with (native / "npu-cycle-trace.jsonl").open() as stream:
        for line in stream:
            if "NPU_WORK" not in line:
                continue
            record = json.loads(line)
            if record.get("kind") != "NPU_WORK":
                continue
            key = stripe_key(record)
            if record.get("scope") in PARENT_SCOPES:
                require(key not in parents, "paired lifecycle: more than one stripe work for one stripe")
                parents[key] = record.get("paired")
            else:
                children[key] += 1
    tally: Counter = Counter()
    for key, paired in parents.items():
        count = children.get(key, 0)
        if paired is None:
            require(count <= 1, "paired lifecycle: compact stripe with more than one residual child work")
            tally["compact stripes"] += 1
            tally["residual children of compact stripes"] += count
        elif paired.get("admitted"):
            require(count == 0, "paired lifecycle: admitted PAIR_LOOP_TAIL stripe owns a residual child work")
            tally["admitted paired stripes"] += 1
        else:
            reason = paired.get("fallback_reason")
            require(count == 1, f"paired lifecycle: {reason} fallback stripe without exactly one residual child work")
            tally[f"{reason} fallback stripes"] += 1
    require(all(key in parents for key in children), "paired lifecycle: residual child work without its stripe")
    return dict(tally)


def validate_collection(native: Path, im2p: Path) -> Record:
    return {"schema": "potal-e2e-collection-validation", "version": 1, "host_stages": host_stages(native),
            "npu_trace": npu_trace(native, im2p), "lifecycle": lifecycle(native)}


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--im2p", type=Path, required=True)
    parser.add_argument("native", type=Path, help="the collection's native/chunk-0 directory")
    args = parser.parse_args()
    try:
        result = validate_collection(args.native.resolve(strict=True), args.im2p.resolve(strict=True))
    except EvaluationError as error:
        print(f"E2E_COLLECTION_VALIDATION_FAIL {error}", file=sys.stderr)
        return 1
    print("E2E_COLLECTION_VALIDATION_PASS " + json.dumps(result, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
