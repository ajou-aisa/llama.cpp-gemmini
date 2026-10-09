#!/usr/bin/env python3
# How to run: python3 -B tests/test-cycle-sim-host-stage-additivity.py
"""P6.3 S5.3 producer regression: canonical PoTal host intervals stay additive.

check-cycle-sim-host-stages.py (the checker every cycle-sim dispatch fixture runs) must accept the paired layout
(non-additive im2p.residual_paired_recomposition parent over disjoint canonical setup, radix recomposition,
finalize and publication stages) and the WS compact layout, and reject nested or overlapping canonical intervals
(the pre-fix paired layout), a canonical paired parent, a paired parent whose cost lacks a canonical stage, and
double measurement. The IM2P.sim consumer check (reconstruct_cpu CpuIndex.validate_potal_host_additivity), which
this change does not touch, must still reject a nested canonical interval and accept disjoint ones."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "tests/check-cycle-sim-host-stages.py"
IM2P = Path(os.environ.get("IM2P_SIM_ROOT", ROOT.parent / "IM2P.sim"))
PARENT = "im2p.residual_paired_recomposition"
SETUP, RADIX = "im2p.residual_paired_setup", "im2p.residual_radix_recomposition"
FINALIZE, PUBLISH = "im2p.residual_paired_finalize", "im2p.residual_output_publish"


def canonical(identity: int, op: str, start: int, end: int, thread: int = 7) -> dict[str, object]:
    return {"record_type": "CYCLE_INTERVAL", "op": op, "duration_role": "POTAL_HOST", "host_stage_id": identity,
            "interval_class": "CANONICAL_ADDITIVE", "additive": True, "cpu_service": True,
            "operation_success": True, "host_execution_id": "e", "thread_id": thread,
            "host_start_ns": start, "host_end_ns": end}


def observation(op: str, start: int, end: int, interval_class: str = "NON_ADDITIVE") -> dict[str, object]:
    return {"record_type": "CYCLE_INTERVAL", "op": op, "duration_role": "OBSERVATION_ONLY", "host_stage_id": None,
            "interval_class": interval_class, "additive": False, "cpu_service": False, "operation_success": True,
            "host_execution_id": "e", "thread_id": 7, "host_start_ns": start, "host_end_ns": end}


def run(cycle: list[dict[str, object]]) -> subprocess.CompletedProcess[str]:
    declared = {r["host_stage_id"]: r["op"] for r in cycle if r["duration_role"] == "POTAL_HOST"}
    trace = [{"kind": "HOST_STAGE", "execution_class": "POTAL_HOST", "event": event, "status": "success",
              "host_stage_id": identity, "stage_name": op}
             for identity, op in declared.items() for event in ("BEGIN", "END")]
    trace.append({"kind": "RUN_END", "status": "success"})
    with tempfile.TemporaryDirectory() as directory:
        paths = [Path(directory) / name for name in ("npu-cycle-trace.jsonl", "cycle-log.jsonl")]
        for path, records in zip(paths, (trace, cycle)):
            path.write_text("".join(json.dumps(r) + "\n" for r in records))
        return subprocess.run([sys.executable, "-B", str(CHECKER), *map(str, paths)],
                              capture_output=True, text=True, check=False)


def expect(name: str, cycle: list[dict[str, object]], passes: bool, detail: str = "") -> None:
    result = run(cycle)
    ok = (result.returncode == 0 and "POTAL_CANONICAL_HOST_INTERVALS_PASS" in result.stdout) if passes else \
        (result.returncode != 0 and detail in result.stderr)
    if not ok:
        raise SystemExit(f"FAIL {name}: rc={result.returncode}\n{result.stdout}{result.stderr}")
    print(f"ok {name}: {(result.stdout or result.stderr).strip().splitlines()[-1]}")


def paired(stages: list[tuple[int, str, int, int]], parent: dict[str, object] | None = None) -> list[dict[str, object]]:
    return [parent or observation(PARENT, 100, 200), *(canonical(*stage) for stage in stages)]


FIXED = [(1, SETUP, 101, 110), (2, RADIX, 112, 150), (3, FINALIZE, 152, 160), (4, PUBLISH, 162, 170)]


def checker_cases() -> None:
    expect("paired layout: non-additive parent over four disjoint canonical stages", paired(FIXED), True)
    expect("touching canonical intervals are disjoint",
           paired([(1, SETUP, 101, 112), (2, RADIX, 112, 150), (3, FINALIZE, 150, 160), (4, PUBLISH, 160, 170)]),
           True)
    expect("WS layout: wait envelope observation over canonical recomposition and publication",
           [canonical(1, "im2p.residual_run_preparation", 80, 95),
            observation("im2p.residual_simulator_host_call", 96, 200, "FUNCTIONAL_EMULATION"),
            canonical(2, RADIX, 150, 160), canonical(3, PUBLISH, 161, 170)], True)
    expect("overlap on different threads is not a nesting",
           [canonical(1, RADIX, 100, 150, thread=7), canonical(2, PUBLISH, 120, 170, thread=8)], True)
    expect("pre-fix paired layout: canonical parent nests radix and publication",
           [canonical(9, PARENT, 100, 200), canonical(2, RADIX, 112, 150), canonical(4, PUBLISH, 162, 170)],
           False, "canonical PoTal host intervals overlap on one thread")
    expect("double counting: setup overlaps radix recomposition",
           paired([(1, SETUP, 101, 120), (2, RADIX, 115, 150), (3, FINALIZE, 152, 160), (4, PUBLISH, 162, 170)]),
           False, "canonical PoTal host intervals overlap on one thread")
    expect("paired parent marked canonical", paired(FIXED, canonical(9, PARENT, 100, 200, thread=8)),
           False, "paired recomposition must be a non-additive diagnostic parent")
    expect("host cost omission: no finalize stage inside the parent",
           paired([(1, SETUP, 101, 110), (2, RADIX, 112, 150), (4, PUBLISH, 162, 170)]),
           False, "paired recomposition host cost is not carried by its canonical stages")
    expect("host cost omission: setup outside the parent",
           paired([(1, SETUP, 90, 99), (2, RADIX, 112, 150), (3, FINALIZE, 152, 160), (4, PUBLISH, 162, 170)]),
           False, "paired recomposition host cost is not carried by its canonical stages")
    duplicate = paired(FIXED)
    duplicate.append(canonical(2, RADIX, 180, 190))
    expect("duplicate canonical measurement of one stage", duplicate, False, "duplicate canonical PoTal host measurement")


def consumer_cases() -> None:
    sys.path.insert(0, str(IM2P))
    from sim.cycle.reconstruct_cpu import CpuIndex  # noqa: E402

    def validate(rows: list[tuple[str, int, int, int]]) -> str:
        index = CpuIndex(sqlite3.connect(":memory:"), None)  # type: ignore[arg-type]
        index.db.executemany("INSERT INTO canonical_host_intervals VALUES(?,?,?,?)", rows)
        try:
            index.validate_potal_host_additivity()
        except ValueError as error:
            return str(error)
        return "PASS"

    nested = validate([("e", 7, 100, 200), ("e", 7, 112, 150), ("e", 7, 162, 170)])
    disjoint = validate([("e", 7, 101, 112), ("e", 7, 112, 150), ("e", 7, 150, 160), ("e", 8, 100, 170)])
    # The message the S5.3 WS_PAIRED join stopped with.
    if nested != "npu-trace: positive canonical host interval overlap" or disjoint != "PASS":
        raise SystemExit(f"FAIL consumer additivity: nested={nested!r} disjoint={disjoint!r}")
    print("ok consumer still rejects a nested canonical interval and accepts disjoint ones")


if __name__ == "__main__":
    checker_cases()
    consumer_cases()
    print("CYCLE_SIM_HOST_STAGE_ADDITIVITY_PASS")
