#!/usr/bin/env python3
# How to run: python3 -B tests/test-cycle-sim-host-stage-additivity.py
"""PoTal host-stage regression: canonical host intervals stay additive and both residual routes carry their host
work in canonical stages.

check-cycle-sim-host-stages.py (run by every cycle-sim dispatch fixture and by the smoke collector) must accept
the paired layout (non-additive im2p.residual_paired_recomposition over disjoint setup, radix recomposition,
finalize, publication and release) and the compact run-aware layout (excluded simulator envelope over request
validation, run preparation, recomposition setup, radix recomposition, finalize, publication and release, with
other canonical stages allowed between them), with one completion probe per residual completion. It must reject
nested or overlapping canonical intervals (the pre-fix paired layout), double counting, a canonical parent, a
missing or misplaced stage on either route, a duplicate publication, a missing completion probe, a missing host
timestamp and an inconsistent host clock. The IM2P.sim consumer check (reconstruct_cpu
CpuIndex.validate_potal_host_additivity), which is not modified, must still reject a nested canonical interval."""
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
PARENT, ENVELOPE = "im2p.residual_paired_recomposition", "im2p.residual_simulator_host_call"
SETUP, RADIX, FINALIZE = "im2p.residual_paired_setup", "im2p.residual_radix_recomposition", "im2p.residual_paired_finalize"
PUBLISH, RELEASE = "im2p.residual_output_publish", "im2p.residual_release"
VALIDATION, PREPARATION = "im2p.residual_request_validation", "im2p.residual_run_preparation"
WS_SETUP, WS_FINALIZE = "im2p.residual_recomposition_setup", "im2p.residual_recomposition_finalize"
_ids = iter(range(1, 10**6))


def canonical(op: str, start: int, end: int, thread: int = 7, identity: int | None = None) -> dict[str, object]:
    return {"record_type": "CYCLE_INTERVAL", "op": op, "duration_role": "POTAL_HOST",
            "host_stage_id": identity if identity is not None else next(_ids), "interval_class": "CANONICAL_ADDITIVE",
            "additive": True, "cpu_service": True, "operation_success": True, "host_execution_id": "e",
            "thread_id": thread, "host_start_ns": start, "host_end_ns": end, "host_elapsed_ns": end - start,
            "host_elapsed_valid": True}


def container(op: str, start: int, end: int, interval_class: str = "NON_ADDITIVE") -> dict[str, object]:
    return {"record_type": "CYCLE_INTERVAL", "op": op, "duration_role": "OBSERVATION_ONLY", "host_stage_id": None,
            "interval_class": interval_class, "additive": False, "cpu_service": False, "operation_success": True,
            "host_execution_id": "e", "thread_id": 7, "host_start_ns": start, "host_end_ns": end}


def probe(parent_op: str) -> dict[str, object]:
    return {"record_type": "HOST_COMPLETION_PROBE", "op": "im2p.residual_completion_probe", "parent_op": parent_op,
            "additive": False, "duration_role": "OBSERVATION_ONLY", "host_stage_id": None, "instrumentation_ns": 5,
            "instrumentation_calls": 9, "dispatch_host_ns": 0, "dispatch_host_calls": 0, "emulation_ns": 0,
            "emulation_calls": 0}


def run(cycle: list[dict[str, object]]) -> subprocess.CompletedProcess[str]:
    declared = {r["host_stage_id"]: r["op"] for r in cycle if r.get("duration_role") == "POTAL_HOST"}
    trace = [{"kind": "HOST_STAGE", "execution_class": "POTAL_HOST", "event": event, "status": "success",
              "host_stage_id": identity, "stage_name": op}
             for identity, op in declared.items() for event in ("BEGIN", "END")]
    trace.append({"kind": "RUN_END", "status": "success"})
    with tempfile.TemporaryDirectory() as directory:
        paths = [Path(directory) / name for name in ("npu-cycle-trace.jsonl", "cycle-log.jsonl")]
        for path, records in zip(paths, (trace, cycle)):
            # Compact separators, as the producer writes them (readers match fields with regular expressions).
            path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records))
        return subprocess.run([sys.executable, "-B", str(CHECKER), *map(str, paths)],
                              capture_output=True, text=True, check=False)


def expect(name: str, cycle: list[dict[str, object]], passes: bool, detail: str = "") -> None:
    result = run(cycle)
    ok = (result.returncode == 0 and "POTAL_CANONICAL_HOST_INTERVALS_PASS" in result.stdout) if passes else \
        (result.returncode != 0 and detail in result.stderr)
    if not ok:
        raise SystemExit(f"FAIL {name}: rc={result.returncode}\n{result.stdout}{result.stderr}")
    print(f"ok {name}: {(result.stdout or result.stderr).strip().splitlines()[-1]}")


PAIRED = [(SETUP, 101, 110), (RADIX, 112, 150), (FINALIZE, 152, 160), (PUBLISH, 162, 170), (RELEASE, 172, 180)]
WS = [(VALIDATION, 201, 205), (PREPARATION, 206, 240), ("im2p.output_reconstruction", 250, 252),
      (WS_SETUP, 260, 262), (RADIX, 263, 290), (WS_FINALIZE, 291, 292), (PUBLISH, 293, 294), (RELEASE, 295, 297)]


def paired(stages=PAIRED, parent: dict[str, object] | None = None, probes: int = 1) -> list[dict[str, object]]:
    return [parent or container(PARENT, 100, 190), *(canonical(*s) for s in stages), *[probe(PARENT)] * probes]


def ws(stages=WS, probes: int = 1) -> list[dict[str, object]]:
    return [container(ENVELOPE, 200, 300, "FUNCTIONAL_EMULATION"), *(canonical(*s) for s in stages),
            *[probe(ENVELOPE)] * probes]


def checker_cases() -> None:
    expect("paired: non-additive parent over five disjoint canonical stages", paired(), True)
    expect("WS: envelope over the run-aware stages, other canonical stages between them", ws(), True)
    expect("both routes in one log", paired() + ws(), True)
    expect("touching canonical intervals are disjoint",
           paired([(SETUP, 101, 112), (RADIX, 112, 150), (FINALIZE, 150, 160), (PUBLISH, 160, 170),
                   (RELEASE, 170, 180)]), True)
    expect("overlap on different threads is concurrency, not nesting",
           [canonical(RADIX, 100, 150, thread=7), canonical(PUBLISH, 120, 170, thread=8)], True)
    expect("pre-fix paired layout: canonical parent nests radix and publication",
           [canonical(PARENT, 100, 200), canonical(RADIX, 112, 150), canonical(PUBLISH, 162, 170)],
           False, "canonical PoTal host intervals overlap on one thread")
    expect("double counting: setup overlaps radix recomposition",
           paired([(SETUP, 101, 120), (RADIX, 115, 150), (FINALIZE, 152, 160), (PUBLISH, 162, 170),
                   (RELEASE, 172, 180)]), False, "canonical PoTal host intervals overlap on one thread")
    expect("paired parent marked canonical", paired(parent=canonical(PARENT, 100, 190, thread=8)),
           False, "must be a non-additive interval around canonical stages")
    expect("paired: missing release (scratch release outside the measured stages)", paired(PAIRED[:-1]),
           False, "paired recomposition host cost is not carried by its canonical stages")
    expect("paired: setup outside the parent",
           paired([(SETUP, 90, 99), *PAIRED[1:]]), False, "paired recomposition host cost is not carried")
    expect("WS: missing request validation", ws([s for s in WS if s[0] != VALIDATION]),
           False, "compact residual host work is not carried by its canonical stages")
    expect("WS: finalize after publication",
           ws([*WS[:5], (PUBLISH, 291, 292), (WS_FINALIZE, 293, 294), WS[7]]),
           False, "compact residual host work is not carried by its canonical stages")
    expect("WS: duplicate publication", ws([*WS[:7], (PUBLISH, 295, 296), (RELEASE, 297, 298)]),
           False, "compact residual host work is not carried by its canonical stages")
    expect("missing completion probe", paired(probes=0), False, "one completion probe per residual completion")
    expect("extra completion probe", ws(probes=2), False, "one completion probe per residual completion")
    spanless = paired()
    spanless[-1]["thread_id"] = 7  # a row naming a thread without a host span breaks the E2E timeline reader
    expect("probe row naming a thread without a span", spanless, False, "not readable by the E2E timeline reader")
    missing = paired()
    del missing[2]["host_start_ns"]
    expect("missing host timestamp", missing, False, "lacks a host interval")
    clock = paired()
    clock[2]["host_elapsed_ns"] = 9_000  # not host_end_ns - host_start_ns: another unit or clock
    expect("inconsistent host clock", clock, False, "inconsistent host clock")
    duplicate = paired()
    duplicate.append(canonical(RADIX, 182, 185, identity=int(str(duplicate[2]["host_stage_id"]))))
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
