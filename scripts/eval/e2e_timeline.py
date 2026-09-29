"""Graph-facing E2E CPU/NPU timeline built on the official execution IR and scheduler.

Raw observations (cycle logs, NPU traces, collection provenance) are never rewritten. This module derives
  * the explicit CPU worker scenario for the official producer-lifecycle stage: host threads share a lane only
    when their observed lifetimes are disjoint, so observed concurrency is kept and the scheduler's bounded
    resource-group limit holds;
  * an isolated-service phase table from the certified isolated replay (SYNTHETIC_ONLY by construction);
  * timeline.jsonl from the official SQLite schedule: one row per CPU worker interval and NPU work, placed where
    the scheduler put them, so CPU/NPU overlap, idle gaps and order are preserved, never summed or packed;
  * TTFT/TPOT from those rows (the single implementation) with explicit timing ownership.
Row field names follow scripts/utils/cycle_timeline.py (kind, op, phase, tid, worker_id, cycle_*, event,
token_index). A reconstructed schedule is a model under its declared clock, never an observation.
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from fractions import Fraction
from pathlib import Path
from statistics import median
from typing import Final

from eval_common import Json, Record, integer, record, records, require, sha256, text

LANE_POLICY: Final = "GREEDY_DISJOINT_OBSERVED_LIFETIME"
THREAD = re.compile(rb'"thread_id":(\d+)')
EXECUTION = re.compile(rb'"host_execution_id":"([^"]+)"')
SPAN = re.compile(rb'"host_start_ns":(\d+),"host_end_ns":(\d+)')
WORKER = re.compile(rb'"worker_id":(\d+)')
STRUCTURAL: Final = frozenset({"OP_ENTER", "OP_EXIT", "BARRIER", "PUBLISH", "FUNCTIONAL_EMULATION", "WAIT", "EXCLUDED"})


def reference(path: Path) -> Record:
    return {"path": str(path), "sha256": sha256(path)}


def thread_lifetimes(log: Path) -> dict[tuple[str, int], tuple[int, int]]:
    """First start and last end of every (host execution, thread) in a cycle log."""
    spans: dict[tuple[str, int], tuple[int, int]] = {}
    with log.open("rb") as stream:
        for line in stream:
            thread = THREAD.search(line)
            if thread is None:
                continue
            execution, span = EXECUTION.search(line), SPAN.search(line)
            require(execution is not None and span is not None, "threaded cycle-log row lacks host execution/span")
            assert execution is not None and span is not None
            key = (execution.group(1).decode(), int(thread.group(1)))
            start, end = int(span.group(1)), int(span.group(2))
            require(end >= start, "cycle-log host span regressed")
            first, last = spans.get(key, (start, end))
            spans[key] = (min(first, start), max(last, end))
    return spans


def pack_lanes(spans: dict[tuple[str, int], tuple[int, int]]) -> dict[tuple[str, int], int]:
    """Greedy interval partitioning: a lane is reused only after its previous thread's last observed end."""
    lanes: dict[tuple[str, int], int] = {}
    ends: list[int] = []
    for key, (first, last) in sorted(spans.items(), key=lambda item: (item[1][0], item[0])):
        lane = next((index for index, end in enumerate(ends) if end < first), len(ends))
        if lane == len(ends):
            ends.append(last)
        else:
            ends[lane] = last
        lanes[key] = lane
    return lanes


def worker_scenario(potal: Path, fullcpu: Path, cpu_policy: str) -> tuple[dict[str, str], Record]:
    """Worker keys -> scheduler resources, enumerated from collected logs only (no hand-written identities)."""
    potal_log, fullcpu_log = potal / "native/chunk-0/cycle-log.jsonl", fullcpu / "native/chunk-0/cycle-log.jsonl"
    application = potal / "native/application-cpu.jsonl"
    mapping: dict[str, str] = {}
    workers: set[int] = set()
    with fullcpu_log.open("rb") as stream:
        for line in stream:
            if b'"cpu_service":true' in line and (worker := WORKER.search(line)) is not None:
                workers.add(int(worker.group(1)))
    for worker in sorted(workers):
        mapping[f"FULL_CPU:{worker}"] = f"cpu:fullcpu-worker-{worker}"
    spans = thread_lifetimes(potal_log)
    lanes = pack_lanes(spans)
    for (execution, thread), lane in sorted(lanes.items()):
        mapping[json.dumps(["POTAL_COLLECTION", "host_thread", execution, thread], separators=(",", ":"))] = \
            f"cpu:potal-lane-{lane}"
    samplers = sorted({(text(row, "host_execution_id"), integer(row, "thread_id")) for row in records(application)})
    require(len(samplers) == 1 and samplers[0] in lanes, "exactly one logged sampler thread required")
    scenario: Record = {"schema": "potal-e2e-cpu-worker-scenario", "version": 1, "provenance": "MEASURED",
        "derivation": "worker keys enumerated from collected cycle logs and application-cpu endpoints; PoTal host "
                      "threads share a scheduler lane only when their observed lifetimes are disjoint",
        "lane_policy": LANE_POLICY, "cpu_policy": cpu_policy, "fullcpu_workers": len(workers),
        "potal_host_threads": len(spans), "potal_lanes": len(set(lanes.values())),
        "sampler_thread_id": samplers[0][1], "sampler_resource": f"cpu:potal-lane-{lanes[samplers[0]]}",
        "sources": {"fullcpu_cycle_log": reference(fullcpu_log), "potal_cycle_log": reference(potal_log),
                    "potal_application_cpu": reference(application)}}
    return dict(sorted(mapping.items())), scenario


def isolated_phase_table(results: Path) -> Record:
    """SYNTHETIC_ONLY service table: each work's certified isolated cycles, independent of acceptance phase."""
    samples: dict[str, Record] = {}
    for row in records(results):
        cycles = integer(record(row.get("modeled")), "total_cycles", 1)
        binding = text(row, "run_view_sha256")
        sample: Record = {"profile": text(row, "profile"), "request_sha256": binding, "phase": 0,
                          "result_ready_cycles": cycles, "resource_ready_cycles": cycles,
                          "evidence_id": "ISOLATED_REPLAY:" + text(row, "cycle_library_sha256") + ":" + binding}
        require(samples.setdefault(binding, sample) == sample, "one isolated binding with two service values")
    require(len({text(sample, "profile") for sample in samples.values()}) == 1, "one hardware profile required")
    table: list[Json] = list(samples.values())
    return {"schema": "im2p-service-phase-table", "version": 1, "scope": "SYNTHETIC_ONLY", "period": 1,
            "samples": table}


def phase_of(value: str) -> tuple[str, int | None]:
    if value.startswith("{"):
        parsed = record(json.loads(value))
        index = parsed.get("decode_index")
        return text(parsed, "kind"), index if isinstance(index, int) else None
    kind, _, index = value.partition(":")
    return kind, None if index in ("", "None") else int(index)


def token_of(kind: str, decode_index: int | None) -> int:
    require(kind in ("prefill", "decode") and (decode_index is None) == (kind == "prefill"), "unknown phase " + kind)
    return 0 if decode_index is None else decode_index + 1


def rational(value: Json) -> Fraction:
    row = record(value)
    return Fraction(integer(row, "numerator"), integer(row, "denominator", 1))


def number(value: Fraction) -> int | float:
    return value.numerator if value.denominator == 1 else float(value)


class Axis:
    """Schedule nanoseconds -> NPU-clock cycles; wall time only under a validated operating clock."""

    def __init__(self, frequency_hz: int, validated: bool) -> None:
        self.frequency_hz = frequency_hz
        self.validated = validated

    def cycles(self, ns: Fraction) -> Fraction:
        return ns * self.frequency_hz / 1_000_000_000

    def place(self, row: Record, start_ns: Fraction, end_ns: Fraction) -> None:
        start, end = self.cycles(start_ns), self.cycles(end_ns)
        row.update(start_cycle=number(start), end_cycle=number(end), duration_cycles=number(end - start),
                   start_ns=number(start_ns) if self.validated else None,
                   end_ns=number(end_ns) if self.validated else None,
                   duration_ns=number(end_ns - start_ns) if self.validated else None)


def cpu_source(kind: str, resource: str) -> str:
    return "application" if kind == "APPLICATION_CPU" else "fullcpu" if resource.startswith("cpu:fullcpu") else "potal"


def timeline_rows(schedule: Path, bundle: Path, npu_results: Path, axis: Axis) -> Iterator[Record]:
    """Every scheduled CPU worker interval, NPU work and application endpoint, in schedule order."""
    works = {integer(row, "sequence"): row for row in records(npu_results)}
    with closing(sqlite3.connect(schedule.resolve().as_uri() + "?mode=ro", uri=True)) as database:
        database.execute("ATTACH DATABASE ? AS ir", (bundle.resolve().as_uri() + "?mode=ro",))
        query = ("SELECT r.ordinal, r.identity, r.body, n.kind, n.body FROM results r "
                 "JOIN ir.nodes n ON n.identity=r.identity ORDER BY r.ordinal")
        for ordinal, identity, body, kind, node_body in database.execute(query):
            scheduled, node = record(json.loads(body)), record(json.loads(node_body))
            require(scheduled.get("node_id") == identity == node.get("node_id"), "schedule/IR node identity mismatch")
            accepted, result, resource = (rational(scheduled.get(name)) for name in
                                          ("accepted_ns", "result_ready_ns", "resource_ready_ns"))
            if identity == "application:request:begin":
                event: Record = {"seq": ordinal, "row_type": "event", "kind": "event", "event": "request_start",
                                 "node_id": identity, "token_index": None}
                axis.place(event, result, result)
                yield event
            if kind in STRUCTURAL:
                continue
            phase, decode_index = phase_of(text(node, "phase"))
            token = (int(identity.rsplit(":", 1)[1]) if identity.startswith("application:sample:")
                     else token_of(phase, decode_index))
            common: Record = {"seq": ordinal, "row_type": "interval", "node_id": identity,
                              "operation_id": node.get("operation_id"), "phase": phase,
                              "decode_index": decode_index, "token_index": token}
            if kind == "NPU":
                work = works[int(identity.removeprefix("npu:"))]
                cycles = integer(record(work.get("modeled")), "total_cycles", 1)
                accepted_cycle = scheduled.get("accepted_cycle")
                require(isinstance(accepted_cycle, int) and axis.cycles(accepted) == accepted_cycle and
                        axis.cycles(resource) - accepted_cycle == cycles, "NPU row differs from its isolated service")
                row: Record = {**common, "kind": "npu", "source": "cycle-sim", "resource": "npu:0",
                               "work_id": work.get("work_id"), "op": work.get("operation"),
                               "layer": work.get("layer"), "evidence_id": scheduled.get("evidence_id"),
                               "npu_cycles": cycles, "npu_ms": None,
                               "result_ready_cycle": number(axis.cycles(result)),
                               "timing_source": "CYCLE_SIM_ISOLATED_CERTIFIED"}
                axis.place(row, accepted, resource)
                yield row
                continue
            require(kind in ("CPU", "APPLICATION_CPU"), "unsupported scheduled node kind " + str(kind))
            workers = scheduled.get("worker_intervals")
            require(isinstance(workers, list) and bool(workers), "CPU node without worker intervals")
            for worker in workers if isinstance(workers, list) else []:
                sample = record(worker)
                measurement = record(sample.get("measurement"))
                cpu: Record = {**common, "kind": "cpu", "source": cpu_source(kind, text(sample, "resource")),
                               "resource": sample.get("resource"), "tid": measurement.get("thread_id"),
                               "worker_id": sample.get("worker_id"),
                               "op": measurement.get("stage") or measurement.get("op"),
                               "layer": measurement.get("layer"), "source_line": measurement.get("source_line"),
                               "host_elapsed_ns": measurement.get("host_elapsed_ns"),
                               "host_thread_cpu_ns": measurement.get("thread_cpu_ns")
                               if measurement.get("thread_cpu_valid") is True else None,
                               "target_cpu_cycles": None, "target_cpu_ms": None,
                               "timing_source": "HOST_MEASURED_DEVELOPMENT"}
                axis.place(cpu, accepted, accepted + rational(sample.get("duration_ns")))
                yield cpu
            if identity.startswith("application:sample:"):
                ready: Record = {"seq": ordinal, "row_type": "event", "kind": "event", "event": "token_ready",
                                 "node_id": identity, "token_index": token}
                axis.place(ready, result, result)
                yield ready


def write_timeline(path: Path, rows: Iterator[Record]) -> int:
    count = 0
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            count += 1
    return count


def merged(intervals: list[tuple[Fraction, Fraction]]) -> list[tuple[Fraction, Fraction]]:
    union: list[tuple[Fraction, Fraction]] = []
    for start, end in sorted(intervals):
        if union and start <= union[-1][1]:
            union[-1] = (union[-1][0], max(union[-1][1], end))
        else:
            union.append((start, end))
    return union


def overlap(left: list[tuple[Fraction, Fraction]], right: list[tuple[Fraction, Fraction]]) -> Fraction:
    total, index = Fraction(0), 0
    for start, end in left:
        while index < len(right) and right[index][1] <= start:
            index += 1
        probe = index
        while probe < len(right) and right[probe][0] < end:
            total += min(end, right[probe][1]) - max(start, right[probe][0])
            probe += 1
    return total


def stats(values: list[Fraction]) -> Record:
    return {"count": len(values), "mean": number(sum(values, Fraction(0)) / len(values)) if values else None,
            "median": number(Fraction(median(values))) if values else None,
            "min": number(min(values)) if values else None, "max": number(max(values)) if values else None}


def ms(cycles: Fraction, frequency_hz: int | None) -> float | None:
    return None if frequency_hz is None else float(cycles * 1000 / frequency_hz)


def performance(timeline: Path, generated: int, clock_hz: int | None) -> tuple[Record, Record]:
    """TTFT/TPOT and timeline checks from timeline.jsonl alone (the single implementation).

    clock_hz is a validated operating clock or None; the schedule's diagnostic clock never produces ms.
    """
    start: Fraction | None = None
    ready: dict[int, Fraction] = {}
    npu: dict[int, Fraction] = {}
    host: dict[int, int] = {}
    thread_cpu: dict[int, int] = {}
    npu_spans: list[tuple[Fraction, Fraction]] = []
    cpu_spans: list[tuple[Fraction, Fraction]] = []
    windows: dict[int, tuple[Fraction, Fraction]] = {}
    order_violations = 0
    for row in records(timeline):
        begin, end = Fraction(row["start_cycle"]), Fraction(row["end_cycle"])  # type: ignore[arg-type]
        if row["kind"] == "event":
            if row["event"] == "request_start":
                require(start is None, "duplicate request start")
                start = begin
            else:
                ready[integer(row, "token_index")] = begin
            continue
        token = integer(row, "token_index")
        if row["kind"] == "npu":
            if npu_spans and begin < npu_spans[-1][1]:
                order_violations += 1
            npu_spans.append((begin, end))
            npu[token] = npu.get(token, Fraction(0)) + integer(row, "npu_cycles")
            low, high = windows.get(token, (begin, end))
            windows[token] = (min(low, begin), max(high, end))
        else:
            cpu_spans.append((begin, end))
            host[token] = host.get(token, 0) + integer(row, "host_elapsed_ns")
            if row.get("host_thread_cpu_ns") is not None:
                thread_cpu[token] = thread_cpu.get(token, 0) + integer(row, "host_thread_cpu_ns")
    require(start is not None and sorted(ready) == list(range(generated)), "timeline lacks request/token endpoints")
    assert start is not None
    boundary: list[Json] = [token for token, (low, high) in windows.items()
                if high > ready[token] or (token > 0 and low < ready[token - 1]) or low < start]
    per_token_npu = [npu.get(token, Fraction(0)) for token in range(1, generated)]
    intervals = [ready[token] - ready[token - 1] for token in range(1, generated)]
    busy = merged(cpu_spans)
    span = ready[generated - 1] - start
    npu_busy = sum((end - begin for begin, end in npu_spans), Fraction(0))
    checks: Record = {
        "npu_rows": len(npu_spans), "cpu_rows": len(cpu_spans),
        "npu_order_or_overlap_violations": order_violations,
        "token_boundary_violations": boundary,
        "cpu_npu_overlap_cycles": number(overlap(merged(npu_spans), busy)),
        "npu_busy_cycles": number(npu_busy), "npu_idle_cycles_in_request": number(span - npu_busy),
        "cpu_busy_union_cycles": number(sum((end - begin for begin, end in busy), Fraction(0))),
        "request_span_cycles": number(span),
        "status": "PASS" if order_violations == 0 and not boundary else "FAIL"}
    ttft_npu = npu.get(0, Fraction(0))
    tpot_npu = stats(per_token_npu)
    unready = {"cpu": "NOT_READY: TARGET_CPU_TIMING (development-host ns only)",
               "interface": "UNMODELED: command/DMA/synchronization/transport",
               "e2e": "NOT_READY: target CPU timing, interface cost and operating clock required"}
    npu_ms_state = "READY: validated operating clock" if clock_hz else "NOT_READY: OPERATING_CLOCK"
    result: Record = {
        "ttft": {"endpoints": "application:request:begin result_ready -> application:sample:0 result_ready",
                 "cpu_cycles": None, "npu_cycles": number(ttft_npu), "interface_cycles": None, "e2e_cycles": None,
                 "cpu_ms": None, "npu_ms": ms(ttft_npu, clock_hz), "interface_ms": None, "e2e_ms": None,
                 "interface_status": "UNMODELED",
                 "readiness": {"npu_cycles": "READY: certified isolated cycle simulator", "npu_ms": npu_ms_state,
                               "cpu_cycles": unready["cpu"], "interface_cycles": unready["interface"],
                               "e2e_cycles": unready["e2e"]},
                 "host": {"cpu_host_elapsed_ns_sum": host.get(0, 0), "cpu_host_thread_cpu_ns_sum": thread_cpu.get(0),
                          "label": "HOST_MEASURED development host; summed CPU work, not wall time or target time"}},
        "tpot": {"intervals": generated - 1, "endpoints": "application:sample:k-1 -> application:sample:k result_ready",
                 "npu_tpot_cycles": tpot_npu["mean"], "npu_tpot_cycles_stats": tpot_npu,
                 "npu_tpot_ms": ms(sum(per_token_npu, Fraction(0)) / len(per_token_npu), clock_hz)
                 if per_token_npu else None,
                 "per_token_npu_cycles": [number(value) for value in per_token_npu],
                 "e2e_tpot_cycles": None, "e2e_tpot_ms": None, "cpu_tpot_cycles": None, "interface_tpot_cycles": None,
                 "interface_status": "UNMODELED",
                 "paper_aggregate": "mean of the 127 per-token intervals ((t127 - t0) / 127 for E2E); NPU-only and "
                                    "E2E values are separate fields",
                 "readiness": {"npu_tpot_cycles": "READY: certified isolated cycle simulator",
                               "npu_tpot_ms": npu_ms_state, "e2e_tpot_cycles": unready["e2e"]},
                 "host": {"per_token_cpu_host_elapsed_ns_sum": [host.get(token, 0) for token in range(1, generated)],
                          "label": "HOST_MEASURED development host; summed CPU work per decode step"}},
        "diagnostic_schedule": {
            "scope": "SYNTHETIC_ONLY", "ttft_axis_cycles": number(ready[0] - start),
            "tpot_axis_cycles_stats": stats(intervals),
            "per_token_axis_cycles": [number(value) for value in intervals],
            "label": "schedule axis under the declared clock with development-host CPU durations and no interface "
                     "cost; for timeline inspection only, never target latency"}}
    return result, checks
