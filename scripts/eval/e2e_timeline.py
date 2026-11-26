"""Host worker lanes and exact interval arithmetic for native evaluation logs."""
from __future__ import annotations

import json
import re
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


def host_cores(measurement: Record) -> Record:
    """Observed Linux cores at the interval endpoints; host_cpu_core is set only when the interval did not migrate."""
    start, end = measurement.get("host_cpu_core_start"), measurement.get("host_cpu_core_end")
    return {"host_cpu_core": start if isinstance(start, int) and start == end else None,
            "host_cpu_core_start": start, "host_cpu_core_end": end, "cpu_migrated": measurement.get("cpu_migrated")}


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
