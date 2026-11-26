#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# Run: uv run scripts/potal-timeline/cpu_parse.py TRACE.csv.cpu.csv OUTPUT.csv
"""Pair CPU sidecar samples; retain inclusive spans, reject incomplete/cross-hart counters."""

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NewType

Slot = NewType("Slot", int)
SpanKey = tuple[Slot, int, int, int, int, int]
FIELDS: Final = ("slot", "stage", "run", "layer", "stripe", "tile", "wall_ns", "thread_ns",
                 "process_ns", "cycles", "instructions", "ipc", "counter_status", "counter_source")


@dataclass(frozen=True, slots=True)
class TraceError(Exception):
    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class Sample:
    key: SpanKey
    seq: int
    event: int
    mono_ns: int
    thread_ns: int
    process_ns: int
    cpu: int
    cycles: int
    instructions: int
    thread_valid: bool
    process_valid: bool
    counters_valid: bool
    source: str


def load(path: Path) -> list[Sample]:
    result: list[Sample] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                flags = tuple(int(row[name]) for name in
                              ("thread_valid", "process_valid", "counters_valid"))
                if any(value not in (0, 1) for value in flags):
                    raise TraceError("validity flag must be 0 or 1")
                key = (Slot(int(row["slot"])), int(row["stage"]), int(row["run"]),
                       int(row["layer"]), int(row["stripe"]), int(row["tile"]))
                result.append(Sample(key, int(row["seq"]), int(row["event"]), int(row["mono_ns"]),
                                     int(row["thread_ns"]), int(row["process_ns"]), int(row["cpu"]),
                                     int(row["cycles"]), int(row["instructions"]), bool(flags[0]),
                                     bool(flags[1]), bool(flags[2]), row["counter_source"]))
            except (KeyError, ValueError, TypeError) as error:
                raise TraceError(f"invalid CPU sample at CSV line {len(result) + 2}: {error}") from error
    if not result:
        raise TraceError("empty CPU sidecar")
    return result


def delta(begin: Sample, end: Sample) -> tuple[int | float | str, ...]:
    if (begin.mono_ns == 0) != (end.mono_ns == 0):
        raise TraceError(f"mixed monotonic boundary availability: {begin.key}")
    if end.mono_ns < begin.mono_ns:
        raise TraceError(f"wall counter regression: {begin.key}")
    wall: int | str = end.mono_ns - begin.mono_ns if begin.mono_ns != 0 else ""
    thread: int | str = ""
    process: int | str = ""
    if begin.thread_valid and end.thread_valid:
        thread = end.thread_ns - begin.thread_ns
        if thread < 0:
            raise TraceError(f"thread counter regression: {begin.key}")
    if begin.process_valid and end.process_valid:
        process = end.process_ns - begin.process_ns
        if process < 0:
            raise TraceError(f"process counter regression: {begin.key}")
    cycles: int | str = ""
    instructions: int | str = ""
    ipc: float | str = ""
    status = "unavailable_or_unpinned"
    if begin.counters_valid and end.counters_valid:
        status = "cross_cpu_or_source"
        if begin.cpu >= 0 and begin.cpu == end.cpu and begin.source == end.source == "riscv_hart_csr":
            status = "counter_regression"
            if end.cycles >= begin.cycles and end.instructions >= begin.instructions:
                status = "valid_hart_counts"
                cycles = end.cycles - begin.cycles
                instructions = end.instructions - begin.instructions
                ipc = instructions / cycles if cycles else ""
    return (*begin.key, wall, thread, process, cycles, instructions,
            ipc, status, begin.source)


def paired(samples: list[Sample]) -> list[tuple[int | float | str, ...]]:
    """Pair by thread and identity, without converting nanoseconds into CPU cycles."""
    pending: dict[SpanKey, list[Sample]] = {}
    previous: dict[Slot, int] = {}
    output: list[tuple[int | float | str, ...]] = []
    for sample in samples:
        slot = sample.key[0]
        if sample.seq <= previous.get(slot, -1):
            raise TraceError(f"non-increasing seq: slot {slot}")
        previous[slot] = sample.seq
        stack = pending.setdefault(sample.key, [])
        if sample.event not in (1, 2):
            raise TraceError(f"invalid CPU boundary event: {sample.event}")
        if sample.event == 1:
            stack.append(sample)
        else:
            if not stack:
                raise TraceError(f"END without BEGIN: {sample.key}")
            output.append(delta(stack.pop(), sample))
    if any(pending.values()):
        raise TraceError("missing END; reject truncated/overflowed trace")
    return output


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, "\nUsage: cpu_parse.py TRACE.csv.cpu.csv OUTPUT.csv", file=sys.stderr)
        return 2
    try:
        rows = paired(load(Path(sys.argv[1])))
        with Path(sys.argv[2]).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(FIELDS)
            writer.writerows(rows)
    except (TraceError, OSError) as error:
        print(f"CPU trace rejected: {error}", file=sys.stderr)
        return 1
    print(f"Paired {len(rows)} inclusive CPU spans; validate the main timeline before aggregation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
