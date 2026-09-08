#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
# How to run: python3 scripts/utils/summarize_host_overlap.py --help
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, TypedDict, Union

JSONValue = Union[None, bool, int, float, str, List["JSONValue"], Dict[str, "JSONValue"]]
Interval = Tuple[int, int]


class GroupKey(NamedTuple):
    execution_id: str
    run_id: int
    layer: str


class Operations(NamedTuple):
    left: frozenset[str]
    right: frozenset[str]


GroupedIntervals = Dict[GroupKey, Tuple[List[Interval], List[Interval]]]


class Skipped(TypedDict):
    unselected_op: int
    missing_host_timing: int
    invalid_host_timing: int
    unknown_identity: int


class SourceSummary(TypedDict):
    path: str
    records: int
    left_matches: int
    right_matches: int
    left_accepted: int
    right_accepted: int
    skipped: Skipped


class GroupSummary(TypedDict):
    execution_id: str
    run_id: int
    layer: str
    left_intervals: int
    right_intervals: int
    left_union_ns: int
    right_union_ns: int
    overlap_ns: Optional[int]
    status: str


class InputError(Exception):
    def __init__(self, location: str, detail: str) -> None:
        self.location = location
        self.detail = detail
        super().__init__(f"{location}: {detail}")


def read_source(path: Path, operations: Operations, groups: GroupedIntervals) -> SourceSummary:
    summary: SourceSummary = {
        "path": str(path), "records": 0, "left_matches": 0, "right_matches": 0,
        "left_accepted": 0, "right_accepted": 0,
        "skipped": {"unselected_op": 0, "missing_host_timing": 0,
                    "invalid_host_timing": 0, "unknown_identity": 0},
    }
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            location = f"{path}:{line_number}"
            try:
                record: JSONValue = json.loads(line)
            except json.JSONDecodeError as error:
                raise InputError(location, error.msg) from None
            if not isinstance(record, dict):
                raise InputError(location, "expected a JSON object")
            summary["records"] += 1
            op = record.get("op")
            left = isinstance(op, str) and op in operations.left
            right = isinstance(op, str) and op in operations.right
            summary["left_matches"] += int(left)
            summary["right_matches"] += int(right)
            if not (left or right):
                summary["skipped"]["unselected_op"] += 1
                continue
            timing = record.get("host_timing")
            if timing is None:
                summary["skipped"]["missing_host_timing"] += 1
                continue
            if not isinstance(timing, dict):
                raise InputError(location, "host_timing must be an object")
            start, end = timing.get("start_ns"), timing.get("end_ns")
            if (timing.get("valid") is not True or type(start) is not int or
                    type(end) is not int or start < 0 or end < start):
                summary["skipped"]["invalid_host_timing"] += 1
                continue
            if timing.get("clock") != "steady_clock" or timing.get("unit") != "nanosecond":
                raise InputError(location, "host_timing requires clock=steady_clock and unit=nanosecond")
            execution = timing.get("execution_id")
            run, layer = record.get("run_id"), record.get("layer")
            if (not isinstance(execution, str) or not execution.strip() or
                    type(run) is not int or run < 0 or not isinstance(layer, str) or not layer.strip()):
                summary["skipped"]["unknown_identity"] += 1
                continue
            sides = groups.setdefault(GroupKey(execution, run, layer), ([], []))
            if left:
                sides[0].append((start, end))
                summary["left_accepted"] += 1
            if right:
                sides[1].append((start, end))
                summary["right_accepted"] += 1
    return summary


def union_duration(intervals: List[Interval]) -> int:
    total, previous_end = 0, 0
    for start, end in sorted(intervals):
        total += max(0, end - max(start, previous_end))
        previous_end = max(previous_end, end)
    return total


class Arguments(argparse.Namespace):
    paths: List[Path]
    left_op: List[str]
    right_op: List[str]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Intersect unions of selected host_timing intervals in cycle JSONL logs.",
        epilog="Only valid steady_clock nanoseconds are used. Unknown execution/run/layer "
               "identities are excluded; different executions and runs are never compared. "
               "A group missing either side has overlap_ns=null, not an observed zero.",
    )
    parser.add_argument("paths", nargs="+", type=Path, help="main and/or ExSIA detail JSONL files")
    parser.add_argument("--left-op", required=True, action="append", help="exact left op; repeat to select a union")
    parser.add_argument("--right-op", required=True, action="append", help="exact right op; repeat to select a union")
    args = Arguments()
    parser.parse_args(namespace=args)
    operations = Operations(frozenset(args.left_op), frozenset(args.right_op))
    groups: GroupedIntervals = {}
    try:
        sources = [read_source(path, operations, groups) for path in args.paths]
    except (OSError, UnicodeError, InputError) as error:
        parser.error(str(error))
    summaries: List[GroupSummary] = []
    for key, (left, right) in sorted(groups.items()):
        left_ns, right_ns = union_duration(left), union_duration(right)
        overlap = left_ns + right_ns - union_duration(left + right) if left and right else None
        summaries.append({
            "execution_id": key.execution_id, "run_id": key.run_id, "layer": key.layer,
            "left_intervals": len(left), "right_intervals": len(right),
            "left_union_ns": left_ns, "right_union_ns": right_ns, "overlap_ns": overlap,
            "status": "complete" if overlap is not None else "missing_side",
        })
    print(json.dumps({
        "clock": "steady_clock", "unit": "nanosecond",
        "left_ops": sorted(operations.left), "right_ops": sorted(operations.right),
        "sources": sources, "groups": summaries,
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
