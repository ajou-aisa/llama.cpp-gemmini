#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

# P6.3 S5.3: an accepted paired companion's host recomposition is a non-additive parent. These
# canonical stages carry its whole cost, in this order, inside it on the same thread.
PAIRED_PARENT = "im2p.residual_paired_recomposition"
PAIRED_STAGES = ("im2p.residual_paired_setup", "im2p.residual_radix_recomposition",
                 "im2p.residual_paired_finalize", "im2p.residual_output_publish")


def rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def host_interval(record: dict[str, object]) -> tuple[str, int, int, int]:
    execution, thread = record.get("host_execution_id"), record.get("thread_id")
    start, end = record.get("host_start_ns"), record.get("host_end_ns")
    if (isinstance(execution, str) and isinstance(thread, int) and isinstance(start, int) and
            isinstance(end, int) and start <= end):
        return execution, thread, start, end
    raise ValueError(f"PoTal host measurement lacks a host interval: {record.get('op')}")


def check_additive(canonical: list[tuple[str, int, int, int, str]]) -> int:
    """Canonical host costs are summed, so within one host execution and thread they must be disjoint:
    the rule IM2P.sim reconstruct_cpu (validate_potal_host_additivity) applies at the join."""
    threads: dict[tuple[str, int], list[tuple[int, int, str]]] = defaultdict(list)
    for execution, thread, start, end, op in canonical:
        if end > start:
            threads[(execution, thread)].append((start, end, op))
    for spans in threads.values():
        spans.sort()
        prior_end, prior_op = None, None
        for start, end, op in spans:
            if prior_end is not None and start < prior_end:
                raise ValueError(f"canonical PoTal host intervals overlap on one thread: {op} starts inside {prior_op}")
            if prior_end is None or end > prior_end:
                prior_end, prior_op = end, op
    return len(threads)


def check_paired_parents(cycle: list[dict[str, object]], canonical: list[tuple[str, int, int, int, str]]) -> int:
    count = 0
    for record in cycle:
        if record.get("op") != PAIRED_PARENT or record.get("record_type") != "CYCLE_INTERVAL":
            continue
        if (record.get("duration_role") != "OBSERVATION_ONLY" or record.get("cpu_service") is not False or
                record.get("interval_class") == "CANONICAL_ADDITIVE" or record.get("additive") is True):
            raise ValueError("paired recomposition must be a non-additive diagnostic parent")
        count += 1
        if record.get("operation_success") is not True:
            continue
        execution, thread, start, end = host_interval(record)
        inside = sorted((stage_start, stage_end, op) for stage_execution, stage_thread, stage_start, stage_end, op
                        in canonical if (stage_execution, stage_thread) == (execution, thread) and
                        start <= stage_start and stage_end <= end)
        if tuple(op for _, _, op in inside) != PAIRED_STAGES:
            raise ValueError("paired recomposition host cost is not carried by its canonical stages: " +
                             ",".join(op for _, _, op in inside))
    return count


def main() -> int:
    trace, cycle_path = map(Path, sys.argv[1:3])
    expected: dict[int, str] = {}
    completed: set[int] = set()
    run_success = False
    for record in rows(trace):
        if record.get("kind") == "RUN_END":
            run_success = record.get("status") == "success"
        if record.get("kind") != "HOST_STAGE" or record.get("execution_class") != "POTAL_HOST":
            continue
        identity = record.get("host_stage_id")
        stage = record.get("stage_name")
        if not isinstance(identity, int) or not isinstance(stage, str):
            raise ValueError("invalid PoTal host-stage identity")
        if record.get("event") == "BEGIN":
            if identity in expected:
                raise ValueError("duplicate PoTal host-stage declaration")
            expected[identity] = stage
        elif record.get("event") == "END" and record.get("status") == "success":
            completed.add(identity)
    measured: dict[int, str] = {}
    cycle = rows(cycle_path)
    canonical: list[tuple[str, int, int, int, str]] = []
    for record in cycle:
        if record.get("duration_role") != "POTAL_HOST":
            continue
        identity = record.get("host_stage_id")
        stage = record.get("op")
        if (not isinstance(identity, int) or not isinstance(stage, str) or
                record.get("interval_class") != "CANONICAL_ADDITIVE"):
            raise ValueError("invalid canonical PoTal host measurement")
        if "additive" in record and record["additive"] is not True:
            raise ValueError("canonical PoTal host measurement is non-additive")
        if identity in measured:
            raise ValueError("duplicate canonical PoTal host measurement")
        if identity not in expected or expected[identity] != stage:
            raise ValueError("undeclared or mismatched canonical PoTal host measurement")
        measured[identity] = stage
        canonical.append((*host_interval(record), stage))
    if run_success and (set(expected) != completed or expected != measured):
        raise ValueError("PoTal host declaration/completion/measurement mismatch")
    threads = check_additive(canonical)
    parents = check_paired_parents(cycle, canonical)
    print(f"POTAL_CANONICAL_HOST_INTERVALS_PASS count={len(expected)} additive_threads={threads} "
          f"paired_parents={parents}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
