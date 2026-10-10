#!/usr/bin/env python3
from __future__ import annotations

import bisect
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

# An accepted paired companion's host recomposition is a non-additive parent. These canonical stages
# carry its whole cost, in this order, inside it on the same thread.
PAIRED_PARENT = "im2p.residual_paired_recomposition"
PAIRED_STAGES = ("im2p.residual_paired_setup", "im2p.residual_radix_recomposition",
                 "im2p.residual_paired_finalize", "im2p.residual_output_publish", "im2p.residual_release")
# A compact residual on the run-aware route runs inside the excluded simulator envelope. Its host work is
# carried by these canonical stages, in this order, inside the envelope on the same thread; other
# canonical stages (output reconstruction during the emulated NPU work) may lie between them.
WS_ENVELOPE = "im2p.residual_simulator_host_call"
WS_STAGES = ("im2p.residual_request_validation", "im2p.residual_run_preparation",
             "im2p.residual_recomposition_setup", "im2p.residual_radix_recomposition",
             "im2p.residual_recomposition_finalize", "im2p.residual_output_publish", "im2p.residual_release")
PROBE = "HOST_COMPLETION_PROBE"
PROBE_COUNTERS = ("instrumentation_ns", "instrumentation_calls", "dispatch_host_ns", "dispatch_host_calls",
                  "emulation_ns", "emulation_calls")
# The fields of a container interval that check_containers reads; nothing else is kept.
CONTAINER_FIELDS = ("op", "duration_role", "cpu_service", "interval_class", "additive", "operation_success",
                    "host_execution_id", "thread_id", "host_start_ns", "host_end_ns", "host_elapsed_ns",
                    "host_elapsed_valid")

# The logs are streamed: memory holds one record plus the canonical intervals and stage identities (a full
# E2E PoTal log has millions of rows). Failures keep the order of a reader that parses every line of a file
# before checking it: a malformed line anywhere in a file is reported before a failed check of that file, and
# the first failed check (in file order, then the post-file checks in a fixed order) is the one reported.


def timeline_rows(path: Path) -> int:
    """The E2E timeline reader's rule (scripts/eval/e2e_timeline.py thread_lifetimes): every row naming a
    thread carries a host execution and span, so every record this log adds stays readable downstream."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/eval"))
    from e2e_timeline import thread_lifetimes  # noqa: E402

    try:
        return len(thread_lifetimes(path))
    except Exception as error:  # EvaluationError and friends
        raise ValueError(f"cycle log is not readable by the E2E timeline reader: {error}") from error


def rows(path: Path):
    """The records of a JSON-lines file in order, one at a time: the lines of str.splitlines(), empty ones skipped."""
    with path.open() as stream:
        for physical in stream:
            for line in physical.splitlines():
                if line:
                    yield json.loads(line)


def host_interval(record: dict[str, object]) -> tuple[str, int, int, int]:
    execution, thread = record.get("host_execution_id"), record.get("thread_id")
    start, end = record.get("host_start_ns"), record.get("host_end_ns")
    if (isinstance(execution, str) and isinstance(thread, int) and isinstance(start, int) and
            isinstance(end, int) and start <= end):
        elapsed = record.get("host_elapsed_ns")
        if record.get("host_elapsed_valid") is False or (elapsed is not None and elapsed != end - start):
            raise ValueError(f"PoTal host measurement has an inconsistent host clock: {record.get('op')}")
        return execution, thread, start, end
    raise ValueError(f"PoTal host measurement lacks a host interval: {record.get('op')}")


Threads = dict[tuple[str, int], list[tuple[int, int, str]]]


def thread_index(canonical: list[tuple[str, int, int, int, str]]) -> tuple[Threads, list[tuple[str, int]]]:
    """Per (host execution, thread): its canonical intervals sorted by (start, end, op); and the threads that have
    a positive-length interval, in the order of their first one."""
    threads: Threads = defaultdict(list)
    positive: dict[tuple[str, int], None] = {}
    for execution, thread, start, end, op in canonical:
        threads[(execution, thread)].append((start, end, op))
        if end > start:
            positive.setdefault((execution, thread))
    for spans in threads.values():
        spans.sort()
    return threads, list(positive)


def check_additive(threads: Threads, order: list[tuple[str, int]]) -> int:
    """Canonical host costs are summed, so within one host execution and thread they must be disjoint:
    the rule IM2P.sim reconstruct_cpu (validate_potal_host_additivity) applies at the join."""
    for key in order:
        prior_end, prior_op = None, None
        for start, end, op in threads[key]:
            if end <= start:
                continue
            if prior_end is not None and start < prior_end:
                raise ValueError(f"canonical PoTal host intervals overlap on one thread: {op} starts inside {prior_op}")
            if prior_end is None or end > prior_end:
                prior_end, prior_op = end, op
    return len(order)


def inside(threads: Threads, container: dict[str, object]) -> list[str]:
    """Canonical stages of the container's thread with start <= s and e <= end, ordered by (s, e, op)."""
    execution, thread, start, end = host_interval(container)
    spans = threads.get((execution, thread), [])
    i = bisect.bisect_left(spans, (start,))
    ops = []
    while i < len(spans) and spans[i][0] <= end:
        if spans[i][1] <= end:
            ops.append(spans[i][2])
        i += 1
    return ops


def subsequence_once(ops: list[str], stages: tuple[str, ...]) -> bool:
    counts = Counter(ops)
    order = [op for op in ops if op in stages]
    return all(counts[stage] == 1 for stage in stages) and tuple(order) == stages


def check_containers(containers: list[dict[str, object]], threads: Threads) -> dict[str, int]:
    found = Counter()
    for record in containers:
        op = record.get("op")
        if (record.get("duration_role") != "OBSERVATION_ONLY" or record.get("cpu_service") is not False or
                record.get("interval_class") == "CANONICAL_ADDITIVE" or record.get("additive") is True):
            raise ValueError(f"{op} must be a non-additive interval around canonical stages")
        found[str(op)] += 1
        if record.get("operation_success") is not True:
            continue
        ops = inside(threads, record)
        if op == PAIRED_PARENT:
            if tuple(ops) != PAIRED_STAGES:
                raise ValueError("paired recomposition host cost is not carried by its canonical stages: " + ",".join(ops))
        elif "im2p.residual_run_preparation" in ops:
            if not subsequence_once(ops, WS_STAGES):
                raise ValueError("compact residual host work is not carried by its canonical stages: " +
                                 ",".join(op for op in ops if op.startswith("im2p.residual_")))
            found["ws_run_aware"] += 1
    return found


def valid_probe(record: dict[str, object]) -> bool:
    return (record.get("additive") is False and record.get("duration_role") == "OBSERVATION_ONLY" and
            record.get("host_stage_id") is None and
            all(isinstance(value := record.get(k), int) and value >= 0 for k in PROBE_COUNTERS))


def check_probes(invalid: bool, per_parent: Counter, found: dict[str, int]) -> None:
    if invalid:
        raise ValueError("invalid residual completion probe")
    if any(found.get(op, 0) != per_parent.get(op, 0) for op in (PAIRED_PARENT, WS_ENVELOPE)):
        raise ValueError(f"one completion probe per residual completion required: intervals {dict(found)}, "
                         f"probes {dict(per_parent)}")


def main() -> int:
    trace, cycle_path = map(Path, sys.argv[1:3])
    expected: dict[int, str] = {}
    completed: set[int] = set()
    run_success = False
    failure: Exception | None = None
    for record in rows(trace):
        if failure is not None:
            continue
        try:
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
                expected[identity] = sys.intern(stage)
            elif record.get("event") == "END" and record.get("status") == "success":
                completed.add(identity)
        except Exception as error:  # noqa: BLE001 - reported once the whole file has parsed
            failure = error
    if failure is not None:
        raise failure
    # Every stored measurement names its declared stage, so the measured map equals the declared one exactly
    # when their identity sets are equal.
    measured: set[int] = set()
    canonical: list[tuple[str, int, int, int, str]] = []
    containers: list[dict[str, object]] = []
    per_parent: Counter = Counter()
    invalid_probe = False
    for record in rows(cycle_path):
        if failure is not None:
            continue
        try:
            if record.get("duration_role") == "POTAL_HOST":
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
                measured.add(identity)
                execution, thread, start, end = host_interval(record)
                canonical.append((sys.intern(execution), thread, start, end, expected[identity]))
            if record.get("record_type") == "CYCLE_INTERVAL" and record.get("op") in (PAIRED_PARENT, WS_ENVELOPE):
                containers.append({key: record[key] for key in CONTAINER_FIELDS if key in record})
            if record.get("record_type") == PROBE:
                per_parent[record.get("parent_op")] += 1
                invalid_probe = invalid_probe or not valid_probe(record)
        except Exception as error:  # noqa: BLE001 - reported once the whole file has parsed
            failure = error
    if failure is not None:
        raise failure
    if run_success and (set(expected) != completed or set(expected) != measured):
        raise ValueError("PoTal host declaration/completion/measurement mismatch")
    threads, order = thread_index(canonical)
    additive_threads = check_additive(threads, order)
    found = check_containers(containers, threads)
    check_probes(invalid_probe, per_parent, found)
    timeline_rows(cycle_path)
    print(f"POTAL_CANONICAL_HOST_INTERVALS_PASS count={len(expected)} additive_threads={additive_threads} "
          f"paired_parents={found.get(PAIRED_PARENT, 0)} ws_run_aware={found.get('ws_run_aware', 0)} "
          f"completion_probes={sum(per_parent.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
