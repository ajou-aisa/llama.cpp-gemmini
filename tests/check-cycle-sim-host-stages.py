#!/usr/bin/env python3
from __future__ import annotations

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


def timeline_rows(path: Path) -> int:
    """The E2E timeline reader's rule (scripts/eval/e2e_timeline.py thread_lifetimes): every row naming a
    thread carries a host execution and span, so every record this log adds stays readable downstream."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/eval"))
    from e2e_timeline import thread_lifetimes  # noqa: E402

    try:
        return len(thread_lifetimes(path))
    except Exception as error:  # EvaluationError and friends
        raise ValueError(f"cycle log is not readable by the E2E timeline reader: {error}") from error


def rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


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


def inside(canonical: list[tuple[str, int, int, int, str]], container: dict[str, object]) -> list[str]:
    execution, thread, start, end = host_interval(container)
    return [op for _, _, op in sorted((s, e, op) for x, t, s, e, op in canonical
                                     if (x, t) == (execution, thread) and start <= s and e <= end)]


def subsequence_once(ops: list[str], stages: tuple[str, ...]) -> bool:
    counts = Counter(ops)
    order = [op for op in ops if op in stages]
    return all(counts[stage] == 1 for stage in stages) and tuple(order) == stages


def check_containers(cycle: list[dict[str, object]], canonical: list[tuple[str, int, int, int, str]]) -> dict[str, int]:
    found = Counter()
    for record in cycle:
        op = record.get("op")
        if record.get("record_type") != "CYCLE_INTERVAL" or op not in (PAIRED_PARENT, WS_ENVELOPE):
            continue
        if (record.get("duration_role") != "OBSERVATION_ONLY" or record.get("cpu_service") is not False or
                record.get("interval_class") == "CANONICAL_ADDITIVE" or record.get("additive") is True):
            raise ValueError(f"{op} must be a non-additive interval around canonical stages")
        found[str(op)] += 1
        if record.get("operation_success") is not True:
            continue
        ops = inside(canonical, record)
        if op == PAIRED_PARENT:
            if tuple(ops) != PAIRED_STAGES:
                raise ValueError("paired recomposition host cost is not carried by its canonical stages: " + ",".join(ops))
        elif "im2p.residual_run_preparation" in ops:
            if not subsequence_once(ops, WS_STAGES):
                raise ValueError("compact residual host work is not carried by its canonical stages: " +
                                 ",".join(op for op in ops if op.startswith("im2p.residual_")))
            found["ws_run_aware"] += 1
    return found


def check_probes(cycle: list[dict[str, object]], found: dict[str, int]) -> int:
    probes = [r for r in cycle if r.get("record_type") == PROBE]
    for record in probes:
        if (record.get("additive") is not False or record.get("duration_role") != "OBSERVATION_ONLY" or
                record.get("host_stage_id") is not None or
                not all(isinstance(value := record.get(k), int) and value >= 0 for k in
                        ("instrumentation_ns", "instrumentation_calls", "dispatch_host_ns", "dispatch_host_calls",
                         "emulation_ns", "emulation_calls"))):
            raise ValueError("invalid residual completion probe")
    per_parent = Counter(r.get("parent_op") for r in probes)
    if any(found.get(op, 0) != per_parent.get(op, 0) for op in (PAIRED_PARENT, WS_ENVELOPE)):
        raise ValueError(f"one completion probe per residual completion required: intervals {dict(found)}, "
                         f"probes {dict(per_parent)}")
    return len(probes)


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
    found = check_containers(cycle, canonical)
    probes = check_probes(cycle, found)
    timeline_rows(cycle_path)
    print(f"POTAL_CANONICAL_HOST_INTERVALS_PASS count={len(expected)} additive_threads={threads} "
          f"paired_parents={found.get(PAIRED_PARENT, 0)} ws_run_aware={found.get('ws_run_aware', 0)} "
          f"completion_probes={probes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
