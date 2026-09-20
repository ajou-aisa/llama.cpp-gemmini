#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
# Basic usage:
#   python3 scripts/utils/cycle_timeline.py cycle-log.jsonl
# Default outputs: normalized rows plus thread/operator Chrome traces.
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

try:
    from .cycle_schema import CycleSchemaError, JsonValue, parse_json_line
except ImportError:
    from cycle_schema import CycleSchemaError, JsonValue, parse_json_line


COMPACT_KINDS = frozenset({"cpu", "segment", "cycle"})
FULL_CPU_TYPES = frozenset({"CPU_INTERVAL", "OPERATOR_SEGMENT"})
DEVICE_INTERVAL_TYPES = frozenset({"IM2P_STRIPE_TELEMETRY"})
DEVICE_SUMMARY_TYPES = frozenset({"IM2P_EXECUTION_TELEMETRY"})
EVENT_TYPE = "INFERENCE_EVENT"


class TimelineError(Exception):
    pass


@dataclass(frozen=True)
class TimelineRow:
    row_type: str
    source_line: int
    execution_id: Optional[str]
    kind: str
    op: Optional[str]
    layer: Optional[str]
    request_id: Optional[int]
    operation_id: Optional[int]
    phase: Optional[str]
    graph_id: Optional[int]
    operator_id: Optional[int]
    operator_kind: Optional[str]
    task_id: Optional[int]
    parent_task_id: Optional[int]
    segment_id: Optional[int]
    parent_segment_id: Optional[int]
    role: Optional[str]
    matmul_invocation_id: Optional[int]
    run_id: Optional[int]
    stripe_id: Optional[int]
    slot: Optional[int]
    node_id: Optional[int]
    worker_id: Optional[int]
    tid: Optional[int]
    tid_start: Optional[int]
    tid_end: Optional[int]
    ns_start: Optional[int]
    ns_end: Optional[int]
    wall_ns: Optional[int]
    cycle_clock: Optional[str]
    cycle_start: Optional[int]
    cycle_end: Optional[int]
    cycles: Optional[int]
    cycle_valid: Optional[bool]
    cycle_reason: Optional[str]
    cpu_interval_sequence: Optional[int]
    cpu_interval_samples: Optional[int]
    expected_stripes: Optional[int]
    event: Optional[str]
    token_index: Optional[int]
    token_id: Optional[int]


@dataclass
class ValidationSummary:
    records: int = 0
    normalized_rows: int = 0
    interval_rows: int = 0
    event_rows: int = 0
    device_rows: int = 0
    compact_rows: int = 0
    full_cpu_rows: int = 0
    invalid_cycle_rows: int = 0
    unplaced_device_rows: int = 0
    inference_rows: int = 0
    operator_rows: int = 0
    task_rows: int = 0
    stripe_rows: int = 0


def _mapping(value: JsonValue, line: int, field: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, dict):
        raise CycleSchemaError(line, f"{field} must be an object")
    return value


def _string(value: JsonValue, line: int, field: str, *, optional: bool = True) -> Optional[str]:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        raise CycleSchemaError(line, f"{field} must be a nonempty string")
    return value


def _uint(value: JsonValue, line: int, field: str, *, optional: bool = True) -> Optional[int]:
    if value is None and optional:
        return None
    if type(value) is not int or value < 0:
        raise CycleSchemaError(line, f"{field} must be a nonnegative integer")
    return value


def _bool(value: JsonValue, line: int, field: str) -> bool:
    if type(value) is not bool:
        raise CycleSchemaError(line, f"{field} must be boolean")
    return value


def _context(record: Mapping[str, JsonValue], line: int) -> tuple[Optional[int], Optional[int], Optional[str]]:
    raw = record.get("inference_context")
    if raw is None:
        return None, None, None
    context = _mapping(raw, line, "inference_context")
    request = _uint(context.get("request_id"), line, "inference_context.request_id", optional=False)
    operation = _uint(context.get("operation_id"), line, "inference_context.operation_id")
    phase = _string(context.get("phase"), line, "inference_context.phase")
    if context.get("included") is not True:
        raise CycleSchemaError(line, "inference_context.included must be true")
    if operation is None:
        if phase is not None:
            raise CycleSchemaError(line, "inference_context phase requires operation_id")
    elif operation == 0 or phase not in {"prefill", "decode"}:
        raise CycleSchemaError(line, "invalid inference operation context")
    if request == 0:
        raise CycleSchemaError(line, "request_id zero is reserved")
    return request, operation, phase


def _operator_context(record: Mapping[str, JsonValue], line: int) -> dict[str, Any]:
    raw = record.get("operator_context")
    if raw is None:
        return {}
    context = _mapping(raw, line, "operator_context")
    result: dict[str, Any] = {}
    for key in ("graph_id", "operator_id", "task_id", "parent_task_id", "segment_id", "parent_segment_id"):
        result[key] = _uint(context.get(key), line, f"operator_context.{key}")
    result["operator_kind"] = _string(context.get("operator_kind"), line, "operator_context.operator_kind")
    result["role"] = _string(context.get("role"), line, "operator_context.role")
    return result


def _identity(record: Mapping[str, JsonValue], line: int) -> dict[str, Optional[int]]:
    return {
        key: _uint(record.get(key), line, key)
        for key in ("matmul_invocation_id", "run_id", "stripe_id", "slot", "node_id", "worker_id")
    }


def _common(record: Mapping[str, JsonValue], line: int) -> dict[str, Any]:
    request_id, operation_id, phase = _context(record, line)
    operator = _operator_context(record, line)
    identity = _identity(record, line)
    return {
        "execution_id": _string(record.get("execution_id"), line, "execution_id"),
        "op": _string(record.get("op"), line, "op"),
        "layer": _string(record.get("layer"), line, "layer"),
        "request_id": request_id,
        "operation_id": operation_id,
        "phase": phase,
        "graph_id": operator.get("graph_id"),
        "operator_id": operator.get("operator_id"),
        "operator_kind": operator.get("operator_kind"),
        "task_id": operator.get("task_id"),
        "parent_task_id": operator.get("parent_task_id"),
        "segment_id": operator.get("segment_id"),
        "parent_segment_id": operator.get("parent_segment_id"),
        "role": operator.get("role"),
        **identity,
    }


def _compact_interval(record: Mapping[str, JsonValue], line: int, execution: Optional[str]) -> TimelineRow:
    kind = _string(record.get("kind"), line, "kind", optional=False)
    if kind not in COMPACT_KINDS:
        raise CycleSchemaError(line, f"unsupported compact kind {kind!r}")
    common = _common(record, line)
    common["execution_id"] = common["execution_id"] or execution
    if common["op"] is None:
        raise CycleSchemaError(line, "compact interval requires op")

    cycle_start = _uint(record.get("start"), line, "start", optional=False)
    cycle_end = _uint(record.get("end"), line, "end", optional=False)
    valid = _bool(record.get("valid"), line, "valid")
    raw_delta = record.get("delta")
    cycles = _uint(raw_delta, line, "delta") if raw_delta is not None else None
    reason = _string(record.get("reason"), line, "reason")

    if valid:
        if cycles is None or cycle_end < cycle_start or cycles != cycle_end - cycle_start:
            raise CycleSchemaError(line, "valid cycle interval has inconsistent endpoints/delta")
        if reason is not None:
            raise CycleSchemaError(line, "valid cycle interval must not carry a reason")
    else:
        if cycles is not None:
            raise CycleSchemaError(line, "invalid cycle interval must have delta=null")
        if reason is None:
            raise CycleSchemaError(line, "invalid cycle interval requires reason")

    ns_start = _uint(record.get("ns_start"), line, "ns_start", optional=False)
    ns_end = _uint(record.get("ns_end"), line, "ns_end", optional=False)
    if ns_end < ns_start:
        raise CycleSchemaError(line, "shared timeline regressed")

    tid = _uint(record.get("tid"), line, "tid")
    tid_start = _uint(record.get("tid_start"), line, "tid_start")
    tid_end = _uint(record.get("tid_end"), line, "tid_end")
    if tid is not None:
        if tid == 0 or tid_start is not None or tid_end is not None:
            raise CycleSchemaError(line, "compact interval must use either tid or tid_start/tid_end")
        tid_start = tid_end = tid
    else:
        if tid_start is None or tid_end is None:
            raise CycleSchemaError(line, "compact interval requires thread identity")
        if tid_start == 0 or tid_end == 0:
            raise CycleSchemaError(line, "thread id zero is unavailable")
    if valid and tid_start != tid_end:
        raise CycleSchemaError(line, "valid cycle interval cannot cross threads")

    return TimelineRow(
        row_type="interval", source_line=line, kind=kind,
        tid=tid, tid_start=tid_start, tid_end=tid_end,
        ns_start=ns_start, ns_end=ns_end, wall_ns=ns_end - ns_start,
        cycle_clock="thread_pmu", cycle_start=cycle_start, cycle_end=cycle_end,
        cycles=cycles, cycle_valid=valid, cycle_reason=reason,
        cpu_interval_sequence=_uint(record.get("cpu_interval_sequence"), line, "cpu_interval_sequence"),
        cpu_interval_samples=None, expected_stripes=None,
        event=None, token_index=None, token_id=None, **common,
    )


def _full_cpu_interval(record: Mapping[str, JsonValue], line: int, execution: Optional[str]) -> TimelineRow:
    record_type = _string(record.get("record_type"), line, "record_type", optional=False)
    common = _common(record, line)
    common["execution_id"] = common["execution_id"] or execution
    if common["op"] is None:
        raise CycleSchemaError(line, "CPU interval requires op")

    host = _mapping(record.get("host_timing"), line, "host_timing")
    if host.get("clock") != "steady_clock" or host.get("unit") != "nanosecond":
        raise CycleSchemaError(line, "host_timing must use steady_clock nanoseconds")
    ns_start = _uint(host.get("start_ns"), line, "host_timing.start_ns", optional=False)
    ns_end = _uint(host.get("end_ns"), line, "host_timing.end_ns", optional=False)
    tid_start = _uint(host.get("start_tid"), line, "host_timing.start_tid", optional=False)
    tid_end = _uint(host.get("end_tid"), line, "host_timing.end_tid", optional=False)
    if host.get("valid") is not True or ns_end < ns_start:
        raise CycleSchemaError(line, "invalid host_timing interval")

    native = _mapping(record.get("native_cycles"), line, "native_cycles")
    valid = _bool(native.get("valid"), line, "native_cycles.valid")
    cycles = _uint(native.get("delta"), line, "native_cycles.delta") if native.get("delta") is not None else None
    reason = _string(native.get("reason"), line, "native_cycles.reason")
    start_obj = _mapping(native.get("start"), line, "native_cycles.start")
    end_obj = _mapping(native.get("end"), line, "native_cycles.end")
    cycle_start = _uint(start_obj.get("value"), line, "native_cycles.start.value")
    cycle_end = _uint(end_obj.get("value"), line, "native_cycles.end.value")
    if valid:
        if cycles is None or cycle_start is None or cycle_end is None or cycles != cycle_end - cycle_start:
            raise CycleSchemaError(line, "valid native_cycles has inconsistent endpoints/delta")
    elif cycles is not None:
        raise CycleSchemaError(line, "invalid native_cycles must have delta=null")

    return TimelineRow(
        row_type="interval", source_line=line,
        kind="segment" if record_type == "OPERATOR_SEGMENT" else "cpu",
        tid=tid_start if tid_start == tid_end else None, tid_start=tid_start, tid_end=tid_end,
        ns_start=ns_start, ns_end=ns_end, wall_ns=ns_end - ns_start,
        cycle_clock="thread_pmu", cycle_start=cycle_start, cycle_end=cycle_end,
        cycles=cycles, cycle_valid=valid, cycle_reason=reason,
        cpu_interval_sequence=_uint(record.get("cpu_interval_sequence"), line, "cpu_interval_sequence"),
        cpu_interval_samples=None, expected_stripes=None,
        event=None, token_index=None, token_id=None, **common,
    )


def _event(record: Mapping[str, JsonValue], line: int, execution: Optional[str]) -> TimelineRow:
    common = _common(record, line)
    common["execution_id"] = common["execution_id"] or execution
    event = _string(record.get("event"), line, "event", optional=False)
    timestamp = _uint(record.get("timestamp_ns"), line, "timestamp_ns", optional=False)
    return TimelineRow(
        row_type="event", source_line=line, kind="event",
        tid=None, tid_start=None, tid_end=None,
        ns_start=timestamp, ns_end=timestamp, wall_ns=0,
        cycle_clock=None, cycle_start=None, cycle_end=None, cycles=None,
        cycle_valid=None, cycle_reason=None,
        cpu_interval_sequence=None,
        cpu_interval_samples=_uint(record.get("cpu_interval_samples"), line, "cpu_interval_samples"),
        expected_stripes=None,
        event=event,
        token_index=_uint(record.get("token_index"), line, "token_index"),
        token_id=_uint(record.get("token_id"), line, "token_id"),
        **common,
    )


def _device_interval(record: Mapping[str, JsonValue], line: int, execution: Optional[str]) -> TimelineRow:
    common = _common(record, line)
    common["execution_id"] = common["execution_id"] or execution
    start = _uint(record.get("publish_cycle"), line, "publish_cycle", optional=False)
    end = _uint(record.get("completion_cycle"), line, "completion_cycle", optional=False)
    delta = _uint(record.get("latency_cycles"), line, "latency_cycles", optional=False)
    if end < start or delta != end - start:
        raise CycleSchemaError(line, "device cycle interval has inconsistent endpoints/delta")
    return TimelineRow(
        row_type="device", source_line=line, kind="device",
        tid=None, tid_start=None, tid_end=None,
        ns_start=None, ns_end=None, wall_ns=None,
        cycle_clock=_string(record.get("source"), line, "source") or "rtl_cycle",
        cycle_start=start, cycle_end=end, cycles=delta,
        cycle_valid=True, cycle_reason=None,
        cpu_interval_sequence=None, cpu_interval_samples=None, expected_stripes=None, event=None,
        token_index=None, token_id=None, **common,
    )


def _device_summary(record: Mapping[str, JsonValue], line: int, execution: Optional[str]) -> TimelineRow:
    common = _common(record, line)
    common["execution_id"] = common["execution_id"] or execution
    expected_stripes = _uint(record.get("rtl_stripes_published"), line, "rtl_stripes_published")
    cycles = _uint(record.get("rtl_work_total_cycles"), line, "rtl_work_total_cycles")
    return TimelineRow(
        row_type="device_summary", source_line=line, kind="device_summary",
        tid=None, tid_start=None, tid_end=None,
        ns_start=None, ns_end=None, wall_ns=None,
        cycle_clock=_string(record.get("source"), line, "source") or "rtl_cycle",
        cycle_start=None, cycle_end=None, cycles=cycles,
        cycle_valid=True if cycles is not None else None, cycle_reason=None,
        cpu_interval_sequence=None, cpu_interval_samples=None,
        expected_stripes=expected_stripes, event=None, token_index=None, token_id=None,
        **common,
    )


def read_rows(path: Path) -> tuple[list[TimelineRow], ValidationSummary]:
    rows: list[TimelineRow] = []
    summary = ValidationSummary()
    execution: Optional[str] = None

    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.endswith("\n"):
                raise CycleSchemaError(line_number, "truncated JSONL line")
            if not line.strip():
                continue
            raw = parse_json_line(line, line_number)
            record = _mapping(raw, line_number, "record")
            summary.records += 1

            record_type = record.get("record_type")
            if record_type == EVENT_TYPE and record.get("event") == "session_start":
                candidate = _string(record.get("execution_id"), line_number, "execution_id", optional=False)
                if execution is not None and execution != candidate:
                    raise CycleSchemaError(line_number, "multiple execution_id values in one log")
                execution = candidate

            row: Optional[TimelineRow] = None
            if record.get("kind") in COMPACT_KINDS and record_type is None:
                row = _compact_interval(record, line_number, execution)
                summary.compact_rows += 1
            elif record_type in FULL_CPU_TYPES:
                row = _full_cpu_interval(record, line_number, execution)
                summary.full_cpu_rows += 1
            elif record_type == EVENT_TYPE:
                row = _event(record, line_number, execution)
            elif record_type in DEVICE_INTERVAL_TYPES:
                row = _device_interval(record, line_number, execution)
            elif record_type in DEVICE_SUMMARY_TYPES:
                row = _device_summary(record, line_number, execution)

            if row is None:
                continue
            rows.append(row)
            summary.normalized_rows += 1
            if row.row_type == "interval":
                summary.interval_rows += 1
            elif row.row_type == "event":
                summary.event_rows += 1
            elif row.row_type in {"device", "device_summary"}:
                summary.device_rows += 1
                if row.row_type == "device" and row.ns_start is None:
                    summary.unplaced_device_rows += 1
            if row.cycle_valid is False:
                summary.invalid_cycle_rows += 1
            if row.request_id is not None:
                summary.inference_rows += 1
            if row.operator_id is not None:
                summary.operator_rows += 1
            if row.task_id is not None:
                summary.task_rows += 1
            if row.stripe_id is not None:
                summary.stripe_rows += 1

    return rows, summary


def validate_relationships(rows: list[TimelineRow], require_valid_cycles: bool) -> list[str]:
    errors: list[str] = []
    segment_owner: dict[int, tuple[Optional[int], Optional[int], Optional[str]]] = {}
    task_operators: dict[int, tuple[Optional[int], Optional[str]]] = {}

    for row in rows:
        if require_valid_cycles and row.row_type == "interval" and row.kind == "cpu" and row.cycle_valid is not True:
            errors.append(f"line {row.source_line}: CPU interval has invalid cycle counter ({row.cycle_reason})")
        if row.segment_id is not None:
            owner = (row.task_id, row.operator_id, row.op)
            previous = segment_owner.setdefault(row.segment_id, owner)
            if previous != owner:
                errors.append(
                    f"line {row.source_line}: segment_id {row.segment_id} reused by different owners"
                )
        if row.task_id is not None:
            owner = (row.operator_id, row.operator_kind)
            previous = task_operators.setdefault(row.task_id, owner)
            if previous != owner:
                errors.append(
                    f"line {row.source_line}: task_id {row.task_id} crosses operator identity"
                )
        if row.parent_task_id is not None and row.task_id is not None and row.parent_task_id == row.task_id:
            errors.append(f"line {row.source_line}: task_id is its own parent")
        if row.parent_segment_id is not None and row.segment_id is not None and row.parent_segment_id == row.segment_id:
            errors.append(f"line {row.source_line}: segment_id is its own parent")

    return errors


def validate_cpu_cardinality(rows: list[TimelineRow]) -> tuple[list[str], dict[str, int | str]]:
    errors: list[str] = []
    sequences: dict[tuple[Optional[str], int, int], list[tuple[int, int]]] = defaultdict(list)
    expected: dict[tuple[Optional[str], int, int], tuple[int, int]] = {}
    operation_ends = 0
    legacy_operation_ends = 0

    for row in rows:
        if row.cpu_interval_sequence is not None:
            if row.request_id is None or row.operation_id is None:
                errors.append(
                    f"line {row.source_line}: cpu_interval_sequence lacks request/operation context"
                )
            else:
                key = (row.execution_id, row.request_id, row.operation_id)
                sequences[key].append((row.cpu_interval_sequence, row.source_line))
        if row.row_type == "event" and row.event == "operation_end":
            operation_ends += 1
            if row.request_id is None or row.operation_id is None:
                errors.append(f"line {row.source_line}: operation_end lacks request/operation context")
                continue
            if row.cpu_interval_samples is None:
                legacy_operation_ends += 1
                continue
            key = (row.execution_id, row.request_id, row.operation_id)
            if key in expected:
                errors.append(f"line {row.source_line}: duplicate operation_end CPU count contract")
            else:
                expected[key] = (row.cpu_interval_samples, row.source_line)

    verified = 0
    for key, (count, end_line) in expected.items():
        observed = sorted(value for value, _ in sequences.get(key, []))
        target = list(range(1, count + 1))
        if observed != target:
            errors.append(
                f"line {end_line}: CPU interval cardinality mismatch for request={key[1]} "
                f"operation={key[2]} expected=1..{count} observed={observed[:16]}"
            )
        else:
            verified += 1

    contracted_keys = set(expected)
    sequence_keys = set(sequences)
    orphan_sequence_keys = sequence_keys - contracted_keys
    # Old logs intentionally lacked cpu_interval_samples. Do not call those holes;
    # they remain explicitly unverified instead of being treated as complete.
    unverified = len(orphan_sequence_keys) + legacy_operation_ends
    status = "verified" if expected and not errors and unverified == 0 else (
        "partial" if expected and not errors else "unverified" if not expected else "invalid"
    )
    return errors, {
        "status": status,
        "operation_end_records": operation_ends,
        "operations_with_count_contract": len(expected),
        "verified_operations": verified,
        "legacy_operation_ends": legacy_operation_ends,
        "unverified_sequence_operations": len(orphan_sequence_keys),
    }


def validate_stripe_cardinality(rows: list[TimelineRow]) -> tuple[list[str], dict[str, int | str]]:
    errors: list[str] = []
    stripes: dict[tuple[Optional[str], int, Optional[str]], list[tuple[int, int]]] = defaultdict(list)
    expected: dict[tuple[Optional[str], int, Optional[str]], tuple[int, int]] = {}

    for row in rows:
        if row.row_type == "device":
            if row.run_id is None or row.stripe_id is None:
                errors.append(f"line {row.source_line}: device stripe lacks run_id/stripe_id")
                continue
            stripes[(row.execution_id, row.run_id, row.layer)].append((row.stripe_id, row.source_line))
        elif row.row_type == "device_summary" and row.expected_stripes is not None:
            if row.run_id is None:
                errors.append(f"line {row.source_line}: device summary stripe contract lacks run_id")
                continue
            key = (row.execution_id, row.run_id, row.layer)
            if key in expected:
                errors.append(f"line {row.source_line}: duplicate device stripe count contract")
            else:
                expected[key] = (row.expected_stripes, row.source_line)

    verified = 0
    for key, (count, source_line) in expected.items():
        observed = sorted(value for value, _ in stripes.get(key, []))
        target = list(range(count))
        if observed != target:
            errors.append(
                f"line {source_line}: IM2P stripe cardinality mismatch for run={key[1]} "
                f"expected=0..{count - 1 if count else -1} observed={observed[:16]}"
            )
        else:
            verified += 1

    unverified = len(set(stripes) - set(expected))
    status = "verified" if expected and not errors and unverified == 0 else (
        "partial" if expected and not errors else "unverified" if not expected else "invalid"
    )
    return errors, {
        "status": status,
        "runs_with_count_contract": len(expected),
        "verified_runs": verified,
        "unverified_stripe_runs": unverified,
        "stripe_rows": sum(len(value) for value in stripes.values()),
    }


def _row_dict(row: TimelineRow) -> dict[str, Any]:
    return {key: value for key, value in asdict(row).items() if value is not None}


def write_rows(path: Path, rows: list[TimelineRow]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(_row_dict(row), separators=(",", ":"), ensure_ascii=False) + "\n")


TRACE_VIEWS = ("thread", "operator", "task", "stripe")


def _trace_lane(row: TimelineRow, view: str) -> Optional[tuple[object, str]]:
    if view == "thread":
        tid = row.tid if row.tid is not None else row.tid_start
        return None if tid is None else (tid, f"tid {tid}")
    if view == "operator":
        if row.operator_id is None:
            return None
        label = f"operator {row.operator_id}"
        if row.operator_kind:
            label += f" {row.operator_kind}"
        return row.operator_id, label
    if view == "task":
        if row.task_id is None:
            return None
        label = f"task {row.task_id}"
        if row.operator_kind:
            label += f" ({row.operator_kind})"
        return row.task_id, label
    if view == "stripe":
        if row.run_id is None or row.stripe_id is None:
            return None
        key = (row.run_id, row.stripe_id, row.layer)
        label = f"run {row.run_id} / stripe {row.stripe_id}"
        if row.layer:
            label += f" / {row.layer}"
        return key, label
    raise ValueError(f"unsupported trace view: {view}")


def write_trace(path: Path, rows: list[TimelineRow], included_only: bool,
                view: str) -> dict[str, int | str]:
    executions = [row.execution_id or "unknown" for row in rows if row.ns_start is not None]
    execution_order = list(dict.fromkeys(executions))
    pid_for = {execution: index + 1 for index, execution in enumerate(execution_order)}
    origins: dict[str, int] = {}
    for row in rows:
        if row.ns_start is None:
            continue
        execution = row.execution_id or "unknown"
        origins[execution] = min(origins.get(execution, row.ns_start), row.ns_start)

    events: list[dict[str, Any]] = []
    for execution in execution_order:
        events.append({
            "ph": "M", "name": "process_name", "pid": pid_for[execution], "tid": 0,
            "args": {"name": execution},
        })

    lane_ids: dict[tuple[str, object], int] = {}
    lane_names: dict[tuple[str, object], str] = {}
    next_lane: dict[str, int] = defaultdict(lambda: 1)
    placed = 0
    skipped_missing_lane = 0
    skipped_context_free = 0

    for row in rows:
        if row.ns_start is None:
            continue
        if included_only and row.request_id is None and row.row_type != "event":
            skipped_context_free += 1
            continue
        execution = row.execution_id or "unknown"
        pid = pid_for[execution]
        origin = origins[execution]
        if row.row_type == "event":
            events.append({
                "ph": "i", "s": "p", "name": row.event or "event", "cat": "inference",
                "pid": pid, "tid": 0, "ts": (row.ns_start - origin) / 1000.0,
                "args": _row_dict(row),
            })
            continue

        lane = _trace_lane(row, view)
        if lane is None:
            skipped_missing_lane += 1
            continue
        lane_key, lane_name = lane
        map_key = (execution, lane_key)
        if view == "thread":
            lane_id = int(lane_key)
        else:
            lane_id = lane_ids.get(map_key, 0)
            if lane_id == 0:
                lane_id = next_lane[execution]
                next_lane[execution] += 1
                lane_ids[map_key] = lane_id
        if map_key not in lane_names:
            lane_names[map_key] = lane_name
            events.append({
                "ph": "M", "name": "thread_name", "pid": pid, "tid": lane_id,
                "args": {"name": lane_name},
            })
        events.append({
            "ph": "X", "name": row.op or row.kind, "cat": row.kind,
            "pid": pid, "tid": lane_id, "ts": (row.ns_start - origin) / 1000.0,
            "dur": ((row.ns_end or row.ns_start) - row.ns_start) / 1000.0,
            "args": _row_dict(row),
        })
        placed += 1

    path.write_text(
        json.dumps({"traceEvents": events, "displayTimeUnit": "ns", "view": view},
                   separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "view": view,
        "placed_intervals": placed,
        "skipped_missing_lane": skipped_missing_lane,
        "skipped_context_free": skipped_context_free,
    }


def report(rows: list[TimelineRow], summary: ValidationSummary, relationship_errors: list[str],
           cpu_interval_coverage: Mapping[str, int | str],
           stripe_coverage: Mapping[str, int | str]) -> dict[str, Any]:
    by_kind = Counter(row.kind for row in rows)
    by_op = Counter(row.op for row in rows if row.op is not None)
    invalid = Counter(row.cycle_reason or "unknown" for row in rows if row.cycle_valid is False)
    threads = sorted({
        tid for row in rows for tid in (row.tid, row.tid_start, row.tid_end)
        if tid is not None
    })
    return {
        **asdict(summary),
        "kinds": dict(sorted(by_kind.items())),
        "ops": dict(sorted(by_op.items())),
        "invalid_cycle_reasons": dict(sorted(invalid.items())),
        "thread_count": len(threads),
        "thread_ids": threads if len(threads) <= 64 else threads[:64],
        "thread_ids_truncated": len(threads) > 64,
        "cpu_interval_coverage": dict(cpu_interval_coverage),
        "stripe_coverage": dict(stripe_coverage),
        "relationship_errors": relationship_errors,
        "status": "ok" if not relationship_errors else "invalid",
        "timeline_semantics": {
            "ns": "shared steady_clock placement/overlap",
            "thread_cycles": "same-thread CPU work only; never a cross-thread absolute clock",
            "device_cycles": "device-local counter; rows without ns are intentionally unplaced",
        },
    }


class Arguments(argparse.Namespace):
    input: Path
    rows: Optional[Path]
    trace_dir: Optional[Path]
    view: Optional[list[str]]
    all_views: bool
    allow_invalid_cycles: bool
    included_only: bool


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and normalize compact cycle logs without aggregating their raw intervals.",
        epilog=(
            "The exported rows preserve raw interval granularity. ns_start/ns_end are the shared observed "
            "timeline; CPU cycle start/end/delta remain thread-local cost. Device-local cycles without host ns "
            "are exported but not placed on the Chrome trace."
        ),
    )
    parser.add_argument("input", type=Path, help="cycle-log.jsonl")
    parser.add_argument("--rows", type=Path,
                        help="normalized raw JSONL output; default: <input>.timeline.jsonl")
    parser.add_argument("--trace-dir", type=Path,
                        help="directory for Chrome traces; default: input directory")
    parser.add_argument("--view", action="append", choices=TRACE_VIEWS,
                        help="trace lane view; repeatable. default: thread + operator")
    parser.add_argument("--all-views", action="store_true",
                        help="write thread, operator, task, and stripe traces")
    parser.add_argument("--allow-invalid-cycles", action="store_true",
                        help="inspect non-PMU logs without failing invalid CPU cycle intervals")
    parser.add_argument("--included-only", action="store_true",
                        help="omit context-free warmup/raw intervals from trace output")
    args = Arguments()
    parser.parse_args(namespace=args)

    base_name = args.input.name[:-6] if args.input.name.endswith(".jsonl") else args.input.stem
    rows_path = args.rows or args.input.with_name(base_name + ".timeline.jsonl")
    trace_dir = args.trace_dir or args.input.parent
    if args.all_views:
        views = list(TRACE_VIEWS)
    elif args.view:
        views = list(dict.fromkeys(args.view))
    else:
        views = ["thread", "operator"]

    try:
        rows, summary = read_rows(args.input)
        errors = validate_relationships(rows, not args.allow_invalid_cycles)
        cardinality_errors, cpu_interval_coverage = validate_cpu_cardinality(rows)
        errors.extend(cardinality_errors)
        stripe_errors, stripe_coverage = validate_stripe_cardinality(rows)
        errors.extend(stripe_errors)
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        write_rows(rows_path, rows)
        trace_dir.mkdir(parents=True, exist_ok=True)
        traces: dict[str, dict[str, int | str]] = {}
        for view in views:
            trace_path = trace_dir / f"{base_name}.timeline.{view}.chrome.json"
            stats = write_trace(trace_path, rows, args.included_only, view)
            traces[view] = {**stats, "path": str(trace_path)}
        result = report(rows, summary, errors, cpu_interval_coverage, stripe_coverage)
        result["rows_output"] = str(rows_path)
        result["traces"] = traces
        print(json.dumps(result, separators=(",", ":"), ensure_ascii=False))
        return 0 if not errors else 2
    except (OSError, UnicodeError, CycleSchemaError, TimelineError) as error:
        print(json.dumps({"status": "invalid", "error": str(error)}, separators=(",", ":")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
