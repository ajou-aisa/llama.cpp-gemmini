#!/usr/bin/env python3
"""Stream and validate repository-emitted Gemmini cycle JSONL records.

Run with: python3 scripts/utils/cycle_schema.py --check cycle-log.jsonl
Requires Python 3.9+ and the standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Mapping
from enum import Enum
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

JsonScalar = Union[None, bool, int, float, str]
JsonValue = Union[JsonScalar, List["JsonValue"], Dict[str, "JsonValue"]]


class RecordType(str, Enum):
    CYCLE_INTERVAL = "CYCLE_INTERVAL"
    WS_LOOP_TELEMETRY = "WS_LOOP_TELEMETRY"
    IM2P_EXECUTION_TELEMETRY = "IM2P_EXECUTION_TELEMETRY"
    IM2P_STRIPE_TELEMETRY = "IM2P_STRIPE_TELEMETRY"
    QUANTIZATION_STRIPE_TELEMETRY = "QUANTIZATION_STRIPE_TELEMETRY"
    PIPELINE_STRIPE_SUMMARY = "PIPELINE_STRIPE_SUMMARY"
    RMD_BACKEND_TELEMETRY = "RMD_BACKEND_TELEMETRY"
    RESIDUAL_HOST_PROFILE = "RESIDUAL_HOST_PROFILE"
    IM2P_RMD_STRIPE_TELEMETRY = "IM2P_RMD_STRIPE_TELEMETRY"
    IM2P_RMD_EXECUTION_TELEMETRY = "IM2P_RMD_EXECUTION_TELEMETRY"
    TIMELINE = "TIMELINE"
    STAGE = "STAGE"


class CycleRecord(NamedTuple):
    line_number: int
    record_type: RecordType
    source: Optional[str]
    unit: Optional[str]
    op: Optional[str]
    layer: Optional[str]
    run_id: Optional[int]
    stripe_id: Optional[int]
    slot: Optional[int]
    node_id: Optional[int]
    worker_id: Optional[int]
    start: Optional[int]
    end: Optional[int]
    delta: Optional[int]
    valid: Optional[bool]
    reason: Optional[str]
    canonical_json: str


class CycleSchemaError(Exception):
    """A cycle JSONL boundary failure tied to its physical input line."""

    __slots__ = ("line_number", "detail")

    line_number: int
    detail: str

    def __init__(self, line_number: int, detail: str) -> None:
        self.line_number = line_number
        self.detail = detail
        super().__init__(line_number, detail)

    def __str__(self) -> str:
        return f"line {self.line_number}: {self.detail}"


class _DuplicateKeyError(Exception):
    pass


class _InvalidConstantError(Exception):
    pass


_COMMON = "schema version record_type op layer run_id stripe_id slot node_id worker_id".split()
_SOURCED = "source unit".split()
_FAMILY_FIELDS = {
    RecordType.CYCLE_INTERVAL: "start end delta valid".split(),
    RecordType.WS_LOOP_TELEMETRY: ("problem_i problem_j problem_k tile_i tile_j tile_k gemmini_outer_i "
        "gemmini_outer_j gemmini_outer_k ws_inner_calls containing_interval_cycles containing_interval_counter_bits "
        "load_occupancy_cycles execute_occupancy_cycles store_occupancy_cycles loop_occupancy_cycles "
        "occupancy_counter_bits valid").split(),
    RecordType.IM2P_EXECUTION_TELEMETRY: "rtl_work_total_cycles".split(),
    RecordType.IM2P_STRIPE_TELEMETRY: "row_begin row_end publish_cycle completion_cycle latency_cycles additive".split(),
    RecordType.QUANTIZATION_STRIPE_TELEMETRY: ("row_begin row_end start end delta valid reason start_ns end_ns duration_ns "
        "overlaps_rtl additive").split(),
    RecordType.PIPELINE_STRIPE_SUMMARY: ("row_begin row_end queue_start_ns queue_end_ns dense_start_ns dense_end_ns "
        "rmd_start_ns rmd_end_ns compose_start_ns compose_end_ns finalize_start_ns finalize_end_ns valid").split(),
    RecordType.RMD_BACKEND_TELEMETRY: ("runtime_bundle_id model_id backend option_source work invocation_total "
        "dispatch timing geometry").split(),
    RecordType.RESIDUAL_HOST_PROFILE: "valid host_timing workload phases tiles workers".split(),
    RecordType.IM2P_RMD_STRIPE_TELEMETRY: "row_begin row_end rmd_dot_calls rmd_work_total_cycles clock_domain additive".split(),
    RecordType.IM2P_RMD_EXECUTION_TELEMETRY: "rmd_dot_calls rmd_work_total_cycles clock_domain additive".split(),
    RecordType.TIMELINE: ("mode start end start_thread_id end_thread_id clock_mode units timer_resolution team_size "
        "elapsed cycle_status").split(),
    RecordType.STAGE: "mode metric value value_units team_size cycle_status".split(),
}


def _pairs_to_mapping(pairs: List[Tuple[str, JsonValue]]) -> Dict[str, JsonValue]:
    result: Dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(key)
        result[key] = value
    return result


def _reject_constant(value: str) -> JsonValue:
    raise _InvalidConstantError(value)


def _require(record: Mapping[str, JsonValue], name: str, line_number: int) -> JsonValue:
    if name not in record:
        raise CycleSchemaError(line_number, f"missing required field {name!r}")
    return record[name]


def _optional_string(record: Mapping[str, JsonValue], name: str, line_number: int) -> Optional[str]:
    value = _require(record, name, line_number)
    if value is None:
        return None
    if type(value) is not str:
        raise CycleSchemaError(line_number, f"field {name!r} must be a string or null")
    return value


def _optional_integer(record: Mapping[str, JsonValue], name: str, line_number: int) -> Optional[int]:
    value = _require(record, name, line_number)
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise CycleSchemaError(line_number, f"field {name!r} must be a non-negative integer or null")
    return value


def _validate_required_types(record: Mapping[str, JsonValue], names: List[str], line_number: int) -> None:
    for name in names:
        value = _require(record, name, line_number)
        if name in {"layer", "op"}:
            if value is not None and type(value) is not str:
                raise CycleSchemaError(line_number, f"field {name!r} must be a string or null")
        elif name in {"valid", "work", "additive", "overlaps_rtl"}:
            if type(value) is not bool:
                raise CycleSchemaError(line_number, f"field {name!r} must be boolean")
        elif name in {"dispatch", "timing", "geometry", "host_timing", "workload", "phases"}:
            if not isinstance(value, dict):
                raise CycleSchemaError(line_number, f"field {name!r} must be an object")
        elif name in {"tiles", "workers"}:
            if not isinstance(value, list):
                raise CycleSchemaError(line_number, f"field {name!r} must be an array")
        elif name in {"schema", "record_type", "source", "unit", "runtime_bundle_id", "model_id", "backend", "option_source",
                      "clock_domain", "mode", "metric", "value_units", "cycle_status",
                      "clock_mode", "units", "reason"}:
            if type(value) is not str:
                raise CycleSchemaError(line_number, f"field {name!r} must be a string")
        elif value is not None and (type(value) is not int or value < 0):
            raise CycleSchemaError(line_number, f"field {name!r} must be a non-negative integer or null")


def rmd_value_status(record: Mapping[str, JsonValue], key: str, line_number: int) -> Tuple[Optional[int], str, str]:
    """Read the producer's flat nullable interval contract; old scalars stay unverified."""
    value = _optional_integer(record, key, line_number)
    suffixes = ("_valid", "_sample_reason", "_count", "_valid_count", "_not_applicable_count")
    if not any(key + suffix in record for suffix in suffixes):
        return value, "unknown", "legacy_validity_unknown"
    valid = _require(record, key + "_valid", line_number)
    reason = _optional_string(record, key + "_reason", line_number)
    sample_reason = _optional_string(record, key + "_sample_reason", line_number)
    counts = [_optional_integer(record, key + suffix, line_number)
              for suffix in ("_count", "_valid_count", "_not_applicable_count")]
    if type(valid) is not bool or any(count is None for count in counts):
        raise CycleSchemaError(line_number, f"field {key!r} requires validity and component counts")
    count, valid_count, not_applicable_count = counts
    if valid_count + not_applicable_count > count:
        raise CycleSchemaError(line_number, f"field {key!r} has inconsistent component counts")
    if valid:
        if value is None or reason or sample_reason or valid_count == 0 or valid_count + not_applicable_count != count:
            raise CycleSchemaError(line_number, f"field {key!r} is not explicitly complete")
        return value, "complete", ""
    if value is not None or not reason:
        raise CycleSchemaError(line_number, f"invalid field {key!r} requires null and reason")
    status = reason if reason in {"not_collected", "not_applicable"} else "invalid"
    return None, status, reason + (":" + sample_reason if sample_reason else "")


def _validate_arithmetic(record: Mapping[str, JsonValue], record_type: RecordType, line_number: int) -> None:
    checks: List[Tuple[str, str, str, bool]] = []
    if record_type == RecordType.CYCLE_INTERVAL:
        valid, delta, reason = record["valid"], record["delta"], record.get("reason")
        if valid and delta is None:
            raise CycleSchemaError(line_number, "valid interval requires a non-null delta")
        if not valid and (delta is not None or type(reason) is not str or not reason):
            raise CycleSchemaError(line_number, "invalid interval requires a null delta and nonempty reason")
        checks.append(("start", "end", "delta", False))
    if record_type == RecordType.PIPELINE_STRIPE_SUMMARY:
        pairs = (("row_begin", "row_end"), ("queue_start_ns", "queue_end_ns"),
                 ("dense_start_ns", "dense_end_ns"), ("rmd_start_ns", "rmd_end_ns"),
                 ("compose_start_ns", "compose_end_ns"), ("finalize_start_ns", "finalize_end_ns"))
        for start_name, end_name in pairs:
            start, end = record[start_name], record[end_name]
            if type(start) is not int or type(end) is not int or end < start:
                raise CycleSchemaError(line_number, f"ordered pair {start_name!r}/{end_name!r} is invalid")
    if record_type == RecordType.IM2P_STRIPE_TELEMETRY:
        checks.append(("publish_cycle", "completion_cycle", "latency_cycles", True))
    if record_type == RecordType.QUANTIZATION_STRIPE_TELEMETRY:
        checks.append(("start_ns", "end_ns", "duration_ns", False))
    if record_type == RecordType.RMD_BACKEND_TELEMETRY:
        rmd_value_status(record, "invocation_total", line_number)
        for key in ("prep", "backend_service", "merge", "residual_total", "queue"):
            if key in record["timing"]:
                rmd_value_status(record["timing"], key, line_number)
    if record_type in {RecordType.TIMELINE, RecordType.STAGE}:
        value = record["elapsed" if record_type == RecordType.TIMELINE else "value"]
        complete = record["cycle_status"] == "complete"
        if complete != (value is not None):
            raise CycleSchemaError(line_number, "CPU detail value must agree with cycle_status")
        for name in ("source", "unit", "sample_reason"):
            if name in record:
                _optional_string(record, name, line_number)
    if record_type == RecordType.TIMELINE:
        checks.append(("start", "end", "elapsed", False))
    for start_name, end_name, delta_name, wraps in checks:
        start = record[start_name]
        end = record[end_name]
        delta = record[delta_name]
        if delta is None:
            continue
        if type(start) is not int or type(end) is not int or type(delta) is not int:
            raise CycleSchemaError(line_number, f"interval {delta_name!r} has invalid endpoint types")
        expected = (end - start) % (1 << 64) if wraps else end - start
        if expected < 0 or delta != expected:
            raise CycleSchemaError(line_number, f"field {delta_name!r} is inconsistent with its interval")


def parse_json_line(line: str, line_number: int) -> JsonValue:
    """Decode JSON while rejecting duplicate keys and non-JSON constants."""
    try:
        return json.loads(line, object_pairs_hook=_pairs_to_mapping, parse_constant=_reject_constant)
    except json.JSONDecodeError as error:
        raise CycleSchemaError(line_number, f"malformed JSON at column {error.colno}") from None
    except _DuplicateKeyError as error:
        raise CycleSchemaError(line_number, f"duplicate key {error.args[0]!r}") from None
    except _InvalidConstantError as error:
        raise CycleSchemaError(line_number, f"invalid JSON constant {error.args[0]!r}") from None


def parse_cycle_line(line: str, line_number: int) -> CycleRecord:
    """Parse one physical JSONL line into a typed immutable record."""
    decoded = parse_json_line(line, line_number)
    if not isinstance(decoded, dict):
        raise CycleSchemaError(line_number, "record must be a JSON object")
    if decoded.get("schema") != "gemmini.cycle" or decoded.get("version") != 2:
        raise CycleSchemaError(line_number, "unsupported schema or version")
    raw_type = decoded.get("record_type")
    try:
        record_type = RecordType(raw_type)
    except (TypeError, ValueError):
        raise CycleSchemaError(line_number, f"unsupported record_type {raw_type!r}") from None
    required = _COMMON + _FAMILY_FIELDS[record_type]
    if record_type not in {RecordType.TIMELINE, RecordType.STAGE}:
        required += _SOURCED
    _validate_required_types(decoded, required, line_number)
    _validate_arithmetic(decoded, record_type, line_number)
    valid_value = decoded.get("valid")
    source = decoded.get("source")
    unit = decoded.get("unit")
    reason = decoded.get("reason")
    return CycleRecord(
        line_number=line_number,
        record_type=record_type,
        source=source if type(source) is str else None,
        unit=unit if type(unit) is str else None,
        op=_optional_string(decoded, "op", line_number),
        layer=_optional_string(decoded, "layer", line_number),
        run_id=_optional_integer(decoded, "run_id", line_number),
        stripe_id=_optional_integer(decoded, "stripe_id", line_number),
        slot=_optional_integer(decoded, "slot", line_number),
        node_id=_optional_integer(decoded, "node_id", line_number),
        worker_id=_optional_integer(decoded, "worker_id", line_number),
        start=_optional_integer(decoded, "start", line_number) if "start" in decoded else None,
        end=_optional_integer(decoded, "end", line_number) if "end" in decoded else None,
        delta=_optional_integer(decoded, "delta", line_number) if "delta" in decoded else None,
        valid=valid_value if type(valid_value) is bool else None,
        reason=reason if type(reason) is str else None,
        canonical_json=json.dumps(decoded, sort_keys=True, separators=(",", ":")),
    )


def parse_cycle_jsonl(path: Path) -> Iterator[CycleRecord]:
    """Yield validated records without retaining prior input lines."""
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise CycleSchemaError(line_number, "blank JSONL line")
            yield parse_cycle_line(line, line_number)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Gemmini v2 cycle JSONL stream")
    parser.add_argument("--check", type=Path, required=True, metavar="JSONL")
    arguments = parser.parse_args()
    counts = {record_type: 0 for record_type in RecordType}
    try:
        for record in parse_cycle_jsonl(arguments.check):
            counts[record.record_type] += 1
    except (CycleSchemaError, OSError) as error:
        print(error, file=sys.stderr)
        return 1
    for record_type, count in counts.items():
        if count:
            print(f"{record_type.value}={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
