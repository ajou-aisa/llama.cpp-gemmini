from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

from scripts.utils.cycle_schema import (
    CycleSchemaError,
    JsonScalar,
    JsonValue,
    parse_json_line,
)
from scripts.utils.residual_profile import _mapping

IDENTITY: Final = ("matmul_invocation_id", "run_id", "layer", "stripe_id", "slot", "node_id", "worker_id")
MIRRORED: Final = frozenset({"TIMELINE", "STAGE", "EXSIA_WORKLOAD"})


@dataclass(frozen=True)
class Record:
    data: Mapping[str, JsonValue]
    path: Path
    line: int
    execution: str
    context: Mapping[str, JsonValue]


@dataclass(frozen=True)
class Capture:
    main: Path
    detail: Path | None
    execution: str
    summary: Mapping[str, JsonValue]


def scalar(value: JsonValue, line: int) -> JsonScalar:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise CycleSchemaError(line, "expected scalar field")


def text(record: Mapping[str, JsonValue], key: str, line: int) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise CycleSchemaError(line, f"{key} must be a nonempty string")
    return value


def integer(value: JsonValue, line: int) -> int:
    if type(value) is not int or value < 0:
        raise CycleSchemaError(line, "expected nonnegative integer")
    return value


def json_records(path: Path) -> Iterator[tuple[int, Mapping[str, JsonValue]]]:
    with path.open(encoding="utf-8") as stream:
        for line, contents in enumerate(stream, 1):
            if not contents.endswith("\n"):
                raise CycleSchemaError(line, f"{path}: truncated JSONL")
            data = _mapping(parse_json_line(contents, line), line)
            if data.get("schema") != "gemmini.cycle" or type(data.get("version")) is not int or data["version"] != 2:
                raise CycleSchemaError(line, f"{path}: unsupported schema/version")
            text(data, "record_type", line)
            yield line, data


def open_capture(main: Path, detail: Path | None, binary: Path) -> Capture:
    result = subprocess.run([str(binary.resolve()), "--json", str(main.resolve())],
                            capture_output=True, text=True, check=False)
    summary = _mapping(parse_json_line(result.stdout, 0), 0)
    if result.returncode or summary.get("available") is not True:
        raise CycleSchemaError(0, f"canonical summary unavailable: {summary.get('reason')}")
    execution = ""
    for line, data in json_records(main):
        if data.get("record_type") == "INFERENCE_EVENT" and data.get("event") == "session_start":
            execution = text(data, "execution_id", line)
            break
    if not execution:
        raise CycleSchemaError(0, "missing session execution identity")
    return Capture(main, detail, execution, summary)


def parse_record(capture: Capture, source: Path, item: tuple[int, Mapping[str, JsonValue]]) -> Record:
    line, data = item
    for key in IDENTITY:
        value = data.get(key)
        if value is not None:
            if key == "layer":
                text(data, key, line)
            else:
                integer(value, line)
    host = data.get("host_timing")
    host_id = _mapping(host, line).get("execution_id") if host is not None else None
    execution = data.get("execution_id", host_id)
    if execution is None and source == capture.main:
        execution = capture.execution
    if execution != capture.execution:
        raise CycleSchemaError(line, f"{source}: missing or mismatched execution_id")
    if host_id is not None and host_id != execution:
        raise CycleSchemaError(line, "conflicting host execution_id")
    raw_context = data.get("inference_context")
    context: Mapping[str, JsonValue] = {} if raw_context is None else _mapping(raw_context, line)
    if context:
        if integer(context.get("request_id"), line) == 0 or context.get("included") is not True:
            raise CycleSchemaError(line, "invalid inference request context")
        operation = context.get("operation_id")
        if operation is None:
            if context.get("phase") is not None:
                raise CycleSchemaError(line, "phase without operation")
        elif integer(operation, line) == 0 or text(context, "phase", line) not in {"prefill", "decode"}:
            raise CycleSchemaError(line, "invalid inference operation context")
    return Record(data, source, line, capture.execution, context)


def identity(record: Record) -> tuple[JsonScalar, ...]:
    data = record.data
    host = data.get("host_timing")
    timing: Mapping[str, JsonValue] = {} if host is None else _mapping(host, record.line)
    fields = ("record_type", "op", *IDENTITY, "metric", "event_sequence", "sequence")
    return (record.execution, *(scalar(data.get(key), record.line) for key in fields),
            *(scalar(timing.get(key, data.get(key)), record.line)
              for key in ("start_ns", "end_ns", "start", "end")))


def fingerprint(record: Record) -> str:
    payload = {key: value for key, value in record.data.items() if key != "inference_context"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def records(capture: Capture) -> Iterator[Record]:
    mirrors: dict[tuple[JsonScalar, ...], tuple[str, Mapping[str, JsonValue]]] = {}
    contexts: dict[tuple[JsonScalar, ...], Mapping[str, JsonValue]] = {}
    for item in json_records(capture.main):
        record = parse_record(capture, capture.main, item)
        if record.data.get("record_type") in MIRRORED:
            key = identity(record)
            if key in mirrors:
                raise CycleSchemaError(record.line, "duplicate main stage identity")
            mirrors[key] = fingerprint(record), record.context
        if record.data.get("record_type") == "EXSIA_RUN_SUMMARY":
            key = (record.execution, scalar(record.data.get("run_id"), record.line),
                   scalar(record.data.get("layer"), record.line))
            if key in contexts and contexts[key] != record.context:
                raise CycleSchemaError(record.line, "ambiguous ExSIA invocation context")
            contexts[key] = record.context
        yield record
    if capture.detail is None:
        return
    seen: set[tuple[JsonScalar, ...]] = set()
    for item in json_records(capture.detail):
        record = parse_record(capture, capture.detail, item)
        if record.data.get("record_type") not in MIRRORED:
            raise CycleSchemaError(record.line, "unsupported detail record type")
        key = identity(record)
        if key in seen:
            raise CycleSchemaError(record.line, "duplicate detail stage identity")
        seen.add(key)
        if key in mirrors:
            digest, context = mirrors[key]
            if digest != fingerprint(record) or ("inference_context" in record.data and record.context != context):
                raise CycleSchemaError(record.line, "conflicting main/detail mirror")
            continue
        invocation = (record.execution, scalar(record.data.get("run_id"), record.line),
                      scalar(record.data.get("layer"), record.line))
        if record.context and invocation not in contexts:
            raise CycleSchemaError(record.line, "included detail invocation has no main-log bridge")
        context = contexts.get(invocation, {})
        if "inference_context" in record.data and invocation in contexts and record.context != context:
            raise CycleSchemaError(record.line, "conflicting detail inference context")
        yield replace(record, context=record.context if "inference_context" in record.data else context)
