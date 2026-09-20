from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Final, TypedDict

from scripts.utils.cycle_schema import CycleSchemaError, JsonScalar, JsonValue
from scripts.utils.potal_records import IDENTITY, Record, integer, scalar, text
from scripts.utils.residual_profile import _cpu, _host, _mapping, _objects, _profile


class CsvRow(TypedDict):
    execution_id: str
    request_id: JsonScalar
    operation_id: JsonScalar
    token_step: JsonScalar
    phase: JsonScalar
    matmul_invocation_id: JsonScalar
    token_index: JsonScalar
    token_id: JsonScalar
    boundary: JsonScalar
    latency_boundary_status: JsonScalar
    run_id: JsonScalar
    layer: JsonScalar
    stripe_id: JsonScalar
    slot: JsonScalar
    node_id: JsonScalar
    worker_id: JsonScalar
    op: JsonScalar
    record_type: JsonScalar
    included: bool
    metric: str
    value: JsonScalar
    unit: str
    valid: bool
    reason: str
    scope: str
    coverage: str
    source_file: str
    source_line: int
    source_field: str
    backend: JsonScalar
    domain: JsonScalar
    source: JsonScalar
    additive: bool


COLUMNS: Final = tuple(CsvRow.__annotations__)
NPU_TYPES: Final = frozenset({"IM2P_EXECUTION_TELEMETRY", "IM2P_STRIPE_TELEMETRY",
    "IM2P_RMD_EXECUTION_TELEMETRY", "IM2P_RMD_STRIPE_TELEMETRY", "WS_LOOP_TELEMETRY"})
WORKLOAD_TYPES: Final = frozenset({"EXSIA_WORKLOAD", "RMD_STRIPE_TELEMETRY",
    "RMD_BACKEND_TELEMETRY", "QUANTIZATION_STRIPE_TELEMETRY", "MATMUL_CONFIGURATION", "INFERENCE_CONFIGURATION"})
META: Final = frozenset({"schema", "version", "record_type", "execution_id", "inference_context",
    "op", *IDENTITY, "host_timing", "thread_cpu_timing", "native_cycles", "valid", "additive", "reason"})


def base(record: Record) -> CsvRow:
    data, context = record.data, record.context
    device: Mapping[str, JsonValue] = _mapping(data["device"], record.line) if "device" in data else {}
    return CsvRow(execution_id=record.execution, request_id=scalar(context.get("request_id"), record.line),
        operation_id=scalar(context.get("operation_id"), record.line), token_step=None, phase=scalar(context.get("phase"), record.line),
        matmul_invocation_id=scalar(data.get("matmul_invocation_id"), record.line),
        token_index=scalar(data.get("token_index"), record.line), token_id=scalar(data.get("token_id"), record.line),
        boundary=scalar(data.get("boundary"), record.line), latency_boundary_status=None,
        run_id=scalar(data.get("run_id"), record.line), layer=scalar(data.get("layer"), record.line),
        stripe_id=scalar(data.get("stripe_id"), record.line), slot=scalar(data.get("slot"), record.line),
        node_id=scalar(data.get("node_id"), record.line), worker_id=scalar(data.get("worker_id"), record.line),
        op=scalar(data.get("op"), record.line), record_type=scalar(data.get("record_type"), record.line),
        included=bool(context), metric="", value=None, unit="", valid=False, reason="", scope="raw_nonadditive",
        coverage="incomplete_no_record_count_contract" if context else "excluded_no_inference_context",
        source_file=str(record.path), source_line=record.line, source_field="",
        backend=scalar(data.get("backend", device.get("backend")), record.line),
        domain=scalar(data.get("domain", data.get("clock_domain", device.get("clock_domain"))), record.line),
        source=scalar(data.get("source"), record.line), additive=False)


def measured(row: CsvRow, field: str, value: JsonScalar) -> CsvRow:
    return {**row, "metric": field, "source_field": field, "value": value,
            "valid": value is not None, "reason": "" if value is not None else "not_collected"}


def native_cycles(record: Record) -> tuple[int | None, str]:
    native = _mapping(record.data.get("native_cycles"), record.line)
    valid = native.get("valid")
    if type(valid) is not bool:
        raise CycleSchemaError(record.line, "native_cycles.valid must be boolean")
    reason, delta = native.get("reason"), native.get("delta")
    if not valid:
        if delta is not None:
            raise CycleSchemaError(record.line, "invalid native cycles must be null")
        return None, text(native, "reason", record.line)
    if reason is not None:
        raise CycleSchemaError(record.line, "valid native cycles cannot have a reason")
    start, end = (_mapping(native.get(key), record.line) for key in ("start", "end"))
    for sample in (start, end):
        if sample.get("valid") is not True or sample.get("source") != "linux_perf_cpu_cycles":
            raise CycleSchemaError(record.line, "native cycles require valid thread perf samples")
    for key in ("owner_token", "generation"):
        if integer(start.get(key), record.line) == 0 or start.get(key) != end.get(key):
            raise CycleSchemaError(record.line, "native counter ownership/generation mismatch")
    value = integer(delta, record.line)
    if integer(end.get("value"), record.line) - integer(start.get("value"), record.line) != value:
        raise CycleSchemaError(record.line, "native counter delta mismatch")
    return value, ""


def cpu_rows(record: Record) -> Iterator[CsvRow]:
    row = base(record)
    if record.data.get("kind") in {"cpu", "cycle"}:
        valid = record.data.get("valid")
        if type(valid) is not bool:
            raise CycleSchemaError(record.line, "compact cycle valid must be boolean")
        value = integer(record.data.get("delta"), record.line) if valid else None
        reason = "" if valid else text(record.data, "reason", record.line)
        yield {**measured(row, "native_cycles", value), "unit": "cycle",
               "reason": reason, "source_field": "delta", "source": None}
        start_ns, end_ns = record.data.get("ns_start"), record.data.get("ns_end")
        if start_ns is not None or end_ns is not None:
            begin = integer(start_ns, record.line)
            end = integer(end_ns, record.line)
            wall = end - begin if end >= begin else None
            yield {**measured(row, "wall_ns", wall), "unit": "nanosecond",
                   "reason": "" if wall is not None else "clock_regression",
                   "source_field": "ns_end-ns_start", "source": "steady_clock"}
        return
    host_raw = record.data.get("host_timing")
    host = _host(host_raw, record.line, same_thread=record.data.get("record_type") != "TIMELINE") if host_raw is not None else None
    wall = measured(row, "wall_ns", host.duration_ns if host else None)
    yield {**wall, "unit": "nanosecond", "source_field": "host_timing.duration_ns",
           "source": scalar(_mapping(host_raw, record.line).get("clock"), record.line) if host_raw is not None else None}
    thread_raw = record.data.get("thread_cpu_timing")
    thread = _cpu(thread_raw, record.line) if thread_raw is not None else None
    if thread is not None and (host is None or host.start_tid != host.end_tid):
        raise CycleSchemaError(record.line, "thread CPU timing requires valid same-thread host endpoints")
    yield {**measured(row, "thread_cpu_ns", thread), "unit": "nanosecond",
           "source_field": "thread_cpu_timing.duration_ns",
           "source": scalar(_mapping(thread_raw, record.line).get("clock"), record.line) if thread_raw is not None else None}
    cycles, reason = native_cycles(record) if "native_cycles" in record.data else (None, "native_counter_not_collected")
    if cycles is not None and (host is None or host.start_tid != host.end_tid):
        raise CycleSchemaError(record.line, "native cycles require valid same-thread host endpoints")
    native_start = (_mapping(record.data["native_cycles"], record.line).get("start")
                    if "native_cycles" in record.data else None)
    yield {**measured(row, "native_cycles", cycles), "unit": "cycle", "reason": reason,
           "source_field": "native_cycles.delta",
           "source": scalar(_mapping(native_start, record.line).get("source"), record.line) if native_start is not None else None}


def scalar_rows(record: Record, payload: Mapping[str, JsonValue], prefix: str = "") -> Iterator[CsvRow]:
    row = base(record)
    for key, value in payload.items():
        if (not prefix and key in META) or key.endswith("_reason"):
            continue
        field = prefix + key
        if isinstance(value, dict):
            yield from scalar_rows(record, value, field + ".")
            continue
        if isinstance(value, list):
            yield {**measured(row, field, json.dumps(value, separators=(",", ":"))), "unit": "setting"}
            continue
        raw = scalar(value, record.line)
        cycle_unit = ("cycle" if key == "native_cycles" or prefix == "cpu_workers." else
                      "rtl_cycle" if "device.counters." in prefix or key.startswith("rtl_") else
                      str(record.data.get("unit", "cycle")))
        unit = (cycle_unit if key.endswith("cycles") else
                "nanosecond" if key.endswith("_ns") else "byte" if key.endswith("_bytes") else
                "bit" if key.endswith("_bits") else "hertz" if key.endswith("_hz") else
                "count" if type(raw) is int else "setting")
        reason_key = "thread_cpu_reason" if prefix == "cpu_workers." and key == "thread_cpu_ns" else key + "_reason"
        reason = payload.get(reason_key, "not_collected" if raw is None else "")
        validity = payload.get(key + "_valid")
        if validity is not None and (type(validity) is not bool or validity != (raw is not None)):
            raise CycleSchemaError(record.line, f"{field} contradicts its validity flag")
        if unit in {"cycle", "rtl_cycle", "tick", "nanosecond", "byte", "bit", "hertz"} and raw is not None:
            integer(raw, record.line)
        invalid = unit != "setting" and (payload.get("valid") is False or record.data.get("valid") is False)
        raw_ws = record.data.get("record_type") == "WS_LOOP_TELEMETRY"
        yield {**measured(row, field, None if invalid and not raw_ws else raw), "unit": unit,
               "valid": raw is not None and not invalid,
               "reason": str(reason or (record.data.get("reason", "invalid_record") if invalid else ""))}


def rows(record: Record) -> Iterator[tuple[str, CsvRow]]:
    data, row = record.data, base(record)
    kind = data.get("record_type")
    if kind in {"CPU_INTERVAL", "CYCLE_INTERVAL", "TIMELINE"}:
        for value in cpu_rows(record):
            yield "cpu-stages", value
    if kind in {"CPU_WORK_SUMMARY", "EXSIA_RUN_SUMMARY", "PIPELINE_STRIPE_SUMMARY"}:
        for value in scalar_rows(record, data):
            yield "cpu-stages", {**value, "scope": "aggregate_nonadditive"}
    if kind == "RESIDUAL_HOST_PROFILE":
        profile = _profile(data, record.line)
        raw_tiles = {integer(tile.get("node_id"), record.line): tile for tile in _objects(data.get("tiles"), record.line)}
        for value in scalar_rows(record, _mapping(data.get("workload"), record.line), "workload."):
            yield "workload", value
        for tile in profile.tiles:
            for stage_name, stage in tile.stages.items():
                raw_stage = _mapping(_mapping(raw_tiles[tile.node_id].get("stages"), record.line)[stage_name], record.line)
                for metric, value in (("calls", stage.calls), ("wall_ns", stage.wall_ns),
                                      ("thread_cpu_ns", stage.thread_cpu_ns), ("cycles", stage.cycles)):
                    yield "cpu-stages", {**measured(row, stage_name + "." + metric, value),
                        "worker_id": tile.worker_id, "node_id": tile.node_id, "scope": "tile_stage_sum_nonadditive",
                        "source_field": "tiles[].stages." + stage_name + "." + metric,
                        "unit": "nanosecond" if metric.endswith("_ns") else "cycle" if metric == "cycles" else "count",
                        "reason": str(raw_stage.get(metric + "_reason") or "not_collected") if value is None else ""}
        for worker in profile.workers:
            for metric, value in (("wall_ns", worker.barrier.host.duration_ns if worker.barrier else None),
                                  ("thread_cpu_ns", worker.barrier.cpu_ns if worker.barrier else None)):
                yield "cpu-stages", {**measured(row, "worker_barrier." + metric, value),
                    "worker_id": worker.worker_id, "unit": "nanosecond", "scope": "worker_barrier_nonadditive",
                    "source_field": "workers[].barrier_" + ("host_timing.duration_ns" if metric == "wall_ns" else "thread_cpu_timing.duration_ns")}
    if kind == "RESOURCE_SAMPLE" and data.get("kind") == "npu":
        collected = data.get("collected") is True
        yield "npu-stages", {**measured(row, text(data, "metric", record.line), scalar(data.get("cycles"), record.line)),
            "valid": collected, "reason": "" if collected else text(data, "reason", record.line),
            "source_field": "cycles", "unit": "cycle", "scope": "canonical_resource",
            "coverage": "verified_resource_sequence"}
    if kind in NPU_TYPES:
        for value in scalar_rows(record, data):
            host_counter = kind == "WS_LOOP_TELEMETRY" and value["metric"].startswith("containing_interval_")
            yield "cpu-stages" if host_counter else "npu-stages", {**value, "scope": "diagnostic_nonadditive",
                "source": scalar(data.get("containing_interval_source"), record.line) if host_counter else value["source"],
                "domain": "cpu_counter" if host_counter else value["domain"]}
    if kind in WORKLOAD_TYPES or (kind == "INFERENCE_EVENT" and data.get("event") == "configuration"):
        for value in scalar_rows(record, data):
            yield "cpu-stages" if value["metric"].startswith("host_stages.") else "workload", ({**value, "scope": "configuration", "included": True,
                                 "coverage": "recorded_configuration"}
                                if "CONFIGURATION" in str(kind) or data.get("event") == "configuration" else value)
    if kind == "STAGE":
        field = text(data, "metric", record.line)
        value = scalar(data.get("value"), record.line)
        yield "workload" if data.get("value_units") == "count" else "cpu-stages", {
            **measured(row, field, value), "source_field": "value", "unit": str(data.get("unit", "unknown")),
            "scope": "stage_sum_nonadditive", "reason": str(data.get("cycle_status", ""))}
    if kind == "INFERENCE_EVENT" and data.get("event") in {"request_start", "token_ready", "request_end"}:
        yield "request-latency", {**measured(row, str(data["event"]) + "_ns", integer(data.get("timestamp_ns"), record.line)),
            "unit": "nanosecond", "source_field": "timestamp_ns", "scope": "observed_event_boundary",
            "coverage": "verified_event_sequence"}


def summary_rows(record: Record) -> Iterator[tuple[str, CsvRow]]:
    row: CsvRow = {**base(record), "included": True, "coverage": "canonical_summary", "scope": "session_summary"}
    for metric in ("request_elapsed_ns", "ttft_ns", "tpot_ns", "ttft_samples", "tpot_gaps", "tokens"):
        value = scalar(record.data.get(metric), record.line)
        yield "request-latency", {**measured(row, metric, value), "reason": str(record.data.get(metric + "_reason") or ""),
            "unit": "nanosecond" if metric.endswith("_ns") else "count"}
    phases = record.data.get("phases")
    if not isinstance(phases, list):
        raise CycleSchemaError(record.line, "summary phases must be an array")
    for raw in phases:
        phase = _mapping(raw, record.line)
        for metric in ("elapsed_ns", "cpu_cycles", "thread_cpu_ns", "cpu_work_wall_ns", "cpu_work_wall_measured_ns"):
            value = scalar(phase.get(metric), record.line)
            yield "cpu-stages", {**measured(row, metric, value), "phase": scalar(phase.get("phase"), record.line),
                "source_field": "phases[]." + metric, "scope": "phase_summary",
                "reason": str(phase.get(metric + "_reason") or ""),
                "unit": "cycle" if metric == "cpu_cycles" else "nanosecond"}
        npu = phase.get("npu")
        if not isinstance(npu, list):
            raise CycleSchemaError(record.line, "summary npu must be an array")
        for raw_device in npu:
            device = _mapping(raw_device, record.line)
            for metric in ("cycles", "time_ns"):
                yield "npu-stages", {**measured(row, text(device, "metric", record.line) + "." + metric,
                    scalar(device.get(metric), record.line)), "phase": scalar(phase.get("phase"), record.line),
                    "backend": scalar(device.get("backend"), record.line), "domain": scalar(device.get("domain"), record.line),
                    "unit": "cycle" if metric == "cycles" else "nanosecond", "scope": "phase_summary",
                    "source_field": "phases[].npu[]." + metric,
                    "reason": str(device.get("cycles_reason" if metric == "cycles" else "time_reason") or "")}
