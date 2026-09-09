#!/usr/bin/env python3
"""Safe streaming metric primitives for Gemmini cycle experiment tables."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

try:
    from .cycle_schema import CycleRecord, RecordType, parse_cycle_jsonl, rmd_value_status
except ImportError:
    from cycle_schema import CycleRecord, RecordType, parse_cycle_jsonl, rmd_value_status


class CycleMetricsError(Exception):
    """A metric input cannot be interpreted without unsafe arithmetic."""


@dataclass(frozen=True)
class MetricSummary:
    count: int
    total: Optional[int]
    median: Optional[Fraction]
    p95: Optional[int]
    valid_count: int = 0
    valid_total: int = 0
    status: str = "complete"
    reason: str = ""


@dataclass(frozen=True)
class OperationMetric:
    source: str
    unit: str
    operation: str
    summary: MetricSummary


@dataclass(frozen=True)
class WorkerMetric:
    source: str
    unit: str
    operation: str
    worker_id: int
    summary: MetricSummary


@dataclass(frozen=True)
class E2EMetrics:
    caller_source: str
    caller_unit: str
    caller: MetricSummary
    pipeline_source: str
    pipeline_unit: str
    pipeline: MetricSummary
    caller_status: str = "unknown"
    caller_reason: str = "legacy_validity_unknown"
    trusted_caller_cycles: Optional[int] = None


@dataclass(frozen=True, order=True)
class InvocationKey:
    layer: str
    run_id: int


@dataclass(frozen=True)
class OperationDefinition:
    canonical_id: str
    name_ko: str
    scope: str
    parent: str = ""
    includes: str = ""


# Aliases describe existing boundaries; raw operations remain separate aggregation keys.
DEFINITIONS = {
    "exsia.local_group": OperationDefinition("activation_local_quantization", "블록별 활성값 분석·양자화", "worker_inclusive", "exsia.local"),
    "exsia.local": OperationDefinition("activation_local_quantization_total", "활성값 분석·양자화 전체", "mode_dependent_inclusive", includes="exsia.local_group"),
    "exsia.mask_assembly": OperationDefinition("outlier_mask_assembly", "이상값 위치 마스크 조립", "cpu_stage"),
    "exsia.exponent_reduction": OperationDefinition("stripe_exponent_selection", "스트라이프 지수 후보 축약", "cpu_stage", includes="block_top_two_reduction;final_selection_in_folding"),
    "exsia.folding": OperationDefinition("activation_stripe_preparation", "활성값 재양자화·잔차 자료 준비", "inclusive", includes="residual_event_capture;residual_event_list_build;residual_packet_build"),
    "exsia.stripe_ready_handoff": OperationDefinition("ready_stripe_submit", "준비된 스트라이프 제출", "caller_inclusive", includes="capacity_wait"),
    "rmd_direct_finish_cycles": OperationDefinition("residual_event_list_build", "잔차 이벤트 목록 생성·정렬", "builder_inclusive", "activation_stripe_preparation", "allocation;copy;sort;validation"),
    "rmd_packet_finish_cycles": OperationDefinition("residual_packet_build", "가속기용 잔차 패킷 생성", "builder_inclusive", "activation_stripe_preparation", "descriptor;compact_index;padding;digits;validation"),
    "rmd_direct_j_tile_interval": OperationDefinition("residual_cpu_output_tile", "CPU 잔차 보정의 출력 타일 계산", "worker_leaf", "residual_backend_host_call"),
    "rmd_packet_compose_cycles": OperationDefinition("residual_result_reconstruction", "잔차 부분합 복원", "cpu_stage", includes="validation;allocation;radix_reconstruction"),
    "rmd_merge_cycles": OperationDefinition("output_correction_apply", "출력 스케일 적용·잔차 보정", "cpu_stage", includes="scale;validation;output_add_store"),
    "matmul_output_commit_cycles": OperationDefinition("output_buffer_copy", "최종 출력 버퍼 복사", "cpu_stage", "matmul_output_validation_and_publish"),
    "dense_backend_host_call": OperationDefinition("dense_backend_host_call", "기본 행렬곱 호출의 CPU 비용", "host_call_inclusive", includes="backend_call;route_specific_preparation"),
    "residual_backend_host_call": OperationDefinition("residual_backend_host_call", "잔차 보정 호출의 CPU 비용", "host_call_inclusive", includes="residual_cpu_output_tile;backend_call"),
    "stripe_input_capture": OperationDefinition("stripe_input_capture", "스트라이프 입력 정보 확보", "metadata_and_handle_capture"),
    "stripe_job_preparation": OperationDefinition("stripe_job_preparation", "스트라이프 실행 작업 준비", "consumer_cpu_stage"),
    "telemetry_stats_compute": OperationDefinition("telemetry_stats_compute", "계측용 통계 계산", "diagnostic_cpu_stage", "legacy_stripe_output_postprocess"),
    "telemetry_hash_compute": OperationDefinition("telemetry_hash_compute", "검증용 해시 계산", "opt_in_diagnostic_cpu_stage", "legacy_stripe_output_postprocess"),
    "stripe_completion_bookkeeping": OperationDefinition("stripe_completion_bookkeeping", "완료 행 수 갱신·슬롯 반환", "cpu_stage", includes="row_accounting;slot_release;lifecycle_notification"),
    "collector_capacity_release": OperationDefinition("collector_capacity_release", "collector 제출 용량 반환", "caller_cpu_control"),
    "pipeline_drain_and_join": OperationDefinition("pipeline_drain_and_join", "남은 작업 대기·작업 스레드 종료의 CPU 비용", "caller_cpu_not_worker_sum"),
    "matmul_output_validation_and_publish": OperationDefinition("matmul_output_validation_and_publish", "전체 출력 검증·결과 전달", "inclusive", includes="finite_validation;output_buffer_copy"),
    "matmul_execution_finish": OperationDefinition("matmul_execution_finish", "행렬곱 완료 확인·결과 전달", "mode_dependent_inclusive", includes="pipeline_output_validation_and_publish"),
    "gemmini.prepare_args": OperationDefinition("legacy_argument_preparation", "입력 인자 준비〔기존 두 경계〕", "legacy_ambiguous"),
    "gemmini.output_preparation": OperationDefinition("output_buffer_preparation", "출력 버퍼·주소 준비", "cpu_stage"),
    "im2p.host_input_preparation": OperationDefinition("npu_input_preparation", "가속기 입력·실행 정보 준비의 CPU 비용", "host_call_inclusive"),
    "im2p.stripe_input_capture": OperationDefinition("stripe_input_capture", "스트라이프 입력 정보 확보", "metadata_and_handle_capture"),
    "im2p.stripe_submit_host_call": OperationDefinition("npu_command_submit", "스트라이프 제출의 CPU 비용", "host_call_inclusive"),
    "im2p.frontend_start_host_call": OperationDefinition("simulator_frontend_start", "시뮬레이터 실행 준비의 CPU 비용", "host_call_inclusive"),
    "im2p.fence_host_call": OperationDefinition("npu_completion_wait", "가속기 완료 대기의 CPU 비용", "caller_cpu_not_worker_sum"),
    "im2p.residual_metadata_preparation": OperationDefinition("residual_metadata_preparation", "잔차 실행 정보 준비", "cpu_stage"),
    "im2p.residual_backend_host_call": OperationDefinition("residual_backend_host_call", "CPU 잔차 보정 호출 비용", "host_call_inclusive", includes="residual_cpu_output_tile"),
    "im2p.residual_simulator_host_call": OperationDefinition("residual_simulator_host_call", "잔차 시뮬레이터 실행의 host CPU 비용", "simulator_host_call_inclusive"),
    "im2p.residual_simulator_setup_host_call": OperationDefinition("residual_simulator_setup", "잔차 시뮬레이터 생성의 CPU 비용", "simulator_host_call_inclusive"),
    "im2p.residual_result_reconstruction": OperationDefinition("residual_result_reconstruction", "잔차 부분합 복원", "cpu_stage", includes="validation;allocation;radix_reconstruction"),
    "im2p.output_correction_apply": OperationDefinition("output_correction_apply", "출력 스케일 적용·잔차 보정", "cpu_stage", includes="scale;validation;output_add_store"),
    "im2p.post_fence_validation": OperationDefinition("accelerator_completion_validation", "가속기 완료 정보 검증의 CPU 비용", "cpu_stage"),
    "im2p.output_authorize_host_call": OperationDefinition("output_commit_authorization", "최종 출력 전달 승인의 CPU 비용", "host_call_inclusive"),
    "im2p.output_buffer_copy": OperationDefinition("output_buffer_copy", "최종 출력 버퍼 복사", "cpu_stage"),
    "exsia.run_total": OperationDefinition("activation_preparation_total", "활성값 준비 전체", "inclusive", includes="exsia.stripe_total"),
    "exsia.stripe_total": OperationDefinition("activation_stripe_total", "스트라이프 활성값 준비 전체", "inclusive", includes="exsia.local;outlier_mask_assembly;stripe_exponent_selection;activation_stripe_preparation"),
}
for _op in ("rmd_cpu_direct_finalize_cycles", "rmd_packet_finalize_cycles", "dense_finalize_cycles"):
    DEFINITIONS[_op] = OperationDefinition("legacy_stripe_output_postprocess", "출력 보정·진단 집계〔기존 범위〕", "inclusive_excludes_slot_release", includes="output_correction_apply;telemetry_stats_compute;telemetry_hash_compute")
for _stage in range(4):
    for _metric in ("sum", "count", "max"):
        DEFINITIONS[f"exsia.stage_metric:local.p{_stage}.{_metric}"] = OperationDefinition(
            f"activation_local_p{_stage}", f"활성값 양자화 P{_stage} 단계",
            "worker_stage_aggregate", "activation_local_quantization")


def operation_definition(operation: str) -> OperationDefinition:
    if operation in DEFINITIONS:
        return DEFINITIONS[operation]
    for definition in DEFINITIONS.values():
        if operation == definition.canonical_id:
            return definition
    return OperationDefinition(operation, "", "unspecified")


@dataclass(frozen=True)
class CpuMeasurement:
    record: CycleRecord
    source: str
    unit: str
    operation: str
    value: Optional[int]
    status: str
    reason: str
    metric: str = ""


def cpu_measurements(path: Path) -> Iterator[CpuMeasurement]:
    """Read measured CPU records without guessing source or repairing identity."""
    for record in parse_cycle_jsonl(path):
        if record.record_type not in {RecordType.CYCLE_INTERVAL, RecordType.TIMELINE, RecordType.STAGE}:
            continue
        if not record.op:
            raise CycleMetricsError(f"line {record.line_number}: operation identity is required")
        raw = json.loads(record.canonical_json)
        metric = ""
        if record.record_type == RecordType.CYCLE_INTERVAL:
            value, unit = record.delta, record.unit
            status = "complete" if record.valid else (record.reason if record.reason in {"not_collected", "not_applicable", "external_completion"} else "invalid")
        else:
            metric = raw.get("metric", "")
            value = raw["value" if record.record_type == RecordType.STAGE else "elapsed"]
            unit = raw["value_units" if record.record_type == RecordType.STAGE else "units"]
            status = raw["cycle_status"]
        if type(value) is not int or status != "complete":
            value = None
        reason = record.reason or raw.get("sample_reason", "")
        if not reason and status != "complete":
            reason = status
        # Legacy ExSIA omitted source; units alone do not establish PMU ownership.
        yield CpuMeasurement(record, record.source or "", {"cycles": "cycle", "ticks": "tick"}.get(unit, unit),
                             record.op + (":" + metric if metric else ""), value, status, reason, metric)


def measurement_summary(records: Sequence[Tuple[Optional[int], str, str]]) -> MetricSummary:
    values = [value for value, _, _ in records if value is not None]
    complete = len(values) == len(records)
    stats = summarize(values) if values else None
    statuses = {status for _, status, _ in records}
    status = "complete" if complete else ("partial" if values else next(iter(statuses)) if len(statuses) == 1 else "incomplete")
    return MetricSummary(len(records), stats.total if complete and stats else None,
                         stats.median if complete and stats else None, stats.p95 if complete and stats else None,
                         len(values), sum(values), status, ";".join(sorted({reason for _, _, reason in records if reason and reason != "none"})))


MEASUREMENT_HEADERS = ("input_file", "line_number", "record_type", "raw_op", "canonical_id", "name_ko", "scope", "parent", "includes",
                       "source", "unit", "raw_unit", "run_id", "layer", "stripe_id", "worker_id", "node_id", "tile_index", "slot",
                       "metric", "value", "status", "reason", "sample_reason", "operation_success", "record_json")


def measurement_cells(path: Path, measurement: CpuMeasurement) -> Tuple[str, ...]:
    record = measurement.record
    raw = json.loads(record.canonical_json)
    definition = operation_definition(measurement.operation)
    tile = raw.get("tile_index", record.node_id if definition.canonical_id == "residual_cpu_output_tile" else None)
    cells = (str(path.resolve()), record.line_number, record.record_type.value, record.op, definition.canonical_id,
             definition.name_ko, definition.scope, definition.parent, definition.includes, measurement.source,
             measurement.unit, raw.get("unit", raw.get("value_units", raw.get("units"))), record.run_id, record.layer,
             record.stripe_id, record.worker_id, record.node_id, tile, record.slot, measurement.metric,
             measurement.value, measurement.status, measurement.reason, raw.get("sample_reason"),
             json.dumps(raw["operation_success"]) if "operation_success" in raw else None, record.canonical_json)
    return tuple("" if value is None else str(value) for value in cells)


def summarize(values: Sequence[int]) -> MetricSummary:
    """Compute total, median, and nearest-rank P95 for nonnegative integers."""
    if not values:
        raise CycleMetricsError("at least one metric value is required")
    if any(value < 0 for value in values):
        raise CycleMetricsError("metric values must be non-negative")
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = Fraction(ordered[middle])
    if len(ordered) % 2 == 0:
        median = Fraction(ordered[middle - 1] + ordered[middle], 2)
    rank = math.ceil(0.95 * len(ordered))
    return MetricSummary(len(ordered), sum(ordered), median, ordered[rank - 1], len(ordered), sum(ordered))


def format_number(value: Optional[Fraction]) -> str:
    """Format an exact integer or half-integer statistic; null remains blank."""
    if value is None:
        return ""
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator // value.denominator}.5"


def operation_metrics(path: Path) -> Tuple[OperationMetric, ...]:
    """Keep raw operations and domains separate; never sum parents with children."""
    grouped: Dict[Tuple[str, str, str], List[Tuple[Optional[int], str, str]]] = defaultdict(list)
    for measurement in cpu_measurements(path):
        if measurement.metric and not measurement.metric.endswith(".sum"):
            continue  # count/max remain in the per-record export, not cycle sums.
        grouped[(measurement.source, measurement.unit, measurement.operation)].append((measurement.value, measurement.status, measurement.reason))
    if not grouped:
        raise CycleMetricsError("CPU measurement records are required")
    return tuple(
        OperationMetric(source, unit, operation, measurement_summary(grouped[(source, unit, operation)]))
        for source, unit, operation in sorted(grouped)
    )


def worker_metrics(path: Path, operation: str) -> Tuple[WorkerMetric, ...]:
    """Aggregate one explicit identity-complete leaf operation by worker."""
    grouped: Dict[Tuple[str, str, str, int], List[Tuple[Optional[int], str, str]]] = defaultdict(list)
    selected = operation_definition(operation)
    if selected.scope not in {"worker_leaf", "worker_inclusive", "unspecified"}:
        raise CycleMetricsError(f"operation {operation!r} is not a worker measurement")
    for measurement in cpu_measurements(path):
        if measurement.operation != operation and operation_definition(measurement.operation).canonical_id != operation:
            continue
        worker_id = measurement.record.worker_id
        if worker_id is None:
            raise CycleMetricsError(f"operation {operation!r} is not an identity-complete leaf")
        grouped[(measurement.source, measurement.unit, measurement.operation, worker_id)].append((measurement.value, measurement.status, measurement.reason))
    if not grouped:
        raise CycleMetricsError(f"operation {operation!r} is unavailable")
    return tuple(
        WorkerMetric(source, unit, raw_op, worker_id, measurement_summary(grouped[(source, unit, raw_op, worker_id)]))
        for source, unit, raw_op, worker_id in sorted(grouped)
    )


def _invocation_key(record: CycleRecord) -> InvocationKey:
    if record.layer is None or record.run_id is None:
        raise CycleMetricsError(f"line {record.line_number}: invocation layer/run identity is required")
    return InvocationKey(record.layer, record.run_id)


def e2e_metrics(path: Path) -> E2EMetrics:
    """Summarize caller envelopes and separate steady-clock pipeline envelopes."""
    callers: Dict[InvocationKey, Tuple[Optional[int], str, str]] = {}
    endpoints: Dict[InvocationKey, Tuple[int, int]] = {}
    stripes = set()
    caller_domain: Optional[Tuple[str, str]] = None
    for record in parse_cycle_jsonl(path):
        if record.record_type == RecordType.RMD_BACKEND_TELEMETRY:
            if record.source is None or record.unit != "cycle" or record.op != "rmd.execute":
                raise CycleMetricsError(f"line {record.line_number}: invalid caller envelope domain or operation")
            if caller_domain is not None and caller_domain != (record.source, record.unit):
                raise CycleMetricsError("records contain mixed source/unit domains")
            caller_domain = (record.source, record.unit)
            key = _invocation_key(record)
            if key in callers:
                raise CycleMetricsError(f"line {record.line_number}: duplicate invocation identity {key!r}")
            callers[key] = rmd_value_status(json.loads(record.canonical_json), "invocation_total", record.line_number)
            continue
        if record.record_type != RecordType.PIPELINE_STRIPE_SUMMARY:
            continue
        if (record.source, record.unit, record.op) != ("steady_clock", "nanosecond", "matmul.pipeline"):
            raise CycleMetricsError(f"line {record.line_number}: invalid pipeline domain or operation")
        if record.valid is not True:
            raise CycleMetricsError(f"line {record.line_number}: invalid pipeline summary")
        key = _invocation_key(record)
        if record.stripe_id is None or record.slot is None:
            raise CycleMetricsError(f"line {record.line_number}: pipeline stripe/slot identity is required")
        stripe = (key, record.stripe_id)
        if stripe in stripes:
            raise CycleMetricsError(f"line {record.line_number}: duplicate pipeline stripe identity")
        stripes.add(stripe)
        raw = json.loads(record.canonical_json)
        start, end = raw["queue_start_ns"], raw["finalize_end_ns"]
        prior = endpoints.get(key)
        endpoints[key] = (start, end) if prior is None else (min(prior[0], start), max(prior[1], end))
    if not callers:
        raise CycleMetricsError("RMD_BACKEND_TELEMETRY records are required")
    if not endpoints:
        raise CycleMetricsError("PIPELINE_STRIPE_SUMMARY records are required")
    if set(callers) != set(endpoints):
        raise CycleMetricsError("caller and pipeline invocation identities do not match")
    if caller_domain is None:
        raise CycleMetricsError("E2E metric domains are unavailable")
    caller_source, caller_unit = caller_domain
    pipeline_values = tuple(endpoints[key][1] - endpoints[key][0] for key in sorted(endpoints))
    values = tuple(value for value, _, _ in callers.values() if value is not None)
    caller = summarize(values) if len(values) == len(callers) else MetricSummary(len(callers), None, None, None)
    statuses = {status for _, status, _ in callers.values()}
    status = next(iter(statuses)) if len(statuses) == 1 else "partial"
    reason = ";".join(sorted({reason for _, _, reason in callers.values() if reason}))
    return E2EMetrics(
        caller_source, caller_unit, caller,
        "steady_clock", "nanosecond", summarize(pipeline_values), status, reason,
        caller.total if status == "complete" else None,
    )
