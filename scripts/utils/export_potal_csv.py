#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
# How to run: python3 scripts/utils/export_potal_csv.py MAIN --summary-binary build/bin/llama-cycle-summary --output-dir CSV
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass, replace
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Final

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.cycle_schema import CycleSchemaError, JsonScalar
from scripts.utils.potal_records import Record, integer, open_capture, records
from scripts.utils.potal_rows import COLUMNS, CsvRow, base, measured, rows, summary_rows
from scripts.utils.residual_profile import _mapping

DATASETS: Final = ("cpu-stages", "npu-stages", "workload", "request-latency")
BLOCK_COUNTS: Final = ("processed_blocks", "eligible_blocks", "reused_blocks", "regenerated_blocks",
                       "forced_recomputed_blocks", "natural_regenerated_blocks")
DEFINITIONS: Final = {
    "ttft_ns": "Existing mean request_start to first token_ready; sampled token, not displayed token.",
    "tpot_ns": "Existing weighted mean consecutive token_ready gap; includes intervening application work.",
    "request_elapsed_ns": "Existing sum request_start to request_end across completed requests.",
    "wall_ns": "Host steady_clock interval; overlapping parent, worker and stage rows are nonadditive.",
    "thread_cpu_ns": "Thread CPU time, independently measured from host wall time and native cycles.",
    "native_cycles": "Per-thread native counter delta with source, ownership and generation validation.",
    "metrics.logical_dot_result_bytes": "Logical INT64 dot-result extent; not measured device transfer traffic.",
    "occupancy_counter_semantics": "WS occupancy fields retain raw modulo counter readings, including zero; valid=false forbids interpreting them as elapsed cycles.",
    "eligible_blocks": "All observed nonempty processed local blocks, including no-integer-outlier and same-scale bypasses; aliases processed_blocks.",
    "regeneration_rate": "regenerated_blocks/eligible_blocks as a fraction. Global rate divides summed counts, never averages stripe rates; zero denominator is unavailable.",
    "natural_regenerated_blocks": "regenerated_blocks-forced_recomputed_blocks; separates natural replay decisions from forced recompute ablation.",
    "required_planes": "Highest required canonical signed radix digit index plus one at this stripe's metrics.digit_bits precision; zero for observed empty residuals.",
    "max_abs_residual": "Maximum absolute observed signed INT32 residual, widened before absolute value; zero for an observed empty residual set.",
    "plane_overflow": "Flag required_planes > (32//digit_bits+1) for observed accepted packets, including CPU-direct input decomposition. Actual full INT32 contract, not a paper four-plane comparison. Failed/rejected input overflow counts remain unknown.",
}
CONTRACT: Final = {
    "identity": "Join executions only by execution_id; request_id and operation_id are explicit request/evaluation identities. token_step joins the producer's operation_start target token ordinal by those three IDs; absent legacy values remain blank. Zero run/stripe/slot/node/worker IDs are valid; blank means absent. matmul_invocation_id joins early stages to MATMUL_CONFIGURATION, never file order.",
    "aggregation": "Use phase_summary for canonical phase totals and session_summary for session latency. Raw, parent, tile and diagnostic scopes overlap: do not sum them or combine CPU and NPU clock domains. additive=false never promises a disjoint partition.",
    "included": "False marks records without an inference context, including warmup; exclude them from inference totals. Configuration records describe the capture regardless of measurement context.",
    "valid_reason": "valid describes this measurement. Blank value plus reason means unavailable/invalid; it is not zero. coverage describes collection completeness separately.",
    "coverage": "verified_cpu_interval_sequence checks counted raw CPU records per operation; it does not prove every CPU stage was instrumented. incomplete_no_record_count_contract means raw stage loss cannot be detected. Canonical resource/event sequences have independent count/order checks.",
    "boundary": "boundary is the producer's event marker. latency_boundary_status reports whether the canonical reader verified those markers; legacy logs remain unverified_legacy_log.",
    "empty_output": "All four datasets are created; an empty category has one unavailable availability row. Existing output directories are rejected; invalid input publishes no CSV directory.",
    "workload_summary": "observed_workload_summary sums included EXSIA_WORKLOAD stripes only, with no record-count completeness contract. Choose these counts or individual stripe counts, never add both. RMD derived metrics use observed_accepted_packets and retain digit precision in the same stripe's metrics.digit_bits row.",
}


class Arguments(argparse.Namespace):
    main: Path = Path()
    detail: Path | None = None
    summary_binary: Path = Path()
    output_dir: Path = Path()


@dataclass(frozen=True)
class Request:
    start: int
    first: int | None = None
    last: int | None = None
    tokens: int = 0
    operation_start: int = 0
    operation_end: int = 0
    token_clock_regression: bool = False


def workload_rows(record: Record, totals: dict[str, int], missing: set[str]) -> list[CsvRow]:
    data = record.data
    row: CsvRow = base(record)
    values: list[tuple[str, JsonScalar, str, str, str]] = []
    if data.get("record_type") == "EXSIA_WORKLOAD":
        counts = {name: integer(data[name], record.line) if data.get(name) is not None and data.get("valid") is True else None
                  for name in BLOCK_COUNTS[:1] + BLOCK_COUNTS[2:5]}
        counts["eligible_blocks"] = counts["processed_blocks"]
        regenerated, forced = counts["regenerated_blocks"], counts["forced_recomputed_blocks"]
        counts["natural_regenerated_blocks"] = regenerated - forced if regenerated is not None and forced is not None else None
        processed, reused = counts["processed_blocks"], counts["reused_blocks"]
        if ((processed is not None and reused is not None and regenerated is not None and processed != reused + regenerated) or
                (counts["natural_regenerated_blocks"] is not None and counts["natural_regenerated_blocks"] < 0)):
            raise CycleSchemaError(record.line, "inconsistent block decision counts")
        if record.context:
            for name, count in counts.items():
                if count is None: missing.add(name)
                else: totals[name] = totals.get(name, 0) + count
        for name, source in (("eligible_blocks", "processed_blocks"),
                             ("natural_regenerated_blocks", "regenerated_blocks-forced_recomputed_blocks")):
            values.append((name, counts[name], source, "count", "counter_not_collected" if counts[name] is None else ""))
        rate = str(Decimal(regenerated) / processed) if regenerated is not None and processed else None
        values.append(("regeneration_rate", rate, "regenerated_blocks/processed_blocks", "fraction",
                       "zero_eligible_blocks" if processed == 0 else "counter_not_collected" if rate is None else ""))
    elif data.get("record_type") == "RMD_STRIPE_TELEMETRY":
        metrics = _mapping(data["metrics"], record.line) if data.get("metrics") is not None else {}
        observed = data.get("valid") is True and data.get("operation_success") is True
        planes, bits, nnz = (metrics.get(name) for name in ("required_planes", "digit_bits", "residual_nnz"))
        minimum, maximum = metrics.get("residual_min"), metrics.get("residual_max")
        magnitude = (max(abs(minimum), abs(maximum)) if observed and type(minimum) is int and type(maximum) is int else
                     0 if observed and type(nnz) is int and nnz == 0 and type(planes) is int and planes == 0 else None)
        known = observed and magnitude is not None and type(planes) is int and type(bits) is int and bits in {4, 8, 16} and type(nnz) is int and nnz >= 0
        for name, value, source in (("required_planes", planes if observed else None, "metrics.required_planes"),
                ("max_abs_residual", magnitude, "max(abs(metrics.residual_min),abs(metrics.residual_max)); observed empty=0"),
                ("plane_overflow", int(integer(planes, record.line) > 32 // integer(bits, record.line) + 1) if known else None, "metrics.required_planes>(32//metrics.digit_bits+1)")):
            values.append((name, integer(value, record.line) if value is not None else None, source, "count",
                           "counter_not_collected" if value is None else ""))
        row = {**row, "scope": "observed_accepted_packets"}
    return [{**measured(row, name, value), "source_field": source, "unit": unit, "reason": reason}
            for name, value, source, unit, reason in values]


def request_rows(record: Record, pending: dict[int, Request]) -> list[CsvRow]:
    if record.data.get("record_type") != "INFERENCE_EVENT" or not record.context:
        return []
    request_id = integer(record.context.get("request_id"), record.line)
    event = record.data.get("event")
    timestamp = integer(record.data.get("timestamp_ns"), record.line)
    if event == "request_start":
        pending[request_id] = Request(timestamp)
    if event == "operation_start":
        pending[request_id] = replace(pending[request_id], operation_start=timestamp)
    if event == "operation_end":
        request = pending[request_id]
        pending[request_id] = replace(request, operation_end=max(request.operation_start, timestamp))
    if event == "token_ready":
        request = pending[request_id]
        pending[request_id] = replace(request, first=request.first if request.first is not None else timestamp,
            last=timestamp, tokens=request.tokens + 1, token_clock_regression=request.token_clock_regression or
            timestamp < max(request.start, request.operation_end, request.last or 0))
    if event != "request_end":
        return []
    request = pending.pop(request_id)
    row: CsvRow = {**base(record), "scope": "request_summary", "coverage": "verified_event_sequence"}
    ttft = None if request.first is None else request.first - request.start
    request_regression = timestamp < max(request.start, request.operation_end, request.last or 0)
    token_reason = "clock_regression" if request_regression or request.token_clock_regression else ""
    if token_reason:
        ttft = None
    tpot = None
    if request.tokens > 1 and request.first is not None and request.last is not None and not token_reason:
        with localcontext() as context:
            context.prec = 30
            tpot = str(Decimal(request.last - request.first) / Decimal(request.tokens - 1))
    return [{**measured(row, name, value), "unit": "count" if name == "tokens" else "nanosecond",
             "reason": reason, "source_field": source}
        for name, value, reason, source in (
            ("request_start_ns", request.start, "", "request_start.timestamp_ns"),
            ("first_token_ready_ns", request.first, "no_generated_tokens" if request.first is None else "", "first token_ready.timestamp_ns"),
            ("last_token_ready_ns", request.last, "no_generated_tokens" if request.last is None else "", "last token_ready.timestamp_ns"),
            ("request_elapsed_ns", None if request_regression else timestamp - request.start,
             "clock_regression" if request_regression else "", "request_end.timestamp_ns-request_start.timestamp_ns"),
            ("tokens", request.tokens, "", "count(token_ready)"),
            ("ttft_ns", ttft, token_reason or ("no_generated_tokens" if ttft is None else ""), "first_token_ready_ns-request_start_ns"),
            ("tpot_ns", tpot, token_reason or ("fewer_than_two_tokens_per_request" if tpot is None else ""),
             "(last_token_ready_ns-first_token_ready_ns)/(tokens-1)"))]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export measured PoTal stages and existing request latency to CSV.",
        epilog="Requires llama-cycle-summary from this checkout. CSV values are long form; never add raw, "
               "canonical, parent and diagnostic scopes together. Blank values are unavailable. "
               "Legacy raw-stage completeness cannot be established. Main/detail mirrors are deduplicated.")
    parser.add_argument("main", type=Path, metavar="MAIN", help="main cycle-log.jsonl from one completed execution")
    parser.add_argument("--detail", type=Path, help="optional exsia-cycle-detail.jsonl from that same execution")
    parser.add_argument("--summary-binary", required=True, type=Path, help="built llama-cycle-summary executable")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = Arguments()
    parser.parse_args(namespace=args)
    try:
        if args.output_dir.exists():
            raise FileExistsError(f"output already exists: {args.output_dir}")
        capture = open_capture(args.main, args.detail, args.summary_binary)
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        mappings: set[tuple[str, ...]] = set()
        requests: dict[int, Request] = {}
        token_steps: dict[tuple[str, JsonScalar, JsonScalar], int | None] = {}
        block_totals: dict[str, int] = {}
        missing_blocks: set[str] = set()
        cpu_routes: list[bool] = []
        counts: dict[str, int] = dict.fromkeys(DATASETS, 0)
        with tempfile.TemporaryDirectory(prefix="potal-csv-", dir=args.output_dir.parent) as temporary:
            staging = Path(temporary) / "export"
            staging.mkdir()
            with ExitStack() as stack:
                writers: dict[str, csv.DictWriter[str]] = {}
                for dataset in DATASETS:
                    stream = stack.enter_context((staging / (dataset + ".csv")).open("w", newline="", encoding="utf-8"))
                    writers[dataset] = csv.DictWriter(stream, fieldnames=COLUMNS)
                    writers[dataset].writeheader()

                def write(dataset: str, row: CsvRow) -> None:
                    row = {**row, "token_step": token_steps.get((row["execution_id"], row["request_id"], row["operation_id"])),
                        "latency_boundary_status": str(capture.summary.get("latency_boundary_status", "unverified_legacy_log"))}
                    if row["record_type"] == "CPU_INTERVAL" and row["included"] and capture.summary.get("cpu_interval_coverage") == "verified":
                        row = {**row, "coverage": "verified_cpu_interval_sequence"}
                    writers[dataset].writerow({key: str(value).lower() if type(value) is bool else value
                                              for key, value in row.items()})
                    counts[dataset] += 1
                    definition = ("Per-request metric at the recorded event boundaries; " + row["source_field"]
                        if row["scope"] == "request_summary" else DEFINITIONS.get(row["metric"],
                        "Recorded field; preserve scope, validity and coverage before aggregation."))
                    mappings.add((dataset, row["metric"], row["source_field"], row["unit"], row["scope"], definition))

                for record in records(capture):
                    if record.data.get("record_type") == "INFERENCE_EVENT" and record.data.get("event") in {"operation_start", "operation_end"}:
                        key = (record.execution, integer(record.context.get("request_id"), record.line),
                               integer(record.context.get("operation_id"), record.line))
                        step = integer(record.data["token_step"], record.line) if "token_step" in record.data else None
                        if record.data["event"] == "operation_start":
                            if key in token_steps:
                                raise CycleSchemaError(record.line, "duplicate operation token identity")
                            token_steps[key] = step
                        elif key not in token_steps or token_steps[key] != step:
                            raise CycleSchemaError(record.line, "conflicting operation token_step")
                    if record.data.get("record_type") == "MATMUL_CONFIGURATION" and record.context:
                        cpu_routes.append(record.data.get("dense_backend") == "cpu" and
                                          record.data.get("residual_backend") in ("disabled", "cpu_direct"))
                    for dataset, row in rows(record):
                        write(dataset, row)
                    for row in workload_rows(record, block_totals, missing_blocks):
                        write("workload", row)
                    for row in request_rows(record, requests):
                        write("request-latency", row)
                summary = Record(capture.summary, capture.main, 0, capture.execution, {})
                for dataset, row in summary_rows(summary):
                    write(dataset, row)
                eligible = block_totals.get("eligible_blocks") if "eligible_blocks" not in missing_blocks else None
                regenerated = block_totals.get("regenerated_blocks") if "regenerated_blocks" not in missing_blocks else None
                rate = str(Decimal(regenerated) / eligible) if regenerated is not None and eligible else None
                for name in (*BLOCK_COUNTS, "regeneration_rate"):
                    value = rate if name == "regeneration_rate" else block_totals.get(name) if name not in missing_blocks else None
                    source = {"eligible_blocks": "processed_blocks", "natural_regenerated_blocks": "regenerated_blocks-forced_recomputed_blocks"}.get(name, name)
                    write("workload", {**measured(base(summary), name, value), "included": True,
                        "scope": "observed_workload_summary", "coverage": "incomplete_no_record_count_contract",
                        "unit": "fraction" if name == "regeneration_rate" else "count",
                        "source_field": "sum(EXSIA_WORKLOAD.regenerated_blocks)/sum(EXSIA_WORKLOAD.processed_blocks)" if name == "regeneration_rate" else "sum(EXSIA_WORKLOAD." + source + ")",
                        "reason": "zero_eligible_blocks" if name == "regeneration_rate" and eligible == 0 else "counter_not_collected" if value is None else ""})
                for dataset in DATASETS:
                    if counts[dataset] == 0:
                        inapplicable = dataset == "npu-stages" and bool(cpu_routes) and all(cpu_routes)
                        write(dataset, {**measured(base(summary), "availability", None), "included": True,
                            "coverage": "inapplicable_observed_cpu_only_routes" if inapplicable else "incomplete_missing_records",
                            "reason": "cpu_only_configuration" if inapplicable else "no_matching_records", "scope": "availability"})
            with (staging / "field-map.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("dataset", "metric", "source_field", "unit", "scope", "definition"))
                writer.writerows(sorted(mappings))
                writer.writerows(("*", name, "", "", "export_contract", definition) for name, definition in CONTRACT.items())
            staging.rename(args.output_dir)
        print("Exported " + ", ".join(f"{name}={count}" for name, count in counts.items()))
        return 0
    except (OSError, UnicodeError, CycleSchemaError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
