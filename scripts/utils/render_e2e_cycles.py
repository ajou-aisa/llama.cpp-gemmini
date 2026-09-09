#!/usr/bin/env python3
"""Render separate caller-cycle and steady-clock pipeline envelope tables.

Run with: python3 scripts/utils/render_e2e_cycles.py INPUT.jsonl OUTPUT_DIR
Requires Python 3.9+ and the standard library only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

try:
    from .cycle_metrics import CycleMetricsError, e2e_metrics, format_number
    from .cycle_schema import CycleSchemaError, RecordType, parse_cycle_jsonl, rmd_value_status
    from .table_output import BarValue, export_csv, render_markdown, render_svg
except ImportError:
    from cycle_metrics import CycleMetricsError, e2e_metrics, format_number
    from cycle_schema import CycleSchemaError, RecordType, parse_cycle_jsonl, rmd_value_status
    from table_output import BarValue, export_csv, render_markdown, render_svg


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render caller invocation cycles and separate pipeline steady-clock nanoseconds",
    )
    parser.add_argument("input", type=Path, metavar="JSONL")
    parser.add_argument("output", type=Path, metavar="OUTPUT_DIR")
    arguments = parser.parse_args(argv)
    try:
        targets = tuple(arguments.output / name for name in (
            "caller-envelope-cycles.csv", "caller-envelope-cycles.md",
            "caller-envelope-cycles.svg", "pipeline-envelope-ns.csv",
            "pipeline-envelope-ns.md", "pipeline-envelope-ns.svg", "caller-measurements.csv",
        ))
        collisions = tuple(target for target in targets if target.exists())
        if collisions:
            raise CycleMetricsError(f"output target already exists: {collisions[0]}")
        metrics = e2e_metrics(arguments.input)
        arguments.output.mkdir(parents=True, exist_ok=True)
        caller_headers = (
            "source", "unit", "invocation_count", "caller_envelope_cycles",
            "median_invocation_cycles", "p95_invocation_cycles", "status", "reason", "trusted_caller_cycles",
        )
        caller_row = (
            metrics.caller_source, metrics.caller_unit, str(metrics.caller.count),
            "" if metrics.caller.total is None else str(metrics.caller.total), format_number(metrics.caller.median),
            "" if metrics.caller.p95 is None else str(metrics.caller.p95), metrics.caller_status, metrics.caller_reason,
            "" if metrics.trusted_caller_cycles is None else str(metrics.trusted_caller_cycles),
        )
        pipeline_headers = (
            "source", "unit", "invocation_count", "pipeline_envelope_ns",
            "median_invocation_ns", "p95_invocation_ns",
        )
        pipeline_row = (
            metrics.pipeline_source, metrics.pipeline_unit, str(metrics.pipeline.count),
            str(metrics.pipeline.total), format_number(metrics.pipeline.median), str(metrics.pipeline.p95),
        )
        outputs = (
            (caller_headers, caller_row, targets[0], targets[1]),
            (pipeline_headers, pipeline_row, targets[3], targets[4]),
        )
        for headers, row, csv_target, markdown_target in outputs:
            export_csv(csv_target, headers, (row,))
            markdown_target.write_text(render_markdown(headers, (row,)), encoding="utf-8")
        targets[2].write_text(
            render_svg("Caller invocation envelope cycles (explicitly complete only)", (
                (BarValue("all invocations", float(metrics.trusted_caller_cycles), metrics.caller_unit),)
                if metrics.trusted_caller_cycles is not None else ()
            )), encoding="utf-8",
        )
        with targets[6].open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("input_file", "line_number", "raw_op", "source", "unit", "run_id", "layer",
                             "invocation_total", "status", "reason", "timing_json", "record_json"))
            for record in parse_cycle_jsonl(arguments.input):
                if record.record_type != RecordType.RMD_BACKEND_TELEMETRY:
                    continue
                raw = json.loads(record.canonical_json)
                value, status, reason = rmd_value_status(raw, "invocation_total", record.line_number)
                writer.writerow((str(arguments.input.resolve()), record.line_number, record.op, record.source, record.unit,
                                 record.run_id, record.layer, value, status, reason,
                                 json.dumps(raw["timing"], sort_keys=True, separators=(",", ":")), record.canonical_json))
        targets[5].write_text(
            render_svg("Pipeline steady-clock envelope", (
                BarValue("all invocations", float(metrics.pipeline.total), "ns"),
            )), encoding="utf-8",
        )
    except (CycleMetricsError, CycleSchemaError, OSError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
