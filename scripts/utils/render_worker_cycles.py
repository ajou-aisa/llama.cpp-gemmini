#!/usr/bin/env python3
"""Render worker-work cycles for one explicit leaf operation.

Run with: python3 scripts/utils/render_worker_cycles.py INPUT.jsonl OUTPUT_DIR --op OP
Requires Python 3.9+ and the standard library only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

try:
    from .cycle_metrics import (CycleMetricsError, format_number, worker_metrics, operation_definition,
                                cpu_measurements, measurement_cells, MEASUREMENT_HEADERS)
    from .cycle_schema import CycleSchemaError
    from .table_output import BarValue, export_csv, render_markdown, render_svg
except ImportError:
    from cycle_metrics import (CycleMetricsError, format_number, worker_metrics, operation_definition,
                              cpu_measurements, measurement_cells, MEASUREMENT_HEADERS)
    from cycle_schema import CycleSchemaError
    from table_output import BarValue, export_csv, render_markdown, render_svg


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Render worker cycles for an explicit leaf operation")
    parser.add_argument("input", type=Path, metavar="JSONL")
    parser.add_argument("output", type=Path, metavar="OUTPUT_DIR")
    parser.add_argument("--op", required=True, help="identity-complete leaf operation")
    arguments = parser.parse_args(argv)
    try:
        targets = tuple(arguments.output / name for name in (
            "worker-cycles.csv", "worker-cycles.md", "worker-cycles.svg", "worker-measurements.csv",
        ))
        collisions = tuple(target for target in targets if target.exists())
        if collisions:
            raise CycleMetricsError(f"output target already exists: {collisions[0]}")
        metrics = worker_metrics(arguments.input, arguments.op)
        arguments.output.mkdir(parents=True, exist_ok=True)
        headers = (
            "source", "unit", "operation", "worker_id", "interval_count",
            "worker_work_cycles", "median_interval_cycles", "p95_interval_cycles",
            "valid_count", "valid_value_sum", "status", "reason", "sum_semantics",
        )
        rows = tuple((
            metric.source, metric.unit, metric.operation, str(metric.worker_id),
            str(metric.summary.count), "" if metric.summary.total is None else str(metric.summary.total),
            format_number(metric.summary.median), "" if metric.summary.p95 is None else str(metric.summary.p95),
            str(metric.summary.valid_count), str(metric.summary.valid_total), metric.summary.status, metric.summary.reason,
            "per_worker_label_not_elapsed_cycles",
        ) for metric in metrics)
        export_csv(targets[0], headers, rows)
        targets[1].write_text(
            render_markdown(headers, rows), encoding="utf-8",
        )
        domains = {(metric.source, metric.unit) for metric in metrics}
        targets[2].write_text(
            render_svg(f"Worker-work cycles: {arguments.op}", tuple(
                BarValue(str(metric.worker_id), float(metric.summary.total), metric.unit)
                for metric in metrics if metric.summary.total is not None and len(domains) == 1
            )),
            encoding="utf-8",
        )
        export_csv(targets[3], MEASUREMENT_HEADERS,
                   (measurement_cells(arguments.input, item) for item in cpu_measurements(arguments.input)
                    if item.operation == arguments.op or operation_definition(item.operation).canonical_id == arguments.op))
    except (CycleMetricsError, CycleSchemaError, OSError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
