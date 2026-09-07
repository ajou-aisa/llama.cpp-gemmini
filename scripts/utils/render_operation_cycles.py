#!/usr/bin/env python3
"""Render safe per-operation worker-work cycle tables.

Run with: python3 scripts/utils/render_operation_cycles.py INPUT.jsonl OUTPUT_DIR
Requires Python 3.9+ and the standard library only.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional, Sequence

try:
    from .cycle_metrics import (CycleMetricsError, format_number, operation_metrics, operation_definition,
                                cpu_measurements, measurement_cells, MEASUREMENT_HEADERS)
    from .cycle_schema import CycleSchemaError
    from .table_output import BarValue, TableRow, render_markdown, render_svg
except ImportError:
    from cycle_metrics import (CycleMetricsError, format_number, operation_metrics, operation_definition,
                              cpu_measurements, measurement_cells, MEASUREMENT_HEADERS)
    from cycle_schema import CycleSchemaError
    from table_output import BarValue, TableRow, render_markdown, render_svg


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Render operation worker-work cycle tables")
    parser.add_argument("input", type=Path, metavar="JSONL")
    parser.add_argument("output", type=Path, metavar="OUTPUT_DIR")
    arguments = parser.parse_args(argv)
    try:
        targets = tuple(arguments.output / name for name in (
            "operation-cycles.csv", "operation-cycles.md", "operation-cycles.svg", "cpu-measurements.csv",
        ))
        collisions = tuple(target for target in targets if target.exists())
        if collisions:
            raise CycleMetricsError(f"output target already exists: {collisions[0]}")
        metrics = operation_metrics(arguments.input)
        arguments.output.mkdir(parents=True, exist_ok=True)
        headers = (
            "source", "unit", "operation", "interval_count", "worker_work_cycles",
            "median_interval_cycles", "p95_interval_cycles", "valid_count", "valid_value_sum", "status", "reason",
            "canonical_id", "name_ko", "scope", "parent", "includes", "sum_semantics",
        )
        rows = tuple(TableRow((
            metric.source, metric.unit, metric.operation, str(metric.summary.count),
            "" if metric.summary.total is None else str(metric.summary.total), format_number(metric.summary.median),
            "" if metric.summary.p95 is None else str(metric.summary.p95), str(metric.summary.valid_count),
            str(metric.summary.valid_total), metric.summary.status, metric.summary.reason,
            operation_definition(metric.operation).canonical_id, operation_definition(metric.operation).name_ko,
            operation_definition(metric.operation).scope, operation_definition(metric.operation).parent,
            operation_definition(metric.operation).includes, "per_raw_operation_not_additive_across_operations",
        )) for metric in metrics)
        with targets[0].open(
            "w", encoding="utf-8", newline="",
        ) as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(headers)
            writer.writerows(row.cells for row in rows)
        targets[1].write_text(
            render_markdown(headers, rows), encoding="utf-8",
        )
        domains = {(metric.source, metric.unit) for metric in metrics}
        targets[2].write_text(
            render_svg("Recorded intervals by operation (not additive)", tuple(
                BarValue(metric.operation, float(metric.summary.total), metric.unit)
                for metric in metrics if metric.summary.total is not None and len(domains) == 1
            )),
            encoding="utf-8",
        )
        with targets[3].open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(MEASUREMENT_HEADERS)
            writer.writerows(measurement_cells(arguments.input, item) for item in cpu_measurements(arguments.input))
    except (CycleMetricsError, CycleSchemaError, OSError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
