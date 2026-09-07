#!/usr/bin/env python3
"""Aggregate the fixed six-cell Gemmini cycle experiment matrix."""
# Project constraint: Python 3.9+ standard library only.
# Run: python3 scripts/utils/summarize_cycle_matrix.py --input RUNS.csv --output SUMMARY_DIR
from __future__ import annotations
import argparse
import csv
import io
import math
import sys
from pathlib import Path
from statistics import fmean, median
from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Set, Tuple
from table_output import BarValue, TableRow, render_markdown, render_svg

CELLS: Tuple[Tuple[str, int, str], ...] = (("q4-baseline", 4, "baseline"), ("q4-hp1", 4, "hp1"), ("q8-baseline", 8, "baseline"), ("q8-hp1", 8, "hp1"), ("q16-baseline", 16, "baseline"), ("q16-hp1", 16, "hp1"))
INPUT_FIELDS: Tuple[str, ...] = ("cell", "width", "variant", "repeat", "required", "status", "failure_reason", "cycle_value", "cycle_unit", "wall_value", "wall_unit", "frequency_mean_mhz", "frequency_weighted_mhz", "affinity_status", "affinity_reason")
SOURCE_FIELDS = ("cycle_source", "wall_source", "cycle_status")
RUN_FIELDS: Tuple[str, ...] = INPUT_FIELDS + ("frequency_reason",) + SOURCE_FIELDS
SUMMARY_FIELDS: Tuple[str, ...] = ("cell", "width", "variant", "status", "expected_count", "success_count", "failure_count", "cycle_unit", "cycle_mean", "cycle_median", "cycle_min", "cycle_max", "cycle_p95", "wall_unit", "wall_mean", "wall_median", "wall_min", "wall_max", "wall_p95", "frequency_mean_mean_mhz", "frequency_weighted_mean_mhz", "frequency_reason", "cycle_hp1_baseline_ratio", "wall_hp1_baseline_ratio") + SOURCE_FIELDS
FAILURE_FIELDS: Tuple[str, ...] = ("cell", "repeat", "required", "reason")
Stats = Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]

class SummaryError(Exception):
    """A typed per-repeat CSV or aggregate contract failure."""
    __slots__ = ("detail",)
    detail: str

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)

    def __str__(self) -> str:
        return self.detail

class Run(NamedTuple):
    cell: str
    width: int
    variant: str
    repeat: int
    required: bool
    status: str
    failure_reason: str
    cycle_value: Optional[float]
    cycle_unit: str
    wall_value: Optional[float]
    wall_unit: str
    frequency_mean_mhz: Optional[float]
    frequency_weighted_mhz: Optional[float]
    affinity_status: str
    affinity_reason: str
    cycle_source: str = ""
    wall_source: str = ""
    cycle_status: str = "unknown"

class Failure(NamedTuple):
    cell: str
    repeat: str
    required: bool
    reason: str


def _number(value: str, field: str) -> Optional[float]:
    if value == "":
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise SummaryError(field + " must be finite and non-negative")
    return number


def _parse_row(row: Mapping[str, str], line: int) -> Run:
    try:
        cell, width, variant = row["cell"], int(row["width"]), row["variant"]
        repeat = int(row["repeat"])
        required_text, status = row["required"], row["status"]
    except (KeyError, TypeError, ValueError) as error:
        raise SummaryError("line %d: malformed identity: %s" % (line, error)) from None
    if (cell, width, variant) not in CELLS or repeat < 1:
        raise SummaryError("line %d: unsupported matrix identity" % line)
    if required_text not in ("true", "false") or status not in ("success", "failure"):
        raise SummaryError("line %d: invalid required/status value" % line)
    try:
        run = Run(cell, width, variant, repeat, required_text == "true", status, row["failure_reason"], _number(row["cycle_value"], "cycle_value"), row["cycle_unit"], _number(row["wall_value"], "wall_value"), row["wall_unit"], _number(row["frequency_mean_mhz"], "frequency_mean_mhz"), _number(row["frequency_weighted_mhz"], "frequency_weighted_mhz"), row["affinity_status"], row["affinity_reason"], row.get("cycle_source", ""), row.get("wall_source", ""), row.get("cycle_status", "unknown"))
    except (KeyError, TypeError, ValueError) as error:
        raise SummaryError("line %d: malformed metrics: %s" % (line, error)) from None
    if status == "success" and (run.cycle_value is None or run.wall_value is None or not run.cycle_unit or not run.wall_unit):
        raise SummaryError("line %d: successful run lacks cycle/wall metrics" % line)
    if status == "failure" and not run.failure_reason:
        raise SummaryError("line %d: failed run lacks failure_reason" % line)
    if run.affinity_status == "conflict_free":
        if status == "success" and (run.frequency_mean_mhz is None or run.frequency_weighted_mhz is None):
            raise SummaryError("line %d: conflict-free run lacks frequency metrics" % line)
        return run
    reason = run.affinity_reason or "affinity mapping unavailable"
    return run._replace(frequency_mean_mhz=None, frequency_weighted_mhz=None, affinity_reason=reason)


def _load(path: Path) -> Tuple[Tuple[Run, ...], Tuple[Failure, ...]]:
    runs: List[Run] = []
    failures: List[Failure] = []
    identities: Set[Tuple[str, int]] = set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) not in (INPUT_FIELDS, INPUT_FIELDS + SOURCE_FIELDS):
            raise SummaryError("CSV header does not match the per-repeat contract")
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise SummaryError("line %d: malformed CSV row" % line)
            run = _parse_row(row, line)
            identity = (run.cell, run.repeat)
            if identity in identities:
                failures.append(Failure(run.cell, str(run.repeat), run.required, "duplicate run identity"))
                continue
            identities.add(identity)
            runs.append(run)
    return tuple(runs), tuple(failures)


def _fmt(value: Optional[float]) -> str:
    return "" if value is None else format(value, ".12g")


def _stats(values: Iterable[float]) -> Stats:
    ordered = sorted(values)
    if not ordered:
        return (None, None, None, None, None)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    return (fmean(ordered), median(ordered), ordered[0], ordered[-1], p95)


def _run_row(run: Run) -> Tuple[str, ...]:
    frequency_reason = "" if run.affinity_status == "conflict_free" else run.affinity_reason
    return (run.cell, str(run.width), run.variant, str(run.repeat), str(run.required).lower(), run.status, run.failure_reason, _fmt(run.cycle_value), run.cycle_unit, _fmt(run.wall_value), run.wall_unit, _fmt(run.frequency_mean_mhz), _fmt(run.frequency_weighted_mhz), run.affinity_status, run.affinity_reason, frequency_reason, run.cycle_source, run.wall_source, run.cycle_status)


def _csv_text(fields: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(fields)
    writer.writerows(rows)
    return stream.getvalue()


def _complete(runs: Sequence[Run], repeats: int) -> Tuple[Tuple[Run, ...], Tuple[Failure, ...]]:
    by_identity: Dict[Tuple[str, int], Run] = {(run.cell, run.repeat): run for run in runs}
    completed: List[Run] = []
    missing: List[Failure] = []
    for cell, width, variant in CELLS:
        for repeat in range(1, repeats + 1):
            run = by_identity.get((cell, repeat))
            if run is None:
                reason = "missing required run"
                run = Run(cell, width, variant, repeat, True, "failure", reason, None, "", None, "", None, None, "unavailable", reason)
            completed.append(run)
    extras = [run for run in runs if run.repeat > repeats]
    for run in extras:
        missing.append(Failure(run.cell, str(run.repeat), run.required, "repeat exceeds expected count"))
    return tuple(completed), tuple(missing)


def _summaries(runs: Sequence[Run], failures: List[Failure]) -> Tuple[Tuple[Tuple[str, ...], ...], Tuple[BarValue, ...]]:
    partial: List[List[str]] = []
    means: Dict[Tuple[int, str, str], float] = {}
    incomplete_cells = {failure.cell for failure in failures if failure.required}
    invalid_cycle: Set[int] = set()
    invalid_wall: Set[int] = set()
    for width in (4, 8, 16):
        successful = [run for run in runs if run.width == width and run.status == "success"]
        cycle_sources = {run.cycle_source for run in successful}
        wall_sources = {run.wall_source for run in successful}
        if len(cycle_sources) > 1:
            invalid_cycle.add(width)
            failures.append(Failure("width-%d" % width, "", True, "cycle source mismatch"))
        if len(wall_sources) > 1:
            invalid_wall.add(width)
            failures.append(Failure("width-%d" % width, "", True, "wall source mismatch"))
        if any(run.cycle_source and run.cycle_status != "complete" for run in successful):
            invalid_cycle.add(width)
            failures.append(Failure("width-%d" % width, "", True, "CPU cycle validity is not complete"))
        cycle_units = {run.cycle_unit for run in successful}
        mixed_cycle_cells = sorted({run.cell for run in successful if len({item.cycle_unit for item in successful if item.cell == run.cell}) > 1})
        if len(cycle_units) > 1:
            invalid_cycle.add(width)
            if mixed_cycle_cells:
                failures.extend(Failure(cell, "", True, "mixed cycle units") for cell in mixed_cycle_cells)
            else:
                failures.append(Failure("width-%d" % width, "", True, "cycle unit mismatch for width %d: %s" % (width, ",".join(sorted(cycle_units)))))
        wall_units = {run.wall_unit for run in successful}
        mixed_wall_cells = sorted({run.cell for run in successful if len({item.wall_unit for item in successful if item.cell == run.cell}) > 1})
        if len(wall_units) > 1:
            invalid_wall.add(width)
            if mixed_wall_cells:
                failures.extend(Failure(cell, "", True, "mixed wall units") for cell in mixed_wall_cells)
            else:
                failures.append(Failure("width-%d" % width, "", True, "wall unit mismatch for width %d: %s" % (width, ",".join(sorted(wall_units)))))
    for cell, width, variant in CELLS:
        cell_runs = [run for run in runs if run.cell == cell]
        successful = [run for run in cell_runs if run.status == "success"]
        if len(successful) != len(cell_runs):
            incomplete_cells.add(cell)
        cycle_units = set() if width in invalid_cycle else {run.cycle_unit for run in successful}
        wall_units = set() if width in invalid_wall else {run.wall_unit for run in successful}
        cycle_stats: Stats = (None, None, None, None, None) if width in invalid_cycle or cell in incomplete_cells else _stats(run.cycle_value for run in successful if run.cycle_value is not None)
        wall_stats: Stats = (None, None, None, None, None) if width in invalid_wall else _stats(run.wall_value for run in successful if run.wall_value is not None)
        affinity_reasons = sorted({run.affinity_reason for run in successful if run.affinity_status != "conflict_free"})
        frequency_ready = bool(successful) and not affinity_reasons
        frequency_mean = _fmt(fmean(run.frequency_mean_mhz for run in successful if run.frequency_mean_mhz is not None)) if frequency_ready else ""
        frequency_weighted = _fmt(fmean(run.frequency_weighted_mhz for run in successful if run.frequency_weighted_mhz is not None)) if frequency_ready else ""
        if cycle_stats[0] is not None: means[(width, variant, "cycle")] = cycle_stats[0]
        if wall_stats[0] is not None: means[(width, variant, "wall")] = wall_stats[0]
        status = "success" if len(successful) == len(cell_runs) else ("partial" if successful else "failure")
        partial.append([cell, str(width), variant, status, str(len(cell_runs)), str(len(successful)), str(len(cell_runs) - len(successful)), next(iter(cycle_units)) if len(cycle_units) == 1 else "", *(_fmt(value) for value in cycle_stats), next(iter(wall_units)) if len(wall_units) == 1 else "", *(_fmt(value) for value in wall_stats), frequency_mean, frequency_weighted, "; ".join(affinity_reasons), "", ""])
    for row in partial:
        if row[2] == "hp1":
            width = int(row[1])
            for offset, metric in ((-2, "cycle"), (-1, "wall")):
                baseline, hp1 = means.get((width, "baseline", metric)), means.get((width, "hp1", metric))
                trusted = metric != "cycle" or all(run.status == "success" and run.cycle_source and run.cycle_status == "complete" and run.cell not in incomplete_cells for run in runs if run.width == width)
                row[offset] = _fmt(hp1 / baseline) if baseline and hp1 is not None and trusted else ""
        successful = [run for run in runs if run.cell == row[0] and run.status == "success"]
        sources = {run.cycle_source for run in successful}
        wall_sources = {run.wall_source for run in successful}
        cycle_status = "incomplete" if row[0] in incomplete_cells else ("complete" if successful and all(run.cycle_source and run.cycle_status == "complete" for run in successful) and int(row[1]) not in invalid_cycle else "unknown")
        row.extend((next(iter(sources)) if len(sources) == 1 else "", next(iter(wall_sources)) if len(wall_sources) == 1 else "", cycle_status))
    rows = tuple(tuple(row) for row in partial)
    bars = tuple(BarValue(row[0], means[(int(row[1]), row[2], "cycle")], row[7]) for row in rows if (int(row[1]), row[2], "cycle") in means and row[-1] == "complete")
    return rows, bars


def summarize(source: Path, output: Path, repeats: int) -> bool:
    failures: List[Failure] = []
    try:
        loaded, load_failures = _load(source)
        failures.extend(load_failures)
    except (OSError, csv.Error, SummaryError) as error:
        loaded = ()
        failures.append(Failure("__input__", "", True, str(error)))
    runs, missing = _complete(loaded, repeats)
    failures.extend(missing)
    for run in runs:
        if run.status == "failure":
            failures.append(Failure(run.cell, str(run.repeat), run.required, run.failure_reason))
    summary_rows, bars = _summaries(runs, failures)
    bar_sources = {row[-3] for row in summary_rows if row[-1] == "complete"}
    if len(bar_sources) > 1:
        failures.append(Failure("__matrix__", "", True, "cycle bar source mismatch: " + ",".join(sorted(bar_sources))))
        bars = ()
    bar_units = {bar.unit for bar in bars}
    if len(bar_units) > 1:
        failures.append(Failure("__matrix__", "", True, "cycle bar unit mismatch: " + ",".join(sorted(bar_units))))
        bars = ()
    artifacts = (
        ("runs.csv", _csv_text(RUN_FIELDS, (_run_row(run) for run in runs))),
        ("matrix-summary.csv", _csv_text(SUMMARY_FIELDS, summary_rows)),
        ("failures.csv", _csv_text(FAILURE_FIELDS, ((failure.cell, failure.repeat, str(failure.required).lower(), failure.reason) for failure in failures))),
        ("matrix-summary.md", render_markdown(SUMMARY_FIELDS, tuple(TableRow(row) for row in summary_rows))),
        ("matrix-summary.svg", render_svg("Mean caller-envelope cycles", bars)),
    )
    output.mkdir(parents=True, exist_ok=True)
    for name, content in artifacts:
        (output / name).write_text(content, encoding="utf-8")
    return not any(failure.required for failure in failures)


def main(argv: Optional[Sequence[str]] = None) -> int:
    contract = "Input CSV columns: " + ",".join(INPUT_FIELDS) + ". One row per measured repeat; success requires cycle/wall metrics, and MHz requires affinity_status=conflict_free."
    parser = argparse.ArgumentParser(description="Summarize the fixed Q4/Q8/Q16 baseline/HP1 matrix.", epilog=contract)
    parser.add_argument("--input", required=True, type=Path, help="canonical per-repeat CSV emitted by the matrix runner")
    parser.add_argument("--output", required=True, type=Path, help="summary artifact directory")
    parser.add_argument("--expected-repeats", type=int, default=3, help="required repeats per each of the six cells (default: 3)")
    args = parser.parse_args(argv)
    if args.expected_repeats < 1:
        parser.error("--expected-repeats must be positive")
    return 0 if summarize(args.input, args.output, args.expected_repeats) else 1

if __name__ == "__main__":
    raise SystemExit(main())
