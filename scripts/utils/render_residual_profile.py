#!/usr/bin/env python3
"""Render CPU_DIRECT host profiles: python3 SCRIPT JSONL --output-dir DIRECTORY.

Requires Python 3.9+ and the standard library only. SVG coordinates subtract
integer nanosecond timestamps before conversion, preserving large host epochs.
"""

from __future__ import annotations

import argparse
import csv
import html
import sys
from pathlib import Path
from typing import Final, Optional, Sequence

try:
    from .cycle_schema import CycleSchemaError
    from .residual_profile import HostTiming, Profile, Tile, Timing, read_profiles
    from .residual_trace import trace
    from .table_output import TableRow, render_markdown
except ImportError:
    from cycle_schema import CycleSchemaError
    from residual_profile import HostTiming, Profile, Tile, Timing, read_profiles
    from residual_trace import trace
    from table_output import TableRow, render_markdown

IDENTITY: Final = ("execution_id", "run_id", "layer", "stripe_id")
PHASES: Final = ("validation", "preparation", "parallel", "finalization")
COLORS: Final = {"compute": "#217c5b", "legacy logging": "#c97818", "barrier": "#7854a8"}
STAGE_COLUMNS: Final = IDENTITY + ("scope", "worker_id", "node_id", "stage", "calls", "wall_ns",
                                  "thread_cpu_ns", "cycles", "cycles_valid", "cycles_reason")


def cell(value: Optional[int | str]) -> str:
    return "" if value is None else str(value)


def identity(profile: Profile) -> tuple[str, ...]:
    return (profile.host.execution_id, cell(profile.run_id), cell(profile.layer), str(profile.stripe_id))


def complete_sum(values: Sequence[Optional[int]]) -> Optional[int]:
    return None if any(value is None for value in values) else sum(value for value in values if value is not None)


def duration(timing: Optional[Timing]) -> Optional[int]:
    return None if timing is None else timing.host.duration_ns


def cpu(timing: Optional[Timing]) -> Optional[int]:
    return None if timing is None else timing.cpu_ns


def export_csv(path: Path, headers: Sequence[str], rows: Sequence[TableRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(row.cells for row in rows)


def worker_rows(profiles: Sequence[Profile]) -> tuple[TableRow, ...]:
    rows = []
    for profile in profiles:
        for worker in profile.workers:
            tiles = [tile for tile in profile.tiles if tile.worker_id == worker.worker_id]
            rows.append(TableRow(identity(profile) + tuple(cell(value) for value in (
                worker.worker_id, worker.tid, worker.timing.host.start_ns, worker.timing.host.end_ns,
                worker.timing.host.duration_ns, worker.timing.cpu_ns, duration(worker.barrier), cpu(worker.barrier),
                len(tiles), sum(tile.timing.host.duration_ns for tile in tiles),
                complete_sum([tile.timing.cpu_ns for tile in tiles]),
                complete_sum([duration(tile.log) for tile in tiles]),
                complete_sum([cpu(tile.log) for tile in tiles]),
            ))))
    return tuple(rows)


def tile_rows(profiles: Sequence[Profile]) -> tuple[TableRow, ...]:
    origins = {profile.host.execution_id: min(item.host.start_ns for item in profiles
               if item.host.execution_id == profile.host.execution_id) for profile in profiles}
    return tuple(TableRow(identity(profile) + tuple(cell(value) for value in (
        tile.node_id, tile.worker_id, tile.j_begin, tile.j_end, tile.timing.host.start_ns, tile.timing.host.end_ns,
        tile.timing.host.start_ns - origins[profile.host.execution_id], tile.timing.host.duration_ns,
        tile.timing.cpu_ns, duration(tile.log), cpu(tile.log), tile.log_calls, tile.log_mutex_wait_ns, tile.log_io_ns,
    ))) for profile in profiles for tile in profile.tiles)


def stage_rows(profiles: Sequence[Profile]) -> tuple[TableRow, ...]:
    rows = []
    for profile in profiles:
        if not profile.tiles[0].stages:
            continue
        groups: list[tuple[str, Optional[int], Optional[int], tuple[Tile, ...]]] = [("stripe", None, None, profile.tiles)]
        groups.extend(("worker", worker.worker_id, None,
                       tuple(tile for tile in profile.tiles if tile.worker_id == worker.worker_id))
                      for worker in profile.workers)
        groups.extend(("tile", tile.worker_id, tile.node_id, (tile,)) for tile in profile.tiles)
        for scope, worker_id, node_id, tiles in groups:
            if not tiles:
                continue
            for name in tiles[0].stages:
                stages = [tile.stages[name] for tile in tiles]
                cycles = complete_sum([stage.cycles for stage in stages])
                reason = "none" if cycles is not None else ";".join(dict.fromkeys(
                    stage.cycles_reason for stage in stages if stage.cycles is None))
                rows.append(TableRow(identity(profile) + tuple(cell(value) for value in (
                    scope, worker_id, node_id, name, sum(stage.calls for stage in stages),
                    complete_sum([stage.wall_ns for stage in stages]),
                    complete_sum([stage.thread_cpu_ns for stage in stages]), cycles,
                    str(cycles is not None).lower(), reason))))
    return tuple(rows)


def summary(profiles: Sequence[Profile]) -> str:
    lines = ["# CPU_DIRECT residual host profile\n",
             "Wall time is measured by steady_clock in nanoseconds. Thread CPU time is a separate measurement; "
             "blank cells mean unavailable or invalid sampling. No cycle-to-time conversion is used.\n",
             "Worker wall time ends at barrier arrival; barrier wait is separate. Tile compute excludes legacy logging. "
             "Sums across workers or stripes are work totals, not elapsed time. The profile excludes its own serialization/output.\n",
             "## Selected execution windows\n"]
    windows = []
    for execution in dict.fromkeys(profile.host.execution_id for profile in profiles):
        group = [profile for profile in profiles if profile.host.execution_id == execution]
        windows.append(TableRow((execution, str(len(group)),
                                str(max(item.host.end_ns for item in group) - min(item.host.start_ns for item in group)),
                                str(sum(item.host.duration_ns for item in group)))))
    lines.append(render_markdown(("execution_id", "stripes", "observed elapsed window (ns)", "sum stripe wall (ns)"), windows))
    lines.append("## Stripe workload\n")
    workload = ("row_begin", "row_count", "logical_j", "logical_k", "event_count", "active_rows", "active_row_blocks", "j_tile_count")
    lines.append(render_markdown(IDENTITY + workload + ("stripe wall (ns)",), tuple(
        TableRow(identity(profile) + tuple(str(profile.workload[key]) for key in workload) + (str(profile.host.duration_ns),))
        for profile in profiles)))
    lines.append("## Stripe phase durations\n")
    lines.append(render_markdown(IDENTITY + ("phase", "wall (ns)", "calling thread CPU (ns)"), tuple(
        TableRow(identity(profile) + (phase, str(profile.phases[phase].host.duration_ns), cell(profile.phases[phase].cpu_ns)))
        for profile in profiles for phase in PHASES)))
    lines.append("## Phase statistics across selected stripes\n")
    statistics = []
    for phase in PHASES:
        values = sorted(profile.phases[phase].host.duration_ns for profile in profiles)
        middle = values[(len(values) - 1) // 2] + values[len(values) // 2]
        median = str(middle // 2) + (".5" if middle % 2 else "")
        statistics.append(TableRow((phase, str(len(values)), str(sum(values)), median,
                                    str(values[(len(values) * 95 + 99) // 100 - 1]),
                                    cell(complete_sum([profile.phases[phase].cpu_ns for profile in profiles])))))
    lines.append(render_markdown(("phase", "count", "sum wall (ns)", "median wall (ns)", "p95 wall (ns)",
                                  "sum calling thread CPU (ns)"), statistics))
    lines.append("## Worker totals per stripe\n")
    lines.append(render_markdown(IDENTITY + ("parallel elapsed (ns)", "sum worker wall before barrier (ns)",
        "sum worker thread CPU before barrier (ns)", "sum barrier wall (ns)", "sum tile compute wall (ns)",
        "sum legacy logging wall (ns)"), tuple(TableRow(identity(profile) + tuple(cell(value) for value in (
            profile.phases["parallel"].host.duration_ns,
            sum(worker.timing.host.duration_ns for worker in profile.workers),
            complete_sum([worker.timing.cpu_ns for worker in profile.workers]),
            complete_sum([duration(worker.barrier) for worker in profile.workers]),
            sum(tile.timing.host.duration_ns for tile in profile.tiles),
            complete_sum([duration(tile.log) for tile in profile.tiles]),
        ))) for profile in profiles)))
    lines.append("The timeline shares a host axis only within one execution_id. Separate execution panels have independent "
                 "origins and scales. Gaps within the gray worker span are unclassified worker overhead. "
                 "Missing logging or barrier measurements have no colored segment.\n")
    lines.append("## Inner J-tile stages\n")
    stages = stage_rows(profiles)
    if stages:
        lines.append("These are measured interval sums, not contiguous timeline spans. event_scan locates each row/block's "
                     "events; weight_dot includes weight reading and integer multiply/accumulate; scale_apply reads scales "
                     "and applies the block result. Deep probes add overhead. Worker and stripe totals aggregate the same "
                     "tile data; do not add scopes together. PMU cycles are independent of wall/thread CPU nanoseconds.\n")
        lines.append(render_markdown(STAGE_COLUMNS, tuple(row for row in stages if row.cells[4] != "tile")))
    missing = [profile for profile in profiles if not profile.tiles[0].stages]
    if missing:
        lines.append(f"{len(missing)} selected stripe(s) have no inner-stage measurements. "
                     "Collect a new capture with GGML_GEMMINI_RESIDUAL_DEEP_PROFILE=1; absent stages are not zero.\n")
    lines.append("Open trace.json using **Open trace file** at https://ui.perfetto.dev. "
                 "It contains measured host spans on OS-thread tracks; aggregate-only stages have no invented timestamps. "
                 "Each execution has an independent zero origin, so cross-execution alignment does not establish overlap.\n")
    lines.append("![Worker wall-time timeline](worker-timeline.svg)\n")
    return "\n".join(lines)


def segment(host: HostTiming, origin: int, span: int, y: int, kind: str, label: str) -> str:
    start, end = host.start_ns - origin, host.end_ns - origin
    x, width = 245 + start / max(span, 1) * 920, (end - start) / max(span, 1) * 920
    return (f'<rect data-kind="{kind}" data-start-ns="{start}" data-end-ns="{end}" '
            f'x="{x:.6f}" y="{y}" width="{width:.6f}" height="16" fill="{COLORS[kind]}">'
            f'<title>{html.escape(label)}; wall {end - start} ns; relative [{start}, {end}) ns</title></rect>')


def timeline(profiles: Sequence[Profile]) -> str:
    body = ['<text x="24" y="32" class="heading">CPU_DIRECT residual worker timeline</text>',
            '<text x="24" y="54">Host wall time; independent panels for distinct execution IDs</text>']
    for index, (kind, color) in enumerate(COLORS.items()):
        x = 24 + index * 190
        body.append(f'<rect x="{x}" y="70" width="14" height="14" fill="{color}"/>'
                    f'<text x="{x + 22}" y="82">{kind}</text>')
    y = 120
    for execution in dict.fromkeys(profile.host.execution_id for profile in profiles):
        group = [profile for profile in profiles if profile.host.execution_id == execution]
        origin = min(profile.host.start_ns for profile in group)
        span = max(profile.host.end_ns for profile in group) - origin
        unit, divisor = ("ns", 1) if span < 1000 else (("µs", 1000) if span < 1000000 else ("ms", 1000000))
        body.append(f'<text x="24" y="{y}" class="heading">Execution {html.escape(execution)}</text>')
        body.append(f'<text x="24" y="{y + 22}">Origin {origin} ns; observed window {span} ns</text>')
        y += 54
        for tick in range(5):
            value = span * tick // 4
            x = 245 + tick * 230
            body.append(f'<text x="{x}" y="{y}" text-anchor="middle">{value / divisor:g} {unit}</text>')
        y += 18
        for profile in group:
            stripe_label = f"Run {cell(profile.run_id) or '(unset)'} / layer {profile.layer or '(unset)'} / stripe {profile.stripe_id}"
            body.append(f'<text x="24" y="{y + 13}">{html.escape(stripe_label)}</text>')
            y += 28
            for worker in profile.workers:
                label = f"stripe {profile.stripe_id} / worker {worker.worker_id} / tid {worker.tid}"
                body.append(f'<text x="24" y="{y + 13}">worker {worker.worker_id} / tid {worker.tid}</text>')
                body.append(f'<path d="M245 {y + 8}h920" stroke="#e6e9ed"/>')
                start = worker.timing.host.start_ns - origin
                end = worker.timing.host.end_ns - origin
                body.append(f'<rect x="{245 + start / max(span, 1) * 920:.6f}" y="{y}" '
                            f'width="{(end - start) / max(span, 1) * 920:.6f}" height="16" fill="#d6dce3"/>')
                for tile in profile.tiles:
                    if tile.worker_id == worker.worker_id:
                        detail = f"{label}; node {tile.node_id}; J [{tile.j_begin}, {tile.j_end})"
                        body.append(segment(tile.timing.host, origin, span, y, "compute", detail))
                        if tile.log is not None:
                            body.append(segment(tile.log.host, origin, span, y, "legacy logging", detail))
                if worker.barrier is not None:
                    body.append(segment(worker.barrier.host, origin, span, y, "barrier", label))
                y += 30
        y += 30
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{y}" viewBox="0 0 1200 {y}" '
            'role="img" aria-labelledby="title desc"><title id="title">Residual worker host timeline</title>'
            '<desc id="desc">Compute, legacy logging and barrier wall intervals across workers and stripes. '
            'Only intervals with the same execution ID share an axis.</desc>'
            '<style>text{font:13px sans-serif;fill:#263445}.heading{font-weight:600;font-size:16px}</style>'
            f'<rect width="1200" height="{y}" fill="white"/>\n' + "\n".join(body) + "\n</svg>\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Render measured CPU_DIRECT residual host profiles (Python 3.9+, stdlib)")
    parser.add_argument("input", type=Path, metavar="JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--layer")
    parser.add_argument("--stripe-id", type=int)
    args = parser.parse_args(argv)
    try:
        profiles = read_profiles(args.input, run_id=args.run_id, layer=args.layer, stripe_id=args.stripe_id)
        targets = [args.output_dir / name for name in
                   ("summary.md", "workers.csv", "tiles.csv", "worker-timeline.svg", "stages.csv", "trace.json")]
        for target in targets:
            if target.exists():
                raise FileExistsError(f"output target already exists: {target}")
        report, svg = summary(profiles), timeline(profiles)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        export_csv(targets[1], IDENTITY + ("worker_id", "tid", "start_ns", "end_ns", "worker_wall_ns",
            "worker_thread_cpu_ns", "barrier_wall_ns", "barrier_thread_cpu_ns", "tile_count", "sum_compute_wall_ns",
            "sum_compute_thread_cpu_ns", "sum_log_wall_ns", "sum_log_thread_cpu_ns"), worker_rows(profiles))
        export_csv(targets[2], IDENTITY + ("node_id", "worker_id", "j_begin", "j_end", "start_ns", "end_ns",
            "start_relative_ns", "compute_wall_ns", "compute_thread_cpu_ns", "log_wall_ns", "log_thread_cpu_ns",
            "log_calls", "log_mutex_wait_ns", "log_io_ns"), tile_rows(profiles))
        targets[0].write_text(report, encoding="utf-8")
        targets[3].write_text(svg, encoding="utf-8")
        export_csv(targets[4], STAGE_COLUMNS, stage_rows(profiles))
        targets[5].write_text(trace(profiles), encoding="utf-8")
    except (CycleSchemaError, OSError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
