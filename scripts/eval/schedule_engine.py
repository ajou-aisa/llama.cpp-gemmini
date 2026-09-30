"""One scheduling pass with selectable output sinks (the shared performance/timeline execution engine).

The IM2P SQLite scheduler runs in this process and hands every scheduled node, in schedule order, to
`ScheduleSinks.observe` right after the node's row is stored. Each node becomes its timeline rows through the single
row builder `e2e_timeline.node_rows`; the rows always feed the performance accumulator and, when a timeline is
requested, are also written as timeline.jsonl. Performance therefore never reads a timeline file back, and turning the
timeline on or off changes only what is written, never what is computed. The stored schedule (SQLite) remains the
compact record from which a timeline can be exported later without scheduling again.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, TextIO

from e2e_timeline import Axis, PerformanceAccumulator, node_rows, npu_works
from eval_common import Record


class ScheduleSinks:
    """Performance accumulator (always) and optional timeline writer fed by the scheduler's node stream."""

    def __init__(self, npu_results: Path, axis: Axis, generated: int, validation: str,
                 timeline: Path | None = None) -> None:
        self.works = npu_works(npu_results)
        self.axis = axis
        self.accumulator = PerformanceAccumulator(generated, axis.frequency_hz if axis.validated else None,
                                                  validation, axis.frequency_hz)
        self.stream: TextIO | None = timeline.open("x", encoding="utf-8") if timeline is not None else None
        self.nodes = self.rows = self.written = 0

    def observe(self, ordinal: int, node: Any, completed: Any, scheduled: Record) -> None:
        del completed  # the stored row carries the same endpoints as exact rationals
        info: Record = {"node_id": node.identity, "operation_id": node.operation, "phase": node.phase}
        for row in node_rows(ordinal, node.identity, scheduled, node.kind.value, info, self.works, self.axis):
            self.accumulator.add(row)
            if self.stream is not None:
                self.stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                self.written += 1
            self.rows += 1
        self.nodes += 1

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None


def run_schedule(im2p: Path, bundle: Path, phase_table: Path, frequency_hz: int, output: Path,
                 sinks: ScheduleSinks) -> Record:
    """SYNTHETIC schedule of the SQLite IR with the phase-table provider (as `execution_cli schedule --synthetic`)."""
    if str(im2p) not in sys.path:
        sys.path.insert(0, str(im2p))
    from sim.cycle.certificate_contract import read_document
    from sim.cycle.execution_cli import phase_table as parse_phase_table
    from sim.cycle.scheduler import Scenario
    from sim.cycle.scheduler_sqlite import SqliteScheduleInputs, schedule_sqlite
    started = time.monotonic()
    try:
        summary = schedule_sqlite(bundle, output, SqliteScheduleInputs(
            parse_phase_table(read_document(phase_table)), Scenario(frequency_hz, "SYNTHETIC")), sinks.observe)
    finally:
        sinks.close()
    return {"summary": summary, "seconds": round(time.monotonic() - started, 3), "nodes": sinks.nodes,
            "rows": sinks.rows, "counters": {"ir_node_bodies_parsed": summary["node_count"],
                                             "schedule_rows_written": sinks.nodes,
                                             "performance_accumulator_events": sinks.rows,
                                             "timeline_rows_written": sinks.written}}
