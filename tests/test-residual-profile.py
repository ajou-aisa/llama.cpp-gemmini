#!/usr/bin/env python3
"""CLI regression checks with synthetic telemetry and an optional live CPU fixture."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/utils/render_residual_profile.py"
EPOCH = (1 << 54) + 1


def host(start: int, end: int, tid: int = 101) -> dict:
    return {"execution_id": "synthetic-test-only", "clock": "steady_clock", "unit": "nanosecond",
            "start_ns": EPOCH + start, "end_ns": EPOCH + end, "start_tid": tid, "end_tid": tid,
            "duration_ns": end - start, "valid": True, "thread_id_kind": "os_tid"}


def cpu(duration: int) -> dict:
    return {"clock": "thread_cpu", "unit": "nanosecond", "start_ns": 100,
            "end_ns": 100 + duration, "duration_ns": duration, "valid": True}


def fixture(stripe: int = 0, offset: int = 0) -> dict:
    """Explicit tiny intervals distinguish integer subtraction from float rounding."""
    phases = {name: {"host_timing": host(offset + start, offset + end),
                     "thread_cpu_timing": cpu(end - start)}
              for name, start, end in (("validation", 0, 10), ("preparation", 10, 20),
                                       ("parallel", 20, 100), ("finalization", 100, 110))}
    workers, tiles = [], []
    for worker, start, end, done, barrier_end in ((0, 20, 50, 70, 98), (1, 25, 75, 85, 99)):
        tid = 101 + worker
        workers.append({"worker_id": worker, "tid": tid,
                        "host_timing": host(offset + start, offset + done, tid),
                        "thread_cpu_timing": cpu(done - start - 5),
                        "barrier_host_timing": host(offset + done, offset + barrier_end, tid),
                        "barrier_thread_cpu_timing": cpu(1)})
        tiles.append({"node_id": worker, "worker_id": worker, "j_begin": worker * 16,
                      "j_end": (worker + 1) * 16,
                      "host_timing": host(offset + start, offset + end, tid),
                      "thread_cpu_timing": cpu(end - start - 1),
                      "log_host_timing": host(offset + end, offset + end + 5, tid),
                      "log_thread_cpu_timing": cpu(2), "log_calls": 1,
                      "log_mutex_wait_ns": 1, "log_io_ns": 2, "log_valid": True})
    return {"schema": "gemmini.cycle", "version": 2, "record_type": "RESIDUAL_HOST_PROFILE",
            "source": "steady_clock", "unit": "nanosecond", "op": "rmd.cpu_direct.profile",
            "layer": "synthetic.layer", "run_id": 7, "stripe_id": stripe, "valid": True,
            "host_timing": host(offset, offset + 110),
            "workload": {"row_begin": stripe * 4, "row_count": 4, "logical_j": 32, "logical_k": 16,
                         "event_count": 3, "active_rows": 2, "active_row_blocks": 2, "j_tile_count": 2},
            "phases": phases, "tiles": tiles, "workers": workers}


def deep_fixture() -> dict:
    record = dict(fixture(), deep_profile=True)
    for index, tile in enumerate(record["tiles"]):
        tile["stages"] = {
            name: {"calls": index + 1, "wall_ns": wall, "thread_cpu_ns": wall - 1,
                   "cycles": (1 << 54) + index + 1, "cycles_valid": True, "cycles_reason": "none"}
            for name, wall in (("event_scan", 3), ("weight_dot", 11), ("scale_apply", 5))}
    return record


class ResidualProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="residual-profile-test-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.input = self.directory / "synthetic.jsonl"
        self.output = self.directory / "rendered"

    def run_cli(self, records: list, *options: str) -> subprocess.CompletedProcess:
        self.input.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
        return subprocess.run([sys.executable, str(SCRIPT), str(self.input),
                               "--output-dir", str(self.output), *options],
                              capture_output=True, text=True, check=False)

    def test_exact_nanoseconds_and_shared_axis_when_workers_overlap(self) -> None:
        # Given mixed legacy telemetry, an epoch above 2^53 and consecutive stripes.
        records = [{"record_type": "RMD_BACKEND_TELEMETRY"}, fixture(), fixture(1, 110)]
        # When the real CLI renders the input.
        result = self.run_cli(records)
        # Then exact differences survive the trace and SVG position calculation.
        self.assertEqual(result.returncode, 0, result.stderr)
        trace = json.loads((self.output / "trace.json").read_text(encoding="utf-8"))
        tiles = [event for event in trace["traceEvents"] if event["name"] == "J-tile compute"]
        self.assertEqual([event["args"]["wall_ns"] for event in tiles], ["30", "50", "30", "50"])
        self.assertEqual(tiles[0]["args"]["start_ns"], str(EPOCH + 20))
        self.assertEqual(tiles[0]["ts"], 0.020)
        tree = ET.parse(self.output / "worker-timeline.svg")
        segments = [node for node in tree.iter() if node.get("data-kind") == "compute"]
        self.assertEqual([node.get("data-start-ns") for node in segments], ["20", "25", "130", "135"])
        first, second = segments[:2]
        self.assertLess(float(second.get("x")), float(first.get("x")) + float(first.get("width")))
        self.assertGreater(float(second.get("x")), float(first.get("x")))
        self.assertEqual({path.name for path in self.output.iterdir()},
                         {"summary.md", "worker-timeline.svg", "trace.json"})

    def test_missing_cpu_and_unavailable_logging_remain_unmeasured(self) -> None:
        # Given unavailable CPU sampling and a platform without legacy logging.
        record = fixture()
        del record["workers"][0]["thread_cpu_timing"]
        tile = record["tiles"][0]
        tile["thread_cpu_timing"] = {"clock": "thread_cpu", "unit": "nanosecond",
                                     "start_ns": None, "end_ns": None, "duration_ns": None, "valid": False}
        tile["log_host_timing"] = dict(host(50, 55), start_ns=None, end_ns=None, start_tid=None,
                                       end_tid=None, duration_ns=None, valid=False)
        tile["log_thread_cpu_timing"] = dict(tile["thread_cpu_timing"])
        for key in ("log_calls", "log_mutex_wait_ns", "log_io_ns"):
            tile[key] = None
        tile["log_valid"] = False
        # When rendered.
        result = self.run_cli([record])
        # Then missing CPU measurements stay null and unavailable logging has no span.
        self.assertEqual(result.returncode, 0, result.stderr)
        trace = json.loads((self.output / "trace.json").read_text(encoding="utf-8"))
        worker_events = [event for event in trace["traceEvents"]
                         if event["ph"] == "X" and event["args"].get("worker_id") == 0]
        for name in ("worker work", "J-tile compute"):
            event = next(event for event in worker_events if event["name"] == name)
            self.assertIsNone(event["args"]["thread_cpu_ns"])
        self.assertNotIn("legacy logging", [event["name"] for event in worker_events])
        summary = (self.output / "summary.md").read_text(encoding="utf-8")
        self.assertIn("| synthetic-test-only | 7 | synthetic.layer | 0 | 80 | 110 |  | 42 | 80 |  |", summary)

    def test_deep_stage_totals_preserve_cycles_and_unavailable_measurements(self) -> None:
        # Given exact large cycle values and one unavailable PMU stage.
        record = deep_fixture()
        record["tiles"][1]["stages"]["scale_apply"].update(
            cycles=None, cycles_valid=False, cycles_reason="unsupported_platform", thread_cpu_ns=None)
        # When rendered through the CLI.
        result = self.run_cli([record])
        # Then worker and stripe totals retain integers and propagate invalid measurements.
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = (self.output / "summary.md").read_text(encoding="utf-8")
        self.assertIn("| synthetic-test-only | 7 | synthetic.layer | 0 | stripe |  |  | weight_dot | "
                      f"3 | 22 | 20 | {(1 << 55) + 3} | true | none |", summary)
        self.assertIn("| synthetic-test-only | 7 | synthetic.layer | 0 | stripe |  |  | scale_apply | "
                      "3 | 10 |  |  | false | unsupported_platform |", summary)
        self.assertEqual(summary.count(" | 0 | worker | "), 6)
        self.assertIn("| synthetic-test-only | 7 | synthetic.layer | 0 | worker | 0 |  | weight_dot | "
                      f"1 | 11 | 10 | {(1 << 54) + 1} | true | none |", summary)

    def test_trace_and_timeline_keep_execution_origins_and_measured_spans(self) -> None:
        # Given two executions with unrelated host epochs and aggregate-only stages.
        other = json.loads(json.dumps(fixture(0, 10000)).replace("synthetic-test-only", "synthetic-second-execution"))
        # When the CLI exports the trace and timeline.
        result = self.run_cli([deep_fixture(), other])
        # Then each execution has its own process identity and exact small relative timestamps.
        self.assertEqual(result.returncode, 0, result.stderr)
        trace = json.loads((self.output / "trace.json").read_text(encoding="utf-8"))
        self.assertEqual(trace["displayTimeUnit"], "ns")
        events = [event for event in trace["traceEvents"] if event["ph"] == "X"]
        tiles = [event for event in events if event["name"] == "J-tile compute"]
        self.assertEqual([(event["ts"], event["dur"]) for event in tiles],
                         [(0.020, 0.030), (0.025, 0.050), (0.020, 0.030), (0.025, 0.050)])
        self.assertEqual([event["tid"] for event in tiles], [101, 102, 101, 102])
        self.assertNotEqual(tiles[0]["pid"], tiles[2]["pid"])
        self.assertEqual(tiles[0]["args"]["start_ns"], str(EPOCH + 20))
        self.assertEqual({event["name"] for event in events},
                         {"CPU_DIRECT", "validation", "preparation", "parallel", "finalization",
                          "worker work", "J-tile compute", "legacy logging", "barrier"})
        tree = ET.parse(self.output / "worker-timeline.svg")
        segments = [node for node in tree.iter() if node.get("data-kind") == "compute"]
        self.assertEqual([node.get("data-start-ns") for node in segments], ["20", "25", "20", "25"])
        self.assertEqual(segments[0].get("x"), segments[2].get("x"))

    def test_corrupt_deep_stage_data_fails_before_publication(self) -> None:
        mutations = (
            lambda row: row.update(deep_profile=1),
            lambda row: row.update(deep_profile=False),
            lambda row: row["tiles"][0]["stages"].pop("weight_dot"),
            lambda row: row["tiles"][0]["stages"]["weight_dot"].update(calls=-1),
            lambda row: row["tiles"][0]["stages"]["weight_dot"].update(cycles_valid=False),
            lambda row: row["tiles"][0]["stages"]["weight_dot"].update(cycles=None),
            lambda row: row["tiles"][0]["stages"]["weight_dot"].update(wall_ns=-1),
            lambda row: row["tiles"][0]["stages"]["weight_dot"].update(calls=0),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                # Given contradictory deep measurements.
                record = deep_fixture()
                mutate(record)
                # When parsed.
                result = self.run_cli([record])
                # Then no partial output is published.
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.output.exists())

    def test_filters_select_identity_before_deep_validation(self) -> None:
        # Given one selected profile and a malformed profile from another run.
        unrelated = dict(fixture(), run_id=8, phases=None)
        # When all identity filters are specified.
        result = self.run_cli([unrelated, fixture()], "--run-id", "7", "--layer", "synthetic.layer", "--stripe-id", "0")
        # Then only the selected stripe is exported.
        self.assertEqual(result.returncode, 0, result.stderr)
        trace = json.loads((self.output / "trace.json").read_text(encoding="utf-8"))
        workers = [event for event in trace["traceEvents"] if event["name"] == "worker work"]
        self.assertEqual([(event["args"]["run_id"], event["args"]["stripe_id"],
                           event["args"]["worker_id"], event["tid"]) for event in workers],
                         [(7, 0, 0, 101), (7, 0, 1, 102)])

    def test_corrupt_selected_telemetry_never_publishes(self) -> None:
        mutations = (
            lambda row: row.update(unit="cycle"),
            lambda row: row["host_timing"].update(end_ns=EPOCH - 1),
            lambda row: row["tiles"][0]["host_timing"].update(execution_id="different-process"),
            lambda row: row["tiles"][0]["host_timing"].update(start_tid=999),
            lambda row: row["workers"][0]["thread_cpu_timing"].update(duration_ns=999),
            lambda row: row["tiles"][0].update(worker_id=99),
            lambda row: row["tiles"][0].update(j_end=99),
            lambda row: row["tiles"][0].update(log_mutex_wait_ns=-1),
            lambda row: row["phases"]["parallel"]["host_timing"].update(unit="cycle"),
            lambda row: row["workload"].update(logical_j=2**68, j_tile_count=2**64),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                # Given one corrupt selected record after a valid record.
                malformed = fixture(1, 110)
                mutate(malformed)
                # When parsed and rendered.
                result = self.run_cli([fixture(), malformed])
                # Then the entire invocation fails before creating output.
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("line 2", result.stderr)
                self.assertFalse(self.output.exists())

    def test_no_profile_capture_requires_rebuild_and_recapture(self) -> None:
        # Given only old telemetry.
        # When the renderer is run.
        result = self.run_cli([{"record_type": "RMD_BACKEND_TELEMETRY"}])
        # Then failure identifies the missing new capture instead of empty success.
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("RESIDUAL_HOST_PROFILE", result.stderr)
        self.assertIn("LOG_CYCLE", result.stderr)
        self.assertFalse(self.output.exists())

    def test_malformed_json_and_missing_input_fail_without_output(self) -> None:
        for content in (None, "{broken\n"):
            with self.subTest(content=content):
                # Given malformed or missing input.
                if content is not None:
                    self.input.write_text(content, encoding="utf-8")
                # When the CLI reads it.
                result = subprocess.run([sys.executable, str(SCRIPT), str(self.input),
                                         "--output-dir", str(self.output)],
                                        capture_output=True, text=True, check=False)
                # Then it fails and publishes nothing.
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.output.exists())

    def test_output_collision_does_not_replace_existing_artifact(self) -> None:
        # Given an existing report.
        self.output.mkdir()
        sentinel = self.output / "summary.md"
        sentinel.write_text("keep", encoding="utf-8")
        # When another rendering targets the directory.
        result = self.run_cli([fixture()])
        # Then the existing artifact stays intact and no partial export appears.
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
        self.assertEqual(list(self.output.iterdir()), [sentinel])

    def test_duplicate_identity_is_rejected(self) -> None:
        # Given the same profile identity twice.
        # When rendered.
        result = self.run_cli([fixture(), fixture()])
        # Then duplicate spans cannot be double counted.
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("line 2", result.stderr)
        self.assertFalse(self.output.exists())


def test_runtime_fixture(executable: Path) -> None:
    # Given the compiled sparse CPU_DIRECT fixture with deep profiling off and on.
    for deep_profile in (False, True):
        with tempfile.TemporaryDirectory(prefix="residual-profile-runtime-") as temporary:
            directory = Path(temporary)
            capture, output = directory / "capture.jsonl", directory / "rendered"
            emitted = subprocess.run(
                [str(executable.resolve()), "--direct-host-profile-output", str(capture),
                 "--direct-host-profile-repeat", "2"],
                env={**os.environ, "GGML_GEMMINI_RESIDUAL_DEEP_PROFILE": str(int(deep_profile))},
                capture_output=True, text=True, check=False)
            assert emitted.returncode == 0, emitted.stderr
            # When the production renderer consumes the real execution's JSONL.
            rendered = subprocess.run([sys.executable, str(SCRIPT), str(capture),
                                       "--output-dir", str(output)],
                                      capture_output=True, text=True, check=False)
            assert rendered.returncode == 0, rendered.stderr
            # Then repeated stripes expose a tile trace and only enabled stages.
            summary = (output / "summary.md").read_text(encoding="utf-8")
            stripes = [line.split(" | ") for line in summary.splitlines() if " | stripe | " in line]
            if deep_profile:
                assert len(stripes) == 12
                assert {row[7] for row in stripes} == {"event_scan", "weight_dot", "scale_apply"}
                assert {row[1] for row in stripes} == {"27159", "27160"}
                assert all(int(row[8]) == (15 if row[3] == "3" else 9) and row[9] != "" for row in stripes)
            else:
                assert not stripes
            trace = json.loads((output / "trace.json").read_text(encoding="utf-8"))
            assert sum(event["ph"] == "X" and event["name"] == "J-tile compute"
                       for event in trace["traceEvents"]) == 12


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--runtime-fixture":
        test_runtime_fixture(Path(sys.argv[2]))
    else:
        unittest.main()
