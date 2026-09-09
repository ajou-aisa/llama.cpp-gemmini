#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.utils.cycle_metrics import format_number, summarize
from scripts.utils.cycle_schema import CycleSchemaError, JsonValue, parse_cycle_jsonl, parse_cycle_line
from scripts.utils.table_output import BarValue, TableOutputError, render_svg

Row = dict[str, JsonValue]
COMMON: Row = dict(schema="gemmini.cycle", version=2, source="linux_perf_cpu_cycles", unit="cycle",
                   op="leaf", layer="blk.0", run_id=0, stripe_id=None, slot=None, node_id=0, worker_id=0)


def interval(value: int = 10, worker: int = 0) -> Row:
    return dict(COMMON, record_type="CYCLE_INTERVAL", start=2**63, end=2**63 + value,
                delta=value, valid=True, worker_id=worker)


def envelopes() -> list[Row]:
    rows: list[Row] = []
    for run, cycles in ((0, 101), (1, 203)):
        rows.append(dict(COMMON, record_type="RMD_BACKEND_TELEMETRY", op="rmd.execute", run_id=run,
                         runtime_bundle_id="bundle", model_id="model", backend="cpu_direct",
                         option_source="build_default", work=True, invocation_total=cycles,
                         dispatch={}, timing={}, geometry={}))
        for stripe in range(3 if run == 0 else 1):
            start = 2**63 + 50 * stripe
            length = 100 if run == 0 else 50
            row = dict(COMMON, record_type="PIPELINE_STRIPE_SUMMARY", source="steady_clock",
                       unit="nanosecond", op="matmul.pipeline", run_id=run, stripe_id=stripe,
                       slot=stripe % 2, row_begin=stripe, row_end=stripe + 1, valid=True)
            for stage in ("queue", "dense", "rmd", "compose", "finalize"):
                row[stage + "_start_ns"], row[stage + "_end_ns"] = start, start + length
            rows.append(row)
    return rows


class CycleLogTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "input.jsonl"

    def write_rows(self, rows: list[Row]) -> None:
        self.source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def cli(self, script: str, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, str(ROOT / "scripts/utils" / script), *arguments],
                              capture_output=True, text=True, check=False)

    def render(self, rows: list[Row], kind: str = "operation") -> Path:
        self.write_rows(rows)
        output = Path(tempfile.mkdtemp(dir=self.root))
        extra = ("--op", "leaf") if kind == "worker" else ()
        result = self.cli(f"render_{kind}_cycles.py", str(self.source), str(output), *extra)
        self.assertEqual(result.returncode, 0, result.stderr)
        return output

    def csv_rows(self, output: Path, name: str) -> list[dict[str, str]]:
        with (output / name).open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_schema_rejects_bad_input_at_the_physical_line(self) -> None:
        good = json.dumps(interval())
        variants = ["{", "", good[:-1] + ',"delta":10}', json.dumps(dict(interval(), delta=9)),
                    json.dumps(dict(interval(), version=3)), json.dumps(dict(interval(), worker_id=-1)),
                    json.dumps({key: value for key, value in interval().items() if key != "op"})]
        for bad in variants:
            with self.subTest(bad=bad):
                self.source.write_text(good + "\n" + bad + "\n", encoding="utf-8")
                with self.assertRaises(CycleSchemaError) as caught:
                    tuple(parse_cycle_jsonl(self.source))
                self.assertEqual(caught.exception.line_number, 2)
        self.assertNotEqual(self.cli("cycle_schema.py", "--check", str(self.source)).returncode, 0)

    def test_valid_zero_invalid_reason_and_exact_integer_contract(self) -> None:
        for row in (dict(interval(), valid=False), dict(interval(), valid=False, delta=None),
                    dict(interval(), valid=False, delta=None, reason="")):
            with self.subTest(row=row), self.assertRaises(CycleSchemaError):
                parse_cycle_line(json.dumps(row), 1)
        record = parse_cycle_line(json.dumps(interval(0)), 1)
        self.assertEqual((record.start, record.end, record.delta, record.run_id), (2**63, 2**63, 0, 0))
        self.assertEqual(format_number(summarize((2**63, 2**63 + 1)).median), "9223372036854775808.5")

    def test_pipeline_rejects_reversed_stage_boundaries(self) -> None:
        stripe = envelopes()[1]
        for stage in ("queue", "dense", "rmd", "compose", "finalize"):
            with self.subTest(stage=stage), self.assertRaises(CycleSchemaError):
                parse_cycle_line(json.dumps(dict(stripe, **{stage + "_end_ns": 0})), 1)

    def test_operation_and_worker_reports_keep_parent_and_worker_sums_separate(self) -> None:
        rows = [interval(value, worker) for value, worker in ((10, 0), (20, 0), (30, 1), (40, 1), (100, 1))]
        rows.append(dict(interval(200), op="parent", worker_id=None))
        output = self.render(rows)
        metrics = {row["operation"]: row for row in self.csv_rows(output, "operation-cycles.csv")}
        self.assertEqual(set(metrics), {"leaf", "parent"})
        self.assertEqual([metrics[name]["worker_work_cycles"] for name in ("leaf", "parent")], ["200", "200"])
        self.assertEqual((metrics["leaf"]["median_interval_cycles"], metrics["leaf"]["p95_interval_cycles"]), ("30", "100"))
        output = self.render(rows, "worker")
        workers = self.csv_rows(output, "worker-cycles.csv")
        self.assertEqual([(row["worker_id"], row["worker_work_cycles"]) for row in workers], [("0", "30"), ("1", "170")])
        self.assertEqual([row["median_interval_cycles"] for row in workers], ["15", "40"])
        for path in output.glob("*.svg"):
            ET.parse(path)

    def test_missing_worker_identity_or_operation_fails_before_output(self) -> None:
        self.write_rows([dict(interval(), worker_id=None)])
        for operation in ("leaf", "missing"):
            result = self.cli("render_worker_cycles.py", str(self.source), str(self.root / "out"), "--op", operation)
            self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "out").exists())

    def test_partial_measurements_preserve_zero_and_failure_reason(self) -> None:
        rows = [interval(0), dict(interval(), end=None, delta=None, valid=False, reason="invalid_end")]
        for kind in ("operation", "worker"):
            with self.subTest(kind=kind):
                metric = self.csv_rows(self.render(rows, kind), f"{kind}-cycles.csv")[0]
                self.assertEqual((metric["interval_count"], metric["valid_count"], metric["status"]), ("2", "1", "partial"))
                self.assertEqual((metric["worker_work_cycles"], metric["valid_value_sum"], metric["reason"]), ("", "0", "invalid_end"))

    def test_domains_aliases_and_raw_statuses_remain_distinct(self) -> None:
        rows = [dict(interval(), op="rmd_direct_finish_cycles", stripe_id=3, node_id=7),
                dict(interval(), op="residual_event_list_build"), dict(interval(), source="host_tick", unit="tick")]
        rows += [dict(interval(), op=reason, delta=None, valid=False, reason=reason)
                 for reason in ("not_collected", "not_applicable", "external_completion")]
        rows.append(dict(interval(), operation_success=False))
        output = self.render(rows)
        metrics = self.csv_rows(output, "operation-cycles.csv")
        self.assertEqual(len(metrics), 7)
        self.assertEqual({row["unit"] for row in metrics}, {"cycle", "tick"})
        self.assertEqual(len([row for row in metrics if row["canonical_id"] == "residual_event_list_build"]), 2)
        details = self.csv_rows(output, "cpu-measurements.csv")
        self.assertEqual((details[0]["run_id"], details[0]["stripe_id"], details[0]["node_id"]), ("0", "3", "7"))
        self.assertEqual(details[0]["parent"], "activation_stripe_preparation")
        self.assertEqual([row["status"] for row in details[3:]], ["not_collected", "not_applicable", "external_completion", "complete"])
        self.assertEqual([row["value"] for row in details[3:]], ["", "", "", "10"])
        self.assertEqual(details[-1]["operation_success"], "false")

    def test_exsia_detail_keeps_unknown_source_and_nullable_stages(self) -> None:
        timeline = dict(COMMON, record_type="TIMELINE", op="exsia.folding", source=None, mode="pipeline",
                        start=10, end=30, elapsed=20, start_thread_id=1, end_thread_id=1, clock_mode="cycle",
                        units="cycles", timer_resolution=1, team_size=2, cycle_status="complete")
        stage = dict(COMMON, record_type="STAGE", op="exsia.stage_metric", mode="pipeline",
                     metric="local.p0.sum", value=9, value_units="cycles", team_size=2, cycle_status="complete")
        rows = [timeline, stage, dict(stage, metric="local.p1.sum", value=None, cycle_status="not_collected")]
        details = self.csv_rows(self.render(rows), "cpu-measurements.csv")
        self.assertEqual(details[0]["source"], "")
        self.assertEqual([row["value"] for row in details], ["20", "9", ""])
        self.assertEqual(details[2]["status"], "not_collected")

    def test_caller_cycles_and_pipeline_ns_use_independent_invocation_envelopes(self) -> None:
        output = self.render(envelopes(), "e2e")
        caller = self.csv_rows(output, "caller-envelope-cycles.csv")[0]
        pipeline = self.csv_rows(output, "pipeline-envelope-ns.csv")[0]
        self.assertEqual((caller["caller_envelope_cycles"], caller["median_invocation_cycles"]), ("304", "152"))
        self.assertEqual((caller["status"], caller["trusted_caller_cycles"]), ("unknown", ""))
        self.assertEqual((pipeline["pipeline_envelope_ns"], pipeline["median_invocation_ns"]), ("250", "125"))

    def test_explicit_caller_validity_controls_trusted_totals(self) -> None:
        for valid in (True, False):
            rows = envelopes()
            for row in rows:
                if row["record_type"] == "RMD_BACKEND_TELEMETRY":
                    row.update(invocation_total_valid=valid, invocation_total_reason=None if valid else "invalid_end",
                               invocation_total_sample_reason=None if valid else "unavailable_event", invocation_total_count=1,
                               invocation_total_valid_count=int(valid), invocation_total_not_applicable_count=0)
                    if not valid:
                        row["invocation_total"] = None
            with self.subTest(valid=valid):
                caller = self.csv_rows(self.render(rows, "e2e"), "caller-envelope-cycles.csv")[0]
                self.assertEqual(caller["status"], "complete" if valid else "invalid")
                self.assertEqual(caller["trusted_caller_cycles"], "304" if valid else "")
                self.assertEqual(caller["caller_envelope_cycles"], "304" if valid else "")

    def test_ambiguous_or_incomplete_invocations_fail(self) -> None:
        rows = envelopes()
        partial = dict(rows[0], invocation_total_valid=True, invocation_total_reason=None,
                       invocation_total_sample_reason=None, invocation_total_count=2,
                       invocation_total_valid_count=1, invocation_total_not_applicable_count=0)
        variants = [rows + [dict(rows[1], slot=99)], rows + [rows[0]], rows[:1],
                    [dict(rows[0], run_id=None)] + rows[1:], [partial] + rows[1:]]
        for records in variants:
            with self.subTest(records=records):
                self.write_rows(records)
                result = self.cli("render_e2e_cycles.py", str(self.source), str(self.root / "out"))
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.root / "out").exists())

    def test_help_and_existing_report_collision(self) -> None:
        for kind, name in (("operation", "cpu-measurements.csv"), ("worker", "worker-cycles.svg"),
                           ("e2e", "pipeline-envelope-ns.csv")):
            with self.subTest(kind=kind):
                script = f"render_{kind}_cycles.py"
                self.assertEqual(self.cli(script, "--help").returncode, 0)
                self.write_rows([interval()] + envelopes())
                output = self.root / kind
                output.mkdir()
                (output / name).write_text("keep", encoding="utf-8")
                extra = ("--op", "leaf") if kind == "worker" else ()
                self.assertNotEqual(self.cli(script, str(self.source), str(output), *extra).returncode, 0)
                self.assertEqual((output / name).read_text(encoding="utf-8"), "keep")
                self.assertEqual(list(output.iterdir()), [output / name])

    def test_svg_rejects_invalid_values_and_escapes_labels(self) -> None:
        for value in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(TableOutputError):
                render_svg("cycles", (BarValue("x", value, "cycle"),))
        with self.assertRaises(TableOutputError):
            render_svg("mixed", (BarValue("a", 1, "cycle"), BarValue("b", 1, "ns")))
        root = ET.fromstring(render_svg("cycles", (BarValue("A<&", 1e308, "cycle"), BarValue("zero", 0, "cycle"))))
        self.assertEqual([node.text for node in root.findall("{*}text")][0], "A<&")
        self.assertEqual([node.attrib["width"] for node in root.findall("{*}rect")], ["300", "0"])


if __name__ == "__main__":
    unittest.main()
