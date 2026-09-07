#!/usr/bin/env python3
"""Red tests for deterministic table output."""

import ast
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.utils.cycle_metrics import format_number, operation_definition, summarize
from scripts.utils.table_output import BarValue, TableOutputError, TableRow, render_markdown, render_svg

METRICS_FIXTURE = Path("tests/fixtures/cycle-experiment/metrics.jsonl")
TODO5_MODULES = (
    Path("scripts/utils/cycle_metrics.py"),
    Path("scripts/utils/render_operation_cycles.py"),
    Path("scripts/utils/render_worker_cycles.py"),
    Path("scripts/utils/render_e2e_cycles.py"),
)


class TableOutputTests(unittest.TestCase):
    def test_output_only_is_deterministic_and_escaped(self) -> None:
        rows = (TableRow(("A<&\"", "7")), TableRow(("B", "0")))
        bars = (BarValue("A<&\"", 7.0, "cycles"), BarValue("B", 0.0, "cycles"))
        markdown = render_markdown(("label", "value"), rows)
        svg = render_svg("cycles", bars)
        self.assertEqual(markdown, render_markdown(("label", "value"), rows))
        self.assertIn("A<&\"", markdown)
        self.assertIn("A&lt;&amp;&quot;", svg)
        root = ET.fromstring(svg)
        self.assertTrue(float(root.attrib["width"]) > 0)
        namespace = "{http://www.w3.org/2000/svg}"
        self.assertTrue(all(float(node.attrib.get("width", "0")) >= 0 for node in root.iter(namespace + "rect")))

    def test_large_finite_values_have_finite_rect_widths(self) -> None:
        svg = render_svg("large", (BarValue("large", 1e308, "cycles"),))
        root = ET.fromstring(svg)
        namespace = "{http://www.w3.org/2000/svg}"
        widths = [float(node.attrib["width"]) for node in root.iter(namespace + "rect")]
        self.assertEqual(widths, [300.0])
        self.assertTrue(all(math.isfinite(value) and value >= 0 for value in widths))

    def test_zero_only_geometry_and_order_and_numeric_labels(self) -> None:
        bars = (BarValue("second", 0.0, "cycles"), BarValue("first", 0.0, "cycles"))
        svg = render_svg("zero", bars)
        root = ET.fromstring(svg)
        namespace = "{http://www.w3.org/2000/svg}"
        texts = [node.text for node in root.iter(namespace + "text")]
        self.assertEqual(texts, ["second", "0 cycles", "first", "0 cycles"])
        self.assertEqual([node.attrib["width"] for node in root.iter(namespace + "rect")], ["0", "0"])

    def test_invalid_bar_values_are_rejected(self) -> None:
        for value in (-1.0, float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(TableOutputError):
                render_svg("invalid", (BarValue("x", value, "cycles"),))

    def test_mixed_units_cli_fails(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/utils/table_output.py", "--output-only", "cycles", "ns"],
            capture_output=True, text=True, check=False,
        )
        self.assertNotEqual(result.returncode, 0)


class CycleMetricsTests(unittest.TestCase):
    def test_cpu_stage_aliases_are_unit_independent(self) -> None:
        aliases = {
            "im2p.fence_host_call": "npu_completion_wait",
            "im2p.stripe_submit_host_call": "npu_command_submit",
            "im2p.output_buffer_copy": "output_buffer_copy",
            "gemmini.output_preparation": "output_buffer_preparation",
            "exsia.stage_metric:local.p0.sum": "activation_local_p0",
            "exsia.stage_metric:local.p0.count": "activation_local_p0",
        }
        for raw, canonical in aliases.items():
            with self.subTest(raw=raw):
                self.assertEqual(operation_definition(raw).canonical_id, canonical)
                self.assertRegex(canonical, r"^[a-z][a-z0-9_]*$")

    def test_actual_python39_imports_metrics_and_all_cli_help_surfaces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = dict(os.environ)
            environment["PYTHONPYCACHEPREFIX"] = str(Path(directory, "pycache"))
            commands = (
                ("-c", "from scripts.utils.cycle_metrics import summarize; assert summarize((1, 2)).total == 3"),
                ("scripts/utils/render_operation_cycles.py", "--help"),
                ("scripts/utils/render_worker_cycles.py", "--help"),
                ("scripts/utils/render_e2e_cycles.py", "--help"),
            )
            for command in commands:
                with self.subTest(command=command):
                    result = subprocess.run(
                        ["/usr/bin/python3", *command], capture_output=True, text=True,
                        check=False, env=environment,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
            happy_commands = (
                ("render_operation_cycles.py", str(METRICS_FIXTURE), f"{directory}/operation"),
                ("render_worker_cycles.py", str(METRICS_FIXTURE), f"{directory}/worker", "--op", "leaf"),
                ("render_e2e_cycles.py", str(METRICS_FIXTURE), f"{directory}/e2e"),
            )
            for command in happy_commands:
                with self.subTest(command=command):
                    result = subprocess.run(
                        ["/usr/bin/python3", f"scripts/utils/{command[0]}", *command[1:]],
                        capture_output=True, text=True, check=False, env=environment,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_every_output_target_collision_fails_before_publication(self) -> None:
        targets = {
            "render_operation_cycles.py": (
                "operation-cycles.csv", "operation-cycles.md", "operation-cycles.svg", "cpu-measurements.csv",
            ),
            "render_worker_cycles.py": (
                "worker-cycles.csv", "worker-cycles.md", "worker-cycles.svg", "worker-measurements.csv",
            ),
            "render_e2e_cycles.py": (
                "caller-envelope-cycles.csv", "caller-envelope-cycles.md",
                "caller-envelope-cycles.svg", "pipeline-envelope-ns.csv",
                "pipeline-envelope-ns.md", "pipeline-envelope-ns.svg", "caller-measurements.csv",
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = dict(os.environ)
            environment["PYTHONPYCACHEPREFIX"] = str(root / "pycache")
            for script, names in targets.items():
                for index, name in enumerate(names):
                    with self.subTest(script=script, target=name):
                        output = root / f"{script}-{index}"
                        output.mkdir()
                        sentinel = output / name
                        unrelated = output / "unrelated.keep"
                        sentinel.write_text("TARGET-SENTINEL", encoding="utf-8")
                        unrelated.write_text("UNRELATED-SENTINEL", encoding="utf-8")
                        arguments = [str(METRICS_FIXTURE), str(output)]
                        if script == "render_worker_cycles.py":
                            arguments.extend(("--op", "leaf"))
                        result = subprocess.run(
                            ["/usr/bin/python3", f"scripts/utils/{script}", *arguments],
                            capture_output=True, text=True, check=False, env=environment,
                        )
                        self.assertNotEqual(result.returncode, 0)
                        self.assertEqual(sentinel.read_text(encoding="utf-8"), "TARGET-SENTINEL")
                        self.assertEqual(unrelated.read_text(encoding="utf-8"), "UNRELATED-SENTINEL")
                        self.assertEqual(set(output.iterdir()), {sentinel, unrelated})

    def test_every_todo5_module_has_python39_grammar(self) -> None:
        for path in TODO5_MODULES:
            with self.subTest(path=path):
                ast.parse(
                    path.read_text(encoding="utf-8"),
                    filename=str(path),
                    feature_version=(3, 9),
                )

    def test_large_even_sample_median_remains_exact(self) -> None:
        summary = summarize((2**63, 2**63 + 1))
        self.assertEqual(format_number(summary.median), "9223372036854775808.5")

    def run_cli(self, script: str, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, f"scripts/utils/{script}", *arguments],
            capture_output=True, text=True, check=False,
        )

    def test_operation_constants_are_independent_of_nested_interval_meaning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_cli("render_operation_cycles.py", str(METRICS_FIXTURE), directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            with Path(directory, "operation-cycles.csv").open(newline="", encoding="utf-8") as stream:
                rows = {row["operation"]: row for row in csv.DictReader(stream)}
        self.assertEqual(rows["leaf"]["worker_work_cycles"], "200")
        self.assertEqual(rows["leaf"]["median_interval_cycles"], "30")
        self.assertEqual(rows["leaf"]["p95_interval_cycles"], "100")
        self.assertEqual(rows["parent"]["worker_work_cycles"], "200")
        self.assertNotIn("latency", rows["leaf"])
        self.assertNotIn("share", rows["leaf"])
        self.assertNotIn("e2e", rows["leaf"])

    def test_worker_constants_require_the_configured_leaf_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_cli(
                "render_worker_cycles.py", str(METRICS_FIXTURE), directory, "--op", "leaf",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with Path(directory, "worker-cycles.csv").open(newline="", encoding="utf-8") as stream:
                rows = {row["worker_id"]: row for row in csv.DictReader(stream)}
        self.assertEqual(rows["0"]["worker_work_cycles"], "30")
        self.assertEqual(rows["0"]["median_interval_cycles"], "15")
        self.assertEqual(rows["0"]["p95_interval_cycles"], "20")
        self.assertEqual(rows["1"]["worker_work_cycles"], "170")
        self.assertEqual(rows["1"]["median_interval_cycles"], "40")
        self.assertEqual(rows["1"]["p95_interval_cycles"], "100")

    def test_e2e_constants_use_caller_and_per_invocation_pipeline_envelopes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_cli("render_e2e_cycles.py", str(METRICS_FIXTURE), directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            with Path(directory, "caller-envelope-cycles.csv").open(newline="", encoding="utf-8") as stream:
                caller = next(csv.DictReader(stream))
            with Path(directory, "pipeline-envelope-ns.csv").open(newline="", encoding="utf-8") as stream:
                pipeline = next(csv.DictReader(stream))
        self.assertEqual(caller["caller_envelope_cycles"], "611")
        self.assertEqual(caller["median_invocation_cycles"], "203")
        self.assertEqual(caller["p95_invocation_cycles"], "307")
        self.assertEqual(pipeline["pipeline_envelope_ns"], "400")
        self.assertEqual(pipeline["median_invocation_ns"], "100")
        self.assertEqual(pipeline["p95_invocation_ns"], "250")

    def test_invalid_and_ambiguous_metric_inputs_fail_closed(self) -> None:
        fixture_text = METRICS_FIXTURE.read_text(encoding="utf-8")
        lines = fixture_text.splitlines()
        variants = {
            "malformed": '{"schema":\n',
            "mixed": "\n".join((lines[0], lines[0].replace(
                '"source":"linux_perf_cpu_cycles","unit":"cycle"',
                '"source":"host_tick","unit":"tick"',
            ))) + "\n",
            "invalid": lines[0].replace(
                '"end":10,"delta":10,"valid":true',
                '"end":null,"delta":null,"valid":false,"reason":"counter_unavailable"',
            ) + "\n",
            "missing-op": lines[0].replace('"op":"leaf"', '"op":null') + "\n",
            "null-invocation": "\n".join((
                lines[6].replace('"run_id":1', '"run_id":null'), lines[9],
            )) + "\n",
            "duplicate": "\n".join((lines[6], lines[6], lines[9])) + "\n",
            "missing-family": lines[6] + "\n",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for name, content in variants.items():
                path = root / f"{name}.jsonl"
                path.write_text(content, encoding="utf-8")
                paths[name] = path
            cases = (
                ("render_worker_cycles.py", (str(METRICS_FIXTURE), str(root / "out-a"), "--op", "missing")),
                ("render_worker_cycles.py", (str(METRICS_FIXTURE), str(root / "out-b"), "--op", "parent")),
                ("render_operation_cycles.py", (str(paths["malformed"]), str(root / "out-c"))),
                # Well-formed invalid and separate source domains now remain visible;
                # the correction tests below assert null totals and distinct rows.
                ("render_operation_cycles.py", (str(paths["missing-op"]), str(root / "out-f"))),
                ("render_e2e_cycles.py", (str(paths["null-invocation"]), str(root / "out-g"))),
                ("render_e2e_cycles.py", (str(paths["duplicate"]), str(root / "out-h"))),
                ("render_e2e_cycles.py", (str(paths["missing-family"]), str(root / "out-i"))),
            )
            for script, arguments in cases:
                with self.subTest(script=script, fixture=arguments[0]):
                    result = self.run_cli(script, *arguments)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(Path(arguments[1]).exists())


class CpuConsumerCorrectionTests(unittest.TestCase):
    def render(self, root, records, script="render_operation_cycles.py", extra=()):
        source = root / "input.jsonl"
        source.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        output = root / script
        result = subprocess.run([sys.executable, "scripts/utils/" + script, str(source), str(output), *extra],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        return output

    def records(self):
        return [json.loads(line) for line in METRICS_FIXTURE.read_text(encoding="utf-8").splitlines()]

    def csv_rows(self, output, name):
        with (output / name).open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def test_invalid_and_valid_zero_survive_operation_and_worker_tables(self):
        leaf = self.records()[0]
        rows = [{**leaf, "end": leaf["start"], "delta": 0},
                {**leaf, "end": None, "delta": None, "valid": False, "reason": "invalid_end"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for script, name, extra in (("render_operation_cycles.py", "operation-cycles.csv", ()),
                                       ("render_worker_cycles.py", "worker-cycles.csv", ("--op", "leaf"))):
                output = self.render(root, rows, script, extra)
                metric = self.csv_rows(output, name)[0]
                self.assertEqual((metric["interval_count"], metric["valid_count"], metric["status"]), ("2", "1", "partial"))
                self.assertEqual(metric["worker_work_cycles"], "")
                self.assertEqual(metric["valid_value_sum"], "0")
                self.assertEqual(metric["reason"], "invalid_end")

    def test_mixed_domains_are_separate_and_aliases_do_not_merge_records(self):
        leaf = self.records()[0]
        rows = [{**leaf, "op": "rmd_direct_finish_cycles"},
                {**leaf, "op": "residual_event_list_build"},
                {**leaf, "source": "host_tick", "unit": "tick"}]
        with tempfile.TemporaryDirectory() as directory:
            output = self.render(Path(directory), rows)
            metrics = self.csv_rows(output, "operation-cycles.csv")
            self.assertEqual(len(metrics), 3)
            self.assertEqual({(row["source"], row["unit"]) for row in metrics},
                             {("linux_perf_cpu_cycles", "cycle"), ("host_tick", "tick")})
            aliases = [row for row in metrics if row["canonical_id"] == "residual_event_list_build"]
            self.assertEqual(len(aliases), 2)
            self.assertEqual([row["interval_count"] for row in aliases], ["1", "1"])

    def test_detail_keeps_raw_identity_domains_status_and_parent_relations(self):
        families = [json.loads(line) for line in Path("tests/fixtures/cycle-experiment/all-families.jsonl").read_text().splitlines()]
        timeline = next(row for row in families if row["record_type"] == "TIMELINE")
        stage = next(row for row in families if row["record_type"] == "STAGE")
        leaf = {**self.records()[0], "op": "rmd_direct_finish_cycles", "run_id": 0, "stripe_id": 3, "node_id": 7}
        rows = [leaf, {**timeline, "op": "exsia.folding", "cycle_status": "complete", "start": 10, "end": 30, "elapsed": 20},
                {**stage, "metric": "local.p0.sum", "value": 9, "value_units": "cycles", "cycle_status": "complete"},
                {**stage, "metric": "local.p1.sum", "value": None, "value_units": "cycles", "cycle_status": "not_collected"}]
        with tempfile.TemporaryDirectory() as directory:
            output = self.render(Path(directory), rows)
            details = self.csv_rows(output, "cpu-measurements.csv")
            self.assertEqual(len(details), 4)
            self.assertEqual((details[0]["raw_op"], details[0]["canonical_id"], details[0]["run_id"], details[0]["stripe_id"], details[0]["node_id"]),
                             ("rmd_direct_finish_cycles", "residual_event_list_build", "0", "3", "7"))
            self.assertEqual(details[0]["parent"], "activation_stripe_preparation")
            self.assertEqual(details[1]["source"], "")
            self.assertEqual((details[3]["value"], details[3]["status"]), ("", "not_collected"))
            operations = self.csv_rows(output, "operation-cycles.csv")
            self.assertEqual(len(operations), 4)
            self.assertNotIn("total", {row["operation"] for row in operations})

    def test_legacy_caller_total_is_exported_but_not_trusted(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.render(Path(directory), self.records(), "render_e2e_cycles.py")
            caller = self.csv_rows(output, "caller-envelope-cycles.csv")[0]
            self.assertEqual(caller["caller_envelope_cycles"], "611")
            self.assertEqual(caller["status"], "unknown")
            self.assertEqual(caller["trusted_caller_cycles"], "")
            self.assertEqual(caller["reason"], "legacy_validity_unknown")

    def test_explicit_rmd_completeness_and_nullable_failures(self):
        for invalid in (False, True):
            rows = self.records()
            for row in rows:
                if row["record_type"] == "RMD_BACKEND_TELEMETRY":
                    row.update(invocation_total_valid=not invalid, invocation_total_reason="invalid_end" if invalid else None,
                               invocation_total_sample_reason="unavailable_event" if invalid else None,
                               invocation_total_count=1, invocation_total_valid_count=0 if invalid else 1,
                               invocation_total_not_applicable_count=0)
                    if invalid:
                        row["invocation_total"] = None
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as directory:
                output = self.render(Path(directory), rows, "render_e2e_cycles.py")
                caller = self.csv_rows(output, "caller-envelope-cycles.csv")[0]
                self.assertEqual(caller["status"], "invalid" if invalid else "complete")
                self.assertEqual(caller["trusted_caller_cycles"], "" if invalid else "611")
                self.assertEqual(caller["caller_envelope_cycles"], "" if invalid else "611")

    def test_not_collected_not_applicable_and_operation_failure_remain_distinct(self):
        leaf = self.records()[0]
        rows = [{**leaf, "op": reason, "delta": None, "valid": False, "reason": reason}
                for reason in ("not_collected", "not_applicable", "external_completion")]
        rows.append({**leaf, "operation_success": False})
        with tempfile.TemporaryDirectory() as directory:
            output = self.render(Path(directory), rows)
            detail = self.csv_rows(output, "cpu-measurements.csv")
            self.assertEqual([row["status"] for row in detail], ["not_collected", "not_applicable", "external_completion", "complete"])
            self.assertEqual(detail[-1]["operation_success"], "false")
            self.assertEqual(detail[-1]["value"], "10")
            self.assertTrue(all(row["value"] == "" for row in detail[:-1]))

    def test_rmd_partial_or_inconsistent_validity_is_not_trusted(self):
        rows = self.records()
        for row in rows:
            if row["record_type"] == "RMD_BACKEND_TELEMETRY":
                row.update(invocation_total_valid=True, invocation_total_reason=None,
                           invocation_total_sample_reason=None, invocation_total_count=2,
                           invocation_total_valid_count=1, invocation_total_not_applicable_count=0)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "input.jsonl")
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = subprocess.run([sys.executable, "scripts/utils/render_e2e_cycles.py", str(source), str(Path(directory, "out"))], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not explicitly complete", result.stderr)

    def test_three_stripes_can_reuse_slot_without_merging_identity(self):
        rows = self.records()
        stripe = next(row for row in rows if row["record_type"] == "PIPELINE_STRIPE_SUMMARY")
        rows.extend(({**stripe, "stripe_id": 1, "slot": 1}, {**stripe, "stripe_id": 2, "slot": 0}))
        with tempfile.TemporaryDirectory() as directory:
            output = self.render(Path(directory), rows, "render_e2e_cycles.py")
            self.assertEqual(self.csv_rows(output, "pipeline-envelope-ns.csv")[0]["pipeline_envelope_ns"], "400")

    def test_slot_change_cannot_hide_duplicate_stripe(self):
        rows = self.records()
        stripe = next(row for row in rows if row["record_type"] == "PIPELINE_STRIPE_SUMMARY")
        rows.append({**stripe, "slot": stripe["slot"] + 99})
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "input.jsonl")
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = subprocess.run([sys.executable, "scripts/utils/render_e2e_cycles.py", str(source), str(Path(directory, "out"))], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("duplicate pipeline stripe identity", result.stderr)


if __name__ == "__main__":
    if "--output-only" in sys.argv:
        sys.argv.remove("--output-only")
    if "--metrics" in sys.argv:
        sys.argv.remove("--metrics")
    unittest.main()
