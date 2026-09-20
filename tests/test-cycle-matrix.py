#!/usr/bin/env python3
"""Acceptance-runner tests; synthetic fixtures never count as model inference."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cycle_matrix", ROOT / "scripts/experiment/cycle_matrix.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def evidence():
    phases = []
    for name, operations, elapsed, cycles, thread in (
        ("prefill", 2, 3_000_000, 90_071_992_547_409_931, 6_000_000),
        ("decode", 4, 5_000_000, 70_000_000, 10_000_000),
    ):
        phases.append({"phase": name, "operations": operations, "failed_operations": 0,
                       "elapsed_ns": elapsed, "cpu_cycles": cycles, "thread_cpu_ns": thread,
                       "cpu_work_wall_ns": None, "cpu_work_wall_ns_reason": "coverage_incomplete",
                       "npu_reason": "no_device_counter_records"})
    summary = {"record_type": "FINAL_INFERENCE_SUMMARY", "available": True,
               "requests": 1, "completed_requests": 1, "tokens": 3,
               "cpu_interval_coverage": "verified", "latency_boundary_status": "verified",
               "ttft_samples": 1, "tpot_gaps": 2, "ttft_ns": 4_000_000, "tpot_ns": 3_000_000,
               "phases": phases}
    runtime = "Inference performance (1 request, 3 tokens)\n"
    for phase in phases:
        runtime += (f"  {phase['phase']}: elapsed={phase['elapsed_ns']/1e6:.3f} ms, "
                    f"CPU cycles={phase['cpu_cycles']}, worker CPU={phase['thread_cpu_ns']/1e6:.3f} ms, "
                    "CPU work wall=n/a (coverage_incomplete), failures=0\n")
    runtime += "  TTFT=4.000 ms, TPOT=3.000 ms\n"
    return summary, runtime


class SummaryTests(unittest.TestCase):
    def test_numeric_metrics_and_coverage_notes(self):
        summary, runtime = evidence()
        self.assertEqual(len(M.validate_summary(summary, 3, runtime)), 4)

    def test_unavailable_cpu_is_not_success(self):
        for value, reason in ((None, "no_cpu_samples"), (0, None), (False, None),
                              (float("nan"), None), (1, "invalid_start")):
            with self.subTest(value=value, reason=reason):
                summary, runtime = evidence()
                summary["phases"][0].update(cpu_cycles=value, cpu_cycles_reason=reason)
                with self.assertRaises(M.CheckError):
                    M.validate_summary(summary, 3, runtime)

    def test_runtime_missing_or_different_is_not_success(self):
        summary, runtime = evidence()
        for output in ("", "Inference performance: n/a (no_cpu_samples)",
                       runtime.replace("CPU cycles=70000000", "CPU cycles=70000001"),
                       runtime.replace("TPOT=3.000", "TPOT=9.000")):
            with self.subTest(output=output[:80]), self.assertRaises(M.CheckError):
                M.validate_summary(summary, 3, output)

    def test_incomplete_or_failed_request_is_not_success(self):
        for key, value in (("tokens", 2), ("completed_requests", 0),
                           ("cpu_interval_coverage", "unverified_legacy_log"), ("tpot_gaps", 0)):
            summary, runtime = evidence()
            summary[key] = value
            with self.subTest(key=key), self.assertRaises(M.CheckError):
                M.validate_summary(summary, 3, runtime)
        summary, runtime = evidence()
        summary["phases"][1]["failed_operations"] = 1
        with self.assertRaises(M.CheckError):
            M.validate_summary(summary, 3, runtime)


class MatrixTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = M.json_read(ROOT / "scripts/experiment/cycle-matrix.json")
        for model in self.config["models"]:
            for bits in self.config["bits"]:
                for family in self.config["families"]:
                    filename = model["files"][0].format(bits=bits, family=family).split("/")[-1]
                    (self.root / filename).write_bytes(b"GGUF-fixture")

    def tearDown(self):
        self.temporary.cleanup()

    def test_default_matrix_is_all_27_cases(self):
        cases = M.input_cases(self.config, self.root, None, None, None)
        self.assertEqual(len(cases), 27)
        self.assertEqual(len({case["name"] for case in cases}), 27)
        self.assertTrue(all(not case["errors"] for case in cases))

    def test_missing_model_fails_without_fallback(self):
        (self.root / "gpt2.Q4_0.gguf").unlink()
        cases = M.input_cases(self.config, self.root, ["gpt2"], [4], ["0", "HP1"])
        self.assertEqual([case["status"] for case in cases], ["failed", "pending"])
        self.assertIn("found 0", cases[0]["errors"][0])

    def test_ambiguous_model_fails(self):
        subdir = self.root / "duplicate"
        subdir.mkdir()
        (subdir / "gpt2.Q8_H1.gguf").write_bytes(b"GGUF-fixture")
        case = M.input_cases(self.config, self.root, ["gpt2"], [8], ["H1"])[0]
        self.assertEqual(case["status"], "failed")
        self.assertIn("found 2", case["errors"][0])

    def test_invalid_gguf_and_unknown_selection_fail(self):
        (self.root / "gpt2.Q16_0.gguf").write_bytes(b"not-a-model")
        case = M.input_cases(self.config, self.root, ["gpt2"], [16], ["0"])[0]
        self.assertEqual(case["status"], "failed")
        with self.assertRaises(M.CheckError):
            M.input_cases(self.config, self.root, ["gpt22"], None, None)

    def test_command_failure_and_timeout_are_recorded(self):
        with self.assertRaises(M.CheckError):
            M.execute([sys.executable, "-c", "raise SystemExit(3)"], self.root,
                      "fail", os.environ.copy(), 5)
        self.assertTrue((self.root / "fail.command.json").is_file())
        with self.assertRaises(subprocess.TimeoutExpired):
            M.execute([sys.executable, "-c", "import time; time.sleep(60)"], self.root,
                      "timeout", os.environ.copy(), 1)

    def test_analyzer_requires_real_normalized_phases_and_all_traces(self):
        summary, _ = evidence()
        rows = self.root / "cycle-log.timeline.jsonl"
        rows.write_text(''.join(json.dumps({"row_type": "interval", "request_id": 1, "phase": phase}) + '\n'
                                for phase in ("prefill", "decode")), encoding="utf-8")
        report = {"status": "ok", "relationship_errors": [], "invalid_cycle_rows": 0,
                  "invalid_cycle_reasons": {}, "interval_rows": 2, "normalized_rows": 2,
                  "cpu_interval_coverage": {"status": "verified", "verified_operations": 6,
                                            "operation_end_records": 6, "unverified_sequence_operations": 0,
                                            "legacy_operation_ends": 0}, "traces": {}}
        for view in ("thread", "operator", "task", "stripe"):
            trace = self.root / f"cycle-log.timeline.{view}.chrome.json"
            trace.write_text('{"traceEvents":[]}', encoding="utf-8")
            report["traces"][view] = {"path": str(trace), "skipped_missing_lane": 0}
        self.assertEqual(M.validate_analyzer(report, rows, summary), {"prefill": 1, "decode": 1})
        for key, value in (("status", "invalid"), ("invalid_cycle_rows", 1), ("normalized_rows", 999)):
            bad = copy.deepcopy(report)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(M.CheckError):
                M.validate_analyzer(bad, rows, summary)
        report["cpu_interval_coverage"]["verified_operations"] = 5
        with self.assertRaises(M.CheckError):
            M.validate_analyzer(report, rows, summary)

    def test_partial_failure_does_not_report_whole_matrix_ok(self):
        cases = [{"name": "a", "status": "ok", "errors": []},
                 {"name": "b", "status": "failed", "errors": ["missing model"]}]
        report = {"scope": "full", "cases": cases}
        M.save_results(self.root, report)
        self.assertEqual(M.json_read(self.root / "results.json")["status"], "failed")
        self.assertIn("missing model", (self.root / "results.csv").read_text())


if __name__ == "__main__":
    unittest.main()
