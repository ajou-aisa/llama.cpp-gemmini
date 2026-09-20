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
from types import SimpleNamespace
from unittest import mock

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
        self.assertEqual(M.json_read(self.root / "fail.result.json")["exit_code"], 3)
        with self.assertRaises(subprocess.TimeoutExpired):
            M.execute([sys.executable, "-c", "import time; time.sleep(60)"], self.root,
                      "timeout", os.environ.copy(), 1)
        self.assertEqual(M.json_read(self.root / "timeout.result.json")["error"], "TimeoutExpired")

    def test_json_rejects_duplicate_keys(self):
        path = self.root / "invalid.json"
        path.write_text('{"status":"failed","status":"ok"}', encoding="utf-8")
        with self.assertRaises(M.CheckError):
            M.json_read(path)

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
            M.json_write(trace, {"view": view, "traceEvents": [
                {"ph": "X", "ts": 0, "dur": 1}, {"ph": "X", "ts": 1, "dur": 1}]})
            report["traces"][view] = {"path": str(trace), "skipped_missing_lane": 0, "placed_intervals": 2}
        self.assertEqual(M.validate_analyzer(report, rows, summary), {"prefill": 1, "decode": 1})
        for key, value in (("status", "invalid"), ("invalid_cycle_rows", 1), ("normalized_rows", 999)):
            bad = copy.deepcopy(report)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(M.CheckError):
                M.validate_analyzer(bad, rows, summary)
        report["cpu_interval_coverage"]["verified_operations"] = 5
        with self.assertRaises(M.CheckError):
            M.validate_analyzer(report, rows, summary)
        report["cpu_interval_coverage"]["verified_operations"] = 6
        trace = Path(report["traces"]["thread"]["path"])
        for contents in ('{', '{"view":"thread","traceEvents":[]}',
                         '{"view":"thread","traceEvents":[{"ph":"X","ts":0,"dur":-1}]}'):
            trace.write_text(contents, encoding="utf-8")
            with self.subTest(contents=contents), self.assertRaises((M.CheckError, ValueError)):
                M.validate_analyzer(report, rows, summary)

    def test_partial_failure_does_not_report_whole_matrix_ok(self):
        cases = [{"name": "a", "status": "ok", "errors": []},
                 {"name": "b", "status": "failed", "errors": ["missing model"]}]
        report = {"scope": "full", "cases": cases}
        M.save_results(self.root, report)
        self.assertEqual(M.json_read(self.root / "results.json")["status"], "failed")
        self.assertIn("missing model", (self.root / "results.csv").read_text())


class FreshBuildTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.args = SimpleNamespace(build_root=str(self.root / "builds"), detail=0, jobs=2, timeout=5)
        self.base = {"GEMMINI_SW_PATH": ("PATH", "/fixture/include"),
                     "GGML_GEMMINI_ACTIVATION_BITS": ("STRING", "16")}
        self.build = None

    def tearDown(self):
        self.temporary.cleanup()

    def compile_fixture(self, command, logs, label, env, timeout):
        # Controlled compiler double. These files never run as model evidence.
        if label == "configure":
            self.build = Path(command[command.index("-B") + 1])
            self.assertEqual(list(self.build.iterdir()), [])
            config = [f"CMAKE_HOME_DIRECTORY:INTERNAL={M.ROOT}"]
            config += [arg[2:].replace("=", ":STRING=", 1) for arg in command if arg.startswith("-D")]
            (self.build / "CMakeCache.txt").write_text("\n".join(config) + "\n")
        else:
            self.assertEqual(label, "build")
            (self.build / "bin").mkdir()
            for name in ("llama-cli", "llama-cycle-summary"):
                path = self.build / "bin" / name
                path.write_text("synthetic-build-fixture")
                path.chmod(0o755)
        M.json_write(logs / f"{label}.result.json", {"success": True, "exit_code": 0})

    def logs(self, run, bits):
        path = self.root / "output" / run / f"build-a{bits}-w{bits}"
        path.mkdir(parents=True)
        return path

    def test_two_runs_always_build_all_widths_in_new_empty_directories(self):
        old = self.root / "builds/a8-w8-detail0/bin/llama-cli"
        old.parent.mkdir(parents=True)
        old.write_text("old-binary-must-not-be-used-or-deleted")
        paths = []
        with mock.patch.object(M, "execute", side_effect=self.compile_fixture) as execute:
            for run in ("first-run", "second-run"):
                for bits in (4, 8, 16):
                    path = M.prepare_build(self.args, self.base, bits, self.logs(run, bits))
                    paths.append(path)
                    self.assertEqual(M.cache_read(path)["GGML_GEMMINI_ACTIVATION_BITS"][1], str(bits))
                    self.assertEqual(M.cache_read(path)["GGML_GEMMINI_WEIGHT_BITS"][1], str(bits))
            self.assertEqual(execute.call_count, 12)  # configure + compile, six times
        self.assertEqual(len(set(paths)), 6)
        self.assertEqual(old.read_text(), "old-binary-must-not-be-used-or-deleted")

    def test_existing_run_directory_is_rejected_before_cmake(self):
        logs = self.logs("collision", 8)
        directory = self.root / "builds/collision/a8-w8-detail0"
        directory.mkdir(parents=True)
        with mock.patch.object(M, "execute") as execute, self.assertRaises(M.CheckError):
            M.prepare_build(self.args, self.base, 8, logs)
        execute.assert_not_called()

    def test_failed_configure_cannot_publish_build_receipt(self):
        logs = self.logs("failure", 4)
        with mock.patch.object(M, "execute", side_effect=M.CheckError("configure failed")) as execute:
            with self.assertRaises(M.CheckError):
                M.prepare_build(self.args, self.base, 4, logs)
        self.assertEqual(execute.call_count, 1)
        self.assertFalse((logs / "build.json").exists())

    def test_removed_no_build_switch_cannot_bypass_compilation(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/experiment/cycle_matrix.py"),
                                 "--no-build"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("unrecognized arguments", result.stderr)
        self.assertNotIn("CYCLE_MATRIX_OK cases=", result.stdout)


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "output/receipt-test"
        self.output.mkdir(parents=True)
        config = M.json_read(M.DEFAULT_CONFIG)
        cases = [{"name": f"{model}-Q{bits}_{family}", "model": model, "bits": bits,
                  "family": family, "status": "ok", "errors": [], **dict.fromkeys(M.CASE_GATES, True)}
                 for model, bits, family in sorted(M.matrix_keys(config))]
        self.report = {"scope": "full", "config": config, "cases": cases, "detail": 0,
                       "source": {"commit": "a" * 40, "sha256": "b" * 64}, "builds": []}
        for bits in config["bits"]:
            path = self.root / f"build/receipt-test/a{bits}-w{bits}-detail0"
            (path / "bin").mkdir(parents=True)
            for name in ("llama-cli", "llama-cycle-summary"):
                (path / "bin" / name).write_text("synthetic-receipt-fixture")
            (path / "CMakeCache.txt").write_text("test cache")
            logs = self.output / f"build-a{bits}-w{bits}"
            logs.mkdir()
            for step in ("configure", "build"):
                M.json_write(logs / f"{step}.result.json", {"success": True, "exit_code": 0})
            self.report["builds"].append({"bits": bits, "detail": 0, "fresh": True,
                                           "path": str(path), "logs": str(logs),
                                           "artifacts": M.build_artifacts(path)})

    def tearDown(self):
        self.temporary.cleanup()

    def finish(self):
        with mock.patch.object(M, "source_snapshot", return_value=self.report["source"]):
            return M.finish_run(self.output, self.report)

    def test_complete_matrix_publishes_one_bound_receipt(self):
        line, code = self.finish()
        self.assertEqual(code, 0)
        self.assertTrue(line.startswith("CYCLE_MATRIX_OK cases=27/27 builds=3/3 fresh=1 detail=0"))
        self.assertEqual((self.output / "OK.txt").read_text(), line + "\n")
        self.assertIn("report_sha256=" + M.file_hash(self.output / "results.json"), line)

    def test_every_gate_is_mandatory(self):
        for gate in M.CASE_GATES:
            with self.subTest(gate=gate):
                saved = copy.deepcopy(self.report)
                self.report["cases"][0][gate] = False
                line, code = self.finish()
                self.assertEqual(code, 1)
                self.assertTrue(line.startswith("CYCLE_MATRIX_FAILED"))
                self.assertFalse((self.output / "OK.txt").exists())
                self.report = saved

    def test_missing_case_is_selection_not_full_ok(self):
        self.report["cases"].pop()
        line, code = self.finish()
        self.assertEqual(code, 0)
        self.assertTrue(line.startswith("CYCLE_MATRIX_SELECTION_PASSED cases=26/26"))
        self.assertFalse((self.output / "OK.txt").exists())
        self.assertEqual(self.report["scope"], "selection")

    def test_failed_case_removes_old_success_receipt(self):
        self.finish()
        self.report["cases"][0].update(status="failed", errors=["PMU unavailable"])
        line, code = self.finish()
        self.assertEqual(code, 1)
        self.assertFalse((self.output / "OK.txt").exists())
        self.assertTrue(line.startswith("CYCLE_MATRIX_FAILED"))

    def test_source_change_prevents_ok(self):
        with mock.patch.object(M, "source_snapshot", return_value={"commit": "new", "sha256": "changed"}):
            line, code = M.finish_run(self.output, self.report)
        self.assertEqual(code, 1)
        self.assertIn("source changed", self.report["audit_errors"][0])
        self.assertFalse((self.output / "OK.txt").exists())

    def test_changed_binary_prevents_ok(self):
        path = Path(self.report["builds"][0]["path"]) / "bin/llama-cli"
        path.write_text("different binary")
        line, code = self.finish()
        self.assertEqual(code, 1)
        self.assertIn("artifacts changed", self.report["audit_errors"][0])

    def test_old_build_or_failed_build_prevents_ok(self):
        for change in ("run", "result", "fresh"):
            with self.subTest(change=change):
                saved = copy.deepcopy(self.report)
                build = self.report["builds"][0]
                if change == "run":
                    build["path"] = str(self.root / "build/old-run/a4-w4-detail0")
                elif change == "result":
                    M.json_write(Path(build["logs"]) / "build.result.json", {"success": False, "exit_code": 2})
                else:
                    build["fresh"] = False
                _, code = self.finish()
                self.assertEqual(code, 1)
                self.assertFalse((self.output / "OK.txt").exists())
                self.report = saved
                M.json_write(Path(self.report["builds"][0]["logs"]) / "build.result.json",
                             {"success": True, "exit_code": 0})

    def test_no_cases_or_missing_build_is_not_success(self):
        self.report["builds"].pop()
        self.assertEqual(self.finish()[1], 1)
        self.report["cases"] = []
        self.report["builds"] = []
        self.assertEqual(self.finish()[1], 1)
        self.assertFalse((self.output / "OK.txt").exists())


if __name__ == "__main__":
    unittest.main()
