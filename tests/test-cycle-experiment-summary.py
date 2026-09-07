#!/usr/bin/env python3
"""Aggregate-contract tests for the six-cell cycle matrix summary."""
from __future__ import annotations
import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/utils/summarize_cycle_matrix.py"
FIXTURE = ROOT / "tests/fixtures/cycle-experiment/task-6-matrix-input.csv"
OUTPUTS = ("runs.csv", "matrix-summary.csv", "matrix-summary.md", "matrix-summary.svg", "failures.csv")
PYTHON39 = Path("/usr/bin/python3")

def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))

class MatrixSummaryTests(unittest.TestCase):
    def invoke(self, source: Path, output: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, str(SCRIPT), "--input", str(source), "--output", str(output), "--expected-repeats", "3"], cwd=ROOT, capture_output=True, text=True, check=False)

    def invoke_python39(self, arguments: List[str], cache: Path) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["PYTHONPYCACHEPREFIX"] = str(cache)
        return subprocess.run([str(PYTHON39), str(SCRIPT), *arguments], cwd=ROOT, env=environment, capture_output=True, text=True, check=False)

    def altered_fixture(self, directory: Path, transform: str) -> Path:
        text = FIXTURE.read_text(encoding="utf-8")
        path = directory / "input.csv"
        if transform == "partial":
            text = text.replace("q4-hp1,4,hp1,2,true,success,,100,cycle,1600,nanosecond,1000,1100,conflict_free,", "q4-hp1,4,hp1,2,true,failure,workload exit 9,,,,,,,,")
        elif transform == "all":
            lines = text.splitlines()
            text = lines[0] + "\n" + "\n".join(",".join((*line.split(",")[:5], "failure", "failed", "", "", "", "", "", "", "", "")) for line in lines[1:]) + "\n"
        elif transform == "missing":
            text = "\n".join(line for line in text.splitlines() if not line.startswith("q4-baseline,")) + "\n"
        elif transform == "affinity":
            text = text.replace(",900,1000,conflict_free,", ",,,conflict,worker 0 remapped", 1)
        elif transform == "mixed":
            text = text.replace("q4-hp1,4,hp1,3,true,success,,150,cycle,", "q4-hp1,4,hp1,3,true,success,,150,nanosecond,")
        elif transform in ("cross-cycle", "cross-wall"):
            lines = text.splitlines()
            column = 8 if transform == "cross-cycle" else 10
            replacement = "rtl_cycle" if transform == "cross-cycle" else "microsecond"
            changed = [line.split(",") for line in lines[1:]]
            text = lines[0] + "\n" + "\n".join(",".join(fields[:column] + [replacement] + fields[column + 1:]) if fields[0] == "q4-hp1" else ",".join(fields) for fields in changed) + "\n"
        elif transform == "precision":
            lines = text.splitlines()
            changed = [line.split(",") for line in lines[1:]]
            values = {"q16-baseline": ("100", "101", "1000"), "q16-hp1": ("50", "51", "500")}
            for fields in changed:
                if fields[0] in values:
                    value = values[fields[0]][int(fields[3]) - 1]
                    fields[7], fields[9] = value, value
            text = lines[0] + "\n" + "\n".join(",".join(fields) for fields in changed) + "\n"
        elif transform == "duplicate":
            text += text.splitlines()[1] + "\n"
        path.write_text(text, encoding="utf-8")
        return path

    def assert_artifacts(self, output: Path) -> None:
        self.assertEqual({path.name for path in output.iterdir()}, set(OUTPUTS))

    def test_happy_six_cells_have_exact_statistics_ratios_and_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            first, second = root / "first", root / "second"
            result = self.invoke(FIXTURE, first)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assert_artifacts(first)
            self.assertEqual(len(read_rows(first / "runs.csv")), 18)
            rows = read_rows(first / "matrix-summary.csv")
            self.assertEqual(len(rows), 6)
            q4_hp1 = next(row for row in rows if row["cell"] == "q4-hp1")
            self.assertEqual({key: q4_hp1[key] for key in ("expected_count", "success_count", "failure_count", "cycle_mean", "cycle_median", "cycle_min", "cycle_max", "cycle_p95", "cycle_hp1_baseline_ratio", "wall_hp1_baseline_ratio")}, {"expected_count": "3", "success_count": "3", "failure_count": "0", "cycle_mean": "100", "cycle_median": "100", "cycle_min": "50", "cycle_max": "150", "cycle_p95": "150", "cycle_hp1_baseline_ratio": "", "wall_hp1_baseline_ratio": "0.8"})
            self.assertEqual(read_rows(first / "failures.csv"), [])
            self.assertEqual(self.invoke(FIXTURE, second).returncode, 0)
            self.assertTrue(all((first / item).read_bytes() == (second / item).read_bytes() for item in OUTPUTS))

    def test_required_failures_are_complete_and_leave_artifacts(self) -> None:
        for scenario, expected in (("partial", 1), ("all", 18), ("missing", 3)):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                result = self.invoke(self.altered_fixture(root, scenario), root / "out")
                self.assertNotEqual(result.returncode, 0)
                self.assert_artifacts(root / "out")
                self.assertEqual(len(read_rows(root / "out/failures.csv")), expected)
                if scenario == "missing":
                    hp1 = next(row for row in read_rows(root / "out/matrix-summary.csv") if row["cell"] == "q4-hp1")
                    self.assertEqual(hp1["cycle_hp1_baseline_ratio"], "")

    def test_affinity_conflict_blanks_frequency_with_reason(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            result = self.invoke(self.altered_fixture(root, "affinity"), root / "out")
            self.assertEqual(result.returncode, 0, result.stderr)
            run = next(row for row in read_rows(root / "out/runs.csv") if row["cell"] == "q4-hp1" and row["repeat"] == "1")
            summary = next(row for row in read_rows(root / "out/matrix-summary.csv") if row["cell"] == "q4-hp1")
            self.assertEqual((run["frequency_mean_mhz"], run["frequency_reason"]), ("", "worker 0 remapped"))
            self.assertEqual((summary["frequency_mean_mean_mhz"], summary["frequency_reason"]), ("", "worker 0 remapped"))

    def test_required_failed_repeats_cannot_produce_trusted_cpu_statistics(self) -> None:
        for scenario in ("partial", "duplicate"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                altered = self.altered_fixture(root, scenario)
                rows = read_rows(altered)
                cell = "q4-hp1" if scenario == "partial" else rows[-1]["cell"]
                source = root / "sourced.csv"
                fields = list(rows[0]) + ["cycle_source", "wall_source", "cycle_status"]
                with source.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(dict(row, cycle_source="linux_perf_cpu_cycles",
                        wall_source="steady_clock", cycle_status="complete"
                        if row["status"] == "success" else "invalid") for row in rows)
                result = self.invoke(source, root / "out")
                self.assertNotEqual(result.returncode, 0)
                summary = read_rows(root / "out/matrix-summary.csv")
                failed = next(row for row in summary if row["cell"] == cell)
                self.assertNotEqual(failed["cycle_status"], "complete")
                for suffix in ("mean", "median", "min", "max", "p95"):
                    self.assertEqual(failed["cycle_" + suffix], "")
                hp1 = next(row for row in summary if row["width"] == failed["width"]
                           and row["variant"] == "hp1")
                self.assertEqual(hp1["cycle_hp1_baseline_ratio"], "")

    def test_mixed_units_and_duplicate_identity_fail_with_artifacts(self) -> None:
        for scenario, fragment in (("mixed", "mixed cycle units"), ("duplicate", "duplicate run identity")):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                result = self.invoke(self.altered_fixture(root, scenario), root / "out")
                self.assertNotEqual(result.returncode, 0)
                self.assert_artifacts(root / "out")
                self.assertTrue(any(fragment in row["reason"] for row in read_rows(root / "out/failures.csv")))

    def test_comparable_cross_cell_unit_mismatch_publishes_safe_complete_outputs(self) -> None:
        for scenario, prefix, reason in (("cross-cycle", "cycle", "cycle unit mismatch for width 4"), ("cross-wall", "wall", "wall unit mismatch for width 4")):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                first, second = root / "first", root / "second"
                source = self.altered_fixture(root, scenario)
                result = self.invoke(source, first)
                self.assertNotEqual(result.returncode, 0)
                self.assert_artifacts(first)
                rows = [row for row in read_rows(first / "matrix-summary.csv") if row["width"] == "4"]
                for row in rows:
                    self.assertEqual(tuple(row[prefix + suffix] for suffix in ("_unit", "_mean", "_median", "_min", "_max", "_p95")), ("", "", "", "", "", ""))
                hp1 = next(row for row in rows if row["variant"] == "hp1")
                self.assertEqual(hp1[prefix + "_hp1_baseline_ratio"], "")
                self.assertTrue(any(row["reason"].startswith(reason) for row in read_rows(first / "failures.csv")))
                self.assertEqual(self.invoke(source, second).returncode, 1)
                self.assertTrue(all((first / item).read_bytes() == (second / item).read_bytes() for item in OUTPUTS))

    def test_cycle_and_wall_ratios_use_unrounded_internal_means(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            result = self.invoke(self.altered_fixture(root, "precision"), root / "out")
            self.assertEqual(result.returncode, 0, result.stderr)
            hp1 = next(row for row in read_rows(root / "out/matrix-summary.csv") if row["cell"] == "q16-hp1")
            self.assertEqual((hp1["cycle_hp1_baseline_ratio"], hp1["wall_hp1_baseline_ratio"]), ("", "0.500416319734"))

    def test_actual_python39_import_help_and_representative_cli_paths(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            help_result = self.invoke_python39(["--help"], root / "cache")
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            happy = root / "happy"
            self.assertEqual(self.invoke_python39(["--input", str(FIXTURE), "--output", str(happy)], root / "cache").returncode, 0)
            self.assert_artifacts(happy)
            happy_row = next(row for row in read_rows(happy / "matrix-summary.csv") if row["cell"] == "q4-hp1")
            self.assertEqual((happy_row["cycle_mean"], happy_row["cycle_p95"], happy_row["cycle_hp1_baseline_ratio"]), ("100", "150", ""))
            for scenario in ("cross-cycle", "all"):
                output = root / (scenario + "-out")
                result = self.invoke_python39(["--input", str(self.altered_fixture(root, scenario)), "--output", str(output)], root / "cache")
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assert_artifacts(output)
                self.assertTrue(read_rows(output / "failures.csv"))

    def test_complete_source_metadata_allows_unrounded_cycle_ratios(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            legacy = self.altered_fixture(root, "precision")
            with legacy.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                fields = list(reader.fieldnames) + ["cycle_source", "wall_source", "cycle_status"]
                rows = list(reader)
            source = root / "complete.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(dict(row, cycle_source="linux_perf_cpu_cycles", wall_source="steady_clock", cycle_status="complete") for row in rows)
            result = self.invoke(source, root / "out")
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = next(row for row in read_rows(root / "out/matrix-summary.csv") if row["cell"] == "q16-hp1")
            self.assertEqual(summary["cycle_hp1_baseline_ratio"], "0.500416319734")
            self.assertEqual(summary["cycle_status"], "complete")

    def test_source_metadata_survives_and_same_unit_different_sources_cannot_compare(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root / "sourced.csv"
            with FIXTURE.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                fields = list(reader.fieldnames) + ["cycle_source", "wall_source", "cycle_status"]
                rows = list(reader)
            for row in rows:
                row.update(cycle_source="riscv_cycle" if row["cell"] == "q4-hp1" else "linux_perf_cpu_cycles",
                           wall_source="steady_clock", cycle_status="complete")
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            result = self.invoke(source, root / "out")
            self.assertEqual(result.returncode, 1)
            output = read_rows(root / "out/runs.csv")
            self.assertEqual(output[0]["cycle_source"], "linux_perf_cpu_cycles")
            summaries = read_rows(root / "out/matrix-summary.csv")
            q4 = next(row for row in summaries if row["cell"] == "q4-hp1")
            self.assertEqual((q4["cycle_mean"], q4["cycle_hp1_baseline_ratio"]), ("", ""))
            self.assertTrue(any("cycle source" in row["reason"] for row in read_rows(root / "out/failures.csv")))

    def test_empty_and_malformed_csv_fail_without_misleading_success(self) -> None:
        for content in (FIXTURE.read_text(encoding="utf-8").splitlines()[0] + "\n", "cell,width\nq4-baseline,nope\n"):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                source = root / "input.csv"
                source.write_text(content, encoding="utf-8")
                result = self.invoke(source, root / "out")
                self.assertNotEqual(result.returncode, 0)
                self.assert_artifacts(root / "out")
                self.assertTrue(read_rows(root / "out/failures.csv"))

if __name__ == "__main__":
    unittest.main()
