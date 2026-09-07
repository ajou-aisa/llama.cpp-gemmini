#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "scripts/utils/capture_cpu_frequency.py"
FIXTURES = ROOT / "tests/fixtures/cycle-experiment"
TEGRAS = FIXTURES / "fake-tegrastats.py"
WORKLOAD = FIXTURES / "fake-workload.py"
HEADER = ["sample_index", "monotonic_ns", "elapsed_ns", "cpu_id", "util_percent", "frequency_mhz", "sample_valid", "raw_line"]


@dataclass(frozen=True)  # noqa: SLOTS_OK -- Python 3.9 compatibility
class Scenario:
    tegra: str = "happy"
    workload: str = "success"
    affinity: str = "normal"
    timeout: str = "2"


class FrequencyCaptureTests(unittest.TestCase):
    def invoke(self, directory: Path, scenario: Scenario = Scenario()) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update({"FAKE_TEGRASTATS_MODE": scenario.tegra, "FAKE_WORKLOAD_MODE": scenario.workload, "FAKE_WORKLOAD_SENTINEL": str(directory / "started"), "FAKE_TEGRASTATS_PID_FILE": str(directory / "tegra.pid"), "FAKE_WORKLOAD_PID_FILE": str(directory / "work.pid"), "FAKE_TEGRASTATS_DESCENDANT_PID_FILE": str(directory / "descendant.pid")})
        env["FAKE_AFFINITY_MODE"] = scenario.affinity
        env["FAKE_CAPTURE_TRACE"] = str(directory / "trace")
        return subprocess.run(self.command(directory, scenario.timeout), env=env, text=True, capture_output=True, timeout=8, check=False)

    @staticmethod
    def command(directory: Path, timeout: str) -> list[str]:
        return [sys.executable, str(CAPTURE), "--tegrastats", str(TEGRAS), "--expected-cpus", "0,1", "--raw-output", str(directory / "raw.txt"), "--csv-output", str(directory / "samples.csv"), "--affinity-output", str(directory / "affinity.csv"), "--ready-timeout", timeout, "--workload-timeout", timeout, "--", sys.executable, str(WORKLOAD)]

    def assert_dead(self, pid_path: Path) -> None:
        if not pid_path.exists():
            return
        with self.assertRaises(ProcessLookupError):
            os.kill(int(pid_path.read_text(encoding="utf-8")), 0)

    def test_system_python39_imports_cli(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            env = os.environ.copy()
            env["PYTHONPYCACHEPREFIX"] = str(Path(name) / "cache")
            result = subprocess.run(["/usr/bin/python3", str(CAPTURE), "--help"], env=env, text=True, capture_output=True, timeout=3, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_happy_sample_zero_mhz_and_affinity_when_ready(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            result = self.invoke(directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            with (directory / "samples.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(list(rows[0]), HEADER)
            self.assertEqual([(row["sample_index"], row["elapsed_ns"], row["cpu_id"], row["util_percent"], row["frequency_mhz"], row["sample_valid"], row["raw_line"]) for row in rows], [("0", "0", "0", "10", "729", "true", "RAM 1/2MB CPU [ 10%@729 , 0%@0 ] GR3D_FREQ 0%"), ("0", "0", "1", "0", "0", "true", "RAM 1/2MB CPU [ 10%@729 , 0%@0 ] GR3D_FREQ 0%")])
            self.assertTrue(all(row["monotonic_ns"].isdigit() for row in rows))
            self.assertEqual((directory / "trace").read_text(encoding="utf-8"), "collector_ready\nworkload_started\n")
            self.assertEqual((directory / "affinity.csv").read_text(encoding="utf-8"), "worker,cpu\n0,0\n1,1\n")
            self.assertRegex((directory / "raw.txt").read_text(encoding="utf-8"), r"^\d+\tRAM")
            self.assert_dead(directory / "tegra.pid")
            self.assert_dead(directory / "work.pid")

    def test_workload_failure_is_propagated_and_cleaned(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            result = self.invoke(directory, Scenario(workload="failure"))
            self.assertEqual(result.returncode, 9)
            self.assert_dead(directory / "tegra.pid")
            self.assert_dead(directory / "work.pid")

    def test_collector_failures_never_start_workload(self) -> None:
        for mode in ("exit", "malformed", "duplicate", "incomplete", "hang"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as name:
                directory = Path(name)
                result = self.invoke(directory, Scenario(tegra=mode, timeout="0.2" if mode == "hang" else "2"))
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((directory / "started").exists())
                self.assert_dead(directory / "tegra.pid")
                self.assert_dead(directory / "work.pid")

    def test_exited_collector_leader_terminates_same_group_descendant(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            result = self.invoke(directory, Scenario(tegra="exited_leader_descendant", timeout="0.2"))
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((directory / "started").exists())
            descendant = int((directory / "descendant.pid").read_text(encoding="utf-8"))
            try:
                os.kill(descendant, 0)
            except ProcessLookupError:
                return
            os.kill(descendant, signal.SIGKILL)
            self.fail(f"collector descendant {descendant} survived exited leader")

    def test_distinct_workers_claiming_one_cpu_fails(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            result = self.invoke(Path(name), Scenario(affinity="shared_cpu"))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("multiple workers", result.stderr.lower())

    def test_worker_remap_fails(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            result = self.invoke(Path(name), Scenario(affinity="remap"))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("conflicting", result.stderr.lower())

    def test_repeated_identical_affinity_regions_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            result = self.invoke(directory, Scenario(affinity="repeated_regions"))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((directory / "affinity.csv").read_text(encoding="utf-8"), "worker,cpu\n0,0\n1,1\n")

    def test_timeout_terminates_both_processes(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            result = self.invoke(directory, Scenario(workload="hang", timeout="0.2"))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("timeout", result.stderr.lower())
            self.assert_dead(directory / "tegra.pid")
            self.assert_dead(directory / "work.pid")

    def test_sigint_and_sigterm_terminate_both_processes(self) -> None:
        for requested_signal in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=requested_signal), tempfile.TemporaryDirectory() as name:
                directory = Path(name)
                env = os.environ.copy()
                env.update({"FAKE_TEGRASTATS_MODE": "happy", "FAKE_WORKLOAD_MODE": "hang", "FAKE_TEGRASTATS_PID_FILE": str(directory / "tegra.pid"), "FAKE_WORKLOAD_PID_FILE": str(directory / "work.pid")})
                process = subprocess.Popen(self.command(directory, "4"), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                self.assertIsNotNone(process.stdout)
                selector = selectors.DefaultSelector()
                selector.register(process.stdout, selectors.EVENT_READ)
                self.assertTrue(selector.select(3), "workload did not start")
                self.assertEqual(process.stdout.readline(), "WORKLOAD_STARTED\n")
                process.send_signal(requested_signal)
                self.assertEqual(process.wait(timeout=3), 128 + requested_signal)
                selector.close()
                process.stdout.close()
                if process.stderr is not None:
                    process.stderr.close()
                self.assert_dead(directory / "tegra.pid")
                self.assert_dead(directory / "work.pid")


if __name__ == "__main__":
    unittest.main()
