#!/usr/bin/env python3
"""Schema-only tests for cycle experiment JSONL data."""

import csv
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.utils.cycle_schema import (
    CycleSchemaError,
    RecordType,
    parse_cycle_jsonl,
    parse_cycle_line,
)


class CycleSchemaTests(unittest.TestCase):
    def test_all_repository_v2_families_stream_as_frozen_typed_records(self) -> None:
        fixture = ROOT / "tests/fixtures/cycle-experiment/all-families.jsonl"

        records = tuple(parse_cycle_jsonl(fixture))

        self.assertEqual({record.record_type for record in records}, set(RecordType))
        self.assertEqual(len(records), 12)
        interval = records[0]
        self.assertEqual(interval.worker_id, 0)
        self.assertIsNone(interval.stripe_id)
        self.assertEqual(interval.delta, 5)
        self.assertTrue(interval.valid)
        rmd = records[6]
        self.assertEqual(
            rmd.nested_json("dispatch"),
            '{"direct_calls":1,"direct_events":2,"packet_calls":0,"ws_calls":0}',
        )
        with self.assertRaises(AttributeError):
            setattr(interval, "op", "changed")

    def test_pipeline_rejects_each_reversed_timing_pair(self) -> None:
        fixture = ROOT / "tests/fixtures/cycle-experiment/all-families.jsonl"
        pipeline = fixture.read_text(encoding="utf-8").splitlines()[5]
        timing_pairs = (
            ("queue_start_ns", "queue_end_ns"),
            ("dense_start_ns", "dense_end_ns"),
            ("rmd_start_ns", "rmd_end_ns"),
            ("compose_start_ns", "compose_end_ns"),
            ("finalize_start_ns", "finalize_end_ns"),
        )
        for start_name, end_name in timing_pairs:
            with self.subTest(start=start_name):
                reversed_pair = pipeline.replace(
                    f'"{start_name}":', f'"{start_name}":10,"unused_original_start":'
                )
                with self.assertRaises(CycleSchemaError) as caught:
                    parse_cycle_line(reversed_pair, 17)
                self.assertEqual(caught.exception.line_number, 17)

    def test_invalid_interval_requires_null_delta_and_nonempty_reason(self) -> None:
        fixture = ROOT / "tests/fixtures/cycle-experiment/all-families.jsonl"
        interval = fixture.read_text(encoding="utf-8").splitlines()[0]
        invalid = interval.replace('"delta":5,"valid":true', '"delta":null,"valid":false')
        malformed_rows = (
            interval.replace('"end":15,"delta":5,"valid":true', '"end":10,"delta":0,"valid":false'),
            invalid,
            invalid[:-1] + ',"reason":null}',
            invalid[:-1] + ',"reason":7}',
            invalid[:-1] + ',"reason":""}',
        )
        for row in malformed_rows:
            with self.subTest(row=row):
                with self.assertRaises(CycleSchemaError) as caught:
                    parse_cycle_line(row, 23)
                self.assertEqual(caught.exception.line_number, 23)

        equal_endpoints = interval.replace('"end":15,"delta":5', '"end":10,"delta":0')
        self.assertEqual(parse_cycle_line(equal_endpoints, 24).delta, 0)

    def test_invalid_inputs_report_the_exact_line(self) -> None:
        fixture_names = (
            "malformed.jsonl",
            "schema-drift.jsonl",
            "duplicate-key.jsonl",
            "missing-field.jsonl",
            "bad-arithmetic.jsonl",
        )
        for fixture_name in fixture_names:
            with self.subTest(fixture=fixture_name):
                with self.assertRaises(CycleSchemaError) as caught:
                    tuple(parse_cycle_jsonl(ROOT / "tests/fixtures/cycle-experiment" / fixture_name))
                self.assertEqual(caught.exception.line_number, 2)

    def test_cli_failure_reports_line_and_returns_nonzero(self) -> None:
        fixture = ROOT / "tests/fixtures/cycle-experiment/malformed.jsonl"

        result = subprocess.run(
            [sys.executable, "scripts/utils/cycle_schema.py", "--check", str(fixture)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("line 2:", result.stderr)


class CycleCsvTests(unittest.TestCase):
    def test_export_preserves_null_presence_nested_and_unknown_fields(self) -> None:
        output = ROOT / "tests/fixtures/cycle-experiment/csv-output"
        output.mkdir(exist_ok=True)
        for path in output.glob("*"):
            path.unlink()
        result = subprocess.run(
            [sys.executable, "scripts/utils/cycle_log_to_csv.py",
             "tests/fixtures/cycle-experiment/csv-lossless.jsonl", str(output)],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        csv_file = output / "RMD_BACKEND_TELEMETRY.csv"
        with csv_file.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        values = dict(zip(rows[0], rows[1]))
        dispatch = "6469737061746368"
        nested = "756e6b6e6f776e5f6e6573746564"
        layer = "6c61796572"
        self.assertEqual(values["field:" + dispatch], '{"a":1,"z":[null]}')
        self.assertEqual(values["field:" + nested], "")
        self.assertEqual(values["present:" + nested], "true")
        self.assertEqual(values["present:" + layer], "true")

    def test_adversarial_keys_have_bijective_collision_free_headers(self) -> None:
        output = ROOT / "tests/fixtures/cycle-experiment/csv-output"
        for path in output.glob("*"):
            path.unlink()
        result = subprocess.run(
            [sys.executable, "scripts/utils/cycle_log_to_csv.py",
             "tests/fixtures/cycle-experiment/csv-lossless.jsonl", str(output)],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        with (output / "RMD_BACKEND_TELEMETRY.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows[0]), len(set(rows[0])))
        decoded = {}
        for header, value in rows[0].items():
            if header.startswith("field:"):
                key = bytes.fromhex(header[6:]).decode("utf-8")
                decoded[key] = (value, rows[0]["present:" + header[6:]])
        self.assertEqual(decoded["probe"], ("", "true"))
        self.assertEqual(decoded["probe_present"], ('"source-value"', "true"))
        self.assertEqual(decoded["unicode-µ"], ('"value"', "true"))
        self.assertEqual(decoded["delimiter,quote"], ('"x"', "true"))

    def test_scalar_unknown_fields_round_trip_json_types(self) -> None:
        output = ROOT / "tests/fixtures/cycle-experiment/csv-output"
        for path in output.glob("*"):
            path.unlink()
        result = subprocess.run(
            [sys.executable, "scripts/utils/cycle_log_to_csv.py",
             "tests/fixtures/cycle-experiment/csv-lossless.jsonl", str(output)],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        with (output / "RMD_BACKEND_TELEMETRY.csv").open(encoding="utf-8", newline="") as stream:
            row = next(csv.DictReader(stream))
        integer_key = "field:756e6b6e6f776e5f696e7465676572"
        string_key = "field:756e6b6e6f776e5f737472696e67"
        self.assertIs(type(json.loads(row[integer_key])), int)
        self.assertIs(type(json.loads(row[string_key])), str)
        self.assertEqual(json.loads(row[integer_key]), 1)
        self.assertEqual(json.loads(row[string_key]), "1")

    def test_existing_destination_collision_preflights_all_publications(self) -> None:
        output = ROOT / "tests/fixtures/cycle-experiment/csv-collision-output"
        output.mkdir(exist_ok=True)
        for path in output.iterdir():
            path.unlink()
        sentinel = output / "STAGE.csv"
        sentinel.write_text("sentinel\\n", encoding="utf-8")
        foreign = output / ".foreign.tmp"
        foreign.write_bytes(b"owned by another invocation\\n")
        result = subprocess.run(
            [sys.executable, "scripts/utils/cycle_log_to_csv.py",
             "tests/fixtures/cycle-experiment/all-families.jsonl", str(output)],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "sentinel\\n")
        self.assertEqual(foreign.read_bytes(), b"owned by another invocation\\n")
        self.assertEqual(sorted(path.name for path in output.iterdir()), [".foreign.tmp", "STAGE.csv"])

    def test_late_malformed_input_publishes_no_csv_or_temps(self) -> None:
        output = ROOT / "tests/fixtures/cycle-experiment/csv-failure-output"
        output.mkdir(exist_ok=True)
        for path in output.iterdir():
            path.unlink()
        result = subprocess.run(
            [sys.executable, "scripts/utils/cycle_log_to_csv.py",
             "tests/fixtures/cycle-experiment/csv-malformed-late.jsonl", str(output)],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(list(output.iterdir()), [])


if __name__ == "__main__":
    for mode in ("--schema-only", "--csv-only"):
        if mode in sys.argv:
            sys.argv.remove(mode)
    unittest.main()
