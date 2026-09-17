#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
# How to run: python3 tests/test-potal-csv.py --summary-binary build/bin/llama-cycle-summary
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]
SCRIPT: Final = ROOT / "scripts/utils/export_potal_csv.py"
LATENCY_METRICS: Final = ("ttft_ns", "request_elapsed_ns", "tokens", "tpot_ns")
COMMON: Final = {
    "execution_id", "request_id", "operation_id", "token_step", "phase", "run_id", "layer",
    "stripe_id", "slot", "node_id", "worker_id", "op", "metric", "value", "unit",
    "valid", "reason", "scope", "coverage", "source_file", "source_line",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-binary", type=Path,
                        default=ROOT / ".cache/exsia-submit-summary-20260915/bin/llama-cycle-summary")
    args = parser.parse_args()
    # Given: the public exporter CLI.
    # When: help is requested without any capture.
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                            capture_output=True, text=True, check=False)
    # Then: all required input/output options are documented.
    assert result.returncode == 0, result.stderr
    for option in ("--detail", "--summary-binary", "--output-dir"):
        assert option in result.stdout

    # Given: an explicit first matmul with zero IDs and unavailable native cycles.
    source_text = Path(__file__).with_name("potal-csv-fixture.jsonl").read_text(encoding="utf-8")
    records = [json.loads(line) for line in source_text.splitlines()]
    events = [record for record in records if record["record_type"] == "INFERENCE_EVENT"]
    resources = [record for record in records if record["record_type"] == "RESOURCE_SAMPLE"]
    stage, timeline, unidentified = (next(record for record in records if record.get("op") == op)
                                     for op in ("cpu.get_rows", "exsia.stripe_total", "cpu.unidentified"))
    context, between = stage["inference_context"], events[1]["inference_context"]
    mirror = {key: value for key, value in timeline.items() if key != "inference_context"}
    mirror_text = json.dumps(mirror) + "\n"
    mirror_text += json.dumps({**mirror, "op": "exsia.bridge_test"}) + "\n"
    workload_mirror = next(record for record in records if record["record_type"] == "EXSIA_WORKLOAD")
    mirror_text += json.dumps({key: value for key, value in workload_mirror.items() if key != "inference_context"}) + "\n"
    encoded = [json.dumps(record) for record in records]
    with tempfile.TemporaryDirectory(prefix="potal-csv-") as temporary:
        source = Path(temporary) / "main.jsonl"
        detail = Path(temporary) / "detail.jsonl"
        destination = Path(temporary) / "csv"
        source.write_text(source_text, encoding="utf-8")
        detail.write_text(mirror_text, encoding="utf-8")
        command = [sys.executable, str(SCRIPT), str(source), "--detail", str(detail),
                   "--summary-binary", str(args.summary_binary), "--output-dir", str(destination)]
        validator = subprocess.run([str(args.summary_binary), "--json", str(source)],
                                   capture_output=True, text=True, check=False)
        assert validator.returncode == 0, validator.stdout + validator.stderr
        assert json.loads(validator.stdout)["ttft_ns"] == 130
        # When: the public CLI exports main plus an identical context-free mirror.
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        # Then: all files preserve the explicit identity and independent measurement domains.
        assert result.returncode == 0, result.stdout + result.stderr
        tables = {}
        for name in ("cpu-stages", "npu-stages", "workload", "request-latency", "field-map"):
            with (destination / f"{name}.csv").open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if name != "field-map":
                    assert COMMON <= set(reader.fieldnames or ()), name
                else:
                    assert {"dataset", "metric", "source_field", "unit", "scope", "definition"} <= set(reader.fieldnames or ())
                tables[name] = list(reader)
        assert tables["field-map"]
        blocks = {(row["stripe_id"], row["metric"]): row for row in tables["workload"]
                  if row["record_type"] == "EXSIA_WORKLOAD"}
        for stripe_id, expected in (("0", ("10", "4", "6", "3", "1")),
                                    ("1", ("2", "1", "1", "0", "1")), ("2", ("0", "0", "0", "0", "0"))):
            assert tuple(blocks[(stripe_id, metric)]["value"] for metric in (
                "eligible_blocks", "regenerated_blocks", "reused_blocks", "forced_recomputed_blocks", "natural_regenerated_blocks")) == expected
        for stripe_id, expected in (("0", 0.4), ("1", 0.5)):
            rate = blocks[(stripe_id, "regeneration_rate")]
            assert float(rate["value"]) == expected and rate["unit"] == "fraction" and rate["valid"] == "true", rate
        zero = blocks[("2", "regeneration_rate")]
        assert (zero["value"], zero["valid"], zero["reason"]) == ("", "false", "zero_eligible_blocks"), zero
        global_blocks = {row["metric"]: row for row in tables["workload"] if row["scope"] == "observed_workload_summary"}
        assert tuple(global_blocks[metric]["value"] for metric in (
            "eligible_blocks", "regenerated_blocks", "reused_blocks", "forced_recomputed_blocks", "natural_regenerated_blocks")) == ("12", "5", "7", "3", "2")
        assert abs(float(global_blocks["regeneration_rate"]["value"]) - 5 / 12) < 1e-12
        packets = {(row["stripe_id"], row["metric"]): row for row in tables["workload"]
                   if row["scope"] == "observed_accepted_packets"}
        for metric, expected in (("required_planes", "8"), ("max_abs_residual", "2147483648"), ("plane_overflow", "0")):
            observed, legacy_packet = packets[("3", metric)], packets[("0", metric)]
            assert (observed["value"], observed["valid"], observed["reason"]) == (expected, "true", ""), observed
            assert (legacy_packet["value"], legacy_packet["valid"], legacy_packet["reason"]) == ("", "false", "counter_not_collected"), legacy_packet
        assert any(row["stripe_id"] == "3" and row["metric"] == "metrics.digit_bits" and row["value"] == "4"
                   for row in tables["workload"])
        diagnostic_rows = {row["metric"]: row for row in tables["cpu-stages"]
                           if row["record_type"] in {"CPU_WORK_SUMMARY", "RMD_STRIPE_TELEMETRY"}}
        for metric, expected in (
            ("cpu_workers.cycles", ("17", "cycle", "true", "")),
            ("cpu_workers.thread_cpu_ns", ("", "nanosecond", "false", "clock_regression")),
            ("host_stages.weight_gather.native_cycles", ("5", "cycle", "true", "")),
        ):
            row = diagnostic_rows[metric]
            assert (row["value"], row["unit"], row["valid"], row["reason"]) == expected, row
        rows = [row for row in tables["cpu-stages"] if row["op"] == "cpu.get_rows"]
        assert len(rows) == 3, rows
        measured = {row["metric"]: row for row in rows}
        assert set(measured) == {"native_cycles", "thread_cpu_ns", "wall_ns"}
        assert (measured["native_cycles"]["value"], measured["native_cycles"]["source"]) == ("", "")
        assert measured["native_cycles"]["valid"] == "false"
        assert measured["native_cycles"]["reason"] == "not_thread_cpu_counter"
        assert (measured["thread_cpu_ns"]["value"], measured["thread_cpu_ns"]["source"]) == ("15", "thread_cpu")
        assert (measured["wall_ns"]["value"], measured["wall_ns"]["source"]) == ("40", "steady_clock")
        mirrored_rows = [row for row in tables["cpu-stages"] if row["op"] == "exsia.stripe_total"]
        assert len(mirrored_rows) == 3, mirrored_rows
        bridged = [row for row in tables["cpu-stages"] if row["op"] == "exsia.bridge_test"]
        assert len(bridged) == 3, bridged
        assert all((row["run_id"], row["request_id"], row["operation_id"], row["token_step"]) == ("0", "1", "1", "0")
                   and Path(row["source_file"]) == detail for row in bridged), bridged
        assert all(row["token_step"] == "0" for row in rows + mirrored_rows)
        assert {row["metric"]: (row["value"], row["valid"]) for row in mirrored_rows} == {
            "wall_ns": ("40", "true"), "thread_cpu_ns": ("", "false"), "native_cycles": ("", "false")}, mirrored_rows
        missing_ids = [row for row in tables["cpu-stages"] if row["op"] == "cpu.unidentified"]
        assert len(missing_ids) == 3, missing_ids
        for row in missing_ids:
            assert all(row[key] == "" for key in ("run_id", "stripe_id", "slot", "node_id", "worker_id"))
        for row in rows:
            assert row["execution_id"] == "fixture-execution"
            assert (row["request_id"], row["operation_id"], row["phase"]) == ("1", "1", "prefill")
            assert all(row[key] == "0" for key in ("run_id", "stripe_id", "slot", "node_id", "worker_id"))
            assert row["coverage"] == "incomplete_no_record_count_contract"
            assert Path(row["source_file"]) == source and row["source_line"] == "4"
        npu = [row for row in tables["npu-stages"] if row["scope"] == "canonical_resource"]
        assert any(row["value"] == "11" and row["backend"] == "im2p" and row["domain"] == "dense"
                   and row["metric"] == "work" for row in npu), npu
        raw = next(row for row in tables["npu-stages"] if row["metric"] == "load_occupancy_cycles")
        assert (raw["value"], raw["unit"], raw["valid"], raw["reason"], raw["scope"]) == (
            "0", "cycle", "false", "device_counter_window_and_wrap_unverified", "diagnostic_nonadditive"), raw
        latency = [row for row in tables["request-latency"] if row["metric"] == "ttft_ns"]
        assert latency and all(row["value"] == "130" for row in latency), latency
        request = {row["metric"]: row for row in tables["request-latency"] if row["scope"] == "request_summary"}
        assert tuple(request[key]["value"] for key in LATENCY_METRICS) == ("130", "140", "1", ""), request
        assert request["tpot_ns"]["valid"] == "false"
        assert request["tpot_ns"]["reason"] == "fewer_than_two_tokens_per_request"
        assert all(row["request_id"] == "1" and row["operation_id"] == "" and row["phase"] == ""
                   for row in request.values()), request
        tokens = [row for row in tables["request-latency"] if row["metric"] == "token_ready_ns"]
        assert len(tokens) == 1 and tokens[0]["value"] == "230", tokens
        assert tokens[0]["token_index"] == "0" and tokens[0]["token_id"] == "0", tokens

        # Given: corrupted mirrors, malformed JSONL, and a missing counted resource.
        bad_inputs = [
            (source_text, json.dumps({**mirror, "cycles": 999}) + "\n"),
            (source_text, json.dumps({**mirror, "inference_context": None}) + "\n"),
            (source_text, json.dumps({**workload_mirror, "run_id": 77, "inference_context": context}) + "\n"),
            (source_text, json.dumps(mirror).replace("fixture-execution", "other-execution") + "\n"),
            (source_text + "{broken\n", mirror_text),
            (source_text, '{"record_type":"CPU_INTERVAL","record_type":"CPU_INTERVAL"}\n'),
            (source_text.rstrip("\n"), mirror_text),
            ("\n".join(line for line in encoded if line != json.dumps(resources[0])) + "\n", mirror_text),
            ("\n".join(encoded[:5] + encoded[4:]) + "\n", mirror_text),
            (source_text.replace(json.dumps(events[3]), json.dumps({**events[3], "token_step": 1})), mirror_text),
        ]
        for index, (bad_main, bad_detail) in enumerate(bad_inputs):
            source.write_text(bad_main, encoding="utf-8")
            detail.write_text(bad_detail, encoding="utf-8")
            # When: the same CLI consumes each damaged capture.
            rejected_output = Path(temporary) / f"invalid-{index}"
            result = subprocess.run(command[:-1] + [str(rejected_output)],
                                    capture_output=True, text=True, check=False)
            # Then: it reports failure rather than producing a plausible complete export.
            assert result.returncode != 0, (index, result.stdout, result.stderr)
            assert result.stderr, index

        # Given: one request has token readiness at 230, 250, and 290 ns, ending at 300 ns.
        decode = {**context, "operation_id": 2, "phase": "decode"}
        legacy = {**decode, "operation_id": 3}
        old = [{key: value for key, value in record.items() if key != "token_step"} for record in (events[2], events[3])]
        multi_events = events[:5] + [
            {**events[2], "event_sequence": 6, "timestamp_ns": 240, "inference_context": decode, "token_step": 1},
            {**timeline, "inference_context": decode, "run_id": 1,
             "host_timing": {**timeline["host_timing"], "start_ns": 240, "end_ns": 244, "duration_ns": 4}},
            {**events[3], "event_sequence": 7, "timestamp_ns": 245, "inference_context": decode, "token_step": 1, "resource_samples": 0},
            {**events[4], "event_sequence": 8, "timestamp_ns": 250, "token_index": 1, "token_id": 17},
            {**old[0], "event_sequence": 9, "timestamp_ns": 270, "inference_context": legacy},
            {**timeline, "inference_context": legacy, "run_id": 2,
             "host_timing": {**timeline["host_timing"], "start_ns": 270, "end_ns": 274, "duration_ns": 4}},
            {**old[1], "event_sequence": 10, "timestamp_ns": 280, "inference_context": legacy, "resource_samples": 0},
            {**events[4], "event_sequence": 11, "timestamp_ns": 290, "token_index": 2, "token_id": 42},
            {**events[5], "event_sequence": 12, "timestamp_ns": 300},
            {**events[1], "event_sequence": 13, "timestamp_ns": 400,
             "inference_context": {**between, "request_id": 2}},
            {**events[4], "event_sequence": 14, "timestamp_ns": 500, "token_id": 9,
             "inference_context": {**between, "request_id": 2}},
            {**events[5], "event_sequence": 15, "timestamp_ns": 550,
             "inference_context": {**between, "request_id": 2}},
            {**events[6], "event_sequence": 16, "timestamp_ns": 560},
        ]
        multiple = multi_events[:3] + [stage, timeline, unidentified] + resources + multi_events[3:]
        source.write_text("\n".join(json.dumps(record) for record in multiple) + "\n", encoding="utf-8")
        detail.write_text(json.dumps(mirror) + "\n", encoding="utf-8")
        multi_output = Path(temporary) / "multi-token"
        # When: the public CLI exports the multiple-token request.
        result = subprocess.run(command[:-1] + [str(multi_output)], capture_output=True, text=True, check=False)
        # Then: TPOT averages the two gaps, preserving request scope and observed token metadata.
        assert result.returncode == 0, result.stdout + result.stderr
        with (multi_output / "cpu-stages.csv").open(newline="", encoding="utf-8") as handle:
            joined = [row for row in csv.DictReader(handle) if row["op"] == "exsia.stripe_total"]
        assert len(joined) == 9, joined
        assert {(row["request_id"], row["operation_id"], row["token_step"]) for row in joined} == {
            ("1", "1", "0"), ("1", "2", "1"), ("1", "3", "")}, joined
        with (multi_output / "request-latency.csv").open(newline="", encoding="utf-8") as handle:
            latency_rows = list(csv.DictReader(handle))
        request = {row["metric"]: row for row in latency_rows
                   if row["scope"] == "request_summary" and row["request_id"] == "1"}
        assert tuple(request[key]["value"] for key in LATENCY_METRICS) == ("130", "200", "3", "30"), request
        assert request["tpot_ns"]["valid"] == "true" and request["tpot_ns"]["reason"] == ""
        second = {row["metric"]: row for row in latency_rows
                  if row["scope"] == "request_summary" and row["request_id"] == "2"}
        assert tuple(second[key]["value"] for key in LATENCY_METRICS) == ("100", "150", "1", ""), second
        assert second["tpot_ns"]["reason"] == "fewer_than_two_tokens_per_request"
        session = [row for row in latency_rows if row["scope"] == "session_summary" and row["metric"] == "ttft_ns"]
        assert len(session) == 1 and session[0]["value"] == "115" and session[0]["request_id"] == "", session
        tokens = [row for row in latency_rows if row["metric"] == "token_ready_ns"]
        assert [(row["request_id"], row["value"], row["token_index"], row["token_id"]) for row in tokens] == [
            ("1", "230", "0", "0"), ("1", "250", "1", "17"),
            ("1", "290", "2", "42"), ("2", "500", "0", "9")], tokens
        assert all(row["token_step"] == "" for row in tokens)
    print("Potal CSV CLI: PASS")


if __name__ == "__main__":
    main()
