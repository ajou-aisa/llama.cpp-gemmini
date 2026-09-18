#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/utils/cycle_timeline.py"


def line(record: dict) -> str:
    return json.dumps(record, separators=(",", ":")) + "\n"


def run(source: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(source), *args],
        capture_output=True, text=True, check=False,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cycle-timeline-") as temporary:
        root = Path(temporary)
        source = root / "cycle-log.jsonl"
        rows_path = root / "cycle-log.timeline.jsonl"
        thread_trace_path = root / "cycle-log.timeline.thread.chrome.json"
        operator_trace_path = root / "cycle-log.timeline.operator.chrome.json"

        inference = {"request_id": 1, "operation_id": 2, "phase": "decode", "included": True}
        operator = {
            "graph_id": 10, "operator_id": 11, "operator_kind": "MUL_MAT",
            "task_id": 12, "parent_task_id": 9, "segment_id": 13,
            "parent_segment_id": 12, "role": "dense",
        }
        records = [
            {
                "schema": "gemmini.cycle", "version": 2, "record_type": "INFERENCE_EVENT",
                "event": "session_start", "event_sequence": 1, "execution_id": "exec-1",
                "timestamp_ns": 1000,
            },
            {
                "schema": "gemmini.cycle", "version": 2, "record_type": "INFERENCE_EVENT",
                "event": "operation_start", "event_sequence": 2, "execution_id": "exec-1",
                "timestamp_ns": 1100, "inference_context": inference,
            },
            {
                "op": "cpu.mul_mat", "kind": "cpu", "layer": "blk.0",
                "run_id": 20, "node_id": 0, "worker_id": 3,
                "start": 100, "end": 160, "delta": 60,
                "ns_start": 1120, "ns_end": 1200, "tid": 42, "valid": True,
                "cpu_interval_sequence": 1,
                "operator_context": operator, "inference_context": inference,
            },
            {
                "op": "worker_queue_wait", "kind": "segment", "layer": "blk.0",
                "run_id": 20, "stripe_id": 0, "worker_id": 3,
                "start": 160, "end": 165, "delta": 5,
                "ns_start": 1200, "ns_end": 1450, "tid": 42, "valid": True,
                "operator_context": {**operator, "segment_id": 14, "parent_segment_id": 13},
                "inference_context": inference,
            },
            {
                "schema": "gemmini.cycle", "version": 2,
                "record_type": "IM2P_STRIPE_TELEMETRY", "source": "im2p_rtl",
                "unit": "rtl_cycle", "op": "im2p.execute", "layer": "blk.0",
                "run_id": 20, "stripe_id": 0, "slot": 0,
                "publish_cycle": 700, "completion_cycle": 900, "latency_cycles": 200,
                "inference_context": inference,
            },
            {
                "schema": "gemmini.cycle", "version": 2,
                "record_type": "IM2P_EXECUTION_TELEMETRY", "source": "im2p_rtl",
                "unit": "rtl_cycle", "op": "im2p.execute", "layer": "blk.0",
                "run_id": 20, "rtl_stripes_published": 1, "rtl_work_total_cycles": 300,
                "inference_context": inference,
            },
            {
                "schema": "gemmini.cycle", "version": 2, "record_type": "INFERENCE_EVENT",
                "event": "operation_end", "event_sequence": 3, "execution_id": "exec-1",
                "timestamp_ns": 1500, "cpu_interval_samples": 1, "inference_context": inference,
            },
        ]
        source.write_text("".join(line(record) for record in records), encoding="utf-8")

        result = run(source)
        assert result.returncode == 0, result.stdout + result.stderr
        summary = json.loads(result.stdout)
        assert summary["status"] == "ok"
        assert Path(summary["rows_output"]) == rows_path
        assert Path(summary["traces"]["thread"]["path"]) == thread_trace_path
        assert Path(summary["traces"]["operator"]["path"]) == operator_trace_path
        assert summary["traces"]["thread"]["placed_intervals"] == 2
        assert summary["traces"]["operator"]["placed_intervals"] == 2
        assert summary["records"] == 7
        assert summary["normalized_rows"] == 7
        assert summary["interval_rows"] == 2
        assert summary["event_rows"] == 3
        assert summary["device_rows"] == 2
        assert summary["unplaced_device_rows"] == 1
        assert summary["kinds"] == {"cpu": 1, "device": 1, "device_summary": 1, "event": 3, "segment": 1}
        assert summary["thread_count"] == 1
        assert summary["thread_ids"] == [42]
        assert summary["thread_ids_truncated"] is False
        assert not summary["relationship_errors"]
        assert summary["cpu_interval_coverage"] == {
            "status": "verified", "operation_end_records": 1,
            "operations_with_count_contract": 1, "verified_operations": 1,
            "legacy_operation_ends": 0, "unverified_sequence_operations": 0,
        }
        assert summary["stripe_coverage"] == {
            "status": "verified", "runs_with_count_contract": 1,
            "verified_runs": 1, "unverified_stripe_runs": 0, "stripe_rows": 1,
        }

        rows = [json.loads(value) for value in rows_path.read_text(encoding="utf-8").splitlines()]
        cpu = next(row for row in rows if row.get("op") == "cpu.mul_mat")
        wait = next(row for row in rows if row.get("op") == "worker_queue_wait")
        device = next(row for row in rows if row.get("row_type") == "device")
        assert (cpu["cycles"], cpu["wall_ns"], cpu["tid"]) == (60, 80, 42)
        assert (wait["cycles"], wait["wall_ns"]) == (5, 250)
        assert (device["cycle_start"], device["cycle_end"], device["cycles"]) == (700, 900, 200)
        assert "ns_start" not in device

        thread_trace = json.loads(thread_trace_path.read_text(encoding="utf-8"))
        spans = [event for event in thread_trace["traceEvents"] if event.get("ph") == "X"]
        assert thread_trace["view"] == "thread"
        assert len(spans) == 2
        assert {event["name"] for event in spans} == {"cpu.mul_mat", "worker_queue_wait"}
        assert all(event["tid"] == 42 for event in spans)
        assert not any(event.get("name") == "im2p.execute" for event in spans)

        operator_trace = json.loads(operator_trace_path.read_text(encoding="utf-8"))
        operator_spans = [event for event in operator_trace["traceEvents"] if event.get("ph") == "X"]
        operator_names = [event for event in operator_trace["traceEvents"] if event.get("name") == "thread_name"]
        assert operator_trace["view"] == "operator"
        assert len(operator_spans) == 2
        assert len({event["tid"] for event in operator_spans}) == 1
        assert any(event["args"]["name"] == "operator 11 MUL_MAT" for event in operator_names)

        all_views_dir = root / "all-views"
        result = run(source, "--all-views", "--trace-dir", str(all_views_dir))
        assert result.returncode == 0, result.stdout
        all_views = json.loads(result.stdout)["traces"]
        assert set(all_views) == {"thread", "operator", "task", "stripe"}
        assert all(Path(info["path"]).exists() for info in all_views.values())
        assert all_views["task"]["placed_intervals"] == 2
        assert all_views["stripe"]["placed_intervals"] == 1

        # Gemmini compact kind=cycle rows keep the shared ns timeline and invocation identity.
        scalar = root / "compact-cycle.jsonl"
        scalar_rows = root / "compact-cycle-rows.jsonl"
        scalar.write_text(line({
            "op": "dense_backend_host_call", "kind": "cycle", "layer": "blk.0",
            "matmul_invocation_id": 77, "run_id": 20,
            "start": 500, "end": 540, "delta": 40,
            "ns_start": 2000, "ns_end": 2100, "tid": 42, "valid": True,
        }), encoding="utf-8")
        result = run(scalar, "--rows", str(scalar_rows))
        assert result.returncode == 0, result.stdout
        scalar_row = json.loads(scalar_rows.read_text(encoding="utf-8"))
        assert scalar_row["matmul_invocation_id"] == 77
        assert scalar_row["op"] == "dense_backend_host_call"
        assert scalar_row["kind"] == "cycle"
        assert (scalar_row["wall_ns"], scalar_row["cycles"]) == (100, 40)

        # Invalid compact cycle arithmetic must fail before timeline export.
        bad_delta = root / "bad-delta.jsonl"
        bad_delta.write_text(line({
            "op": "cpu.add", "kind": "cpu", "start": 10, "end": 20, "delta": 7,
            "ns_start": 1, "ns_end": 2, "tid": 7, "valid": True,
        }), encoding="utf-8")
        result = run(bad_delta)
        assert result.returncode == 1
        assert "inconsistent endpoints/delta" in result.stdout

        # Global monotonic timeline regression is a hard hole.
        bad_ns = root / "bad-ns.jsonl"
        bad_ns.write_text(line({
            "op": "cpu.add", "kind": "cpu", "start": 10, "end": 20, "delta": 10,
            "ns_start": 9, "ns_end": 8, "tid": 7, "valid": True,
        }), encoding="utf-8")
        result = run(bad_ns)
        assert result.returncode == 1
        assert "shared timeline regressed" in result.stdout

        # Invalid PMU samples remain inspectable by default, but Nano experiment
        # validation can promote them to failure.
        invalid_cycle = root / "invalid-cycle.jsonl"
        invalid_cycle.write_text(line({
            "op": "cpu.add", "kind": "cpu", "start": 0, "end": 0, "delta": None,
            "ns_start": 1, "ns_end": 2, "tid": 7, "valid": False,
            "reason": "unavailable_event",
        }), encoding="utf-8")
        strict = run(invalid_cycle)
        assert strict.returncode == 2
        assert "invalid cycle counter" in strict.stdout
        relaxed = run(invalid_cycle, "--allow-invalid-cycles")
        assert relaxed.returncode == 0

        # Dropping one counted CPU interval is a hard cardinality hole.
        missing = root / "missing-counted-cpu.jsonl"
        missing.write_text("".join([
            line({
                "schema": "gemmini.cycle", "version": 2, "record_type": "INFERENCE_EVENT",
                "event": "operation_end", "execution_id": "exec-1", "timestamp_ns": 10,
                "cpu_interval_samples": 1, "inference_context": inference,
            }),
        ]), encoding="utf-8")
        result = run(missing)
        assert result.returncode == 2
        assert "CPU interval cardinality mismatch" in result.stdout

        # A dropped IM2P stripe is detected against rtl_stripes_published.
        missing_stripe = root / "missing-stripe.jsonl"
        missing_stripe.write_text("".join([
            line({
                "schema": "gemmini.cycle", "version": 2,
                "record_type": "IM2P_STRIPE_TELEMETRY", "source": "im2p_rtl",
                "op": "im2p.execute", "layer": "blk.0", "run_id": 31,
                "stripe_id": 0, "slot": 0, "publish_cycle": 10,
                "completion_cycle": 20, "latency_cycles": 10,
            }),
            line({
                "schema": "gemmini.cycle", "version": 2,
                "record_type": "IM2P_EXECUTION_TELEMETRY", "source": "im2p_rtl",
                "op": "im2p.execute", "layer": "blk.0", "run_id": 31,
                "rtl_stripes_published": 2, "rtl_work_total_cycles": 40,
            }),
        ]), encoding="utf-8")
        result = run(missing_stripe)
        assert result.returncode == 2
        assert "IM2P stripe cardinality mismatch" in result.stdout

        # Reusing a structural segment ID for another owner is an assembly hole.
        reused = root / "reused.jsonl"
        reused.write_text("".join([
            line({
                "op": "a", "kind": "segment", "start": 1, "end": 2, "delta": 1,
                "ns_start": 1, "ns_end": 2, "tid": 8, "valid": True,
                "operator_context": {"task_id": 4, "segment_id": 9, "operator_id": 5},
            }),
            line({
                "op": "b", "kind": "segment", "start": 2, "end": 3, "delta": 1,
                "ns_start": 2, "ns_end": 3, "tid": 8, "valid": True,
                "operator_context": {"task_id": 6, "segment_id": 9, "operator_id": 7},
            }),
        ]), encoding="utf-8")
        result = run(reused)
        assert result.returncode == 2
        assert "segment_id 9 reused" in result.stdout

    print("cycle timeline: PASS")


if __name__ == "__main__":
    main()
