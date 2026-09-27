# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: PYTHONPATH=. uv run pytest -q tests/test-evaluation-v3-stateful.py
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/eval"))

import certified_reconstruction as certified
import scheduled_endpoints as endpoints
from eval_common import Json, Record, sha256, write_json
from offline_pipeline import add_arguments


def inputs(root: Path) -> Record:
    result: Record = {"scenario": {"profile": "a8w8-d16-hp1", "initial_scratchpad_half": 0,
                                   "initial_accumulator_half": 0},
                      "stateful_evidence_root": {"path": str(root)}}
    for name in ("library", "npu_trace", "timing", "clock_selection", "cycle_certificate",
                 "run_aware_certificate", "stateful_sequence_certificate"):
        path = root / (name + ".json")
        write_json(path, {"selected_frequency_hz": 1_000_000_000} if name == "clock_selection" else {})
        result[name] = certified.artifact_reference(path)
    return result


def schedule(root: Path, storage: str) -> Path:
    zero: Record = {"numerator": 0, "denominator": 1}
    rows: list[Record] = [{"node_id": identity, "accepted_ns": zero, "result_ready_ns": zero}
                          for identity in ("application:request:begin", "application:prefill-prep:0", "dispatch:0:begin")]
    rows.extend({"node_id": f"application:sample:{index}",
                 "result_ready_ns": {"numerator": 100 + index, "denominator": 1}} for index in range(128))
    header: Record = {"schema": "im2p-execution-schedule" + ("-sqlite" if storage == "sqlite" else ""),
                      "version": 2, "scope": "RECONSTRUCTED", "service_validation_scope": "STATEFUL_SEQUENCE_PRODUCTION",
                      "service_binding": {"scope": "CURRENT_STATEFUL_SEQUENCE"}}
    path = root / ("schedule." + storage)
    if storage == "json":
        nodes: list[Json] = list(rows)
        write_json(path, {**header, "nodes": nodes})
    else:
        with sqlite3.connect(path) as database:
            database.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, body TEXT)")
            database.execute("CREATE TABLE results(identity TEXT PRIMARY KEY, body TEXT)")
            database.execute("INSERT INTO metadata VALUES('manifest', ?)", (json.dumps(header),))
            database.executemany("INSERT INTO results VALUES(?,?)", ((row["node_id"], json.dumps(row)) for row in rows))
    return path


def test_stateful_service_arguments_when_exact_inputs_bound(tmp_path: Path) -> None:
    # Given hash-bound inputs; this unit test proves forwarding, not production admission.
    bound = inputs(tmp_path)
    # When constructing the common schedule/verifier options.
    arguments = certified.service_arguments(bound)
    # Then the certificate/root are forwarded and the old certificate cannot shadow them.
    assert arguments[arguments.index("--stateful-sequence-certificate") + 1] == str(tmp_path / "stateful_sequence_certificate.json")
    assert arguments[arguments.index("--stateful-evidence-root") + 1] == str(tmp_path)
    assert "--service-certificate" not in arguments
    assert "--stateful-diagnostic" not in arguments


@pytest.mark.parametrize("storage", ["json", "sqlite"])
def test_v2_endpoint_reduction_when_typed_verifier_receipt_matches(tmp_path: Path, storage: str) -> None:
    # Given an endpoint fixture and a typed receipt (isolated reduction test, not certification).
    path = schedule(tmp_path, storage)
    receipt = endpoints.VerifiedStatefulSchedule(sha256(path), {"scope": "CURRENT_STATEFUL_SEQUENCE"})
    # When reducing after verification.
    result = endpoints.scheduled_application_result(path, [0], receipt)
    # Then exact fractions are preserved for both storage forms.
    assert result["ttft_ns"] == {"numerator": 100, "denominator": 1}
    assert result["tpot_ns"] == {"numerator": 1, "denominator": 1}


@pytest.mark.parametrize("storage", ["json", "sqlite"])
def test_v2_endpoint_rejection_when_marker_copied(tmp_path: Path, storage: str) -> None:
    # Given only copied production strings without a callable verifier result.
    path = schedule(tmp_path, storage)
    # When reducing without fresh verification, then no TTFT is returned.
    with pytest.raises(ValueError):
        endpoints.scheduled_application_result(path, [0])


def test_v2_endpoint_rejection_when_changed_after_verification(tmp_path: Path) -> None:
    # Given a receipt bound to the pre-tamper schedule.
    path = schedule(tmp_path, "json")
    receipt = endpoints.VerifiedStatefulSchedule(sha256(path), {"scope": "CURRENT_STATEFUL_SEQUENCE"})
    path.write_text(path.read_text() + "\n")
    # When reducing a changed file, then no endpoint survives the binding check.
    with pytest.raises(ValueError, match="verified stateful"):
        endpoints.scheduled_application_result(path, [0], receipt)


@pytest.mark.parametrize("clock_status", ["PASS", "DIAGNOSTIC_ONLY", "SYNTHETIC_ONLY"])
def test_fresh_official_verification_when_forged_clock(tmp_path: Path, clock_status: str) -> None:
    # Given fully bound files, but a copied clock marker rather than post-route evidence.
    bound = inputs(tmp_path)
    clock = tmp_path / "clock_selection.json"
    clock.write_text(json.dumps({"schema": "im2p-operating-clock", "version": 1,
                                 "status": clock_status, "selected_frequency_hz": 1_000_000_000}))
    bound["clock_selection"] = certified.artifact_reference(clock)
    bound["consumer_sources"] = certified.consumer_sources()
    path = schedule(tmp_path, "json")
    bundle = tmp_path / "bundle.json"
    write_json(bundle, {})
    # When the real official CLI is invoked, then its clock gate rejects before native use.
    with pytest.raises(ValueError, match="post-route operating clock"):
        certified.verify_official_schedule({"schedule": path, "bundle": bundle}, bound, ROOT.parent / "IM2P.sim")


def test_fresh_official_verification_when_consumer_source_stale(tmp_path: Path) -> None:
    # Given a stale wrapper/loader closure on an otherwise bound input record.
    bound = inputs(tmp_path)
    bound["consumer_sources"] = {"offline_pipeline.py": "0" * 64}
    # When loading, then source drift rejects before any external verifier is run.
    with patch("certified_reconstruction.subprocess.run") as run, pytest.raises(ValueError, match="consumer source"):
        certified.verify_official_schedule({}, bound, ROOT.parent / "IM2P.sim")
    run.assert_not_called()


@pytest.mark.parametrize("name", ["stateful_sequence_certificate", "library", "npu_trace", "timing"])
def test_service_arguments_reject_when_bound_input_changes(tmp_path: Path, name: str) -> None:
    # Given an input altered after its binding was recorded.
    bound = inputs(tmp_path)
    (tmp_path / (name + ".json")).write_text("{\"changed\":true}\n")
    # When forwarding exact bound inputs, then stale evidence never reaches scheduling.
    with pytest.raises(ValueError, match="artifact binding mismatch"):
        certified.service_arguments(bound)


def test_diagnostic_receipt_cannot_reduce_application_endpoints(tmp_path: Path) -> None:
    # Given a production label copied around an explicitly diagnostic binding.
    path = schedule(tmp_path, "json")
    receipt = endpoints.VerifiedStatefulSchedule(sha256(path), {"scope": "DIAGNOSTIC_STATEFUL_SEQUENCE"})
    # When requesting latency, then the diagnostic receipt is rejected.
    with pytest.raises(ValueError, match="verified stateful"):
        endpoints.scheduled_application_result(path, [0], receipt)


def test_mutually_exclusive_certificates_when_public_parser_used() -> None:
    # Given all required parser options and conflicting certificate branches.
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    required = [option for action in parser._actions if action.required for option in action.option_strings[:1]]
    arguments = [part for option in required for part in (option, "/unused")]
    arguments += ["--lifecycle", "/unused", "--service-certificate", "/old", "--stateful-sequence-certificate", "/new"]
    # When parsing, then argparse rejects the conflict with its normal nonzero exit.
    with pytest.raises(SystemExit) as error:
        parser.parse_args(arguments)
    assert error.value.code == 2


def test_public_cli_help_when_stateful_requested() -> None:
    # Given the real official wrapper executable.
    command = [sys.executable, "-B", str(ROOT / "scripts/eval/end_to_end.py"), "reconstruct", "--help"]
    # When the public parser is invoked.
    completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    # Then each stateful option is available.
    assert completed.returncode == 0
    assert all(option in completed.stdout for option in
               ("--stateful-sequence-certificate", "--stateful-evidence-root", "--stateful-diagnostic"))


def test_official_wrapper_rejects_before_stages_when_target_host_admission_missing(tmp_path: Path) -> None:
    # Given stateful production options without any validated target-host admission interface.
    output = tmp_path / "output"
    command = [sys.executable, "-B", str(ROOT / "scripts/eval/end_to_end.py"), "--output", str(output),
               "reconstruct", "--im2p", str(ROOT.parent / "IM2P.sim"), "--stateful-evidence-root", str(tmp_path)]
    for name in ("full-cpu-log", "full-cpu-graph", "full-cpu-provenance", "potal-log", "potal-graph", "potal-provenance",
                 "npu-trace", "library", "cycle-certificate", "run-aware-certificate", "application", "lifecycle",
                 "stateful-sequence-certificate"):
        command.extend(("--" + name, str(tmp_path / "unused.json")))
    # When the actual public wrapper runs, then it stops before replay or normal publication.
    completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    assert completed.returncode == 1
    assert "validated target-host/application admission is unavailable" in completed.stderr
    assert {item.name for item in output.iterdir()} == {"failure.json"}
