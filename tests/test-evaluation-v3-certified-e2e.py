#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 -B tests/test-evaluation-v3-certified-e2e.py
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/eval"))

from application_results import (aggregate_results, application_result, application_services,
                                 load_measurement, load_potal_collection)
from certified_reconstruction import artifact_reference, reconstructed_row
from eval_common import Record, sha256
from scheduled_endpoints import scheduled_application_result


def schedule_rows() -> list[dict[str, object]]:
    start = {"numerator": 11, "denominator": 2}
    zero = {"numerator": 0, "denominator": 1}
    rows: list[dict[str, object]] = [
        {"node_id": "application:request:begin", "accepted_ns": zero, "result_ready_ns": zero},
        {"node_id": "application:prefill-prep:0", "accepted_ns": zero,
         "result_ready_ns": {"numerator": 4, "denominator": 1}},
        {"node_id": "dispatch:0:begin", "accepted_ns": start, "result_ready_ns": start}]
    rows.extend({"node_id": f"application:sample:{index}",
                 "result_ready_ns": {"numerator": 100 + 3 * index, "denominator": 2}}
                for index in range(128))
    return rows


def potal_fixture(root: Path, pipeline: bool = True) -> tuple[Path, Path, Path, Path]:
    run = root / "repetition-00"
    (run / "native").mkdir(parents=True)
    endpoint = {"schema": "potal-application-endpoints", "version": 1, "source_role": "potal_collection",
                "chunk_id": 0, "workload": "E2E_GENERATION_256_128", "complete": True, "cost_only": False,
                "samples": 128, "decode_calls": 127, "warmup": 0, "excludes_terminal_io": True,
                "timing_source": "steady_clock", "timing_unit": "ns", "t0_ns": 1,
                "sample_accept_ns": list(range(2, 130)), "generated_tokens": list(range(128))}
    endpoint_path = run / "native/application.jsonl"
    endpoint_path.write_text(json.dumps(endpoint) + "\n")
    measured = application_result(endpoint)
    service_path = run / "native/application-cpu.jsonl"
    preparation = [{"schema": "potal-application-cpu", "version": 2,
        "source_role": "potal_collection", "stage": "prefill_batch_prepare", "batch_index": 0,
        "dispatch_id": 0, "sample_index": None, "token_id": None, "chunk_id": 0,
        "phase": "prefill", "decode_index": None, "cpu_work_cycles_unit": "cycle",
        "host_elapsed_valid": True, "host_start_ns": 1, "host_end_ns": 2, "host_elapsed_ns": 1,
        "cpu_work_cycles_valid": False, "cpu_work_cycles": None,
        "cpu_work_cycles_reason": "fixture"}] if pipeline else []
    samples = [{"schema": "potal-application-cpu", "version": 2 if pipeline else 1,
        "source_role": "potal_collection", "stage": "sample_accept", "sample_index": index,
        "token_id": index, "chunk_id": 0, "phase": "prefill" if index == 0 else "decode",
        "decode_index": None if index == 0 else index - 1, "cpu_work_cycles_unit": "cycle",
        "host_elapsed_valid": False, "host_elapsed_ns": None, "host_elapsed_reason": "fixture",
        "cpu_work_cycles_valid": False, "cpu_work_cycles": None,
        "cpu_work_cycles_reason": "fixture"} for index in range(128)]
    service_path.write_text("".join(json.dumps(row) + "\n" for row in [*preparation, *samples]))
    proof_path = run / "collection-provenance.json"
    proof_path.write_text(json.dumps({"schema": "im2p-collection-provenance", "version": 2,
        "source_role": "POTAL_COLLECTION", "collection_success": True,
        "actual_sampler_calls": 128, "decode_calls": 127, "chunk_id": 0,
        "input_tokens_sha256": "input", "output_tokens_sha256": measured["generated_tokens_sha256"],
        "recipe_id": "recipe", "model_sha256": "model",
        "artifacts": {"application_endpoints": {"sha256": sha256(endpoint_path)},
                      "application_cpu": {"sha256": sha256(service_path)}}}))
    join_path = root / "join-summary.json"
    join_path.write_text(json.dumps({"status": "PASS", "scope": "structural-three-source-reconstruction",
        "model_sha256": "model",
        "potal_provenance_sha256": sha256(proof_path),
        "decode_token_fingerprint_matches": {str(index): True for index in range(127)}}))
    (root / "request.json").write_text(json.dumps({"model_sha256": "model", "dataset_sha256": "dataset",
        "host": {"host_id": "host"}, "build_info": {"activation_metrics": 0, "residual_metrics": 0,
            "backend": "IM2P_SIM", "hp1": True, "cycle_sim": 1, "log_cycle": 1}}))
    result_path = run / "result.json"
    result_path.write_text(json.dumps({"schema": "potal-e2e-run", "version": 1, "role": "potal",
        "measurement_kind": "COLLECTION_OBSERVATION_ONLY", "chunk_id": 0,
        "input_tokens_sha256": "input", "model_sha256": "model", "dataset_sha256": "dataset",
        "host_id": "host", "recipe_id": "recipe",
        "application_sha256": sha256(endpoint_path),
        "generated_tokens_sha256": measured["generated_tokens_sha256"],
        "collection_observation": measured,
        "application_services": application_services(service_path, endpoint),
        "collection_provenance": {"sha256": sha256(proof_path)}}))
    return result_path, service_path, proof_path, join_path


class CertifiedE2ETests(unittest.TestCase):
    def test_official_cli_exposes_certified_inputs(self) -> None:
        # Given the public reconstruct command.
        command = [sys.executable, "-B", str(ROOT / "scripts/eval/end_to_end.py"), "reconstruct", "--help"]
        # When inspecting its actual parser.
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10)
        # Then a caller can supply all externally certified inputs.
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for option in ("--service-certificate", "--clock-selection", "--profile", "--potal-result",
                       "--timing", "--initial-scratchpad-half", "--initial-accumulator-half"):
            self.assertIn(option, completed.stdout)

    def test_schedule_uses_request_start_before_prefill_prep_and_exact_128_samples(self) -> None:
        # Given a reconstructed request start and measured prep before the first dispatch.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schedule.json"
            path.write_text(json.dumps({"schema": "im2p-execution-schedule", "version": 1,
                                        "scope": "RECONSTRUCTED", "service_validation_scope": "CURRENT_CERTIFIED_SEQUENCE",
                                        "nodes": schedule_rows()}))
            # When reducing its application endpoints.
            result = scheduled_application_result(path, [0])
            # Then TTFT includes prep, and TPOT uses all 127 decode intervals.
            self.assertEqual(result["ttft_ns"], {"numerator": 50, "denominator": 1})
            self.assertEqual(result["tpot_ns"], {"numerator": 3, "denominator": 2})
            self.assertEqual(result["samples"], 128)
            self.assertEqual(result["decode_calls"], 127)

    def test_synthetic_or_incomplete_schedule_has_no_application_result(self) -> None:
        # Given an otherwise plausible schedule with diagnostic scope or one absent sample.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schedule.json"
            for scope, rows in (("SYNTHETIC", schedule_rows()),
                                ("RECONSTRUCTED", schedule_rows()[:-1]),
                                ("RECONSTRUCTED", schedule_rows()[1:]),
                                ("RECONSTRUCTED", [*schedule_rows()[:1], *schedule_rows()[2:]])):
                with self.subTest(scope=scope, count=len(rows)):
                    path.write_text(json.dumps({"schema": "im2p-execution-schedule", "version": 1,
                                                "scope": scope, "service_validation_scope": "CURRENT_CERTIFIED_SEQUENCE",
                                                "nodes": rows}))
                    # When reducing application endpoints, then publication fails closed.
                    with self.assertRaises(ValueError):
                        scheduled_application_result(path, [0])

    def test_sqlite_schedule_keeps_the_same_rational_endpoints(self) -> None:
        # Given the official SQLite schedule storage shape with 128 samples.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schedule.sqlite"
            with sqlite3.connect(path) as database:
                database.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, body TEXT)")
                database.execute("CREATE TABLE results(identity TEXT PRIMARY KEY, body TEXT)")
                database.execute("INSERT INTO metadata VALUES(?,?)", ("manifest", json.dumps({
                    "schema": "im2p-execution-schedule-sqlite", "version": 1, "scope": "RECONSTRUCTED",
                    "service_validation_scope": "CURRENT_CERTIFIED_SEQUENCE"})))
                database.executemany("INSERT INTO results VALUES(?,?)",
                                     ((row["node_id"], json.dumps(row)) for row in schedule_rows()))
            # When reducing the stored schedule, then endpoint fractions match JSON.
            result = scheduled_application_result(path, [0])
            self.assertEqual(result["ttft_ns"], {"numerator": 50, "denominator": 1})
            self.assertEqual(result["tpot_ns"], {"numerator": 3, "denominator": 2})

    def test_ten_reconstructed_ttft_values_keep_rational_precision(self) -> None:
        # Given ten distinct official run values with half-nanosecond TTFT.
        rows: list[Record] = [dict(repetition=index, chunk_id=index, measurement_id=str(index),
                     application_sha256=str(index), host_id="host", role="potal",
                     comparison_contract="same", measurement_kind="VALIDATED_RECONSTRUCTION",
                     ttft_ns={"numerator": 2 * index + 1, "denominator": 2},
                     tpot_ns={"numerator": 3, "denominator": 2}) for index in range(10)]
        # When aggregating by existing median-of-ten policy.
        result = aggregate_results(rows)
        # Then the two central run values produce an exact median.
        self.assertEqual(result["ttft_ns"], {"numerator": 5, "denominator": 1})
        self.assertEqual(result["tpot_ns"], {"numerator": 3, "denominator": 2})

    def test_potal_collection_requires_bound_native_endpoints_and_services(self) -> None:
        # Given one complete PoTal collection, its native sidecars, and the official join summary.
        with tempfile.TemporaryDirectory() as directory:
            result_path, service_path, proof_path, join_path = potal_fixture(Path(directory))
            # When loading the bound source, then its real 128-sample identity is admitted.
            source = load_potal_collection(result_path, service_path, proof_path, join_path)
            self.assertEqual(source["role"], "potal")

    def test_potal_collection_rejects_changed_sampler_sidecar(self) -> None:
        # Given a complete collection whose sampler sidecar changes after provenance.
        with tempfile.TemporaryDirectory() as directory:
            result_path, service_path, proof_path, join_path = potal_fixture(Path(directory))
            service_path.write_text("{}\n")
            # When the sampler sidecar changes, then the source is rejected.
            with self.assertRaises(ValueError):
                load_potal_collection(result_path, service_path, proof_path, join_path)

    def test_old_sample_only_potal_collection_cannot_certify(self) -> None:
        # Given an older PoTal sampler stream without source-owned prefill preparation.
        with tempfile.TemporaryDirectory() as directory:
            result_path, service_path, proof_path, join_path = potal_fixture(Path(directory), pipeline=False)
            # When loading as a certified source, then the missing preparation proof rejects it.
            with self.assertRaisesRegex(ValueError, "prefill preparation"):
                load_potal_collection(result_path, service_path, proof_path, join_path)

    def test_invalid_prefill_preparation_interval_cannot_certify(self) -> None:
        # Given a v2 PoTal source with an invalid measured prefill preparation interval.
        with tempfile.TemporaryDirectory() as directory:
            result_path, service_path, proof_path, join_path = potal_fixture(Path(directory))
            rows = [json.loads(line) for line in service_path.read_text().splitlines()]
            rows[0]["host_elapsed_valid"] = False
            service_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            # When the native collection is bound, then invalid preparation timing rejects publication.
            with self.assertRaisesRegex(ValueError, "invalid prefill preparation"):
                load_potal_collection(result_path, service_path, proof_path, join_path)

    def test_standalone_loader_rejects_forged_clock_and_service_certificate(self) -> None:
        # Given a plausible run row whose clock and service certificate are only forged files.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collection, service, provenance, join = potal_fixture(root)
            fake: dict[str, Path] = {}
            for name in ("clock_selection", "service_certificate", "library", "npu_trace", "timing",
                         "cycle_certificate", "run_aware_certificate"):
                path = root / (name + ".json")
                path.write_text("{}\n")
                fake[name] = path
            npu_result = root / "npu-result.jsonl"
            npu_result.write_text("")
            join_row = json.loads(join.read_text())
            join_row["source_artifacts"] = {"npu_results": {"sha256": sha256(npu_result)}}
            join.write_text(json.dumps(join_row))
            inputs: Record = {"im2p": {"path": str(ROOT.parent / "IM2P.sim")},
                              "scenario": {"profile": "a8w8-d16-hp1", "initial_scratchpad_half": 0,
                                           "initial_accumulator_half": 0}}
            for name, path in {**fake, "potal_result": collection, "application": service,
                               "potal_provenance": provenance}.items():
                inputs[name] = artifact_reference(path)
            bindings = root / "input-bindings.json"
            bindings.write_text(json.dumps(inputs))
            schedule = root / "schedule.json"
            schedule.write_text(json.dumps({"schema": "im2p-execution-schedule", "version": 1,
                "scope": "RECONSTRUCTED", "service_validation_scope": "CURRENT_CERTIFIED_SEQUENCE",
                "nodes": schedule_rows()}))
            bundle, lifecycle = root / "bundle.json", root / "lifecycle.json"
            lifecycle.write_text(json.dumps({"schema": "im2p-execution-lifecycle", "version": 2,
                "source_kind": "PRODUCER_DECLARED",
                "producer_binding": {"provenance_sha256": sha256(provenance),
                                     "join_summary_sha256": sha256(join)},
                "application": {"sha256": sha256(service), "expected_samples": 128,
                                "prefill_steps": [{"batch_index": 0, "dispatch_id": 0, "graph_begin": 0}]},
                "npu_results_sha256": sha256(npu_result)}))
            bundle.write_text(json.dumps({"lifecycle_sha256": sha256(lifecycle)}))
            proof: Record = {"schema": "potal-e2e-reconstruction-proof", "version": 1,
                "input_bindings": artifact_reference(bindings), "schedule": artifact_reference(schedule),
                "bundle": artifact_reference(bundle), "join_summary": artifact_reference(join),
                "lifecycle": artifact_reference(lifecycle), "npu_results": artifact_reference(npu_result)}
            source = load_potal_collection(collection, service, provenance, join)
            row = reconstructed_row(source, scheduled_application_result(schedule, [0]), proof)
            candidate = root / "forged-result.json"
            candidate.write_text(json.dumps(row))
            # When loading through the public reducer, then the official verifier rejects the fake proof.
            with self.assertRaisesRegex(ValueError, "current fixed-policy post-route operating clock required"):
                load_measurement(candidate)

    def test_official_service_validator_rejects_forged_certificate(self) -> None:
        # Given a certificate file without current RTL service evidence.
        sys.path.insert(0, str(ROOT.parent / "IM2P.sim"))
        from sim.cycle.service_certificate import validate_service_certificate
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in ("service.json", "library.bin", "trace.jsonl",
                                             "base.json", "run.json")]
            for path in paths:
                path.write_text("{}\n")
            # When admitted by the official producer boundary, then the fake proof is rejected.
            with self.assertRaisesRegex(ValueError, "schema, corpus or library binding mismatch"):
                validate_service_certificate(paths[0], paths[1], paths[2], base_certificate=paths[3],
                                             run_certificate=paths[4], timing={},
                                             initial_scratchpad_half=0, initial_accumulator_half=0)


if __name__ == "__main__":
    unittest.main()
