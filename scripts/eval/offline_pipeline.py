from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

from application_results import load_measurement, load_potal_collection
from certified_reconstruction import artifact_reference, reconstructed_row
from eval_common import Json, Record, integer, read_json, require, sha256, write_json
from scheduled_endpoints import prefill_dispatches, scheduled_application_result

JSON_INPUT_LIMIT_BYTES = 64 * 1024 * 1024
DISK_RESERVE_BYTES = 256 * 1024 * 1024


def write_rejection(output: Path, reason: str) -> None:
    write_json(output / "result.json", {"schema": "potal-offline-evaluation", "version": 1,
        "replay": "PASS", "three_source_join": "PASS", "execution_ir": "PASS",
        "schedule": "NOT_READY_CERTIFICATION_REJECTED", "reason": reason,
        "E2E_RECONSTRUCTION_READY": False, "TTFT": None, "TPOT": None,
        "scope": "official offline dataset; certified inputs rejected"})


def add_arguments(parser: argparse.ArgumentParser) -> None:
    for name in ("im2p", "full-cpu-log", "full-cpu-graph", "full-cpu-provenance",
                 "potal-log", "potal-graph", "potal-provenance", "npu-trace", "library",
                 "cycle-certificate", "run-aware-certificate", "application"):
        parser.add_argument("--" + name, type=Path, required=True)
    lifecycle = parser.add_mutually_exclusive_group(required=True)
    lifecycle.add_argument("--lifecycle", type=Path, help="already bound official lifecycle artifact")
    lifecycle.add_argument("--lifecycle-sidecar", type=Path, help="native producer sidecar; build bound lifecycle after join")
    parser.add_argument("--worker-resources", type=Path, help="explicit CPU worker/resource scenario for lifecycle projection")
    parser.add_argument("--cpu-policy", choices=("THREAD_CPU_NS_GANG", "HOST_ELAPSED_NS_GANG"))
    parser.add_argument("--sampler-resource")
    parser.add_argument("--service-certificate", type=Path, help="current official phase/state service proof")
    parser.add_argument("--clock-selection", type=Path, help="validated operating-clock selection")
    parser.add_argument("--profile", help="exact hardware profile shared by service and clock proof")
    parser.add_argument("--potal-result", type=Path, help="native PoTal repetition result and endpoint binding")
    parser.add_argument("--timing", type=Path, help="source-bound reference memory timing for service replay")
    parser.add_argument("--initial-scratchpad-half", type=int, choices=(0, 1))
    parser.add_argument("--initial-accumulator-half", type=int, choices=(0, 1))
    storage = parser.add_mutually_exclusive_group()
    storage.add_argument("--streaming-ir", dest="streaming_ir", action="store_true",
                         help="alias for the default bounded-memory SQLite IR and schedule")
    storage.add_argument("--json-ir", dest="streaming_ir", action="store_false",
                         help="JSON IR and schedule for small diagnostic inputs (64 MiB maximum)")
    parser.set_defaults(streaming_ir=True)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--diagnostic-phase-table", type=Path,
                        help="explicit SYNTHETIC_ONLY service table; never final target latency")
    parser.add_argument("--diagnostic-frequency-hz", type=int)


def reconstruct(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        _reconstruct(args)
    except (OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as error:
        if output.is_dir() and not (output / "failure.json").exists():
            write_json(output / "failure.json", {"schema": "potal-offline-failure", "version": 1,
                "boundary": "reconstruct", "status": "FAILED", "reason": str(error)})
        raise


def _reconstruct(args: argparse.Namespace) -> None:
    source = args.im2p.resolve(strict=True)
    require((source / "sim/cycle/npu_trace.py").is_file(), "IM2P official replay entrypoint missing")
    require(args.timeout > 0, "finite positive offline timeout required")
    require((args.diagnostic_phase_table is None) == (args.diagnostic_frequency_hz is None),
            "diagnostic schedule requires both phase table and explicit frequency")
    certified = ("service_certificate", "clock_selection", "profile", "potal_result", "timing",
                 "initial_scratchpad_half", "initial_accumulator_half")
    require(args.diagnostic_phase_table is None or any(getattr(args, name) is None for name in certified),
            "diagnostic phase table cannot participate in certified PoTal publication")
    if args.lifecycle_sidecar is not None:
        require(args.worker_resources is not None and args.cpu_policy is not None and args.sampler_resource,
                "producer lifecycle projection requires explicit worker resources, CPU policy, and sampler resource")
    input_names = ("full_cpu_log", "full_cpu_graph", "full_cpu_provenance", "potal_log",
                   "potal_graph", "potal_provenance", "npu_trace", "library", "cycle_certificate",
                   "run_aware_certificate", "application")
    input_names += ("lifecycle",) if args.lifecycle is not None else ("lifecycle_sidecar", "worker_resources")
    input_names += tuple(name for name in ("service_certificate", "clock_selection", "potal_result", "timing")
                         if getattr(args, name) is not None)
    paths = {name: Path(getattr(args, name)).resolve(strict=True) for name in input_names}
    output = args.output.resolve()
    workload_bytes = sum(path.stat().st_size for name, path in paths.items()
                         if name not in ("library", "cycle_certificate", "run_aware_certificate",
                                         "service_certificate", "clock_selection", "potal_result", "timing"))
    required_free = DISK_RESERVE_BYTES + 8 * workload_bytes
    available_free = shutil.disk_usage(output).free
    reason = None
    if not args.streaming_ir and workload_bytes > JSON_INPUT_LIMIT_BYTES:
        reason = f"JSON IR input exceeds {JSON_INPUT_LIMIT_BYTES} bytes; use default SQLite IR"
    elif available_free < required_free:
        reason = f"offline storage requires {required_free} free bytes; available {available_free}"
    if reason is not None:
        write_json(output / "failure.json", {"schema": "potal-offline-failure", "version": 1,
            "boundary": "preflight", "status": "REJECTED", "reason": reason,
            "workload_bytes": workload_bytes, "required_free_bytes": required_free,
            "available_free_bytes": available_free})
        require(False, reason)
    hashes = {name: sha256(path) for name, path in paths.items()}
    identities: Record = {name: {"path": str(path), "sha256": hashes[name]} for name, path in paths.items()}
    identities["im2p"] = {"path": str(source)}
    identities["scenario"] = {"profile": args.profile, "cpu_policy": args.cpu_policy,
                              "sampler_resource": args.sampler_resource,
                              "initial_scratchpad_half": args.initial_scratchpad_half,
                              "initial_accumulator_half": args.initial_accumulator_half}
    write_json(output / "input-bindings.json", identities)

    def stage(name: str, arguments: list[str]) -> None:
        command = [sys.executable, "-B", "-m", *arguments]
        try:
            with (output / (name + ".log")).open("x") as log:
                process = subprocess.run(command, cwd=source, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=args.timeout, check=False)
        except subprocess.TimeoutExpired as error:
            write_json(output / (name + "-command.json"), {"argv": list(command), "cwd": str(source),
                "exit_code": None, "timeout_seconds": args.timeout, "status": "TIMEOUT"})
            write_json(output / "failure.json", {"schema": "potal-offline-failure", "version": 1,
                "boundary": name, "status": "TIMEOUT", "reason": str(error)})
            raise
        except OSError as error:
            write_json(output / "failure.json", {"schema": "potal-offline-failure", "version": 1,
                "boundary": name, "status": "FAILED", "reason": str(error)})
            raise
        write_json(output / (name + "-command.json"), {"argv": list(command), "cwd": str(source),
            "exit_code": process.returncode, "timeout_seconds": args.timeout,
            "status": "PASS" if process.returncode == 0 else "FAILED"})
        if process.returncode != 0:
            write_json(output / "failure.json", {"schema": "potal-offline-failure", "version": 1,
                "boundary": name, "status": "FAILED", "exit_code": process.returncode,
                "reason": "official offline boundary failed: " + name})
        require(process.returncode == 0, "official offline boundary failed: " + name)

    artifacts = ["--library", str(paths["library"]), "--cycle-certificate", str(paths["cycle_certificate"]),
                 "--run-aware-certificate", str(paths["run_aware_certificate"])]
    npu = output / "npu-cycle-result.jsonl"
    stage("replay", ["sim.cycle.npu_trace", str(paths["npu_trace"]), *artifacts,
                     "--output", str(npu), "--summary", str(output / "npu-summary.json")])
    dataset, join_summary = output / "dataset.jsonl.gz", output / "join-summary.json"
    join = ["sim.cycle.reconstruct", *artifacts, "--npu-trace", str(paths["npu_trace"]),
            "--npu-results", str(npu), "--output", str(dataset), "--summary", str(join_summary)]
    for name in input_names[:6]:
        join.extend(("--" + name.replace("_", "-"), str(paths[name])))
    stage("join", join)
    lifecycle_path = paths["lifecycle"] if args.lifecycle is not None else output / "execution-lifecycle.json"
    if args.lifecycle_sidecar is not None:
        stage("producer-lifecycle", ["sim.cycle.execution_lifecycle_cli", "--sidecar", str(paths["lifecycle_sidecar"]),
              "--semantic-graph", str(paths["potal_graph"]), "--provenance", str(paths["potal_provenance"]),
              "--application", str(paths["application"]), "--dataset", str(dataset), "--npu-results", str(npu),
              "--join-summary", str(join_summary), "--worker-resources", str(paths["worker_resources"]),
              "--cpu-policy", args.cpu_policy, "--sampler-resource", args.sampler_resource,
              "--output", str(lifecycle_path)])
    bundle = output / ("execution.sqlite" if args.streaming_ir else "execution-bundle.json")
    stage("execution-ir", ["sim.cycle.execution_cli", "adapt", "--dataset", str(dataset),
          "--lifecycle", str(lifecycle_path), "--npu-results", str(npu), "--join-summary",
          str(join_summary), "--application", str(paths["application"]), "--output", str(bundle),
          *(["--streaming"] if args.streaming_ir else [])])
    missing: list[Json] = [name for name in certified if getattr(args, name) is None]
    schedule_status = "NOT_RUN_MISSING_CERTIFIED_INPUTS"
    reconstructed: Record | None = None
    if not missing:
        try:
            frequency = integer(read_json(paths["clock_selection"]), "selected_frequency_hz", 1)
            schedule = output / ("reconstructed-schedule.sqlite" if args.streaming_ir else "reconstructed-schedule.json")
            stage("certified-schedule", ["sim.cycle.execution_cli", "schedule", "--bundle", str(bundle),
                  "--cycle-library", str(paths["library"]), "--npu-trace", str(paths["npu_trace"]),
                  "--timing", str(paths["timing"]), "--initial-scratchpad-half", str(args.initial_scratchpad_half),
                  "--initial-accumulator-half", str(args.initial_accumulator_half),
                  "--service-certificate", str(paths["service_certificate"]),
                  "--cycle-certificate", str(paths["cycle_certificate"]),
                  "--run-aware-certificate", str(paths["run_aware_certificate"]),
                  "--clock-selection", str(paths["clock_selection"]), "--profile", args.profile,
                  "--frequency-hz", str(frequency), "--output", str(schedule)])
            source_row = load_potal_collection(paths["potal_result"], paths["application"],
                                               paths["potal_provenance"], join_summary)
            metrics = scheduled_application_result(schedule, prefill_dispatches(read_json(lifecycle_path)))
            proof: Record = {"schema": "potal-e2e-reconstruction-proof", "version": 1,
                             "input_bindings": artifact_reference(output / "input-bindings.json"),
                             "schedule": artifact_reference(schedule), "bundle": artifact_reference(bundle),
                             "join_summary": artifact_reference(join_summary),
                             "lifecycle": artifact_reference(lifecycle_path),
                             "npu_results": artifact_reference(npu)}
            reconstructed = reconstructed_row(source_row, metrics, proof)
            schedule_status = "RECONSTRUCTED_CERTIFIED"
        except (OSError, sqlite3.Error, subprocess.SubprocessError) as error:
            if not (output / "failure.json").exists():
                write_json(output / "failure.json", {"schema": "potal-offline-failure", "version": 1,
                    "boundary": "certified-reconstruction", "status": "FAILED", "reason": str(error)})
            raise
        except ValueError as error:
            if not (output / "failure.json").exists():
                write_rejection(output, str(error))
            raise
    if args.diagnostic_phase_table is not None:
        schedule = output / ("schedule.sqlite" if args.streaming_ir else "schedule.json")
        stage("synthetic-schedule", ["sim.cycle.execution_cli", "schedule", "--bundle", str(bundle),
              "--phase-table", str(args.diagnostic_phase_table.resolve(strict=True)), "--frequency-hz",
              str(args.diagnostic_frequency_hz), "--synthetic", "--output", str(schedule)])
        schedule_status = "SYNTHETIC_ONLY"
    require(all(sha256(path) == hashes[name]
                for name, path in paths.items()), "offline source inputs changed")
    if reconstructed is not None:
        try:
            with tempfile.TemporaryDirectory(prefix="candidate-run-", dir=output) as directory:
                candidate = Path(directory) / "result.json"
                write_json(candidate, reconstructed)
                load_measurement(candidate)
                os.link(candidate, output / "reconstructed-result.json")
        except (OSError, sqlite3.Error, subprocess.SubprocessError) as error:
            if not (output / "failure.json").exists():
                write_json(output / "failure.json", {"schema": "potal-offline-failure", "version": 1,
                    "boundary": "certified-publication", "status": "FAILED", "reason": str(error)})
            raise
        except ValueError as error:
            if not (output / "failure.json").exists():
                write_rejection(output, str(error))
            raise
    status: Record = {"schema": "potal-offline-evaluation", "version": 1,
        "replay": "PASS", "three_source_join": "PASS", "execution_ir": "PASS",
        "execution_ir_format": "SQLITE" if args.streaming_ir else "JSON",
        "schedule": schedule_status, "missing_certified_inputs": missing,
        "E2E_RECONSTRUCTION_READY": reconstructed is not None,
        "TTFT": None if reconstructed is None else reconstructed["ttft_ns"],
        "TPOT": None if reconstructed is None else reconstructed["tpot_ns"],
        "reconstructed_result": None if reconstructed is None else artifact_reference(output / "reconstructed-result.json"),
        "paper_campaign": "NOT_RUN",
        "scope": "official one-run reconstruction" if reconstructed is not None
                 else "official offline dataset and optional synthetic schedule"}
    write_json(output / "result.json", status)
