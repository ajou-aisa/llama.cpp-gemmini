from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from eval_common import Record, require, sha256, write_json


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
    parser.add_argument("--streaming-ir", action="store_true", help="bounded-memory SQLite execution IR and optional diagnostic schedule")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--diagnostic-phase-table", type=Path,
                        help="explicit SYNTHETIC_ONLY service table; never final target latency")
    parser.add_argument("--diagnostic-frequency-hz", type=int)


def reconstruct(args: argparse.Namespace) -> None:
    source = args.im2p.resolve(strict=True)
    require((source / "sim/cycle/npu_trace.py").is_file(), "IM2P official replay entrypoint missing")
    require(args.timeout > 0, "finite positive offline timeout required")
    require((args.diagnostic_phase_table is None) == (args.diagnostic_frequency_hz is None),
            "diagnostic schedule requires both phase table and explicit frequency")
    if args.lifecycle_sidecar is not None:
        require(args.worker_resources is not None and args.cpu_policy is not None and args.sampler_resource,
                "producer lifecycle projection requires explicit worker resources, CPU policy, and sampler resource")
    input_names = ("full_cpu_log", "full_cpu_graph", "full_cpu_provenance", "potal_log",
                   "potal_graph", "potal_provenance", "npu_trace", "library", "cycle_certificate",
                   "run_aware_certificate", "application")
    input_names += ("lifecycle",) if args.lifecycle is not None else ("lifecycle_sidecar", "worker_resources")
    paths = {name: Path(getattr(args, name)).resolve(strict=True) for name in input_names}
    hashes = {name: sha256(path) for name, path in paths.items()}
    identities: Record = {name: {"path": str(path), "sha256": hashes[name]} for name, path in paths.items()}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "input-bindings.json", identities)

    def stage(name: str, arguments: list[str]) -> None:
        command = [sys.executable, "-B", "-m", *arguments]
        with (output / (name + ".log")).open("x") as log:
            process = subprocess.run(command, cwd=source, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=args.timeout, check=False)
        write_json(output / (name + "-command.json"), {"argv": list(command), "cwd": str(source),
            "exit_code": process.returncode, "timeout_seconds": args.timeout})
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
    lifecycle_path = paths.get("lifecycle", output / "execution-lifecycle.json")
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
    schedule_status = "NOT_RUN_MISSING_VALIDATED_CLOCK_AND_SERVICE_PROVIDER"
    if args.diagnostic_phase_table is not None:
        schedule = output / ("schedule.sqlite" if args.streaming_ir else "schedule.json")
        stage("synthetic-schedule", ["sim.cycle.execution_cli", "schedule", "--bundle", str(bundle),
              "--phase-table", str(args.diagnostic_phase_table.resolve(strict=True)), "--frequency-hz",
              str(args.diagnostic_frequency_hz), "--synthetic", "--output", str(schedule)])
        schedule_status = "SYNTHETIC_ONLY"
    require(all(sha256(path) == hashes[name]
                for name, path in paths.items()), "offline source inputs changed")
    write_json(output / "result.json", {"schema": "potal-offline-evaluation", "version": 1,
        "replay": "PASS", "three_source_join": "PASS", "execution_ir": "PASS",
        "execution_ir_format": "SQLITE" if args.streaming_ir else "JSON",
        "schedule": schedule_status, "E2E_RECONSTRUCTION_READY": False,
        "TTFT": None, "TPOT": None, "scope": "official offline dataset and optional synthetic schedule"})
