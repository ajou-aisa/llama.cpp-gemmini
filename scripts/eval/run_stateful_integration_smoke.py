#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 -B scripts/eval/run_stateful_integration_smoke.py --potal-runner P --fullcpu-runner F --model M --dataset D --im2p ../IM2P.sim --output ROOT
"""INTEGRATION_ONLY metrics-OFF host-timing collection for one 256+1 PoTal/FullCPU pair.

Never a paper E2E run: the 128-sample/127-decode application contract stays
untouched; this collects one greedy 256-prompt 1-token trajectory with metric
hooks compiled OFF and CPU cycle logging ON, then a FullCPU forced replay of
the same single generated token as the ordinary-CPU cost source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Final

from eval_common import (
    EvaluationError,
    Record,
    clean_environment,
    compiled_info,
    integer,
    read_json,
    record,
    records,
    require,
    sha256,
    text,
    write_json,
)
from evaluation_host import host_facts

SCOPE: Final = "INTEGRATION_ONLY_NOT_PAPER_E2E"
SCHEMA: Final = "potal-stateful-integration-collection-v1"
WORKLOAD: Final = "E2E_GENERATION_256_128"
PAIRED_WORKLOAD_KEYS: Final = (
    "workload", "manifest_sha256", "tokens", "model_file_type", "model_description", "model_parameters",
    "complete_chunks", "selected_chunks", "first_chunk", "chunk_index", "dropped_tail_tokens",
    "context_tokens", "batch", "ubatch", "runtime_context", "threads", "threads_batch", "add_special",
    "parse_special", "trailing_lf_removed", "add_bos", "bos_token", "bos_policy", "output_mask",
    "kv_policy", "warmup", "seed", "temperature", "sampler_policy", "recipe_id", "top_k", "top_p",
    "min_p", "typical_p", "vocab_size", "diagnostic_smoke", "requested_generated_tokens",
    "gpu_layers_requested",
)


def argv_for(runner: Path, model: Path, dataset: Path, native: Path, forced: Path | None) -> list[str]:
    argv = [str(runner), "--model", str(model), "--file", str(dataset), "--output-dir", str(native),
            "--workload", WORKLOAD, "--max-chunks", "1", "--chunk-index", "0",
            "--smoke-generated-tokens", "1", "--seed", "1234", "--temp", "0",
            "--batch-size", "256", "--ubatch-size", "256", "--threads", "1", "--threads-batch", "1",
            "--gpu-layers", "0"]
    if forced is not None:
        argv += ["--forced-token-ids", str(forced)]
    return argv


def run_collection(role: str, runner: Path, model: Path, dataset: Path, output: Path,
                   timeout: int, forced: Path | None, pairing: Record | None = None) -> Path:
    from sim.cycle.collection_native import (
        finish_native_collection,
        prepare_native_build,
        start_native_collection,
    )

    output.mkdir(parents=True, exist_ok=False)
    if pairing is not None:
        write_json(output / "paired-trajectory.json", pairing)
    build = prepare_native_build(runner.parent.parent, output)
    command = argv_for(runner, model, dataset, output / "native", forced)
    collection = start_native_collection(build, output, tuple(command))
    write_json(output / "host-before.json", host_facts())
    with (output / "process.log").open("x") as log:
        process = subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT,
                                 timeout=timeout, check=False)
    write_json(output / "host-after.json", host_facts())
    require(read_json(output / "host-before.json").get("host_id") ==
            read_json(output / "host-after.json").get("host_id"),
            "host identity changed during " + role + " collection")
    receipt: Record = {"argv": [str(part) for part in command], "cwd": str(output),
                       "exit_code": process.returncode}
    write_json(output / "command.json", receipt)
    require(process.returncode == 0, role + " native collection failed; see process.log")
    return finish_native_collection(collection)


def token_list(endpoint: Record) -> list[int]:
    value = endpoint.get("generated_tokens")
    require(isinstance(value, list), "generated_tokens array required")
    assert isinstance(value, list)
    return [integer({"id": item}, "id") for item in value]


def chunk0(workload: Record) -> Record:
    value = workload.get("chunks")
    require(isinstance(value, list) and len(value) == 1, "one workload chunk required")
    assert isinstance(value, list)
    return record(value[0])


def endpoint_of(output: Path) -> Record:
    rows = list(records(output / "native/application.jsonl"))
    require(len(rows) == 1, "exactly one integration trajectory required")
    return rows[0]


def duration_rows(path: Path) -> list[Record]:
    rows: list[Record] = []
    for row in records(path):
        if row.get("interval_class") == "CANONICAL_ADDITIVE" and row.get("duration_role") == "POTAL_HOST":
            rows.append(row)
    return rows


def validate_potal(output: Path, info: Record) -> Record:
    endpoint = endpoint_of(output)
    require(endpoint.get("execution_kind") == "FREE_GENERATION" and
            integer(endpoint, "actual_sampler_calls") == 1 and integer(endpoint, "decode_calls") == 0 and
            endpoint.get("diagnostic_smoke") is True and integer(endpoint, "requested_generated_tokens") == 1,
            "PoTal integration trajectory must be free generation of exactly one token")
    log_root = output / "native/chunk-0"
    check = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve().parents[2] /
                            "tests/check-cycle-sim-host-stages.py"),
                            str(log_root / "npu-cycle-trace.jsonl"), str(log_root / "cycle-log.jsonl")],
                           capture_output=True, text=True, timeout=600, check=False)
    require(check.returncode == 0 and "POTAL_CANONICAL_HOST_INTERVALS_PASS" in check.stdout,
            "declared PoTal host stages and canonical measurements do not match one-to-one: " + check.stderr)
    canonical = duration_rows(log_root / "cycle-log.jsonl")
    counts: dict[str, int] = {}
    invalid = 0
    for row in canonical:
        counts[text(row, "op")] = counts.get(text(row, "op"), 0) + 1
        valid = (row.get("host_elapsed_valid") is True and integer(row, "host_end_ns") >=
                 integer(row, "host_start_ns") and
                 integer(row, "host_elapsed_ns") == integer(row, "host_end_ns") - integer(row, "host_start_ns"))
        if not valid:
            invalid += 1
    require(bool(canonical), "PoTal collection produced no canonical POTAL_HOST intervals")
    require(invalid == 0, f"{invalid} canonical intervals lack a valid host_elapsed authority")
    require(info.get("log_cycle") == 1 and info.get("cycle_sim") == 1, "PoTal build lost required flags")
    report: Record = {"canonical_rows": len(canonical), "per_stage_counts": dict(counts),
                      "host_stage_check": check.stdout.strip()}
    return report


def validate_fullcpu(output: Path, forced_ids: list[int]) -> Record:
    endpoint = endpoint_of(output)
    require(endpoint.get("execution_kind") == "FORCED_CPU_COST_ONLY" and
            endpoint.get("trajectory_source") == "POTAL" and
            integer(endpoint, "actual_sampler_calls") == 0 and integer(endpoint, "decode_calls") == 0 and
            token_list(endpoint) == forced_ids,
            "FullCPU forced replay must consume exactly the PoTal token without sampling")
    require((output / "native/application-cpu.jsonl").stat().st_size == 0,
            "forced CPU collection must not emit sampler service")
    ordinary = [row for row in records(output / "native/chunk-0/cycle-log.jsonl")
                if row.get("cpu_service") is True]
    require(bool(ordinary), "FullCPU ordinary CPU services missing")
    report: Record = {"ordinary_cpu_service_rows": len(ordinary)}
    return report


def paired_trajectory(potal: Path, provenance: Path, forced_ids: list[int], forced: Path,
                      dataset: Path) -> Record:
    source = read_json(provenance)
    application = sha256(potal / "native/application.jsonl")
    require(record(record(source.get("artifacts")).get("application_endpoints")).get("sha256") == application,
            "PoTal endpoint is not bound by its collection provenance")
    binding: Record = {"schema": "potal-paired-trajectory", "version": 1,
        "execution_kind": "FORCED_CPU_COST_ONLY", "trajectory_source": "POTAL",
        "source_potal_provenance_sha256": sha256(provenance), "source_potal_application_sha256": application,
        "token_vector_sha256": hashlib.sha256(json.dumps(forced_ids, separators=(",", ":")).encode()).hexdigest(),
        "forced_file_sha256": sha256(forced), "input_tokens_sha256": text(source, "input_tokens_sha256"),
        "chunk_id": integer(source, "chunk_id"), "model_sha256": text(source, "model_sha256"),
        "dataset_sha256": sha256(dataset), "actual_potal_samples": len(forced_ids), "actual_cpu_samples": 0}
    return binding


def pairing(potal: Path, fullcpu: Path, potal_prov: Path, fullcpu_prov: Path) -> Record:
    from sim.cycle.reconstruct_pairing import validate_source_pair

    left, right = read_json(potal_prov), read_json(fullcpu_prov)
    for key in ("model_sha256", "input_tokens_sha256", "output_tokens_sha256", "chunk_id", "recipe_id",
                "requested_generated_tokens", "diagnostic_smoke"):
        require(key in left and key in right and left[key] == right[key], "paired source differs: " + key)
    validate_source_pair(right, left, sha256(potal_prov))
    lw = read_json(potal / "native/workload.json")
    rw = read_json(fullcpu / "native/workload.json")
    for key in PAIRED_WORKLOAD_KEYS:
        require(key in lw and key in rw and lw[key] == rw[key], "paired workload setting differs: " + key)
    lc, rc = chunk0(lw), chunk0(rw)
    require(lc.get("input_tokens") == rc.get("input_tokens"), "paired prompt token IDs differ")
    prompt_fingerprint = hashlib.sha256(json.dumps(lc.get("input_tokens"),
                                                   separators=(",", ":")).encode()).hexdigest()
    lh, rh = read_json(potal / "host-before.json"), read_json(fullcpu / "host-before.json")
    require(lh.get("host_id") == rh.get("host_id"), "paired host differs")
    left_kernel = read_json(potal / "native-build-receipt.json").get("cpu_kernel_contract_sha256")
    right_kernel = read_json(fullcpu / "native-build-receipt.json").get("cpu_kernel_contract_sha256")
    report: Record = {"host_id": lh.get("host_id"), "phase_input_fingerprint": prompt_fingerprint,
                      "cpu_kernel_contract_sha256": {"potal": left_kernel, "fullcpu": right_kernel},
                      "generated_token_sha256": left.get("output_tokens_sha256")}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("potal-runner", "fullcpu-runner", "model", "dataset", "im2p", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    clean_environment()
    sys.path.insert(0, str(args.im2p.resolve(strict=True)))
    output = args.output.resolve()
    model = args.model.resolve(strict=True)
    dataset = args.dataset.resolve(strict=True)
    try:
        potal_runner = args.potal_runner.resolve(strict=True)
        fullcpu_runner = args.fullcpu_runner.resolve(strict=True)
        potal_info = compiled_info(potal_runner)
        fullcpu_info = compiled_info(fullcpu_runner)
        for info in (potal_info, fullcpu_info):
            require(all(info.get(key) == 0 for key in
                        ("activation_metrics", "residual_metrics", "scale_metrics")),
                    "integration collection requires all metric hooks compiled OFF")
            require(info.get("log_cycle") == 1 and info.get("ggml_cpu_cycle_log") == 1,
                    "integration collection requires LOG_CYCLE=1 and CPU cycle instrumentation")
        require(potal_info.get("cycle_sim") == 1 and potal_info.get("backend") == "IM2P_SIM" and
                potal_info.get("hp1") is True and potal_info.get("matmul_mode") == "STRIPE_PIPELINE" and
                potal_info.get("dim") == 32 and potal_info.get("activation_bits") == 8,
                "PoTal integration runner must be the production STRIPE_PIPELINE A8W8 DIM32 build")
        require(fullcpu_info.get("cpu_only") is True and fullcpu_info.get("cycle_sim") == 0,
                "FullCPU integration runner must be a CPU-only CYCLE_SIM=0 build")
        host = host_facts()
        potal_dir, fullcpu_dir = output / "potal/collection", output / "fullcpu/collection"
        potal_prov = run_collection("potal", potal_runner, model, dataset, potal_dir,
                                    args.timeout, None)
        potal_host = validate_potal(potal_dir, potal_info)
        endpoint = endpoint_of(potal_dir)
        forced_ids = token_list(endpoint)
        require(len(forced_ids) == 1, "integration trajectory must contain exactly one generated token")
        forced_path = output / "fullcpu/forced-token-ids.json"
        forced_path.parent.mkdir(parents=True, exist_ok=True)
        with forced_path.open("x", encoding="utf-8") as stream:
            json.dump(forced_ids, stream)
            stream.write("\n")
        fullcpu_prov = run_collection("fullcpu", fullcpu_runner, model, dataset, fullcpu_dir,
                                      args.timeout, forced_path,
                                      paired_trajectory(potal_dir, potal_prov, forced_ids, forced_path, dataset))
        fullcpu_stats = validate_fullcpu(fullcpu_dir, forced_ids)
        pair = pairing(potal_dir, fullcpu_dir, potal_prov, fullcpu_prov)
        potal_entry: Record = {"collection": str(potal_dir),
                               "provenance": {"path": str(potal_prov), "sha256": sha256(potal_prov)},
                               "build_info_sha256": hashlib.sha256(
                                   json.dumps(potal_info, sort_keys=True).encode()).hexdigest()}
        potal_entry.update(potal_host)
        fullcpu_entry: Record = {"collection": str(fullcpu_dir),
                                 "provenance": {"path": str(fullcpu_prov), "sha256": sha256(fullcpu_prov)},
                                 "forced_token_ids": [int(value) for value in forced_ids]}
        fullcpu_entry.update(fullcpu_stats)
        summary: Record = {
            "schema": SCHEMA, "scope": SCOPE, "workload": WORKLOAD,
            "recipe": {"prompt_tokens": 256, "generated_tokens": 1, "decode_calls": 0,
                       "sampler": "greedy", "seed": 1234, "chunk_index": 0},
            "paper_contract_untouched": "128 samples / 127 decode requirements remain in "
                                        "application_results.py and e2e_run.py",
            "host": {"host_id": host.get("host_id")},
            "potal": potal_entry,
            "fullcpu": fullcpu_entry,
            "pairing": pair,
            "METRICS_OFF_POTAL_HOST_COLLECTION_READY": "READY",
            "PAIRED_FULLCPU_COST_SOURCE_READY": "READY",
            "TTFT": None, "TPOT": None,
        }
        write_json(output / "collection-summary.json", summary)
        print(json.dumps({key: summary[key] for key in
                          ("METRICS_OFF_POTAL_HOST_COLLECTION_READY", "PAIRED_FULLCPU_COST_SOURCE_READY")}))
    except (EvaluationError, OSError, ValueError, subprocess.SubprocessError) as error:
        write_json(output / "failure.json", {"schema": SCHEMA, "status": "FAILED", "reason": str(error)})
        print("integration collection failed: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
