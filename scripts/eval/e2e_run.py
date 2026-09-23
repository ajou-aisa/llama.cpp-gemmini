from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import uuid

from application_results import application_result, application_services, cuda_placement, integer_array
from eval_common import (Record, artifact_snapshot, clean_environment, compiled_info, integer,
    read_json, record, records, require, run, sha256, validate_recipe, write_json)
from evaluation_host import host_facts


def settings_contract(settings: Record) -> None:
    allowed = {"schema", "version", "seed", "temperature", "threads", "threads_batch", "batch_size",
               "ubatch_size", "sampler_policy", "split", "chunk_ids", "expected_host_id", "scope",
               "top_k", "top_p", "min_p", "repeat_penalty", "repeat_last_n", "grammar", "eos_stopping", "warmup"}
    require(not set(settings) - allowed, "unsupported evaluation setting; native defaults cannot replace it")
    require(allowed - {"expected_host_id", "scope"} <= set(settings), "approved evaluation settings are incomplete")
    require(settings.get("schema") == "potal-evaluation-settings" and settings.get("version") == 2,
            "unsupported evaluation settings schema/version")
    require(integer(settings, "seed") == 1234, "approved E2E seed is 1234")
    temperature = settings.get("temperature")
    require(isinstance(temperature, (int, float)) and not isinstance(temperature, bool)
            and math.isfinite(temperature) and temperature == 0, "approved E2E temperature is 0")
    for name in ("threads", "threads_batch", "batch_size", "ubatch_size"):
        integer(settings, name, 1)
    require(integer(settings, "ubatch_size") <= integer(settings, "batch_size") <= 256,
            "require ubatch <= batch <= 256")
    require(settings.get("sampler_policy") == "user-confirmed-greedy-v3", "unsupported sampler policy")
    require(settings.get("split") == "test", "approved E2E split is WikiText-2 test")
    require(integer_array(settings, "chunk_ids") == list(range(10)), "approved repetition mapping is chunks 0..9 once")
    require(all(isinstance(settings[name], (int, float)) and not isinstance(settings[name], bool) and
                settings[name] == value for name, value in (("top_p", 1), ("min_p", 0), ("repeat_penalty", 1))),
            "invalid fixed sampler probability/penalty")
    require(integer(settings, "top_k") == 0 and integer(settings, "repeat_last_n") == 0 and
            settings.get("grammar") is None and settings.get("eos_stopping") is False and
            integer(settings, "warmup") == 0, "approved greedy/EOS/no-warmup policy mismatch")


def validate_native_recipe(workload: Record) -> None:
    expected: Record = {"recipe_id": "wikitext2-test-256x128-greedy-seed1234-v1",
        "sampler_policy": "user-confirmed-greedy-v3", "seed": 1234, "temperature": 0,
        "top_k": 0, "top_p": 1, "min_p": 0, "penalty_repeat": 1, "penalty_last_n": 0,
        "grammar": "", "eos_stopping": False, "eos_logit_suppression": False, "warmup": 0}
    require(all(key in workload and workload[key] == value for key, value in expected.items()),
            "actual native sampler/workload differs from user-approved recipe")


def collect_runs(args: argparse.Namespace) -> None:
    clean_environment()
    settings = read_json(args.settings)
    settings_contract(settings)
    require(1 <= args.repetitions <= 10, "repetitions must be a finite 1..10")
    binary, model, dataset = (Path(value).resolve(strict=True) for value in
                              (args.runner, args.model, args.dataset))
    info = compiled_info(binary)
    validate_recipe(info, "e2e")
    role = args.role
    require(role not in ("fullcpu", "fullcpu-cost-only") or (info.get("cpu_only") is True and integer(info, "cycle_sim") == 0),
            "FullCPU requires CPU-only CYCLE_SIM=0 artifact")
    require(role != "potal" or (integer(info, "cycle_sim") == 1 and info.get("backend") == "IM2P_SIM" and
                               info.get("hp1") is True), "PoTal requires CPU-functional CYCLE_SIM=1 HP1 artifact")
    require(role == "cuda" or integer(info, "log_cycle") == 1, "FullCPU/PoTal sources require existing CPU cycle log")
    require(role != "cuda" or (info.get("cuda_evaluation_supported") is True and integer(info, "cuda") == 1
                              and integer(info, "cycle_sim") == 0), "CUDA native runner capability unavailable")
    require(role != "cuda" or (integer(info, "log_cycle") == 0 and integer(info, "ggml_cpu_cycle_log") == 0),
            "direct CUDA application timing requires instrumentation-OFF artifact")
    require((role == "fullcpu-cost-only") == (args.paired_potal is not None),
            "forced FullCPU cost-only role requires --paired-potal; other roles must not receive it")
    before, host = artifact_snapshot(binary), host_facts()
    if "expected_host_id" in settings:
        require(settings["expected_host_id"] == host["host_id"], "same-host preflight mismatch")
    identities: Record = {"model_sha256": sha256(model), "dataset_sha256": sha256(dataset),
                           "settings_sha256": sha256(args.settings), "artifacts": before,
                           "build_info": info, "host": host, "settings": settings}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    native_build = None
    if role != "cuda":
        require(args.im2p is not None, "FullCPU/PoTal collection requires --im2p for official native provenance")
        sys.path.insert(0, str(args.im2p.resolve(strict=True)))
        from sim.cycle.collection_native import prepare_native_build
        native_build = prepare_native_build(binary.parent.parent, output)
        require(compiled_info(binary) == info, "native rebuild changed requested role/metric configuration")
        before = artifact_snapshot(binary)
        identities["artifacts"] = before
    write_json(output / "request.json", identities)
    results: list[Record] = []
    comparison = hashlib.sha256(json.dumps({"model": identities["model_sha256"],
        "dataset": identities["dataset_sha256"], "settings": settings}, sort_keys=True).encode()).hexdigest()
    for repetition in range(args.repetitions):
        destination = output / f"repetition-{repetition:02d}"
        destination.mkdir()
        command = [str(binary), "--model", str(model), "--file", str(dataset), "--output-dir",
                   str(destination / "native"), "--workload", "E2E_GENERATION_256_128", "--max-chunks", "1",
               "--run-id", f"{output.name}-{repetition}"]
        command.extend(("--chunk-index", str(repetition)))
        for key, option in (("seed", "--seed"), ("temperature", "--temp"), ("threads", "--threads"),
                            ("threads_batch", "--threads-batch"), ("batch_size", "--batch-size"),
                            ("ubatch_size", "--ubatch-size")):
            command.extend((option, str(settings[key])))
        if role == "fullcpu-cost-only":
            from paired_inputs import prepare_forced_tokens
            forced = prepare_forced_tokens(args.paired_potal / f"repetition-{repetition:02d}", destination,
                                           {**identities, "chunk_id": repetition})
            command.extend(("--forced-token-ids", str(forced)))
        if role == "cuda":
            command.extend(("--gpu-layers", "-1"))
        native_collection = None
        if native_build is not None:
            from sim.cycle.collection_native import start_native_collection
            native_collection = start_native_collection(native_build, destination, tuple(command))
        host_before = host_facts()
        require(host_before["host_id"] == host["host_id"], "host identity changed before measurement")
        write_json(destination / "host-before.json", host_before)
        try:
            run(command, destination, args.timeout)
        finally:
            host_after = host_facts()
            write_json(destination / "host-after.json", host_after)
        require(host_after["host_id"] == host["host_id"], "host identity changed during measurement")
        provenance = None
        require(before == artifact_snapshot(binary), "native artifacts changed between actual repetitions")
        application = list(records(destination / "native/application.jsonl"))
        require(len(application) == 1, "one mapped prompt required per repetition")
        require(application[0].get("source_role") == {"fullcpu": "full_cpu", "fullcpu-cost-only": "full_cpu", "potal": "potal_collection", "cuda": "cuda"}[role],
                "native application source role mismatch")
        if role == "fullcpu-cost-only":
            from paired_inputs import forced_result
            measured = forced_result(application[0])
            require((destination / "native/application-cpu.jsonl").stat().st_size == 0,
                    "forced CPU collection must not emit sampler service")
            sampling: Record = {"samples": 0, "duration_authority": "EXCLUDED_FORCED_CPU_NO_SAMPLING"}
        else:
            measured = application_result(application[0])
            sampling = application_services(destination / "native/application-cpu.jsonl", application[0])
        workload = read_json(destination / "native/workload.json")
        validate_native_recipe(workload)
        require(workload.get("complete") is True and workload.get("output_mask") == "last_token",
                "incomplete or PPL-masked E2E workload")
        raw_chunks = workload.get("chunks")
        require(isinstance(raw_chunks, list) and len(raw_chunks) == 1, "wrong native prompt mapping")
        chunks = raw_chunks if isinstance(raw_chunks, list) else []
        prompt = integer_array(record(chunks[0]), "input_tokens")
        require(integer(record(chunks[0]), "chunk_id") == repetition and
                integer(record(chunks[0]), "token_offset") == repetition * 256,
                "native prompt/chunk mapping mismatch")
        require(len(prompt) == 256, "E2E prompt must contain exactly 256 native token IDs")
        if native_collection is not None:
            from sim.cycle.collection_native import finish_native_collection
            provenance = finish_native_collection(native_collection)
        result: Record = {"schema": "potal-e2e-run", "version": 1, "repetition": repetition,
            "measurement_id": str(uuid.uuid4()), "role": role, "host_id": host["host_id"],
            "chunk_id": repetition,
            "comparison_contract": comparison, "scope": "DEVELOPMENT_MEASUREMENT",
            "model_sha256": identities["model_sha256"], "dataset_sha256": identities["dataset_sha256"],
            "input_tokens_sha256": hashlib.sha256(json.dumps(prompt,separators=(",", ":")).encode()).hexdigest(),
            "generated_tokens_sha256": measured["generated_tokens_sha256"],
            "application_sha256": sha256(destination / "native/application.jsonl"),
            "application_services": sampling,
            "collection_provenance": None if provenance is None else {"path": str(provenance), "sha256": sha256(provenance)},
            "host_observations": {"before_sha256": sha256(destination / "host-before.json"),
                                  "after_sha256": sha256(destination / "host-after.json")},
            "workload_sha256": sha256(destination / "native/workload.json"),
            "recipe_id": workload["recipe_id"],
            "tokenizer_identity": {"model_sha256": identities["model_sha256"],
                "vocab_size": workload.get("vocab_size"), "add_special": workload.get("add_special"),
                "parse_special": workload.get("parse_special"), "bos_policy": workload.get("bos_policy")},
            "measurement_kind": "NATIVE_APPLICATION" if role == "cuda" else "COLLECTION_OBSERVATION_ONLY"}
        if role != "cuda":
            result["collection_observation"] = measured
            result["target_latency"] = "NOT_READY_REQUIRES_CERTIFIED_REPLAY_JOIN_CLOCK_AND_SERVICE_PROOF"
        else:
            result.update(measured)
        if role == "cuda":
            result["actual_placement"] = cuda_placement(destination / "process.log")
            result["arithmetic_matching"] = "PRACTICAL_REFERENCE_NOT_A4W4_A8W8_MATCHED"
        write_json(destination / "result.json", result)
        results.append(result)
    require(identities["model_sha256"] == sha256(model) and identities["dataset_sha256"] == sha256(dataset)
            and identities["settings_sha256"] == sha256(args.settings), "workload changed during repetitions")
    write_json(output / "campaign-status.json", {"actual_repetitions": len(results), "warmup": 0,
        "paper_campaign": "NOT_RUN", "scope": "native development collection; no Jetson campaign claim",
        "result_paths": [f"repetition-{index:02d}/result.json" for index in range(len(results))]})
