from __future__ import annotations

import hashlib
import json
from pathlib import Path

from application_results import application_result, integer_array
from eval_common import Record, integer, read_json, record, records, require, sha256, write_json


def token_digest(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def prepare_forced_tokens(source: Path, output: Path, expected: Record) -> Path:
    result = read_json(source / "result.json")
    request = read_json(source.parent / "request.json")
    proof_path = source / "collection-provenance.json"
    proof = read_json(proof_path)
    require(result.get("role") == "potal" and result.get("measurement_kind") == "COLLECTION_OBSERVATION_ONLY",
            "forced CPU source must be actual PoTal collection")
    require(proof.get("source_role") == "POTAL_COLLECTION" and proof.get("collection_success") is True,
            "paired PoTal source provenance failed")
    require(record(result.get("collection_provenance")).get("sha256") == sha256(proof_path),
            "paired PoTal provenance hash mismatch")
    require(request.get("settings") == expected.get("settings"), "paired native settings differ")
    require(result.get("host_id") == record(expected.get("host")).get("host_id"), "paired host differs")
    for key in ("model_sha256", "dataset_sha256"):
        require(result.get(key) == expected.get(key), "paired source differs: " + key)
    endpoint_path = source / "native/application.jsonl"
    endpoints = list(records(endpoint_path))
    require(len(endpoints) == 1, "paired PoTal must have exactly one application trajectory")
    endpoint = endpoints[0]
    require(integer(endpoint, "chunk_id") == integer(expected, "chunk_id"), "paired chunk mapping differs")
    application_result(endpoint)
    application_hash = sha256(endpoint_path)
    require(result.get("application_sha256") == application_hash and
            record(record(proof.get("artifacts")).get("application_endpoints")).get("sha256") == application_hash,
            "paired endpoint is not bound by producer provenance")
    tokens = integer_array(endpoint, "generated_tokens")
    target = output / "forced-token-ids.json"
    write_json(target, list(tokens))
    binding: Record = {"schema": "potal-paired-trajectory", "version": 1,
        "execution_kind": "FORCED_CPU_COST_ONLY", "trajectory_source": "POTAL",
        "source_potal_provenance_sha256": sha256(proof_path), "source_potal_application_sha256": application_hash,
        "token_vector_sha256": token_digest(tokens), "forced_file_sha256": sha256(target),
        "input_tokens_sha256": result["input_tokens_sha256"], "chunk_id": integer(expected, "chunk_id"),
        "model_sha256": result["model_sha256"], "dataset_sha256": result["dataset_sha256"],
        "actual_potal_samples": 128, "actual_cpu_samples": 0}
    write_json(output / "paired-trajectory.json", binding)
    return target


def forced_result(row: Record) -> Record:
    require(row.get("schema") == "potal-application-endpoints" and row.get("version") == 1 and
            row.get("complete") is True and row.get("source_role") == "full_cpu", "invalid forced CPU completion")
    require(row.get("execution_kind") == "FORCED_CPU_COST_ONLY" and row.get("trajectory_source") == "POTAL" and
            row.get("cost_only") is True, "forced CPU execution ownership mismatch")
    require(integer(row, "samples") == integer(row, "actual_samples") == 0 and
            integer(row, "decode_calls") == 127 and row.get("sample_accept_ns") == [],
            "forced CPU must not claim actual sampler calls or endpoints")
    tokens = integer_array(row, "generated_tokens")
    require(len(tokens) == 128, "forced CPU trajectory requires exactly 128 token IDs")
    return {"execution_kind": "FORCED_CPU_COST_ONLY", "trajectory_source": "POTAL",
            "actual_samples": 0, "forced_steps": 128, "decode_calls": 127,
            "generated_tokens_sha256": token_digest(tokens), "target_latency": "NOT_A_FREE_GENERATION_MEASUREMENT"}
