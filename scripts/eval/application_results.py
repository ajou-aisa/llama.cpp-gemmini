from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
from statistics import median

from eval_common import Record, integer, read_json, record, records, require, sha256, text, validate_recipe


def integer_array(row: Record, name: str) -> list[int]:
    values = row.get(name)
    require(isinstance(values, list), "missing integer array: " + name)
    if not isinstance(values, list):
        return []
    result: list[int] = []
    for value in values:
        require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                "invalid integer array value: " + name)
        if isinstance(value, int):
            result.append(value)
    return result


def rational(value: Fraction) -> Record:
    return {"numerator": value.numerator, "denominator": value.denominator}


def application_result(row: Record) -> Record:
    require(row.get("schema") == "potal-application-endpoints" and row.get("version") == 1,
            "unsupported application endpoint schema")
    require(row.get("workload") == "E2E_GENERATION_256_128" and row.get("complete") is True,
            "incomplete or wrong application workload")
    require(row.get("cost_only") is not True, "forced CPU cost collection is not free generation")
    require(integer(row, "samples") == 128 and integer(row, "decode_calls") == 127,
            "application requires 128 actual samples and 127 decode calls")
    require(integer(row, "warmup") == 0 and row.get("excludes_terminal_io") is True,
            "warmup or terminal I/O violates endpoint contract")
    require(row.get("timing_source") == "steady_clock" and row.get("timing_unit") == "ns",
            "unsupported application time source/unit")
    start = integer(row, "t0_ns")
    samples, tokens = integer_array(row, "sample_accept_ns"), integer_array(row, "generated_tokens")
    require(len(samples) == len(tokens) == 128, "sample endpoint/token coverage mismatch")
    require(all(a <= b for a, b in zip([start, *samples[:-1]], samples)), "non-monotonic application endpoints")
    token_bytes = json.dumps(tokens, separators=(",", ":")).encode()
    return {"ttft_ns": samples[0] - start,
            "tpot_ns": {"numerator": samples[-1] - samples[0], "denominator": 127},
            "samples": 128, "decode_calls": 127, "warmup": 0,
            "generated_tokens_sha256": hashlib.sha256(token_bytes).hexdigest(),
            "timing_source": "steady_clock", "timing_unit": "ns"}


def aggregate_results(rows: list[Record]) -> Record:
    require(len(rows) == 10, "paper aggregation requires ten actual repetitions")
    require({integer(row, "repetition") for row in rows} == set(range(10)),
            "missing/duplicate repetition identity")
    require(all(integer(row, "chunk_id") == integer(row, "repetition") for row in rows),
            "ten repetitions must use native WikiText test chunks 0..9 once")
    require(len({text(row, "measurement_id") for row in rows}) == 10, "repeated host measurement")
    require(len({text(row, "application_sha256") for row in rows}) == 10, "reused application endpoint sample")
    for key in ("host_id", "comparison_contract", "role"):
        require(len({text(row, key) for row in rows}) == 1, "repetition contract mismatch: " + key)
    require(all(row.get("measurement_kind") in {"NATIVE_APPLICATION", "VALIDATED_RECONSTRUCTION"}
                for row in rows), "collection/replay is not an application latency measurement")
    require(all(row.get("role") not in {"potal", "fullcpu"} or row.get("measurement_kind") == "VALIDATED_RECONSTRUCTION"
                for row in rows), "instrumented collection elapsed time is not target latency")
    ttft = [Fraction(integer(row, "ttft_ns")) for row in rows]
    tpot = [Fraction(integer(record(row.get("tpot_ns")), "numerator"),
                     integer(record(row.get("tpot_ns")), "denominator", 1)) for row in rows]
    return {"schema": "potal-e2e-aggregate", "version": 1, "repetitions": 10,
            "ttft_ns": rational(median(ttft)), "tpot_ns": rational(median(tpot)),
            "aggregation": "median-of-ten-run-ttft-and-run-mean-tpot",
            "role": rows[0]["role"], "host_id": rows[0]["host_id"],
            "comparison_contract": rows[0]["comparison_contract"], "first_run_discarded": False}


def validate_pair(full_cpu: Record, potal: Record) -> None:
    for key in ("host_id", "model_sha256", "dataset_sha256", "input_tokens_sha256",
                "generated_tokens_sha256", "cpu_kernel_contract_sha256", "comparison_contract"):
        require(text(full_cpu, key) == text(potal, key), "FullCPU/PoTal pairing mismatch: " + key)


def cuda_placement(log: Path) -> Record:
    matches = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", log.read_text(errors="replace"))
    require(len(matches) == 1, "missing/ambiguous GPU layer-count evidence")
    offloaded, total = map(int, matches[0])
    require(offloaded == total and total > 0, "requested maximum GPU layer offload incomplete")
    return {"offloaded_layers": offloaded, "total_layers": total,
            "source": "native load_tensors model-load log", "requested_gpu_layers": -1,
            "process_log_sha256": sha256(log), "coverage": "LAYER_COUNT_ONLY",
            "placement_complete": False, "verified_backend": None,
            "tensor_placement": None, "fallback_coverage": None,
            "publication_status": "OBSERVATION_ONLY_PLACEMENT_INCOMPLETE"}


def load_measurement(path: Path) -> Record:
    row = read_json(path)
    require(row.get("schema") == "potal-e2e-run" and row.get("version") == 1,
            "unsupported application result schema")
    require(row.get("role") == "cuda" and row.get("measurement_kind") == "NATIVE_APPLICATION",
            "only bound native CUDA direct latency currently admitted; reconstructed publication requires service/clock proof")
    root = path.parent
    endpoint_path = root / "native/application.jsonl"
    require(row.get("application_sha256") == sha256(endpoint_path), "application result/source hash mismatch")
    endpoints = list(records(endpoint_path))
    require(len(endpoints) == 1 and endpoints[0].get("source_role") == "cuda", "wrong application source")
    measured = application_result(endpoints[0])
    require(all(row.get(key) == value for key, value in measured.items()), "application result differs from measured endpoints")
    request = read_json(root.parent / "request.json")
    info = record(request.get("build_info"))
    validate_recipe(info, "e2e")
    require(info.get("cuda") == 1 and info.get("cycle_sim") == 0 and info.get("log_cycle") == 0
            and info.get("ggml_cpu_cycle_log") == 0, "direct latency artifact contains instrumentation")
    require(row.get("host_id") == record(request.get("host")).get("host_id"), "result host binding mismatch")
    placement = cuda_placement(root / "process.log")
    require(row.get("actual_placement") == placement, "CUDA placement result binding mismatch")
    require(placement.get("placement_complete") is True,
            "CUDA placement proof is LAYER_COUNT_ONLY; verified campaign requires actual CUDA backend, tensor placement, and fallback coverage")
    return row


def application_services(path: Path, endpoint: Record) -> Record:
    services = list(records(path))
    tokens = integer_array(endpoint, "generated_tokens")
    require(len(services) == len(tokens) == 128, "missing/extra application sampling service")
    for index, row in enumerate(services):
        require(row.get("schema") == "potal-application-cpu" and row.get("version") == 1
                and row.get("stage") == "sample_accept", "unsupported application service schema/stage")
        require(integer(row, "sample_index") == index and integer(row, "token_id") == tokens[index],
                "application service/token identity mismatch")
        require(row.get("source_role") == endpoint.get("source_role") and
                integer(row, "chunk_id") == integer(endpoint, "chunk_id"), "application service ownership mismatch")
        require(row.get("phase") == ("prefill" if index == 0 else "decode") and
                row.get("decode_index") == (None if index == 0 else index - 1), "application phase mismatch")
        require(row.get("cpu_work_cycles_unit") == "cycle", "invalid CPU cycle unit")
        if row.get("host_elapsed_valid") is True:
            require(integer(row, "host_end_ns") - integer(row, "host_start_ns") == integer(row, "host_elapsed_ns"),
                    "application host elapsed interval mismatch")
        else:
            require(row.get("host_elapsed_ns") is None and bool(row.get("host_elapsed_reason")),
                    "invalid host sample lacks reason")
        if row.get("cpu_work_cycles_valid") is True:
            integer(row, "cpu_work_cycles")
            require(row.get("cpu_work_cycles_source") == "linux_perf_cpu_cycles", "invalid CPU work counter source")
        else:
            require(row.get("cpu_work_cycles") is None and bool(row.get("cpu_work_cycles_reason")),
                    "invalid CPU work sample lacks reason")
    return {"path": str(path), "sha256": sha256(path), "samples": len(services),
            "duration_authority": "POTAL_APPLICATION_ONLY" if endpoint.get("source_role") == "potal_collection"
                                  else "NON_POTAL_REFERENCE_ONLY",
            "host_valid_samples": sum(row.get("host_elapsed_valid") is True for row in services)}
