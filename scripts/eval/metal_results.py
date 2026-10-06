"""Validate observed Metal tensor placement against completed custom kernel calls."""
from __future__ import annotations

import math
from pathlib import Path
from typing import TypedDict

from eval_common import Record, integer, read_json, record, require, sha256


class MetalPlacement(TypedDict):
    verified_backend: str
    producer: str
    placement_complete: bool
    coverage: str
    execution_sha256: str
    process_log_sha256: str
    observed_quantized_matmuls: int
    completed_quantized_matmuls: int
    fallback_coverage: str
    tensor_placement: list[Record]
    arithmetic_contract: str
    timing_scope: str


def validate_metal_build(info: Record) -> None:
    require(info.get("metal_quantized_evaluation_supported") is True and
            integer(info, "metal") == integer(info, "metal_quantized") == 1,
            "Metal quantized runner capability unavailable")
    require(integer(info, "cuda") == integer(info, "gemmini") == 0,
            "Metal quantized evaluation requires CUDA and Gemmini OFF")
    require(all(integer(info, key) == 0 for key in
                ("cycle_sim", "log_cycle", "ggml_cpu_cycle_log", "activation_metrics", "residual_metrics", "scale_metrics")),
            "direct Metal application timing requires instrumentation-OFF artifact")
    require(integer(info, "activation_bits") == integer(info, "weight_bits") and
            integer(info, "activation_bits") in (4, 8) and integer(info, "dim") in (16, 32, 64),
            "Metal quantized profile requires matched A4W4 or A8W8")
    require(info.get("activation_mode") in ("BLOCK", "EXSIA") and integer(info, "block_size") == 32,
            "Metal quantized profile requires BLOCK32 or EXSIA32")
    require(info.get("quantization_producer") == "cpu", "unsupported Metal quantization producer")


def metal_placement(proof_path: Path, log_path: Path, info: Record) -> MetalPlacement:
    validate_metal_build(info)
    proof = read_json(proof_path)
    require(proof.get("schema") == "metal-quantized-execution" and proof.get("version") == 1 and
            proof.get("backend") == "Metal" and proof.get("producer") == "cpu" and
            proof.get("complete") is True and proof.get("placement_verified") is True,
            "missing complete Metal custom kernel execution evidence")
    require(all(proof.get(key) == info.get(key) for key in
                ("activation_bits", "weight_bits", "dim", "activation_mode")) and
            proof.get("rmd_enabled") is bool(integer(info, "rmd_enabled")),
            "Metal execution profile differs from requested build")
    observed = integer(proof, "observed_quantized_matmuls", 1)
    block, hp1 = integer(proof, "block_calls"), integer(proof, "hp1_calls")
    require(observed == integer(proof, "metal_quantized_matmuls") == block + hp1 and
            integer(proof, "fallback_calls") == integer(proof, "failed_calls") == 0,
            "Metal quantized matmul fallback or incomplete execution coverage")
    require(integer(proof, "prefill_matmuls", 1) + integer(proof, "decode_matmuls", 1) == observed,
            "Metal execution requires prefill and decode coverage")
    residual = integer(proof, "residual_launches")
    require(integer(proof, "dense_launches") == observed and integer(proof, "merge_launches") == residual,
            "missing Metal dense or residual merge kernel launches")
    require(block == 0 or residual == 0, "BLOCK correction must be fused into its dense kernel")
    require(proof.get("rmd_enabled") is True or residual == 0, "unexpected residual kernel execution")
    expected_type = f'q{integer(info, "weight_bits")}_' + ("hp1" if info["activation_mode"] == "EXSIA" else "0")
    require((block == observed and hp1 == 0) if info["activation_mode"] == "BLOCK" else
            (hp1 == observed and block == 0), "Metal kernel family differs from compiled activation mode")
    raw_tensors = proof.get("tensor_placement")
    require(isinstance(raw_tensors, list) and bool(raw_tensors), "missing Metal tensor placement")
    tensors = [record(value) for value in raw_tensors] if isinstance(raw_tensors, list) else []
    require(sum(integer(row, "calls", 1) for row in tensors) == observed and all(
            row.get("weight_backend") == row.get("output_backend") == "Metal" and
            row.get("type") == expected_type and isinstance(row.get("weight"), str) and row["weight"]
            for row in tensors), "Metal tensor placement, weight format, or coverage mismatch")
    timings = record(proof.get("seconds"))
    require(all(isinstance(timings.get(key), (int, float)) and not isinstance(timings[key], bool) and
                math.isfinite(timings[key]) and timings[key] >= 0 for key in
                ("producer", "transfer", "dense_gpu", "residual_gpu", "merge_gpu", "matmul_total")),
            "invalid Metal execution stage timing")
    require(proof.get("scheduler_observer") == "ask_only_with_split_synchronization",
            "Metal fallback coverage observer unavailable")
    return {"verified_backend": "Metal", "producer": "cpu", "placement_complete": True,
            "coverage": "QUANTIZED_MATMUL_TENSORS_AND_COMPLETED_CUSTOM_KERNELS",
            "execution_sha256": sha256(proof_path), "process_log_sha256": sha256(log_path),
            "observed_quantized_matmuls": observed, "completed_quantized_matmuls": block + hp1,
            "fallback_coverage": "ALL_OBSERVED_QUANTIZED_MATMULS",
            "tensor_placement": tensors,
            "arithmetic_contract": "HP1_EXSIA_ORDERED_SCU" if hp1 else "BLOCK32_INTEGER_DOT",
            "timing_scope": "CPU_PRODUCER_METAL_COMPUTE_WITH_PLACEMENT_OBSERVER"}
