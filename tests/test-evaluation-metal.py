from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/eval"))

from eval_common import Record, EvaluationError
from application_results import application_result
from metal_results import metal_placement, validate_metal_build


@pytest.fixture
def profile() -> Record:
    return {"metal_quantized_evaluation_supported": True, "metal": 1, "metal_quantized": 1,
            "cuda": 0, "gemmini": 0, "cycle_sim": 0, "log_cycle": 0, "ggml_cpu_cycle_log": 0,
            "activation_metrics": 0, "residual_metrics": 0, "scale_metrics": 0,
            "activation_bits": 4, "weight_bits": 4, "dim": 16,
            "activation_mode": "EXSIA", "block_size": 32, "rmd_enabled": 1,
            "quantization_producer": "cpu"}


@pytest.fixture
def proof() -> Record:
    return {"schema": "metal-quantized-execution", "version": 1, "backend": "Metal", "producer": "cpu",
            "complete": True, "placement_verified": True, "activation_bits": 4, "weight_bits": 4,
            "dim": 16, "activation_mode": "EXSIA", "rmd_enabled": True,
            "observed_quantized_matmuls": 2, "metal_quantized_matmuls": 2, "prefill_matmuls": 1,
            "decode_matmuls": 1, "block_calls": 0, "hp1_calls": 2, "dense_launches": 2,
            "residual_launches": 1, "merge_launches": 1, "failed_calls": 0, "fallback_calls": 0,
            "tensor_placement": [{"weight": "blk.0.attn_q.weight", "type": "q4_hp1",
                "weight_backend": "Metal", "output_backend": "Metal", "calls": 2}],
            "scheduler_observer": "ask_only_with_split_synchronization",
            "seconds": {"producer": 0.1, "transfer": 0.01, "dense_gpu": 0.2,
                        "residual_gpu": 0.03, "merge_gpu": 0.02, "matmul_total": 0.4}}


def test_complete_kernel_and_tensor_coverage_is_admitted(tmp_path: Path, profile: Record, proof: Record) -> None:
    # Given: two observed matmuls have successful custom kernels and actual Metal buffers.
    proof_path, log = tmp_path / "proof.json", tmp_path / "process.log"
    proof_path.write_text(json.dumps(proof))
    log.write_text("fixture log\n")
    # When: the collector binds the evidence to the compiled profile.
    result = metal_placement(proof_path, log, profile)
    # Then: only quantized matmul execution is certified, with CPU producer explicitly retained.
    assert result["placement_complete"] is True
    assert result["completed_quantized_matmuls"] == 2
    assert result["producer"] == "cpu"


@pytest.mark.parametrize("mutation", [
    {"hp1_calls": 1}, {"metal_quantized_matmuls": 1}, {"failed_calls": 1},
    {"fallback_calls": 1}, {"merge_launches": 0}, {"decode_matmuls": 0},
    {"activation_bits": 8}, {"block_calls": 2, "hp1_calls": 0},
    {"tensor_placement": [{"weight": "blk.0.attn_q.weight", "type": "q4_hp1",
        "weight_backend": "Metal", "output_backend": "CPU", "calls": 2}]},
    {"tensor_placement": [{"weight": "blk.0.attn_q.weight", "type": "q4_0",
        "weight_backend": "Metal", "output_backend": "Metal", "calls": 2}]},
])
def test_incomplete_or_changed_execution_is_rejected(
        tmp_path: Path, profile: Record, proof: Record, mutation: Record) -> None:
    # Given: layer offload looks successful, but numerical kernel/placement coverage is wrong.
    proof_path, log = tmp_path / "proof.json", tmp_path / "process.log"
    proof_path.write_text(json.dumps({**proof, **mutation}))
    log.write_text("offloaded 13/13 layers to GPU\n")
    # When/Then: generic GPU layer counts cannot hide the missing custom execution.
    with pytest.raises(EvaluationError):
        metal_placement(proof_path, log, profile)


def test_a16_profile_is_not_admitted_as_a4_or_a8(profile: Record) -> None:
    # Given: the compile profile requests an unverified accumulator width.
    profile.update(activation_bits=16, weight_bits=16)
    # When/Then: E2E collection rejects it before launching a model.
    with pytest.raises(EvaluationError, match="matched A4W4 or A8W8"):
        validate_metal_build(profile)


def test_block_fused_restoration_needs_no_separate_merge(
        tmp_path: Path, profile: Record, proof: Record) -> None:
    # Given: BLOCK performs correction and scale restoration inside its dense dispatch.
    profile.update(activation_mode="BLOCK")
    proof.update(activation_mode="BLOCK", block_calls=2, hp1_calls=0,
                 residual_launches=0, merge_launches=0,
                 tensor_placement=[{"weight": "blk.0.attn_q.weight", "type": "q4_0",
                     "weight_backend": "Metal", "output_backend": "Metal", "calls": 2}])
    proof_path, log = tmp_path / "proof.json", tmp_path / "process.log"
    proof_path.write_text(json.dumps(proof))
    log.write_text("fixture log\n")
    # When: the actual fused execution is validated.
    result = metal_placement(proof_path, log, profile)
    # Then: it satisfies coverage without invented separate merge launches.
    assert result["arithmetic_contract"] == "BLOCK32_INTEGER_DOT"


def test_diagnostic_run_cannot_become_full_application_measurement() -> None:
    # Given: even a 128-token diagnostic is explicitly marked as smoke.
    endpoint: Record = {"schema": "potal-application-endpoints", "version": 1,
        "workload": "E2E_GENERATION_256_128", "complete": True,
        "diagnostic_smoke": True, "samples": 128, "decode_calls": 127}
    # When/Then: application latency admission preserves that diagnostic scope.
    with pytest.raises(EvaluationError, match="diagnostic smoke"):
        application_result(endpoint)
