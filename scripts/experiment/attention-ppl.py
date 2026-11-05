#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# How to run: uv run scripts/experiment/attention-ppl.py --run --config FILE --output DIR
"""Run PoTal attention ablations using the established corpus queue and provenance."""
from collections.abc import Mapping
from importlib import import_module
import json
import math
import os
from pathlib import Path
import re
from typing import Final, TypedDict

queue = import_module("cuda-ppl-queue")
ROOT: Final = Path(__file__).resolve().parents[2]


class Job(TypedDict):
    id: str
    kind: str
    family: str
    bits: int
    anchor: int
    attention_scope: str
    binary: str
    model: str
    full_chunks: int


Proof = dict[str, str | int | bool]


class Result(TypedDict):
    chunks: int
    scored_tokens: int
    nll: float
    ppl: float
    cuda_proof: Proof
    scu_proof: Proof
    attention_proof: Proof


class Manifest(TypedDict):
    environment: dict[str, str]
    command: list[str]


original_settings: Final = queue.settings


def inputs(job: Job, dataset: Path) -> dict[str, str]:
    """Pin the exact executable, libraries, inputs, driver and enabled build profile."""
    binary = Path(job["binary"]).resolve()
    cache = binary.parent.parent / "CMakeCache.txt"
    config = dict(re.findall(r"^([A-Za-z_][A-Za-z0-9_]*):[A-Za-z_]+=(.*)$", cache.read_text(), re.M))
    expected = dict(GGML_GEMMINI_ENABLE_A4NKS_ATTENTION="ON", GGML_CUDA="OFF",
                    GGML_GEMMINI_OFFLOAD_A4NKS_ATTENTION="OFF", CYCLE_SIM="0",
                    GGML_GEMMINI_CUDA_CPU_EXACT="ON", GGML_GEMMINI_CUDA_RTN_ATTENTION="ON",
                    GGML_GEMMINI_ACTIVATION_QUANT="EXSIA", GGML_GEMMINI_ENABLE_RMD="ON",
                    GGML_GEMMINI_ACTIVATION_BITS=str(job["bits"]), GGML_GEMMINI_WEIGHT_BITS=str(job["bits"]),
                    GGML_GEMMINI_DIM="32", GGML_GEMMINI_BLOCK_SIZE="32", GGML_GEMMINI_EXSIA_SIGMA="2",
                    GGML_GEMMINI_DEFAULT_RMD_BACKEND="CPU", GGML_GEMMINI_DEFAULT_MATMUL_MODE="STRIPE_PIPELINE",
                    GGML_GEMMINI_COMPUTE_TYPE="INT", GGML_GEMMINI_DEQUANT_FP_TEST="OFF",
                    GGML_GEMMINI_EXSIA_STRIPE_ANCHOR=str(job["anchor"]), CMAKE_CUDA_ARCHITECTURES="86")
    queue.require(all(config.get(k) == v for k, v in expected.items()), "Wrong attention build: " + job["id"])
    queue.require(job["attention_scope"] in ("none", "qk", "pv", "both"), "Invalid attention scope")
    paths = [binary, ROOT / job["model"], dataset, cache, Path(__file__),
             ROOT / "scripts/experiment/cuda-ppl-queue.py", *sorted(binary.parent.glob("*.so*"))]
    queue.require(all(path.is_file() for path in paths) and os.access(binary, os.X_OK), "Missing attention input")
    return {str(path): queue.digest(path) for path in paths}


def settings(job: Job, dataset: Path, phase: str, identity: Mapping[str, str]) -> Manifest:
    """Retain the common PPL policy while selecting the quantized attention nodes."""
    manifest = original_settings(job, dataset, phase, identity)
    # This fork has an enable-only --flash-attn flag; its default is false.
    manifest["environment"].update(GGML_GEMMINI_ATTENTION_SCOPE=job["attention_scope"],
                                    GGML_GEMMINI_QK_SCHEDULE="full")
    return manifest


def verify(log: Path, job: Job, chunks: int) -> Result:
    """Reject partial corpora, incorrect node coverage, wrong anchors and SCU saturation."""
    text = log.read_text(errors="replace")
    totals = re.findall(r"^PPL_TOTALS (.*)$", text, re.M)
    planned = re.findall(r"calculating perplexity over (\d+) chunks", text)
    queue.require(len(totals) == len(planned) == 1, "Missing final totals")
    row = dict(field.split("=", 1) for field in totals[0].split())
    count, tokens = int(row["chunks"]), int(row["total_scored_tokens"])
    nll, ppl = float(row["total_nll"]), float(row["corpus_ppl"])
    entries = [dict(field.split("=", 1) for field in entry.split())
               for entry in re.findall(r"\bPPL_CHUNK ([^\r\n]*)", text)]
    queue.require(count == chunks == int(planned[0]), "Incomplete attention corpus")
    queue.require([int(c["index"]) for c in entries] == list(range(count)), "Missing or repeated chunks")
    queue.require(all(int(c["scored_tokens"]) == 255 and math.isfinite(float(c["nll"])) for c in entries)
                  and tokens == count * 255 and math.isfinite(nll) and math.isfinite(ppl) and ppl > 0,
                  "Invalid scored-token population")
    queue.require(math.isclose(sum(float(c["nll"]) for c in entries), nll, rel_tol=1e-10)
                  and math.isclose(math.log(ppl), nll / tokens, rel_tol=1e-10), "NLL/PPL mismatch")

    def proof(prefix: str) -> Proof:
        records = re.findall(r"^" + prefix + r" (\{[^{}]*\})\r?$", text, re.M)
        queue.require(len(records) == 1, "Missing or repeated " + prefix)
        return json.loads(records[0].replace("\r", "").replace("\n", ""))

    layers = {"qwen2-0.5B": 24, "llama3.2-1B": 16}[job["family"]]
    qk = layers * count if job["attention_scope"] in ("qk", "both") else 0
    pv = layers * count if job["attention_scope"] in ("pv", "both") else 0
    floating = 2 * layers * count - qk - pv
    dense = (7 * layers + 1) * count
    p, s, a = proof("CUDA_CPU_EXACT_PROOF"), proof("HP1_SCU_PROOF"), proof("POTAL_ATTENTION_PROOF")
    expected: Proof = dict(schema="cuda-cpu-exact-ppl", version=3, complete=True,
                           graph="cpu_main_equivalent_cuda", compute="INT", activation="EXSIA",
                           bits=job["bits"], dim=32, scored_tokens=tokens, activation_fp16=False,
                           float_gpu_calls=0, cpu_q6_head_matmuls=0, gpu_q6_head_matmuls=0,
                           observed_matmuls=dense, verified_matmuls=dense,
                           attention_backend="cuda_rtn_baseline", attention_gpu_calls=floating,
                           observed_attention_matmuls=floating, verified_attention_matmuls=floating)
    queue.require(all(p.get(k) == v for k, v in expected.items()), "Wrong dense/floating attention CUDA coverage")
    expected_attention: Proof = dict(schema="potal-attention-v1", complete=True,
                                     scope=job["attention_scope"], bits=job["bits"], anchor=job["anchor"],
                                     block_size=32, backend="cpu_quantized_scu", qk_nodes=qk, pv_nodes=pv,
                                     verified_nodes=qk + pv)
    queue.require(a == expected_attention, "Wrong quantized attention scope, anchor or execution")
    queue.require(p["integer_gpu_launches"] > 0 and p["residual_gpu_launches"] > 0
                  and s.get("schema") == "hp1-scu-v1" and s.get("dim") == 32
                  and s.get("final_merge") == "legacy_fp32_main_plus_residual"
                  and s.get("main_launches") == p["integer_gpu_launches"]
                  and s.get("residual_launches") == p["residual_gpu_launches"], "Missing dense main/residual SCU work")
    queue.require(all(s.get(key) == 0 for key in ("main_shift_saturations", "main_acc_saturations",
                                               "residual_shift_saturations", "residual_acc_saturations")),
                  "Dense SCU saturation")
    return dict(chunks=count, scored_tokens=tokens, nll=nll, ppl=ppl,
                cuda_proof=p, scu_proof=s, attention_proof=a)


if __name__ == "__main__":
    queue.inputs, queue.settings, queue.verify = inputs, settings, verify
    queue.main()
