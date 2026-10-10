#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# How to run: python3 scripts/experiment/verify-cuda-quality-ppl.py RESULT_DIRECTORY
# Also supported: uv run scripts/experiment/verify-cuda-quality-ppl.py RESULT_DIRECTORY

from __future__ import annotations

import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path


class InvalidResult(ValueError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Rejected CUDA PPL: {reason}")


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise InvalidResult(reason)


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    model: str
    method: str
    bits: int
    dim: int

    @classmethod
    def read(cls, path: Path) -> Case:
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        require(len(rows) == 1, "expected exactly one case")
        row = rows[0]
        result = cls(row["case"], row["model"], row["method"], int(row["bits"]), int(row["dim"]))
        require(result.model in ("gpt2", "llama"), "unknown model")
        require(result.method in ("RTN-WA1", "RTN-WA2", "PoTal"), "unknown method")
        require(result.bits in (4, 8) and result.dim in (16, 32, 64), "unsupported precision or DIM")
        require(result.method != "RTN-WA1" or result.dim == 16, "WA1 has no DIM sweep")
        return result


@dataclass(frozen=True, slots=True)
class Result:
    case: Case
    chunks: int
    tokens: int
    ppl: float
    seconds: float

    def tsv(self) -> str:
        return "\t".join(map(str, (self.case.name, self.case.method, self.case.model,
            self.case.bits, self.case.dim, self.chunks, self.tokens, self.ppl, self.seconds)))


def verify(directory: Path) -> Result:
    case = Case.read(directory / "case.tsv")
    require((directory / "exit-status.txt").read_text().strip() == "0", "process did not exit successfully")
    text = (directory / "ppl.log").read_text(encoding="utf-8", errors="replace")
    totals = re.findall(r"^PPL_TOTALS (.*)$", text, re.M)
    planned = re.findall(r"calculating perplexity over (\d+) chunks", text)
    proof = re.findall(r"^CUDA_CPU_EXACT_PROOF (.*)$", text, re.M)
    require(len(totals) == len(planned) == len(proof) == 1, "missing or duplicate final records")
    values = dict(field.split("=", 1) for field in totals[0].split())
    chunks, tokens = int(values["chunks"]), int(values["total_scored_tokens"])
    ppl, nll = float(values["corpus_ppl"]), float(values["total_nll"])
    require(chunks > 1 and chunks == int(planned[0]), "incomplete full-corpus chunks")
    records = [dict(field.split("=", 1) for field in row.split())
               for row in re.findall(r"\bPPL_CHUNK ([^\r\n]*)", text)]
    require([int(row["index"]) for row in records] == list(range(chunks)), "missing or duplicated chunk")
    require(all(int(row["scored_tokens"]) == 255 and math.isfinite(float(row["nll"])) for row in records),
            "invalid chunk score or context")
    require(tokens == chunks * 255 and math.isfinite(ppl) and ppl > 0 and math.isfinite(nll), "invalid totals")
    require(math.isclose(sum(float(row["nll"]) for row in records), nll, rel_tol=1e-10), "NLL total mismatch")
    require(math.isclose(math.log(ppl), nll / tokens, rel_tol=1e-10), "PPL is not corpus NLL/token")
    payload: dict[str, str | int | bool] = json.loads(proof[0])
    require(type(payload.get("complete")) is bool and type(payload.get("activation_fp16")) is bool, "invalid proof flags")
    counters = ("version", "bits", "dim", "scored_tokens", "observed_matmuls", "verified_matmuls",
        "float_gpu_calls", "integer_gpu_launches", "residual_gpu_launches", "attention_gpu_calls",
        "observed_attention_matmuls", "verified_attention_matmuls", "cpu_q6_head_matmuls", "gpu_q6_head_matmuls")
    require(all(type(payload.get(key)) is int and int(payload[key]) >= 0 for key in counters), "invalid proof counters")
    hp1, residual = case.method != "RTN-WA1", case.method == "PoTal"
    expected = {"schema": "cuda-cpu-exact-ppl", "version": 2, "complete": True,
        "graph": "cpu_equivalent_cuda", "compute": "INT", "activation": "EXSIA" if hp1 else "BLOCK",
        "bits": case.bits, "dim": case.dim, "scored_tokens": tokens, "activation_fp16": False,
        "float_gpu_calls": 0, "gpu_q6_head_matmuls": 0,
        "cpu_q6_head_matmuls": chunks if not hp1 and case.bits == 4 else 0}
    require(all(payload.get(key) == value for key, value in expected.items()), "proof profile mismatch")
    require(int(payload["observed_matmuls"]) > 0 and payload["observed_matmuls"] == payload["verified_matmuls"]
        and int(payload["integer_gpu_launches"]) > 0, "quantized matmul CPU fallback")
    attention = 2 * {"gpt2": 12, "llama": 16}[case.model] * chunks
    require(all(payload[key] == attention for key in
        ("attention_gpu_calls", "observed_attention_matmuls", "verified_attention_matmuls")), "attention coverage")
    require((int(payload["residual_gpu_launches"]) > 0) == residual, "wrong residual route")
    cache = dict(re.findall(r"^([^/#][^:\n]*):[^=\n]+=(.*)$", (directory / "CMakeCache.txt").read_text(), re.M))
    settings = {"GGML_GEMMINI_CUDA_CPU_EXACT": "ON", "GGML_CUDA": "OFF", "GGML_METAL": "OFF",
        "GGML_GEMMINI": "ON", "GGML_GEMMINI_METAL_CPU_EXACT": "OFF", "GGML_BACKEND_DL": "OFF",
        "GGML_GEMMINI_COMPUTE_TYPE": "INT", "GGML_GEMMINI_ACTIVATION_BITS": str(case.bits),
        "GGML_GEMMINI_WEIGHT_BITS": str(case.bits), "GGML_GEMMINI_DIM": str(case.dim),
        "GGML_GEMMINI_BLOCK_SIZE": "32", "GGML_GEMMINI_ACTIVATION_QUANT": "EXSIA" if hp1 else "BLOCK",
        "GGML_GEMMINI_ENABLE_RMD": "ON" if residual else "OFF",
        "GGML_GEMMINI_EXSIA_OUTLIER_SELECTION": "ON" if residual else "OFF"}
    require(all(cache.get(key) == value for key, value in settings.items()), "build profile mismatch")
    command = (directory / "command.txt").read_text()
    require("--chunks -1 " in command and "--ctx-size 512 " in command, "not a full-corpus command")
    elapsed = re.findall(r"^real\s+([0-9.]+)$", text, re.M)
    require(len(elapsed) == 1 and math.isfinite(float(elapsed[0])), "missing elapsed time")
    return Result(case, chunks, tokens, ppl, float(elapsed[0]))


def verify_smoke(cpu: Path, gpu: Path) -> None:
    cpu_text, gpu_text = cpu.read_text(), gpu.read_text()
    reference = re.findall(r"^PPL_TOTALS (.*)$", cpu_text, re.M)
    candidate = re.findall(r"^PPL_TOTALS (.*)$", gpu_text, re.M)
    require(len(reference) == 1 and reference == candidate, "CPU/CUDA smoke PPL totals differ")
    require("chunks=1 " in reference[0], "smoke did not finish one chunk")
    proof = re.findall(r"^CUDA_CPU_EXACT_PROOF (.*)$", gpu_text, re.M)
    require(len(proof) == 1, "missing CUDA smoke proof")
    payload: dict[str, str | int | bool] = json.loads(proof[0])
    require(payload.get("complete") is True and payload.get("graph") == "cpu_equivalent_cuda" and
            int(payload.get("integer_gpu_launches", 0)) > 0, "smoke did not use CUDA")
    require(payload.get("observed_matmuls") == payload.get("verified_matmuls") and
            payload.get("observed_attention_matmuls") == payload.get("verified_attention_matmuls"), "smoke GPU fallback")
    print("PASS CPU/CUDA one-chunk corpus PPL totals match exactly")


def main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] == "--smoke":
        verify_smoke(Path(sys.argv[2]), Path(sys.argv[3]))
        return
    if len(sys.argv) != 2:
        raise SystemExit("Usage: verify-cuda-quality-ppl.py RESULT_DIRECTORY")
    try:
        result = verify(Path(sys.argv[1]))
    except (InvalidResult, OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(str(error)) from error
    print(result.tsv())


if __name__ == "__main__":
    main()
