#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pytest"]
# ///
# How to run: python3 -B -m pytest tests/test-cuda-quality-ppl.py
from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cuda_ppl_verify", ROOT / "scripts/experiment/verify-cuda-quality-ppl.py")
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VERIFY
SPEC.loader.exec_module(VERIFY)


@pytest.fixture
def run_directory(tmp_path: Path) -> Path:
    (tmp_path / "case.tsv").write_text("case\tmodel\tmethod\tbits\tdim\ngpt2-potal4-d16\tgpt2\tPoTal\t4\t16\n")
    (tmp_path / "exit-status.txt").write_text("0\n")
    (tmp_path / "command.txt").write_text("llama-perplexity --ctx-size 512 --chunks -1 --no-warmup\n")
    settings = {"GGML_GEMMINI_CUDA_CPU_EXACT": "ON", "GGML_CUDA": "OFF", "GGML_METAL": "OFF",
        "GGML_GEMMINI": "ON", "GGML_GEMMINI_METAL_CPU_EXACT": "OFF", "GGML_BACKEND_DL": "OFF",
        "GGML_GEMMINI_COMPUTE_TYPE": "INT", "GGML_GEMMINI_ACTIVATION_BITS": "4",
        "GGML_GEMMINI_WEIGHT_BITS": "4", "GGML_GEMMINI_DIM": "16", "GGML_GEMMINI_BLOCK_SIZE": "32",
        "GGML_GEMMINI_ACTIVATION_QUANT": "EXSIA", "GGML_GEMMINI_ENABLE_RMD": "ON",
        "GGML_GEMMINI_EXSIA_OUTLIER_SELECTION": "ON"}
    (tmp_path / "CMakeCache.txt").write_text("".join(f"{k}:STRING={v}\n" for k, v in settings.items()))
    proof = {"schema": "cuda-cpu-exact-ppl", "version": 2, "complete": True, "graph": "cpu_equivalent_cuda",
        "compute": "INT", "activation": "EXSIA", "bits": 4, "dim": 16, "scored_tokens": 510,
        "observed_matmuls": 6, "verified_matmuls": 6, "float_gpu_calls": 0, "integer_gpu_launches": 8,
        "residual_gpu_launches": 4, "attention_gpu_calls": 48, "observed_attention_matmuls": 48,
        "verified_attention_matmuls": 48, "cpu_q6_head_matmuls": 0, "gpu_q6_head_matmuls": 0,
        "activation_fp16": False}
    (tmp_path / "ppl.log").write_text("calculating perplexity over 2 chunks\n"
        "PPL_CHUNK index=0 nll=1000 scored_tokens=255\nPPL_CHUNK index=1 nll=1000 scored_tokens=255\n"
        f"PPL_TOTALS total_nll=2000 total_scored_tokens=510 chunks=2 corpus_ppl={math.exp(2000/510):.17g}\n"
        f"CUDA_CPU_EXACT_PROOF {json.dumps(proof)}\nreal 20.3\n")
    return tmp_path


def test_full_run_returns_corpus_ppl(run_directory: Path) -> None:
    result = VERIFY.verify(run_directory)
    assert result.ppl == math.exp(2000/510) and result.tokens == 510


@pytest.mark.parametrize(("filename", "before", "after"), [
    ("ppl.log", "index=1", "index=0"),
    ("ppl.log", "over 2 chunks", "over 3 chunks"),
    ("ppl.log", '"verified_matmuls": 6', '"verified_matmuls": 5'),
    ("ppl.log", '"attention_gpu_calls": 48', '"attention_gpu_calls": 47'),
    ("ppl.log", '"residual_gpu_launches": 4', '"residual_gpu_launches": 0'),
    ("ppl.log", '"integer_gpu_launches": 8', '"integer_gpu_launches": true'),
    ("ppl.log", "total_nll=2000", "total_nll=2001"),
    ("ppl.log", '"complete": true', '"complete": false'),
    ("ppl.log", '"complete": true', '"complete": 1'),
    ("command.txt", "--chunks -1", "--chunks 2"),
    ("CMakeCache.txt", "EXSIA_OUTLIER_SELECTION:STRING=ON", "EXSIA_OUTLIER_SELECTION:STRING=OFF"),
    ("exit-status.txt", "0", "1"),
])
def test_rejects_incomplete_or_different_execution(run_directory: Path, filename: str, before: str, after: str) -> None:
    artifact = run_directory / filename
    artifact.write_text(artifact.read_text().replace(before, after))
    with pytest.raises(VERIFY.InvalidResult):
        VERIFY.verify(run_directory)


def test_list_contains_only_the_pending_22_cases() -> None:
    command = ["bash", str(ROOT / "scripts/experiment/run-cuda-quality-ppl.sh"), "--list"]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    rows = [line.split("\t") for line in result.stdout.splitlines()[1:]]
    assert len({row[0] for row in rows}) == 22
    assert Counter(row[2] for row in rows) == {"PoTal": 6, "RTN-WA1": 4, "RTN-WA2": 12}
    assert all(row[1] == "gpt2" for row in rows if row[2] == "PoTal")


def test_unknown_case_fails_before_gpu_work() -> None:
    result = subprocess.run(["bash", str(ROOT / "scripts/experiment/run-cuda-quality-ppl.sh"),
        "--run", "--case", "llama-rtnw8"], text=True, capture_output=True)
    assert result.returncode != 0


def test_smoke_rejects_any_corpus_ppl_difference(tmp_path: Path) -> None:
    cpu, gpu = tmp_path / "cpu.log", tmp_path / "gpu.log"
    cpu.write_text("PPL_TOTALS total_nll=1000 total_scored_tokens=255 chunks=1 corpus_ppl=50\n")
    gpu.write_text("PPL_TOTALS total_nll=1001 total_scored_tokens=255 chunks=1 corpus_ppl=50\n")
    with pytest.raises(VERIFY.InvalidResult):
        VERIFY.verify_smoke(cpu, gpu)


def test_smoke_requires_observed_cuda_work(tmp_path: Path) -> None:
    cpu, gpu = tmp_path / "cpu.log", tmp_path / "gpu.log"
    totals = "PPL_TOTALS total_nll=1000 total_scored_tokens=255 chunks=1 corpus_ppl=50\n"
    cpu.write_text(totals)
    gpu.write_text(totals)
    with pytest.raises(VERIFY.InvalidResult):
        VERIFY.verify_smoke(cpu, gpu)
