"""Fresh, independently configured measurement builds and command receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Final

from eval_common import (
    Record,
    artifact_snapshot,
    compiled_info,
    require,
    sha256,
    write_json,
)

REPO: Final = Path(__file__).resolve().parents[2]
TARGETS: Final = ["llama-eval-workload", "test-evaluation-workload", "test-evaluation-trace",
                 "test-gemmini-evaluation-metrics", "test-cycle-sim-reader",
                 "test-gemmini-cycle-sim-log", "test-cycle-sim-coverage"]


def command(argv: list[str], directory: Path, name: str, timeout: int = 1800) -> None:
    """Run one bounded command, retaining its exact argv, output and exit status."""
    with (directory / (name + ".log")).open("x", encoding="utf-8") as log:
        try:
            result = subprocess.run(argv, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            write_json(directory / (name + ".json"), {"argv": list(argv), "exit_code": None,
                       "status": "TIMEOUT", "timeout_seconds": timeout})
            raise
    write_json(directory / (name + ".json"), {"argv": list(argv), "cwd": str(REPO),
               "exit_code": result.returncode})
    require(result.returncode == 0, "command failed: " + str(directory / (name + ".log")))


def build(kind: str, precision: str, dim: int, output: Path, im2p: Path, jobs: int) -> Path:
    """Configure, compile and verify a new directory; never reuse a user's build cache."""
    require(kind in ("cycle", "activation", "residual", "scu"), "invalid measurement kind")
    require(precision in ("a4w4", "a8w8") and dim in (16, 32, 64) and jobs > 0,
            "invalid measurement build profile")
    output.mkdir(parents=True, exist_ok=False)
    producer_head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    source_diff = subprocess.check_output(["git", "-C", str(REPO), "diff", "--binary"])
    with (output / "producer.diff").open("xb") as stream:
        stream.write(source_diff)
    bits = 4 if precision == "a4w4" else 8
    options = {
        "CMAKE_BUILD_TYPE": "Release", "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
        "BUILD_SHARED_LIBS": "OFF", "LLAMA_BUILD_TESTS": "ON", "LLAMA_BUILD_TOOLS": "ON",
        "LLAMA_CURL": "OFF", "GGML_GEMMINI": "ON", "GGML_METAL": "OFF",
        "GGML_ACCELERATE": "OFF", "GGML_BLAS": "OFF", "GGML_CUDA": "OFF",
        "GGML_OPENMP": "OFF", "CYCLE_SIM": "1", "LOG_CYCLE": "0", "LOG_DEBUG": "0",
        "CYCLE_DETAIL": "0", "GGML_CPU_CYCLE_LOG": "OFF", "GGML_BACKEND_DL": "OFF",
        "GGML_GEMMINI_EXECUTION_BACKEND": "IM2P_SIM", "IM2P_SIM_IMPLEMENTATION": "GEMMINI_HP1",
        "IM2P_SIM_ROOT": str(im2p), "GGML_GEMMINI_OPTION": "WS",
        "GGML_GEMMINI_ACTIVATION_QUANT": "EXSIA", "GGML_GEMMINI_BLOCK_SIZE": "32",
        "GGML_GEMMINI_ACTIVATION_BITS": str(bits), "GGML_GEMMINI_WEIGHT_BITS": str(bits),
        "GGML_GEMMINI_DIM": str(dim), "GGML_GEMMINI_DEFAULT_MATMUL_MODE": "FULL",
        "GGML_GEMMINI_DEFAULT_RMD_BACKEND": "WS", "GGML_GEMMINI_ENABLE_RMD": "ON",
        "GGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE": "OFF",
        "GGML_GEMMINI_ACT_METRICS": str(int(kind == "activation")),
        "GGML_GEMMINI_ACT_QUANT_METRICS": "0",
        "GGML_GEMMINI_RESIDUAL_METRICS": str(int(kind == "residual")),
        "GGML_GEMMINI_SCALE_METRICS": str(int(kind == "scu")),
    }
    command(["cmake", "-S", str(REPO), "-B", str(output),
             *(f"-D{key}={value}" for key, value in options.items())], output, "configure")
    command(["cmake", "--build", str(output), "--parallel", str(jobs), "--target", *TARGETS],
            output, "build")
    command(["ctest", "--test-dir", str(output), "--output-on-failure", "-R",
             ("^(test-evaluation-(workload|trace|build-options|metric-framework)|"
             "test-gemmini-(evaluation-metrics|cycle-sim-log)|"
             "test-cycle-sim-(reader|coverage|build-contract))$")], output, "verify")
    runner = output / "bin/llama-eval-workload"
    info = compiled_info(runner)
    expected = (int(kind == "activation"), int(kind == "residual"), int(kind == "scu"))
    require(tuple(info.get(key) for key in ("activation_metrics", "residual_metrics", "scale_metrics"))
            == expected, "compiled collectors are not independent")
    require(info.get("dim") == dim and info.get("activation_bits") == bits and
            info.get("weight_bits") == bits and info.get("cycle_sim") == 1,
            "compiled build profile differs from request")
    write_json(output / "build-info.json", info)
    write_json(output / "artifacts.json", artifact_snapshot(runner))
    write_json(output / "build-receipt.json", {"kind": kind, "options": dict(options),
               "runner_sha256": sha256(runner), "verification": "PASS", "producer_git_sha": producer_head,
               "producer_diff_sha256": hashlib.sha256(source_diff).hexdigest()})
    return runner


def snapshot(output: Path) -> None:
    """Preserve requested pre-change git evidence without modifying any checkout."""
    output.mkdir(parents=True, exist_ok=False)
    rows: Record = {}
    for name in ("llama.cpp-gemmini", "IM2P.sim", "RISC-V-DynDNN-gemmini-include"):
        repo = REPO.parent / name
        fields: Record = {}
        for label, args in (("status", ["status", "--short"]), ("diff", ["diff", "--binary"]),
                            ("branch", ["branch", "--show-current"]), ("head", ["rev-parse", "HEAD"])):
            result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                                    text=True, check=True)
            with (output / (name + "-" + label + ".txt")).open("x", encoding="utf-8") as stream:
                stream.write(result.stdout)
            fields[label] = result.stdout.strip()
        rows[name] = fields
    write_json(output / "repositories.json", rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("cycle", "activation", "residual", "scu"), required=True)
    parser.add_argument("--precision", choices=("a4w4", "a8w8"), required=True)
    parser.add_argument("--dim", type=int, choices=(16, 32, 64), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--im2p", type=Path, default=REPO.parent / "IM2P.sim")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    runner = build(args.kind, args.precision, args.dim, args.output.resolve(), args.im2p.resolve(), args.jobs)
    print(json.dumps({"runner": str(runner), "sha256": sha256(runner)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
