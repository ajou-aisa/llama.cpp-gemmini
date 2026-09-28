#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: scripts/eval/run_cycle_trace_capture.sh --help
"""Build, run actual inference, capture the production NPU trace and certify it by exact equivalence."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final

from campaign_build import REPO, build, command
from campaign_inputs import checksums, dataset_input, model_metadata
from eval_common import (
    EvaluationError,
    Json,
    Record,
    clean_environment,
    compiled_info,
    read_json,
    record,
    require,
    sha256,
    text,
    write_json,
)

sys.path.insert(0, str(REPO))
from evaluation.manifest import Manifest

MODELS: Final = {"gpt2": ("GPT-2 124M", "GPT-2-124M", ("gpt2", 12)),
                 "llama3.2-1B": ("Llama-3.2-1B", "Llama-3.2-1B", ("llama", 16))}
MATMUL_MODE: Final = "STRIPE_PIPELINE"
WORKLOAD_POLICY: Final = ("CERTIFIED_CYCLE_PREFILL_256_PLUS_1:split=test:chunk=0:context=256:"
                          "output=last_token:sample=1:decode=0:seed=1234:temp=0:matmul=STRIPE_PIPELINE")
METRIC_POLICY: Final = "METRIC_PREFILL_256:split=test:max_chunks=1:context=256:tail=drop:output=second_half"
EXIT_REJECTED: Final = 3


def configuration_path(model: str, precision: str, dim: int) -> Path:
    return Path(MODELS[model][1]) / precision.upper() / f"DIM{dim}"


def git_identity(repository: Path) -> Record:
    def git(*args: str) -> bytes:
        return subprocess.run(["git", "-C", str(repository), *args], capture_output=True, check=True).stdout

    return {"root": str(repository), "head": git("rev-parse", "HEAD").decode().strip(),
            "branch": git("branch", "--show-current").decode().strip(),
            "tracked_diff_sha256": hashlib.sha256(git("diff", "--binary")).hexdigest(),
            "staged_diff_sha256": hashlib.sha256(git("diff", "--cached", "--binary")).hexdigest(),
            "status_sha256": hashlib.sha256(git("status", "--short")).hexdigest()}


def compiler(build_dir: Path) -> Record:
    cache = (build_dir / "CMakeCache.txt").read_text(encoding="utf-8").splitlines()
    values = dict(line.split("=", 1) for line in cache if line.startswith(("CMAKE_C_COMPILER:", "CMAKE_CXX_COMPILER:")))
    result: Record = {}
    for key, value in sorted(values.items()):
        version = subprocess.run([value, "--version"], capture_output=True, text=True, check=True).stdout
        result[key.split(":", 1)[0]] = {"path": value, "version": version.strip().splitlines()[0]}
    return result


def compress(raw: Path, target: Path) -> Record:
    content = hashlib.sha256()
    with (raw.open("rb") as source, target.open("xb") as sink,
          gzip.GzipFile(filename="", mode="wb", fileobj=sink, mtime=0, compresslevel=6) as stream):
        while chunk := source.read(1 << 24):
            content.update(chunk)
            stream.write(chunk)
    restored = hashlib.sha256()
    with gzip.open(target, "rb") as stream:
        while chunk := stream.read(1 << 24):
            restored.update(chunk)
    require(content.hexdigest() == restored.hexdigest() == sha256(raw), "compressed trace differs from native trace")
    return {"path": str(target), "sha256": sha256(target), "content_sha256": content.hexdigest(),
            "content_bytes": raw.stat().st_size, "compression": "gzip:mtime=0:level=6:no-filename"}


def prepared_runner(path: Path, precision: str, dim: int) -> Path:
    prepared = path.resolve(strict=True)
    receipt = read_json(prepared / "build-receipt.json")
    runner = prepared / "bin/llama-eval-workload"
    options = record(receipt["options"])
    require(receipt.get("kind") == "cycle" and receipt.get("verification") == "PASS" and
            receipt.get("runner_sha256") == sha256(runner) and
            options.get("GGML_GEMMINI_DEFAULT_MATMUL_MODE") == MATMUL_MODE and
            options.get("GGML_GEMMINI_DIM") == str(dim) and
            options.get("GGML_GEMMINI_ACTIVATION_BITS") == precision[1], "prepared cycle build receipt mismatch")
    return runner


def certify(args: argparse.Namespace, trace: Path, runner: Path, manifest: Path, directory: Path,
            im2p: Path) -> tuple[int, Record]:
    directory.mkdir(parents=True, exist_ok=False)
    attempts: list[Json] = []
    output = directory / "cycle_trace_certificate.json"
    for index, (parent, evidence_root) in enumerate(args.parent or []):
        name = f"build-{index}"
        try:
            command([sys.executable, "-B", "-m", "sim.cycle.cycle_trace_certificate", "build",
                     "--trace", str(trace), "--parent-certificate", str(Path(parent).resolve(strict=True)),
                     "--evidence-root", str(Path(evidence_root).resolve(strict=True)), "--producer", str(runner),
                     "--evaluation-manifest", str(manifest), str(output)], directory, name, args.timeout, im2p)
        except EvaluationError:
            lines = (directory / (name + ".log")).read_text(encoding="utf-8").strip().splitlines()
            attempts.append({"parent": parent, "status": "REJECTED", "reason": lines[-1] if lines else "no output"})
            continue
        command([sys.executable, "-B", "-m", "sim.cycle.cycle_trace_certificate", "verify", str(output)],
                directory, "verify", args.timeout, im2p)
        attempts.append({"parent": parent, "status": "VERIFIED"})
        result: Record = {"status": "VERIFIED", "certificate": {"path": str(output), "sha256": sha256(output)},
                          "attempts": attempts}
        write_json(directory / "verification.json", result)
        checksums(directory)
        return 0, result
    result = {"status": "REJECTED", "certificate": None, "attempts": attempts,
              "reason": "no supplied certified corpus is exactly equivalent to this production trace"
              if attempts else "no certified parent corpus supplied",
              "cycle_execution": "NOT_RUN"}
    write_json(directory / "rejection.json", result)
    checksums(directory)
    return EXIT_REJECTED, result


def capture(args: argparse.Namespace) -> int:
    clean_environment()
    require(0 <= args.seed < 4294967295 and args.timeout > 0 and args.jobs > 0, "invalid capture bounds")
    root = args.campaign_root.resolve()
    relative = configuration_path(args.model, args.precision, args.dim)
    config = root / "trace/evaluation-cycle" / relative
    config.mkdir(parents=True, exist_ok=False)
    im2p = args.im2p.resolve(strict=True)
    bits = int(args.precision[1])
    model = (args.model_path or REPO / "models" / args.model / f"{args.model}.Q{bits}_HP1.gguf").resolve(strict=True)
    tokenizer, architecture, blocks, file_type = model_metadata(model)
    require(file_type == (44 if bits == 4 else 40), "model is not matching HP1 precision")
    require((architecture, blocks) == MODELS[args.model][2], "model is not the requested architecture")
    dataset = dataset_input(args.dataset_manifest, REPO / "wikitext-2-raw/wiki.test.raw", config / "dataset_manifest.json")
    runner = (prepared_runner(args.prepared_build, args.precision, args.dim) if args.prepared_build
              else build("cycle", args.precision, args.dim, config / "build", im2p, args.jobs, MATMUL_MODE))
    info = compiled_info(runner)
    require(all(info.get(key) == 0 for key in ("activation_metrics", "residual_metrics", "scale_metrics")) and
            info.get("cycle_sim") == 1 and info.get("backend") == "IM2P_SIM" and info.get("matmul_mode") == MATMUL_MODE
            and info.get("dim") == args.dim and info.get("activation_bits") == bits == info.get("weight_bits"),
            "capture requires metrics-OFF CYCLE_SIM IM2P_SIM STRIPE_PIPELINE build of the requested profile")
    receipt_path = runner.parent.parent / "build-receipt.json"
    receipt = read_json(receipt_path)
    llama = git_identity(REPO)
    capture_dir = config / "capture"
    capture_dir.mkdir()
    manifest_path = capture_dir / "evaluation-manifest.json"
    write_json(manifest_path, {"model": MODELS[args.model][0], "dataset": "WikiText-2", "tokenizer_sha256": tokenizer,
               "chunk_policy": METRIC_POLICY, "precision": args.precision.upper(), "dim": args.dim, "BK": 32,
               "seed": args.seed, "git_sha": text(llama, "head")})
    manifest = Manifest.load(manifest_path)
    shared = root / "manifest" / relative / "evaluation_manifest.json"
    shared.parent.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(manifest_path, shared)
    require(sha256(shared) == manifest.sha256, "shared manifest copy differs")
    native = config / "native"
    argv = [str(runner), "--model", str(model), "--file", str(dataset), "--output-dir", str(native),
            "--workload", "E2E_GENERATION_256_128", "--max-chunks", "1", "--chunk-index", "0",
            "--smoke-generated-tokens", "1", "--seed", str(args.seed), "--temp", "0", "--batch-size", "256",
            "--ubatch-size", "256", "--threads", "1", "--threads-batch", "1", "--gpu-layers", "0",
            "--manifest-sha256", manifest.sha256]
    command(argv, config, "inference", args.timeout)
    workload = read_json(native / "workload.json")
    require(workload.get("complete") is True and workload.get("sampling_executed") is True and
            workload.get("requested_generated_tokens") == 1 and workload.get("output_mask") == "last_token" and
            workload.get("chunk_index") == 0 and workload.get("seed") == args.seed and
            workload.get("context_tokens") == 256, "native workload differs from certified cycle recipe")
    raw = native / "chunk-0/npu-cycle-trace.jsonl"
    trace_ref = compress(raw, capture_dir / "trace.jsonl.gz")
    raw.unlink()
    trace = capture_dir / "trace.jsonl.gz"
    options = record(receipt["options"])
    write_json(capture_dir / "producer-manifest.json", {
        "schema": "im2p-cycle-trace-producer-v1", "manifest_sha256": manifest.sha256,
        "producer": {"path": str(runner), "sha256": sha256(runner)},
        "build_receipt": {"path": str(receipt_path), "sha256": sha256(receipt_path)}, "build_info": info,
        "inference": {"argv": list(argv), "receipt": {"path": str(config / "inference.json"),
                      "sha256": sha256(config / "inference.json")},
                      "log_sha256": sha256(config / "inference.log")},
        "workload": {"path": str(native / "workload.json"), "sha256": sha256(native / "workload.json")},
        "workload_policy": WORKLOAD_POLICY, "trace": trace_ref,
        "native_trace_retained_as": "capture/trace.jsonl.gz (byte-exact content; raw copy removed for disk)",
        "model": {"path": str(model), "sha256": sha256(model)},
        "dataset": {"path": str(dataset), "sha256": sha256(dataset)},
        "manifest_chunk_policy_scope": "metric input/output policy shared by metric runs; cycle recipe is workload_policy",
        "cycle_count_is_not_latency_ms": True})
    write_json(capture_dir / "source-binding.json", {
        "schema": "im2p-cycle-trace-source-binding-v1",
        "llama_cpp_gemmini": llama, "build_producer_git_sha": receipt.get("producer_git_sha"),
        "build_producer_diff_sha256": receipt.get("producer_diff_sha256"),
        "im2p_sim": git_identity(im2p), "gemmini_headers": git_identity(REPO.parent / "RISC-V-DynDNN-gemmini-include"),
        "compiler": compiler(runner.parent.parent), "build_flags": options,
        "metric_options": {key: options[key] for key in ("GGML_GEMMINI_ACT_METRICS", "GGML_GEMMINI_ACT_QUANT_METRICS",
                                                          "GGML_GEMMINI_RESIDUAL_METRICS", "GGML_GEMMINI_SCALE_METRICS")},
        "cycle_options": {key: options[key] for key in ("CYCLE_SIM", "GGML_GEMMINI_EXECUTION_BACKEND",
                          "IM2P_SIM_IMPLEMENTATION", "IM2P_SIM_ROOT", "GGML_GEMMINI_DEFAULT_MATMUL_MODE",
                          "GGML_GEMMINI_DIM", "GGML_GEMMINI_ACTIVATION_BITS", "GGML_GEMMINI_WEIGHT_BITS",
                          "GGML_GEMMINI_BLOCK_SIZE", "GGML_GEMMINI_OPTION", "GGML_GEMMINI_ENABLE_RMD",
                          "GGML_GEMMINI_DEFAULT_RMD_BACKEND")},
        "host": {"python": sys.version, "platform": platform.platform()}})
    checksums(capture_dir)
    status, certification = certify(args, trace, runner, manifest_path,
                                    root / "certificate" / relative, im2p)
    write_json(config / "capture-status.json", {
        "configuration": str(relative), "EVALUATION_CYCLE_TRACE_CAPTURE": "PASS",
        "trace_sha256": trace_ref["sha256"], "trace_content_sha256": trace_ref["content_sha256"],
        "manifest_sha256": manifest.sha256, "certificate_status": certification["status"]})
    print(json.dumps({"configuration": str(relative), "trace": str(trace),
                      "certificate": certification["status"]}))
    return status


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", choices=tuple(MODELS), required=True)
    result.add_argument("--model-path", type=Path)
    result.add_argument("--precision", type=str.lower, choices=("a4w4", "a8w8"), required=True)
    result.add_argument("--dim", type=int, choices=(16, 32, 64), required=True)
    result.add_argument("--campaign-root", type=Path, required=True,
                        help="evaluation-cycle-campaign-<timestamp> root; trace/, manifest/, certificate/ are created")
    result.add_argument("--parent", nargs=2, action="append", metavar=("CERTIFICATE", "EVIDENCE_ROOT"),
                        help="independently certified corpus to test for exact equivalence (repeatable)")
    result.add_argument("--dataset-manifest", type=Path)
    result.add_argument("--prepared-build", type=Path, help="verified cycle capture build directory to reuse")
    result.add_argument("--seed", type=int, default=1234)
    result.add_argument("--jobs", type=int, default=8)
    result.add_argument("--timeout", type=int, default=7200)
    result.add_argument("--im2p", type=Path, default=REPO.parent / "IM2P.sim")
    return result


def main() -> int:
    try:
        return capture(parser().parse_args())
    except (EvaluationError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"cycle trace capture failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
