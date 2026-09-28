#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 -B scripts/eval/campaign.py activation --help
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from campaign_build import REPO, build, command
from campaign_inputs import checksums, dataset_input, model_metadata
from eval_common import (
    EvaluationError,
    Record,
    clean_environment,
    compiled_info,
    integer,
    read_json,
    record,
    require,
    sha256,
    text,
    write_json,
)

sys.path.insert(0, str(REPO))
from campaign_metrics import outputs
from metric_run import collect_metric, parser_for, validate_metric_recipe

from evaluation.manifest import Manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Independent build/run/manifest/evidence measurement; no PPL or latency.")
    result.add_argument("kind", choices=("cycle", "activation", "residual", "scu"))
    result.add_argument("--model", choices=("gpt2", "llama3.2-1B"), required=True)
    result.add_argument("--model-path", type=Path)
    result.add_argument("--precision", type=str.lower, choices=("a4w4", "a8w8"), required=True)
    result.add_argument("--dim", type=int, choices=(16, 32, 64), required=True)
    result.add_argument("--dataset-manifest", type=Path)
    result.add_argument("--output", type=Path)
    result.add_argument("--max-chunks", type=int, default=1, help="1: smoke; 0: all complete WikiText-2 test chunks")
    result.add_argument("--seed", type=int, default=1234)
    result.add_argument("--jobs", type=int, default=4)
    result.add_argument("--timeout", type=int, default=1800)
    result.add_argument("--im2p", type=Path, default=REPO.parent / "IM2P.sim")
    result.add_argument("--prepared-build", type=Path, help="explicit previously verified build_measurement.sh directory")
    result.add_argument("--workload-manifest", type=Path, help="optional exact native chunk identity from another metric")
    result.add_argument("--evaluation-manifest", type=Path,
                        help="shared v1 evaluation_manifest.json (e.g. from cycle trace capture); bytes are reused")
    result.add_argument("--trace-source", type=Path, help="existing actual-inference trace; omission captures a fresh prefill")
    result.add_argument("--source-provenance", type=Path, help="required for existing trace replay")
    result.add_argument("--certificate", type=Path, help="stateful production certificate admitting exact trace")
    result.add_argument("--evidence-root", type=Path, help="existing certificate evidence root")
    result.add_argument("--library", type=Path, help="explicit certified cycle runtime library")
    return result


def expected_layers(architecture: str, blocks: int) -> set[str]:
    attention = ("qkv_proj", "out_proj") if architecture == "gpt2" else ("q_proj", "k_proj", "v_proj", "out_proj")
    mlp = ("up_proj", "down_proj") if architecture == "gpt2" else ("up_proj", "gate_proj", "down_proj")
    return {"lm_head"} | {f"blk.{index}.{group}.{role}" for index in range(blocks)
                          for group, roles in (("attn", attention), ("mlp", mlp)) for role in roles}


def run_campaign(args: argparse.Namespace) -> Path:
    clean_environment()
    require(args.max_chunks >= 0 and 0 <= args.seed < 4294967295 and args.timeout > 0, "invalid campaign bounds")
    if args.kind == "cycle":
        require(args.certificate is not None and args.evidence_root is not None,
                "cycle campaign requires an exact stateful --certificate and --evidence-root; no implicit certification")
    else:
        require(all(value is None for value in (args.trace_source, args.source_provenance, args.certificate,
                args.evidence_root, args.library)), "metric campaigns do not accept cycle-provider inputs")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output or REPO.parent / ("evaluation-campaign-" + stamp) / args.kind).resolve()
    output.mkdir(parents=True, exist_ok=False)
    im2p = args.im2p.resolve(strict=True)
    bits = 4 if args.precision == "a4w4" else 8
    model = (args.model_path or REPO / "models" / args.model / f"{args.model}.Q{bits}_HP1.gguf").resolve(strict=True)
    tokenizer, architecture, blocks, file_type = model_metadata(model)
    require(file_type == (44 if bits == 4 else 40), "model is not matching HP1 precision")
    require((architecture, blocks) == (("gpt2", 12) if args.model == "gpt2" else ("llama", 16)),
            "model is not the requested GPT-2 124M or Llama-3.2-1B architecture")
    dataset = dataset_input(args.dataset_manifest, REPO / "wikitext-2-raw/wiki.test.raw", output / "dataset_manifest.json")
    if args.prepared_build is None:
        runner = build(args.kind, args.precision, args.dim, output / "build", im2p, args.jobs)
    else:
        prepared = args.prepared_build.resolve(strict=True)
        receipt = read_json(prepared / "build-receipt.json")
        runner = prepared / "bin/llama-eval-workload"
        require(receipt.get("kind") == args.kind and receipt.get("verification") == "PASS" and
                receipt.get("runner_sha256") == sha256(runner), "prepared build receipt mismatch")
    info = compiled_info(runner)
    require(info.get("dim") == args.dim and info.get("activation_bits") == bits and
            info.get("weight_bits") == bits, "prepared build precision/DIM mismatch")
    build_receipt = read_json(runner.parent.parent / "build-receipt.json")
    head = text(build_receipt, "producer_git_sha")
    policy = f"METRIC_PREFILL_256:split=test:max_chunks={args.max_chunks}:context=256:tail=drop:output=second_half"
    producer_hash = sha256(runner)
    if args.kind == "cycle" and args.trace_source is not None:
        require(args.max_chunks == 1, "existing source replay is exactly one certified chunk")
        require(args.source_provenance is not None, "source replay requires original inference provenance")
        original = read_json(args.source_provenance)
        head = text(record(record(original["source"])["llama_cpp_gemmini"]), "head")
        producer_hash = text(record(original["build"]), "binary_sha256")
        policy = "CERTIFIED_SOURCE_PREFILL_256:split=test:chunk=0:output=last_token:sample=1:decode=0"
    manifest_path = output / "evaluation_manifest.json"
    model_name = "GPT-2 124M" if args.model == "gpt2" else "Llama-3.2-1B"
    if args.evaluation_manifest is None:
        write_json(manifest_path, {"model": model_name,
            "dataset": "WikiText-2", "tokenizer_sha256": tokenizer, "tokenizer_hash": tokenizer,
            "precision": args.precision.upper(), "dim": args.dim, "DIM": args.dim, "BK": 32,
            "seed": args.seed, "git_sha": head, "build_hash": producer_hash, "chunk_policy": policy})
    else:
        require(args.kind != "cycle", "cycle campaigns bind the manifest through their trace certificate")
        shared = read_json(args.evaluation_manifest.resolve(strict=True))
        require(shared.get("model") == model_name and shared.get("tokenizer_sha256") == tokenizer and
                shared.get("precision") == args.precision.upper() and shared.get("dim") == args.dim and
                shared.get("seed") == args.seed and shared.get("git_sha") == head and
                shared.get("chunk_policy") == policy, "shared evaluation manifest differs from this measurement")
        with manifest_path.open("xb") as stream:
            stream.write(args.evaluation_manifest.resolve(strict=True).read_bytes())
    manifest = Manifest.load(manifest_path)
    binding: Record = {"manifest_sha256": manifest.sha256, "kind": args.kind,
        "model_path": str(model), "model_sha256": sha256(model), "dataset_path": str(dataset),
        "dataset_sha256": sha256(dataset), "tokenizer_hash_algorithm": "sorted-gguf-tokenizer-key-type-value-bytes-v1",
        "runner": str(runner), "build_hash": sha256(runner), "build_info": info,
        "build_receipt_sha256": sha256(runner.parent.parent / "build-receipt.json"),
        "command": [sys.executable, *sys.argv], "scope": "ONE_CHUNK_SMOKE" if args.max_chunks == 1 else
        "ALL_COMPLETE_CHUNKS" if args.max_chunks == 0 else "BOUNDED_CHUNKS",
        "E2E_RECONSTRUCTION_READY": "NOT_READY", "PAPER_CAMPAIGN_COMPLETE": "NOT_RUN"}
    if args.kind == "cycle":
        options: Record = {key: str(getattr(args, key)) if getattr(args, key) is not None else None
            for key in ("trace_source", "source_provenance", "certificate", "evidence_root", "library")}
        options.update({"im2p": str(im2p), "jobs": args.jobs, "max_chunks": args.max_chunks,
                        "seed": args.seed, "timeout": args.timeout})
        request = output / "cycle-request.json"
        write_json(request, {"options": options, "runner": str(runner), "model": str(model),
            "dataset": str(dataset), "manifest": manifest.identity(), "output": str(output), "binding": binding})
        try:
            command([sys.executable, "-B", str(Path(__file__).with_name("campaign_cycle.py")), str(im2p), str(request)],
                    output, "stateful-provider", args.timeout)
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            write_json(output / "failure.json", {"manifest_sha256": manifest.sha256, "error": str(error),
                       "EVALUATION_CYCLE_CAMPAIGN_READY": "NOT_READY"})
            checksums(output)
            raise
        binding = read_json(output / "cycle-binding.json")
    else:
        recipe = "scale" if args.kind == "scu" else args.kind
        validate_metric_recipe(info, recipe)
        argv = ["--runner", str(runner), "--model", str(model), "--dataset", str(dataset),
                "--split", "test", "--max-chunks", str(args.max_chunks), "--output", str(output / "collection"),
                "--evaluation-manifest", str(manifest_path), "--timeout", str(args.timeout)]
        if args.workload_manifest:
            argv.extend(["--workload-manifest", str(args.workload_manifest)])
        if recipe == "scale":
            argv.append("--gzip")
        raw, collected = collect_metric(parser_for(recipe).parse_args(argv), recipe)
        workload = record(collected["workload"])
        raw_chunks = workload["chunks"]
        require(isinstance(raw_chunks, list), "native chunks missing")
        chunks: set[int] = {integer(record(row), "chunk_id") for row in raw_chunks} if isinstance(raw_chunks, list) else set()
        outputs(args.kind, raw, manifest, output, expected_layers(architecture, blocks), chunks)
        binding["native_workload_identity"] = collected["native_workload_identity"]
        binding["layer_count"] = len(expected_layers(architecture, blocks))
        binding["lm_head_included"] = True
    require(binding["model_sha256"] == sha256(model) and binding["dataset_sha256"] == sha256(dataset) and
            binding["build_hash"] == sha256(runner), "campaign inputs changed")
    write_json(output / "manifest.json", binding)
    checksums(output)
    return output


def main() -> int:
    args = parser().parse_args()
    try:
        print(run_campaign(args))
        return 0
    except (EvaluationError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"campaign failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
