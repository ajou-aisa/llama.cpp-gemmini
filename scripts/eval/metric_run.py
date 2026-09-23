from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from eval_common import (EvaluationError, Record, artifact_snapshot, clean_environment,
    compiled_info, read_json, record, require, run, sha256, validate_recipe, write_json)
from metric_reducers import activation_summary, residual_summary


def parser_for(recipe: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Independent {recipe} metric collection; native WikiText-2 prefills.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--runner", type=Path, help="built llama-eval-workload executable")
    source.add_argument("--reduce", type=Path, help="reduce an existing complete dedicated JSONL; no new collection claim")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--workload-manifest", type=Path, help="prior workload-binding.json; require same native chunk identities")
    parser.add_argument("--output", type=Path, required=True, help="fresh output directory; never overwritten")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--threads-batch", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ubatch-size", type=int, default=256)
    parser.add_argument("--max-chunks", type=int, default=0, help="0 means all complete native 256-token chunks")
    parser.add_argument("--timeout", type=int, default=600, help="finite collection timeout in seconds")
    if recipe == "residual":
        parser.add_argument("--accept-proposed-weighting", action="store_true",
                            help="record explicit use of proposed-v3-DTR-v1; does not change raw shapes")
    return parser


def workload_identity(workload: Record) -> str:
    keys = ("workload", "tokens", "complete_chunks", "selected_chunks", "dropped_tail_tokens",
            "context_tokens", "batch", "ubatch", "threads", "threads_batch", "add_special",
            "parse_special", "trailing_lf_removed", "add_bos", "bos_token", "bos_policy",
            "output_mask", "kv_policy")
    require(all(name in workload for name in keys), "incomplete native workload manifest")
    raw = workload.get("chunks")
    require(isinstance(raw, list), "native chunk metadata missing")
    chunk_rows = raw if isinstance(raw, list) else []
    chunks = [record(item) for item in chunk_rows]
    require(all(row.get("complete") is True for row in chunks), "incomplete native chunk")
    identity: Record = {name: workload[name] for name in keys}
    identity["chunks"] = [{key: row[key] for key in ("chunk_id", "token_offset", "input_tokens")}
                           for row in chunks]
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def collect_metric(args: argparse.Namespace, recipe: str) -> tuple[Path, Record]:
    require(args.model is not None and args.dataset is not None and args.split is not None,
            "collection requires --model, --dataset, and explicit --split")
    require(args.max_chunks >= 0 and all(value > 0 for value in
            (args.threads, args.threads_batch, args.batch_size, args.ubatch_size)), "invalid workload dimensions")
    binary, model, dataset = (Path(value).resolve(strict=True) for value in (args.runner, args.model, args.dataset))
    clean_environment()
    info = compiled_info(binary)
    validate_recipe(info, recipe)
    before = artifact_snapshot(binary)
    identities: Record = {"model_sha256": sha256(model), "dataset_sha256": sha256(dataset),
                           "split": args.split, "build_info": info, "artifacts": before}
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    activation = recipe == "activation"
    filename = "activation-quant-metrics.jsonl" if activation else "residual-path-metrics.jsonl"
    raw = output / filename
    command = [str(binary), "--model", str(model), "--file", str(dataset),
               "--output-dir", str(output / "native"), "--workload", "METRIC_PREFILL_256",
               "--max-chunks", str(args.max_chunks), "--threads", str(args.threads),
               "--threads-batch", str(args.threads_batch), "--batch-size", str(args.batch_size),
               "--ubatch-size", str(args.ubatch_size), "--run-id", output.name,
               "--activation-output" if activation else "--residual-output", str(raw)]
    write_json(output / "request.json", {**identities, "recipe": recipe, "command": list(command)})
    run(command, output, args.timeout)
    require(before == artifact_snapshot(binary) and identities["model_sha256"] == sha256(model)
            and identities["dataset_sha256"] == sha256(dataset), "collection inputs changed during execution")
    forbidden = output / ("residual-path-metrics.jsonl" if activation else "activation-quant-metrics.jsonl")
    require(raw.is_file() and not forbidden.exists(), "dedicated metric sink isolation failed")
    workload_path = output / "native/workload.json"
    workload = read_json(workload_path)
    require(workload.get("workload") == "METRIC_PREFILL_256", "metric output is not a native prefill workload")
    require(workload.get("complete") is True and workload.get("output_mask") == "second_half",
            "metric runner did not preserve complete native perplexity output mask")
    identities.update({"workload": workload, "native_workload_sha256": sha256(workload_path),
                       "native_workload_identity": workload_identity(workload),
                       "coverage": "FULL_COMPLETE_CHUNKS" if args.max_chunks == 0 else "BOUNDED_CHUNK_SUBSET",
                       "requested_max_chunks": args.max_chunks})
    if args.workload_manifest is not None:
        expected = read_json(args.workload_manifest)
        require(all(expected.get(key) == identities[key] for key in
                    ("model_sha256", "dataset_sha256", "split", "native_workload_identity")),
                "native ACT/RES workload manifest mismatch")
    write_json(output / "workload-binding.json", identities)
    return raw, identities


def main(recipe: str) -> int:
    parser = parser_for(recipe)
    args = parser.parse_args()
    try:
        output = args.output.resolve()
        binding: Record
        if args.reduce is None:
            raw, binding = collect_metric(args, recipe)
        else:
            raw = args.reduce.resolve(strict=True)
            output.mkdir(parents=True, exist_ok=False)
            binding = {"coverage": "RAW_REDUCTION_ONLY", "source": str(raw)}
        summary = (activation_summary(raw) if recipe == "activation" else
                   residual_summary(raw, args.accept_proposed_weighting))
        summary["provenance"] = binding
        summary_path = output / ("activation-quant-summary.json" if recipe == "activation" else
                                  "residual-path-summary.json")
        write_json(summary_path, summary)
        os.link(summary_path, output / "summary.json")
        if args.reduce is None:
            os.link(raw, output / ("counts.jsonl" if recipe == "activation" else "shapes.jsonl"))
        print(str(output))
        return 0
    except (EvaluationError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"evaluation failed: {error}", file=sys.stderr)
        return 1
