from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Final

from eval_common import (
    EvaluationError,
    Json,
    Record,
    artifact_snapshot,
    clean_environment,
    compiled_info,
    decode,
    integer,
    read_json,
    record,
    require,
    run,
    sha256,
    validate_recipe,
    write_json,
)
from metric_reducers import activation_summary, residual_summary

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation import activation, residual, weight_alignment
from evaluation.manifest import Manifest
from evaluation.reducer import metric_rows


def parser_for(recipe: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Independent {recipe} metric collection; native WikiText-2 prefills.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--runner", type=Path, help="built llama-eval-workload executable")
    source.add_argument("--reduce", type=Path, help="reduce an existing complete dedicated JSONL; no new collection claim")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--workload-manifest", type=Path, help="prior workload-binding.json; require same native chunk identities")
    parser.add_argument("--evaluation-manifest", type=Path,
                        help="validated evaluation_manifest.json; required for new collection and scale reduction")
    parser.add_argument("--output", type=Path, required=True, help="fresh output directory; never overwritten")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--threads-batch", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ubatch-size", type=int, default=256)
    parser.add_argument("--max-chunks", type=int, default=0, help="0 means all complete native 256-token chunks")
    parser.add_argument("--timeout", type=int, default=600, help="finite collection timeout in seconds")
    if recipe == "scale":
        parser.add_argument("--gzip", action="store_true", help="stream exact SCU observations to gzip without a raw disk copy")
        parser.add_argument("--scale-mode", choices=("detailed", "aggregate"), default="detailed",
                            help="detailed: one record per SCU coordinate; aggregate: producer-validated integer "
                                 "sums per chunk/layer/work type (same summary)")
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
    validate_metric_recipe(info, recipe)
    require(args.evaluation_manifest is not None, "collection requires --evaluation-manifest")
    manifest_path = Path(args.evaluation_manifest).resolve(strict=True)
    manifest = Manifest.load(manifest_path)
    bits = integer(info, "activation_bits")
    require(manifest.precision == f"A{bits}W{bits}" and manifest.dim == integer(info, "dim"),
            "evaluation manifest differs from compiled metric profile")
    require(manifest.chunk_policy == f"METRIC_PREFILL_256:split={args.split}:max_chunks={args.max_chunks}:"
            "context=256:tail=drop:output=second_half", "evaluation manifest differs from requested chunk policy")
    before = artifact_snapshot(binary)
    identities: Record = {"model_sha256": sha256(model), "dataset_sha256": sha256(dataset),
                           "split": args.split, "build_info": info, "artifacts": before,
                           "manifest_sha256": manifest.sha256}
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "evaluation_manifest.json").open("xb") as target:
        target.write(manifest_path.read_bytes())
    require(sha256(output / "evaluation_manifest.json") == manifest.sha256,
            "evaluation manifest changed before collection")
    aggregate = recipe == "scale" and args.scale_mode == "aggregate"
    require(not (aggregate and args.gzip), "aggregate SCU output is small; --gzip is for detailed observations")
    filename = {"activation": "activation-quant-metrics.jsonl", "residual": "residual-path-metrics.jsonl",
                "scale": "scale-alignment-aggregate.jsonl" if aggregate else "scale-alignment-metrics.jsonl"}[recipe]
    compressed = recipe == "scale" and args.gzip
    if compressed:
        filename += ".gz"
    raw = output / filename
    command = [str(binary), "--model", str(model), "--file", str(dataset),
               "--output-dir", str(output / "native"), "--workload", "METRIC_PREFILL_256",
               "--max-chunks", str(args.max_chunks), "--threads", str(args.threads),
               "--threads-batch", str(args.threads_batch), "--batch-size", str(args.batch_size),
               "--ubatch-size", str(args.ubatch_size), "--run-id", output.name,
               "--manifest-sha256", manifest.sha256, "--seed", str(manifest.seed)]
    if not compressed:
        command.extend(["--" + recipe + "-output", str(raw)])
    if aggregate:
        command.extend(["--scale-mode", "aggregate"])
    write_json(output / "request.json", {**identities, "recipe": recipe, "command": list(command)})
    if compressed:
        from campaign_stream import run_compressed
        run_compressed(command, output, raw, args.timeout)
    else:
        run(command, output, args.timeout)
    require(before == artifact_snapshot(binary) and identities["model_sha256"] == sha256(model)
            and identities["dataset_sha256"] == sha256(dataset), "collection inputs changed during execution")
    sinks = {"activation-quant-metrics.jsonl", "residual-path-metrics.jsonl", "scale-alignment-metrics.jsonl",
             "scale-alignment-aggregate.jsonl"}
    require(raw.is_file() and not any((output / other).exists() for other in sinks - {filename}),
            "dedicated metric sink isolation failed")
    require(sha256(manifest_path) == manifest.sha256, "evaluation manifest changed during collection")
    workload_path = output / "native/workload.json"
    workload = read_json(workload_path)
    require(workload.get("workload") == "METRIC_PREFILL_256", "metric output is not a native prefill workload")
    require(workload.get("complete") is True and workload.get("output_mask") == "second_half",
            "metric runner did not preserve complete native perplexity output mask")
    require(integer(workload, "seed") == manifest.seed and integer(workload, "context_tokens") == 256,
            "native workload differs from manifest seed/chunk policy")
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


# Combined collection (`metric all`): metric kind -> (runner sink option, raw file name of one shard).
COMBINED_SINKS: Final = {"activation": ("--activation-output", "activation-quant-metrics.jsonl"),
                         "residual": ("--residual-output", "residual-path-metrics.jsonl"),
                         "scu": ("--scale-output", "scale-alignment-aggregate.jsonl")}
# The native workload fields a shard owns; every other field is identical in all shards of one collection.
SHARD_FIELDS: Final = ("first_chunk", "selected_chunks", "chunks")
STREAM_SCHEMAS: Final = {"activation": "im2p-activation-quant-metrics", "residual": "im2p-residual-path-metrics"}


def chunk_policy(max_chunks: int) -> str:
    return f"METRIC_PREFILL_256:split=test:max_chunks={max_chunks}:context=256:tail=drop:output=second_half"


def validate_combined(info: Record) -> None:
    """A combined build: all three sinks compiled in, and each recipe's own route requirements."""
    sinks = {"activation_metrics": "activation", "residual_metrics": "residual", "scale_metrics": "scale"}
    require(all(integer(info, key) == 1 for key in sinks), "combined collection requires all three metric sinks")
    for key, recipe in sinks.items():
        validate_metric_recipe({**info, **dict.fromkeys(sinks, 0), key: 1}, recipe)


def native_plan(binary: Path, model: Path, dataset: Path, max_chunks: int) -> Record:
    """The runner's own chunk population (its tokenizer and chunking), without running the workload."""
    done = subprocess.run([str(binary), "--model", str(model), "--file", str(dataset), "--plan-only",
                           "--max-chunks", str(max_chunks)], capture_output=True, text=True, timeout=600, check=False)
    require(done.returncode == 0 and bool(done.stdout.strip()),
            "native chunk plan failed: " + (done.stderr.strip().splitlines() or ["no output"])[-1])
    plan = decode(done.stdout.strip().splitlines()[-1])
    require(plan.get("schema") == "potal-native-chunk-plan" and integer(plan, "first_chunk") == 0,
            "unexpected native chunk plan")
    return plan


def shard_plan(selected: int, workers: int) -> list[tuple[int, int]]:
    """Contiguous (first chunk, count) shards covering chunks 0..selected-1 once; sizes differ by at most one."""
    require(selected > 0 and workers > 0, "invalid shard plan")
    count, extra = divmod(selected, min(workers, selected))
    shards: list[tuple[int, int]] = []
    for index in range(min(workers, selected)):
        shards.append((sum(size for _, size in shards), count + int(index < extra)))
    return shards


def run_shards(jobs: list[tuple[list[str], Path]], timeout: int) -> list[Record]:
    """Start every shard process at once, each in its own directory with its own log; wait for all and stop all at the
    first failure or the shared deadline. Per shard: exit code, wall seconds and peak RSS (from wait4)."""
    started = time.monotonic()
    results: list[Record] = [{"exit_code": None} for _ in jobs]
    running: dict[int, tuple[int, subprocess.Popen[bytes]]] = {}
    logs = [(directory / "process.log").open("x", encoding="utf-8") for _, directory in jobs]
    failure = ""
    try:
        for index, ((argv, directory), log) in enumerate(zip(jobs, logs)):
            process = subprocess.Popen(argv, cwd=directory, stdout=log, stderr=subprocess.STDOUT)
            running[process.pid] = (index, process)
        while running and not failure:
            pid, status, usage = os.wait4(-1, os.WNOHANG)
            if pid == 0:
                failure = "timeout" if time.monotonic() - started > timeout else ""
                time.sleep(0.2)
                continue
            index, process = running.pop(pid)
            process.returncode = os.waitstatus_to_exitcode(status)
            results[index] = {"exit_code": process.returncode, "wall_seconds": round(time.monotonic() - started, 3),
                              "peak_rss_bytes": usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)}
            failure = f"shard {index} exited {process.returncode}" if process.returncode else ""
    finally:
        for _, process in running.values():
            process.kill()
            process.wait()
        for log in logs:
            log.close()
        for (argv, directory), result in zip(jobs, results):
            write_json(directory / "command.json", {"argv": list(argv), "cwd": str(directory), **result,
                                                    "timeout_seconds": timeout, "status": failure or "PASS"})
    require(not failure, f"native combined collection failed ({failure}); see shard process.log")
    return results


def reusable_shard(previous: Path, argv: list[str], directory: Path, manifest: Manifest) -> bool:
    """A shard of an earlier, unfinished attempt counts only if it ran exactly this command (its own directory aside),
    exited 0, and its three streams pass the reducers' own stream validation."""
    done = read_json(previous / "command.json") if (previous / "command.json").is_file() else {}
    old = done.get("argv")
    if done.get("exit_code") != 0 or not isinstance(old, list) or \
            [str(value).replace(str(previous), str(directory)) for value in old] != argv:
        return False
    try:
        for kind, (_, name) in COMBINED_SINKS.items():
            stream = (weight_alignment.rows(previous / name, manifest) if kind == "scu" else
                      metric_rows(previous / name, STREAM_SCHEMAS[kind], manifest))
            for _ in stream:
                pass
    except (EvaluationError, OSError, ValueError):
        return False
    return True


def link_shard(previous: Path, directory: Path) -> None:
    """Hard links of a reused shard: the earlier attempt stays untouched and nothing is copied."""
    directory.mkdir()
    for path in sorted(previous.rglob("*")):
        target = directory / path.relative_to(previous)
        if path.is_dir():
            target.mkdir()
        else:
            os.link(path, target)


def merge_workloads(workloads: list[Record]) -> Record:
    """One native workload over chunk-disjoint shards: every run field equal, the shards' chunks joined in order."""
    common = [{key: value for key, value in workload.items() if key not in SHARD_FIELDS} for workload in workloads]
    require(all(value == common[0] for value in common), "metric shards ran different workloads")
    chunks: list[Json] = []
    for workload in workloads:
        rows = workload.get("chunks")
        if not isinstance(rows, list):
            raise EvaluationError("native chunks missing")
        chunks.extend(rows)
    return {**workloads[0], "selected_chunks": len(chunks), "chunks": chunks}


def collect_combined(binary: Path, model: Path, dataset: Path, manifest_path: Path, output: Path, max_chunks: int,
                     workers: int, threads: int, threads_batch: int, timeout: int,
                     reuse: Path | None = None) -> tuple[dict[str, list[Path]], Record]:
    """Activation, residual and aggregate SCU from one forward: the runner's chunk population split into contiguous
    shards that run as parallel processes with every sink on, each validated like a single collection, then joined
    into one workload binding. `reuse` is the collection of an earlier unfinished attempt whose finished shards are
    linked instead of run again. Returns the shard streams of each metric kind and that binding."""
    require(max_chunks >= 0 and min(workers, threads, threads_batch, timeout) > 0, "invalid combined collection")
    binary, model, dataset = (path.resolve(strict=True) for path in (binary, model, dataset))
    clean_environment()
    info = compiled_info(binary)
    validate_combined(info)
    manifest = Manifest.load(manifest_path)
    bits = integer(info, "activation_bits")
    require(manifest.precision == f"A{bits}W{bits}" and manifest.dim == integer(info, "dim"),
            "evaluation manifest differs from compiled metric profile")
    require(manifest.chunk_policy == chunk_policy(max_chunks), "evaluation manifest differs from requested chunk policy")
    before = artifact_snapshot(binary)
    plan = native_plan(binary, model, dataset, max_chunks)
    shards = shard_plan(integer(plan, "selected_chunks", 1), workers)
    identities: Record = {"model_sha256": sha256(model), "dataset_sha256": sha256(dataset), "split": "test",
                          "build_info": info, "artifacts": before, "manifest_sha256": manifest.sha256,
                          "native_plan": plan}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "evaluation_manifest.json").open("xb") as target:
        target.write(manifest_path.read_bytes())
    require(sha256(output / "evaluation_manifest.json") == manifest.sha256, "evaluation manifest changed before collection")
    planned: list[tuple[list[str], Path]] = []
    jobs: list[tuple[list[str], Path]] = []
    reused: dict[Path, str] = {}
    requests: list[Json] = []
    for number, (first, count) in enumerate(shards):
        directory = output / f"shard-{number:03d}"
        argv = [str(binary), "--model", str(model), "--file", str(dataset), "--output-dir", str(directory / "native"),
                "--workload", "METRIC_PREFILL_256", "--first-chunk", str(first), "--max-chunks", str(count),
                "--threads", str(threads), "--threads-batch", str(threads_batch), "--batch-size", "256",
                "--ubatch-size", "256", "--run-id", directory.name, "--manifest-sha256", manifest.sha256,
                "--seed", str(manifest.seed), "--scale-mode", "aggregate"]
        for option, filename in COMBINED_SINKS.values():
            argv.extend([option, str(directory / filename)])
        planned.append((argv, directory))
        previous = reuse / directory.name if reuse is not None else None
        if previous is not None and previous.is_dir() and reusable_shard(previous, argv, directory, manifest):
            link_shard(previous, directory)
            reused[directory] = str(previous)
        else:
            directory.mkdir()
            jobs.append((argv, directory))
        requests.append({"first_chunk": first, "chunks": count, "command": [*argv],
                         "reused_from": reused.get(directory)})
    write_json(output / "request.json", {**identities, "recipe": "combined", "shards": requests})
    started = time.monotonic()
    fresh = dict(zip((directory for _, directory in jobs), run_shards(jobs, timeout)))
    native_seconds = round(time.monotonic() - started, 3)
    results: list[Record] = []
    for _, directory in planned:
        if directory in reused:
            done = read_json(directory / "command.json")
            results.append({key: done.get(key) for key in ("exit_code", "wall_seconds", "peak_rss_bytes")} |
                           {"reused_from": reused[directory]})
        else:
            results.append(fresh[directory])
    require(before == artifact_snapshot(binary) and identities["model_sha256"] == sha256(model) and
            identities["dataset_sha256"] == sha256(dataset) and sha256(manifest_path) == manifest.sha256,
            "collection inputs changed during execution")
    workloads: list[Record] = []
    for (first, count), (_, directory) in zip(shards, planned):
        require(sorted(path.name for path in directory.glob("*.jsonl*")) == sorted(name for _, name in
                COMBINED_SINKS.values()), "combined metric sinks missing or unexpected: " + directory.name)
        workload = read_json(directory / "native/workload.json")
        rows = workload.get("chunks")
        require(workload.get("workload") == "METRIC_PREFILL_256" and workload.get("complete") is True and
                workload.get("output_mask") == "second_half" and integer(workload, "seed") == manifest.seed and
                integer(workload, "context_tokens") == 256, "metric shard is not the complete native prefill workload")
        require((integer(workload, "first_chunk"), integer(workload, "selected_chunks")) == (first, count) and
                isinstance(rows, list) and [integer(record(row), "chunk_id") for row in rows] ==
                list(range(first, first + count)) and all(record(row).get("complete") is True for row in rows),
                "metric shard chunk coverage differs from its plan: " + directory.name)
        require((integer(workload, "threads", 1), integer(workload, "threads_batch", 1)) == (threads, threads_batch),
                "metric shard thread settings differ from the request")
        workloads.append(workload)
    merged = merge_workloads(workloads)
    rows = merged["chunks"]
    require(isinstance(rows, list) and [integer(record(row), "chunk_id") for row in rows] ==
            list(range(integer(plan, "selected_chunks", 1))), "combined chunk coverage differs from the native plan")
    write_json(output / "workload.json", merged)
    index: list[Json] = []
    for (first, count), (_, directory), result in zip(shards, planned, results):
        streams: Record = {kind: {"path": f"{directory.name}/{name}", "sha256": sha256(directory / name)}
                           for kind, (_, name) in COMBINED_SINKS.items()}
        index.append({"directory": directory.name, "first_chunk": first, "chunks": count, "streams": streams,
                      "native_workload_sha256": sha256(directory / "native/workload.json"), **result})
    write_json(output / "shards.json", {"manifest_sha256": manifest.sha256, "threads": threads,
                                        "threads_batch": threads_batch, "shards": index})
    identities.update({"workload": merged, "native_workload_sha256": sha256(output / "workload.json"),
                       "native_workload_identity": workload_identity(merged),
                       "coverage": "FULL_COMPLETE_CHUNKS" if max_chunks == 0 else "BOUNDED_CHUNK_SUBSET",
                       "requested_max_chunks": max_chunks, "threads": threads, "threads_batch": threads_batch,
                       "workers": len(shards), "native_seconds": native_seconds,
                       "reused_shards": {directory.name: source for directory, source in reused.items()}})
    write_json(output / "workload-binding.json", identities)
    return {kind: [directory / name for _, directory in planned] for kind, (_, name) in COMBINED_SINKS.items()}, identities


def validate_metric_recipe(info: Record, recipe: str) -> None:
    if recipe != "scale":
        validate_recipe(info, recipe)
        require(info.get("scale_metrics", 0) == 0, "unexpected scale metric sink enabled")
        return
    require(integer(info, "scale_metrics") == 1 and integer(info, "activation_metrics") == 0 and
            integer(info, "residual_metrics") == 0, "scale recipe/build metric flag mismatch")
    require(integer(info, "cycle_sim") == 1 and info.get("backend") == "IM2P_SIM" and
            info.get("hp1") is True and integer(info, "gemmini") == 1 and
            info.get("gemmini_option") == "WS", "scale collection requires functional IM2P_SIM HP1 WS route")
    require(integer(info, "activation_bits") == integer(info, "weight_bits") and
            integer(info, "activation_bits") in (4, 8) and integer(info, "dim") in (16, 32, 64) and
            info.get("activation_mode") == "EXSIA" and integer(info, "block_size") == 32,
            "unsupported scale collector profile")


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
        if args.evaluation_manifest is not None:
            manifest = Manifest.load(args.evaluation_manifest)
            summary = {"activation": activation.reduce, "residual": residual.reduce,
                       "scale": weight_alignment.reduce}[recipe](raw, manifest)
        else:
            require(recipe != "scale", "scale reduction requires --evaluation-manifest")
            summary = (activation_summary(raw) if recipe == "activation" else
                       residual_summary(raw, args.accept_proposed_weighting))
        summary["provenance"] = binding
        filename = {"activation": "activation-quant-summary.json", "residual": "residual-path-summary.json",
                    "scale": "scale-alignment-summary.json"}[recipe]
        summary_path = output / filename
        write_json(summary_path, summary)
        os.link(summary_path, output / "summary.json")
        if args.evaluation_manifest is not None:
            metric_filename = {"activation": "activation_metrics.json", "residual": "residual_metrics.json",
                               "scale": "scale_alignment_metrics.json"}[recipe]
            os.link(summary_path, output / metric_filename)
        if args.reduce is None:
            os.link(raw, output / (("counts.jsonl" if recipe == "activation" else "shapes.jsonl") +
                                  (".gz" if raw.suffix == ".gz" else "")))
        print(str(output))
        return 0
    except (EvaluationError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"evaluation failed: {error}", file=sys.stderr)
        return 1
