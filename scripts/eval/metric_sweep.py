#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 scripts/eval/run_measurement.py metric all --dry-run
"""metric all: activation, residual and SCU over models x precisions x DIMs, aggregated (orchestration only).

  run_measurement.py metric all [--models gpt2,llama3.2-1B] [--precisions a4w4,a8w8] [--dims 16,32,64]
                                [--max-chunks 0] [--seed 1234] [--workers N] [--threads N] [--output DIR]
  run_measurement.py metric all --resume SWEEP_DIR     continue with the recorded matrix and options
  run_measurement.py metric all --dry-run              print the plan; nothing is built or run

Combined collection (default): per configuration one `campaign.py metrics-all` process; one build per precision x
DIM with all three metric sinks serves both models, the runner's chunk population is split into --workers contiguous
shards that run in parallel, and the unchanged reducers reduce the shard streams as one stream. By default the
combined collection runs the terminal lm_head metrics-only (--terminal-lm-head): observed completely, its logits never
computed; --terminal-lm-head full computes them as before. Separate collection
(--collection separate): one unchanged `campaign.py KIND` run per metric kind, the activation run being the workload
anchor of residual and SCU. Either way the sweep only reads the finished runs: it checks their shared identity (and
the chunk population across DIMs) and writes metric-summary.json, metric-summary.csv and the stdout tables. It
defines no fourth metric and changes none.
"""
from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from campaign_build import COMBINED, METRIC_SINKS, REPO, cached_llama_build, llama_plan
from campaign_verify import verify
from eval_common import (
    EvaluationError,
    Json,
    Record,
    clean_environment,
    integer,
    read_json,
    record,
    require,
    sha256,
    text,
    write_json,
)
from measurement_identity import compare, run_identity
from metric_table import metric_table, render

EVAL: Final = Path(__file__).resolve().parent
MODELS: Final = ("gpt2", "llama3.2-1B")
PRECISIONS: Final = ("a4w4", "a8w8")
DIMS: Final = ("16", "32", "64")
KINDS: Final = tuple(METRIC_SINKS)  # activation, residual, scu: activation anchors the workload of the other two
REQUEST: Final = "potal-metric-sweep-request"
# Summary fields copied from each run's summary.json; the first ones are the paper columns (CSV, stdout).
ACTIVATION: Final = ("recall", "jaccard", "requant_ratio", "residual_fraction", "fp_fraction", "fp_selected",
                     "potal_selected", "intersection", "union", "valid_positions", "residual_nnz",
                     "eligible_logical_blocks", "unique_actual_requantized_blocks", "p3_requantization_events")
SCU: Final = ("avg_delta_w", "max_delta_w", "scu_update_fraction", "delta_w_sum", "alignment_count",
              "updated_partial_sum_count", "total_partial_sum_count")
SCU_POPULATIONS: Final = ("dense", "residual", "overall")
RESIDUAL: Final = ("retained_row_factor", "retained_k_factor", "logical_ratio", "padded_ratio",
                   "logical_main_macs", "logical_residual_macs", "physical_main_macs", "physical_residual_macs",
                   "zero_limb_pruned_count", "original_radix_rows", "retained_rows", "original_k", "compact_k")
CSV_COLUMNS: Final = ("model", "precision", "dim", *(f"activation_{name}" for name in ACTIVATION[:4]),
                      *(f"scu_{population}_{name.removeprefix('scu_')}" for population in SCU_POPULATIONS
                        for name in SCU[:3]),
                      *(f"residual_{name}" for name in RESIDUAL[:4]))
# The only shared-identity field that may differ between the DIMs of one model and precision.
DIM_DEPENDENT: Final = ("dim",)
Config = tuple[str, str, int]  # (model, precision, DIM)
Build = tuple[str, str, int]  # (metric kind, precision, DIM)


@dataclass(frozen=True, slots=True)
class Sweep:
    """The matrix and the result-defining options of one sweep, recorded in its manifest.json."""
    models: tuple[str, ...]
    precisions: tuple[str, ...]
    dims: tuple[int, ...]
    max_chunks: int
    seed: int
    dataset_manifest: str | None
    im2p: str
    build_cache: str
    scu_mode: str = "aggregate"
    collection: str = "combined"  # combined: one forward per configuration; separate: one per metric kind
    threads: int = 1
    threads_batch: int = 1
    terminal_lm_head: str = "full"  # combined: "metrics-only" observes the lm_head but never computes the logits

    def configurations(self) -> list[Config]:
        return [(model, precision, dim) for model in self.models for precision in self.precisions for dim in self.dims]

    def collections(self) -> tuple[str, ...]:
        """The native collections of one configuration: one combined forward, or one per metric kind."""
        return (COMBINED,) if self.collection == "combined" else KINDS

    def builds(self) -> list[Build]:
        return [(kind, precision, dim) for kind in self.collections() for precision in self.precisions
                for dim in self.dims]

    def record(self) -> Record:
        return {"schema": REQUEST, "version": 1, "models": list(self.models), "precisions": list(self.precisions),
                "dims": list(self.dims), "max_chunks": self.max_chunks, "seed": self.seed,
                "dataset_manifest": self.dataset_manifest, "im2p": self.im2p, "build_cache": self.build_cache,
                "scu_mode": self.scu_mode, "collection": self.collection, "threads": self.threads,
                "threads_batch": self.threads_batch, "terminal_lm_head": self.terminal_lm_head}


def subset(given: str | None, allowed: tuple[str, ...], option: str) -> tuple[str, ...]:
    """A comma list in the canonical matrix order; empty or unknown values are refused."""
    if given is None:
        return allowed
    names = [name.strip() for name in given.split(",") if name.strip()]
    unknown = [name for name in names if name not in allowed]
    require(bool(names) and not unknown, f"{option} takes a comma list of {','.join(allowed)}; got {given!r}")
    return tuple(name for name in allowed if name in names)


def recorded(root: Path) -> Sweep:
    value = read_json(root / "manifest.json")
    require(value.get("schema") == REQUEST and value.get("version") == 1, "not a metric sweep directory: " + str(root))
    lists: dict[str, str] = {}
    for key in ("models", "precisions", "dims"):
        items = value.get(key)
        if not isinstance(items, list):
            raise EvaluationError("recorded sweep lacks " + key)
        lists[key] = ",".join(str(item) for item in items)
    dataset = value.get("dataset_manifest")
    return Sweep(subset(lists["models"], MODELS, "models"), subset(lists["precisions"], PRECISIONS, "precisions"),
                 tuple(int(dim) for dim in subset(lists["dims"], DIMS, "dims")), integer(value, "max_chunks"),
                 integer(value, "seed"), dataset if isinstance(dataset, str) else None, text(value, "im2p"),
                 text(value, "build_cache"),
                 # Sweeps recorded before SCU modes, combined collection and terminal modes: detailed, separate, one
                 # thread, the lm_head computed in full.
                 str(value.get("scu_mode", "detailed")), str(value.get("collection", "separate")),
                 integer(value, "threads", 1) if "threads" in value else 1,
                 integer(value, "threads_batch", 1) if "threads_batch" in value else 1,
                 str(value.get("terminal_lm_head", "full")))


def resolve(args: argparse.Namespace) -> tuple[Sweep, Path | None]:
    """(sweep, its directory); --resume takes everything that defines results from the recorded sweep."""
    if args.resume is not None:
        given = [option for option in ("models", "precisions", "dims", "max_chunks", "seed", "dataset_manifest",
                                        "im2p", "build_cache", "scu_mode", "collection", "threads", "threads_batch",
                                        "terminal_lm_head")
                 if getattr(args, option) is not None]
        require(not given, "--resume uses the recorded matrix and options; remove --" +
                ", --".join(option.replace("_", "-") for option in given))
        root = args.resume.resolve(strict=True)
        return recorded(root), root
    sweep = Sweep(subset(args.models, MODELS, "--models"), subset(args.precisions, PRECISIONS, "--precisions"),
                  tuple(int(dim) for dim in subset(args.dims, DIMS, "--dims")),
                  0 if args.max_chunks is None else args.max_chunks, 1234 if args.seed is None else args.seed,
                  str(args.dataset_manifest.resolve(strict=True)) if args.dataset_manifest is not None else None,
                  str((args.im2p or REPO.parent / "IM2P.sim").resolve(strict=True)),
                  str((args.build_cache or Path("runs/.build-cache")).resolve()), args.scu_mode or "aggregate",
                  args.collection or "combined", args.threads or 1, args.threads_batch or args.threads or 1,
                  args.terminal_lm_head or ("full" if args.collection == "separate" else "metrics-only"))
    require(sweep.max_chunks >= 0 and 0 <= sweep.seed < 4294967295, "invalid --max-chunks or --seed")
    require(sweep.threads > 0 and sweep.threads_batch > 0, "invalid --threads or --threads-batch")
    require(sweep.collection == "separate" or sweep.scu_mode == "aggregate",
            "the combined collection collects aggregate SCU; detailed SCU needs --collection separate")
    require(sweep.collection == "combined" or (sweep.threads, sweep.threads_batch) == (1, 1),
            "--threads and --threads-batch apply to the combined collection")
    require(sweep.collection == "combined" or sweep.terminal_lm_head == "full",
            "--terminal-lm-head metrics-only applies to the combined collection")
    # One detailed SCU chunk is ~25M records / ~330 MB (GPT-2 A8W8 DIM 32); the full corpus is ~1118 chunks.
    require(sweep.scu_mode == "aggregate" or sweep.max_chunks != 0 or args.allow_large_raw_scu,
            "Refusing full-corpus detailed SCU collection. Use --scu-mode aggregate or explicitly acknowledge "
            "large raw output with --allow-large-raw-scu.")
    return sweep, args.output.resolve() if args.output is not None else None


def progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def build_name(kind: str, precision: str, dim: int) -> str:
    return f"{kind}-{precision}-d{dim}"


def campaign_argv(sweep: Sweep, config: Config, kind: str, build: str, anchor: str | None, timeout: int,
                  workers: int = 1) -> list[str]:
    """The campaign.py command line of one collection (a metric kind, or the combined one), without its --output."""
    model, precision, dim = config
    argv = [kind, "--model", model, "--precision", precision, "--dim", str(dim), "--max-chunks", str(sweep.max_chunks),
            "--seed", str(sweep.seed), "--timeout", str(timeout), "--im2p", sweep.im2p, "--prepared-build", build]
    if kind in ("scu", COMBINED):
        argv += ["--scu-mode", sweep.scu_mode]
    if kind == COMBINED:
        argv += ["--workers", str(workers), "--threads", str(sweep.threads), "--threads-batch", str(sweep.threads_batch),
                 "--terminal-lm-head", sweep.terminal_lm_head]
    if sweep.dataset_manifest is not None:
        argv += ["--dataset-manifest", sweep.dataset_manifest]
    if anchor is not None:  # the activation run's exact manifest bytes and native chunk identity
        argv += ["--evaluation-manifest", f"{anchor}/evaluation_manifest.json",
                 "--workload-manifest", f"{anchor}/collection/workload-binding.json"]
    return argv


def dry_run(sweep: Sweep, root: Path | None, timeout: int, workers: int = 1) -> str:
    base = str(root) if root is not None else "OUTPUT"
    configurations = sweep.configurations()
    collections = len(configurations) * len(sweep.collections())
    lines = ["metric all: dry run (nothing is built or run)", f"{len(configurations)} configurations",
             f"{len(configurations) * len(KINDS)} metric runs" + ("" if sweep.collection == "separate" else
                                                                  f" from {collections} combined collections"),
             f"{len(sweep.builds())} build configurations",
             f"collection = {sweep.collection}" + ("" if sweep.collection == "separate" else
                f" (threads = {sweep.threads}, threads_batch = {sweep.threads_batch}, workers = {workers}, "
                f"terminal lm_head = {sweep.terminal_lm_head})"),
             "models = " + ", ".join(sweep.models), "precisions = " + ", ".join(sweep.precisions),
             "DIMs = " + ", ".join(map(str, sweep.dims)), f"max_chunks = {sweep.max_chunks}" +
             (" (all complete chunks)" if sweep.max_chunks == 0 else ""), f"seed = {sweep.seed}",
             f"SCU mode = {sweep.scu_mode}",
             "", f"builds (campaign_build.cached_llama_build, cache {sweep.build_cache}):"]
    for kind, precision, dim in sweep.builds():
        semantic = llama_plan(kind, precision, dim, Path(sweep.im2p)).semantic_options_sha256
        lines.append(f"  {build_name(kind, precision, dim)}  semantic_options_sha256={semantic}")
    lines += ["", "metric runs (python3 -B scripts/eval/campaign.py ...):"]
    for config in configurations:
        model, precision, dim = config
        cell = f"{base}/{model}/{precision}/d{dim}"
        for kind in sweep.collections():
            argv = campaign_argv(sweep, config, kind, f"{base}/builds/{build_name(kind, precision, dim)}",
                                 None if kind in ("activation", COMBINED) else f"{cell}/activation", timeout, workers)
            lines.append("  " + shlex.join([*argv, "--output", f"{cell}/{kind}"]))
    return "\n".join(lines)


def failure(model: str | None, precision: str, dim: int, stage: str, error: object) -> Record:
    return {"model": model, "precision": precision, "dim": dim, "stage": stage, "error": str(error)}


def prepare_builds(root: Path, sweep: Sweep, jobs: int, failures: list[Record], keep_going: bool,
                   timing: Record) -> dict[Build, Path]:
    """One verified build per collection kind x precision x DIM from the build authority's cache, linked from
    builds/; `timing` receives each build's seconds and cache hit.

    Both models use the same build. A resumed sweep keeps the builds it linked, the exact verified runners its
    earlier configurations used, even after the sources moved on (a commit or merge): nothing is rebuilt or mixed."""
    prepared: dict[Build, Path] = {}
    (root / "builds").mkdir(exist_ok=True)
    for build in sweep.builds():
        kind, precision, dim = build
        link = root / "builds" / build_name(kind, precision, dim)
        try:
            started = time.monotonic()
            if link.is_symlink():
                path, state = link.resolve(strict=True), "recorded"
                receipt = read_json(path / "build-receipt.json")
                require(receipt.get("kind") == kind and receipt.get("verification") == "PASS" and
                        receipt.get("runner_sha256") == sha256(path / "bin/llama-eval-workload"),
                        f"{link.name} no longer holds the verified build this sweep started with")
            else:
                path, hit = cached_llama_build(Path(sweep.build_cache), kind, precision, dim, Path(sweep.im2p), jobs)
                link.symlink_to(path.resolve(), target_is_directory=True)
                state = "cache hit" if hit else "built"
            timing[link.name] = {"seconds": round(time.monotonic() - started, 3), "cache_hit": state != "built"}
            prepared[build] = link
            progress(f"build {link.name}: {state} {path.resolve()}")
        except (EvaluationError, OSError, ValueError, subprocess.SubprocessError) as error:
            failures.append(failure(None, precision, dim, "build " + kind, error))
            progress(f"build {link.name}: FAILED {error}")
            if not keep_going:
                break
    return prepared


def log_of(run: Path) -> Path:
    return run.with_name(run.name + ".campaign.log")


def campaign(argv: list[str], output: Path) -> None:
    """One unchanged campaign.py process; its stdout and stderr are kept next to (not inside) its run directory."""
    done = subprocess.run([sys.executable, "-B", str(EVAL / "campaign.py"), *argv, "--output", str(output)],
                          capture_output=True, text=True, check=False)
    with log_of(output).open("x", encoding="utf-8") as log:
        log.write(f"argv: {shlex.join(argv)}\nexit: {done.returncode}\n--- stdout\n{done.stdout}--- stderr\n{done.stderr}")
    reason = (done.stderr.strip().splitlines() or ["no output"])[-1]
    require(done.returncode == 0 and done.stdout.strip() == str(output), f"campaign.py {argv[0]} failed: {reason}")


def metric_run(cell: Path, kind: str, argv: list[str], label: str) -> Path:
    """The run directory of one metric. A complete earlier run is verified (SHA256SUMS, manifest binding) and reused;
    otherwise campaign.py writes a new directory and any incomplete attempt stays untouched next to it."""
    attempts: list[Path] = []
    while True:  # KIND, KIND.retry-1, ...: every name a run directory or its campaign log already took
        output = cell / (f"{kind}.retry-{len(attempts)}" if attempts else kind)
        if not (output.exists() or log_of(output).exists()):
            break
        attempts.append(output)
    for path in attempts:
        if (path / "SHA256SUMS").is_file():
            verify(path)
            progress(f"{label}: reused {path}")
            return path
    unfinished = attempts[-1] / "collection" if attempts else None
    if argv[0] == COMBINED and unfinished is not None and unfinished.is_dir():
        argv = [*argv, "--reuse-shards", str(unfinished)]  # its finished shards are linked, not run again
    progress(f"{label}: running {output}")
    campaign(argv, output)
    progress(f"{label}: PASS")
    return output


def workload(run: Path, identity: Record) -> Record:
    """Shared identity of one finished run: measurement_identity's model/configuration part plus its workload."""
    evaluation = read_json(run / "evaluation_manifest.json")
    binding = read_json(run / "collection/workload-binding.json")
    chunks = record(binding.get("workload")).get("chunks")
    if not isinstance(chunks, list):
        raise EvaluationError("native chunks missing: " + str(run))
    chunk_ids: list[Json] = [*sorted(integer(record(chunk), "chunk_id") for chunk in chunks)]
    return {**record(identity["shared"]), "model": text(evaluation, "model"), "dataset": text(evaluation, "dataset"),
            "tokenizer_sha256": text(evaluation, "tokenizer_sha256"), "seed": integer(evaluation, "seed"),
            "chunk_policy": text(evaluation, "chunk_policy"), "max_chunks": integer(binding, "requested_max_chunks"),
            "native_workload_identity": text(binding, "native_workload_identity"), "chunk_ids": chunk_ids}


def fields(summary: Record, names: tuple[str, ...], what: str) -> Record:
    missing = [name for name in names if name not in summary]
    require(not missing, f"{what} lacks {', '.join(missing)}")
    return {name: summary[name] for name in names}


def configuration(root: Path, sweep: Sweep, config: Config, runs: dict[str, Path], builds: dict[Build, Path]) -> Record:
    """One aggregate row from the three finished runs of a configuration, after checking that they share one model,
    configuration and native workload (their builds differ by design and are only recorded)."""
    model, precision, dim = config
    combined = sweep.collection == "combined"
    identities = {kind: run_identity(run) for kind, run in runs.items()}
    require(all(identities[kind]["measurement"] == (COMBINED if combined else kind) for kind in KINDS),
            "run directory of another collection kind")
    comparison = compare(list(identities.values()))
    require(comparison["same_model_and_configuration"] is True,
            "metric runs differ in shared identity: " + ", ".join(record(comparison["differing"])))
    shared = {kind: workload(runs[kind], identities[kind]) for kind in KINDS}
    anchor = shared["activation"]
    differing = sorted({key for row in shared.values() for key in anchor if row.get(key) != anchor[key]})
    require(not differing, "metric runs of one configuration differ in: " + ", ".join(differing))
    expected: Record = {"precision": precision, "dim": dim, "seed": sweep.seed, "max_chunks": sweep.max_chunks}
    require(all(anchor.get(key) == value for key, value in expected.items()), "runs do not match configuration " +
            f"{model} {precision} d{dim} (seed {sweep.seed}, max_chunks {sweep.max_chunks})")
    builds_used: Record = {}
    for kind in KINDS:
        built = COMBINED if combined else kind
        prepared = builds[(built, precision, dim)]
        runner = Path(text(read_json(runs[kind] / "manifest.json"), "runner"))
        require(runner.parent.parent == prepared.resolve(), f"{kind} run did not use the sweep's build {prepared.name}")
        build = record(record(identities[kind]["builds"])[built])
        builds_used[kind] = {"name": prepared.name, "path": str(prepared.resolve()),
                             "build_receipt_sha256": build["build_receipt_sha256"],
                             "runner_sha256": build["binary_sha256"],
                             "semantic_options_sha256": build["semantic_options_sha256"]}
    results = {kind: runs[kind] / kind if combined else runs[kind] for kind in KINDS}  # each metric's summary.json
    summaries = {kind: read_json(results[kind] / "summary.json") for kind in KINDS}
    require(all(isinstance(summaries["scu"].get(population), dict) for population in SCU_POPULATIONS),
            "SCU summary has no dense/residual/overall split (reduced before the split; run scu again): " +
            str(runs["scu"]))
    binding = read_json(runs["scu"] / "manifest.json")
    mode = binding.get("scu_collection_mode", "detailed")
    require(mode == sweep.scu_mode, f"SCU run was collected in {mode} mode, this sweep uses {sweep.scu_mode}")
    require(binding.get("scu_reducer_sha256") == sha256(REPO / "evaluation/weight_alignment/__init__.py"),
            "SCU run was reduced by another SCU reducer source")
    scu: Record = {"collection_mode": mode, **{population: fields(record(summaries["scu"][population]), SCU,
                                                                  "SCU " + population) for population in SCU_POPULATIONS}}
    collection: Record = {"mode": sweep.collection}
    if combined:  # what a reused combined run must share with this sweep besides the build
        sources = binding.get("collector_sha256")
        require(isinstance(sources, dict) and all(sha256(REPO / name) == value for name, value in sources.items()) and
                binding.get("reducer_sha256") == {kind: sha256(REPO / "evaluation" / module / "__init__.py") for
                kind, module in (("activation", "activation"), ("residual", "residual"), ("scu", "weight_alignment"))},
                "combined run was collected or reduced by other collector/reducer sources")
        require((binding.get("threads"), binding.get("threads_batch")) == (sweep.threads, sweep.threads_batch),
                "combined run used other thread settings")
        # Runs collected before terminal modes computed the lm_head in full.
        execution = binding.get("metric_execution") or {"terminal_lm_head": "full"}
        require(isinstance(execution, dict) and execution.get("terminal_lm_head") == sweep.terminal_lm_head,
                f"combined run ran the terminal lm_head in another mode than {sweep.terminal_lm_head}")
        shards = read_json(runs["scu"] / "collection/shards.json").get("shards")
        collection.update({"threads": sweep.threads, "threads_batch": sweep.threads_batch,
                           "workers": binding.get("workers"), "metric_execution": execution,
                           "timing": binding.get("timing"), "shards": shards})
    return {"model": model, "precision": precision, "dim": dim, "shared_identity": anchor, "builds": builds_used,
            "collection": collection, "runs": {kind: str(results[kind].relative_to(root)) for kind in KINDS},
            "summary_sha256": {kind: sha256(results[kind] / "summary.json") for kind in KINDS},
            "activation": fields(summaries["activation"], ACTIVATION, "activation summary"), "scu": scu,
            "residual": fields(summaries["residual"], RESIDUAL, "residual summary")}


def same_workload_across_dims(previous: list[Record], row: Record) -> None:
    """DIM changes tiling, scale sharing, SCU alignment and padding only: one model and precision keep their chunks."""
    if not previous:
        return
    reference, current = record(previous[0]["shared_identity"]), record(row["shared_identity"])
    differing = [key for key in reference if key not in DIM_DEPENDENT and current.get(key) != reference[key]]
    require(not differing, f"DIM {row['dim']} workload differs from DIM {previous[0]['dim']} in: " + ", ".join(differing))


def execute(root: Path, sweep: Sweep, args: argparse.Namespace,
            timing: Record) -> tuple[list[Record], list[Record], list[Record]]:
    """(passing configurations, failures, workload identity per model and precision); fail-fast unless keep_going."""
    failures: list[Record] = []
    builds = prepare_builds(root, sweep, args.jobs, failures, args.keep_going, timing)
    passed: list[Record] = []
    groups: dict[tuple[str, str], list[Record]] = {}
    configurations = sweep.configurations()
    for index, config in enumerate(configurations):
        if failures and not args.keep_going:
            break
        model, precision, dim = config
        cell = root / model / precision / f"d{dim}"
        runs: dict[str, Path] = {}
        if sweep.collection == "combined":
            label = f"[{index + 1}/{len(configurations)}] {model} {precision} d{dim} {COMBINED}"
            build = builds.get((COMBINED, precision, dim))
            try:
                require(build is not None, "build failed")
                run = metric_run(cell, COMBINED, campaign_argv(sweep, config, COMBINED, str(build), None,
                                                               args.timeout, args.workers), label)
                runs = dict.fromkeys(KINDS, run)
            except (EvaluationError, OSError, ValueError, subprocess.SubprocessError) as error:
                failures.append(failure(*config, COMBINED, error))
                progress(f"{label}: FAILED {error}")
        for offset, kind in enumerate(KINDS if sweep.collection == "separate" else (), 1):
            step = f"[{index * len(KINDS) + offset}/{len(configurations) * len(KINDS)}]"
            label = f"{step} {model} {precision} d{dim} {kind}"
            build = builds.get((kind, precision, dim))
            if build is None or (kind != "activation" and "activation" not in runs):
                reason = "build failed" if build is None else "no activation workload anchor"
                failures.append(failure(*config, kind, reason))
                progress(f"{label}: SKIPPED ({reason})")
            else:
                anchor = str(runs["activation"]) if kind != "activation" else None
                try:
                    runs[kind] = metric_run(cell, kind, campaign_argv(sweep, config, kind, str(build), anchor,
                                                                      args.timeout), label)
                except (EvaluationError, OSError, ValueError, subprocess.SubprocessError) as error:
                    failures.append(failure(*config, kind, error))
                    progress(f"{label}: FAILED {error}")
            if failures and not args.keep_going:
                break
        if len(runs) != len(KINDS):
            continue
        try:
            row = configuration(root, sweep, config, runs, builds)
            same_workload_across_dims(groups.get((model, precision), []), row)
        except (EvaluationError, OSError, ValueError) as error:
            failures.append(failure(*config, "identity", error))
            progress(f"{model} {precision} d{dim}: identity FAILED {error}")
            continue
        passed.append(row)
        groups.setdefault((model, precision), []).append(row)
    identity: list[Record] = []
    for (model, precision), rows in groups.items():
        shared = record(rows[0]["shared_identity"])
        identity.append({"model": model, "precision": precision, "dims": [row["dim"] for row in rows],
                         "native_workload_identity": shared["native_workload_identity"],
                         "chunk_ids": shared["chunk_ids"], "status": "SAME" if len(rows) > 1 else "SINGLE_DIM"})
    return passed, failures, identity


def csv_row(row: Record) -> list[Json]:
    scu = record(row["scu"])
    values: list[Json] = [row["model"], row["precision"], row["dim"],
                          *(record(row["activation"])[name] for name in ACTIVATION[:4]),
                          *(record(scu[population])[name] for population in SCU_POPULATIONS for name in SCU[:3]),
                          *(record(row["residual"])[name] for name in RESIDUAL[:4])]
    return ["" if value is None else value for value in values]


def report(rows: list[Record], failures: list[Record], plain: bool, scu_breakdown: str) -> str:
    """The stdout tables: display only, in the configuration order (model, precision, DIM)."""
    def entries(values: str, population: str | None = None) -> list[tuple[str, str, int, Record]]:
        return [(text(record(row["shared_identity"]), "model"), text(row, "precision").upper(), integer(row, "dim"),
                 record(record(row[values])[population]) if population else record(row[values])) for row in rows]

    populations = ("dense",) if scu_breakdown == "dense" else SCU_POPULATIONS
    blocks = [metric_table("activation", entries("activation"), plain),
              *(metric_table("scu", entries("scu", population), plain, population.title()) for population in populations),
              metric_table("residual", entries("residual"), plain)]
    if failures:
        blocks.append(render("Failed configurations", ["Model", "Prec", "DIM", "Stage", "Error"],
                             [[str(row["model"] or "*"), str(row["precision"]).upper(), str(row["dim"]),
                               str(row["stage"]), str(row["error"])] for row in failures], plain, text_columns=5))
    return "\n\n".join(blocks)


def retire(root: Path) -> None:
    """A resumed sweep keeps the summaries of its earlier invocation as metric-summary.attempt-N.*."""
    current = root / "metric-summary.json"
    if not current.exists():
        return
    require(read_json(current).get("status") != "PASS", "sweep already PASS, nothing to resume: " + str(root))
    number = 1 + len(list(root.glob("metric-summary.attempt-*.json")))
    for suffix in (".json", ".csv"):
        if (root / f"metric-summary{suffix}").exists():
            (root / f"metric-summary{suffix}").rename(root / f"metric-summary.attempt-{number}{suffix}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--models", help="comma list (default gpt2,llama3.2-1B)")
    result.add_argument("--precisions", help="comma list (default a4w4,a8w8)")
    result.add_argument("--dims", help="comma list (default 16,32,64)")
    result.add_argument("--max-chunks", type=int, help="0 (default): all complete WikiText-2 test chunks; 1: smoke")
    result.add_argument("--seed", type=int, help="default 1234")
    result.add_argument("--dataset-manifest", type=Path, help="passed to every campaign.py run")
    result.add_argument("--im2p", type=Path, help="IM2P.sim checkout (default ../IM2P.sim)")
    result.add_argument("--build-cache", type=Path, help="campaign_build cache (default runs/.build-cache)")
    result.add_argument("--scu-mode", choices=("aggregate", "detailed"),
                        help="SCU collection: aggregate (default; integer sums) or detailed (every coordinate)")
    result.add_argument("--collection", choices=("combined", "separate"),
                        help="combined (default): activation, residual and SCU from one forward of one build per "
                             "precision x DIM; separate: one build and run per metric kind (the individual paths)")
    result.add_argument("--threads", type=int, help="combined: runner threads per shard (default 1; recorded)")
    result.add_argument("--threads-batch", type=int, help="combined: runner batch threads (default --threads; recorded)")
    result.add_argument("--workers", type=int, default=1,
                        help="combined: parallel contiguous chunk shards per configuration, merged exactly (default 1)")
    result.add_argument("--terminal-lm-head", choices=("metrics-only", "full"),
                        help="combined (default metrics-only): observe the terminal lm_head completely without "
                             "computing its logits, which no metric reads; full computes them (recorded)")
    result.add_argument("--allow-large-raw-scu", action="store_true",
                        help="accept detailed SCU collection of the full corpus (hundreds of GB of raw output)")
    target = result.add_mutually_exclusive_group()
    target.add_argument("--output", type=Path, help="new sweep directory")
    target.add_argument("--resume", type=Path, metavar="SWEEP_DIR",
                        help="continue a sweep: verified complete runs are reused, incomplete ones run again")
    result.add_argument("--jobs", type=int, default=4, help="build parallelism")
    result.add_argument("--timeout", type=int, default=1800, help="campaign.py --timeout of every run")
    result.add_argument("--keep-going", action="store_true",
                        help="run every configuration after a failure; the sweep still fails")
    result.add_argument("--scu-breakdown", choices=("dense", "all"), default="dense",
                        help="SCU stdout tables: dense (paper default) or dense, residual and overall")
    result.add_argument("--plain", action="store_true", help="ASCII column tables instead of box drawing")
    result.add_argument("--dry-run", action="store_true", help="print the plan; build and run nothing")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        sweep, root = resolve(args)
        require(args.workers > 0, "invalid --workers")
        if args.dry_run:
            print(dry_run(sweep, root, args.timeout, args.workers))
            return 0
        if root is None:
            raise EvaluationError("--output DIR or --resume SWEEP_DIR is required")
        progress(dry_run(sweep, root, args.timeout, args.workers).split("\n\n", 1)[0].replace(
            "dry run (nothing is built or run)", "preflight"))
        clean_environment()
        if args.resume is None:
            require(not root.exists(), f"sweep directory exists: {root} (continue it with --resume)")
            root.mkdir(parents=True)
            write_json(root / "manifest.json", {**sweep.record(), "command": [sys.executable, *sys.argv]})
        else:
            retire(root)
        build_timing: Record = {}
        passed, failures, identity = execute(root, sweep, args, build_timing)
        complete = not failures and len(passed) == len(sweep.configurations())
        summary: Record = {
            "schema": "potal-metric-sweep", "version": 1, "status": "PASS" if complete else "FAILED",
            "matrix": {"models": list(sweep.models), "precisions": list(sweep.precisions), "dims": list(sweep.dims)},
            "max_chunks": sweep.max_chunks, "seed": sweep.seed,
            "counts": {"configurations": len(sweep.configurations()),
                       "metric_runs": len(sweep.configurations()) * len(KINDS), "builds": len(sweep.builds()),
                       "native_collections": len(sweep.configurations()) * len(sweep.collections()),
                       "passed_configurations": len(passed)},
            "collection": {"mode": sweep.collection, "threads": sweep.threads, "threads_batch": sweep.threads_batch,
                           "workers": args.workers, "terminal_lm_head": sweep.terminal_lm_head},
            "build_timing": build_timing,
            "configurations": [*passed], "failures": [*failures], "dim_identity": [*identity],
            "csv": "metric-summary.csv", "aggregation": "copied from each run's summary.json; nothing recomputed"}
        write_json(root / "metric-summary.json", summary)
        with (root / "metric-summary.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(CSV_COLUMNS)
            writer.writerows(csv_row(row) for row in passed)
        print(report(passed, failures, args.plain, args.scu_breakdown))
        print(f"\nstatus: {summary['status']}\nsummary: {root / 'metric-summary.json'}\ncsv: {root / 'metric-summary.csv'}")
        return 0 if complete else 1
    except (EvaluationError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"metric sweep failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
