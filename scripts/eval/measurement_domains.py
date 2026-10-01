"""The measurement interface of the PoTal evaluation scripts, defined once: exactly three domains.

`run_measurement.py` dispatches from these definitions, the README tables are generated from them and the tests check
both. Terminology is fixed here:

  Performance  TTFT / TPOT / CPU-NPU timing (run_cycle_evaluation.py --run performance)
  Timeline     optional/export view of the stored scheduling result (same engine: --timeline compact, --run timeline)
  Metrics      activation / residual / SCU (campaign.py activation|residual|scu)

Perplexity/model-quality evaluation is intentionally outside this measurement interface and is run with
llama-perplexity.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from campaign_build import CYCLE_MODEL_OPTIONS, METRIC_SINKS, REPO, llama_plan
from eval_common import Json, Record

EVAL: Final = Path(__file__).resolve().parent
RUNNER: Final = "run_cycle_evaluation.py"
CAMPAIGN: Final = "campaign.py"
# The official domains (the wrapper's commands) and the metric subtypes; nothing else is a measurement domain.
DOMAINS: Final = ("performance", "timeline", "metric")
METRIC_KINDS: Final = tuple(METRIC_SINKS)  # activation, residual, scu
TERMINOLOGY: Final = {"performance": ("Performance", "TTFT / TPOT / CPU-NPU timing"),
                      "timeline": ("Timeline", "optional/export view of the stored scheduling result"),
                      "metric": ("Metrics", "activation / residual / SCU")}
OUTSIDE_INTERFACE: Final = ("Perplexity/model-quality evaluation is intentionally outside this measurement interface "
                            "and is run with llama-perplexity.")

# Dependency matrix columns: (key, question).
MATRIX_COLUMNS: Final = (
    ("cycle_model", "Needs cycle model?"), ("potal", "Needs PoTal?"), ("fullcpu", "Needs FullCPU?"),
    ("pmu", "Needs PMU?"), ("scheduler", "Needs scheduler?"), ("npu_clock", "Needs NPU clock?"),
    ("wikitext", "Needs WikiText?"), ("metric_instrumentation", "Needs metric instrumentation?"),
    ("timeline", "Produces timeline?"))
# Row names of the dependency matrix where they differ from the measurement name.
MATRIX_LABELS: Final = {"timeline": "timeline-only"}
# Build flags shown in the build mapping (read from campaign_build.llama_plan, never restated).
KEY_FLAGS: Final = (*(option for option, _ in METRIC_SINKS.values()),
                    "CYCLE_SIM", "LOG_CYCLE", "GGML_CPU_CYCLE_LOG", "GGML_GEMMINI_OPTION",
                    "GGML_GEMMINI_EXECUTION_BACKEND", "GGML_GEMMINI_DEFAULT_MATMUL_MODE")
Build = tuple[str, str | None]  # (build kind, matmul mode); no matmul mode = the cycle-model library


@dataclass(frozen=True, slots=True)
class Measurement:
    domain: str
    title: str
    command: tuple[str, ...]
    delegate: tuple[str, ...]
    example: str
    builds: tuple[Build, ...]
    build_note: str
    measures: str
    never: str
    inputs: str
    outputs: tuple[str, ...]
    reuse: str
    default_output: str
    needs: dict[str, str]

    def record(self) -> Record:
        builds: list[Json] = [{"kind": kind, "matmul_mode": matmul} for kind, matmul in self.builds]
        needs: Record = {key: value for key, value in self.needs.items()}
        return {"domain": self.domain, "title": self.title, "command": list(self.command),
                "delegate": list(self.delegate), "example": self.example, "builds": builds,
                "build_note": self.build_note, "measures": self.measures, "never": self.never,
                "inputs": self.inputs, "outputs": list(self.outputs), "reuse": self.reuse,
                "default_output": self.default_output, "needs": needs}


def _metric(kind: str, measures: str, raw: str, summary: str, layers: str,
            extra: tuple[str, ...] = ()) -> Measurement:
    flag = METRIC_SINKS[kind][0]
    return Measurement(
        domain="metric",
        title={"activation": "Activation metrics", "residual": "Residual metrics", "scu": "SCU metrics"}[kind],
        command=("metric", kind), delegate=(CAMPAIGN, kind),
        example=(f"python3 scripts/eval/run_measurement.py metric {kind} \\\n"
                 "  --model gpt2 --precision a8w8 --dim 32 [--max-chunks 1] [--model-manifest MANIFEST.json] "
                 "[--output DIR]"),
        builds=((kind, "FULL"),),
        build_note=(f"fresh `{kind}` build in `<output>/build` (`{flag}=1`, the other two metric sinks 0); "
                    "`--prepared-build DIR` reuses a verified `build_measurement.sh` build"),
        measures=measures,
        never="TTFT/TPOT, NPU cycle replay, scheduling, timeline; the other two metric sinks",
        inputs=("HP1 GGUF model (`--model gpt2|llama3.2-1B` or `--model-path`), WikiText-2 test text "
                "(`--dataset-manifest`), IM2P.sim checkout (functional backend sources)"),
        outputs=("manifest.json", "request.json", "evaluation_manifest.json", "dataset_manifest.json", "build/",
                 f"collection/{raw}", f"raw.jsonl{'.gz' if raw.endswith('.gz') else ''}",
                 f"{summary} (= summary.json)", f"{layers} (= layers.json)", *extra, "SHA256SUMS"),
        reuse="`--prepared-build`, `--evaluation-manifest`, `--workload-manifest` (same native chunks across metrics)",
        default_output=f"runs/metrics/{kind}/<utc>-<model>-<precision>-d<dim>",
        needs={"cycle_model": "no", "potal": "no", "fullcpu": "no", "pmu": "no", "scheduler": "no",
               "npu_clock": "no", "wikitext": "yes", "metric_instrumentation": "yes", "timeline": "no"})


MEASUREMENTS: Final[dict[str, Measurement]] = {
    "performance": Measurement(
        domain="performance", title="Performance",
        command=("performance",), delegate=(RUNNER, "--run", "performance"),
        example=("python3 scripts/eval/run_measurement.py performance \\\n"
                 "  --validation-mode nano-local --local-validation RECEIPT.json --model-manifest MANIFEST.json \\\n"
                 "  --model models/gpt2/gpt2.Q8_HP1.gguf --prompt-file wikitext-2-raw/wiki.test.raw \\\n"
                 "  --precision a8w8 --dim 32 [--reuse-collection RUN] [--timeline none|compact] \\\n"
                 "  [--stage-cache DIR] [--npu-task-cache FILE|none] [--clock-selection CLOCK.json] [--output DIR]"),
        builds=(("cycle-model", None), ("potal-host", "STRIPE_PIPELINE"), ("fullcpu-host", "FULL")),
        build_note=("cycle model (nano-local: the library the validation receipt admits, never rebuilt); PoTal and "
                    "FullCPU only for a fresh collection, none with `--reuse-collection`; all metric sinks 0"),
        measures=("TTFT, 127 TPOT intervals and token-ready endpoints from the CPU/NPU schedule; CPU service PMU "
                  "cycles, host elapsed ns and thread ns; NPU isolated-service cycles; optional timeline"),
        never="activation/residual/SCU metrics; target E2E ms without clock, interface and target-host evidence",
        inputs=("HP1 GGUF model and its model manifest, WikiText-2 test text, validation authority (nano-local "
                "receipt or certificate set), optionally a retained collection and a clock selection"),
        outputs=("manifest.json", "performance.json", "provenance.json", "timing-provenance.json", "build/",
                 "replay/ (NPU results, dataset, lifecycle, IR, schedule.sqlite, schedule-performance.json)",
                 "timeline/ (only with --timeline compact)", "SHA256SUMS"),
        reuse=("`--build-cache`, `--stage-cache` (content-addressed stages), `--npu-task-cache` (per-work NPU "
               "answers), `--reuse-collection` (CPU timing is then the reused collection's, not a new measurement)"),
        default_output="runs/performance/<utc>-<model>-<precision>-d<dim>-hp1",
        needs={"cycle_model": "yes", "potal": "fresh collection only", "fullcpu": "fresh collection only",
               "pmu": "fresh collection only", "scheduler": "yes", "npu_clock": "optional (ms only)",
               "wikitext": "yes", "metric_instrumentation": "no", "timeline": "optional (--timeline compact)"}),
    "timeline": Measurement(
        domain="timeline", title="Timeline",
        command=("timeline",), delegate=(RUNNER, "--run", "timeline"),
        example="python3 scripts/eval/run_measurement.py timeline --from-run PERFORMANCE_RUN [--format jsonl] [--output DIR]",
        builds=(),
        build_note="nothing is built",
        measures=("nothing new: the rows of the stored schedule (CPU worker intervals, NPU works, request and token "
                  "events with overlap, idle gaps and core provenance), the same rows `--timeline compact` writes"),
        never=("collection, replay, join, lifecycle, IR, scheduling, metrics; a run without its stored schedule "
               "fails instead of recomputing"),
        inputs=("a completed performance run whose schedule, IR and NPU results are still stored (run directory "
                "with `--keep-raw`, or its stage cache)"),
        outputs=("manifest.json", "timeline.jsonl", "timeline-rows.json", "export.json", "SHA256SUMS"),
        reuse="stored schedule, IR and NPU results, each verified against its stage receipt",
        default_output="runs/timeline/<utc>-<source run name>",
        needs={"cycle_model": "no", "potal": "no", "fullcpu": "no", "pmu": "no", "scheduler": "no (stored schedule)",
               "npu_clock": "no (source run's axis)", "wikitext": "no", "metric_instrumentation": "no",
               "timeline": "yes"}),
    "activation": _metric(
        "activation",
        "activation quantization and outlier statistics per layer and in aggregate (signed-row BK32 2-sigma "
        "reference, selection overlap, requantization ratio)",
        "activation-quant-metrics.jsonl", "activation_metrics.json", "layer_activation_metrics.json"),
    "residual": _metric(
        "residual",
        "residual / error-compensation path statistics (main, radix and compact stripes, zero-limb pruning, "
        "inactive-K compaction)",
        "residual-path-metrics.jsonl", "residual_metrics.json", "layer_residual_metrics.json",
        ("compact-shape-summary.json",)),
    "scu": _metric(
        "scu",
        "scale / block alignment statistics of the SCU path (HP1 block-PoT to channel-anchor shifts, partial-sum "
        "update fraction)",
        "scale-alignment-metrics.jsonl.gz", "scale_alignment_metrics.json", "layer_scale_metrics.json"),
}

# Every file of scripts/eval (and the build scripts next to it): (script, status, role, called by).
STATUSES: Final = ("CANONICAL", "WRAPPER", "INTERNAL", "LEGACY", "CERTIFICATION_ONLY", "DEBUG/RESEARCH")
SCRIPTS: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("run_measurement.py", "CANONICAL", "user entry point of the three domains; dispatch only", "user"),
    ("run_cycle_evaluation.py", "CANONICAL", "performance and timeline export engine (`--run`)",
     "run_measurement.py"),
    ("campaign.py", "CANONICAL",
     "metric engine: `activation`, `residual`, `scu` (its `cycle` kind is the legacy stateful replay adapter)",
     "run_measurement.py, run_*_metrics.sh"),
    ("run_activation_metrics.sh", "WRAPPER", "`campaign.py activation`", "user (kept)"),
    ("run_residual_metrics.sh", "WRAPPER", "`campaign.py residual`", "user (kept)"),
    ("run_scu_metrics.sh", "WRAPPER", "`campaign.py scu`", "user (kept)"),
    ("build_measurement.sh", "WRAPPER", "`campaign_build.py` CLI (separately verifiable build, `--dry-run`)", "user"),
    ("activation_quant_metrics.py", "WRAPPER", "standalone activation collect/reduce with a prepared runner",
     "user (advanced)"),
    ("residual_path_metrics.py", "WRAPPER", "standalone residual collect/reduce with a prepared runner",
     "user (advanced)"),
    ("scale_alignment_metrics.py", "WRAPPER", "standalone SCU collect/reduce with a prepared runner",
     "user (advanced)"),
    ("measurement_domains.py", "INTERNAL", "domain definitions for the wrapper, README tables and tests",
     "run_measurement.py"),
    ("measurement_identity.py", "INTERNAL",
     "read-only shared identity of finished runs (`run_measurement.py identity RUN...`)", "run_measurement.py"),
    ("campaign_build.py", "INTERNAL",
     "the build authority: every CMake option of every measurement build, receipts, build cache",
     "run_cycle_evaluation.py, campaign.py, cycle scripts"),
    ("metric_run.py", "INTERNAL", "native metric collection with sink isolation; body of the standalone reducers",
     "campaign.py, *_metrics.py"),
    ("campaign_metrics.py", "INTERNAL", "aggregate and per-layer metric outputs", "campaign.py"),
    ("campaign_inputs.py", "INTERNAL", "model/tokenizer/dataset identity, SHA256SUMS", "campaign.py, cycle scripts"),
    ("campaign_stream.py", "INTERNAL", "streaming gzip sink for SCU observations", "metric_run.py"),
    ("campaign_verify.py", "INTERNAL", "verifier CLI for a finished metric run (manifest binding, checksums)", "user"),
    ("e2e_timeline.py", "INTERNAL", "the one timeline row builder and performance accumulator",
     "run_cycle_evaluation.py, schedule_engine.py"),
    ("schedule_engine.py", "INTERNAL", "one scheduling pass with performance and timeline sinks",
     "run_cycle_evaluation.py"),
    ("stage_cache.py", "INTERNAL", "content-addressed offline stage cache", "run_cycle_evaluation.py"),
    ("model_manifest.py", "INTERNAL", "frozen model manifest (create/verify CLI, `model_entry`)",
     "run_cycle_evaluation.py, campaign.py"),
    ("evaluation_host.py", "INTERNAL", "read-only host observations", "run_cycle_evaluation.py, e2e_run.py"),
    ("pmu_probe.c", "INTERNAL", "PMU readiness probe compiled by the performance preflight",
     "run_cycle_evaluation.py"),
    ("eval_common.py", "INTERNAL", "shared helpers (certificate-pinned)", "all"),
    ("end_to_end.py", "INTERNAL", "native collection (`collect`) and certified reconstruction (`reconstruct`) CLI",
     "run_cycle_evaluation.py"),
    ("e2e_run.py", "INTERNAL", "native collection logic", "end_to_end.py"),
    ("run_stateful_integration_smoke.py", "INTERNAL", "256+1 smoke pair collector (`--smoke`)",
     "run_cycle_evaluation.py"),
    ("paired_inputs.py", "INTERNAL", "PoTal/FullCPU pairing inputs", "e2e_run.py"),
    ("application_results.py", "INTERNAL", "application endpoint results (certificate-pinned)",
     "end_to_end.py, offline_pipeline.py"),
    ("target_admission.py", "INTERNAL", "fail-closed publication gates (clock, target host, interface)",
     "run_cycle_evaluation.py"),
    ("metric_reducers.py", "LEGACY", "unbound ACT/RES reducers for historical streams (`--reduce` without a manifest)",
     "metric_run.py"),
    ("offline_pipeline.py", "CERTIFICATION_ONLY", "certified offline replay/join/IR (`--validation-mode certified`)",
     "end_to_end.py"),
    ("certified_reconstruction.py", "CERTIFICATION_ONLY", "certified reconstruction results", "offline_pipeline.py"),
    ("scheduled_endpoints.py", "CERTIFICATION_ONLY", "scheduled endpoints of certified schedules",
     "certified_reconstruction.py"),
    ("e2e_cost_ownership.py", "CERTIFICATION_ONLY", "cost-ownership and interface contracts", "target_admission.py"),
    ("e2e_integration_prep.py", "CERTIFICATION_ONLY", "stateful host+NPU integration prep on a certified trace",
     "user"),
    ("campaign_cycle.py", "CERTIFICATION_ONLY", "stateful provider replay behind `campaign.py cycle`",
     "campaign.py, cycle_campaign.py"),
    ("cycle_campaign.py", "CERTIFICATION_ONLY", "certified trace cycle accounting", "run_cycle_campaign.sh"),
    ("run_cycle_campaign.sh", "CERTIFICATION_ONLY", "`cycle_campaign.py`", "user"),
    ("cycle_trace_capture.py", "CERTIFICATION_ONLY", "actual-inference trace capture and certificate",
     "run_cycle_trace_capture.sh"),
    ("run_cycle_trace_capture.sh", "CERTIFICATION_ONLY", "`cycle_trace_capture.py`", "user"),
    ("cycle_campaign_bundle.py", "CERTIFICATION_ONLY", "evaluation-cycle campaign bundle", "user"),
    ("actual_cycle_bundle.py", "CERTIFICATION_ONLY", "actual-cycle campaign bundle", "user"),
    ("cycle_accounting.py", "CERTIFICATION_ONLY", "cycle accounting definitions", "cycle_campaign.py"),
)
# Outside scripts/eval: never a measurement build or entry point.
OUTSIDE: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("build-arm64.sh, build-arm64-cpu.sh, build-arm64-fpga-uart.sh, build-x86.sh, build-riscv.sh", "DEBUG/RESEARCH",
     "developer builds with their own option defaults; never a measurement build", "developer"),
    ("scripts/experiment/, scripts/utils/", "DEBUG/RESEARCH", "cycle matrix experiments and log rendering helpers",
     "developer"),
    ("evaluation/ (python -m evaluation)", "INTERNAL", "ACT/RES/SCU reducers, schemas and manifest type",
     "campaign_metrics.py, metric_run.py"),
)

# Build audit: (script, caller, build kinds, semantic options, platform options, output, status). Every measurement
# build takes its options from campaign_build.py; no other script of this table defines a measurement option.
BUILD_SCRIPT_FIELDS: Final = ("script", "caller", "build_kinds", "semantic_options", "platform_options", "output",
                              "status")
BUILD_SCRIPTS: Final[tuple[tuple[str, str, str, str, str, str, str], ...]] = (
    ("scripts/eval/campaign_build.py", "all measurement scripts; CLI through build_measurement.sh",
     "`cycle`, `activation`, `residual`, `scu`, `potal-host`, `potal-host-nocpulog`, `fullcpu-host`, `cycle-model`",
     "defined here and nowhere else: `llama_plan()`, `METRIC_SINKS`, `CYCLE_MODEL_OPTIONS`",
     ("`platform_profile()`: library suffix, binary format, install name; adds no CMake option today and may never "
      "add a semantic one (`with_platform`)"),
     ("build directory with configure/build/verify logs, `build-info.json`, `artifacts.json`, `build-receipt.json`; "
      "`cached_*` entries under `--build-cache`"),
     "ACTIVE (authority)"),
    ("scripts/eval/build_measurement.sh", "user", "same kinds (`--kind`, `--dry-run`)", "from campaign_build.py",
     "from campaign_build.py", "`--output DIR`", "ACTIVE (wrapper)"),
    ("scripts/eval/run_cycle_evaluation.py", "run_measurement.py performance, timeline",
     "performance: `cycle-model`, `potal-host` (STRIPE_PIPELINE), `fullcpu-host` (FULL); timeline: none",
     "none of its own: `--precision`, `--dim` select the `llama_plan()` profile", "from campaign_build.py",
     "`--build-cache/<kind>-<identity>`; per-build summary in `<run>/build/*.json`", "ACTIVE"),
    ("scripts/eval/campaign.py", "run_measurement.py metric KIND, run_*_metrics.sh",
     "`activation`, `residual`, `scu` (FULL); `cycle` only for its legacy adapter",
     "none of its own: `--precision`, `--dim` select the `llama_plan()` profile", "from campaign_build.py",
     "`<output>/build` (fresh) or a verified `--prepared-build`", "ACTIVE"),
    ("scripts/eval/cycle_trace_capture.py", "run_cycle_trace_capture.sh", "`cycle` (STRIPE_PIPELINE)",
     "none of its own", "from campaign_build.py", "`<config>/build` or `--prepared-build`", "CERTIFICATION_ONLY"),
    ("scripts/eval/campaign_cycle.py", "campaign.py cycle", "`cycle-model`", "`CYCLE_MODEL_OPTIONS`",
     "from campaign_build.py", "`<output>/cycle-library-build`", "CERTIFICATION_ONLY (legacy adapter)"),
    ("build-arm64.sh, build-arm64-cpu.sh, build-arm64-fpga-uart.sh, build-x86.sh, build-riscv.sh", "developer",
     "none (developer builds, not a measurement kind)",
     ("own environment defaults resolved by `scripts/im2p-build-options.py`; they differ from every measurement "
      "profile (backend, DIM, bits, OpenMP, runtime matmul override)"),
     "host and toolchain specific (native flags, OpenMP, cross toolchain)",
     ("`build-arm64/`, `build-arm64-cpu/`, `build-arm64-fpga-uart/`, `build-x86/`, `build-riscv[-static]/`; no "
      "receipt"),
     "DEBUG/RESEARCH (never admitted by a runner)"),
)


def measurement(command: tuple[str, ...]) -> tuple[str, Measurement]:
    for name, row in MEASUREMENTS.items():
        if row.command == command:
            return name, row
    raise KeyError(" ".join(command))


def dependency_matrix() -> dict[str, dict[str, str]]:
    return {name: {key: row.needs[key] for key, _ in MATRIX_COLUMNS} for name, row in MEASUREMENTS.items()}


def build_mapping(precision: str = "a8w8", dim: int = 32) -> list[Record]:
    """Build kinds of every measurement with their key CMake flags, read from campaign_build (the authority)."""
    rows: list[Record] = []
    for name, row in MEASUREMENTS.items():
        if not row.builds:
            rows.append({"measurement": name, "build_kind": None, "flags": {}, "binary": None})
        for kind, matmul in row.builds:
            if matmul is None:
                flags: Record = {key: value for key, value in CYCLE_MODEL_OPTIONS.items()}
                rows.append({"measurement": name, "build_kind": kind, "flags": flags,
                             "binary": "libim2p_cycle_model (IM2P.sim sim/cycle)"})
                continue
            plan = llama_plan(kind, precision, dim, REPO.parent / "IM2P.sim", matmul)
            shown: Record = {key: plan.options[key] for key in KEY_FLAGS if key in plan.options}
            rows.append({"measurement": name, "build_kind": kind, "flags": shown,
                         "binary": "llama-eval-workload",
                         "semantic_options_sha256": plan.semantic_options_sha256})
    return rows


def description() -> Record:
    measurements: Record = {name: row.record() for name, row in MEASUREMENTS.items()}
    scripts: list[Json] = [{"script": script, "status": status, "role": role, "called_by": callers}
                           for script, status, role, callers in (*SCRIPTS, *OUTSIDE)]
    columns: list[Json] = [{"key": key, "question": question} for key, question in MATRIX_COLUMNS]
    rows: Record = {name: {key: value for key, value in needs.items()} for name, needs in dependency_matrix().items()}
    builds: list[Json] = [row for row in build_mapping()]
    terminology: Record = {key: {"title": title, "meaning": meaning} for key, (title, meaning) in TERMINOLOGY.items()}
    kinds: list[Json] = list(METRIC_KINDS)
    domains: list[Json] = list(DOMAINS)
    sinks: Record = {kind: {"cmake_option": option, "build_info_key": key} for kind, (option, key) in METRIC_SINKS.items()}
    audit: list[Json] = [dict(zip(BUILD_SCRIPT_FIELDS, row)) for row in BUILD_SCRIPTS]
    return {"schema": "potal-measurement-domains", "version": 1, "domains": domains, "metric_kinds": kinds,
            "terminology": terminology, "outside_interface": OUTSIDE_INTERFACE, "measurements": measurements,
            "dependency_matrix": {"columns": columns, "rows": rows,
                                  "labels": {name: MATRIX_LABELS.get(name, name) for name in MEASUREMENTS}},
            "build_mapping": builds, "metric_sinks": sinks, "build_scripts": audit, "scripts": scripts}


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


SECTIONS: Final = ("commands", "matrix", "builds", "build-scripts", "outputs", "scripts")


def markdown(section: str) -> str:
    """One generated README block (a name of SECTIONS)."""
    if section == "commands":
        return "\n".join(
            f"### {row.title}\n\n```bash\n{row.example}\n```\n\n"
            f"- Builds: {row.build_note}.\n- Measures: {row.measures}.\n- Does not run: {row.never}.\n"
            f"- Inputs: {row.inputs}.\n- Cache / reuse: {row.reuse}.\n"
            f"- Outputs (default `{row.default_output}`): {', '.join(row.outputs)}.\n"
            for row in MEASUREMENTS.values())
    if section == "matrix":
        return _table(["Measurement", *(question for _, question in MATRIX_COLUMNS)],
                      [[MATRIX_LABELS.get(name, name), *(needs[key] for key, _ in MATRIX_COLUMNS)]
                       for name, needs in dependency_matrix().items()])
    if section == "builds":
        rows: list[list[str]] = []
        for build in build_mapping():
            flags = build["flags"]
            assert isinstance(flags, dict)
            rows.append([str(build["measurement"]), f"`{build['build_kind']}`" if build["build_kind"] else "none",
                         "<br>".join(f"`{key}={value}`" for key, value in flags.items()) or "-",
                         str(build["binary"] or "-")])
        return _table(["Measurement", "Build kind", "Key CMake flags (a8w8, DIM 32)", "Binary"], rows)
    if section == "build-scripts":
        return _table(["Script", "Caller", "Build kinds", "Semantic options", "Platform options", "Output", "Status"],
                      [[f"`{script}`" if " " not in script else script, *rest] for script, *rest in BUILD_SCRIPTS])
    if section == "outputs":
        return _table(["Measurement", "Default directory", "Artifacts"],
                      [[name, f"`{row.default_output}`",
                        "<br>".join(f"`{item}`" if " " not in item else item for item in row.outputs)]
                       for name, row in MEASUREMENTS.items()])
    if section == "scripts":
        return _table(["Script", "Status", "Role", "Called by"],
                      [[f"`{script}`" if " " not in script else script, status, role, callers]
                       for script, status, role, callers in (*SCRIPTS, *OUTSIDE)])
    raise KeyError(section)


def update_readme(path: Path) -> bool:
    """Rewrite the generated blocks of the README in place; True when the file changed."""
    text = path.read_text()
    updated = text
    for section in SECTIONS:
        begin, end = f"<!-- BEGIN GENERATED {section} -->\n", f"<!-- END GENERATED {section} -->"
        start, stop = updated.index(begin) + len(begin), updated.index(end)
        updated = updated[:start] + markdown(section) + updated[stop:]
    if updated != text:
        path.write_text(updated)
    return updated != text
