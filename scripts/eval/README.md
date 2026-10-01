# PoTal measurement scripts

Run from the llama checkout with Python 3.11 or newer. One entry point, `scripts/eval/run_measurement.py`, is the
measurement interface. It has exactly three domains, and the domains never run each other: a performance run does
not collect metrics and a metric run does not replay or schedule.

| Domain | Command | What it is | Engine |
|---|---|---|---|
| **Performance** | `performance` | TTFT / TPOT / CPU-NPU timing: CPU hardware cycles and host timing, NPU simulated cycles, CPU/NPU pipelining reconstruction | `run_cycle_evaluation.py --run performance` |
| **Timeline** | `timeline` | optional/export view of the stored scheduling result (overlap, idle gaps, token boundaries, core provenance): written by the same scheduling pass or exported later | `--timeline compact`, `run_cycle_evaluation.py --run timeline` |
| **Metrics** | `metric activation\|residual\|scu` | activation / residual / SCU | `campaign.py activation\|residual\|scu` |
| | `metric all` | the three metrics over models × precisions × DIMs, aggregated (orchestration, not a fourth metric) | `metric_sweep.py` |

```text
run_measurement.py
├── performance
├── timeline
└── metric
    ├── activation
    ├── residual
    ├── scu
    └── all          orchestration of the three over a matrix (see "Metric sweep")
```

"Metrics" always means activation, residual and SCU. Perplexity/model-quality evaluation is intentionally outside
this measurement interface and is run with llama-perplexity.

`python3 scripts/eval/run_measurement.py describe [--json]` prints the definitions below; they come from
`measurement_domains.py`, and `describe --update-readme` regenerates the marked tables of this file.
`run_measurement.py identity RUN [RUN ...]` compares the model/configuration identity of finished runs of the three
domains (see "Shared identity").

## Canonical commands

`run_measurement.py DOMAIN ...` only chooses the engine and a default output directory; every other option is
passed through, so `run_measurement.py performance --help` (or `metric activation --help`) shows the real options.

<!-- BEGIN GENERATED commands -->
### Performance

```bash
python3 scripts/eval/run_measurement.py performance \
  --validation-mode nano-local --local-validation RECEIPT.json --model-manifest MANIFEST.json \
  --model models/gpt2/gpt2.Q8_HP1.gguf --prompt-file wikitext-2-raw/wiki.test.raw \
  --precision a8w8 --dim 32 [--reuse-collection RUN] [--timeline none|compact] \
  [--stage-cache DIR] [--npu-task-cache FILE|none] [--clock-selection CLOCK.json] [--output DIR]
```

- Builds: cycle model (nano-local: the library the validation receipt admits, never rebuilt); PoTal and FullCPU only for a fresh collection, none with `--reuse-collection`; all metric sinks 0.
- Measures: TTFT, 127 TPOT intervals and token-ready endpoints from the CPU/NPU schedule; CPU service PMU cycles, host elapsed ns and thread ns; NPU isolated-service cycles; optional timeline.
- Does not run: activation/residual/SCU metrics; target E2E ms without clock, interface and target-host evidence.
- Inputs: HP1 GGUF model and its model manifest, WikiText-2 test text, validation authority (nano-local receipt or certificate set), optionally a retained collection and a clock selection.
- Cache / reuse: `--build-cache`, `--stage-cache` (content-addressed stages), `--npu-task-cache` (per-work NPU answers), `--reuse-collection` (CPU timing is then the reused collection's, not a new measurement).
- Outputs (default `runs/performance/<utc>-<model>-<precision>-d<dim>-hp1`): manifest.json, performance.json, provenance.json, timing-provenance.json, build/, replay/ (NPU results, dataset, lifecycle, IR, schedule.sqlite, schedule-performance.json), timeline/ (only with --timeline compact), SHA256SUMS.

### Timeline

```bash
python3 scripts/eval/run_measurement.py timeline --from-run PERFORMANCE_RUN [--format jsonl] [--output DIR]
```

- Builds: nothing is built.
- Measures: nothing new: the rows of the stored schedule (CPU worker intervals, NPU works, request and token events with overlap, idle gaps and core provenance), the same rows `--timeline compact` writes.
- Does not run: collection, replay, join, lifecycle, IR, scheduling, metrics; a run without its stored schedule fails instead of recomputing.
- Inputs: a completed performance run whose schedule, IR and NPU results are still stored (run directory with `--keep-raw`, or its stage cache).
- Cache / reuse: stored schedule, IR and NPU results, each verified against its stage receipt.
- Outputs (default `runs/timeline/<utc>-<source run name>`): manifest.json, timeline.jsonl, timeline-rows.json, export.json, SHA256SUMS.

### Activation metrics

```bash
python3 scripts/eval/run_measurement.py metric activation \
  --model gpt2 --precision a8w8 --dim 32 [--max-chunks 1] [--model-manifest MANIFEST.json] [--output DIR]
```

- Builds: fresh `activation` build in `<output>/build` (`GGML_GEMMINI_ACT_METRICS=1`, the other two metric sinks 0); `--prepared-build DIR` reuses a verified `build_measurement.sh` build.
- Measures: activation quantization and outlier statistics per layer and in aggregate (signed-row BK32 2-sigma reference, selection overlap, requantization ratio).
- Does not run: TTFT/TPOT, NPU cycle replay, scheduling, timeline; the other two metric sinks.
- Inputs: HP1 GGUF model (`--model gpt2|llama3.2-1B` or `--model-path`), WikiText-2 test text (`--dataset-manifest`), IM2P.sim checkout (functional backend sources).
- Cache / reuse: `--prepared-build`, `--evaluation-manifest`, `--workload-manifest` (same native chunks across metrics).
- Outputs (default `runs/metrics/activation/<utc>-<model>-<precision>-d<dim>`): manifest.json, request.json, evaluation_manifest.json, dataset_manifest.json, build/, collection/activation-quant-metrics.jsonl, raw.jsonl, activation_metrics.json (= summary.json), layer_activation_metrics.json (= layers.json), SHA256SUMS.

### Residual metrics

```bash
python3 scripts/eval/run_measurement.py metric residual \
  --model gpt2 --precision a8w8 --dim 32 [--max-chunks 1] [--model-manifest MANIFEST.json] [--output DIR]
```

- Builds: fresh `residual` build in `<output>/build` (`GGML_GEMMINI_RESIDUAL_METRICS=1`, the other two metric sinks 0); `--prepared-build DIR` reuses a verified `build_measurement.sh` build.
- Measures: residual / error-compensation path statistics (main, radix and compact stripes, zero-limb pruning, inactive-K compaction).
- Does not run: TTFT/TPOT, NPU cycle replay, scheduling, timeline; the other two metric sinks.
- Inputs: HP1 GGUF model (`--model gpt2|llama3.2-1B` or `--model-path`), WikiText-2 test text (`--dataset-manifest`), IM2P.sim checkout (functional backend sources).
- Cache / reuse: `--prepared-build`, `--evaluation-manifest`, `--workload-manifest` (same native chunks across metrics).
- Outputs (default `runs/metrics/residual/<utc>-<model>-<precision>-d<dim>`): manifest.json, request.json, evaluation_manifest.json, dataset_manifest.json, build/, collection/residual-path-metrics.jsonl, raw.jsonl, residual_metrics.json (= summary.json), layer_residual_metrics.json (= layers.json), compact-shape-summary.json, SHA256SUMS.

### SCU metrics

```bash
python3 scripts/eval/run_measurement.py metric scu \
  --model gpt2 --precision a8w8 --dim 32 [--max-chunks 1] [--model-manifest MANIFEST.json] [--output DIR]
```

- Builds: fresh `scu` build in `<output>/build` (`GGML_GEMMINI_SCALE_METRICS=1`, the other two metric sinks 0); `--prepared-build DIR` reuses a verified `build_measurement.sh` build.
- Measures: scale / block alignment statistics of the SCU path (HP1 block-PoT to channel-anchor shifts, partial-sum update fraction).
- Does not run: TTFT/TPOT, NPU cycle replay, scheduling, timeline; the other two metric sinks.
- Inputs: HP1 GGUF model (`--model gpt2|llama3.2-1B` or `--model-path`), WikiText-2 test text (`--dataset-manifest`), IM2P.sim checkout (functional backend sources).
- Cache / reuse: `--prepared-build`, `--evaluation-manifest`, `--workload-manifest` (same native chunks across metrics).
- Outputs (default `runs/metrics/scu/<utc>-<model>-<precision>-d<dim>`): manifest.json, request.json, evaluation_manifest.json, dataset_manifest.json, build/, collection/scale-alignment-metrics.jsonl.gz, raw.jsonl.gz, scale_alignment_metrics.json (= summary.json), layer_scale_metrics.json (= layers.json), with `--scu-mode aggregate`: collection/scale-alignment-aggregate.jsonl and raw.jsonl instead of the gzip observations, SHA256SUMS.
<!-- END GENERATED commands -->

The former entry points stay valid: `run_activation_metrics.sh`, `run_residual_metrics.sh`, `run_scu_metrics.sh`
(`campaign.py KIND`) and `run_cycle_evaluation.py` itself. `campaign.py` still prints only its run directory on
stdout; the one-row result table of a metric run goes to stderr.

### Metric sweep (`metric all`)

```bash
# One metric
python3 scripts/eval/run_measurement.py metric activation --model gpt2 --precision a8w8 --dim 32 --max-chunks 1

# Full paper metric sweep (these are the defaults: `metric all` alone runs this matrix)
python3 scripts/eval/run_measurement.py metric all \
  --models gpt2,llama3.2-1B \
  --precisions a4w4,a8w8 \
  --dims 16,32,64 \
  --max-chunks 0

# Smoke subset, the plan only, and an interrupted sweep
python3 scripts/eval/run_measurement.py metric all --models gpt2 --precisions a8w8 --dims 32 --max-chunks 1
python3 scripts/eval/run_measurement.py metric all --dry-run
python3 scripts/eval/run_measurement.py metric all --resume runs/metrics/sweep-<utc>
```

`metric all` is an orchestration layer. It does not define a fourth metric and does not change individual metric
semantics: every run is an unchanged `campaign.py activation|residual|scu` process with its own run directory, and the
sweep only reads finished runs (`metric_sweep.py`).

- Builds: one per metric kind × precision × DIM (18 for the full matrix) from `campaign_build.cached_llama_build`
  (`--build-cache`, default `runs/.build-cache`), passed as `campaign.py --prepared-build`; GPT-2 and Llama use the
  same build. `<sweep>/builds/` links them, and a resumed sweep must resolve to the same builds.
- Workload: in each configuration (model × precision × DIM) the activation run is the anchor; residual and SCU get its
  `evaluation_manifest.json` (`--evaluation-manifest`) and `collection/workload-binding.json` (`--workload-manifest`).
  The sweep then requires the same model, dataset, tokenizer, precision, DIM, seed, chunk policy, native workload
  identity and chunk IDs in the three runs (`measurement_identity` plus the workload files), and the same native
  workload and chunk IDs across the DIMs of one model and precision. Builds differ by design; each configuration
  records the build receipt, runner and semantic-options SHA-256 it used.
- Output (`runs/metrics/sweep-<utc>`): `manifest.json` (the recorded matrix and options), `builds/`,
  `<model>/<precision>/d<dim>/{activation,residual,scu}/` (unchanged campaign run directories, `<kind>.campaign.log`
  next to each), `metric-summary.json` and `metric-summary.csv`.
- `metric-summary.json` (`potal-metric-sweep`): status, matrix, counts, one entry per passing configuration (shared
  identity, builds, runs, activation fields with their counts, SCU `dense`/`residual`/`overall`, residual fields),
  the failures and the workload identity per model and precision. `metric-summary.csv`: one row per passing
  configuration with the paper columns. Every value is copied from a run's `summary.json`; nothing is recomputed.
- stdout: the Activation Adaptation, SCU Weight-Scale Alignment (dense; `--scu-breakdown all` adds residual and
  overall) and Residual Overhead tables, ordered by model, precision and DIM; `--plain` prints ASCII columns. The
  tables are for reading; exact values are in the JSON/CSV.
- Failures: a failed build, metric run or identity check makes the sweep `FAILED` (exit 1) and is listed. The default
  stops at the first failure; `--keep-going` runs every configuration and prints a failed-configuration table. A
  failed configuration never enters the tables, the CSV or the configuration list.
- `--resume DIR` keeps the recorded matrix and options. A complete run (`SHA256SUMS`) is verified with
  `campaign_verify` and reused; an incomplete run stays untouched and its metric runs again in `<kind>.retry-N`; a
  complete run that fails verification stops the sweep. Earlier summaries are kept as `metric-summary.attempt-N.*`.
- SCU collection mode (`--scu-mode`, default `aggregate` for `metric all`; `metric scu` keeps `detailed` unless
  `--scu-mode aggregate` is given). `detailed` streams one `SCALE_ALIGNMENT` record per (invocation, work type,
  stripe, column, original block): about 25M records / 330 MB per GPT-2 A8W8 DIM 32 chunk. `aggregate` runs the same
  per-coordinate checks in the producer (scale/offset/update consistency, finite scales, duplicate coordinates per
  invocation) and streams only integer sums per (chunk, layer, work type) (`im2p-scale-alignment-aggregate`); the
  reducer turns both into the same summary, so ratios and the paper columns are identical. `metric all` refuses
  detailed SCU with `--max-chunks 0` unless `--allow-large-raw-scu` is given. A resumed sweep reuses an SCU run only
  when its `scu_collection_mode` and `scu_reducer_sha256` match; the summary records `scu.collection_mode`.
- `--dry-run` prints the counts, the builds with their `semantic_options_sha256` and every `campaign.py` command;
  a real run prints the same preflight header first.
  `--timeout` is passed to every `campaign.py` run (its per-command limit, default 1800 s).

## Dependency matrix

What a measurement needs before it can run, so nothing unnecessary is installed, built or executed.

<!-- BEGIN GENERATED matrix -->
| Measurement | Needs cycle model? | Needs PoTal? | Needs FullCPU? | Needs PMU? | Needs scheduler? | Needs NPU clock? | Needs WikiText? | Needs metric instrumentation? | Produces timeline? |
|---|---|---|---|---|---|---|---|---|---|
| performance | yes | fresh collection only | fresh collection only | fresh collection only | yes | optional (ms only) | yes | no | optional (--timeline compact) |
| timeline-only | no | no | no | no | no (stored schedule) | no (source run's axis) | no | no | yes |
| activation | no | no | no | no | no | no | yes | yes | no |
| residual | no | no | no | no | no | no | yes | yes | no |
| scu | no | no | no | no | no | no | yes | yes | no |
<!-- END GENERATED matrix -->

- Cycle model: the IM2P cycle-model library (`libim2p_cycle_model`). Metric builds compile the functional IM2P_SIM
  backend from the IM2P.sim checkout but never load or call the cycle model.
- PMU: Linux aarch64 collection needs `kernel.perf_event_paranoid<=1` and `kernel.perf_user_access=1` (the runner
  checks and fails fast; it never changes a system setting). A reused collection carries its own PMU contract.
- NPU clock: `--clock-selection CLOCK.json` (validated operating clock). Without it the schedule is placed on the
  configured 1 GHz test clock and every ms value stays null; a new clock reschedules, it never rescales.

## Builds

`campaign_build.py` is the only place that turns a measurement configuration into CMake options: `llama_plan()`
for every llama build kind and `CYCLE_MODEL_OPTIONS` for the cycle-model library. Only `platform_profile()` may
differ between hosts. `build_measurement.sh --kind KIND [--dry-run]` is its CLI. The table is read from it.

<!-- BEGIN GENERATED builds -->
| Measurement | Build kind | Key CMake flags (a8w8, DIM 32) | Binary |
|---|---|---|---|
| performance | `cycle-model` | `CMAKE_BUILD_TYPE=Release`<br>`CMAKE_EXPORT_COMPILE_COMMANDS=ON` | libim2p_cycle_model (IM2P.sim sim/cycle) |
| performance | `potal-host` | `GGML_GEMMINI_ACT_METRICS=0`<br>`GGML_GEMMINI_RESIDUAL_METRICS=0`<br>`GGML_GEMMINI_SCALE_METRICS=0`<br>`CYCLE_SIM=1`<br>`LOG_CYCLE=1`<br>`GGML_CPU_CYCLE_LOG=ON`<br>`GGML_GEMMINI_OPTION=WS`<br>`GGML_GEMMINI_EXECUTION_BACKEND=IM2P_SIM`<br>`GGML_GEMMINI_DEFAULT_MATMUL_MODE=STRIPE_PIPELINE` | llama-eval-workload |
| performance | `fullcpu-host` | `GGML_GEMMINI_ACT_METRICS=0`<br>`GGML_GEMMINI_RESIDUAL_METRICS=0`<br>`GGML_GEMMINI_SCALE_METRICS=0`<br>`CYCLE_SIM=0`<br>`LOG_CYCLE=1`<br>`GGML_CPU_CYCLE_LOG=ON`<br>`GGML_GEMMINI_OPTION=CPU`<br>`GGML_GEMMINI_EXECUTION_BACKEND=HARDWARE`<br>`GGML_GEMMINI_DEFAULT_MATMUL_MODE=FULL` | llama-eval-workload |
| timeline | none | - | - |
| activation | `activation` | `GGML_GEMMINI_ACT_METRICS=1`<br>`GGML_GEMMINI_RESIDUAL_METRICS=0`<br>`GGML_GEMMINI_SCALE_METRICS=0`<br>`CYCLE_SIM=1`<br>`LOG_CYCLE=0`<br>`GGML_CPU_CYCLE_LOG=OFF`<br>`GGML_GEMMINI_OPTION=WS`<br>`GGML_GEMMINI_EXECUTION_BACKEND=IM2P_SIM`<br>`GGML_GEMMINI_DEFAULT_MATMUL_MODE=FULL` | llama-eval-workload |
| residual | `residual` | `GGML_GEMMINI_ACT_METRICS=0`<br>`GGML_GEMMINI_RESIDUAL_METRICS=1`<br>`GGML_GEMMINI_SCALE_METRICS=0`<br>`CYCLE_SIM=1`<br>`LOG_CYCLE=0`<br>`GGML_CPU_CYCLE_LOG=OFF`<br>`GGML_GEMMINI_OPTION=WS`<br>`GGML_GEMMINI_EXECUTION_BACKEND=IM2P_SIM`<br>`GGML_GEMMINI_DEFAULT_MATMUL_MODE=FULL` | llama-eval-workload |
| scu | `scu` | `GGML_GEMMINI_ACT_METRICS=0`<br>`GGML_GEMMINI_RESIDUAL_METRICS=0`<br>`GGML_GEMMINI_SCALE_METRICS=1`<br>`CYCLE_SIM=1`<br>`LOG_CYCLE=0`<br>`GGML_CPU_CYCLE_LOG=OFF`<br>`GGML_GEMMINI_OPTION=WS`<br>`GGML_GEMMINI_EXECUTION_BACKEND=IM2P_SIM`<br>`GGML_GEMMINI_DEFAULT_MATMUL_MODE=FULL` | llama-eval-workload |
<!-- END GENERATED builds -->

- Semantic options (precision, DIM, backend, quantization, metric sinks) form `semantic_options_sha256` in every
  build receipt; `LOG_CYCLE`, `GGML_CPU_CYCLE_LOG`, `CYCLE_DETAIL`, `LOG_DEBUG` are instrumentation only.
- Metric sinks are named once, in `campaign_build.METRIC_SINKS`: `activation` = `GGML_GEMMINI_ACT_METRICS`,
  `residual` = `GGML_GEMMINI_RESIDUAL_METRICS`, `scu` = `GGML_GEMMINI_SCALE_METRICS`. A metric build sets its own
  option to 1 and the other two to 0; every other kind sets all three to 0. `build()` refuses a binary whose
  `--build-info` reports anything else and `metric_run.validate_metric_recipe` refuses to collect with a foreign
  or doubly enabled build.
- Performance builds are cached by source/configuration/toolchain identity (`--build-cache`); nano-local uses the
  cycle library its validation receipt admits. Metric runs build fresh into `<output>/build` or reuse
  `--prepared-build`.

Build scripts and where their options come from (the root `build-*.sh` scripts are developer builds; their binaries
are not admitted by any runner):

<!-- BEGIN GENERATED build-scripts -->
| Script | Caller | Build kinds | Semantic options | Platform options | Output | Status |
|---|---|---|---|---|---|---|
| `scripts/eval/campaign_build.py` | all measurement scripts; CLI through build_measurement.sh | `cycle`, `activation`, `residual`, `scu`, `potal-host`, `potal-host-nocpulog`, `fullcpu-host`, `cycle-model` | defined here and nowhere else: `llama_plan()`, `METRIC_SINKS`, `CYCLE_MODEL_OPTIONS` | `platform_profile()`: library suffix, binary format, install name; adds no CMake option today and may never add a semantic one (`with_platform`) | build directory with configure/build/verify logs, `build-info.json`, `artifacts.json`, `build-receipt.json`; `cached_*` entries under `--build-cache` | ACTIVE (authority) |
| `scripts/eval/build_measurement.sh` | user | same kinds (`--kind`, `--dry-run`) | from campaign_build.py | from campaign_build.py | `--output DIR` | ACTIVE (wrapper) |
| `scripts/eval/run_cycle_evaluation.py` | run_measurement.py performance, timeline | performance: `cycle-model`, `potal-host` (STRIPE_PIPELINE), `fullcpu-host` (FULL); timeline: none | none of its own: `--precision`, `--dim` select the `llama_plan()` profile | from campaign_build.py | `--build-cache/<kind>-<identity>`; per-build summary in `<run>/build/*.json` | ACTIVE |
| `scripts/eval/campaign.py` | run_measurement.py metric KIND, run_*_metrics.sh | `activation`, `residual`, `scu` (FULL); `cycle` only for its legacy adapter | none of its own: `--precision`, `--dim` select the `llama_plan()` profile | from campaign_build.py | `<output>/build` (fresh) or a verified `--prepared-build` | ACTIVE |
| `scripts/eval/metric_sweep.py` | run_measurement.py metric all | `activation`, `residual`, `scu` (FULL), one per precision x DIM, shared by both models | none of its own: `--precisions`, `--dims` select the `llama_plan()` profiles | from campaign_build.py | `--build-cache/<kind>-<identity>` (`cached_llama_build`), linked from `<sweep>/builds/`; passed to `campaign.py --prepared-build` | ACTIVE |
| `scripts/eval/cycle_trace_capture.py` | run_cycle_trace_capture.sh | `cycle` (STRIPE_PIPELINE) | none of its own | from campaign_build.py | `<config>/build` or `--prepared-build` | CERTIFICATION_ONLY |
| `scripts/eval/campaign_cycle.py` | campaign.py cycle | `cycle-model` | `CYCLE_MODEL_OPTIONS` | from campaign_build.py | `<output>/cycle-library-build` | CERTIFICATION_ONLY (legacy adapter) |
| build-arm64.sh, build-arm64-cpu.sh, build-arm64-fpga-uart.sh, build-x86.sh, build-riscv.sh | developer | none (developer builds, not a measurement kind) | own environment defaults resolved by `scripts/im2p-build-options.py`; they differ from every measurement profile (backend, DIM, bits, OpenMP, runtime matmul override) | host and toolchain specific (native flags, OpenMP, cross toolchain) | `build-arm64/`, `build-arm64-cpu/`, `build-arm64-fpga-uart/`, `build-x86/`, `build-riscv[-static]/`; no receipt | DEBUG/RESEARCH (never admitted by a runner) |
<!-- END GENERATED build-scripts -->

## Outputs

<!-- BEGIN GENERATED outputs -->
| Measurement | Default directory | Artifacts |
|---|---|---|
| performance | `runs/performance/<utc>-<model>-<precision>-d<dim>-hp1` | `manifest.json`<br>`performance.json`<br>`provenance.json`<br>`timing-provenance.json`<br>`build/`<br>replay/ (NPU results, dataset, lifecycle, IR, schedule.sqlite, schedule-performance.json)<br>timeline/ (only with --timeline compact)<br>`SHA256SUMS` |
| timeline | `runs/timeline/<utc>-<source run name>` | `manifest.json`<br>`timeline.jsonl`<br>`timeline-rows.json`<br>`export.json`<br>`SHA256SUMS` |
| activation | `runs/metrics/activation/<utc>-<model>-<precision>-d<dim>` | `manifest.json`<br>`request.json`<br>`evaluation_manifest.json`<br>`dataset_manifest.json`<br>`build/`<br>`collection/activation-quant-metrics.jsonl`<br>`raw.jsonl`<br>activation_metrics.json (= summary.json)<br>layer_activation_metrics.json (= layers.json)<br>`SHA256SUMS` |
| residual | `runs/metrics/residual/<utc>-<model>-<precision>-d<dim>` | `manifest.json`<br>`request.json`<br>`evaluation_manifest.json`<br>`dataset_manifest.json`<br>`build/`<br>`collection/residual-path-metrics.jsonl`<br>`raw.jsonl`<br>residual_metrics.json (= summary.json)<br>layer_residual_metrics.json (= layers.json)<br>`compact-shape-summary.json`<br>`SHA256SUMS` |
| scu | `runs/metrics/scu/<utc>-<model>-<precision>-d<dim>` | `manifest.json`<br>`request.json`<br>`evaluation_manifest.json`<br>`dataset_manifest.json`<br>`build/`<br>`collection/scale-alignment-metrics.jsonl.gz`<br>`raw.jsonl.gz`<br>scale_alignment_metrics.json (= summary.json)<br>layer_scale_metrics.json (= layers.json)<br>with `--scu-mode aggregate`: collection/scale-alignment-aggregate.jsonl and raw.jsonl instead of the gzip observations<br>`SHA256SUMS` |
<!-- END GENERATED outputs -->

- Every run directory is new (existing directories are refused) and ends with `SHA256SUMS` over its files.
- Every run names its domain: performance `manifest.json` markers `MEASUREMENT_DOMAIN=performance`,
  `HARDWARE_METRICS_RUN=NOT_RUN`; timeline export markers `MEASUREMENT_DOMAIN=timeline`, `TIMELINE_EXPORT_ONLY=PASS`,
  `PERFORMANCE_RUN=NOT_RUN`, `HARDWARE_METRICS_RUN=NOT_RUN`; metric `manifest.json` `measurement_domain=metric`
  with its `kind`.
- `performance.json` holds `ttft`, `tpot`, `e2e` (scheduled token-ready endpoints,
  TTFT, 127 TPOT intervals), `components` (CPU and NPU sums per clock domain, never added), `timing_model`,
  `timeline_checks`, `publication` and `execution` (fresh or reused collection, NPU task-cache counters, stage
  cache hits). Target E2E values stay null without clock, interface and target-host evidence.
- Metric runs keep their metric-specific file names and add the uniform hard-link names `summary.json`,
  `layers.json`, `request.json` and `raw.jsonl[.gz]`; `campaign_verify.py DIR` re-checks the manifest binding and
  `SHA256SUMS`. Residual adds `compact-shape-summary.json`; SCU raw observations are gzip only.

### Shared identity

The frozen model manifest (`model_manifest.py create|verify`) is the common identity of the model across domains:
performance records it in `provenance.json`/`performance.json` (`workload.model.manifest`) and a metric run in
`manifest.json` (`model_manifest`, with `--model-manifest`); a timeline export has the identity of its source run.
Both engines resolve the model through `model_manifest.model_entry`, which rejects a file that is not an artifact of
the manifest. Precision, DIM, block size and quantization are part of every build receipt.
Build identities are deliberately separate: each domain records its own `semantic_options_sha256` and binary
SHA-256, and nothing requires them to be equal (same model and configuration, different instrumentation build).

```bash
python3 scripts/eval/run_measurement.py identity PERFORMANCE_RUN TIMELINE_RUN METRIC_RUN [METRIC_RUN ...]
```

reads finished run directories of the three domains and prints, per run, the shared part (`model_sha256`,
`model_manifest_sha256`, `model_artifact`, `architecture`, `quantization`, `precision`, `dim`, `block_size`,
`dataset_sha256`) and the builds that run recorded. It exits 1 when a shared field differs between the runs; a field
a run did not record (a metric run without `--model-manifest`) is listed as unrecorded, not as a mismatch.

## Performance options

| Option | Meaning |
|---|---|
| `--validation-mode certified\|nano-local` | NPU cycle authority: reviewed certificate set (`--certificates`) or a host-local validation receipt (`--local-validation`, `NANO_LOCAL_VALIDATED`, never publication) |
| `--model`, `--model-manifest`, `--prompt-file` | HP1 GGUF model, its frozen manifest and the WikiText-2 test text (256 prompt tokens + 128 generated) |
| `--precision a4w4\|a8w8`, `--dim 16\|32\|64` | hardware profile `<precision>-d<dim>-hp1` |
| `--reuse-collection RUN` | reuse the completed PoTal/FullCPU collections of an earlier run: no llama build, no PMU preflight; CPU timing is that collection's measurement, not a new one |
| `--timeline none\|compact` | `none` (default): no timeline file and no row objects; `compact`: `timeline/timeline.jsonl` from the same scheduling pass |
| `--stage-cache DIR` | content-addressed outputs of worker-scenario, replay, join, lifecycle, IR, schedule, timeline; keyed by input hashes, parameters and the code of the stage and its upstream stages (default `<run>/stage-cache`) |
| `--npu-task-cache FILE\|none` | persistent per-work NPU answers keyed by the full model document, library and semantic options (default `<stage-cache>/npu-tasks.sqlite`) |
| `--clock-selection CLOCK.json` | validated operating clock: schedules on that clock and fills NPU ms |
| `--target-host-timing`, `--target-interface-cost` | publication gates; without them publication stays `NOT_READY` |
| `--smoke` | `SMOKE_256P1` connectivity workload (256 + 1 token); TPOT is `NOT_APPLICABLE` |
| `--stop-after STAGE` | end after an offline stage (`STOPPED_AFTER_<STAGE>`) |
| `--keep-raw` | keep cycle logs, traces, IR and schedule in the run directory (the stage cache keeps IR and schedule either way) |
| `--build-cache`, `--jobs`, `--replay-workers`, `--storage-factor`, `--timeout`, `--reconstruct-timeout` | build cache location and execution bounds |
| `--dry-run [--system S --machine M]` | print the platform and the build plan; build nothing |

Timeline-only options are `--from-run RUN` and `--format jsonl`.

Validation modes: `nano-local` admits the cycle library named by a `sim.cycle.local_validation` receipt (library
bytes, compiled source closure and authority sources are re-checked; a changed C++ closure fails) and labels every
output `NANO_LOCAL_VALIDATED`, `publication_certified=false`. `certified` builds the cycle model, requires the fresh
library to equal the CURRENT library of the certificate set (`im2p-evaluation-certificate-set` v1: library SHA-256
plus base, run-aware and transition certificates) and reconstructs through `end_to_end.py reconstruct`. Neither
mode publishes target latency: `target_admission.py` keeps publication closed until the operating-clock,
target-host, target-interface, workload-identity and reconstruction gates all pass.

Deprecations and removals:

| Item | Status | Replacement |
|---|---|---|
| `run_cycle_evaluation.py --run metrics`, `--metric` | removed: `--run` accepts `performance` and `timeline` only | `run_measurement.py metric activation\|residual\|scu` for metrics |
| marker `METRICS_AUTOMATION_RUN` in performance and timeline manifests | no longer written; old runs keep it | `MEASUREMENT_DOMAIN`, `HARDWARE_METRICS_RUN` |
| `--prompt-tokens`, `--generate` | accepted only as 256 / 128; proposed for removal | the recipe is fixed (`E2E_GENERATION_256_128`, `--smoke`) |
| `--timeline full` (internal NPU event log) | not implemented | certification-only event replay (`cycle_campaign.py`) |
| `campaign.py cycle` | legacy certified replay adapter, not a metric | `run_cycle_campaign.sh` (certification), performance runs (evaluation) |
| `end_to_end.py reconstruct --streaming-ir` | alias of the default | omit |

## Timeline

`e2e_timeline.visit_node` is the single interpretation of a stored schedule row; the in-pass writer
(`--timeline compact`) and the export (`--run timeline`) both use its row builder, so they produce the same bytes.
Row fields:

- identity: `seq`, `row_type`, `kind` (`cpu`, `npu`, `event`), `node_id`, `operation_id`, `work_id`, `op`, `layer`,
  `phase`, `decode_index`, `token_index`, `event` (`request_start`, `token_ready`)
- placement: `start_cycle`, `end_cycle`, `duration_cycles` on the schedule axis; `start_ns`, `end_ns`,
  `duration_ns`, `npu_ms` only under a validated operating clock
- resources: `resource`, `resource_class`, `scheduler_lane` (a synthetic scheduler lane, never a CPU core),
  `host_thread_id`/`tid`, `host_cpu_core_start`, `host_cpu_core_end`, `host_cpu_core` (set only when the interval
  did not migrate), `cpu_migrated`, `target_cpu_core` (null: no target mapping exists)
- costs: `cpu_cycles` (+ `cpu_cycles_valid`, `cpu_cycle_source`, `cpu_cycle_scope`), `host_elapsed_ns`,
  `host_thread_cpu_ns`, `npu_cycles`, `target_cpu_cycles`/`target_cpu_ms` (null)
- provenance: `timing_source`, `evidence_id`, `source_line`, `worker_id`

CPU PMU cycles and NPU cycles are different clock domains and are never summed. Overlap and idle gaps are kept
exactly where the scheduler placed them.

## Script classification

`CANONICAL` user entry points, `WRAPPER` thin aliases, `INTERNAL` modules, `LEGACY` kept for old evidence,
`CERTIFICATION_ONLY` reviewed-certificate tooling, `DEBUG/RESEARCH` developer helpers. Nothing was removed.

<!-- BEGIN GENERATED scripts -->
| Script | Status | Role | Called by |
|---|---|---|---|
| `run_measurement.py` | CANONICAL | user entry point of the three domains; dispatch only | user |
| `run_cycle_evaluation.py` | CANONICAL | performance and timeline export engine (`--run`) | run_measurement.py |
| `campaign.py` | CANONICAL | metric engine: `activation`, `residual`, `scu` (its `cycle` kind is the legacy stateful replay adapter) | run_measurement.py, run_*_metrics.sh |
| `run_activation_metrics.sh` | WRAPPER | `campaign.py activation` | user (kept) |
| `run_residual_metrics.sh` | WRAPPER | `campaign.py residual` | user (kept) |
| `run_scu_metrics.sh` | WRAPPER | `campaign.py scu` | user (kept) |
| `build_measurement.sh` | WRAPPER | `campaign_build.py` CLI (separately verifiable build, `--dry-run`) | user |
| `activation_quant_metrics.py` | WRAPPER | standalone activation collect/reduce with a prepared runner | user (advanced) |
| `residual_path_metrics.py` | WRAPPER | standalone residual collect/reduce with a prepared runner | user (advanced) |
| `scale_alignment_metrics.py` | WRAPPER | standalone SCU collect/reduce with a prepared runner | user (advanced) |
| `measurement_domains.py` | INTERNAL | domain definitions for the wrapper, README tables and tests | run_measurement.py |
| `metric_sweep.py` | INTERNAL | `metric all`: unchanged `campaign.py` runs over models x precisions x DIMs (one build per metric/precision/DIM, activation as workload anchor), identity checks and the aggregate JSON/CSV/tables; computes no metric | run_measurement.py |
| `metric_table.py` | INTERNAL | deterministic stdout tables of metric summaries (display only) | metric_sweep.py, campaign.py |
| `measurement_identity.py` | INTERNAL | read-only shared identity of finished runs (`run_measurement.py identity RUN...`) | run_measurement.py |
| `campaign_build.py` | INTERNAL | the build authority: every CMake option of every measurement build, receipts, build cache | run_cycle_evaluation.py, campaign.py, cycle scripts |
| `metric_run.py` | INTERNAL | native metric collection with sink isolation; body of the standalone reducers | campaign.py, *_metrics.py |
| `campaign_metrics.py` | INTERNAL | aggregate and per-layer metric outputs | campaign.py |
| `campaign_inputs.py` | INTERNAL | model/tokenizer/dataset identity, SHA256SUMS | campaign.py, cycle scripts |
| `campaign_stream.py` | INTERNAL | streaming gzip sink for SCU observations | metric_run.py |
| `campaign_verify.py` | INTERNAL | verifier CLI for a finished metric run (manifest binding, checksums) | user |
| `e2e_timeline.py` | INTERNAL | the one timeline row builder and performance accumulator | run_cycle_evaluation.py, schedule_engine.py |
| `schedule_engine.py` | INTERNAL | one scheduling pass with performance and timeline sinks | run_cycle_evaluation.py |
| `stage_cache.py` | INTERNAL | content-addressed offline stage cache | run_cycle_evaluation.py |
| `model_manifest.py` | INTERNAL | frozen model manifest (create/verify CLI, `model_entry`) | run_cycle_evaluation.py, campaign.py |
| `evaluation_host.py` | INTERNAL | read-only host observations | run_cycle_evaluation.py, e2e_run.py |
| `pmu_probe.c` | INTERNAL | PMU readiness probe compiled by the performance preflight | run_cycle_evaluation.py |
| `eval_common.py` | INTERNAL | shared helpers (certificate-pinned) | all |
| `end_to_end.py` | INTERNAL | native collection (`collect`) and certified reconstruction (`reconstruct`) CLI | run_cycle_evaluation.py |
| `e2e_run.py` | INTERNAL | native collection logic | end_to_end.py |
| `run_stateful_integration_smoke.py` | INTERNAL | 256+1 smoke pair collector (`--smoke`) | run_cycle_evaluation.py |
| `paired_inputs.py` | INTERNAL | PoTal/FullCPU pairing inputs | e2e_run.py |
| `application_results.py` | INTERNAL | application endpoint results (certificate-pinned) | end_to_end.py, offline_pipeline.py |
| `target_admission.py` | INTERNAL | fail-closed publication gates (clock, target host, interface) | run_cycle_evaluation.py |
| `metric_reducers.py` | LEGACY | unbound ACT/RES reducers for historical streams (`--reduce` without a manifest) | metric_run.py |
| `offline_pipeline.py` | CERTIFICATION_ONLY | certified offline replay/join/IR (`--validation-mode certified`) | end_to_end.py |
| `certified_reconstruction.py` | CERTIFICATION_ONLY | certified reconstruction results | offline_pipeline.py |
| `scheduled_endpoints.py` | CERTIFICATION_ONLY | scheduled endpoints of certified schedules | certified_reconstruction.py |
| `e2e_cost_ownership.py` | CERTIFICATION_ONLY | cost-ownership and interface contracts | target_admission.py |
| `e2e_integration_prep.py` | CERTIFICATION_ONLY | stateful host+NPU integration prep on a certified trace | user |
| `campaign_cycle.py` | CERTIFICATION_ONLY | stateful provider replay behind `campaign.py cycle` | campaign.py, cycle_campaign.py |
| `cycle_campaign.py` | CERTIFICATION_ONLY | certified trace cycle accounting | run_cycle_campaign.sh |
| `run_cycle_campaign.sh` | CERTIFICATION_ONLY | `cycle_campaign.py` | user |
| `cycle_trace_capture.py` | CERTIFICATION_ONLY | actual-inference trace capture and certificate | run_cycle_trace_capture.sh |
| `run_cycle_trace_capture.sh` | CERTIFICATION_ONLY | `cycle_trace_capture.py` | user |
| `cycle_campaign_bundle.py` | CERTIFICATION_ONLY | evaluation-cycle campaign bundle | user |
| `actual_cycle_bundle.py` | CERTIFICATION_ONLY | actual-cycle campaign bundle | user |
| `cycle_accounting.py` | CERTIFICATION_ONLY | cycle accounting definitions | cycle_campaign.py |
| build-arm64.sh, build-arm64-cpu.sh, build-arm64-fpga-uart.sh, build-x86.sh, build-riscv.sh | DEBUG/RESEARCH | developer builds with their own option defaults; never a measurement build | developer |
| scripts/experiment/, scripts/utils/ | DEBUG/RESEARCH | cycle matrix experiments and log rendering helpers | developer |
| evaluation/ (python -m evaluation) | INTERNAL | ACT/RES/SCU reducers, schemas and manifest type | campaign_metrics.py, metric_run.py |
<!-- END GENERATED scripts -->

```text
run_measurement.py
 |- performance | timeline -> run_cycle_evaluation.py
 |     |- campaign_build.py            builds, receipts, build cache          (performance)
 |     |- model_manifest.py, evaluation_host.py, pmu_probe.c                  (identity, host, PMU preflight)
 |     |- end_to_end.py collect -> e2e_run.py | run_stateful_integration_smoke.py (--smoke)
 |     |- IM2P.sim sim.cycle.local_replay, nano_local_execution               (nano-local replay, join, lifecycle, IR)
 |     |     or end_to_end.py reconstruct -> offline_pipeline.py              (certified)
 |     |- stage_cache.py, schedule_engine.py -> e2e_timeline.py -> IM2P scheduler_sqlite
 |     `- target_admission.py          publication gates
 |- metric activation | residual | scu -> campaign.py
 |     |- campaign_build.py            metric build (one sink)
 |     |- campaign_inputs.py, model_manifest.py
 |     |- metric_run.py -> llama-eval-workload METRIC_PREFILL_256, campaign_stream.py (SCU gzip)
 |     `- campaign_metrics.py -> evaluation/ reducers
 |- metric all -> metric_sweep.py      campaign.py per model x precision x DIM; reads and aggregates the runs
 |     |- campaign_build.py            cached_llama_build: one build per metric/precision/DIM for both models
 |     |- campaign_verify.py, measurement_identity.py   resume verification, shared identity
 |     `- metric_table.py              stdout tables (also campaign.py's one-row table)
 |- identity RUN... -> measurement_identity.py   reads finished runs of any domain; measures nothing
 `- describe -> measurement_domains.py           the definitions behind this file's generated tables
```

The two engines share only `campaign_build.py`, `model_manifest.py` and `eval_common.py`; the metric engine imports no
replay, scheduler, timeline or stage-cache module and the performance engine imports no metric collector or reducer
(`tests/test-measurement-domains.py` checks both import closures).

Removal candidates for a later change (not done here): `metric_reducers.py` once no unbound historical stream
needs `--reduce`; the `cycle` kind of `campaign.py` together with `campaign_cycle.py` (superseded by
`cycle_campaign.py`).

## Verification

```sh
python3 -B -m pytest -q tests/test-measurement-domains.py tests/test-metric-sweep.py tests/test-cycle-evaluation.py
python3 -B -m evaluation.tests.test_campaign && python3 -B -m evaluation.tests.test_framework
python3 -B scripts/eval/run_measurement.py describe --update-readme
```

## Reference

Metric campaigns, cycle trace capture and certificates are described in `CAMPAIGNS.md`; the reducers and metric
definitions in `evaluation/README.md`. No Python tokenizer, NPU timing equation, new numerical executor or added
package dependency is used. `llama-eval-workload --build-info` reports the compiled metric flags and target recipe;
scripts bind the executable and input hashes before and after collection, and runtime paths never enable
compiled-out hooks.

### Standalone activation and residual collection (prepared runner)

Use separate build artifacts with ACT/RES flags `1/0` and `0/1`, respectively.
Both scripts require the observed production EXSIA block32 route and an explicit
CPU-functional `CYCLE_SIM=1`, `IM2P_SIM`, HP1 artifact. Device/RTL metric collection
is rejected, not implicitly selected. RES additionally
requires enabled HP1 WS run-aware residual work; a CPU fallback is not silently
reported as a residual-free invocation. CPU-functional progression can be selected
explicitly with `CYCLE_SIM=1`; metric flags do not enable it.

```sh
python3 -B scripts/eval/activation_quant_metrics.py --help
python3 -B scripts/eval/residual_path_metrics.py --help

python3 -B scripts/eval/activation_quant_metrics.py \
  --runner /absolute/act-build/bin/llama-eval-workload \
  --model models/gpt2/gpt2.Q8_HP1.gguf \
  --dataset wikitext-2-raw/wiki.test.raw --split test \
  --max-chunks 1 --output /absolute/fresh-activation-output

python3 -B scripts/eval/residual_path_metrics.py \
  --runner /absolute/res-build/bin/llama-eval-workload \
  --model models/gpt2/gpt2.Q8_HP1.gguf \
  --dataset wikitext-2-raw/wiki.test.raw --split test \
  --max-chunks 1 --output /absolute/fresh-residual-output \
  --workload-manifest /absolute/fresh-activation-output/workload-binding.json
```

`--max-chunks 0` requests all complete native chunks; a positive value is a
bounded smoke subset. Workload binding compares native input token IDs, offsets,
BOS, output mask, batch/ubatch and thread settings. Statistics retain the native
perplexity second-half output mask: lm_head can have fewer than 256 rows.

ACT files are `activation-quant-metrics.jsonl` (`counts.jsonl` alias) and
`activation-quant-summary.json` (`summary.json` alias). Integer sufficient counts
are summed before division. Undefined denominators remain null. The current
user-confirmed reference uses signed original FP, separately for each logical row
and original BK32 block, actual tail coordinates only, population divisor N,
and strict `x > mean + 2σ`. Revision:
`signed-row-original-bk32-population-2sigma-v1`. Nonfinite input leaves
F/intersection/union null and blocks reference publication. Historical candidate
streams remain diagnostic; a marker does not upgrade their definition.

RES files are `residual-path-metrics.jsonl` (`shapes.jsonl` alias) and
`residual-path-summary.json` (`summary.json` alias). Every main stripe contributes
to D, including stripes without residual work. Compact rectangles produce T/R
without DIM padding or nonzero-payload discount. `proposed-v3-DTR-v1` ratios remain
explicitly proposed; `--accept-proposed-weighting` records an explicit choice, not
an assertion of historical equivalence. In particular, `retained_K_factor=R/T` is
not silently named the research scripts' hypothetical K-first `weighted_k_retention`.

`--reduce existing.jsonl --output fresh-directory` validates and reduces a
completed dedicated stream without a model or other metric's dependencies.
This is `RAW_REDUCTION_ONLY`, not a new native collection claim. Missing RUN_END,
duplicate invocation/stripe, malformed run coverage and failed streams do not
publish a summary. Existing directories/files are never overwritten.

### Native collection and certified offline reconstruction (`end_to_end.py`)

```sh
python3 -B scripts/eval/end_to_end.py --help
python3 -B scripts/eval/end_to_end.py collect --help
python3 -B scripts/eval/end_to_end.py reconstruct --help

python3 -B scripts/eval/end_to_end.py --output /absolute/fresh-e2e-output collect \
  --runner /absolute/metrics-off-build/bin/llama-eval-workload \
  --model models/gpt2/gpt2.Q8_HP1.gguf \
  --dataset wikitext-2-raw/wiki.test.raw --settings /absolute/evaluation-settings.json \
  --role fullcpu --im2p ../IM2P.sim --repetitions 1 --timeout 600
```

Current settings use schema `potal-evaluation-settings`, version 2:

```json
{
  "schema": "potal-evaluation-settings", "version": 2,
  "seed": 1234, "temperature": 0, "top_k": 0, "top_p": 1, "min_p": 0,
  "repeat_penalty": 1, "repeat_last_n": 0, "grammar": null,
  "eos_stopping": false, "warmup": 0,
  "sampler_policy": "user-confirmed-greedy-v3", "split": "test",
  "chunk_ids": [0,1,2,3,4,5,6,7,8,9],
  "threads": 1, "threads_batch": 1, "batch_size": 256, "ubatch_size": 256,
  "scope": "DEVELOPMENT_SMOKE_NOT_PAPER_CAMPAIGN"
}
```

Ten repetitions use native test chunks 0..9 once each, not chunk 0 ten times.
`--repetitions 1` uses only chunk 0. EOS does not stop generation and is not
logit-suppressed. Missing chunks, changed sampling settings or prompt replacement
fail. Optional `expected_host_id` enforces a recorded same-host identity.

Roles are `fullcpu`, `fullcpu-cost-only`, `potal`, and `cuda`. All reject stats-enabled artifacts.
FullCPU/PoTal require existing CPU logging. CUDA requests maximum offload and
requires actual native `offloaded X/Y layers to GPU` proof with X=Y>0;
partial placement, OOM and absent CUDA capability fail without fallback.
CUDA Q4_0/Q8_0 are practical references, not arithmetic-matched A4W4/A8W8 baselines.
The current generic GPU loader line proves only `LAYER_COUNT_ONLY`: it does not
identify the actual CUDA backend, tensor placement, or fallback coverage.
Direct application measurements remain observations, and verified campaign
aggregation rejects this incomplete proof. Detailed placement instrumentation
is an implementation gap separate from unavailable Jetson/CUDA hardware.

Each repetition starts a real process, uses 256 native prompt tokens, and requires
128 sample/accept completions and exactly 127 decode calls. Short runs fail.
No warmup or first-run discard occurs. Native endpoints exclude terminal I/O.
Both instrumented FullCPU and PoTal observed collection durations are explicitly
**not** target latency. Their per-operation/canonical service tables are the cost
sources; synchronous trace serialization can contaminate collection wall time.
Direct CUDA timing requires all CPU/cycle instrumentation OFF.

Collect PoTal first, then obtain ordinary CPU costs on its exact generated IDs:

```sh
python3 -B scripts/eval/end_to_end.py --output /absolute/fresh-fullcpu-cost-output collect \
  --runner /absolute/fullcpu-build/bin/llama-eval-workload --im2p ../IM2P.sim \
  --model models/gpt2/gpt2.Q8_HP1.gguf --dataset wikitext-2-raw/wiki.test.raw \
  --settings /absolute/evaluation-settings.json --role fullcpu-cost-only \
  --paired-potal /absolute/completed-potal-output --repetitions 1 --timeout 600
```

The native runner consumes a hash-bound 128-ID array directly. This performs
127 decode calls but zero actual sampler calls, is labeled `FORCED_CPU_COST_ONLY`,
and never claims free-generation TTFT/TPOT. Official joining permits only this
declared argument difference, with PoTal endpoint/provenance/token bindings,
prompt/chunk/model/kernel equality and actual phase-input fingerprint equality.

`aggregate --result ...` requires ten unique measurement and endpoint identities,
matching host/configuration, and computes median(run TTFT) and median(run mean
TPOT), using rational nanoseconds. Replaying one NPU artifact ten times does not
provide ten host measurements.

`reconstruct` invokes the existing official IM2P replay, certificate admission,
three-source join and execution-IR adapter. It requires `--application` for the
measured PoTal sampling sidecar and explicit lifecycle
semantics and collection provenance, never invents missing edges. Optional
`--diagnostic-phase-table` plus `--diagnostic-frequency-hz` invokes only the
synthetic scheduler. Without a validated clock and current service-boundary
certificate, `result.json` lists missing certified inputs, TTFT/TPOT remain null,
and no `reconstructed-result.json` is written.

This is the complete diagnostic reconstruction argv for one bound repetition;
replace each `/absolute/...` input with its existing artifact. SQLite IR and
schedule are the default, so no storage flag is needed:

```sh
python3 -B scripts/eval/end_to_end.py --output /absolute/fresh-reconstruction-output reconstruct \
  --im2p ../IM2P.sim \
  --full-cpu-log /absolute/fullcpu/cpu-log.jsonl \
  --full-cpu-graph /absolute/fullcpu/semantic-graph.json \
  --full-cpu-provenance /absolute/fullcpu/provenance.json \
  --potal-log /absolute/potal/cpu-log.jsonl \
  --potal-graph /absolute/potal/semantic-graph.json \
  --potal-provenance /absolute/potal/provenance.json \
  --npu-trace /absolute/potal/npu-trace.jsonl \
  --library /absolute/im2p/libim2p_cycle.so \
  --cycle-certificate /absolute/im2p/cycle-certificate.json \
  --run-aware-certificate /absolute/im2p/run-aware-certificate.json \
  --application /absolute/potal/application.jsonl \
  --lifecycle /absolute/potal/execution-lifecycle.json \
  --diagnostic-phase-table /absolute/im2p/phase-table.json \
  --diagnostic-frequency-hz 1000000000 --timeout 600
```

`--streaming-ir` remains an alias for the default. `--json-ir` selects the
small/debug JSON bundle and schedule; it rejects more than 64 MiB of workload
inputs before replay. Both modes preflight free space for a 256 MiB reserve plus
eight times the workload input bytes. A timeout or failed command records its
boundary in `failure.json` and does not publish a normal `result.json`.

#### Stateful session options and current boundary

The actual `end_to_end.py reconstruct --help` surface exposes
`--stateful-sequence-certificate` as mutually exclusive with
`--service-certificate`, plus `--stateful-evidence-root` and
`--stateful-diagnostic`. Certificate and evidence root are required together.
Stateful diagnostic mode also requires `--diagnostic-frequency-hz`, forbids
`--clock-selection`, and is a configured test-clock schedule only. The existing
`--cycle-certificate`, `--run-aware-certificate`, `--timing`, `--profile`,
`--initial-scratchpad-half`, and `--initial-accumulator-half` inputs remain
required by the underlying stateful provider. There is no `--all-profiles` or
implicit stateful fallback option.

The lower-level `execution_cli schedule` and `verify-schedule` surfaces use the
same `--stateful-sequence-certificate`, `--stateful-evidence-root`, and
`--stateful-diagnostic` options. They open one native session for the ordered
trace, preserve request-availability, electrical-offer, accepted, result-ready,
final-scale-release, and resource-ready epochs, and reject changed work/source
bindings. Verification starts a fresh provider/session and recomputes JSON or
SQLite results. SQLite remains the official wrapper default; `--json-ir` is the
explicit small/debug path.

Production stateful admission is typed and source-bound; a scope string or an
exact-v2 drained certificate is insufficient. The official wrapper currently
rejects stateful target result publication because validated
target-host/application admission is unavailable. `--stateful-diagnostic`
therefore produces only `STATEFUL_DIAGNOSTIC` schedule evidence and never
TTFT/TPOT. Current NPU parent certificates do not replace target-host evidence.

Current evidence is intentionally narrower: the actual GPT-2 run validated 3
works from a 16-work subset. Work 3 reached tag peak 5, above the reviewed limit
4, while row peak was 4. The first cold 374-work attempt also stopped after
3 validated works at that boundary, without publishing a result; the second
full attempt is `NOT_RUN`. The latest
exploratory RTL attempt reached the unchanged 128 MiB queue-edge limit and
produced no parity verdict. None of these observations authorizes a wider
stateful domain or a completion marker.

The separate diagnostic `sim.cycle.sequence_trace_cli watchdog` treats atomic
publication of a validated `result.json` as its completion boundary. A fresh
verifier checks `pending-result.json`; the publisher checks its inode, digest,
and deadline before hard-linking it to the final name. The pending file remains
available as evidence. Receipt emission or cleanup failures after publication
are reported separately and do not revoke the published result. A receipt
write failure is reported as JSON on stderr. Failures before atomic publication
still leave the run incomplete; this contract does not authorize the official
wrapper to publish production latency.

The current IM2P drained v1 certificate covers fixed two-work RTL fixtures only;
it yields `DRAINED_FIXTURE_PARITY`, not production sequence admission. Supplying
that certificate as `--service-certificate` still fails closed until a
producer-generated same-instance sequence/phase certificate exists.
Rejected certified inputs leave `NOT_READY_CERTIFICATION_REJECTED` scope status
and a nonzero CLI exit. The diagnostic phase table cannot be combined with a
complete certified publication request.

For one certified PoTal repetition, supply `--service-certificate`,
`--clock-selection`, `--profile`, `--timing`,
`--initial-scratchpad-half`, `--initial-accumulator-half`, and
`--potal-result` pointing at that repetition's native collection `result.json`,
in addition to the existing replay, join, lifecycle, and `--application`
arguments. The timing file and initial halves are explicit service scenario
inputs; the NPU frequency comes only from the validated operating-clock file.
The official scheduler and `verify-schedule` command check source, service,
scenario, and clock bindings before publication. The standalone aggregate loader
repeats that verification. `reconstructed-result.json` requires source-bound
prefill preparation, uses scheduled `application:request:begin` before that
preparation as t0, all 128 scheduled sample completions, and exact rational
nanoseconds for TTFT and TPOT. Legacy sampler-only sidecars cannot publish.
Ten distinct validated results are still
needed for the median-of-ten aggregate. A single reconstructed run is neither
a Jetson measurement nor a full paper campaign or quality PPL claim.

`--lifecycle-sidecar` plus explicit `--worker-resources`, `--cpu-policy`, and
`--sampler-resource` invokes the official producer-bound lifecycle builder after
joining. Adding both
`--diagnostic-phase-table /absolute/phase-table.json` and
`--diagnostic-frequency-hz 1000000000` explicitly requests a synthetic schedule
in `schedule.sqlite`; the numeric frequency is a diagnostic scenario, not a
selected hardware clock. `--json-ir` writes `schedule.json` for small diagnostic
inputs. Neither path publishes target TTFT/TPOT.
