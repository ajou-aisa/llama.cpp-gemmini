# Measurement scripts

Run independent activation, residual and scale measurements with `run_measurement.py metric`.
Each build uses internal NPU headers and an explicit REFERENCE transport.
Metrics run the complete numerical forward and materialize logits.

`python3 -B scripts/eval/run_measurement.py describe` prints commands and build identities.
`python3 -B scripts/eval/run_measurement.py identity RUN [RUN ...]` compares finished metric runs.
Native CPU and CUDA application observations remain available through `end_to_end.py collect`.

## commands

<!-- BEGIN GENERATED commands -->
### Activation metrics

```bash
python3 scripts/eval/run_measurement.py metric activation \
  --model gpt2 --precision a8w8 --dim 32 [--max-chunks 1] [--model-manifest MANIFEST.json] [--output DIR]
```

- Builds: fresh `activation` build in `<output>/build` (`GGML_GEMMINI_ACT_METRICS=1`, the other two metric sinks 0); `--prepared-build DIR` reuses a verified `build_measurement.sh` build.
- Measures: activation quantization and outlier statistics per layer and in aggregate (signed-row BK32 2-sigma reference, selection overlap, requantization ratio).
- Does not run: TTFT/TPOT, NPU cycle replay, scheduling, timeline; the other two metric sinks.
- Inputs: HP1 GGUF model (`--model gpt2|llama3.2-1B` or `--model-path`), WikiText-2 test text (`--dataset-manifest`), internal Gemmini SDK.
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
- Inputs: HP1 GGUF model (`--model gpt2|llama3.2-1B` or `--model-path`), WikiText-2 test text (`--dataset-manifest`), internal Gemmini SDK.
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
- Inputs: HP1 GGUF model (`--model gpt2|llama3.2-1B` or `--model-path`), WikiText-2 test text (`--dataset-manifest`), internal Gemmini SDK.
- Cache / reuse: `--prepared-build`, `--evaluation-manifest`, `--workload-manifest` (same native chunks across metrics).
- Outputs (default `runs/metrics/scu/<utc>-<model>-<precision>-d<dim>`): manifest.json, request.json, evaluation_manifest.json, dataset_manifest.json, build/, collection/scale-alignment-metrics.jsonl.gz, raw.jsonl.gz, scale_alignment_metrics.json (= summary.json), layer_scale_metrics.json (= layers.json), with `--scu-mode aggregate`: collection/scale-alignment-aggregate.jsonl and raw.jsonl instead of the gzip observations, SHA256SUMS.
<!-- END GENERATED commands -->

## matrix

<!-- BEGIN GENERATED matrix -->
| Measurement | Needs cycle model? | Needs PoTal? | Needs FullCPU? | Needs PMU? | Needs scheduler? | Needs NPU clock? | Needs WikiText? | Needs metric instrumentation? | Produces timeline? |
|---|---|---|---|---|---|---|---|---|---|
| activation | no | no | no | no | no | no | yes | yes | no |
| residual | no | no | no | no | no | no | yes | yes | no |
| scu | no | no | no | no | no | no | yes | yes | no |
<!-- END GENERATED matrix -->

## builds

<!-- BEGIN GENERATED builds -->
| Measurement | Build kind | Key CMake flags (a8w8, DIM 32) | Binary |
|---|---|---|---|
| activation | `activation` | `GGML_GEMMINI_ACT_METRICS=1`<br>`GGML_GEMMINI_RESIDUAL_METRICS=0`<br>`GGML_GEMMINI_SCALE_METRICS=0`<br>`LOG_CYCLE=0`<br>`GGML_CPU_CYCLE_LOG=OFF`<br>`GGML_GEMMINI_OPTION=CPU`<br>`GGML_GEMMINI_EXECUTION_BACKEND=HARDWARE`<br>`GGML_GEMMINI_DEFAULT_MATMUL_MODE=FULL` | llama-eval-workload |
| residual | `residual` | `GGML_GEMMINI_ACT_METRICS=0`<br>`GGML_GEMMINI_RESIDUAL_METRICS=1`<br>`GGML_GEMMINI_SCALE_METRICS=0`<br>`LOG_CYCLE=0`<br>`GGML_CPU_CYCLE_LOG=OFF`<br>`GGML_GEMMINI_OPTION=CPU`<br>`GGML_GEMMINI_EXECUTION_BACKEND=HARDWARE`<br>`GGML_GEMMINI_DEFAULT_MATMUL_MODE=FULL` | llama-eval-workload |
| scu | `scu` | `GGML_GEMMINI_ACT_METRICS=0`<br>`GGML_GEMMINI_RESIDUAL_METRICS=0`<br>`GGML_GEMMINI_SCALE_METRICS=1`<br>`LOG_CYCLE=0`<br>`GGML_CPU_CYCLE_LOG=OFF`<br>`GGML_GEMMINI_OPTION=CPU`<br>`GGML_GEMMINI_EXECUTION_BACKEND=HARDWARE`<br>`GGML_GEMMINI_DEFAULT_MATMUL_MODE=FULL` | llama-eval-workload |
<!-- END GENERATED builds -->

## build-scripts

<!-- BEGIN GENERATED build-scripts -->
| Script | Caller | Build kinds | Semantic options | Platform options | Output | Status |
|---|---|---|---|---|---|---|
| `scripts/eval/campaign_build.py` | all measurement scripts; CLI through build_measurement.sh | `activation`, `residual`, `scu`, `metrics-all` (all three sinks, `metric all`), `potal-host`, `potal-host-nocpulog`, `fullcpu-host` | defined here and nowhere else: `llama_plan()`, `METRIC_SINKS` | `platform_profile()`: library suffix, binary format, install name; adds no CMake option today and may never add a semantic one (`with_platform`) | build directory with configure/build/verify logs, `build-info.json`, `artifacts.json`, `build-receipt.json`; `cached_*` entries under `--build-cache` | ACTIVE (authority) |
| `scripts/eval/build_measurement.sh` | user | same kinds (`--kind`, `--dry-run`) | from campaign_build.py | from campaign_build.py | `--output DIR` | ACTIVE (wrapper) |
| `scripts/eval/campaign.py` | run_measurement.py metric KIND, run_*_metrics.sh | `activation`, `residual`, `scu` (FULL); `cycle` only for its legacy adapter | none of its own: `--precision`, `--dim` select the `llama_plan()` profile | from campaign_build.py | `<output>/build` (fresh) or a verified `--prepared-build` | ACTIVE |
| `scripts/eval/metric_sweep.py` | run_measurement.py metric all | `metrics-all` (FULL), one per precision x DIM shared by both models; with `--collection separate` `activation`, `residual`, `scu` each | none of its own: `--precisions`, `--dims` select the `llama_plan()` profiles | from campaign_build.py | `--build-cache/<kind>-<identity>` (`cached_llama_build`), linked from `<sweep>/builds/`; passed to `campaign.py --prepared-build` | ACTIVE |
| build-arm64.sh, build-arm64-cpu.sh, build-x86.sh, build-riscv.sh | developer | none (developer builds, not a measurement kind) | environment build defaults resolved by cmake/npu-config.cmake | host and toolchain specific (native flags, OpenMP, cross toolchain) | `build-arm64/`, `build-arm64-cpu/`, `build-x86/`, `build-riscv[-static]/`; no receipt | DEBUG/RESEARCH (never admitted by a runner) |
<!-- END GENERATED build-scripts -->

## outputs

<!-- BEGIN GENERATED outputs -->
| Measurement | Default directory | Artifacts |
|---|---|---|
| activation | `runs/metrics/activation/<utc>-<model>-<precision>-d<dim>` | `manifest.json`<br>`request.json`<br>`evaluation_manifest.json`<br>`dataset_manifest.json`<br>`build/`<br>`collection/activation-quant-metrics.jsonl`<br>`raw.jsonl`<br>activation_metrics.json (= summary.json)<br>layer_activation_metrics.json (= layers.json)<br>`SHA256SUMS` |
| residual | `runs/metrics/residual/<utc>-<model>-<precision>-d<dim>` | `manifest.json`<br>`request.json`<br>`evaluation_manifest.json`<br>`dataset_manifest.json`<br>`build/`<br>`collection/residual-path-metrics.jsonl`<br>`raw.jsonl`<br>residual_metrics.json (= summary.json)<br>layer_residual_metrics.json (= layers.json)<br>`compact-shape-summary.json`<br>`SHA256SUMS` |
| scu | `runs/metrics/scu/<utc>-<model>-<precision>-d<dim>` | `manifest.json`<br>`request.json`<br>`evaluation_manifest.json`<br>`dataset_manifest.json`<br>`build/`<br>`collection/scale-alignment-metrics.jsonl.gz`<br>`raw.jsonl.gz`<br>scale_alignment_metrics.json (= summary.json)<br>layer_scale_metrics.json (= layers.json)<br>with `--scu-mode aggregate`: collection/scale-alignment-aggregate.jsonl and raw.jsonl instead of the gzip observations<br>`SHA256SUMS` |
<!-- END GENERATED outputs -->

## scripts

<!-- BEGIN GENERATED scripts -->
| Script | Status | Role | Called by |
|---|---|---|---|
| `run_measurement.py` | CANONICAL | user entry point of the metric domains; dispatch only | user |
| `campaign.py` | CANONICAL | metric engine: `activation`, `residual`, `scu` (independent metric sinks) | run_measurement.py, run_*_metrics.sh |
| `run_activation_metrics.sh` | WRAPPER | `campaign.py activation` | user (kept) |
| `run_residual_metrics.sh` | WRAPPER | `campaign.py residual` | user (kept) |
| `run_scu_metrics.sh` | WRAPPER | `campaign.py scu` | user (kept) |
| `build_measurement.sh` | WRAPPER | `campaign_build.py` CLI (separately verifiable build, `--dry-run`) | user |
| `activation_quant_metrics.py` | WRAPPER | standalone activation collect/reduce with a prepared runner | user (advanced) |
| `residual_path_metrics.py` | WRAPPER | standalone residual collect/reduce with a prepared runner | user (advanced) |
| `scale_alignment_metrics.py` | WRAPPER | standalone SCU collect/reduce with a prepared runner | user (advanced) |
| `measurement_domains.py` | INTERNAL | domain definitions for the wrapper, README tables and tests | run_measurement.py |
| `metric_sweep.py` | INTERNAL | `metric all`: `campaign.py metrics-all` (one forward, all three sinks, chunk shards) or separate metric runs over models x precisions x DIMs, identity checks and the aggregate JSON/CSV/tables; computes no metric | run_measurement.py |
| `metric_table.py` | INTERNAL | deterministic stdout tables of metric summaries (display only) | metric_sweep.py, campaign.py |
| `measurement_identity.py` | INTERNAL | read-only shared identity of finished runs (`run_measurement.py identity RUN...`) | run_measurement.py |
| `campaign_build.py` | INTERNAL | the build authority: every CMake option of every measurement build, receipts, build cache | campaign.py, metric scripts |
| `metric_run.py` | INTERNAL | native metric collection with sink isolation; body of the standalone reducers | campaign.py, *_metrics.py |
| `campaign_metrics.py` | INTERNAL | aggregate and per-layer metric outputs | campaign.py |
| `campaign_inputs.py` | INTERNAL | model/tokenizer/dataset identity, SHA256SUMS | campaign.py, metric scripts |
| `campaign_stream.py` | INTERNAL | streaming gzip sink for SCU observations | metric_run.py |
| `campaign_verify.py` | INTERNAL | verifier CLI for a finished metric run (manifest binding, checksums) | user |
| `stage_cache.py` | INTERNAL | content-addressed offline stage cache | native profiling tools |
| `model_manifest.py` | INTERNAL | frozen model manifest (create/verify CLI, `model_entry`) | campaign.py |
| `evaluation_host.py` | INTERNAL | read-only host observations | e2e_run.py |
| `pmu_probe.c` | INTERNAL | PMU readiness probe compiled by the performance preflight | run_cycle_evaluation.py |
| `eval_common.py` | INTERNAL | shared metric and artifact helpers | all |
| `end_to_end.py` | INTERNAL | native CPU and CUDA collection (`collect`) CLI | native measurement tools |
| `e2e_run.py` | INTERNAL | native collection logic | end_to_end.py |
| `paired_inputs.py` | INTERNAL | PoTal/FullCPU pairing inputs | e2e_run.py |
| `application_results.py` | INTERNAL | native application endpoint results | end_to_end.py |
| `metric_reducers.py` | LEGACY | unbound ACT/RES reducers for historical streams (`--reduce` without a manifest) | metric_run.py |
| `plot_potal_pipeline.py` | INTERNAL | native PoTAL pipeline plotting | native profiling tools |
| `cycle_accounting.py` | INTERNAL | event interval accounting with caller-supplied event layout | native profiling tools |
| `e2e_timeline.py` | INTERNAL | host worker lanes and exact interval arithmetic | native profiling tools |
| `target_admission.py` | INTERNAL | target host and interface measurement validation | native measurement tools |
| `certified_reconstruction.py` | INTERNAL | hash-bound artifacts and publication rows | native measurement tools |
| `cpu_profiling.py` | INTERNAL | Native CPU timing validity and PMU profiling | native profiling tools |
| build-arm64.sh, build-arm64-cpu.sh, build-x86.sh, build-riscv.sh | DEBUG/RESEARCH | developer builds with their own option defaults; never a measurement build | developer |
| scripts/experiment/, scripts/utils/ | DEBUG/RESEARCH | cycle matrix experiments and log rendering helpers | developer |
| evaluation/ (python -m evaluation) | INTERNAL | ACT/RES/SCU reducers, schemas and manifest type | campaign_metrics.py, metric_run.py |
<!-- END GENERATED scripts -->
