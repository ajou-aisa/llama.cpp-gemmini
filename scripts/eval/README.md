# Independent PoTal evaluation commands

Run from the llama checkout with Python 3.11 or newer. No Python tokenizer,
NPU timing equation, new numerical executor, or added package dependency is used.

`llama-eval-workload --build-info` reports the compiled metric flags and target
recipe. Scripts bind the executable/shared-library bytes and input hashes before
and after collection. Runtime paths never enable compiled-out hooks.

## ACT and RES

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

## Application and offline reconstruction

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
synthetic scheduler. Until a validated clock and service-boundary provider are
available, target TTFT/TPOT remain null and E2E reconstruction is NOT_READY.
This does not claim a Jetson measurement, a full paper campaign, or quality PPL.

`--lifecycle-sidecar` plus explicit `--worker-resources`, `--cpu-policy`, and
`--sampler-resource` invokes the official producer-bound lifecycle builder after
joining. `--streaming-ir` publishes bounded-memory SQLite IR. Adding both
`--diagnostic-phase-table /absolute/phase-table.json` and
`--diagnostic-frequency-hz 1000000000` explicitly requests a synthetic schedule
in `schedule.sqlite`; the numeric frequency is a diagnostic scenario, not a
selected hardware clock. Without `--streaming-ir`, the same diagnostic path
writes `schedule.json`. Neither path publishes target TTFT/TPOT.
