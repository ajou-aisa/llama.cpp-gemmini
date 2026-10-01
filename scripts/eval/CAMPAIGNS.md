# Independent measurement campaigns

Metric campaigns (activation, residual, SCU) and the certification-only cycle campaigns. The canonical entry point
for metrics is `scripts/eval/run_measurement.py metric activation|residual|scu` (see `README.md` for the three
measurement domains: performance, timeline, metric); the shell wrappers below are equivalent and stay supported.
`run_measurement.py metric all` runs these three over models × precisions × DIMs, by default from one combined forward
per configuration with parallel chunk shards, and writes one aggregate summary (README "Metric sweep"); it adds no
metric of its own.

Each command configures a fresh CMake build, compiles the native llama backend,
runs focused cycle/metric CTests, executes native inference, validates the metric
stream and layer coverage, and writes bound JSON plus SHA256SUMS. Requires Python
3.11+, CMake, a C++ compiler, local HP1 GGUF models and WikiText-2 test text. No new
runtime Python package is required. Existing build scripts, model/cache files,
headers and historical evidence are not modified. Existing outputs are refused.

Run from the llama checkout:

```sh
python3 scripts/eval/run_measurement.py metric activation --model gpt2 --precision a8w8 --dim 16 --output /absolute/fresh/activation-results
python3 scripts/eval/run_measurement.py metric residual --model gpt2 --precision a8w8 --dim 16 --output /absolute/fresh/residual-results
python3 scripts/eval/run_measurement.py metric scu --model gpt2 --precision a8w8 --dim 16 --output /absolute/fresh/scale-results
# equivalent: scripts/eval/run_activation_metrics.sh | run_residual_metrics.sh | run_scu_metrics.sh with the same options
```

Without `--output` the wrapper writes to `runs/metrics/<kind>/<utc>-<model>-<precision>-d<dim>`. Every run also
carries the uniform names `summary.json`, `layers.json`, `request.json` and `raw.jsonl[.gz]` (hard links to the
metric-specific files), and `--model-manifest MANIFEST.json` records the frozen model identity that performance
runs reference.

Default is one 256-token chunk. `--max-chunks 0` selects every complete test chunk.
Tokenization uses the model's native tokenizer, removes one final LF, replaces
chunk-first BOS when configured, clears KV per chunk, and requests second-half
logits. All expected linear layers including `lm_head` must appear in **every**
chunk (49 GPT-2, 113 Llama-3.2-1B). PPL is not computed. Optional
`--workload-manifest previous/collection/workload-binding.json` enforces the same
native token/chunk identity between independent metrics.

The default files are `models/gpt2/gpt2.Q{4,8}_HP1.gguf`,
`models/llama3.2-1B/llama3.2-1B.Q{4,8}_HP1.gguf`, and
`wikitext-2-raw/wiki.test.raw`. Override model location with `--model-path`.
`--dataset-manifest FILE` accepts a hash-bound input:

```json
{"dataset":"WikiText-2","split":"test","path":"wiki.test.raw","sha256":"<actual SHA256>"}
```

The path is relative to that manifest. Without it, a manifest is generated from
the existing default dataset. No dataset or model is downloaded.

`evaluation_manifest.json` extends the existing nine-field manifest with
`tokenizer_hash`, `DIM` aliases and `build_hash`. Aliases must agree. Tokenizer
hash covers sorted exact GGUF tokenizer.* key/type/value bytes, excluding weights;
Q4/Q8 variants therefore share tokenizer identity. Build hash is the producing
executable SHA256. Build receipts preserve producing git SHA and dirty diff.
The exact manifest bytes are hashed into every measurement JSON. `manifest.json`
additionally binds model/dataset/build/workload identities; SHA256SUMS covers
all retained output bytes. Metrics remain OFF by default; each build enables
only ACT, RES or SCALE. `build_measurement.sh --kind activation|residual|scu|cycle`
supports a separately verifiable build step; pass its directory through
`--prepared-build` to reuse its verified binary and recorded producing revision.

SCU observations use a dedicated inherited pipe and streaming gzip to avoid an
uncompressed multi-gigabyte disk copy. Raw observations remain exact and auditable.
Layer partitions also use gzip temporary files. No observation sampling occurs.
The existing reducers retain integer-sum-before-division definitions. Undefined
denominators remain null. RES `radix_limb_count` sums radix counts per main stripe;
pruning/compaction fields contain explicit removed counts and fractions. SCU
delta is HP1 block-PoT-to-channel-anchor offset, not unavailable pre-PoT FP error.

## Cycle trace capture, certificate and campaign

```sh
ROOT="/absolute/evaluation-cycle-campaign-$(date -u +%Y%m%dT%H%M%SZ)"
python3 -B scripts/eval/cycle_campaign_bundle.py init "$ROOT"
scripts/eval/run_cycle_trace_capture.sh --model gpt2 --precision a8w8 --dim 16 --campaign-root "$ROOT" \
  --parent /absolute/stateful-full374-current.json /absolute/tag6-evidence-root
scripts/eval/run_cycle_campaign.sh \
  --trace "$ROOT/trace/evaluation-cycle/GPT-2-124M/A8W8/DIM16/capture/trace.jsonl.gz" \
  --certificate "$ROOT/certificate/GPT-2-124M/A8W8/DIM16/cycle_trace_certificate.json" \
  --output "$ROOT/cycle/GPT-2-124M/A8W8/DIM16"
python3 -B scripts/eval/cycle_campaign_bundle.py finalize "$ROOT"   # after regression/results.json
```

Capture configures a fresh metrics-OFF `CYCLE_SIM=1` `STRIPE_PIPELINE` build, runs actual
inference (WikiText-2 test chunk0, 256 prefill tokens plus one greedy sample, seed 1234), and writes
`trace/evaluation-cycle/<Model>/<Precision>/DIM<n>/capture/` with `trace.jsonl.gz` (byte-exact
deterministic gzip of the native trace; content SHA recorded), `producer-manifest.json`,
`evaluation-manifest.json` (also copied to `manifest/.../evaluation_manifest.json`),
`source-binding.json` (git SHAs/diffs, compiler, build/metric/cycle options) and `SHA256SUMS`.
It then builds `certificate/.../cycle_trace_certificate.json` with IM2P.sim
`sim.cycle.cycle_trace_certificate`. A fresh trace is admitted only when its ordered NPU work
descriptors (geometry, tile shape, compact runs, row map, work classes) equal a supplied,
independently certified corpus that itself passes current `validate`/`admit`; it inherits that
corpus's state-domain revision and offer policy. Otherwise `rejection.json` is written, the script
exits 3 and no cycle run is possible. The certificate is re-verified by recomputation at every use.

The campaign verifies the certificate, admits it through the unchanged
`StatefulSequenceProvider` admission path, executes every work back-to-back in one session with
passive native event recording, and writes `cycle-results.json`, `per-work-cycle.jsonl`,
`scu-cycle-accounting.json`, `window-parity.json` and `SHA256SUMS`. `--reference-per-work` requires
exact offered/accepted/result/final-scale/resource windows against an earlier replay. Load, store,
scale and SCU cycles are unions of closed intervals of existing native events
(ReadRequest/Response, WriteRequest/Completion, ScaleRequest/Response, ScaleLane, ScaleRelease)
inside each work's accepted-to-resource-ready window; event counts must equal native counters.
SCU execution timing is separate from the SCU scale-alignment metric. Metric campaigns can bind the
same manifest bytes with `--evaluation-manifest manifest/.../evaluation_manifest.json`.

## Actual-inference evaluation cycles

The paper recipe is `--recipe evaluation`: WikiText-2 test chunk0, 256 prompt tokens, one greedy
sample (seed 1234, temperature 0), zero decode calls, STRIPE_PIPELINE, metrics OFF. It writes
`traces/<model>/<precision>/dim<n>/` with `trace.jsonl.gz` (raw content SHA, compressed SHA and a
decompression verification), `recipe.json`, `cycle_manifest.json` (model, precision, dim, BK, trace,
producer, build, tokenizer and recipe SHA256), producer/evaluation manifests and source binding.

```sh
ROOT="/absolute/actual-cycle-campaign-$(date -u +%Y%m%dT%H%M%SZ)"
python3 -B scripts/eval/actual_cycle_bundle.py init "$ROOT"
scripts/eval/run_cycle_trace_capture.sh --model gpt2 --precision a8w8 --dim 32 --recipe evaluation \
  --campaign-root "$ROOT"
```

Without an equivalent certified corpus the certificate stays `PENDING_INDEPENDENT_CERTIFICATION`.
IM2P.sim `sim/tests/cycle/actual_trace_{stimulus,capture,compare,replay,milestone}.py` then produce
independent evidence for that exact trace: RTL versus model on every probe-admissible work (at most
4096 works, m/n/k <= 8192; `lm_head` stays native-only), two complete native replays with record
parity, and per-edge versus boundary milestone parity. Reviewed evidence hashes are pinned in
`sim/cycle/actual_trace_pins.py`; `sim.cycle.actual_trace_certificate build` then issues an
`im2p-actual-trace-certificate-v1` whose finite state domain is exactly the RTL-observed tag/row range.
The per-trace certificate follows through the unchanged admission path:

```sh
scripts/eval/run_cycle_trace_capture.sh --model gpt2 --precision a8w8 --dim 32 --recipe evaluation \
  --campaign-root "$ROOT" --certify-existing --parent /absolute/actual_trace_certificate.json /absolute/evidence
scripts/eval/run_cycle_campaign.sh --evaluation \
  --trace "$ROOT/traces/gpt2/a8w8/dim32/trace.jsonl.gz" \
  --certificate "$ROOT/certificates/gpt2/a8w8/dim32/cycle_trace_certificate.json" \
  --reference-replay /absolute/evidence/replay-first/report.json --output "$ROOT/cycles/gpt2/a8w8/dim32"
python3 -B scripts/eval/actual_cycle_bundle.py finalize "$ROOT"   # after regression/results.json
```

Recipe mode and `--evaluation` accept only real-inference parents (the full374 replay corpus or an
actual-trace certificate); synthetic producer certificates remain regression-only.
`--reference-replay` requires every per-work window to equal the independent native replay records.

## Legacy cycle replay adapter

```sh
python3 -B scripts/eval/campaign.py cycle --model gpt2 --precision a8w8 --dim 16 \
  --certificate /absolute/stateful-certificate.json \
  --evidence-root /absolute/existing-certified-evidence \
  --library /absolute/certified/libim2p_cycle_model.dylib \
  --trace-source /absolute/actual-inference/npu-cycle-trace.jsonl \
  --source-provenance /absolute/actual-inference/provenance.json \
  --output /absolute/fresh/cycle-results
```

This builds/verifies both the native inference runner and cycle library, then uses
one persistent production `StatefulSequenceProvider` session and verifies complete
work coverage. Library hashes record fresh and certified binaries separately.
The certified runtime may require its original path even when a fresh build is
byte-identical. Explicit source replay retains the original producer's revision,
binary, seed and workload policy; it never claims fresh inference at current HEAD.

Omitting `--trace-source` collects a fresh zero-generation 256-token prefill with
metrics OFF. The certificate must admit that **exact** trace. There is no implicit
recertification or isolated-work fallback. Rejected admission leaves native trace,
command receipts, failure.json and checksums; it does not publish cycle totals.

The current provider admits finite, exact trace corpora. The verified reference
is GPT-2 A8W8 DIM16, chunk0, 256 prefill tokens and one sample, zero decode calls;
its output mask is last-token, unlike the independent metrics' second-half mask.
Other model/configuration traces require their own admitted certificate. This
campaign adapter does not enlarge that domain or generate RTL certification.

Dense/residual cycles are result-ready minus accepted. Submission count is native
planner loop/frame count; logical work count is separate. Traffic uses native
request/response transactions, not guessed bytes. Resource cycles use the
resource-ready minus offered interval. SCU drain tail is reported separately;
this legacy adapter records no events, so its `scu_cycles` is null with an explicit
status (use `run_cycle_campaign.sh` above for event-based SCU timing). Prefill work is batch-attributed in `per-token-cycle.json`;
individual prefill-token cycles are not observable and remain null.

**Cycle count != latency(ms).** No frequency conversion, TTFT/TPOT publication,
Jetson, CUDA comparison, FPGA programming, synthesis or post-route work occurs.
`E2E_RECONSTRUCTION_READY=NOT_READY`, `PAPER_CAMPAIGN_COMPLETE=NOT_RUN` remain fixed.
A configuration produces cycle evidence only after its own production trace is admitted;
configurations without an equivalent certified corpus stop at `rejection.json`.

## Twelve configurations

Run each metric independently. The following executes all complete chunks; it is
an execution recipe, not a claim these campaigns have already run:

```sh
ROOT="/absolute/evaluation-campaign-$(date -u +%Y%m%dT%H%M%SZ)"
for MODEL in gpt2 llama3.2-1B; do
  for PRECISION in a4w4 a8w8; do
    for DIM in 16 32 64; do
      for METRIC in activation residual scu; do
        "scripts/eval/run_${METRIC}_metrics.sh" --model "$MODEL" \
          --precision "$PRECISION" --dim "$DIM" --max-chunks 0 --timeout 86400 \
          --output "$ROOT/$METRIC/$MODEL-$PRECISION-d$DIM"
      done
    done
  done
done
```

Use `--timeout` to bound each native inference run. Cycle runs use their matching
production trace/certificate inputs independently; substituting the reference
certificate for another model, seed, precision or DIM is rejected.

```sh
python3 -B -m evaluation.tests.test_campaign
python3 -B -m evaluation.tests.test_cycle_campaign
python3 -B -m evaluation.tests.test_framework
pyright -p scripts/eval/campaign-pyright.json
pyright -p evaluation/pyrightconfig.json
uv run --no-project --offline --with ruff ruff check scripts/eval/campaign*.py scripts/eval/cycle_*.py evaluation
git diff --check
```

## E2E cost ownership and integration prep

`scripts/eval/e2e_cost_ownership.py build DIR` emits and validates three
machine-readable contracts: `cost-ownership-contract.json` (every target stage
carries exactly one authority among HOST_MEASURED, NPU_MODELED,
EXCLUDED_WITH_REASON, UNMODELED, DIAGNOSTIC_ONLY; source files are hash-bound),
`memory-interface-scenario.json` (REFERENCE_MEMORY semantics; actual DRAM stays
unmodeled and unmeasured; interface transports stay UNMODELED until declared)
and `cycle-accounting-definitions.json` (frozen field names with
`definition_revision`, e.g. `scu_active_cycles` =
`SCALE_PATH_FETCH_LANE_RELEASE_UNION_V1`, never "SCU ALU utilization").

`scripts/eval/e2e_integration_prep.py` walks one certified actual trace through
the unchanged stateful provider, derives `request_available` from declared
host-stage dependencies (zero-duration; ordering evidence only, never latency),
records `port_offer`/`accepted`/`result_ready`/`final_scale_release`/
`resource_ready` per work, re-verifies in a fresh session and writes
`stateful-e2e-integration-prep.json` with publication kept fail-closed
(`NOT_READY_MISSING_TARGET_HOST_ADMISSION`, `NOT_READY_MISSING_OPERATING_CLOCK`).

```sh
python3 -B -m evaluation.tests.test_publication_gates
```
