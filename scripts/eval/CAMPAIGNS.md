# Independent measurement campaigns

Each command configures a fresh CMake build, compiles the native llama backend,
runs focused cycle/metric CTests, executes native inference, validates the metric
stream and layer coverage, and writes bound JSON plus SHA256SUMS. Requires Python
3.11+, CMake, a C++ compiler, local HP1 GGUF models and WikiText-2 test text. No new
runtime Python package is required. Existing build scripts, model/cache files,
headers and historical evidence are not modified. Existing outputs are refused.

Run from the llama checkout:

```sh
scripts/eval/run_activation_metrics.sh --model gpt2 --precision a8w8 --dim 16 --output /absolute/fresh/activation-results
scripts/eval/run_residual_metrics.sh --model gpt2 --precision a8w8 --dim 16 --output /absolute/fresh/residual-results
scripts/eval/run_scu_metrics.sh --model gpt2 --precision a8w8 --dim 16 --output /absolute/fresh/scale-results
```

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

## Cycles and certification boundary

```sh
scripts/eval/run_cycle_campaign.sh --model gpt2 --precision a8w8 --dim 16 \
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
the provider does **not** expose total SCU-active cycles, so `scu_cycles` is null
with an explicit status. Prefill work is batch-attributed in `per-token-cycle.json`;
individual prefill-token cycles are not observable and remain null.

**Cycle count != latency(ms).** No frequency conversion, TTFT/TPOT publication,
Jetson, CUDA comparison, FPGA programming, synthesis or post-route work occurs.
`E2E_RECONSTRUCTION_READY=NOT_READY`, `PAPER_CAMPAIGN_COMPLETE=NOT_RUN` remain fixed.
The complete 12-configuration cycle campaign cannot be marked READY while exact
trace admissions and total SCU cycle accounting are unavailable.

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
python3 -B -m evaluation.tests.test_framework
pyright -p scripts/eval/campaign-pyright.json
pyright -p evaluation/pyrightconfig.json
uv run --no-project --offline --with ruff ruff check scripts/eval/campaign*.py evaluation
git diff --check
```
