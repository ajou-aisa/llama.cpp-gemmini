# Separated PoTal evaluation

The native `llama-eval-workload` target shares tokenization and non-strided batch
construction with `llama-perplexity`. Perplexity scoring remains a separate tool.
Metric prefills use native 256-token chunks and the second-half logits mask;
generation uses the last-token mask, 128 actual samples and 127 subsequent decodes.
EOS tokens remain selectable and do not stop generation. Neither path warms up the model.

## Independent builds and outputs

`GGML_GEMMINI_ACT_QUANT_METRICS` and `GGML_GEMMINI_RESIDUAL_METRICS` accept only
`0` or `1`, default to `0`, and use the existing build-script resolver. They do not
enable cycle logging, detailed telemetry, validation, or functional execution.
Use ACT/RES `10` for activation, `01` for residual, `00` for E2E; `11` is diagnostic.

Example native functional ACT build, from the llama repository:

```sh
BUILD_DIR="$HOME/aisa-lab/build/im2p-gemmini/eval-act" \
GGML_GEMMINI=ON GGML_GEMMINI_OPTION=WS \
GGML_GEMMINI_EXECUTION_BACKEND=IM2P_SIM IM2P_SIM_IMPLEMENTATION=GEMMINI_HP1 \
CYCLE_SIM=1 LOG_CYCLE=0 GGML_CPU_CYCLE_LOG=0 CYCLE_DETAIL=0 \
GGML_GEMMINI_ACT_QUANT_METRICS=1 GGML_GEMMINI_RESIDUAL_METRICS=0 \
GGML_GEMMINI_ACTIVATION_BITS=8 GGML_GEMMINI_WEIGHT_BITS=8 GGML_GEMMINI_DIM=16 \
GGML_GEMMINI_ENABLE_RMD=ON GGML_GEMMINI_DEFAULT_RMD_BACKEND=WS \
GGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE=OFF \
bash build-arm64.sh -DGGML_METAL=OFF -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF

python3 -B scripts/eval/activation_quant_metrics.py \
  --runner "$HOME/aisa-lab/build/im2p-gemmini/eval-act/bin/llama-eval-workload" \
  --model models/gpt2/gpt2.Q8_HP1.gguf --dataset wikitext-2-raw/wiki.test.raw \
  --split test --max-chunks 1 --output /tmp/potal-act-fresh
```

Use a fresh output directory. For residual, build `ACT=0,RES=1` and invoke
`scripts/eval/residual_path_metrics.py` with the same arguments. Its optional
`--workload-manifest /tmp/potal-act-fresh/workload-binding.json` checks identical
native chunks, tokenizer/BOS/mask policy, model and dataset. `--max-chunks 0`
selects all complete chunks; it is a campaign, not a smoke.

Each entrypoint writes its own dedicated JSONL and summary (`counts.jsonl` or
`shapes.jsonl` and `summary.json` aliases). The actual binary, input hashes and
native workload are bound separately. Statistics-only execution retains semantic
identity but does not emit the large auxiliary NPU timing trace. CYCLE_SIM still
selects the existing CPU-functional numerical implementation; no online timing
model is linked or called.

ACT observes original FP, final folding selection, actual residual nonzeros and
the union of genuinely recomputed original row/block identities. The confirmed
reference revision is `signed-row-original-bk32-population-2sigma-v1`: each logical
row and original BK32 block uses all prequantization signed FP values, population
variance divided by the actual N, and strict `x > mean + 2*sigma`. Tail padding,
absolute values and top1 exclusion are absent. The historical top1-excluded
magnitude detector is comparison context only. Non-finite inputs are not filtered:
raw counts remain, but incomplete F counts and final F ratios are withheld.
The obsolete `--activation-reference-candidate` option is rejected.

RES records every MAIN_STRIPE, including residual-free stripes, and actual compact
requests/runs/row maps. Raw D/T/R are always preserved. The proposed v3 weighting
is explicitly selected with `--accept-proposed-weighting`, not substituted for
historical weighted-K retention. Ratios use integer sums, never layer-ratio means.
Zero denominators are null; row expansion can produce a ratio greater than one.

The evaluation scripts require Python 3.11 or newer. Check this isolated script
surface explicitly; the repository-wide Python default remains 3.9 for legacy
tools:

```sh
python3 -B tests/test-evaluation-v3.py
python3 -B tests/test-evaluation-v3-pairing.py
pyright --pythonversion 3.11 scripts/eval
pyright scripts/evaluation_build_options.py scripts/im2p-build-options.py
```

## E2E source collection

```sh
python3 -B scripts/eval/end_to_end.py --help
python3 -B scripts/eval/end_to_end.py --output /tmp/potal-e2e-fresh collect \
  --runner /absolute/build/bin/llama-eval-workload --im2p ../IM2P.sim \
  --model models/gpt2/gpt2.Q8_HP1.gguf --dataset wikitext-2-raw/wiki.test.raw \
  --settings /absolute/evaluation-settings.json --role potal --repetitions 1
```

Settings schema version 2 fixes `sampler_policy=user-confirmed-greedy-v3`, seed
1234, temperature 0, top_k 0, top_p 1, min_p 0, repeat_penalty 1, repeat_last_n 0,
grammar null, eos_stopping false and warmup 0. Threads/batch settings remain
explicit. Split is test and chunk_ids is exactly [0,1,2,3,4,5,6,7,8,9]. Run r uses
native non-overlapping chunk r once: these are ten distinct prompts, not ten copies
of chunk 0. The token-vector hash and BOS policy are bound per model/run.
One repetition covers only chunk 0. No failed prompt substitution or automatic retry.

PoTal's actual generated token IDs define the reconstruction trajectory. An explicit
FullCPU `FORCED_CPU_COST_ONLY` run replays those IDs when needed, performs no sampling,
and contributes ordinary CPU costs only. It is not a free-generation latency run.
PoTal alone supplies sampling service; CUDA generates its own sequence.

FullCPU and PoTal builds require LOG_CYCLE=1 and GGML_CPU_CYCLE_LOG=1, with both
metrics OFF. PoTal uses CYCLE_SIM=1. HP1 FullCPU uses GEMMINI=ON, OPTION=CPU,
HARDWARE, RMD backend CPU, CYCLE_SIM=0 and runtime matmul overrides OFF; disabling
the Gemmini CPU backend makes HP1 matmul weights unsupported. Other accelerators
and dynamic backend loading must be OFF for the FullCPU role proof.

Native provenance v2 binds an actual configure/build-once receipt, selected
executable compiler dependencies, CPU kernels/objects and runtime artifacts.
Repetitions reuse those bytes, not copied host timings. Legacy provenance is not
silently upgraded. A changed trajectory/kernel/host or missing source file fails
the join. Application sampling service belongs to the PoTal source exactly once.

Instrumented FullCPU/PoTal wall endpoints are collection observations, **not**
target latency: semantic serialization remains present. The aggregate rejects
them. CUDA direct application measurement requires logging and CYCLE_SIM OFF,
actual complete GPU placement, and the same host. CUDA is a practical Q4_0/Q8_0
reference, not an arithmetic-matched A4W4/A8W8 result.

## Offline boundary

Use the existing certified NPU replay, three-source join, and the explicit
execution lifecycle/application adapters; see
[execution IR](../../IM2P.sim/docs/EVALUATION_EXECUTION_IR.md) and
[clock selection](../../IM2P.sim/docs/EVALUATION_CLOCK.md).
FULL lifecycle declarations come from real synchronized native dispatches.
Unsupported PIPELINE semantics fail closed. Isolated NPU accounting is retained;
the additive service API separates result readiness from final resource release.

The evaluation hardware reference is U250 part `xcu250-figd2104-2L-e`, not a deployment
claim. An OOC compute-core route uses CLK, no board pins/PCIe/DDR/I/O delays or
bitstream. Only verified post-route timing closure can select the clock. The sizing
reference is 33 dense INT8 TOPS, one array and one MAC/PE/cycle for both precisions;
peak is `2*DIM*DIM*f/1e12`. An unreachable target selects the highest verified tested
frequency, with finite search bounds/resolution reported; unsupported part never falls back.

Synthetic scheduler tests are not a Jetson measurement or validated overlap.
No paper latency is available without a verified operating clock, service/scenario
contract, matched host data and complete repetitions. Missing Vivado or Jetson
access does not block independent metric counts.
