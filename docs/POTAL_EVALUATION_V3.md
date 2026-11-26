# Separated PoTal evaluation

The native `llama-eval-workload` target shares tokenization and non-strided batch
construction with `llama-perplexity`. Perplexity scoring remains a separate tool.
Metric prefills use native 256-token chunks and the second-half logits mask;
generation uses the last-token mask, 128 actual samples and 127 subsequent decodes.
EOS tokens remain selectable and do not stop generation. Neither path warms up the model.

## Independent builds and outputs

`GGML_GEMMINI_ACT_QUANT_METRICS` and `GGML_GEMMINI_RESIDUAL_METRICS` accept only
`0` or `1`, default to `0`, and are resolved by CMake. They do not enable cycle
logging, detailed telemetry, or validation.
Use ACT/RES `10` for activation, `01` for residual, `00` for E2E; `11` is diagnostic.

Example native functional ACT build, from the llama repository:

```sh
cmake --preset gemmini-int8-d16-reference -S . -B build-eval-act \
  -DGGML_GEMMINI_ACT_QUANT_METRICS=1 -DGGML_GEMMINI_RESIDUAL_METRICS=0 \
  -DLOG_CYCLE=0 -DGGML_CPU_CYCLE_LOG=OFF -DCYCLE_DETAIL=0 \
  -DGGML_GEMMINI_ENABLE_RMD=ON -DGGML_GEMMINI_DEFAULT_RMD_BACKEND=CPU \
  -DGGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE=OFF -DLLAMA_CURL=OFF
cmake --build build-eval-act --target llama-eval-workload -j16

python3 -B scripts/eval/activation_quant_metrics.py \
  --runner build-eval-act/bin/llama-eval-workload \
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
identity and executes the configured internal Gemmini REFERENCE numerical path.
No online timing model is linked or called.

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
pyright --project scripts/eval/campaign-pyright.json
```

## E2E source collection

```sh
python3 -B scripts/eval/end_to_end.py --help
python3 -B scripts/eval/end_to_end.py --output /tmp/potal-e2e-fresh collect \
  --runner /absolute/build/bin/llama-eval-workload \
  --model models/gpt2/gpt2.Q8_HP1.gguf --dataset wikitext-2-raw/wiki.test.raw \
  --settings /absolute/evaluation-settings.json --role fullcpu --repetitions 1
```

Settings schema version 2 fixes `sampler_policy=user-confirmed-greedy-v3`, seed
1234, temperature 0, top_k 0, top_p 1, min_p 0, repeat_penalty 1, repeat_last_n 0,
grammar null, eos_stopping false and warmup 0. Threads/batch settings remain
explicit. Split is test and chunk_ids is exactly [0,1,2,3,4,5,6,7,8,9]. Run r uses
native non-overlapping chunk r once: these are ten distinct prompts, not ten copies
of chunk 0. The token-vector hash and BOS policy are bound per model/run.
One repetition covers only chunk 0. No failed prompt substitution or automatic retry.

PoTal's actual generated token IDs can define a paired replay trajectory. An explicit
FullCPU `FORCED_CPU_COST_ONLY` run replays those IDs when needed, performs no sampling,
and contributes ordinary CPU costs only. It is not a free-generation latency run.
The supplied PoTal record owns its sampling service; CUDA generates its own sequence.

FullCPU builds require LOG_CYCLE=1 and GGML_CPU_CYCLE_LOG=1, with both metrics OFF.
HP1 FullCPU uses the internal Gemmini REFERENCE backend, OPTION=CPU, HARDWARE,
RMD backend CPU and runtime matmul overrides OFF; disabling the Gemmini CPU backend
makes HP1 matmul weights unsupported. Other accelerators and dynamic backend
loading must be OFF for the FullCPU role proof.

Native collection records executable and runtime artifact hashes and checks that
the artifact snapshot stays unchanged between repetitions. Each repetition runs
the native workload again. Model, dataset, settings and token trajectory are
bound to its result; copied host timings are not additional repetitions.

Instrumented FullCPU wall endpoints are collection observations, **not**
target latency: semantic serialization remains present. The aggregate rejects
them. CUDA direct application measurement requires logging OFF,
actual complete GPU placement, and the same host. CUDA is a practical Q4_0/Q8_0
reference, not an arithmetic-matched A4W4/A8W8 result.

## Hardware timing scope

Metric counts and native application observations remain separate from hardware
timing claims. Hardware timing needs a verified operating clock, an explicit
service/scenario contract, matched host data and complete repetitions.

The evaluation hardware reference is U250 part `xcu250-figd2104-2L-e`, not a deployment
claim. An OOC compute-core route uses CLK, no board pins/PCIe/DDR/I/O delays or
bitstream. Only verified post-route timing closure can select the clock. The sizing
reference is 33 dense INT8 TOPS, one array and one MAC/PE/cycle for both precisions;
peak is `2*DIM*DIM*f/1e12`. An unreachable target selects the highest verified tested
frequency, with finite search bounds/resolution reported; unsupported part never falls back.

Independent metric counts do not establish Jetson latency or validated overlap.
Missing Vivado or Jetson access does not block independent metric counts.
