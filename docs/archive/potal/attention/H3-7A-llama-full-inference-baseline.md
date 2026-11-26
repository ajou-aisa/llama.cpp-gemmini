# H3-7A: llama.cpp Full Inference Baseline

Date: 2026-10-25. Tests stopped at the user's request.
Overall verdict: **B -- Partial Verification** of the requested full scope.
Real end-to-end inference succeeded in the completed cases; the uncompleted
E3 cross-mode checks were canceled, not counted as passed. Performance
comparison completion is separately partial.

## 1. Executive Summary

The existing GPT-2 model completed real model loading, 16-token prefill, seven
KV-cached autoregressive decode steps, and eight generated tokens on the existing
FireSim U280. Serial, Paired RTL, and Paired Conservative all returned exit code
0. All 402,056 captured logits per mode are bitwise identical across the three
hardware modes. Their eight greedy token IDs also match an independent CPU FP32
reference using the original checkpoint.

The actual model has **124,439,808 parameters**, not 163 million. The existing
model was audited and retained; no alternate GPT-2 checkpoint was substituted.

Hardware evidence includes completed Main and Residual GEMMs, Scale commands,
and, in both Paired modes, Index commands and actual Paired stripes. This is
neither a MatMul-only test nor CPU-only generation. No hardware failure or
capacity fallback was reported in E1/E2.

E1/E2 demonstrate successful full inference for the tested 16+8 workload.
Serial additionally completed 256+32, including one captured request and three
uncaptured primary requests. Paired E3 was stopped after 11 emitted tokens;
Conservative E3 was not started. The complete three-mode E3 performance
comparison is therefore **incomplete by user cancellation**.
No FP32 logit tolerance was invented: FP32 errors are quantified, while exact
Serial/Paired equivalence and token agreement are reported separately.

One instrumentation issue remains relevant to interpreting the first run:
llama.cpp overwrites a directly installed scheduler callback during decode.
Consequently, the original E1/E2 CPU-operation arrays are empty and are **not**
valid zero-fallback counts. The test-only collector was corrected and checked on
the host before E3. Production PoTal code and the hardware image were not changed.

## 2. Baseline

### Checkpoint and checkout

- Branch: `potal`.
- Origin: `git@github.com:ajou-aisa/llama.cpp-gemmini.git`.
- D1 checkpoint: `70fa117d58ae6cfa966fe08fc64019267180827c`.
- Commit title: `test(potal): add H3-6D1 planning diagnostics and analysis`.
- Parent: `26ec82756feedb131dc5acc6e93e16a0b04512ca`.
- Exactly seven requested D1 paths were committed, with 1,722 insertions and
  three deletions. Model files, build products, and H7 test files were excluded.
- Both supplied checkout paths resolved to the same real directory and Git
  root, branch, HEAD, and remote. This was checked before staging.

Canonical checkout:

```text
/home/alveo/firesim/target-design/chipyard/generators/potal/software/workspace/llama.cpp-potal
```

The checkpoint was made after inspecting unstaged/staged changes and checking
the staged names, statistics, and whitespace. No push, new branch, or additional
project commit was made.

### Build and hardware

The existing `build-potal-h3-6a-d16` Release configuration was reused.
`cmake --build build-potal-h3-6a-d16 -j4 --target llama-cli llama-bench` passed.
The compiler is the existing RISC-V GCC/G++ 12.2.0 toolchain under
`/home/alveo/firesim/.conda-env/riscv-tools/bin/`.

Configuration checked in CMakeCache and generated compiler definitions:

- A8W8 ExSIA, HP1 block32, residual/error compensation enabled.
- DIM16; `STRIPE_PIPELINE`; Residual backend `WS`.
- `GGML_GEMMINI_EXECUTION_BACKEND_HARDWARE=1`.
- `POTAL_PLANNING_DIAGNOSTICS=OFF`.
- `LOG_CYCLE=0`, `CYCLE_SIM=0`, and ExSIA profiling scope OFF.

The legacy Gemmini `OPTION=CPU` cache entry does not select the PoTal execution
route. PoTal has its own RISC-V RoCC issuer; actual successful command counts and
results below establish execution rather than relying on that cache label.

The same existing U280 driver and bitstream were used:

```text
/home/alveo/FIRESIM_RUNS_DIR/sim_slot_0/FireSim-xilinx_alveo_u280
```

The driver reported PCI device `0000:01:00.0`, vendor/device `10ee:903f`, and
FireSim fingerprint `0x46697265`. No FPGA/RTL rebuild was performed. TraceRV was
disabled because no trace file was supplied.

The original D1 rootfs was preserved. Its separate H7 copy,
`/tmp/h3-7a-run/rootfs.img`, was expanded from a full 16 GiB filesystem to 24 GiB
using offline filesystem checks/resizing. After staging it had approximately
7 GiB available. The guest reported 16,005 MiB RAM and no swap. Dynamic loader
dependency checks passed before inference.

All shells used:

```bash
source /home/alveo/miniforge3/etc/profile.d/conda.sh
cd /home/alveo/firesim/target-design/chipyard
source env.sh
```

`env.sh` emits a pre-existing `EnvironmentLocationNotFound` for Chipyard's
`.conda-env` path. The already-installed toolchain and builds still worked.

The production RISC-V `libggml-potal.so` SHA256 remained:

```text
1aca66631382f52856d4a187822e456f03a7522a9c828c968811619377c240b6
```

## 3. Model and Quantization

### Actual assets

| Asset | Bytes | Tensor inventory |
|---|---:|---|
| `models/gpt2/gpt2.fp16.gguf` | 252,470,496 | 49 F16 + 99 F32 |
| `models/gpt2/gpt2.Q8_HP1.gguf` | 159,821,472 | 49 Q8_HP1 + 99 F32 |
| Cached original `model.safetensors` | 548,105,171 | 148 parameters + 12 causal-mask buffers |

SHA256:

```text
FP16 GGUF:
5bd663a1e1d303f9404dd2edd92e4e3ca3e24bcf80d19c655880f0e455c8f4df
HP1 GGUF:
cf23a45e4db23736d7d1f67ab42018803a198cfcb01db36daca69ff56bdbc7d4
Original FP32 safetensors:
248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707
```

The original checkpoint is the existing Hugging Face cache snapshot
`607a30d783dfa663caf39e06633721c8d4cfcd7e` for `gpt2`.
Every one of the 148 FP16-GGUF parameter tensors was compared against that
snapshot, with the known GPT-2 Conv1D transposes and the GGUF tensor's storage
dtype conversion. All 124,439,808 elements matched exactly. The only excluded
safetensors entries were the 12 non-parameter attention causal masks. Thus the
independent CPU reference is tied to the actual checkpoint, not merely its name
or architecture.

The model has 12 layers, embedding size 768, FFN size 3,072, 12 heads, vocabulary
50,257, and trained context 1,024. Output weights are tied to token embeddings.
The GGUF files have the same GPT-2 BPE token/merge arrays: 50,257 vocabulary
entries, 50,000 merges, and BOS/EOS ID 50,256.

Both files are GGUFv3. HP1 file type is 40 and tensor type is 41. HP1 block32
stores 32 signed codes, an int16 exponent, two padding bytes, and a float32
channel scale in 40 bytes. The audit checked all 3,860,376 blocks:
finite positive power-of-two scales, constant channel scale within each row,
zero padding, no -128 code, no sentinel exponent, and observed exponents 0--9.
All HP1 K dimensions are divisible by 32. The 99 F32 tensors are byte-identical
between the FP16 and HP1 GGUF files.

Relevant implementation:
`src/llama-model.cpp:2676` loads GPT-2/tied output tensors;
`ggml/src/ggml-common.h:249` defines HP1 storage;
`ggml/src/ggml-quants.c:612` reconstructs its weights;
`ggml/src/ggml-potal/matmul/dense.cpp:100` checks the supported HP1 contract.

### Runtime preparation

Each cold E1/E2 process prepared 49 weights exactly once:
49 cache misses, 49 validations, 49 packs, 139,543,936 packed bytes, followed by
343 cache hits. There were no uncached/bypass preparations. The loader reported
147.72 MiB in the POTAL model buffer.

The warning that token embeddings and F32 tensors use CPU storage is expected:
embedding lookup is not a PoTal MatMul, and tied output projection has its own
PoTal weight buffer. It is not evidence of an HP1 MatMul CPU fallback.

No model file was changed, converted in place, deleted, or added to Git.

## 4. End-to-End Execution

The existing `llama-cli`, `llama-bench`, and `examples/simple` path were inspected;
actual `--help`, version, device listing, and loader dependencies were checked.
The CLI did not expose all per-step logits and executed CPU-operation counts
needed here. A standalone acceptance runner therefore uses the existing
`llama_model_load_from_file`, `llama_init_from_model`, `llama_decode`,
KV-cache, and backend APIs. It does not implement an alternate transformer,
quantizer, MatMul, or hardware path.

The test source is
`ggml/src/ggml-potal/tests/potal-full-inference.cpp`.
It is compiled separately against the unchanged production libraries:

```bash
riscv64-unknown-linux-gnu-g++ -O3 -DNDEBUG -std=gnu++17 -Wall -Wextra \
  -Iinclude -Isrc -Iggml/include -Iggml/src -Icommon \
  ggml/src/ggml-potal/tests/potal-full-inference.cpp \
  -Lbuild-potal-h3-6a-d16/bin \
  -lllama -lggml-potal -lggml-cpu -lggml -lggml-base \
  -lggml-gemmini-utils -pthread -Wl,-rpath,'$ORIGIN' \
  -o potal-full-inference
```

All target libraries are staged alongside the executable. The corrected E3
collector SHA256 is:

```text
6b7027f56c7e6130ec53f3e3b858902b973fae4c8cdd7079b7924643795587c7
```

Shared inference settings: context 512, batch/ubatch 256, three CPU threads,
F16 KV cache, flash attention OFF, K/Q/V CPU attention offload disabled, greedy
selection with lowest-ID ties, and EOS excluded to retain the exact requested
generation length. In all hardware modes PoTal operation offload is enabled.

The prompt fixture is the existing `prompts/gpt2-256.txt`, which tokenizes to
exactly 256 IDs. E1 uses its first 16:

```text
37065 12 3106 27665 16936 4780 2989 357
1565 8035 8 16113 389 6768 973 284
```

All three E1/E2 hardware modes and both CPU references generated:

```text
5911 290 5911 262 16936 4780 13 383
```

One request performs one multi-token prefill and seven single-token decode
calls, feeding the preceding selected token at each next position. Logits are
synchronized before selection. At every captured step the runner checked
KV count `16 + step` and maximum position `15 + step`; the final state was
23 cached tokens and maximum position 22. The eighth generated token is emitted
but is not fed back, explaining the final cache count.

E3 uses the same fixture, checkpoint, quantization, sampler, and context
settings, with 256 prompt tokens and 32 generated tokens. Its captured warmup
has one prefill plus 31 decode calls; final KV expectations are 287 tokens and
maximum position 286. The original protocol scheduled three subsequent
uncaptured requests per mode. Serial completed all three; Paired was interrupted
in its captured request and Conservative was canceled before execution.

Runtime routes:

| Mode | `GGML_POTAL_PAIRED_AUTO` | `GGML_POTAL_PAIRED_FENCES` |
|---|---|---|
| Serial | OFF | RTL |
| Paired RTL | ON | RTL |
| Paired Conservative | ON | CONSERVATIVE |

Profiling, raw tracing, probing, and diagnostic environment variables were
unset. `H7_REQUEST_EXECUTED` means that a real request and its runtime guards
completed; it is not a replacement for the independent numerical checks.

## 5. Backend Execution Evidence

### Observed E1/E2 command and completion counts

| Counter per 16+8 request | Serial | Paired RTL | Conservative |
|---|---:|---:|---:|
| HP1 MatMuls completed | 392 | 392 | 392 |
| Main LOOP_WS | 9,328 | 17,504 | 17,504 |
| Residual LOOP_WS | 7,455 | 97,922 | 97,922 |
| Paired stripes | 0 | 392 | 392 |
| Paired capacity fallbacks | 0 | 0 | 0 |
| Paired Index configs | 0 | 97,922 | 97,922 |
| Paired Index writes | 0 | 1,889,544 | 1,889,544 |
| Scale writes | 56,210,201 | 57,120,281 | 57,120,281 |
| Scale releases | 56,210,201 | 57,120,281 | 57,120,281 |
| Paired explicit Fences | 0 | 392 | 228,940 |
| Paired drains | 0 | 115,818 | 344,366 |
| Radix reconstructions / merges | 392 / 392 | 392 / 392 | 392 / 392 |

HP1 failures, GEMM failures, stripe-pipeline failures, and compensation failures
were zero in every mode. Last hardware status was zero. The issuer was ready
on Linux CPU 1. Each request completed the expected `49 * 8` weighted MatMuls,
not just device registration. Scale writes/releases balance.

These are existing software counters for commands actually issued/accepted by
the RISC-V hardware path, not independent FPGA cycle counters. In
`ggml/src/ggml-potal/potal.c:778`, the RISC-V dispatch executes the RoCC
instructions and returns checked Scale/Index/Fence statuses. The backend
propagates a failed MatMul instead of substituting CPU computation
(`ggml/src/ggml-potal/ggml-potal.cpp:142`).

Capacity refusal is a Paired-to-Serial **hardware** route, not a CPU fallback
(`ggml/src/ggml-potal/matmul/stripe.cpp:724`). No such refusal occurred in E1/E2.
Serial naturally has no Paired Index counts; its hardware Residual GEMMs are
recorded separately.

### CPU operation accounting

**Attention is not fully offloaded.** Q/K/V and O projections, FFN weights, and
the output projection use PoTal HP1 MatMul. QK-transpose, Softmax, Attention-times-V,
and KV-cache storage remain on CPU. The runner sets `offload_kqv=false`
(`ggml/src/ggml-potal/tests/potal-full-inference.cpp:379`), and the current backend accepts only
Q8_HP1-weight x F32-input -> F32 2D MatMul
(`ggml/src/ggml-potal/matmul/dense.cpp:110`). The floating-point attention MatMuls do not meet that
contract; enabling a flag alone would not add this missing backend support.

Original E1/E2 operation arrays are empty because of the collector callback
issue in section 9. Their `cpu_hp1_matmul=0` fields must not be interpreted as
measured CPU counts. The corrected Serial E3 capture contains actual assignments
for all 32 forward passes:

| Executed MatMul node occurrences | Total | Per forward |
|---|---:|---:|
| POTAL, Q8_HP1 left operand | 1,568 | 49 |
| CPU, F16 attention left operand | 768 | 24 |
| CPU HP1 MatMul fallback | 0 | 0 |

Other recorded CPU node occurrences were ADD 3,136; MUL 800; NORM 800;
CPY 768; CONT 1,536; RESHAPE 1,536; VIEW 1,152; PERMUTE 768; TRANSPOSE 384;
GET_ROWS 128; SOFT_MAX 384; and UNARY 384. VIEW/RESHAPE and related metadata
nodes are not separate arithmetic or hardware commands. POTAL also owned
1,536 metadata VIEW nodes.

These direct CPU counts apply to the completed Serial E3 capture, not to the
uncaptured primary requests or the interrupted Paired E3 request. The original
E1/E2 capture limitation remains disclosed rather than relabeled as zero work.

The host-only collector check observed 73 MatMuls per forward and every
prefill/decode step, while recording zero hardware completions. This validates
the observer, not the U280 CPU-operation count. Metadata-only VIEW nodes may
be assigned to a POTAL backend even in a CPU reference; that assignment alone
does not imply hardware work.

## 6. Numerical Validation

### Independent reference and checkpoint identity

`potal-full-inference-reference.py` uses the existing cached original checkpoint
through Transformers 4.46.1 and Torch 2.5.1 on CPU FP32. It performs real prefill
and `past_key_values` cached autoregressive decode, checking cache growth on
every step. No package was installed. The original tensors were independently
matched to the FP16 GGUF as described in section 3.

The native llama.cpp FP16 CPU reference additionally checks the conversion and
tokenizer path. Its tensor buffers are forced to CPU and operation offload is
disabled; merely supplying an empty device list is insufficient because this
tree registers PoTal as an accelerator. CPU references recorded zero PoTal
MatMul completions.

### E1/E2 results

Each comparison covers all 50,257 logits at the last prefill position and at
each of seven successive decode steps.

| Reference vs candidate | Max absolute | RMSE | Max relative | Mean relative | Normalized RMSE | Tokens |
|---|---:|---:|---:|---:|---:|---|
| Serial vs Paired RTL | 0 | 0 | 0 | 0 | 0 | 8/8 |
| Serial vs Conservative | 0 | 0 | 0 | 0 | 0 | 8/8 |
| Independent FP32 vs Serial HP1 | 2.376983643 | 0.401450281 | 0.015825966 | 0.002627304 | 0.003172007 | 8/8 |
| Native FP16 vs Serial HP1 | 2.375701904 | 0.395514078 | 0.015989366 | 0.002576088 | 0.003125010 | 8/8 |
| Independent FP32 vs native FP16 | 0.068527222 | 0.018780524 | 0.000546924 | 0.000128322 | 0.000148392 | 8/8 |

Serial/Paired and Serial/Conservative logits are bitwise identical. No token
divergence exists in these eight steps. FP32 and quantized PoTal logits are
not bitwise identical, as expected; token agreement does not erase their
measured numerical differences.

For independent FP32 vs Serial, the last-prefill row has max absolute error
1.597015381 and RMSE 0.360592911. The first decode row has max absolute error
1.247238159 and RMSE 0.289093588. Full per-step metrics are retained in JSON.

Definitions:

- Relative error: `abs(candidate - reference) / max(abs(reference), 1e-6)`.
- Normalized RMSE: `RMSE / max(RMS(reference), 1e-12)`.
- Reductions use float64; captured values are raw little-endian float32.

The denominator floors define metrics, not acceptance thresholds. No existing
full-model logit-error acceptance limit was found. No arbitrary tolerance or
FP32 bitwise-equality condition was introduced. Cross-mode exactness is verified;
FP32 quantized-model quality beyond this trajectory is not certified.

All captured logits were finite, with zero NaN/Inf entries, and all captured
KV count/position checks passed. No hardware status indicated an execution
failure. There is no dedicated full-model integer-overflow telemetry in this
run; absence of NaN/Inf is not presented as proof about every integer
intermediate.

### E3 completed Serial comparison and canceled modes

Independent FP32 and native FP16 CPU references both completed 256+32 with
identical 32-token greedy trajectories. The native FP16-vs-FP32 comparison has
max absolute error 0.266256332, RMSE 0.067201201, and normalized RMSE
0.000545916. The completed Serial hardware capture was compared against both
the free-greedy FP32 reference and the already-prepared same-history replay.

The Serial E3 trajectory first differs from free-greedy CPU FP32 at
generated step 1 (zero-based): CPU selects 464 and HP1 selects 1135. Both
histories are still identical at that step: the same 256-token prompt followed
by token 198. The FP32 logits are -111.503051758 for 464 and -111.516830444 for
1135, a margin of 0.013778687. At the same step, HP1 logits are
-111.717155457 for 464 and -111.274169922 for 1135. Later free-greedy steps
have different input histories and cannot be treated as aligned numerical-error
samples. No tolerance or numerical algorithm was changed to hide this divergence.

Serial's captured 256+32 warmup completed with 1,568 successful HP1
MatMuls, no hardware/compensation failure, and no capacity fallback. Its
free-greedy token agreement with CPU FP32 is 2/32; this sequence-level result is
not a same-history logit-error measurement. Generated IDs:

```text
198 1135 635 905 326 262 11862 318
517 6942 621 262 584 734 16113 287
262 976 27039 13 220 198 1135 635
905 326 262 11862 318 517 6942 621
```

An independent CPU FP32 replay on this actual Serial history completed, with
all 32 input histories and KV lengths checked. Its next-token policy agrees
with the hardware on 28/32 same-history steps; the differing positions are
1, 8, 12, and 13. All 32 captured logit rows were compared with these
identical input histories. The replay is teacher-forced by the real hardware
output IDs; it does not replace the original free-greedy CPU reference or the
real hardware request.

| FP32 reference vs Serial HP1 | Same-history rows | Max absolute | RMSE | Max relative | Mean relative | Normalized RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Free-greedy reference, aligned prefix only | 2 | 2.073074341 | 0.619543379 | 0.014877664 | 0.003935890 | 0.004495578 |
| Replay on actual Serial input history | 32 | 9.421075821 | 1.881615686 | 6.040606290 | 0.020806962 | 0.014503456 |

For the last-prefill row, max absolute error is 2.073074341 and RMSE
0.711783468. For the first decode row, they are 1.526031494 and 0.510913195.
All 1,608,224 Serial E3 captured logits are finite. Every saved KV check passed;
the captured and three primary requests ended at 287 cached tokens,
maximum position 286.

These are recorded differences between FP32 inference and the quantized PoTal
configuration, including their different KV precision and implementations.
They are not declared an accuracy PASS under an invented threshold.

The interrupted Paired E3 run emitted the first 11 Serial-matching token IDs
but did not finish its captured request or write a full logit capture.
Conservative E3 was not started. There is therefore no 256+32 cross-mode
bitwise-equivalence claim; the completed 16+8 cross-mode result remains valid.

## 7. Performance

All times below are **guest CPU-observed wall-clock milliseconds**, not pure
FPGA execution or waiting time. Loading and context creation are excluded from
request timing. No D1 Decode/FFN microbenchmark is substituted for inference.

Definitions implemented by the runner:

- Model loading: model-loader call.
- Weight preparation: existing backend preparation/allocation counters,
  reported separately and included in a cold first request.
- Prefill: `llama_decode` plus synchronized logits for the full prompt.
- First-token generation: the post-prefill scan/selection/output interval.
- TTFT: post-load request start, before KV reset, through the first token's
  flushed output. It includes reset/setup, prefill, and selection/output, so
  it is not merely prefill latency.
- TPOT: `(last_output_time - first_output_time) / (generated_tokens - 1)`.
  E1 uses seven intervals; E3 uses 31.
- Total: request start through the last token's flushed output.

### E1/E2 diagnostic samples

These are single cold capture requests, not the primary repeated baseline.
No median/MAD is inferred from one sample.

| Mode | Model loading ms | Context initialization ms |
|---|---:|---:|
| Serial | 1,418.418 | 170.334 |
| Paired RTL | 1,365.621 | 173.303 |
| Conservative | 1,358.551 | 170.837 |

| Mode | Prompt | Generated | Prefill ms | First generation ms | TTFT ms | TPOT ms | Total ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Serial | 16 | 8 | 7,699.017 | 2.641 | 7,724.151 | 1,602.768 | 18,943.530 |
| Paired RTL | 16 | 8 | 10,059.490 | 2.634 | 10,084.622 | 3,118.422 | 31,913.576 |
| Conservative | 16 | 8 | 10,100.483 | 2.682 | 10,125.618 | 3,156.459 | 32,220.834 |

Prepared-weight packing was 3,524.295 / 3,527.284 / 3,524.506 ms respectively;
allocation was 708.350 / 726.995 / 717.870 ms. Packing and allocation counters
must not be assumed disjoint without their source definitions. They are not
added to total latency again.

### E3 primary results and stopped coverage

Completed-mode cells below are **median / unscaled MAD in milliseconds** over
three complete primary requests, excluding the captured warmup.

| Mode | Prompt tokens | Generated tokens | TTFT | TPOT | Total time |
|---|---:|---:|---|---|---|
| Serial | 256 | 32 | 39,506.765 / 28.902 | 1,648.501 / 1.682 | 90,581.409 / 23.233 |
| Paired RTL | 256 | 11/32 in interrupted capture | Not measured | Not measured | Interrupted by user request |
| Paired Conservative | 256 | 0/32, not started | Not measured | Not measured | Canceled before execution |

The original protocol used one captured request as a separate warmup/accuracy
sample, excluded from primary timing, followed by three uncaptured requests
reusing the model/context and prepared weights while clearing KV state.
Only Serial completed this protocol. Scheduler callbacks were disabled for its
three primary requests. Actual primary counts are Serial 3, Paired 0, and
Conservative 0. No statistics are fabricated for the canceled modes.

All three Serial primary requests completed with capture disabled and the
Serial process returned 0. Every 32-token output sequence exactly matches the
captured warmup. Each completed 1,568 hardware HP1 MatMuls, with 1,568
prepared-weight cache hits, zero cache misses/packs/validations, and status 0.

| Serial primary request | TTFT ms | TPOT ms | Total ms |
|---|---:|---:|---:|
| 0 | 39,806.803 | 1,665.322226 | 91,431.792 |
| 1 | 39,477.863 | 1,648.501484 | 90,581.409 |
| 2 | 39,506.765 | 1,646.819710 | 90,558.176 |

For E3 Serial, model loading was 1,445.421 ms and context initialization was
173.680 ms, outside request timing. Primary prefill median/MAD was
39,481.304 / 28.981 ms; post-prefill first-token selection/output was
2.271 / 0.036 ms. The 93 post-first-token intervals across the three requests
ranged from 1,550.596 to 1,890.177 ms; TPOT above is the median/MAD of the
three per-request means, each over 31 intervals.

The completed Serial captured warmup measured TTFT 45,311.837 ms, TPOT
1,716.266 ms over 31 intervals, and total 98,516.096 ms. These are diagnostic
warmup timings, not primary samples. Its existing counters recorded 49 weight
packs, 41,890 Main loops, 34,791 Residual loops, and 242,848,783 Scale writes
and releases each. The same loop and Scale command counts occurred in each
completed primary request. The real RISC-V capture passed the nonempty-observation
guard at every step; its actual operation counts are recorded in section 5.

Existing Scale/Issue/Fence timers are CPU-observed API intervals. Pipeline wall
time and these stage intervals can overlap; they must not be summed as
independent contributions or labeled as hardware wait time.

The first boot/smoke/three-bringup suite consumed 3,983.3 host wall seconds.
The interrupted E3 simulator session reported 16,878.1 host wall seconds.
Its simulator cycle/performance summary concerns the whole emulation and the
fastest target clock, not an individual inference kernel.

Optional 1+32 and 32+32 requests were not run: long emulation time first favored
the primary workload, and the user then ended testing. Multi-row prefill and
single-row cached decode were exercised by the completed 16+8 and Serial
256+32 requests. The user subsequently requested `--no-warmup` for future
inference launches; historical warmup samples remain identified and excluded
from the Serial primary statistics. No new inference was launched after the
stop request.

## 8. Regression

Nine relevant existing Host binaries passed with actual exit code 0:

```text
potal-api-test
potal-trace-test
potal-residual-unit-test
potal-paired-plan-test
potal-paired-schedule-test
potal-paired-unit-test
potal-paired-auto-test
potal-paired-execution-test
potal-stripe-pipeline-test
```

Injected error/cleanup cases in these tests are expected assertions, not
suppressed failures. The tests were executed directly rather than treating
an empty CTest registry as successful testing.

The guest `potal-paired-smoke --quick` passed before full inference. Its PASS
is recorded separately and is not used as proof of full-model success.
The three real 16+8 requests each returned 0, and the outer suite returned 0.

The first guest was synced and powered off normally. The kernel printed
`reboot: Power down`; the simulator returned 0 and printed:

```text
Simulation complete.
*** PASSED *** after 2502046958252 cycles
```

The E3 simulator was interrupted at the user's request during Paired's captured
request. It exited 0 and printed `*** PASSED *** after 2984315231417 cycles`,
but no guest `Power down` or complete E3-suite exit marker was present.
That simulator PASS is not an inference PASS or proof of graceful guest shutdown.

A cleanup-only boot then ran `sync; poweroff`, without loading or executing a
model. Its kernel printed `reboot: Power down`; the simulator exited 0 and
printed `*** PASSED *** after 2991664699907 cycles`. The filesystem was then
mounted read-only on the host solely to extract the existing results and was
unmounted afterwards. All started simulator sessions are finished.

The cleanup boot warned that the filesystem was unchecked and had no journal.
A full filesystem check was not performed in this reduced-scope closeout;
the existing files were readable and the final sync/poweroff completed.

## 9. Issues and Fixes

| Finding | Classification | Bounded action / result |
|---|---|---|
| Existing model called "163M" but actually 124,439,808 parameters | Asset identity | Audited actual data and retained the existing checkpoint |
| Copied rootfs had no free space | Guest resource/staging | Preserved original; checked and expanded only the H7 copy |
| FP32 GGUF converter raises `gguf.SpecialVocab` AttributeError | Reference tooling | Used installed Transformers/Torch with the verified original cached checkpoint; did not change converter or models |
| Decode resets directly installed scheduler callback | Test instrumentation | Changed only acceptance-runner callback wiring; empty MatMul observations now fail explicitly |
| Extraction copy preserved root ownership on its directory | Host staging | Used privileged copy for its UART log; unmount trap completed; no model/backend change |
| Fallback clang LSP lacks standalone runner include paths | Editor configuration | Actual native and RISC-V `-Wall -Wextra` compile/link succeeded; LSP result is not claimed clean |

The callback cause is explicit in `src/llama-context.cpp:982`: each decode graph
reloads the scheduler callback from context parameters. The version-bound
standalone test owns a non-const context and changes only its callback fields
between synchronized requests using the existing internal accessor. It changes
neither production llama.cpp nor PoTal code. Native verification observed all
73 MatMuls at each of eight capture steps and then completed three requests
with observation disabled.

The user relaxed verification strictness, requested `--no-warmup` for future
launches, and then explicitly ended testing. Remaining work used only existing
captures, counters, source-preservation checks, and cleanup. No additional
inference, regression rerun, edge-case sweep, or measurement repetition followed
that stop request. The proposed helper flag addition was canceled rather than
triggering another build or test.

H7 project additions:

```text
H3-7A-llama-full-inference-baseline.md
ggml/src/ggml-potal/tests/potal-full-inference.cpp
ggml/src/ggml-potal/tests/potal-full-inference-reference.py
ggml/src/ggml-potal/tests/potal-full-inference-compare.py
```

These are intentionally not committed. No PoTal quantization, Scale API,
planning, validation, deep-copy, weight-cache, RTL, Gemmini-submodule, or
`software/host-init.sh` implementation was changed for H7. Final preservation
checks passed: HEAD is still the D1 checkpoint, the index is empty, and tracked
diffs are empty. Both original model hashes and the production library hash
match the initial snapshot. Protected additional hashes:

```text
potal.c:
7dec7c2a9cfc0bc12cb3dcdb9abd93302c0dc69e813a812f6d332c87e16850bd
software/host-init.sh:
e96d846238946925fd4a1c52f9b5e24ea144e15b3d748d15f8420e1c71c70834
Gemmini existing binary diff:
407b6271652f19485acaa919b81e5fb697083ac73a894d7b217de883c9e71b84
```

Gemmini HEAD remains `185293e87cbbc2a336410f49a3fd73d01ed8d04e`; its pre-existing
changes are preserved. `models/gpt2/` remains untracked and was never staged.

Evidence locations:

- First guest UART: `/tmp/h3-7a-run/uartlog`.
- Extracted E1/E2 requests: `/tmp/h3-7a-results/bringup/`.
- CPU references and comparison JSON: `/tmp/h3-7a-host/results/`.
- Original-parameter identity checker: `/tmp/h3-7a/check-checkpoint.py`.
- Historical E3 guest script: `/tmp/h3-7a/guest-primary.sh`.
- Interrupted E3 UART: `/tmp/h3-7a-primary-run/uartlog`.
- Cleanup-only UART: `/tmp/h3-7a-cleanup-run/uartlog`.
- Final extracted files and all three UART logs: `/tmp/h3-7a-results/final/`.

For each capture prefix, `.json` records prompt IDs, generated IDs, per-step
KV state, and counters; `.tokens` records generated IDs; `.logits.f32` stores
the full vocabulary row for every sampled step. Comparison JSON records
per-step errors and divergence, rather than only a summary token sequence.

## 10. Recommendation

**Hold the D2 implementation decision; no D2 work was started.** E1/E2 and
Serial E3 establish real hardware full-inference evidence, but the long-sequence
Paired/Conservative comparison was stopped and FP32 fidelity has only been
quantified, not accepted against a pre-existing model-quality threshold.

Paired E1/E2 has substantially more Residual LOOP_WS and Index work than Serial,
and Conservative adds many explicit Fences. These observed counts identify
additional actual work; they do not prove how much end-to-end time D2 can
recover. Full-inference planning time has not been separately isolated with
diagnostics OFF, so the D1 microbenchmark planning percentage must not be
substituted for a measured percentage of this inference request.

The overall result is **B -- Partial Verification**, not a hardware-failure
verdict: completed full-inference cases succeeded, while the requested complete
E3 cross-mode coverage remains incomplete by user choice. Serial performance
measurement is complete at n=3; the full three-mode performance table is not.
Attention core arithmetic remains CPU-side. No new experiments are queued;
any later continuation should use the user's requested no-warmup mode.
