# H3-8D0: matched DIM16/DIM32 Performance baseline

Date: 2026-10-29.

## Result and checkpoint

Both actual FPGA runs passed the direct Main/Indexed Residual golden,
production Paired smoke, and both repeated microbenchmarks. DIM32 is slower
on these matched H2 workloads: steady Decode graph latency increases
5.681676%, and FFN increases 8.015729%. This negative result is retained.
Neither array size nor fewer readback tiles establishes a speedup.

Authorized checkpoints, both on `potal`, without push:

- Software: `df8482e9fec038dd1f5db48c8b3ab9c1c11c3df9`,
  `feat(potal): validate DIM32 FireSim full-model inference`.
- RTL: `8c839ea348a42398da6822440209955b3a015ff8`,
  `feat(potal): add DIM32 quad-Rocket FireSim target`.

The software commit contains 35 reviewed files. The RTL commit contains
eight configuration/test files. H3-8D's historical E4C1/E4C2 document remains
a plan. No new optimization is included in either checkpoint.
`/tmp/h3-8d-checkpoint.md` records inclusion, excluded working-tree changes,
header/bitstream/report hashes and validation limits.

Fresh checkpoint validation passed 11 native tests per DIM and the profile
regression. All five preserved artifact/report manifests matched. Historical
RTL evidence is 34 passing tests in six suites, not a full-suite pass.
The documented reduced-cache ingress failures remain. The changed drain
harness has compilation evidence, not execution proof in those six suites.

## Matched method

Both new builds use software commit `df8482e9`, the same compiler, seed
20261023, and the unchanged `bench_one` input generator. Weight generation
precedes activation generation. No model, prompt, HP1 packing, ExSIA folding,
RMD arithmetic, saturation or tolerance changed.

Builds:

```text
build-potal-h3-8d0-d16-performance
build-potal-h3-8d0-d32-performance
```

Both actual CMake caches confirm:

```text
LOG_DEBUG=0 LOG_CYCLE=1 CYCLE_DETAIL=0 CYCLE_SIM=0
POTAL_PLANNING_DIAGNOSTICS=OFF
GGML_GEMMINI_ACTIVATION_QUANT=EXSIA
GGML_GEMMINI_ENABLE_RMD=ON
GGML_GEMMINI_DEFAULT_MATMUL_MODE=STRIPE_PIPELINE
GGML_POTAL_PAIRED_AUTO=ON
```

Primary measurements unset Raw Trace, scheduler debug, planning profile and
Paired profile. The Paired route uses `GGML_POTAL_PAIRED_FENCES=RTL`.
Serial and Conservative Paired are additional controls; they are not mixed
into the H2 row. Each route has seven cold-cache and seven steady-cache
samples per workload. Route order rotates/reverses. Untimed warmups and
every measured output are checked against the existing golden.

The unchanged CLI is:

```sh
./bin/potal-paired-smoke --bench --case decode_1x768x768 --repeat 7
./bin/potal-paired-smoke --bench --case gpt2_ffn_up_16x3072x768 --repeat 7
```

The nominal target clock remains 1 GHz; the preserved FPGA implementations
use approximately 30 MHz host clocks. These tables use guest steady-clock
graph intervals and backend pipeline wall time, not host-run duration or
accelerator cycles. Canonical guest monotonic TTFT/TPOT code is unchanged.
No full GPT-2 256/128 rerun occurred.

Both guests confirm Linux CPU1 is Hart0. Direct API gates run on CPU1;
the backend's issuer reports CPU1. The inherited shell mask is `0xF`.
The existing producer-affinity guard excludes the issuer CPU
(`matmul/execution.cpp:626-669`). These microbenchmarks do not instantiate
the full model's three-worker token-generation pools.

## Measured graph latency

All values are milliseconds, shown as median / MAD.

| Workload and route | DIM16 steady | DIM32 steady |
|---|---:|---:|
| Decode Serial | 10.027 / 0.019 | 10.349 / 0.043 |
| Decode H2 Paired | 15.946 / 0.041 | 16.852 / 0.022 |
| Decode Conservative Paired | 16.219 / 0.028 | 17.069 / 0.039 |
| FFN Serial | 54.816 / 0.207 | 55.548 / 0.055 |
| FFN H2 Paired | 82.650 / 0.138 | 89.275 / 0.386 |
| FFN Conservative Paired | 83.744 / 0.165 | 90.369 / 0.455 |

| H2 measurement | DIM16 | DIM32 |
|---|---:|---:|
| Decode cold graph median / MAD | 31.400 / 0.043 | 32.181 / 0.014 |
| FFN cold graph median / MAD | 145.676 / 0.209 | 150.592 / 0.155 |
| Decode steady pipeline median / MAD | 15.364 / 0.024 | 16.262 / 0.018 |
| FFN steady pipeline median / MAD | 81.983 / 0.140 | 88.610 / 0.348 |
| Decode activation quantization median | 4.017 | 4.087 |
| FFN activation quantization median | 10.573 | 10.983 |
| Decode Scale API aggregate median | 8.143 | 8.118 |
| FFN Scale API aggregate median | 36.227 | 36.296 |

All 168 primary samples across both DIMs report exact output success.
Every per-sample Scale Write/Release count balances; unexpected Paired
fallback is zero. Six summaries per workload/DIM were independently
recomputed from individual samples, including graph and pipeline Median/MAD.
Serial being faster is retained, not used to silently change production H2.

## Command counts and geometry

These H2 counts are stable across the seven steady samples. They are command
counts and declared tile counts, not observed DMA traffic or hardware time.

| Counter | Decode D16 | Decode D32 | FFN D16 | FFN D32 |
|---|---:|---:|---:|---:|
| Main LOOP_WS | 12 | 12 | 44 | 48 |
| Indexed Residual LOOP_WS | 60 | 60 | 264 | 288 |
| Index Config | 60 | 60 | 264 | 288 |
| Index Write | 978 | 1830 | 6314 | 11496 |
| Scale Write = Release | 33024 | 33024 | 147456 | 147456 |
| ACC readback commands | 12 | 12 | 44 | 48 |
| ACC readback tiles | 96 | 48 | 384 | 192 |
| Explicit Paired fences | 1 | 1 | 1 | 1 |
| Paired drains | 73 | 73 | 309 | 337 |
| Unexpected fallback | 0 | 0 | 0 | 0 |

The same byte capacities give DIM16 4096 SP rows/bank and 1024 total ACC
rows, versus DIM32 2048 and 512. Consequently, one ACC half holds 32 DIM16
tiles but only eight DIM32 tiles. The tiler tests `(I+J)*K*DIM` SP rows and
`I*J*DIM` ACC rows, then grows J/I/K in that order
(`matmul/tiling.cpp:21-46,176-242`).

Paired K is additionally bounded by B bank capacity, Main A plus reserved
Residual A, and simultaneous Main/Residual Scale slots
(`matmul/paired-auto.cpp:360-467`). Source-derived schedules for these shapes
use I1/J9/K28 at DIM16 and I1/J4/K16 at DIM32. They imply two K windows,
six J groups for Decode, and 22 versus 24 J groups for FFN, matching the
observed Main command counts. These K/J values are source-derived; the
Performance benchmark does not print independent runtime K/J counters.

Index64 still holds only two physical fragments at DIM32, just as Index32
holds two DIM16 fragments. Padding/filler and full physical fragment width
can increase Index writes without reducing window count. Scale metadata
remains attached to K32 blocks and output columns, so doubling physical DIM
does not halve all Scale work. The exactly equal Scale counts above are
measured; these geometry rules explain why equality is possible.

## Separate CPU attribution runs

After primary timing, each workload ran with `POTAL_PAIRED_PROFILE=1` and
three repeats. The following medians cover 16 profiled Paired/Conservative
executions per workload, including warmups. They are not the seven primary
H2 samples and must not be subtracted from primary graph medians.

| CPU interval, ms | Decode D16 | Decode D32 | FFN D16 | FFN D32 |
|---|---:|---:|---:|---:|
| Residual decode | 0.1650 | 0.2410 | 1.9015 | 2.0835 |
| Route planning | 1.5950 | 1.8885 | 8.2970 | 10.9120 |
| Executor planning/validation | 0.8770 | 1.0325 | 5.5530 | 7.0985 |
| Residual staging pack | 0.0335 | 0.0410 | 0.2220 | 0.2500 |
| Scale Write API | 4.2425 | 4.3470 | 18.8320 | 19.4385 |
| Scale Release API | 3.8870 | 3.7690 | 17.3440 | 16.8545 |
| Index Config/Write API | 0.2000 | 0.3250 | 1.2415 | 2.1175 |
| Main issue API | 0.1495 | 0.1475 | 0.6900 | 0.6220 |
| Residual issue API | 0.1015 | 0.1015 | 0.5110 | 0.5285 |
| ACC readback issue API | 0.0180 | 0.0170 | 0.1405 | 0.1165 |
| Explicit Fence API | 0.0490 | 0.0485 | 0.2175 | 0.2425 |

Every profile record reports OK. The increased planning and Index API costs
are measured observations. Their share of graph latency, and whether a
command response contains hardware waiting, are not isolated by this run.
Scale Release retains global drain semantics; a small explicit Fence time
does not prove negligible hardware waiting.

The baseline does not separately expose radix reconstruction or final merge
duration with LOG_DEBUG=0. Completion/output checks cover their correctness,
but their time remains unavailable. No physical DMA/PE overlap, NPU cycles,
bus bytes or per-controller waiting was measured. Cycle JSONL contains
diagnostic intervals, including invalid `counter_regression` records across
CPU placement; none is converted into accelerator cycles or wall latency.
Overlapping intervals are never added to synthesize graph time.

## Relationship to the older Functional results

H3-8C recorded TTFT decreasing 4.796957%, TPOT increasing 6.914878%, more
Main/Residual loops and Index writes, more QK loops, and nearly equal Scale
counts. Those were different software revisions and one Functional run each.

The new matched experiment independently reproduces a Decode/FFN latency
regression, greater Index work, and equal Scale counts for these two inputs.
It does not reproduce TTFT or attention timing. Larger prefill work may use
the larger array more effectively, but that remains a structural hypothesis
for the old TTFT improvement, not a measured attribution.

QK keeps complete low/high passes and frozen doubled-K tile selection.
Its physical ACC capacity, J grouping and divisor-constrained K panels can
increase LOOP_WS count (`matmul/attention.cpp:521-535`). This explains a
mechanism consistent with the historical QK count increase; no new QK
performance claim is made. Attention arithmetic is unchanged.

## Reproduction, artifacts and first failures

Commands and environment cleanup: `/tmp/h3-8d0/run-benchmark.sh`.
Build logs: `/tmp/h3-8d0-d{16,32}-build-first.log`, both exit 0.
Actual caches and manifests are in the two build directories above.
Separate runtime configs are `deploy/config_runtime_h3_8d0_d{16,32}.yaml`;
separate slots are `/home/alveo/FIRESIM_RUNS_DIR_H3_8D0_D{16,32}`.
Existing H3-8C slots, bitstreams, rootfs and results were not overwritten.

Primary and attribution raw logs:

```text
/tmp/h3-8d0/d16/extracted/results/
/tmp/h3-8d0/d32/extracted/results/
/tmp/h3-8d0/d16/results.sha256
/tmp/h3-8d0/d32/results.sha256
/tmp/h3-8d0/matched-summary.json
/tmp/h3-8d0/artifacts-before-run.sha256
/tmp/h3-8d0/completed-evidence.sha256
```

Both guests and drivers exited 0 after sync/poweroff. Complete UART/driver
logs are `d16/run` and `d32/run-uart`.

The first DIM32 boot could not receive login: the launcher had not created
the UART input file before the driver's unchecked `open()`. The attempted
stdin workaround did not reach the file-backed UART. No workload ran in
that boot. Its logs and image were preserved, the launcher was corrected,
and a fresh staged image booted under `run-uart`. Details and original/
corrected script hashes are in `first-infrastructure-failure.md` and
`uart-recovery-scripts.sha256`. This was a launch failure, not a numerical
gate failure. No measured workload was rerun to discard a bad sample.

Read-only debugfs extraction reported inability to reproduce root ownership
under the unprivileged host user. File contents were extracted, completed
markers and all samples parsed, and hashes recorded. The ownership warning
does not establish a filesystem content failure.

The matched numerical and performance baseline is established. D1 may now
evaluate an explicit opt-in change against H2; no optimization, default
promotion or prefetch implementation is claimed by this report.
