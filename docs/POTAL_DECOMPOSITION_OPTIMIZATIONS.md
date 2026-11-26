# Decomposition preparation: candidate decisions (2026-11-16)

Baseline: `fbbcab5e6d4f44f1059a0a2e88cda633d691ce90`, now included in
`potal` through PR #83. This change retains candidates 1+5 together and enables
candidate 3 for packets with at least 32 source rows. These are PoTal preparation
changes, not CUDA kernels or optimizations of the floating-point baseline.

## Decisions

| Candidate | Decision | Evidence and scope |
|---|---|---|
| 1. Avoid decomposing W4 Attention upper limbs twice | Keep with 5 | Paired execution materializes only the main digit during folding. The upper digits are generated once during job preparation. Standalone timing was mixed; the combined path improved every measured case. |
| 2. Pack W4 residual digits directly into DMA storage | Exclude | Individual trials ranged from 22.2% faster to 5.2% slower. Product packing remains unchanged. |
| 3. Existing fast packet decoder | Keep for >=32 source rows, W4 and W8 | Packet build+decode fell 52.2-76.3% for 32-row cases. Sparse one-row W4 cases became 128.6-133.0% slower; one-row decoding retains the reference default. |
| 4. Cache compact K ranks instead of repeated popcounts | Exclude | Individual trials ranged from 22.3% faster to 9.0% slower. This experiment cached K ranks, not a new row-rank table. |
| 5. Generate execution-ready upper residual rows directly | Keep with 1, W4 Attention | Avoids all-selected bitmap creation, StripePacket construction and subsequent decoding. Preserves ascending block/K and lane/source-row order. |

The decoder override remains available: `GGML_POTAL_FAST_DECODE=ON` or `1`
forces fast decode; any other explicitly supplied value forces reference decode.
With the variable absent, rows 1-31 retain reference decode. This cutoff is a
measured policy, not a claim that 32 is optimal on Rocket.

## How W4 Attention changes

Previously folding generated all dense limbs. Job preparation then recovered the
upper value from the code and main digit, decomposed it again through the bitmap
builder, emitted a compact packet and decoded that packet back to execution rows.

The paired sequential and strict pipelines now keep only the main digit at
folding. For each source code, preparation computes
`upper = (code - main - bias) / 16`, where `bias=8` for PV and zero for QK.
It decomposes the upper value once into signed radix-16 digits. Nonzero events
record K, source row, lane and digit; maps then place them into the same dense
execution rows consumed by the existing planner. Each event occupies eight
bytes on the measured ABI. Empty residuals produce no residual work.

The public reference preparation path still materializes all limbs. No rounding,
anchor, clipping, PV bias, reconstruction, saturation order, weight preload,
Index/Scale command or fence semantics are changed. The new path uses scalar
integer operations and ordinary C++ containers, with no RVV or GPU dependency.
Direct residual construction applies to W4 Attention only. W8 and Linear retain
their existing packet producers and receive the row-gated decoder change.

## Measured preparation time

Local x86 CPU under WSL, Release GCC 13.3, DIM32/TOP3/A4W4. These are synthetic
inputs through the production preparation/planning/staging code. The timed scope
includes activation loading, local quantization, folding, upper residual
generation, plans and staging. Stationary preparation, GEMM/device execution and
output finalization are excluded. Each sample prepares four jobs with four live
slots; three timed repetitions follow warm-up. Nine alternating baseline/final
process pairs were run on one pinned CPU. Values below are medians.

The Attention translation unit disables loop/SLP vectorization; supporting
objects use their ordinary Release build. Floating-point fast math is disabled.
This is not a Rocket instruction or cycle simulation.

| Operation | Rows | Context length | Before (us) | After (us) | Time reduction |
|---|---:|---:|---:|---:|---:|
| QK | 1 | 128 | 660.15 | 541.08 | 18.0% |
| QK | 32 | 128 | 4457.91 | 1826.55 | 59.0% |
| PV | 1 | 128 | 270.81 | 194.01 | 28.4% |
| PV | 32 | 128 | 3023.16 | 884.67 | 70.7% |
| QK | 1 | 1024 | 4676.22 | 4140.57 | 11.5% |
| QK | 32 | 1024 | 8247.06 | 6241.80 | 24.3% |
| PV | 1 | 1024 | 1379.55 | 1106.85 | 19.8% |
| PV | 32 | 1024 | 23527.56 | 6195.15 | 73.7% |

QK uses K=128, N=context length; PV uses K=context length, N=64.
The digest covers packed A/B, residual digits, row/lane/K mapping and residual
staging. Every baseline/final digest matched. These improvements cannot be added
to the decoder microbenchmark percentages or interpreted as TTFT/TPOT gains.

For candidate 3, the separate benchmark times builder construction, emission,
packet sealing and decoding together. It uses rows 1/32, K=128/1024, residual
density 5%/100%, and W4/W8, with seven alternating pairs and 20 repetitions per
sample. All digests matched. At 32 rows W4 time fell 55.0-76.3% and W8 fell
52.2-68.3%. One-row results justify retaining the reference default there.

Raw measurements: [Attention](potal-decomposition-local.csv),
[packet decoding](potal-packet-decode-local.csv), and
[individual candidates](potal-decomposition-individual.csv).

## Correctness and limits

The W4 native golden suite passes 24 cases, including independent dense and
Attention arithmetic, sequential/strict execution, GQA, batches, stripe tails,
decode, failed fences and command counters. The added direct-upper oracle
compares against the existing builder/packet decoder for QK/PV, empty residuals,
INT32 extremes, nonzero row offsets, rows 1/3/33 and K=1/31/32/33/257.
The W8 residual unit suite also passes with the updated decoder dispatch.
A mutation that flips a generated upper digit makes the new oracle fail; the
mutation is confined to a separate evidence build, not product code.

Native tests use an emulated DIM32 hardware geometry fixture, SHA256
`a8104a56eb0c0322f38498fa401fdbea7391ebfac84c28432cb4ee03f7748616`.
It is not certification against a deployed FPGA bitstream. Full PPL, Nano timing,
Rocket TTFT/TPOT and target peak memory were not measured for this change.
The old main worktree and previous experiment branch are preserved.

Local scripts, source snapshots and logs are retained at
`../.omo/evidence/decomposition-candidates-20261116/` relative to the worktree:
`bench.cpp`, `build.py`, `run.py`, `run-final.py`,
`packet-bench.cpp`, `build-packet.py`, `run-packet.py` and the validation logs.
