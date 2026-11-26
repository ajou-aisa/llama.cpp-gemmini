# H3-8D2: checked next-K prefetch and physical observation

Date: 2026-10-29.

## Result

The explicit DIM32 Prefetch route passed native contract/ownership tests,
nine actual FPGA raw numerical cases, physical cross-pair overlap, attention
and PV-range regression, and matched repeated Decode/FFN benchmarks.
The successful full-row fixture measured **103 target-clock cycles** with
an actual B SRAM DMA write and a useful MAC on the opposite pair's observed
PE(0,0): 31 in one direction and 72 in the other. Its H2 and K-tile controls
measured zero. The two smaller fixtures still measured zero.

This is a bounded physical witness, not a claim about every shape, the entire
array, DMA duration, or the benchmark workloads' overlap.

**Prefetch is slower than K-tile on both measured workloads:** steady graph
latency increases 7.071692% for Decode and 5.116037% for FFN. Keep it opt-in;
do not promote it as the preferred performance route. Default H2 remains.
No full-model inference, TTFT or TPOT measurement is claimed here.

```sh
export GGML_POTAL_PAIRED_AUTO=ON
export GGML_POTAL_PAIRED_KTILE=ON
export GGML_POTAL_PAIRED_PREFETCH=ON
```

Invalid selection, missing K-tile/Paired prerequisites, DIM16, or an
unsupported candidate plan is refused rather than hidden by Serial fallback.
Unset/OFF preserves the default. D1/D2 changes remain uncommitted after the
authorized H3-8 checkpoints; no push occurred.

## Actual mechanism

The existing H2 plan remains the arithmetic and dependency manifest. The
executor reserves the opposite, already-retired pair's next K panel before
issuing the current panel's last Residual LOOP_WS. Immediately after that
LOOP_WS, it submits the prepared B transfer. It does not prefetch across J
groups or overwrite a live residency.

The checked C11 interface binds the real immutable B allocation and checks
all panel bounds before reservation. `potal_paired_h2_panel_begin(...,0,...)`
prepares an unused reference; `potal_paired_h2_panel_prefetch(session,ref)`
loads that exact owner/generation/pair. The original combined
`panel_begin(...,1,...)` remains supported through the same emitter.

```text
B base = (2*pair + 2)*BANK_ROWS - Ktiles*Jtiles*DIM
SP row = B base + (k*Jtiles + j)*DIM
DRAM   = original_B + (k_begin + k*DIM)*ldb + j_begin + j*DIM
```

Each prefetch uses ordinary CONFIG_LD state 0 and tile MVINs. After a checked
global drain publishes B, Main still supplies its real ordinary descriptor
to the existing encoder. Only the encoded B address is suppressed. No
fabricated pointer, new resident ISA command, DIM16 resident capability, or
unchecked escape was introduced.

Main and Residual retain their own I shapes. Original K32 carriers, compact
fragment order, partial/ACC saturation, Main/Residual ACC epochs, readback,
INT64 reconstruction and F32 merge remain unchanged.

### Ownership and completion

- Pending ownership precedes the first CONFIG/MVIN, including failed attempts.
- Preparation is not readiness. Queries and physical-progress observations
  cannot publish a panel.
- Only successful checked Fence or valid Scale Release publishes a fully
  issued load and completes issued readers.
- Retirement requires every declared reader completed, no global pending
  work, closed Index transaction and no live Scale columns. Saved retirement
  receipts prevent an old logical retire from clearing a newer reservation.
- Pure validation/BUSY refusals retain prepared identities for corrected
  continuation. Preparation is not cancellable and does not relax close.
- Answered failures allow one cleanup drain and actual live-column release.
  Cleanup discards both current and prepared identities. Unknown completion
  is tracked separately from the first sticky status and retains ownership
  and DMA dependencies without retry or fallback.
- Native no-sink admission fails before allocating managed state. A sink
  returning NO_HARDWARE after a command attempt is an effectful failure, not
  a no-effects shortcut.

### CPU ordering and observation effects

The initial attempts passed numerical tests but measured no physical
intersection. Investigation found two serialization paths:

1. `potal_cpu_fence_rw` emits `fence rw,rw`. Rocket stalls decoded FENCE
   while RoCC is busy (`RocketCore.scala:354-358`); PoTal's busy signal
   includes generators, reservation station and scratchpad.
2. The configured guest VDSO's CLOCK_MONOTONIC read uses `smp_rmb`, compiled
   as `fence r,r`. Thus a clock query between Residual issue and prefetch
   can also destroy the intended overlap.

The B publication fence now executes once at idle managed begin, justified
by the entire allocation's immutable-until-close contract. A publication
fences remain in ordinary loop issue. Actual PoTal Fence/Release semantics
are unchanged.

The last Residual and its prefetch have no clock query between them. Their
combined CPU-observed interval is `residual_prefetch_issue_s`; it is included
once in issue time, not counted again as standalone Residual time. It is
not hardware DMA/compute time. Pre-issue live-slot scanning was removed,
and descriptor validation/reservation moved before current GEMM issue.

LoopMatmul still forwards raw commands only after configured generators
finish issuing. This is not necessarily physical Execute completion.
No RTL admission bypass or Release relaxation was added.

## Observation image and verification

The separate image is
`FireSimQuadRocketSinglePoTalD32PrefetchObservationConfig`, with unchanged
DIM32/SP4x2048/ACC2x256/Scale256/Index64 geometry and 30 MHz host target.
Observation wiring is opt-in; the arithmetic datapath is not changed.

COUNTER126 query `0x504f425300000000` must return
`0x504f425301200007`. Software clears legacy counter0 before probing, so an
unsupported image's old counter value cannot masquerade as capability.
The checked response path preserves all 64 bits.

Selectors 45..51 observe B0/B1 physical SRAM DMA writes, pair0/pair1
qualified PE00 MAC, the two cross-pair intersections, and their union.
Writes require the DMA-selected physical port and a nonempty actual mask.
Operand provenance follows buffered handshakes, mesh latency and stationary
weight registers. Preload-only, flush, padding, garbage, zero-fill,
same-pair, empty-mask and unsupported transpose/ACC/im2col cases cannot
satisfy the cross-pair event.

| FPGA fixture | H2 overlap | K-tile overlap | Prefetch overlap | Prefetch loads / resident Main |
|---|---:|---:|---:|---:|
| M17/N65/K544, consecutive sparse runs | 0 | 0 | 0 | 8 / 8 |
| M17/N65/K512, gapped sparse runs | 0 | 0 | 0 | 6 / 6 |
| M63/N129/K544, I2/J4/K4, 63 Residual rows | 0 | 0 | 103 | 8 / 8 |

All nine raw Main/Residual INT32 cases match independent wide arithmetic and
explicit saturation. The full-row case checks 8127 values in each output;
the smaller cases check 1105 Main and 130 Residual values each.
For the full-row fixture, all three routes count B writes 1408/1312 and
qualified MAC cycles 5544/5166. Prefetch alone adds intersections 31/72.
These counts are not DRAM bytes, full-array utilization or elapsed stage time.

Final native gates pass 11/11 for DIM16 and 11/11 for DIM32, including
prepared-reference, partial-issue, stale/reentrant, cleanup/quarantine,
no-sink owner release, exact command/numerical and fallback regressions.
Native command emulation is software-only.

RTL observation/capability tests pass 4/4; production DIM32 ingress passes
1/1; DIM32 real writeback/SRAM/ROB drain tests pass 3/3. The original
eight-slot drain fixture still fails its three scenarios because its helper
uses 256-slot key offsets. Those failures are preserved, not edited green
or described as a full-suite pass.

Actual FPGA API smoke, full attention smoke and PV-range-only smoke pass.
The PV-range witness reports exact=1, status=0, 49 jobs, 16 bitplane jobs,
11 corrections, 22 bias MVINs, 49 LOOP_WS, 71 fences, and 28224 balanced
Scale writes/releases.

Routed timing: WNS +0.029 ns, WHS +0.010 ns; zero failing endpoints,
zero no-clock checks and zero unconstrained internal endpoints. The
historical generated-XDC and address-map warnings recur and remain recorded.

## Matched benchmark

All routes use the same final binary, observation image, seed 20261023 and
unchanged `bench_one` input generation. Actual cache settings retain
LOG_DEBUG=0, LOG_CYCLE=1, CYCLE_DETAIL=0, CYCLE_SIM=0,
POTAL_PLANNING_DIAGNOSTICS=OFF, EXSIA and compensation ON.
Primary runs disable Raw Trace and Paired/planning/scheduler profiling.
Guest CPU1 maps to Hart0; direct API tests are pinned there and the existing
single-issuer affinity checks remain.

```sh
./bin/potal-paired-smoke --prefetch-observe
./bin/potal-paired-smoke --bench-prefetch --case decode_1x768x768 --repeat 7
./bin/potal-paired-smoke --bench-prefetch --case gpt2_ffn_up_16x3072x768 --repeat 7
```

Seven cold and seven steady samples per route, rotating/reversed order.
All 84 primary samples are exact, Scale counts balance and unexpected
fallback is zero. Median/MAD/min/max were independently recomputed for every
graph and pipeline summary. No outlier was removed. Separate
POTAL_PAIRED_PROFILE=1 runs add 36 exact samples and are not mixed into the
primary table.

All times are guest milliseconds, not host simulation duration or pure
accelerator time.

| Workload / route | Steady graph median / MAD | Graph min..max | Steady pipeline median / MAD | Cold graph median / MAD |
|---|---:|---:|---:|---:|
| Decode H2 | 17.908 / 0.028 | 17.880..18.914 | 17.331 / 0.025 | 33.169 / 0.013 |
| Decode K-tile | 14.353 / 0.030 | 14.322..14.426 | 13.762 / 0.011 | 29.596 / 0.019 |
| Decode Prefetch | 15.368 / 0.053 | 15.302..15.522 | 14.763 / 0.031 | 30.546 / 0.043 |
| FFN H2 | 92.861 / 0.142 | 92.719..93.256 | 92.201 / 0.221 | 154.276 / 0.132 |
| FFN K-tile | 73.123 / 0.124 | 72.695..73.323 | 72.489 / 0.115 | 134.249 / 0.147 |
| FFN Prefetch | 76.864 / 0.077 | 76.425..76.942 | 76.164 / 0.077 | 137.974 / 0.099 |

Within this experiment, K-tile reduces H2 latency by 19.851463%/21.255425%;
Prefetch reduces H2 latency by 14.183605%/17.226823% but regresses K-tile.
Do not combine these percentages with earlier binaries/images' results.

| Steady counter | Decode H2 / K-tile / Prefetch | FFN H2 / K-tile / Prefetch |
|---|---:|---:|
| Main LOOP_WS | 12 / 12 / 12 | 48 / 48 / 48 |
| Residual LOOP_WS and Index Config | 60 / 60 / 60 | 288 / 288 / 288 |
| Index Write | 1830 / 1830 / 1830 | 11496 / 11496 / 11496 |
| Scale Write = Release | 33024 / 19968 / 19968 | 147456 / 73728 / 73728 |
| Readbacks / tiles | 12/48 for all | 48/192 for all |
| Explicit Fence | 1 / 55 / 55 | 1 / 289 / 289 |
| Logical drain boundaries | 73 for all | 337 for all |
| Prefetch loads / resident Main | 0 / 0 / 6 | 0 / 0 / 24 |
| Raw prefetch MVIN attempts | 0 / 0 / 192 | 0 / 0 / 768 |
| Prefetch command-declared bytes | 0 / 0 / 196608 | 0 / 0 / 786432 |

The last row is a logical descriptor extent, not measured bus traffic.
No GEMM, Index or Scale reduction beyond D1 occurs.

CPU-observed steady Scale API medians are 8.957/5.378/6.171 ms for Decode
and 39.863/19.784/22.543 ms for FFN. Issue API medians are
0.641/0.639/0.741 and 3.652/3.671/4.164 ms. The combined
Residual+Prefetch intervals are 0.080 and 0.362 ms, already included in
Prefetch issue time. These observations are consistent with added managed
checks/raw submissions outweighing the small overlap; they do not uniquely
attribute the regression to hardware waiting. No overlapping intervals are
summed into graph wall time.

## Artifacts, failures and reproducibility

Evidence root: `/tmp/h3-8d2`.

- Final output: `extracted-final/results`, `run-prepared/uartlog`,
  `final-results.sha256`. Guest validation and simulator both exit 0.
- Native/cross gates: `prepared-prefetch-gates.log`.
- RTL gates: `rtl-first.log`, `rtl-dim32-drain-first.log`.
- Source/binaries: `accepted-software.patch`, `accepted-artifacts.sha256`,
  `rtl-source.patch`, `rtl-new-tests.tar.gz`.
- Final smoke SHA256:
  `c266e738dfcf4936d3fb6997ee1f03c2bb5dee813e67be441eda3032c59fa730`.
- Final libggml-potal SHA256:
  `820a25045f296f917bb40d3186c748722ee3062e97e5aa54e09765ca7981afd5`.
- Header SHA256:
  `3580b84f15ed8de9f9d311edea64eaa7c1f9bb308b1552a8f79453ec2c05ea33`.
- Bitstream SHA256:
  `aac9dc7b1b2c9932c09e758d8fb3e67271e0e5d683ee1b24c1c02ed544c4e83d`.
- Packaged image SHA256:
  `0c35bf90abe56714e47ec25818c40be96760eb9091360536343c1fc42bd1a46a`.
- Driver SHA256:
  `ba77be72324070ae7d8336780bafb7d44bcc61951e08e7eccf8ecc35969adae0`.
- Complete Vivado build: `/home/alveo/build_firesim_h3_8d2_d32`.
- Runtime slot: `/home/alveo/FIRESIM_RUNS_DIR_H3_8D2_D32/sim_slot_0`.

The manager initially exited 1 while duplicating completed Vivado results:
disk full, not synthesis/timing failure. The incomplete new duplicate was
moved intact to `/dev/shm/h3-8d2-incomplete-results` (temporary RAM storage);
the complete authoritative build and failure logs remain on disk. No prior
artifact was removed. The existing bit/MCS was packaged in the official
layout, gzip-checked and extracted byte-for-byte against the originals.
The recovered package and corrected infrasetup pass.

Other first failures are retained in `first-failures.md`: required CMake
DIM variables, a native-only missing smoke target, the old drain fixture,
an interrupted launcher with bad exit-status capture, missing workload
directory, the test wrapper's missing generation pointer, and each
numerically exact zero-overlap attempt. Each hardware attempt has its own
UART log, extracted results and frozen rootfs. Failed physical gates never
ran the subsequent benchmarks.

No full-model rerun or DIM32 E4C1 hardware claim was added. The existing
DIM16-only resident extension remains separate. The result is a safe,
physically witnessed experimental prefetch mechanism, not a performance
promotion over D1.
