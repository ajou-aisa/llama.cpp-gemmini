# H3-8F/G: matched performance and adoption record

Updated: 2026-10-30 (local +0900). Actual FPGA execution completed.

## Result and decision

Keep production H2 and all optimization defaults unchanged. The opt-in
Early LdB path is numerically correct and has measured physical overlap,
but its added benefit over F1 is small: steady median reductions of 0.31%
Decode, 0.66% FFN and 0.68% tail. Early remains 5.20% slower than K-tile
on Decode and 3.70% slower on FFN. Tail cold results do not improve.
These measurements do not justify default adoption or another full GPT-2
inference run. No full GPT-2 rerun was performed.

This completes a bounded opt-in experiment, not a claim that all prefetch
bottlenecks have been eliminated. The implementation preserves real
Fence/Release synchronization and immutable operand ownership.

## Source, image and execution provenance

- Software base: b5683e474b6b8708fe90f3d5a150a56322f64081, uncommitted changes.
- Chipyard base: bf1d98ca8a291a53845648ae475c80e01fd0b83d, scoped dirty changes.
- Gemmini pinned commit: 185293e87cbbc2a336410f49a3fd73d01ed8d04e; incoming dirty worktree preserved.
- Configuration: FireSimQuadRocketSinglePoTalD32EarlyPrefetchObservationConfig.
- Final corrected build: /home/alveo/build_firesim_h3_8g2_d32, 30 MHz host target.
- Package: /tmp/h3-8fg/firesim-g2.tar.gz.
- Package SHA-256: 8b1f652e8b5ba092e15c134babb332ee621888d2a6a396c4c8d6f8ba0bf99991.
- Image/driver/software/source hashes: bitstream-g2.sha256, driver-g2.sha256,
  staged-g2.sha256 and source-g2.sha256 under /tmp/h3-8fg.
- Program log: /tmp/h3-8fg/program-g2.log, exit 0 after explicit user approval.
- Rootfs: /tmp/h3-8fg/rootfs-g2.img; execution logs: /tmp/h3-8fg/g2-run.
- Guest result H3_8G2_EXIT=0; enclosing simulator exit 0 after clean poweroff.
- Extracted results: /tmp/h3-8fg/extracted-g2/results.
- Independent summary/order checks: /tmp/h3-8fg/analysis-g2.json.
- Raw-file hashes: /tmp/h3-8fg/results-g2.sha256 and analysis-g2.sha256.

The original architecture audit is H3-8F0-original-gemmini-prefetch-audit.md.
F1's earlier separate-image experiment is H3-8F1-managed-completion-optimization.md.
The command/ownership/RTL contract and failed first-run history are in
H3-8G-gemmini-based-early-prefetch-implementation.md. Outer FireSim metadata
alone does not identify the dirty source; the source snapshots above do.

## Matched comparison contract

All five routes use one final g2 image and the same executable/tensors
within each workload process. Early combines F1 completion reuse with
scheduler LdB prefetch; its difference from F1 isolates the prefetch-path
change, while its difference from D2 includes F1.

| Route | K-tile reuse | Raw-MVIN prefetch | F1 completion reuse | Scheduler LdB |
|---|---|---|---|---|
| H2 (paired) | No | No | No | No |
| K-tile | Yes | No | No | No |
| D2 (prefetch) | Yes | Yes | No | No |
| F1 (completion) | Yes | Yes | Yes | No |
| Early | Yes | No | Yes | Yes |

Decode 1x768x768 and FFN 16x3072x768 retain seed 20261023 and existing
benchmark helper generation order. tail_63x129x544 is a pipeline benchmark
at the requested tail dimensions, distinct from the independent raw INT32
63x129x544 fixture. Both passed, without claiming identical input tensors.
The original raw fixture also received its own five-route seven-repeat
measurement, recorded in the supplement below.

Each route has seven cold and seven steady samples, rotating/reversing
order, with untimed warmups. Cold clears the prepared-weight cache;
steady reuses it. All 210 samples have ok=1. Every group has exactly seven
unique iteration numbers and the expected cache miss/hit. F1/Early reject
fallback. Every reported median/MAD was independently recomputed from
raw samples and matched the executable's summary.

Primary runs disable raw/physical tracing and diagnostic profiling.
Physical observation runs are separate and explicitly not performance.
The guest pins the issuer using the existing owner path; its CPU mapping
was checked before execution (CPU 1 is hart 0). No numerical, quantization,
attention or saturation semantics were changed for these measurements.

## Graph compute latency

Milliseconds, median (MAD), seven samples per cell. These are CPU-observed
graph-compute intervals, not accelerator cycle counts.

### Steady prepared weights

| Workload | H2 | K-tile | D2 | F1 | Early |
|---|---:|---:|---:|---:|---:|
| decode_1x768x768 | 18.705 (0.020) | 14.833 (0.058) | 15.894 (0.021) | 15.652 (0.006) | 15.604 (0.015) |
| gpt2_ffn_up_16x3072x768 | 97.375 (0.348) | 75.555 (0.105) | 79.400 (0.090) | 78.872 (0.526) | 78.352 (0.060) |
| tail_63x129x544 | 43.286 (0.354) | 43.469 (0.188) | 43.795 (0.390) | 43.101 (0.174) | 42.809 (0.395) |

### Cold prepared-weight cache

| Workload | H2 | K-tile | D2 | F1 | Early |
|---|---:|---:|---:|---:|---:|
| decode_1x768x768 | 33.815 (0.044) | 29.945 (0.037) | 30.998 (0.029) | 30.780 (0.044) | 30.750 (0.012) |
| gpt2_ffn_up_16x3072x768 | 159.391 (0.506) | 137.408 (0.091) | 141.344 (0.284) | 140.896 (0.434) | 140.561 (0.038) |
| tail_63x129x544 | 45.356 (0.201) | 45.024 (0.234) | 44.354 (0.154) | 45.170 (0.692) | 45.337 (0.148) |

Minimum/maximum, pipeline-wall median/MAD, every individual sample and CPU
stage interval remain in each workload's bench.out. bench.err is retained
alongside it. The JSON evidence includes all 30 summary groups. Cold tail
Early is 45.337 ms versus F1 45.170 ms; its small steady improvement does
not establish a broadly beneficial optimization. FFN F1 variability is
also comparable to its small improvement; no confidence/significance
claim is made from seven samples.

## Physical timing evidence

Separate --observe-early runs use the same workload tensors, with one cold
and one steady measured invocation plus warmups (four Early invocations).
They record 24/96/8 prefetch intervals for Decode/FFN/tail and twice as many
Main intervals, including final panels. Each prefetch was matched to the
next Main by run, stripe, next pair, panel and generation. All matches and
orders passed:

start <= scheduler admission <= first DMA-row request <= first B SRAM write;
last B SRAM write and current last qualified MAC <= accepted Fence;
accepted Fence < next Main first qualified MAC.

All snapshots were frozen, non-overflowed and had a seen accepted Fence.
The DMA event is accepted LoadController-to-Scratchpad row request, not
TileLink issue or DRAM completion. Last masked SRAM write and global Fence
are separate endpoints. Representative qualified MAC events do not denote
full-array utilization. Target-clock intervals are not CPU rdcycle values.

| Workload | Intervals | Positive overlap | Simultaneous cycles min / median / max | Total simultaneous cycles |
|---|---:|---:|---:|---:|
| Decode | 24 | 3 | 0 / 0 / 1 | 3 |
| FFN | 96 | 96 | 1 / 3 / 4 | 277 |
| Tail pipeline | 8 | 8 | 4 / 9.5 / 22 | 89 |

These are physical same-cycle cross-pair SRAM/MAC witnesses, not saved
cycles. Decode has zero overlap in 21 of 24 intervals. The raw full-row
63x129x544 fixture separately records 355 cycles, with B0/B1 write counts
1408/1312; H2 and K-tile controls report zero. Small/gapped raw fixtures
report zero and are not relabeled as positive overlap.

Median target-clock deltas show admission is early but actual B movement
arrives near the end of current Main:

| Workload | Admission before current last MAC | First DMA-row request before last MAC | First B write before last MAC | Last B write to next first MAC |
|---|---:|---:|---:|---:|
| Decode | 12238.5 | 74 | 17 | 711422 |
| FFN | 13657 | 89 | 23 | 729058 |
| Tail pipeline | 12404.5 | 188 | 128.5 | 425736.5 |

The final column includes diagnostic readbacks, host work and intermediate
commands, so it is not a production pipeline bubble estimate. Fence
request-to-accept medians are 10102.5/10901/3971.5 target cycles. They are
actual observed protocol intervals, not a decomposition of DMA/Execute
service times and are not added to graph time. The counters do not locate
which downstream arbitration or dependency caused late request acceptance;
claiming a specific stall source would require further instrumentation.

## Work reduction and remaining cost

The CPU submits six descriptor words per load-only admission instead of
D2's per-tile raw MVIN loop. These CPU counts are not internal LOAD2 counts
or measured bus traffic. Successful per-graph counts are:

| Workload | Prefetched panels | D2/F1 raw MVIN | Early raw MVIN | Early LdB descriptors |
|---|---:|---:|---:|---:|
| Decode | 6 | 192 | 0 | 6 |
| FFN | 24 | 768 | 0 | 24 |
| Tail pipeline | 2 | 5 | 0 | 2 |

Scale writes/releases stay balanced and identical across K-tile/D2/F1/Early:
19968 each for Decode, 73728 for FFN, 2193 for tail. H2 uses 33024,
147456 and 4386 respectively. Thus Early's small gain over F1 is not a
second Scale-reuse change. The extra real completion boundary remains.

Steady median CPU stage intervals (ms), D2 / F1 / Early:

| Workload | Scale API interval | Issue interval | Separate Fence interval | Early combined Main/prefetch/drain subinterval |
|---|---|---|---|---:|
| Decode | 6.757 / 6.522 / 6.533 | .760 / .750 / .729 | .050 / .051 / .050 | .161 |
| FFN | 24.858 / 23.989 / 23.967 | 4.206 / 4.206 / 4.116 | .281 / .281 / .281 | .665 |
| Tail pipeline | .745 / .729 / .717 | .298 / .292 / .285 | .017 / .020 / .018 | .043 |

Combined Main/prefetch/drain is a subset of issue time, not an independent
extra contribution. Stage intervals can overlap pipeline work and are not
summed into graph wall time. Scale API time includes software checks,
command responses and hardware synchronization; it is not Scale hardware
waiting in isolation. F1 reduces redundant host reconciliation but retains
CPU fences and all checked releases. K-tile remains faster on the two
requested large workloads; additional overlap alone did not recover D2's
extra host/command cost. Default promotion is not warranted.

## Verification, limitations and preserved failures

The final g2 actual-FPGA run passed API smoke, nine independent raw integer
goldens, attention and PV-range (49 jobs, 16 bitplane jobs, 71 fences,
28224 balanced Scale writes/releases, exact=1). All workload trace commands
and 210 checked performance samples passed. Guest and driver exit 0;
read-only e2fsck passed after shutdown.

Local evidence: 12 targeted native tests before diagnostic correction;
after correction API/execution tests passed 2/2 on each DIM32/DIM16.
The corrected Early RTL suite passed 7/7, including real COUNTER queue,
held response, older DMA and unchanged Fence/Release gates. Eight existing
DIM32 regressions passed before the diagnostic-only correction. These
native/RTL results are not themselves hardware proof.

g2 routed timing: WNS .047 ns, WHS .010 ns, WPWS 0, no failing endpoints,
no no_clock or unconstrained internal endpoints. Bus skew: 28 paths MET,
minimum slack 2.868 ns. Host clock reports 33.330 ns / 30.003 MHz. Placed
resources: 523734 LUTs, 487119 registers, 563.5 BRAM tiles, 212 DSPs.
Inherited XDC/methodology warnings and 572 warning-severity routed DRC
entries remain; no warning or timing constraint was suppressed.

The first Early image failed diagnostic reset at h2_early_trace_begin:
COUNTER counted its own queue head as pending. Capability probing also
reconfigured B0's observation counter. Those defects were fixed at their
owners, regression-tested and verified on g2. First-run samples are not
included; original image, rootfs, logs and frozen binaries remain under
/tmp/h3-8fg and extracted-g-v1. The failed guest exited 1 while its cleanly
powered-off simulator exited 0; those statuses are not conflated.

F1's earlier D2-image run had a driver timeout after successful guest work;
that limitation remains in its separate report. Initial broad DIM32 tests
selected pre-existing DIM16-specific microtile expectations and an unbuilt
GGUF test; no test was weakened. D2's old initial-rootfs hash predates its
pre-session modification and does not match; the exact hash/timestamps and
missing resume-time hash baseline are documented in the G report.
Existing dirty Gemmini, host-init, overlays, D2/F1 images and CDE stream
state were preserved. No commits, push or default promotion were made.

## Completion-audit supplement: original fixture and separated MAC counts

Final requirement comparison identified two gaps in the first report:
the pipeline tail case was not a substitute for timing the original raw
fixture, and Main/Residual MAC-active cycles needed separate attribution.
Both were measured on the **same programmed g2 image**, without RTL changes
or reprogramming. Original g2 performance binaries are frozen in
`/tmp/h3-8fg/g2-frozen/bin`; the supplementary binaries and source snapshot
are identified by `staged-g3.sha256` and `source-software-g3-tracked.patch`.
The 210 primary samples above are not recomputed or mixed with this run.

The raw benchmark reuses the exact independent-golden fixture: M63/N129/
K544, seed 20261023, carriers 24 on every fourth block and column modulo 3
otherwise, local residual K positions 1/7/29, alternating -128/127 digits,
and the original frozen tile-plan input I2/J4/K4. One untimed invocation
per route precedes seven rotated/reversed measured invocations.
Every invocation checks Main and Residual INT32 independently, balanced
Scale and expected reuse/prefetch descriptors. All 35 recorded samples
passed, and median/MAD were independently recomputed.

| Original raw fixture route | Median ms | MAD ms |
|---|---:|---:|
| H2 | 2.737 | .042 |
| K-tile | 1.966 | .011 |
| D2 | 2.195 | .016 |
| F1 | 2.183 | .029 |
| Early | 2.122 | .038 |

These intervals measure the checked paired runner on the issuer hart,
excluding thread bring-up, fixture construction and golden calculation.
They include its planning/session/command/drain work, not graph compute
or isolated accelerator execution. This prepared raw fixture has no graph
weight-cache cold/steady distinction. Early is still slower than K-tile;
the adoption conclusion does not change.

Separate diagnostic runs use `POTAL_EARLY_TRACE=1` for Main and `=2` for
Residual. Residual mode explicitly drains before reset and after each
Residual command to isolate its representative MAC events. This extra
serialization is diagnostic-only: its latency and overlap are not used
as production performance evidence. Main-to-prefetch command issue remains
uninstrumented between the two submissions. Snapshots are read after real
Fence completion and never substitute for completion.

| Workload | Main MAC-active cycles per invocation | Residual MAC-active cycles per invocation | Main / Residual intervals captured |
|---|---:|---:|---:|
| Decode | 576 | 456 | 48 / 240 |
| FFN | 36864 | 29952 | 192 / 1152 |
| Tail pipeline | 5355 | 4590 | 16 / 72 |

Each count is identical across four independently checked Early
invocations (warmups plus one cold and one steady diagnostic invocation).
These are representative qualified PE events, not all-PE operation counts.
Every snapshot is frozen/non-overflowed and has an accepted Fence;
first/last MAC timestamps lie within the interval. Window, K panel,
pair, generation, run and stripe identify each record.

Supplementary guest and simulator both exited 0; read-only filesystem
checking passed. Evidence: `/tmp/h3-8fg/extracted-g3/results`, `g3-run`,
`analysis-g3.json`, `results-g3.sha256`, and `analysis-g3.sha256`.
The latest execution regression passed before the hardware supplement;
the supplementary raw golden nine-case gate ran before the 35 samples.
