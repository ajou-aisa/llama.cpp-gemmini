# H3-6E3: Paired Bottleneck Isolation

Date: 2026-10-26. DIM16, fixed Paired K28.

The same-build n=7 controls show Paired remains slower at the same K28:
Decode +5.481 ms and FFN +25.679 ms. Serial's K48-to-K28 changes are much
smaller: +0.040 ms and +0.175 ms. These are fresh measurements, not substituted
H3-6E results. All 12 separate hardware diagnostic runs passed mapping,
command-count, INT32 golden and safe-final-state checks.

## 1. Baseline and Hardware Configuration

The branch is `potal`, HEAD
`dcab70ff8a76348a531c3f89785a29f13a2e8413`. Existing uncommitted H3-7B/H3-6E
work is preserved. Production scheduling, Attention, numerical contracts,
cached trace control, packet decode reuse and default-OFF D1 are unchanged.
The existing U280 FPGA image, PoTal RTL, Gemmini and host-init are unchanged.
No commit, push or Git staging was performed.

Release DIM16 builds use A8W8, ExSIA block32, compensation and stripe pipeline.
The opt-in `POTAL_E3_TESTS` switch builds test-only executables. The controlled
benchmark includes the unchanged backend through a test-local translation unit
and redirects only tile selection and preparation capture. Production
`ops.cpp`, `potal.c` and the planner/executor were not edited.

Artifacts:

| Artifact | SHA256 |
|---|---|
| Controlled benchmark | `0fb3fd2d3a236ea1fe70710700c39d99f3ba425c78b48c86bc6b93a9b4ff9ba5` |
| Window diagnostic | `118370231dfc829f16439f4a329a83f88e705ba7622bf2015640fd1298cd9333` |
| Current production library | `630fcc334458c7cad69506be1d8588f2bd0253c7becc2d73c7ceb9835a16e513` |
| Frozen H3-6E benchmark | `c694e07a3349929f6b3568cb3d6cc3867fbbd4323bed619482ca3f3a10c79c43` |
| Frozen H3-6E production library | `1aca66631382f52856d4a187822e456f03a7522a9c828c968811619377c240b6` |
| Hardware parameter header | `b4cd8b9efef9f3cd9b598ae3cc84af6478f83e628ed73d94a77fb1c452df1fc7` |

The current library includes preserved H3-7B work and is not falsely identified
as the old frozen library. A fresh frozen-binary run is separate from the
same-build K-control experiment. The guest uses a separate rootfs copy.

Both builds retain the same existing LOG_DEBUG setting. Additional paired,
window, raw-trace and planning diagnostics are disabled for baseline timings.
Window wrapping, prefencing and profiling are separate diagnostic runs.

The guest suite and driver exited 0. Raw results are in `/tmp/h3-e3-results/`;
UART evidence is `/tmp/h3-e3-run/uartlog`. Native sources and separate builds
are under `tests/potal-e3-*`, `build-potal-e3-host` and `build-potal-e3-d16`.
The driver ran for 3608.6 host seconds. Its fastest-target-clock cycle total is
not an accelerator-cycle or workload-time measurement.

## 2. Serial K48/K28 vs Paired K28

The benchmark reuses `potal-paired-smoke` fixtures with seed 20261023, preserving
weight/activation RNG order, outlier insertion, quantization, packet lowering,
stripe size, I/J, cache policy and merge path. Seven measured samples per mode
rotate/reverse execution order; warmup and preparation fingerprint capture are
outside timed samples.

| Fixture | I/J | Original K | Restricted K | Quantization/packet hash |
|---|---|---:|---:|---|
| Decode 1x768x768 | 1/9 | 48 | 28 | `e460e0fce470e311` |
| FFN 16x3072x768 | 1/9 | 48 | 28 | `18e2b73e075c3bbe` |
| Capacity fallback 160x320x512 | 5/6 | 32 | 28 | `b11dc236d3133c1b` |

Native and guest software verification proved equal quantization/sealed packets
and exact independent Main integer outputs. Serial Main LOOP_WS counts changed
6 to 12, 22 to 44 and 8 to 16 respectively; corresponding Scale write counts
remained 18432, 73728 and 10240.

The Paired benchmark's `frozen_requested_K=48` is its unchanged request, not
executed K. The active planner applies the B-bank limit and executes K28.

Report median and unscaled median absolute deviation (MAD), in milliseconds:

```text
A = median(T_serial_K28) - median(T_serial_K48)
B = median(T_paired_K28) - median(T_serial_K28)
```

| Fixture | Serial Original median / MAD | Serial K28 median / MAD | Paired Original median / MAD | A | B |
|---|---:|---:|---:|---:|---:|
| Decode | 8.770 / 0.003 | 8.810 / 0.020 | 14.291 / 0.024 | +0.040 | +5.481 |
| FFN | 52.509 / 0.129 | 52.684 / 0.026 | 78.363 / 0.168 | +0.175 | +25.679 |
| Capacity fallback | 142.062 / 0.925 | 142.958 / 0.376 | 142.455 / 0.409 | +0.896 | -0.503 |

The fallback row starts from K32, not K48. Every Paired-selected sample falls
back for both stripes and issues zero Paired loops, Index commands, readbacks
or fences. Its actual Serial counts remain Main8/Residual16, while the forced
Serial K28 control uses Main16/Residual16. Its negative B is therefore **not**
a Paired K28 speedup and is not included in the two-case K-matched conclusion.

All 63 samples report success, zero timed weight packs/misses and one weight
cache hit. Each seven-sample mode has stable command counts. Decode/FFN Paired
use Main 12/44, Residual 60/264, Index writes 978/6314, readbacks 12/44,
readback tiles 96/384 and one explicit Fence. Reported grouped drains are 73/309;
these are not the count of individual Scale Release instructions.

Paired is 62.21% slower than Serial K28 for Decode and 48.74% slower for FFN.
B includes different scratchpad placement, Residual lowering, planning,
validation, staging, issue and completion behavior. It is not pure instruction
issue overhead. A controls K tiling within Serial; it does not convert Serial's
memory layout into Paired's layout.

Fresh frozen H3-6E binary steady results, also n=7 median / MAD in ms:

| Fixture | Serial | Paired | Paired conservative |
|---|---:|---:|---:|
| Decode | 8.760 / 0.013 | 14.415 / 0.015 | 14.576 / 0.025 |
| FFN | 52.237 / 0.169 | 78.270 / 0.039 | 79.301 / 0.077 |

They retain the same qualitative outcome. They are a separate binary/run and
are not mixed into the same-build A/B calculation.

## 3. Residual Window Work Breakdown

The unchanged planner and exact fixture inventory establish:

| Quantity | Decode | FFN |
|---|---:|---:|
| Main B residencies | 12 | 44 |
| J groups | 6 | 22 |
| K panels per J group | 2 | 2 |
| Residual windows | 60 | 264 |
| Main LOOP_WS | 12 | 44 |
| Residual LOOP_WS with B=NULL | 60 | 264 |
| Fragments in the two panels, per J group | 13 / 6 | 14 / 10 |
| Current windows per panel | 7 / 3 | 7 / 5 |
| Index32 lower bound with preserved DIM partials | 7 / 3 | 7 / 5 |

Each fragment retains a DIM16 partial boundary and its block/carrier identity.
Index32 admits at most two such fragments. Therefore these cases already reach
`ceil(fragment_count/2)` in each residency. Cross-panel merging would also cross
B residency ownership. Simply merging current windows cannot reduce their
number while preserving these contracts. Removing padding by mixing fragments
inside a DIM partial can change SCU application and saturation/cancellation.

Per-window work consists of compact-to-padded Residual A staging, Scale window
programming, Index configuration and original-K entries, one checked loop,
completion-dependent Scale release, and ownership retirement. INT32 ACC1
persists across windows; the active schedule does not spill/restore ACC at
every window. Final readback is separate.

The new diagnostic records window/panel/J-group identity, K extents, fillers,
lane geometry, carriers, generations, Scale slots, Index entries and checked
API counts. It aggregates before printing, after execution. Source:
`ggml/src/ggml-potal/tests/potal-e3-window-diag.cpp`.

Planner evidence:
`ggml/src/ggml-potal/matmul/paired-auto.cpp:149` (fragment construction),
`:470` (two-fragment window packing). The preserved full inventory remains in
`H3-6E-paired-dataflow-scheduling-analysis.md`.

Actual hardware window wrappers confirm the inventory. Their n=3 median
CPU-observed costs are below, in ms. These heavily instrumented values describe
only that diagnostic, not production wall-time contributions.

| API | Decode Main | Decode Residual | FFN Main | FFN Residual |
|---|---:|---:|---:|---:|
| Scale Write | 13.997 | 11.011 | 56.523 | 56.887 |
| Scale Release | 13.174 | 10.804 | 53.316 | 54.270 |
| Index Config | 0 | 0.046 | 0 | 0.210 |
| Index Write | 0 | 0.678 | 0 | 4.418 |
| LOOP_WS issue | 0.162 | 0.119 | 0.736 | 0.584 |
| Final readback issue | 0.013 | 0.009 | 0.085 | 0.073 |

Main/Residual Scale writes and releases are 18432/14592 for Decode and
73728/73728 for FFN. Readbacks are 6/6 and 22/22, not one per window.
Medians of subgroups need not sum to the median of their total.

## 4. Scale/Index/LOOP_WS Issue Costs

GNU linker wrappers call each real checked API exactly once with unchanged
arguments and status. They time Scale Write/Release, Index Config/Write,
LOOP_WS, ACC readback, Fence and session close. The executor envelope also
includes unwrapped staging, validation and wrapper bookkeeping.

These intervals are CPU-observed checked API elapsed time. They include any
instruction response delay encountered by that caller but do not isolate FPGA
execution time. A returned LOOP_WS is not a completed matrix operation.

The separate existing profiler measures broader phases, including plan/pack
and Scale loops. Its intervals must not be added to the wrapper intervals or
subtracted from a different diagnostics-OFF graph run as disjoint costs.
All diagnostic calls succeeded. Normal raw command/drain counts were
67916/33025 for Decode and 305000/147457 for FFN.

Existing always-collected API metrics in the diagnostics-OFF controls give
the following n=7 median / MAD, in milliseconds:

| CPU-observed metric | Decode Serial K28 | Decode Paired | FFN Serial K28 | FFN Paired |
|---|---:|---:|---:|---:|
| Scale API | 3.190 / 0.008 | 7.448 / 0.001 | 14.269 / 0.023 | 33.227 / 0.017 |
| Issue API | 0.205 / 0.003 | 0.467 / 0.001 | 1.336 / 0.003 | 2.554 / 0.004 |

The Scale interval grows by 4.258 ms and 18.958 ms. Individual Scale write
and release counts are nevertheless equal between modes: 33024 each for Decode
and 147456 each for FFN. The evidence points to the cost of the checked Paired
Scale path and its synchronization semantics, not more Scale element commands.
The broader issue interval includes different work across routes; it is not
a matched pure LOOP_WS hardware execution timer.

The separate phase profiler gives n=7 median / MAD, excluding its warmup:

| Paired phase, CPU ms | Decode | FFN |
|---|---:|---:|
| Executor planning/validation scope | 0.857 / 0.012 | 5.520 / 0.015 |
| Residual staging/pack | 0.034 / 0.001 | 0.222 / 0.001 |
| Scale Write loop | 4.138 / 0.005 | 18.358 / 0.035 |
| Scale Release loop | 3.305 / 0.002 | 14.863 / 0.030 |
| Index program | 0.198 / 0.003 | 1.232 / 0.008 |
| Main issue | 0.148 / 0.001 | 0.686 / 0.002 |
| Residual issue | 0.099 / 0.001 | 0.490 / 0.006 |
| Readback issue | 0.019 / 0.001 | 0.140 / 0.002 |
| Final Fence | 0.001 / 0 | 0.001 / 0 |

Profile-enabled graph medians are 14.377 / MAD 0.024 and 78.781 / MAD 0.015 ms,
versus diagnostics-OFF 14.291 and 78.363 ms. This is observed run-to-run
perturbation, not an exact instrumentation correction.

Per-call wrapping is much more intrusive. Its executor envelope medians/MAD
are 126.483/0.663 ms Decode and 578.795/1.835 ms FFN. Even the wrapped Scale
Write/Release totals are 25.008/24.006 ms and 113.413/107.517 ms. Timer reads,
recording, changed issue pacing and wrapper bookkeeping materially affect the
experiment. Do not subtract these from the unwrapped baseline or interpret
their absolute values as baseline FPGA waits. Their scope is also narrower
than the graph while their instrumentation is heavier.

Current RISC-V disassembly retains an 80-byte Scale Write stack frame and a
64-byte Scale Release frame. The untraced route emits the raw custom instruction
directly; Release still performs the required software outstanding-state
updates after success. No new instruction-count claim is copied from an older
binary, and checked validation/synchronization is not proposed for removal.

## 5. Scale Release and Drain Analysis

Before the first Release of each window, the diagnostic queries C11 state and
labels it `software_pending` or `software_idle`. Those are software bookkeeping
categories, not hardware-wait categories. The accelerator may already have
completed work while the software count still says pending.

Source establishes that successful Release and Fence responses require
`allDrained` (`generators/potal/src/main/scala/potal/Controller.scala:787-808`).
This is a completion guarantee, not a duration measurement.

The opt-in prefence experiment inserts one checked Fence before a first Release
with software outstanding work. Its count/time is reported separately. It is
intrusive: moving the synchronization point changes the command stream and
cannot reveal a pure Release hardware-wait time by simple subtraction.

In every normal and prefence run, the first-Release query finds 60 pending and
12 idle windows for Decode, and 264 pending and 44 idle windows for FFN.
Pending cases are Residual windows; Main Scale releases occur after preceding
Residual releases have drained the outstanding state.

In all groups the individual first-Release elapsed median is 1 microsecond,
MAD 0, with measurements quantized at microsecond granularity. That does not
establish a 1-microsecond hardware wait or a zero wait.

| n=3 experiment, ms median / MAD | Decode | FFN |
|---|---:|---:|
| Normal: sum of first Releases queried pending | 0.050 / 0.004 | 0.211 / 0.002 |
| Prefence: same subsequent Release calls | 0.046 / 0 | 0.207 / 0 |
| Extra Fence calls' total CPU interval | 0.045 / 0.002 | 0.197 / 0.002 |
| Prefence executor envelope | 123.327 / 0.150 | 575.690 / 0.102 |

Prefence adds exactly 60/264 commands and drains, giving 67976/33085 and
305264/147721 raw totals. Its somewhat lower envelope is not a production
speedup: the experiment deliberately changes pacing and uses heavy timing
instrumentation. No large, separately measurable first-Release completion
stall is established by these observations.

Actual Release completion-wait duration: **Not measured**.
Separate DMA, Execute and readback wait durations: **Not measured**.

## 6. Weight Reuse Benefit

Paired Main commands declare total B payloads of 589824 bytes for Decode and
2359296 bytes for FFN. The two K panels partition K; doubling Main loop count
does not double that payload. All Indexed Residual commands use B=NULL and
consume the resident B bank.

This proves the command-level reuse and absence of an additional Residual B
DRAM pointer. It does not measure actual bus bytes, burst count, DMA duration or
time saved. Address-span sums include stride gaps and must not be substituted
for transferred bytes. A net Serial/Paired timing difference contains many
other changes and cannot isolate the time benefit of reuse.

Actual saved DMA time and physical bus bytes: **Not measured**.
Source-visible DMA-write/mesh-input activity counters, if available in this
image, still would not by themselves prove simultaneous Main/Residual PE MACs.

No `POTAL_DATA_OVERLAP` or corresponding PoTal summary records were observed in
the actual driver output. `memory_stats0.csv` contains a header only;
`latency_histogram.csv` has no samples. Heartbeats report fastest-target cycles,
not per-window accelerator events. These files cannot fill the missing
DMA/Execute overlap or saved-weight-transfer measurements.

## 7. Additional Paired Overhead

K-controlled differential B is +5.481 ms for Decode and +25.679 ms for FFN. Its
candidate contributors are repeated Residual window setup and Index programs,
checked API calls, repeated loop setup, staging and planning/validation.
The same fixture, packets and I/J control removes an important confounder but
does not make the two lowering paths identical.

Window count explains repeated work structurally. Cross-case correlation
between window count and total latency is not a causal per-window cost:
Decode and FFN also differ in rows, columns, residual lanes and total work.
No unimplemented window-merging speedup is claimed.

## 8. Measured vs Unmeasured Costs

| Quantity | Interpretation |
|---|---|
| Diagnostics-OFF graph latency | Fresh n=7 CPU-observed wall interval, table in section2 |
| Checked API intervals | Instrumented CPU elapsed time, not pure accelerator execution |
| Prefence interval | Intrusive completion response interval |
| Software outstanding query | Bookkeeping, not physical busy duration |
| Logical/command-declared bytes | Descriptor-derived extent, not measured bus traffic |
| Release/DMA/Execute/readback wait components | Not measured |
| Physical Main/Residual PE overlap | Not measured |
| Complete microtile hardware pipeline speedup | Not measured; not implemented |

No overlapping timing scopes are summed as independent graph-wall-time
contributions. Software emulator pass results remain software-only.

## 9. Bottleneck Ranking

The strongest measured difference is the CPU-observed Scale API path:
+4.258 ms Decode and +18.958 ms FFN, despite equal Scale command counts.
The issue interval increase is smaller: +0.262 ms and +1.218 ms. The K-split
control A is smaller again. Rank the evidence as follows:

1. **Checked Scale programming/release path:** dominant measured added API
   interval. Preserve checked validation, trace caching and Release completion.
2. **Owner-side planning/validation and repeated window control:** measurable
   Paired scopes (0.857/5.520 ms planning), with Index and issue costs separately
   shown above. A matched Serial planning scope is unavailable, so those
   absolute values are not attributed wholesale to B.
3. **Residual staging and final readback issue:** smaller measured CPU scopes.
   Actual execution/readback completion time remains unseparated.
4. **Main K subdivision:** small controlled A; not the principal explanation
   for the Paired gap in these fixtures.

Physical Release/DMA/Execute waiting cannot be ranked quantitatively.
Window subdivision is a structural source of repeated control, but the current
Index32/DIM16 boundaries already attain the per-residency window-count lower
bound. Neither unconditional merging nor B-bank expansion follows from the
measurements.

Next performance work should target proven CPU work in the checked Scale
path, with invariants and assembly evidence before implementation. This task
does not start D2/report-copy optimization or weaken validation. Pipeline work
first needs the explicit resident-tile/SCU interfaces described in the companion
report; an assumed overlap speedup is not an alternative to measurement.

The companion `H3-6E3-microtile-pipeline-feasibility.md` records the implemented
software prototype, bounded raw ACC primitive, missing resident-tile/metadata
interfaces, Scale capacity conflict and separate A/B/C/D feature verdicts.
No production schedule adoption is justified by an unconnected software trace.

Preservation checks: 54 protected source/artifact hashes match the initial
snapshot; `src/CMakeLists.txt` also matches. Removing only the new opt-in E3
CMake block recovers the initial backend CMake hash. The 67-file manifest under
PoTal's Scala directory, Gemmini HEAD/diff and host-init hashes also match.
The Git index remains empty, and the branch/HEAD are unchanged.

Validation used separate native and DIM16 builds, nine preserved Host regression
binaries, the expanded software prototype, 63 controlled hardware samples,
fresh frozen-binary runs, 12 actual window diagnostics and seven raw ACC
hardware tests. LSP lacks this target's generated include configuration; actual
C/C++ builds passed. Existing Chipyard `.conda-env` activation warnings did not
prevent the configured toolchains or tests from running. Hardware fault
injection and a connected microtile hardware pipeline were not run.
