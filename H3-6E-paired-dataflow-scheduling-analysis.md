# H3-6E: Paired Dataflow and Scheduling Analysis

Date: 2026-10-26.

The active H2 hardware path shares resident weights, but does **not** implement
the complete requested Weight Reuse Pipeline. Fresh unchanged-binary measurements
still favor Serial. No Paired schedule, RTL, FPGA image, or Report Copy
optimization was changed.

The paper requirements below are the requirements supplied in the work request.
No separate paper PDF was supplied. They are not rewritten to match the code.

## 1. Paper Dataflow Requirements

| Requirement | Active hardware route |
|---|---|
| Main/Residual share weights | Present within each B residency |
| Split activation scratchpad and INT32 ACC 1:1 | ACC halves are fixed; A is disjoint but not fixed equal halves |
| Preserve Main stripe and I/J tiles | Present, except reducing I for the existing stripe's tail |
| Do not shrink Main K because of Residual | Planner contains this coupling; current measured cases are instead limited by the B bank |
| Subdivide only Residual K as necessary | Residual windows exist, but Main also uses a smaller common K |
| Original-K indexed access to resident B | Present, encoded relative to the resident panel |
| Alternate Main/Residual microtiles | Not implemented by the active whole-LOOP_WS schedule |
| Double-buffered next-K prefetch | Two physical pairs exist; software does not schedule next-panel prefetch ahead of the current releases |
| Group-level ACC save/restore | Not implemented by the active route |
| Reclaim capacity after one path completes | No dynamic A/ACC borrowing in a mixed stripe |
| Drain the surviving path in Single mode | Empty Residual panels omit work; no general microtile-tail transition/retiling |
| Overlap Host preparation/reconstruction with execution | Producer preparation can overlap; owner-side planning, staging and reconstruction remain serial |

The separate `paired_tiled_matmul_auto` software schedule/emulator implements
several missing behaviors. It explicitly refuses its hardware route. Its
microtile trace is not evidence that `potal_paired_matmul_auto` issues that trace.

Evidence: `ggml/src/ggml-potal/matmul/paired.cpp:85`,
`paired-schedule.cpp:310`, `paired-schedule.cpp:332`,
`paired-auto.cpp:832`, and `stripe.cpp:757`. Fully qualified software paths are
relative to the checkout. Bare C++ filenames in citations mean
`ggml/src/ggml-potal/matmul/<filename>`. RTL paths are relative to Chipyard;
bare `ExecuteController.scala` means
`generators/potal/src/main/scala/potal/ExecuteController.scala`.

## 2. Current Implementation

The measured software is the H3-7A checkpoint
`dcab70ff8a76348a531c3f89785a29f13a2e8413`, whose production library is unchanged
from the D1 checkpoint. The existing Release DIM16 configuration has A8W8,
HP1 block32, ExSIA, compensation ON, STRIPE_PIPELINE, and Planning Diagnostics OFF.

Frozen measurement artifacts:

```text
/tmp/h3-6e-baseline/bin/
potal-paired-smoke SHA256:
c694e07a3349929f6b3568cb3d6cc3867fbbd4323bed619482ca3f3a10c79c43
libggml-potal.so SHA256:
1aca66631382f52856d4a187822e456f03a7522a9c828c968811619377c240b6
parameter-header SHA256:
b4cd8b9efef9f3cd9b598ae3cc84af6478f83e628ed73d94a77fb1c452df1fc7
```

The existing FireSim U280 driver/image was reused. The H3-6E rootfs is a
separate copy; neither the D1 nor H3-7A image was overwritten.

Actual order is J group, K panel, one Main LOOP_WS, then that residency's
Indexed Residual windows. The Main scale window is deliberately released after
the Residual windows have been issued: releasing it earlier would add a global
drain before Residual could join the stream. Each Residual scale window still
has a completion release. All residency releases precede the next residency.

Evidence: `paired-auto.cpp:774`, `paired-auto.cpp:832`,
`paired-auto.cpp:879`, and `paired-auto.cpp:1005`.

## 3. Main/Residual K Tiling Comparison

Both fixed fixtures reset `mt19937(20261023)` and reproduce the exact existing
benchmark's weight generation, activation generation, and outlier insertion.
The inventory runs the actual planner and checked C11 translation on the H2
emulator. It does not replace the planner or issue hardware commands.

K values in this table are DIM16 tiles; multiply by 16 for elements.

| Field | Decode 1x768x768 | FFN 16x3072x768 |
|---|---:|---:|
| Serial Main I/J/K | 1 / 9 / 48 | 1 / 9 / 48 |
| Requested Paired Main K | 48 | 48 |
| Paired hardware-only K limit | 28 | 28 |
| Residual-dependent K limit | 28 | 28 |
| Executed Paired Main K | 28 | 28 |
| Actual K panels | 2 | 2 |
| J groups | 6 | 22 |
| Serial Main LOOP_WS | 6 | 22 |
| Paired Main LOOP_WS | 12 | 44 |
| Indexed Residual windows executed | 60 | 264 |

The common-K source coupling is real:

```cpp
plan.tile_K = std::min({
    plan.tile_K_main,
    plan.tile_K_residual_limit,
    plan.tile_K_hardware_limit
});
```

However, **Residual reservation is not the active reason for K=28 in these two
fixtures**. Both residual and hardware-only bounds equal 28. The resident B
bank is the binding resource. Removing the residual term alone cannot restore
K=48 here.

With D=16, one B bank has 4,096 rows. Preserving J=9 permits
`floor(4096/(9*16))=28` K tiles after whole-block rounding. K=48/J=9 would
need 6,912 B rows, or 108 KiB, versus the existing 64 KiB weight bank.
Serial fits its combined A+B allocation in a 128 KiB loop half; Paired instead
requires separate A and B banks. These are different capacity constraints.

For a general mixed stripe, the current planner subtracts Residual A rows
and Scale slots before computing another Main K bound. Decoupling Residual K
is useful when that bound is tighter, but that benefit has not been measured
for other inputs here.

Evidence: `paired-auto.cpp:326`, `paired-auto.cpp:368`,
`paired-auto.cpp:382`, `paired-auto.cpp:420`, and `tiling.cpp:20`.

## 4. Weight Residency and Reuse

| Quantity | Decode | FFN |
|---|---:|---:|
| B residencies / Main LOOP_WS with nonzero B | 12 | 44 |
| Indexed Residual LOOP_WS with B=0 | 60 | 264 |
| Consumers per residency, range | 4--8 | 6--8 |
| Consumers per residency, mean including Main | 6 | 7 |
| Residual windows per residency, range | 3--7 | 5--7 |
| Main command-declared B payload bytes | 589,824 | 2,359,296 |
| Sum of Main B address-span bytes, including stride gaps | 3,531,264 | 51,775,488 |
| Main+Residual command-declared A payload bytes | 5,586 | 352,418 |

These are **command-declared payloads/spans**, not measured bus traffic or DMA
transaction counts. Address-span sums include gaps and repeated address
envelopes; they must not be reported as bytes actually transferred.

The two Main panels partition K without overlap. Their total logical B payload
is N*K, not twice N*K. Doubling Main LOOP_WS therefore does not establish a
doubling of weight bytes. Indexed Residual supplies B=NULL and uses the resident
scratchpad copy; no Residual B DRAM load is requested by those commands.

For pair p, A occupies bank 2p and B bank 2p+1. The resident B origin is:

```text
b_sp_begin = (2*p+2)*BANK_ROWS - K_panel_tiles*J_group_tiles*DIM
```

The inventory checks this origin, bounds, pair selection, and every
`resident_k = original_k - panel.k_begin` mapping. Mapping errors were zero.
Pairs 0 and 1 alternate; alternation alone is not proof of overlapped prefetch.

Logical `load_weight`, `load_main_a`, and `load_residual_a` operations issue no
separate commands in the executor. A Main LOOP_WS causes the RTL load
generators to emit real loads; an Indexed Residual LOOP_WS has no B load.
Mesh PRELOAD is another operation, reading resident scratchpad weights, not
another synonym for DRAM load.

Actual DMA transaction counts, physical B bus bytes/time, and measured
Residual B-DMA savings are unavailable from these runs. No software estimate
is labeled as those measurements.

Evidence: `paired-auto.cpp:662`, `paired-auto.cpp:895`,
`paired-auto.cpp:1990`, `paired-auto.cpp:2073`, `paired-auto.cpp:2158`;
`generators/potal/src/main/scala/potal/LoopMatmul.scala:177,433,1247`;
`ExecuteController.scala:361`.

## 5. Residual Window Efficiency

The physical Index table has 32 entries per pair. Each fragment preserves its
original block/carrier identity and occupies a DIM-wide partial boundary.

| Aggregate over executed windows | Decode | FFN |
|---|---:|---:|
| Active lane rows | 1 | 13 |
| Padded lane rows | 16 | 16 |
| Unique windows before repeating across J groups | 10 | 12 |
| Executed windows / Residual LOOP_WS | 60 | 264 |
| Active compact K sum | 216 | 4,312 |
| Padded K sum | 1,824 | 8,448 |
| Active/padded K | 11.8421% | 51.0417% |
| Filler Index entries | 762 | 2,002 |
| Index writes, including fillers | 978 | 6,314 |
| Active K range per window | 2--7 | 9--23 |
| Padded K per window | 16 or 32 | 32 |
| I tiles | 1 | 1 |
| J tiles | 9, or 3 in the final group | 9, or 3 in the final group |

These utilization ratios describe the compact rectangular K geometry. They
are not measured PE utilization, nor the count of nonzero MACs: different lane
rows can contain zeros within a shared compact rectangle.

The complete 60/264 window records also give this command-work inventory:

| Rectangular MAC sites | Decode | FFN |
|---|---:|---:|
| Active rows * active compact K * valid columns, summed | 27,648 | 7,827,456 |
| DIM-padded I * K * J, summed | 3,735,552 | 18,874,368 |
| Active rectangle per issued Residual LOOP_WS, mean | 460.8 | 29,649.45 |
| Padded rectangle per issued Residual LOOP_WS, mean | 62,259.2 | 71,493.82 |
| Active/padded rectangle | 0.7401% | 41.4714% |

These are derived from the executed window descriptors and valid J tails.
They quantify nominal work per issued GEMM command, not per individual CONFIG,
Scale, or Index command. Active rectangles still contain structural zeros;
their size is not a measured count of useful nonzero MACs or physical cycles.

Small Residual jobs repeat Index, Scale, LOOP_WS configuration, and completion
work. Decode expands four Serial Residual LOOP_WS into sixty Indexed commands;
FFN expands twenty-two into 264. This is measured issued work, not an inferred
array-cycle count.

The full per-window records include original/resident K ranges, tail padding,
filler counts, Scale generation/window, dependent residency, ACC epoch, first/
last flags, and A/ACC footprints in `/tmp/h3-6e-inventory.log`.

Merging windows cannot simply discard padding. Packing fragments with different
carriers into one raw partial would change SCU placement; combining even
equal-carrier partials can change saturation order. A candidate must preserve
the original partial sequence or prove bounds excluding every changed
saturation. Index capacity alone is not the only constraint.

## 6. Index/Scale/LOOP_WS Issue Cost

Fresh diagnostics-OFF steady medians below are guest CPU-observed API
milliseconds. They are not independent FPGA execution intervals.

| Case / route | Scale Write+Release API | LOOP issue API | Explicit Fence API |
|---|---:|---:|---:|
| Decode Serial | 3.155 | 0.197 | 0.008 |
| Decode Paired | 7.452 | 0.459 | 0.001 |
| Decode Conservative | 7.437 | 0.463 | 0.097 |
| FFN Serial | 14.092 | 1.308 | 0.031 |
| FFN Paired | 33.176 | 2.552 | 0.001 |
| FFN Conservative | 33.168 | 2.547 | 0.426 |

Scale counts are identical across routes: 33,024 writes and releases each for
Decode, 147,456 each for FFN. Paired's larger Scale API interval does not mean
it transfers more Scale entries, nor prove that the excess is hardware waiting.

A separate `POTAL_PAIRED_PROFILE=1` run used three repeats. Paired steady
medians:

| Instrumented API/group interval, ms | Decode | FFN |
|---|---:|---:|
| Packet decode caller envelope | 0.159 | 1.863 |
| Route plan construction/validation envelope | 1.549 | 8.213 |
| Executor request/validation/copy envelope | 0.878 | 5.561 |
| Residual staging/pack | 0.034 | 0.222 |
| Scale Write | 4.155 | 18.485 |
| Scale Release | 3.307 | 14.691 |
| Index Config+Write | 0.196 | 1.234 |
| Main LOOP_WS program issue | 0.147 | 0.685 |
| Residual LOOP_WS program issue | 0.096 | 0.504 |
| ACC readback issue | 0.017 | 0.142 |
| Explicit Fence | 0.001 | 0.001 |

CONFIG, bounds, addresses, strides and LOOP_WS are grouped in the issue APIs;
their individual elapsed costs were not separately measured. Each successful
LOOP_WS program has four CONFIG commands, five LOOP_WS_CONFIG commands and
one LOOP_WS command. Paired readback adds CONFIG_ST plus MVOUT per result tile.
Those are source/translated-command counts, not ten independent GEMMs.

The issue-profile join uses run IDs, not adjacent log lines. stderr profile
printing can precede buffered stdout samples in the redirected guest log.
The unchanged benchmark's steady Paired IDs are 17/21/24; Conservative IDs
are 18/20/22. All expected IDs were present in each separate case process.

## 7. Drain and Completion Dependencies

| Steady per graph | Decode Serial / Paired / Conservative | FFN Serial / Paired / Conservative |
|---|---:|---:|
| Main LOOP_WS | 6 / 12 / 12 | 22 / 44 / 44 |
| Residual LOOP_WS | 4 / 60 / 60 | 22 / 264 / 264 |
| Index Config | 0 / 60 / 60 | 0 / 264 / 264 |
| Index Write | 0 / 978 / 978 | 0 / 6,314 / 6,314 |
| Paired ACC readback groups | 0 / 12 / 12 | 0 / 44 / 44 |
| Paired MVOUT tiles | 0 / 96 / 96 | 0 / 384 / 384 |
| Explicit Fences | 10 / 1 / 145 | 44 / 1 / 617 |
| Backend Paired drain groups | 0 / 73 / 217 | 0 / 309 / 925 |

There are two distinct drain counters:

- The backend executor counts a released **Scale window** once, plus explicit
  fences. Thus 12+60 release windows plus one fence give Decode's 73.
- The C11 session counts each accepted SCALE_RELEASE command and each fence.
  The same Decode stream has 33,024 individual Release commands, not 72.

An accepted Release uses the same global `allDrained` condition as Fence.
One explicit fence is therefore not one synchronization point. Reusing Index
windows, staged Residual A, a pair, Scale entries, and ACC epochs relies on
completion dependencies and accepted releases. Later idle releases are not
permission to remove public synchronization or liveness checks.

Main's scale window remains live until after its Residual windows are issued.
Each same-buffer Index replacement and Residual A reuse must respect the
previous consumer's completion. Final readback is CPU-readable only after its
covering drain. Fatal DMA uncertainty keeps buffers alive and disables reuse;
a failed fence is not retried by cleanup.

Evidence: `paired-auto.cpp:1734,1958,1965,2170`;
`generators/potal/src/main/scala/potal/Controller.scala:788`.
No drain relocation, Scale-state shortcut, or unchecked API was implemented.

## 8. Pipeline Overlap

The software producer can prepare later activation stripes while the accelerator
owner consumes an earlier stripe. That is a supported producer/consumer overlap,
not proof of Main/Residual arithmetic overlap.

The active schedule submits a complete Main loop before its Residual windows.
The single Execute generator accepts another request only after traversing its
current request. Logical alternation in `paired-schedule.cpp` belongs to the
separate emulator and must not be projected onto this route.

Two physical pairs do not establish next-K prefetch: the next residency is
generated after the current releases. Owner-side packet decode, planning and
staging precede issue; reconstruction/merge follows the final drain on that
same owner before the next stripe is consumed.

Source RTL has a `[POTAL_LOOP_OVERLAP]` monitor at
`generators/potal/src/main/scala/potal/LoopMatmul.scala:933`. Its own comments
limit it to generators and shared Execute reservation-station occupancy, not
physical DMA/PE activity. A print additionally requires observing a qualifying
next-loop load. Absence of such a record cannot be interpreted as zero overlap.
This run does not provide a per-tile hardware occupancy timeline, DMA timing,
or measured PE overlap. No RTL/counter build was authorized or performed.

## 9. Scratchpad/ACC Capacity

The selected generated geometry, not an upstream default:

```text
DIM=16; SP banks=4; rows/bank=4096; total SP=256 KiB
ACC rows=1024; INT32; total ACC=64 KiB
Index entries/pair=32; Scale slots=256; work IDs/loop=64
```

For both measured fixtures, maximum paired footprint is:

| Resource | Rows | Bytes |
|---|---:|---:|
| Main A | 448 | 7,168 |
| Residual A | 32 | 512 |
| Shared B | 4,032 | 64,512 |
| Main ACC reservation | 144 | 9,216 |
| Residual ACC reservation | 144 | 9,216 |

Residual A begins after the maximum Main A reservation, even in a shorter tail
panel. The unused gap is not borrowed. A is not divided into fixed equal halves.
ACC0 and ACC1 remain fixed physical halves. Excess residual lane rows cause a
capacity refusal rather than grouped ACC context switching.

B residency persists through every consuming window. Main may replace a pair
only after old readers retire; replacement invalidates its Index descriptor.
Index configuration/writes form a transaction, with original-K bounds checked
against the resident descriptor. Scale slots retain column/generation ownership
until released; two physical pairs do not provide independent Scale memories.

## 10. Proposed Scheduling Architecture

These candidates are designs, not implemented or measured speedups.

### Main K preservation

- **Location/difference:** `paired-auto.cpp:420`; common K differs from the
  requirement to freeze Main geometry independently.
- **Measured reason:** Main loops double, but B-bank capacity already limits
  K to 28 in these fixtures.
- **Schedule/software:** freeze stripe/I/J and hardware-feasible Main K, then
  attach smaller residual windows to the whole residency.
- **RTL/ISA:** decoupling the planner is software where capacity permits.
  K=48/J=9 requires 108 KiB B residency; the present 64 KiB bank cannot supply it.
  A larger/repartitioned B region and indexed-address support need a separate
  hardware design, or J changes must be acknowledged as a different contract.
- **Ordering/proof:** retain each output's original K partial order, SCU
  boundaries, first/last flags, and checked refusal behavior.
- **Test/performance:** sweep pure and residual K bounds, compare command fields
  and exact integers, then use diagnostics-OFF matched fixtures.

### Independent Residual K subdivision

- **Location/difference:** `paired-auto.cpp:475,493`; residual fragment/window
  construction exists but feeds a bound back into Main.
- **Measured reason:** 60/264 windows and 11.84%/51.04% compact-K utilization.
- **Schedule/software:** bind original-K fragments to a frozen resident Main
  panel; independently group lane rows and subdivide compact K.
- **RTL/ISA:** current Index mapping works inside one bank. Wider resident B
  or more groups needs explicit descriptor/ACC lifetime support.
- **Ordering/proof:** preserve original block carriers, padded partial boundaries,
  and per-lane source identities; no compact-K/model-global-K confusion.
- **Test/performance:** mixed runs, tails, multiple lanes and refusal paths,
  followed by actual Index/Scale/LOOP counts and latency.

### Residual Window merging

- **Location/difference:** `paired-auto.cpp:493,895`; many small windows cost
  repeated checked command programs.
- **Measured reason:** Decode has only 2--7 active K positions per window;
  FFN has 9--23. Index capacity is 32.
- **Schedule/software:** merge only admissible ordered fragments while retaining
  every required SCU boundary and residency.
- **RTL/ISA:** wider tables/windows require a parameter/protocol change; filling
  holes without changing table size still requires numerical equivalence proof.
- **Ordering/proof:** do not reassociate saturating sums merely because carriers
  match. Prove overflow bounds or retain the exact partial recurrence.
- **Test/performance:** adversarial cancellation/saturation plus compact/sparse/
  dense windows; measure commands eliminated and diagnostics-OFF latency.

### Main/Residual microtile alternation

- **Location/difference:** `paired-auto.cpp:832,879` versus
  `paired-schedule.cpp:332`; only the latter alternates microtiles.
- **Measured reason:** whole Main requests precede 60/264 residual windows;
  software issue order supplies no array-overlap percentage.
- **Schedule/software:** two ordered ready streams, an explicit resource
  scoreboard, alternating eligible microtiles and draining only the survivor.
- **RTL/ISA:** verify how a Main microtile selects its original resident B
  subrange and how SCU/work metadata accompanies it. The current single Execute
  generator cannot interleave two whole requests; a checked lower-level
  interface or generator/arbiter extension is required.
- **Ordering/proof:** preserve K order within each ACC epoch; never merge Main
  and Residual's distinct integer scales or reorder saturating contributions.
- **Test/performance:** compare emitted microtile trace with the existing
  behavioral model, then prove actual occupancy/completion on hardware.

### Double-buffered K prefetch

- **Location/difference:** `paired-auto.cpp:774,1005,2158`; logical loads are
  not independent prefetch commands.
- **Measured reason:** both pairs are used, but releases precede the next
  residency; full-K B also exceeds one current bank.
- **Schedule/software:** explicit current/next descriptors, load-next while
  current computes, and retire only after every reader completes.
- **RTL/ISA:** add checked load-only/resident-Main lowering if existing raw
  load and address controls suffice; otherwise extend the interface. Do not
  assume a LOOP_WS that also computes is a harmless prefetch.
- **Ordering/proof:** disjoint A/B ranges, Scale budget for both panels, Index
  descriptor identity, and no overwrite before completion.
- **Test/performance:** delayed-completion ownership tests plus hardware DMA/
  Execute traces. No prefetch speedup is currently measured.

### Drain placement

- **Location/difference:** `paired-auto.cpp:1734,1965,1005`; existing drains
  enforce required safety and also constrain overlap.
- **Measured reason:** 73/309 logical RTL drain groups; individual Releases
  remain 33,024/147,456. Their API time does not isolate hardware waiting.
- **Schedule/software:** move a drain only with explicit last-consumer,
  same-buffer Index, staged-A, ACC, and Scale ownership dependencies.
- **RTL/ISA:** purely redundant software fences may be removable after proof;
  making Release non-global would be a hardware/ISA semantic change.
- **Ordering/proof:** audit every producer, consumer, readiness, sticky failure,
  cleanup, safe-close and fence path. Do not infer safety from no-GEMM loops.
- **Test/performance:** full success/error command streams, injected failures,
  unknown-DMA abandonment, and matched unprofiled measurement.

### Single-mode reclamation and ACC groups

- **Location/difference:** `paired-auto.cpp:368,463`; fixed ACC halves and
  one residual epoch prevent general group/tail reclamation.
- **Measured reason:** current fixtures use only 144 of 512 rows per ACC half;
  this does not remove their B-bank bound.
- **Schedule/software:** group residual lane rows, save only unfinished epochs,
  restore exact raw INT32, then switch partition only after the other path
  and all its readers/stores complete.
- **RTL/ISA:** raw full-width ACC MVIN/overwrite exists in the base ISA.
  A checked restore wrapper with row stride, ranges and epoch ownership is
  missing; it is not evidence of missing arithmetic RTL. Full-capacity
  borrowing still needs partition/addressing validation and possibly RTL.
- **Ordering/proof:** restore without SCU or rounding, resume with accumulate,
  and preserve partial sums across save/restore and failure.
- **Test/performance:** raw INT32 round trips, group boundary saturation,
  early Main/Residual completion, fatal DMA, and actual transfer/latency costs.

## 11. Required RTL/ISA Changes

No hardware change was made. Some missing features are software interfaces,
not absent arithmetic:

- H3-7B's new checked bias wrapper demonstrates the existing full-width ACC
  MVIN-add ISA; a future raw overwrite/restore wrapper is a separate operation.
- Independent residual subdivision can be software within existing capacities.
- Preserving K=48/J=9 cannot fit the current paired B bank. Wider residency
  requires capacity/addressing changes or an explicitly different tiling.
- Microtile arbitration, resident Main subrange selection, load-only prefetch,
  and dynamic partition reclamation require individually validated interfaces.
  This report does not claim all require new opcodes, nor that all can be
  achieved through existing public wrappers.
- Hardware DMA/PE overlap measurements need available counters or a separately
  approved instrumentation image. A generator-busy print is not a DMA timeline.

## 12. Measurement Results

Fresh diagnostics-OFF values are median / unscaled MAD in milliseconds, n=7:

| Case | Serial steady | Paired RTL steady | Conservative steady |
|---|---:|---:|---:|
| Decode 1x768x768 | 8.762 / 0.013 | 14.325 / 0.020 | 14.494 / 0.016 |
| FFN 16x3072x768 | 52.423 / 0.088 | 78.001 / 0.121 | 79.014 / 0.134 |

Paired/Serial is 1.635 for Decode and 1.488 for FFN: these are slowdowns, not
achieved speedups. Cold medians are respectively 23.557/29.105/29.251 and
115.216/141.195/141.880 ms. Cold samples clear the prepared-weight cache;
steady samples hit it with zero packing/validation. No sample was discarded.
All 84 baseline cold+steady samples passed the existing bitwise golden and
counter guards with zero fallback.

The existing microbenchmark has route bring-up/cache warmups, excluded from
statistics, and rotates/reverses order. These are not Full Inference warmups.
H3-7B's new real-inference launch uses `--no-warmup`.

Separate active D1 diagnostics, n=3, Paired steady exclusive buckets:

| Active diagnostic quantity | Decode | FFN |
|---|---:|---:|
| Validation ms | 0.912 | 6.231 |
| Copy ms | 0.745 | 3.760 |
| Allocation ms | 2.413 | 10.426 |
| Deallocation ms | 0.840 | 3.397 |
| Construction ms | 1.883 | 8.662 |
| Other ms | 0.018 | 0.020 |
| Instrumentation bookkeeping ms | 3.251 | 12.936 |
| Route+executor root envelope ms | 10.101 | 45.424 |
| Logical copied bytes | 154,139 | 726,853 |
| Two report-copy inclusive envelopes, ms | 2.268 | 10.404 |

Root scopes are disjoint; inclusive site envelopes contain nested work and are
not additional contributions. Medians of buckets need not sum to the median
root. Active graph medians are 22.288/111.383 ms, 55.59%/42.80% above the
production controls. Neither bookkeeping subtraction nor inclusive copy timing
recovers an uninstrumented component cost. D2 was not started.

The issue-profile graph medians, also separate, are 14.546/78.557 ms.
They are not mixed into the primary table.

Evidence:

```text
/tmp/h3-6e-guest.sh
/tmp/h3-6e-results.log
/tmp/h3-6e-inventory.log
/tmp/h3-6e-conformance-notes.md
ggml/src/ggml-potal/tests/potal-paired-inventory.cpp
```

The guest suite printed `__H36E_SUITE_PASS__`, synced and powered off.
The simulator exited 0 after 1,690.0 host seconds. Its whole-emulation cycle
summary is not an individual kernel-cycle or DMA measurement. The rootfs
mounted with the pre-existing unchecked/no-journal warning; extraction was
read-only and the image was unmounted afterwards.

## 13. Optimization Priority

1. Define the intended resident B capacity and address contract before promising
   Main K preservation. The measured K reduction is B-bank-limited.
2. Address tiny Residual window work with an order-preserving design. Count
   actual Index/Scale/program submissions and padding, not only GEMMs.
3. Design explicit load/residency/ACC-epoch ownership and microtile arbitration
   before attempting prefetch or loop interchanges.
4. Obtain hardware completion/DMA/occupancy evidence for that design. Existing
   API times quantify caller cost, not the hardware portion to be recovered.
5. Treat Report Copy, Scale micro-optimizations, and Attention as separate
   experiments. No D2 or Paired schedule replacement is included here.

The conclusion is measured Paired overhead plus specific dataflow gaps, not a
claim that weight reuse is absent or that a proposed pipeline has already
achieved speedup. The amount recoverable by a future schedule remains unmeasured.
