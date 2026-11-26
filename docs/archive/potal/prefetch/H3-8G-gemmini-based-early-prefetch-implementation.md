# H3-8G: scheduler-owned early LdB prefetch

Updated: 2026-10-30 (+0900).

## Implementation and current verification boundary

The opt-in early path reuses the original `LoopMatmulLdB`, LOAD2 arbitration,
reservation station, LoadController, DMA and Scratchpad. It is not an earlier
raw-MVIN bypass and does not execute the future Main.

The new six-test RTL suite passed on Verilator. The existing twelve-test
native acceptance scope passed for DIM32, with completion reuse off and on;
DIM16 preservation and the added diagnostic API passed twelve tests each.
Native numerical/command emulation remains software-only.
The new RISC-V hardware-test binaries compiled successfully.

The corrected g2 image subsequently passed complete generation, FPGA build,
timing/bus-skew, actual numerical/attention gates, workload physical traces
and all 210 checked performance samples. Guest and simulator both exited
0. These are separate actual-FPGA evidence, not inferred from software or
Verilator tests. The diagnostic correction increased the Early RTL suite
to seven passing tests. See the final g2 section and
`H3-8FG-performance-comparison.md`; small measured gains do not justify
changing production defaults. Earlier F1 measurements used the old D2 image.

Sources remain uncommitted. Software base:
`b5683e474b6b8708fe90f3d5a150a56322f64081`.
Chipyard base: `bf1d98ca8a291a53845648ae475c80e01fd0b83d`.
Pinned Gemmini: `185293e87cbbc2a336410f49a3fd73d01ed8d04e`.
The original architecture audit is `H3-8F0-original-gemmini-prefetch-audit.md`.

## Why this seam

Original two-slot LdB/Execute load-ahead remains in PoTal. Existing ordinary
LOOP_WS is nevertheless a complete compute descriptor: NULL A/C do not
disable arithmetic. Current Main Scale slots are reused after Release,
and the checked ordinary emitter has CPU publication and CONFIG commands.
A drop-in real-next-Main submission would change Scale allocation/lifetime,
ACC contribution/epoch order and emitter admission. A load-only scheduler
slot is the smaller contract.

The chosen conservative schedule is:

```text
prepare adjacent opposite-pair next-B identity
current Main: existing checked ordinary LOOP_WS
next B: five scheduler configuration words + checked LdB-only admission
one additional real global Fence: current Main and future B drain
current Index/Residual/readback/Release: original order and commands
next Main: real checked descriptor, only encoded B address suppressed
```

No CPU clock read, logging operation or CPU fence is inserted between Main
submission and the load-only descriptor. The extra Fence is included once
in `main_prefetch_issue_s`/`issue_s`, not also added to `fence_s`.
That interval is CPU-observed, not isolated DMA or accelerator execution
time. Diagnostic snapshots and printing occur after the added Fence.

The extra Fence deliberately does not advance the executor's
`drained_upto`: it cannot replace an existing manifest drain boundary.
This preserves the current pending-B exclusion and all old Fence/Release
calls. It is a bounded early-Main pipeline with an explicit synchronization
cost, not uninterrupted concurrent future compute.

## Protocol and lifecycle

`include/potal.h`, `include/potal_rocc.h`, and `potal.c` define:

- `POTAL_FUNCT_LDB_ONLY=29`.
- Exact capability query `0x504f4c4400000000`, response
  `0x504f4c4401200001`.
- `potal_paired_h2_early_enable(session,counter)` probes while idle.
  Old images receive only safe COUNTER commands; no new opcode is sent
  without an exact match.
- Existing `potal_paired_h2_panel_prefetch` selects the new emitter only
  after successful enable. Existing D2 CONFIG_LD/MVIN lowering remains.

The five fresh slot words encode I=1, positive J/K, padI=padK=0, valid
padJ, zero A/D/C pointers and strides, and real B pointer/stride.
Canonical run has xd=1, rs1=`0x10|pair`, rs2=0. Admission acknowledges
ownership, not DMA completion. Malformed/unsupported runs have no load
effects; rejected staging is consumed and must be completely restaged.
xd=0 cannot start the load-only generator or consume a response slot.

C11 validates original allocation extents, exact adjacent K and same J,
opposite owned pair, issued current Main, no target readers, and fresh
owner/generation. Ownership and pending-B precede the first staging word.
Only a successful admission sets `load_issued`; only a later checked
global Fence/Release publishes the fully issued B. Queries never publish.
All attempted words invalidate F1 completion reuse before effects.

Answered partial failure retains the original error and permits the
existing single cleanup drain. Failed publication is forbidden.
Unknown responses or failed Fence/Release retain quarantine and borrowed
source ownership without retry, free, close or fallback.

Production selection is explicit:

```sh
export GGML_POTAL_PAIRED_AUTO=ON
export GGML_POTAL_PAIRED_KTILE=ON
export GGML_POTAL_PAIRED_PREFETCH=ON
export GGML_POTAL_PAIRED_EARLY_PREFETCH=ON
```

Invalid dependencies or unsupported geometry/capability are refused.
Unset/OFF keeps existing paths and production H2 default.

## RTL reuse, layout and first failure

`generators/potal/src/main/scala/potal/LoopMatmul.scala` adds a per-slot
`loadOnly` discriminator and fresh-word validity. It marks A/D/EX/store
stages skipped, suppresses their requests explicitly and clears all
compute/Index/Scale metadata. LdB alone starts and completes its real
generator. Generator idle/slot retirement does not imply DMA completion.

The scheduler shadows canonical LOAD2 state1 only on an actually forwarded
CONFIG handshake. It checks stride, identity scale, block stride, repeats,
source representation, widened final address, and B-bank capacity before
admission. Existing LOAD2 issue and returned-byte completion remain ordered.
No new DMA engine or load state was added.

DIM32 placement:

| Resource | Pair 0 | Pair 1 |
|---|---|---|
| Main/Residual A bank | bank0, rows 0..2047 | bank2, rows 4096..6143 |
| B bank | bank1, rows 2048..4095 | bank3, rows 6144..8191 |
| B rear base | `4096-K*J*32` | `8192-K*J*32` |
| Index table | 0 | 1 |
| Main ACC / Residual ACC | ACC0 / ACC1, shared across pairs | same ordered epochs |

Each B footprint must fit 2048 rows. Physical bank separation alone is not
ownership: admission checks configured future operand ranges, while the RS
retains allocated physical dependencies. Current Main and compact Residual
order, K32 carriers, SCU/ACC Sat32 semantics and final reconstruction stay
with the original plan.

**First RTL failure:** a load-only tail never requests LdA. The original
weighted arbiter therefore retained head's idle A loop ID and forced that
invalid stream, starving tail B while head Execute was intentionally
stalled. The first Treadle run timed out at the unchanged credit test.
It is preserved in `/tmp/h3-8fg/rtl-ldb-only.log`; the outdated run was
stopped after diagnosis.

The cause-level fix relaxes force-A only for an admitted load-only tail
after the head's A generator has started and become idle. Range admission
and real RS dependencies remain. Ordinary full-loop and resadd arbitration
are unchanged. `/tmp/h3-8fg/rtl-ldb-only-v2.log` records all six tests passing
on Verilator after this fix.

## Physical diagnostic contract

The separate early image also exposes an idle-reset, Fence-frozen
28-field COUNTER snapshot. No such capability is claimed for D2.
`POTAL_EARLY_TRACE=1` is sampled before execution, not in the critical
Main-to-admission interval.

Reset command is tag OR `0x100`, returning 1 only when the actual global
drain conjunction is true; otherwise it returns 0 without clearing data.
Read command is tag OR `0x200` plus field index. Special reads do not
change existing counter configuration/reset/snapshot state.

Fields preserve absolute 64-bit target-clock timestamps across K windows:

- Actual scheduler admission, separately labeled.
- First/last accepted LoadController-to-Scratchpad B DMA row requests,
  distinguished from TileLink issue or DRAM response completion.
- First/last selected, masked B SRAM writes per pair.
- First/last qualified representative PE MAC events per pair.
- Seven B-write/MAC/cross-pair cycle counts.
- Fence reaching the ingress head and canonical acceptance after
  `olderGemminiDrained && wb.io.drained`.
- First/last busy events, interval start/end, seen flags and saturation
  overflow flag.

Snapshots freeze on the first qualified Fence. Current-Main identity follows
the chosen command order: no current Residual is issued before that Fence.
Host output binds each interval to run, stripe, K panel, pair and software
generation. A subsequent panel's first MAC gives next-Main start on the same
absolute clock. Diagnostic mode also snapshots the final Main with an extra
real Fence, so the last prefetched panel has a next-Main witness. Primary
timing enables neither the snapshots nor that diagnostic-only Fence.
These events are not CPU rdcycle values or full-array
utilization. Cross-pair counts are not saved cycles.

`potal-paired-smoke --early-prefetch-observe` checks independent raw goldens.
`--observe-early --case NAME --repeat 1` reuses the exact benchmark tensors
and prints diagnostic intervals; its timing is explicitly not performance.
`--bench-early --case NAME --repeat 7` compares H2, K-tile, D2, F1 and early
LdB with rotating/reversed route order. Primary timing rejects enabled
physical trace. No fallback is accepted for F1/early routes.

## Initial verification artifacts and subsequent gates

- Native: `/tmp/h3-8fg/g-native-{default,reuse}.log`,
  `/tmp/h3-8fg/g-d16-native.log`, `/tmp/h3-8fg/g-d32-trace-native.log`.
- Verilator: `/tmp/h3-8fg/rtl-ldb-only-v2.log`, new outputs under
  `/tmp/h3-8fg/rtl-run/`.
- Cross build: `/tmp/h3-8fg/g-d32-performance`.
- Final software regression: `/tmp/h3-8fg/g-final-native-tests.log`,
  12/12 passed; final cross-build completed with exit 0.
- Existing DIM32 regressions: `/tmp/h3-8fg/rtl-existing-d32.log`,
  8/8 passed across ingress, real drain and physical observation suites.
- Isolated guest rootfs: `/tmp/h3-8fg/rootfs-g.img`; all eight executable/
  library payloads and the script/hash file were extracted and compared
  byte-for-byte after staging; read-only `e2fsck` passed.
- Guest hashes: `/tmp/h3-8fg/staged-g.sha256`.
- New configuration:
  `FireSimQuadRocketSinglePoTalD32EarlyPrefetchObservationConfig`.
- New recipes/HWDB/runtime/workload paths use H3-8G names under
  `/home/alveo/firesim/deploy/`.
- New build directory: `/home/alveo/build_firesim_h3_8g_d32`.

The tests cover malformed/stale descriptors, both pairs and tails,
same-pair future hazards, held responses/output, exhausted load/EX budgets,
slot reuse, true returned-byte drain, frozen event snapshots, busy reset,
capability isolation, exact commands, independent numerical goldens,
partial failures, cleanup and quarantine.

The first full FireSim elaboration caught a diagnostic integration error:
the representative MAC event is `Valid(Bool)` carrying pair identity, not
`Vec(2, Bool)`. `Controller` now qualifies each pair by valid and pair bit,
matching the existing physical counter helper. The failing elaboration log
is preserved at `/tmp/h3-8fg/generate-early-rtl.log`; the corrected retry
uses `generate-early-rtl-v2.log`. Harness-only tests did not prove that
top-level connection.

Actual workload physical overlap, numerical/attention gates and five-route
measurements are recorded in the final g2 section below. They remain
separate from the source and harness evidence.

### Inherited implementation-constraint warnings

During new-image implementation Vivado emitted `12-4739` for the generated
`host_clock`/`clock_1000MHz,harnessbinder_clock,reference` clock and its
multicycle paths. The generated implementation XDC is byte-identical to
the preserved D2 file. D2's `impl_1/runme.log:106-122` contains the same
three Critical Warnings. Its routed report has zero `no_clock` and zero
`unconstrained_internal_endpoints`, but retains methodology warnings
TIMING-4, TIMING-27 and TIMING-54 plus missing I/O-delay warnings.

This is an inherited integration limitation, not a new Early-specific
warning and not a reason to claim a warning-free build. The new routed
report was independently checked for actual clock periods,
setup/hold/pulse slack and unconstrained endpoints, alongside placed
resource use, as recorded below. No timing constraint or warning was
suppressed or changed.

### New-image routed timing and placed resources

The new `impl_1/design_1_wrapper_timing_summary_routed.rpt` reports
WNS **0.062 ns**, TNS **0**, WHS **0.009 ns**, THS **0**, WPWS **0** and
zero failing setup, hold and pulse-width endpoints. Vivado states all
user-specified timing constraints are met. `no_clock`, unconstrained
internal endpoints and combinational loops are all zero.
The separate routed bus-skew report has 28 reported paths, all `MET`;
minimum reported skew slack is 3.124 ns.

The actual generated host clock `clk_out1_design_1_clk_wiz_0_0` has period
33.330 ns (30.003 MHz as reported from rounded input constraints), setup
slack 6.845 ns and hold slack 0.009 ns. This is the 30 MHz FPGA host
clock, not a claim that the simulated target runs at that architectural
frequency. The inherited methodology/I/O warnings above remain.

Matched **placed** utilization, not a post-route resource claim:

| Resource | Preserved D2 | Early | Delta |
|---|---:|---:|---:|
| CLB LUTs | 521700 (40.02%) | 523748 (40.17%) | +2048 |
| CLB registers | 485093 (18.60%) | 487122 (18.68%) | +2029 |
| Block RAM tiles | 563.5 (27.95%) | 563.5 (27.95%) | 0 |
| DSPs | 202 (2.24%) | 212 (2.35%) | +10 |

Both source reports are `impl_1/design_1_wrapper_utilization_placed.rpt`
under their separate D2/Early build directories. The Early total includes
the timestamp observation circuit, descriptor validation and scheduler
changes; it is not an isolated prefetch-only area estimate.

The routed DRC summary contains warning-severity entries only: 572
violations versus D2's 540. The warning categories are identical.
DPIP-2 increases 183 to 195, DPOP-3 154 to 164, DPOP-4 150 to 160;
the other category counts are unchanged. The two AVAL-318 warnings refer
to the same existing LdD/StC DSP nodes in both builds. No warning was
waived. Router finalization reports zero failed, unrouted, partially
routed or overlapping nets.

### Final package and execution boundary

The complete FireSim bitstream build finished with exit 0 and
`H3_8G_BITSTREAM_MANAGER_EXIT=0`. The final package is
`/tmp/h3-8fg/firesim-g.tar.gz`, SHA-256
`f34acbf9336c4e39e9e0d813f7d88fe66adfa7a1fa3661f3b4f8cf7876ab2718`.
It passed gzip integrity checking and byte comparison with the manager
result. Extracted bit/mcs files match the generated originals.
`/tmp/h3-8fg/bitstream-g.sha256` records all three hashes; driver hash is
in `/tmp/h3-8fg/driver-g.sha256`.

The package metadata names the exact Early configuration and outer
FireSim commit `5c3458a3f54917ab9502bda4a8b3cc3dd513304d`. That outer
commit alone does not identify the dirty Chipyard/PoTal/software source.
Reproduction evidence therefore also includes:

- `source-hardware-tracked.patch`: scoped Chipyard/PoTal tracked changes.
- `source-preserved-gemmini.patch`: read-only snapshot of incoming dirty
  Gemmini changes, without modifying that worktree.
- `source-new-rtl-test.tar.gz`: the new untracked RTL test.
- `source-software-tracked.patch`: the software changes.
- `source-g.sha256`: hashes of those four artifacts.

All paths above are under `/tmp/h3-8fg`. These are evidence artifacts,
not Git commits. Current-session `git diff --check` passed in both source
repositories. The incoming dirty CDE stream hash remains unchanged.
All nine frozen F1 executable/library hashes still match their staging
manifest.

An extra check against D2's old `initial-rootfs.sha256` failed. That
reference was written at 11:40:39 +0900, while the D2 rootfs mtime and
ctime are both 16:15:03 +0900, before this resumed session began at
18:14:41 +0900. Its current SHA-256 is
`afd5c74f6d16c837cfc7c74a66e591368783d40bb7460dab5cfcb730f65c2c83`.
The old reference is therefore not a resume-time baseline; do not claim
it passed or rewrite it to hide the mismatch. This session used separate
F1/G rootfs copies and did not write the D2 source image. The timestamps
are preservation evidence, not a substitute for a missing resume-time
cryptographic baseline.

The user subsequently authorized programming and the remaining hardware
verification. Programming U280 BDF `01:00.0` completed with manager exit 0,
recorded as `H3_8G_PROGRAM_MANAGER_EXIT=0` in
`/tmp/h3-8fg/program-g-v2.log`.

The first attempt stopped before flashing while copying the boot binary:
`WorkloadConfig.bootbinary_path` prefixes the configured relative path with
`workloads/h3-8g-d32/`, but that input directory did not exist. Creating
the directory and checking the resolved existing boot file fixed the
path contract. The initial failure remains in `program-g.log`; the
original boot image was not changed.

At that first programming checkpoint the execution gates were pending.
The first failure and corrected successful run are recorded below, separate
from build/programming PASS. All scripts are under `/tmp/h3-8fg`.

### First actual-FPGA run and diagnostic integration correction

The first programmed image passed API smoke, all nine raw Main/Residual
goldens, full attention and PV-range smoke. The PV-range witness retained
49 jobs, 16 bitplane jobs, 71 fences and 28224 balanced Scale
writes/releases with `exact=1 status=0`. The raw 63x129x544 fixture reported
379 cross-pair SRAM/MAC cycles. This is not a Decode/FFN timing result.

The subsequent diagnostic Decode run failed at the first
`h2_early_trace_begin` with `POTAL_ERR_BUSY`. The benchmark process then
disabled its issuer, so all samples in that failed process are invalid.
The guest script exited 1. Linux was synced and powered off; the enclosing
simulator then exited 0. These statuses are deliberately separate.
Evidence is under `/tmp/h3-8fg/extracted-g-v1/results`,
`g-run/uartlog`, and `g-run/driver.{log,exit}`.

Two observation integration defects were found:

1. COUNTER commands travel through `unrolled_cmd`. The diagnostic reset
   checked the same `allDrained` conjunction as ingress Fence, including
   `!unrolled_cmd.valid`. Unlike ingress Fence, reset therefore counted
   itself as pending and could never be accepted. The correction is
   diagnostic-only and must not weaken canonical Fence/Release drain.
2. Early capability probing used legacy counter0 setup `0xA` after the
   seven-counter observation had begun. That reconfigured B0's counter to
   event NONE, explaining the first run's zero B0-write count despite
   nonzero cross-pair evidence. C11 now retains counter0's full-width
   prefix while idle and restores its owned B0 selector after probing,
   including unsupported-capability responses. Other counters are not
   reset. A focused callback regression models the real configuration
   reset, prior counts and later events; DIM32 and DIM16 API/execution
   regressions each passed 2/2.

The first image, package, rootfs/results and driver/software binaries
remain preserved. Frozen first-run binaries are in `g-v1-frozen`.
The corrected build uses `/home/alveo/build_firesim_h3_8g2_d32` and a
separate manager recipe. Its completed RTL and actual-hardware results
follow; the initial harness tests alone did not prove the integration.

## Final corrected g2 result

The diagnostic reset now excludes only its own supported, canonical
COUNTER head from diagnostic occupancy. All other scheduler, queue,
RS/controller, Scratchpad and writeback predicates remain required.
`olderGemminiDrained` for canonical Fence/Release retains exactly the
original conjunction. `PoTalTraceReset` is shared with the real-queue
regression rather than reproducing an independent test-only predicate.
The new test routes COUNTER through LoopMatmul and its output queue,
checking idle acceptance, held responses, malformed commands, queued
work, outstanding DMA, preserved epochs and qualified Fence freezing.
All seven Early tests passed in `/tmp/h3-8fg/rtl-ldb-only-g2.log`.

Final image configuration is unchanged; source correction and build
provenance distinguish g2. The complete build and programming exited 0:
`build-g2-manager.log` and `program-g2.log`. Final package
`/tmp/h3-8fg/firesim-g2.tar.gz` has SHA-256
`8b1f652e8b5ba092e15c134babb332ee621888d2a6a396c4c8d6f8ba0bf99991`.
Extracted bit/mcs match the generated originals. Source snapshots and
hashes are `source-*-g2-tracked.patch`, `source-g2-rtl-test.tar.gz` and
`source-g2.sha256`. `staged-g2.sha256` identifies all eight guest binaries/
libraries and was checked inside the guest.

g2 routed timing reports WNS 0.047 ns, WHS 0.010 ns, WPWS 0, all summed
violations zero, no failing endpoints, no unclocked registers and no
unconstrained internal endpoints. The host clock is 33.330 ns / 30.003 MHz;
its setup/hold slack is 6.657 / 0.010 ns. All 28 bus-skew paths are MET,
minimum slack 2.868 ns. Placed resources are 523734 LUTs, 487119 registers,
563.5 BRAM tiles and 212 DSPs. The same methodology warning categories
and 572 warning-severity DRC entries remain. No constraint was waived.

Actual FPGA gates passed again: API, all nine independent raw integer
goldens, full attention and PV-range. The latter retained exact=1,
49 jobs, 16 bitplane jobs, 71 fences and 28224 balanced Scale operations.
The corrected B0 physical counter reports 832/768/1408 writes across the
three Early raw fixtures, matching the respective H2/K-tile controls.
The full-row raw fixture has 355 simultaneous cross-pair cycles; both
smaller fixtures and H2/K-tile controls have zero.

Decode/FFN/tail pipeline traces contain 24/96/8 prefetch intervals.
Positive overlap occurs in 3/96/8 intervals; simultaneous-cycle totals
are 3/277/89. Every interval was matched to next Main by run, stripe,
pair, panel and generation, and admission/DMA/SRAM/Fence/next-Main
ordering was checked. Decode's 21 zero-overlap intervals remain zero.
Admission occurs early, but accepted DMA requests arrive only 74/89
cycles before current last qualified MAC for Decode/FFN. This is not
proof of DRAM waiting or a particular downstream stall mechanism.

All 210 primary performance samples passed: three workloads, five routes,
two cache phases, seven unique iterations each. Every median/MAD was
independently recomputed. Full data are under
`/tmp/h3-8fg/extracted-g2/results`; `analysis-g2.json` records the checks
and summaries, with `results-g2.sha256` and `analysis-g2.sha256` preserving
provenance. Guest `H3_8G2_EXIT=0` and driver exit 0 are both recorded in
`g2-run`; read-only e2fsck passed after sync and poweroff.

The final comparison report contains all route medians/MADs, stage
intervals, physical timing distinctions and the adoption decision.
Early's steady gain over F1 is only 0.31% Decode, 0.66% FFN, 0.68% tail.
K-tile remains faster on Decode/FFN; tail cold does not improve. Preserve
all opt-in/default boundaries and do not promote Early or repeat full
GPT-2 inference on this evidence. No commits or push were made.

### Final MAC attribution and original-fixture supplement

The completion audit required the original raw 63x129x544 fixture itself
to be timed and Residual MAC counts to be separated from Main. Both now
passed on the same g2 FPGA, with supplementary software only.
`POTAL_EARLY_TRACE=2` isolates Residual commands with diagnostic-only real
Fence boundaries before reset and after issue. Primary timing has tracing
unset and keeps its original scheduling. `=1` additionally reports Main
MAC-active counts; both output the window and existing identity fields.

Four checked invocations give identical per-invocation Main/Residual
representative MAC-active counts: Decode 576/456, FFN 36864/29952 and tail
5355/4590. Raw fixture five-route timing has seven samples each, independent
Main/Residual golden checks and rotated/reversed route order. Medians
H2/K-tile/D2/F1/Early are 2.737/1.966/2.195/2.183/2.122 ms. This is checked
runner time, not graph time or isolated accelerator service time.

All 35 supplementary samples and all six Main/Residual trace processes
passed; guest and simulator both exited 0. Evidence is in
`/tmp/h3-8fg/extracted-g3/results` and `analysis-g3.json`, with separate
g3 hashes/source snapshot. g2 primary binaries remain in `g2-frozen/bin`
and its 210 samples remain unchanged. The comparison report contains
the full supplementary median/MAD table and measurement limitations.
