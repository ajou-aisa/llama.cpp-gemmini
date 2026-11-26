# H3-6E4B1: General Resident-B Micro-tile Contract

Date: 2026-10-26. Fixed DIM16, I1/J9/K28. E4B1 remains uncommitted.

## 1. Executive Summary

Select **Option A: extend generated LOOP_WS with resident B and independent
ACC addressing**. Preserve generated PRELOAD reservation, SCU/writeback and
completion. Do not introduce a raw untagged compute path.

Original block32 phase does not require a new SCU arithmetic interface.
Existing `passes_per_block=1` and checked per-partial carrier rebinding preserve
the original DIM16 partials, including odd-phase starts. This costs more Scale
entries than P2, so admission and measured command overhead remain explicit.

General addressing and indexed Main ACC0 still require RTL changes: **C**.
The software contract/prototype is separate from actual hardware. The unchanged
E4A subset passed a fresh actual RoCC regression: **A within that subset only**.
Native general-model validation passes, as do all 14 preserved Host gates.
The new target also cross-builds for RISC-V; that model binary was not run
on the guest and is not the actual hardware regression executable.
No production scheduler, Attention, quantization, RTL or FPGA image was changed.

## 2. H3-6E4A Checkpoint SHA

```text
4f2ca97d325ea5598d565bdec39730177053e02f
test(potal): validate H3-6E4A resident-B microtile interface
```

Parent: `d27563ba6041ac8234f2eac164bae841c425c7d7`.
Exactly the E4A report, four resident interface/test files and backend CMake
were staged: six paths, 2842 insertions. Cached whitespace/names/statistics
passed. The index was empty after commit; existing `models/gpt2/` remained
untracked. No push or branch creation occurred.

Earlier reports' uncommitted/HEAD statements describe their historical handoffs.
They were not rewritten to imply that those reports ran against this checkpoint.

## 3. Existing E4A Supported Domain

Fresh hardware regression retains K28/J2 B residency, Main K2/P2 native suffix,
Indexed Residual K1/P1, generated SCU, ACC0/ACC1 isolation, ordered continuation,
B-only completion-qualified publication and checked retirement, on both pairs.
It does not establish arbitrary native interior K/J or general output offsets.

Source-derived opportunities outside this wrapper are not automatically absent
from RTL. K2/J9 native suffix addressing is source-expressible, but is not a
new hardware-verified fixture here. Shifted Index descriptors can represent
some interior Residual J subviews; section 4 states their conservative limit.

Evidence audits:
`/tmp/h3-e4b1-rtl-audit.md`, `/tmp/h3-e4b1-software-audit.md`,
`/tmp/h3-e4b1-ownership-design.md`.
RTL references below are relative to Chipyard
`generators/potal/src/main/scala/potal/`; software references are relative to
`ggml/src/ggml-potal/`.

## 4. Resident-B Addressing Limitation

Current native Execute uses:

```text
B_native_begin = selected_half_end - Klocal*Jlocal*16
B_native_row(k,j,r) = B_native_begin + (k*Jlocal+j)*16 + r
```

`LoopMatmul.scala:393-446,1377`. B DRAM zero suppresses LdB, not that address
calculation (`:208-213`). Changing local J also changes its pitch.
For K28/J9 pair 0, K1/J1 selects row 8176, the final K/J tile.
Requested interior K tile 3/J tile 5 instead starts at row 4672.

Indexed gathering uses the full descriptor's base/pitch, but emits local J
without an origin (`ExecuteController.scala:357-392`;
`LoopMatmul.scala:1091-1092`). Existing ISA can shift INDEX_CONFIG Bbase:
base 4240, Jtiles 9, sourceKRows 432, indices 48..63 selects original K tile 3,
J tile 5. Its region ends at 8128, within bank 1.
Keeping sourceKRows 448 instead would end beyond the bank. This bounded
workaround loses the final K tile for nonzero J origin.

The pure Index checker admits that bounded geometry; current paired issue
requires full resident equality and local J equality (`potal.c:1186-1193`).
Thus this is source-expressible ISA with unverified wrapper integration,
not a general full-extent implementation or a Hardware PASS.

## 5. General K/J Micro-tile Addressing

Resident geometry is immutable until retirement:
pair/bank, base, K/J valid extents, resident pitch, original K/J origins,
weight identity/version and nonwrapping model incarnation.

```text
D = 16; rows_per_bank = 4096
bank(p) = 2*p+1
resident_rows = ceil(Kvalid/16)*ceil(Jvalid/16)*16
B_base = (2*p+2)*4096 - resident_rows
B_pitch = ceil(Jvalid/16)*16              # SP rows per K tile

q = original_k - resident.k_begin
z = original_j - resident.j_begin
B_row(q,z) = B_base + floor(q/16)*B_pitch + floor(z/16)*16 + q%16
B_lane(z) = z%16
```

Full K28/J9 uses 4032 rows: pair 0 base 4160, pair 1 base 12352,
pitch 144. Logical B payload is 64512 bytes, not measured physical traffic.
K28/J10 needs 4480 rows and is rejected.

Execution geometry has independent K/J selection and valid shape, activation
allocation offset/stride, physical ACC placement, Scale/Index incarnations,
output identity and first/accumulate/last.
Selection starts at DIM tile boundaries. A shortened Main partial is permitted
only at the canonical input tail; splitting an interior dot into two shifted,
saturating partials changes the numerical contract and is rejected.
Nominal K28/J9 with valid K443/J139 exercises both final shape tails.

Proposed native metadata uses an effective B origin plus retained resident
pitch. Indexed mode keeps the full Index descriptor, adds original resident
J origin to the gather metadata, and keeps panel-relative Index values.
The reverse physical preload counter maps through logical row `15-counter`;
offsets apply to logical coordinates, not to the reversed counter.

## 6. SCU Metadata and Original-K Phase

Each canonical Main partial has:

```text
original_block = floor(original_k/32)
original_phase = floor(original_k/16)%2
partial = signed DIM16 dot
shifted = Sat32(partial * 2^carrier)
ACC = first ? shifted : Sat32(ACC + shifted)
```

Starting at K112 means block 3/phase 1; the next partial uses block 4/phase 0.
Restarting a P2 block counter at phase 0 would incorrectly reuse block 3.
The selected mapping uses existing P1:

```text
slot(p,j) = scale_base + p*Jlocal + j
slot carrier = original carrier for partial[p].original_block and original J
```

Two halves of a block may duplicate the same carrier, but never merge their
dots or saturation. Local J repacking avoids requiring a new Scale J-stride
field. Existing Scale encoders and generation/liveness checks remain in use.
`PoTalScaleBlockCounter.scala:23-48`, `LoopMatmul.scala:1094-1111`,
`PoTalWriteback.scala:150-159,222-250`.

Residual retains ordered compact fragments within one original block32.
A fragment may span both original DIM halves; it is still one compact dot.
Different blocks cannot be merged into one SCU partial. Two short fragments
retain interior DIM padding with valid Index entries and zero activation
digits; only the final missing suffix becomes the shape tail.
Indexed Main retains canonical contiguous DIM partials, not Residual compaction.

Carriers are unsigned shifts 0..32767 or `0x80000000` zero sentinel.
Shift 0 is identity. Negative mathematical carriers are rejected before
unsigned conversion; no right-shift rounding contract is invented.
Signed negative dots and negative saturation remain supported.
`PoTalSCU.scala:5-67`, `potal.c:136-137`.

## 7. Indexed B and ACC Destination

Current admission requires indexed => paired Residual, then chooses ACC1:
`LoopMatmul.scala:1235-1258,1381-1382`.
Execute gather itself does not require ACC1. New-mode admission separates
Index enable from output destination. Required model combinations:
native/indexed Main -> ACC0; indexed Residual -> ACC1.

```text
ACC_row(i,j,r) = bank*512 + acc_base
                + ((output_i+i)*output_J_pitch + output_j+j)*16 + r
```

`acc_base` is bank-relative and tile-aligned. Capacity uses the greatest
addressed row, not merely local tile count. Each bank holds 32 DIM tiles.
Pair selection affects A/B banks, not ACC bank.
For I1/J9, J tile 5 starts at row 80 in ACC0 and 592 in ACC1.

Execute, optional StC and checked readback must agree on that placement.
The new-mode paired contract keeps D and implicit C store disabled initially;
readback is explicit. Arbitrary ACC-offset snapshots remain semantic model
operations here, not falsely encoded using the zero-origin readback API.
For the required local I1 domain, software precombines output I/J origin and
pitch into one effective ACC base. A new hardware output-pitch field is needed
only if a later extension admits multiple local I tiles; it is not part of
the strict I1 minimum.

## 8. ACC Epoch/Work Identity

An epoch belongs to a physical ACC tile and binds output identity,
original row/J origin, valid shape, weight identity/version, logical reduction
extent, contribution count and sequence. Each tile has a distinct model token.
First refuses an unfinished epoch; continuation requires the matching token
and next contribution; last is accepted exactly at the declared final partial.
Main preserves contiguous canonical K order; Residual preserves fragment order.

Completed contribution values may survive B retirement. A drained unfinished
epoch may be snapshotted. Pending readback/restore blocks overlapping mutation,
and restore copies raw INT32 without SCU or bias. It restores logical progress
with a new physical incarnation, not a stale physical token.
Metadata-bearing GEMM after restore is a software model transition only.

Hardware Work is different: current IDs are scheduler_slot*64+local_output_tile.
Each invocation seals on its last local K; writeback retires only after physical
ACC commit. Preserve that behavior and drain between dependent invocations.
`PoTalWorkTracker.scala:32-96`, `PoTalWriteback.scala:253-347`.
Keeping Work unsealed across a current Fence/Release would prevent allDrained.
Persistent Work is not part of the minimum E4B2 address extension.

## 9. Selected ISA/RTL Interface Design

**Select Option A.** Existing generated loops already provide ordered
PRELOAD/COMPUTE, atomic reservation, Scale generation, indexed hazards and
physical writeback completion. Option B would add raw-ingress metadata
snapshot/pairing and Work ownership, without inherently solving admission
or overlap.

The proposed resident-view configuration carries:

| Field | Meaning |
|---|---|
| Explicit enable/valid | Opt-in descriptor; legacy behavior remains the default |
| Effective B base and resident B row pitch | Native B addresses independent of local loop bounds |
| Indexed J origin and independent Index enable | Full resident gather without ACC1 coupling |
| ACC bank and effective bank-relative base | Stable I1 placement; software folds output origins/pitch into the base |
| Existing P1, Scale base/generation | Software-rebound original block/column carriers |
| Existing accumulate and invocation-final Work | Preserve overwrite/order and present drain protocol |

Original identities, weight generations and logical ACC epochs remain checked
software metadata. No unimplemented hardware generation tag is claimed.
Original block/phase are explicit provenance in the prototype, not mandatory
new SCU bits.

Concrete transport budget: one proposed resident-view configuration carrying
two 64-bit operand words, plus an explicit run-mode selection. This is a
semantic interface/command-count budget, not an allocated current funct.
The prototype never serializes proposed metadata into a hardware opcode.
E4B2 must finalize and test transport allocation before enabling issuance.

Actual namespace inspection: funct 0..27 are occupied by CONFIG/load/store,
raw compute/preload, loops/convolution, clock and PoTal controls; 126 is counter.
28..125 and 127 have no named allocation in the inspected `GemminiISA.scala`,
but source-unused is not an allocated opcode.
LOOP_WS has source-unused rs1[7:3], rs1[15:11], rs2[63:32];
BOUNDS has unused high 16 bits in each operand. These are candidate spaces,
not currently defined extension fields. Existing Scale/Index mandatory-zero
bits must not be repurposed.

Legacy encoders stay unchanged. New configuration is immutable per scheduler
slot, valid only when complete, and cleared on consume/reset/slot reuse.
Legacy run ignores disabled extension payload deterministically. New run
rejects missing/inconsistent metadata. No unknown-command probing or proposed
run is sent to the existing image.

## 10. Required Scala/C11 Changes

E4B2 locations, not changes made in E4B1:

- `LoopMatmul.scala:330-353,393-446`: captured B pitch/base and ACC placement.
- `:750-815,1043-1055,1157-1307`: opt-in configuration, atomic slot snapshot,
  defaults, validation and reset.
- `:1078-1112,1235-1258`: indexed J origin and decoupled Index/ACC admission.
  Keep Scale P1 and current Work semantics.
- `:1340-1435`: consistent Execute/load/store routing; resident B load remains
  disabled and D remains unsupported.
- `Controller.scala:328-400`: preserve all indexed pending/queue/RS/lookup
  protections for new-mode indexed Main as well as Residual.
- `ExecuteController.scala:357-392`: propagate indexed J origin with bounds;
  preserve logical reverse-read row mapping.
- `ReservationStation.scala:251-298`: preserve full descriptor indexed hazards
  and actual addressed native tile hazards.
- `include/potal.h`, `potal.c:258-284,456-540,1148-1212`: separate opt-in pure
  descriptor/encoder and retained-resident issue, without altering legacy APIs.
- `potal.c:570-597`: separately checked ACC subview readback if exposed.
- `include/potal_rocc.h`: add raw transport only after its allocation is fixed.

No new ScaleBlockCounter, SCU or Writeback arithmetic is needed for P1.
No early raw-load bypass or persistent Work is included in this minimum.
No Scala source was edited in E4B1.

## 11. Checked Software Prototype

New `tests/potal-e4b1.hpp/.cpp` contain pure descriptors, checks and translation.
`tests/potal-e4b1-test.cpp` supplies the separate conformance/data model.
`POTAL_E4B1_TESTS` is default OFF; the target links one ordinary `potal-api`
for pure encoder reuse, not the E4A replacement core or E3 context core.

Model tokens are a separate C++ type, not convertible E4A tokens.
The state owns its software images and never acquires an accelerator owner,
borrows a session, installs a paired sink or dispatches proposed commands.
Scale keys and full-resident Index configuration/writes use actual pure
encoders. General tile records have no funct/opcode and remain explicitly
`requires_rtl`. Real E4A execution is a separate executable.

Validation checks full layout, source extent, bank capacity, canonical K cuts,
J subranges, A range/reservation, all touched Scale columns/generations,
Index publication/incarnation, physical ACC range and output progression.
Results are published transactionally; a rejected pure translation leaves
its output record unchanged.

The object-level undefined-symbol audit found only four direct production
dependencies: `potal_carrier_is_valid`, `potal_scale_key_encode`,
`potal_index_config_encode`, `potal_index_write_encode`. No issuing or session
function is referenced by either new C++ object. The linked ordinary API
object still contains other functions; this is a call-path audit, not a claim
that the entire ELF contains no issuing code. `--hardware` rejects with exit 2.

## 12. Integer Exactness Tests

General conformance passed with 666000 assertions. Coverage includes both pairs,
K28/J9 full coverage, first/interior/final K and J, joint K112/J48,
K443/J139 tails, odd phase/block crossings, indexed Main and compact Residual.
The oracle derives addresses/values from original tensors independently of
the translated physical rows. Host source poisoning checks actual model SP reuse.

Verified distinguishing witnesses:

```text
MAX, MAX, MIN in order -> -1
MAX, MIN, MAX reordered -> 2147483646
compact (+1,-1), carrier31 -> 0; separate dense partials -> -1
Sat32(2032*2^24) + Sat32(-127*2^24), saturated -> 16777215
merged dot at carrier24 -> INT32_MAX
```

Negative signed carriers reject; shift0 and zero sentinel differ.
Stale bindings, replacement, independent ACC tiles, raw snapshot/resume,
unknown completion and exactly-one cleanup passed separate lifecycle checks.
These include partial load, first/middle/final compute, readback/restore and
Index failures; five isolated terminal cases cover direct/cleanup drain,
Release, uncertain Scale acceptance and cleanup Release after a known drain.
Live-resident overwrite, live-Scale retirement, healthy cleanup discard and
software-incarnation resurrection are refused.
No software result is a Hardware PASS.

Logical successful parent-process totals, excluding terminal child processes:
38 microtile requests, 1399 partial/J-tile contributions, 25006 Scale records,
249 Index records, 12288 snapshot-copy bytes and 2048 restore-copy bytes.
They describe a mixed conformance suite, not one workload or actual DMA.

All 14 preserved native executables passed:
`potal-api-test`, `potal-trace-test`, `potal-residual-unit-test`,
`potal-paired-plan-test`, `potal-paired-schedule-test`, `potal-paired-unit-test`,
`potal-paired-auto-test`, `potal-paired-execution-test`,
`potal-stripe-pipeline-test`, `potal-attention-test`,
`potal-resident-test`, `potal-resident-model-test`,
`potal-e3-prototype-test`, `potal-e3-context-test`.
Existing binaries were executed without changing their build directories;
protected source hashes and E4A checkpoint files match.

New target native and RISC-V builds passed in separate
`build-potal-e4b1-host` and `build-potal-e4b1-d16` directories.
Both new C++ sources also pass `-Wall -Wextra -Werror -fsyntax-only` with
the correct include paths. LSP still lacks those target paths and reports
cascading header/toolchain diagnostics; actual compiler success is the evidence.
The pre-existing generated `POTAL_SCALE_ENTRIES` warning remains unsuppressed.
An initial new-validator check compared a partial's first source against its
not-yet-bound origin; the cause was fixed, retaining the negative checks.

Final executable SHA256:

```text
native: a72bdc5954cec0076b8217a50c374848ccb717f719339a0d5c71c1e5fb5f3709
riscv:  213ab821f87864a32b9f18ed9b8d88c427bb77a0a86bf7f4e6d064395c1cbd04
```

Logs: `/tmp/h3-e4b1-analysis/native-final.log`,
`/tmp/h3-e4b1-analysis/regressions/`,
`/tmp/h3-e4b1-analysis/{hardware-refusal,undefined-symbols}.log`.

## 13. Actual Hardware Regression

Fresh unchanged DIM16 E4A execution passed on Linux CPU1/hart0.
Binary SHA256:
`b0c2fec76cb8e002b86d8d4b7f9d354a559ee1b0c1b896baa31e1fdfd8aedfb2`.
Target software-only regression passed first; actual trace run then passed.
Guest suite and simulator exited 0. No full inference was repeated.

All 2074 raw ENTER/EXIT pairs returned status 0:
504 B MVIN, 16 LOOP_WS (10 Main/6 Residual), 30 ACC MVOUT,
608 Scale writes and 608 Releases, 35 Fence,
6 Index configurations/96 writes, 88 CONFIG and 3 FLUSH.
Each of five LOOP configuration kinds occurs 16 times.
Every AB record has B DRAM address 0.

Exact fixtures retain initial Main 1760/3648 and Residual 960/136;
saturation Main -1 and Residual INT32_MIN;
continued Main 3648/7552 and Residual 2688/368.
They validate both pair banks, poisoned-source reuse, independent ACCs,
first overwrite, ordered accumulation, Release completion and retirement.

Evidence: `/tmp/h3-e4b1-results/{software-target,hardware-trace}.log`,
`/tmp/h3-e4b1-run/uartlog`, `/tmp/h3-e4b1-guest.sh`.
The first launch used an incorrect FIFO UART input and was stopped before
the test ran. The successful launch restored the existing regular-file method.
The stopped driver's generic PASSED banner was not counted as test evidence.

General interior addressing, new ACC fields, restore/GEMM integration and
hardware fault injection remain **Not verified** on hardware.

## 14. Ownership and Failure Semantics

The prototype's completion domain is software-only. A model drain cannot
complete or release any E4A handle. No second accelerator owner exists.
Reuse E4A's invariants, not its opaque core through a borrowed pointer.

Premark each potentially partial data effect. Issue return does not publish.
Only successful complete model drain publishes matching generations.
Readback and restore retain their sources/destinations and block conflicting
operations. Rejected Release is not a drain.

Recoverable failure preserves the first status and permits one cleanup drain,
even if no final compute record was accepted. A separate unknown-completion
flag survives an earlier sticky issue error. Failed drain or uncertain cleanup
quarantines state and buffers without retry or close. Terminal software tests
must retain that state until their isolated process exits.

Actual E4A protection remains a closed single-issuer contract: legacy Scale,
raw RoCC, foreign processes and process death are not made safe by this model.

## 15. Scale256/Index32 Compatibility

```text
Existing P2 current Main126 + Residual18 + next Main126 =270 >256
Full P1 current Main252 + Residual18                  =270 >256
One-partial P1 Main J9 + two-fragment Residual J9      =27 slots
```

The first two simultaneous allocations must reject before mutation.
The last is a capacity example, not a pipeline speedup.
P1 uses duplicated entries when adjacent Main phases share a carrier.
Bounded windows and completion-qualified releases keep the live high-water
mark within 256. B-only load does not require next Scale programming.

Index holds at most 32 rows per pair, tied to full resident identity.
Reprogramming invalidates the old incarnation; an incomplete transaction is
not readable. New indexed Main can use canonical identity indices, but cannot
place all 448 rows in a single Index32 window.

Keep the single-pending guard for this prototype. Future relaxation requires
separate load generations/readers, A regions, Index transactions/readers,
Scale readers/live columns, ACC writers/readers and context-buffer dependencies.
Successful global drain retires the older acceptance prefix. Resource-local
completion must never be promoted to global allDrained.
Current Release remains global; raw loads still cannot bypass configured loops.

## 16. Unmeasured Performance

No general hardware speedup is required or claimed.
Proposed command budget is 11 per new addressed LOOP versus existing 10,
if the selected descriptor uses one additional configuration command.
This is a proposed budget, not an issued command count or finalized opcode.

For full resident Main, one DIM K pass across J9 uses 9 P1 Scale slots,
144 per-column writes and corresponding releases when retired.
Twenty-eight such invocations cover the canonical Main K extent.
Splitting each into J1 instead yields 252 invocations. Whole existing Main
P2 needs 126 Scale slots/2016 per-column writes; full P1 traversal needs
252 slot-program instances/4032 writes, unless a separately proven reuse
policy changes the command stream. Residual adds its own fragment windows.

One full I1/J9 raw INT32 context contains 16*144*4 = 9216 logical bytes.
One save plus restore declares 18432 bytes before any bus framing.
These are logical extents, not actual DMA bus traffic.

The traced E4A regression test bundle took 85.759 ms, n=1, with logging and
host checks. It is not a performance distribution or comparable general-loop
benchmark.

Native x86-64 CPU validation used seven batches of 512 calls on immutable
I1/J2/K3 metadata, selecting original K112/J48. The timer includes translation,
output-record copying, status checks and checksum work; no numerical GEMM runs
inside it. All seven batches were retained:

```text
batch CPU ms: 0.633153 0.553506 0.553411 0.553184 0.551616 0.558678 0.542755
batch median / unscaled MAD: 0.553411 / 0.001795 ms
per-call batch-mean median / MAD: 1.080881 / 0.003506 microseconds
```

These are medians of batch means, not individual-call latency percentiles,
RISC-V issue latency or a production hot-path speedup.
FPGA cycles, DMA bus bytes/time and PE overlap are **Not measured**.

## 17. H3-6E4B2 Implementation Plan

1. Freeze the selected semantic descriptor and allocate its transport after
   checking the inspected funct/bit census. Bind enablement to a known new
   bitstream; never send a proposed extended run to the old image.
2. Add immutable per-slot resident B/ACC geometry, explicit mode validity and
   reset/legacy defaults. Keep new data out of existing mandatory-zero fields.
3. Implement native resident pitch/base, indexed J origin, independent ACC
   placement and consistent hazard/readback routing. Preserve P1 Scale and
   per-invocation Work retirement.
4. Add pure C11 encoding and checked retained-resident issue under existing
   ownership. Prevalidate the full program and retain partial-issue failures.
5. Compare RTL command traces with these semantic records, then run legacy
   E4A plus arbitrary interior/tails/odd-phase/indexed-Main/ACC-offset goldens.
   Check backpressure, stale descriptor reset, Index hazards and physical
   writeback completion before dependent reuse.
6. Only after exact integers and ownership gates, measure matched whole-loop
   versus microtile commands/CPU cost with diagnostics OFF and n>=7.
   Actual DMA/PE overlap needs independent hardware evidence and a later
   dependency/admission design; it is not an address-extension side effect.

Final feature classification:

| Feature | Verdict and boundary |
|---|---|
| General K/J resident addressing | C; checked software prototype B, new hardware addressing required |
| Original block/phase SCU binding | B; P1/local-J rebinding verified without new SCU arithmetic fields; general hardware execution awaits addressing |
| Independent indexed B and ACC destination | C; software exactness B, new-mode admission/address routing required |
| General ownership/epochs/failures | B; isolated software conformance verified; unchanged E4A actual primitive remains A |
| Future prefetch compatibility | B for explicit lifecycle/capacity contract; C for guaranteed early admission or independent non-draining completion |
| 270 simultaneous live Scale slots | D under unchanged Scale256 |
| Existing E4A bounded execution | A; fresh actual RoCC regression, not a general-interface A |

No production schedule adoption is proposed.

Final preservation: all 50 protected hashes, the 67-file Scala manifest,
Gemmini HEAD/pre-existing diff, host-init and models match. All E4A checkpoint
files except the additive backend CMake block are byte-identical. E4B1 changes
are three new test/contract files, this report and a default-OFF 13-line CMake
block, all uncommitted. The index is empty and branch `potal` remains at
`4f2ca97d325ea5598d565bdec39730177053e02f`. No E4B2 RTL work or push occurred.
