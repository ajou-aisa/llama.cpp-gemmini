# H3-6E4B2: Resident Micro-tile RTL/ISA Implementation

Implementation started October 26, 2026. Validation continued October 27, 2026.
The B2 changes are uncommitted. This report distinguishes source implementation,
software command interpretation, RTL execution, and physical FPGA execution.

## 1. Executive Summary

Option A is implemented: extend generated `LOOP_WS`, retaining generated
PRELOAD/COMPUTE, existing SCU arithmetic, reservation dependencies, physical ACC
writeback and invocation-local Work completion. No raw compute fallback,
production schedule replacement, prefetch, or permanent unsealed Work was added.

Native micro-tiles use an effective resident B base and the full resident pitch.
Indexed Main selects ACC0 independently of Index enable; indexed Residual selects
ACC1. Both use the original full Index descriptor and an independent J origin.
The destination has a bank-local, tile-aligned ACC offset. Checked C11 sessions
bind resident, Scale, Index, output epoch and readback receipt incarnations.

The fixed supported domain remains DIM16, I1, J at most 9, K at most 28,
four 4096-row SP banks, two 512-row ACC banks, Scale256 and Index32 per pair.
K28/J9 residency occupies 4032 B rows, with bases 4160/12352 and pitch 144.
The K3/J5 address is 4672 on pair 0 and 12864 on pair 1.

Established evidence:

- The native C11/independent-command-model test passes 544655 assertions,
  32 numerical invocations and 18740 checked result cells. Its label is
  `hardware=0`, including when run as RISC-V guest software.
- Eighteen rebuilt host targets pass, including E3, E4A, B1, B5 attention,
  paired planning/execution and stripe API regressions.
- Nine new RTL scenarios and 47 existing scenarios pass, with the backend and
  prior failed attempts disclosed in section 12.
- Six integrated generated-LOOP saturation/cancellation invocations pass 3072
  INT32 comparisons through actual PE/SCU/ACC RTL and addressed MVOUT.
- The ordinary DIM16 processor RTL simulator, new FireSim quadcore DIM16
  generated target, and matching new FireSim driver build successfully.
- The existing FPGA returns zero to the safe capability probe. Both the raw
  and checked diagnostics refuse without issuing an extension.
- A distinct new U280 image is built, timing-qualified, packaged and programmed.
  Actual RoCC returns capability `0x504f544102100007`.
- On that image the raw generated-LOOP fixture passes 42 invocations and
  193024 addressed INT32 comparisons. Production checked C11 passes one untimed
  warmup plus seven measured suites, each with 16 invocations and 4404 cells.
- Unchanged E4A passes before and after extended execution. Each trace contains
  2074 issued commands, 16 legacy LOOPs with xd=0, and no extension.

General resident micro-tile dataflow and integer numerics are Hardware A.
Physical DMA fault injection remains unverified; its failure/quarantine
semantics have software/RTL evidence, not a physical fault-injection PASS.
The optional complete processor RTL address sweep was stopped after its
passing prefix once required component/arithmetic RTL and full physical
hardware acceptance were complete. It is not counted as a 42-case RTL pass.
The initial Linux raw fixture needed writable, pinned DMA output backing;
that test-only correction preserves its oracle, encoding and RTL.

Verdict definitions follow the requested categories:

- A: implemented and actual FPGA hardware verified.
- B: software/RTL simulation verified, new FireSim FPGA execution pending.
- C: implementation blocked by an interface/RTL issue.
- D: unsupported under the fixed current geometry.

| Feature | Current verdict and boundary |
|---|---|
| General resident K/J | A: both pairs, full K28/J9, first/interior/last and K443/J139 tails pass actual hardware |
| Indexed Main to ACC0 | A: actual full-descriptor/J-origin computation and addressed output |
| Indexed Residual to ACC1 | A: actual sparse compact computation with original-K bindings |
| Original-K phase and saturation | A: K112 and all four independent saturated/cancellation witnesses pass |
| ACC effective offset/epoch | A: nonzero destinations and checked first/continue/last sequences 0/1/2 pass |
| Safe ownership | A for completed publication, retirement and close; B for injected failure/quarantine paths, without physical DMA error injection |
| Legacy compatibility | A for unchanged E4A before/after and preserved command behavior; scoped host/RTL regressions pass; two pre-existing Scala harness compile defects and the identical old/new smoke mismatch remain disclosed |
| I greater than 1, J greater than 9, K greater than 28, Index greater than 32, more than 256 live Scale slots | D: rejected by this fixed interface |
| Prefetch overlap / full paired pipeline | Not implemented; outside this task, with no performance or A claim |

No interface/RTL blocker C remains. The A verdict comes from separately executed
extended hardware fixtures, not promotion of the old E4A subset, a software sink,
elaboration, or the capability response.

Path notation used below:

```text
F = /home/alveo/firesim
C = F/target-design/chipyard
P = C/generators/potal
L = P/software/workspace/llama.cpp-potal
S = /home/alveo/h3-e4b2/source
Q = xilinx_alveo_u280-firesim-FireSim-FireSimQuadRocketSinglePoTalD16Config-BaseXilinxAlveoConfig
LOG = /tmp/h3-e4b2-analysis
```

## 2. H3-6E4B1 Checkpoint SHA

```text
fac2439aed676067287ad79e6ccc6b2ff28ac43e
test(potal): validate H3-6E4B1 general resident microtile contract
parent: 4f2ca97d325ea5598d565bdec39730177053e02f
```

Exactly these five authorized paths were staged, checked and committed, with
2275 insertions and the required co-author trailer:

```text
H3-6E4B1-general-resident-microtile-contract.md
ggml/src/ggml-potal/CMakeLists.txt
ggml/src/ggml-potal/tests/potal-e4b1-test.cpp
ggml/src/ggml-potal/tests/potal-e4b1.cpp
ggml/src/ggml-potal/tests/potal-e4b1.hpp
```

The index was empty afterwards. The historical B1 report's uncommitted statement
describes its earlier handoff; it was not rewritten. Existing `models/gpt2/`
remains untracked. There is no B2 commit, push, or new branch.

Chipyard remains on `potal` at
`c705d8a7090933a15652165ce64eac196a0a5e4a`; FireSim remains detached at
`5c3458a3f54917ab9502bda4a8b3cc3dd513304d`. These commit IDs do not identify
the uncommitted B2 source; the separate generated artifacts and logs matter.

## 3. ISA Transport Allocation

The existing custom3 instruction namespace was rechecked. New funct **28**
is `POTAL_RESIDENT_VIEW`; extended generated `LOOP_WS` retains funct **8**
and selects the extension with **rs1 bit 3**. Canonical view/run instructions
have `xd=1` and return the existing status encoding. Legacy LOOP retains
`xd=0`, extension bit zero and its original words.

Resident View funct 28:

| Word | Bits | Meaning |
|---|---|---|
| rs1 | 15:0 | Effective B SP base, in rows |
| rs1 | 31:16 | Full resident B pitch, in rows per K tile |
| rs1 | 47:32 | Indexed J origin, in tiles |
| rs1 | 63:48 | Bank-local effective ACC base, in rows |
| rs2 | 0 | Enable |
| rs2 | 1 | Physical pair, 0 or 1 |
| rs2 | 2 | Index enable |
| rs2 | 3 | ACC bank |
| rs2 | 7:4 | Execution mode, exactly 1 |
| rs2 | 63:8 | Mandatory zero |

A disabled view is exactly `rs1=0, rs2=0`. Native views require indexed J
origin zero. Bases and pitch are aligned to 16 rows; positive pitch is at most
144 rows. The selected B bank and ACC capacity are checked again at run time.

Extended run retains existing shape, Scale base/generation, P1, accumulate,
pair, path and Index fields. It requires rs1[15:4], rs1[2], rs2[63:32] and
rs2[2:0] zero. rs2[31:3] remains the existing A SP offset in tiles.
It requires real A, zero B/D/C DRAM addresses, I1, J<=9, K<=28 and P1.
Residual requires Index; Index itself does not select the ACC bank.

Four baseline CONFIG commands, five LOOP configuration words, one view and one
run form an eleven-command invocation. Legacy remains ten commands.

For pair 0 K3/J5 with ACC base 80, the native view words are
`rs1=(80<<48)|(144<<16)|4672`, `rs2=0x11`.
The indexed Main equivalent retains full base 4160 and uses
`rs1=(80<<48)|(5<<32)|(144<<16)|4160`, `rs2=0x15`.
Indexed Residual uses the same address fields with `rs2=0x1d`.

Each generator slot has independent staged-valid, five-word configuration mask,
view data and active mode. State changes occur only on `fire`. Consuming any
run, including a refused run or legacy run, consumes its staging and mask.
An admitted slot retains its active snapshot until generator retirement.
Reset/slot reuse clear it. Ordered FLUSH invalidates staging only: it does not
erase active snapshots, Index readers, Scale state or Work.

Responses are buffered with stable destination register/status under response
backpressure. Missing/invalid metadata refuses before generated effects.
Malformed `xd=0` new instructions have no unsolicited register response and
no extension effects. An incomplete indexed window refuses rather than
waiting behind a transaction that it prevents from completing.

### Safe actual-image gate

Only the already implemented COUNTER funct **126** is used before admission:

```text
1. rs1=0x000000000000000a, rs2=0, xd=1
2. rs1=0x504f544100000000, rs2=0, xd=1
required second response: 0x504f544102100007
```

Step 1 disables counter 0 and clears snapshot-read mode; its sampled response is
discarded. On the known old PoTal counter implementation the high query bits
are ignored and the now-disabled counter returns zero. The new counter
recognizes only the exact tagged word and rs2 zero, and advertises the signature
only for the complete supported configuration in `PoTalResidentCapability`.
Canonical counter operations retain their prior behavior.

This is version discrimination for the attached, known PoTal accelerator, not
safe discovery of an arbitrary unknown accelerator. Probe on its attached hart
under the issuer boundary, before counter profiling. An environment opt-in
cannot bypass a mismatch. Both old-image diagnostics observed zero and issued
no funct 28 or extended funct 8.

## 4. RTL Changes

All RTL implementation changes are under `P/src/main/scala/potal/`:

| File | Change |
|---|---|
| `GemminiISA.scala` | Allocate resident view funct 28 |
| `PoTalResidentView.scala` | Slot view bundle, exact capability query/signature, fixed supported-geometry predicate |
| `CounterFile.scala` | Exact tagged capability response, preserving ordinary counter setup and sampling |
| `Controller.scala` | Gate by real config; wire full Index descriptors/completeness and buffered extension responses |
| `LoopMatmul.scala` | Slot-local transactions, admission/refusal, resident PRELOAD base/pitch, indexed J metadata, independent ACC base/bank, consumed-run lifetime |
| `PoTalScaleCommandIngress.scala` | Export complete Index descriptor/window validity without discarding transaction/reader protection |

`LoopMatmul.scala:426` selects effective resident B addressing instead of
recomputing B from the invocation's local geometry. `:1123` adds indexed J
origin. Admission is at `:1154-1245`; generated Execute/ACC address propagation
is at `:1525-1594`.

`ExecuteController.scala` and `ReservationStation.scala` were read together
with their producers. Their existing explicit J metadata, encoded ACC addresses,
full-panel Index dependency and pending-reader consumers already support the
new admitted values. They therefore need no production change. New tests use
the actual modules, not cloned consumer logic. Indexed native/Main and Residual
addresses remain dependencies of the real reservation station.

Index dispatch ownership is canceled on every consumed indexed run, including
refusal. Active readers and descriptor transactions are not canceled by staging
invalidation. Existing generated PRELOAD reservation, reverse-row/lane mapping,
SCU partial order, ACC RMW and physical Work completion remain in use.
`SCALE_RELEASE` still waits for global allDrained.

New RTL test files, all under `P/src/test/scala/potal/`:

```text
PoTalResidentTestSupport.scala
PoTalResidentExecuteSpec.scala
PoTalResidentLoopSpec.scala
PoTalResidentCounterSpec.scala
PoTalResidentConsumerSpec.scala
PoTalResidentControlSpec.scala
```

The integrated build's existing test whitelist is unchanged. Section 12 gives
explicit source selections for these opt-in suites and the existing regressions.
No Gemmini source, array capacity, arithmetic block, production schedule or
generated RTL was hand-edited.

## 5. C11 Checked API

Modified software paths relative to L:

```text
ggml/src/ggml-potal/CMakeLists.txt
ggml/src/ggml-potal/include/potal.h
ggml/src/ggml-potal/include/potal_rocc.h
ggml/src/ggml-potal/potal.c
```

New software test/diagnostic paths:

```text
ggml/src/ggml-potal/tests/potal-e4b2-c11-test.c
ggml/src/ggml-potal/tests/potal-e4b2-test.cpp
ggml/src/ggml-potal/tests/potal-e4b2-rocc.c
ggml/src/ggml-potal/tests/potal-e4b2-hardware.cpp
ggml/src/ggml-potal/tests/support/e4b2-emulator.hpp
ggml/src/ggml-potal/tests/support/e4b2-fixtures.h
```

This report is the remaining new L path. Private build recipes, logs, the
isolated source/build tree and UART/guest orchestration are outside L.

`potal.h:674-823` declares resident references and checked load, view, Scale,
Index, epoch, run, readback and status descriptors. Pure functions are:

```text
potal_resident_ws_shift_carrier
potal_resident_ws_load_check
potal_resident_ws_view_encode
potal_resident_ws_run_check
potal_resident_ws_run_encode
potal_resident_ws_readback_encode
```

The session operations are `potal_paired_resident_{open,prepare,buffer_view,
load_issue,get,scale_program,index_program,epoch_begin,run_check,issue,
readback_issue,result,retire,close,query}`. They reuse the opaque
`potal_paired_session`, existing checked Scale/Fence entry points and the same
issuer exclusion, with a distinct resident mode. No legacy signature changed.
No test-only E4A opaque handle is adopted.

The production encoder validates the complete physical descriptor before
publishing an output program. Session validation additionally binds every
resident/Scale/Index/epoch incarnation, original K/J, expected contribution,
live slot, buffer extent, ACC collision and pending state before the first
baseline CONFIG. The eleven issue positions can still fail after acceptance;
that is an issue failure with retained ownership, not ordinary validation.

`POTAL_E4B2_TESTS` defaults OFF. The software-only conformance executable works
in API-only builds. RISC-V diagnostics additionally provide raw generated
execution and checked C11 hardware execution. The checked hardware diagnostic
uses the existing `issuer_owner`, NULL sink options and the actual gate.
A non-NULL sink is explicitly software-only and uses a narrow 64-bit counter
callback, preserving the legacy command-sink ABI.

## 6. Resident B Addressing

For immutable resident origins and valid extents:

```text
D = 16
pitch = ceil(resident_J / D) * D
base = (pair + 1)*8192 - ceil(resident_K/D)*ceil(resident_J/D)*D
q = original_k - resident_k_begin
z = original_j - resident_j_begin
B_row = base + floor(q/D)*pitch + floor(z/D)*D + q%D
B_lane = z%D
```

Native C11 lowering folds the K/J tile origins into the effective view base;
generated local k/j then use the retained pitch. Indexed lowering retains the
full base/pitch/K extent in INDEX_CONFIG and folds J origin into metadata.
The reverse PRELOAD counter still maps logical rows; it does not reverse
the selection of resident K/J tiles.

K448/J144 uses 252 tiles and 4032 SP rows. Pair 0 is B bank 1 with base 4160;
pair 1 is B bank 3 with base 12352. K3/J5 adds `3*144+5*16=512` rows.
K443/J139 has the same padded residency but valid final extents 11/11.

B-only load issues one CONFIG_LD and 252 MVIN commands, then requires qualified
completion before publication. A micro-tile emits no B MVIN, keeps LOOP B
DRAM zero and actually selects the existing SP data. The tests poison the
completed load source and use pair/K/J-distinguishing values. B address zero
alone is not treated as reuse proof.

Starts are DIM-tile aligned. A shortened native partial is admitted only at
the canonical resident tail; splitting an interior DIM16 dot would change the
saturation order and is rejected. A general unaligned lane-level subview is
not claimed.

## 7. Indexed Main/Residual

Admitted combinations are native Main/ACC0, indexed Main/ACC0, and indexed
Residual/ACC1. View Index enable and ACC bank are separate bits, cross-checked
with the run's legacy path/index fields. The logical path remains Main 0 or
Residual 1; arbitrary path/bank permutations are not exposed.

The full descriptor retains resident base, J tile pitch, original K extent,
active compact extent and pair. J origin is added separately. Neither
INDEX_CONFIG's K extent nor pitch is shortened to manufacture an interior J
subview. Last-resident-K accesses remain legal.

The C11 wrapper rejects more than 32 Index rows per pair. Compact Residual
fragments retain declared original K order and block identity. Interior padding
uses in-range Index fillers and requires every corresponding A byte zero.
Same-value reprogramming still creates a new incarnation; an old token never
becomes live again merely because descriptor values match.

New software numerical tests cover both pairs, sparse multi-fragment Residual,
Main, nonzero J origins and final K/J tails. Actual Execute and reservation
station tests cover full-panel indexed reads and independent ACC dependencies.
New-image indexed Main/ACC0 and Residual/ACC1 numerical results pass on both
pairs in raw and production-C11 execution; section 15 records the evidence.

## 8. SCU Original-K Binding

No new SCU arithmetic field is added:

```text
original_block = floor(original_k / 32)
original_phase = floor(original_k / 16) % 2
partial = signed DIM16 dot
shifted = Sat32(partial * 2^carrier)
ACC = first ? shifted : Sat32(ACC + shifted)
```

P1 binds each partial to its original block/J carrier. At K112, block 3 phase 1
is followed by block 4 phase 0 at K128. Local P2 phase restart is not used.
The binding copies original block/J identity as well as generation and carrier
values. Adjacent equal carriers do not permit combining dots.

Shift 0 is carrier 0, not the zero sentinel `0x80000000`.
`potal_resident_ws_shift_carrier(int64_t, ...)` rejects negative mathematical
shifts. Existing checked carrier validation remains active.

Independent adversaries include:

- Ordered `INT32_MAX`, `INT32_MAX`, `INT32_MIN` contributions result in -1,
  distinguishing wide reassociation and merged dots.
- Same-block compact `[+1,-1]` at carrier 31 results in 0. Two dense DIM16
  partials from that block saturate separately and result in -1.
- A positive carrier-24 partial followed by the negative partial results in
  16777215, not a merged INT32_MAX.

The original-tensor oracle uses independent int64 products/shifts and per-partial
clamping. It does not use production SCU/lowering functions. Accelerator results
come only from ACC MVOUT in the raw RTL/hardware fixture.
The six integrated numerical RTL invocations completed with exit 0 and exactly
`ordered=-1 compact=0 dense=-1 carrier24=16777215`.

## 9. ACC Destination/Epoch

Effective destination row is:

```text
physical_ACC_row = bank*512 + acc_base + local_j_tile*16 + local_i_row
```

The view base is aligned, bank-local and capacity checked. I1 software translates
output tile placement into this effective base; there is no new multi-I ACC
pitch field. Explicit readback uses the same bank/tile region, real rows,
columns and caller-checked stride.

Each physical output tile has an immutable declaration: output ID, original I/J,
shape, logical K domain or ordered Residual fragment manifest, and incarnation.
First overwrites; subsequent invocations must present the matching next sequence;
last seals the declared contribution span after successful completion.
The private state copies manifests, so caller mutation cannot change ownership.
Initialized unfinished epochs prevent close, even when B and Scale are retired.

The existing invocation Work is completed at the existing physical writeback
boundary. A later invocation accumulates into the completed INT32 ACC value,
not a permanent unsealed Work across Fence. Readback receipts bind both epoch
and sequence; pending, reused or modified outputs cannot yield an old result.
Numerical tests check all physical ACC cells, including untouched regions and
bank isolation. The raw tail fixtures first seed full destinations and then
check that partial writes preserve padded rows/columns.

## 10. Scale256/Index32 Admission

Full Main P1 uses `28*9=252` slots. Adding the specified 18-slot Residual window
would require 270 live slots and is refused before issue. Scale storage was not
enlarged. Each micro-tile programs only its live P1 partial/J window.
At most 32 compact Index rows per pair remain supported.

Scale validation checks base plus complete live window, generation, original
block/J binding, columns and existing allocation. Duplicate or conflicting
live allocations refuse. Generation values do not replace owner/incarnation
checks. Reuse follows completed consumers and matching checked Release.
Release retains the global allDrained guarantee; it is not weakened to a local
slot acknowledgement.

Index programming validates and copies the entire descriptor and every entry
before INDEX_CONFIG. Pending readers reject same-pair reprogramming; the real
RS additionally protects queued/gather dependencies. Incomplete descriptors
cannot run. Source/lane fillers are checked independently of compact counts.

## 11. Ownership and Failure Semantics

One resident session holds the existing issuer exclusion. It owns six
64 KiB pools, each 64-byte aligned: two B sources, Main/Residual A and two
readback destinations.
Its 393216-byte allocation remains owned until safe close, or permanently
retained on unknown completion. Raw/legacy bypasses are refused only while
resident mode owns the accelerator; legacy mode itself keeps its old behavior.

Before any potentially accepted B DMA, baseline/view/run sequence, Index
programming or readback, the session records pending ownership and invalidates
old publication. Success is published only after qualified allDrained and a
complete issue. Successful command acknowledgement alone does not publish.

The first error/status/stage is sticky. Recoverable issue failure allows one
explicit cleanup drain. Successful cleanup invalidates failed residency and
releases only known-live resources, without replacing the original error.
Failed direct or cleanup Fence is never retried. Unknown Fence, ambiguous
Scale acceptance or failed cleanup Release quarantines context, pools and owner.
Close cannot free or transfer a quarantined session.

Deterministic software fault tests cover all eleven issue positions, first/
middle/final B MVIN, partial Index, partial readback, stale tokens, callback
reentry, externally pending operations and six terminal cases. Terminal cases
run in separate forked processes to avoid pretending that poisoned ownership
can be recovered. They are not included in the parent's assertion total.

The RTL completion harness tests delayed memory readiness/drain, response
backpressure and successful Fence/Release qualification. It does not inject a
physical DMA timeout or prove recovery from a real FPGA bus error.

## 12. RTL Simulation

Every shell used this setup:

```bash
source /home/alveo/miniforge3/etc/profile.d/conda.sh
cd /home/alveo/firesim/target-design/chipyard
source env.sh
source /home/alveo/firesim/env.sh
```

The mandatory Chipyard env emits `EnvironmentLocationNotFound` because
`C/.conda-env` is missing. The subsequent FireSim env supplies the verified
RISCV/tools environment; no setup file was edited.

### Component results

Pinned environment: Scala 2.13.10, Chisel 3.5.6, chisel-iotesters 2.5.1,
sbt 1.8.2, Verilator 5.008.

| Log under LOG | Result |
|---|---|
| `rtl-resident-context.log` | Seven scenarios pass; RS scenario fails under Treadle because its plusarg blackbox returns zero |
| `rtl-rs-verilator.log` | The actual RS scenario passes under Verilator, 528 cycles |
| `rtl-control-final.log` | Completion-qualified Fence/Release scenario passes |
| `rtl-legacy-components.log` | 47 existing scenarios in ten unchanged suites pass |

Thus nine distinct new scenarios and 47 existing scenarios pass. An accidental
duplicate RS invocation also passed and is not counted twice. RTL simulated
cycle counts are test evidence, not FPGA performance.

The full original suite initially failed compilation at two pre-existing
stale `PoTalScaleCommandIngress` constructors:
`PoTalScaleCommandIngressSpec.scala` and `PoTalDrainHandshakeSpec.scala` omit
`indexEntries` and `spRows`. Those tests were not edited green. The same first
attempt exposed a new ScalaTest `run()` name collision, which was corrected;
all new suites subsequently compiled and ran. New test Parameters use full
Rocket DefaultConfig plus a real negotiated metadata TL edge, not guessed
missing keys one at a time.

Reproduction uses S as a real isolated source snapshot and a fresh classpath:

```bash
cd /home/alveo/h3-e4b2/source/target-design/chipyard
sbt 'set Global / concurrentRestrictions := Seq(Tags.limitAll(1))' \
 'set potal / Test / sources := { val d = (potal / Test / scalaSource).value; Seq("PoTalResidentTestSupport.scala","PoTalResidentExecuteSpec.scala","PoTalResidentLoopSpec.scala","PoTalResidentCounterSpec.scala","PoTalResidentConsumerSpec.scala").map(n => d / n) }' \
 'potal/testOnly potal.PoTalResidentExecuteSpec potal.PoTalResidentLoopSpec potal.PoTalResidentCounterSpec potal.PoTalResidentConsumerSpec'
```

The source now selects Verilator for the RS scenario. For the additional control
suite, select `PoTalResidentTestSupport.scala` and
`PoTalResidentControlSpec.scala`, then run
`potal/testOnly potal.PoTalResidentControlSpec`.
For the 47-case legacy gate, explicitly select the ten original suites named in
`rtl-legacy-components.log`, excluding only the two unmodified noncompiling
harnesses. This is disclosed scoped evidence, not a full-suite green claim.

### Integrated generated execution

The ordinary `PoTalSat32GemminiRocketConfig` simulator builds successfully in
S with fresh `.classpath_cache-e4b2`, 12 GiB Java heap and eight build workers.
No previous generated jar is used as B2 proof. The actual processor fixture
issues generated LOOP, Scale/Index and addressed MVOUT, never raw PRELOAD/
COMPUTE and never a CPU-result fallback.

```bash
cd /home/alveo/h3-e4b2/source/target-design/chipyard/sims/verilator
make -j8 CONFIG=PoTalSat32GemminiRocketConfig \
 CLASSPATH_CACHE=/home/alveo/h3-e4b2/source/target-design/chipyard/.classpath_cache-e4b2 \
 JAVA_HEAP_SIZE=12G JAVA_TOOL_OPTIONS='-Xmx12G -Xss8M -XX:ActiveProcessorCount=8'
make run-binary-fast CONFIG=PoTalSat32GemminiRocketConfig \
 CLASSPATH_CACHE=/home/alveo/h3-e4b2/source/target-design/chipyard/.classpath_cache-e4b2 \
 JAVA_HEAP_SIZE=12G JAVA_TOOL_OPTIONS='-Xmx12G -Xss8M -XX:ActiveProcessorCount=8' \
 BINARY=/tmp/h3-e4b2-bin/potal-e4b2-rocc-baremetal \
 LOADMEM=1 timeout_cycles=100000000
```

Bare-metal linking uses the existing benchmark common runtime:

```bash
cd /home/alveo/firesim/target-design/chipyard/generators/potal/software/workspace/llama.cpp-potal
COMMON=/home/alveo/firesim/target-design/chipyard/generators/potal/software/gemmini-rocc-tests/riscv-tests/benchmarks/common
riscv64-unknown-elf-gcc -mcmodel=medany -std=gnu99 -O2 -fno-common \
 -fno-builtin-printf -fno-tree-loop-distribute-patterns -march=rv64gc -mabi=lp64d \
 -DPREALLOCATE=1 -DMULTITHREAD=1 -DBAREMETAL=1 -DID_STRING='"h3-e4b2"' \
 -I"$COMMON/../.." -I"$COMMON/../../env" -I"$COMMON" \
 -nostdlib -nostartfiles -static -T "$COMMON/test.ld" \
 ggml/src/ggml-potal/tests/potal-e4b2-rocc.c "$COMMON"/*.c "$COMMON"/*.S \
 -lgcc -lm -o /tmp/h3-e4b2-bin/potal-e4b2-rocc-baremetal
```

`rocc-baremetal-final-build.log` records successful linking and the pre-existing
linker script's RWX-segment warning. The separate saturation build uses this
same command plus `-DE4B2_RTL_SATURATION`, output
`/tmp/h3-e4b2-bin/potal-e4b2-saturation-baremetal`, and a distinct run log.
Before the Linux DMA preparation correction, relinking the full source changed
a nonallocated temporary-object name
in the ELF symbol strings. Disassembly and loadable-section comparison, rather
than full-ELF hash equality, verify that the default 42-case executable is
unchanged by adding the saturation-only compile selection. The recorded
bare-metal ELFs predate the Linux preparation correction in section 15.
Their original-tensor data, oracle and command-packing semantics are unchanged;
they are not claimed byte-identical to the corrected Linux executable.

`LOADMEM=1` uses the existing direct ELF loader; it changes loading cost only,
not the computation or oracle. An earlier serial-load run exceeded its one-hour
monitor and was stopped, not counted as a full pass. The optional full 42-case
direct-load run was also stopped after required RTL and physical hardware
acceptance completed. Prefix results include actual capability, source poisoning,
view refusal, first/interior K3/J5 row 4672, last, K112 odd phase and last J on
pair 0. A prefix is not a complete 42-case pass.

A read-only runtime snapshot on October 27 at 03:36:37 +09:00 observes
test-harness trace_count 38258335, Rocket PC 0x8000268c and allDrained=1.
The exact running ELF maps that PC to the independent oracle's multiply loop
inside `execute`, after compute Fence and before addressed result comparison.
At that snapshot the processor is not waiting in RoCC/Fence. This is a bounded
simulator-state observation, not FPGA performance or a full-case numeric pass.
The debugger detached without changing simulator data; evidence is
`LOG/rtl-runtime-state.log` and `LOG/rtl-runtime-journal.md`.

The separate saturation-only build completed the same real generated execution
path without the long address sweep. `classic-saturation-numeric.log` ends:

```text
E4B2_ROCC_SATURATION PASS ordered=-1 compact=0 dense=-1 carrier24=16777215
E4B2_ROCC_PASS runs=6 checks=3072 B_only_MVIN=756 view_configurations=6
scope=generated_LOOP_numeric actual_source=ACC_MVOUT
```

The monitor reports exit 0. The 756 B-only MVINs are three distinct 252-tile
resident loads with different witness weights, not micro-tile B reloads.
Both physical pairs and indexed ACC1 participate in these witnesses.

## 13. Legacy Regression

Eighteen host targets were rebuilt against the changed production API and ran
successfully. Individual logs are under `LOG/regressions/`:

```text
potal-api-test                 potal-attention-api-test
potal-trace-test               potal-residual-unit-test
potal-paired-plan-test         potal-paired-schedule-test
potal-paired-unit-test         potal-paired-auto-test
potal-paired-execution-test    potal-stripe-pipeline-test
potal-attention-test           potal-resident-test
potal-resident-model-test      potal-e3-prototype-test
potal-e3-context-test          potal-e4b1-test
potal-stripe-api-test          potal-e4b2-test
```

The new suite's latest output is `native-binding-guard.log`; the separate earlier
regression invocation is not substituted for that final result.

Software reproduction, after the environment setup in section 12:

```bash
cd /home/alveo/firesim/target-design/chipyard/generators/potal/software/workspace/llama.cpp-potal
PARAMS=/home/alveo/firesim/target-design/chipyard/generators/potal/software/gemmini-rocc-tests/include/gemmini_potal_params.h
cmake -S . -B build-potal-e4b2-host \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_C_COMPILER=/home/alveo/firesim/.conda-env/bin/cc \
 -DCMAKE_CXX_COMPILER=/home/alveo/firesim/.conda-env/bin/c++ \
 -DGGML_POTAL=ON -DGGML_GEMMINI=OFF -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF \
 -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
 -DGGML_GEMMINI_ACTIVATION_BITS=8 -DGGML_GEMMINI_WEIGHT_BITS=8 \
 -DGGML_GEMMINI_ACTIVATION_QUANT=EXSIA -DGGML_GEMMINI_BLOCK_SIZE=32 \
 -DGGML_GEMMINI_DIM=16 -DGGML_GEMMINI_ENABLE_RMD=ON \
 -DGGML_GEMMINI_ENABLE_STRIPE_MATMUL=ON -DGGML_GEMMINI_ENABLE_STRIPE_PIPELINE=ON \
 -DGGML_GEMMINI_DEFAULT_MATMUL_MODE=STRIPE_PIPELINE \
 -DGGML_GEMMINI_DEFAULT_RMD_BACKEND=WS \
 -DGGML_GEMMINI_EXSIA_DEFAULT_MODE=SEQUENTIAL \
 -DGGML_GEMMINI_EXSIA_PROFILE_SCOPE=OFF -DGGML_GEMMINI_ENABLE_OPENMP=OFF \
 -DPOTAL_API_ONLY=OFF -DPOTAL_ENABLE_OPENMP=OFF -DPOTAL_PLANNING_DIAGNOSTICS=OFF \
 -DPOTAL_HW_PARAMS="$PARAMS" -DPOTAL_EXPECTED_DIM=16 \
 -DPOTAL_E3_TESTS=ON -DPOTAL_E4A_TESTS=ON \
 -DPOTAL_E4B1_TESTS=ON -DPOTAL_E4B2_TESTS=ON
TARGETS='potal-api-test potal-attention-api-test potal-trace-test potal-residual-unit-test potal-paired-plan-test potal-paired-schedule-test potal-paired-unit-test potal-paired-auto-test potal-paired-execution-test potal-stripe-pipeline-test potal-attention-test potal-resident-test potal-resident-model-test potal-e3-prototype-test potal-e3-context-test potal-e4b1-test potal-stripe-api-test potal-e4b2-test'
cmake --build build-potal-e4b2-host -j8 --target $TARGETS
for t in $TARGETS; do build-potal-e4b2-host/bin/"$t" || exit "$?"; done
cmake -S . -B build-potal-e4b2-d16 -DCMAKE_TOOLCHAIN_FILE=cmake/riscv.cmake \
 -DCMAKE_BUILD_TYPE=Release -DGGML_POTAL=ON -DGGML_GEMMINI=OFF \
 -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
 -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_GEMMINI_ACTIVATION_BITS=8 \
 -DGGML_GEMMINI_WEIGHT_BITS=8 -DGGML_GEMMINI_ACTIVATION_QUANT=EXSIA \
 -DGGML_GEMMINI_BLOCK_SIZE=32 -DGGML_GEMMINI_DIM=16 \
 -DGGML_GEMMINI_ENABLE_RMD=ON -DGGML_GEMMINI_ENABLE_STRIPE_MATMUL=ON \
 -DGGML_GEMMINI_ENABLE_STRIPE_PIPELINE=ON \
 -DGGML_GEMMINI_DEFAULT_MATMUL_MODE=STRIPE_PIPELINE \
 -DGGML_GEMMINI_DEFAULT_RMD_BACKEND=WS \
 -DGGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE=ON \
 -DGGML_GEMMINI_ENABLE_OPENMP=OFF -DGGML_GEMMINI_EXSIA_DEFAULT_MODE=SEQUENTIAL \
 -DGGML_GEMMINI_EXSIA_PROFILE_SCOPE=OFF -DGGML_POTAL_PAIRED_AUTO=ON \
 -DPOTAL_PLANNING_DIAGNOSTICS=OFF -DPOTAL_E3_TESTS=OFF \
 -DPOTAL_HW_PARAMS="$PARAMS" -DPOTAL_EXPECTED_DIM=16 \
 -DPOTAL_E4A_TESTS=ON -DPOTAL_E4B1_TESTS=ON -DPOTAL_E4B2_TESTS=ON
cmake --build build-potal-e4b2-d16 -j8 --target \
 potal-e4b2-test potal-e4b2-rocc potal-e4b2-hardware potal-resident-test potal-api-smoke
```

These commands expose the acceptance-relevant cache settings; the complete
observed caches remain in the two build directories. Configure and build logs
are `host-configure.log`, `host-regression-build.log`,
`cross-target-configure.log` and `cross-binding-guard-build.log`.
All listed final builds exit 0. An earlier cross invocation before regenerating
CMake reported an unknown new target; regenerating the unchanged graph fixed
that configuration-state error.

Forty-seven protected source/artifact fingerprints match their original values
in `LOG/final-protected-source-verification.log`. The nine separately hashed
image/model/header/rootfs artifacts and all current code/ELF identities pass
`LOG/final-boundary-verification.log`.
They include the H3-6B2 trace path, H3-6C public lowering/decode reuse, production
planner/scheduler, B5 attention/emulator, PV correction and Linear paths.
`POTAL_RAW_TRACE` remains cached at successful initialization; it is not looked
up in each command or Scale hot path. D1 diagnostics stay default OFF.
The existing attention tests retain independent limb/cancellation/saturation
goldens. No obsolete test assertion was removed or weakened.

On the existing physical FPGA, raw and checked probe-only diagnostics pass their
unsupported-image checks. The existing API smoke fails:

```text
FAIL A hw invalid column: got=6(HW_INVALID_ENCODING) expected=1(HW_INVALID_COLUMN)
```

This is the unchanged smoke's raw column-16 key on DIM16. The original ingress
uses a four-bit column and key width 20; that key sets reserved bit 20, so encoding
priority returns 6. This behavior exists in the pre-B2 ingress. It is not a new
extension instruction or a new header mismatch. The test remains failing and
unmodified; a full legacy FPGA smoke pass is not claimed.

An initial unpinned smoke ran on Linux CPU3 and trapped at `potal_tlb_flush`.
The attached PoTal hart0 is Linux CPU1. Later runs pin the smoke and raw probe to
CPU1; checked diagnostics use `issuer_owner`.
Running E4A after the intentionally failing smoke left its slot 3 allocated,
causing `HW_SCALE_CONFLICT`. That contaminated attempt is not E4A evidence;
the clean E4A-first regression passes on CPU1/hart0 with `repeats=1`, exact
independent numerics, retirement and preflight refusal checks. Its trace and
`POTAL_RESIDENT actual RoCC PASS` are saved under
`/tmp/h3-e4b2-old-attempt6-results/legacy-e4a-hardware.log`.
This is the original K28/J2, native Main K2/P2 and indexed Residual K1/P1 subset
on both pairs, not new general K28/J9 hardware evidence.
The observed trace has 2074 issued commands, 16 legacy LOOPs with `xd=0`,
zero Resident Views and zero extended LOOPs.
The following unchanged smoke still fails at the raw reserved-key expectation,
so the aggregate old-image guest suite deliberately exits 1. Generic FireSim driver
`*** PASSED ***` after guest shutdown is never substituted for a guest test pass.

The new image reproduces that same smoke status mismatch. In its separate
clean numerical boot, E4A passes both before and after extended execution;
each trace again has 2074 commands, 16 legacy xd=0 LOOPs, zero funct-28
instructions and zero extended LOOPs. This demonstrates legacy/extended
switching without promoting the malformed-key smoke to PASS.

## 14. FireSim Build and Image Identity

The build uses a separate real source snapshot S, private recipes and a distinct
alias:

```text
/home/alveo/h3-e4b2/config_build.yaml
/home/alveo/h3-e4b2/config_build_recipes.yaml
alveo_u280_firesim_potal_quadcore_d16_h3_e4b2_no_nic
default_build_dir=/home/alveo/h3-e4b2/build_firesim
TARGET_CONFIG=FireSimQuadRocketSinglePoTalD16Config
PLATFORM_CONFIG=BaseXilinxAlveoConfig
PLATFORM=xilinx_alveo_u280
fpga_frequency=30 MHz; build_strategy=TIMING
```

FireSim Scala/Chisel target generation and matching driver build pass:
`firesim-target-restored.log`, exit 0. New driver:

```text
S/sim/output/xilinx_alveo_u280/Q/FireSim-xilinx_alveo_u280
sha256=2122ed72e53859aa45f7f86db20c91b7712d6be2d5321d96aef49d45f05bf666
```

DIM16 generated header matches the software's original geometry header:
`b4cd8b9efef9f3cd9b598ae3cc84af6478f83e628ed73d94a77fb1c452df1fc7`.
Geometry equality is ABI evidence, not extension image identity.
The six implemented Scala source files are byte-identical between C and S.
`LOG/b2-source-and-executable.sha256` records all 22 B2 code paths and the
diagnostic ELFs. The full bare-metal fixture hash is
`5d787d4f4fb62264cd61466f67da232d65c1b058ec7d0fece8ffb6bd9008d602`;
the saturation-only fixture is
`37366726ba9a9cc1f7ebd0d5f0cea4939209a73856c44960231c184ac2d545bd`.

Two snapshot-filter defects were corrected by copying original infrastructure
bytes, not by editing generated RTL. The broad `models` exclusion had omitted
MIDAS Scala/resource models. The broad `build-*` exclusion had omitted
`deploy/build-farm-recipes` and platform build scripts. The corrected target
build passes and the corrected FPGA manager invocation reaches Vivado 2021.1:

```bash
cd /home/alveo/h3-e4b2/source/deploy
./firesim -b /home/alveo/h3-e4b2/config_build.yaml \
 -r /home/alveo/h3-e4b2/config_build_recipes.yaml buildbitstream
```

`fpga-build.log` records the initial missing-recipe failure;
`fpga-build-restored.log` records the retry. The restored infrastructure is
byte-for-byte original source. Vivado route, bitgen and write_cfgmem complete
successfully. The manager subsequently exits 1 because its rsync receiver
cannot create the timestamped directory under the omitted private
`source/deploy/results-build` parent. This is a postbuild copy failure, not a
Vivado or timing failure. Creating the missing parent and replaying that exact
rsync step succeeds without another RTL or FPGA build.
Synthesis, placement, physical optimization and routing complete.
`Router Completed Successfully` and the routed checkpoint are observed with
zero errors; routed DRC and methodology reports are generated.
The subsequent bitstream and cfgmem steps also complete with exit 0.
The BD 41-1356 address-assignment and Vivado 12-4739 generated-clock/multicycle
critical warnings also occur verbatim in the successful October 24 archived
`vivado.log` and `impl_1/runme.log`. They are reported, not suppressed or
manually patched. The completed `design_1_wrapper_route_status.rpt` reports
636893/636893 routable nets fully routed and zero routing errors.
`design_1_wrapper_timing_summary_routed.rpt` reports WNS 0.070 ns,
WHS 0.010 ns, TNS/THS zero and all user specified timing constraints met.
The stock `implementation_2021.1.tcl` checks WNS/WHS before its next-step
bitstream run.
The report still lists TIMING-4, TIMING-27 and TIMING-54 methodology critical
warnings; no constraint or warning policy was changed.

The remaining SDK post-copy package layout and HWDB entry are recovered
separately, with `postbuild-package-recovery.log` exiting 0. The original
manager's exit 1 is retained, not relabeled. The isolated snapshot contains no
Git metadata; its SDK-format metadata records the verified original FireSim
base `5c3458a3f54917ab9502bda4a8b3cc3dd513304d-dirty` as source provenance,
not a claimed snapshot checkout HEAD. Exact B2 code remains SHA256-bound in
the separate source/ELF manifest. No source, timing constraint, global HWDB or
old package is edited by recovery.

The new result root is:

```text
S/deploy/results-build/2026-10-26--17-24-35-alveo_u280_firesim_potal_quadcore_d16_h3_e4b2_no_nic/cl_Q
```

| New artifact | SHA256 |
|---|---|
| `firesim.tar.gz` | `5aa1794e9924bf0b6ff987e38eaf845023ce3f0a495f503a1e95584a9f891609` |
| `xilinx_alveo_u280/firesim.bit` | `c4c71ef648f5c5a73a40f810fed98fdfb3862f1433b00592720f49631972beae` |
| `xilinx_alveo_u280/firesim.mcs` | `ee0ee3cf66c3efba8467255405bea9ae32272ed227810ffd50de69811f6e3d2a` |
| `xilinx_alveo_u280/metadata` | `b15281c4688e4535e0194278a50a51b93b0bec07f665b0045b8cc6023d8b96c1` |

Source, recovered and packaged bit/mcs hashes match. After a fresh no-open-user
check, the existing SDK's single-BDF volatile programming path programs only
the new `.bit` to 01:00.0, UID `Xilinx/217602178016A`. XDMA is reloaded using
the stock module and poll_mode=1. The actual log records successful programming,
`__E4B2_NEW_IMAGE_PROGRAM_PASS__` and matching-driver simulation start.
This is deployment evidence, not numerical Hardware PASS.

The preserved October 24 build is:

```text
F/deploy/results-build/2026-10-24--02-24-42-alveo_u280_firesim_potal_quadcore_d16_no_nic/cl_Q
```

| Preserved artifact | SHA256 |
|---|---|
| `firesim.tar.gz` | `07fcd2cf6fc4c520229ab2e96a8043ba6e1070d362d16c2690ed36b81a4fbd46` |
| `xilinx_alveo_u280/firesim.bit` | `f4c465a451385cbda44c254fb038fc2ede3a25ea12d5e829c2a501b7d495b3de` |
| `xilinx_alveo_u280/firesim.mcs` | `b6c622f76319f73cced7d367bc062812c5235ac73dc90d13a9a233aae94e9187` |
| Original and deployed old driver | `ebbc201ee188d1553912aafb3835b5637f247eed60a696499dcf74a27b110a0d` |
| Original B1 rootfs | `5962811af071fd50337abcd2128daa4129e98a2e66ea3d5d5a350234092bf055` |
| GPT2 Q8_HP1 | `cf23a45e4db23736d7d1f67ab42018803a198cfcb01db36daca69ff56bdbc7d4` |
| GPT2 FP16 | `5bd663a1e1d303f9404dd2edd92e4e3ca3e24bcf80d19c655880f0e455c8f4df` |

Final checks are recorded in `protected-artifacts-final.sha256`.
The original 24 GiB rootfs inode 73405652 was not mounted for writes; guest
tests use distinct copied inode 73405612 at `/tmp/h3-e4b2-old-run/rootfs.img`.
Only that copy required offline fsck after a timed-out boot; fsck returned 1
(corrected), with the wrapper explicitly accepting that documented status.
No original image, model, host initialization or Gemmini working change was
replaced. Protected PoTal `software/host-init.sh` remains
`e96d846238946925fd4a1c52f9b5e24ea144e15b3d748d15f8420e1c71c70834`;
Gemmini's dirty binary diff remains
`407b6271652f19485acaa919b81e5fb697083ac73a894d7b217de883c9e71b84`.

## 15. Actual Hardware Verification

No extension is issued on the existing image. Raw `--probe-only` and checked
`--probe-only` both observe zero, return unsupported, and prove their respective
`no_extended_command_issued=1` / `extension_issued=0` markers.
The checked raw trace contains exactly two funct-126 instructions.

The cross-built raw diagnostic uses actual custom3 instructions and ACC MVOUT.
The checked diagnostic uses the production resident C11 API under the existing
issuer owner with no sink or emulator. Its default run checks both pairs,
native/interior/indexed Main/indexed Residual, nonzero offsets, ordered epochs,
tail shapes, source poisoning, safe retirement and results. It has 16 numerical
cases, including a three-invocation first/continue/last Main epoch on each pair.
Those invocations preserve one declared K[0,48) output, use sequences 0/1/2,
read back after each contribution, and compare independently accumulated
INT32 expectations. Raw full execution has 42 cases including saturation witnesses.
Both have now run on the matching new image, with results below.
The updated checked diagnostic compiles and links with exit 0 in
`cross-checked-continuation-build.log`; its ELF SHA256 is
`55f66631665f105d8409758c4052cc4c45fbac64839abf4a832d25f1d7cd6ba7`.
The earlier old-image probe used the previous diagnostic ELF; the gate code
and probe-only instruction sequence were not changed by this positive-path test.
Separate boot scripts `/tmp/h3-e4b2-new-legacy-guest.sh` and
`/tmp/h3-e4b2-new-main-guest.sh` stage fresh binaries/libraries under
`/root/h3-e4b2-new` in the private copied rootfs. Both require the actual raw
capability first. The first boot records the unchanged smoke result; the second
starts with clean target state, runs E4A, raw42, checked n=7 and E4A again after
extended execution. This prevents the aborting malformed-key smoke from
contaminating the numerical acceptance.
Distinct host UART inputs are prepared under
`/tmp/h3-e4b2-new-legacy-run` and `/tmp/h3-e4b2-new-main-run`.
The prepared launch preserves all original boot arguments, replacing only
the executable with the matching snapshot driver, the block image with the
private copied rootfs, and the two UART paths. The driver SHA above is
reverified. Successful final numerical evidence is under
`/tmp/h3-e4b2-new-main-attempt2-results/`; its host UART/driver logs are
`/tmp/h3-e4b2-new-main-run2/{uartlog,driver.log}`.

### Actual gate and failure recovery

Raw and production-C11 probe-only executions both observe the actual response
`0x504f544102100007`. The checked trace has exactly the two existing funct-126
instructions and no extension during probing. Logs are under
`/tmp/h3-e4b2-new-legacy-results/`.
The unchanged smoke reproduces the old column-16 mismatch: status 6 versus
expected 1. That isolated guest exits 1, while its host driver exits 0.

The first clean numerical attempt passes E4A, then the raw fixture fails at
`bank=0 row=16 col=1 expected=-160 actual=0`. Results are retained under
`/tmp/h3-e4b2-new-main-attempt1-results/`.
Its static Linux output BSS had not been CPU-written or pinned before MVOUT.
Production C11 already initializes and mlocks its owned DMA pools.
The test-only correction prepares writable output backing, pins all three
Linux DMA buffers, and requires every bank readback to replace CPU poison.
The oracle, expected integers, view/run encoding and RTL are unchanged.
Page tables were not directly captured; the decisive regression proof is the
complete successful rerun on the same bitstream and driver.

Cross build (`LOG/cross-raw-prefault-build.log`) and private-rootfs staging
both exit 0. Corrected raw Linux ELF:

```text
aa5c15f779765c5d06e5019b9414b4394847b752edf92335d7d6b4fcb9289c53
```

The failed ELF `63e644...` is preserved at
`LOG/potal-e4b2-rocc-before-prefault`.
The production-C11 diagnostic remains `55f666...`; no production code or
FPGA image was changed to resolve the raw Linux memory precondition.

### Actual numerical acceptance

The final guest records all of these markers, then exits 0:

```text
__E4B2_NEW_E4A_BEFORE_PASS__
__E4B2_NEW_RAW_42_PASS__
__E4B2_NEW_CHECKED_N7_PASS__
__E4B2_NEW_E4A_AFTER_PASS__
__E4B2_NEW_MAIN_PASS__
__E4B2_NEW_MAIN_EXIT__=0
```

| Actual run | Result |
|---|---|
| Raw generated LOOP | 42 successful invocations, 193024 INT32 comparisons |
| Raw saturation witnesses | ordered=-1, compact=0, dense=-1, carrier24=16777215 |
| Production checked C11 | 16 invocations and 4404 cells per suite; one untimed warmup plus seven measured suites |
| All checked executions | 128 invocations and 35232 checked cells, including warmup |
| Legacy E4A before/after | Both PASS; 2074 commands and 16 legacy xd=0 LOOPs each, no extension |
| Final checked ownership | No pending/control operation, live Scale slots, readable residency or sticky failure; retirement and close succeed |

Both pairs pass full K28/J9, first/interior/last, K112 odd phase, native and
indexed K443/J139 tails, indexed Main/ACC0, sparse Residual/ACC1, nonzero ACC
offsets and ordered contributions. Tail destinations are seeded so untouched
rows/columns are also checked. Checked epochs execute first/continue/last
sequences 0/1/2 and read back after each completed contribution.
Raw hardware also refuses missing, reserved and disabled view configurations.
Physical DMA error injection is not performed.

### B reuse and physical addressing proof

Every raw compute case records `B_dram=0`. The seven explicit resident loads
issue 1764 B-only MVINs: 252 tiles per resident generation, not a B reload for
each micro-tile. Each load completes before its CPU weight source is poisoned.
Actual generated-LOOP results remain correct after poisoning.

The physical address mapping is checked at the generated RTL PRELOAD consumer,
including row 4672/12864 for K3/J5 on pairs 0/1. The built image is bound to
those exact Scala sources. The actual K3/J5 hardware case uses that view and
matches its independent INT32 oracle. Its tensor markers are not a repeated
constant panel: all 504 pair/K/J tile payloads are distinct. An independent
finite projection through the K48/J80 DIM16 activation/carrier fixture also
produces 504 distinct output signatures, so selecting a different aligned
resident tile cannot pass that fixture.
This combines explicit physical RTL addressing with discriminating actual
hardware results; it is not a claim based only on B DRAM address zero.
An internal FPGA PRELOAD bus trace is not captured, and physical bus bytes/time
are not inferred from these command-declared addresses or extents.

All host drivers and UART writers are stopped. The private rootfs is unmounted;
the board remains programmed with the new bitstream. No old image or original
rootfs was overwritten.
No physical FPGA A is inferred from software on the target CPU or from ordinary
Verilator simulation.

### Recorded hardware reproduction commands

Use the environment from section 12. These are the recorded image/driver
paths; a rerun uses a fresh RUN directory and regular UART input stream.
UART bytes must append to the same open inode, not replace that file.
The successful programming sequence used the no-open-user check before removal:

```bash
F=/home/alveo/firesim
S=/home/alveo/h3-e4b2/source
Q=xilinx_alveo_u280-firesim-FireSim-FireSimQuadRocketSinglePoTalD16Config-BaseXilinxAlveoConfig
IMAGE="$S/deploy/results-build/2026-10-26--17-24-35-alveo_u280_firesim_potal_quadcore_d16_h3_e4b2_no_nic/cl_$Q/xilinx_alveo_u280/firesim.bit"
if sudo -n fuser -v /dev/xdma* /sys/bus/pci/devices/0000:01:00.0/resource0; then
    exit 17
else
    test "$?" -eq 1
fi
sudo -n rmmod xdma
sudo -n /usr/bin/python3 "$F/platforms/xilinx_alveo_u280/scripts/fpga-util.py" \
 --bdf 01:00.0 --bitstream "$IMAGE" \
 --vivado-bin /tools/Xilinx/Vivado/2021.1/bin/vivado \
 --hw-server-bin /tools/Xilinx/Vivado/2021.1/bin/hw_server
sudo -n insmod /lib/modules/$(uname -r)/extra/xdma.ko poll_mode=1
sudo -n udevadm settle --timeout=60
```

The following records the final successful driver launch. The original boot
binary/DTB/ROM are read-only inputs; the old driver and original rootfs are not
used for this launch:

```bash
BOOT=/home/alveo/FIRESIM_RUNS_DIR/sim_slot_0/gemmini-tests-interactive0-gemmini-tests-interactive-bin
RUN=/tmp/h3-e4b2-new-main-run2
DRIVER="$S/sim/output/xilinx_alveo_u280/$Q/FireSim-xilinx_alveo_u280"
cd "$RUN"
sudo -n "$DRIVER" +permissive +macaddr0=00:12:6D:00:00:02 \
 +blkdev0=/tmp/h3-e4b2-old-run/rootfs.img +niclog0=niclog0 +blkdev-log0=blkdev-log0 \
 +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 \
 +dwarf-file-name="$BOOT-dwarf" +autocounter-readrate=0 \
 +autocounter-filename-base=AUTOCOUNTERFILE \
 +drj_dtb="$BOOT.dtb" +drj_bin="$BOOT" +drj_rom="$BOOT.rom" \
 +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 \
 +shmemportname0=default +domain=0x0000 +bus=0x01 +device=0x00 \
 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f \
 +uart-in0="$RUN/uart-in" +uart-out0="$RUN/uartlog" \
 +permissive-off "$BOOT"
```

Before launch, login, authenticated-prompt and guest-result subscriptions were
armed. `/tmp/h3-e4b2-uart-writer.py "$RUN/uart-in"` supplies root login and the
guest command only after the corresponding prompt. The guest commands are:

```sh
# Separate first boot; this records the known smoke failure.
sh /root/h3-e4b2-new/legacy-guest.sh
# Separate clean reset boot, used for numerical acceptance.
sh /root/h3-e4b2-new/main-guest.sh
```

The latter runs unchanged E4A, `taskset -c 1 bin/potal-e4b2-rocc`,
`bin/potal-e4b2-hardware --repeat 7`, and unchanged E4A again. It requires
each actual PASS marker and exactly seven checked samples before exit 0.
After shutdown the private image is mounted read-only to copy results, then
unmounted. The old failed and final successful logs remain separate.
Host lifecycle cycle totals include boot and interactive idle time; they are
not accelerator performance measurements.

## 16. Integer Golden Results

Native final result:

```text
E4B2_SOFTWARE_PASS assertions=544655 runs=32 cells=18740 destroyed=34
scope=production_C11_and_independent_command_model hardware=0
```

RISC-V guest software result is the same source-level conformance class, not
accelerator execution. It reports 544654 assertions because the platform-specific
host-only branch differs, with the same 32 runs, 18740 cells and 34 destroyed
contexts. Its observed log is under `/tmp/h3-e4b2-old-attempt4-results/`.

Fixtures use original-tensor int64 math, signed int8 operands, original K/J and
pair-distinguishing weights, per-DIM16 SCU clamping and ordered ACC clamping.
They cover full K28/J9, interior K3/J5, odd K112, K443/J139, both pairs, sparse
compact Residual, zero/shift-zero, negative dots, positive/negative saturation
and cancellation. All physical ACC cells are also checked against untouched
sentinels in the software interpreter.

The integrated raw fixtures additionally seed destination padding before tail
writes. The stopped optional RTL prefix is in `classic-fast-numeric.log`.
The separate completed six-witness RTL result checks 3072 INT32 values and the
four adversarial constants in `classic-saturation-numeric.log`, exit 0.
The full physical FPGA run passes 193024 comparisons, with both pairs and the
same four adversarial constants. No incomplete RTL log is reported as a full
numerical pass, and no software-only result is relabeled as hardware.

## 17. Command/Performance Comparison

### Command accounting

These are exact encoder/source counts, asserted by the C11 software test, not
measurements of physical bus traffic:

| Operation | Issued commands |
|---|---|
| Capability gate | 2 existing COUNTER |
| Prepare | FLUSH + FENCE |
| Full resident B load | 1 CONFIG_LD + 252 B MVIN; explicit completion FENCE |
| Legacy generated invocation | 4 CONFIG + 5 LOOP_CONFIG + 1 LOOP_WS = 10 |
| Extended generated invocation | Same + 1 Resident View = 11 |
| Extended B reload | 0 B MVIN; LOOP B DRAM remains zero |
| Scale window with P partials and N valid output columns | P*N SCALE_WRITE and matching eventual P*N SCALE_RELEASE |
| Index window with X compact entries | 1 INDEX_CONFIG + X INDEX_WRITE, X<=32 |
| Addressed readback with T J tiles | 1 CONFIG_ST + T MVOUT; completion FENCE or qualifying Release |
| CPU output epoch/reference validation | 0 accelerator commands |
| ACC save/restore for normal continuation | 0; completed INT32 values stay in place |

Release is itself completion qualified. It is not counted as both a new
independent wait interval and overlapping Fence wall time. Each standalone
conformance invocation normally uses an explicit compute completion and
readback completion; software tests also check Release-qualified readback.
The raw fixture's final funct counters include its extra refusal tests and
repeated sentinel readbacks, so they must not be substituted for minimal C11
invocation counts.

The completed six-witness integrated RTL run reports CONFIG 39, B MVIN 756,
MVOUT 12, FLUSH 1, LOOP_WS 6, each LOOP_CONFIG funct 9..13 six times,
SCALE_WRITE 128, SCALE_RELEASE 128, FENCE 23, INDEX_WRITE 19, INDEX_CONFIG 2,
Resident View 6 and COUNTER 2. These are observed issued instructions, not DMA
bus bytes or a minimal performance schedule.

The complete physical FPGA raw run reports these issued RoCC counters:

```text
CONFIG(0)=275        B_MVIN(2)=1764       MVOUT(3)=754
FLUSH(7)=1           LOOP_WS(8)=46       LOOP_CONFIG(9..13)=46 each
SCALE_WRITE(23)=9272 SCALE_RELEASE(24)=9272
FENCE(25)=137        INDEX_WRITE(26)=175 INDEX_CONFIG(27)=10
RESIDENT_VIEW(28)=46 COUNTER(126)=2
```

There are 42 successful invocations and four refused run attempts; the view
counter also includes refusal/disable checks. These counters do not count
accelerator-internal generated PRELOAD/COMPUTE operations. No raw PRELOAD or
COMPUTE instruction is issued by the fixture.
The checked suite's final pre-close snapshot reports core.commands=2090,
core.drains=401, B_MVIN=1008, runs=16 and cells=4404. Commands already include
completion instructions; drains count completed allDrained qualifications,
including Release. They must not be added as another 401 commands or Fences.

Full Main at P2 uses 126 slots and 2016 per-column writes/releases. P1 uses
252 slots and 4032 writes/releases: +126 slots, +2016 writes and +2016 releases
when the whole live Main window is programmed once. Including the specified
18-slot Residual window would use 144 slots at P2 versus 270 at P1; the latter
is rejected. Splitting live windows fixes admission, not the extra P1 work.
No Scale reuse/P2 optimization was added.

### Timing

The separate software-only check/encode measurements use n=7. Each
sample is one 512-call batch. Timings are CPU-observed elapsed API intervals,
including validation/output or status/checksum work. They contain no accelerator
command issue, DMA, Execute, or readback. `hardware=0` is explicit.

| Platform / primitive | Median ms per 512 calls | MAD ms |
|---|---:|---:|
| x86 physical check/encode/output/checksum | 0.020186 | 0.000018 |
| x86 live-session binding check/status/checksum | 0.075293 | 0.000017 |
| RISC-V guest physical check/encode/output/checksum | 0.294000 | 0.001000 |
| RISC-V guest binding check/status/checksum | 1.075000 | 0.002000 |

Logs: `native-binding-guard.log` and
`/tmp/h3-e4b2-old-attempt4-results/software-only.log`.
Checksum 19124224 prevents dead-code removal. Diagnostics are off in these
measurements. Batch medians are not per-call percentiles. Platforms are not a
paired speedup comparison, and these absolute results are not compared with
the different B1 semantic checker or with FPGA execution.

The actual new-image `--repeat 7` diagnostic completes one untimed warmup and
seven validated suites with diagnostics/raw trace off. Each suite has the same
16 cases and 4404 result cells on the same image. Target guest CPU-observed
suite elapsed times are 14.007, 13.928, 13.858, 13.860, 13.845, 13.935 and
13.957 ms: median **13.928 ms**, MAD **0.068 ms**.

The following values are aggregate CPU-observed API intervals per suite,
not per-call percentiles and not accelerator execution or physical FPGA wall time:

| Scope | Calls per suite | Median ms | MAD ms |
|---|---:|---:|---:|
| prepare | 1 | 0.002 | 0.000 |
| B load issue | 4 | 0.056 | 0.000 |
| B completion | 4 | 0.004 | 0.000 |
| resident handoff | 4 | 0.003 | 0.000 |
| Index setup | 6 | 0.021 | 0.000 |
| epoch setup | 16 | 0.028 | 0.000 |
| Scale setup | 16 | 0.068 | 0.000 |
| binding validation | 16 | 0.051 | 0.002 |
| physical encoding | 16 | 0.026 | 0.000 |
| generated LOOP issue | 16 | 0.075 | 0.002 |
| ACC completion | 16 | 0.018 | 0.001 |
| readback issue | 16 | 0.032 | 0.001 |
| readback completion | 16 | 0.016 | 0.000 |
| result receipt | 16 | 0.017 | 0.001 |
| Scale Release | 364 | 0.276 | 0.008 |
| residency retirement | 4 | 0.015 | 0.000 |
| close | 1 | 0.422 | 0.002 |

Source is `new-checked-profile.log` in the successful result directory.
Zero MAD is at the logged timer resolution, not proof of zero hardware variance.
Generated issue includes all eleven commands, including Resident View;
standalone funct-28 latency is not separately isolated from that atomic API.
Binding validation and physical encoding are extra pure diagnostic calls and
emit no commands. Suite time includes staging, validation, independent comparison
and case reporting; interval output is printed after its timer stops.
Default/probe/warmup paths do not read the profiling clock.
The intervals are not independent graph-wall-time contributions: do not sum
them or subtract them to infer hardware waiting.
DMA bus bytes/time, PE utilization, internal PRELOAD bus traces and pipeline
overlap are **Not measured**. No matched old/new performance speedup is claimed.

Historical K28 context only:

| Case | Serial | Paired |
|---|---:|---:|
| Decode | 8.810 ms | 14.291 ms |
| FFN | 52.684 ms | 78.363 ms |

These are earlier whole-graph CPU-observed results, not matched B2 primitives.
The production paired scheduler is unchanged. There is no B2 full-MatMul
speedup or pipeline-completion claim.

## 18. Remaining Limitations

- New-image raw42, checked warmup+n7 and E4A before/after pass. The optional
  full processor RTL address sweep was stopped after its passing prefix;
  it is not a complete 42-case RTL pass. Required 56 component cases and
  six integrated numerical saturation invocations pass separately.
- The checked domain is fixed I1/J<=9/K<=28, tile-aligned origins, canonical
  native K tails, Index32/pair and Scale256. Unsupported geometry is rejected.
- More than 256 live Scale slots, 33 compact Index rows, negative mathematical
  shifts, stale incarnations, nonzero interior compact A filler and unresolved
  ownership do not get fallback execution.
- Unknown real DMA completion is quarantined by policy. Physical bus-error
  injection and recovery are not verified.
- Two original Scala harnesses do not compile against their pre-existing ingress
  constructor contract. Both old/new-image API smoke have the reserved-key status
  expectation mismatch described above. None was edited to claim green.
- The default SBT whitelist does not automatically include the opt-in resident
  test sources; use the explicit commands in section 12.
- No prefetch, micro-tile alternation, production pipeline replacement,
  Scale fast path, report-copy optimization or Attention arithmetic change
  was implemented.
- LSP could not resolve the configured C include/standard-library environment
  and no Scala language server was available. Actual C/C++ and Scala builds
  provide the compilation evidence. The existing header's
  `POTAL_SCALE_ENTRIES` redefinition warning was not suppressed.

## 19. H3-6E4C Pipeline Integration Recommendation

The matching new-image numerical and E4A before/after gates now pass. E4C can
use this checked primitive as a baseline; it is not a completed pipeline.
Keep the production scheduler unchanged until separate pipeline acceptance.

Before introducing overlap, define a scheduler-owned mapping from original
K/J and ordered Residual fragments to immutable resident, Scale/Index and output
epochs. Preserve the independent arithmetic oracle and distinguish compact
fragment order from dense Main order. Use the existing 256/32 admission limits;
do not assume the 270-slot full P1 pair can stay live.

Make buffer readiness and consumer retirement explicit before allowing another
B load or Index update. Continue to premark potentially accepted commands.
Unknown completion must keep ownership quarantined; it must not trigger retries,
new DMA, output publication or reassignment of the six pools.

Measure P1 Scale programming and the extra view/config/Fence/readback commands
before choosing a future reuse/P2 optimization. The diagnostics-off n=7
primitive baseline is recorded here; extend it to matched schedule comparisons,
then collect completion and bus
instrumentation if claiming hardware waiting or overlap. CPU issue intervals
alone cannot establish PE utilization, DMA overlap or a full graph speedup.

Only after those correctness, ownership and measurement gates should a separate
change implement micro-tile alternation/prefetch. This B2 change supplies the
generated execution interface, not the completed paired pipeline.
