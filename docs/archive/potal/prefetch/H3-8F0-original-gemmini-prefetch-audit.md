# H3-8F0: pinned Gemmini versus PoTal loop/prefetch architecture

Date: 2026-10-29. Read-only source audit; no builds, tests, guest access,
FPGA work, Git writes, submodule updates, or production edits.

## 1. Result and provenance

**PoTal preserves the original two-slot next-loop LdB mechanism. It does not
preserve a nonexistent load-only LOOP_WS instruction.** A real second loop
can generate B DMA while the first loop executes, subject to arbitration,
capacity, address dependencies and downstream backpressure. Ordinary raw
MVIN cannot pass configured LoopMatmul slots. D2 deliberately uses that raw
path to load B without also launching the next Main's computation.

Source architecture alone does not prove physical overlap or performance.
The D2 report records physical overlap for one fixture; this audit did not
rerun it. No new Hardware PASS is claimed.

Root: `/home/alveo/firesim/target-design/chipyard`.
All source paths below are relative to this root, unless explicitly introduced
as software-relative.

- First inspected `GIT_MASTER=1 git show
  185293e87cbbc2a336410f49a3fd73d01ed8d04e:src/main/scala/gemmini/LoopMatmul.scala`
  inside `generators/gemmini`, before working-tree source investigation.
- Gemmini pinned and observed HEAD:
  `185293e87cbbc2a336410f49a3fd73d01ed8d04e`.
  A read-only diff against that object for `src/main/scala/gemmini` and the
  requested header was empty. Subsequent file reads therefore provide exact
  pinned line evidence, including the leading blank lines.
- At that revision `software/gemmini-rocc-tests` is a **040000 tree**, not a
  gitlink: tree `49875959dc98d8e2a1683bb17719fcd9a5fada56`.
  Header `software/gemmini-rocc-tests/include/gemmini.h` is tracked directly
  as blob `e877f187c5be49a362510c4ada20126a94743005`.
  There is no separate nested header submodule revision to substitute.
  This pinned header includes local ggml/debug additions; it is not asserted
  to be an unmodified upstream release.
- PoTal RTL belongs to the enclosing Chipyard repository, not a separate
  repository at `generators/potal`. Observed enclosing HEAD:
  `bf1d98ca8a291a53845648ae475c80e01fd0b83d`.
  Scoped status for `generators/potal/src/main/scala/potal` was empty.
- Software checkout:
  `generators/potal/software/workspace/llama.cpp-potal`;
  observed HEAD `b5683e474b6b8708fe90f3d5a150a56322f64081`.
  Scoped status for the audited `potal.c` and `matmul/paired-auto.cpp` was
  empty. This is the audit snapshot, not a promise that the parent's shared
  checkout will remain unchanged afterward.
- Read the applicable Chipyard `AGENTS.md`; directory discovery found no
  deeper `AGENTS.md` under PoTal/software. Every shell sourced conda.sh,
  changed to Chipyard, then sourced env.sh. The missing `.conda-env` printed
  `EnvironmentLocationNotFound`; read-only Git still returned exit 0.
  Dirty unrelated Gemmini/Chipyard paths were left untouched.

## 2. Original slots, generators, loop IDs and completion

### Slot allocation is not a two-command DMA queue

`generators/gemmini/src/main/scala/gemmini/LoopMatmul.scala:784-795`
declares two complete descriptor states, a head, and tail `~head`.
Configuration targets the head until configured, otherwise the tail.
The input Queue can consume loop configuration/run instructions while a
slot is available; ordinary non-loop instructions require no configured
loop (`:850-863`).

Each of LdA, LdB, LdD, Execute and StC is one generator with its own stored
request and loop ID. Each independently selects head until that head's
corresponding `*_started` flag is true, then selects tail:

- LdA `LoopMatmul.scala:958-975`;
- LdB `:978-995`;
- Execute `:998-1024`;
- LdD `:1027-1050`;
- StC `:1052-1074`.

LdA/LdB request validity requires configured and not started. Their local
`req.ready` additionally requires an idle generator. Execute requires
configured, not started, and **LdA/LdB/LdD started**, not DMA completed
(`:1015-1016`). Its single generator can accept tail only after finishing
head's command generation. StC additionally requires Execute started.

LdA and LdB generate LOAD and LOAD2 respectively; zero DRAM address returns
the generator to idle without issuing the load. Their iterators advance on
command `fire`, not memory responses:
`LoopMatmul.scala:74-122,174-238`.

Execute emits PRELOAD then COMPUTE, iterating i fastest, then j, then k.
The generator's eligibility is:

- `lda_completed || ld_ka > k || (ld_ka == k && ld_i > i)`;
- `ldb_completed || ld_kb > k || (ld_ka == k && ld_j > j)`;
- `ldd_completed`;
- all three, not ROB-overloaded, and not skipped.

These are the literal pinned expressions, including the `ld_ka` reference
in the second expression; this audit does not silently rewrite it to
`ld_kb`. Evidence: `LoopMatmul.scala:451-491`.
The enclosing module defines these completion inputs as generator idle or
a different generator loop ID (`:872-879`). Once a load generator moves
to the other slot, all its former loop's load commands have been generated;
that does not mean their data has arrived.

`*_completed` slot flags are set from generator idle, and head resets/flips
when all five flags are set (`LoopMatmul.scala:1083-1115`).
Thus **slot retirement, generator completion, DMA completion and final ACC
commit are different events**.

### Actual DMA and dependency timeline

Verified partial order, not measured cycles:

1. LdB emits LOAD2 into the reservation station; its generation iterator
   advances immediately on that handshake.
2. ReservationStation decodes destination row intervals and establishes
   load/execute/store dependencies. Execute reads cannot issue through an
   overlapping older load. Same-queue ordering is released on issue;
   cross-queue data dependencies survive until completion.
   `generators/gemmini/src/main/scala/gemmini/ReservationStation.scala:216-292,309-338,390-435,449-478`.
3. LoadController selects load state 1 for LOAD2, allocates a byte-counted
   DMA command tag, emits row requests and can dequeue the command once
   row requests are sent. Completion instead comes from returned bytes:
   `generators/gemmini/src/main/scala/gemmini/LoadController.scala:60-76,83-130,152-170`.
4. StreamReader reserves transaction metadata, receives TileLink D beats,
   repacks them, and carries command ID/byte counts/last through its response.
   Finishing request issue is not finishing TileLink:
   `generators/gemmini/src/main/scala/gemmini/DMA.scala:73-109,205-264,275-296`.
5. Scratchpad accepts the scaled/repeated data into the selected physical
   bank. Its final response is generated on the accepted last output, not
   merely a DRAM request or a TileLink beat:
   `generators/gemmini/src/main/scala/gemmini/Scratchpad.scala:364-436,512-549`.
6. DMACommandTracker subtracts returned bytes, reports completion at zero,
   then clears its entry on completion handshake:
   `generators/gemmini/src/main/scala/gemmini/DMACommandTracker.scala:61-95`.
   Controller routes that completion back to ReservationStation and the
   appropriate matmul load-utilization counter:
   `generators/gemmini/src/main/scala/gemmini/Controller.scala:233-250,312-330`;
   `ReservationStation.scala:455-460`.

Original ExecuteController separately reads SP into the array, can overlap
compute/preload, and retires output-bearing commands on tagged final mesh
output. Other EX commands can use pending completion IDs:
`generators/gemmini/src/main/scala/gemmini/ExecuteController.scala:585-621,632-679,874-898,905-988`.

### Arbitration and resource counters

`generators/gemmini/src/main/scala/gemmini/LoopMatmul.scala:812-870`:

- A/B arbiter forces the head loop when generators carry different loop IDs.
- With `weightA=0` and static weighting enabled it uses generator idle and
  K/J progress, not a programmable round-robin fairness guarantee.
  `generators/gemmini/src/main/scala/gemmini/WeightedArbiter.scala:28-69`.
- Global fixed-priority inputs are StC, Execute, LdD, then the A/B result.
- Global load/store/execute utilization increments on emitted command
  handshakes and decrements on RS completion counts. These are outstanding
  command counts, not per-slot DMA-done flags or elapsed-time counters.

Consequently next-loop B generation may be delayed by head A/B generation,
valid higher-priority commands, a full load reservation budget, dependencies,
load-controller queues, translation or DMA/SRAM backpressure. The design
permits overlap; it does not guarantee early admission or positive speedup.
Original Controller busy-combination counters (`Controller.scala:336-345`)
are controller activity intersections, not qualified PE-MAC overlap.

## 3. Scratchpad selectors are pair selectors, not loop IDs

Let M be total scratchpad rows and H=M/2:

| Selector | A base | B exclusive end | B actual start |
|---|---:|---:|---:|
| 0 | slot's default base | slot's default end | end - K*J*DIM |
| 1 | 0 | H | H - K*J*DIM |
| 2 | H | M | M - K*J*DIM |

Evidence:
`generators/gemmini/src/main/scala/gemmini/LoopMatmul.scala:174-180,967-987,1007-1008,1111-1115`.
Physical bank comes from the high row-address bits:
`generators/gemmini/src/main/scala/gemmini/LocalAddr.scala:33-39`.
A/B selectors are independently encoded; they are not the head/tail bit.
Default selectors alternate regions with slots; explicit selectors can
retain the same region across multiple slots for reuse.

PoTal retains these equations:
`generators/potal/src/main/scala/potal/LoopMatmul.scala:1472-1498,1519-1534`.
For four equal banks, H is two banks. If each operand fits its designated
bank, selector 1 means A bank0/B bank1 and selector 2 A bank2/B bank3.
This is a footprint condition, not a property that prevents an oversized
legacy operand from spanning banks.

For the DIM32 256-KiB SP configuration there are 8192 rows total, 2048 per
bank: A bases 0/4096, B ends 4096/8192, B bases end-K*J*32.
Configuration: `generators/potal/src/main/scala/potal/PoTalConfigs.scala:29-52,127-146`.
H2 enforces separate A and B bank capacities before issue:
software-relative `ggml/src/ggml-potal/potal.c:1794-1827`.
These DIM32 addresses must not be replaced by the DIM16 resident-view
decoder's hardcoded bank addresses.

Disjoint next-pair ownership remains necessary: the RS protects already
allocated dependencies, not future head-loop reads that have not yet been
generated. A selector alone is not a residency generation or lifetime proof.

## 4. Original software really submits adjacent full loops

Pinned header:
`generators/gemmini/software/gemmini-rocc-tests/include/gemmini.h`.

- `tiled_matmul_auto`, `:2940-3046`: WS chooses double-buffered capacities,
  limits total SP rows to half and ACC rows to half, grows legal tile sizes,
  then calls `tiled_matmul`.
- `tiled_matmul`, `:1687-1698`: hardware WS/OS routes to `tiled_matmul_outer`.
- `tiled_matmul_outer`, `:797-801,843-847,888-946`: configures execution,
  store and three load states, loops i0/j0/k0 with k0 innermost.
  Bias is supplied only for first K; output C only for last K.
- Reuse is enabled for B when `J0*K0 <= 2`, A when `I0*K0 <= 2`, in WS.
  Explicit selectors choose 1 for the first panel and 2 for the other.
  Later j0 sets A=NULL; later i0 sets B=NULL (`:891-922`).
  The compile-time disable-reuse branch is at `:848-853`.
- Active `sp_tiled_matmul_ws`, `:744-751`, calls `gemmini_loop_ws`.
  The preceding long manually unrolled body is commented out.
- Macro `:412-420` emits five configuration instructions and one LOOP_WS
  run. Flags contain accumulation, activation, low/full precision,
  transpose, selectors and resadd; **no normal-GEMM load-only bit exists**.
- One ordinary CPU `gemmini_fence` follows the outer loop nest (`:956`);
  not one between every inner loop. Its RISC-V definition is
  `asm volatile("fence")` (`:339-345`), and is a no-op off RISC-V.
  This is distinct from checked response-bearing POTAL_FENCE.

Normal original timeline:

```text
configure/run L0 -> load/execute/store generators issue portions of L0
configure/run L1 in other slot
LdB generator finishes L0 issue -> may issue L1 B to disjoint region
Execute generator continues L0; RS waits for each relevant L0 DMA dependency
Execute generator finishes L0 issue -> may start L1 instruction generation
physical controllers/array/ACC can still contain older work
end-of-call CPU fence (not a per-inner-loop software drain)
```

When C is NULL, automatic ACC-half advancement is suppressed
(`LoopMatmul.scala:1022-1024,1047-1050,1071-1074`), supporting successive K
contributions to one output region. A NULL A or B suppresses that operand's
DMA only; the loop still computes using resident data.

## 5. What PoTal preserves and what blocks progress

### Preserved: next-loop LdB

`generators/potal/src/main/scala/potal/LoopMatmul.scala` retains:

- two slots and opposite tail `:869-892`;
- head-forced A/B arbitration and global priority `:917-939`;
- utilization completion accounting `:942-949,1248-1252`;
- LdA/LdB request selection without an EX-done gate `:1463-1505`;
- Execute started conditions `:1507-1544`;
- generator-based completion and slot reset `:1618-1641`.

Its load-ahead predicates and normal pre/compute generator remain at
`:438-525`. Additional scale-block iteration is metadata generation, not
an independent DMA completion mechanism. No blanket "PoTal disabled the
original next-loop B prefetch" conclusion is supported.

PoTal LoadController and DMA retain the same completion mechanism
(`generators/potal/src/main/scala/potal/LoadController.scala:60-128`;
`DMA.scala:73-109,275-296`).
Scratchpad retains final accepted output accounting and now exports
physical DMA-write observation (`Scratchpad.scala:419-439,516-550`).

### Blocked: raw commands during configured loop generation

`generators/potal/src/main/scala/potal/LoopMatmul.scala:1070-1096,1150-1152`
forwards normal raw commands only when `!loop_configured`. A second loop
configuration/run can enter an available slot; a CONFIG_LD/MVIN cannot
bypass those configured loops. This is inherited from original
`generators/gemmini/src/main/scala/gemmini/LoopMatmul.scala:850-863`.

It is a **generator-admission barrier**, not necessarily a physical
Execute-completion barrier. Once all generators finish, raw B DMA may
overlap older array/ACC work still downstream.

### Additional ordering: Scale, Index, ACC and writeback

1. Scale/Index control commands are routed before LoopConv/LoopMatmul,
   at the shared FIFO head:
   `generators/potal/src/main/scala/potal/Controller.scala:235-275`.
   Free Scale allocation or an inactive peer Index update can proceed
   while older execution exists; bypass is not unlimited command reordering.
2. PoTal PRELOAD atomically reserves output context and EX acceptance:
   `Controller.scala:565-614`;
   `PoTalPreloadReservationGate.scala:48-73`.
   A generation-qualified Scale hit, context capacity and row reservations
   are prerequisites:
   `PoTalWriteback.scala:148-205`.
3. Scale memory cannot overwrite a valid slot even with another generation:
   `generators/potal/src/main/scala/potal/PoTalScaleMemory.scala:101-123,127-153,174-201`.
   Release and POTAL_FENCE require global older-command drain plus writeback
   drain (`Controller.scala:801-825`;
   `PoTalScaleCommandIngress.scala:361-368,451-470,516-525`).
   Release is not just "this output has reached the mesh."
4. Indexed B protection covers configured loops, pending dispatched runs,
   queued PRELOADs, outstanding RS entries and actual EX lookup:
   `Controller.scala:343-418,455-490`.
   Indexed RS read hazards conservatively cover the full resident B extent:
   `generators/potal/src/main/scala/potal/ReservationStation.scala:251-298`.
   Normal address-dependency handling remains at `:386-414`.
5. Each generated PoTal PRELOAD carries its own loop-derived Scale
   generation/base/block, Index buffer/J/K, work ID and first/last flags:
   `generators/potal/src/main/scala/potal/LoopMatmul.scala:1098-1142`.
6. Paired Main always targets ACC0; paired Residual ACC1, independently of
   which A/B pair is selected:
   `LoopMatmul.scala:1531-1534,1561-1565,1591-1595`.
   Switching pair does not provide a second Main ACC epoch.
7. Final PoTal RS retirement requires both EX raw completion and scaled ACC
   write latency completion, through the delayed ROB path:
   `Controller.scala:723-798`;
   `generators/potal/src/main/scala/potal/PoTalWriteback.scala:253-325`.
   Work drain also requires sealed contributions and no active work:
   `PoTalWriteback.scala:201-205,343-347`;
   `PoTalWorkTracker.scala:37-64`.

These gates prevent several physical hazards. They do not prove that
reordering two high-level Main/Residual windows preserves saturating
contribution order, readback epochs or metadata ownership.

## 6. D2 raw path: exact meaning and timeline

All paths in this section are software-relative.

`ggml/src/ggml-potal/matmul/paired-auto.cpp:2073-2111` prepares the opposite
retired pair for the next K panel before issuing the current panel's last
Residual. Immediately after that Residual issue, without an intervening
clock read, it calls prefetch. It requires same J group, next K panel and
different pair. This is not Main-K0/B1 early prefetch.

`ggml/src/ggml-potal/potal.c:1830-1852`:

- marks panel loading/pending before CONFIG;
- emits ordinary CONFIG_LD state0 and K*J tile MVINs;
- uses checked rear-packed B addresses in the target odd bank;
- performs no PRELOAD, COMPUTE, Scale allocation or ACC update.

The immutable entire B allocation is bound and published once at idle H2
begin (`potal.c:1753-1790`); its source extent/layout is checked before
commands (`:1794-1827`). It is not an unchecked raw issuer.

```text
prepare next-panel reference (not readable)
submit current last Residual LOOP_WS
submit raw CONFIG_LD/MVIN for next B
    FIFO/LoopMatmul waits for configured generators to finish issuing
raw B requests enter LD/RS/DMA while older physical execution may remain
first checked global Fence/Release covers BOTH current work and next B DMA
paired_drained -> h2_completed publishes B and completes declared readers
next Main uses real checked ordinary descriptor, with encoded B address = 0
```

Publication is `potal.c:1185-1196,1725-1751`; source queries cannot publish.
Next Main validates the real descriptor and identical rear SP placement,
then suppresses only encoded B DMA (`:1909-1945`).
Pending-B blocks another managed loop issue (`:1908-1920`).
The normal loop emitter also issues load/EX/store CONFIGs and a CPU
publication fence (`:1489-1506`), so naively issuing it after an active loop
does not automatically exploit the RTL's available second slot. Even
removing that CPU fence alone would not suffice: its leading ordinary
CONFIG commands encounter the same configured-loop admission barrier.
The original outer tiler avoids this particular barrier by configuring
once before its sequence of loop-only submissions.

D2 chooses raw MVIN because it needs **only next B residency** while the
current Main/Residual/Scale/readback order stays intact. It does not choose
raw MVIN because the old two-slot load generator was removed.

The existing D2 report records 103 cross-pair qualified MAC/B-write
intersection cycles in its full-row fixture and zero for two shorter
fixtures: `H3-8D2-next-k-prefetch-implementation.md:130-143`.
Those are historical reported observations, not this audit's measurements
or evidence that every shape overlaps. Its command-declared byte counts
are not measured DMA bus bytes (`:203-217`).

## 7. Why existing LOOP_WS is not a safe B-only prefetch

For normal GEMM there is no independent EX-disable flag.

- A=NULL means no A DMA, not no A reads or compute.
- C=NULL means no MVOUT, not no ACC writes.
- D=NULL means no bias DMA, not no output.
- `ex_accumulate` selects overwrite versus accumulation; it does not skip EX.
- Zero I/J/K is not a supported PoTal no-op: positive dimensions are
  asserted (`generators/potal/src/main/scala/potal/LoopMatmul.scala:1424-1448`).
- The only EX skip is resadd; that mode redirects A/B loads to ACC and
  makes B accumulate, so it is not a B-to-SP load-only workaround.
  Original `LoopMatmul.scala:89-92,194-197,1013,1077-1081`;
  PoTal `LoopMatmul.scala:1537,1612-1616`.
- A resident-view extended run changes address selection, not these compute
  semantics (`generators/potal/src/main/scala/potal/LoopMatmul.scala:1524-1540`).

Queuing the **real next full compute** can be legal for correctly prepared
adjacent loops; original software deliberately does this. It is not
equivalent to B-only prefetch and is not automatically invalid in every
PoTal use. For the present paired schedule, it would require new proof of:

- ACC0/ACC1 contribution and first/last/readback epoch ordering;
- free, already written correct Scale slots/generations for the next loop;
- retained current B/Index tables until every indexed consumer finishes;
- no Release needed to obtain resources held by a configured stalled tail.

A concrete conditional deadlock follows directly from current gates:

```text
tail loop configured, PRELOAD waits for missing/conflicting next Scale
software needs old Scale Release before writing that next Scale
Release waits allDrained, which includes that configured tail and RS/WB
tail cannot finish until next Scale exists
```

Similarly an Index update placed after an indexed run targeting that same
busy table stalls until its consumers finish; updates needed by that run
must precede dispatch. These are conditional source-derived hazards, not
claims that the current D2 implementation triggers them.

DMA fills SP; PRELOAD transfers resident B into the WS array;
COMPUTE consumes A and stationary B; resident reuse suppresses repeated
SP DMA; next-K prefetch fills a future K panel in another owned pair.
These five operations must not be conflated.

## 8. A/B/C/D feasibility ranking and unresolved decisions

Grades here rank feasibility, not new validation results:
A = existing implemented path with historical hardware evidence;
B = source-feasible software scheduling candidate, unverified here;
C = needs a separately settled ownership/admission/completion contract;
D = incompatible with the present encoding or stated semantics.
Do not mistake B for a newly tested software prototype.

| Grade | Candidate | Evidence and limit |
|---|---|---|
| A | Existing checked D2 late raw-MVIN next-K prefetch, then resident Main | Emitter, owner/publication and caller order traced above; historical D2 hardware report only, not rerun. No performance improvement inferred. |
| B | Exploit preserved next-loop LdB by submitting a real complete next compute with all resources prepared | Original and PoTal gates permit it. Existing C11 emitter fences/configs and managed pending-B rules prevent treating it as a drop-in optimization. Exact numeric, resource and CPU ordering proof is still required. |
| C | Turn that capability into an early paired Main/Residual pipeline or independently completed B-load transaction | Must settle ACC epochs, concurrent Scale capacity/lifetime, Index ownership, FIFO placement and whether global publication remains acceptable. This audit selects no new contract or RTL/ISA. |
| D | Encode B-only prefetch as an ordinary LOOP_WS using NULL A/C, zero dimensions, resadd, or resident-view selection | It still computes, is invalid, or redirects load semantics to ACC. None supplies checked B-only SP residency with independent completion. |

Original next-loop LdB capability itself is source-proven, not blocked;
physical overlap for a newly proposed full-loop schedule remains unverified.
No finding justifies automatic RTL rebuild or ISA changes.

Unresolved parent decisions:

1. Is the target still late B-only prefetch, or a different full-compute
   schedule? They have different acceptance contracts.
2. Can the real paired Scale live set coexist without releasing an active
   generation, and can ACC contributions/readbacks retain exact order?
3. Can existing CPU fence/configuration and managed completion boundaries
   support any intended earlier submission without changing their contracts?
4. Would earlier admission repay command/bookkeeping cost on matched
   workloads? Source cannot decide performance.

These are explicit decision/verification gaps, not blockers to this completed
architecture audit. The missing environment blocks build validation, which
was out of scope anyway. Parent retains all checkpoint, environment and
completion-optimization work. Only this note was created.

## 9. Disk-interruption recovery

This report restores the completed read-only audit left at
`/tmp/h3-8f0-original-audit.md` by the interrupted session. On resumption,
the software HEAD was `b5683e474b6b8708fe90f3d5a150a56322f64081`,
Chipyard HEAD was `bf1d98ca8a291a53845648ae475c80e01fd0b83d`, and
the Gemmini gitlink and HEAD were both the pinned revision above. The
unrelated dirty state matches the preserved H3-8E inventory. No reset,
submodule update, cleanup, commit or push was performed in this session.

The filesystem has 30 GiB available as reported by `df -h`, with inode
usage at 5%. Chipyard `.conda-env` is missing; sourcing its `env.sh` still
sets project paths but reports `EnvironmentLocationNotFound`. Activating
the existing `/home/alveo/firesim/.conda-env` afterward restores CMake
3.26.3, Verilator 5.008, Java and SBT. This supersedes the historical
environment limitation in section 8; it does not retroactively add tests.

## 10. Implemented follow-through

Sections above preserve the pre-implementation audit and its then-open
questions. The selected implementation is narrow scheduler-owned B-only
LdB, not speculative execution of the next full Main descriptor. Existing
Scale lifetime, real Release/Fence, ACC order and owner contracts remain.
The managed-completion and Early reports document the implementation,
diagnostic integration corrections and actual FPGA evidence. Final
seven-sample route comparisons are in `H3-8FG-performance-comparison.md`.
They show physical overlap but only small incremental Early gains;
production defaults remain unchanged.
