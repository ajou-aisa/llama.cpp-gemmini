# H3-6E3: Micro-tile Pipeline Feasibility

Date: 2026-10-26. Fixed DIM16, K28. No production schedule or RTL replacement.

The K28 alternating schedule is software-verified. The complete hardware
pipeline is **not implemented**: raw PRELOAD cannot carry PoTal SCU metadata,
native tiny LOOP_WS cannot select arbitrary resident K/ACC tiles, and indexed
LOOP_WS binds its output to ACC1. These are explicit interface constraints,
not a reason to weaken saturation, reload every tiny B tile, or call a Main
GEMM "prefetch-only."

## 1. Paper Dataflow Contract

Retain shared resident B, separate activation/INT32 ACC regions, frozen Main
stripe/I/J, ordered original-K indexed Residual fragments, microtile alternation,
independent next-pair prefetch, group ACC context switching, single-path tails,
and Host preparation/reconstruction overlap. K28 is accepted; restoring K48
by enlarging B storage is outside this task.

For every output, retain:

```text
partial = signed DIM16 dot
shifted = Sat32(partial * 2^carrier)
acc = Sat32(acc + shifted)
```

Main K order and each Residual lane's compact-fragment order are independent
invariants. Alternating paths with separate ACCs does not authorize reordering
contributions within either path.

## 2. Current Hardware/Software Structure

There are two distinct implementations:

- `matmul/paired-auto.cpp`: the active checked hardware route. Whole Main
  LOOP_WS followed by its Index windows, with completion/ownership dependencies.
- `matmul/paired-schedule.cpp` and `paired-emulator.cpp`: an older functional
  microtile schedule with groups, logical prefetch and raw ACC spill/restore.
  Its hardware selection refuses execution.

The older header's statement that current RTL has no paired execution is stale.
It describes that model's missing hardware backend, not active paired-auto.
No stale comment was changed outside this task.

Current geometry: four 4096-row scratchpad banks, two pairs, 1024 INT32 ACC
rows, Index32 per pair, shared Scale256 and 64 work IDs per scheduler slot.
At I1/J9/K28, resident B uses 4032 rows. Arbitrary extra B capacity is not assumed.

Software paths below are relative to `ggml/src/ggml-potal/`.
RTL filenames mean Chipyard `generators/potal/src/main/scala/potal/`.
Detailed producer/consumer citations are retained in
`/tmp/h3-e3-hardware-audit.md` and `/tmp/h3-e3-software-schedule-audit.md`.

## 3. Proposed Micro-tile Schedule

Reuse `build_paired_schedule` rather than adding a second planner. For each
stripe, J group and K28 window:

1. Prepare Main A, shared B, Scale context and the first Residual group.
2. Begin current pair; logically prepare the next window in the other pair.
3. Alternate Main and Residual microtiles while both have work.
4. Execute only the surviving stream's real tail.
5. Save unfinished Residual groups, restore their raw INT32 values when resumed,
   and complete finished groups without unnecessary saves.
6. Retire the residency after all its consumers; keep ACC epochs independent
   from scratchpad window lifetimes.

The test freezes I1/J9/K28 for M17/N145/K928. This crosses stripe, J and K
tails: rows16+1, columns144+1, and K448+448+32.
The planner's returned K must equal28, not silently shrink.

The old equal-A-half model supports K28 only while `I*28*16<=2048`.
I>=5 cannot use that exact layout at K28. This is not evidence that the active
dynamic Main-A-plus-window layout fails. The capacity-fallback control retains
its original stripe/I/J rather than disguising a smaller fixture.

## 4. Weight Residency and Bank Ownership

Pair p owns A bank2p and B bank2p+1. The current resident B layout is:

```text
B_start = selected_half_end - K_panel_tiles * J_group_tiles * DIM
B(k,j) = B_start + (k * J_group_tiles + j) * DIM
```

One B residency survives all Main and Residual consumers. Pair replacement
requires their completion, not just submission. Residual A replacement also
requires its preceding group's readers to complete.

Logical model loads are synchronous Host copies. Their ordering and disjoint
bounds can be verified, but they do not establish actual DMA concurrency.
The prototype records logical traffic only.

## 5. Main/Residual Interleaving

`LoopMatmul.scala:484-520` has one Execute generator. A request traverses its
whole I/J/K before the generator accepts another. Two scheduler slots and
multiple outstanding requests do not imply concurrent Main/Residual PE MACs.

Raw CPU PRELOAD/COMPUTE receives zero PoTal metadata at
`Controller.scala:215-222`. Only generated PRELOAD attaches Scale base/generation,
block, work ID, first/last contribution and indexed context
(`LoopMatmul.scala:1068-1113`). Untagged results do not follow the same SCU path.
A software wrapper cannot fabricate this missing RTL metadata.

For native tiny LOOP_WS, B start is recomputed from the tiny invocation's own
K/J geometry. B=NULL suppresses DMA but does not add arbitrary resident K/J
offsets or preserve a larger panel's row stride. A matching layout or suffix
is possible; a general interior K28 microtile is not.

Indexing provides original-K selection, but requires paired Residual and ACC1
(`LoopMatmul.scala:1235-1267,1382-1384`). It cannot independently select ACC0.
Therefore a general alternating resident Main/Residual microtile hardware
schedule needs an explicit addressed/metadata-bearing interface.

## 6. Double-buffered K Prefetch

Raw B-only signed8 MVIN exists: CONFIG_LD sets identity/byte stride; each MVIN
loads a valid DIM tile into the free pair's B layout. This is not a GEMM.

What is missing from the checked session:

- Load-only DMA ownership and readiness.
- Publication of resident B base/J stride/K extent and identity.
- Exact-layout resident Main consumption without reloading B.
- Index invalidation and last-reader protection on replacement.

Raw non-loop commands also pass LoopMatmul only when no loop is configured
(`LoopMatmul.scala:1060-1062,1465-1482`). A later raw load can potentially overlap
previous outstanding execution after generation ends, but cannot guarantee early
admission throughout the current whole loop.

The existing-ISA wrapper opportunity is therefore **B-only load plus compatible
consumption**, not proven overlapped prefetch. Arbitrary resident Main selection
and guaranteed early admission remain separate RTL/interface work.

## 7. ACC Save/Restore

Existing full-width MVOUT supplies the save operation, followed by a successful
fence. Raw full-width MVIN with ACC address, `shrink=0`, `accumulate=0`, and
nonzero source row stride overwrites signed32 values exactly.

```text
ACC tile row = bank*(ACC_ROWS/2) + (i*J_tiles+j)*DIM
DRAM origin = source + i*DIM*ldc + j*DIM
CONFIG_LD byte stride = ldc*sizeof(int32_t)
```

Bit29 is a readback flag, not MVIN input-width selection. H3-7B bias MVIN uses
stride0 and accumulate1; it broadcasts/adds and is not restore.

Restore must preflight all fields before issue, track partial DMA before every
MVIN, preserve buffers until a successful drain, and keep sticky failures.
After restore, a continued GEMM must accumulate rather than overwrite.
SCU is not applied to restored values.

The bounded prototype adds this wrapper in a test-only C translation unit that
includes unchanged `potal.c`, reusing the existing internal session/error
mechanisms. It does not expose a new production API.

## 8. Single-mode Drain

**Main completes first:** retain resident B and unfinished Residual ACC/group
contexts. Its dead A region and released Scale slots may be reclaimed after
last-reader completion. Current session `main_a_tiles` admission needs an
explicit retirement contract before Residual may use that region.

**Residual completes first:** Main continues its original-K partial sequence.
No dummy Residual work is required. Borrowing ACC1 is not achieved by deleting
a capacity check; current Main destination and per-loop output layout remain
restricted.

Indexed B descriptors must remain inside bank1 or3. Indexed output targets
ACC1. Free SRAM/ACC capacity does not change those address/selection contracts.
Group save/restore virtualizes values at transfer/command cost; it does not
increase one invocation's physical addressing range.

The software fixtures exercise Main-only and Residual-only tails. They retain
fixed partitions, so dynamic capacity reclamation is not claimed implemented.

## 9. Scale/Index/ACC Dependencies

The old emulator has separate per-pair carrier vectors, not the physical shared
256-slot Scale memory. The new test overlay allocates two disjoint 126-slot
contexts at K28/J9: 252 live slots. It verifies no overwrite while live and
cycles generations through 255 to 1 only after ownership release.

This logical shared-carrier model is **not** the current command encoding.
Active Main uses passes_per_block=2, whereas Indexed Residual uses 1 with its
own rebased Scale window. Current Main 126 + Residual 18 + next Main 126 needs
270 slots, exceeding 256. Even a one-fragment Residual window gives 261.
Simply preprogramming two Main windows while keeping current Residual scales
live therefore fails. Delayed programming, narrower window reservation or an
explicit microtile Scale mapping must be designed and measured; the old
emulator's independent contexts do not solve this hardware constraint.

Each Residual microtile uses at most 16 original-K entries, fitting Index32.
The overlay validates original/resident K and block identity. It does not
pretend that model gather operations are real INDEX_WRITE instructions.
Actual checked Index32 command translation is separately exercised by the H2
window diagnostic and preserved active executor regressions.

Release and Fence both require `allDrained` (`Controller.scala:787-808`).
Releasing after each microtile would add global synchronization. Index reader
protection spans queued loops/PRELOADs, reservation station and Execute lookup.
No Scale validation or Release synchronization was weakened.

Failure tests preserve original first errors, avoid retrying failed fences,
retain unknown DMA ownership and prohibit accelerator reuse. A purely
functional emulator cannot itself prove physical DMA lifetime.

## 10. Required C11/RTL Interfaces

| Operation | Source-supported classification |
|---|---|
| Whole Main then Indexed Residual | Existing checked API |
| Full-width raw ACC save | Existing checked readback API |
| Full-width raw ACC overwrite/restore | Checked wrapper on existing ISA |
| B-only DMA into a free pair | Checked wrapper on existing ISA |
| Publish independently loaded B and consume matching native layout | Checked ownership/residency wrapper |
| Native arbitrary resident K/J and ACC output offsets | RTL/ISA fields or lower-level metadata-bearing tile command |
| Indexed selection into ACC0 | Separate Index mode from ACC-bank selection in RTL/ISA |
| Raw scaled PRELOAD/COMPUTE | Explicit PoTal metadata-binding interface |
| Guaranteed raw-load admission while current loop generates | Admission/generator scheduling interface change |
| Dead-A/Scale reclamation | Explicit software retirement and revised checked admission |
| One enlarged indexed operation spanning ACC banks | Destination/extent interface change |
| Indexed B spanning freed A banks | Index descriptor/addressing interface change |

A future tile command must bind B base/stride/offset, A range, ACC epoch,
Scale generation, work identity and first/last semantics atomically.
It must preserve writeback reservation and physical commit-qualified completion.
No opcode, new RTL or FPGA image was introduced here.

## 11. Prototype Implementation and Tests

`tests/potal-e3-prototype-test.cpp` reuses the existing schedule, emulator and
three-way fixtures, then adds active paired-auto H2 comparison and separate
wide-arithmetic SCU/ACC/reconstruction checks.

K28 fixtures passed for zero, sparse, dense multi-lane and mixed
saturation/cancellation inputs. Every Main INT32, Residual INT32 and reconstructed
INT64 value agreed across applicable paths. Full F32 merge modes were compared
separately, not required to equal each other.

The logical overlay independently enumerates all contributions, rejects
duplicates, checks shared Scale capacity and generation wrap, Index demand and
bank retirement. For the dense multi-lane fixture it recorded 37120 logical
save bytes and 37120 restore bytes. These are not measured bus transactions.

Additional negatives remove or duplicate a Main contribution, admit an exact
logical 256-slot two-context reservation, reject its overflow, and reject
silently retaining K28 with the old I5 equal-half layout. Native tests passed.

Existing spill-corruption, first/middle/last failure, post-restore saturation,
overflow and transactional output tests passed. All nine existing Host
regression binaries also passed, including active C11 fatal/unknown ownership.
No Host test result is labeled hardware pipeline success.

## 12. Actual Hardware Verification

The complete alternating hardware pipeline is blocked by section5 interfaces.
No forced CPU fallback, repeated B reload workaround or untagged raw arithmetic
is offered as its implementation.

The raw ACC-context test passed **seven actual RoCC runs** on the unchanged
DIM16 U280 image. Source:
`tests/potal-e3-context.c`, `tests/potal-e3-context.h` and
`tests/potal-e3-context-test.cpp`.

Each run verified both ACC banks, full and tailed 2x2 tile shapes, padded host
strides, INT32_MIN/MAX and signed bit patterns, save/clobber/restore, preservation
of the other bank, switching back to an earlier saved group, and exact
saturating bias accumulation after restore. Restore does not apply SCU again.
The hardware continuation tested is a checked ACC bias-add, **not a resumed
metadata-bearing microtile GEMM**. Numerical GEMM continuation remains covered
by the software schedule; the full hardware connection remains absent.

Every run issued 240 commands: 72 MVIN, 88 MVOUT and 40 Fence commands.
Logical/command-declared totals were 51560 restore bytes, 14456 bias-lane bytes
and 80472 readback bytes. These are not measured DMA bus transfers.
Status checks and clean session close passed; locked static DMA buffers
remained live through completion.

Software-only failure tests also passed partial-MVIN sticky failure,
one cleanup drain, idempotent cleanup, failed-Fence no-retry behavior and
permanently retained unknown ownership/buffers. Hardware fault injection was
not performed. No software fault test is described as hardware failure proof.

Binary SHA256:
`047db122046f6f5c2dffa8ede9c4266f410b673a5c9ec11a65e787caa0c5ecba`.
Evidence: `/tmp/h3-e3-context-run/uartlog`, with seven actual-RoCC PASS records
and `__E3_CONTEXT_EXIT__=0`. This proves the bounded raw context primitive,
not prefetch or microtile pipeline overlap.

## 13. Performance Comparison

See `H3-6E3-paired-bottleneck-isolation.md` for measured Serial Original,
Serial Restricted and Paired Original comparisons.

**Paired microtile hardware prototype latency: Not measured.**
There is no complete hardware prototype to benchmark. Functional-emulator
execution time is not substituted. Adoption conditions for a new default
therefore do not hold: no verified pipeline RoCC stream, physical overlap
evidence, reproducible latency benefit or instruction-overhead budget.

No production path was replaced. Any bounded ACC test proves a primitive,
not a pipeline speedup.

The seven checked raw-context tests took 0.838 ms median, 0.013 ms MAD
(samples 0.833, 0.825, 0.896, 0.830, 0.863, 0.838, 0.898 ms).
This CPU-observed interval includes validation, host comparisons and all
240 commands. It is neither isolated DMA latency nor a matched end-to-end
pipeline comparison. There is no existing equivalent whole-pipeline hardware
baseline for this primitive, so no speedup/adoption claim follows.

The K-controlled production comparison is n=7 per mode:
Serial K28 versus Paired K28 is 8.810 versus 14.291 ms for Decode and
52.684 versus 78.363 ms for FFN. The new primitive was not inserted into those
measurements. A future pipeline must beat those matched baselines after paying
its additional load, Scale, context-transfer and synchronization instructions.

## 14. Recommended Next Implementation Step

First define a metadata-bearing resident microtile interface or decouple
indexed B selection from ACC destination. Preserve K28 and frozen I/J.
Add checked load-only residency ownership and independently observable load
completion before attempting overlapping prefetch. Keep raw ACC overwrite
as a separate reviewed interface.

Feature verdicts:

| Feature | Status | Scope |
|---|---|---|
| Existing K28 shared-B whole-loop execution | A | Existing checked hardware path, not a new pipeline |
| K28 alternating microtile plan and ordered single tails | B | Software prototype verified |
| General resident Main microtile selection | C | Address/metadata interface missing |
| Indexed Main targeting ACC0 | C | Current Index mode forces ACC1 |
| Logical next-pair prefetch schedule | B | Software model verified; checked B-only residency connection missing |
| Guaranteed next-K overlapped prefetch | C | Admission/interface and hardware evidence missing |
| Raw INT32 ACC save/restore and bias-add continuation | A | Seven real RoCC runs; complete pipeline/GEMM continuation not claimed |
| Ordered single-path tails with fixed partitions | B | Software verified; dynamic capacity borrowing not implemented |
| Enlarged indexed ACC/B operation after reclamation | C | Current destination/descriptor contract excludes it |
| Old equal-A-half K28 layout with I>=5 | D | Main half-bank footprint does not fit |
| Physical Main/Residual PE overlap | C | Not measured; single generator is not proof of overlap |

A means implemented and hardware-verified in the stated scope. B means the
stated software prototype is verified but its hardware connection is incomplete.
Independent checked B-only load/residency publication and dead-A retirement
remain software interface designs, not separately verified features.
No pending physical behavior is promoted to A based on the Host emulator.

Adoption verdict: **do not replace the production schedule**. The functional
prototype satisfies its integer and logical-resource checks, and raw ACC
overwrite is hardware-verified. General resident microtile addressing/SCU
metadata, independent checked prefetch ownership and physical overlap evidence
are still missing. The active Scale mapping also cannot hold 270 live slots in
Scale256. Address these contracts before implementing a connected pipeline;
then require exact outputs, safe failure/cleanup and reproducible matched n>=7
latency improvement. K48 restoration, RTL edits and FPGA rebuilds were not
performed in this task.
