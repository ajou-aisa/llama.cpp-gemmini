# H3-6E4C2-0 - Independent B-Prefetch Admission and Completion Contract

Date: October 27, 2026.

## 1. Checkpoint verdict

**The existing Resident C11 API cannot safely execute this concurrent
experiment. No physical experiment was attempted.** It admits one global
pending load, compute or readback, and exposes no checked independent
completion query.

**An FPGA rebuild is not automatically required for a restricted late-prefetch
experiment.** The current RTL has fresh, ordered aggregate RS load/EX count
reads. With a new checked C11 sealed-prefix protocol, they can certify a
whole admitted load/EX prefix independently of the other class. This is a
source/generated-RTL feasibility result, not an implemented API or hardware
PASS. The prefix restrictions and validation obligations below are mandatory.

**Unrestricted, generation-tagged per-pair completion and earlier raw-MVIN
admission require additional RTL/interface support.** Existing counters do
not identify a particular pair or transfer generation. They cannot certify
one completed B while another load remains pending.

Proceed only with a separately verified checked-interface change. The
existing image is a candidate for the bounded prefix protocol, not approval
to issue concurrent commands through today's API. No speedup, DMA duration,
PE overlap or Hardware A is claimed for C2-0.

The delivered default-OFF software model passes 12 deterministic scenarios,
1,608 assertions and 94 expected refusals, with zero ownership violations.
It distinguishes existing global completion from a proposed prefix query.

## 2. Baseline, roots and changed files

```text
F = /home/alveo/firesim
C = F/target-design/chipyard
P = C/generators/potal
L = P/software/workspace/llama.cpp-potal
A = L/ggml/src/ggml-potal
S = /home/alveo/h3-e4b2/source
R = P/src/main/scala/potal
Q = xilinx_alveo_u280-firesim-FireSim-FireSimQuadRocketSinglePoTalD16Config-BaseXilinxAlveoConfig
G = S/sim/generated-src/xilinx_alveo_u280/Q/FireSim-generated.sv
```

The requested RTL repository is the actual C root; P is a directory in it.
L is an independent Git root. Short software source references below are
relative to A.

| Root | Branch | Unchanged HEAD |
|---|---|---|
| Software L | `potal` | `8fb19aa5b66ace4cc50014581e5b3fe5ad7fc546` |
| RTL C | `potal` | `1dfbe85aea66e0afc0e11072b983335a475a5b14` |
| Outer FireSim F | `aisa-customizations` | `5c3458a3f54917ab9502bda4a8b3cc3dd513304d` |

Preserved B2 bit identity:
`c4c71ef648f5c5a73a40f810fed98fdfb3862f1433b00592720f49631972beae`.
No FPGA programming, driver/guest run, RTL generation or RTL build occurs in
this phase. The capability and physical image execution evidence remain the
previous C1 evidence; they are not represented as a new probe.

C2-0 changes only:

```text
H3-6E4C2-0-prefetch-admission-completion-feasibility.md
ggml/src/ggml-potal/CMakeLists.txt
ggml/src/ggml-potal/tests/potal-e4c2-0-test.cpp
```

The CMake addition registers only the isolated `POTAL_E4C2_0_TESTS` option,
default OFF, before the existing API-only return. The complete C1 CMake block
is preserved byte-for-byte. No production API, scheduler, arithmetic, RTL,
ISA or capacity is changed. No commit or push is made.

## 3. Current C11 contract

The authoritative interface is `include/potal.h:640-839`; implementation is
`potal.c`, not the old test-local E4A handle.

| Mechanism | Current evidence | Consequence |
|---|---|---|
| Pending admission | `potal.c:983-1003`, `potal.c:1616-1624` | One `kind` plus one pending-pair/bank/tile/count/ref set. `resident_ws_ready(s,0)` rejects any pending/control-pending work globally. |
| Load reservation | `potal.c:1957-2003` | Pair identity and loading/source ownership are premarked before CONFIG or any MVIN. A successful issue return sets `issue_complete`, not readable B. |
| Resident publication | `potal.c:1649-1688`, `potal.c:2006-2014` | A successful global drain publishes only a matching fully issued load. `get` also requires global non-pending state. |
| Compute ownership | `potal.c:2204-2320` | Checked B/Scale/Index/epoch bindings precede effects. Epoch updates are staged under the single pending compute. |
| ACC readback | `potal.c:2323-2388` | Issue reserves a receipt; result is unreadable until a successful global drain and matching epoch/contribution count. |
| Retirement | `potal.c:2129-2142` | Pending work and resident-bound live Scale slots block retirement. Retirement clears B/Index but deliberately retains ACC epochs. |
| Buffer aliases | `potal.c:1905-1914`; header ownership comment | New views are globally refused while pending. Retained aliases must also remain untouched by the caller. |
| Release | `potal.c:1308-1332`, `potal.c:1837-1851` | A successful column Release invokes global drain reconciliation. It is not pair-local completion. |
| Failure | `potal.c:1040-1046`, `potal.c:1638-1643`, `potal.c:1817-1899` | First error is sticky. One recoverable cleanup drain is allowed; unknown Fence or failed cleanup quarantines the owner and pools. |
| Close | `potal.c:1931-1955` | Pending work, live B/Scale or unfinished ACC epochs refuse close. Safe cleaned close returns the original error. Quarantine cannot close/free. |

The C1 executor remains sequential. Its `load` issues, globally fences and
gets a token before consumers (`tests/support/e4c1-sequential.cpp:156-190`).
It releases/fences each consumer before subsequent work, retires each B panel,
and reads ACC after the K panels (`:365-445`).

`potal_paired_resident_query` copies software bookkeeping. It does not poll
hardware. The new focused test finishes the independent command interpreter's
internal load effects and proves the C11 session still reports pending and
refuses `get` until the checked Fence.

The existing 64-bit counter helper is private and reaches only the two frozen
open-time commands (`potal.c:1695-1728`, `:1783-1787`). No public resident call
reads LD/EX occupancy. A status-only sink or truncated `potal_status` result
must not be substituted for a counter value.

## 4. Verified RTL admission and completion boundaries

G is the actual 347,700-line generated B2 target. It was inspected with
bounded reads beyond the search tool's first-4-MiB coverage limit.
Its SHA256 is
`af3d40bae7ecd430af3c148621bbe04d73c8af13379515d5f92902e840a25290`.

| Boundary | Source / generated evidence | What is established |
|---|---|---|
| CPU command acceptance | `R/Controller.scala:216-226`; `G:284107` | The two-entry raw queue accepts the instruction. This is not DMA launch or completion. |
| Raw B MVIN admission | `R/LoopMatmul.scala:1087-1092,1150-1152`; `G:233628-233648,233868` | Ordinary commands, including raw MVIN/counters, cannot pass while either loop slot is configured or a loop response is pending. |
| Generator completion | `R/LoopMatmul.scala:203-227,1618-1641`; `G:229774-229780,233262-233273` | Final command emission clears generator ownership. RS/controller work may remain active. |
| Extended LOOP response | `R/LoopMatmul.scala:1227-1232,1336-1452`; `G:232606-232608,234948-234970` | The response reports validation/admission on command fire, not completed GEMM. |
| Physical addresses | `R/LocalAddr.scala:33-39`; `R/LoopMatmul.scala:1472-1496,1519-1527`; `G:133356-133406,233114-233147` | A/B banks are 0/1 for pair0 and 2/3 for pair1. B ownership must cover actual address ranges, not a software label. |
| Load/EX hazards | `R/ReservationStation.scala:248-298,337-400,489-545`; `G:182000-182033,182725-182764,183104-183112` | RAW/WAR dependencies use physical ranges. Indexed B reads conservatively cover the full source descriptor. |
| TL return | `R/DMA.scala:81-109,290-296`; `G:110922,114618,114700` | Final bus-beat acceptance is earlier than the final qualified SP-write path. |
| B SRAM completion | `R/Scratchpad.scala:419-439,523-550`; `G:136971-137029` | SRAM write enables/data and returned-byte accounting are tied to the accepted repeater output. |
| Whole MVIN retirement | `R/LoadController.scala:94-130`; `R/DMACommandTracker.scala:69-93`; `G:236961-236963,237039-237040` | ROB-tagged completion requires valid command and zero remaining bytes after returned SP writes. MVIN is not complete-on-issue. |
| Qualified compute completion | `R/PoTalWriteback.scala:253-325,343-347`; `G:167205,167270,168815-168816,170989-170997` | `pendingRob && rawDone && writeDone` gates completion after scaled ACC writes traverse the commit pipeline. |
| ACC read versus host store | `R/Scratchpad.scala:297-315,691-706`; `R/StoreController.scala:181-196`; `R/DMA.scala:390-401`; `G:136175-136176,116349-116355,117808` | Store command/row acceptance is earlier than acknowledged host-memory write completion. |
| CPU responses | `R/Controller.scala:947-982`; `G:285711-285718` | Counter, PoTal-ingress and LOOP status are exported. There is no pair/generation-tagged MVIN completion response. |

For a full K28/J9 B descriptor, pair0 base is4160 and pair1 base is12352,
with pitch144. An inactive pair must have no queued, RS or Execute/gather
reader, no live conflicting metadata, and no pending replacement.
Generator idle alone does not establish those conditions.

## 5. Existing counters: useful prefix certificates, not pair tickets

This distinction avoids unnecessarily declaring that every prefetch needs a
new bitstream.

`CounterExternal` defines live RS load/EX occupancy selectors1/3
(`R/CounterFile.scala:77-91`). The inputs are PopCounts of valid entries, not
cumulative byte counters (`R/ReservationStation.scala:570-576`). MVIN entries
remain valid until completion; only appropriate configuration commands can
complete on issue (`:419`, `:489-545`).

Counter commands reach the same ordered unrolled FIFO after LoopMatmul.
Its head goes either to `counters.io.in` or to RS allocation, never past an
unaccepted preceding command (`R/Controller.scala:308-309,1139-1184`).
`CounterFile.io.data` selects a live external value unless snapshot mode is
enabled (`R/CounterFile.scala:175-203`; `G:138144`). The response registers
that value on input fire (`:258-266`; `G:138795-138809`).

A new checked wrapper could therefore certify a *sealed admitted load
prefix* when all of these hold:

1. One issuer owns the whole command/configuration stream and both B pools.
2. The entire intended B descriptor has been issued; a zero before full
   descriptor admission is not completion.
3. Every relevant generated command precedes the query in the shared FIFO.
   Ordinary query admission occurs only after the configured generators clear.
4. No new load-producing LOOP/raw command enters the certified prefix while
   the query/publication is being resolved.
5. The wrapper owns the selected counters, explicitly configures the external
   selector, clears snapshot mode, discards the configuration response, and
   performs a later separate live read.
6. The sample is tied to the current owner, immutable resident generation,
   descriptor and sealed software issue cut. Old samples cannot authorize a
   replacement generation.
7. The result is LD_COUNT=0 with no unresolved error/unknown completion.
   The required CPU memory ordering occurs before source reuse/publication.

The certificate means **all loads in that prefix retired**, including old
Main A loads if present. It does not name B1. If B1 has finished but another
load remains, nonzero count cannot identify which one is pending. The model
has a dedicated test for this limitation.

An analogous sealed EX_COUNT=0 certificate can qualify the known complete
resident-GEMM prefix after PoTal's delayed ACC commit. It must include the
whole generated command prefix; a transient empty queue or solitary raw
PRELOAD is not sufficient. ST_COUNT=0 is not a host-store completion
certificate; readback retains global drain qualification.

`RDMA_BYTES_REC` increments on aggregate `tl.d.fire` and is neither pair
tagged nor a final-SP-write acknowledgement (`R/DMA.scala:314-321`).
Expected payload bytes must not be compared to this counter as a B-ready
test. Capability `0x504f544102100007` identifies the interface, not readiness.

This prefix protocol is the minimum *bounded existing-image candidate*.
Its checked implementation and focused RTL/counter regression are still
needed before a physical experiment. Arbitrary independent pair completion
requires a stronger tagged interface.

## 6. Per-pair state machine and authorized transitions

```text
FREE
  -> PREFETCH_ISSUED
  -> PREFETCH_COMPLETE
  -> RESIDENT
  -> READERS_IN_FLIGHT
  -> RESIDENT / RETIRE_SAFE
  -> FREE

Any live state -> QUARANTINED on unresolved completion or failed cleanup.
```

| Transition | Required authorizing event |
|---|---|
| FREE to PREFETCH_ISSUED | Checked descriptor/range validation and exclusive pair/source reservation before any possible effect. This premarked state includes incomplete admission. |
| Admission progress | Each required MVIN is accepted into the ordered execution prefix; configured generators may backpressure it. Track full descriptor issuance separately. |
| PREFETCH_ISSUED to PREFETCH_COMPLETE | Existing successful global drain covering the matching fully issued load, or the proposed qualified live LD-prefix-zero certificate. A future tagged pair ticket could replace the aggregate certificate. |
| PREFETCH_COMPLETE to RESIDENT | Explicit checked publication of the same owner/generation/descriptor after memory ordering. No producer may replace that identity. |
| RESIDENT to READERS_IN_FLIGHT | Validated Main/indexed-Residual submission with immutable B, Index/Scale bindings and the expected ACC epoch/contribution sequence. Ownership starts before effects, not at PE start. |
| READERS_IN_FLIGHT to RESIDENT | Qualified EX/ACC completion, not generator idle or raw EX done. Advance the matching ACC sequence exactly once. Future consumers and metadata may still retain B. |
| RESIDENT to RETIRE_SAFE | Every declared consumer has completed, no reader remains, and every resident-bound Scale/Index lifetime has retired under the preserved completion contract. |
| RETIRE_SAFE to FREE | Explicit retirement invalidates B/Index identity. ACC epochs remain independent. |
| Live to QUARANTINED | Uncertain DMA/Fence/Release completion or failed cleanup. Retain source, pair, ACC dependencies and shared issuer; no reuse, retry or close/free. |

A definite recoverable issue error uses a separate sticky failed-session
status, not a readable residency. One successful cleanup global drain may
discard failed identities and release known-safe resources. It never
publishes failed B. Cleaned close still returns the first error.

Because failure isolation between pairs is not established, uncertainty
poisons the shared issuer and all live pair/ACC ownership. An inactive pair
is not permission to recover one half of an unknown accelerator.

## 7. Lifetime invariants

- A pair has one immutable owner/generation/weight-version/physical descriptor.
  Generation counters do not reset on retirement or alias an old session.
- DMA source storage is held from preissue reservation through observed,
  matching completion. Physical completion hidden from software is not enough.
- Source reuse after completion is distinct from B-bank reuse. Poisoning a
  released host source must not change the resident SP payload.
- No load overwrites physical B while queued/RS/Execute/indexed readers own it.
  A dependency stall is not a substitute for a checked ownership lease.
- Completed Main does not retire B if a future Residual consumer is declared.
- ACC0/Main and ACC1/Residual epochs survive B retirement and next-pair
  publication. A new weight version, stale sequence or conflicting writer
  cannot continue an epoch.
- ACC readback admission is not a readable host result. Its destination and
  receipt remain owned until acknowledged stores and the checked global drain.
- A blocked Scale Release is an in-order barrier. Later CPU commands cannot
  bypass it; asynchronous hardware completion events may still occur.
- All success transitions preserve the single issuer. All uncertain transitions
  retain it. A second owner cannot make quarantined storage safe.

## 8. Global Release dependency timeline

Actual source (`R/Controller.scala:801-819`, `G:280903-280919,284573-284574`):

```text
allDrained =
  !LoopConv.busy && !LoopMatmul.busy &&
  !conv_cmd.valid && !loop_cmd.valid && !unrolled_cmd.valid &&
  RS.drained &&
  !LoadController.busy && !StoreController.busy &&
  !ExecuteController.busy && !Scratchpad.busy &&
  PoTalWriteback.drained
```

`RS.drained` means all entries invalid, not the weaker `RS.busy` indication
that treats a solitary PRELOAD specially. Valid Scale Release uses
`releaseSafe`; Fence uses `fenceDrained`
(`R/PoTalScaleCommandIngress.scala:468-470,516-524`,
`G:142537,142551-142559`).

The current C1 timeline cannot admit B1 while Main0/Residual0 is pending:
the C11 call returns BUSY before issuing anything.

The following is a *proposed late-prefetch* timeline, not an executed trace:

| Step | Pair0 Main/Indexed Residual and ACC | Pair1 B | Dependency |
|---|---|---|---|
| A | B0 is resident; Main0 runs and completes; its Scale Release succeeds. | No prefetch yet. | Existing checked boundaries. |
| B | Last selected Indexed Residual0 is admitted. Its generator emits commands; EX/SCU/ACC work can remain. | B1 source/identity is reserved. | A raw MVIN may queue upstream but cannot pass configured LOOP slots. |
| C | No new load-producing work is admitted after sealing the prefix. | B1 MVIN prefix enters RS after generator ownership clears. | Physical B-bank ranges must be disjoint. Queue acceptance is earlier than TL request launch. |
| D1 | Residual0 remains in flight. | DMA1 finishes first; a qualified LD-prefix-zero observation permits B1 completion/publication. | This is independent of EX completion only under the section5 prefix restrictions. |
| D2 | Alternatively, qualified EX completion occurs first. ACC sequence can advance. | DMA1 is still pending. | Release0 cannot succeed yet; source1 stays owned. |
| E | Scale Release0 joins all older work, including DMA1. | Any not-yet-observed completed B can be reconciled by the global certificate. | Do not bypass allDrained. |
| F | Retire B0 after all panel consumers and metadata finish. ACC epochs remain. | B1 is resident before its first reader. | B retirement does not reclaim ACC state. |
| G | Optional partial/final ACC readback is issued. Fence must qualify acknowledged host stores before receipt use. | No B reuse depends on that host result unless a consumer requires it. | Partial K28 output is not final K768 output. |
| H | B1 K20 consumers continue the same ACC epochs in canonical order. Final readback follows complete logical K. | B1 retires only after its own readers/metadata. | No spill/restore is added by this phase. |

If DMA1 is issued during Main0 before Main0's Release, that Release can stall
before Residual0 is admitted. Issuing during the final Residual avoids moving
that wait in front of the entire Residual phase, but the generator-to-ACC tail
may be too short to overlap useful data movement. No timing benefit follows
from source inspection.

Real opportunity exists only if generator emission finishes while old
execution still has useful work, and the inactive-pair load actually proceeds
then. If it does not, the sequence merely moves waiting into admission or
Release/Fence. DMA1 finishing after old compute does not by itself disprove
some earlier overlap; actual qualified intervals/activity are needed.

## 9. Observability and the physical experiment gate

| Requested observation | Current verified boundary | C2-0 action |
|---|---|---|
| B command admission | Raw queue fire; a later ordered counter acceptance can establish preceding RS allocation. Neither is TL DMA launch. | Static audit and software model only. |
| B DMA completion | Internal SP-write-qualified load retirement. Public resident API exposes only global reconciliation. Aggregate LD-prefix query is an unimplemented checked extension candidate. | No independent physical measurement. |
| Current compute completion | Delayed PoTal ACC-qualified ROB completion; proposed EX-prefix query can cover a sealed complete prefix. | No new hardware observation. |
| Scale Release/Fence acceptance | Global allDrained, including inactive-pair DMA and stores. | Preserved, never relaxed. |
| Next resident publication | Current C11 requires the single matching pending load and global drain. | No concurrent publication through current API. |

Therefore the minimal actual concurrent RoCC experiment is **not authorized
by the current checked interface**, and is not performed. No raw counter or
MVIN command is issued as a workaround. `--hardware` on the new executable
refuses with exit2 before any model/API activity.

Existing controller counters include same-cycle LD/EX busy events5/7
(`R/Controller.scala:984-1002`). These measure controller-busy overlap, not
DMA bus transfer or every PE MAC-active cycle.

Source also contains same-cycle physical SP-DMA-write/other-pair Mesh-input
monitoring (`R/Controller.scala:1004-1135`). Its report is once-only,
not transaction/generation tagged or an independent completion API.
Its capture availability on the preserved FPGA image was not established
in this phase. Do not substitute its source presence for a hardware result.

Future measurement must establish exact signal meanings and capture the
same-clock activity/interval evidence. CPU API elapsed time and issue order
are not substitutes. Load-prefix/EX occupancy overlap alone is not proof
of simultaneously active DMA and PE datapaths.

## 10. Software-only deterministic model and results

`tests/potal-e4c2-0-test.cpp` contains:

- A synthetic event world, typed per-pair references, source leases, separate
  ACC epochs and a shared issuer lease.
- The proposed ownership reducer with and without checked prefix observations.
  Its pending flags are ownership, not hardware timing.
- Independent assertions on state, source retention, poisoned-source/SP
  separation, epoch progress, failure status and issuer exclusion.
- A focused real-C11/software-transport gate using the existing B2 command
  interpreter. It does not issue RoCC instructions.

The completion reducer uses only a validated owner/cut-bound certificate and
its software pending records. It does not consult the synthetic hardware's
private DMA/EX identity or busy flags to decide completion. Independent
ownership assertions compare the resulting records against that oracle.
Old zero samples and foreign-owner zero samples are rejected with otherwise
ready conditions, so other guards cannot mask a missing scope check.

The abstraction uses two admission units and a 64-byte sample payload per B,
one Main and one indexed-Residual consumer per residency, and one representative
Scale column per reader. It is not a DIM16 numerical/traffic simulator and
does not define a batch-Release opcode. Only zero/nonzero occupancy matters.

Coverage includes both DMA/compute completion orders, delayed notifications,
generator backpressure, two pending load prefixes, snapshot/config/stale-cut
refusal, repeated pair reuse, unmasked stale-generation rejection,
premature readers/retirement, ACC continuation, failed first/partial
admission, one recoverable cleanup, uncertain completion and active-peer
quarantine. Later apparent completion cannot heal a poisoned issuer.

The aggregate-scope case deliberately refuses to certify completed B1 while
B0 still loads. This prevents the model from pretending an aggregate
counter is a pair-tagged event.

The current C11 gate separately proves:

1. Load pending rejects peer load without another command.
2. Internal interpreter completion does not change public pending state.
3. Compute pending rejects peer prefetch and retirement.
4. B can retire while its independent completed ACC epoch remains.
5. Readback pending rejects peer load and unreadable receipt access.
6. Checked global completion enables the receipt; the independent all-ones
   Main witness is exactly16 for every INT32 cell.
7. The unchanged sequential path closes safely and destroys its adopted
   software context.

Final focused run, `/tmp/h3-e4c2-0-certificate-focused.log`, exit0:

```text
E4C2_0_SOFTWARE_PASS scenarios=12 assertions=1608 expected_refusals=94 ownership_violations=0 hardware=0 overlap_measured=0
100% tests passed, 0 tests failed out of 1
E4C2_0_REFUSED hardware execution is unavailable; software-only contract test.
__C2_0_CERTIFICATE_FOCUSED_PASS__
```

The 94 expected protocol refusals are not ownership violations. Tests use
explicit deterministic events, not sleeps, timed polling or random ordering.
Earlier focused versions also passed. Final coverage includes unmasked
generation checks, aggregate-prefix ambiguity, active-peer quarantine and
certificate-only completion with independently checked oracle state.
No failing test was weakened.

Only the new API-only target was compiled/tested. LSP lacks the generated
include context for this new test file; the actual GNU12.2 C/C++ build passes.
The existing `POTAL_SCALE_ENTRIES` macro redefinition warning remains,
unchanged and unsuppressed.

## 11. Minimal next implementation proposal

For a bounded late-prefetch experiment on the existing image:

1. Add a separate opt-in checked C11 interface, not a raw issuing escape.
   Replace the single pending-kind assumption with per-pair load records and
   independently owned compute/readback records.
2. Preserve preissue source ownership, immutable descriptor/generation and
   complete-admission accounting. A successful CPU instruction return is
   never the completion certificate.
3. Add constrained 64-bit counter reads for reserved live LD/EX selectors.
   Bind every query to the current owner and sealed issue cut; explicitly
   control snapshot/configuration lifetime. Do not truncate into a status.
4. Permit only the bounded no-new-producer prefix protocol initially.
   Validate complete descriptor admission, generation, success/error state
   and CPU memory ordering before publication/source reuse.
5. Make successful global Release/Fence reconcile *all matching pending
   records*, not overwrite or clear just one `kind`. Keep global allDrained.
   A blocked barrier cannot be bypassed by later commands.
6. Retain one cleanup attempt, first-error stickiness and shared-issuer
   quarantine. Do not assume pair-local recovery or reclaim from a timeout.
7. Before hardware use, add focused counter/queue/RS retirement tests proving
   no early zero, no overtaking, snapshot rejection, delayed SP/ACC completion
   and failed/unknown completion handling. Then use a tiny checked experiment,
   not full Decode/FFN, with the established image/capability guard.

This is a proposal, not code added to `potal.c`. A new public prefix query
requires its own review and validation before the existing image is used.

For arbitrary independently completing loads, add a nonwrapping pair/command/
generation-qualified completion record after the last SP write, with defined
success/error semantics. Independently qualify compute after scaled ACC
commit. Add response/readout plumbing and capability identification.
Earlier MVIN admission while generators are configured also needs validated
arbitration and hazard/drain accounting. Those changes require RTL and a
new image; no such rebuild is approved or performed here.

## 12. Numerical risks and regression requirements

No arithmetic or packet code changes. Before any future checked implementation
is accepted, retain C1's independent Main/Residual INT32, reconstructed INT64
and bitwise F32 checks; exact packet hashes; K28+K20/J-tail; both pairs;
canonical compact fragments; nonzero ACC offsets; and fault/legacy tests.

Prefetch must not alter original-K identity, block32 carrier, Index fillers,
Scale generation, lane ordering, DIM16 saturation or ACC contribution order.
RAW/WAR safety alone does not prove those numerical properties. B publication
does not publish ACC output, and ACC read request acceptance does not complete
host readback.

The separate P2 opportunity is documented, not implemented:

- P1 full K28/J9 Main uses252 slot-program instances and4032 column writes;
  phase-aligned P2 metadata would use126 and2016, if its contract is proven.
- Equal carrier values do not prove equivalent P1/P2 execution. Prove original
  K phase/alignment, exact two-partial block identity, J/padding bindings,
  separate DIM16 dot/shift/saturation/accumulation and first/last behavior.
- Do not fuse two dots before saturation. Preserve witnesses MAX/MAX/MIN=-1,
  MAX/MIN/MAX=2147483646, compact0 versus dense-1 and carrier24=16777215.
- The current resident encoder/checked bindings are P1. A P2 fast path needs
  separate descriptor/encoder/consumer validation; metadata value caching
  cannot silently cross resident generations or original block/J identities.
- Begin with a bounded Main-only proof if pursued. Residual compact fragments
  are not interchangeable with dense block32 halves.

## 13. Performance interpretation

Historical C1 diagnostics-OFF n=7 medians, not new C2 measurements:

| Workload | Legacy Paired K28 | C1 sequential K28 |
|---|---:|---:|
| Decode | 13.189 ms | 23.697 ms |
| FFN | 79.623 ms | 122.927 ms |

C1's additional raw Scale commands are92.303%/92.429% of its additional
command counts, not latency percentages. Prefetch cannot remove that metadata
work. No speedup estimate is derived from CPU profile intervals or this
software model.

If a checked experiment later becomes safe, record separate command/RS
admission, qualified DMA completion, qualified compute completion, global
Release/Fence acceptance and publication observations. State whether evidence
is only in-flight/controller overlap or actual data-path activity. Keep trace
and detailed profiling out of any new primary n=7 measurement.

## 14. Preservation and regression boundary

The baseline inventory is `/tmp/h3-e4c2-0-ground-truth.log`.
All C1 code/report files remain unchanged; the C1 CMake block is unchanged.
The new cache is `build-potal-e4c2-0-host`; C1/B2 caches and guest evidence
are not reconfigured or rerun.

`/tmp/h3-e4c2-0-preservation.log` exits0: all13 protected hashes match
(six C1 source/report files, three existing executables, three benchmark logs
and the bitstream). Removing only the new C2 CMake block reproduces the
entire pre-C2 CMake file exactly. Tracked C11/matmul and RTL main/test sources
have no diff; both heads/branches and empty indexes are unchanged.
The unrelated host-init and dirty Gemmini diff hashes also retain their
earlier values. No rootfs/model rewrite was performed.

Retained evidence, not newly rerun suites:

| Evidence | Existing location |
|---|---|
| B2 actual raw42, checked warmup+n7, E4A before/after | `/tmp/h3-e4c1-baseline-results/` |
| C1 five exact numerical gates, K28+K20/J-tail | `/tmp/h3-e4c1-gates-results/` |
| Four-mode exact integers and bitwise output, primary/profile/raw trace | `/tmp/h3-e4c1-benchmark-results/` |
| C1 deterministic failure/quarantine and native/reference checks | `/tmp/h3-e4c1-native-final.log` |
| Existing scoped RTL/legacy checks | `/tmp/h3-e4c1-rtl-final.log` |

The known legacy smoke status6/1 assertion and two stale Scala constructors
are unchanged. An incomplete optional RTL sweep is not promoted to an
FPGA result. Physical DMA fault injection remains unperformed.

Reproduction from L after sourcing conda, Chipyard and FireSim environments:

```bash
cmake -S . -B build-potal-e4c2-0-host \
  -DCMAKE_BUILD_TYPE=Release -DPOTAL_API_ONLY=ON \
  -DPOTAL_HW_PARAMS=/home/alveo/firesim/target-design/chipyard/generators/potal/software/gemmini-rocc-tests/include/gemmini_potal_params.h \
  -DPOTAL_EXPECTED_DIM=16 -DPOTAL_E4B2_TESTS=OFF -DPOTAL_E4C2_0_TESTS=ON
cmake --build build-potal-e4c2-0-host -j4 --target potal-e4c2-0-test
ctest --test-dir build-potal-e4c2-0-host -V -R '^potal-e4c2-0-test$'
```

## 15. Stop decision

The focused software test and preservation checks pass. This feasibility
checkpoint is complete; it does not authorize production adoption.

**As-is C11 plus existing FPGA: do not proceed with concurrency.**
**Checked sealed-prefix C11 extension plus existing FPGA: bounded late
prefetch is a justified next validation candidate, without an automatic
rebuild requirement.** It is not yet hardware verified.
**General tagged per-pair completion or earlier generator-time admission:
RTL/interface modification and a separately approved image are required.**

No unchecked RoCC sequence, physical overlap claim, production replacement,
Scale/Fence/Release relaxation, premature reclamation, commit or push is
part of C2-0.
