# H3-6E4A: Resident-B Micro-tile Interface

Date: 2026-10-26. Fixed DIM16/K28. E4A changes remain uncommitted.

## 1. Executive Summary

E4A adds an isolated checked resident interface without changing production
Attention, Paired scheduling, C11 semantics, RTL or the FPGA image.
The supported execution subset is end-aligned K28/J2 residency, native Main
K2/J2 suffix into ACC0, and one indexed DIM16 Residual fragment into ACC1.
It is not arbitrary interior Main addressing or a full pipelined schedule.

Native numerical, ownership and fault regressions pass. The actual FireSim
trace run and seven subsequent profiled repetitions also pass. Both physical
pairs, generated SCU, separate ACC results, saturation and ordered continuation
through another B residency are verified for this subset.

**Bounded bring-up: A. General arbitrary resident microtiles: C.**
No full pipeline or general interior Main hardware interface is declared A.

## 2. H3-7B5 Checkpoint SHA

The authorized checkpoint is:

```text
d27563ba6041ac8234f2eac164bae841c425c7d7
feat(potal): implement H3-7B5 QK host INT64 limb reconstruction
```

Its parent is `dc257bcbb707913edb0af5a1d623970276051528`; the earlier H3-7B
checkpoint is `3e373948d44d0eccbc7c52fd52f65750da703314`.
Only the twelve B5 source/test/report paths were explicitly staged.
Cached whitespace, names and statistics passed: 1481 insertions, 131 deletions.
The index was empty afterward; only `models/gpt2/` remained untracked.
No push, new branch, model, rootfs, binary or unrelated change entered the commit.
The B5 report's uncommitted statements describe its historical handoff.

## 3. Existing Hardware/Software Interface

The selected generated parameter header retains SHA256
`b4cd8b9efef9f3cd9b598ae3cc84af6478f83e628ed73d94a77fb1c452df1fc7`.

```text
DIM16; signed8 A/B; signed32 ACC
4 SP banks; 4096 rows/bank; 16 bytes/SP row
B bank64 KiB; ACC1024 rows, 512 rows/path
Index32/pair; shared Scale256; Work IDs64/loop
Paired Main K28 tiles =448 elements
```

For I1/J9/K28, B uses 4032 rows and Main Scale uses 126 slots. K48/J9
requires 6912 B rows and is outside this unchanged geometry. Each ACC half
holds 32 DIM16 output tiles, a stricter limit than 64 Work IDs.

The active `paired-auto.cpp` route issues whole Main then Indexed Residual.
`paired-schedule.cpp`/`paired-emulator.cpp` implement a separate functional
microtile schedule whose hardware route refuses execution. Neither planner
nor schedule is replaced here.

Software paths below are relative to `ggml/src/ggml-potal/`; RTL filenames
refer to Chipyard `generators/potal/src/main/scala/potal/`.
Source audits: `/tmp/h3-e4a-{c11,rtl,software}-audit.md` and
`/tmp/h3-e4a-ownership-contract.md`.

## 4. Resident-B Descriptor Contract

`tests/potal-resident.h` defines:

- Pair and explicit weight bank identity.
- Physical B base and packed K-tile row stride.
- Original K/J origins and valid extents.
- Logical weight identity/generation.
- A nonwrapping, handle-scoped resident incarnation.
- Main A reservation, load/readability state and pending consumers.
- Completion-qualified retirement, separate from ACC lifetime.

For pair p, DIM D=16 and bank rows R=4096:

```text
B bank =2*p+1
bank_end =(2*p+2)*R
J =ceil(j_columns/D)
K =ceil(k_rows/D)
B_base =bank_end-K*J*D
B_row_stride =J*D
B_SP(kt,jt,row) =B_base+kt*B_row_stride+jt*D+row
```

The wrapper checks exact bank/base/stride, K<=28 tiles, valid source strides,
owned allocation bounds, overflow and bank capacity before issuing anything.
Each handle owns six aligned 64 KiB DMA pools: two B, Main/Residual A and two
private readback destinations. Caller aliases cannot free or touch them while
DMA is pending. Quarantine retains them and the adopted heap sink context.

A token is unavailable at load return. Successful Fence or qualifying Release
publishes only an entirely successful matching load. Partial failed loads are
invalidated after recovery, never published. Retirement refuses pending readers
and associated live Scale columns. Replacement receives a new incarnation.

## 5. Micro-tile Addressing and SCU Binding

Native Execute recomputes B from its own geometry, even when B DRAM is zero:

```text
B_native_start =bank_end-K_local*J_local*16
B_native(k,j) =B_native_start+(k*J_local+j)*16
```

With unchanged J stride, this selects a K suffix. Pair 0 K28/J2 has base 7296;
native K2/J2 starts 8128 and consumes original rows 416..447.
Pair 1 translates SP addresses by 8192. Source:
`LoopMatmul.scala:393-446,1371-1384`.

E4 constructs a real ordinary checked Main descriptor for the retained source
suffix. After pure checked encoding and independent SP-layout equivalence
checks, it changes only `program.addrs_ab.rs2` to zero. It retains CONFIG_EX,
pair selectors, Scale metadata and actual A DMA. It never calls ordinary
Main issue and then repairs its state: that would reload B.

Full resident K/J geometry remains unchanged after the small consumer.
Indexed Residual configures that full descriptor, not the Main K2 view.
Index entries are checked original-K positions, encoded panel-relative, and
bound to an Index incarnation and resident token.

Local Scale block 0 for Main binds original block 13. Residual binds the source
block of its compact DIM partial. All required columns, original block/J
identity and live hardware generation must match. Carrier 0 means identity;
`POTAL_CARRIER_ZERO` is distinct.

The execution API currently admits full I1/J2, Main K2/P2 and Residual K1/P1
with 16 strictly increasing indices in one original K32 block. General load
geometry is wider than this execution subset. Other valid execution shapes
return UNSUPPORTED; malformed fields still receive the existing argument checks.
Arbitrary interior native K, general J subviews and hardware execution tails
are not claimed verified by this subset.

Raw PRELOAD/COMPUTE starts with zero PoTal metadata
(`Controller.scala:215-222`). E4 uses generated metadata-bearing LOOP_WS,
not raw unscaled arithmetic or Host correction.

## 6. Main/Residual ACC Epoch

Paired Main targets ACC0 base 0; Indexed Residual targets ACC1 base 512.
Their bank-local output base is 0. This selection follows
`LoopMatmul.scala:1382-1384`, not the SP pair number.

An output epoch binds path/bank, output identity, fixed I/J shape, original
J origin and weight identity/generation. First initializes; a continued epoch
requires its matching token, next sequence and ACCUMULATE. Main requires the
next contiguous original-K interval; Residual preserves ordered fragment
sequence with allowed gaps. First cannot discard an unfinished epoch.
Last closes it; another accumulation into that closed epoch is rejected.

Completed ACC values survive retirement of their consumed B residency.
They contain values, not live references to the old SP bytes. Continuation
through another residency still requires the same matrix and J identity.
All actual writer/readback DMA must complete before bank reuse.

RTL Work IDs belong to each invocation and scheduler slot, not this software
output epoch. E4 drains every invocation before reuse. Every invocation's
last K seals its hardware work, even when its logical output epoch continues.

The unchanged E3 raw ACC save/restore and bias-continuation proof is reused
as historical evidence. E4 adds no restore API and does not claim newly verified
metadata-bearing GEMM continuation after a raw restore.

## 7. Checked B-only Load and Ownership

`tests/potal-resident.c` includes unchanged `potal.c` once, behind an owning
opaque handle. No core pointer or raw issuing callback is exposed.
The executable links that C11 object instead of another `potal-api` object.
The driver object references only E4 operations and the pure status formatter;
issuer startup occurs before E4 acquires its core.

This is a closed, single-issuer harness. Ordinary Scale APIs and raw RoCC
can bypass `paired_owner`; they are explicitly forbidden during an E4 lifetime.
An opaque pointer alone is not protection from arbitrary C code, another
process or a foreign accelerator user.

Raw B load uses existing CONFIG_LD state 0 and MVIN funct 2:

```text
CONFIG_LD rs1 =0x00100101
CONFIG_LD rs2 =source byte stride
MVIN source =B+kt*16*source_stride+jt*16
MVIN destination =B_base+(kt*J+jt)*16
MVIN rows/columns =valid DIM tile extents
```

The handle records pending state before CONFIG and increments attempted MVIN
count before every issue. The core has no independent B counter; zero core
counters therefore never establish safe close or publication.

Normal completion uses checked Fence or successful checked Release, followed
by CPU ordering before exposing data. An issue failure permits one explicit
cleanup Fence even when core counters are zero. Successful drain invalidates
failed results, then core cleanup releases tracked Scale state.

An independent unknown-Fence bit records failure even when core sticky stage
still names the earlier MVIN/LOOP error. Unknown Fence, uncertain Scale response
or failed cleanup quarantines the handle. No retry, close, free, flush or later
issue follows. Safe cleaned close consumes the handle while returning its
original sticky error.

## 8. Scale/Index Lifetime

Physical Scale memory is shared:

```text
Current Main126 + Residual18 + Next Main126 =270 >256
```

The earlier logical 252-slot overlay uses a different carrier-sharing model.
It does not solve the active 270-slot encoding. The new tests separately
exercise capacity refusal; repeated block bindings in its synthetic capacity
witness are not represented as a real J9 execution.

B loading requires no next Scale programming. Two residents can coexist with
independent weight generations, while Scale is allocated only when needed.
However, this E4 wrapper conservatively allows one DMA-bearing transaction
at a time. It establishes lifetime separation, not overlapped prefetch.

Current compute completion, next B issue/completion, Release, next Scale setup
and pair switch have explicit boundaries. Release uses global allDrained,
so an older queued next-B DMA can delay current Scale retirement. Raw loads
also cannot bypass configured loops. Early prefetch can serialize execution;
no benefit is inferred from its issue order.

Index remains 32 entries/pair. Configuration/writes are prevalidated as one
transaction; publication requires every ordered write. The executable consumes
16 entries per supported Residual microtile. Stale weight/Index tokens and
reprogramming while a consumer is pending are rejected.

## 9. Existing ISA Capabilities

Source-supported capabilities:

- Raw B-only load to a free paired B bank, with checked completion.
- Native B=NULL consumption when recomputed layout exactly matches residency.
- Original-K Index gather into ACC1 with generated SCU metadata.
- Independent first/accumulate behavior for the fixed ACC0/ACC1 banks.
- Raw INT32 readback, Fence and completion-qualified Scale Release.
- Existing standalone full-width ACC restore, as previously verified in E3.

Unsupported in this E4 execution API does not automatically mean absent in RTL.
For example, other native suffix shapes can be source-expressible but remain
outside the verified narrow wrapper.

## 10. Required RTL/ISA Extensions

General interior microtiles need effective B start and resident J stride,
independent of local K/J loop shape. Narrow I/J outputs also need independent
ACC destination offsets so the output epoch remains correctly placed.

Index enable must be separable from Residual/ACC1 selection to target ACC0.
Scale addressing needs its original J stride/origin and initial K phase,
or explicit per-partial rebinding through a proven lower-level interface.
Non-draining persistent output work needs stable Work identity and explicit
first/last lifecycle protection.

A raw tile alternative must atomically bind Scale slot/generation, output
shape/destination, Work identity and indexed B context to its PRELOAD.
Guaranteed raw-load admission during a configured loop is a separate scheduling
change. No opcode number, RTL change or FPGA rebuild is invented here.

## 11. Implementation Changes

- `ggml/src/ggml-potal/tests/potal-resident.h`: test-only descriptors, tokens,
  status, owned-buffer access and checked operations.
- `ggml/src/ggml-potal/tests/potal-resident.c`: existing-core mediation,
  B-only loading, matching resident consumption, lifetime validation and quarantine.
- `ggml/src/ggml-potal/tests/potal-resident-test.cpp`: independent packed-SP
  command emulator, exact fixtures, negative/fault tests and actual RoCC driver.
- `ggml/src/ggml-potal/tests/potal-resident-model-test.cpp`: existing Serial,
  old microtile and active H2 integer comparisons without planner changes.
- `ggml/src/ggml-potal/CMakeLists.txt`: default-OFF `POTAL_E4A_TESTS` targets.
- This report.

Production source bodies and all B5 numerical/Attention behavior remain unchanged.
The hardware target reuses the existing issuer on Linux CPU1/hart0.
The driver never invokes its production planned/paired runner while E4 owns
the core. Backend dependencies do not authorize an alternate command path.

## 12. Host Regression

The requested nine regressions passed:
`potal-api-test`, `potal-trace-test`, `potal-residual-unit-test`,
`potal-paired-plan-test`, `potal-paired-schedule-test`, `potal-paired-unit-test`,
`potal-paired-auto-test`, `potal-paired-execution-test` and
`potal-stripe-pipeline-test`.
Preserved `potal-attention-test` and the new model executable passed. The resident
ownership executable also passed after correcting a malformed negative fixture.

New negatives cover invalid pair/bank/base/stride/extent, K/J/tail mismatch,
wrong generations and SCU identity, Scale capacity/missing columns, Index
mapping/incarnation/order, epoch conflicts, pending overwrite/retirement and
unpublished readback. Eight-bit Scale reuse is distinct from checked
nonwrapping resident/owner generation arithmetic.

Fork-isolated software faults cover first/middle/final B MVIN, all ten positions
of Main/Residual command issue, both MVOUTs, partial Index writes, failed
prepare Fence, a cleanup Fence failure hidden behind an earlier sticky error,
uncertain Scale responses, failed Release and cleanup Release failure.
They assert retained allocations/owner, no retry, exact one-drain recovery and
safe error-close/reopen. No fault case is claimed actual hardware injection.

Logs: `/tmp/h3-e4a-results/potal-*.log`.
Native and DIM16 builds pass. LSP still lacks the target include configuration;
the existing generated `POTAL_SCALE_ENTRIES` warning remains unsuppressed.

Reproduction uses separate `build-potal-e4a-host` and `build-potal-e4a-d16`
directories with `POTAL_E4A_TESTS=ON`. The hardware target uses the unchanged
`cmake/riscv.cmake` toolchain and generated DIM16 parameters.

```text
build-potal-e4a-host/bin/potal-resident-model-test
build-potal-e4a-host/bin/potal-resident-test
# On the DIM16 guest, through the executable's issuer_owner:
POTAL_RAW_TRACE=1 ./bin/potal-resident-test --hardware --repeat 1
./bin/potal-resident-test --hardware --repeat 7
```

## 13. Actual FireSim Bring-up

Actual DIM16 RoCC bring-up passed. Binary SHA256:
`b0c2fec76cb8e002b86d8d4b7f9d354a559ee1b0c1b896baa31e1fdfd8aedfb2`.

The isolated guest script is `/tmp/h3-e4a-guest.sh`, using a copied rootfs.
It gated a raw-trace hardware run and seven profiled repetitions after the
target software-only regression. Script and simulator exited 0. No Full
Inference request was repeated. The simulator's 740.6-second host wall interval
and fastest-target-clock summary are not primitive execution measurements.

Observed exact outputs, across every valid row/column:

| Fixture | Main J0 / J1 | Residual J0 / J1 |
|---|---:|---:|
| Initial K28/J2 residency | 1760 / 3648 | 960 / 136 |
| Carrier 14 saturation | -1 / -1 | INT32_MIN / INT32_MIN |
| Ordered continuation across a new residency | 3648 / 7552 | 2688 / 368 |

The second Main contribution is 1888/3904, added to the existing ACC0.
Its different original-K data in the other pair also distinguishes a wrong
B-bank selection from the correct one. Host B is poisoned only after load
completion; the correct nonzero outputs therefore cannot come from reloading
that poisoned source. ACC0 remains exact after ACC1 execution/readback.

The separate trace records 2074 matched ENTER/EXIT pairs in three owned
sessions, every returned status 0:

| Raw command | Count |
|---|---:|
| CONFIG | 88 |
| B-only MVIN | 504 |
| LOOP_WS | 16: Main 10, Residual 6 |
| LOOP_WS_CONFIG, each of five kinds | 16 |
| Full-width ACC MVOUT | 30 |
| SCALE_WRITE / SCALE_RELEASE | 608 / 608 |
| FENCE | 35 |
| INDEX_CONFIG / INDEX_WRITE | 6 / 96 |
| FLUSH | 3 |

All sixteen AB command records have B DRAM address 0. All nine resident loads
have 56 DIM16 MVINs, with every source/destination/stride checked against the
packed layout. Only B banks 1/3 are targeted. Main uses passes_per_block 2;
Residual uses 1; every LOOP generation is 255. Two LOOPs carry ACCUMULATE.
Generation 0 programming/release after 255 is also exercised, separately from
nonwrapping software token identities.

There is no hidden B-bearing Main LOOP, raw PRELOAD/COMPUTE workaround, or CPU
dot-product fallback. The trace does not expose generated internal DMA bursts.
Command-declared payload is not physical bus-byte measurement.

Hardware preflight refusals and healthy retirement/close were exercised.
Hardware fault injection, arbitrary interior Main/J selection, execution
tails, other host strides and resumed GEMM after raw ACC restore remain
**Not verified**. The broader B-load geometry and injected failures have native
software evidence only; their status is not promoted by the narrow hardware run.

Evidence: `/tmp/h3-e4a-results/{hardware-trace,hardware-profile,software-target}.log`,
`/tmp/h3-e4a-run/uartlog`, and
`/tmp/h3-e4a-analysis/hardware-evidence.json`.

## 14. Numerical Exactness

Six M17/N145/K928 software fixtures retain microtile/active I1/J9/K28:
zero, sparse, dense three-lane and three saturation adversaries.
Serial retains its existing automatic tiling. Raw Main, Residual/lane identity
and reconstructed INT64 agree with independent wide arithmetic.
Packet decoding is shared; arithmetic and manual witness constants are not.

```text
A: Sat32(Sat32(MAX+MAX)+MIN) =-1
   reordered MAX,MIN,MAX =2147483646
B: compact dot(+1-1) at carrier31 =0
   original-DIM zero expansion gives Sat32(MAX+MIN) =-1
C: Sat32(2032*2^24) =MAX
   Sat32(-127*2^24) =-2130706432
   accumulated result =16777215, not merged-partial MAX
```

The new packed-SP emulator independently decodes actual command fields and
computes from its loaded SP bytes, not the wrapper's addresses or arithmetic.
Source poisoning after load completion checks no B reload. All model evidence
remains software-only. H3-7B5's separate stationary-limb contract is untouched.

## 15. Performance and Unmeasured Costs

Raw tracing is OFF for the following seven samples. API timers remain enabled:
these are instrumented CPU envelopes, not isolated FPGA waiting or DMA time.
Each sample runs the same bundle of ordinary, saturation and continuation
fixtures. Values below are sums of the named sequential API intervals within
one bundle, then median / unscaled MAD across seven bundles.

| Scope | Calls/bundle | Median / MAD ms |
|---|---:|---:|
| B-only load issue | 9 | 0.056 / 0 |
| B completion Fence | 9 | 0.007 / 0.001 |
| Resident token handoff | 9 | 0.005 / 0 |
| Microtile issue | 16 | 0.046 / 0.001 |
| ACC completion Fence | 12 | 0.010 / 0.001 |
| Scale window programming | 19 | 0.095 / 0.001 |
| Index window programming | 6 | 0.026 / 0 |
| ACC readback issue | 15 | 0.020 / 0.001 |
| Readback completion Fence | 11 | 0.010 / 0.001 |
| Residency retirement | 9 | 0.051 / 0.001 |
| Individual Scale Release | 608 | 0.468 / 0.007 |

Four compute and four readback completions use successful Release instead of
an extra explicit Fence. Their intervals belong to the Release row, not both
completion rows. The displayed values are not single-call latency medians.
Observed timer values are quantized to 0.001 ms; zero MAD is not zero cost.

Full checked-test bundle median/MAD is **8.847 / 0.031 ms**, including
allocation/mlock, preparation, comparisons, validation and instrumentation:

```text
10.498 8.915 8.868 8.847 8.816 8.833 8.783 ms
```

All seven samples are retained; none is manufactured from a single capture.
The separate raw-trace bundle took 84.725 ms and is excluded from these
statistics. Its logging changes issue pacing. Do not subtract timings between
these modes to infer a pure hardware wait or baseline component cost.

Physical DMA bytes/time, PE overlap and complete pipeline speedup are
**Not measured**. New ACC restore timing is not collected; the E3 historical
checked context test remains a separate experiment.

Historical diagnostics-OFF reference only:
Decode Serial K28 8.810 ms versus Paired 14.291 ms;
FFN Serial K28 52.684 ms versus Paired 78.363 ms.
Those are not E4A achieved timings or a matched comparison with this small
suffix fixture.

## 16. H3-6E4B Implementation Recommendation

Feature verdicts:

| Feature | Current status |
|---|---|
| Checked K28/J2 B-only load/publication/retirement, both pairs | A; actual RoCC subset verified |
| Matching native Main suffix plus Indexed Residual, SCU and ACC0/ACC1 | A; exact hardware results, no B reload |
| Ordered accumulation across completed residencies | A; actual Main and Residual continuation verified |
| Broader B-load geometry and unsupported execution shapes | B; native validation, hardware outside tested subset |
| General native interior K/J selection | C; independent addressing metadata required |
| Indexed selection targeting ACC0 | C; selection/destination coupling remains |
| Resource, epoch and stale-token refusal checks | B; native checks, selected hardware preflight checks |
| Injected failure/unknown-completion quarantine | B; software-only fault evidence |
| Raw ACC restore primitive | Historical A from E3, not newly integrated GEMM continuation |
| Concurrent 270 live Scale slots | D under Scale256 |
| K28/J10 single-bank residency | D under 64 KiB B bank |
| Guaranteed early prefetch and physical overlap | Not verified / Not measured; no pipeline claim |

Do not replace production Paired scheduling. E4B can build on a verified
bounded resident primitive only after retaining the exact address/SCU/epoch
contract. General tile coverage requires the section 10 fields; overlapped
ownership needs more than the current single-pending-transaction guard.
Keep first-error retention, unknown-completion quarantine and exact saturating
order. Hardware fault recovery, arbitrary caller/raw interference and process
death are not solved by this test-local API.

Final preservation check: all 50 protected source/artifact hashes match.
All tracked production/B5 files match the checkpoint; the only tracked
working-tree change is the default-OFF E4A CMake block. The 67-file PoTal Scala
manifest, Gemmini HEAD/pre-existing diff, host-init, models and frozen artifacts
remain unchanged. E4A's four new test/interface files and this report remain
uncommitted, with an empty index. Branch `potal` remains at
`d27563ba6041ac8234f2eac164bae841c425c7d7`. No push or E4B implementation occurred.
