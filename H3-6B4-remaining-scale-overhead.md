# H3-6B4 - Remaining Paired Scale Overhead Attribution

Date: 2026-10-25. Repository: ajou-aisa/llama.cpp-gemmini. Branch: `potal`.
Baseline: retained H3-6B2 + H3-6C. Hardware: unchanged H3-5B FireSim U280 DIM16.

## Decision

**Choose B: investigate Paired planning/validation next. End the H3-6B software
Scale micro-optimization series on the present evidence.**

The unloaded Paired penalty is reproducible with the unchanged executable.
The guest supports both `rdcycle` and `rdinstret`. Separate counter measurements
show substantially more retired work in the Paired loops than in Serial,
consistent with the checked session bookkeeping visible in disassembly.
They do not isolate a safely removable cycle budget.

Repeated idle Release clearing executes three zeroing stores, not six stores.
Those stores are logically redundant when every affected counter is already
zero. Their separate latency has not been measured. An all-zero guard needs
loads and branches, and may cost more than the stores. The observable drain
counter must still increment. No such change was implemented.

A is not justified by a demonstrated net reduction in executed work. B has
measured planning/preparation intervals and visible repeat validation/copies,
but its removable fraction still needs attribution. C is a larger structural
investigation, not a proved software micro-optimization or permission to change
Scale/Index ordering or synchronization in this phase.

## 1. Baseline preservation and scope

No repository production, API, benchmark or numerical source was edited.
The existing seven-repeat benchmark remained byte-for-byte unchanged. New
measurement code was confined to `/tmp/h3-6b4/scale-path-counters.c`, a diagnostic
copy of that benchmark, and was linked against the existing DIM16
`libpotal-api.a` and clock-bridge object. No production library rebuild or
CMake reconfiguration was needed. The only new repository artifact is this report.

The full fingerprint of matching C/C++/header/CMake files under
`ggml/src/ggml-potal` matched before and after collection:
`eaaa28e23c227dba362ab91dd95e3a03d9c1059c50b6645c91158bcb80198688`.
The fingerprint hashes the sorted per-file SHA256 list, including existing
local diagnostics and tests. The individual retained identities also matched:

| Artifact | Unchanged SHA256 |
|---|---|
| `potal.c` | 7dec7c2a9cfc0bc12cb3dcdb9abd93302c0dc69e813a812f6d332c87e16850bd |
| Original Scale benchmark C source | 0d1e6e773865e7ff3fb0c7f00a59b1895585259ea65546f425623e8259ff6f95 |
| Original DIM16 Scale benchmark executable | 1f13bd8aa8a12fbc59e45625be32b900904a6336727555c09fba18656c96c758 |
| H3-6C smoke executable | c694e07a3349929f6b3568cb3d6cc3867fbbd4323bed619482ca3f3a10c79c43 |
| H3-6C `libggml-potal.so` | 1aca66631382f52856d4a187822e456f03a7522a9c828c968811619377c240b6 |
| B2 report | 6dc70672d6b4356c61e85bb00898ddfc1387d5f19169172c033385904e5239db |
| C report | 421d65520bcc412b9c3323e195f52d67614901613bfa82e6d59ccd006eb97e15 |

B3's rejected Sink/Trace-helper experiment was not repeated. Its smaller
reserved frames left executed hardware paths at 62/69/64 instructions and
retained the same save/restore work. It remains a rejected generated-code
experiment, not a measured speedup or slowdown.

## 2. Measurement method and targeted instrumentation

The unchanged benchmark ran first:

```sh
./potal-scale-path-bench-b4-baseline --cycles
```

The separate diagnostic copy then ran:

```sh
./potal-scale-path-counters-b4 --cycles --instret
```

Both use the original stream: 36 batches, 256 slots, 16 columns, carrier 0,
generations 1 through 36, ascending Write then Release order. Each phase has
**147,456 commands per repetition**. Each route has one full untimed warmup
and seven measured repetitions with rotating route order. The routes are
checked Serial hardware, checked Paired hardware, and checked Paired with a
success-only software Sink. The Sink issues no hardware commands.

Both programs verified physical hart 0 maps to Linux CPU 1 and pinned there.
`POTAL_RAW_TRACE` and detailed profiling/diagnostic controls were unset.
There was no per-command printing and no Main, Residual or readback work.
The original initial hardware Fence remained outside timing. All returned
statuses, per-batch exact command totals, zero live Scale slots, zero
outstanding loops/readbacks, failure state and session close were checked.

The diagnostic copy adds two `rdinstret` samples around each batch-phase
command loop, alongside the existing cycle samples. Its boundaries are:

```text
steady-clock start; rdcycle start; rdinstret start
    original command loop, returned-status checks and loop control
rdinstret end; rdcycle end; steady-clock end
```

An untimed SIGILL-safe probe checks that `instret` advances across 64 NOPs.
A separate `--probe-self-test` raises and catches SIGILL and restores the
previous handler. Host and guest self-tests passed. The non-RISC-V Host
software-only run correctly reported `instret_supported=0` and passed all
seven repetitions. The real guest reported **`instret_supported=1`** and
completed all 42 diagnostic route/phase samples.

The counter copy was built with the existing RISC-V GCC/G++ 12.2.0 tools,
Release `-O3 -DNDEBUG -std=c11` for its C caller, the configured DIM16 headers,
and the retained API archive/clock object. Its executable SHA256 is
`caf9d8486b5630cbfb90c6f522b30940b1e567b44dcd6c82f867b89c436cf842`.
Staged and guest binary hashes matched.

RISC-V counter reference: RISC-V International, *The RISC-V Instruction Set
Manual, Volume I*, chapter on the Zicntr/Zihpm counter extensions, version 2.0.
On RV64 these reads access the full counters; availability is environment
dependent, hence the actual guest probe rather than an assumption.

### What the counters mean

Retired instructions and cycles below cover the **whole batch loop**, not
only the C11 API body. They include calls/argument setup, status checks,
loop control and counter-sampling overhead. No privilege filtering or
interrupt-work subtraction was configured or established. Counts can include
OS work on the hart. Timer/counter overhead was not subtracted.

Cycles are architectural guest CPU cycles, **not FPGA accelerator cycles**.
Neither elapsed time nor `rdcycle` separates CPU execution, cache/pipeline
stalls, command admission and hardware waiting. `rdinstret` distinguishes
retired work from elapsed cycles but does not assign cycles to individual
validation or state-update instructions.

## 3. Fresh unchanged-executable latency control

All values are CPU-observed ns/command except the last column. MAD is median
absolute deviation over seven repetitions. The B2 columns are the preserved
historical medians, not replaced by the new run.

| Route | Phase | B2 median ns | B4 control median ns | MAD ns | Min ns | Max ns | Median guest cycles/command |
|---|---|---:|---:|---:|---:|---:|---:|
| Serial hardware | write | 37.835 | 37.910 | 0.061 | 37.808 | 37.984 | 37.773 |
| Serial hardware | release | 32.905 | 32.823 | 0.014 | 32.789 | 32.905 | 32.684 |
| Paired hardware | write | 87.029 | 86.968 | 0.034 | 86.873 | 87.104 | 86.821 |
| Paired hardware | release | 82.147 | 82.147 | 0.014 | 82.113 | 82.303 | 81.993 |
| Paired software-only Sink | write | 91.614 | 91.736 | 0.149 | 91.532 | 91.939 | 91.560 |
| Paired software-only Sink | release | 86.948 | 86.846 | 0.142 | 86.704 | 87.111 | 86.730 |

The fresh Paired-minus-Serial gaps are 49.059 ns/Write and 49.323 ns/Release.
Paired Release reproduces the historical median to the printed precision.
There is no outstanding GEMM/readback whose completion could explain this
unloaded gap. The software-only Sink remains at least as costly as hardware
in this control, supporting substantial software overhead. Its different
callback path prevents treating Sink-minus-hardware as a measured hardware
wait budget.

## 4. Separately instrumented retired instructions and cycles

The table uses seven-repeat medians and dispersion from the diagnostic copy.
The final column is shown to expose measurement/layout perturbation; it is
not a production optimization result.

| Route | Phase | Retired instructions/command | MAD instructions | Min instructions | Max instructions | Guest cycles/command | MAD cycles | Diagnostic ns/command |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Serial hardware | write | 31.235 | 0.045 | 31.190 | 31.281 | 36.692 | 0.092 | 36.831 |
| Serial hardware | release | 26.281 | 0.068 | 26.190 | 26.349 | 31.755 | 0.176 | 31.901 |
| Paired hardware | write | 78.014 | 0.000 | 77.969 | 78.083 | 85.431 | 0.021 | 85.578 |
| Paired hardware | release | 72.390 | 0.009 | 72.344 | 72.435 | 82.604 | 0.023 | 82.777 |
| Paired software-only Sink | write | 85.014 | 0.045 | 84.969 | 85.118 | 88.918 | 0.196 | 89.084 |
| Paired software-only Sink | release | 79.390 | 0.000 | 79.390 | 79.435 | 83.679 | 0.009 | 83.828 |

The measured whole-loop Paired-minus-Serial retired-work differences are
46.779 instructions/Write and 46.109 instructions/Release. Sink medians add
seven retired instructions per command relative to hardware in this diagnostic
run. These are differences of route medians, not separately isolated callback
or hardware costs.

The diagnostic copy changes caller code/register layout and adds sampling.
For example its Serial Write median is 36.831 ns versus 37.910 ns for the
unchanged control, while its Paired Release median is 82.777 versus 82.147 ns.
No API optimization occurred. These differences must not be presented as
speedups or added to the historical B2/C performance baselines.

## 5. Static hardware-path attribution

Source: `ggml/src/ggml-potal/potal.c:755-881,984-1036`.
Assembly: `/tmp/h3-6b3-before/disassembly.txt`.
Selected paths: valid nonfailed session, carrier 0, valid DIM16 key, no Sink,
cached Trace OFF, hardware status OK. Compressed instructions count once.
Other carriers, invalid inputs and error/Trace/Sink paths differ.

The following disjoint classification covers every selected PC exactly once.
It refines B3's combined validation category to expose readiness separately.
**These are static API-body counts, not measured runtime category costs.**

| Associated work | Write empty slot | Write live slot, new column | Release |
|---|---:|---:|---:|
| Session readiness: NULL / sticky-failed checks | 3 | 3 | 3 |
| Scale key/carrier range checks and argument encoding | 17 | 17 | 13 |
| Generation, liveness, mask addressing and successful state updates | 13 | 20 | 17 |
| Command-counter load/add/store | 3 | 3 | 3 |
| `paired_drained()` state/count updates | 0 | 0 | 6 |
| Optional Sink/Trace selection | 4 | 4 | 4 |
| Register preservation, stack, argument moves and return | 17 | 17 | 13 |
| One RoCC issue and hardware-status handling | 5 | 5 | 5 |
| **Total** | **62** | **69** | **64** |

Serial checked success paths contain 23 instructions for Write and 19 for
Release, with no stack frame. In this stream each slot has one empty-slot
Write and 15 live-slot Writes, giving a static Paired API-body average of
68.5625 instructions/Write. That is not the diagnostic whole-loop value of
78.014; caller and asynchronous work have different boundaries.

### Validation, liveness and ownership

Key range checks already combine slot/generation bounds efficiently. Write
also validates the carrier; Release has no carrier argument. A live Write
checks the slot generation and duplicate-column mask before issue. Release
checks live-column presence and matching generation before issue.

`paired_ready()` checks NULL and sticky failure. It does not perform a thread
ID lookup, lock or per-command global ownership check. The process-global
`paired_owner` gate is in session open/close. API thread/hart discipline is a
caller contract; it is not an additional hot-path lock cost
(`potal.h:15-22`, `potal.c:876-936`).

The command count increments exactly once before raw issue, including an
issued command that returns an error. Successful liveness updates happen only
after OK. Status values outside the hardware range map to bad response;
non-OK results retain the first sticky error/stage. No check or status handling
was bypassed.

### Register and issue costs

Write reserves 80 bytes and saves/restores `ra,s0,s1,s2,s3,s4`: six stores
and six loads. Release reserves 64 bytes and saves/restores `ra,s0,s1,s2`:
four stores and four loads. The counts above also include two frame adjustments,
two argument-preservation moves and return. The ordinary hardware success
paths execute **zero function or library calls**, despite preserving registers
needed by optional paths. B3 already proved that moving optional dispatch to
a helper did not remove this work.

The hardware block consists of one RoCC instruction plus status-range and
success handling. Trace-disabled Scale commands do not call `getenv`, format
records, or enter `paired_raw_traced`. The software-only Sink does construct
a record and invoke its callback. Its measured counts are not hardware timings.

## 6. Repeated Release state updates

`paired_drained()` clears aggregate loops/readbacks and every pair's outstanding
count, then increments the public session drain count. In this fixed two-pair
build it compiles to:

```text
12b22: ld    a5,56(s0)       # drains
12b24: sd    zero,40(s0)     # both aggregate 32-bit counters
12b28: sw    zero,140(s0)    # pair 0 outstanding
12b2c: addi  a5,a5,1
12b2e: sw    zero,188(s0)    # pair 1 outstanding
12b32: sd    a5,56(s0)
```

Only **three** instructions are zeroing stores. The other three implement
observable `drains++` and cannot be dropped merely because the session was idle.
With no GEMM/readback ever issued, every timed Release encounters already-zero
outstanding counters. The stream therefore entails 442,368 redundant zero-store
operations per Release repetition, derived from the known path and command
count. This is not a measured per-PC/store event count or a nanosecond budget.

**Separate zero-store cycle cost remains unmeasured.** B4 did not remove stores
or compare a synchronization-changing Release. The measured Release cycle
interval contains all API/caller work. Static three-of-64 instructions does
not imply three-of-64 elapsed time. A strict guard must test enough state;
its additional loads, logic and branch can exceed three unconditional stores.

### Invariants and error/cleanup coverage

| Existing behavior | Consequence for any future proposal |
|---|---|
| `calloc` initializes counters; successful complete LOOP_WS/readback issue increments aggregate and pair state (`potal.c:899,1156-1180`) | Idleness is true in the isolated stream, not an API-wide constant. |
| Successful public Release or Fence clears both aggregates and all pair counters (`potal.c:867,1034,1231`) | Preserve all fields and the accepted-command drain count. Failed commands do not clear them. |
| Loop counters are `uint32_t` without an explicit no-wrap guard | Aggregate-zero alone does not prove every pair counter is zero for the full API domain. |
| Readbacks have independent outstanding state | Checking loops alone cannot establish idleness. |
| Existing test issues another Main between Release columns (`tests/potal-api-test.c:831-845`) | Neither a later column nor a still-live slot proves no intervening work. |
| Resident B/geometry/Index metadata survive a drain | Do not clear or invalidate unrelated resident state. |
| Query, flush and close consume these counters; healthy busy close refuses; failed close can retain accelerator ownership (`potal.c:913-981`) | State errors can cause unsafe DMA reuse or ownership release. |
| Cleanup bypasses ordinary readiness, fences tracked work, then releases live columns directly (`potal.c:1184-1221`) | Cleanup Releases do not call `paired_drained()` or increment drains. Do not merge this with public Release accounting. |
| First error/stage is sticky; a stored first `fence` error suppresses cleanup retry (`potal.c:755,1191`) | Preserve first-error behavior; do not reinterpret it as a universal later-fence-failure latch. |
| Partial raw-issue errors may precede complete-issue counter updates | Software zero state is not a stronger hardware-idleness guarantee. |

The trace regression explicitly expects different cleanup/public drain totals
(`tests/potal-trace-test.c:97-103`). Executor settlement preserves DMA buffers
on fatal/unknown outcomes before wrapper close. These paths were inspected,
not changed or newly exercised by a production optimization.

Hardware Release acceptance still waits for
`releaseSafe && scaleRelease.ready`. Controller wires `releaseSafe` and
`fenceDrained` to the same `allDrained` condition
(`generators/potal/src/main/scala/potal/Controller.scala:803-809` and
`PoTalScaleCommandIngress.scala:450-455`). CPU zero counters never authorize
removing that synchronization. Release has a compiler memory clobber; explicit
Fence additionally performs the CPU `fence rw,rw`. Both remain unchanged.

A guard checking every skipped field is already zero can be state-equivalent
when placed after successful Release, preserving liveness and unconditional
`drains++`. That logical observation does not establish fewer executed
instructions, fewer cycles, or a profitable optimization. No guard or new dirty
flag was implemented.

## 7. Decode and GPT-2 FFN execution overhead

The unprofiled B2 seven-repeat records remain the current retained performance
baseline for these two normal-Paired workloads. C optimized capacity fallback;
it did not supply new normal-Paired detailed profiles.

| B2 steady per graph | Decode 1x768x768 | GPT-2 FFN up 16x3072x768 |
|---|---:|---:|
| Serial compute median ms | 9.283 | 53.500 |
| Paired compute median ms | 14.699 | 79.623 |
| Serial aggregate Scale median ms | 3.214 | 14.338 |
| Paired aggregate Scale median ms | 7.482 | 33.369 |
| Writes / Releases | 33,024 / 33,024 | 147,456 / 147,456 |
| Paired Main / Residual LOOP_WS | 12 / 60 | 44 / 264 |
| Index Config / Write | 60 / 978 | 264 / 6,314 |
| Readback calls / tiles | 12 / 96 | 44 / 384 |
| Paired fences / report drains | 1 / 73 | 1 / 309 |
| Fallback stripes | 0 | 0 |

Original sources: `/tmp/h3-6b2-run/uartlog:399,498` for graph summaries;
`H3-6B2-trace-cold-path-report.md:352-379` for Scale and command counts.
The Scale median gaps are 4.268 ms Decode and 19.031 ms FFN. They are not
proved removable work or pure accelerator-wait intervals.

B2 did not run a new detailed profiling pass. For finer boundaries, the
following values use the separately enabled **B1** diagnostic pass: medians
of steady iterations 2 and 3, one Paired graph each, not B2 component costs.
They are retained diagnostic evidence, not newly measured B4 graph timings.

| Diagnostic field, ms/Paired graph | Decode | FFN |
|---|---:|---:|
| `scale_write_ms` | 4.5985 | 20.6555 |
| `scale_release_ms` | 3.6745 | 16.1965 |
| Route `[POTAL_PLAN_PROFILE] plan_ms` | 1.4820 | 8.1890 |
| Executor `[POTAL_PROFILE] plan_ms` | 0.8525 | 5.5375 |
| Combined `index_ms` | 0.2440 | 1.5555 |
| `main_issue_ms` | 0.1560 | 0.7285 |
| `residual_issue_ms` | 0.1275 | 0.6205 |
| `readback_ms` | 0.0190 | 0.1455 |

Raw records were reread at `/tmp/h3-6b1-run/uartlog:600-613,682-695` and the
values recalculated. H3-6A's older separate diagnostic Write/Release values
were 19.125/16.888 ms Decode and 90.331/83.883 ms FFN. Those pre-B1/B2 costs
must not be used as current savings budgets. H3-6A's original route timer also
included packet decode; its 1.522/8.232 ms route figures are not pure planning
measurements. Later boundaries separate `decode_ms` and `plan_ms`.

### Boundaries, overlaps and hardware waiting

- **Scale:** timers surround carrier access/caller loops, checked API calls,
  bookkeeping and hardware admission (`matmul/paired-auto.cpp:1800-1865`).
  Release may wait for earlier hardware work. Its interval does not distinguish
  waiting from CPU execution. Conservative pre-drain work is outside this
  Release-loop timer and can move waiting to another interval.
- **Planning:** route planning builds and internally validates the plan
  (`paired-auto.cpp:928`). The executor receives that plan rather than
  regenerating it, validates it again, checks the request and deep-copies plan
  and tile (`paired-auto.cpp:2146-2205`, `matmul/stripe.cpp:775`). Executor
  `plan_ms` is not a validation-only timer. Removable validation/copy fractions
  have not been separated.
- **Index:** `index_ms` includes Config, all Writes and loop bookkeeping
  (`paired-auto.cpp:1916-1948`). Existing data does not separate Config versus
  Write latency. Dividing by all commands is not an individual command cost.
- **LOOP_WS:** each logical issue includes checked encoding/state checks,
  a CPU memory fence and a ten-RoCC-command configuration/issue bundle
  (`potal.c:1096-1110`). Main/Residual issue intervals are neither the final
  instruction alone nor GEMM execution time.
- **ACC readback:** the interval covers checked encoding plus CONFIG/MVOUT
  issue, not DMA completion (`potal.c:1161-1182`,
  `paired-auto.cpp:2001-2027`). Later Release/Fence drainage establishes
  completion before CPU consumption.
- **Graph wall time:** pipeline wall is nested inside graph compute. Activation
  production overlaps the issuer and can include capacity waiting
  (`matmul/stripe.cpp:1202-1239`). Diagnostic logging also perturbs execution.
  Do not add pipeline or activation time to graph time, add Scale subgroups to
  aggregate Scale, add issue subgroups to `issue_s`, or add internal validation
  again to its enclosing planning interval. Even distinct issuer intervals
  do not automatically become independent graph critical-path contributions.

The data establishes substantial software work with no GEMM outstanding. It
does **not** measure actual hardware wait fractions for Decode or FFN. In
particular, small explicit-Fence intervals do not prove negligible waiting:
a prior accepted Release may already have absorbed it.

## 8. Ranked opportunities and proof requirements

Ranking favors a credible next investigation, not a claimed speedup. Static
instruction upper bounds below are not removable cycle or nanosecond estimates.

| Rank / direction | Suspected mechanism | Supporting evidence | Estimated removable work | Correctness / synchronization risks | Software-only sufficient? | Experiment required to demonstrate benefit |
|---|---|---|---|---|---|---|
| 1 / B | Repeated execution-plan validation and deep copies | Distinct validated route plan and executor revalidation/copies; B1 diagnostic route/executor intervals 1.482/0.8525 ms Decode, 8.189/5.5375 ms FFN. | At most a demonstrated redundant validation/copy; removable fraction and graph saving unmeasured. | Plan provenance, immutability, request matching, failure paths and buffer lifetimes must retain checked behavior. | Plausibly software-only, with checked public boundaries preserved. | Separate validation/copy counters, same-input matched codegen, complete invalid-plan/state regressions and unprofiled seven-repeat benefit. |
| 2 / C | Scale and Index window traffic volume | 33024 or 147456 Writes and Releases; 978 or 6314 Index Writes; paired loop counts exceed Serial. | Only commands proven unnecessary by legal window/residency reuse; no established removable count. | Scale generation lifetime, global Release drains, B residency, Index ordering and bitwise saturation order. | Not established; must use current checked commands or be a separate hardware/protocol design study, not B4 implementation. | Prove legal smaller command stream and identical numeric/order constraints before matched cycle and graph tests. |
| 3 / A not justified | Repeated successful Write generation stores | Current live-slot check proves the same generation; success path still executes address construction plus generation store. | Upper bound 3 static instructions on live Writes before guard/register cost; 15/16 of this isolated stream. No measured removable cycles. | Must retain empty-slot initialization, post-success state and error/Sink/Trace behavior. | Possible in principle; no net generated-code reduction demonstrated. | First prove fewer executed instructions with all checks and complete state equivalence; only then seven-repeat hardware/Sink timing. |
| 4 / A not justified | Repeated paired_drained zero stores | 3 zeroing stores in a 6-instruction drain update; all counters already zero in this no-GEMM stream. | At most 3 zeroing stores per Release before guard cost; drains++ remains. Isolated per-store cycle cost unmeasured. | All aggregate and per-pair state must be zero; readbacks, interleaved work, 32-bit wrap, sticky errors, cleanup and DMA ownership. | A strict all-zero guard can be state-equivalent, but may execute more work than 3 stores. | No removal in B4. Future proof must include all state fields and cleanup paths, then codegen reduction and real-work hardware benefit. |
| 5 / Reject repetition | Sink/Trace helper extraction and smaller reserved frame | B3 already left hardware paths 62/69/64 instructions with same saves/restores. | Zero demonstrated executed instructions; smaller frame alone is not reduced work. | Shared successful state transitions and diagnostic/status equivalence. | Existing failed experiment is software-only but unjustified. | Do not repeat B3; any future approach must be materially different and first reduce actual executed work. |
| 6 / Protected work | Validation, liveness, command counters and accepted Release synchronization | Checked API and query/error contracts require these updates and exactly one issue per command. | None established within current contract. | Unchecked accesses, stale generations, duplicate writes, Release misses, command-accounting drift or early DMA reclamation. | Not by bypassing checks or weakening Fence/Release. | No such weakening is permitted. |

The Write generation-store observation comes from the existing live-generation
check and the post-success address/store sequence at `129c0-129d4` and
`12a20-12a24`. It is a candidate to test, not evidence that a conditional
rewrite will improve compiled code. No current candidate has passed both a
net-work-reduction gate and a checked end-to-end benefit gate.

For B, first distinguish validation traversal from vector copies/allocations
within the two measured planning boundaries. Preserve checked public entry
behavior and require provenance/immutability proof before any internal reuse.
Then compare actual work and unprofiled seven-repeat graph results. Do not
infer a 5.5375 or 8.189 ms removable FFN budget from enclosing intervals.

For C, measured command volume motivates a separate residency/window design
study. It does not prove that mandatory per-entry Scale commands or required
Release drain points can be removed under the existing contract. No bulk or
unchecked API, reordered stream or synchronization change is proposed here.

## 9. Verification, shutdown and artifacts

Targeted instrumentation verification passed:

- Cross compilation/link and native Host compilation/link: exit 0.
- Host and guest SIGILL-probe self-test: PASS.
- Host software-only seven-repeat validation: PASS; no hardware timing claim.
- Unchanged guest baseline: seven repeats for all three routes, all statuses
  and batch-state/count checks passed, `no_gemms=1 live_entries=0`.
- Guest counter copy: seven repeats for all routes, `instret_supported=1`,
  all 42 rows complete, same stream and checks, suite exit 0.
- Before/after source, benchmark, H3-6C executable/library and B2/C report
  fingerprints matched. No previous baseline was overwritten.

Existing generated-header macro-redefinition warnings remained. The temporary
measurement source has no configured editor include paths; actual cross and
native compilers succeeded. No clean LSP/include-resolution gate is claimed.
No production code changed. Existing Host/API/Trace evidence and B2/C hardware
numerical gates were retained; no full hardware numerical suite was repeated.
No model suite ran.

Only one B4 guest was started, from a new copy of the already cleanly powered-off
C image. It received `poweroff`, reported `reboot: Power down`, completed
simulation with `*** PASSED ***`, and the driver returned 0. All B4 milestone
watchers were stopped or completed. Existing B2/C images and logs were preserved.

| Artifact | Content |
|---|---|
| `H3-6B4-remaining-scale-overhead.md` | This report |
| `/tmp/h3-6b4/scale-path-counters.c` | Diagnostic-only benchmark copy and safe counter probe |
| `/tmp/h3-6b4/guest-suite.sh` | Saved guest command sequence |
| `/tmp/h3-6b4/baseline-hashes.json` | Retained executable/source identities |
| `/tmp/h3-6b4/static-attribution.json` | Every selected PC and its disjoint category |
| `/tmp/h3-6b4/measurements.json` | All 84 measured route/phase rows, summaries, historical profile fields and ranked candidates |
| `/tmp/h3-6b4-run/uartlog` | Guest hashes, affinity, self-test, both full measurements and clean poweroff |
| `/tmp/h3-6b3-before/disassembly.txt` | Retained B2 hardware-path assembly |
| `/tmp/h3-6b1-run/uartlog` | Separate historical detailed profile |
| `/tmp/h3-6b2-run/uartlog` | Preserved B2 Scale/Decode/FFN baseline |
| `/tmp/h3-6c-after-run/uartlog` | Preserved C fallback validation and timing baseline |

No RTL modification, FPGA rebuild, numerical/quantization change, unchecked or
bulk API, command-order or synchronization change, project commit or push was
performed. Stop here: the evidence supports **B**, and does not justify another
H3-6B checked Scale micro-optimization on present data.
