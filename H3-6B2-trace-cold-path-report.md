# H3-6B2: isolate the paired trace cold path

Date: October 25, 2026. Repository branch: `potal`.
Hardware: existing H3-5B FireSim U280 DIM16, unchanged.
Baseline: the current uncommitted H3-6B1 implementation.

## Result

Outlining enabled trace output measurably simplifies the disabled-trace path and
reduces command latency beyond the observed seven-sample dispersion:

* Paired Write: **101.969 -> 87.029 ns/command**, 14.65% lower.
* Paired Release: **93.343 -> 82.147 ns/command**, 12.00% lower.
* Decode Paired steady compute: **15.622 -> 14.699 ms**, 5.91% lower.
* FFN Paired steady compute: **83.242 -> 79.623 ms**, 4.35% lower.

All trace tests, full sink-record comparisons, focused host regressions, target
trace tests, isolated benchmarks, and bitwise Golden checks passed. Command,
Fence, Drain, and fallback counts matched B1. Guest shutdown and simulator exit
completed cleanly with status zero.

This removes only part of the remaining Paired penalty. Unloaded Paired hardware
still exceeds Serial by approximately **49.194 ns/Write** and
**49.242 ns/Release**. The evidence does not attribute all remaining overhead to
tracing or imply that checked-session work is unnecessary.

Only `ggml/src/ggml-potal/potal.c` was modified as production source. Existing
tests were reused without edits. H3-3/H3-4/H3-6A/H3-6B0/B1 work was preserved.
No RTL change, FPGA rebuild, bulk/unchecked Scale API, release/drain semantic
change, numerical execution change, full model suite, commit, or push occurred.

## Minimal source diff

The complete one-hunk diff against the preserved B1 `potal.c` is below. Its only
functional change is routing enabled tracing through a separate helper:

```diff
@@ -806,42 +806,53 @@
 }
 
 
-static potal_status paired_raw(
+#if defined(__riscv) || defined(POTAL_TEST_RAW_TRACE)
+#if defined(__GNUC__) || defined(__clang__)
+__attribute__((noinline, cold))
+#endif
+static potal_status paired_raw_traced(
     potal_paired_session *s,
     uint32_t funct,
     uint64_t rs1,
     uint64_t rs2,
     int xd
 ) {
-#if defined(__riscv) || defined(POTAL_TEST_RAW_TRACE)
-    const int enabled = s->trace_enabled;
     const unsigned long long seq =
         (unsigned long long)s->commands + 1;
 
-    if (enabled) {
-        fprintf(stderr,
-            "[POTAL_RAW] ENTER seq=%llu funct=%u xd=%d "
-            "rs1=0x%016llx rs2=0x%016llx\n",
-            seq, funct, xd,
-            (unsigned long long)rs1,
-            (unsigned long long)rs2);
-        fflush(stderr);
-    }
-#endif
+    fprintf(stderr,
+        "[POTAL_RAW] ENTER seq=%llu funct=%u xd=%d "
+        "rs1=0x%016llx rs2=0x%016llx\n",
+        seq, funct, xd,
+        (unsigned long long)rs1,
+        (unsigned long long)rs2);
+    fflush(stderr);
 
     const potal_status status =
         paired_raw_untraced(s, funct, rs1, rs2, xd);
 
+    fprintf(stderr,
+        "[POTAL_RAW] EXIT seq=%llu funct=%u status=%d\n",
+        seq, funct, (int)status);
+    fflush(stderr);
+
+    return status;
+}
+#endif
+
+static potal_status paired_raw(
+    potal_paired_session *s,
+    uint32_t funct,
+    uint64_t rs1,
+    uint64_t rs2,
+    int xd
+) {
 #if defined(__riscv) || defined(POTAL_TEST_RAW_TRACE)
-    if (enabled) {
-        fprintf(stderr,
-            "[POTAL_RAW] EXIT seq=%llu funct=%u status=%d\n",
-            seq, funct, (int)status);
-        fflush(stderr);
+    if (s->trace_enabled) {
+        return paired_raw_traced(s, funct, rs1, rs2, xd);
     }
 #endif
-
-    return status;
+    return paired_raw_untraced(s, funct, rs1, rs2, xd);
 }
 
 #define PAIRED_RAW(session, funct, rs1, rs2, xd, stage)                                  \
```

`noinline` and `cold` are guarded for GCC/Clang. Other compilers see an ordinary
C declaration; the separate non-inlined helper was verified in this task's GCC
output. The existing RISC-V/private host-test guards are retained.

The cached session-open trace selection is unchanged. The helper computes the
same pre-issue sequence number, prints the same ENTER record, invokes the
unchanged `paired_raw_untraced` exactly once, prints the same EXIT/status record,
and returns that status. Disabled tracing returns directly through
`paired_raw_untraced`. The command-counter increment remains solely in that
existing dispatcher, exactly once per issued command.

No environment access, formatting, library output, allocation, extra RoCC
instruction, or extra Fence was added to the disabled hardware path.
`paired_drained`, validation, liveness/generation checks, sticky failure,
cleanup, ownership, encoders, and synchronization were not edited.

## Trace and command equivalence

The unchanged H3-6B1 `potal-trace-test` passed on the host and RISC-V guest.
Coverage includes:

* Successive OFF -> ON -> OFF sessions and a successful ON session.
* Unset, empty, `"0"`, `"true"`, `"1"`, and `"1suffix"` selections.
* Environment changes during a session, which do not change its snapshot.
* Parsed ENTER/EXIT sequence, funct, xd, rs1, rs2, and returned status fields.
* No records from disabled sessions.
* Injected Scale hardware error, sticky failure, cleanup releases, and safe close.
* Duplicate write, wrong generation, release miss, live-column state, outstanding
  work, command count, and drain-count assertions.

The saved B1 host trace executable and rebuilt B2 executable were compared in
`--dump` mode. All **42 complete command records across seven sessions** matched
byte-for-byte, including injected statuses and cleanup commands:

```text
POTAL_TRACE_TEST PASS
B2_FULL_SINK_RECORDS_IDENTICAL
```

Seven existing host executables also passed, each executed directly with exit
status zero:

```text
potal-api-test
potal-paired-plan-test
potal-paired-schedule-test
potal-paired-unit-test
potal-paired-auto-test
potal-paired-execution-test
potal-stripe-pipeline-test
```

No absence of registered CTest tests was counted as a pass. Existing generated
header macro-redefinition warnings and editor include-resolution diagnostics
remain; actual host and DIM16 RISC-V compiler builds succeeded.

## RISC-V code-generation comparison

Compiler and flags were unchanged: GCC 12.2.0 (`g2ee5e430018`), default
`rv64imafdc`, `lp64d`, Rocket tuning, Release `-O3 -DNDEBUG -std=c11`, and the
existing PIC/visibility/warning settings.

### Counting method

Two different counts are reported:

1. **Static function instructions** include all instructions in the emitted
   function, including cold/error blocks.
2. **Selected executed-path instructions** follow a valid trace-disabled hardware
   success path, including prologue/epilogue and one RoCC command. The raw status
   is `POTAL_OK`, carrier is 0, and no sink is bound. Write is counted separately
   for an empty slot and a same-generation live slot with an unwritten column.

Each compressed instruction counts as one instruction. These are walks of the
generated control-flow paths, not hardware `instret` measurements or cycle
predictions. Invalid arguments and trace-enabled/sink paths have different counts.

| Function/path | B1 instructions | B2 instructions | B1 conditional branches | B2 conditional branches |
|---|---:|---:|---:|---:|
| Write, empty slot, disabled hardware success | 76 | 62 | 11 | 10 |
| Write, live slot, disabled hardware success | 83 | 69 | 13 | 12 |
| Release, disabled hardware success | 75 | 64 | 11 | 10 |

| Emitted function | B1 static instructions/bytes | B2 static instructions/bytes | B1 frame | B2 frame |
|---|---|---|---:|---:|
| `potal_paired_scale_write` | 145 / 410 | 123 / 306 | 128 B | 80 B |
| `potal_paired_scale_release` | 129 / 360 | 109 / 268 | 96 B | 64 B |
| `paired_raw` | 138 / 364 | fully inlined | 112 B | no standalone frame |
| `paired_raw_untraced` | fully inlined | fully inlined | no standalone frame | no standalone frame |
| `paired_raw_traced` | absent | 117 / 324 | absent | 96 B |

Prologue register preservation:

* Write: B1 saves `ra,s0..s9` (11 registers); B2 saves `ra,s0..s4` (6).
* Release: B1 saves `ra,s0..s6` (8 registers); B2 saves `ra,s0..s2` (4).
* The old emitted `paired_raw` saves `ra,s0..s7` (9). Its disabled hardware
  SCALE_WRITE/SCALE_RELEASE paths have 49/50 instructions and five conditional
  branches. Those are separate wrapper counts, not additions to the inlined
  Scale API counts above.
* The new cold helper saves `ra,s0..s6` (8). Its frame and output calls are
  entered only when the cached trace flag is enabled.

B1 already skipped output calls when tracing was disabled. Its larger frame and
register-preservation requirements still affected the ordinary path. B2 removes
the second trace-flag branch and the need to carry trace sequence/printing state
through ordinary issue. It does not eliminate safety checks to obtain the lower
instruction counts.

### Call sites and instruction evidence

B1 Write/Release contain two `fprintf` and two `fflush` call sites apiece in
their enabled branches. Their disabled hardware paths execute none.
B2 Write/Release contain no formatting/output library call sites. Their optional
sink callback remains; a taken trace branch invokes the new cold helper:

```text
0000000000010ce0 <paired_raw_traced>:
   10d1e: fa3ff0ef  jal ra,10cc0 <fprintf@plt>
   10d26: efbff0ef  jal ra,10c20 <fflush@plt>
   ...
   10e02: ebfff0ef  jal ra,10cc0 <fprintf@plt>
   10e0a: e17ff0ef  jal ra,10c20 <fflush@plt>

000000000001295c <potal_paired_scale_write>:
   12962: 715d      addi sp,sp,-80
   129f2: e659      bnez a2,12a80
   12a38: 2ed7f57b  .4byte 0x2ed7f57b
   ...
   12a88: a58fe0ef  jal ra,10ce0 <paired_raw_traced>

0000000000012a8e <potal_paired_scale_release>:
   12a92: 7139      addi sp,sp,-64
   12aee: ef51      bnez a4,12b8a
   12b40: 30d7f57b  .4byte 0x30d7f57b
   ...
   12b94: 94cfe0ef  jal ra,10ce0 <paired_raw_traced>
```

Neither selected disabled hardware path calls the helper or any library
function. No replacement environment lookup appears. The cached flag is still
read from the session.

RoCC words change CPU register allocation, not the command:

| Operation | B1 word | B2 word | Opcode | funct7 | funct3 |
|---|---|---|---|---:|---:|
| Scale Write | `0x2f39797b` | `0x2ed7f57b` | `0x7b` | 23 | 7 |
| Scale Release | `0x30e4f4fb` | `0x30d7f57b` | `0x7b` | 24 | 7 |

Both retain `xd=1` with both source operands present. The unchanged encoder and
full-record comparison preserve `rs1 = slot | generation<<8 | column<<16`,
Write `rs2 = carrier`, and Release `rs2 = 0`.

Full baseline and optimized assembly, including trace-enabled dispatch/output,
is saved at:

```text
/tmp/h3-6b2-before/disassembly.txt
/tmp/h3-6b2-after/disassembly.txt
/tmp/h3-6b2-after/path-evidence.json
```

The JSON records the exact selected instruction PCs so the path counts can be
checked against the archived assembly.

## Isolated Scale benchmark

The benchmark source and streams were unchanged: DIM16, 36 complete batches,
256 slots, 16 columns, carrier 0, generations 1..36, ascending Write/Release
order, and **147,456 commands per phase per repetition**.
All routes received the existing warmup and seven measured repetitions with
rotating route order. There were no outstanding Main/Residual GEMMs.

The benchmark verified physical hart 0 as Linux CPU 1 and pinned itself there.
Guest `steady_clock` and supported `rdcycle` were enabled. Timed sections had
no printing or enabled tracing. Every C11 status and independent per-batch
session-state/count check passed, with zero live entries and outstanding work.

All times below are **CPU-observed guest command latency**, not pure FPGA
execution time. The software-only sink performs no RoCC access.
MAD is median absolute deviation across the seven samples.

| Route | Phase | B1 median ns | B1 MAD ns | B2 median ns | B2 MAD ns | B2 min..max ns |
|---|---|---:|---:|---:|---:|---|
| Serial hardware | Write | 37.835 | 0.027 | 37.835 | 0.068 | 37.727..37.923 |
| Serial hardware | Release | 33.203 | 0.014 | 32.905 | 0.014 | 32.803..32.918 |
| Paired hardware | Write | 101.969 | 0.264 | 87.029 | 0.047 | 86.928..87.287 |
| Paired hardware | Release | 93.343 | 0.434 | 82.147 | 0.014 | 81.909..82.235 |
| Paired software-only | Write | 107.320 | 0.047 | 91.614 | 0.034 | 91.505..91.810 |
| Paired software-only | Release | 98.016 | 0.061 | 86.948 | 0.054 | 86.785..87.097 |

Measured hardware savings are **14.940 ns/Write** and **11.197 ns/Release**.
Every B2 Paired hardware sample is faster than the fastest B1 sample in its
phase; the separation is much larger than either run's observed MAD. The
software-only path improves similarly, supporting the software-overhead
interpretation. Serial Write is unchanged; Serial Release changes by -0.90%.

Median B2 guest cycles/command corroborate timing: Serial Write/Release
37.685/32.753; Paired hardware 86.868/81.993; Paired software-only 91.472/86.776.
These are architectural guest cycles, not physical FPGA execution cycles.
The counts do not partition cache effects, CPU stalls, or hardware admission.

## Decode and FFN performance

After the isolated benchmark passed, the same guest ran the target trace test,
`potal-paired-smoke --quick`, and the existing Decode/FFN benchmarks with seven
repetitions. Input generation, seed `20261023`, weight-cache policy, shapes,
route ordering, status checks, and independent bitwise F32 Golden checks were
unchanged. The existing CLI also runs cold samples and Conservative controls.

Quick smoke passed every route with `failed_nodes=0`. Both benchmark cases
returned zero and reported `POTAL_BENCH PASS ... failures=0`; all cold and steady
executions retained bitwise Golden checking.
The following values use only **unprofiled seven-repeat steady samples**.
No new detailed profiling pass or full model suite was run.

| Workload | Route | B1 median ms | B1 MAD ms | B2 median ms | B2 MAD ms | B2 min..max ms |
|---|---|---:|---:|---:|---:|---|
| Decode | Serial | 9.294 | 0.065 | 9.283 | 0.053 | 8.872..11.344 |
| Decode | Paired | 15.622 | 0.045 | 14.699 | 0.037 | 14.642..14.759 |
| Decode | Conservative | 15.808 | 0.007 | 14.903 | 0.023 | 14.862..14.960 |
| FFN | Serial | 53.572 | 0.072 | 53.500 | 0.189 | 52.860..53.724 |
| FFN | Paired | 83.242 | 0.139 | 79.623 | 0.177 | 79.446..79.941 |
| FFN | Conservative | 84.148 | 0.030 | 80.397 | 0.157 | 80.097..80.617 |

| Workload | Paired saved ms | Paired time reduction | B1 Paired/Serial | B2 Paired/Serial |
|---|---:|---:|---:|---:|
| Decode | 0.923 | 5.91% | 1.681 | 1.583 |
| FFN | 3.619 | 4.35% | 1.554 | 1.488 |

Serial compute medians change by -0.12% Decode and -0.13% FFN.
Paired remains slower than Serial; the result is an incremental measured
improvement, not evidence that Paired has become the faster route.

### Unprofiled Scale time

Scale time combines Writes and Releases at the existing caller timing boundary.
The per-command denominator includes both command counts.

| Workload | Route | B1 Scale ms | B2 Scale ms | B1 ns/command | B2 ns/command |
|---|---|---:|---:|---:|---:|
| Decode | Serial | 3.217 | 3.214 | 48.707 | 48.662 |
| Decode | Paired | 8.266 | 7.482 | 125.151 | 113.281 |
| Decode | Conservative | 8.267 | 7.485 | 125.167 | 113.327 |
| FFN | Serial | 14.363 | 14.338 | 48.703 | 48.618 |
| FFN | Paired | 36.829 | 33.369 | 124.881 | 113.149 |
| FFN | Conservative | 36.876 | 33.345 | 125.041 | 113.068 |

Paired Scale time falls 9.48% for Decode and 9.39% for FFN. These caller-level
timings are not identical to the standalone C11 loop's boundaries and working
set, and should not be treated as pure hardware time.

### Command, Fence, Drain, and fallback counts

Every non-timing counter in the B1 steady statistics matched across all seven
B2 samples for all three routes, including Serial Fence fields:

| Workload/route | Writes/Releases | Serial Main/Residual | Paired Main/Residual | Index config/write | Readbacks/tiles | Serial Main/Residual Fences | Paired Fences/Drains |
|---|---|---|---|---|---|---|---|
| Decode Serial | 33024/33024 | 6/4 | 0/0 | 0/0 | 0/0 | 6/4 | 0/0 |
| Decode Paired | 33024/33024 | 0/0 | 12/60 | 60/978 | 12/96 | 0/0 | 1/73 |
| Decode Conservative | 33024/33024 | 0/0 | 12/60 | 60/978 | 12/96 | 0/0 | 145/217 |
| FFN Serial | 147456/147456 | 22/22 | 0/0 | 0/0 | 0/0 | 22/22 | 0/0 |
| FFN Paired | 147456/147456 | 0/0 | 44/264 | 264/6314 | 44/384 | 0/0 | 1/309 |
| FFN Conservative | 147456/147456 | 0/0 | 44/264 | 264/6314 | 44/384 | 0/0 | 617/925 |

Fallback stripes and producer capacity waits remained zero. Each graph completed
one semantic stripe. No command reduction or synchronization relaxation explains
the timing improvement.

## Residual overhead and recommendation

The remaining unloaded Paired-minus-Serial medians are approximately
49.194 ns/Write and 49.242 ns/Release. The software-only sink remains slightly
slower than hardware. Outlining helps, but checked validation, state
bookkeeping, optional dispatch, and register preservation remain; their
individual timing contributions were not isolated here.

In FFN, Paired Scale still takes 33.369 ms versus Serial 14.338 ms, a
19.031 ms stage gap. The graph compute gap is 26.123 ms. These overlapping
intervals are not a strict additive causal decomposition, but they support
continued attention to checked command overhead.

**Keep B2. Do not remove checks or introduce a bulk/unchecked API.** A next
candidate is reducing common-path register preservation around optional sink/
trace issue, using an internal checked hardware path rather than a new public
API. First demonstrate a smaller generated frame/path without duplicating or
weakening liveness, generation, sticky-failure, counter, or drain handling; then
repeat the same hardware/sink and bitwise tests. If that code-generation proof
does not hold, choose a measured bookkeeping attribution task instead of adding
another speculative specialization.

The remaining approximately 49 ns is not a measured removable budget.
`paired_drained()` and all synchronization should remain unchanged until a
separate measurement supports a specific safe change. No further optimization
was implemented in B2.

## Reproduction and saved evidence

The existing host and cross build directories and flags were reused:

```sh
source /home/alveo/miniforge3/etc/profile.d/conda.sh
cd /home/alveo/firesim/target-design/chipyard
source env.sh
cd generators/potal/software/workspace/llama.cpp-potal
cmake --build build-potal-h3-6a-d16 \
  --target potal-scale-path-bench potal-paired-smoke potal-trace-test -j8
```

Host tests were rebuilt and executed directly. The pre-existing missing
`chipyard/.conda-env` activation message did not prevent use of the configured
compiler under `/home/alveo/firesim/.conda-env`.

Before artifacts and exact source/assembly evidence:

```text
/tmp/h3-6b2-before/potal.c
/tmp/h3-6b2-before/potal-scale-path-bench
/tmp/h3-6b2-before/potal-paired-smoke
/tmp/h3-6b2-before/libggml-potal.so
/tmp/h3-6b2-before/potal-trace-test
/tmp/h3-6b2-before/disassembly.txt
/tmp/h3-6b2-before/hot-path-counts.json
/tmp/h3-6b2-after/source.diff
/tmp/h3-6b2-after/disassembly.txt
/tmp/h3-6b2-after/path-evidence.json
```

Preserved B1 source SHA-256:
`36e524a36179fbb030c1b4a2eed4c8a51b8c50ed6a3795d958226649c8f2a30a`.

The unchanged FPGA driver used a separate copied image at
`/tmp/h3-6b2-run/rootfs.img`; earlier images and logs were not overwritten.
Guest commands were success-gated in `/tmp/h3-6b2-guest-suite.sh`:

```sh
unset POTAL_RAW_TRACE POTAL_PAIRED_PROFILE
./potal-scale-path-bench --cycles
./potal-trace-test
# In the existing benchmark bin directory:
./potal-paired-smoke --quick
./potal-paired-smoke --bench --case decode_1x768x768 --repeat 7
./potal-paired-smoke --bench --case gpt2_ffn_up_16x3072x768 --repeat 7
```

Raw guest evidence is `/tmp/h3-6b2-run/uartlog`, with stage markers and final:

```text
__H36B2_SUITE_PASS__
__H36B2_SUITE_EXIT__=0
```

Host, staged image, and running guest hashes matched:

| B2 artifact | SHA-256 |
|---|---|
| `potal-scale-path-bench` | `1f13bd8aa8a12fbc59e45625be32b900904a6336727555c09fba18656c96c758` |
| `potal-paired-smoke` | `5350576c8dc32a0a4ce270c8e0d48e36020311955a4908859f261680231e0c58` |
| `libggml-potal.so` | `75efab94831166f795c71a628d60dc207588104d5c1e9df674e8333c2ef80c8b` |

The DIM16 parameter hash remains
`b4cd8b9efef9f3cd9b598ae3cc84af6478f83e628ed73d94a77fb1c452df1fc7`.
The fixed H3-5B RTL revision remains
`dc2f52a5bd0236bc4e02a8d1109151945661755b`.
