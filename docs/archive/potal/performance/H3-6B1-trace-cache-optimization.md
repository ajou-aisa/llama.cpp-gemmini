# H3-6B1: cache paired trace selection at session open

Date: October 25, 2026. Branch: `potal`. Hardware: existing H3-5B fixed
FireSim U280, DIM16.

## Result

Caching `POTAL_RAW_TRACE` at session open removed `getenv` from the Paired command
hot paths without changing the issued command stream or hardware synchronization.

* Isolated Paired Write: **318.515 to 101.969 ns/command**, 68.0% lower.
* Isolated Paired Release: **306.322 to 93.343 ns/command**, 69.5% lower.
* Decode Paired median: **41.417 to 15.622 ms**, a 2.65x speedup.
* FFN Paired median: **209.738 to 83.242 ms**, a 2.52x speedup.
* Serial controls remained stable. All requested tests, target trace tests,
  isolated passes, quick smoke, and Decode/FFN runs passed.

Paired remains slower than Serial: 1.681x for Decode and 1.554x for FFN.
No bulk/unchecked Scale API, RTL change, FPGA rebuild, release/drain change,
GGML execution change, quantization change, full model suite, commit, or push
was performed. Existing H3-3/H3-4 diagnostics and H3-6A/H3-6B0 work were preserved.

## Minimal production change

`ggml/src/ggml-potal/potal.c` now holds a private `trace_enabled` field in the
opaque session. After validation and successful allocation, session open reads
the environment once. No subsequent initialization step can fail.
`paired_raw` reads that field instead of calling `getenv`.

The functional diff is:

```diff
 struct potal_paired_session {
     potal_paired_sink sink;
     int               failed;
+    int               trace_enabled;        /* POTAL_RAW_TRACE snapshot at successful open */
```

```diff
-    const char *env = getenv("POTAL_RAW_TRACE");
-    const int enabled = env != NULL && env[0] == '1';
+    const int enabled = s->trace_enabled;
```

```diff
     if (sink != NULL) {
         s->sink = *sink;
     }
+    const char *env = getenv("POTAL_RAW_TRACE");
+    s->trace_enabled = env != NULL && env[0] == '1';
     paired_owner = 1;
     *out = s;
```

The environment interpretation is unchanged: only a value whose first character
is `'1'` enables tracing. `include/potal.h` documents the new session-open
snapshot contract: changing the environment affects the next session, not an
active session.

ENTER/EXIT formatting, sequence calculation, command counters, returned status,
sticky failure handling, and `paired_raw_untraced` dispatch are unchanged.
The complete source comparison against the preserved baseline contains no
changes to `paired_drained`, Scale validation/liveness/generation handling,
cleanup, ownership, encoding, Fence, or CPU memory fences.

For host trace testing only, the three existing RISC-V guards around the stdio
include and ENTER/EXIT printing also accept `POTAL_TEST_RAW_TRACE`.
The new `potal-trace-test` target compiles its own `potal.c` with this private
definition. Normal host `potal-api` retains its previous non-tracing behavior;
real RoCC dispatch remains RISC-V-only. No runtime control API was added.

## C11, trace, and command-stream regressions

These seven existing host executables were built and run directly, each with
exit status zero:

* `potal-api-test`
* `potal-paired-plan-test`
* `potal-paired-schedule-test`
* `potal-paired-unit-test`
* `potal-paired-auto-test`
* `potal-paired-execution-test`
* `potal-stripe-pipeline-test`

The new `tests/potal-trace-test.c` passed on both the host and the RISC-V guest.
It exercises successive OFF -> ON -> OFF sessions, a successful ON session, and
unset, empty, `"0"`, `"true"`, `"1"`, and `"1suffix"` selections. Each session
changes the environment to the opposite selection after opening, proving that
the active snapshot remains unchanged and that the next session refreshes it.

The test captures stderr and parses the machine-consumed ENTER/EXIT records.
It verifies every sequence number, funct, xd, rs1, rs2, returned status, and
record count. Sequence numbers restart at 1 for each session. Disabled sessions
emit no records; enabled sessions trace successful commands, an injected
hardware error, and cleanup releases.

Success/error cases also check duplicate-write rejection, wrong-generation
release rejection, release-miss rejection, live-entry close refusal, sticky
failure, cleanup, final liveness, outstanding work, drain/command counts, and
safe close. Existing tests cover the wider loop/readback/ownership/failure
contracts, including failed-fence cleanup behavior.

For baseline equivalence, the new harness was compiled separately against the
unchanged, saved pre-optimization `potal.c`. Its `--dump` mode emits every full
sink record, including the returned injected status. A direct diff against the
optimized harness was empty:

```text
POTAL_TRACE_TEST PASS
FULL_SINK_RECORDS_IDENTICAL
```

All **42 command records across seven sessions** matched, including failure and
cleanup commands. The comparison includes all encoded arguments, not just
function counts or a checksum. It skips trace-snapshot assertions only in dump
mode so the old implementation can serve as the command-stream baseline.

No CTest-registration result was used as a substitute for executing tests.
Existing `POTAL_SCALE_ENTRIES` header-redefinition warnings remain. The editor's
existing include/standard-header configuration still produces unresolved-header
diagnostics; actual host and RISC-V compiler builds succeeded.

## Before/after RISC-V code generation

The existing Release configuration was retained: GCC 12.2.0
(`g2ee5e430018`), `rv64imafdc`, `lp64d`, Rocket tuning, `-O3 -DNDEBUG`,
and C11. The API retains its existing PIC/visibility flags. No compiler
optimization flag was changed.

Disassembly of the preserved pre-change DIM16 executable:

```text
00000000000126dc <potal_paired_session_open>:
    [allocation and initialization; no getenv call]

0000000000012a16 <potal_paired_scale_write>:
   12aca: 9e6fe0ef  jal ra,10cb0 <getenv@plt>

0000000000012bbe <potal_paired_scale_release>:
   12c32: 87efe0ef  jal ra,10cb0 <getenv@plt>
```

Disassembly of the rebuilt DIM16 executable:

```text
00000000000126be <potal_paired_session_open>:
   126f8: cd8fe0ef  jal ra,10bd0 <calloc@plt>
   12712: d9efe0ef  jal ra,10cb0 <getenv@plt>

0000000000012a06 <potal_paired_scale_write>:
   [cached flag test; no getenv call]
   12ae0: 9702      jalr a4                 # existing sink dispatch
   12b04: 2f39797b  .4byte 0x2f39797b       # hardware SCALE_WRITE

0000000000012ba0 <potal_paired_scale_release>:
   [no getenv call]
   12c32: 9702      jalr a4                 # existing sink dispatch
   12c7a: 30e4f4fb  .4byte 0x30e4f4fb       # hardware SCALE_RELEASE
```

The shared `paired_raw` disassembly also has no `getenv` call. No replacement
library call appears in the disabled hardware hot path. The indirect call is
the existing sink callback; `fprintf`/`fflush` remain only on trace-enabled
branches.

The former Write instruction word `0x2f79777b` and new `0x2f39797b` both decode
to custom-3 opcode `0x7b`, funct7 **23**, funct3 **7** (`xd=1`, both sources
present). The changed bits select a different CPU source register after register
allocation, not a different command argument. Release retains
`0x30e4f4fb`, funct7 **24**, funct3 **7**. Runtime full-record comparisons verify
the argument values:

```text
rs1 = slot | (generation << 8) | (column << 16)
Write rs2 = carrier
Release rs2 = 0
```

The original raw issue functions and encoders were not edited. No extra RoCC
instruction or Fence was inserted.

## Isolated Scale measurements

The H3-6B0 benchmark source and command stream were unchanged: DIM16, 36 complete
Write/Release batches per repetition, slots 0..255, columns 0..15, generations
1..36, carrier 0, and 147,456 commands per phase. Each route received its existing
warmup and seven measured repetitions with rotating route order.

The benchmark verified Linux CPU 1 as physical hart 0 and pinned itself there.
Both sink-only controls used that same physical hart. Guest `steady_clock` and
supported `rdcycle` were used, without timed printing or enabled raw tracing.
Every returned status and each batch's independent session-state checks passed:
zero live entries, zero outstanding work, and exact command counts. Hardware
Write/Release values, generations, and order were unchanged.

All intervals are **CPU-observed guest latency**, not pure FPGA execution time.
The software-only sink does not issue RoCC commands.

| Route | Phase | Before ns/command | After ns/command | After min..max ns | After MAD ns | Reduction |
|---|---|---:|---:|---|---:|---:|
| Serial hardware | Write | 37.842 | 37.835 | 37.808..37.977 | 0.027 | 0.02% |
| Serial hardware | Release | 33.813 | 33.203 | 33.190..33.271 | 0.014 | 1.81% |
| Paired hardware | Write | 318.515 | 101.969 | 101.705..102.505 | 0.264 | 67.99% |
| Paired hardware | Release | 306.322 | 93.343 | 92.787..93.777 | 0.434 | 69.53% |
| Paired software-only | Write | 322.571 | 107.320 | 107.157..107.456 | 0.047 | 66.73% |
| Paired software-only | Release | 312.683 | 98.016 | 97.914..98.124 | 0.061 | 68.65% |

Actual Paired hardware savings were **216.546 ns/Write** and
**212.979 ns/Release**, larger than the earlier approximately 109 ns
environment-sensitive projection. That projection measured only the component
responsive to environment scanning, not the entire library call and associated
generated-code cost. The experiment does not partition those remaining
instruction costs individually.

Matched sink-only controls, seven repetitions each:

| Phase | Before normal | Before empty | Before normal-minus-empty | After normal | After empty | After normal-minus-empty |
|---|---:|---:|---:|---:|---:|---:|
| Write, ns/command | 321.445 | 212.280 | 109.165 | 107.374 | 108.514 | -1.139 |
| Release, ns/command | 311.537 | 202.542 | 108.995 | 98.070 | 98.090 | -0.020 |

The positive approximately 109 ns environment-sensitive penalty disappeared.
The small negative post-change differences are not negative lookup costs:
session initialization is outside the command timers, and sequential runs still
have minor layout/cache/timing variation. Empty/normal post-change MADs were
0.061/0.088 ns for Write and 0.068/0.054 ns for Release.

Normal guest environments had 11 entries in both studies, although the suite's
noninteractive shell replaced exported `PS1` with `OLDPWD` and changed `SHLVL`.
The matched post-change controls share the same suite environment. All comparisons
retain the unchanged stream and fixed physical hart; no claim requires every
historical environment string or address to be identical.

## End-to-end Decode and FFN

The existing input generator, seed `20261023`, shapes, seven repetitions,
independent software Golden, cache warmups, rotating/reversing route order,
and bitwise F32 checks were retained. The existing CLI runs cold and steady
phases and includes Conservative as a control. The tables below use only the
**unprofiled seven-repeat steady phase**.

Every cold and steady sample reported `ok=1`, and both cases ended with
`POTAL_BENCH PASS ... failures=0`. Quick smoke also passed all routes with
`failed_nodes=0`. The RISC-V trace regression passed between the isolated
benchmark and quick smoke.

| Workload | Route | H3-6A median ms | H3-6B1 median ms | After min..max ms | After MAD ms |
|---|---|---:|---:|---|---:|
| Decode | Serial | 9.317 | 9.294 | 8.888..11.379 | 0.065 |
| Decode | Paired | 41.417 | 15.622 | 15.405..15.684 | 0.045 |
| Decode | Conservative | 42.245 | 15.808 | 15.786..15.845 | 0.007 |
| FFN | Serial | 53.724 | 53.572 | 52.995..53.712 | 0.072 |
| FFN | Paired | 209.738 | 83.242 | 82.956..83.727 | 0.139 |
| FFN | Conservative | 210.708 | 84.148 | 83.851..84.300 | 0.030 |

| Workload | Before Paired/Serial | After Paired/Serial | Paired speedup | Paired time reduction |
|---|---:|---:|---:|---:|
| Decode | 4.445 | 1.681 | 2.651x | 62.28% |
| FFN | 3.904 | 1.554 | 2.520x | 60.31% |

Serial median changes were -0.25% Decode and -0.28% FFN. The reduced Paired
time is measured, not extrapolated from isolated command latency.

Unprofiled Scale counters, with ns/command computed over Writes plus Releases:

| Workload | Route | Before Scale ms | After Scale ms | Before ns/command | After ns/command |
|---|---|---:|---:|---:|---:|
| Decode | Serial | 3.141 | 3.217 | 47.556 | 48.707 |
| Decode | Paired | 33.326 | 8.266 | 504.572 | 125.151 |
| Decode | Conservative | 33.891 | 8.267 | 513.127 | 125.167 |
| FFN | Serial | 14.201 | 14.363 | 48.153 | 48.703 |
| FFN | Paired | 159.183 | 36.829 | 539.764 | 124.881 |
| FFN | Conservative | 158.901 | 36.876 | 538.808 | 125.041 |

The model-level Scale timings include their existing caller boundaries and
working-set effects; they are not identical to the standalone C11 loop.
The old H3-6A baseline predates its final activation-timer instrumentation.
That existing instrumentation was preserved, not removed to improve this result.

### Command and synchronization counts

Every counter available in the original steady logs matched in all seven new
steady samples for all three routes, not just in their medians:

| Workload/route | Writes/Releases | Serial Main/Residual | Paired Main/Residual | Index config/write | Paired readbacks/tiles | Paired Fences/Drains |
|---|---|---|---|---|---|---|
| Decode Serial | 33024/33024 | 6/4 | 0/0 | 0/0 | 0/0 | 0/0 |
| Decode Paired | 33024/33024 | 0/0 | 12/60 | 60/978 | 12/96 | 1/73 |
| Decode Conservative | 33024/33024 | 0/0 | 12/60 | 60/978 | 12/96 | 145/217 |
| FFN Serial | 147456/147456 | 22/22 | 0/0 | 0/0 | 0/0 | 0/0 |
| FFN Paired | 147456/147456 | 0/0 | 44/264 | 264/6314 | 44/384 | 1/309 |
| FFN Conservative | 147456/147456 | 0/0 | 44/264 | 264/6314 | 44/384 | 617/925 |

The new logs also report Serial Main/Residual Fences: 6/4 for Decode and 22/22
for FFN, with zero Serial Fences on Paired routes. Those individual Serial Fence
fields were not printed in the original baseline log. Fallback stripes and
producer capacity waits remained zero; each graph completed one semantic stripe.

### Separate detailed operation profiles

Detailed profiles ran afterward, with `POTAL_PAIRED_PROFILE=1` and three
repetitions per case. As in H3-6A, these operation-group values use the median of
steady iterations 2 and 3. Their Paired run IDs are 21 and 24; the two-value
median is their arithmetic mean. Profile logging is excluded from the baseline
compute-time comparison above.

| Paired operation | Decode before ms | Decode after ms | FFN before ms | FFN after ms |
|---|---:|---:|---:|---:|
| Scale Write | 19.125 | 4.5985 | 90.331 | 20.6555 |
| Scale Release | 16.888 | 3.6745 | 83.883 | 16.1965 |
| Index | 0.702 | 0.2440 | 4.633 | 1.5555 |
| Main issue | 0.204 | 0.1560 | 0.915 | 0.7285 |
| Residual issue | 0.364 | 0.1275 | 1.784 | 0.6205 |
| Readback | 0.059 | 0.0190 | 0.292 | 0.1455 |
| Fence | 0.002 | 0.0005 | 0.002 | 0.0010 |

All paired raw commands benefit from the cached selection, so improvements are
not limited to Scale. These are CPU-observed issue/group intervals; Release may
include hardware admission wait, and stages may overlap. Do not sum these
separate profile medians into a graph execution time.

## Remaining overhead and next recommendation

Unloaded Paired hardware still exceeds Serial by approximately **64.134
ns/Write** and **60.140 ns/Release**. The success-only sink remains slightly
slower than hardware, consistent with callback/record construction cost.
Thus substantial remaining unloaded overhead is software, not evidence of a
large hardware drain stall.

Within FFN, Paired Scale is still 36.829 ms versus Serial 14.363 ms, a
22.466 ms stage gap; the graph compute gap is 29.670 ms. These are overlapping
timers rather than a strict additive attribution, but Scale remains a measured
priority. Separate FFN profiles also retain approximately 8.189 ms route
planning and 5.538 ms executor validation.

The next narrow candidate is to measure outlining enabled trace printing into
a cold helper so disabled tracing can retain a smaller issue path and fewer
register-preservation requirements. Keep the checked APIs, exact command stream,
sticky failures, and all synchronization intact. Measure generated instruction
count/spills and the same hardware/sink benchmark before claiming a benefit.
This experiment does not isolate the individual remaining costs of validation,
liveness stores, raw dispatch, or `paired_drained`; it does not justify weakening
checks or changing drain semantics. No further optimization was implemented.

## Reproduction and evidence

Before artifacts are preserved at `/tmp/h3-6b1-before/`, including the original
`potal.c`, DIM16 benchmark, smoke executable, and backend library.
Before source SHA-256:
`c08c0e134780767fb77219e547e52dc144c10f5b2804b6384651991475ef572d`.

The existing host and cross build directories were reused:

```sh
source /home/alveo/miniforge3/etc/profile.d/conda.sh
cd /home/alveo/firesim/target-design/chipyard
source env.sh
cd generators/potal/software/workspace/llama.cpp-potal
cmake -S . -B build-potal-paired-host
cmake -S . -B build-potal-h3-6a-d16
cmake --build build-potal-h3-6a-d16 \
  --target potal-scale-path-bench potal-paired-smoke potal-trace-test -j8
```

The pre-existing missing `chipyard/.conda-env` activation message remains; the
configured compiler under `/home/alveo/firesim/.conda-env` worked. Both builds
completed successfully.

Guest commands, in order, using the existing physical-hart mapping:

```sh
unset POTAL_RAW_TRACE POTAL_PAIRED_PROFILE
./potal-scale-path-bench --cycles
taskset -c "$hart0" env -i /root/potal-scale-path-bench --software-only --cycles
taskset -c "$hart0" /root/potal-scale-path-bench --software-only --cycles
./potal-trace-test
# In the existing benchmark bin directory:
./potal-paired-smoke --quick
./potal-paired-smoke --bench --case decode_1x768x768 --repeat 7
./potal-paired-smoke --bench --case gpt2_ffn_up_16x3072x768 --repeat 7
POTAL_PAIRED_PROFILE=1 ./potal-paired-smoke --bench --case decode_1x768x768 --repeat 3
POTAL_PAIRED_PROFILE=1 ./potal-paired-smoke --bench --case gpt2_ffn_up_16x3072x768 --repeat 3
```

All stages were success-gated. Raw evidence is
`/tmp/h3-6b1-run/uartlog`, with `__H36B1_*_BEGIN__` / `*_PASS__` markers:

```text
__H36B1_SUITE_PASS__
__H36B1_SUITE_EXIT__=0
```

The suite script is `/tmp/h3-6b1-guest-suite.sh`. A separate copied image,
`/tmp/h3-6b1-run/rootfs.img`, received the new artifacts; prior images were not
overwritten. The existing FPGA driver/bitstream were reused. Host, staged-image,
and running-guest SHA-256 values matched:

| Artifact | SHA-256 |
|---|---|
| `potal-scale-path-bench` | `53bcacc03f265c3e2526bdb3e75556511eef638a4785f36832722f0bb165b3af` |
| `potal-paired-smoke` | `b7320d4d2f0db17ce8e64927fa4d3448f32ea3eb7c8e5570a7ed2701415e90a3` |
| `libggml-potal.so` | `0dbc924b211ce46c1b525915a6f314b2792862bce601b60b8b699b8f612dd8e0` |

The unchanged DIM16 parameter-header hash is
`b4cd8b9efef9f3cd9b598ae3cc84af6478f83e628ed73d94a77fb1c452df1fc7`;
the H3-5B RTL revision remains
`dc2f52a5bd0236bc4e02a8d1109151945661755b`.

After all stages passed, guest poweroff completed cleanly and the simulator
exited with status zero.
