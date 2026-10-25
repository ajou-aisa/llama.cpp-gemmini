# H3-6B0: isolated Scale command overhead

Date: October 25, 2026. Hardware: existing H3-5B FireSim U280, DIM16.
No optimization, RTL change, FPGA rebuild, full model suite, commit, or push was performed.

## Result

**Paired Scale overhead persists without any GEMM outstanding. Software dominates this isolated experiment.**

The checked Paired hardware path takes 8.42 times the Serial Write latency and
9.06 times the Serial Release latency. The same Paired API feeding a success-only
software sink is slightly slower than the hardware path. Thus outstanding GEMMs
and hardware Release admission cannot explain the large unloaded Paired penalty.

All numbers below are **CPU-observed guest intervals**, not pure FPGA execution
time. The software-only rows issue no hardware commands.

## Benchmark and measurement contract

New target: `potal-scale-path-bench`, with C11 caller
`ggml/src/ggml-potal/tests/potal-scale-path-bench.c` and a small C++ guest
`std::chrono::steady_clock` bridge, `tests/potal-scale-clock.cpp`.

Routes:

* `serial`: checked `potal_scale_write` and `potal_scale_release`.
* `paired`: checked `potal_paired_scale_write` and
  `potal_paired_scale_release` in an explicitly opened/closed session.
* `paired_software_only`: the same session API with the existing command sink
  callback returning `POTAL_OK`, without accessing hardware.

Each repetition has 36 complete batches. Each batch writes slots 0 through 255,
columns 0 through 15 in ascending slot/column order, with carrier 0 and generation
`batch + 1` (1 through 36). It then releases the same entries in the same order.
All routes use identical arguments and command order. Each phase therefore has
147,456 commands per repetition, matching the H3-6A FFN per-phase count.
No Main or Residual GEMM is issued.

Every returned C11 status is checked. After every Paired batch, an untimed query
checks zero live Scale slots, no failure, zero outstanding loops/readbacks, and
the exact cumulative command count. Successful checked Serial releases establish
that every Serial batch released its entries. Session close is checked.
One initial hardware Fence is outside all timings; no Fence behavior or RoCC
encoding was changed.

Each route receives one full untimed warmup, followed by seven measured
repetitions with rotating route order. Write and Release loops have separate
timers. Allocation, session open/close, queries, sorting, and printing are outside
the timed loops. Status branches, loop control, and clock/counter sampling remain
included. There is no per-command printing; `POTAL_RAW_TRACE` must be unset.

The measured process verified Linux CPU 1 maps to physical hart 0 and pinned
itself there. Linux CPU 0 is not assumed to be hart 0. Guest `rdcycle` was
supported and enabled using `--cycles`. Guest steady-clock readings have
microsecond granularity; each reported interval sums 36 batch-phase intervals.
Timer overhead was not subtracted.

## Measured Serial, Paired, and software-only latency

MAD is median absolute deviation across the seven per-repetition ns/command
values. Cycles are the median guest cycles/command, not FPGA clock cycles.

| Route | Phase | Median elapsed ms | Median ns/command | Min ns | Max ns | MAD ns | Median cycles/command |
|---|---|---:|---:|---:|---:|---:|---:|
| Serial hardware | Write | 5.580 | 37.842 | 37.808 | 37.937 | 0.034 | 37.686 |
| Serial hardware | Release | 4.986 | 33.813 | 33.807 | 33.929 | 0.007 | 33.667 |
| Paired hardware | Write | 46.967 | 318.515 | 317.641 | 363.376 | 0.610 | 318.324 |
| Paired hardware | Release | 45.169 | 306.322 | 305.718 | 337.972 | 0.604 | 306.157 |
| Paired software-only | Write | 47.565 | 322.571 | 322.157 | 346.544 | 0.414 | 322.412 |
| Paired software-only | Release | 46.107 | 312.683 | 311.049 | 338.874 | 0.990 | 312.517 |

The median of each repetition's Write-plus-Release sum is 10.578 ms Serial,
92.141 ms Paired hardware, and 93.679 ms software-only. A sum of separately
computed phase medians need not equal the median of the sums.

All primary samples, in guest milliseconds:

| Repetition | Serial Write | Serial Release | Paired Write | Paired Release | Sink Write | Sink Release |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 5.576 | 5.000 | 46.887 | 45.084 | 47.518 | 46.161 |
| 2 | 5.594 | 4.985 | 52.506 | 49.285 | 51.100 | 49.969 |
| 3 | 5.592 | 4.986 | 46.838 | 45.356 | 47.731 | 46.053 |
| 4 | 5.593 | 4.986 | 46.877 | 45.080 | 47.504 | 45.866 |
| 5 | 5.576 | 4.989 | 53.582 | 49.836 | 47.810 | 46.253 |
| 6 | 5.575 | 5.003 | 46.967 | 45.095 | 47.520 | 45.955 |
| 7 | 5.580 | 4.985 | 46.972 | 45.169 | 47.565 | 46.107 |

The slower repetitions remain included. Architectural cycles corroborate wall
time; neither measurement distinguishes CPU execution from CPU stalls.

## Environment lookup and generated call paths

`ggml/src/ggml-potal/potal.c` has these relevant paths:

* `potal_scale_write` / `potal_scale_release`: key/carrier validation followed by
  one status-returning RoCC instruction.
* `potal_paired_scale_write`: readiness and argument checks, per-column liveness
  and generation checks, raw issue, then liveness/generation stores.
* `potal_paired_scale_release`: readiness, argument/liveness/generation checks,
  raw issue, column clear, then `paired_drained`.
* `paired_raw`: **calls `getenv("POTAL_RAW_TRACE")` for every command on RISC-V,
  even when tracing is unset**. It retains trace branches and values needed by
  the diagnostic printing path.
* `paired_raw_untraced`: increments the command count and selects the sink or
  hardware command. GCC specializes/inlines this path into the Scale functions.
* `paired_drained`: clears aggregate and per-pair outstanding counters and
  increments drains on every successful Release, even when already idle.

The per-entry safety checks are part of the checked API contract; this experiment
does not justify removing them. Repeated environment scanning is unnecessary for
a process whose diagnostic selection remains constant during a session.
Repeated zero stores in `paired_drained` are another candidate, but their
individual cost was not isolated.

The actual RISC-V executable was disassembled. Both checked Paired Scale
functions contain a `jal getenv@plt` before issue. Their prologues reserve
112 bytes (Write) and 96 bytes (Release) and save multiple registers. Serial
Write is a leaf function without that stack frame or environment lookup.
The compiler inlines key validation and specializes the hardware issue; there is
not an unoptimized generic dispatcher call on every Scale command.

Compiler: `riscv64-unknown-linux-gnu-gcc` 12.2.0, revision
`g2ee5e430018`; default ISA `rv64imafdc`, ABI `lp64d`, tuning `rocket`.
Both Serial and Paired API functions are in the same Release `potal.c`
translation unit with `-O3 -DNDEBUG -std=c11 -fPIC -fvisibility=hidden`.
The benchmark C caller also uses `-O3 -DNDEBUG -std=c11`; the clock bridge uses
the existing C++17 build flags. There is no Serial/Paired optimization-level
mismatch and no new optimization flag.

An additional software-only pass used the same binary, command stream, seven
repetitions, cycle sampling, and physical hart, but an empty environment:

```sh
taskset -c 1 env -i /root/potal-scale-path-bench --software-only --cycles
```

| Empty-environment software-only phase | Median ns/command | Min ns | Max ns | MAD ns |
|---|---:|---:|---:|---:|
| Write | 212.280 | 211.358 | 217.943 | 0.183 |
| Release | 202.542 | 202.162 | 209.886 | 0.176 |

This pass also returned zero, reported PASS, and left no live entries. Removing
the normal guest's 11 environment entries reduced the sink latency by about
110 ns/command relative to the primary run. This is a software effect: neither
sink run accesses RoCC hardware. It supports targeting environment scanning,
rather than assuming hardware admission is the cause of the whole gap.
It does not measure the entire cost of `getenv` or prove that all remaining
software cost is removable.

A matched normal-environment control then ran the same sink-only invocation
under `taskset -c 1`, without `env -i`, and passed all seven repetitions:

| Matched normal-environment software-only phase | Median ns/command | Min ns | Max ns | MAD ns |
|---|---:|---:|---:|---:|
| Write | 321.445 | 321.248 | 321.940 | 0.163 |
| Release | 311.537 | 311.327 | 311.720 | 0.115 |

Relative to the empty-environment pass, these matched medians differ by
109.165 ns/Write and 108.995 ns/Release. This confirms the effect without
comparing sink-only execution against the rotating three-route process.
The environment comparison is sequential, not a randomized causal partition of
every instruction cost; the unchanged binary, fixed hart, and small dispersion
make repeated environment lookup the supported explanation for this component.

## Comparison with H3-6A and hardware-wait assessment

H3-6A unprofiled FFN Scale time was 14.201 ms Serial versus 159.183 ms Paired
for 147,456 Writes and 147,456 Releases: approximately 48.2 versus 539.8
ns/command averaged across phases. The isolated Paired phase medians average
312.4 ns/command versus 35.8 ns Serial. The isolated gap is approximately
81.57 ms at that command count, so a large penalty exists without any GEMM.
These are different workloads, environments, and timing boundaries; the
81.57 ms is not a measured model speedup.

The separate, logging-perturbed H3-6A profile reported FFN Write 90.331 ms and
Release 83.883 ms, or 612.6 and 568.9 ns/command respectively. Both phases
were slower there, not just Release. Those profile figures must not be combined
as if they came from the unprofiled baseline.

Answers to the four questions:

1. **Does unloaded Paired overhead persist? Yes**, by 280.7 ns/Write and
   272.5 ns/Release relative to Serial phase medians.
2. **Is software-only overhead significant? Yes.** It reproduces the whole
   unloaded latency scale and exceeds Paired hardware by approximately 4.1
   ns/Write and 6.4 ns/Release. The sink has a callback/record construction cost,
   so subtracting it does not yield a pure hardware latency.
3. **Does substantial Release wait occur only with GEMMs outstanding?**
   No substantial extra unloaded Release wait is visible. In the inspected
   `Controller.scala`, `releaseSafe` and `fenceDrained` both equal `allDrained`,
   which includes the older Gemmini work and writeback drain conditions.
   Thus older GEMMs can delay Release. This experiment does **not** quantify
   that loaded delay or establish that it explains H3-6A's remaining excess.
   Doing so requires a matched outstanding-GEMM experiment, outside this
   no-GEMM benchmark. Write also slowed in H3-6A, and environment sensitivity
   provides a concrete competing explanation.
4. **Compiler or C11 work?** The build is already optimized uniformly.
   Generated Paired code performs substantially more CPU work, including
   measured environment-sensitive diagnostic lookup. The relative shares of
   liveness bookkeeping, call preservation, and drain stores are not isolated.

## Minimal H3-6B1 proposal; not implemented

First move `POTAL_RAW_TRACE` selection out of the per-command path into session
initialization and use the stored selection for Paired raw issue. Keep the
existing ENTER/EXIT records, status handling, command sequence, RoCC encoding,
Fence/Release drain semantics, and checked entry validation unchanged.

The matched environment-sensitive component is about 109 ns/command in this
guest, or approximately 32.17 ms across 294,912 Scale commands. This supports a
material expected benefit from eliminating repeated scanning. It is a projection,
not a guaranteed FFN saving. The approximately 81.57 ms total unloaded Paired
excess is an upper bound for eliminating all excess at this stream size, not
the expected benefit of this one change. Even an empty environment leaves
substantial checked-session overhead.

Define and test trace selection at session open. If existing diagnostics require
environment changes to take effect mid-session, preserve that capability through
an explicit diagnostic control rather than silently changing the contract.
Do not introduce a bulk unchecked Scale API or remove generation/liveness checks
until measurements after this first change justify a second step.

Correctness risks and required regressions:

* Preserve live-column tracking, padded windows, generation reuse/wrap, wrong
  generation rejection, duplicate writes, and release-miss rejection.
* Preserve sticky hardware errors, cleanup, safe close, ownership, command/drain
  counters, and outstanding-work clearing only after accepted drains.
* Preserve H3-3/H3-4 diagnostic trace records when enabled, including failures.
  Test selection across successive sessions and its chosen mid-session contract.
* Run focused C11/API, Paired plan/schedule/unit/auto/execution, and stripe
  regressions. Compare complete sink command records before/after; timing alone
  cannot prove command equivalence.
* Repeat this seven-repetition hardware/sink benchmark with normal and empty
  environments. Require lower Paired timing, unchanged status/count/cleanup
  behavior, and stable Serial controls.
* Then run targeted FireSim correctness and H3-6A decode/FFN profiling with
  unchanged inputs and bitwise Goldens. Confirm command counts, diagnostic
  behavior, and loaded Release/Fence correctness before claiming model speedup.

## Validation, build, and evidence

The DIM16 RISC-V target and host target built successfully. The host success-only
sink run passed. Six focused binaries passed with exit code zero:
`potal-api-test`, `potal-paired-execution-test`, `potal-paired-unit-test`,
`potal-paired-schedule-test`, `potal-paired-plan-test`, and
`potal-paired-auto-test`.

The existing host build has no registered CTest tests; its CTest invocation
reported "No tests were found". The six executables were subsequently run
directly, not counted as passing based on that CTest result. Existing backend
build warnings about `POTAL_SCALE_ENTRIES` redefinition remain unchanged.
Editor/LSP diagnostics could not resolve the generated include paths and
standard headers; the actual host and cross compilers built the new target.

Reproduction using the existing build configuration:

```sh
source /home/alveo/miniforge3/etc/profile.d/conda.sh
cd /home/alveo/firesim/target-design/chipyard
source env.sh
cd generators/potal/software/workspace/llama.cpp-potal
cmake -S . -B build-potal-h3-6a-d16
cmake --build build-potal-h3-6a-d16 --target potal-scale-path-bench -j8
```

This machine's Chipyard `env.sh` reports a missing
`chipyard/.conda-env`; the existing compiler paths under
`/home/alveo/firesim/.conda-env` were usable, and both builds exited zero.

The executable was copied into a separate rootfs image at
`/tmp/h3-6b0-run/rootfs.img`, leaving the existing image untouched.
The existing FPGA driver and hardware were reused without programming or
rebuilding. An initial boot was shut down without issuing Scale commands to
stage the corrected physical-hart mapping. The measured boot ran only this
benchmark, including its software-only controls.

Guest primary command:

```sh
unset POTAL_RAW_TRACE
/root/potal-scale-path-bench --cycles
```

Guest and host executable SHA-256:
`b4974fe0cd007ecd4700c961080109cd861ec6ef1df64b3c8354675338be07f6`.
Parameter-header SHA-256:
`b4cd8b9efef9f3cd9b598ae3cc84af6478f83e628ed73d94a77fb1c452df1fc7`.
The build manifest records DIM16, 256 Scale slots, RTL repository head
`dc2f52a5bd0236bc4e02a8d1109151945661755b`, and software head
`0d3363590343af24c4ac73dedf8edd05901c4517`, with pre-existing working changes.

Primary raw evidence and control passes are in
`/tmp/h3-6b0-run/measured-uartlog`, delimited by `__H36B0_BEGIN__`,
`__H36B0_EMPTY_BEGIN__`, `__H36B0_CONTROL_BEGIN__`, and their exit markers.
Both controls reported PASS and exit zero.
Primary completion:

```text
POTAL_SCALE_BENCH PASS no_gemms=1 live_entries=0
__H36B0_EXIT__=0
```

H3-3/H3-4 diagnostic implementations and H3-6A profiling were preserved.
