# W4A4 issuer CPU and command diagnostics

The W4A4 issuer uses the existing ISA and one accelerator owner. These changes
add no hardware, command types, speculative execution, or extra issuer core.

## Software defaults and optimizations

Scale-window issue is enabled by default. `GGML_POTAL_WINDOW_SCALE_ISSUE=OFF`
or `0`, or `request.window_scale_issue=false`, selects scalar lowering.
The window API still sends a WRITE/RELEASE for every original column; it
hoists invariant CPU checks and batches slot bookkeeping, not RoCC commands.
Managed modes and the existing serialized diagnostics retain their contracts.

Ordinary Index installation uses a checked window call: one existing
INDEX_CONFIG followed by every existing ordered INDEX_WRITE. The public API
checks its input extent once, preserves per-entry range checks and successful
prefixes, and publishes a valid Index only after the last successful write.
Managed/reentrant paths use their existing scalar admission checks. The W4
Index Queue remains unsupported.

Residual staging has a prepared op-to-window map. Strict jobs also prepare
immutable LOOP descriptors on the producer. The issuer binds generation and
Scale aliases and retains the checked encoder and session-dependent
resident-B, Index, outstanding-reader and failure checks. This is not raw
command replay. Prepared descriptors and staging borrow the original Main
operand and immutable weight for the job's lifetime.

## Enable the optional diagnostic build

Append `-DPOTAL_ISSUER_DIAGNOSTICS=ON` to the existing W4A4 configure command,
keeping its hardware parameters, algorithm, rounding and model settings.
GNU/Clang and RV64, x86, or AArch64 counter support are required.

The default build compiles the hooks out. An instrumented build with the
runtime option off still has hook-call overhead; measure that separately.

Before starting a new process:

```sh
export POTAL_ISSUER_CPU_PROFILE=1
export POTAL_ISSUER_CPU_SAMPLE_LOG2=10
export POTAL_PAIRED_TIMING=0
unset POTAL_TIMELINE POTAL_RAW_TRACE POTAL_SCALE_CPU
```

Run the normal runner with its original model/arguments. The existing
hart-0 owner flushes `[POTAL_ISSUER_CPU]` records after its task returns.
There is no printing, allocation, CLOCK_MONOTONIC, or scheduling call in
the new per-command hooks.

`SAMPLE_LOG2` is an integer from 0 through 20. Zero samples every scope;
10 samples one in 1024 calls of each scope, beginning with its first call.
Environment configuration is cached per thread. Use a new process when
changing options. Sampling is systematic, not a random unbiased estimator;
report the sample size rather than extrapolating a few samples to total
model time.

The original timeline remains available for producer preparation and job
handoff timing. Use a separate run for it: wait markers call the monotonic
clock, whose RISC-V VDSO CPU fences can stall on RoCC busy. Leaving the
legacy per-op timer enabled can likewise move waits into a clock read.
The new diagnostic reports both legacy timing/trace flags.

## Counter sources and units

| `source` | `ticks` | `instructions` |
|---|---|---|
| `riscv_rdcycle` | Existing hart cycle counter | Existing `rdinstret` |
| `x86_tsc` | TSC ticks, **not CPU cycles or nanoseconds** | Unavailable; `instructions_valid=0` |
| `aarch64_cntvct` | Architectural timer ticks, **not CPU cycles** | Unavailable; `instructions_valid=0` |

Keep a sample on the same hart/CPU. Production uses the existing pinned
issuer; pin a CPU-only benchmark to a verified idle CPU. Hart instructions
can include interrupts. Do not convert cycles to time by assuming 1 GHz.

`empty_ticks` and `empty_instructions` are the minimum of 32 adjacent
counter-read pairs. They describe counter-read overhead, not all hook or
sampling overhead. The collector never subtracts them automatically.

## Recorded boundaries

Request, Main-op and Residual-op scopes cover the C++ owner work. Checked
API scopes cover Scale windows/scalars, Index windows/scalars, LOOP issue,
readback and FENCE. Nested scopes are inclusive; do not add their totals.
Raw records are counted once even when several sampled parent scopes are
active.

Each sampled scope has this partition:

```text
scope entry
  local software before first raw instruction or CPU fence       pre
  raw command call and response                                 raw
  local software between raw commands / explicit CPU fences      between
  explicit publication or completion-ordering CPU fence          cpu_fence
  local software after the final raw command / CPU fence         post
scope return
```

Within one row:

```text
ticks = pre_ticks + between_ticks + post_ticks + raw_ticks + cpu_fence_ticks
```

The corresponding instruction fields have the same identity on RV64.
Local software includes validation, encoding, metadata/loop traversal,
bookkeeping, cache effects and instrumentation. It is not a measurement of
pure arithmetic execution. In a window, validation and publication between
commands belong to `between`, not exclusively `pre` or `post`.

| Record | Interpretation |
|---|---|
| `api=request` | Entire strict/unprepared C++ executor request, including setup and its own cleanup |
| `api=main_op`, `residual_op` | Descriptor construction/binding and issue for that logical op |
| Other `api=` rows | Checked API interval; counts/samples are per scope |
| `funct=23`, `26`, `8`, etc. | Sampled Scale, Index, LOOP and other raw round trips |
| `funct=25` | Raw checked accelerator FENCE round trip |
| `funct=24` | Scale RELEASE round trips, including any completion wait |
| `cpu_fence=publication` | Explicit input-memory publication fence |
| `cpu_fence=completion` | Explicit CPU ordering after a completion response |

**Raw round trip is not pure admission waiting.** It includes dispatch,
normal transfer/response cost, backpressure, and any interrupt during the
interval. It also measures the recording/emulation sink when one replaces
hardware. Existing hardware does not supply a ready/accept timestamp here.
FENCE/RELEASE include their normal handling as well as drain waiting.
These components are separated from surrounding software, not from each
other inside the hardware interface.

## CPU-only checks and measurements

Build these targets with the same W4 configuration:

```text
potal-index-window-test
potal-scale-window-test
potal-index-payload-test
potal-w4a4-golden-test
potal-issuer-profile-test          (diagnostic build)
potal-index-window-bench
potal-issuer-prepared-bench
```

The profile test drives real checked APIs and command sinks, including the
Scale direct path, invalid Index prefixes and answered raw failures. W4
payload/golden tests check commands, generation binding, Main/Residual
results, INT64 reconstruction and final F32 values. Existing W8-only tests
are not a substitute for these packed-W4 fixtures.

The Index benchmark compares scalar/window APIs with identical command
digests. The prepared benchmark has `staging`, `descriptor`, and `issuer`
modes. The first two isolate lookup/construction work on a real prepared
W4 job; they exclude encoding, session state and producer preparation.
Issuer mode executes the real strict path against a cheap recording sink;
it includes setup, checked emission and cleanup, but has **no accelerator
admission, DMA or compute wait**. DMA addresses are masked only in its
cross-job digest; functional regressions separately check the protocol.

Compare legacy timing ON/OFF and Scale-window ON/OFF as four separate
processes. For observer overhead compare:

1. Diagnostics compiled out.
2. Diagnostics compiled in, runtime off.
3. Runtime on with sample log2=10.
4. Runtime on with sample log2=0.

Alternate order, repeat trials, and report median and spread. Neither
these host measurements nor Nano dummy-RoCC replay establish FPGA speedup,
FPGA cycle inflation, full-model PPL or model latency.
