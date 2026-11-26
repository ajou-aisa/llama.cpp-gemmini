# H3-6B3: checked Paired command dispatch

Date: October 25, 2026. Branch: `potal`.
Baseline: current uncommitted H3-6B2.
Hardware configuration: fixed H3-5B FireSim U280 DIM16, unchanged.

## Decision

**Reject the specialization at the Phase 3 generated-code gate. Keep B2.**

One minimal internal hardware specialization was implemented and compiled.
It moved optional Sink/Trace dispatch outside the Scale API bodies, reducing
reserved Write/Release frames from 80/64 bytes to 48/32 bytes.
It did **not** reduce the hardware success path's executed instructions,
branches, saved registers, function calls, or library calls:

| Valid hardware path | B2 instructions | Experiment instructions | B2 branches | Experiment branches |
|---|---:|---:|---:|---:|
| Write, empty slot | 62 | 62 | 10 | 10 |
| Write, live slot, new column | 69 | 69 | 12 | 12 |
| Release | 64 | 64 | 10 | 10 |

Both versions execute zero function/library calls on these paths, preserve the
same registers, and issue exactly one RoCC instruction. The smaller reservation
does not remove any of the actual save/restore operations.

For this gate, moving cold record construction and shrinking reserved stack
space, without reducing ordinary hardware-path work, is insufficient to justify
36 additional source lines and another dispatch boundary. This is a
generated-code rejection, **not a measured latency regression or a claim that
stack/cache layout cannot affect timing**.

The experimental changes were removed using an inverse patch limited to the
three edited hunks. The restored `potal.c` and rebuilt Scale benchmark match the
saved B2 artifacts byte-for-byte. Existing uncommitted H3-3/H3-4/H3-6A/B0/B1/B2
work remains in place.

Per the requested gate, no B3 Host regression run, Sink-stream comparison,
FireSim benchmark, quick smoke, or Decode/FFN run followed this rejection.
No simulator or guest was started for B3, so no guest shutdown was needed.
No second optimization was attempted.

## 1. Baseline disassembly analysis

### Compiler and artifact preservation

The current B2 source and executables were saved before editing:

```text
/tmp/h3-6b3-before/potal.c
/tmp/h3-6b3-before/potal-scale-path-bench
/tmp/h3-6b3-before/potal-paired-smoke
/tmp/h3-6b3-before/potal-trace-test
/tmp/h3-6b3-before/potal-trace-test-host
/tmp/h3-6b3-before/libggml-potal.so
/tmp/h3-6b3-before/disassembly.txt
/tmp/h3-6b3-before/path-evidence.json
```

The compiler is the existing
`/home/alveo/firesim/.conda-env/riscv-tools/bin/riscv64-unknown-linux-gnu-gcc`,
GCC 12.2.0 (`g2ee5e430018`). The same configured DIM16 build directory and
compiler defaults were reused; there was no CMake reconfiguration or change to
architecture/ABI/tuning flags.

The C API target's existing flags are:

```text
-Wno-format -Wno-unused-variable -Wno-unused-function
-O3 -DNDEBUG -std=c11 -fPIC -fvisibility=hidden
-Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes
-Werror=implicit-int -Werror=implicit-function-declaration
-Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wdouble-promotion
```

Includes still select the existing `include` and DIM16 `generated` directories.
Both experimental and restoration builds completed with exit status zero.

### Requested functions

| Function | B2 emitted form | Static instructions | Function bytes | Frame |
|---|---|---:|---:|---|
| `potal_paired_scale_write` | standalone | 123 | 306 | 80 B |
| `potal_paired_scale_release` | standalone | 109 | 268 | 64 B |
| `paired_raw` | fully inlined; no symbol | not separately countable | not applicable | not separate |
| `paired_raw_untraced` | fully inlined; no symbol | not separately countable | not applicable | not separate |
| `paired_raw_traced` | separate cold helper | 117 | 324 | 96 B |
| `paired_drained` | inlined into callers; no symbol | counted in caller | not applicable | not separate |

The baseline symbol table contains `paired_raw_traced` at `0x10ce0` but no
standalone `paired_raw`, `paired_raw_untraced`, or `paired_drained`, including
compiler-renamed variants.

The hardware Scale path already has a constant funct and one direct RoCC
instruction. It does not execute the generic dispatch switch or an indirect
call. The source-level existence of `paired_raw_untraced` is therefore not
evidence of an executed function-call cost.

### Path-count method and attribution

The counted paths use a valid nonfailed session, valid DIM16 key, carrier 0,
no Sink, cached Trace OFF, and returned hardware status `POTAL_OK`.
Write distinguishes an initially empty slot from a same-generation live slot
with a previously unwritten column.

Counts include prologue/epilogue, compressed instructions counted once, one RoCC
instruction, and the chosen branch edges. They are static control-flow walks,
not measured `instret` values. Other carriers, invalid inputs, injected errors,
Trace, and Sink paths can execute different instructions.

The following categories are disjoint assignments of those selected
instructions. Validation includes key encoding and the live-slot
generation/duplicate checks; the live-column category includes addressing,
the initial presence test, mask preparation, and successful state stores.
Register/return includes argument-preservation moves and the return itself.

| Associated work | Write empty | Write live | Release |
|---|---:|---:|---:|
| Key/carrier validation, generation/duplicate checks, argument encoding | 20 | 27 | 21 |
| Live-column addressing, presence test, mask/state bookkeeping | 13 | 13 | 12 |
| Command counter and drain-state updates | 3 | 3 | 9 |
| Optional Sink selection | 2 | 2 | 2 |
| Optional Trace selection | 2 | 2 | 2 |
| Register preservation, frame adjustment, argument preservation, return | 17 | 17 | 13 |
| RoCC issue and hardware-status normalization/branching | 5 | 5 | 5 |
| **Total** | **62** | **69** | **64** |

These counts do **not** assign nanoseconds to the categories. Loads, stores,
branches, cache state, pipeline stalls, and hardware admission have different
costs; this experiment did not measure those contributions separately.

### Optional Sink and Trace control flow

Baseline Write:

```text
129de: lw    a2,20(s0)        # cached trace selection
129f2: bnez  a2,12a80        # enabled trace branch
129f4: ld    a4,48(s0)       # commands
129f6: ld    a2,0(s0)        # sink.cmd
129f8: addi  a4,a4,1
129fa: sd    a4,48(s0)
129fc: beqz  a2,12a38        # direct hardware when no Sink
...
12a0e: jalr  a2              # Sink only
...
12a38: .4byte 0x2ed7f57b     # SCALE_WRITE
...
12a88: jal   ra,10ce0        # paired_raw_traced, Trace only
```

Baseline Release:

```text
12aec: lw    a4,20(s0)
12aee: bnez  a4,12b8a        # Trace only
12af0: ld    a4,48(s0)
12af2: ld    a3,0(s0)
12af4: addi  a4,a4,1
12af6: sd    a4,48(s0)
12af8: beqz  a3,12b40        # direct hardware when no Sink
...
12b0a: jalr  a3              # Sink only
...
12b40: .4byte 0x30d7f57b     # SCALE_RELEASE
...
12b94: jal   ra,10ce0        # Trace only
```

Each Sink block constructs a 24-byte record in the Scale function's frame
before its callback. The ordinary hardware path does not construct the record.
Nevertheless, the frame reserves its space.

The cold Trace helper retains two `fprintf` and two `fflush` call sites, the
existing Sink callback and hardware dispatch, and an eight-register
`ra,s0..s6` save set. The disabled hardware path never enters it and performs
no diagnostic environment lookup or formatting.

### Validation, liveness, counting, and drain accounting

Write checks readiness, slot/generation/column ranges, carrier validity, and
live-slot generation/duplicate-column conditions before any issue. A successful
issue then updates the live mask and slot generation once.

Release checks readiness, key validity, live-column presence, and generation
before issue. Only a successful status clears the live column and invokes the
unchanged drain accounting.

In baseline Release, `paired_drained()` appears as six instructions:

```text
12b22: ld    a5,56(s0)       # drains
12b24: sd    zero,40(s0)     # outstanding loops/readbacks
12b28: sw    zero,140(s0)    # pair 0 outstanding
12b2c: addi  a5,a5,1
12b2e: sw    zero,188(s0)    # pair 1 outstanding
12b32: sd    a5,56(s0)
```

Together with the three command-counter instructions, these account for the
nine counting/drain instructions in the Release row. No change to this
function or to accepted-Release synchronization was proposed.

### Register preservation

Write saves/restores `ra,s0,s1,s2,s3,s4`: six registers. Release saves/restores
`ra,s0,s1,s2`: four registers. The normal path still executes those operations
even though its own taken route makes no function call. The optional callback/
trace paths must preserve values used by the shared successful state update.

This supported one narrow experiment: remove record construction and optional
dispatch from the public Scale bodies without duplicating the state updates.
It did not establish that the compiler would remove the save/restore work.

## 2. Proposed and rejected changes

The experiment added a RISC-V-only `noinline,cold` helper,
`paired_scale_raw_optional`, with GCC/Clang attribute guards. It routed optional
Scale dispatch through the existing `paired_raw`; it did not replace Trace or
Sink semantics.

After the existing public checks, each Scale operation selected direct hardware
only for `sink.cmd == NULL` and cached Trace disabled. The Write issue block was:

```c
#if defined(__riscv)
    potal_status raw_status;
    if (s->sink.cmd == NULL && !s->trace_enabled) {
        s->commands++;
        raw_status = hw_status(potal_rocc_scale_write(rs1, carrier));
    } else {
        raw_status = paired_scale_raw_optional(s, POTAL_FUNCT_SCALE_WRITE, rs1, carrier);
    }
    if (raw_status != POTAL_OK) {
        return paired_fail(s, raw_status, "scale_write");
    }
#else
    PAIRED_RAW(s, POTAL_FUNCT_SCALE_WRITE, rs1, carrier, 1, "scale_write");
#endif
```

Release used the same structure with `potal_rocc_scale_release(rs1, 0)` and
the original `"scale_release"` failure stage.

The direct branch incremented the command count once; the mutually exclusive
fallback used the existing dispatcher's increment. Successful live-mask and
generation updates, error handling, and `paired_drained()` remained shared,
not copied into separate success routes. Non-RISC-V behavior was unchanged.

The complete three-hunk, 36-added-line experiment is preserved in
`/tmp/h3-6b3-experiment/source.diff`. **None of it is retained in production.**
No public header, test source, GGML numerical code, or RTL was edited.

## 3. Actual generated-code gate

| Metric | B2 Write | Experiment Write | B2 Release | Experiment Release |
|---|---:|---:|---:|---:|
| Executed instructions, empty-slot Write / valid Release | 62 | 62 | 64 | 64 |
| Executed instructions, live-slot Write | 69 | 69 | not applicable | not applicable |
| Conditional branches, empty-slot Write / valid Release | 10 | 10 | 10 | 10 |
| Conditional branches, live-slot Write | 12 | 12 | not applicable | not applicable |
| Saved registers | 6 | 6 | 4 | 4 |
| Frame bytes | 80 | 48 | 64 | 32 |
| Hardware-path function calls | 0 | 0 | 0 | 0 |
| Hardware-path library calls | 0 | 0 | 0 | 0 |
| Hardware RoCC instructions | 1 | 1 | 1 | 1 |
| Whole-function static instructions | 123 | 113 | 109 | 97 |
| Whole-function bytes | 306 | 284 | 268 | 244 |

The static reductions are in the removed in-function Sink block, not in the
selected hardware success path. The same six/four registers still spill and
reload. Both versions still execute two optional-selection loads and two
conditional branches.

The compiler emitted an additional 82-instruction, 222-byte `paired_raw`
function for the experiment. The new cold helper became a two-instruction,
six-byte tail wrapper:

```text
10e24: li a4,1
10e26: j  11628 <paired_raw>
```

The existing Trace helper stayed at 117 instructions and 324 bytes. Ordinary
Scale hardware paths never call either helper. This does not demonstrate a
net runtime benefit from the extra dispatch structure.

### RoCC and status paths

| Operation | B2 word | Experiment word | Opcode | funct7 | funct3 |
|---|---|---|---|---:|---:|
| Write | `0x2ed7f57b` | `0x2ee5757b` | `0x7b` | 23 | 7 |
| Release | `0x30d7f57b` | `0x30e5757b` | `0x7b` | 24 | 7 |

The different words reflect CPU register allocation. Both retain `xd=1` and
both source operands. The checked key encoder and the RoCC primitives were
unchanged: `rs1 = slot | generation<<8 | column<<16`, Write `rs2 = carrier`,
Release `rs2 = 0`. Each counted hardware path has one counter increment and
one hardware instruction.

Both retain the `rd <= 7` status normalization and reject larger responses as
`POTAL_ERR_BAD_RESPONSE`. The candidate's common non-OK branch still enters
`paired_fail` before any successful state mutation. This is source/assembly
inspection, not an executed injected-hardware-error test of the candidate.

**Gate result: failed.** The candidate was rejected rather than advancing to
Host or hardware testing based only on smaller cold blocks/reserved frames.
No nanosecond estimate was derived from these counts.

## 4. Command-stream and error-path equivalence status

The requested new Host, Trace, Sink, injected-error, sticky-failure, cleanup,
generation/duplicate, live-slot, outstanding-operation, order/argument/count
regressions were conditional on Phase 3 succeeding. They were **not run for
this rejected candidate**, and no candidate Sink-stream equivalence pass is
claimed.

The final retained implementation is the original B2, not the untested
candidate. Restoration checks used `cmp -s` on both source and rebuilt binary:

```text
H36B3_SOURCE_RESTORED_EXACTLY
H36B3_B2_BINARY_RESTORED_EXACTLY
H36B3_RESTORE_PASS
```

SHA-256 before and after restoration:

| Artifact | Preserved B2 and final retained value |
|---|---|
| `ggml/src/ggml-potal/potal.c` | `7dec7c2a9cfc0bc12cb3dcdb9abd93302c0dc69e813a812f6d332c87e16850bd` |
| `build-potal-h3-6a-d16/bin/potal-scale-path-bench` | `1f13bd8aa8a12fbc59e45625be32b900904a6336727555c09fba18656c96c758` |

The B2 report remains the record of its previously passed regression and
full-command equivalence tests; those are not presented as new B3 test runs.
The restored source, rebuilt API object, and benchmark retain B2 behavior.

## 5. Isolated benchmark applicability

**Not run.** The generated-code gate failed, so the Host gate and FireSim
benchmark were not entered. No new seven-repeat latency, MAD, cycle count, or
Paired improvement was measured.

The following are only the supplied B2 reference medians:

| Operation | B2 Serial ns/command | B2 Paired ns/command |
|---|---:|---:|
| Write | 37.835 | 87.029 |
| Release | 32.905 | 82.147 |

They are CPU-observed guest command latencies, not pure FPGA execution times.
There is no B3 hardware or software-only Sink timing result to compare.

## 6. Decode/FFN applicability

**Not run.** Isolated improvement was not established, so quick smoke and
seven-repeat Serial/Paired/Conservative Decode/FFN measurements were not
eligible. No Golden, command-count, Fence/Drain, fallback, or failed-node claim
is made for execution of the rejected candidate.

Supplied B2 steady compute references, not new B3 measurements:

| Workload | B2 Serial ms | B2 Paired ms |
|---|---:|---:|
| `decode_1x768x768` | 9.283 | 14.699 |
| `gpt2_ffn_up_16x3072x768` | 53.500 | 79.623 |

No detailed profile or full RISC-V model suite was run.

## 7. Remaining overhead and limits

The B2 reference gap remains approximately 49.194 ns/Write and
49.242 ns/Release, using the underlying B2 unrounded measurements. No B3 timing
was collected to repartition that gap.

The baseline analysis distinguishes four optional-dispatch selection
instructions from validation, argument encoding, live-state bookkeeping,
status processing, command/drain updates, and ABI preservation. It does not
establish a separate time cost for any category or prove that the remaining
work is removable.

The rejected candidate demonstrates a limit of this particular specialization:
outlining optional behavior did not shorten the counted hardware paths or
remove their call-preserved live values. It does not prove that every possible
software optimization is exhausted.

Removing validation, weakening generation/liveness checks, skipping drain
stores/accounting, changing Release synchronization, or reducing semantic
command counts would not be a valid way to claim success for this task.

## 8. H3-6B4 versus H3-6C

**Prefer a bounded H3-6B4 measurement/attribution checkpoint, not another
dispatch rewrite.** The approximately 49 ns reference gap is still unpartitioned,
and this experiment provides no evidence that a new generic/unchecked issue
layer would help.

B4 should retain all checked behavior and use matched batch-level guest
cycle/retired-instruction measurements, with hardware and software-Sink
controls, to test a concrete mechanism. It should distinguish caller-loop cost,
checked state/encoding work, and register preservation before proposing another
production change. Instrumentation belongs in a separate diagnostic run, not
the unprofiled timing baseline.

Another optimization should require a demonstrable reduction of actual
hardware-path work and then the same correctness/performance gates. If B4
cannot identify safely removable work, stop command-dispatch tuning and move to
a separately specified H3-6C scope.

No existing B4/C definition was found in the local H3 reports or the wider
POTAL Markdown documentation. This recommendation therefore does not assume or
authorize a particular C-phase hardware, protocol, API, or numerical redesign.

## Reproduction and evidence

Both experiment and restoration used the same build command:

```sh
source /home/alveo/miniforge3/etc/profile.d/conda.sh
cd /home/alveo/firesim/target-design/chipyard
source env.sh
cd generators/potal/software/workspace/llama.cpp-potal
cmake --build build-potal-h3-6a-d16 --target potal-scale-path-bench -j8
```

The existing missing `chipyard/.conda-env` activation message, generated
`POTAL_SCALE_ENTRIES` macro-redefinition warning, and editor include-resolution
diagnostics remain. They were not suppressed or repaired as part of B3.
The configured actual cross compiler built both variants successfully.

Disassembly was captured with the configured toolchain's
`riscv64-unknown-linux-gnu-objdump -d --disassemble=<symbol>` for all six
requested names; the experimental optional helper was captured too.
The baseline symbol absence was checked with the same toolchain's `nm -a`.
Instruction PCs and gate counts are preserved in the JSON artifacts:

```text
/tmp/h3-6b3-before/disassembly.txt
/tmp/h3-6b3-before/path-evidence.json
/tmp/h3-6b3-experiment/source.diff
/tmp/h3-6b3-experiment/disassembly.txt
/tmp/h3-6b3-experiment/codegen-gate.json
/tmp/h3-6b3-experiment/potal-scale-path-bench
```

The rejected experimental benchmark SHA-256 is
`7bb324e1221ec4c66a60f93a277bfcaccf0ad016e57a866030b4a2143302adb2`.
It is archived separately and is not the final benchmark in the build directory.

Final retained change for B3: this report only. No production-source change,
public API change, RTL/FPGA rebuild, numerical change, synchronization change,
commit, or push is retained or performed.
