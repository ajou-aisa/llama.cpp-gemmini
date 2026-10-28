# H3-6E4C1 - Scale-aware Sequential Paired Micro-tile Integration

Date: October 27, 2026.

Path notation:

```text
F = /home/alveo/firesim
C = F/target-design/chipyard
P = C/generators/potal
L = P/software/workspace/llama.cpp-potal
S = /home/alveo/h3-e4b2/source
Q = xilinx_alveo_u280-firesim-FireSim-FireSimQuadRocketSinglePoTalD16Config-BaseXilinxAlveoConfig
```

## 1. Executive Summary

The default-OFF test scheduler is implemented. It alternates real checked
resident Main and indexed Residual operations while preserving each path's
DIM16 contribution order. Production scheduling, quantization, C11, RTL and
ISA are unchanged. No prefetch or physical overlap is implemented.

Native and RISC-V builds pass. Native software-only validation passes
8,395,015 assertions, 60 new executions, 174,580 compared cells and 40
Serial/Paired/E3 reference executions. All 56 scoped RTL tests pass.
Actual C1 FPGA execution passes all five progressive gates, including the
exact Decode/FFN fixtures, with independent raw INT32/INT64 and bitwise final
F32 agreement. Guest and driver both exit 0. Separate actual raw trace
auditing passes 945,304 commands with matching ENTER/EXIT and no bad response,
address, Scale-generation or live-consumer binding.

Fresh diagnostics-OFF n=7 medians are 23.697 ms Decode and 122.927 ms FFN,
versus legacy Paired K28 at 13.189/79.623 ms. C1 is 79.672%/54.386% slower,
not a performance improvement. Scale writes/releases account for
92.303%/92.429% of its additional raw commands. The implementation remains
test-only; correctness Hardware A does not justify production adoption.

## 2. B2 Software/RTL Checkpoint SHA

| Root | Branch | B2 checkpoint |
|---|---|---|
| L, independent software root | `potal` | `8fb19aa5b66ace4cc50014581e5b3fe5ad7fc546` |
| C, actual PoTal RTL root | `potal` | `1dfbe85aea66e0afc0e11072b983335a475a5b14` |
| F, unchanged outer root | `aisa-customizations` | `5c3458a3f54917ab9502bda4a8b3cc3dd513304d` |

The software commit contains only the ten completed B2 code files and its
report. The RTL commit contains only six completed implementation files and
six B2 test files. Each staged diff passed `--check`; name-status and stat
were inspected. Each index was empty before staging and after committing.
There was no push, new branch or FireSim commit. The actual outer branch is
`aisa-customizations`; the historical B2 report's detached-HEAD description
does not match the current checkout and was not used to change it.

C1 changes remain uncommitted:

```text
H3-6E4C1-sequential-paired-microtile-integration.md
ggml/src/ggml-potal/CMakeLists.txt
ggml/src/ggml-potal/tests/support/e4c1-sequential.hpp
ggml/src/ggml-potal/tests/support/e4c1-sequential.cpp
ggml/src/ggml-potal/tests/potal-e4c1-test.cpp
ggml/src/ggml-potal/tests/potal-e4c1-bench.cpp
ggml/src/ggml-potal/tests/potal-e4c1-backend.cpp
```

## 3. FPGA Image Provenance

The preserved B2 image is reused without programming, module reload or FPGA
rebuild. Its generated target is `Q`, configuration
`FireSimQuadRocketSinglePoTalD16Config`, platform `BaseXilinxAlveoConfig`.
The build directory is
`S/deploy/results-build/2026-10-26--17-24-35-alveo_u280_firesim_potal_quadcore_d16_h3_e4b2_no_nic/cl_Q`.
The driver is `S/sim/output/xilinx_alveo_u280/Q/FireSim-xilinx_alveo_u280`.
All 22 checkpoint code identities matched the completed B2 manifest before
committing.

The reused generation/build recipe is
`/home/alveo/h3-e4b2/config_build_recipes.yaml`, build alias
`alveo_u280_firesim_potal_quadcore_d16_h3_e4b2_no_nic`, with 30 MHz FPGA
frequency and TIMING strategy. Historical B2 routed evidence reports
WNS 0.070 ns / WHS 0.010 ns, zero TNS/THS and fully routed nets; C1 does not rerun
or relabel that build. A fresh byte comparison of all six implemented RTL
files between C and S passes and records their complete SHA256 values in
`/tmp/h3-e4c1-reused-build-identity.log`. This links the checkpoint source,
unchanged generated-build snapshot, matching driver and actual signature.

| Artifact | Recomputed full SHA256 |
|---|---|
| B2 `xilinx_alveo_u280/firesim.bit` | `c4c71ef648f5c5a73a40f810fed98fdfb3862f1433b00592720f49631972beae` |
| Matching snapshot driver | `2122ed72e53859aa45f7f86db20c91b7712d6be2d5321d96aef49d45f05bf666` |
| B2 checked guest ELF | `55f66631665f105d8409758c4052cc4c45fbac64839abf4a832d25f1d7cd6ba7` |
| B2 raw guest ELF | `aa5c15f779765c5d06e5019b9414b4394847b752edf92335d7d6b4fcb9289c53` |
| C1 benchmark ELF | `b6b30860da0dfd69a35c9a2c04f0e840acb898d35fd6431f9537efc33e039b56` |
| C1 `libggml-base.so` | `870e8aac024cdce8ad85d7b5f4625a4069117322674a9093f579db8967e8d0c5` |
| C1 `libggml-cpu.so` | `4ab8c61bc27c9a120a8b16ff87100f193c35f70b04049cb7f3bcb99185897341` |
| C1 `libggml-gemmini-utils.so` | `d232c406a0a655e70b9e71dcc8a1a5c2782125754be31e451ba7e40fac839896` |
| C1 `libggml-potal.so` | `5be73897347aeb48b51f7bddecc8b476ef5e4a43c8a6d5e9e6fc02720de999fc` |
| C1 `libggml.so` | `5d0091640800bdfb7e3916596757d6938482f6562b78738d3ebd246d56afaefe` |

`/tmp/h3-e4c1-provenance-baseline.log` records full checks and fresh exclusive
device use. Every extended suite starts with actual capability
`0x504f544102100007`; a mismatch refuses before any extension. Only
`/tmp/h3-e4b2-old-run/rootfs.img`, the already-private mutable copy, is used.
The original `/tmp/h3-e4b1-run/rootfs.img`, old/new bitstreams, old driver,
models and prior result directories are preserved.

`/tmp/h3-e4c1-code-identity.log` binds all six uncommitted C1 code/build files
to their full SHA256 values and the executed ELF/libraries. It also confirms
the unchanged outer FireSim HEAD and `aisa-customizations` branch.

## 4. E4B2 Regression and Known Issues

Fresh actual B2 execution on this image passes raw 42 invocations/193,024
INT32 comparisons, all four saturation witnesses and checked warmup plus
n=7 suites of 16 invocations/4,404 cells. Unchanged E4A passes before and after:
each trace has 2,074 commands, 16 legacy LOOPs with xd=0 and no extension.
Evidence is `/tmp/h3-e4c1-baseline-results/` and
`/tmp/h3-e4c1-baseline-run/{uartlog,driver.log}`. Guest and driver exit 0.

The independent unchanged smoke boot reproduces:

```text
FAIL A hw invalid column: got=6(HW_INVALID_ENCODING) expected=1(HW_INVALID_COLUMN)
```

`key(6,1,16)` is `0x00100106`. Column 16 sets reserved key bit20 because the
column field has four bits. Encoding validation precedes column validation;
the hardware correctly returns status 6 for that encoding. Pre-B2 ingress
has the same priority, and archived old/new image runs show the same failure.
The smoke assertion was not changed. Fresh evidence:
`/tmp/h3-e4c1-smoke-results/new-legacy-api-hardware.log`; guest exit 1,
host driver exit 0. Host driver PASS alone is not a guest numerical PASS.

Two pre-existing Scala harnesses omit required `indexEntries`/`spRows`:
`PoTalScaleCommandIngressSpec.scala` and `PoTalDrainHandshakeSpec.scala`.
They remain unchanged. The optional full processor RTL address sweep remains
an incomplete passing prefix, distinct from actual FPGA 42-case success.
Physical FPGA DMA/bus fault injection remains unperformed.

## 5. Sequential Scheduler Architecture

`tests/support/e4c1-sequential.cpp` consumes the existing validated active H2
tile/fragment plan. It does not duplicate packet lowering or tile selection.
Its state is stripe, J group, K panel, resident token/pair, canonical Main
position, compact Residual batch/group, ACC epochs, checked Scale allocations,
Index incarnation and pending/completion state.

The benchmark's owner proxy replaces only `owner_runners.paired` inside the
existing issuer callback. Production owner establishment, thread pinning,
producer queue, packet preparation, raw result handling, reconstruction and
merge remain the existing implementation. Actual execution passes NULL
transport to the checked B2 resident API; native testing supplies an explicitly
software-only independent command model. No new unchecked RoCC emitter exists.

`potal-e4c1-backend.cpp` redirects the tile selector/pipeline only in this test
executable. It compiles the existing `ops.cpp` once through the adapter.
`POTAL_E4C1_TESTS` defaults OFF. Production Paired remains unchanged.

## 6. Quantization/Packet Reuse

The benchmark reuses `potal-paired-smoke.cpp` fixture helpers: fresh
`std::mt19937(20261023)` per case, weights first, then activations, with the
same separate distributions and outlier insertion order.

Production HP1 packing, ExSIA sequential local adaptation, stripe folding,
signed8 Main, signed radix Residual and sealed packets are reused.
Fingerprint serialization follows E3 and covers prepared activation bytes,
theta, sealed packet metadata, block/group/lane layout, compact indices and
digit storage. Expected exact fingerprints are Decode `e460e0fce470e311` and
FFN `18e2b73e075c3bbe`. Every warmup checks the independently prepared packet
against the actual production graph preparation. Timed runs verify immutable
original weights/activations and stable cache/command counts.

## 7. K/J/Stripe Scheduling

Geometry stays DIM16/I1, J<=9, K<=28, four SP banks of 4096 rows, two ACC
paths of 512 rows, Scale256 and Index32 per pair.

K768 uses panels K448 and K320, or 28+20 tiles. Decode uses five J144 groups
and one J48 group: 6 groups, 12 residencies. FFN uses 21 J144 groups and one
J48 group: 22 groups, 44 residencies. Existing I1 stripe splitting is kept;
M17 is tested as M16 plus M1, not new I2 support.

Pair is `(j_group * k_panel_count + panel) % 2`. Each resident has one B load,
all consumers complete, then it retires before the next load. Thus both pairs
participate, but peak concurrent live B residency is one, not double buffering.
The default Main grouping is four DIM16 partials. K20 has five groups; K28
has seven. Last columns/rows retain padding and checked origin validation.

## 8. Main/Residual Alternation

Within each residency, the scheduler starts with Main and alternates one
Main group with one ready Residual I1 group. It never reorders contributions
within either path. When one path is empty, the other runs to completion
without dummy consumers. Switch and tail counts reset at residency boundaries.

Residual batches preserve canonical compact fragments. Each batch contains
at most two fragments, fitting Index32. Groups whose exact active-fragment
sets match share one Index/Scale window; group-zero fragments are skipped.
The sharing does not combine fragment dots, merge carriers from different
blocks or change per-group accumulation order.

Native command-transcript validation independently checks view/LOOP path
switches, every residency, both pairs, original-K order, tails and exact counts.
Actual C1 switch counts are recorded in the hardware results, not inferred
from the software model.

## 9. Scale-aware Window Management

Full K28/J9 P1 Main requires 252 slots; adding a two-fragment Residual
window would require 270 and is not admitted. Instead Main K4 uses at most
36 slots; the disjoint Residual window uses at most 18. Capacity is checked
before effects and actual live high-water is counted.

Main slots start at zero. Residual starts at `main_k_tiles * local_J_tiles`.
Carriers bind each original block32 and J column. P1 retains both DIM16
partials even when adjacent carriers match. New checked Scale tokens bind
slot, wire generation, resident, original block and consumer; wire generations
wrap only through the existing API interpretation.

The first successful Main Release proves global completion. All remaining
column Releases retain the same allDrained semantics. A Residual cohort uses
Fence between non-final consumers and Release after its final consumer.
Scale/Index are never overwritten while a reader is pending.

Software-only M17/N145/K928 comparison, per first stripe:

| Fixture | K1 commands | K2 commands | K4 commands | Main loops K1/K2/K4 | Peak Scale K1/K2/K4 |
|---|---:|---:|---:|---|---|
| Zero Residual | 18,706 | 18,068 | 17,760 | 116/58/30 | 9/18/36 |
| Sparse | 32,932 | 32,294 | 31,986 | 116/58/30 | 18/18/36 |
| Dense three-lane | 39,512 | 38,874 | 38,566 | 116/58/30 | 27/36/54 |

Every candidate writes and releases 8,410 Main columns. Sparse Residual adds
6,235 writes and releases; dense adds 8,410. K4 saves 946 commands versus K1,
but does not reduce P1 Scale traffic. These are software command counts, not
hardware timing. K4 is selected for bounded control reduction, not a claimed
measured speedup over the other candidates.

## 10. Resident-B Ownership

Every load uses the C11-owned pinned B pool and the original checked descriptor.
Publication follows successful Fence. Diagnostic runs overwrite the completed
host source with 93 before Main/Residual consumers. Each new residency recopies
its own correct panel; consumers do not issue a replacement B DMA.

Full K28/J9 uses pair0 base4160 or pair1 base12352 and pitch144. Interior native
Main advances base by `(k_offset/16)*pitch`; indexed Residual retains full-panel
base, pitch and original-K gather. B DRAM LOOP address is zero.
The conjunction of command-level load count, decoded view/LOOP, unchanged RTL
PRELOAD addressing, source poison and exact independent INT32 output is the
reuse proof. A zero DRAM address alone is insufficient.

The API owns pool lifetimes and global issuer exclusion. Recoverable cleanup
may release only after known completion. Unknown completion retains pools,
sink context and accelerator ownership; there is no automatic retry.

## 11. ACC Epoch and Saturation

The B2 epoch contract already permits continuation across B residency changes:
retirement leaves ACC epochs intact, and epochs bind logical weight identity,
generation, output coordinates and canonical contributions rather than one
resident incarnation. No C11/API change was needed.

Main uses ACC0; Residual uses ACC1. The nonzero bank-local tile offset is one.
Residual I1 groups occupy disjoint `1 + group * local_J_tiles` tile ranges.
The total fixed ACC footprint is checked before open; oversized cases are
rejected, not spilled or rebound. Each J group begins new checked epochs;
K panels continue those epochs with immutable manifests and sequences.

Readback issues addressed MVOUT, awaits successful Fence, validates its
receipt and copies to private host vectors. Caller buffers are published only
after all readbacks, retirement and successful close.

Arithmetic stays:

```text
partial = signed DIM16 dot
shifted = Sat32(partial * 2^carrier)
acc     = Sat32(acc + shifted)
```

Independent wide arithmetic verifies ordered MAX/MAX/MIN=-1,
MAX/MIN/MAX=2147483646, compact cancellation=0 versus dense=-1,
carrier24=16777215 and negative saturation=INT32_MIN. No merged-dot oracle is
used. Raw restore followed by metadata-bearing GEMM resume is not verified.

## 12. Host and RTL Regression

Eighteen unchanged Host targets rebuilt and passed in one run:
`/tmp/h3-e4c1-baseline-host.log`. They include API, attention/B5, trace,
Residual, Paired planning/schedule/unit/auto/execution, stripe pipeline,
resident/model, E3 prototype/context, B1, stripe API and B2.

C1 native evidence is `/tmp/h3-e4c1-native-final.log`:
60 executions across ten numerical fixtures, three groupings and two stripes,
40 existing-model executions, 8,395,015 assertions and 174,580 compared cells.
This is software-only. `--probe-only` on the native benchmark explicitly
refuses with exit2; it cannot manufacture hardware success.

Both actual compiler builds pass, including the RISC-V executable. New
missing-declaration/aggregate-initialization warnings were fixed. The existing
generated `POTAL_SCALE_ENTRIES` redefinition warning remains disclosed.
LSP lacks the generated compilation include context for new test files;
the real native and cross compilers, not the cascaded LSP include errors,
are the compilation evidence.

The scoped RTL rerun passes all 56 tests: nine resident scenarios and 47
unchanged legacy scenarios. Evidence: `/tmp/h3-e4c1-rtl-final.log`, exit 0.
Its first command selected a Control test absent from the isolated snapshot
and failed before running tests. The successful command supplies the unchanged
current Control test by explicit absolute path. The two known stale
constructors remain excluded and unchanged; this is not a full-suite green.

The preserved C1 build caches explicitly enable `POTAL_E4C1_TESTS=ON`;
the default remains OFF. Reproduction from L after sourcing conda, Chipyard
and FireSim environments:

```bash
cmake --build build-potal-e4c1-host -j8 --target potal-e4c1-test potal-e4c1-bench
LD_LIBRARY_PATH="$PWD/build-potal-e4c1-host/bin" build-potal-e4c1-host/bin/potal-e4c1-test
cmake --build build-potal-e4c1-d16 -j8 --target potal-e4c1-test potal-e4c1-bench
```

## 13. Actual RoCC Execution

All five actual-hardware gates pass. Evidence:
`/tmp/h3-e4c1-gates-results/gates-numeric.log`,
`gates-{checked-capability,capability,e4a-before,e4a-after}.log`,
and `/tmp/h3-e4c1-gates-run/{uartlog,driver.log}`.
Raw capability matches `0x504f544102100007`; checked capability reports
`software_only=0 extension_commands=0`. Both precede extended work.
Each graph completes one real Paired stripe with zero fallback and exact
independent raw results. E4A before/after passes. Guest exit0 and driver exit0
are separately observed.

| Gate | Shape M/N/K | B residencies | Main/Residual LOOPs | Main-to-Residual / reverse | Scale/Index peak |
|---|---|---:|---|---|---|
| 1 | 16/144/448 | 1 | 7/7 | 7/6 | 36/27 |
| 2 | 16/144/768 | 2 | 12/24 | 12/10 | 36/32 |
| 3 | 16/155/768 | 4 | 24/24 | 24/20 | 36/27 |
| 4 Decode | 1/768/768 | 12 | 72/60 | 60/54 | 36/19 |
| 4 FFN | 16/3072/768 | 44 | 264/264 | 264/220 | 36/27 |

Gate2 continues the same nonzero ACC tile1 epoch across original-K origins
0 and 448 and both pairs. Gate3 uses original-J origins 0/144 and pitch 144/16.
Full Decode uses J origins 0,144,288,432,576,720 and pitch 144/48. FFN uses
22 J groups through 3024 and the same full/tail pitches. Every residency
retires safely and is loaded once. These are command-declared origins and
extents, not physical DMA timing.

Both single-tail directions execute on actual hardware: Decode has 12
Main-only and six Residual-only microtiles; FFN has zero and 44 respectively.
A terminal Residual after the last Main counts as Residual-only even when
the two complete loop counts are equal. No dummy execution is introduced.

The separate benchmark/profile/trace boot also ends with guest exit 0 and
driver exit 0. `/tmp/h3-e4c1-benchmark-results/` contains the preserved logs;
`/tmp/h3-e4c1-benchmark-run/` contains its UART/driver records. Its final
unchanged E4A passes 2,074 matched commands, 16 legacy LOOPs, no extension
and no raw-audit error. Both private-rootfs collections finish unmounted;
the UART writers are terminated after driver exit.

## 14. Decode/FFN Integer Golden

The diagnostic graph observer checks raw Main INT32, each canonical Residual
lane/group INT32 and reconstructed INT64 against packet-direct wide math.
It verifies lane/source-row identity, not just final tensor values.
Final output uses the existing independent pipeline golden and existing
same-policy merge. Native synthetic cases additionally compare bitwise F32
against Serial/E3 under the same legacy merge.

Both full hardware fixtures pass raw Main INT32, per-lane Residual INT32,
checked reconstructed INT64 and bitwise final F32. Decode prepared hash is
`e460e0fce470e311`; FFN is `18e2b73e075c3bbe`, exactly the E3 values.
Original weights/activation hashes are respectively
`8da7f13cc88c7f2d`/`2fec58fb268724ec` and
`44d4bab36efd4b0d`/`ed83bb311e5e1d18`.
The unchanged smoke comparator uses `memcmp` on each output float, not an
FP tolerance. The independent raw observer separately prevents an integer
error from being hidden by scaling.

The separate actual profile run has also completed with its diagnostic
success sentinel. It exercises all four modes on both workloads and requires
the same raw Main/Residual/reconstructed integer oracle for every mode.
Thus integer agreement is checked against actual Serial K48, forced Serial
K28 and legacy Paired K28, not only their software models.

Primary timing runs do not run the raw oracle inside measured graph work;
diagnostic integer verification is a separate execution.

## 15. Command-count Comparison

Every successful C1 run reports checked-session command/drain totals plus
Main/Residual microtiles, original partials, B load/MVIN/retirement, Scale
programs/slots/writes/releases/generations, Index, view, LOOP, readback/MVOUT,
switches, tails and actual high-water. Native transcripts independently match
the complete funct counts to these statistics.

Hardware trace runs are separate from timing. The complete 126,950,321-byte
actual trace has 1,891,399 nonempty lines. A single streaming audit takes
9.17 host seconds and matches 945,304 ENTERs to 945,304 successful EXITs.
There are 660 extended LOOPs and 660 funct28 views, plus 380 legacy Paired
LOOPs. Serial uses a different existing low-level path and is not included
in these paired-session raw totals; its route counts are reported separately.

The independent audit checks extension bit3/xd1, P1, ACC/path/indexed bits,
pair A/B SP identifiers, B DRAM zero, local bounds/pitch, full B-bank range,
nonzero bank-local ACC offset, live Scale slot/generation/column coverage,
matching capability responses and full-width addressed ACC0/ACC1 MVOUT.
No error is found. Both extended sessions finish with zero live Scale slots.
Raw command totals exactly match the C1 statistics below.
Audit details are `/tmp/h3-e4c1-raw-trace-audit.md`; the original raw file is
preserved, not rewritten to repair stdout/stderr interleaving.
`drains` counts completion qualifications; it is not an additional disjoint
command category and must not be added to command totals.

Actual checked C1 statistics per full fixture, also matched to raw trace:

| Counter | Decode | FFN |
|---|---:|---:|
| Main / Residual partials, over J groups | 288 / 114 | 1056 / 528 |
| Main / Residual microtiles | 72 / 60 | 264 / 264 |
| B loads / B-only MVIN commands | 12 / 2304 | 44 / 9216 |
| B retirements | 12 | 44 |
| Scale program instances | 132 | 528 |
| Scale slot-program instances | 3216 | 13824 |
| Main Scale writes, and equal releases | 36864 | 147456 |
| Residual Scale writes, and equal releases | 14592 | 73728 |
| Scale generations / wire wraps | 132 / 0 | 528 / 2 |
| Logical Release boundaries | 132 | 528 |
| Index config / write | 60 / 978 | 264 / 6314 |
| View / LOOP | 132 / 132 | 528 / 528 |
| Explicit Fence commands | 25 | 89 |
| Accepted global drain qualifications | 51481 | 221273 |
| Readback requests / MVOUT commands | 12 / 96 | 44 / 384 |
| Checked command total | 107854 | 464534 |
| Main-to-Residual / reverse switches | 60 / 54 | 264 / 220 |
| Main-only / Residual-only tail | 12 / 6 | 0 / 44 |
| Live Scale / Index / B residency high-water | 36 / 19 / 1 | 36 / 27 / 1 |

High-water means live checked allocations during this actual execution,
not an internal on-chip occupancy or traffic measurement. Scale capacity is
256 and Index capacity 32; the two workloads do not exercise their maxima.
Native dense multi-lane execution separately reaches 54/32, software-only.

Fresh comparison from the same primary run and separate actual trace:

| Route | Decode Main/Residual LOOPs | FFN Main/Residual LOOPs | Decode Scale write/release each | FFN Scale write/release each |
|---|---|---|---:|---:|
| Serial originalK48 | 6/4 | 22/22 | 33024 | 147456 |
| Serial forcedK28 | 12/4 | 44/22 | 33024 | 147456 |
| Legacy Paired K28 | 12/60 | 44/264 | 33024 | 147456 |
| C1 resident K28, Main groups K4 | 72/60 | 264/264 | 51456 | 221184 |

Serial Residual keeps its existing automatic compact tiling; forcedK28
controls the Main plan, exactly as the E3 controlled baseline does.

| Actual paired-session raw counter | Decode whole / C1 | FFN whole / C1 |
|---|---|---|
| Commands | 67916 / 107854 | 305000 / 464534 |
| CONFIG funct0 | 300 / 552 | 1276 / 2200 |
| B-only raw MVIN funct2 | 0 / 2304 | 0 / 9216 |
| View funct28 | 0 / 132 | 0 / 528 |
| LOOP funct8 | 72 / 132 | 308 / 528 |
| Each LOOP config funct9..13 | 72 / 132 | 308 / 528 |
| SCALE_WRITE funct23 | 33024 / 51456 | 147456 / 221184 |
| SCALE_RELEASE funct24 | 33024 / 51456 | 147456 / 221184 |
| Fence funct25 | 1 / 25 | 1 / 89 |
| Index config / writes | 60/978 / 60/978 | 264/6314 / 264/6314 |
| MVOUT funct3 | 96 / 96 | 384 / 384 |
| Live Scale high-water | 144 / 36 | 144 / 36 |

Legacy B loads occur inside generated Main LOOPs, not raw CPU funct2.
Zero raw MVIN in that column does not mean zero B DMA. Both schedules have
the same 12/44 logical B residencies; C1 exposes their 2304/9216 tile loads
as checked CPU-issued commands instead of reloading B per microtile.

Using the same accepted Release-plus-Fence definition, global completion
qualifications are whole/C1 33025/51481 for Decode and 147457/221273 for FFN.
The older report's `paired_drains=73/309` counts logical scheduler drain
boundaries, not every accepted column Release. It must not be compared to
C1's core drain counter as if they were identical measurements.

## 16. Matched Performance Benchmark

One executable contains all four routes: Serial originalK48, test-only Serial
forcedK28, legacy Paired whole-loop K28 and C1 sequential K28. The production
planner is not changed. K48 is a Serial baseline, not an attempt to restore
unsupported Paired K48.

The same backend, cached weight, input, prepared packet, image and pinned
issuer are checked. Each mode has separate warmup. Measured order rotates
and reverses across seven rounds. Primary build has
LOG_DEBUG/INFO/TRACE=0, CYCLE_DETAIL=0 and no externally enabled diagnostic
environment variables. Poison, raw oracle, trace events and detailed clocks
are disabled during primary measurements.

All 56 measured samples and eight n=7 summaries pass. Independent
recalculation from individual samples exactly reproduces each median and
unscaled median absolute deviation. No sample is discarded. Evidence:
`/tmp/h3-e4c1-benchmark-results/benchmark-primary.log`. Historical E3 timings
are not reused.

| Route | Decode median / MAD, ms | FFN median / MAD, ms |
|---|---|---|
| Serial originalK48 | 7.485 / 0.004 | 52.713 / 0.082 |
| Serial forcedK28 | 7.499 / 0.004 | 52.690 / 0.078 |
| Legacy Paired whole-loop K28 | 13.189 / 0.027 | 79.623 / 0.087 |
| C1 sequential resident K28, Main K4 | 23.697 / 0.051 | 122.927 / 0.126 |

C1 adds 10.508 ms Decode and 43.304 ms FFN versus matched whole-loop:
79.672% and 54.386% slower. It is 216.002%/133.302% slower than forced
Serial K28. These are CPU-observed graph-compute latencies, not accelerator
cycles or hardware waiting intervals.

Order, with S48/S28/P/C denoting the table rows, is:
`S48,S28,P,C`; `S48,C,P,S28`; `P,C,S48,S28`; `P,S28,S48,C`;
then the first three orders repeat. This reverses/rotates but is not a fully
balanced Latin square. S48/P occupy positions1/3 and S28/C positions2/4.
C1 position2/4 medians are 23.6915/23.757 ms Decode and
122.9035/123.053 ms FFN. First-three/last-four C1 medians are
23.755/23.6915 and 123.053/122.8825 ms. These small observed shifts do not
remove the whole-loop gap; they do not prove universal order independence.

Outliers are retained: Decode S48 sample 1 is 9.280 ms and S28 sample 2 is
8.596 ms. Whole-loop ranges are 13.151..13.289 and 79.508..79.803 ms;
C1 ranges are 23.646..23.932 and 122.766..123.187 ms. Every C1 sample is
slower than every whole-loop sample within its workload.

Actual RISC-V guest invocations, after the separate raw capability probe:

```bash
bin/potal-e4c1-bench --probe-only
bin/potal-e4c1-bench --gates --group 4
bin/potal-e4c1-bench --bench --case all --repeat 7 --group 4
bin/potal-e4c1-bench --profile --case all --group 4
bin/potal-e4c1-bench --trace --case all --group 4
```

The guest scripts record ELF/library hashes and clear external diagnostic
environment variables. Profile and trace are distinct process invocations,
not added to or subtracted from the primary samples.

## 17. Scale/Index/Drain Bottleneck Analysis

P1 Main necessarily programs twice the per-column metadata of phase-aligned
P2 whole-loop under the current interface. Increasing K grouping amortizes
view/LOOP/configuration calls but not those Scale writes/releases.
Index32 limits each indexed batch to two DIM16 fragments. Exact cohorts
reuse a window across I1 lane groups without capacity expansion.

Actual checked C1 command totals already show the metadata floor:
Decode has 102,912 Scale write/release commands out of 107,854 total,
or 95.418%; FFN has 442,368 out of 464,534, or 95.228%.
These are command proportions, not time proportions.
Compared with phase-aligned P2 Main metadata, P1 alone adds at least 36,864
Decode and 147,456 FFN write/release commands. Additional view/configuration
and explicit B-only issue commands cannot be hidden by grouping Main work.
In the independent dense three-lane fixture, moving K1 to K4 reduces total
commands by only 2.394% despite reducing Main loops from 116 to 30. Sparse
reduces 2.873%. These candidate results remain software-only counts.

The actual additional commands are 39,938 Decode and 159,534 FFN,
58.805%/52.306% above whole-loop. Scale writes/releases contribute
36,864/147,456, or 92.303%/92.429% of those additions. The remaining
3,074/12,078 decompose exactly into:

| Added raw commands | Decode | FFN |
|---|---:|---:|
| Explicit B-only MVIN | 2304 | 9216 |
| CONFIG funct0 | 252 | 924 |
| LOOP plus five LOOP configs | 360 | 1320 |
| Resident view | 132 | 528 |
| Explicit Fence | 24 | 88 |
| Capability commands | 2 | 2 |

This is a measured command-cost floor for the implemented sequential path,
not a conversion to milliseconds. B reuse avoids per-microtile reloads,
but does not reduce the logical B loads versus the already-resident
whole-loop baseline. Next-K prefetch alone does not remove this metadata
or control work.

Separate instrumented C1 CPU intervals, in milliseconds:

| Scope | Decode | FFN |
|---|---:|---:|
| Internal planning/manifest preparation | 1.392 | 8.057 |
| Checked preparation calls | 1.576 | 3.612 |
| B issue / completion calls | 0.129 / 0.021 | 0.535 / 0.070 |
| Scale program | 4.390 | 18.326 |
| Index program | 0.207 | 1.095 |
| Main / Residual issue | 0.774 / 0.605 | 3.235 / 3.858 |
| Scale Release | 38.056 | 164.358 |
| Readback calls | 0.072 | 0.298 |
| Retirement / close | 0.045 / 0.313 | 0.149 / 0.307 |

Evidence is `benchmark-profile.log`. Its stdout/stderr interleave splits
the Decode Residual `calls=60` field; the complete value and ns field remain
in the preserved file. Profile has zero separate non-final cohort completion
calls here: actual lane groups complete through Release. That zero is not
zero completion cost.

The largest instrumented C1 interval is per-column Scale Release. Legacy
profile reports Scale program/release at 4.242/3.829 ms Decode and
19.038/17.359 ms FFN. C1 clocks each of 51,456/221,184 Release calls, while
legacy clocks a whole window. Instrumentation granularity is different:
the Release interval alone exceeds primary C1 latency. It cannot be
subtracted from a primary sample or treated as a causal decomposition.

The structural CPU path is
`executor::release -> potal_paired_scale_release -> resident_ws_release ->
paired_scale_release_core -> paired_drained -> resident_ws_completed`.
The latter preserves the RISC-V CPU fence and checked resident publication
on every accepted Release (`potal.c:1837`, `potal.c:1308`, `potal.c:1156`,
`potal.c:1649`).
Together with P1's extra commands, this identifies Scale/control work as the
main evidenced overhead candidate, not DMA/PE waiting. No clear, validation
or completion guarantee is removed based on this benchmark.

Outer planning intervals can enclose internal work. Untimed host packing
and reconstruction are not exhaustively partitioned; reconstruction duration
is not separately reported by this harness. These CPU API intervals are
not hardware wait measurements, are not added into graph wall time, and are
not subtracted from the diagnostics-OFF results.

## 18. Failure and Cleanup

Deterministic software command-index injection covers before/after first,
middle and replacement B MVIN, Main/Residual LOOP, six configuration/view
boundaries, Scale writes, Index configure/writes, first/middle MVOUT,
Release, prepare/load/final completion Fence and uncertain cleanup.
Failures are armed from an observed successful transcript, not elapsed time.

Every failed invocation leaves caller output unpublished and preserves its
first status. Recoverable issue failures use the existing single cleanup
drain and close once. Uncertain Scale/Release/Fence responses and cleanup
errors retain the context and global issuer; a new open is BUSY without any
additional command. Fork-isolated test processes terminate without unwinding
possible DMA request storage. Capability and pre-open capacity refusals
issue no extended work. Existing B2 regression covers stale generation,
reader protection, epoch collision, receipt expiry and unfinished close.

These are software-injected failures. They are not physical FPGA DMA fault
injection or proof of recovery from an actual bus error.

## 19. Remaining Limitations

I>1 per microtile, J>9, K>28, Index>32 or Scale>256 are unsupported.
Fixed ACC placement admits bounded lane groups; it does not introduce spill.
Metadata-bearing GEMM resume after raw ACC restore is unverified.
Physical DMA duration, bus bytes, PE overlap and accelerator wait attribution
are not measured. Next-K prefetch, dynamic capacity reclamation, Scale P2
optimization, production replacement and report-copy optimization are absent.

Existing smoke, stale Scala constructors and incomplete optional RTL sweep
remain as described in section 4. They were not edited to manufacture green.

Grades: A means implemented and actual hardware verified; B means
software/RTL verified with hardware pending; C means blocked by the current
interface/ownership contract; D means unsupported in this implementation.
Explicit out-of-scope/unmeasured entries do not receive a fabricated grade.

| Feature | Verdict and boundary |
|---|---|
| Single-residency alternation | A: Gate1 actual RoCC and independent integers |
| Multi-K continuation | A: K28+K20, same output epochs across resident replacement |
| Multi-J execution | A: J9+tail, origin/pitch/ACC offset and independent results |
| Decode/FFN exactness | A: raw INT32/INT64 plus bitwise final output |
| Scale/Index admission | A for actual accepted 36/19 or 36/27; B for injected boundary/ownership refusals |
| Main-only and Residual-only tail | A: both occur in actual Decode; Residual-only also in FFN |
| Multiple I1 stripes | B: M17 native regression; no new multi-stripe hardware claim |
| Failure/quarantine | B: deterministic software and existing RTL, no physical bus-fault injection |
| Raw ACC restore plus metadata-bearing GEMM resume | Not verified; outside C1, no A claim |
| Next-K prefetch | C: not implemented; current admission/completion ownership needs a separate contract |
| Physical DMA/PE overlap | Not measured or implemented; no overlap grade or performance claim |
| Dynamic capacity reclamation | D in this fixed implementation |
| Production Paired replacement | Not adopted; default-OFF test-only integration |

These verdicts concern the listed features, not a full prefetch pipeline.

## 20. E4C2 Prefetch/Pipeline Recommendation

First preserve the completed C1 numerical/ownership contracts. A next-K DMA
experiment needs explicit proof of LoopMatmul generator admission versus raw
MVIN admission, observable completion for the next pair, immutable resident
generations, checked pending publication, old-pair reader completion and
failure quarantine for both DMA and consumers.

Global Release currently waits for allDrained. Issuing next-pair DMA earlier
can delay current Scale release; earlier issue alone is not overlap.
Any next step must keep that guarantee or separately design and verify a
completion contract, rather than removing its check.

Measured P1 metadata overhead should determine whether a separate
original-K phase-aligned P2 fast path, exact carrier-binding reuse or bounded
Scale batching is worthwhile. Such a path must preserve DIM16 saturation,
compact-fragment order and token lifetimes. Equal carrier values alone do
not authorize reuse across different original block/J identities.

No E4C2 implementation or production adoption is part of C1.

The minimum interface work before a safe prefetch experiment is explicit
per-pair pending-B state plus independent compute/readback completion:
the current resident C11 session has one pending-kind admission boundary.
Keep the single issuer and immutable generations, but prove publication,
reader retirement and quarantine when more than one operation is pending.
Then test actual LoopMatmul/raw-MVIN admission and completion visibility.
Do not assume that a second pair or spare Scale slots establish overlap.

The measured metadata floor makes phase-aligned P2 or an equivalently proven
reduction of Scale transactions a stronger first performance candidate than
prefetch alone. Any batched completion/bookkeeping change needs a complete
producer/consumer/ownership audit and preservation proof, not only a faster
no-GEMM benchmark. C1 deliberately leaves all such semantics unchanged.

Final preservation evidence, `/tmp/h3-e4c1-final-protection.log`, exits 0.
All nine protected legacy image/driver/header/model/original-rootfs hashes
match. The B2 bit/driver hashes remain unchanged. Both B2 checkpoint HEADs,
both `potal` branches and both empty indexes are verified. The pre-existing
host-init hash remains
`e96d846238946925fd4a1c52f9b5e24ea144e15b3d748d15f8420e1c71c70834`;
the Gemmini tracked-diff hash remains
`407b6271652f19485acaa919b81e5fb697083ac73a894d7b217de883c9e71b84`.
Existing overlay deletion/generated files, CDE target-stream state,
`test_run_dir/` and untracked `models/gpt2/` remain untouched.
