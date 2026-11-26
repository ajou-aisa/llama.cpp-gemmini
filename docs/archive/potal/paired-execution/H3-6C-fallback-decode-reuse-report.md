# H3-6C - Paired Fallback Decode Reuse

Date: 2026-10-25. Repository: ajou-aisa/llama.cpp-gemmini, branch `potal`.
Baseline: retained H3-6B2, after the rejected H3-6B3 dispatch prototype was restored.
Hardware: unchanged H3-5B FireSim U280 DIM16 image.

## Result

The capacity-fallback path now lowers its already validated decoded packet into
the original Serial K32 request instead of decoding the packet a second time.
There is one shared lowering implementation. The public packet builder still
validates and decodes before calling it.

Fresh, unprofiled seven-repeat steady graph-compute medians for
`dense_fallback_160x320x512` changed as follows:

- Serial: 139.542 -> 139.773 ms (MAD 0.707 -> 0.558 ms).
- Paired with two capacity-fallback stripes: 225.650 -> 141.265 ms
  (MAD 0.592 -> 0.357 ms), a reduction of **84.385 ms / 37.40%**.
- Paired Conservative with two capacity-fallback stripes: 225.651 -> 141.633 ms
  (MAD 1.415 -> 0.852 ms), a reduction of 84.018 ms / 37.23%.

The Paired-minus-Serial difference of medians fell from **86.108 to 1.492 ms**.
The current difference is 1.07% of the current Serial median. These are guest
CPU-observed graph-compute and pipeline wall times, not pure FPGA execution
times. No historical 83.102 ms saving was assumed.

Actual-call diagnostics prove four -> two decodes per two-stripe all-fallback
graph. A deterministic Host regression independently failed with three calls
before reuse and passed with two after reuse on a mixed success/fallback graph.
All nine Host regression executables, the DIM16 build, independent bitwise
Goldens, unchanged command-count checks and normal Paired quick smoke passed.
Both guests started for this task were powered off cleanly.

## Change and ownership contract

Production changes are limited to:

- `ggml/src/ggml-potal/residual/stripe-residual.cpp`: public packet wrapper,
  shared decoded-packet lowering, and a guarded actual-decode diagnostic timer.
- `ggml/src/ggml-potal/residual/stripe-residual.hpp`: module-private trusted
  lowering declaration/contract and test-only invocation-counter declarations.
- `ggml/src/ggml-potal/matmul/stripe.cpp`: one fallback request-builder call.
- `ggml/src/ggml-potal/CMakeLists.txt`: enable the existing testing macro only
  for the Paired execution regression target.

The new `build_request_from_decoded(decoded_packet &&, const hp1_weight &, request &)`
requires the result of a successful `decode_packet` for the same sealed,
immutable packet and the same prepared weight. It is an internal C++ Residual
entry, not a C11/RoCC API. It resets the destination, moves runs and lane rows,
then copies compact digits into the DMA A buffer while those digits still
belong to the caller's live `decoded_packet`. No borrowed digit pointer is
stored in the request. B and HP1 carriers are copied from the prepared weight
as before. The request continues to own its DMA allocations.

The entire old lowering body, from `const size_t runs = decoded.runs.size();`
through the existing allocation/length exception handlers, is byte-identical
(**1,805 characters**) to the retained B2 body. It still performs the same
run-ordered K32 expansion, zero padding, weight gather, carrier mapping,
allocation/alignment, dimensions/strides and tile selection. There is no second
implementation of the lowering algorithm.

The original public `build_request(packet, weight, out)` still resets out,
validates/decodes, returns the decode error unchanged, and only then calls this
shared lowerer. Decode and lowering retain their allocation/overflow handlers;
only the redundant outer wrapper handler was removed.

In `matmul/stripe.cpp:978`, the existing Serial branch now chooses:

```cpp
const rmd::RmdStatus built = cfg_.paired
    ? residual::build_request_from_decoded(std::move(decoded), w_, req)
    : residual::build_request(*s.packet, w_, req);
```

This condition is justified by the preceding control flow, not an unchecked
new route: an unsuccessful decode returns immediately; a non-capacity planning
refusal returns immediately; a successful Paired plan takes the Paired branch.
Thus, reaching this Serial branch with `cfg_.paired` and `s.packet` means a
recognized capacity refusal after successful decode of this same immutable
packet. The refusal status/detail and `serial_fallback` event are emitted
before this call, unchanged. Serial Main, the Residual descriptor/GEMM,
reconstruction/merge, fatal-abandon behavior and cleanup remain on their
original paths. With Paired disabled, the original packet builder is used.
Successful Indexed execution still consumes the original decoded packet.

The C11 source `ggml/src/ggml-potal/potal.c` remains byte-identical to the
retained B2 source:
`7dec7c2a9cfc0bc12cb3dcdb9abd93302c0dc69e813a812f6d332c87e16850bd`.
H3-3/H3-4 diagnostics, H3-6A profiling, the B0 benchmark and B1/B2 trace
optimizations were retained. No RTL, FPGA image, Scale/Fence/Drain semantics,
RoCC encoding, quantization, tolerance or model-execution behavior was changed.

## Exact invocation-count regression

The counter increments at entry to the real `decode_packet`, under
`GGML_GEMMINI_TESTING`, using a relaxed atomic. It counts actual invocations,
not planner events or inferred timing. The execution regression resets it only
when no decode is active, brackets `run_stripe_pipeline`, joins the owner,
and reads the difference. Fixture construction and independent Golden work
are outside the bracket. No sleeps or timing threshold determine the result.

The mixed dense fixture in `tests/potal-paired-execution-test.cpp:1463`
(path relative to `ggml/src/ggml-potal/`) has two residual packets, one
successful Paired stripe and one capacity fallback:

| Execution | Before actual decodes | After actual decodes | Expected |
|---|---:|---:|---:|
| Paired disabled / Serial | 2 | 2 | One per packet |
| Paired, one success + one fallback | 3 | 2 | One per packet, including fallback |

Before changing the fallback call, the new regression failed specifically at
`S dense Serial decodes once per packet; capacity-rejected Paired stripes reuse that decode`.
The run returned 1 and the expected-red wrapper reported
`H36C_EXPECTED_PRE_FIX_FAILURE_CONFIRMED`. Shared-lowering equivalence was
already passing at that point. After the fallback change the same regression
passed, reporting two actual calls. Successful Paired and Serial controls
also assert one decode per nonempty packet. Existing event-order, capacity
refusal, fatal failure and cleanup checks remain active.

## Numerical and lowering equivalence

`tests/potal-residual-unit-test.cpp:353` compares the public packet builder
against explicit successful decode plus moved decoded lowering in each
`run_case`. The destination is first populated with a request, dirty C and
extra run/row entries, proving that the trusted lowerer resets rather than
appends. Both resulting requests are executed independently in the existing
emulator and compared with independent numerical references.

The checks cover:

- Every block ID/local-K vector, run order, lane/source-row pair, compact and
  execution K, M/N, strides and every tile-plan field.
- Full padded A, gathered B, carrier and initial C buffers; DMA alignment;
  zero row/stride padding and K32 holes; per-run HP1 carrier association.
- Residual INT32 against the independent compact-run Golden, INT64 radix
  reconstruction against independent composition, and final F32 bitwise
  against the independent merge Golden and existing checked Gemmini merge.
- Compact lengths 1, 15, 16, 17, 31 and 32; sparse nonconsecutive blocks/K/rows;
  multiple runs; lanes 0..4 and mixed signs; a new dense two-run, 12-row,
  multi-lane case; zero/0/5/31/40 carriers and signed saturation.
- Existing invalid version, A4, foreign-shape and invalid-carrier rejection,
  ordering/range validation, INT64 limits/overflow and merge-error cases.

Existing cancellation-order and merge tests were retained rather than weakened.
The Host pipeline suite additionally passed its independent Main/Residual
INT32, INT64 correction and bitwise F32 checks, immutable accepted-input checks,
failure cleanup and deterministic scheduling/event checks.

## Host and DIM16 gates

All nine affected Host executables were rebuilt and run directly with
success-gated execution. The terminal returned 0 and ended with
`H36C_HOST_ALL_PASS`:

```text
potal-api-test
potal-residual-unit-test
potal-paired-plan-test
potal-paired-schedule-test
potal-paired-unit-test
potal-paired-auto-test
potal-paired-execution-test
potal-stripe-pipeline-test
potal-trace-test
```

An empty CTest registry was not treated as successful testing. The C11 and
Trace regressions ran even though their production source was unchanged.

The configured DIM16 tree `build-potal-h3-6a-d16` successfully rebuilt
`potal-paired-smoke`, `ggml-potal`, `potal-residual-unit-test` and
`potal-paired-execution-test` (exit 0, `H36C_DIM16_BUILD_PASS`). It retained
the existing RISC-V GCC 12.2.0 toolchain under
`/home/alveo/firesim/.conda-env/riscv-tools/bin/`, Release `-O3 -DNDEBUG`,
C++ `-std=gnu++17`, hardware execution, DIM16, Paired Auto, A8W8/ExSIA,
block K32 and job capacity 2. Existing compile/warning flags were not changed.
The generated hardware-parameter SHA256 printed in both guests was:
`b4cd8b9efef9f3cd9b598ae3cc84af6478f83e628ed73d94a77fb1c452df1fc7`.

The target build can be reproduced using the retained configuration:

```sh
source /home/alveo/miniforge3/etc/profile.d/conda.sh
cd /home/alveo/firesim/target-design/chipyard
source env.sh
cd generators/potal/software/workspace/llama.cpp-potal
cmake --build build-potal-h3-6a-d16 --target \
  potal-paired-smoke ggml-potal potal-residual-unit-test \
  potal-paired-execution-test -j8
```

Editor/LSP include resolution is pre-existingly incomplete; it is not claimed
as a clean diagnostics gate. Existing generated-header macro warnings and the
missing `chipyard/.conda-env` activation message were left in place. The
configured Host/cross compilers and actual executable gates succeeded. No full
RISC-V model suite was run.

## FireSim method and preserved executable identities

The first performance run used the retained B2 executable/library, copied
before any instrumentation or refactor, on the existing FPGA image. The C
run used the same driver, image, benchmark source, fixed input seed 20261023,
case, route ordering and seven repeats. The benchmark supplies Serial,
Paired and Conservative routes, cold runs and steady-cache runs. The primary
comparison below uses only the seven steady samples per route. C quick smoke
reported the ready issuer on Linux CPU 1, consistent with the existing owner
configuration. No new affinity or image configuration was introduced.

Unprofiled command in both guests:

```sh
./potal-paired-smoke --bench --case dense_fallback_160x320x512 --repeat 7
```

`POTAL_RAW_TRACE`, `POTAL_PAIRED_PROFILE`, `POTAL_PAIRED_PROBE`,
`POTAL_PAIRED_WINDOW_DIAG`, `POTAL_PAIRED_DIAG_INDEX_ENTRIES` and
`POTAL_PAIRED_STAGE_DIAG` were unset. There are no
`POTAL_DECODE_PROFILE` records in the unprofiled C section. Per-command
printing/tracing was not enabled.

| Artifact | SHA256 |
|---|---|
| Retained B2 smoke | 5350576c8dc32a0a4ce270c8e0d48e36020311955a4908859f261680231e0c58 |
| Retained B2 libggml-potal.so | 75efab94831166f795c71a628d60dc207588104d5c1e9df674e8333c2ef80c8b |
| Diagnostic-only B2 smoke | cc3877cc680652df86a928ac14d1eef0c2eadb7461c29c14822e666c3c163c90 |
| Diagnostic-only B2 library | 05471db24f27d815746330404e2d81ef1a0f1d90722dc2bf50812514077cd664 |
| C smoke | c694e07a3349929f6b3568cb3d6cc3867fbbd4323bed619482ca3f3a10c79c43 |
| C libggml-potal.so | 1aca66631382f52856d4a187822e456f03a7522a9c828c968811619377c240b6 |

Staged and guest hashes matched. The diagnostic-only B2 artifact was built
after adding the same actual-decode timer, but before extracting the lowerer
or changing fallback. It retained duplicate decode. It is used only for the
separate count/time observations, never as the unprofiled B2 baseline.

## Seven-repeat unprofiled results

All values are ms; MAD is median absolute deviation. Each row compares seven
steady samples with the same generated tensor pair. The C benchmark recorded
42 cold/steady route samples, all `ok=1`, and `failures=0`.

| Route | B2 compute median | B2 MAD | B2 min | B2 max | C compute median | C MAD | C min | C max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| serial | 139.542 | 0.707 | 138.497 | 141.483 | 139.773 | 0.558 | 139.215 | 141.620 |
| paired | 225.650 | 0.592 | 223.292 | 226.242 | 141.265 | 0.357 | 140.747 | 142.110 |
| paired_conservative | 225.651 | 1.415 | 222.595 | 227.149 | 141.633 | 0.852 | 139.456 | 142.485 |

| Route | B2 pipeline median | B2 MAD | C pipeline median | C MAD |
|---|---:|---:|---:|---:|
| serial | 138.955 | 0.896 | 139.111 | 0.476 |
| paired | 224.982 | 0.494 | 140.499 | 0.252 |
| paired_conservative | 224.993 | 1.415 | 140.910 | 0.867 |

The Paired samples do not overlap between versions (B2 minimum 223.292 ms,
C maximum 142.110 ms). Serial changed by +0.231 ms, within the observed
sample dispersion. Paired Conservative's extra median over Serial fell from
86.109 to 1.860 ms. These are differences of route medians, not medians of
per-iteration differences.

The remaining Paired fallback difference is not a second decode: actual-call
proof below excludes that mechanism. Planning, route bookkeeping and ordinary
run variation remain in this CPU-observed comparison. This experiment does
not assign an exact nanosecond budget to those components or propose another
optimization.

### Golden and unchanged work checks

The benchmark's `golden()` uses `potal_test::pipeline_golden`, not a prior
hardware output as its reference. Each `run_route` checks every output F32
bit with `memcmp`, graph success, unchanged failed-node counters, semantic
stripe count and merges equal to independent Golden packet count. Paired
additionally checks `paired_stripes + fallback_stripes == stripes` and
logical Serial Main GEMMs equal to fallback count. These checks are included
in `ok=1` and the final zero-failure gate, even where the compact benchmark
stats line does not print the corresponding field.

All 21 steady command/stat rows were compared with their corresponding fresh
B2 rows. Every count below matched in every repetition:

| Per-graph counter | Serial | Paired fallback | Conservative fallback |
|---|---:|---:|---:|
| Semantic stripes | 2 | 2 | 2 |
| Paired stripes | 0 | 0 | 0 |
| Capacity-fallback stripes | 0 | 2 | 2 |
| Serial Main LOOP_WS | 8 | 8 | 8 |
| Serial Residual LOOP_WS | 16 | 16 | 16 |
| Serial Main fences | 8 | 8 | 8 |
| Serial Residual fences | 16 | 16 | 16 |
| SCALE_WRITE | 30720 | 30720 | 30720 |
| SCALE_RELEASE | 30720 | 30720 | 30720 |
| Paired Main/Residual LOOP_WS | 0 / 0 | 0 / 0 | 0 / 0 |
| Paired Index configs/writes | 0 / 0 | 0 / 0 | 0 / 0 |
| Paired readbacks/tiles | 0 / 0 | 0 / 0 | 0 / 0 |
| Paired fences/drains | 0 / 0 | 0 / 0 | 0 / 0 |
| Producer capacity waits | 0 | 0 | 0 |

There were no failed nodes or mismatched corrections/merges. The capacity
reason remained `POTAL_PAIRED_RESIDUAL_ACC_CAPACITY` in diagnostics, and
fallback remained a fully Serial GEMM route. No work or synchronization was
removed to obtain the timing change.

## Separate actual-decode diagnostics

The actual `decode_packet` entry now emits one
`[POTAL_DECODE_PROFILE] stripe=... elapsed_ms=...` record only when
`POTAL_PAIRED_PROFILE=1`. Disabled diagnostics do not read the clock or print.
The guard performs one environment lookup per decode; it is not a per-RoCC
trace operation. The test atomic is absent from production builds. Existing
planner/profile output remains available.

The matched diagnostic runs used:

```sh
POTAL_PAIRED_PROFILE=1 ./potal-paired-smoke \
  --bench --case dense_fallback_160x320x512 --repeat 2
```

The selected observation is **steady iteration 2**, one clean graph per route,
after warmups. Actual decode records preceding that sample are counted and
summed; warmup records are excluded. This is not a seven-repeat median or a
component decomposition of the unprofiled graph timing.

| Route | B2 calls | C calls | B2 total decode ms | C total decode ms | Stripe sequence B2 -> C |
|---|---:|---:|---:|---:|---|
| serial | 2 | 2 | 85.312 | 84.957 | 0, 1 -> 0, 1 |
| paired_conservative | 4 | 2 | 169.153 | 85.823 | 0, 0, 1, 1 -> 0, 1 |
| paired | 4 | 2 | 169.721 | 84.336 | 0, 0, 1, 1 -> 0, 1 |

For B2 Paired, first decodes totaled 84.916 ms; the second decodes totaled
84.805 ms. C Paired records are stripe 0 = 43.729 ms and stripe 1 = 40.607 ms,
with no duplicate entry. Both versions still decoded each Serial packet once.
B2 Conservative's first/duplicate totals were 84.595/84.558 ms; C totaled
85.823 ms across two calls. All selected diagnostic graphs also passed Golden
and work-count checks. This establishes the removed mechanism independently
of the unprofiled 84.385 ms saving; profiled and unprofiled times are not mixed.

## Normal Paired smoke and shutdown

`./potal-paired-smoke --quick` passed the independent Golden for Serial,
Paired and Conservative on `small_48x96x256`. Both Paired routes executed
one Paired stripe and zero fallback stripes, with Main/Residual LOOP_WS 1/7,
Index configs/writes 7/150, readbacks/tiles 2/36, Scale writes/releases
2112/2112, and one merge matching one Golden packet. Paired fences/drains
were 1/9; Conservative fences/drains were 18/26. The final smoke stats reported
`failed_nodes=0`, and `POTAL_PAIRED_SMOKE PASS` and suite exit 0 were recorded.
These single quick-run timings are not used as performance evidence.

The B2 baseline guest and the C validation guest both received `poweroff`,
reported `reboot: Power down`, completed simulation with `*** PASSED ***`
and exited 0. The C UART archive ends with
`Script done ... [COMMAND_EXIT_CODE="0"]`. No guest or milestone watcher
started for this task remains running.

## Evidence files

Paths below are local artifacts, not new commits:

- `/tmp/h3-6c-before/`: original B2 smoke/library and source snapshots.
- `/tmp/h3-6c-before-run/uartlog`: fresh retained-B2 seven-repeat measurement
  and original existing-profile control, with clean shutdown.
- `/tmp/h3-6c-b2-diagnostic/`: old duplicate-decode executable/library with
  only the matched actual-call timer added.
- `/tmp/h3-6c-after-run/uartlog`: diagnostic B2 control, C unprofiled seven
  repeats, C separate diagnostics, C quick smoke, hashes and clean shutdown.
- `/tmp/h3-6c-after/`: C executable/library, exact production/CMake
  `source.diff`, `residual-unit-test.diff`, `verification.json` and
  `results.json` (steady samples, summaries, count comparison, actual decode
  records, quick-smoke counters and gate outcomes).
- `/tmp/h3-6c-before/decode-profile.json`: clean selected B2 diagnostic records.

The Paired execution test already contained earlier local diagnostic work;
its whole repository diff must not be mistaken for a C-only patch. The new
counter bracket and fallback assertion are identified above. The exact
production/CMake diff below is against the saved retained-B2 files, not HEAD.
Existing unrelated local changes were preserved. No commit or push was made.
Final scoped `git diff --check` passed on all six changed code/build/test files;
the branch remained `potal`, and the C11 source hash still matched retained B2.

## Exact production/CMake diff against retained B2

```diff
--- /tmp/h3-6c-before/stripe-residual.cpp	2026-10-25 15:50:36.473910750 +0900
+++ ggml/src/ggml-potal/residual/stripe-residual.cpp	2026-10-25 15:57:33.150776393 +0900
@@ -5,7 +5,9 @@
 #include "residual/rmd/rmd-builder.hpp"
 
 #include <algorithm>
+#include <chrono>
 #include <cmath>
+#include <cstdio>
 #include <cstdlib>
 #include <cstring>
 #include <limits>
@@ -13,6 +15,10 @@
 #include <stdexcept>
 #include <utility>
 
+#if defined(GGML_GEMMINI_TESTING)
+#include <atomic>
+#endif
+
 namespace ggml::potal::residual {
 
 namespace {
@@ -25,6 +31,32 @@
 static_assert(rmd::kBlockSize == RUN_K, "a residual run is one original weight block = one HP1 scale block");
 static_assert(RUN_K % HW_DIM == 0, "a run is a whole number of DIM-wide passes");
 
+#if defined(GGML_GEMMINI_TESTING)
+std::atomic<uint64_t> decode_calls{ 0 };
+#endif
+
+struct decode_profile {
+    size_t stripe;
+    bool enabled;
+    std::chrono::steady_clock::time_point start;
+
+    explicit decode_profile(size_t id) : stripe(id) {
+        const char * env = std::getenv("POTAL_PAIRED_PROFILE");
+        enabled = env != nullptr && std::strcmp(env, "1") == 0;
+        if (enabled) {
+            start = std::chrono::steady_clock::now();
+        }
+    }
+
+    ~decode_profile() {
+        if (enabled) {
+            const double ms = std::chrono::duration<double, std::milli>(
+                std::chrono::steady_clock::now() - start).count();
+            std::fprintf(stderr, "[POTAL_DECODE_PROFILE] stripe=%zu elapsed_ms=%.6f\n", stripe, ms);
+        }
+    }
+};
+
 __extension__ typedef __int128 wide_t; // checked radix composition (no INT64 narrowing in between)
 
 // Position of `lane` among the block's active lanes, or -1 when the block has no digit of it.
@@ -55,6 +87,16 @@
 
 } // namespace
 
+#if defined(GGML_GEMMINI_TESTING)
+void reset_decode_packet_calls() {
+    decode_calls.store(0, std::memory_order_relaxed);
+}
+
+uint64_t decode_packet_calls() {
+    return decode_calls.load(std::memory_order_relaxed);
+}
+#endif
+
 request::~request() {
     for (void * p : blocks) {
         std::free(p);
@@ -98,6 +140,10 @@
 }
 
 rmd::RmdStatus decode_packet(const rmd::StripePacket & packet, const hp1_weight & w, decoded_packet & out) try {
+#if defined(GGML_GEMMINI_TESTING)
+    decode_calls.fetch_add(1, std::memory_order_relaxed);
+#endif
+    const decode_profile profile(packet.stripe_id);
     out = decoded_packet{};
     if (rmd::validate_packet(packet) != rmd::RmdStatus::success) {
         return rmd::RmdStatus::invalid_packet;
@@ -174,13 +220,18 @@
     return rmd::RmdStatus::overflow;
 }
 
-rmd::RmdStatus build_request(const rmd::StripePacket & packet, const hp1_weight & w, request & out) try {
+rmd::RmdStatus build_request(const rmd::StripePacket & packet, const hp1_weight & w, request & out) {
     out.reset();
     decoded_packet       decoded;
     const rmd::RmdStatus status = decode_packet(packet, w, decoded);
     if (status != rmd::RmdStatus::success) {
         return status;
     }
+    return build_request_from_decoded(std::move(decoded), w, out);
+}
+
+rmd::RmdStatus build_request_from_decoded(decoded_packet && decoded, const hp1_weight & w, request & out) try {
+    out.reset();
     const size_t runs = decoded.runs.size();
     out.runs          = std::move(decoded.runs);
     out.rows          = std::move(decoded.rows);

--- /tmp/h3-6c-before/stripe-residual.hpp	2026-10-25 15:50:36.769908564 +0900
+++ ggml/src/ggml-potal/residual/stripe-residual.hpp	2026-10-25 15:57:33.142776454 +0900
@@ -129,12 +129,25 @@
 // nonzero digit is invalid_packet; unsupported_route: not an A8 packet.
 rmd::RmdStatus decode_packet(const rmd::StripePacket & packet, const hp1_weight & w, decoded_packet & out);
 
+#if defined(GGML_GEMMINI_TESTING)
+// Test-only invocation counter; reset only while no decode is active.
+void reset_decode_packet_calls();
+uint64_t decode_packet_calls();
+#endif
+
 // validate_packet, the run-aware rows/runs, the padded operands gathered from
 // the prepared weight and the residual plan, into `out` (reset first).
 // RmdStatus::success only with a complete request; a packet without any
 // nonzero digit is invalid_packet; unsupported_route: not an A8 packet.
 rmd::RmdStatus build_request(const rmd::StripePacket & packet, const hp1_weight & w, request & out);
 
+// Module-private K32 lowering, without a second packet validation/decode.
+// `decoded` must come from a successful decode_packet of the same sealed,
+// immutable packet and prepared weight w. Consumes runs/rows; compact digits
+// remain alive through A staging. Resets out and retains build_request's
+// allocation, tile-selection and failure behavior.
+rmd::RmdStatus build_request_from_decoded(decoded_packet && decoded, const hp1_weight & w, request & out);
+
 // The Residual GEMM's C (req.c) -> the stripe's BlockScaledInt64Correction,
 // row_count x n values (stripe-local rows). Rows must be strictly increasing
 // in (lane, source row): a duplicate or missing ordering is invalid_packet.

--- /tmp/h3-6c-before/stripe.cpp	2026-10-25 15:50:36.913907501 +0900
+++ ggml/src/ggml-potal/matmul/stripe.cpp	2026-10-25 16:01:24.984595358 +0900
@@ -972,7 +972,11 @@
             // 2. Residual GEMM of the sealed packet, on the same accelerator.
             if (s.packet) {
                 emit(advance(s, slot_state::residual_prepare, stripe_event_kind::residual_prepare_begin));
-                const rmd::RmdStatus built = residual::build_request(*s.packet, w_, req);
+                // In this Serial branch, cfg_.paired means a recognized capacity
+                // refusal after a successful decode of this same immutable packet.
+                const rmd::RmdStatus built = cfg_.paired
+                    ? residual::build_request_from_decoded(std::move(decoded), w_, req)
+                    : residual::build_request(*s.packet, w_, req);
                 if (built != rmd::RmdStatus::success) {
                     return failed("residual_prepare", POTAL_OK, false, rmd::rmd_status_message(built));
                 }

--- /tmp/h3-6c-before/CMakeLists.txt	2026-10-25 15:50:37.129905906 +0900
+++ ggml/src/ggml-potal/CMakeLists.txt	2026-10-25 15:52:56.052872464 +0900
@@ -386,6 +386,9 @@
     endif()
 endforeach()
 
+# Counts real Residual decode calls in the fallback regression, not elapsed time.
+target_compile_definitions(potal-paired-execution-test PRIVATE GGML_GEMMINI_TESTING)
+
 # Benchmark (host and target, not a test): Q8_HP1 weight preparation cost per
 # stage and with the backend's weight cache, synthetic or from a GGUF (--gguf).
 # GGML_GEMMINI_TESTING enables the Gemmini reader's call counters in this target only.
```
