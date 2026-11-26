# Phase 2B test results

Common extraction is implemented and regression compatibility is PASS. This is
not an all-green test-suite claim: 67 pre-existing failures and 16 existing
NOT RUN outcomes remain. The original INT4 attention fixture is still BLOCKED
and its full build still fails with the same 25 compiler diagnostics. Its
bounded migration attempt, independent scratch proof and 181-assertion gap
ledger are in test-readiness.md.

Evidence root E = build-npu/phase2b-evidence. Final logs are E/final.
Phase 2A logs and per-name results were copied to E/pre before any rerun.
The verified phase2a-complete checkpoint is referenced in
build-npu/checkpoints/README.txt. The Git index was not changed; no commit,
push, reset, stash, checkout or clean was performed.

## Comparable matrix

| Lane | Phase 2A PASS / FAIL / NOT RUN | Phase 2B PASS / FAIL / NOT RUN | Build |
| --- | --- | --- | --- |
| macOS CPU-only | 56 / 4 / 0 | 56 / 4 / 0 | PASS |
| Gemmini API-only | 5 / 0 / 0 | 5 / 0 / 0 | PASS |
| macOS Gemmini INT8 | 110 / 9 / 0 | 110 / 9 / 0 | PASS |
| macOS Gemmini A4W4-on-INT8 | 104 / 9 / 0 | 104 / 9 / 0 | PASS |
| Linux CPU-only | 55 / 6 / 0 | 55 / 6 / 0 | PASS |
| Linux Gemmini INT8 | 107 / 13 / 0 | 107 / 13 / 0 | PASS |
| PoTAL INT8 D16 | 75 / 9 / 1 | 76 / 8 / 1 | PASS |
| PoTAL INT8 D32 | 71 / 11 / 1 | 72 / 10 / 1 | PASS |
| PoTAL INT4 D32 | 66 / 8 / 14 | 66 / 8 / 14 | FAIL, unchanged stale fixture |
| Total, 734 registrations | 649 / 69 / 16 | 651 / 67 / 16 | qualified |

No test was added, removed, renamed or newly disabled. The two FAIL-to-PASS
changes are potal-stripe-pipeline-test in D16 and D32 after the deterministic
thread-exit repair. All 67 retained failures have the same failing assertions
or error messages. E/per-test-comparison.json records every name, status,
raw before/after failure reason and evidence path. Only source-root/line
locations and declared namespace renames are normalized; numeric expectations
and assertion messages are retained. Counts alone were not used as proof.

Three source-contract tests initially failed because their file paths still
named moved owners. Their paths and split-header coverage were corrected,
without dropping assertions. Only affected tests were rerun. The final ledger
overlays final/{mac-int8,mac-a4,linux-int8}-contract-recheck.ctest.log on each
full log. The retained profiled-domain failure still says:
'F4: non-pipeline scalar-start behavior retains both domains: expected 1, found 2'.
Initial attempts remain available; none is silently labelled PASS.

INT4 attention has exactly the same 25 compiler diagnostic sites/messages before
and after, comprising 12 distinct message texts: E/int4-build-comparison.json.
Its existing w8-only disabled registration remains intact. The supported W4
numerical golden/QK-transfer/QK-residual tests pass; this does not repair or
replace the stale attention fixture.

## Numerical, command and PB checks

| Check | Result | Evidence |
| --- | --- | --- |
| Activation values and complete numeric metadata, including optional run identity | Bitwise identical, A4/A8 | E/bitwise-final.log and bitwise-{snapshot,current}-a{4,8}.txt |
| Weight codes/carrier/column scale, residual bitmap/group descriptors and limbs | Bitwise identical in deterministic fixtures | E/bitwise.cpp and the same outputs |
| Original 52 SDK command streams, ordering and drain markers | PASS; FULL_STREAM_GOLDEN_PASS | final/golden-full-verified.log |
| Existing CPU numeric golden hashes | PASS via original functional tests | native matrix and PB functional logs |
| PB Q8_0 block/attention plus model isolation/sticky failure | 3/3 PASS | final/gemmini-int8-d32-pb-reference.ctest.log |
| PB MXINT8 block/attention plus functional model | 3/3 PASS | final/gemmini-int8-d32-pb-mxint8-reference.ctest.log |
| PB Q8_0 five-core block and attention | 2/2 PASS, all CPU mismatch counters zero | final/gemmini-pb-q8-0-*-test.five-core.log |
| PB runner --help | PASS | final/pb-e2e-help.log |

The differential harness uses the restored checkpoint's source, the same
compiler/options and selected geometry, and field-wise hashes rather than
object padding. It covers 24 activation configurations (six algorithms,
two residual routes, two widths) plus two HP1 weight fixtures. It is fixture
coverage, not a universal proof over every possible input. Existing tests
supply exact partial/reconstruction and FP32 output contracts. Existing golden
files were not regenerated. The SDK capture adapter's historical namespace was
restored verbatim after an over-broad internal rename initially broke capture.

## Standalone APIs and cross-builds

Gemmini standalone API: 5/5 PASS. PoTAL INT8 D16 standalone: 4/4 PASS.
PoTAL INT8 D32 and INT4 D32 standalone: build PASS, 3/4 tests PASS; potal-api-test
retains the macOS 'stream failure isolation requires fork' failures. These were
reproduced on phase2a-complete, not assumed pre-existing:
final/snapshot-api-INT8-d32.* and snapshot-api-INT4-d32.*.
Supplemental D64 standalone full test build FAILS on the unchanged assertion
'stripe fixtures require DIM16 or DIM32'; the checkpoint reproduces it in
final/snapshot-api-INT8-d64.build.log. The standalone API libraries themselves
build. E/build-boundaries.json records these comparisons and actual source sets.

All four required RISC-V builds PASS (configure=0, build=0): Gemmini API,
Gemmini RoCC, PoTAL RoCC and Gemmini PB. Evidence: final/riscv-summary.log and
final/linux-logs/rv-*.{configure,build}.log. They are cross-builds, not hardware
execution. CPU-only/API-only compile databases contain no common numerical
sources; CPU-only contains no hardware API sources. Actual dynamic linkage is
recorded in final/linked-libraries.log.

## Order, commands and acceptance

The ordered change units were: existing checkpoint reference; separate test
readiness repair and baseline; argument/storage split; activation/weight move;
residual/buffer move; profiling/options and observation callbacks; common-target
and namespace integration; bounded INT4 attempt; final matrix and affected
rechecks. Stage logs are types-*, quant-*, residual-*, profiling-*, common-* and
extraction-* under E. Representative numerical tests ran after each stage;
the full matrix ran once, followed only by affected rechecks.

The existing socm-baseline:ubuntu24 image and build-npu/run-npu-matrix.sh were
reused. E/final-mac.sh, final-linux.sh and final-pb.sh record the exact final
commands. Linux builds reuse build-npu/phase2a-linux after preserving its logs;
new logs are copied by lane into E/final/linux-logs. No image or dependency was
downloaded. No old log was reused as a new result.

| Gate group | Verdict and scope |
| --- | --- |
| Real common ownership, neutral types, quantization, weight/dequant, residual, profiling, software policy; no ISA calls | PASS; common-software.md, common-types.md, common-profiling.md |
| Real common target, removed PoTAL Gemmini sources/shim, deduplicated identical compilation, isolated test ABI | PASS; common-dependencies.md and E/layering-audit.json |
| Single precision, unchanged generator/validation, supported API and CPU-only builds | PASS builds; standalone fixture exceptions above remain FAIL |
| Independent APIs, explicit transport ownership, real PB command consumption, PoTAL issuer ownership and completion contracts | PASS preservation evidence; command/PB/PoTAL regressions above |
| IM2P/CYCLE_SIM/legacy simulator implementation absent | PASS active-source audit; no new external SDK path |
| Original command and CPU numeric goldens, Gemmini/PB/PoTAL comparisons and RISC-V cross-builds | PASS compatibility; INT4 full-build fixture remains FAIL |
| Existing failures distinguished from regressions | PASS, all names and 67 retained failure reasons compared |
| INT4 fixture readiness | BLOCKED after bounded attempt; no weakened or removed assertions |
| Physical hardware, DMA coherence, hardware performance/cycles | NOT RUN; no physical target supplied |
| LLM generation | NOT RUN / BLOCKED; no model-weight GGUF or prompt-token fixture supplied |
| Full FP32 overlay and CUDA accuracy execution | NOT RUN, explicitly outside Phase 2B; static wiring checked |

Remaining Phase 2B work is the complete INT4 attention fixture migration.
Phase 2C retains FP32/CUDA/E3/E4 isolation; Phase 2D retains final public option
and naming cleanup; Phase 3 retains PoTAL INT8/INT4 consolidation; Phase 4
retains the common NPU protocol. None is represented as already implemented.
