# Phase 2C results

PASS: isolation and production regression compatibility. This is not an
all-green test-suite claim. The comparable Phase 2B matrix retains 734
registrations: 651 PASS, 67 FAIL, 16 NOT RUN. Full INT4 build still FAILS
on its unchanged attention fixture. FP32 and CUDA numerical execution are
BLOCKED by absent dependencies; physical hardware and LLM execution are
NOT RUN.

Evidence root E: `build-npu/phase2c-evidence/`. Phase 2B final logs and
per-test results were copied into E/pre before any rerun. New results are
E/final; none is an old log relabelled as new evidence.

| Comparable lane | PASS / FAIL / NOT RUN | Build |
| --- | --- | --- |
| macOS CPU | 56 / 4 / 0 | PASS |
| Gemmini API | 5 / 0 / 0 | PASS |
| macOS Gemmini INT8 | 110 / 9 / 0 | PASS |
| macOS A4W4 on INT8 | 104 / 9 / 0 | PASS |
| Linux CPU | 55 / 6 / 0 | PASS |
| Linux Gemmini INT8 | 107 / 13 / 0 | PASS |
| PoTAL INT8 D16 | 76 / 8 / 1 | PASS |
| PoTAL INT8 D32 | 72 / 10 / 1 | PASS |
| PoTAL INT4 D32 | 66 / 8 / 14 | FAIL, existing fixture |

`per-test-comparison.json` compares every name, status and retained failure
reason. There are no added, removed, renamed or newly disabled production
registrations. All 67 retained failure reasons match. ANSI/whitespace and
source-root/line locations are normalized, never numerical expectations.
The GGUF test's buffered per-case messages vary; both saved baseline and
current logs terminate on the same `ggml_is_contiguous(&info.t)` assertion.

One initial Linux geometry source scan enumerated an old report path just
before archival and then failed to open it. After archival completed, the
unchanged test passed in `final/linux-geometry-recheck.ctest.log`. The ledger
overlays this single recheck and keeps the initial failure log. No assertion
was changed to hide the race. Lead review subsequently identified four live
paired contract documents, restored unchanged to their backend locations.
The affected source scan passed again in
`final/linux-contract-restoration.ctest.log`; no rebuild was needed.

`int4-build-comparison.json` retains all 25 original diagnostic sites and
12 distinct messages, identical before/after. No INT4 attention fixture
change or target exclusion was made. Standalone PoTAL INT8 D16 passes 4/4;
INT8 D32 and INT4 D32 each retain 3/4 PASS and the macOS fork-isolation
failure. Supplemental D64 build still FAILS on the existing DIM16/DIM32
stripe-fixture static assertion. These remain FAIL, not compatibility PASS
substituted for test success.

| Additional check | Status | New evidence |
| --- | --- | --- |
| Original 52 complete command streams and drains | PASS | `final/golden-full.log`, `FULL_STREAM_GOLDEN_PASS` |
| PB Q8_0 block/attention and functional model | PASS, 3/3 | `final/gemmini-int8-d32-pb-reference.ctest.log` |
| PB MXINT8 block/attention and functional model | PASS, 3/3 | `final/gemmini-int8-d32-pb-mxint8-reference.ctest.log` |
| PB Q8_0 five-core block and attention | PASS, 2/2 | `final/gemmini-pb-q8-0-*.five-core.log` |
| PB runner help | PASS | `final/pb-e2e-help.log` |
| Four RISC-V cross-builds | PASS | `final/riscv-summary.log`, `final/linux-logs/rv-*` |
| Production dependency audit | PASS | `production-isolation.json`, `dependencies/` |
| CUDA-enabled host C++ adapters | PASS, 8/8 syntax checks | `cuda-host-syntax-summary.log`, `optional/*.cuda-syntax.log` |
| FP32 independent lane diagnostics | PASS | `optional/fp32-{reference,rocc-linux}.configure.log` |
| FP32 native/target numerical and RTL vector execution | BLOCKED | Missing verified SDK overlay/vectors |
| CUDA kernels and numerical tests | BLOCKED | `optional/cuda-{exact,rtn}.configure.log`: missing nvcc |
| Physical hardware, DMA coherence, hardware timing | NOT RUN | No target supplied |
| LLM generation | NOT RUN | No model-weight GGUF/prompt fixture supplied |
| Fresh A4/A8 differential metadata harness | NOT RUN | Common numerical/type sources unchanged; `preservation.json` plus existing matrix numeric tests |

## Explicit experiments

`check-experiments.sh` enables all six existing experiment options and builds
eleven host targets per precision/geometry. INT8 D16 has six resident/E4
CTest passes and two direct E3 acceptance passes. Two experiment-only
compile defects were exposed: the obsolete index-capacity macro spelling
and fixture `main` macro pollution of a production header. Their causes
and repairs are documented in `potal-experiments.md`. All eleven D16 target
builds succeed after the two affected targets are rebuilt; the initial
failed build and successful repairs remain separately logged.

INT8 D32 and INT4 D32 experiment builds remain FAIL for four unchanged
DIM16-only static assertions in E3 context, E4A resident/model and E4C1
fixtures. In each lane the E4 CTest group has three FAIL and three NOT RUN.
The newly executable E3 prototype also FAILS its existing larger-geometry
or precision assertions. These optional experiments were not in the 734
default Phase 2B registrations. They are newly exercised limitations, not
passing production results and not a reason to remove coverage. No
arithmetic or oracle was changed to make unsupported fixtures pass.
The rebuilt E4C1 benchmark compiles in all three lanes; physical benchmark
execution is NOT RUN.

Detailed commands/statuses: `experiment-summary.log`,
`experiment-repair-summary.log`, `experiments/*.build.log`,
`experiments/*.repair-build.log`, `experiments/*.ctest.log`,
`experiments/*.potal-e3-*.log`. Original target names remain; the E4C2-0
test's implementation moves to `experiments/prefetch-admission/`.
E3 context/prototype remain direct commands because they were never CTest
registrations. Every moved path is in `source-moves.json`.

## Commands and scope

The shared build definitions changed, so the full Phase 2B matrix ran once,
using the existing `socm-baseline:ubuntu24` image and
`build-npu/run-npu-matrix.sh`. No image or dependency was downloaded.
`final-mac.sh`, `final-linux.sh`, and `final-pb.sh` under E record the exact
commands and reused build directories. The only subsequent production
test rerun is the affected geometry scan; experiment compile repairs have
their own narrow rechecks. Source changes do not touch numerics or command
generation. Original goldens are unchanged and were not regenerated.

The acceptance ledger is `acceptance-gate.json`. Remaining work is P2B-R
INT4 attention fixture migration, Phase 2D public options/naming, Phase 3
PoTAL consolidation, and Phase 4 common protocol. None is represented as
implemented by this isolation phase.
