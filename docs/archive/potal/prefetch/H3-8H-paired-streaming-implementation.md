# H3-8H Paired Streaming implementation

Status: the requested implementation and validation are complete, with
negative overlap/performance results. All new routes remain opt-in; no
default was promoted. A/B and the separate C image passed U280 numerical gates. The A/B
cohort's 24 diagnostic graphs showed zero qualified future-B-write /
current-Residual PE00 coincidence. Its six-route primary logs and all 42
GPT-2 E2E requests are collected and analyzed. Baseline A/B regressed versus
K-tile in both TTFT and mean TPOT. C's 35-case/API/Attention gates and its
closed physical trace are collected. All 36 C-image diagnostic graphs passed
identity/conservation checks but showed zero qualified coincidence.
C primary performance is verified: 294 samples and 42 summaries; C is slower
than K-tile at steady state by 35.81% on decode, 19.15% on FFN and 1.67% on
tail. On 2026-10-31 the user reduced C E2E to one request per route.
All seven requests, output comparisons and timestamp checks passed. C's
single observed TTFT/mean-TPOT were 33.12%/91.69% slower than same-image
K-tile; repeat MAD is unmeasured. These results do not establish a useful
optimization. Existing broad-test, methodology and RAM_STYLE findings below
remain disclosed; this is not warning-free platform signoff.

## Final artifacts and reproduction

All paths below are under `/tmp/h3-8h/` unless stated otherwise.

| Evidence | Artifact |
|---|---|
| Final source, dirty status and patches | `source-v17-status.txt`, `source-v17.sha256`, `software-v17.patch`, `rtl-v17.patch`, `*-source-v17.tar.gz` |
| Scripts, configuration, actual collected data and failed/successful root logs | `reproduction-v17.tar.gz`; the active capture log is excluded |
| Final repository preservation audit | `final-scope-audit.log`; starting HEADs/gitlink and the explicit CDE checksum are checked |
| Scalar deployed software | `early-stream-software-v13.tar.gz`, `early-stream-software-v13.sha256`, `early-stream-software-v13-archive.sha256` |
| C bitstream and driver packages | `firesim-early-stream.tar.gz`, `early-stream-driver.tar.gz`, `early-stream-bitstream.sha256`, `early-stream-driver.sha256` |
| Routed reports and inherited RAM finding | `early-stream-cad-reports.tar.gz`, `early-stream-route-audit.md`, `early-stream-ram-finding.md` |
| Exact FPGA gates | `early-stream-gates-collected/gates`, `collect-early-stream-gates.log` |
| Physical capture and audit | `early-stream-physical.tar.gz`, `early-stream-physical.sha256`, `early-stream-physical-results.md`, `analyze-early-stream.log` |
| Primary timing and logical Tile counts | `early-primary-results.md`, `analyze-early-primary.log`, both `*-performance-collected` directories |
| Separate CPU API diagnostic | `early-admission-results.md`, `analyze-early-admission.log` |
| E2E results, boundaries and full per-request data | `e2e-final-results.md`, `analyze-e2e.log`, `baseline-e2e-results.md`, `baseline-e2e-analysis.json`, both `*-e2e-collected` directories |

C software archive SHA-256:
`b4af0332e2be104b7187b33a3c84bb6d1f27d335487ab5332ccbcfb3af26bfba`.
C bitstream:
`0bb86df67bfeb343082599f4b4224179dc9c64a2127ee4dd341b9144bb1195c9`.
C driver:
`06976e57db0373813cd1c0b63da32f54666507c3c77fc73d2289341667b12545`.
The deployed C/C++ executables remain v13. The subsequent one-repeat change
replaced only the guest orchestration script and adjusted collection/analysis;
`reconfigure-e2e-once.log` records backup, fsck and round-trip comparison.
The model is `gpt2.Q8_HP1.gguf`, SHA-256
`cf23a45e4db23736d7d1f67ab42018803a198cfcb01db36daca69ff56bdbc7d4`.
`model.sha256` and exact `prompt.txt` are in the reproduction archive;
the final source manifest also hashes those input files.

Reproduce analysis of the retained closed data without rerunning hardware:

```bash
source /home/alveo/miniforge3/etc/profile.d/conda.sh
cd /home/alveo/firesim/target-design/chipyard
source env.sh
conda activate /home/alveo/firesim/.conda-env
bun=/home/alveo/.omo/agent/bundled-bun/ca66da8b3b88cc87/bun
"$bun" /tmp/h3-8h/analyze-early-primary.ts
"$bun" /tmp/h3-8h/analyze-early-admission.ts
"$bun" /tmp/h3-8h/analyze-e2e.ts
"$bun" /tmp/h3-8h/analyze-early-stream.ts
```

The exact guest entry point was
`H3_8H_RESULT_ROOT=results-early-stream sh /root/h3-8h/run-guest.sh e2e early`.
It prints `routes=7 repeats=1 requests=7`; the fixed model/prompt, eight tokens,
seed, masks, flags and no-warmup CLI are in the archived script.
No further inference run is required to reproduce the analysis.
Failed builds, interrupted preflight and pre-existing failures were retained,
not overwritten or relabeled PASS.

## Authoritative design correction

The latest direct correction restores Pair-based GEMM Tile streaming as the
final H3-8H goal. Independent A/B selectors and Residual I-tiling are deferred,
not acceptance requirements. Their earlier requirement note is preserved but
marked superseded. Current contract:
`/tmp/h3-8h/tile-streaming-correction.md`.

The intervening independent-bank work was read-only source analysis and
documentation. A byte comparison against frozen source archives confirmed
that neither the C/C++ issuer nor production RTL was changed for it.
The existing paired-bank FPGA recovery therefore remains relevant to the
corrected goal. No implementation was reset, reverted or discarded.

The actual H3 path is `paired-auto.cpp` and `potal.c`, not the separate
`paired-schedule.cpp` emulated microtile scheduler. `main_gemm` emits one
LOOP_WS with the logical Main operation's I/J/K. `residual_gemm` emits one
LOOP_WS per compact Index window, using residual row tiles and that window's
K. It does not subdivide I. Main and Residual share original B residency while
their I and execution-K may differ. Existing capacity refusal remains intact.

Main/Residual DIM operations inside a LOOP_WS are not additional GEMM Tiles
or Activation prefetch operations. A is loaded by GEMM Tile; B is loaded by
K/J residency and reused across its Residual windows. Same-pair A/window
reuse may still require a local completion boundary. Occupancy remains the
checked I/K/J footprint, not a fixed Main-A:Residual-A:Weight ratio.

The production host path first reconstructs Main INT32 into F32, reconstructs
Residual radix terms into `BlockScaledInt64Correction`, then performs the
existing checked F32 merge. Correction scaling uses double intermediates and
a float delta added to the Main float. This is not full Main-plus-Residual
INT64 summation. `stripe.cpp:1085-1114` and
`residual/stripe-residual.cpp:332-407` define the selected path; it is unchanged.

Matched measurement now includes six routes: H2 baseline, K-tile, D2,
Early LdB, bounded Tile streaming and continuous Tile streaming. GEMM Tile,
Activation Tile and Weight Tile counts will be reported separately, with
command-declared counts distinguished from physical write observations.

## Baseline and preservation

- Software baseline and starting HEAD:
  `8bd379b1bc48d57825b93d5cb3bee450b34591b6`.
- Chipyard baseline and starting HEAD:
  `ce8e34cef08f48fe2a24db26a64af27aaf07da50`.
- Original Gemmini pin and checkout:
  `185293e87cbbc2a336410f49a3fd73d01ed8d04e`.
- FireSim starting HEAD: `6204ce53bdf970db757d47cf936b6f6ba4b1a4c9`,
  on `potal`. The earlier misplaced commit on `aisa-customizations` was
  already reverted. This implementation authorizes no commits or pushes.
- No interrupted H3-8H edits or running build were found at startup.
  The process-search command returned 1 because no matching process existed.

Existing Gemmini software/submodule dirt, host-init changes, overlay files,
deleted smoke/runOutput files, CDE streams, test output and model files are
outside this change. Original Gemmini and all submodule pins stay unchanged.

Every shell sources conda.sh, then Chipyard env.sh, then activates
`/home/alveo/firesim/.conda-env`. Chipyard env.sh references a missing local
`.conda-env`; its existing `EnvironmentLocationNotFound` warning is retained.
The available FireSim environment supplies the build tools.

## Implementation seams

The existing pure paired encoder validates addresses, geometry and Scale
fields. Legacy emission inserts four CONFIGs and a CPU publication fence per
loop. H2 ownership separately tracks pending B and publishes readability only
after a checked global drain. These legacy meanings are preserved.

The existing immutable execution plan determines tile shapes, Main/Residual
contribution order, compact fragments, Scale alias eligibility and ACC
first-contribution flags. Streaming compiles a separate metadata/completion
schedule from that plan; it does not change the plan or independent goldens.

The implementation adds equal maximum aligned residual row pitch across all
windows. Original compact column order, filler zeroes and each window's K
padding remain unchanged. Host pitch padding and command-declared load
extents are distinct counters; neither is measured DMA traffic.

## Selection

The new explicit environment option is
`GGML_POTAL_PAIRED_STREAM=OFF|BOUNDED|CONTINUOUS|CONTINUOUS_EARLY`, default `OFF`.
It requires DIM32 Paired K-tile execution. Legacy prefetch, Early and
conservative-fence flags are incompatible with the basic stream route.
Unsupported capability or failed explicit selection must fail, not run Serial,
H2, D2 or Early as a substitute.

Use `GGML_POTAL_PAIRED_AUTO=ON`, `GGML_POTAL_PAIRED_KTILE=ON`,
`GGML_POTAL_PAIRED_FENCES=RTL`, and select `BOUNDED` or `CONTINUOUS`.
Keep `GGML_POTAL_PAIRED_PREFETCH=OFF` and
`GGML_POTAL_PAIRED_EARLY_PREFETCH=OFF`. No default route was promoted.
`CONTINUOUS_EARLY` separately negotiates B-only composition; it does not enable
either legacy prefetch flag or silently fall back on an older image.

## Verification ledger

| Gate | Current result |
| --- | --- |
| Baseline/current SHA and pin inspection | Passed |
| Fixed-pitch and stream scheduling source | Implemented; command-model tests passed |
| Checked C11 API and direct tests | DIM16/DIM32 focused gates passed |
| RTL epochs, LOAD3 and scoped completion | Compiled; focused module gates passed |
| Software numerical/command regression | Focused gates passed; existing broad-suite failures below |
| RTL adversarial concurrency regression | Module/composition gates passed; not full-SoC proof |
| Synthesis/timing and FPGA golden | A/B image passed 30-case gates; C image passed routed timing and 35-case/API/Attention/PV-range gates; inherited methodology/RAM findings retained |
| Identity-linked physical overlap | A/B image: zero in 24 graphs; C image: zero in all 36 graphs, including 12 C graphs; identity/conservation checks passed |
| Same-bitstream matched latency | A/B cohort: 252 exact samples audited; C-image cohort: 294 samples and 42 summaries audited; A/B/C steady medians regressed versus K-tile |
| End-to-end TTFT/TPOT | Baseline 42/42 and C 7/7 requests analyzed; all 490 events and token identities checked; C repeat MAD unmeasured; A/B/C regressed versus same-image K-tile |
| Conditional Early composition | Native/directed RTL, numerical and physical conservation checks passed; 128 B-only producers observed, zero qualified coincidence; steady elapsed time regressed versus K-tile |

### Separate A, B and C verification status

| Variant | Native / directed RTL | FPGA numerical gates | Qualified physical coincidence | Primary elapsed time versus K-tile | GPT-2 E2E |
| --- | --- | --- | --- | --- | --- |
| A: bounded | Focused gates passed; broader existing findings retained below | Passed on A/B and C images | None in its 12 diagnostic graphs on either image | C-image steady cohort, slower: decode 34.20%, FFN 20.41%, tail 0.82% | C-image single request: TTFT +32.41%, mean TPOT +90.32% versus K-tile |
| B: continuous | Focused gates passed; broader existing findings retained below | Passed on A/B and C images | None in its 12 diagnostic graphs on either image | C-image steady cohort, slower: decode 26.89%, FFN 18.30%, tail 1.30% | C-image single request: TTFT +29.66%, mean TPOT +84.07% versus K-tile |
| C: continuous Early B-only | DIM32 software numerics, DIM16 compatibility and 27/27 directed RTL tests passed | Passed: five C fixtures within the 35-case cohort, plus API/Attention gates | None in all 12 C graphs; 128 real B-only producers across the C cohort | C-image steady cohort, slower: decode 35.81%, FFN 19.15%, tail 1.67% | C-image single request: TTFT +33.12%, mean TPOT +91.69% versus K-tile |

Software numerical proof, directed RTL proof, FPGA numerical proof,
physical coincidence and elapsed-time improvement are separate claims.
The C row has its own FPGA numerical evidence, not inherited A/B acceptance.
Both E2E cohorts and the combined audit are complete. C's one observation
per route supplies no request-repeat variability estimate.

The C physical audit correlated **7,724,388 event records**, **5,288 commands**
and **36 graphs**, including **128 B-only producers**. Epoch/reference
conservation, unchanged compute layouts, producer/Main binding and absence
of duplicate prefetched-Main B SRAM writes passed. Qualified future-B-write /
current-Residual-PE00 coincidence was zero in every graph. The raw capture
SHA-256 is `e0f0eaf85d3514bc7181c5f3cddececa64d4f3993e3b81da99f1ce49cebd26c3`.
Full results, observed interval definitions and limitations are in
`/tmp/h3-8h/early-stream-physical-results.md` and `analyze-early-stream.log`.
The separate CPU API diagnostic checked 128 calls across 12 C graphs.
Cold/steady median API duration was 0.009 ms for decode and FFN, and
0.0025/0.0035 ms for tail. This includes staging but excludes the later
explicit pair wait; it is not producer completion or attributed hardware
waiting. Per-phase MAD/ranges, parser proof and TypeScript-tooling limitations
are in `/tmp/h3-8h/early-admission-results.md`. Physical intervals and CPU
diagnostics are not additive contributions to graph time.

### C-image matched primary timing

All seven routes ran on the same C image with seven cold and seven steady
samples per workload. Input hashes, rotation/reversal, cache behavior,
numerical success, no fallback and all 42 printed median/MAD summaries were
independently checked. Physical trace and internal profiling were disabled.
Zero stage fields mean unmeasured, not free. The separate request-level E2E
measurement is not replaced by this benchmark.

Steady CPU-observed graph elapsed time, median / MAD in milliseconds:

| Route | Decode 1x768x768 | FFN 16x3072x768 | Tail 63x129x544 |
|---|---:|---:|---:|
| H2 baseline | 15.163 / 0.017 | 90.990 / 0.088 | 40.767 / 0.182 |
| K-tile | 11.529 / 0.005 | 70.381 / 0.081 | 40.127 / 0.203 |
| D2 prefetch | 12.367 / 0.010 | 73.556 / 0.072 | 40.409 / 0.095 |
| Early LdB | 12.383 / 0.006 | 73.591 / 0.046 | 40.457 / 0.084 |
| A bounded | 15.472 / 0.277 | 84.744 / 1.095 | 40.458 / 0.227 |
| B continuous | 14.629 / 0.541 | 83.263 / 1.999 | 40.647 / 0.134 |
| C continuous Early | 15.657 / 0.060 | 83.856 / 0.408 | 40.797 / 0.091 |

All routes retained per-graph issued GEMM / command-declared Activation /
command-declared Weight Tile totals of 72/72/12, 336/336/48 and 22/22/4,
respectively. D2, Early LdB and C each had 6, 24 and 2 separate B prefetch
Tiles; those are already included in the Weight totals. They are not actual
DMA byte counts or hardware time.

The complete cold/steady ranges, median/MAD values, relative changes and
per-route Tile counts are in `/tmp/h3-8h/early-primary-results.md`.
`analyze-early-primary.log` records the actual successful audit. The runner
verifies collected-log, trace-off, driver and bitstream manifests. Guest
preflight separately checked the staged executable hashes.
The negative overlap and performance results do not justify changing defaults.

### C-image end-to-end: one request per route

The user-requested seven runs completed with guest, driver, collector and
analysis exit 0. All seven generated outputs matched same-image H2; H2
stdout also matched the old baseline image. All 70 C request events passed
the typed timestamp, token-index, token-identity and monotonicity checks.
`analyze-e2e.log` contains `H3_8H_E2E_LATENCY_AUDIT_COMPLETE`.

Single observations in target CPU steady-clock milliseconds:

| Route | TTFT | Mean TPOT | Request duration |
|---|---:|---:|---:|
| H2 baseline | 10343.808 | 4229.507 | 39950.395 |
| K-tile | 9445.967 | 3444.197 | 33555.384 |
| D2 prefetch | 9596.173 | 3687.822 | 35410.967 |
| Early LdB | 9616.551 | 3668.918 | 35299.021 |
| A bounded | 12507.237 | 6554.954 | 58391.958 |
| B continuous | 12247.716 | 6339.699 | 56625.650 |
| C continuous Early | 12574.663 | 6602.292 | 58790.751 |

Request-repeat MAD is NULL/unmeasured, not zero. These are not confidence
estimates or repeated medians. Seven inter-token intervals within one
request do not create seven independent requests.
`e2e-final-results.md` records all ratios, exact boundaries and reproduction.
The old baseline remains a separate six-route/seven-repeat cohort.

### Completed A/B-image end-to-end baseline

The closed baseline has 42 real requests, seven per route, and 420 checked
request events. All eight generated token IDs matched the same-cohort H2
request. Collector/checksum, event-count, timestamp-order and bounds checks
passed. This analysis reran no inference. It is separate from the C-image
one-request-per-route cohort.

Target CPU steady-clock milliseconds, median / MAD across seven requests:

| Route | TTFT | Per-request mean TPOT |
|---|---:|---:|
| H2 baseline | 10378.861 / 2.198 | 4181.052 / 3.187 |
| K-tile | 9517.680 / 7.405 | 3412.412 / 3.023 |
| D2 prefetch | 9693.070 / 4.059 | 3668.710 / 11.324 |
| Early LdB | 9682.407 / 6.796 | 3669.546 / 1.344 |
| A bounded | 12378.188 / 59.664 | 6330.159 / 73.060 |
| B continuous | 12452.300 / 46.855 | 6400.598 / 27.856 |

TTFT starts with the tokenized prompt before inference. Mean TPOT divides
last-token-ready minus first-token-ready by seven intervals. Model loading,
prompt tokenization and process startup are outside these intervals. These
are neither FPGA wall-clock duration nor attributed hardware waiting.

Full ranges, request duration, relative differences, per-request values and
inter-token distributions are in `/tmp/h3-8h/baseline-e2e-results.md` and
`baseline-e2e-analysis.json`. The combined final audit subsequently passed
both complete cohorts. No cross-image timing samples are pooled.

Compiler database discovery is currently incomplete in the language server:
it reports missing `potal.h`/generated include paths. These diagnostics are
not a passing compile; the real configured build must verify the changes.

Native configuration reproduction:

```bash
bash /tmp/h3-8h/configure-native.sh 32
bash /tmp/h3-8h/configure-native.sh 16
```

## Adopted wire contract and geometry

Generated header:
`generators/potal/software/gemmini-rocc-tests/include/gemmini_potal_d32_params.h`,
SHA-256 `3580b84f15ed8de9f9d311edea64eaa7c1f9bb308b1552a8f79453ec2c05ea33`.
DIM32, signed A8/W8, signed INT32 accumulator, CUSTOM3, four SP banks of
2048 rows, 512 ACC rows, 256 Scale entries, 64 Index entries, MAX_BYTES=64.
Main/Residual keep ACC halves 0/1. Pair A/B banks remain 0/1 and 2/3.
All software address/capacity calculations use the selected parameter header.

The safe probe first resets legacy counter 0 with COUNTER funct126, rs1=10.
It then issues COUNTER xd=1, rs1=`0x4853384800000000`, rs2=0. Exact supported
response is `0x4853384801200803` (version 1, DIM32, eight epochs, flags 3).
An unsupported image receives no new stream opcode.

Control funct30 uses xd=1 and existing checked statuses:

| Operation | rs1 | rs2 |
| --- | --- | --- |
| BEGIN_BOUNDED | 1 | 0 |
| BEGIN_CONTINUOUS | 2 | 0 |
| BEGIN_CONTINUOUS_EARLY | 6 | 0 |
| END | 3 | 0 |
| WAIT_PAIR | `4 | (pair << 4)` | 0 |
| scoped Scale release | 5 | `slot | (generation << 8) | (column << 16)` |

Reserved bits must be zero. Legacy funct24 release and funct25 Fence remain
global. Ordinary LOOP_WS fields stay unchanged; the five configuration words
use xd=0 and the final LOOP_WS uses xd=1 with a checked admission response.
Admission is not execution/DMA completion.

`potal_stream_config` borrows the weight view and immutable Main/Residual
base/byte-extent/row-stride views; `continuous` selects A or B. The public
`potal_paired_stream_*` functions are `begin`, `panel_begin`, `loop_issue`,
`wait_pair`, `scale_release`, `panel_retire`, `end`, and `query`. Existing
Scale write, Index, readback and Fence APIs use stream-specific guards when
this owner is active. Absent Residual accepts `{NULL,0,0}`. The loop emitter
reuses the checked pure encoder, not a second arithmetic/tiling implementation.

## Actual command sequence and barriers

The executor prepares fixed-pitch windows, Scale allocations and replay
actions before BEGIN. Optional diagnostic manifests also print before BEGIN.
An idle legacy TLB flush precedes the checked capability/mode setup.
Stream begin publishes input memory once and emits BEGIN followed by
CONFIG_EX and CONFIG_LOAD state0/state1/state2. State0 carries Main lda,
state1 the real B ldb, and state2 the maximum aligned Residual row pitch.
Residual with null B never clears the B stride.

Loops replay `M(k), R(k,0..last), M(k+1), ...` in the frozen plan order.
Main A uses LOAD, B LOAD2, and stream Residual A LOAD3. Required Scale/Index
metadata precedes each contiguous six-word loop submission. No ordinary
CONFIG, CPU publication fence, issuer clock, output, or source allocation
occurs inside that six-word submission.

Remaining completion boundaries are deliberate:

- A retains global Fence/release at bounded batches, at most two panels.
  Capacity pressure may split a batch without changing tile dimensions.
- B waits for the older owner of a pair before reuse and uses scoped releases.
  These waits do not declare the opposite pair drained.
- Multiple Residual windows still wait before replacing the same Index table
  and A subrange. There is no new shadow Index storage.
- Readback remains conservative: global checked Fence before CONFIG_ST/MVOUT,
  then a final checked Fence and CPU consumption ordering before host reads.
  A global Scale release cannot substitute for that final Fence.
- END requires global hardware drain excluding its own raw FIFO head.

The emitter's absence of clocks is tested by wrapping the actual libstdc++
`steady_clock::now` entry on Linux/GNU. A positive control proves the wrapper
observes a real call. A libc-only hook was insufficient: this host libstdc++
calls `syscall` directly. The initial failed observation run is retained in
`/tmp/h3-8h/native-counters.log`; corrected tests pass in
`native-counters-v2.log`.

## Ownership and physical completion

Each software panel has owner/generation/identity, A/B ranges, declared and
submitted consumers, and sealed/completed state. Scheduled B permits ordered
consumers; it never changes legacy H2 readable into a scheduling-only promise.
Main and its Residuals share the identical rear-packed B placement.

Raw dispatch allocates `{id,serial,pair,generation}` before the generator
queues. Eight entries exist; A uses two and B all eight. Exhaustion stalls
admission, not retirement. A 32-bit serial cannot wrap while any earlier
identity survives; a two-bit test instance exercises this boundary.
Command counts cover RS capacity plus four generated-queue positions.
Work counts are bounded by 128; overflow/underflow assertions are tested.

Generator close does not retire an epoch by itself. Each generated command
adds a reference, removed exactly once by either complete-on-issue or response
retirement. Tags traverse RS, load tracking, DMA/XactTracker, returned-byte
accounting and actual Scratchpad writes. Tagged work references survive raw
completion through scaled ACC commit and workDone.

`startsWork=(k==0)` is independent of ACC `firstContribution`. The preload
reservation gate refuses a recycled work ID until old work ends, while old
raw/commit/completion paths continue. Follow-on contributions require matching
active, unsealed ownership.

Scoped release checks both Controller epoch references and Writeback context/
work ownership. It conservatively protects an entire matching generation and
also rejects active untagged legacy work. Unrelated tagged generation writes
can continue. Distinct live Scale allocations cannot reuse a generation;
proven Main/Residual aliases remain one allocation. Scale capacity uses the
unique live set, not a fixed half split.

Index update protection remains buffer-local. Indexed PRELOAD hazards use
the selected descriptor's B base and
`ceil(sourceKRows / DIM) * jTiles * DIM` range in ReservationStation.
Execute entries depend on all older unissued entries of the same queue, so
load-ahead does not reorder EX contributions. Opposite-pair pending B is not
a global software Index admission gate.

Partial six-word issue or lost checked response quarantines ownership and
borrowed sources. Cleanup must not retry, publish results, release uncertain
resources, or silently fall back.

## Verification evidence and limitations

Native emulation is software-only. The new tests compare unchanged independent
goldens and old K-tile output for Main INT32, Residual INT32, radix INT64 and
final F32. They cover saturation/cancellation, tails, sparse/discontinuous
active-K, absent Residual, multiple windows/panels, 260-panel generation wrap,
full Main/Residual ACC halves and full B-bank capacity. Logical B loads are
checked for duplication; these counts are not measured bus traffic.

- DIM32 broad regression: 15 targets, 11 passed, three failed and one Not Run.
  `potal-gguf-load-test` was Not Run because its executable was missing.
  Logs: `/tmp/h3-8h/native-d32-regression.log`.
- DIM16 broad regression: 14 targets, one `potal-log-test` failure.
  Logs: `/tmp/h3-8h/native-d16-regression.log`.
- DIM32 failures are `potal-log-test`, `potal-paired-schedule-test`
  and `potal-paired-unit-test`. The retained schedule/unit baseline logs
  reproduce their failures. `/tmp/h3-8h/baseline-potal-log-test.log` instead
  ends `POTAL_LOG_TEST PASS`, so that receipt does not establish the log
  failure as pre-existing. Its broad-suite failure remains unresolved.
  Those tests/goldens were not edited green.
- Later focused DIM32/DIM16 API, attention API, tiling and execution gates
  each passed 4/4: `native-d32-final.log`, `native-d16-final.log`.
- Diagnostic counter/clock regressions passed DIM32 2/2 and DIM16 4/4:
  `native-counters-v2.log`, `native-d16-counters.log`.
- Timing-off pipeline plus paired execution passed DIM32 2/2:
  `native-quiet.log`, and DIM16 2/2: `native-d16-quiet.log`.
  Weight/pipeline/stripe timing fields are explicitly
  unmeasured when disabled; numerical results and event order remain intact.

RTL verification is component/composition simulation, not FPGA:

- Stream table, real RS/LoadController delayed-byte return, safe capability,
  serial wrap, static LOAD/LOAD3 scheduler admission and held rejection:
  seven cases, plus four physical-prefetch observation regressions, 11/11
  in `/tmp/h3-8h/rtl-mac-stream.log`.
- Four counter overflow/underflow tests passed in
  `/tmp/h3-8h/rtl-counter-bounds-v2.log`.
- Writeback reuse/scoped-release, reservation gate and legacy LDB-only gates
  are documented with exact commands in `/tmp/h3-8h/rtl-contract.md`.
  Actual ACC RMW, DIM32 ingress and indexed scheduler cases also passed.
- Three unchanged reduced-cache `PoTalDrainHandshakeSpec` fixtures reject
  their hardcoded Scale key before work starts: `rd=11 status=6`. They use
  eight entries but encode column at bit16 instead of bit11. The production
  256-entry DIM32 cases pass. These failures are not suppressed.
- These component/composition regressions are not a whole-SoC simulation
  suite. Subsequent FPGA numerical/liveness gates passed as recorded in the
  current ledger. Their separate physical captures showed zero qualified
  coincidence; module proofs alone do not establish workload overlap.

Additional real RS/LoadController tests passed. They withhold all
future-pair returned bytes and check that a disjoint indexed PRELOAD can
issue. An intersecting B descriptor is the negative control and must wait.
No EX completion is fabricated: the current epoch must retain its reference.
This observes RS issue eligibility, not actual MAC execution. The first build
failed because the helper was absent from the explicit SBT test-source list;
that registration is corrected. Logs are `rtl-indexed-dma.log` and
`rtl-indexed-dma-v2.log`. The corrected run passed both dependency cases
(2/2) and the existing tagged returned-byte retirement case (1/1), exit 0.
The production RTL used by the FPGA checkpoint was not changed by these tests.

The C test trace grew from 512 to 2048 records because the new final counter
snapshot exceeded the original buffer. Bounds are asserted before inspecting
snapshot records. Initial failure logs are retained.

## Instrumentation and measurement protocol

The validation history below preserves earlier checkpoint-specific failures
and pending states. Later completed results in the current verification
ledger supersede those pending observations. Failed logs and intermediate
artifacts remain available; historical build progress is not a current gate.

### Print-synthesis integration correction

The first diagnostic guest completed all three cases and all six routes,
but produced no physical trace file. Its generated driver header contained
no PrintBridge. Source inspection found two missing requirements:
`PoTalStream.trace` used ordinary Chisel `printf` without `SynthesizePrintf`,
and its compiler configuration was `BaseXilinxAlveoConfig` without
`WithPrintfSynthesis`. A runtime plusarg cannot add a missing bridge.
The previous numerical FPGA proof remains valid; the diagnostic run is
not physical-overlap evidence and its instrumented timings are not primary
performance results. Its counters/manifests are preserved in
`trace-collected-v3/trace` and `trace-software-v3.sha256`.

The helper now wraps the existing printf in
`midas.targetutils.SynthesizePrintf`, using the existing transitive
targetutils dependency. An isolated compiler configuration,
`WithPrintfSynthesis_BaseXilinxAlveoConfig`, leaves other images/defaults
unchanged. Golden Gate and the driver build passed with a generated
`synthesized_prints_t` instance and all 18 required event names.
The generated metadata has 32 printf sites, all `[POTAL_STREAM]`; no CPU
commit-log printf was added. Their existing runtime gate remains disabled
for primary timing.
`printf-generated-header.sha256` records header SHA-256
`b946cfe491ec21194c02f6df5c612625b302bf9ebe9138acc5ceab8cdc589496`.
That generated-header check is now mandatory before starting Vivado.
Scala LSP is unavailable; the configured Scala/Golden Gate build is the
compile evidence.

`generate-rtl-printf.sh` places the new generated/output namespace in tmpfs.
`build-printf-image.sh` uses the platform's normal project synthesis and
implementation constraint setup through opt_design, then the previously
successful single-thread place/phys-opt/route flow and setup/hold/bitgen
gates. It avoids manager bulk copyback into the nearly full root filesystem.
The first staging attempt moved the CL directory and broke relative links
to sibling-board Tcl files, failing before Vivado started. That move was
reversed; staging now matches the manager's dereferencing `rsync -L`.
Shared platform templates remain unchanged. The failed staging log is
`build-printf-image.log`; the active build log is `build-printf-image-v2.log`.
All FPGA gates, physical capture and matched performance measurements will
use the resulting new image; no old-image timing is substituted.

Synthesis and project-mode `opt_design` completed successfully. The first
implementation wrapper then rejected a valid checkpoint because it expected
the display status `opt_design Complete*`. Vivado 2021.1 actually reported
`Not started place_design`, progress 40%, after the requested `-to_step`.
The generated run script proves completion using `.opt_design.end.rst`,
absence of `.opt_design.error.rst`, and the nonempty opt checkpoint. The
wrapper now uses those checks. `resume-printf-implementation.sh` verifies
the original generated inputs, records checkpoint/script hashes and resumes
that printf-enabled checkpoint; it does not rebuild synthesis or borrow the
older non-print image. The failed log remains `build-printf-image-v2.log`.

The opt log also contains three warnings for the generated `host_clock`
alias constraints. `probe-printf-opt.tcl` loaded the real checkpoint and
found exactly one target clock buffer, timed by
`clk_out1_design_1_clk_wiz_0_0` at 33.330 ns. The generated alias requested
divide-by-1, setup 1 and hold 0. The recovery checks the actual clock binding
and period before placement and retains the nonnegative setup/hold gate;
it adds no timing relaxation. Evidence: `printf-opt-probe.log` and
`printf-opt-clocks.rpt`.

Placement and routing subsequently completed. The final routed timing gate
reported WNS +0.055 ns and WHS +0.010 ns, with zero failed setup/hold endpoints.
The timing report lists zero unclocked register/latch pins and zero
unconstrained internal endpoints. Routed-net verification passed with zero
failed, unrouted, partially routed or overlapping nets. Bit generation
completed and `resume-printf-implementation.log` ended with
`H3_8H_PRINTF_BUILD_EXIT=0`.

`package-printf.sh` passed its setup/hold, generated-header, checkpoint/script
hash, archive comparison and preserved-C/D/E checks. The package hash is
`9d35b08774292b8893b6a5bc6e6503f8783f09271a891a27dee5934500d4aec9`;
the bitstream hash is
`e7b144629b3a665df5f8be8dcc1e951b10821b1bd66449315d3e53521d6633e0`.
The matching driver hash is
`009f37213489aee7dfb5d29a1f7c225ade851902c3da4146ab7ec2eec4bf6dff`.
`program-printf.sh` programmed U280 slot 0, rechecked the driver identity,
and returned 0. Evidence: `package-printf.log`, `printf-bitstream.sha256`,
`printf-driver.sha256`, and `program-printf.log`.
The new image passed all 30 raw golden cases, API smoke, Attention smoke and
PV-range checks. Guest exit, simulator exit and read-only fsck were all
successful; stderr logs were empty. `printf-gates-collected`,
`printf-gates-software.sha256` and `collect-printf-gates.log` preserve the
matched-image proof. Its raw-golden output hash matches the v3-r2 output.

The actual driver created `printf-gates-run/synthesized-prints.out0`.
After shutdown it contained only the clock-domain header, with relative
frequency 1/1 and no `[POTAL_STREAM]` events, proving trace-off for this run.
`printf-gates-disabled-trace.sha256` records the file. A separate boot
requested tracing, but also produced only that header. The three software
diagnostic cases passed; the physical capture did not. Those software
results are not overlap or primary-performance evidence.

### Runtime trace-enable correction

The generated image had a PrintBridge but no `plusargs_t` constructor or
`potal_stream_trace` argument in its driver header. Rocket Chip's ordinary
`plusarg_reader.v` assigns `out = DEFAULT` under `SYNTHESIS`; the PoTal
enable therefore became constant 0. The driver's actual PlusArgs parser
does accept `+potal_stream_trace=1`; argument spelling was not the cause.
`printf-trace-collected`, `printf-trace-software.sha256`,
`printf-physical-trace.tar.gz` and `collect-printf-trace.log` preserve this
failed attempt. The collector deliberately returned nonzero because no
physical event existed.

The correction uses a PoTal-local RawModule annotated with the existing
MIDAS PlusArgs bridge. Its native body retains the original 32-bit
Rocket Chip PlusArg and default 0; FireSim extracts that body and uses the
startup-initialized host bridge instead. Startup configuration is sufficient:
the workflow boots separately for trace-on and trace-off runs and never
requires live mutation. PoTal now directly depends on `midas`; the actual
SBT compile completed without a dependency cycle. No ISA, arithmetic,
ownership or production default was changed.

The first native run caught annotation-clock removal by native DCE.
Retaining the annotated IO with `dontTouch` fixed that specific problem;
all 13 existing PoTalStreamSpec tests then passed unchanged on Treadle/
Verilator (`runtime-bridge-native-r2.log`). The failed first log remains.
The small `PoTalStreamTraceModule` integration target uses the real trace
helper and checks both host bridge instances. Its metasim build/run passed:
default and 0 produced no records; 1 and 2 each produced exactly the 64
requested records. The actual records also decoded with serial/local cycle
0..63, bridge cycle 5..68, epoch/pair 0 and generation 1.
`validate-runtime-trace-metasim.sh --resume` is the reproduction command for
the prepared namespace. This is software-only FPGA-model validation, not a
board PASS. `runtime-trace-metasim-r2.log` and
`runtime-trace-metasim.sha256` preserve the result. The first build's wrong
header filename check was fixed without rebuilding or replacing its generated
design; its failed log is retained.

The related legacy RTL batch completed 32 tests: 29 passed and the same three
pre-existing reduced-cache drain fixtures failed at `rd=11 status=6`.
`PoTalDrainHandshakeSpec.scala` still shifts the column by 16 for its
eight-entry cache, rather than 11. These are the failures already recorded
in the earlier RTL evidence, not new trace-bridge failures. The batch retains
exit code 1 in `runtime-bridge-legacy-rtl.log`; no test was weakened or
skipped to manufacture a green batch.

The full runtime-enabled rebuild uses the separate
`PoTalRuntimeTraceXilinxAlveoConfig` image namespace with unchanged target
geometry/arithmetic. `generate-rtl-runtime-trace.sh` requires the passing
native stream and metasim gates and checks the full generated header for
32 runtime trace-enable arguments, a PlusArgs bridge, a PrintBridge and all
18 event kinds before Vivado is allowed to start. The prior constant-disabled
image and all its artifacts remain intact.

Full-image generation completed with all 32 runtime bridge arguments. Each
constructor has default 0 and width 32. The generated header SHA-256 is
`de7f1453bb6fe8cea27b7a7836b90a9ad4a154d09642d6e4f3e32b651010d3b3`.
The generated executable driver SHA-256 is
`fd95293a7e2059fca88d09692887997024cd0c60304e9e3ff7570bf77adbd054`.
`generate-rtl-runtime-trace-r2.log` records exit 0. Its initial preflight
mistakenly searched for a monitor marker printed outside `tee`; it now checks
the actual SBT 13/13 result. The first log remains intact.

`build-runtime-trace-image.sh` is running in
`/dev/shm/h3-8h-runtime-trace-build`. It uses the previously checked single-thread
Vivado 2021.1 flow and requires the actual 33.330 ns target clock plus nonnegative
setup/hold slack. Routed timing, native bit generation and the matched board
gates have now passed.

The new runtime/HWDB YAMLs retain every execution setting from the prior
printf image except the image key and isolated run directory. PyYAML parsed
both and their normalized structures compared equal. The YAML language server
is unavailable; actual parsing was used instead. The package/program/launch/
send/collect scripts passed `bash -n`. Guest software and its input hashes
remain unchanged, with a new `results-runtime-trace` result namespace.
Every launch records the current rootfs, boot binary and driver hashes before
boot, rather than reusing an earlier mutable rootfs hash.

After the build and final timing/report review, the prepared reproduction
commands are:

```bash
set -o pipefail
bash /tmp/h3-8h/package-runtime-trace.sh 2>&1 | tee /tmp/h3-8h/package-runtime-trace.log
bash /tmp/h3-8h/program-runtime-trace.sh 2>&1 | tee /tmp/h3-8h/program-runtime-trace.log
bash /tmp/h3-8h/launch-runtime-trace.sh gates
```

The launch script records its driver output and exit status in the phase run
directory. In a second terminal, after the guest login prompt:

```bash
bash /tmp/h3-8h/send-runtime-trace.sh gates /tmp/h3-8h/uart-runtime-trace-login.txt
bash /tmp/h3-8h/send-runtime-trace.sh gates /tmp/h3-8h/uart-runtime-trace-gates.txt
```

After the guest phase exit marker, send `uart-runtime-trace-shutdown.txt`,
wait for simulator exit, then run `collect-runtime-trace.sh gates`. Repeat
with `trace`, `performance` and `e2e` using their matching UART payloads.
Only the trace launch sets `+potal_stream_trace=1`; other launches set 0,
and collection rejects any unexpected hardware event in those trace-off runs.
Packaging, programming and matched gates have now completed. The trace guest
is booting; primary performance and E2E phases have not run yet.

### Disk-full recovery during implementation

The root filesystem filled while Vivado was running. The outer `tee` reported
`No space left on device`; `build-runtime-trace-image.log` is therefore
incomplete and must not be used alone as the final build receipt. Vivado's
complete native log remained in the RAM build directory. A separate monitor
now copies that log from its beginning and follows new output into
`build-runtime-trace-vivado-recovered.log`. The original failed log is retained.
The outer pipeline returned 1 because of the earlier `tee` failure, while the
native script emitted `H3_8H_RUNTIME_TRACE_BUILD_EXIT=0` after bit generation
and its protected-input checks. Both statuses were observed in `bash_309`.
After completion, the recovered log compared byte-for-byte equal to Vivado's
original, and the now-obsolete follow monitor was stopped.

`runtime-trace-native-exit.txt` records that observed native completion and
outer logging failure. The package gate now reads the complete native
`vivado.log` and this explicit exit receipt, rather than the truncated outer
log. The build script writes the receipt directly for subsequent reproductions.
No CAD stages were rerun merely to repair logging.

Tool hooks also initially failed with ENOSPC. Moving the newly written trace
correlation document temporarily to RAM released enough space for the existing
kernel to resume. The document was then restored. With no active guest, the
working rootfs was moved to `/dev/shm/h3-8h-runtime-rootfs.img`, retaining
`/tmp/h3-8h/rootfs.img` as a symlink. SHA-256 before and after matched:
`3bf69cd775e3f0193c9fd4bd24e477bf30a54cfc1082e589ef215871721156a6`.

The user subsequently requested deletion of unnecessary past builds.
`archive-obsolete-build-evidence.sh` saved the old printf build's reports,
logs, messages and completion markers, then compared the archive with its
originals. `remove-obsolete-builds.sh` removed only:

- `/dev/shm/h3-8h-print-build/vivado_proj`: 4,937,568 KiB of obsolete Vivado
  working files.
- 1,040 object files, totaling 73,784,736 bytes, beneath the completed
  `/tmp/h3-8h/{d16-host,d32-host,cli-cpu}` builds.

Source files, failed-run logs, native executables, the prior bitstream package,
driver package and current runtime-enabled build remain. Archive/bitstream and
protected CDE checks passed after deletion. The detailed deletion record is
`remove-obsolete-builds.log`; the removed object paths are in
`removed-obsolete-objects.txt`. The observed remaining space after cleanup was
84 GiB on the root filesystem and 19 GiB in `/dev/shm`.

The new runtime build continued through opt_design and verified its actual
target clock as `clk_out1_design_1_clk_wiz_0_0`, period 33.330 ns.

### Runtime-enabled routed result

The full runtime-enabled design routed all 988,172 routable nets with zero
routing errors. The final timing report and Tcl gate agree: WNS +0.070 ns,
WHS +0.010 ns, zero failing setup/hold/pulse-width endpoints and zero total
negative slack. It reports zero unclocked register/latch pins and zero
unconstrained internal endpoints. Two input ports and one output port still
lack delays.

The DRC report lists 586 warnings and no errors. Methodology reports 41
findings, including four critical warnings: TIMING-4, TIMING-27 and two
TIMING-54 instances. Comparing all methodology detail lines with the retained
v3 report found the same findings, ignoring trailing whitespace. These
platform findings remain; the result is not warning-free signoff.

The reports are under `/dev/shm/h3-8h-runtime-trace-build/vivado_proj/`.
The complete recovered Vivado log contains the actual
`H3_8H_PRINTF_ROUTED_TIMING WNS=0.070 WHS=0.010` result. Bitstream/MCS generation
then completed with `H3_8H_PRINTF_BITSTREAM_PASS` and native script exit 0.
Packaging and FPGA programming both subsequently returned 0.

The new package identities are:

- Package: `0c01327a3648a54cb4dcaf419d78fbf31d494d2acb4871f8c86d749f32e67505`
- Bitstream: `2a21b2a87baa9839af465ebce53cc2dc03348dafd05d88a4c7a31be3909b511d`
- Driver: `fd95293a7e2059fca88d09692887997024cd0c60304e9e3ff7570bf77adbd054`

`runtime-trace-bitstream.sha256`, `runtime-trace-driver.sha256`,
`package-runtime-trace.log` and `program-runtime-trace.log` preserve the
identity checks. The manager programmed slot 0, PCI `01:00.0`, from this
package in the isolated runtime-trace run directory. Source/reproduction
snapshot v11 is retained; FireSim remains on `potal`.

The new image passed all 30 raw cases, API, Attention and PV-range gates.
Guest and driver exits were both 0, the read-only filesystem check passed,
and all four gate stderr files were empty. The raw golden output is byte-for-byte
identical to the prior printf image, SHA-256
`f8bc40e8b4a9ca80e7819991f30103b2bd6c1aa4970470f2ca70477023d85e4f`.
`runtime-trace-gates-collected`, `runtime-trace-gates-software.sha256` and
`collect-runtime-trace-gates.log` preserve the board result.

The trace-off file contains only the expected clock-domain header at relative
frequency 1/1. A new boot of this same bitstream now enables
`+potal_stream_trace=1`. Physical identity/conservation/overlap results are still
pending; the successful numerical gate is not overlap evidence.

This is not a warning-free platform signoff. Routed DRC contains 586 warnings
and no errors. The methodology report contains the same 41 findings and the
same four critical warnings as the preserved v3 report:
TIMING-4 (clock redefinition), TIMING-27 (hierarchical primary clock), and
two TIMING-54 scoped clock-group constraints in the platform clock/PCIe IP.
The comparison used `timing-v3/design_1_wrapper_methodology_drc_routed.rpt`.
The final timing report also records two inputs and one output without I/O
delays. These existing platform constraints were not suppressed or relaxed;
the reports are retained with the image evidence.

The trace protocol carries epoch ID, serial, pair and Scale generation.
The runtime-enabled rebuild must prove `+potal_stream_trace=1` reaches those
event gates. Events include raw/scheduler admission,
generated commands, RS allocate/issue/terminal, load and DMA request,
returned bytes, actual A/B SRAM write, qualified PE(0,0) MAC, mesh output,
scaled ACC commit, work start/done and local retire.
`mesh_output` is not used as a MAC event.

`POTAL_STREAM_MANIFEST=1` prints run/stripe/J-group/panel/window/path,
pair/generation, tile dimensions and addresses before issue. The single issuer's
manifest order must be matched against ordered raw admissions, with pair and
generation agreement; missing, rejected or extra records invalidate matching.
Current consumer loads and future-panel loads must be classified from this
identity, not from mere temporal coincidence.

The manifest's `ordinal` is the GEMM Tile execution sequence within a
run/stripe. It is distinct from `(group,panel)` B residency and from internal
DIM operations. Match every ordered `raw_admit` to exactly one manifest
record, verify pair/generation, then propagate its complete
`(epoch,serial,pair,generation)` key through all later events. Missing,
duplicate, rejected or orphan records invalidate a complete-run claim.
Separate benchmark processes need their case/file namespace even when
their software run IDs repeat.

Classify future Activation writes by a later GEMM ordinal in the same
run/stripe. Classify next-residency Weight writes by the next panel of the
same J group, rather than by opposite pair alone. Report same-pair and
opposite-pair cases separately. Overlap requires equal target cycles for a
qualified current `pe00_mac` and the classified physical SRAM write.
Use the print driver's `CYCLE:` timestamp within its declared clock domain
for that comparison; retain each event's local `cycle` field as a consistency
check. Do not silently join different clock domains or treat host arrival
time as a target cycle. The driver emits a one-line clock-domain/frequency
header and keeps the cycle prefix enabled in these launch commands.
Admission intervals and first/last stage spans cannot establish that overlap.

`trace-parser.sql` provides strict DuckDB decoders for the padded driver
format and software manifests. Unsigned 64-bit cycles/window sentinels and
32-bit serials retain their widths. Missing fields, unknown events/modes,
out-of-range epoch/pair/generation and zero or out-of-width dimensions are rejected.
Materialize every decoded column before aggregate queries so projection
pruning cannot bypass field checks.

`bun test /tmp/h3-8h/trace-parser.test.js` passed 39 tests in one run;
`trace-parser-test.log` preserves the output. This is software parser proof
only, using the installed DuckDB v2.0.0 binding in the user cache. The
optional JavaScript language server is unavailable; Bun executed the actual
SQL and test source successfully.

The parser also decoded all 3,440 manifests from the previous non-print
diagnostic without duplicate identities or ordinal gaps. Each case contains
four graphs per stream mode: untimed warmup, cold sample, untimed steady-cache
priming, and steady sample. Their graph IDs are 5/11/17/23 for bounded and
6/12/18/24 for continuous. Do not mistake warmup/priming traces for measured
samples. These records are software command identities, not physical events.
PE(0,0) observations do not establish whole-array utilization; SRAM write
events do not provide bus-byte counts.

Hardware print records use the trace-only override
`/dev/shm/h3-8h-trace-prints/synthesized-prints.out<bridge-number>`,
not `driver.log`. Root storage had 1.7 GiB free before packaging, so the
diagnostic stream uses tmpfs to avoid repeating the disk-full failure.
The primary performance launches retain the original output settings.
The FireSim print driver writes each clock domain's header; keep that header
when collecting evidence. Its `finish()`
calls `flush()`; require normal simulator completion before claiming a
complete trace. The matching software manifests are in the guest's
`/root/h3-8h/results/trace/<case>/bench.err`. Extract those files after guest
shutdown, retaining the three cases' command order and file namespace.
After the driver stops, preserve the trace with
`tar -czf /tmp/h3-8h/trace-run/target-trace.tar.gz -C /dev/shm/h3-8h-trace-prints .`;
retain the tmpfs originals until the archive is verified, including on failure.
`--observe-stream` explicitly enables manifests/counters and labels its
timing non-performance; primary benchmark invocations reject these diagnostics.

Diagnostic `potal_paired_stream_counters_begin/end` select existing events
10,19,20,41,27,29,31 using the seven counters guaranteed by capability.
Setup/reset is before panel declaration; reads require final global Fence.
These are overlapping predicate/event counts, not additive graph latency,
DMA bytes, or saved cycles. In particular the DMA TLB response-miss predicate
must not automatically be described as request-qualified TLB stall time.
Software callback values are explicitly labelled software-only.

Primary measurement builds use LOG_DEBUG/LOG_CYCLE/CYCLE_SIM/CYCLE_DETAIL=0,
no manifest/raw trace/counter profiling, and `POTAL_PAIRED_TIMING=0`.
The scalar Rocket configuration also sets GGML_RVV/GGML_RV_ZFH=OFF.
The smoke benchmark rejects logging-enabled primary builds. The timing switch
also disables weight, pipeline and stripe-event timing; the paired executor
does not time individual issue/Scale/Fence calls. Default reporting is retained
when the switch is absent.

The primary microbenchmark quantity is `compute_ms`, measured immediately
around synchronous `ggml_backend_graph_compute`. Preparation, issuance,
drain/readback and host reconstruction inside the graph are included.
`pipeline_wall_ms`, weight/quant timing and related internal timing fields
are zero/unmeasured in this build; they must not be interpreted as zero costs
or used instead of the external graph interval.

`POTAL_REQUEST_LATENCY=1` in noninteractive `llama-cli` reuses the existing
tokenized-prompt and sampling/accept timestamps. It buffers token boundaries
and emits `POTAL_REQUEST_EVENT` records only after inference, so LOG_CYCLE can
stay zero. TTFT is first token minus request start; TPOT is last minus first
token divided by the number of token gaps. These are guest CPU monotonic
latencies, not accelerator cycles.

The actual CPU-only FP16 CLI passed the logging toggle check with identical
generated output, eight token events, and strictly increasing start/token/end
timestamps. Logs: `/tmp/h3-8h/host-cli-cpu-{0,1}.{out,err}` and
`validate-latency.log`. This validates the observation surface only, not PoTal
hardware or Q8_HP1 performance. The earlier host attempt with PoTal registered
failed at attention with `ERR_NO_HARDWARE`; `host-cli.err` is retained.

`/tmp/h3-8h/run-guest.sh` defines separate gates, trace, performance and E2E
phases. The three microbenchmark shapes are `decode_1x768x768`,
`gpt2_ffn_up_16x3072x768`, and `tail_63x129x544`. Six routes share one binary,
bitstream, inputs and frozen plan: H2, K-tile, D2 prefetch, Early LDB-only,
bounded stream and continuous stream. Seven measured repetitions rotate route
order. Raw golden destinations are poisoned before each route to detect a
missing readback. Whole graph timing includes staging, metadata, readback and
host combination.

E2E uses the same GPT-2 Q8_HP1 model/prompt, eight greedy tokens with EOS
suppressed, seven rotated repetitions per route, and fixed CPU masks. Numerical
token identity, event count, median/MAD and negative results still need actual
guest runs. No microbenchmark substitutes for E2E.

## Build and artifact paths

- Native configuration: `/tmp/h3-8h/configure-native.sh`.
- Cross configuration: `/tmp/h3-8h/configure-riscv.sh`.
- Primary cross build: `/tmp/h3-8h/d32-performance`; latest build log
  `build-riscv-scalar.log`.
- Generated build manifest:
  `/tmp/h3-8h/d32-performance/generated/potal_build_manifest.txt`.
- RTL generation/build: `/tmp/h3-8h/generate-rtl.sh`, `build-fpga.sh`.
- Isolated FPGA build root: `/home/alveo/build_firesim_h3_8h_d32`.
- FPGA config: `FireSimQuadRocketSinglePoTalD32PairedStreamingConfig`.
- Build manager raw log: `/tmp/h3-8h/build-fpga-manager.log`.
- Rootfs staging: `/tmp/h3-8h/stage-rootfs.sh` and `stage-rootfs.debugfs`;
  it copies the existing image rather than modifying it, verifies every binary
  by round-trip extraction, checks model identity and runs read-only fsck.
  Initial byte-for-byte staging passed, but the later ISA audit rejected that
  artifact as described below. Its original identities and image are retained.
  Deployment/launch helpers are `program.sh`,
  `launch.sh`, and `send.sh`. Infrasetup and the golden guest passed;
  the physical-trace guest is now booting.
  `uart-{gates,trace,performance,e2e}.txt` select the guest phase;
  `uart-poweroff.txt` is sent only after recording that phase's exit status.

`package-recovery.sh` creates `timing-accepted` itself after checking the final
bitstream-success record and exactly one finite, nonnegative routed WNS/WHS
record. This removes a missing manual bootstrap step without accepting
intermediate timing. Missing, malformed, duplicated or negative final timing
fails before the certificate is written. The updated script passed `bash -n`;
the actual embedded awk parser also passed nine synthetic input cases:
positive and zero slack accepted; negative setup, negative hold, missing
record, N/A, infinity, duplicate record and extra field rejected. These are
parser tests, not FPGA results. Real packaging subsequently passed, exit 0,
in `package-recovery-v3.log`, after Vivado successfully generated both
`.bit` and `.mcs` and exited 0. Archive/bit/MCS/metadata identities are in
`bitstream.sha256`; packaging-time driver and report identities are in
`driver.sha256` and `timing-v3.sha256`. The bitstream archive SHA-256 is
`1bfecfcecd5b7b62974cfc5c430b01a33a703895c630b16e0e84c3d058ea43cd`.
`runtime-v3.sha256` binds the launch, programming, guest scripts and runtime
configuration used for this deployment.

`program-v3.log` records flashing slot 0, BDF `01:00.0`, with the packaged
`firesim.bit`, followed by `H3_8H_PROGRAM_MANAGER_EXIT=0`. The manager rebuilt
software during infrasetup; the post-deployment driver still matches the
packaging-time SHA-256, and the protected CDE input checksum is unchanged.
The expected bootbinary exists. Golden simulator `bash_215` exited 0 after
normal guest shutdown. Trace simulator `bash_227` is a separate launch of
the same bitstream and driver, with only diagnostic tracing enabled.
Programming success and each later numerical/trace result remain separate.

The first UART send failed before writing input because the root-owned
driver's `/proc/<pid>/fd` directory is inaccessible to the invoking user.
The file-mode UART bridge opens its input once and calls `read()` again
after EOF; it does not unlink or replace that inode. `send.sh` now appends
to the existing owned input file instead of inspecting root descriptors.
The unchanged single-driver check remains. `bash -n` passed, the retried
send succeeded, and actual UART output showed `buildroot login: root`
followed by the root shell prompt. No RTL or guest binary changed.
`runtime-v4.sha256` preserves the corrected sender and all UART payload
identities; the earlier v3 identity record remains intact.

### Actual FPGA numerical and liveness gate

The guest verified all ten staged binaries and the GPT-2 model checksum,
then reported `H3_8H_HARDWARE_GATES_PASS` and
`H3_8H_EXIT phase=gates status=0`. Normal shutdown produced simulator exit 0.
Logs are `/tmp/h3-8h/gates-collected/gates/`; their identities, UART log,
driver log and exit record are in `hardware-gates.sha256`.

- API smoke passed DIM32 Main and indexed Residual goldens, encoding/status
  rejection, in-flight issue/flush, Scale lifecycle and final cleanup.
- Raw Main/Residual INT32 passed 18/18: H2, K-tile, D2, Early LdB,
  bounded and continuous across three fixtures. Each stream mode passed
  three fixtures. Checked Main/Residual cell counts were 1105/130 for the
  contiguous and gapped fixtures and 8127/8127 for the tail fixture.
  Every route balanced Scale writes/releases. Early used only LdB-only
  prefetch commands in its three cases.
- Attention passed original-K raw signed32 low/high QK comparisons,
  overwrite/accumulate/store epoch and Fence checks, K33/K65/K4097 tails,
  strided/head/batch cases, PV bias handling and saturation witnesses.
  The destructive QK witness matched its independent integer-policy golden,
  not the different CPU FP32 result; no universal FP32-equivalence claim
  is made.
- PV range passed at M1/N64/K288 with 12 heads: 49 jobs, 16 bitplane jobs,
  11 corrections and balanced 28224 Scale writes/releases, `exact=1 status=0`.

All four stderr files were empty. Read-only fsck passed after shutdown.
Unprivileged `debugfs rdump` emitted guest-owner restoration warnings; all
ten output files were present, read and hashed. No guest image repair or
ownership relaxation was performed. These are actual FPGA numerical and
completion results, not yet physical-overlap or primary-performance results.

### Additional coverage prepared for the printf-enabled image

The three completed raw fixtures did not exercise unequal Main/Residual
I-Tile counts. Two additional `--stream-golden` fixtures are now compiled
and staged, while the original three are unchanged. One has M17 and four
lane groups (68 Residual rows), so Main I=1 and Residual I=3. Its digits vary
by row and its non-saturating carriers expose wrong row/tile addressing.
The other supplies no Residual packet, source or output.

The test checks every Main operation's frozen I, each Residual operation's
full I, one Main per K/J residency and one Residual per Index window; it
cannot silently accept I-subdivision. All six routes run all five fixtures,
for 30 required raw passes. Initial RISC-V compilation and scalar-ISA/
byte-for-byte rootfs staging passed. Logs: `build-smoke-extended.log`,
`update-rootfs-extended.log`, and `rootfs-extended-inputs.sha256`.

While the printf image was building, the existing v3 FPGA image ran the
extension. API smoke passed; 26 raw cases passed, including all six unequal-I
routes. The next case, zero-Residual D2, failed the new test's prefetch-count
expectation: it required a positive count, but the executor correctly issued
zero prefetches. `paired-auto.cpp:2439` launches D2 prefetch only after the
last Residual; Early instead launches after Main. The test now checks the
exact eligible count, `j_groups * (k_panels - 1)`, and zero for D2 without
Residual. All numerical, full-I, operation-count and Scale-balance checks
remain intact. Production code was not changed for this correction.

The failed run, binary and inputs remain in `gates-extended-v3-collected`,
`gates-extended-v3-run`, `extended-check`, and source v5. Guest shutdown and
read-only fsck passed; the collector intentionally returned nonzero because
the gate failed. The corrected test passed RISC-V compilation
(`build-smoke-extended-r2.log`) and the complete 30-case FPGA retry.
All six unequal-I and all six zero-Residual cases passed. Zero-Residual D2
issued zero prefetches; Early issued exactly eight LdB-only prefetches.
API, Attention and PV-range gates also passed, all stderr logs were empty,
and guest exit, simulator exit and read-only fsck were successful.
`gates-extended-v3-r2-collected`, `gates-extended-v3-r2.sha256` and
`collect-extended-v3-r2.log` preserve that proof. The printf-image repeat
remains required.

After the successful guest shutdown, `rootfs-preprintf-inputs.sha256`
records the updated image including both preserved result namespaces;
the image hash is
`1e26ed3fb31df9c041b7354c97d8224f528202228aad8a57086d665475d1496a`.
The printf gate launcher verifies this checkpoint before first boot.

The new payloads use `H3_8H_RESULT_ROOT=results-printf`, leaving the earlier
`results` tree intact. The guest protocol requires the same namespace's
golden gate before diagnostics, performance or E2E. The three primary
benchmark shapes, inputs and route rotation remain unchanged.

The completed non-print-image diagnostic recorded these separate Tile
counts, identical across all six routes and cold/steady phases (36 records):

| Case | GEMM Tiles issued | Activation Tiles, command-declared | Weight Tiles, command-declared |
| --- | ---: | ---: | ---: |
| decode_1x768x768 | 72 | 72 | 12 |
| gpt2_ffn_up_16x3072x768 | 336 | 336 | 48 |
| tail_63x129x544 | 22 | 22 | 4 |

D2/Early's separate B-prefetch Tile counts were 6, 24 and 2 respectively,
not the total Weight Tile counts. These are software command counts from
completed diagnostic FPGA runs, not measured DMA traffic or final
performance results. They will be repeated on the printf-enabled image.

The six-route source/reproduction snapshot passed with
`bash /tmp/h3-8h/capture-sources.sh v3`, exit 0
(`capture-sources-v3.log`). `source-v3.sha256` identifies both source archives,
the reproduction archive, dirty patches, CMake cache, generated build manifest
and parameter header. Existing v1/v2 artifacts were retained. Software,
Chipyard, FireSim and original Gemmini HEADs still match the baseline above;
FireSim remains on `potal`. This snapshot does not establish an accepted
bitstream identity or hardware execution.

The printf integration correction, extended fixtures, scoped Vivado overrides
and new runtime helpers/configs are preserved in source/reproduction v4
(`capture-sources-v4.log`, `source-v4.sha256`), also exit 0. Source archives
include new untracked files; the Git patches describe tracked changes only.
The new helper scripts passed shell syntax validation, and the printf
runtime/HWDB YAML parsed with a matching configuration key and metasimulation
disabled. The new package's actual timing parser passed the same nine
accept/reject fixtures. None of these preparation checks substitutes for the
pending printf-enabled FPGA run.

### RISC-V staging correction

The final ELF audit found V/Zve/Zvl attributes in `libggml-cpu.so`, while the
other nine staged ELF files advertised scalar RV64GC. The cache had
`GGML_RVV:BOOL=ON`: `ggml-cpu/CMakeLists.txt` appends `-march=rv64gcv` for
that setting, independently of the toolchain's default scalar ISA. The
Rocket-specific configuration now explicitly sets `GGML_RVV=OFF` and
`GGML_RV_ZFH=OFF`; no global production default or toolchain file was changed.

The original image is `/tmp/h3-8h/rootfs-rvv-v1.img`, and its extracted
binaries are in `stage-check-rvv-v1`. Original cache, compile commands, ELF
attributes, checksums and staging log are retained with `rvv-v1` names.
This was a declared-ISA rejection, not an observed guest SIGILL or FPGA run.
The scalar rebuild and restaging passed, exit 0, in
`build-riscv-scalar.log` and `stage-rootfs-scalar.log`. All ten ELF files
now advertise scalar RV64GC. Current identities are `staged.sha256` and
`rootfs-inputs.sha256`; the new image is `rootfs.img`. This does not establish
FPGA execution. Restaging now requires the scalar cache setting, RV64 attributes for all ten
ELFs, and absence of V/Zv/Zfh attributes before copying any binary into a new
image.

### Initial six-route staging update (non-print image)

The corrected comparison adds H2 and the raw golden gate adds Early LdB.
That version required 18 raw golden successes across three fixtures and
subsequently passed them on FPGA, as recorded above. The printf-enabled
replacement requires 30 raw passes. The primary measurement protocol still
requires 84 cold/steady samples per case and six E2E routes; those performance
runs have not been completed.

The RISC-V hardware-smoke rebuild passed in `build-smoke-six-riscv.log`.
The native build attempt found no `potal-paired-smoke` target:
`CMakeLists.txt:604` registers it only for a RISC-V processor. Existing native
emulator gates remain the software proof; no native hardware-smoke success
is claimed.

`update-rootfs-six.sh` preserved replaced five-route inputs, proved the
other nine binaries unchanged, checked scalar attributes, and updated only
the smoke executable, guest script and checksum manifest. Round-trip byte
comparisons and read-only fsck passed in `update-rootfs-six.log`, exit 0.
Smoke SHA-256:
`fafa09d3d6cfb76e660db1f83313f428441bba7bf152cca5d335b4af804acf30`.
Rootfs SHA-256:
`c3dc2389907cb46896a72d512dc07bf9c7935f9f749591a57db5aa9154912c79`.

`POTAL_BENCH_STATS` records issued whole-GEMM Tile counts separately from
command-declared Activation Tiles and Weight Tiles. The latter equals
Main loops minus resident-Main loops plus separate B-prefetch Tiles.
Separate B-prefetch count alone is not total Weight loading. These fields
are emitted only for a successful paired run with no fallback and are not
physical DMA counts.

### First FPGA build failure and recovery

Placement completed with reported WNS +0.314 ns, but this is not final routed
timing acceptance. Vivado 2021.1 then terminated with signal 11 during
checkpoint creation. The stack passes through `HAPRDesignCheckpoint::create`,
`HANUSdcObjectHierNameCache::clearCache` and bundled tcmalloc. The exact cause
of that crash is not established.

The manager subsequently failed to copy results with `No space left on
device (28)`. Root storage had 186 MiB available. These are separate observed
failures; no causal claim equates the SIGSEGV with the full disk.
Original manager output, `impl_1/runme.log`, the crash stack and successful
`design_1_wrapper_opt.dcp` remain in their original locations.

Only this attempt's incomplete duplicate result copy was moved to
`/dev/shm/h3-8h-partial-results-v1`, verifying every regular file against
`partial-results-v1.sha256`. Original sources and primary build artifacts
remain on disk. This freed approximately 1.8 GiB without deleting prior
builds or logs. The RAM copy is temporary; its primary source remains on disk.

`resume-implementation.sh` verifies the opt checkpoint ZIP and hashes it,
then runs `resume-implementation.tcl` in `/dev/shm/h3-8h-impl-v2`.
It keeps the original Explore directives, setup and hold gates, DRC and
bitstream generation. New logs are `/tmp/h3-8h/recovery-v2.log`; the failed
manager run is not overwritten or relabelled successful. No RTL or constraint
change was made for this retry. Its outcome is recorded below.

The RAM retry also terminated with signal 11, this time inside
`sta::TagGroupBldr::clear` / `sta::Search::clearTagBldrs` during placement,
before checkpoint writing. Thus disk exhaustion alone does not explain the
Vivado failure. `/tmp/h3-8h/recovery-v2.log` and the RAM attempt's crash stack
are preserved. A third attempt uses `resume-single-thread.sh` and
`resume-single-thread.tcl` with `general.maxThreads=1`, the same input
checkpoint, directives, constraints and final setup/hold gates. This tests a
parallel STA/placement hypothesis; it is not a confirmed root-cause fix.
Outputs are isolated in `/dev/shm/h3-8h-impl-v3`, with persistent command
output in `/tmp/h3-8h/recovery-v3.log`.

The wait-stall audit found the original monitor `bash_187`
(`mon_9MX6PM7JDYDDJYB6`) still running. Vivado PID 3119678 used 100% CPU
after approximately 2 hours 22 minutes; its log was updating. The terminal
confirmed successful routed-net verification and completion of Phase 9,
Depositing Routes. The latest intermediate WNS/WHS were +0.058/+0.010 ns.
This is active build progress, not a hung wait or final timing acceptance.
At that audit point, the routed-checkpoint, final setup/hold gate and
bitstream steps were still pending.
The live build was retained; no replacement implementation was launched.

Routing subsequently completed successfully and wrote the routed checkpoint.
Its ZIP integrity check passed, exit 0, in
`/tmp/h3-8h/routed-checkpoint-v3-integrity.log`. SHA-256 is
`e5b8aa80ecb39f4616ce92a79f4ececab52914d22099496df4124b7c8bcc07bb`,
recorded in `routed-checkpoint-v3.sha256`. Original failed runs remain intact.
The Fully Routed DRC report lists 586 warnings across nine rules, with no
Error or Critical Warning entries in its summary. These warnings are retained,
not suppressed or assumed pre-existing: DSP pipelining accounts for 533,
with additional DSP placement, clock-group, LUT, RAM collision and no-load
advisories. DRC command completion is not a warning-free build or an FPGA
numerical result. The report is
`/dev/shm/h3-8h-impl-v3/design_1_wrapper_drc_routed.rpt`.

The separate methodology report is not clean: it has four Critical Warnings
(TIMING-4 x1, TIMING-27 x1, TIMING-54 x2), 36 warnings and one advisory.
The prior H3-8G2 routed methodology report has the same rule counts and the
same four critical clock diagnostics, including the clk_wiz primary-clock
redefinition and PCIe scoped clock groups. These are confirmed existing
findings, not new stream-specific findings, and have not been suppressed or
resolved. Positive routed slack is evaluated under the existing constraints;
it is not a clean methodology or CDC sign-off. Comparison baseline:
`/home/alveo/build_firesim_h3_8g2_d32/platforms/xilinx_alveo_u280/cl_xilinx_alveo_u280-firesim-FireSim-FireSimQuadRocketSinglePoTalD32EarlyPrefetchObservationConfig-BaseXilinxAlveoConfig/vivado_proj/firesim.runs/impl_1/design_1_wrapper_methodology_drc_routed.rpt`.

The final routed timing gate printed `WNS=0.058 WHS=0.010`.
The timing summary confirms zero failing setup, hold and pulse-width
endpoints, TNS/THS/TPWS of zero and minimum pulse-width slack of zero.
It reports no unclocked registers, unconstrained internal endpoints or
combinational loops. Two input ports and one output port lack delays; the
methodology findings above remain. Route status confirms all 947403 routable
nets fully routed, with zero routing errors. This establishes the existing
routed timing gate. The later bitstream/package completion is recorded above;
FPGA numerical proof remains separate.

### Runtime-enabled six-route primary measurement

This is the completed A/B cohort, not a C result. Three cases each contain
six routes, seven cold and seven steady samples: 252 samples and 252 matching
statistics records. Every sample passed exact output comparison, reported no
fallback and balanced Scale writes/releases. Cold samples had one weight-cache
miss; steady samples had one hit. The route order rotated and reversed exactly
as specified. All 36 median/MAD summaries were independently recalculated from
the individual samples.

The timer surrounds `ggml_backend_graph_compute`, including its preparation,
metadata, readback and host combination work. Internal stage/paired clocks and
logging were disabled. Stage fields equal zero because they were unmeasured,
not because those operations were free. The physical print file contains only
the clock-domain header. These are CPU-observed graph elapsed milliseconds,
not accelerator cycles or a sum of overlapping stage intervals.

Each cell below is median / MAD in milliseconds.

| Phase and route | Decode 1x768x768 | FFN 16x3072x768 | Tail 63x129x544 |
| --- | ---: | ---: | ---: |
| Cold H2 | 30.115 / 0.014 | 152.214 / 0.123 | 42.246 / 0.173 |
| Cold K-tile | 26.547 / 0.016 | 131.963 / 0.192 | 41.479 / 0.342 |
| Cold D2 | 27.747 / 0.015 | 136.629 / 0.262 | 41.541 / 0.264 |
| Cold Early LdB | 27.749 / 0.017 | 136.976 / 0.354 | 41.857 / 0.306 |
| Cold bounded | 28.901 / 0.209 | 148.771 / 0.911 | 41.993 / 0.213 |
| Cold continuous | 29.051 / 0.462 | 148.291 / 3.529 | 41.702 / 0.626 |
| Steady H2 | 14.890 / 0.014 | 89.475 / 0.258 | 38.776 / 0.095 |
| Steady K-tile | 11.315 / 0.013 | 69.186 / 0.131 | 38.225 / 0.133 |
| Steady D2 | 12.531 / 0.022 | 73.676 / 0.097 | 38.335 / 0.020 |
| Steady Early LdB | 12.556 / 0.009 | 73.780 / 0.130 | 38.597 / 0.297 |
| Steady bounded | 13.647 / 0.319 | 86.032 / 1.533 | 38.662 / 0.103 |
| Steady continuous | 13.896 / 0.250 | 84.224 / 1.153 | 38.709 / 0.118 |

In steady state, bounded is 20.61%, 24.35% and 1.14% slower than K-tile for decode, FFN and
tail respectively. Continuous is 22.81%, 21.74% and 1.27% slower. Thus A/B
does not establish a speedup over K-tile, despite improving decode/FFN relative
to H2. Defaults remain unchanged.

Every route/sample has the same whole-GEMM / Activation / Weight Tile totals:
decode 72 / 72 / 12, FFN 336 / 336 / 48, tail 22 / 22 / 4. These are issued or
command-declared counts, not DIM micro-operations, measured DMA bytes or bus
traffic. Weight total is ordinary Main loads plus separate prefetch loads,
not just the prefetch count. The separately captured A/B physical trace has
zero qualified future-B-write / current-Residual PE00 coincidence, including
decode; elapsed-time differences are not evidence of hardware overlap.

All routes reuse each case's fixed tensors, seed 20261023. Input FNV-1a64:
decode `a16d3ef162f8bbf4`, FFN `0ba73f4a34b75fa0`, tail `821c829ca69f2f68`.
Raw evidence is under
`/tmp/h3-8h/runtime-trace-performance-collected/performance/<case>/bench.out`.
The existing `run-guest.sh performance` protocol uses `--bench-stream` and
seven repetitions on the runtime-enabled image. Checksums are retained in
`runtime-trace-bitstream.sha256`, `runtime-trace-driver.sha256`,
`runtime-trace-performance-software.sha256` and
`runtime-trace-performance-disabled-trace.sha256`. GPT-2 TTFT/TPOT is a
separate, still-running gate; these graph measurements do not replace it.

### Conditional Early composition: native implementation

The measured A/B capture justified C: all 24 graphs had zero qualified
next-panel B-write / previous-panel Residual PE00 coincidence. The selected
contract is `/tmp/h3-8h/early-stream-contract.md`.

`CONTINUOUS_EARLY` preserves the existing enum values and default `OFF`.
It uses exact query/signature `0x4853384801000000` /
`0x4853384801200807` and begin operation 6. The checked C11 API appends
default-zero `early_b_only` and adds `potal_paired_stream_panel_prefetch`.
Admission is not DMA completion. Each B-only command has its own raw epoch,
pair and generation zero, with real generated/RS/DMA terminal ownership and
no fabricated compute work. Local wait completes a producer before Main;
the checked normal Main encoder then suppresses only the equivalent B address.

The executor inserts a next-panel declaration and B-only command directly
after Main only when the opposite pair is already free and K/J adjacency
matches. It moves no retirement, release, Scale, Index or numerical operation
and inserts no drain to manufacture eligibility. A busy target retains
ordinary Main loading. The original six benchmark routes stay unchanged.
New explicit `--bench-stream-early`, `--observe-stream-early` and
`--stream-early-golden` additionally exercise C. Manifest `path=b_prefetch`
identifies its generation-zero, zero-compute epoch; A/ACC use unused sentinels.

`POTAL_STREAM_ADMISSION_PROFILE=1` is an additional, default-off CPU diagnostic.
It times the B-only API, including staging and the admission response, into
preallocated storage; records print after END. The separate guest command
`sh /root/h3-8h/run-guest.sh admission early` requires completed Early gates.
Physical trace and primary timing explicitly unset this option. The primary
benchmark refuses it. These CPU durations are not hardware waiting or DMA
time, and are not mixed into the passive physical-overlap capture.

Evidence under `/tmp/h3-8h/`:

- `early-stream-api-d32.log` and `early-stream-api-geometry.log` retain checked
  API lifecycle, refusal, failure/quarantine and geometry regression evidence.
- `early-stream-execution-d32.log` ends `PASS: 0 failure(s)`, exit 0.
  This is software-only exact Main/Residual INT32, host reconstruction and
  final output proof, including command ordering, capacity, tails, absent
  Residual and actual hot-clock interception. It is not hardware timing.
- The DIM16 compatibility execution regression also exited 0. Scalar RISC-V
  `potal-paired-smoke` and `llama-cli` compiled successfully in the separate
  `d32-early-performance` directory; neither result is a C hardware run.
- `admission-test-d32.log` and `admission-test-d16.log` both end
  `PASS: 0 failure(s)`, exit 0. The observer verifies exactly two clock calls
  per explicitly profiled B-only API and zero hot clocks in the ordinary
  stream runs. The diagnostic preserves exact software numerical outputs.
  The first command attempt stopped before compiling because of a directory
  typo; its failure output was retained rather than relabelled a test result.
- `early-stream-rtl-r3.log` has 27 succeeded, zero failed, exit 0.
  The first two logs remain intact. Their sole failure was a fixture that
  withheld all B0 responses while requiring B1 DMA. DIM32 has one real load
  command credit, `(16 / 32) + 1`. The corrected fixture retains both epochs,
  returns B0 bytes and proves pair-0 wait completes while pair 1 stays busy,
  then completes B1. No production credit/arbitration change or timeout
  relaxation was made. Details: `early-stream-debug-notes.md`.

`PoTalEarlyStreamXilinxAlveoConfig` gives the rebuild a separate artifact
identity while preserving the existing target geometry and runtime trace
control. `generate-rtl-early-stream.sh` checks all required trace bridges;
`build-early-stream-image.sh` retains the prior timing/DRC gates. The old A/B
image, binaries and measurements are not replaced.

The separate `analyze-early-stream.ts` and five `early-stream-*.sql` analysis
files are ready for the closed C capture. Their audit checks all 36 streaming
graphs, preserves the 5160 expected compute commands, and counts B-only
producers from observed identities rather than predicting their number.
Zero-work effects, producer/Main binding, terminal reference conservation and
actual same-cycle B-SRAM/Residual-PE00 coincidence have synthetic positive and
negative checks. This is not a physical result. The event schema cannot prove
individual generated opcodes or untraced Scale/Index effects; source/RTL gates
provide that evidence. Details and the existing Node/Bun declaration conflicts
from strict TypeScript checking are in `early-stream-analysis-notes.md`.

Pending: C FPGA numerical/physical/timing proof and matched controls,
complete GPT-2 TTFT/TPOT and final source/binary/bitstream hashes.
No performance gain is inferred from CPU command latency or expected overlap.

The new `early-stream-host.sh` prepares isolated package/stage/program/launch/
send/collect operations. Its shell syntax passed. The new Early deployment
YAML parsed successfully and differs from the baseline runtime settings only
in artifact identity and simulation directory. YAML LSP is unavailable; no
dependency was installed to conceal that limitation. These helpers have not
yet deployed the C image.

The staging operation refuses an active guest and requires completed baseline
E2E collection. It clones, rather than modifies, the baseline rootfs, installs
the sealed v13 software, compares extracted bytes and checks the filesystem.
Package/program/launch retain routed-timing, checksum, branch and exclusive
FPGA guards. Collector checks 35 golden cases, 98 primary samples per case,
or seven deterministic E2E runs as appropriate. The separate admission phase
requires CPU diagnostic records while physical tracing remains disabled.

The final `analyze-e2e.ts` runner and `e2e-{parser,load,metrics}.sql` are
prepared for both closed E2E cohorts. They preserve uint64 timestamps,
require the completed 42-request baseline and the new seven-request C cohort,
with matching token IDs. Baseline request median/MAD remains available.
C has one request per route: its TTFT and mean TPOT are single measurements,
and request-level MAD is unmeasured (NULL), not zero. Inter-token
distributions are reported separately and are not repeat variability.
TTFT starts with an already tokenized prompt, before inference; model load,
tokenization and process startup are excluded. Archived host parsing and
synthetic positive/negative checks passed. The completed baseline's 42
actual requests have now passed separate timestamp analysis; C's seven
requests and the combined final audit remain pending. Strict TypeScript
checking was unavailable on the discovered
paths; Bun transpilation passed. See `e2e-analysis-notes.md`.

The current C build emitted `[Project 1-498]` while hierarchy black boxes
were unresolved. Its deferred wrapper XDC contains five queue `RAM_STYLE
ULTRA` attributes, not timing-path exceptions. The project file marks the
source XDC for both synthesis and implementation. Final application/object
binding remains a separate open verification item; no constraints were
edited or warning suppressed. See `early-stream-deferred-constraints.md`.

The pre-deployment scope audit passed: all four recorded HEADs and the
original Gemmini gitlink are unchanged, FireSim remains on `potal`, no staged
diff exists, the software whitespace check passed, and the saved CDE checksum
matched. Current changes remain uncommitted. Early deployment YAML is ignored
by `deploy/.gitignore`, so the reproduction archive includes it explicitly.
Unrelated dirty/untracked paths were left untouched; historical byte identity
is not claimed for paths lacking startup content hashes. See
`predeploy-scope-audit.md` and its retained log. Later source/scope changes
require reopening this check.

During C leaf-clock-delay optimization, available host RAM fell below 6 GiB.
Vivado alone was reversibly paused. Six inactive, task-owned RAM artifact
directories were copied to disk, SHA-256/structure-verified, and relocated
with their original paths preserved by symlink. Existing header, physical
trace, build-input and failed partial-copy manifests all passed afterward.
Available RAM recovered from 4.9 GiB to about 14 GiB; the same Vivado PID was
resumed. FireSim/E2E continued, and no RTL, constraint, directive or executable
changed. This host-resource intervention is recorded, not treated as timing
acceptance or proof of unchanged host wall time. See `ram-recovery.md` and
`evacuate-inactive-ram.log`.

The C image's actual final routed timing marker reports WNS **+0.006 ns**
and WHS **+0.010 ns**. All 988,484 routable nets are routed, with zero routing
errors. The DRC summary retains 586 warnings, matching baseline rule/counts;
the methodology report retains four Critical Warnings. Its complete body
and five selected constraint/build input files match the baseline. These
pre-existing findings remain disclosed; positive slack under the existing
constraints is not warning-free signoff or FPGA-functional acceptance.
See `early-stream-route-audit.md`. Bitstream/native build completion, final
queue-RAM inspection, and C hardware testing remain pending at this point.

The C bitstream/MCS generation subsequently completed, and the native build
exited 0 with `H3_8H_EARLY_STREAM_BUILD_EXIT=0`. Final checkpoint inspection
and packaging are running separately. This is build acceptance only;
C FPGA programming, numerical/physical gates and full E2E remain unverified.

The C package then completed with `H3_8H_EARLY_STREAM_PACKAGE_PASS`, exit 0.
Retained identities under `/tmp/h3-8h`:

| Artifact | SHA-256 |
|---|---|
| `firesim-early-stream.tar.gz` | `83ed304460de5b881ad7ebced4fb076c7469ef8679cbf43aaa4fd81e9982c971` |
| `early-stream-package/xilinx_alveo_u280/firesim.bit` | `0bb86df67bfeb343082599f4b4224179dc9c64a2127ee4dd341b9144bb1195c9` |
| `early-stream-package/xilinx_alveo_u280/firesim.mcs` | `afed8cc63fbb82e3f74d0a7a48d81eeddbbb5a0f543a486c8fa1a83a273b0e0e` |
| Early FireSim driver | `06976e57db0373813cd1c0b63da32f54666507c3c77fc73d2289341667b12545` |
| `early-stream-cad-reports.tar.gz` | `4da5d0825a6403fb7a9a02b9df6142ba043dc4d835538b207f47ee2d96a57d30` |
| `early-stream-driver.tar.gz` | `575f8aecb0f68bee83bb84df27964eb6079b96173ea7d0f951bb9ea5b2e16983` |

The package records FireSim `6204ce53bdf970db757d47cf936b6f6ba4b1a4c9-dirty`;
the compiler alias is `PoTalEarlyStreamXilinxAlveoConfig`. The full driver
path is in `early-stream-driver.sha256`. Packaging is not deployment or
hardware acceptance; the existing baseline E2E guest remains active.

**Final RAM mapping check: FAILED (inherited platform finding).** The
unchanged checker exited 1: all five host-stream queue patterns resolve to
BRAM, not URAM, and RAM_STYLE values are empty. No black boxes remain.
The source XDC's relative `top/...` queries fail to bind in the wrapper
hierarchy; the tracked platform loader is unchanged from HEAD. Baseline
has the same five no-match warnings and zero placed URAM, with matching
global BRAM counts. The failed logs and expectations are preserved.
See `early-stream-ram-finding.md` for per-queue counts, source evidence and
scope limits. Verification is complete with an explained pre-existing
failure, not a RAM_STYLE pass. Platform defaults were not changed; C FPGA
numerical, physical and E2E results remain separate pending gates.

The completed v15 source/reproduction capture preserves the unchanged
failing RAM checker and its diagnosis. `source-v15.sha256` records all
archive/diff identities; `reproduction-v15.tar.gz` has SHA-256
`5584eea72e434d0eaa348c9d88444be69eef7ac8fa0bb5fea0ec25cad2a1d9d5`.
Executable C/C++ remains the sealed v13 software; later changes here are
progress documentation. The existing baseline E2E has reached 17/42
completed requests, not a completed end-to-end comparison.

The baseline six-route E2E subsequently completed **42/42 requests**.
Guest and actual simulator exits were zero. After normal sync/poweroff,
`collect-runtime-trace.sh e2e` exited 0: all 42 records had eight token
events and a request-end event, all generated outputs matched baseline H2,
the filesystem/checksums passed, and internal hardware trace remained off.
The completion observer's exit 141 is retained separately from those
successful guest/simulator/collector results. Final timestamp arithmetic
and cross-cohort latency analysis still require the complete C cohort.

C staging then passed with exit 0: sealed v13 binaries were installed in
the separate `early-stream-rootfs.img`, extracted bytes matched, filesystem
checks passed, and the original parent image remained unchanged.
C FPGA programming is in progress in `program-early-stream.log`; no C
hardware numerical or performance result is claimed yet.

C FPGA programming subsequently completed with
`H3_8H_EARLY_STREAM_PROGRAM_EXIT=0`, exit 0. The first C numerical-gate
guest is booting in the separate Early slot. Phase-specific UART payloads
are retained as `uart-early-stream-*.txt`; gate results remain pending.

Login and shell readiness completed, and the C gate command was delivered
after subscribing to its exit marker. API smoke, 35 golden cases and the
Attention gates are now running. No C gate pass has yet been observed.
