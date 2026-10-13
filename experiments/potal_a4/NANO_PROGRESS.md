# Nano continuation

## Local preparation for vanilla Gemmini QK/PV replay (2026-10-13)

The user requested local code for remote execution, with vanilla QK/PV first and
removable optimization experiments. `im2p_measure.py` defaults to no mask and
prepares the existing quantization, BK/lane schedule, int8 buffers and independent
NumPy references before executing. `--experiments` enables direct host wide
accumulation and the existing QK-only mask plan. No upper-lane specialization or
quantization change is introduced. PV retains the full zero-point domain.

Reuse: the existing raw probe owns the real provider and strict numerical-runtime
identity checks; it now also builds `potal-im2p-replay`. The existing CPU batch
loop moved to `raw_batch.hpp`; CPU and Gemmini share its checked wide shifts,
row scatter and publication-after-success contract. Prepared replay subclasses
the existing descriptor packer rather than adding another packet format.
Product activation main/residual and stacked RMD code were not edited here.

Measurement is component-level: fresh runtime per variant/repetition; real RTL
cycles and read/write/preload counters are collected over all raw requests.
Preparation, native simulator wall time and reference/publication work are
separate. The raw request sequence is the existing per-BK/per-lane profiling
schedule, not certification of the product stacked RMD scheduler. No new NPU
cross-request operand reuse, whole-model latency or new PPL result is claimed.

Local evidence: rebuilt `potal-native-replay` and `potal-native-model`; CTest 7/7
passed, including the shared batch failure-publication and >int32 accumulation
cases. Added prepared-job checks cover row/K/output tails, repeated full/direct
execution, all-masked QK and zero P. GPT-2 and Llama layer0/head0 QK/PV captures
from `results-upper-compare-gpt2-10chunk` and `results-qk-batch-10chunk` passed exact
NumPy int64 accumulation, reconstructed float64 and saved float32 comparisons.
Both C++ preprocessor modes pass `c++ -std=c++17 -Wall -Wextra -Werror
-fsyntax-only` using the actual IM2P headers. Python rule audit passed.

For each sampled QK head, dense has 96 raw calls; the optional mask plan has
192. Main fragments are 512 to 272 per K nibble pass. Thus extra request cost
must remain in the RTL measurements, even after eliminating Python setup from
the execution interval. Each sampled PV head has 108 raw calls; it remains dense.
Full-partial and direct modes make identical raw calls and preserve exact values.

The raw hardware probe now has 18 cases (old 15 plus three strided-buffer cases).
These updated RTL cases and the new shared-library link/run have **not** run
locally: there is no matching macOS A4/W4/DIM32/ACC512 numerical archive.
Nano's recorded compatible source `20b82f0` exposes the same used ABI fields.
Remote build and vanilla-first commands are in [README.md](README.md#gemmini-qkpv-원격-검사-vanilla-기본값).

## Local common ExSIA source-zero policy (2026-10-13)

The adopted source-zero population policy now applies to all supported ExSIA
activation widths (4, 8 and 16). The earlier nano change had an A4-only compile
condition. Both product callers still use the existing `LocalStage`; optimized
full/tail and validation reference paths now use the same population. Original
nonzero inputs that round to integer zero remain in the population.

Git history places the A4-only guard in nano preservation commit
`63e166da635bcbedaf9edb8cb84bd3e88851c3dc` (2026-10-12 13:26:38 +0900).
Its implementation plan scoped changes to A4/a4nks and preserved legacy A8;
that scope restriction had also been applied to the common population policy.
The activation-width build option itself predates that patch and is retained.

The optimized statistics loop reuses the exponent already stored in `blk.e`.
A source zero writes `q_out[i] = 0` before skipping preliminary quantization and
statistics. There is no new allocation, mask or scan. Three sum-only
`WideQuantizer` overloads no longer test source zero: zero contributes nothing
to S/SS, and those helpers do not count samples. The strict one-sided 2-sigma
formula, top selection, output layout and PE operand representation are retained.
The full product/test/config change is [source-zero.patch](results-source-zero-common/source-zero.patch).

The existing source-zero fixture is generalized using `meta.rho`, not duplicated
per width. Eight cases cover signed source zero, nonzero rounded zero, all-zero
and singleton blocks, full BK32 blocks, one/seven-column tails and prefilled
output storage. Independent expected count/S/SS/exponent/mask values are checked
along with optimized/reference code equality. The initial generalized fixture
passed on pre-fix A4 and failed on pre-fix A8. Final Release CTests pass A4 9/9
with OpenMP OFF and 9/9 with ON, A8 14/14 and A16 9/9.

Native GPT-2 WikiText-2 test, context512 x10, 5,120 input / 2,550 scored tokens:

| Product configuration | Before PPL | After PPL | Delta |
| --- | ---: | ---: | ---: |
| A4/W4, existing source-zero exclusion | 40.563992429741681 | 40.563992429741752 | +7.1054e-14 |
| A8/W8, source-zero exclusion newly applied | 29.357533546738331 | 29.356037576203310 | -0.001495970535021 |

Both before/after runs completed all ten chunks. A4's maximum absolute
per-chunk NLL delta is 1.819e-12; this does not claim bitwise logit or per-token
identity. A8 has mixed-sign chunk deltas and a -0.00510% aggregate PPL change.
Totals, chunk deltas and log hashes are in
[comparison.json](results-source-zero-common/comparison.json).

The pair explicitly selects `--device GEMMINI`, CPU main/RMD and FP attention,
DIM/BK32, physical geometry `(4,4096,1024)`, four threads and sequential ExSIA.
Within each width the model, token prefix, CMake cache and generated geometry
hashes match. A4 uses Gemmini OpenMP ON, A8 OFF; compare before/after within a
width. Earlier default-device BLAS probes and the differently configured A4
OpenMP-OFF PPL probe are excluded and retained with reasons in
[inputs.json](results-source-zero-common/inputs.json). PPL runs overlapped builds
and other runs, so their times do not establish a speedup. This is not an a4nks
attention or Gemmini/IM2P hardware measurement. A16 has functional coverage only.
The previous nano A4 include/exclude experiment below remains a separate result.

## Local native batching and direct accumulation (2026-10-12)

The follow-up modifies the existing `native_model.cpp` raw CPU kernel,
`profile_gemm.py` scheduler, `native_measure.py` replay/measurement runner and
`verify.py`; it does not add a parallel attention implementation. A stripe is
submitted once, with one descriptor for each original BK32 block, lane and K
nibble pass. The descriptor retains the original row map and per-column shift.
Native execution reuses the existing int8 x int8 -> int32 raw dot. It accumulates
into private int64 storage using checked int128 scaling, and copies the result
out only after all work succeeds. Int64 overflow is rejected, not saturated.
Native input values must be signed nibbles stored in int8; a full folded Q code
cannot be silently narrowed to int8. Prepared Python oracle arrays remain int64.

The direct variant uses one reusable raw-region buffer instead of a full-width
partial for every block/lane. A full-width native control is retained to measure
that difference. Both batch variants move SCU into native code, so the first
comparison includes packing/native-merge changes as well as reduced FFI calls.
The original Q/K/P/V quantization, logical stripes, mask planning and weighted
PV zero point are preserved. PV never skips masked tiles in this experiment.

Release build and CTest pass **7/7**, including new batch checks for per-column
and lane shifts, a result above int32, invalid rows/regions/overflow with unchanged
published output, nibble-range validation, reordered residual rows, masks and
PV zero-point cancellation. Llama-3.2-1B Q4_HP1, context512 x10, gives
**13.801307254545433** in both dense and batch-direct modes (5,120 input / 2,550
scored tokens). Packets, scales, valid QK, softmax, PV, logits and token NLL are
exactly equal. All 10 dense records also match the previous QK experiment's
dense records. Tail33, prefix31, decode1, cached suffix17 and packed-sequence34
real graph scenarios all pass in both modes, including position/sequence mask
checks. The new raw batch therefore has evidence beyond equal aggregate PPL.

Apple M5 CPU component timing, 15 QK / 15 PV captures, one warmup and five
rotated-order repeats; sum of per-capture medians, excluding common quantization:

| Component | Dense ms | Existing QK skip ms | Native batch, full partial ms | Native batch, direct ms |
| --- | ---: | ---: | ---: | ---: |
| QK | 399.586459 | 397.461962 | 368.414042 | **326.162122** |
| PV | **160.280377** | 161.176461 | 171.007750 | 175.863913 |

QK direct accumulation reduces time **17.94%** against existing skip and
**18.38%** against dense. Direct accumulation contributes a further **11.47%**
against the full-partial batch. QK submissions fall from 2,880 to **120** per
15 captures while the same 2,880 retained raw regions execute. Its DIM32 work
counts also match the previous skip experiment exactly; batching itself does
not remove additional MACs. All 15 QK captures improve against dense. PV time
increases **9.72%** despite submissions falling 1,660 -> 60; 14/15 captures
regress. Retain QK as the implementation candidate; do not enable PV batching
by default. Product dispatch/defaults have not been changed.

An earlier independent component run reaches the same directional result:
QK existing skip 404.402 -> direct 319.964 ms; PV dense 161.413 -> direct
173.078 ms. Dense versus unbatched skip varies between runs, so use within-run
controls rather than mixing old and new medians. Final repeat sums for QK
direct are 321.916..336.091 ms, versus dense 393.267..420.857 ms. These ranges
use a different aggregation from the table's sum of per-capture medians.

The full-partial batch initializes 188,612,608 bytes of full-width QK partials
across the 15 captures; the direct variant removes those initializations.
This is cumulative buffer work, not peak memory. Workspace counter definitions
and both runs' raw measurements are in [EXECUTION.md](results-qk-batch-10chunk/EXECUTION.md)
and [component-summary.json](results-qk-batch-10chunk/component-summary.json).
The final [report.json](results-qk-batch-10chunk/report.json),
[chunks.jsonl](results-qk-batch-10chunk/chunks.jsonl) and
[mask-scenarios.json](results-qk-batch-10chunk/mask-scenarios.json) retain exact
results and model/token/source/library identities. All measured Python/C++
source and library hashes match the final working tree.

Small-token QA took longer in the existing CPU linear route; a read-only
[stack sample](results-qk-batch-10chunk/prefix-diagnostic.txt) locates active
weight reads/native matched-width dense execution. It is not a QK batch timing.
No product linear code was changed to speed up the checks.

Remaining work is product StreamQuant/direct-limb and linear integration,
arbitrary-width exact merge, actual raw Gemmini/IM2P submission, a4nks role/cache/
GQA dispatch and hardware timing/coverage. The current batch is a CPU replay
prototype; its timings establish neither accelerator cycles nor whole-model
speedup. Mask-plan reuse and per-KV-head preparation reuse are separate possible
follow-ups. See the existing implementation plan; its product tasks remain open.

## Local continuation: QK masked tiles (2026-10-12)

The active local test checkout is
`/Users/chan/Projects/AISA/llama.cpp-gemmini/.worktrees/potal-qk-mask`, branch
`work/potal-qk-mask-local`. The dirty parent `develop` checkout is preserved.
Nano SSH authentication was unavailable, so 38 successful source-edit records
from the previous session were replayed on `f588ad4`; all 18 recovered files
matched the reconstructed contents byte-for-byte before this increment.
That initial recovery excluded generated measurements. The user's nano commit
`63e166da635bcbedaf9edb8cb84bd3e88851c3dc` has now been fetched and compared:
all 13 execution/test/build files and `ORIGINAL_SPEC.md` match the recovery
byte-for-byte. Four handoff documents gained transfer notes, now incorporated.
The local test branch was fast-forwarded to this commit with its QK changes
preserved. Git's 63 inventoried measurement files match their recorded hashes.
The 20 local source/document files, 71 local result files and measured runtime
hashes were checked after restoration. Only provenance/handoff text changed
after that check; no numerical experiment was rerun for synchronization.
The named pre-sync stash remains as a recovery copy. The separate transfer
archive was not received: SCP still fails SSH authentication, so its payload
has not been verified locally. See [nano-sync.json](results-qk-mask-10chunk/nano-sync.json)
and [source-recovery.json](results-qk-mask-10chunk/source-recovery.json).

The QK-only experiment extends existing owners:

- `build_attn_mha` supplies the actual softmax mask as an optional QK dependency;
  the replay ABI is version 2 and retains that dependency's lifetime.
- `profile_gemm` skips a DIM32 row/key tile only when every mask entry is
  negative infinity. Residual plans use original query-row IDs after compaction.
  Both K nibble passes use the same mask. Original Q/K preparation, stripe
  geometry, scales and digits are unchanged.
- Skipped scores are published as finite zero before the existing FP softmax.
  PV and its full weighted zero-point correction remain unchanged.
- `native_measure qk-mask` compares the existing dense integer replay with the
  candidate, including packet, scale, valid-QK, softmax, PV, logits and token-NLL
  hashes. This reuses the native model/tokenizer/RoPE/GQA/linear path.

The local Llama-3.2-1B Q4_HP1 checkpoint SHA256 is
`d483ca08fdfd166b6afba4cf146ea865d51519a80cab01022f7ab71694bc63a5`.
The paired context512 x10 result is **13.801307254545433** for both modes:
5,120 input tokens, 2,550 scored tokens, and exact per-token/hash equality.
These are local paired results, not a PPL improvement over the older nano model.
Nano's model-file SHA256 is
`48efffb2bd7b0f5fbd2414d536ff28d637145ac99c19f2f6c2a57f3239ee24e8`,
different from the local file. The text and token hashes match exactly. The
other local `models/default` Q4_HP1 file also has a different hash. No tensor-level
equivalence between these model files has been established, and the absolute
nano/local PPL values must not be interpreted as an optimization comparison.
See [chunks.jsonl](results-qk-mask-10chunk/chunks.jsonl) and the pre-run
[runtime-inputs.json](results-qk-mask-10chunk/runtime-inputs.json).

Per K pass, QK main work falls from 2,621,440 to 1,392,640 DIM32 equivalents
(46.875% removed), and upper work from 5,241,024 to 2,795,090. These count actual
CPU replay submatmuls with DIM padding; they are not observed Gemmini submissions
or the specification's pooled lane-row cost. K-high repeats both categories.
The dense and candidate use the same ordinary integer CPU dot; unit-add is off.

The context64 smoke also passes tail33, prefix31, single-token decode, a
17-token cached suffix and two packed sequences. It checks the actual graph mask
against absolute positions and sequence IDs, then compares both executions.
The full run passes the same five native scenarios in both modes. Actual masks
match the independent position/sequence check, and valid QK, softmax, PV and
logits remain exact. Functional CTest passes 6/6, including the independent
arithmetic oracle and QK causal/tail/row-map/finite-bias fixtures. Nearby ExSIA
baseline, stripe-geometry and source-zero CTests pass 3/3; direct meta-rho and
non-outlier-clipping cases also pass. Build and test logs remain under
`build-potal-qk` and `build-potal-replay`.

CPU timing uses Apple M5, 15 QK and 15 PV captures from chunk 0, one warmup and
five alternating-order measurements per mode/capture. The table sums the
per-capture medians. Common quantization/unpacking and softmax are excluded;
mask classification/planning, packing/conversion, host SCU and finite-zero
publication are included. No other experiment or build ran during these timings.

| CPU replay component | Dense ms | QK-skip ms | Time change |
| --- | ---: | ---: | ---: |
| QK, 15 heads | 365.941457 | 403.154123 | +10.1690% |
| PV, 15 heads, unchanged path | 162.060042 | 160.476208 | -0.9773% |
| QK + PV, 30 heads | 528.001499 | 563.630331 | +6.7479% |

QK repeat sums range from 373.183 to 379.774 ms for dense and from 402.836 to
407.215 ms for the candidate. These are sums at each repeat index, a different
aggregation from the sum of per-capture medians. PV's small difference is timing
variation on identical code, not a PV optimization.

For each capture's median QK run, native dot time totals 159.840 -> 95.886 ms,
while native dot calls double from 1,440 to 2,880. Conversion/allocation/FFI time
grows from 48.501 to 70.451 ms, and time outside dot calls from 157.541 to
236.692 ms. The latter includes mask planning, slicing, SCU and publication; it
must not all be attributed to mask classification. The lower MAC work is real,
but these overheads outweigh it in the current CPU replay implementation.

Keep the original candidate opt-in through `qk-mask`; this historical run alone
does not support a CPU speedup. The batching follow-up is now recorded above.
The numerical gate
supports future Gemmini scheduling work but does not establish its performance.
All measured source/library hashes still match the pre-run snapshot. See
[report.json](results-qk-mask-10chunk/report.json),
[mask-scenarios.json](results-qk-mask-10chunk/mask-scenarios.json) and
[component-summary.json](results-qk-mask-10chunk/component-summary.json).

Product a4nks dispatch remains incomplete. Original native FP QK/PV dots still
run before replay replacement, so replay wall time is diagnostic. No new
Gemmini/IM2P latency, CUDA latency, PV tile skip or full-model speedup is claimed.

## Historical nano checkpoint

This branch is the test workspace `work/potal-attn-im2p-nano`. On 2026-10-12 the
user requested a commit and normal push of the existing work and evidence on
this branch for local continuation. Do not merge into develop/hotfix, discard
existing changes, or start another implementation/experiment during this
transfer. Moving the work to hotfix remains a later task.

At the initial audit, fetched `origin/develop` was
`3b2eaa3cb6f7adc36094b7f78662141825cc4ef4`. The product tree at the pre-change
HEAD `f588ad4bb280147a152c0d23f624526fff64d45a` was identical to that revision:
the 81 differing tracked paths were only the PoTal
experiment directory and handoff document. Nano's local `develop` is stale at
`e5eb928e80382cbc89d8ae1d7953d502ffd3632d` (359 product-file differences); it was
not used as the baseline. The product tree was unchanged at that initial audit.
It now contains the A4 source-zero exclusion patch in `quants/act/exsia/local.cpp`
and its tests; no other product algorithm change has been made in this session.

The accepted policy remains `potal-a4-preserve-v1`: selective linear, all required
Q limbs, original geometry P uint8, K [-128,119], and block/channel V4. Product
attention dispatch and fused limb emission remain incomplete.

## Adopted zero policy and specification audit

The user explicitly adopted excluding source zero from the sigma population
after reviewing the result below. Keep this change in future A4 baselines;
it is no longer waiting for an adoption decision. A nonzero source that rounds
to q=0 remains included. The original Nano patch covered A4 only. The user has
since clarified that this population policy applies across supported ExSIA widths (A4/A8/A16);
the A4-only implementation and test guards have been removed in the local worktree.
Local validation is recorded separately from the historical Nano measurements below.

`results-outlier-zero/` contains the paired native product measurements:

| Scope | Before | After |
| --- | ---: | ---: |
| Sigma population | Includes source zero | Excludes source zero (adopted) |
| Context / chunks | 512 / 10 | 512 / 10 |
| Input / scored tokens | 5,120 / 2,550 | 5,120 / 2,550 |
| PPL | 40.18882947104453 | 40.20531899910632 |

The relative change is +0.0410301%; four chunk NLLs improve and six worsen.
Model/text/binary/source hashes and chunk NLLs are in `inputs.json`, `before.json`,
`after.json` and `comparison.json`. The new population fixture failed before the
patch; the patched full/tail/reference fixture and nearby ExSIA CTest gate passed
5/5. Both PPL builds use A4/W4/DIM32/BK32, CPU main and CPU-direct RMD, FP attention,
batch/ubatch512 and 4 threads. Generated geometry is `(4,4096,1024)` in both, not
the original `(4,4096,512)`. The pair measures this policy change on the existing
product path; it does not prove a4nks, original geometry or RTL parity. CPU elapsed
times are not performance comparisons because compilation overlapped portions of
measurement. A layer/token-level cause analysis of the PPL difference has not
been run; changing U can change selection and scales but does not guarantee an
improvement in PPL.

The product's folding stage may still mark a zero as a D>0 correction candidate.
That is permitted by the supplied corr rule and does not make zero a sigma outlier
or a nonzero upper digit. Source-zero population, top, sigma, correction candidates
and actual upper work must be reported separately.

The re-supplied A-D specification has been mapped to the ten implementation tasks
in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). Missing explicit gates now
include geometry, full Q4_HP1 scale decoding, 10-chunk paired comparisons, timing
prediction versus the actual supported opcode, and cleanup after each adopted
increment. The seven-test scalar oracle is partial task 1 evidence; source-zero
handling is partial task 4 evidence. No complete product task is checked off.

The current checkpoint has 48 Q4_HP1 tensors, 2,654,208 blocks, no zero blocks,
and no nonzero block with a nonpositive/non-power-of-two channel_scale (direct
GGUF field scan). Its spec scale can therefore be represented as
`thetaW=m+ilogb(channel_scale)`. Native decoder/full-GEMM comparison remains a gate;
the field scan alone does not establish that integration.

## Existing implementation audit before further edits

The user requested checking existing code before adding implementations.
[The reuse map](IMPLEMENTATION_PLAN.md#existing-implementation-and-reuse-audit)
now traces ExSIA local/folding stages, balanced digits/bitmap packets, run-aware
packing, validated weight readers, matmul output ownership, the raw IM2P provider,
the shared attention graph, geometry and evaluation/cycle records to their
existing owners and tests. Existing `BLOCK` quantization, contiguous F16 handling
and saturating SCU/checked-int64 compose were inspected: their contracts alone
do not implement the requested K/P/V, strided views or exact wide accumulation.

Tasks 1-10 now extend existing source/test owners first. Unconditional new PoTal
executor, attention module, test executable and duplicate contract-document
instructions were removed. A small private extraction remains possible when a
specific missing responsibility and real callers justify it. Recheck callers
and remove replaced unused paths after each adopted increment. This audit changed
planning/handoff documents only; it adds no numerical or runtime validation claim.

## Local environment and inputs

- Linux `nano2`, aarch64, kernel `6.8.12-1021-tegra`, 7.4 GiB RAM.
- Python 3.12.3, NumPy 1.26.4. Isolated `/tmp/potal-nano-venv` adds tokenizers
  0.23.3. `rtk` was not installed, so commands ran directly.
- `models/gpt2-base/gpt2.Q4_HP1.gguf`, downloaded
  `models/gpt2-base/tokenizer.json`, and both WikiText-2 sources match every SHA256
  specified in the handoff. Model/tokenizer locations differ from the original
  model path in the handoff. Tokens and captures are local generated artifacts.
- IM2P source: `5da6a4cf0d1da38211b14cacf409bb1abc8d458a`;
  Gemmini header repository: `fcfcf5ce7c4670bd789365f11810177f97016f81`.
  The installed IM2P checkout is untouched; the compatible isolated source used
  for the completed build is `20b82f058aa8b0e69c48d1d00790d8e19eaa6f66` below.

## Numerical and native preparation evidence

Linux `ru_maxrss` is now normalized from KiB to bytes in both `run.py` and
`native_probe.cpp`. Darwin retains its byte interpretation. Historical results
are untouched.

`tests/potal_a4_reference.py` is an independent Python-integer/Fraction primitive
oracle, not a complete reference forward. Seven tests cover full int32 radix
digits, selective +8, K and P code reconstruction, real-tail/source-zero sigma
populations, strict integer arithmetic, catastrophic cancellation, double
rounding, subnormal, infinity and signed-zero publication. The oracle is registered
in the experimental probe CTest. It does not complete plan task 1's linear/full
masked-attention oracle and model fixture requirements.

`results-nano/verify.log`: 141,076 integer roundtrips.
`results-nano/ctest.log`: 5/5 passed, including bitmap LOG0/LOG1.
`results-nano/native-packets.json`: 30 fresh-process probes, three repetitions per
fixture and representation. Median peak RSS and elapsed seconds:

| Fixture | Clipped retained MiB | Direct retained MiB | Clipped RSS MiB | Direct RSS MiB | Clipped seconds | Direct seconds |
|---|---:|---:|---:|---:|---:|---:|
| narrow | 10.0000 | 2.0000 | 13.1133 | 11.2500 | 0.073139 | 0.063907 |
| sparse | 12.0899 | 4.0843 | 16.2617 | 11.2500 | 0.080605 | 0.071224 |
| dense | 17.0875 | 6.9606 | 22.2422 | 11.6133 | 0.276462 | 0.254331 |
| wide | 17.0875 | 4.8336 | 22.2422 | 11.2500 | 0.295756 | 0.200147 |
| carry | 27.7223 | 19.7223 | 33.7227 | 25.7227 | 0.663819 | 0.644372 |

These are the existing **clipped-main/full-correction versus direct-all**
representation probe, with its scalar bridge and reconstruction scratch still
present. They are neither selective-policy parity nor fused-emitter or Gemmini
timing results. The probe uses stripe32, DIM32, and the generated software header
ACC_ROWS=1024. Hardware schedule parity is not claimed.

The fresh `results-nano-baseline/test512-original_a4nks.json` completed 8 chunks,
2,040 scored tokens: PPL **35.378030265292224**, top1 **0.3475490196078431**,
peak RSS **913,309,696 bytes**. Token SHA256 matches the historical run.
The historical macOS PPL was 35.37806; cross-platform bitwise equality is not
claimed. This is the unchanged original-policy NumPy forward, not optimized
product/native inference or an accuracy-equivalence pass for a new implementation.

## Further multiplication-reduction candidates

The user asked whether the many digit products can be reduced. Reused the existing
`profile_attention.prepare` / `profile_gemm.unpack` on the saved context512 capture
(144 heads), verified its SHA against the existing report, and matched the old
fragment totals before comparing schedules. Results and digit histograms are in
[optimization-opportunities.json](results-nano-profile/optimization-opportunities.json).
This is one-chunk component analysis, not a new PPL comparison or native timing.

The supplied cost formula uses the stripe-wide union of active `(lane,row)` pairs
for every surviving K block. Counting only pairs active in each original BK32
block gives `sum_b ceil(m_b/32)*ceil(N/32)*ceil(|C_b|/32)`, with the same codes,
scales, original block boundaries and column unions:

| Component | Reference schedule | Block-local candidate |
| --- | ---: | ---: |
| Q upper fragments, one K pass | 147,456 | 147,168 |
| P upper fragments | 44,724 | 13,782 |
| Total QK / full main | 6.000 | 5.9921875 |
| Total PV / full main | 1.6066081 | 1.1869303 |

P upper work falls 69.1843% and total PV fragments fall 26.1220% in this analytical
comparison. Nine selected heads / 108 stripe upper sums were recomputed with
block-local packed dots and scatter, after the existing int64 bound check; their
integer sums before float conversion matched the original lane loops exactly.
Main and zero-point terms were unchanged. This does not certify native execution,
the full graph, PPL or latency. Gather/call overhead can offset fragment savings.

**Existing implementation to reuse:** `RmdBitmapBuilder::finish` already creates
per-block lane groups/row IDs, and `rmd-executor.cpp` has a compact-dot path using
those row IDs. Its run-aware path instead builds the global row union. The
candidate is to preserve/use the fitting existing path during A4 attention
integration, not add another compactor or claim the current native product is
69% slower. Nominal spec counters and actual submitted counts must remain
separate if this schedule is adopted; the existing acceptance contract is not
silently changed by this experiment.

The later user decision selects fully masked QK tile skipping as the next
implementation scope; only its investigation is complete (see the final section).
At 512 tokens/DIM32, ordinary causal prefill retains 136 of 256 square tiles
including the diagonal. Sparse sign/add handling of upper digits remains an
experimental candidate, evaluated below. In this capture
97.7855% of nonzero Q lane-2 digits are +/-1; nonzero P lane-2 digits are all +1.
These identities can replace multiplications with additions on a suitable host
or vector path, but current signed-4 GEMM hardware does not automatically gain
cycles from them. Transfer/merge costs and actual RTL cycles need measurement.
For PV mask skipping, adjust the weighted zero-point column-sum domain too;
otherwise dropping main=-8 terms breaks cancellation. Lower precision/limb caps
remain unadopted because they change reconstructed operands.

## IM2P build correction and timing authority

**Subsequent raw-provider admission:** the selected manifest and native link now
pass. The corrected `im2p_raw_probe.cpp` passes all 15 independent integer-dot
checks (K1/K31/K32, row/column tails, three repetitions, normal and unit digits),
and returns integrated RTL cycles. The previous pointer-only public API call
worked for K1 but failed for K31: in `sim/src/simulator/matmul.rs`, absent scale
metadata causes `scale_block_size=K`, whereas the HP1 raw runtime requires 1.
The existing product `Im2pCompactDot` provider path preserves explicit block_size;
the probe now uses that same public provider route with checked weight/output
callbacks. No dummy scale buffers or simulator source changes were added.

For 32x32x32, both ordinary signed-nibble inputs and zero/+1/-1 inputs take
**492 RTL cycles** in each of three repetitions. A change in coefficient values
alone does not reduce Gemmini latency. CPU transfer/packing/merge, concurrent
scheduling and a full native a4nks execution remain separate requirements.
Evidence is in `results-unit-add-llama512/rtl-raw.json`. The earlier construction
history below remains useful; its former pending raw-admission gate is now met.

The first provisioning command incorrectly passed the header repository as
`GEMMINI_ROOT`; the cache fingerprint requires the llama source repository there.
The corrected arguments use separate `GEMMINI_ROOT` and `GEMMINI_PARAMS_ROOT`.

The next attempt placed artifacts inside the llama Git repository. IM2P's
`gemmini_vendor.py` calls `git -C OVERLAY apply PATCH`; inside a parent Git
repository this returned success but **skipped all four patch paths**. A verbose
check reproduced `Skipped patch` for GemminiConfigs, LoadController, LoopMatmul,
and StoreController. The resulting unpatched `inputType.getWidth / 8` caused
division by zero for A4. This was a build-location problem, not proof of missing
Gemmini A4 support. The failed logs are under `build-potal-hp1-diagnostic/`.

The corrected numerical artifact build uses `/tmp/potal-nano-im2p-artifacts`,
outside any Git repository. Patched `dma_maxbytes * 8 / inputType.getWidth` was
inspected in the actual generated overlay. A4/W4/DIM32 RTL elaboration succeeded.
At that stage, manifest, native-link, and raw numerical execution checks were
still required; the later compatible provider-path run above completes them.

That build subsequently failed in the C++ frontend because current IM2P checkout
`5da6a4c` still consumes retired native `q8_h1`/`q4_h1` descriptor fields. Develop
retired those contracts in `8a38141`; its commit message explicitly names the
matching IM2P commit **`20b82f058aa8b0e69c48d1d00790d8e19eaa6f66`**. That commit
exists locally and descends from the installed IM2P source through its merge.
The compatible checkout is isolated at `/tmp/potal-nano-im2p-compatible`, leaving
the original IM2P checkout untouched. Its numerical artifacts are built at
`/tmp/potal-nano-hp1-matched`, again outside all Git roots. The failed intermediate
compiler log is `/tmp/potal-nano-im2p-build.log` and the corrected build log is
`/tmp/potal-nano-hp1-matched-build.log`. Do not add retired types or dummy descriptor
padding to make an incompatible frontend compile.

The compatible build has now completed: release compilation finished in
28m59s and selected `GEMMINI_HP1/a4-w4-d32/current` contains both
`libim2p_gemmini_frontend.a`, `libim2p_sim.a` and `real-lib.json`.
This was build evidence only. The subsequent manifest/native-link admission and
raw-provider probe described above now supply numerical and RTL-cycle evidence.

The native cycle predictor is separately built in `/tmp/potal-nano-cycle-model`.
Its CTest passed 16/16. `results-nano/cycle-scaled-smoke.json` binds the built
library hash and resolved geometry `(4,4096,512)` to a 2x3x32 planner-block request;
it returned 331 logical cycles for the predictor's HP1 scaled-work contract. This
is a diagnostic estimate, **not raw-op0 timing parity or whole-model latency**.

The numerical runtime (`sim/backends/gemmini_hp1/runtime.cpp`) admits op0/BYPASS
only with block_size=1 and K<=32. Its elapsed-cycle authority is integrated RTL.
The separate `sim/cycle/` predictor consumes value-free geometry and explicit
backing-memory timing. `CYCLE_SIM=1` in llama selects CPU-functional numerical
execution and trace collection; it does not link or invoke that predictor online.
Existing scaled RMD/SCU certificates cannot certify a new raw-op0/host-exact merge
path. No frequency or application latency is inferred from these cycle counts.

`im2p_raw_probe.cpp` is a real-runtime gate: it verifies implementation, ABI5,
A4/W4/DIM32/ACC512, runs K1/K31/K32 with row/column tails against an independent
integer dot oracle, and requires reported RTL cycles. It offers no CPU fallback.
Configure the probe project with the verified `IM2P_SELECTED_GENERATION` and
`IM2P_SIM_ROOT` to build it. Passing this gate does not imply product QK/PV dispatch.

## Unit-add candidate: native Llama numerical validation complete

The user's candidate 3 is evaluated without dropping or clipping digits. Upper
lanes >=2 have three CPU implementations: dense multiplication, sparse
multiplication, and the same sparse metadata with +/-1 replaced by add/subtract.
Other coefficients are multiplied exactly. Keeping the sparse-multiply control
separates zero skipping from the effect of the unit specialization itself.

Reused owners: `native_model.cpp` retains llama.cpp's model/tokenizer/RoPE/GQA/
mask/softmax/linear execution and exposes the existing evaluation callback;
`profile_ops.py` supplies the original Q/K/P/V policy; `profile_gemm.gemm` accepts
a dot callable; `native_measure.py attention` drives paired evaluations and
component timings. This is an experimental output-replacement seam, not product
attention dispatch. The original floating QK/PV dots still execute, so replay
wall time is diagnostic and must not be called integrated backend speedup.

The input is a public FP16 conversion of Llama-3.2-1B, repository
`am00r/Llama-3.2-1B-GGUF`, revision `8e3690a604735388daca0ca1d51f6c92e932e698`.
Its 2,479,591,584-byte file matches the Hub's SHA256
`92cce0cbec5e029be0742a1dfff93254da3393ff4e47f5c07c6539de2dc445d8`.
The current-source quantizer produced Q4_HP1 while preserving F16 embedding and
output tensors; provenance is under `models/llama3.2-1B/` and the quantizer log
is `/tmp/potal-llama-quantize.log`.

GPT-2 and Llama context32 smoke runs pass 36 independent dot comparisons and
bitwise packet/scale/attention/logits/token-NLL comparisons. Llama's diagnostic
one-chunk PPL is 43.818175515399254 for both modes; this short-context smoke is
not the requested 10-chunk result. Its nonzero Q lane-2 unit fraction is 92.3894%,
and P lane-2 digits are all +1. These are context32 observations only.

The existing CPU tile OpenMP option reduced Llama smoke replay time from about
132 to 21 seconds, with attention/logit/NLL hashes unchanged. This is a test
runtime configuration, not a gain attributed to the unit-add candidate. The
serial build and prior outlier measurements are preserved. Context512 x10
paired evaluation completed in `results-unit-add-llama512/`; runtime libraries,
configuration and source hashes are recorded in `runtime-inputs.json`.
Both dense and sparse-unit-add PPL are **21.486062983190273**, with 5,120 input
tokens and 2,550 scored tokens. Packets/scales/attention/logits and every token
NLL are bitwise identical. Each mode evaluates 5,120 QK and 5,120 PV heads.
This proves the incremental arithmetic replacement in this native callback
experiment, not original-spec native linear or complete Gemmini integration.
The executed runner is archived as `native_measure.executed.py`; its SHA256
matches the pre-run input record. A later shared-helper extraction also passed
the prior context32 smoke's packet/attention/logit/token-NLL hashes exactly.

On the actual Llama context512 layer0/head0 P/V capture, dropping fully future
DIM32 main products while retaining the all-token zero-point correction gives
max absolute output error **2.91015625**. Restricting the weighted correction to
the exact retained key domain restores bitwise equality (error 0). The retained
domain extends through the end of the query tile, so masked entries inside the
diagonal tile still cancel. This is an analytical component check, not adopted
PV dispatch; see `results-unit-add-llama512/mask-domain-check.json`.

The same check now passes bitwise on all 15 selected P/V heads (layers
0/5/8/11/15, heads 0/16/31). The wrong all-token correction gives maximum
absolute errors from 2.28515625 to 29.828125 across these heads; the matching
retained-domain correction gives zero on each. The unmodified stripe scales,
K/V block exponents and residual digits are used in both calculations.
See `results-unit-add-llama512/mask-sample-check.json`.

For clarity, let H[i,j] mark the key positions whose GEMM tiles are retained,
D_s be the diagonal matrix of original P stripe scales, and Vhat the original
weighted reconstruction of V. With M=lo-8 and R=q-lo, the identity is
`O = D_s ((H*M + H*R) Vhat + 8 H Vhat)`, where `*` is elementwise.
H may include masked positions inside diagonal tiles: M=-8 and R=0 there,
so the compensation must include those same positions. Retaining `8 J Vhat`
instead (J all ones) adds the erroneous term `8 D_s (J-H) Vhat`.

Aligned equal-size causal prefill gives block-lower-triangular P. Strictly
upper blocks contain only masked entries, lower blocks are fully valid, and
diagonal blocks contain both. Each retained QK entry is an independent channel
dot; future entries affect neither softmax numerator nor denominator. For
prefix/decode, sequence identity and absolute query/key positions define the
boundary. In `src/llama-graph.cpp`, the ordinary QK node does not receive the
mask that is supplied to softmax; the existing flash-attention route already
receives it. A future skip implementation must reuse this actual mask meaning,
preserve original query IDs through residual row compaction, and keep original
quantization stripe groups. No skip implementation exists in this snapshot.
The current CPU softmax (`ggml-cpu/ops.cpp`) copies and scales the entire score
row before adding the mask. Merely omitting future QK stores can therefore read
uninitialized values: NaN plus -inf stays NaN. A separate-node implementation
must write safe finite placeholders for skipped scores, or reuse a fused path
that never reads those entries. This is another integration cost beyond fewer
GEMM submissions.

Thirty chunk-0 QK/PV samples also separate work reductions. Among Q upper
lanes >=2 nonzeros, 90.6364% are +/-1. Relative to all unpadded QK scalar dot
products, zero skipping removes 29.7893% and unit specialization replaces
another 3.2738% with adds/subtracts. For PV those shares are 4.1712% and
0.0734%. These are contribution counts, not latency estimates. Future-tile
skipping overlaps some zero elimination; the two savings must not be added.
See `candidate-work-counts.json` and `mask-sample-check.json` in the same result
directory. Full 10-chunk histograms are now in `report.json`: Q lane-2 has
18,819,606 nonzero digits, of which 92.2348853% are +/-1; Q lane-3 has 993 nonzero
digits, all +/-1. P lane-2 has 1,023,054 nonzero digits, all +1. These counts
describe this checkpoint/input/policy, not a guarantee for arbitrary models.

## CUDA component comparison and CPU candidate decision

The user's subsequent CUDA request is tested in the existing `native_measure.py`
owner. `CpuDot` and `prepare_head` are shared with the native accuracy replay;
`CudaDot` reuses cuBLAS `cublasGemmEx/CUBLAS_COMPUTE_32I` and supplies int64 host
partials to the unchanged checked SCU merge. Signed nibble values are stored in
int8 operands. This is not a CUDA PoTal model port or a new quantization policy.
The GGML CUDA tree has no current Q4_HP1/Q8_HP1/EXSIA implementation to substitute
for the existing native linear path automatically.

Hardware: Orin (nvgpu), SM8.7, driver 595.78; CUDA runtime/driver APIs 13020,
cuBLAS version 130401. Shapes use DIM32 zero padding and column-major TN layout.
All 84 independent CPU/CUDA raw-dot comparisons passed, including nibble values,
zeros/units, tails and cached padding reuse. All 30 saved Llama QK/PV heads match
the independent integer GEMM in float64 and the published float32 output exactly.

CPU/CUDA measurements ran together after the full numerical replay and earlier
CUDA functional check had exited. Each saved head/mode has one warmup and five
measurements with rotated mode order. The table sums per-head medians; common
quantization preparation is excluded, while conversion/padding/copies/device
synchronization/original host SCU and PV zero point are included.

| Mode | 15 QK heads | 15 PV heads | Combined |
| --- | ---: | ---: | ---: |
| CPU dense | 830.984 ms | 339.361 ms | 1170.345 ms |
| CPU sparse multiply, upper lanes >=2 | 735.405 ms | 336.155 ms | 1071.560 ms |
| CPU sparse unit add, upper lanes >=2 | 733.686 ms | 333.380 ms | 1067.066 ms |
| CUDA dense per K32 dot | 911.831 ms | 607.669 ms | 1519.500 ms |

Zero skipping plus unit handling reduces the combined component time by 8.82%,
but specializing +/-1 adds only 0.42% over the sparse-multiply control. Median
within-capture repeat ranges are 2.23% and 1.31% for those CPU modes. This does
not establish an additional performance benefit from unit specialization alone.
The existing fixed-shape Gemmini raw kernel also takes 492 cycles for ordinary
and unit digits alike. Keep these as measured candidates, not adopted product
defaults or a claimed integrated speedup.

Per-dot CUDA is **29.83% slower** than dense CPU here (QK +9.73%, PV +79.06%).
For example, layer0/head0 needs 96 GPU calls and 14,352,384 copied bytes for QK,
and 108 calls and 3,594,240 copied bytes for PV. Retained operand/output buffers
are 256,000 bytes each on host and device across tested shapes, excluding CUDA/
cuBLAS internal memory and common host outputs. Copies and synchronization occur
for every small dot, and SCU remains on the CPU. GPU-resident batched products
and combined SCU/quantization are plausible follow-ups, not measured gains.

In the complete dense diagnostic replay, tracked attention preparation/execution
accounts for 24.87% of wall time; raw dot kernel work is 5.31%, and raw-dot calls
including conversion/wrapping are 9.28%. The remainder includes native work,
the original floating attention still executed by the callback seam, and other
untracked overhead. Do not label the whole remainder as measured linear time or
infer production model speedup from these diagnostic fractions.

Use `cuda-component-timing.json` and `component-summary.json` for performance.
The original `report.json` CPU component timings overlapped the first CUDA
functional run and are superseded for timing comparisons; its completed
10-chunk numerical/PPL evidence remains valid. No full-model CUDA PPL or
integrated Gemmini unit-add/causal-skip implementation is claimed.

## QK fully masked tile skip investigation — implementation pending

This section preserves the nano checkpoint at `63e166d`. The local CPU replay
implementation and measurements are recorded at the top of this document;
product Gemmini/IM2P integration remains pending.

This is the final investigation state before the 2026-10-12 transfer. The user
selected **QK-only** omission of physical output tiles for which every actual
mask entry is negative infinity. No implementation, new PPL pair or skip timing
has been run. Q/K quantization, original stripe partitions, all required digits,
valid inner products and FP softmax must retain their existing values. PV remains
unchanged; this is not a FlashAttention-2 integration.

1. **Use the actual mask and its graph lifetime.** In `src/llama-graph.cpp`,
   `llm_graph_input_attn_kv_unified::set_input` tests sequence membership and absolute
   query/key positions (around lines 439–447), with additional SWA and padding
   rules. The ordinary QK `ggml_mul_mat` at line 1248 has no mask input; the
   mask reaches `ggml_soft_max_ext` at line 1275. The layer's selected mask must
   become an explicit, invocation-correct dependency before the backend can
   omit work. A cached pointer or an assumed `key_index > query_index` rule is
   insufficient for reused graphs, prefixes, cache layouts or multiple sequences.
   Finite negative bias (including ALiBi) is not an all-masked entry.
2. **Keep original row identities after RMD packing.** Existing
   `ggml/src/ggml-gemmini/residual/rmd/rmd-types.hpp` owns `row_ids` and
   `row_begin`. Map packed rows back through their original query rows before
   looking up the mask. For example, original query positions `[0,100]` cannot
   be treated as compressed positions `[0,1]`: keys 32–63 are valid for query
   100. QK output columns are key tokens; residual `C_b` columns are reduction
   channels, so the same index does not describe both axes.
3. **Quantize before choosing execution tiles.** `profile_ops.py::stripe_rows`
   depends on the original GEMM dimensions, including N. Cropping a future tile
   before quantization can change stripe boundaries and E2. Preserve whole
   original packets/scales and apply the mask decision to execution only. Main,
   all Q residual lanes, and both K nibble passes need the same valid-output
   domain, with the original-row mapping applied to each submitted packet.
4. **Publish safe skipped scores.** The current CPU softmax in
   `ggml/src/ggml-cpu/ops.cpp` copies/scales the score row before adding the mask.
   Skipping a store into uninitialized memory can introduce NaNs even when the
   mask is -inf. A separate QK node must publish finite placeholders such as
   zero for omitted entries. Fully masked rows are a separate existing boundary
   (`assert(sum > 0.0)` around line 4763), not something this optimization may
   silently redefine.
5. **Reuse the existing graph and dispatch owners.** The generic Gemmini matmul
   gate in `ggml/src/ggml-gemmini/ops.cpp` around line 557 restricts stationary
   head/batch dimensions and layout. It does not already admit arbitrary Llama
   attention heads/views. Product a4nks role dispatch is still absent. Do not
   merely relax the generic weight gate or build a duplicate executor and claim
   attention integration.
6. **Adjust the future validation gate, not the policy.** The current callback
   runner hashes all QK entries. A correct skip may replace only masked entries
   with zeros, so its future gate must compare full operand packets/scales and
   every valid QK entry, then the entire P, PV, logits and token NLL. A matching
   10-chunk PPL pair remains required. The completed unit-add pair above does
   not validate tile skipping. `native_model.cpp` currently clears KV and starts
   at position zero for each single-sequence chunk; prefix/decode, multiple
   sequences, GQA, tails, changed masks and graph reuse still need coverage.
7. **Keep work counts distinct from latency.** Ordinary aligned T512/DIM32
   causal prefill retains 136 of 256 square main tiles: 46.875% fewer main tiles.
   Residual row packing requires its own submitted-work count. Mask inspection,
   gathers, calls and score initialization have costs; no skip latency has
   been measured. Single-token decode over valid cached keys does not have this
   half-matrix saving. PV skipping also needs a matching weighted zero-point
   correction domain; the existing component checks establish that algebra,
   not an adopted PV backend. Online softmax would additionally need to preserve
   C4's normalized-P stripe maximum before it could replace the present path.

## Transfer checkpoint: completed and remaining

**Completed and preserved:** adopted A4 source-zero population fix and its
full/tail/reference regression cases; native GPT-2 before/after 10-chunk result;
Linux RSS unit correction; partial independent scalar oracle; compatible real
IM2P build/admission and 15 raw-dot checks; separate cycle-predictor diagnostics;
Llama native-callback unit-add 10-chunk bitwise comparison; CPU/CUDA component
comparison; specification/reuse audit and the QK investigation above. Evidence
and its limitations are in the preceding sections and result directories.

**Remaining:** complete A–D primitive/linear/masked-attention oracle coverage;
native original-geometry selective low-digit emission without the dense scalar
residual bridge; actual Q4_HP1 scale decoding/full-GEMM parity; checked/exact-wide
merge and final rounding; product Q/K/P/V dispatch with views/cache/GQA/decode;
the selected QK-only skip implementation and paired correctness/work/timing
measurement; raw-op0 predictor validation and full-path Gemmini measurement.
Unit-add, block-local packing, PV skipping and GPU-resident fusion are not
adopted product defaults. See `IMPLEMENTATION_PLAN.md` for the existing owners
and per-task gates. No complete product task is newly marked finished here.

The transfer itself only updates records, inventories existing files and
preserves them. Historical result JSON and logs have not been overwritten with
new measurements. Small results/validation logs accompany the source commit;
captures, token dumps and large logs are delivered in the separate archive.
The archive also preserves older local cycle logs under an explicitly historical
category; they are not evidence for this PoTal implementation.

[`results-nano/transfer-20261012/provenance.json`](results-nano/transfer-20261012/provenance.json)
records current file/hash checks, dependency commits, resolved IM2P generation,
artifact hashes and CMake settings. Historical commands are in `commands.json`;
per-file sizes, hashes, categories and Git/archive selection are in
`inventory.json`. Hashes first collected during transfer are labelled as such.
The native outlier PPL pair did not save a separate token-ID hash/dump, and the
older NumPy baseline did not bind a complete interpreter/library closure; these
gaps remain explicit. Model weights, caches and compiled executables/libraries
are excluded. Configuration/manifest snapshots are evidence, not bundled builds.

The transfer archive is
`/home/dyndnn/Projects/llama.cpp-gemmini/output/potal-nano-transfer-20261012.tar.gz`
(278,803,401 bytes; 409 members). SHA256:
`458d8ec4eec37419d9615f3be02de8657fe30fa6555f7e5e3e2c305c608e3cc4`.
Every archived member's size and SHA256 was checked after compression. Details
are in [`archive.json`](results-nano/transfer-20261012/archive.json); its adjacent
`.sha256` sidecar is outside Git alongside the archive.

| Preserved archive category | Files | Uncompressed bytes |
| --- | ---: | ---: |
| Current PoTal measurements, including captures/token dumps | 207 | 92,225,254 |
| Older local measurements, not revalidated | 139 | 2,928,587,549 |
| Historical build/validation logs | 24 | 5,666,852 |
| Historical analysis scripts/results | 3 | 16,606 |
| Failed-build diagnostic metadata | 7 | 47,623 |
| Build configuration metadata | 11 | 202,444 |
| WikiText input text | 1 | 1,290,590 |
| Transfer metadata and copied validation logs | 16 | 287,088 |

The additional archive-root `inventory.json` describes these 408 payload files.
`workspace/` contains repository-relative paths; `supplemental/` contains
historical external evidence. Extract into a temporary directory first, then
restore only the needed `workspace/` paths into the local checkout. The bundle
includes the small measurement results as well as the Git-excluded data.
Models, model caches, compiled `build-*` outputs, IM2P libraries/caches,
`.parity` binaries/GGUF fixtures, generated mock build commands, and raw Codex
session/account files are neither committed nor included in this archive.
Original local files remain in place.

Transfer checks: existing result JSON/JSONL parsed successfully; historical
hash expectations for the checked models, inputs, binaries and selected IM2P
artifacts matched; the source/document diff passed whitespace validation. No
build, test suite, PPL replay or performance experiment was started for transfer.
The experimental `.gitattributes` preserves this snapshot's raw measurement
bytes, including CSV line endings and whitespace in logs/diff evidence. Source
and handoff documents retain the normal whitespace checks.

## 2026-10-13: upper-lane specialization closed; retain NPU execution

The user excluded the zero/±1 specialization from adoption and selected NPU
operand reuse and tile omission as the optimization direction. Existing
activation main/residual generation, stacking, PE execution and accumulation
must remain. No product execution code was changed in this comparison.

Before that decision, the existing CPU runner completed GPT-2 context512 x10:
dense and exact sparse-unit-add PPL both 34.50487565853101, with bitwise packet,
scale, attention, softmax, logits and token-NLL equality. The historical Nano
Llama x10 pair remains 21.486062983190273 both ways; it was rechecked from saved
records, not rerun locally. New local component checks reused 24 GPT-2 and 30
Llama captures and the existing quantization/GEMM/measurement helpers. All
matched independent reconstructed float64 products and integer raw-dot results.
Adding unit specialization to zero skipping did not establish an extra gain:
the sums of per-capture medians were 424.513 -> 443.214 ms for GPT-2 and
483.143 -> 490.183 ms for Llama. These are CPU measurements, not NPU cycles.

Full x10 nonzero Q lane-2 digits were ±1 in 97.556% of GPT-2 and 92.235% of
historical Nano Llama observations. K's high nibble was ±1 in only 43.151% and
43.535% of the respective local saved-head samples. P lane 2 is exactly 0/+1
for all 256 uint8 codes; V has one nibble pass. A deliberately lossy sign-only
negative control changed valid QK outputs in both models. It was never used
in the exact candidate or PPL replay.

Evidence: [GPT-2 x10](results-upper-compare-gpt2-10chunk/report.json) and
[comparison, histograms and individual timings](results-upper-comparison/comparison.json).
The adjacent `compare.py` reuses the existing helpers and preserves the capture
and source identities. Actual A4/W4/DIM32 NPU latency for this candidate was
not measured. Further unit-specialization experiments are stopped. Reuse/tile
work must preserve the original codes/scales and be assessed through actual
NPU submissions, transfers and cycles; PV omission also requires a matching
zero-point correction domain.
