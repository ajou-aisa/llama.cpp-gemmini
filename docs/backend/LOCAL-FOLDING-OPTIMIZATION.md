# Applied Local/Folding and KV preparation optimizations

Runtime baseline: `fc30c0a` on `hotfix/optimize-local-folding`, with the
subsequent single-query decode workspace change described below.
This branch starts at develop `1cf4697`. It changes `ggml-gemmini`, not
`ggml-potal`, and keeps the existing hardware operand widths and execution
order. It does not redistribute work across Core1/2/3.
Publication to the hotfix branch does not mean it was merged into develop.

Distinguish code inclusion, activation in the selected build, tested output
equivalence, and measured performance improvement. One does not establish
the others.

## Applied summary

Source paths below are relative to `ggml/src/ggml-gemmini/`.

| Optimization | Applied behavior | Main source |
| --- | --- | --- |
| QKV/Gate-Up preparation reuse | Conditional reuse of compatible Local/Folding results; activation guards are listed below | `quants/act/preparation-cache.cpp`, `ops.cpp` |
| Incremental KV preparation | Retains develop's reuse of unchanged K tokens and V blocks, subject to cache lifecycle checks | `matmul/attention.cpp`, `matmul/attention-stationary.cpp` |
| KV limb repacking reduction | K retains a geometric physical column pitch; V rewrites changed blocks. CPU, CUDA upload and IM2P consumers receive physical stride | `matmul/attention-stationary.*`, `matmul/attention.cpp`, `residual/rmd/rmd-im2p-executor.*` |
| Local bit-based exponent extraction and quantization | Replaces per-element ilogb/ldexp work while retaining ties-to-even and INT32 saturation | `quants/act/exsia/exsia_float.hpp`, `quants/act/exsia/local.cpp` |
| Attention output scale reuse | Computes scales once per column only when `st.rows > 1`; one-row decode keeps ldexp | `matmul/attention.cpp` |
| Block shift preparation | Computes shift constants once per folding block | `quants/act/exsia/exsia_shift.hpp`, `quants/act/exsia/folding.cpp` |
| Promoted-block mask updates | Updates selected valid columns by word-range OR | `quants/act/exsia/folding.cpp` |
| Residual early exit | Returns zero residual for in-range values; skips a zero low lane only when radix divisibility holds | `quants/act/exsia/folding.cpp`, `residual/rmd/rmd-bitmap-builder.cpp` |
| General builder/scratch reuse | Retains builder capacity and visits/clears touched bitmap words; completed packets keep ownership | `residual/residual-capture.hpp`, `residual/rmd/rmd-bitmap-builder.*` |
| Decode attention workspace reuse | Reuses bitmap capacity and builder scratch across heads/batches within one call when the query tensor has one row | `matmul/attention.cpp` |

The attention workspace is distinct from the general residual builder reuse
and from token-to-token KV caching. Eligibility is the query shape
`b.ne[1] == 1`, normally single-token decode. Multi-row calls, including a
one-row tail stripe within prefill, keep stripe-local storage.

Only working memory is reused. Each head's input is quantized again, the
selection bitmap is filled with its original valid pattern, and builder
reset clears previous metadata, digits and touched masks. Completed packets
retain independent ownership. The workspace ends with the attention call,
so nothing is shared between invocations or backend contexts. QK shift and
digit emission remain separate, with unchanged anchor/rounding/order.

### Decode workspace validation

At A4 and A8, 48 public attention-execution cases per width produced identical
raw output bytes and logical work counts before/after this change. Cases cover
QK/PV, query rows 1/3/65, K=33/65, N=5/67, four query heads, two KV heads, two
batches, alternating zero/residual-heavy inputs and repeated calls. Invalid
input leaves the previously published output unchanged. Retained packets also
remain unchanged after a shared workspace resets for a zero-input stripe.

Instrumented ordinary C++ allocation calls decreased in all 16 one-row cases
per width: 77-126 fewer at A4 and 77-128 fewer at A8. All 32 multi-row cases
per width had unchanged counts. These are allocation counts, not wall-time
or allocator-time measurements. Both widths also passed the four selected
integration tests and cached-pitch/masked-residual checks again.

An isolated A4 mutant that skipped reset after a nonempty builder failed with
invalid arguments during the public execution test. The normal source was not
modified for this mutation. Local evidence in the existing A4/A8 build roots:
`decode-workspace-tests.log`, `decode-workspace-pitched-mask.log`,
`decode-before.log`, `decode-after.log`, `decode-before.bin`, `decode-after.bin`,
and `decode-mutant.log` (A4). Test code and snapshots remain unpublished.

## Historical stream candidates

QK shift/digit fusion and attention-call bitmap/builder reuse were published
in `8166496`, then explicitly deferred by an earlier scope decision.
They were removed by published follow-up `fc30c0a`; published history is retained.
The current follow-up restores workspace reuse only for one-query-row calls;
QK fusion and multi-row workspace reuse remain excluded.
Their historical checks at each of A4/A8 covered 144 stream preparation cases
and 96 reconstructed outputs whose digests matched the frozen pre-change implementation.
The cases cover Top2/Top3, tail K, shrinking/growing stripe sizes, zero inputs,
wide Q values and retained-packet lifetime across workspace reset.
At that historical revision, preparation-reuse, projection-reuse and
attention-mask tests passed (3/3 at each width). The unchanged attention test still fails its initial
`wide right-shift boundaries` assertion: it expects -1 for
`shift_q_i32(INT32_MIN, -32)`, while the existing ties-even implementation
returns 0. The failing test was not altered or suppressed.
Logs are `omission-*.log` in the A4/A8 build directories listed below;
the local driver is `verify-omissions.sh` beside `sigma64-audit.md`.

The additional empty-mask block fast path, new K-index ctz enumeration,
single-loop replacement, stripe-wide guard, and naive bitmap assignment are
not included. Their supplied PC measurements did not establish consistent
improvement. K-index enumeration remains ascending; ctz would also enumerate
ascending indices, so memory-order changes are not a reason for this decision.
Existing row/digit ctz operations remain unchanged.

## Width and rounding contracts

The input is IEEE binary32. The bit-based quantizer reads its representation as
uint32 and computes a magnitude with uint64 intermediates. A significand has
at most 24 bits and a non-saturating left shift is bounded by 31, so this
intermediate needs fewer than 56 bits. PreparedShift's largest product is
2^31 * 2^31 = 2^62. Both fit uint64.

These are host intermediates, not wider accelerator operands. The quantizer
still saturates to INT32. Existing clipping/decomposition then supplies signed
4-bit or 8-bit values to the existing data path. No MAC width, accumulator
width, SCU saturation, scale field or checkpoint format is changed.

Right-shift rounding compares the discarded fraction with half a quantum.
An exact tie increments only an odd retained integer. The same rule applies
to negative magnitudes before restoring their sign: this is ties-to-even,
not ties-away. Subnormals, zeros, nonfinite inputs and INT32 limits have
explicit handling matching the existing scalar contract.

The older separate `potal-cpu-20261112` experiment preserved its own ties-away
policy. Its helper must not be copied into this ties-even develop branch.

For output scaling, the original operation is:

    float(ldexp(double(accumulator), exponent))

The multi-row path computes the positive double power-of-two factor once per
column and uses:

    float(double(accumulator) * factor)

Finite F32 operand scales keep the double factors and nonzero INT64 products
in the normal double range. The accumulator conversion, exact power-of-two
scaling and final F32 conversion retain the same value. One-row calls retain
direct ldexp. Floating-point exception-status flags are not claimed to be an
equivalence property.

## PV limbs are digits, not a loss of precision

For operand width b, let R=2^b. The current ggml-gemmini PV code is clipped
to 0 <= q <= R^2-1. Its existing low-digit bias representation is:

    d0 = (q mod R) - R/2
    u  = floor(q/R)
    d1 = u if u < R/2, otherwise u-R
    d2 = 0 if u < R/2, otherwise 1
    q  = R/2 + d0 + R*d1 + R^2*d2

All digits fit the existing signed b-bit range. For A8, 65535 is exactly
128 + 127 + 256*(-1) + 65536*1. For A4, 255 is exactly
8 + 7 + 16*(-1) + 256*1. No value is discarded.

This bound applies to PV's constrained codes, not arbitrary INT32 Q/folding
values. The generic residual builder retains its full lane capacity.
ggml-gemmini already emits only the required nonzero residual digits; this
branch does not import the separate ggml-potal five-array-to-three-array
storage change.

## Projection reuse conditions and lifetime

Reuse is bypassed at compile time if any of `CYCLE_SIM`,
`EXSIA_OBSERVATION_ENABLED`, `GGML_GEMMINI_ACT_QUANT_METRICS`,
`GGML_GEMMINI_RESIDUAL_METRICS`, or `GGML_GEMMINI_SCALE_METRICS` is enabled.
Runtime sensitivity profiling/shadow also bypasses it.
Do not assume reuse is active in FireSim from code presence alone: inspect the
actual build flags, runtime compatibility and reuse evidence.

The backend owns one bounded projection-family cache. A hit requires the same
layer/family, input tensor and data addresses, exact input bytes, I/K, stripe
row count, anchor, residual route and floating rounding mode. A projection
role cannot hit twice in the same family sequence.
`GGML_GEMMINI_EXSIA_FORCE_RECOMPUTE=1` bypasses
reuse. Unsupported layouts and observation/simulation modes use the original
preparation path.

Same X alone is insufficient. If different output widths select different
stripe extents, Q/K/V are prepared separately. If stripe extents match,
output-column-dependent residual metadata is copied and rebound for the
current J; original packets are not mutated.

FULL execution can share the owned main buffer. For streaming execution, the
collector may already own its main backing, so cached bytes are copied into
that backing before publishing stripes in their original order. Each reuse
gets a fresh run ID and does not report old Local/Folding work as newly run.
Sink failure clears the cache and propagates failure.

Input snapshots, shared payload lifetime, byte comparisons and rebinding have
costs. This is not a zero-memory or zero-overhead optimization.

## Baseline fairness and KV

Generic K/V buffer sharing is not added here. The existing CUDA baseline
implementation already reserves/reuses buffers and uses zero weight stride
to share a KV head across grouped-query GEMMs.

The current RTN/MXINT/NVFP4 evaluation paths use FP16 attention. They therefore
do not perform the PoTal KV quantization/folding that incremental preparation
avoids. Existing develop KV caching is retained. The latest scope additionally
includes physical-pitch limb repacking reduction, distinct from quantization
reuse; no cross-device cache is introduced. A future quantized-KV
baseline should receive the same opportunity to reuse unchanged blocks,
subject to its own scale axis and numerical schedule.

Projection preparation reuse can also be relevant to other quantized dense
methods. Report it as an implementation optimization, not as proof of an
exclusive PoTal algorithmic advantage or as a term that automatically cancels
out of latency ratios.

## Evidence boundaries

The main-checkout primitive comparison already passed A4/A8 quantizer,
prepared-shift, scale, bitmap and attention checks. Its Local+fold medians
fell from 34.752 to 25.277 ms (A4) and 28.631 to 22.953 ms (A8), while some
isolated bitmap/PV cases regressed. These are native synthetic preparation
measurements of that bundle, not this final branch's model TTFT/TPOT.

The supplied projection reuse report measured roughly 62% lower one-row QKV
preparation and 48-49% lower one-row Gate/Up preparation, including cache
lookup/copy costs. It also documents broad timing variance and bypass modes.
Those existing measurements are retained as bounded evidence; they are not
new Rocket measurements and are not added to other improvement percentages.

Local integration checks cover floating-rounding changes and forced
recomputation, plus actual projection output comparisons in FULL and
STRIPE_PIPELINE. The test build enables runtime matmul-mode override so those
two requested modes really execute. A test configuration with that override
disabled cannot establish environment-mode rejection behavior.

The initial projection-reuse A4 and A8 builds completed successfully. Each passed
`test-gemmini-preparation-reuse` and `test-gemmini-projection-reuse` (2/2).
Logs are under `/home/mojoi/potal-reviews/hotfix-local-folding-20261113/a4`
and `a8`, in `verification.log`. Existing pedantic warnings about `__int128`
remain. The final KV-repacking checks are listed below. Test sources, test-only
CMake changes, dumps, logs, baseline copies and mutation files stay local;
this document summarizes their evidence without publishing those artifacts.

The existing verified boundary and isolated-variant experiments are not
repeated just to reproduce their records. Full-model PPL and Rocket/FPGA
TTFT/TPOT have not been remeasured for these changes. Finite bitwise tests
must not be described as a measured 0% PPL loss.

## Subsequent preparation candidates

Sigma64, row writes and residual batching remain experimental and are not
included in this hotfix. The candidate report's Local baseline is byte-identical
to this hotfix's Local source. Its Folding baseline includes the excluded
empty-mask specialization, and its bitmap baseline includes the excluded
K-index ctz loop; whole-file imports would reintroduce those changes.

The bounded Sigma64 prototype reduces measured standalone Local time by
17-24%, but returns failure outside rho 0..14, block size <=128 and sigma=2.
Any future integration needs the existing production int128 path for other
settings. The supplied 8-bit sigma-only complete preparation median increased
from 17.800015 to 18.134690 ms (1.88%). Row writes regress on residual-heavy
4-bit prefill, and batching regresses on sparse input.

The combined sigma/row-write medians decrease by 6.67% and 6.30% at 4/8 bits.
Contrary to the original report's broad overlap statement, the three-process
median ranges overlap only at 8 bits, not at 4 bits. These small synthetic
host samples still do not establish model accuracy or Rocket TTFT/TPOT.
The audit and corrected verification reports are retained under
`C:/Users/mojoi/AppData/Local/Temp/potal-preparation-import-20261113/`
as `sigma64-audit.md` and `sigma64-verified.md`.

## Latest scope decision

| Item | Current decision |
| --- | --- |
| QKV/Gate-Up preparation reuse | Keep |
| Develop incremental KV preparation | Keep |
| Local bit-based exponent/quantization and output scales | Keep |
| Block shift/mask, residual early exit, general builder/scratch reuse | Keep |
| KV limb repacking reduction | Added; native A4/A8 integration checks passed |
| QK shift/digit fusion | Deferred; removed from default path in `fc30c0a` |
| Attention stripe bitmap/builder reuse | Reintroduced only for one-query-row decode; multi-row prefill reuse remains excluded |
| PV temporary arrays/intermediate passes | Reference worktree only; not imported |
| Sigma64 | Experimental; bounded guard and production wide fallback required |
| Folding row writes | Not included; input-dependent regressions |
| Residual block batching | Not included; sparse regressions |
| Single-loop folding | Excluded |
| Additional empty-mask specialization | Excluded |
| Additional K-index ctz enumeration | Excluded |
| Core redistribution | Excluded; stationary preparation still uses one worker |
| Additional K/V preparation buffer sharing | Excluded |
| New SiLU fusion or approximation | Not added |
| Direct Stripe Quantization | Excluded |
| Residual Scale command reuse | Conditional reuse already exists in an older NPU snapshot; not newly implemented or counted as a new gain here |
| Additional residual Index command omission | Deferred; missing command-payload and installation-state evidence |

KV repacking's reported 58-86% reduction is a native synthetic append-preparation
measurement from the reference worktree, not a new measurement of this hotfix.
Geometric K capacity increases host limb/carrier memory. NPU use requires
the pitched operand readers and column views to be validated with the real
provider; actual device validation and Rocket TTFT/TPOT remain unverified.

The independent stride audit found no omitted CPU/IM2P physical-pitch consumer.
CUDA upload validates spare physical columns as well as logical ones; current
producer padding contains valid codes/carriers, but this does not imply that
CUDA accepts poisoned padding as the IM2P reader test does. Optional CUDA
execution was not run. The configured SDK has no IM2P simulator headers or
archives, so provider pitch execution was not run either. A provider test's
zero exit alone would be insufficient because unsupported implementations
can skip its body; require the actual case PASS marker on the target.

Both A4/A8 native builds and the preparation-reuse, projection-reuse,
attention-cache and attention-mask tests passed (4/4 each). These are the four
selected tests, not the entire test suite. The cache test
compares F16/F32 cached QK/PV with independent uncached outputs through pitch
growth, partial/full blocks, changing V scales and cache invalidation.
Logs are `kv-repack-build.log` and `kv-repack-tests.log` in the existing A4/A8
build directories.

The added cached-pitch/masked-residual check also passed at A4/A8. Logical
N=65,66,129 uses physical pitch 96,96,192, with a live column view [32,64),
two residual submissions per case, and reused old KV blocks on append.
Cached and independent uncached output bytes and logical work counts match.
The initial fixture accidentally changed its storage strides with logical N,
causing legitimate cache invalidation; the fixture now retains the actual
fixed-capacity storage strides. No production behavior was changed for it.
Logs: `kv-pitched-mask.log`; local harness:
`tests/test-gemmini-attention-pitched-mask.cpp`.

An isolated A4 mutant changed only the CPU residual consumer's physical pitch
back to logical N. The new test exited 1 with
`pitched cached mask/residual output differs`; production remained unchanged.
Mutation logs: `kv-stride-mutant-build.log` and `kv-stride-mutant.log`.

## Later real-model-input replay

The separate experimental report `potal-dump-replay-results.md` in
`gemmini-ttft-develop-20261112/docs/` records 168 input tensors captured from
FP16 Qwen2-0.5B and Llama-3.2-1B. Capture used the first 128 input tokens,
two decode steps and three selected layers per model. These are real FP16
activations replayed through PoTal preparation on a PC, not activations
produced by running preceding layers in PoTal or a full-model accuracy test.

| Candidate | Observed preparation result | Current action |
| --- | --- | --- |
| Attention bitmap/builder workspace reuse | Decode QK 5.9-15.8% and PV 8.5-13.6% lower; prefill benefit unproven | Supports the single-query decode change; these are earlier replay measurements, not new target measurements of this commit |
| QK folding/digit fusion | No consistent gain; Qwen W4 decode 5.1% slower | Keep deferred |
| Sigma64 | Qwen prefill Local+Folding W4 3.1% and W8 6.7% slower | Do not enable by default |
| Residual Scale reuse | Older NPU source contains conditional `ktile_scale_reuse`; past `PAIRED_KTILE` setting was ON | Preserve existing behavior; do not count twice |
| Additional Index omission | No ordered payload/install-state evidence | Keep deferred |

The replay reports matching output digests for 636 initial comparison keys
and 72 paired attention keys. These checks cover the captured input set;
they are not an exhaustive proof or a full PPL/ACC comparison. Decode timing
aggregates the selected inputs and two steps, not end-to-end ms/token.
IQRs are not confidence intervals.

The older NPU Scale implementation does not establish the current remote
build's exact configuration. Safe additional Index omission needs command
payload, destination pair, generation, installation completion, invalidation
and accepted-command order. Write counts alone do not establish duplicates.

No Rocket TTFT/TPOT improvement or full-model PPL before/after equivalence
has been established for the complete optimization set. Do not sum preparation
speedup percentages or present finite output checks as measured 0% accuracy loss.

## Commit history

| Commit | Runtime change |
| --- | --- |
| `1cf4697` | Develop incremental KV preparation base |
| `3424a5a` | Local/Folding primitives, bit quantization and output scales |
| `9ed414a` | Compatible projection preparation reuse |
| `8166496` | Historical QK fusion and attention workspace implementation |
| `fc30c0a` | KV limb repacking/stride support; removes the two deferred stream changes |
