# Phase 2B test readiness

Baseline: `build-npu/checkpoints/README.txt` identifies the already verified
`phase2a-complete` checkpoint. It was not recreated. Phase 2A logs and the
per-test ledger were copied before reruns to `build-npu/phase2b-evidence/pre/`.
No commit, index update, or production API change accompanies this repair.

## Stripe lifecycle

Before: FAIL, CTest exit 8, in `pre/stripe-d32.ctest.log`.
The Phase 2A diagnostic and the new reproduction identify the same combined
ROLES assertions. The old lifecycle predicate queried `/proc/self/task/<tid>`
after the thread owners had joined. Procfs directory disappearance is not
the thread join contract, and numeric thread identifiers may be reused.

The test now records a shared exit token from each observed thread. Its
thread-local destructor publishes termination with release ordering; the
post-run predicate loads each token with acquire ordering. The normal
production and test-owner joins remain unchanged. Tokens survive thread
destruction and do not rely on Linux task-directory teardown.
The predicate rejects an empty observation set and any live observed thread.
No sleeps, retries, success-path logging, numeric assertion changes, or
scheduling changes were introduced.

After: PASS, build and CTest exit 0, in
`build-npu/phase2b-evidence/readiness-stripe.{build,ctest}.log`.
This result is the separate test-repair baseline, before common extraction.

## INT4 attention fixture

Before: FAIL, build exit 1, in `pre/int4-attention.build.log`.
The existing fixture assumes the INT8 attention contract: five radix-256
digits, unsigned 16-bit probabilities, an INT32 column bias, a planned-stripe
callback, and hardware bias injection via the removed `run_corrected`.
Its `support/attention-golden.hpp` also implements those INT8 semantics.

The current INT4 production contract in `matmul/attention.hpp` uses nine
radix-16 digits, unsigned eight-bit probabilities, INT64 CPU column bias,
and `paired_runner`. It explicitly refuses the legacy PV algorithm.
Changing only vector types and callback signatures would compile a fixture
whose oracle and many numeric expectations describe a different algorithm.
Reintroducing the removed production interface would violate this phase's
contract.

Status: BLOCKED pending a complete assertion-preserving migration of this
fixture to the supported INT4 contract. The test and its independent oracle
remain intact; no test is disabled or removed and no golden is regenerated.
The attention registration was already disabled as w8-only; that existing
policy is preserved. Common extraction may proceed independently, as permitted
by directive section 6.3. This is not a claim of complete INT4 attention regression.

## Bounded INT4 migration attempt

After the common extraction gates passed, a separate scratch change unit tested
an independent division/remainder radix-16 oracle written from the public
attention contract and Sat32 partial/accumulator rule. It does not call or copy
the production numerical implementation to derive expected results.

`build-npu/phase2b-evidence/attention-int4-attempt.cpp` builds and passes 12
QK/PV scalar cases (K=1/32/64, carriers 0/20), including INT32 extremes and
INT64 outputs. Nine further paired/strict/failed-publication/refusal checks pass.
Logs: `attention-int4-attempt.build.log` and `attention-int4-attempt.log` in that
evidence directory. These are scratch results, not a repaired CTest fixture.

The attempt exposed an important refusal distinction: execute/execute_strict
reject legacy as INVALID_ARGUMENT at attention_contract; prepare_stream(PV)
returns UNSUPPORTED and retains its prior output. The prototype was corrected
to test those actual contracts, rather than changing production.

The complete migration is still BLOCKED: the prototype does not establish
assertion-preserving counterparts for all 181 old check call sites. In
particular it does not replace the independent strided/broadcast preparation
oracle, removed bitplane/certificate witnesses, all raw ACC/command captures,
or all fatal cleanup cases. No replacement of the source fixture was retained;
the attempt was confined to scratch evidence. No assertion, test registration,
tolerance, production API, or golden fixture was removed or weakened.

The existing INT8-semantics attention-golden.hpp must remain: the INT4-tree
attention benchmark, smoke fixture, and support/attention-emulator.hpp still
consume it in addition to the original attention test. Their migration is not
claimed here. The scratch subset is not a substitute for those consumers.

The following ledger covers every original check call site, using line numbers
from phase2a-complete. Full expressions are retained in
`build-npu/phase2b-evidence/attention-assertion-gaps.json`. All rows remain
NOT MIGRATED; partial prototype evidence is stated without upgrading a row to
PASS.

| Original line | Assertion expression/name | Counterpart or precise remaining gap |
| --- | --- | --- |
| 152 | `name+" independent golden fits INT64"` | Independent radix-16 scalar oracle passes 12 cases; original prepared fixtures and all exact vectors remain unmigrated. |
| 154 | `name+" reference status"` | Independent radix-16 scalar oracle passes 12 cases; original prepared fixtures and all exact vectors remain unmigrated. |
| 155 | `name+" independent integer values"` | Independent radix-16 scalar oracle passes 12 cases; original prepared fixtures and all exact vectors remain unmigrated. |
| 162 | `"INT32_MAX uses fifth Q limb"` | Fifth-byte/16-bit-probability witnesses need ninth-nibble/u8 counterparts; overflow, carry, bias cancellation and rejection vectors remain unmigrated. |
| 163 | `"INT32_MIN stays exact"` | Fifth-byte/16-bit-probability witnesses need ninth-nibble/u8 counterparts; overflow, carry, bias cancellation and rejection vectors remain unmigrated. |
| 184 | `"QK independent low/high signed raw ACCs"` | Fifth-byte/16-bit-probability witnesses need ninth-nibble/u8 counterparts; overflow, carry, bias cancellation and rejection vectors remain unmigrated. |
| 187 | `"QK independent and legacy witnesses evaluate"` | Fifth-byte/16-bit-probability witnesses need ninth-nibble/u8 counterparts; overflow, carry, bias cancellation and rejection vectors remain unmigrated. |
| 189 | `"QK independent passes reject old shared result"` | Fifth-byte/16-bit-probability witnesses need ninth-nibble/u8 counterparts; overflow, carry, bias cancellation and rejection vectors remain unmigrated. |
| 196 | `"QK oracle derives Q digits from original codes"` | Fifth-byte/16-bit-probability witnesses need ninth-nibble/u8 counterparts; overflow, carry, bias cancellation and rejection vectors remain unmigrated. |
| 202 | `"P upper 255 has signed digit and carry"` | Fifth-byte/16-bit-probability witnesses need ninth-nibble/u8 counterparts; overflow, carry, bias cancellation and rejection vectors remain unmigrated. |
| 204 | `"P low uses unsigned byte minus 128"` | Fifth-byte/16-bit-probability witnesses need ninth-nibble/u8 counterparts; overflow, carry, bias cancellation and rejection vectors remain unmigrated. |
| 211 | `"PV exact block bias cancels before INT32 check"` | Fifth-byte/16-bit-probability witnesses need ninth-nibble/u8 counterparts; overflow, carry, bias cancellation and rejection vectors remain unmigrated. |
| 212 | `"PV carrier24 independent cancellation expects zero"` | Fifth-byte/16-bit-probability witnesses need ninth-nibble/u8 counterparts; overflow, carry, bias cancellation and rejection vectors remain unmigrated. |
| 228 | `"PV exact cancellation beyond INT128 width"` | Fifth-byte/16-bit-probability witnesses need ninth-nibble/u8 counterparts; overflow, carry, bias cancellation and rejection vectors remain unmigrated. |
| 231 | `"PV noncancelling wide bias explicitly unsupported"` | Fifth-byte/16-bit-probability witnesses need ninth-nibble/u8 counterparts; overflow, carry, bias cancellation and rejection vectors remain unmigrated. |
| 235 | `"PV outside-INT32 exact bias rejected by golden"` | Fifth-byte/16-bit-probability witnesses need ninth-nibble/u8 counterparts; overflow, carry, bias cancellation and rejection vectors remain unmigrated. |
| 236 | `"PV outside-INT32 bias now selects exact bitplanes"` | Fifth-byte/16-bit-probability witnesses need ninth-nibble/u8 counterparts; overflow, carry, bias cancellation and rejection vectors remain unmigrated. |
| 241 | `"PV negative outside-INT32 bias now selects exact bitplanes"` | Fifth-byte/16-bit-probability witnesses need ninth-nibble/u8 counterparts; overflow, carry, bias cancellation and rejection vectors remain unmigrated. |
| 245 | `"PV exact INT32_MIN bias remains supported"` | Fifth-byte/16-bit-probability witnesses need ninth-nibble/u8 counterparts; overflow, carry, bias cancellation and rejection vectors remain unmigrated. |
| 253 | `"PV mixed saturation independent expected minus one"` | Fifth-byte/16-bit-probability witnesses need ninth-nibble/u8 counterparts; overflow, carry, bias cancellation and rejection vectors remain unmigrated. |
| 263 | `"PV repeated mixed signs expose fixed microtile order"` | Fifth-byte/16-bit-probability witnesses need ninth-nibble/u8 counterparts; overflow, carry, bias cancellation and rejection vectors remain unmigrated. |
| 267 | `"PV reversed repeated signs have different saturation"` | Fifth-byte/16-bit-probability witnesses need ninth-nibble/u8 counterparts; overflow, carry, bias cancellation and rejection vectors remain unmigrated. |
| 280 | `"PV positive final correction saturates exactly once"` | Fifth-byte/16-bit-probability witnesses need ninth-nibble/u8 counterparts; overflow, carry, bias cancellation and rejection vectors remain unmigrated. |
| 285 | `"PV negative final correction saturates exactly once"` | Fifth-byte/16-bit-probability witnesses need ninth-nibble/u8 counterparts; overflow, carry, bias cancellation and rejection vectors remain unmigrated. |
| 303 | `"independent completed-limb reconstruction overflows"` | Fifth-byte/16-bit-probability witnesses need ninth-nibble/u8 counterparts; overflow, carry, bias cancellation and rejection vectors remain unmigrated. |
| 304 | `"reference rejects INT64 overflow"` | Fifth-byte/16-bit-probability witnesses need ninth-nibble/u8 counterparts; overflow, carry, bias cancellation and rejection vectors remain unmigrated. |
| 305 | `"overflow publishes no partial reference output"` | Fifth-byte/16-bit-probability witnesses need ninth-nibble/u8 counterparts; overflow, carry, bias cancellation and rejection vectors remain unmigrated. |
| 318 | `"QK later cancellation would fit final INT64"` | Fifth-byte/16-bit-probability witnesses need ninth-nibble/u8 counterparts; overflow, carry, bias cancellation and rejection vectors remain unmigrated. |
| 321 | `"QK intermediate overflow rejects before later cancellation"` | Fifth-byte/16-bit-probability witnesses need ninth-nibble/u8 counterparts; overflow, carry, bias cancellation and rejection vectors remain unmigrated. |
| 324 | `"QK reference intermediate overflow publishes nothing"` | Fifth-byte/16-bit-probability witnesses need ninth-nibble/u8 counterparts; overflow, carry, bias cancellation and rejection vectors remain unmigrated. |
| 368 | `"prepare K strided head/batch"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 369 | `"K dimensions and block bases"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 371 | `"K exact block exponent differences"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 378 | `"K independently calculated signed limbs and padding"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 380 | `"prepare V signed8"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 385 | `"V independently calculated signed8 and padding"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 392 | `"prepare simple Q"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 398 | `"Q simple F16/F32 scale, codes and limbs"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 401 | `"prepare zero stationary"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 402 | `"zero blocks have zero carriers"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 407 | `"prepare zero stream"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 408 | `"zero Q codes and padding"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 416 | `"prepare Q with three block exponents"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 422 | `"Q anchor follows GGML_GEMMINI_EXSIA_STRIPE_ANCHOR=" + std::to_string(GGML_GEMMINI_EXSIA_STRIPE_ANCHOR)` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 433 | `"set caller rounding mode"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 436 | `"P quantization restores caller rounding"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 437 | `"prepare P ties"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 438 | `"P ties round half to even, unsigned upper carry"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 449 | `"Q quantization restores caller rounding"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 450 | `"prepare Q ties"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 451 | `"Q signed ties round half to even"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 455 | `"restore original rounding mode"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 459 | `"prepare P unsigned16 clipping"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 460 | `"P clips RNE overflow to 65535 with upper carry"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 471 | `"prepare K clipping"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 472 | `"K asymmetric clipping endpoints"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 476 | `"prepare V clipping"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 477 | `"V signed8 endpoints"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 484 | `"P negative/nonfinite rejected"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 485 | `"failed P prepare restores rounding"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 486 | `"failed P prepare leaves output unchanged"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 493 | `"nonfinite stationary rejected"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 494 | `"failed stationary prepare publishes nothing"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 496 | `"nonfinite Q rejected"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 497 | `"failed Q prepare publishes nothing"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 500 | `"zero requested rows rejected"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 501 | `"out-of-range stream row rejected"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 502 | `"out-of-range head rejected"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 503 | `"out-of-range batch rejected"` | Independent expected INT4 preparation for F16/F32 strided/broadcast tensors, finite checks, rounding modes, Q/K scaling and residual policy is not yet mapped. |
| 581 | `"execute contract accepts strided/repeated graph"` | Paired runner passes exact power-of-two QK/PV probes; original tails, heads, gaps and command-accounting fixtures remain unmigrated. |
| 585 | `name+" emulator execution"` | Paired runner passes exact power-of-two QK/PV probes; original tails, heads, gaps and command-accounting fixtures remain unmigrated. |
| 596 | `name+" independent values across all heads/batches"` | Paired runner passes exact power-of-two QK/PV probes; original tails, heads, gaps and command-accounting fixtures remain unmigrated. |
| 597 | `name+" independent low/high descriptors and frozen tiles"` | Paired runner passes exact power-of-two QK/PV probes; original tails, heads, gaps and command-accounting fixtures remain unmigrated. |
| 599 | `name+" QK has no PV correction"` | Paired runner passes exact power-of-two QK/PV probes; original tails, heads, gaps and command-accounting fixtures remain unmigrated. |
| 600 | `name+" balanced scales and no live DMA"` | Paired runner passes exact power-of-two QK/PV probes; original tails, heads, gaps and command-accounting fixtures remain unmigrated. |
| 602 | `name+" returned command counts"` | Paired runner passes exact power-of-two QK/PV probes; original tails, heads, gaps and command-accounting fixtures remain unmigrated. |
| 605 | `name+" actual LOOP command count"` | Paired runner passes exact power-of-two QK/PV probes; original tails, heads, gaps and command-accounting fixtures remain unmigrated. |
| 619 | `"QK leaves causal mask to the later CPU softmax"` | Raw low/high ACC snapshots, corruptions, later-pass failures, overflow publication and mask/broadcast cases still need paired-request counterparts. |
| 632 | `"zero-block execute status"` | Raw low/high ACC snapshots, corruptions, later-pass failures, overflow publication and mask/broadcast cases still need paired-request counterparts. |
| 636 | `"zero-block output is zero"` | Raw low/high ACC snapshots, corruptions, later-pass failures, overflow publication and mask/broadcast cases still need paired-request counterparts. |
| 651 | `"emulator injected failure propagates"` | Raw low/high ACC snapshots, corruptions, later-pass failures, overflow publication and mask/broadcast cases still need paired-request counterparts. |
| 652 | `"emulator failure publishes no partial output"` | Raw low/high ACC snapshots, corruptions, later-pass failures, overflow publication and mask/broadcast cases still need paired-request counterparts. |
| 667 | `"failure between low/high preserves status and completed count"` | Raw low/high ACC snapshots, corruptions, later-pass failures, overflow publication and mask/broadcast cases still need paired-request counterparts. |
| 671 | `"later-job failure publishes no earlier results"` | Raw low/high ACC snapshots, corruptions, later-pass failures, overflow publication and mask/broadcast cases still need paired-request counterparts. |
| 680 | `"fatal high completion preserves fence failure and stops"` | Raw low/high ACC snapshots, corruptions, later-pass failures, overflow publication and mask/broadcast cases still need paired-request counterparts. |
| 683 | `"fatal high completion retains pending scales and sticky owner"` | Raw low/high ACC snapshots, corruptions, later-pass failures, overflow publication and mask/broadcast cases still need paired-request counterparts. |
| 686 | `"fatal high completion publishes no output"` | Raw low/high ACC snapshots, corruptions, later-pass failures, overflow publication and mask/broadcast cases still need paired-request counterparts. |
| 707 | `"QK execute saturation witness fits"` | Raw low/high ACC snapshots, corruptions, later-pass failures, overflow publication and mask/broadcast cases still need paired-request counterparts. |
| 709 | `"QK execute exact independent and obsolete shared witnesses"` | Raw low/high ACC snapshots, corruptions, later-pass failures, overflow publication and mask/broadcast cases still need paired-request counterparts. |
| 712 | `"QK execute independent active-limb raw ACC witnesses"` | Raw low/high ACC snapshots, corruptions, later-pass failures, overflow publication and mask/broadcast cases still need paired-request counterparts. |
| 728 | `"QK execute independent-pass saturation result"` | Raw low/high ACC snapshots, corruptions, later-pass failures, overflow publication and mask/broadcast cases still need paired-request counterparts. |
| 733 | `"QK execute every completed raw pass matches independent oracle"` | Raw low/high ACC snapshots, corruptions, later-pass failures, overflow publication and mask/broadcast cases still need paired-request counterparts. |
| 735 | `"QK execute saturation panels and balanced scales"` | Raw low/high ACC snapshots, corruptions, later-pass failures, overflow publication and mask/broadcast cases still need paired-request counterparts. |
| 753 | `"QK execute overflow fixture preserves folded codes and Delta"` | Raw low/high ACC snapshots, corruptions, later-pass failures, overflow publication and mask/broadcast cases still need paired-request counterparts. |
| 757 | `"QK execute overflow is independently required"` | Raw low/high ACC snapshots, corruptions, later-pass failures, overflow publication and mask/broadcast cases still need paired-request counterparts. |
| 764 | `"QK execute rejects overflow after completed passes"` | Raw low/high ACC snapshots, corruptions, later-pass failures, overflow publication and mask/broadcast cases still need paired-request counterparts. |
| 767 | `"QK execute overflow publishes nothing and leaves no DMA"` | Raw low/high ACC snapshots, corruptions, later-pass failures, overflow publication and mask/broadcast cases still need paired-request counterparts. |
| 809 | `name + " status"` | Removed INT32 bias-MVIN/run_corrected and u16-P expectations require CPU-INT64-bias checks or explicit legacy refusals; original cases remain unmigrated. |
| 817 | `name + " independent full output"` | Removed INT32 bias-MVIN/run_corrected and u16-P expectations require CPU-INT64-bias checks or explicit legacy refusals; original cases remain unmigrated. |
| 818 | `name + " unsigned upper carry uses separate jobs"` | Removed INT32 bias-MVIN/run_corrected and u16-P expectations require CPU-INT64-bias checks or explicit legacy refusals; original cases remain unmigrated. |
| 819 | `name + " exact once-only bias operand"` | Removed INT32 bias-MVIN/run_corrected and u16-P expectations require CPU-INT64-bias checks or explicit legacy refusals; original cases remain unmigrated. |
| 821 | `name + " real checked commands interpreted"` | Removed INT32 bias-MVIN/run_corrected and u16-P expectations require CPU-INT64-bias checks or explicit legacy refusals; original cases remain unmigrated. |
| 823 | `name + " total command counts"` | Removed INT32 bias-MVIN/run_corrected and u16-P expectations require CPU-INT64-bias checks or explicit legacy refusals; original cases remain unmigrated. |
| 827 | `name + " old representable bias never selects bitplanes"` | Removed INT32 bias-MVIN/run_corrected and u16-P expectations require CPU-INT64-bias checks or explicit legacy refusals; original cases remain unmigrated. |
| 842 | `"PV failed bias MVIN propagates without partial publication"` | Removed INT32 bias-MVIN/run_corrected and u16-P expectations require CPU-INT64-bias checks or explicit legacy refusals; original cases remain unmigrated. |
| 845 | `"PV successful cleanup releases accelerator ownership"` | Removed INT32 bias-MVIN/run_corrected and u16-P expectations require CPU-INT64-bias checks or explicit legacy refusals; original cases remain unmigrated. |
| 846 | `"PV cleanup session closes"` | Removed INT32 bias-MVIN/run_corrected and u16-P expectations require CPU-INT64-bias checks or explicit legacy refusals; original cases remain unmigrated. |
| 853 | `name+" original-code plane oracle"` | Old u16 bitplane/certificate oracle has no direct INT4 API counterpart; per-case refused-contract or radix-16 witness remains unmapped. |
| 864 | `name+" unsigned digits reconstruct original codes"` | Old u16 bitplane/certificate oracle has no direct INT4 API counterpart; per-case refused-contract or radix-16 witness remains unmapped. |
| 868 | `name+" certificate refuses before publication"` | Old u16 bitplane/certificate oracle has no direct INT4 API counterpart; per-case refused-contract or radix-16 witness remains unmapped. |
| 874 | `name+" certificate and independent wide dot"` | Old u16 bitplane/certificate oracle has no direct INT4 API counterpart; per-case refused-contract or radix-16 witness remains unmapped. |
| 878 | `name+" raw planes reconstruct exact wide quantized dot"` | Old u16 bitplane/certificate oracle has no direct INT4 API counterpart; per-case refused-contract or radix-16 witness remains unmapped. |
| 879 | `name+" production reference equals independent planes"` | Old u16 bitplane/certificate oracle has no direct INT4 API counterpart; per-case refused-contract or radix-16 witness remains unmapped. |
| 896 | `"PV zero codes retain one unsigned plane"` | Old bitplane exactness/capacity/cancellation fixtures need supported u8/radix-16 counterparts; independent boundary oracle remains incomplete. |
| 908 | `"PV largest positive multiple128 bias endpoint"` | Old bitplane exactness/capacity/cancellation fixtures need supported u8/radix-16 counterparts; independent boundary oracle remains incomplete. |
| 913 | `"PV exact negative bias endpoint"` | Old bitplane exactness/capacity/cancellation fixtures need supported u8/radix-16 counterparts; independent boundary oracle remains incomplete. |
| 931 | `"PV boundary independent balanced candidate and minimal unsigned planes"` | Old bitplane exactness/capacity/cancellation fixtures need supported u8/radix-16 counterparts; independent boundary oracle remains incomplete. |
| 942 | `"PV unrestricted balanced256 counterexample"` | Old bitplane exactness/capacity/cancellation fixtures need supported u8/radix-16 counterparts; independent boundary oracle remains incomplete. |
| 977 | `"PV large carrier exact zero has no undefined shift"` | Old bitplane exactness/capacity/cancellation fixtures need supported u8/radix-16 counterparts; independent boundary oracle remains incomplete. |
| 983 | `"PV ZERO raw plane outputs"` | Old bitplane exactness/capacity/cancellation fixtures need supported u8/radix-16 counterparts; independent boundary oracle remains incomplete. |
| 991 | `"PV independent wide cancellation beyond carrier width"` | Old bitplane exactness/capacity/cancellation fixtures need supported u8/radix-16 counterparts; independent boundary oracle remains incomplete. |
| 994 | `"PV exact wide near-limit cancellation remains representable"` | Old bitplane exactness/capacity/cancellation fixtures need supported u8/radix-16 counterparts; independent boundary oracle remains incomplete. |
| 998 | `"PV wide cross-block bias cancellation remains old"` | Old bitplane exactness/capacity/cancellation fixtures need supported u8/radix-16 counterparts; independent boundary oracle remains incomplete. |
| 1009 | `"PV invalid original codes/metadata reject atomically"` | Old bitplane exactness/capacity/cancellation fixtures need supported u8/radix-16 counterparts; independent boundary oracle remains incomplete. |
| 1053 | `"PV fixture stripe selection"` | Old planned-stripe descriptors and hardware bias injection cannot be adapted by a signature cast; per-case paired request/output evidence remains absent. |
| 1056 | `"PV real V preparation"` | Old planned-stripe descriptors and hardware bias injection cannot be adapted by a signature cast; per-case paired request/output evidence remains absent. |
| 1060 | `"PV real P preparation"` | Old planned-stripe descriptors and hardware bias injection cannot be adapted by a signature cast; per-case paired request/output evidence remains absent. |
| 1064 | `"PV extension trigger is only strict column bias"` | Old planned-stripe descriptors and hardware bias injection cannot be adapted by a signature cast; per-case paired request/output evidence remains absent. |
| 1069 | `"PV prepared unsigned planes"` | Old planned-stripe descriptors and hardware bias injection cannot be adapted by a signature cast; per-case paired request/output evidence remains absent. |
| 1071 | `"PV rejected fixture requires interval refusal"` | Old planned-stripe descriptors and hardware bias injection cannot be adapted by a signature cast; per-case paired request/output evidence remains absent. |
| 1075 | `"PV accepted fixture independent certificate"` | Old planned-stripe descriptors and hardware bias injection cannot be adapted by a signature cast; per-case paired request/output evidence remains absent. |
| 1079 | `"PV accepted fixture exact wide integer equality"` | Old planned-stripe descriptors and hardware bias injection cannot be adapted by a signature cast; per-case paired request/output evidence remains absent. |
| 1086 | `"PV normalized guard fixture constants"` | Old planned-stripe descriptors and hardware bias injection cannot be adapted by a signature cast; per-case paired request/output evidence remains absent. |
| 1099 | `"PV normalized exact output is signed 2^-16"` | Old planned-stripe descriptors and hardware bias injection cannot be adapted by a signature cast; per-case paired request/output evidence remains absent. |
| 1102 | `"PV old path independent arithmetic"` | Old planned-stripe descriptors and hardware bias injection cannot be adapted by a signature cast; per-case paired request/output evidence remains absent. |
| 1170 | `name+" refuses before any command or output"` | Old planned-stripe descriptors and hardware bias injection cannot be adapted by a signature cast; per-case paired request/output evidence remains absent. |
| 1183 | `name+" bitwise F32 independent output"` | Old planned-stripe descriptors and hardware bias injection cannot be adapted by a signature cast; per-case paired request/output evidence remains absent. |
| 1184 | `name+" raw completed command C and descriptors"` | Old planned-stripe descriptors and hardware bias injection cannot be adapted by a signature cast; per-case paired request/output evidence remains absent. |
| 1185 | `name+" selects only required stripes and completed planes"` | Old planned-stripe descriptors and hardware bias injection cannot be adapted by a signature cast; per-case paired request/output evidence remains absent. |
| 1190 | `name+" historical guard class preserves first eleven heads"` | Old planned-stripe descriptors and hardware bias injection cannot be adapted by a signature cast; per-case paired request/output evidence remains absent. |
| 1193 | `name+" no corrected runner or bias MVIN"` | Old planned-stripe descriptors and hardware bias injection cannot be adapted by a signature cast; per-case paired request/output evidence remains absent. |
| 1195 | `name+" balanced scales and clean DMA"` | Old planned-stripe descriptors and hardware bias injection cannot be adapted by a signature cast; per-case paired request/output evidence remains absent. |
| 1236 | `"PV later-plane failure preserves completed jobs and publication"` | Old fallback/bitplane dispatch cases require explicit unsupported-contract checks or supported signed-digit counterparts; not yet migrated. |
| 1241 | `"PV fatal plane retains unknown DMA and Scale ownership"` | Old fallback/bitplane dispatch cases require explicit unsupported-contract checks or supported signed-digit counterparts; not yet migrated. |
| 1246 | `"PV recoverable plane failure leaves completed DMA clean"` | Old fallback/bitplane dispatch cases require explicit unsupported-contract checks or supported signed-digit counterparts; not yet migrated. |
| 1385 | `"B actual production all65536 prepare"` | Signed-digit preparation must compare independent u8/radix-16 digits and metadata for every original input/stride case; not yet migrated. |
| 1402 | `"B all65536 unchanged codes independent digits carry32640 zero positions padding"` | Signed-digit preparation must compare independent u8/radix-16 digits and metadata for every original input/stride case; not yet migrated. |
| 1404 | `"default PV preparation selects balanced encoding"` | Signed-digit preparation must compare independent u8/radix-16 digits and metadata for every original input/stride case; not yet migrated. |
| 1408 | `"B allzero minimum one limb"` | Signed-digit preparation must compare independent u8/radix-16 digits and metadata for every original input/stride case; not yet migrated. |
| 1413 | `"QK ignores PV selector including encoded stream tag"` | Signed-digit preparation must compare independent u8/radix-16 digits and metadata for every original input/stride case; not yet migrated. |
| 1484 | `"B stripe plan"` | Small exact paired outputs pass; original balanced metadata, command counts, no-live-DMA and overflow cases remain unmigrated. |
| 1492 | `name+" V prepare"` | Small exact paired outputs pass; original balanced metadata, command counts, no-live-DMA and overflow cases remain unmigrated. |
| 1515 | `name+" P prepare"` | Small exact paired outputs pass; original balanced metadata, command counts, no-live-DMA and overflow cases remain unmigrated. |
| 1529 | `name+" independent B/C representable"` | Small exact paired outputs pass; original balanced metadata, command counts, no-live-DMA and overflow cases remain unmigrated. |
| 1578 | `name+" independent prepared metadata and nonsat equality"` | Small exact paired outputs pass; original balanced metadata, command counts, no-live-DMA and overflow cases remain unmigrated. |
| 1579 | `name+" actual multiple production stripes"` | Small exact paired outputs pass; original balanced metadata, command counts, no-live-DMA and overflow cases remain unmigrated. |
| 1630 | `name+(checked ? " checked" : " planned")+" default execution=B observedINT32 compositionalINT64 exactF32"` | Small exact paired outputs pass; original balanced metadata, command counts, no-live-DMA and overflow cases remain unmigrated. |
| 1632 | `name+" B no corrected bias or bitplanes and fenced Scale completion"` | Small exact paired outputs pass; original balanced metadata, command counts, no-live-DMA and overflow cases remain unmigrated. |
| 1636 | `name+" actual multi K panel"` | Small exact paired outputs pass; original balanced metadata, command counts, no-live-DMA and overflow cases remain unmigrated. |
| 1664 | `name+" explicit legacy execution matches A status and values"` | Small exact paired outputs pass; original balanced metadata, command counts, no-live-DMA and overflow cases remain unmigrated. |
| 1708 | `"B between-limb failure seam "+std::to_string(seam)+" no partial publication"` | Legacy execute refusal is proven (INVALID_ARGUMENT, attention_contract, zero calls); original randomized balanced cases remain unmigrated. |
| 1712 | `"B safe between-limb refusal leaves completed DMA clean"` | Legacy execute refusal is proven (INVALID_ARGUMENT, attention_contract, zero calls); original randomized balanced cases remain unmigrated. |
| 1715 | `"B failed fence retains unknown DMA ownership without retry"` | Legacy execute refusal is proven (INVALID_ARGUMENT, attention_contract, zero calls); original randomized balanced cases remain unmigrated. |
| 1723 | `"metadata role QK"` | Original backend shape/type/data/refusal matrix still needs assertion-by-assertion migration; production API was not changed. |
| 1725 | `"metadata role PV"` | Original backend shape/type/data/refusal matrix still needs assertion-by-assertion migration; production API was not changed. |
| 1726 | `"null role"` | Original backend shape/type/data/refusal matrix still needs assertion-by-assertion migration; production API was not changed. |
| 1728 | `name+" rejected"` | Original backend shape/type/data/refusal matrix still needs assertion-by-assertion migration; production API was not changed. |
| 1735 | `name+" no command or partial output"` | Original backend shape/type/data/refusal matrix still needs assertion-by-assertion migration; production API was not changed. |
| 1757 | `"metadata support check does not require data"` | Original backend shape/type/data/refusal matrix still needs assertion-by-assertion migration; production API was not changed. |
| 1777 | `"invalid P rejected before runner"` | Original backend shape/type/data/refusal matrix still needs assertion-by-assertion migration; production API was not changed. |
| 1850 | `name + " sequential reference executes"` | Strict power-of-two outputs and zero CPU work on issuer pass; original shape/broadcast cases, fatal failure and every count comparison remain unmigrated. |
| 1858 | `name + " executes"` | Strict power-of-two outputs and zero CPU work on issuer pass; original shape/broadcast cases, fatal failure and every count comparison remain unmigrated. |
| 1859 | `name + " bitwise equal to execute()"` | Strict power-of-two outputs and zero CPU work on issuer pass; original shape/broadcast cases, fatal failure and every count comparison remain unmigrated. |
| 1860 | `name + " same GEMMs and command counts"` | Strict power-of-two outputs and zero CPU work on issuer pass; original shape/broadcast cases, fatal failure and every count comparison remain unmigrated. |
| 1864 | `name + " no CPU numeric step on the owner thread"` | Strict power-of-two outputs and zero CPU work on issuer pass; original shape/broadcast cases, fatal failure and every count comparison remain unmigrated. |
| 1866 | `name + " balanced scales, no live DMA"` | Strict power-of-two outputs and zero CPU work on issuer pass; original shape/broadcast cases, fatal failure and every count comparison remain unmigrated. |
| 1877 | `name + (fatal ? " fatal" : " nonfatal") + " injected failure propagates, nothing published"` | Strict power-of-two outputs and zero CPU work on issuer pass; original shape/broadcast cases, fatal failure and every count comparison remain unmigrated. |
| 1895 | `"strict legacy PV refused before any command"` | Current execute/execute_strict reject legacy as INVALID_ARGUMENT; prepare_stream(PV) rejects it as UNSUPPORTED. Original unavailable-owner fixture remains unmigrated. |
| 1900 | `"strict unavailable owner refused, nothing published"` | Current execute/execute_strict reject legacy as INVALID_ARGUMENT; prepare_stream(PV) rejects it as UNSUPPORTED. Original unavailable-owner fixture remains unmigrated. |
| 1908 | `"GGML initialization"` | Scratch initialization passes; original test entrypoint stays intact and unbuildable until complete fixture migration. |
| 1909 | `"DIM16/DIM32 attention contract"` | Scratch initialization passes; original test entrypoint stays intact and unbuildable until complete fixture migration. |
