# Gemmini ExSIA numeric contracts and fix scope

This note records source-level contracts and fix scope. It reports no build,
test, model, accuracy, or performance results.

## Numeric behavior

- **RN policy:** New builds use nearest, ties-to-even for initial activation
  integers and folding right shifts, independently of caller rounding mode.
  Half ties `+/-0.5` become zero, `+/-1.5` become `+/-2`, and `+/-2.5`
  become `+/-2`; adjacent values must not double-round. Scale selection,
  clipping, code ranges, radix compensation, saturation, reduction order,
  batching and datasets retain their existing definitions. See
  `ROUNDING_POLICY.md` for conversion owners and validation status.
- **RMD-off folding:** Outlier decomposition is gated by both the outlier mark
  and `GGML_GEMMINI_ENABLE_RMD`. Without RMD, values use signed activation
  saturation and leave no residual; they do not retain an uncompensated low
  limb. ( `quants/act/exsia/folding.cpp:139-168` )
- **Stripe scale:** `e_s` is selected from the stripe's top exponents: `e1`
  when alone, `e2` normally, or `e3` when the runtime anchor is 3 and present.
  Each block is aligned by `delta = e_b - e_s`; stripe scale is
  `theta = e_s - rho`. ( `folding.cpp:82-96,120-145`; `exsia_shift.hpp:8-14` )
- **Rho and attention:** Ordinary ExSIA uses `rho = activation_bits - 2`.
  Attention has a separate logical code width `code_bits = 2*pe_bits` and
  uses `rho = code_bits - 2`; it emits upper-limb compensation after extracting
  the transport-width low digit. Do not treat this as an ordinary-path rho
  override. ( `ggml-gemmini-config.hpp:144-145`; `types.hpp:68-78`;
  `matmul/attention.cpp:32-36,163-166,192-225` )
- **Anchor precedence:** The compile-time anchor initializes the runtime
  fallback. Runtime selection can override it: thread-local override, then
  per-key policy, then fallback. Folding consumes the selected args value.
  ( `ggml-gemmini-args.h:150`; `ops.cpp:984-985`;
  `potal-sensitivity.cpp:132-140`; `folding.cpp:89-94` )
- **Detector semantics:** BLOCK/RMD tests signed two-sided deviation from the
  signed mean. ExSIA tests integer `abs(q)` against its magnitude population
  mean, selecting only the high side. Their populations and detectors differ;
  neither implies a universal equivalence. ExSIA sigma is a positive integer;
  fractional values such as 2.5 are unsupported. ( `block/block.cpp:55-90`;
  `exsia/local.cpp:179-202,263-286`; `types.hpp:12-13`;
  `cmake/ggml-gemmini-options.cmake:190,219-227` )
- **Shift limits:** Right-shift rounding can discard low bits before residual
  extraction, so later residual capture cannot restore that loss. Positive
  shifts clamp to signed INT32; decomposition reconstructs that shifted,
  clamped integer, not an unclamped integer or the original floating-point
  activation. No universal exactness guarantee follows. ( `exsia_shift.hpp:17-46`;
  `folding.cpp:27-45,120-168`; `exsia.cpp:1864-1869` )

## Fixes and observability

- The RMD gate fixes the no-consumer case above. Runtime configuration and
  metrics implement their named contracts; this document makes no performance
  or accuracy claim.
- The intended profile counters `right_shift_elements`,
  `right_shift_zeroed_elements`, and `left_shift_saturated_elements` count,
  respectively, nonzero logical integer inputs shifted right, those shifted
  inputs that become zero, and inputs whose mathematical left shift exceeds
  signed INT32 range. Merely obtaining `INT32_MIN` or `INT32_MAX` without
  mathematical overflow does not count as saturation. Padding is excluded.
  Collection is gated by `EXSIA_STAGE_PROFILE_ENABLED`, which requires
  `CYCLE_DETAIL` and `GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE == 2`.
  Folding now collects these counters, the stage state carries them to the
  profile, and the profile emitter serializes them. This source observation
  does not establish executed counter tests, accuracy or performance.
- The dense validation plane is debug/validation-only. Production reconstruction
  routes through residual packets or direct payloads, according to the selected
  route. ( `exsia-state.hpp:590-594`; `exsia.cpp:1810-1832`;
  `matmul/execution.cpp:1682-1706` )

## Historical results and pending coverage

Newly compiled `POTAL_ANCHOR_POLICY_PROOF` reports `ties_even`. Frozen
ties-away DSOs, models, receipts, caches and results retain their old
identities. An RN runtime consuming ties-away MXINT weights is a mixed recipe;
full RN additionally requires newly RN-requantized weights and fresh model
and build fingerprints. Archived RNE filenames alone prove neither.

GPU numerical execution, integrated CPU/CUDA parity with RN dependencies,
RN checkpoint conversion and whole-model accuracy remain separate pending
acceptance work. Isolated compilation and historical ties-away results do
not close those checks. This document does not claim they were executed.

## References

Source references above are repository-relative and identify the contracts
covered here. This document makes no measured-accuracy or speedup claim.
