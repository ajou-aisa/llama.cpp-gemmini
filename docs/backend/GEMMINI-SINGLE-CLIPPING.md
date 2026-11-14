# Original-source stripe quantization

ExSIA dense activation preparation applies the final stripe scale directly to
the original F32 values. This avoids passing materialized block integers,
including their rounding and saturation losses, through a second rescale.

The final conversion is:

```
retained = saturate_int32(round_ties_even(double(source) * 2^(-stripe_theta)))
```

The implementation uses exact power-of-two scaling in double over the
reachable F32/stripe exponent range. It does not change the caller's floating
point rounding mode. Nonfinite inputs and absent scales retain the existing
zero behavior.

## Preserved numerical behavior

The exponent/sigma analysis still chooses masks and block exponents. TOP2 and
TOP3 retain their existing distinct-exponent selection and fallback rules.
The dense bit-width clamp, selected residual decomposition, and final scales
are unchanged. Residuals never recover magnitude discarded by final clipping.

"Single clipping" names the removal of the intermediate block conversion.
It does not mean that every clamp in the backend has been removed: final
INT32 retention and dense bit-width limiting remain.

## Runtime policy

`GGML_GEMMINI_EXSIA_DIRECT_STRIPE` selects the policy at each ExSIA invocation:

| Value | Behavior |
| --- | --- |
| Unset or `1` | Original F32 to final stripe RN conversion |
| `0` | Previous materialized block RN followed by integer stripe rescaling |
| Other present values | Invalid-input diagnostic and failed invocation |

The policy applies to sequential, local-parallel and local/folding-pipeline
dense preparation, for both supported stripe anchors. The direct path retains
preliminary sigma integers only in worker scratch and does not allocate or
fill slot-wide block integer outputs or replay final block quantization.

Materialized Local consumers, including attention preparation, retain their
existing block-code contract. No CUDA kernel or attention algorithm change
is part of this feature.

## Coverage and evaluation boundary

The `direct-stripe-rn` test case drives the real producer and public activation
dispatch against an independent original-input oracle. It covers both
anchors, both policies and the default, signed ties, clipping, intermediate
block loss, tails and logical source stride, slot reuse, execution modes,
packet/direct residuals, malformed policies and recovery.

An earlier isolated CUDA evaluation measured 21 chunks / 5355 scored targets
per arm using the same quantization policy:

| Model | Bits | Anchor | Block-derived PPL | Direct PPL |
| --- | --- | --- | ---: | ---: |
| Qwen2-0.5B | 4 | TOP2 | 25.55509 | 24.12284 |
| Qwen2-0.5B | 4 | TOP3 | 22.47427 | 21.81672 |
| Llama3.2-1B | 4 | TOP2 | 20.85489 | 19.63090 |
| Llama3.2-1B | 4 | TOP3 | 17.95201 | 17.13202 |
| Qwen2-0.5B | 8 | TOP2 | 14.83996 | 14.83496 |
| Qwen2-0.5B | 8 | TOP3 | 14.82715 | 14.82931 |
| Llama3.2-1B | 8 | TOP2 | 11.32597 | 11.31281 |
| Llama3.2-1B | 8 | TOP3 | 11.31905 | 11.32338 |

These are reference measurements from the isolated evaluation source, not a
new PPL run on this develop-based branch or FPGA measurements. W4 improved
in all four pairs; W8 effects were small and mixed. Shared-GPU elapsed times
do not isolate CPU quantization overhead or establish a CUDA speedup.
