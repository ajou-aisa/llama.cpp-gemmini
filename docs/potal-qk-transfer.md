# Selective Q residuals with two stationary K digits

The main candidate uses an n-bit Q representation with selective compensation
and a 2n-bit K representation decomposed into two n-bit digits. W4 uses Q4/K8;
W8 uses Q8/K16. Both the ordinary and strict five-hart execution paths support
this policy. Reciprocal channel transfer is a separate ablation, not the design's
central claim.

Enable it on the existing PoTal attention run:

```sh
export POTAL_QK_POLICY=residual
# Run the usual PoTal executable and its existing attention-offload arguments.
# Optional channel-transfer ablation:
export POTAL_QK_POLICY=residual-transfer
```

Unset the variable or set it to `legacy` for the existing full-digit Q policy
(rho6 in W4, rho14 in W8). Other values are rejected for QK.
This is a quality/performance policy change,
not a bit-exact optimization of the previous quantizer.

## Design rationale

Q is streamed and its dynamic Local/folding/digit preparation repeats for each
stripe. K's two stationary digits are reused across query rows and grouped heads.
Reducing Q's base precision targets dynamic host preparation, while retaining
selected residuals for accuracy. Extra regular K-digit GEMMs remain on the NPU.
This motivates the asymmetric representation, but does not by itself prove a
speedup: padded residual work and preparation time must both be measured.

The transfer ablation aims to reduce quantization difficulty on Q by moving
channel range to K. It preserves the unquantized dot product, not its quantized
approximation. With quantization errors EQ/EK, the logit error is
`EQ * K'^T + Q' * EK^T + EQ * EK^T`; decreasing Q's error can increase K's.
Therefore neither improved PPL nor fewer residual positions is guaranteed.

## Calculation

The `residual` policy uses zero channel shifts. For `residual-transfer`, each KV
head and batch computes a channel exponent from the largest absolute
Q value across its grouped query heads/broadcast batches and the first
`min(128, query_rows)` rows. Let `d = -clamp(exponent, -16, 16)`; zero channels use 0.
Apply `Q'[i,c] = Q[i,c] * 2^d[c]` and `K'[j,c] = K[j,c] * 2^-d[c]` before quantization.
Thus `Q' K'^T = Q K^T` in exact arithmetic. Finite, round-trip-exact binary32
scaling is checked; unrepresentable scaling fails instead of silently clipping.

Q uses Local rho(n-2), the configured stripe anchor, and the existing fold rounding.
Only Local-selected positions and blocks above the anchor retain clipping
residuals. Other positions clamp to the signed n-bit range. K keeps block absmax
rho(2n-2): [-128,119] in W4 and [-32768,32639] in W8, represented by two signed
digits of the physical operand width. Main/residual GEMMs,
stationary-pass ordering, SCU/INT32 saturation and checked INT64 reconstruction
use the existing NPU runner. Existing PV preparation is unchanged (P8/V4 for W4,
P16/V8 for W8), as is FP16 KV storage. W8 retains its existing dense digit-job
lowering; this patch does not introduce W4's sparse packet scheduler into W8.

The plan is shared across all Q stripes using a stationary K. In the strict
pipeline Local worker 0 computes it before publishing the stripe to Local worker
1; the fold worker reads the same immutable plan. No CPU numerical work moves
onto the issuer. Plans live until all workers finish.

## Scope and evidence

The policy recomputes its plan for each QK invocation and prepares K for that
plan. It does not reuse previously transformed K across decode calls. The
128-row prefix is a runtime rule, not offline calibration; changing query
batching can change the plan. It does not establish token-wise causal quantizer
equivalence or prefill/decode equivalence. Scaling is applied to the supplied
post-RoPE operands, not folded into projection weights.

User-supplied full PPL results for the experimental implementation were
22.07 / 17.11 (Qwen / Llama), versus 22.19 / 17.53 without transfer and
21.61 / 16.18 for the old attention policy. Those are not a new full-PPL
measurement of this NPU port. Host validation compares independent quantization
goldens, the public software reference, the paired-command emulator, and the
strict multithreaded execution. It does not certify a live bitstream or report
Rocket TTFT/TPOT.
There is no new full-PPL or latency claim for the W8 candidate in this patch.

Local validation on 2026-11-16 passed all four W4 tests (legacy golden,
residual, residual-transfer, counters) and all four W8 tests (C API, legacy
attention, residual, residual-transfer). Coverage includes independent W4
integer goldens, a literal Q8 clipping/compensation fixture, command emulation,
strict-pipeline agreement, GQA and batch broadcast, partial blocks, prefix
boundaries, and rejection of nonrepresentable transfer values. Deliberately
restoring the legacy Q rho in W4 made both new policy tests fail; restoring the
implementation returned them to PASS.

Reproduce after building the corresponding test targets:

```sh
ctest --test-dir <w4-build> -R potal-w4a4 --output-on-failure
ctest --test-dir <w8-build> -R 'potal-(qk|attention)' --output-on-failure
```

## Related transformations

- [SmoothQuant](https://proceedings.mlr.press/v202/xiao23c.html): reciprocal
  channel scaling of activations and weights.
- [OmniQuant, Eq. 5](https://arxiv.org/html/2308.13137v3): learned equivalent
  scaling of Q and K before quantization.
- [QServe SmoothAttention, Sec. IV-B](https://arxiv.org/html/2405.04532v1):
  scales K down and Q up to protect quantized keys; this candidate uses the
  opposite direction to protect lower-precision Q. The underlying reciprocal
  transformation is prior art. Our policy uses runtime power-of-two scales and
  existing low-bit residual execution, not the complete recipe of those papers.
