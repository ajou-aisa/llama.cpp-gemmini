# Supplied PoTal A4 specification: linear + a4nks attention

Source: the user's initial implementation specification and accompanying NumPy `potal.py` pasted in this conversation. This transcription preserves the A-D arithmetic contract so continuation does not require the chat transcript. The [implementation plan](IMPLEMENTATION_PLAN.md) specifies finite-input boundaries, exact accumulation and the accepted representation optimization. The supplied `sf_tradeoff.py` / `attn_quant.py` dependencies were not available; parity with those external modules has not been established.

## Constants and primitives

`n=4`, signed PE codes `[-8,7]`, `BK=DIM=32`, `tau=2`, exact-zero exponent `NEG`, distinct stripe anchor rank `E2`. Linear uses `rho=2`; Q uses `rho=6` on the same 4-bit PEs.

```text
eps(x)       = floor(log2(abs(x))) for nonzero x; NEG for zero
qround(x,t)  = round_half_even(x * 2^-t), saturated to int32
shift(q,D)   = D>0: sat_int32(q << D)
               D<0: sign(q) * ((abs(q) + 2^(-D-1)) >> -D)
               D=0: q
sat(u)       = clip(u,-8,7)
lowdigit(u)  = ((u+8) mod 16)-8
digits(r)    = repeatedly emit d=lowdigit(r), then r=(r-d)/16 until zero
               r_initial = sum_l d_l * 16^l; each digit is in [-8,7]
SCU(acc,part,s) = acc += part * 2^s
```

`stripe_rows(I,N,K,DIM,n)` is the GEMM tile-height schedule from scratchpad/accumulator geometry. The supplied Python memory tuple for A4/DIM32 is `(4,4096,512)`; `packet.py::stripe_rows` implements this reference geometry. Distinguish logical quantization stripes from later hardware subdivision.

## A. StreamQuant: linear inputs and Q

Input `X[rows,K]` is float32. For each row and K block:

```text
e[k]  = eps(X[k])
e1    = max e
e2    = largest exponent strictly below e1, or NEG
top[k]= e[k]==e1 and e[k]!=NEG and e2!=NEG
e_pre = e2 if e2!=NEG else e1
q[k]  = qround(X[k], e_pre-rho)
U     = nonzero source positions that are not top
cnt   = |U|; S=sum_U abs(q); SS=sum_U q^2
var   = cnt*SS-S^2
sig[k]= k in U and cnt>0 and var>0 and cnt*abs(q[k])-S>0
        and (cnt*abs(q[k])-S)^2 > tau^2*var
E     = max e[k] over U excluding sig, if any sig; otherwise e_pre
q[k]  = qround(X[k], E-rho)
out[k]= top[k] or sig[k]
```

For each stripe, select `e_s` as the second-largest distinct non-NEG block exponent E, or the last available if fewer exist. For each element of each block:

```text
D      = E-e_s
u      = shift(q,D)
corr   = correct_all or out or D>0
M      = lowdigit(u) if corr else sat(u)
r      = u-M if corr else 0
D_l    = balanced digit l of r; lane 0 is zero
L_l    = rows with a nonzero digit in lane l
C_b    = union of columns with any upper digit in original K block b
scale  = 2^(e_s-rho)
X_hat  = (M+r)*scale
```

Pad each selected C_b to whole DIM fragments. Keep original block and lane identities. Linear sets `correct_all=false`; Q sets `correct_all=true`.

Accepted optimization: linear selects `v=corr ? u : sat(u)`, Q selects `v=u`; decompose v once and emit upper digits directly. This preserves the original main/digits exactly and avoids storing r. It does not change linear to unconditional correction. Keep every required limb.

## B. Linear: Y = X W

StreamQuant uses `rho=2`, `correct_all=false`. W already supplies stationary codes `QW` and exponents `thetaW` along K. For each stripe/output tile and original K block:

```text
main:  acc += (M . QW) * 2^((e_s-2)+thetaW)
upper: acc[L_l] += (D_l[:,C_b] . QW[C_b])
                   * 2^((e_s-2)+thetaW+4*l)
Y = acc
```

Weights are not re-quantized to make a comparison pass. The plan uses raw integer dots plus exact host scaling/merging because existing hardware SCU-final saturation does not implement the requested wide sum.

## C. Attention per head (a4nks)

**C1 Q:** StreamQuant with `rho=6`, `correct_all=true`. All required signed 4-bit digit planes are kept.

**C2 K:** Per token j / channel block b, `thetaK[j,b]=max eps(K[j,b])-6`. `k8=clip(qround(K,thetaK),-128,119)`; `K_lo=lowdigit(k8)`; `K_hi=(k8-K_lo)/16`. Both passes are signed `[-8,7]`. The upper bound 119 is necessary for the high nibble.

**C3 QK:** Compute Q main and every upper lane against both K passes. The exponent for each original channel block/output column is `e_s-6+thetaK+sp+4*l`, where `sp` is 0 or 4 and main has l=0. Then `P=softmax(S/sqrt(d)+causal_mask)` in floating point. Product integration retains the model graph's own attention scale and masks.

**C4 P:** Per logical stripe, `e_s=eps(max(P))`; `q=clip(round_half_even(P*2^-(e_s-7)),0,255)`. There is no block adaptation, outlier selection or folding shift. `lo=q&15`, `M=lo-8`, `r=q-lo`. Balanced digits of r have lanes 1/2; lane 2 is 1 iff q>=128. Reuse active-row/K-column compaction. `P_hat=q*2^(e_s-7)`. Decomposing q directly would change this main/zero-point contract.

**C5 V:** Per channel c / token block b, `thetaV[b,c]=max eps(V[b,c])-2`; `V4=clip(qround(V,thetaV),-8,7)`. Precompute `colsumV[c]=sum_t V4[t,c]*2^thetaV[b(t),c]` once per head.

**C6 PV:** Main and upper dot exponents are `e_s-7+thetaV+4*l`. Publish `O=acc+8*2^(e_s-7)*colsumV`, broadcasting the offset across the stripe's rows. Include the same masked-token domain so q=0 cancels its main offset exactly. For GQA, the continuation plan reuses colsum across query heads sharing a KV head within one invocation.

## D. Fragment and add accounting

For each GEMM, using DIM-padded fragments:

```text
main     = ceil(rows/DIM) * ceil(N/DIM) * sum_b ceil(|b|/DIM)
residual = ceil(m/DIM)    * ceil(N/DIM) * sum_b ceil(|C_b|/DIM)
m        = number of nonzero (lane,row) pairs, packed together
matmul x = 1 + residual/main
```

Sum stripe contributions. QK's second K pass is charged to additional/residual work, adding another Q main plus Q upper work. P zero-point adds are weighted colsum plus output broadcast: T*d each for equal lengths; `Tkv*d+Tq*d` for unequal lengths with KV-head reuse recorded separately. These are work counts, not latency estimates.

## Boundaries found in the supplied Python

- NEG entries must not become real E2 anchors; all-zero blocks use zero codes and a neutral stored exponent. Padding must not enter source-nonzero statistics; reduction tails need explicit handling. An all-zero streamed stripe chooses anchor 0; all-zero P uses zero unsigned codes with its cancelling main offset.
- Clipping a negative shift to 31 in the pasted helper was incorrect. `shift(2^30,-32)=0`; `shift(INT32_MIN,-32)=-1`; right shifts >=33 give zero.
- `INT32_MAX-lowdigit(INT32_MAX)=2^31` cannot be a signed int32 residual. Balanced representation needs digit positions 0..8; do not force the upper sum through a legacy scalar emitter.
- The pasted int64 GEMM is not a general exact accumulator: exponent spread and cancellation can overflow it. The accepted plan uses checked int64 or exact host accumulation, then the specified F64->F32 conversion.
- The pasted `verify()` supplies captured original P to `attention_head`, so that particular test does not validate recomputed QK->softmax->PV accuracy. Current CPU model experiments do recompute P, but use reconstructed FP32 matmul. Integer profiling is separate component replay.
- No native attention implementation, simulator parity, or external `sf_tradeoff`/`attn_quant` parity follows merely from the supplied file's docstring. See the handoff for observed checks and remaining work.
