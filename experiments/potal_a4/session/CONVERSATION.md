# PoTal A4: nano continuation conversation

Source session: `01a12521-397c-7d92-b5bf-c38ac78e1886`. Portable session: `5d07d5b8-15f4-5d29-a2d5-01e2398bb920`.

2026-10-10 대화에서 사용자 메시지 29개와 공개 답변·진행 설명 70개를 원문 그대로 보존했다. 처음 제공한 명세와 Python 코드, 선택 질문의 답변도 포함한다. 시각은 UTC다. 환경 초기화·복구 메시지, 도구 실행 로그, 계정 설정은 포함하지 않는다.

과거의 가설과 예비 수치도 당시 발언 그대로다. 최신 정책과 최종 측정 결과는 [POTAL_NANO_HANDOFF.md](../../../POTAL_NANO_HANDOFF.md)를 먼저 따른다. 이후 정확도 유지 결정이 선형 direct-all 제안을 대체했다. 복원은 [README.md](README.md)를 따른다.

## 001 · user · 2026-10-10T09:27:21.910Z

 git branch "hotfix/potal-attn"
fatal: cannot lock ref 'refs/heads/hotfix/potal-attn': 'refs/heads/hotfix' exists; cannot create 'refs/heads/hotfix/potal-attn' 이거 뭐임

## 002 · assistant · 2026-10-10T09:27:24.654Z commentary

Git 오류 원인을 묻는 걸로 이해했어요. 브랜치 이름이 충돌하는 이유와 해결 명령을 설명하고 마칠게요.

LazyCodex는 v5.1.21로 업데이트됐고, v5.1.29가 백그라운드에서 설치 중이에요. 완료 후 새 Codex 세션을 시작하면 적용됩니다.


## 003 · assistant · 2026-10-10T09:27:41.937Z final_answer

이미 **`hotfix`라는 브랜치가 있어서** 이름이 충돌한 거예요. Git은 `hotfix`와 `hotfix/potal-attn`을 동시에 만들 수 없어요. 파일과 폴더 이름이 겹치는 것과 비슷합니다.

새 브랜치 이름에서 `/`를 빼면 해결돼요:

```bash
rtk git branch hotfix-potal-attn
```

꼭 `hotfix/potal-attn`을 쓰려면 기존 `hotfix` 브랜치 이름을 먼저 바꿔야 해요.

명령은 실행하지 않았어요.

## 004 · user · 2026-10-10T09:31:40.303Z

hotfix 브랜치 이미 있는거 이름 뭔데?

## 005 · assistant · 2026-10-10T09:31:42.177Z commentary

현재 저장소에서 `hotfix`로 시작하는 브랜치 이름을 확인할게요. 목록이 나오면 마칠게요.


## 006 · assistant · 2026-10-10T09:31:49.543Z final_answer

이름이 그냥 **`hotfix`**예요. `hotfix/무언가`가 아니라 `hotfix` 자체가 브랜치 이름입니다.

로컬 목록에서 확인했고, 변경은 하지 않았어요.

## 007 · user · 2026-10-10T09:31:56.769Z

로컬이야? 누가 만든거야

## 008 · assistant · 2026-10-10T09:31:59.729Z commentary

로컬 `hotfix` 브랜치가 맞아요. `omo:git-master` 스킬로 생성 기록을 확인해서 누가 만들었는지 알 수 있는 범위까지 확인할게요.


## 009 · assistant · 2026-10-10T09:32:17.062Z final_answer

네, 충돌을 일으킨 건 **로컬 `hotfix` 브랜치**예요.

Git 생성 기록(reflog)은 이렇게 남아 있어요:

- 생성 시각: **2026-09-21 05:53:50, 한국시간**
- Git 사용자 이름: **`Chan`**
- 내용: `branch: Created from HEAD`

직접 만든 건지, 같은 Git 설정을 쓰는 에이전트가 만든 건지는 이 기록만으론 구분할 수 없어요. 조회만 했고 변경은 하지 않았어요.

## 010 · user · 2026-10-10T09:33:11.573Z

없애. 내 로컬 브랜치도 이제 정리할거야. develop이랑 최근에 만든 metal버전이랑 cuda버전인가 그거만 두고 나머지 정리해

## 011 · assistant · 2026-10-10T09:33:15.461Z commentary

로컬 브랜치에서 `develop`과 최근 Metal·CUDA 작업 브랜치를 찾고, 나머지를 삭제하라는 요청으로 이해했어요. `omo:git-master` 절차로 각 브랜치의 커밋과 워크트리 사용 여부를 확인한 뒤, 지정한 브랜치만 남으면 마칠게요.


## 012 · assistant · 2026-10-10T09:34:04.395Z commentary

최근에 만든 쌍은 `work/metal-ablation-20261010`과 `work/cuda-ppl-20261010`으로 확인했어요. 이 둘을 로컬에도 만들고 `develop`과 함께 남기겠습니다. 다른 브랜치를 사용 중인 워크트리 세 곳은 현재 커밋에서 연결만 해제해, 파일과 미커밋 작업을 보존하겠습니다.


## 013 · user · 2026-10-10T09:35:05.694Z

응 그러면서 저장소 관리도 해.

## 014 · assistant · 2026-10-10T09:35:18.438Z commentary

브랜치 정리와 함께 Git 무결성, 오래된 원격 추적 정보, 경로가 사라진 워크트리 등록도 점검하고 정리할게요. 현재 남아 있는 파일과 미커밋 작업은 보존하겠습니다.


## 015 · assistant · 2026-10-10T09:36:16.035Z commentary

`hotfix`를 포함한 로컬 브랜치 15개를 삭제했고, 요청한 세 개만 남았습니다. 미커밋 변경 내용도 그대로인 걸 확인했어요. Git 무결성 검사는 통과했고, 서버에서 이미 삭제된 원격 추적 정보 16개와 경로가 사라진 워크트리 등록 4개를 정리 중입니다.


## 016 · assistant · 2026-10-10T09:37:18.111Z final_answer

정리 완료했어요. **로컬 브랜치는 3개만** 남았습니다.

- `develop` — 현재 브랜치
- `work/metal-ablation-20261010`
- `work/cuda-ppl-20261010`

`hotfix` 포함 15개를 삭제했고, 오래된 원격 추적 정보 16개와 사라진 워크트리 등록 4개도 정리했습니다. Git 무결성 검사도 통과했어요.

원격 브랜치와 미커밋 작업은 보존했습니다. 기존 워크트리 3곳은 `detached HEAD` 상태이므로, 그곳에서 작업을 이어가려면 브랜치를 연결해야 합니다.

## 017 · user · 2026-10-10T09:39:42.654Z

이제 작업 시작 준비하자. ================================================================================
  PoTal A4 quantization — implementation spec (linear + attention, a4nks)
  ================================================================================
  Constants
    n      = 4                      PE operand width (signed: [-8, 7])
    BK     = 32                     block length along the reduction axis (= DIM)
    tau    = 2                      2-sigma threshold
    NEG    = -inf exponent marker   (exact zero)
    anchor = E2                     stripe anchor rank for folded streamed operands

  Primitives (integer unless noted)
    eps(x)            = floor(log2|x|) for x != 0, NEG for x == 0
    qround(x, t)      = round_half_even(x * 2^-t), saturated to int32
    shift(q, D)       = D > 0: sat_int32(q << D)
                        D < 0: sign(q) * ((|q| + 2^(-D-1)) >> -D)      // round half away
                        D = 0: q
    sat(u)            = clip(u, -2^(n-1), 2^(n-1) - 1)
    lowdigit(u)       = ((u + 2^(n-1)) mod 2^n) - 2^(n-1)             // signed low n-bit digit
    digits(r)         = balanced radix-2^n digits of r:  r = sum_l d_l 2^(n l), d_l in [-8, 7]
                        loop: d = r & (2^n-1); if d >= 2^(n-1): d -= 2^n; emit d; r = (r - d) >> n
    stripes(rows)     = row groups of stripe_rows(I, N, K, DIM, n)    // GEMM tile geometry
    SCU(acc, part, s) = acc += part << s     (column-bottom shift, then accumulate)

  --------------------------------------------------------------------------------
  A. Streamed operand quantization (linear inputs; attention Q)        StreamQuant
  --------------------------------------------------------------------------------
  input : X (rows x K, float32), rho (code width - 2), correct_all (bool)
  output: per stripe s: anchor e_s; main codes M (rows x K, in [-8,7]);
          residual digit planes D_l (l >= 1) with active-row sets L_l and column set C_b

  for each row i, block b (BK columns):                           // 1. block adaptation
      e[k]   = eps(X[i,k])
      e1     = max e;  e2 = max{e[k] : e[k] < e1}
      top[k] = (e[k] == e1) and e[k] != NEG and e2 != NEG
      e_pre  = (e2 != NEG) ? e2 : e1
      q[k]   = qround(X[i,k], e_pre - rho)
      U      = {k : not top[k] and e[k] != NEG}                   // zeros excluded
      cnt=|U|; S=sum_U |q|; SS=sum_U q^2; var = cnt*SS - S^2
      sig[k] = k in U and cnt>0 and var>0 and (cnt*|q[k]| - S) > 0
               and (cnt*|q[k]| - S)^2 > tau^2 * var
      if any(sig): E[i,b] = max{e[k] : k in U, not sig[k]}  else E[i,b] = e_pre
      q[k]   = qround(X[i,k], E[i,b] - rho)                       // final block codes
      out[i,k] = top[k] or sig[k]

  for each stripe s:                                              // 2. stripe folding
      e_s = anchor-th largest distinct E[i,b] in s (last one if fewer)
      for i in s, block b, k in b:
          D      = E[i,b] - e_s
          u      = shift(q[i,k], D)                               // int32
          corr   = correct_all or out[i,k] or D > 0
          M[i,k] = corr ? lowdigit(u) : sat(u)                    // 3. low-digit main
          r      = corr ? u - M[i,k] : 0                          // multiple of 2^n
          (d_0 = 0, d_1, d_2, ...) = digits(r)
          for l >= 1 with d_l != 0: D_l[i,k] = d_l; add i to L_l; add k to C_b
      pad each C_b to whole DIM fragments                         // K-compaction
      scale of s = 2^(e_s - rho)
  // reconstruction (reference): X_hat[i,k] = (M[i,k] + r) * 2^(e_s - rho)

  --------------------------------------------------------------------------------
  B. Linear layer  Y = X W                                          (rho = n - 2 = 2)
  --------------------------------------------------------------------------------
  (M, {D_l, L_l, C_b}, e_s) = StreamQuant(X, rho=2, correct_all=false)
  W: stationary codes Q^W with block exponents theta^W along K (as given)
  for each stripe s, output tile:
      acc = 0
      for each K block b:                                          // main GEMM
          SCU(acc, M[s, b] . Q^W[b], (e_s - rho) + theta^W_b)
      for each lane l >= 1, rows i in L_l, K block b:              // RMD, row-pruned
          SCU(acc[i], D_l[i, C_b] . Q^W[C_b], (e_s - rho) + theta^W_b + n*l)
      Y[s] = acc                                                   // (or host wide add, Alg.1 l.22)

  --------------------------------------------------------------------------------
  C. Attention, per head  (a4nks)
  --------------------------------------------------------------------------------
  Q (T x d), K (T x d), V (T x d)

  C1. Q  (streamed, 8-bit codes on 4-bit PEs)
      (M^Q, {D^Q_l, L_l, C_b}, e_s) = StreamQuant(Q, rho=6, correct_all=true)

  C2. K  (stationary, 8-bit, two nibble passes)
      for each token j, channel block b:
          theta^K[j,b] = max_k eps(K[j,k]) - 6
          k8 = clip(qround(K[j,k], theta^K[j,b]), -128, 119)
          K_lo[j,k] = lowdigit(k8)                                  // [-8, 7]
          K_hi[j,k] = (k8 - K_lo[j,k]) / 16                          // [-8, 7] (needs k8 <= 119)

  C3. S = Q K^T
      for each stripe s, key tile:
          acc = 0
          for (Kp, sp) in ((K_lo, 0), (K_hi, 4)):                    // two passes, dense
              for channel block b:
                  SCU(acc, M^Q[s,b] . Kp[:,b]^T, (e_s-6) + theta^K_b + sp)
                  for lane l >= 1, rows i in L_l:
                      SCU(acc[i], D^Q_l[i,C_b] . Kp[:,C_b]^T, (e_s-6) + theta^K_b + sp + 4l)
          S[s] = acc
      P = softmax(S / sqrt(d) + causal_mask)                          // FP

  C4. P  (streamed, unsigned 8-bit per stripe; no block stage, no outlier selection)
      for each stripe s (rows):
          e_s = eps(max P[s])                                        // stripe max
          q   = clip(round_half_even(P[s] * 2^-(e_s-7)), 0, 255)    // int32, no shift
          lo  = q & 15                                               // main, unsigned
          M^P = lo - 8                                               // PE operand [-8, 7]
          r   = q - lo                                               // 16 * high nibble
          (0, d_1, d_2) = digits(r)                                  // d_2 = 1 iff q >= 128
          active rows L_1, L_2 and columns C_b as in A
  // reconstruction: P_hat = q * 2^(e_s - 7)

  C5. V  (stationary, 4-bit)
      for each channel c, token block b:
          theta^V[b,c] = max_t eps(V[t,c]) - 2
          V4[t,c] = clip(qround(V[t,c], theta^V[b,c]), -8, 7)
      colsumV[c] = sum_t V4[t,c] * 2^theta^V[b,c]       // once per head: T*d adds

  C6. O = P V
      for each stripe s, channel tile:
          acc = 0
          for token block b:
              SCU(acc, M^P[s,b] . V4[b], (e_s-7) + theta^V_b)
              for lane l in {1,2}, rows i in L_l:
                  SCU(acc[i], D_l[i,C_b] . V4[C_b], (e_s-7) + theta^V_b + 4l)
          O[s] = acc + 8 * 2^(e_s-7) * colsumV        // zero point: broadcast add (rows*d adds)

  --------------------------------------------------------------------------------
  D. Cost accounting (per GEMM, DIM-padded fragments)
  --------------------------------------------------------------------------------
  main      = ceil(rows/DIM) * ceil(N/DIM) * sum_b ceil(|b|/DIM)
  residual  = ceil(m/DIM)    * ceil(N/DIM) * sum_b ceil(|C_b|/DIM),
              m = number of (lane, row) pairs with a nonzero digit
  K second pass (C3, sp = 4): counted with residual (= main + Q residual again)
  matmul x  = 1 + residual / main;   adds = colsumV + zero-point broadcast (C5, C6) #!/usr/bin/env python3
"""
PoTal A4 quantization, executed in the integer domain as the implementation spec states: block adaptation with
outlier selection, stripe folding, low-digit main + RMD residual lanes (row-pruned, K-compacted) for streamed
operands, block codes (8-bit as two signed nibble passes) for stationary operands, column-bottom SCU shifts into one
accumulator, and the P zero point as adds. Linear GEMM inputs and the a4nks attention (Q, K, P, V).

Independent of sf_tradeoff.py / attn_quant.py (the float reference: fake quantization + fragment counts); running
this file verifies it against them on GPT-2 tensors:
    python3 potal.py            # chunk 0, layers 0, 5, 11
Checked: reconstructions and fragment / add counts equal the reference bitwise; every integer GEMM (main pass + lanes
+ nibble passes + zero point, SCU-shifted) equals the float64 GEMM of the reconstructed operands.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field

import numpy as np

N_PE = 4                                  # PE operand width (signed)
BK = 32                                   # block length along the reduction axis
TAU = 2                                   # 2-sigma threshold
NEG = -32768                              # exponent of an exact zero
INT32_MIN, INT32_MAX = -(2 ** 31), 2 ** 31 - 1
RANK = {"TOP": 0, **{f"E{r}": r - 1 for r in range(2, 9)}}
MEMORY = {(4, 32): (4, 4096, 512), (8, 32): (4, 2048, 512)}  # (banks, bank rows, accumulator rows) at DIM 32


# ---------------------------------------------------------------------------
# Primitives

def eps(x):
    """floor(log2 |x|); NEG for 0."""
    x = np.asarray(x)
    return np.where(x != 0, np.frexp(np.abs(x))[1] - 1, NEG).astype(np.int64)


def qround(x, t, lo=INT32_MIN, hi=INT32_MAX):
    """round_half_even(x / 2^t) of float32 x, saturated (scaling by 2^-t is exact in float32)."""
    with np.errstate(over="ignore"):
        v = np.rint(np.ldexp(np.asarray(x, np.float32), -np.asarray(t, np.int32)))
    big = ~(np.abs(v) < 2.0 ** 31)
    return np.where(big, np.where(v > 0, hi, lo), np.clip(np.where(big, 0, v), lo, hi)).astype(np.int64)


def shift(q, d):
    """d > 0: left shift saturated to int32; d < 0: round-half-away right shift; d = 0: q."""
    q, d = np.asarray(q, np.int64), np.asarray(d, np.int64)
    mag = np.abs(q)
    rb = np.clip(-d, 0, 31)
    down = (mag + np.where(rb > 0, 1 << np.maximum(rb - 1, 0), 0)) >> rb
    up = mag << np.clip(d, 0, 31)
    left = np.where(q >= 0, np.minimum(up, INT32_MAX), np.where(up >= 2 ** 31, INT32_MIN, -up))
    return np.where(d > 0, left, np.where(d < 0, np.where(q >= 0, down, -down), q))


def sat(u, n=N_PE):
    return np.clip(u, -(1 << (n - 1)), (1 << (n - 1)) - 1)


def lowdigit(u, n=N_PE):
    """Signed low n-bit digit: u - lowdigit(u) is a multiple of 2^n."""
    return ((u + (1 << (n - 1))) & ((1 << n) - 1)) - (1 << (n - 1))


def digits(r, n=N_PE, lanes=16):
    """Balanced radix-2^n digits of r (lane 0 first), each in [-2^(n-1), 2^(n-1))."""
    out, r = [], np.asarray(r, np.int64)
    for _ in range(lanes):
        d = lowdigit(r, n)
        out.append(d)
        r = (r - d) >> n
    assert not r.any(), "residual exceeds the lanes"
    return out


def stripe_rows(I, J, K, dim, bits):
    """Rows of one stripe: the output tile height of the GEMM schedule (scratchpad / accumulator geometry)."""
    banks, bank_rows, acc_rows = MEMORY[(bits, dim)]
    max_ij = int(math.sqrt((acc_rows // 2) // dim))
    max_k = (((banks * bank_rows // 2) // 2) // dim) // max_ij
    Ip, Jp, Kp = (math.ceil(v / dim) * dim for v in (I, J, K))
    spad_max, acc_max = banks * bank_rows // 2, acc_rows // 2
    spad = lambda i, j, k: (i * k + k * j) * dim                                       # noqa: E731
    ti, tj, tk = min(Ip // dim, max_ij), min(Jp // dim, max_ij), min(Kp // dim, max_k)
    while True:
        grew = False
        if spad(ti, tj + 1, tk) <= spad_max and ti * (tj + 1) * dim <= acc_max and (tj + 1) * dim <= Jp:
            tj, grew = tj + 1, True
        if spad(ti + 1, tj, tk) <= spad_max and (ti + 1) * tj * dim <= acc_max and (ti + 1) * dim <= Ip:
            ti, grew = ti + 1, True
        if spad(ti, tj, tk + 1) <= spad_max and (tk + 1) * dim <= Kp:
            tk, grew = tk + 1, True
        if not grew:
            return ti * dim


def frags(rows, n, cols_per_block, dim):
    """DIM-padded GEMM fragments: rows x n output, the given column count per K block."""
    return math.ceil(rows / dim) * math.ceil(n / dim) * sum(math.ceil(c / dim) for c in cols_per_block)


# ---------------------------------------------------------------------------
# A. Streamed operand: block adaptation, stripe folding, low-digit main + RMD lanes

@dataclass
class Stripe:
    rows: np.ndarray                      # row indices of the operand
    scale: int                            # exponent of the stripe scale (codes x 2^scale)
    main: np.ndarray                      # rows x K main codes, in [-2^(n-1), 2^(n-1))
    lanes: dict = field(default_factory=dict)   # l >= 1 -> (active row positions, digit plane rows x K)
    cols: list = field(default_factory=list)    # per K block: columns holding a residual (K-compaction)
    zero_point: int = 0                   # main operand = code - zero_point (P: 8)


def block_adapt(X, rho):
    """Spec A.1 per row and BK block: final block exponents E (rows x blocks), codes q, outlier mask."""
    R, K = X.shape
    x = X.reshape(R, K // BK, BK)
    e = eps(x)
    e1 = e.max(-1, keepdims=True)
    e2 = np.where(e < e1, e, NEG).max(-1, keepdims=True)
    top = (e == e1) & (e != NEG) & (e2 != NEG)
    e_pre = np.where(e2 != NEG, e2, e1)
    q = np.where(e_pre == NEG, 0, qround(x, np.where(e_pre == NEG, 0, e_pre - rho)))
    U = ~top & (e != NEG)
    mag = np.where(U, np.abs(q), 0)
    cnt, S = U.sum(-1, keepdims=True), mag.sum(-1, keepdims=True)
    var = cnt * (mag * mag).sum(-1, keepdims=True) - S * S
    cen = cnt * mag - S
    sig = U & (cnt > 0) & (var > 0) & (cen > 0) & (cen * cen > TAU * TAU * var)
    E = np.where(sig.any(-1, keepdims=True), np.where(U & ~sig, e, NEG).max(-1, keepdims=True), e_pre)
    q = np.where(E == NEG, 0, qround(x, np.where(E == NEG, 0, E - rho)))
    return E[..., 0], q.reshape(R, K), (top | sig).reshape(R, K)


def stream_quant(X, N, rho, anchor, correct_all, dim=32):
    """Spec A for one operand (rows x K) of a GEMM with output width N: its stripes."""
    R, K = X.shape
    E, q, out = block_adapt(X, rho)
    rs = stripe_rows(R, N, K, dim, N_PE)
    stripes = []
    for r0 in range(0, R, rs):
        sl = slice(r0, min(r0 + rs, R))
        Es = E[sl]
        ex = np.unique(Es[Es != NEG])[::-1]
        e_s = int(ex[min(RANK[anchor], len(ex) - 1)]) if len(ex) else 0
        D = np.repeat(np.where(Es != NEG, Es - e_s, 0), BK, axis=1)
        u = shift(q[sl], D)
        corr = np.ones_like(u, bool) if correct_all else out[sl] | (D > 0)
        main = np.where(corr, lowdigit(u), sat(u))
        st = Stripe(np.arange(sl.start, sl.stop), e_s - rho, main)
        _lanes(st, np.where(corr, u - main, 0))
        stripes.append(st)
    return stripes


def _lanes(st, residual):
    """Residual digit planes l >= 1 with their active rows (row pruning) and residual columns (K-compaction)."""
    ds = digits(residual)
    assert not ds[0].any(), "low-digit main leaves lane 0 empty"
    for l, d in enumerate(ds[1:], 1):
        active = np.flatnonzero(d.any(1))
        if len(active):
            st.lanes[l] = (active, d[active])
    nz = residual != 0
    st.cols = [np.flatnonzero(nz[:, b:b + BK].any(0)) + b for b in range(0, nz.shape[1], BK)]


def reconstruct(stripes, shape):
    out = np.zeros(shape)
    for st in stripes:
        v = (st.main + st.zero_point).astype(np.float64)
        for l, (rows, d) in st.lanes.items():
            v[rows] += d * float(1 << (N_PE * l))
        out[st.rows] = np.ldexp(v, st.scale)
    return out


def stream_frags(stripes, N, K, dim=32):
    """(main, residual) fragments: residual rows = (lane, row) pairs, columns = K-compacted."""
    main = sum(frags(len(st.rows), N, [min(BK, K - b) for b in range(0, K, BK)], dim) for st in stripes)
    res = sum(frags(sum(len(r) for r, _ in st.lanes.values()), N, [len(c) for c in st.cols], dim)
              for st in stripes if st.lanes)
    return main, res


# ---------------------------------------------------------------------------
# Stationary operands: block codes along the reduction axis

@dataclass
class Stationary:
    passes: list                          # [(codes K x N in [-8, 7], extra shift)]: 4-bit one pass, 8-bit two
    theta: np.ndarray                     # (K blocks) x N block exponents: value = code x 2^theta


def stationary_quant(W, bits, hi_code=None):
    """W (K x N, reduction axis first): per column and BK block of K, codes at the block's top exponent; 8-bit
    codes (saturated to hi_code) run as two signed nibble passes 16 hi + lo."""
    K, N = W.shape
    rho = bits - 2
    w = W.T.reshape(N, K // BK, BK)
    E = eps(w).max(-1)                                                # N x blocks
    hi = (1 << (rho + 1)) - 1 if hi_code is None else hi_code
    c = np.where(E[..., None] == NEG, 0, qround(w, np.where(E == NEG, 0, E - rho)[..., None], -(1 << (rho + 1)), hi))
    codes = c.reshape(N, K).T
    theta = np.where(E == NEG, 0, E - rho).T                          # blocks x N
    if bits == N_PE:
        return Stationary([(codes, 0)], theta)
    lo = lowdigit(codes)
    hi_n = (codes - lo) >> N_PE
    assert hi_n.min() >= -8 and hi_n.max() <= 7, "8-bit code outside the signed nibble pair (saturate to 119)"
    return Stationary([(lo, 0), (hi_n, N_PE)], theta)


def stationary_value(st):
    return np.sum([np.ldexp(c.astype(np.float64), np.repeat(st.theta, BK, axis=0) + s) for c, s in st.passes], axis=0)


# ---------------------------------------------------------------------------
# GEMM execution: integer passes, SCU shift per output column, one accumulator per stripe

def gemm(stripes, W, n_out):
    """Y = X W for a streamed operand (stripes) and a stationary one (W: Stationary). Every partial sum of a K block
    is shifted by the SCU (stripe exponent + block exponent of each column + nibble / lane weight) before it is
    accumulated; the accumulator is int64 at the stripe's lowest shift. Returns float64 Y and pass counters."""
    Kb = W.theta.shape[0]
    Y = np.zeros((sum(len(s.rows) for s in stripes), n_out))
    for st in stripes:
        base = st.scale + int(W.theta.min())
        acc = np.zeros((len(st.rows), n_out), np.int64)

        def scu(rows, part, b, extra):
            sh = st.scale + W.theta[b] + extra - base                     # per output column, >= 0
            acc[rows] += part << sh.astype(np.int64)

        for codes, sp in W.passes:
            for b in range(Kb):
                k = slice(b * BK, (b + 1) * BK)
                scu(slice(None), st.main[:, k] @ codes[k], b, sp)                       # main pass
                for l, (rows, d) in st.lanes.items():                                    # RMD lanes
                    cols = st.cols[b]
                    if len(cols):
                        scu(rows, d[:, cols] @ codes[cols], b, sp + N_PE * l)
        if st.zero_point:                                              # + zero_point * colsum(W): adds only
            for codes, sp in W.passes:
                for b in range(Kb):
                    cs = codes[b * BK:(b + 1) * BK].sum(0)
                    acc += (st.zero_point * cs) << (st.scale + W.theta[b] + sp - base).astype(np.int64)
        Y[st.rows] = np.ldexp(acc.astype(np.float64), base)
    return Y


# ---------------------------------------------------------------------------
# B. Linear layer and C. attention (a4nks), per head

def linear(X, W_codes, anchor="E2", dim=32):
    """Y = X W, X streamed at A4 (rho 2); W given as Stationary codes (K x N)."""
    N = W_codes.passes[0][0].shape[1]
    st = stream_quant(X, N, 2, anchor, False, dim)
    m, r = stream_frags(st, N, X.shape[1], dim)
    return gemm(st, W_codes, N), st, {"main_frag": m, "clip_frag": r}


def p_quant(P, n_out, dim=32):
    """Spec C4: P (rows x keys) unsigned 8-bit per stripe at the stripe max's exponent; main = low nibble - 8."""
    R, K = P.shape
    rs = stripe_rows(R, n_out, K, dim, N_PE)
    stripes = []
    for r0 in range(0, R, rs):
        sl = slice(r0, min(r0 + rs, R))
        m = P[sl].max()
        e_s = int(eps(m)) if m > 0 else 0
        q = np.where(m > 0, qround(P[sl], e_s - 7, 0, 255), 0)
        lo = q & 15
        st = Stripe(np.arange(sl.start, sl.stop), e_s - 7, lo - 8, zero_point=8)
        _lanes(st, q - lo)
        stripes.append(st)
    return stripes


def attention_head(Q, K, V, P_fn, anchor="E2", dim=32):
    """a4nks for one head: S = Q K^T (Q streamed 8-bit nibble RMD, K 8-bit two passes), P = P_fn(S), O = P V (P per
    stripe uint8 + zero point, V 4-bit). Returns S, O, operands and costs."""
    T, d = Q.shape
    sq = stream_quant(Q, T, 6, anchor, True, dim)
    Kc = stationary_quant(K.T, 8, 119)                                # K^T: reduction axis (channels) first
    S = gemm(sq, Kc, T)
    P = P_fn(S)
    sp = p_quant(P, d, dim)
    Vc = stationary_quant(V, 4)
    O = gemm(sp, Vc, d)
    qm, qr = stream_frags(sq, T, d, dim)
    pm, pr = stream_frags(sp, d, T, dim)
    cost = {"main_frag": qm + pm, "clip_frag": qr + (qm + qr) + pr,   # K hi pass counted with the residual GEMMs
            "add_ops": T * d + T * d}                                 # colsum(V) + zero-point broadcast
    return S, O, {"q": sq, "k": Kc, "p": sp, "v": Vc, "P": P}, cost


# ---------------------------------------------------------------------------
# Verification against sf_tradeoff.py / attn_quant.py on GPT-2

def verify(layers=(0, 5, 11)):
    import types
    import sf_tradeoff as sf
    import attn_quant as aq
    sd, chunks, _ = sf.load_model("gpt2", types.SimpleNamespace(cache=sf.ROOT / "cache", text=None))
    tok = chunks[0]
    lin, att = [], {}
    sf.MODEL["nll"](sd, tok, lambda u, X, n, l: (lin.append((u, X.copy(), n)) if u != "embedding" and l in layers else None) or X,
                    lambda op, X, n, l: (att.setdefault(l, {}).__setitem__(op, (X.copy(), n)) if l in layers else None) or X)
    rng = np.random.default_rng(0)
    worst, n_chk = 0.0, 0

    def close(a, b):
        nonlocal worst, n_chk
        err = np.abs(a - b).max() / max(np.abs(b).max(), 1e-300)
        worst, n_chk = max(worst, err), n_chk + 1
        assert err < 1e-12, err

    for u, X, N in lin:                                                # B. linear
        Wf = (rng.standard_normal((X.shape[1], N)) * np.exp2(rng.integers(-6, 2, (1, N)))).astype(np.float32)
        Wc = stationary_quant(Wf, 4)
        Y, st, cost = linear(X, Wc)
        ref_cost = dict.fromkeys(sf.COST_KEYS, 0)
        Xr = sf.fold_heads(X, N, 4, "E2", 32, ref_cost, True)
        assert np.array_equal(reconstruct(st, X.shape), Xr), u
        assert (cost["main_frag"], cost["clip_frag"]) == (ref_cost["main_frag"], ref_cost["clip_frag"]), (u, cost, ref_cost)
        close(Y, Xr @ stationary_value(Wc))
    for l, ops in att.items():                                         # C. attention a4nks, every head
        Qh, Kh, Vh, Ph = (ops[op][0] for op in ("q", "k", "v", "p"))
        ref = dict.fromkeys(aq.COST_KEYS, 0)
        rq, rk, rv, rp = (aq.attention("a4nks", op, X, n, "E2", 32, ref) for op, (X, n) in
                          (("q", ops["q"]), ("k", ops["k"]), ("v", ops["v"]), ("p", ops["p"])))
        tot = dict.fromkeys(("main_frag", "clip_frag", "add_ops"), 0)
        for h in range(Qh.shape[0]):
            S, O, parts, cost = attention_head(Qh[h], Kh[h], Vh[h], lambda S, h=h: Ph[h])
            assert np.array_equal(reconstruct(parts["q"], Qh[h].shape).astype(np.float32), rq[h])
            assert np.array_equal(stationary_value(parts["k"]).T.astype(np.float32), rk[h])
            assert np.array_equal(stationary_value(parts["v"]).astype(np.float32), rv[h])
            assert np.array_equal(reconstruct(parts["p"], Ph[h].shape).astype(np.float32), rp[h])
            close(S, rq[h].astype(np.float64) @ rk[h].astype(np.float64).T)
            close(O, rp[h].astype(np.float64) @ rv[h].astype(np.float64))
            for k_ in tot:
                tot[k_] += cost[k_]
        assert tot == {k_: ref[k_] for k_ in tot}, (l, tot, {k_: ref[k_] for k_ in tot})
    print(f"verified on GPT-2 chunk 0, layers {list(layers)}: {len(lin)} linear GEMMs, "
          f"{sum(v['q'][0].shape[0] for v in att.values())} attention heads; reconstructions and fragment / add "
          f"counts equal the reference; {n_chk} integer GEMMs vs float64 GEMMs: max relative error {worst:.1e}")


if __name__ == "__main__":
    sys.exit(verify())
 계획 세우고 뭘 해야할지 구체적으로 봐.

## 018 · assistant · 2026-10-10T09:39:48.570Z commentary

주신 명세와 Python 기준 구현을 현재 코드에 대조해서, 구현 위치·차이점·검증 순서가 명확한 작업 계획을 만들겠습니다. `omo:ulw-plan` 절차를 적용하고, 구현에 들어가기 전에 결정해야 할 사항까지 정리되면 마칠게요.


## 019 · assistant · 2026-10-10T09:40:43.465Z commentary

ULW-PLAN MODE ENABLED!

현재 범위는 구현 전 계획입니다. 기존 선형 양자화 경로, attention 호출 경로, 기준 코드의 수치 경계조건을 나눠 조사하겠습니다. 주신 요청에 따라 계획 문서까지 작성하고, 실제 코드 변경은 다음 실행 단계로 남기겠습니다.


## 020 · assistant · 2026-10-10T09:42:13.644Z commentary

현재 선형 경로에는 ExSIA·RMD가 이미 있어 재사용할 수 있지만, 명세의 영점 제외 통계와 low-digit main 규칙은 기존 동작과 다릅니다. attention은 현재 선형 GEMM 경로에 그대로 붙일 수 없어 별도 연결 설계가 필요합니다.

의도는 CLEAR이며 기본 계획 검토로 진행합니다. 완료 기준은 수치 일치, 실제 attention 실행 확인, 실제 실행량과 맞는 비용계수로 잡겠습니다.


## 021 · user · 2026-10-10T09:42:31.735Z

<send_user_message_question_reply>
[{"answer":"Gemmini 구현 + CPU 기준 검증 (추천)","question":"이번 PoTal A4 구현의 대상 백엔드는 어디인가요? 현재 develop의 Gemmini 경로와 별도 Metal/CUDA 실험 브랜치가 있어서, 이 선택에 따라 구현 위치와 검증 장비가 달라집니다.","questionItemId":"[\"request_user_input_async\",\"call_773dc539327f485090261aa900828de5\",0]"}]
</send_user_message_question_reply>

## 022 · user · 2026-10-10T09:42:50.242Z

 git branch "hotfix/potal-attn"

## 023 · assistant · 2026-10-10T09:43:00.367Z commentary

Gemmini 구현과 CPU 기준 검증으로 확정하겠습니다. 요청한 `hotfix/potal-attn` 브랜치를 현재 커밋에서 만들고 계획 작성을 이어갈게요.


## 024 · assistant · 2026-10-10T09:45:48.390Z commentary

`hotfix/potal-attn`은 이미 `develop`과 같은 커밋으로 만들어져 있네요. 계획의 작업 브랜치로 기록하겠습니다.

가장 큰 쟁점은 누적 정밀도입니다. 기존 Gemmini SCU는 int32 포화 누적이고, 주신 명세는 넓은 정수 누적을 요구합니다. 기준 Python에도 큰 우측 이동과 int64 overflow 반례가 있어, 먼저 기준 계산을 고정한 뒤 이식하는 순서가 필요합니다.


## 025 · assistant · 2026-10-10T09:47:27.978Z commentary

중단 전 조사 결과와 수치 반례는 확보됐습니다. 저장된 초안과 브랜치 상태를 확인한 뒤, 구현 순서와 검증 항목을 담은 최종 계획을 마무리하겠습니다.


## 026 · assistant · 2026-10-10T09:53:19.641Z commentary

기존 Gemmini가 포화되지 않은 4비트 raw dot 결과를 반환하는 경로는 확인했습니다. 이를 재사용하고, 명세의 지수·lane 가중치는 host에서 넓게 합산하는 방향으로 계획을 잡았습니다.

작업은 10개로 나눴습니다. 기준 계산과 residual 표현을 먼저 고친 뒤 선형 경로, attention 연결, 비용계수와 모델 검증으로 이어집니다. 지금은 계획의 빠진 의존성과 검증 조건을 점검하고 있습니다.


## 027 · assistant · 2026-10-10T09:57:09.491Z commentary

계획 검토에서 최종 float 변환 규칙이 모호한 점을 찾았습니다. 제공한 Python의 반환 형식에 맞춰 정확한 누적값을 F64로 반올림한 뒤, GGML 출력 시 F32로 변환하도록 명시하겠습니다. A8 호환성을 위해 통계·반올림 변경도 A4 정책에 한정하겠습니다.

## 028 · assistant · 2026-10-10T09:58:49.603Z final_answer

`hotfix/potal-attn`은 이미 생성돼 있고, `develop`과 같은 `3b2eaa3`를 가리킵니다. 현재 체크아웃은 `develop`입니다. `git branch`는 생성만 합니다.

**[구현 계획](/Users/chan/Projects/AISA/llama.cpp-gemmini/.omo/plans/potal-a4nks-linear-attention.md)**을 작성하고 검토했습니다. Gemmini 구현 + CPU 기준 검증으로, 10개 작업을 다음 순서로 진행합니다.

| 단계 | 구체적인 작업 |
|---|---|
| 1. 수치 기준 확정 | Python 기준 구현 수정, 경계값 fixtures, residual int64 전파, attention 역할·옵션 추가 |
| 2. 연산 구현 | 공통 StreamQuant, Gemmini raw dot + 넓은 host 누적, K/P/V 양자화, 실제 fragment/add 계수 |
| 3. 통합·검증 | 선형 경로 연결, QK→softmax→PV 연결, prefill/decode·GQA, GPT-2/Llama 및 기존 A8 회귀 검증 |

먼저 해결해야 할 문제도 확인했습니다.

- 예제의 `shift(2^30, -32)`는 **0이어야 하는데 1**이 됩니다.
- `INT32_MAX`를 low digit으로 분리하면 residual이 **+2³¹**이라 int32에 담기지 않습니다.
- 예제의 int64 GEMM과 기존 Gemmini의 int32 포화 누적 모두 명세의 넓은 합산을 보장하지 못합니다.
- 기존 검증은 저장된 P를 사용하므로, **양자화된 QK 결과에서 softmax까지 이어지는 검증**이 추가로 필요합니다.

첫 구현은 **수정된 기준 계산과 실패 사례 고정**입니다. 작업은 해당 브랜치의 별도 worktree에서 진행하도록 계획했습니다.

제품 코드 수정·커밋과 전체 모델·IM2P 실행은 아직 하지 않았습니다. Gemmini raw dot 이후 host 누적 비용은 별도로 측정해야 합니다.

## 029 · user · 2026-10-10T09:58:49.866Z

rmd도 줄일거야. 굳이 클리핑 안하고 바로 limb decomposition하는 식의 계획도 세우자

## 030 · assistant · 2026-10-10T09:58:53.822Z commentary

RMD 경로를 줄이면서 값을 바로 limb로 분해하는 대안을 기존 계획에 추가하겠습니다. 전체 folded code를 분해하는 방법과 필요한 값만 분해하는 방법을 비교하고, 정확도와 실행량에 미치는 영향까지 계획에 반영하겠습니다.

## 031 · assistant · 2026-10-10T10:01:12.028Z commentary

기존 분해기에 folded int32 값을 바로 넣으면 `lane 0`은 main으로 쓰고, 상위 limb만 packet에 담을 수 있습니다. 그러면 별도의 residual 배열과 앞서 계획했던 전체 int64 확장을 피할 수 있습니다.

다만 이는 RMD 준비 과정과 메모리를 줄이는 효과입니다. 모든 값을 보존하면 기존에 `8→7`로 잘리던 값에도 상위 limb가 생기므로, GEMM 횟수는 별도로 비교해야 합니다.

## 032 · user · 2026-10-10T10:01:52.737Z

<send_user_message_question_reply>
[{"answer":"선형·Q 모두 전체 값을 직접 limb 분해 (추천)","question":"클리핑을 없애는 범위는 어느 쪽으로 잡을까요? 선형에서도 모든 folded code를 바로 분해하면 기존 명세에서 잘리던 값(예: 8→7)을 보존하지만, 상위 limb 연산이 늘 수 있습니다.","questionItemId":"[\"request_user_input_async\",\"call_d846d4181ae74857aac5a04bb2042818\",0]"}]
</send_user_message_question_reply>

## 033 · assistant · 2026-10-10T10:01:58.475Z commentary

선형·Q 모두 전체 값을 직접 분해하는 것으로 확정하겠습니다. `u → main limb + 상위 limb packet`을 기본 경로로 삼고, 기존의 선택적 클리핑 명세는 정확도·실행량 비교 기준으로 남기겠습니다.

## 034 · assistant · 2026-10-10T10:07:07.333Z commentary

직접 분해식은 int32 경계값을 포함한 141,075개 값에서 원래 값을 복원하는지 확인했습니다. signed 4비트 limb는 carry 때문에 최대 9개 위치가 필요합니다.

계획에서는 공용 residual int64 확장을 제거하고, 직접 digit을 넣는 경로와 packet 자르기·해시·복원 처리를 수정하도록 바꿨습니다. 기존 A8 경로를 유지하면서 빠진 연결이 없는지 검토 중입니다.

## 035 · assistant · 2026-10-10T10:08:00.476Z final_answer

[계획에 반영했습니다.](/Users/chan/Projects/AISA/llama.cpp-gemmini/.omo/plans/potal-a4nks-linear-attention.md) **선형·Q 모두 folded 값을 바로 분해**하는 방식입니다.

```text
folded int32 u → balanced limb 분해
                 ├─ d0     → main GEMM
                 └─ d1…d8 → compact packet → 추가 GEMM
```

기존 계획에서 다음을 바꿨습니다.

- 4비트 클리핑과 별도 `u-main` 계산을 제거합니다.
- 전체 residual 배열과 scalar capture를 생략합니다. 앞서 계획한 공용 int64 확장도 필요 없어집니다.
- 기존 packet의 row pruning·K compaction을 재사용합니다.
- 준비 시간·메모리 절감과 실제 추가 GEMM 횟수를 따로 검증합니다.

Python 계산으로 **141,075개 값의 정확한 복원**을 확인했고, 수정 계획 검토도 완료했습니다.

아직 구현·성능 측정은 하지 않았습니다. Q의 fragment 수는 기존 명세와 같고, 선형은 이전에 잘리던 `+8`을 보존하면서 추가 GEMM이 늘 수 있습니다.

## 036 · user · 2026-10-10T10:16:00.216Z

양자홥ㅇ식이 달라졌나?

## 037 · user · 2026-10-10T10:16:10.151Z

리니어 레이어랑 어텐션 레이어는 다르니까?

## 038 · assistant · 2026-10-10T10:16:39.490Z final_answer

맞아요. **선형과 attention의 양자화 정책은 다르고, 공통으로 쓰는 것은 4비트 limb 분해 코드입니다.**

| 대상 | 양자화 정책 | 이번 변경 |
|---|---|---|
| 선형 X | `rho=2`, 블록 적응·E2 folding | 모든 folded 값을 보존하도록 변경 |
| Attention Q | `rho=6`, 블록 적응·E2 folding | 원래 전체 보정이므로 수치 결과 동일 |
| K | 8비트, `[-128,119]`, 두 nibble pass | 유지 |
| P | stripe별 unsigned 8비트, zero point 보정 | 유지 |
| V | 블록별 signed 4비트 | 유지 |

**선형은 수치 규칙도 일부 바뀝니다.** 원래 보정 대상이 아닌 `u=8`은 `7`로 잘렸지만, 새 계획에서는 `-8 + 16 = 8`로 보존합니다. Q는 원래부터 이렇게 전체 값을 보존했습니다.

따라서 준비 과정을 단순화하는 변경에 더해, 선형에는 클리핑 제거라는 변경이 있습니다. 현재 계획만 수정했으며, 선형의 모델 정확도·추가 연산량 변화는 아직 측정하지 않았습니다.

## 039 · user · 2026-10-10T10:18:31.825Z

좀 더 쉽게 설명해줘 양자화정책이 어떻게 달라진건지.

## 040 · assistant · 2026-10-10T10:19:00.735Z final_answer

핵심은 **선형에서 일부 값을 잘라 버리던 것을, 여러 조각으로 나눠 보존하도록 바꾼 것**입니다.

4비트 숫자 하나에는 **-8부터 7까지**만 담을 수 있습니다. 양자화한 값이 `8`이면 그대로 담을 수 없죠.

- **기존 선형:** 일부 값만 추가 계산으로 보정했습니다. 보정 대상으로 뽑히지 않은 `8`은 `7`로 잘랐습니다.
- **새 선형:** 그 `8`도 `-8 + 1×16`이라는 두 조각으로 담습니다. 계산 후 합치면 `8`이 복원됩니다. 이 조각이 **limb**입니다.
- **Attention의 Q:** 원래부터 모든 값을 조각으로 보존했습니다. 이번에는 같은 결과를 만드는 중간 과정을 줄입니다.

**선형은 비교적 거칠게, Q는 더 촘촘하게 양자화한다는 차이는 유지됩니다.** K·P·V의 양자화 규칙도 그대로입니다.

선형에서 버리는 값은 줄지만, 처리할 조각과 계산량은 늘 수 있습니다. 실제 모델 정확도와 속도 변화는 아직 측정하지 않았습니다.

## 041 · user · 2026-10-10T10:21:11.431Z

P가 더 예민했던 것으로 들었는데 그렇지 않나?

## 042 · assistant · 2026-10-10T10:21:14.782Z commentary

P가 양자화 오차에 민감하다는 점과, 지금 명세가 P를 어떻게 처리하는지 함께 확인하겠습니다.

## 043 · assistant · 2026-10-10T10:22:09.248Z final_answer

맞아요. **P는 작은 확률을 얼마나 잘 보존하느냐가 중요해서, 양자화에 민감할 수 있습니다.** 다만 항상 Q·K보다 더 민감한지는 모델과 양자화 방식에 따라 달라집니다.

P는 **“각 토큰의 정보를 얼마나 섞을지” 정하는 비중**입니다. 작은 확률들이 양자화하면서 전부 0이 되면, 그 토큰들의 기여가 사라집니다. 이런 편중된 분포 때문에 P 전용 양자화가 필요하다는 연구도 있습니다. [FQ-ViT](https://arxiv.org/abs/2111.13824)

그래서 현재 명세에서도 P는 별도로 처리합니다.

- **softmax 계산은 부동소수점으로 유지**
- **P는 unsigned 8비트로 양자화**
- 그 **8비트 값을 4비트 조각으로 나눠 계산**

즉, P를 4비트 값 하나로 줄이는 계획은 아닙니다. 다만 **조각을 합치면 8비트 코드는 정확히 복원해도, 처음 P를 8비트로 바꾸며 잃은 작은 확률은 복원하지 못합니다.**

우리 모델에서 8비트 P가 충분한지는 아직 측정하지 않았습니다. Q·K·V를 고정하고 P만 양자화해 출력 오차를 따로 확인해야 합니다.

## 044 · user · 2026-10-10T10:22:36.204Z

Q랑 P랑 양자화 방식이 같아?

## 045 · assistant · 2026-10-10T10:23:02.545Z final_answer

아니요. **Q와 P는 정수로 바꾸는 규칙이 다릅니다.** 둘 다 4비트 조각으로 계산하지만, 양자화 기준은 따로입니다.

| | Q | P |
|---|---|---|
| 원래 값 | 양수·음수가 섞인 벡터 | 0~1 사이의 확률 |
| 양자화 기준 | 32개씩 살펴 큰 값의 영향을 조정한 뒤, stripe 기준으로 맞춤 | stripe 전체의 최댓값으로 기준을 정함 |
| 큰 값 처리 | 이상치를 구분해 기준을 조정 | 이상치 선별 없음 |
| 정수 표현 | 부호 있는 정수, 필요한 만큼 limb로 보존 | unsigned 8비트 `0~255` |

쉽게 말해, **Q는 큰 값 때문에 나머지 값이 뭉개지지 않도록 조정하고, P는 확률들을 최댓값에 맞춘 일정한 간격으로 표현합니다.**

P는 음수가 없어서 4비트 계산기에 넣을 때 별도의 `-8` 이동과 보정도 필요합니다. Q의 분해 방식을 그대로 적용하지 않습니다.

현재 계획은 두 정책을 구분하며, 실제 모델에서의 오차 비교는 아직 하지 않았습니다.

## 046 · user · 2026-10-10T10:24:57.035Z

그럼 사이즈가 오히려 커진건가? 양자화해서 4비트사이즈가 아니라 8비트 사이즈거나 limb개수만큼의 사이즈로 배가 된건가?

## 047 · assistant · 2026-10-10T10:25:11.243Z commentary

4비트 연산 폭과 실제 저장 크기는 다를 수 있습니다. 현재 코드가 main과 추가 limb를 어떻게 저장하는지 확인해서 설명하겠습니다.

## 048 · user · 2026-10-10T10:25:19.004Z

그런 분석도 해봐. 원래 의미가 퇴색되는 구현사항들이 우리꺼에 있는지 추적해.

## 049 · assistant · 2026-10-10T10:25:35.233Z commentary

확인한 코드에서는 **4비트 main과 limb도 메모리에 각각 1바이트로 저장**합니다. 현재 A4는 연산 폭을 뜻하며, 값 하나를 항상 4비트 공간에 저장한다는 뜻은 아닙니다.

`repo-understanding` 스킬로 실제 경로를 추적하겠습니다. 저장 공간, 추가 GEMM, host 처리, 비용 집계를 살펴보고, 현재 구현과 새 계획에서 PoTal의 의도가 약해지는 지점을 근거와 함께 정리하겠습니다.

## 050 · assistant · 2026-10-10T10:29:32.600Z commentary

저장 공간 이득을 줄이는 지점이 확인됐습니다. 현재 A4 경로는 main을 값당 1바이트로 저장하면서, 별도로 값당 4바이트짜리 residual 배열도 유지합니다. 둘만 합쳐도 FP32 원본의 값당 4바이트보다 큽니다.

A4를 바이트 단위로 전송하는 것은 저장소 문서에 명시된 설계 선택입니다. 불필요한 중간 배열, 전송 형식의 제약, 정확도를 위해 필요한 추가 연산을 구분해서 판단하겠습니다.


## 051 · assistant · 2026-10-10T10:30:39.300Z commentary

정확도 쪽에서도 반례를 재현했습니다. 주신 P 정책으로 `T=256, d=32`, `Q=K=0`, `V=1`인 causal attention을 계산하면, 마지막 행의 확률 `1/256`이 전부 0으로 양자화되어 출력이 1에서 0으로 바뀝니다.

이때 PV의 추가 GEMM 비용은 약 1.6%뿐입니다. 따라서 RMD가 적다는 것만으로 성공을 판단하면 안 됩니다. 이 문제는 limb 구현보다 앞단인, 여러 행이 하나의 P scale을 공유하는 정책에서 발생합니다.


## 052 · assistant · 2026-10-10T10:31:41.378Z final_answer

있습니다. **4비트 PE로 계산한다는 의미는 유지되지만, 저장 공간과 추가 처리 때문에 이득이 줄어드는 부분이 확인됐습니다.** 다만 Q·K·P를 여러 limb로 계산하는 것 자체는 원래 명세에 들어 있는 설계입니다.

현재 코드에서 확인한 주요 지점입니다.

| 발견 | 실제 영향 |
|---|---|
| **A4도 원소당 1바이트 저장·전송** | 이 버퍼에서는 A8 대비 공간이 절반으로 줄지 않습니다. 문서에도 명시된 설계 선택입니다. [버퍼 구현](/Users/chan/Projects/AISA/llama.cpp-gemmini/ggml/src/ggml-gemmini/quants/act/buffer.hpp:27) |
| **main 외에 전체 int32 residual 배열 유지** | main 1바이트 + residual 4바이트로 **원소당 5바이트**입니다. 패킷·임시 버퍼까지 제외한 수치인데도 FP32의 4바이트보다 큽니다. direct-limb 계획에서 먼저 없앨 대상입니다. [할당 코드](/Users/chan/Projects/AISA/llama.cpp-gemmini/ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:447) |
| **RMD용 weight를 int32로 펼쳤다가 int8로 전달** | `compactK=1024, N=4096`이면 중간 weight 배열만 16MiB입니다. 같은 코드를 int8로 담으면 4MiB입니다. **host 준비 공간의 낭비**이며, 실제 PE가 32비트라는 뜻은 아닙니다. [준비 코드](/Users/chan/Projects/AISA/llama.cpp-gemmini/ggml/src/ggml-gemmini/residual/rmd/rmd-run-aware.cpp:182) |
| **희소한 limb도 타일 크기로 패딩** | DIM32에서는 digit 하나만 남아도 패킷이 32×32, 즉 1,024바이트가 될 수 있습니다. BK=DIM32라면 블록 안에서 열을 32개→1개로 줄여도 GEMM fragment는 하나입니다. [패딩 코드](/Users/chan/Projects/AISA/llama.cpp-gemmini/ggml/src/ggml-gemmini/residual/rmd/rmd-bitmap-builder.cpp:332) |
| **현재 일반적인 multi-head attention은 기존 Gemmini 경로에 들어가지 못함** | shared-weight 조건이 head 차원과 비연속 view를 거부합니다. 현재 선형 레이어의 결과를 attention까지 가속한 결과로 해석하면 안 됩니다. [지원 조건](/Users/chan/Projects/AISA/llama.cpp-gemmini/ggml/src/ggml-gemmini/ops.cpp:557) |

그래서 **limb를 늘리면 FP16보다 커질 수도 있습니다.** 1,024개 값을 모두 저장한다고 하면, 메타데이터 제외 기준으로:

- packed 4비트: **512바이트**
- 현재 방식의 main 한 장: **1,024바이트**
- main + 상위 limb 한 장: **2,048바이트**, FP16과 같음
- main + 상위 limb 두 장: **3,072바이트**, FP16보다 큼

추가 limb가 충분히 희소하고, **패딩 후에도 작아야** 공간과 연산 이득이 남습니다. 원본 Q4 weight의 nibble packing은 유지되고 있으므로 모델 weight 전체가 이런 크기로 바뀐 것은 아닙니다.

**더 중요한 발견은 P의 정확도 반례입니다.** 주신 명세를 작은 계산으로 재현했습니다.

`T=256, d=32, Q=K=0, V=1`인 causal attention에서는 원래 출력이 모두 1입니다. 그런데 명세의 stripe 배치에서는 첫 행의 `P=1`과 마지막 행의 `P=1/256`이 같은 scale을 공유합니다.

```text
마지막 행: q = round_half_even((1/256) × 128)
             = round_half_even(0.5)
             = 0
결과: 출력 1 → 0
```

이때 **PV의 추가 fragment 비용은 약 1.6%뿐**입니다. RMD 비용은 작지만 결과는 크게 틀릴 수 있습니다. 이는 원래 P 양자화 정책의 한계이며, 이미 0이 된 확률은 정확한 limb 분해로도 복구되지 않습니다.

[현재 계획](/Users/chan/Projects/AISA/llama.cpp-gemmini/.omo/plans/potal-a4nks-linear-attention.md)에도 남는 비용이 있습니다. FP KV cache와 FP softmax를 유지하고, 정확한 누적을 위해 host에서 scale 적용과 합산을 합니다. 따라서 이 계획만으로 **KV 메모리 절감이나 모든 누적의 하드웨어 실행**까지 얻지는 못합니다.

우선순위는 **P 정확도 반례 검증과 scale 정책 재검토 → 전체 residual 배열 제거 → int32 weight 중간 배열 축소 → 실제 메모리·전송량·지연시간 측정**으로 잡는 게 맞습니다. 직접 limb 분해는 중간 처리를 줄이지만, 추가 GEMM까지 항상 줄여주지는 않습니다.

소스 추적과 작은 수치 반례까지 확인했습니다. 실제 모델 PPL, 최대 메모리, Gemmini 실행시간은 아직 측정하지 않았습니다.

## 053 · user · 2026-10-10T10:34:07.880Z

직접 한번 작업 해보고 공간은 얼마나 이득이거나 손해인지, 정확도는 어는정도 바뀔지 제한상황 걸어가며 측정해.

## 054 · assistant · 2026-10-10T10:34:16.688Z commentary

직접 구현하고 제한 조건을 바꿔가며 측정하겠습니다. 기존 방식과 direct-limb 방식의 저장량·실행 중 최대 메모리·수치 오차를 비교하고, P의 scale 공유 범위도 별도로 실험하겠습니다. 재현 가능한 결과와 적용 가능한 조건이 나올 때까지 진행하겠습니다.


## 055 · assistant · 2026-10-10T10:35:11.862Z commentary

로컬에 GPT-2 모델과 기존 tensor 수집 도구가 있어, 인공 행렬뿐 아니라 실제 모델 입력에서도 비교할 수 있습니다. `ponytail`·`programming` 스킬로 실험 구현을 작게 유지하고, `mandela` 기준에 따라 원본 모델 출력을 독립 비교 기준으로 두겠습니다. 원소별 오차와 최종 perplexity는 따로 측정하겠습니다.


## 056 · assistant · 2026-10-10T10:37:45.426Z commentary

실험용 quantizer와 실제 바이트 배열을 만드는 코드까지 작성했습니다. 다음으로 동일한 GPT-2 weight와 텍스트를 고정해, P의 bit 수·scale 공유 행 수·허용 limb 수를 바꿔 실행하겠습니다.

자동 주석 검사에서 지적한 불필요한 모듈 설명 네 개는 `comment-checker` 지침에 따라 제거하겠습니다.


## 057 · assistant · 2026-10-10T10:39:16.209Z commentary

실행 파일의 의존성 선언과 실행 명령은 결과 재현에 필요한 정보라 유지하겠습니다. 비교 조건은 개발용 텍스트에서 먼저 좁히고, 선택한 조건은 별도 test 텍스트에서 다시 확인하겠습니다.


## 058 · assistant · 2026-10-10T10:41:34.468Z commentary

첫 실제 모델 실행에서는 동일한 Q4 weight 기준으로, 선형 입력까지 direct-limb 양자화했을 때 PPL이 26.48에서 31.34로 올랐습니다. 아직 개발용 254개 예측 토큰의 예비 결과입니다.

현재는 attention까지 포함한 12개 조건을 실행 중입니다. Python 배열의 크기와 별개로, 기존 C++ RMD packet builder에서도 메모리를 직접 할당해 비교하겠습니다.


## 059 · assistant · 2026-10-10T10:44:58.410Z commentary

개발용 조건 비교에서는 두 가지 제한이 특히 손해였습니다. 상위 limb를 하나로 제한하자 PPL이 52.40으로 올랐고, P를 4비트로 줄인 조건에서는 확률이 전부 0이 된 행이 13개 나왔습니다.

P의 scale을 8행씩 공유하는 조건은 원래 stripe 공유보다 PPL이 낮았습니다. 다만 작은 행 묶음마다 타일 패딩이 붙으므로, 공간과 GEMM 비용까지 합쳐서 판단하고 있습니다.


## 060 · assistant · 2026-10-10T10:48:38.064Z commentary

별도 test 텍스트 2,040개 예측 토큰에서도 확인했습니다. 원래 a4nks는 PPL 35.81, direct-limb와 원래 P stripe 조합은 37.10, P를 8행씩 묶은 조합은 35.71입니다. 마지막 두 정책의 작은 차이는 표본 오차도 함께 표시하겠습니다.

C++ 측정에서는 전체 residual 배열을 없애는 것만으로 512×4096 입력당 8MiB가 줄었습니다. 다만 limb가 드물게 남는 경우에는 패딩 때문에 digit 개수가 줄어도 패킷 크기는 그대로였습니다.


## 061 · assistant · 2026-10-10T10:52:32.482Z commentary

최악의 저장 조건도 확인했습니다. 모든 값에 9개 digit이 필요한 경우에는 중간 residual 배열을 없애도 입력 패킷이 19.72MiB로, 같은 값의 FP32 저장량 8MiB보다 컸습니다.

최종 비교에서는 모델에 원래 FP16으로 저장된 출력층을 보존하고, Gemmini 대상인 Q4 선형층만 바꾸도록 조건을 맞추고 있습니다. 앞서 얻은 예비 PPL에는 이 출력층의 입력 양자화도 포함돼 있었으므로, 최종 수치는 다시 집계하겠습니다.


## 062 · assistant · 2026-10-10T11:01:53.324Z commentary

최종 26개 조건의 측정이 끝났습니다. 기존 명세의 PPL은 35.38, 전체 값을 직접 limb로 분해하면 36.55였습니다. P의 스케일을 8행마다 잡으면 35.36까지 회복됐지만, PV 연산 조각 수가 약 3배 늘었습니다. 저장 공간과 정확도를 함께 비교한 결과를 정리하고 있습니다.

## 063 · assistant · 2026-10-10T11:02:57.024Z commentary

limb 개수를 강제로 줄이는 쪽은 정확도 손실이 컸습니다. 선형·Q의 상위 limb를 1개로 제한하자 PPL이 89.74로 올라갔고, P를 4비트로 줄인 개발용 실험에서는 확률 행 13개가 전부 0이 됐습니다. 현재 결과로는 limb를 버리기보다 중간 int32 residual 배열부터 없애는 것이 안전한 절감 방향입니다.

## 064 · assistant · 2026-10-10T11:05:11.597Z final_answer

실험 구현과 측정을 마쳤습니다. **중간 residual 배열을 없애는 효과는 컸지만, limb 개수를 줄이면 정확도가 크게 나빠졌습니다.**

공간은 실제 C++ 패킷으로 측정했습니다.

| 입력 조건 | 기존 → direct 저장량 |
|---|---:|
| 드문 큰 값 | 12.09 → **4.08 MiB, 66% 감소** |
| 전체가 8비트 범위의 큰 값 | 17.09 → **6.96 MiB** |
| 9개 digit이 필요한 극단값 | 27.72 → **19.72 MiB** |

마지막 조건은 같은 원소 수의 **FP32 배열보다 2.47배 큽니다.** 실제 모델에서도 Q 패킷은 평균 **3.19 bytes/값**이어서, 4비트 PE 사용이 4비트 저장을 뜻하지 않았습니다.

GPT-2의 동일한 Q4 weight로 비교한 정확도입니다. PPL은 낮을수록 좋습니다.

| 정책 | PPL |
|---|---:|
| 기존 명세 | **35.38** |
| 전체 값 direct limb 분해 | 36.55 |
| direct + P 스케일을 8행마다 설정 | **35.36** |
| 위 조건에서 P를 6비트로 축소 | 38.00 |
| 선형·Q 상위 limb를 1개로 제한 | **89.74** |

P 스케일을 세분화하면 정확도가 회복됐지만 **PV 연산 조각 수가 약 3배** 늘었습니다. 따라서 우선은 **필요한 limb를 유지하면서 중간 int32 residual 배열을 제거하는 방향**이 타당합니다.

26개 모델 조건과 128개 합성 제약 실험, 재현 코드·원자료를 [결과 보고서](/Users/chan/Projects/AISA/worktrees/potal-attn/experiments/potal_a4/RESULTS.md)에 정리했습니다.

정확도는 CPU 실험값입니다. 제품 Gemmini 경로 통합과 실제 하드웨어 속도·전력, native Q4 전체 추론 일치는 아직 검증하지 않았습니다.

## 065 · user · 2026-10-10T11:09:41.306Z

더 쉽게 정리

## 066 · assistant · 2026-10-10T11:10:02.446Z final_answer

핵심은 **“숫자를 덜 저장하기”와 “불필요한 중간 복사본을 없애기”를 구분해야 한다**는 겁니다.

- **중간 복사본 없애기:** 같은 값을 유지하면서 공간을 줄일 수 있었습니다. 큰 값이 드문 실험에서는 **약 66% 절약**됐습니다.
- **숫자를 구성하는 조각 줄이기:** 공간은 줄지만 값이 부정확해져서 **모델 성능이 크게 나빠졌습니다.**
- **어텐션의 확률 P를 더 세밀하게 처리하기:** 정확도는 회복됐지만, 현재 구조에서는 **PV 계산량이 약 3배** 늘었습니다.

그리고 지금은 **4비트 조각으로 계산하되, 조각 하나를 메모리에서는 1바이트로 저장**합니다. 큰 숫자는 조각도 여러 개 필요합니다. 그래서 경우에 따라 원래 FP32보다 커질 수도 있습니다.

**지금 할 일은 필요한 조각은 유지하고, 중간 복사본부터 없애는 것입니다.** 조각 개수를 억지로 줄이는 것은 손해가 컸습니다.

실제 Gemmini에서의 속도와 전체 모델 메모리 절감량은 아직 확인하지 않았습니다.

## 067 · user · 2026-10-10T11:13:25.486Z

- **숫자를 구성하는 조각 줄이기:** 공간은 줄지만 값이 부정확해져서 **모델 성능이 크게 나빠졌습니다.** 이거 이유?

- **어텐션의 확률 P를 더 세밀하게 처리하기:** 정확도는 회복됐지만, 현재 구조에서는 **PV 계산량이 약 3배** 늘었습니다. 이게 는 이유?

## 068 · assistant · 2026-10-10T11:14:08.612Z final_answer

**1. 조각을 줄이면 왜 정확도가 떨어졌나?**

조각은 **숫자의 자릿수**입니다. 큰 값을 표현하려면 높은 자리도 필요합니다.

현재 방식에서 조각 2개로 표현할 수 있는 양수의 최대는 다음과 같습니다.

`7 + 7 × 16 = 119`

따라서 조각을 2개로 제한하면 **127 같은 값도 119로 깎입니다.** 실제 실험에서는 Q 값의 약 **15.5%가 표현 범위를 넘어 깎였습니다.**

Q는 “어떤 단어를 얼마나 참고할지” 결정하는 데 쓰입니다. Q가 변하면 `QK → 어텐션 확률 → 출력`이 함께 달라집니다. 필요한 높은 자리를 없앴으니 모델 성능도 크게 나빠진 것입니다.

**2. P를 세밀하게 처리하면 왜 계산량이 늘었나?**

원인은 **32행짜리 계산 타일에 빈칸이 늘었기 때문**입니다.

| 처리 방식 | 실제 데이터 | 계산에 잡히는 공간 |
|---|---:|---:|
| 32행을 함께 처리 | 32행 | 32행 타일 1개 |
| 8행씩 따로 처리 | 8행 × 4묶음 | 32행 타일 4개 |

P의 스케일을 8행마다 따로 잡으면 작은 확률을 더 잘 보존합니다. 그런데 현재 실험 방식은 각 묶음을 따로 처리하므로, **8행만 들어 있어도 32행 타일 하나를 차지**합니다.

그래서 main 계산은 4배가 됐고, 보정 계산까지 합친 전체 PV 작업량은 **약 3배**가 됐습니다.

즉, **조각 제한은 정보를 잃는 문제이고, P 세분화는 타일을 비효율적으로 채우는 문제**입니다.

3배는 타일 수로 산출한 작업량입니다. 실제 실행시간이 3배인지는 아직 확인하지 않았습니다.

## 069 · user · 2026-10-10T11:15:43.326Z

기존에는 어떻게 했는데?

## 070 · assistant · 2026-10-10T11:15:45.932Z commentary

기존 코드와 처음 주신 명세를 나눠 확인하겠습니다. 앞서 말한 “기존”은 비교 항목에 따라 기준이 달랐습니다.

## 071 · user · 2026-10-10T11:16:01.268Z

내가 처음에 준 그 코드랑 프롬프트를 기준으로 어떻게 처리하라고 되어있었길래 이렇게 된거지?

## 072 · assistant · 2026-10-10T11:16:35.930Z final_answer

**처음 주신 명세에는 limb를 2개로 제한하거나 P를 8행씩 나누라는 지시가 없었습니다.** 앞서 말한 **PPL 89.74와 PV 계산량 3배는 제가 추가한 제약 실험에서 나온 결과**입니다. 이 구분을 명확히 설명하지 못했습니다.

원문은 이렇게 되어 있습니다.

| 대상 | 원래 지시 |
|---|---|
| 선형 입력 | 일반 값은 4비트 범위로 자르고, 이상치나 스케일 조정으로 커진 값은 상위 limb로 보정 |
| Q | `correct_all=true`: **모든 값을 필요한 limb만큼 분해해서 보존** |
| P | **stripe 전체에서 스케일 하나**를 정하고, unsigned 8비트로 양자화 |

**limb 제한은 원문에 없었습니다.**

원문의 `digits(r)`는 남은 값이 없어질 때까지 필요한 자릿수를 만듭니다. 그런데 “공간을 더 줄이면 어떨까?”를 시험하면서 제가 **선형·Q의 상위 limb를 최대 1개로 제한**했습니다. 표현하지 못하는 큰 값이 깎이면서 정확도가 무너진 것입니다. 원문대로 필요한 limb를 유지하는 경우와 구분해야 합니다.

**P의 8행 분할도 추가 실험이었습니다.**

원문은 `stripe_rows(...)`가 정한 묶음 전체에 스케일 하나를 씁니다. 이번 입력 크기에서는 **128행이 한 묶음**입니다.

- **원문:** 128행에 스케일 하나 → 32행 타일 **4개**
- **추가 실험:** 8행마다 별도 스케일 → 16묶음이 각각 타일 하나를 차지 → **16개**

작은 확률을 더 잘 보존하려고 스케일을 세분화했는데, 타일의 빈칸이 늘었습니다. main은 4배, 보정까지 합친 PV 작업량은 약 3배가 됐습니다.

사용자님이 선택한 **“선형도 전체 값을 직접 limb 분해”**는 별도 변경입니다. 이는 기존에 잘리던 값까지 보존하자는 것이며, **limb를 잘라 버리자는 변경은 아닙니다.**

이번 수치는 CPU 정책 실험이며, 원문 명세 그대로의 Gemmini 하드웨어 실행은 아직 검증하지 않았습니다.

## 073 · user · 2026-10-10T11:20:06.927Z

정확도가 더 떨어지면 안 되는디?

## 074 · assistant · 2026-10-10T11:20:09.941Z commentary

맞습니다. 정확도가 떨어지는 안은 채택하면 안 됩니다. 앞으로는 **기존 명세의 양자화 결과를 유지하면서 저장 방식만 바꾸는 것**을 기준으로 잡겠습니다. direct 분해 때문에 선형의 값이 달라지는 부분과, 값은 그대로 두고 중간 배열만 없앨 수 있는 부분을 확인하겠습니다.

## 075 · assistant · 2026-10-10T11:21:22.590Z commentary

선형의 “모든 값 보존”도 원문과 결과가 달라집니다. 원문이 7로 자르던 값을 8로 살리는 경우가 있는데, 이 변화가 모델 정확도까지 좋아지게 하지는 않았습니다. 이 변경도 보류하고, 원문의 값과 스케일을 그대로 유지하는 최적화만 통과하도록 계획을 수정하겠습니다.

## 076 · assistant · 2026-10-10T11:23:13.384Z final_answer

맞습니다. **처음 주신 명세보다 정확도가 떨어지는 변경은 채택하지 않겠습니다.**

그래서 기준을 이렇게 수정했습니다.

- **선형·Q·K·P·V의 양자화 정책은 원문 그대로 유지**
- **필요한 limb는 전부 보존**
- **중간 residual 배열과 불필요한 복사만 제거**
- 최적화 전후 **양자화된 값과 모델 출력이 같은지 검증한 뒤 채택**

선형의 무조건적인 전체 값 보존, limb 개수 제한, P의 8행 분할은 채택하지 않은 실험으로 남깁니다. PPL이 우연히 비슷하게 나온 것만으로도 통과시키지 않겠습니다.

이 조건을 [구현 계획](/Users/chan/Projects/AISA/llama.cpp-gemmini/.omo/plans/potal-a4nks-linear-attention.md)에 반영했습니다. 제품 Gemmini 경로에서의 동일성 검증은 아직 남아 있습니다.

## 077 · user · 2026-10-10T11:25:50.486Z

그렇게 한다면 얼마나 늦어지고 나빠지지?

## 078 · assistant · 2026-10-10T11:25:53.715Z commentary

원문 정확도를 유지하는 안이 기존보다 느려지거나 커지는지 확인하겠습니다. 먼저 저장량과 연산 조각 수를 비교하고, CPU에서 측정할 수 있는 패킷 준비 시간도 확인하겠습니다.

## 079 · user · 2026-10-10T11:26:26.941Z

attention에 대해서 limb가 얼마나 생기는지도 궁금해. 그리고 여기서 아웃라이어 셀렉션이랑 compensation하는 비용이 얼마나 비중이 큰지도 걱정되고. measure해줘.

## 080 · assistant · 2026-10-10T11:26:44.142Z commentary

어텐션의 Q·K·P·V별로 필요한 limb 수와 패딩 후 작업량을 집계하겠습니다. CPU에서는 이상치 선택, limb 패킷 생성, 보정 연산에 걸리는 시간을 나눠 재겠습니다. 정확도는 원문 정책으로 고정하고, 실제 측정 시간과 Gemmini 타일 수로 계산한 작업량을 구분해 보고하겠습니다.

## 081 · assistant · 2026-10-10T11:29:28.226Z commentary

Q에서는 모든 값을 보정하므로, 이상치로 선택된 값만 limb가 생기는 것은 아닙니다. 작은 값에도 높은 자릿수가 필요하면 보정 연산이 생깁니다. 이 둘을 구분해서, 실제 GPT-2 입력의 limb 분포와 `이상치 선택 → 패킷 준비 → 보정 곱셈·합산` 시간을 재고 있습니다.

## 082 · assistant · 2026-10-10T11:34:01.930Z commentary

512토큰 측정에서는 Q 값의 약 91%가 상위 limb를 필요로 했습니다. Q는 main 외에 두 개의 상위 limb 면이 만들어졌고, K의 두 번째 패스까지 합치면 QK 작업량은 단일 4비트 곱셈의 6배였습니다. CPU 측정에서는 이상치 선택보다 패킷 생성과 보정 곱셈의 비중이 훨씬 컸습니다.

## 083 · assistant · 2026-10-10T11:37:39.418Z commentary

P에서는 상위 limb가 필요한 값이 약 0.96%였지만, 패딩된 보정 패킷에서 실제 0이 아닌 부분은 1.69%뿐이었습니다. 희소한 값을 타일에 담으면서 공간과 연산이 많이 늘어나는 것으로 확인됐습니다. 같은 CPU 재생 측정에서 이상치 선택은 측정 구간 합계의 약 0.65%, 상위 limb 보정 곱셈·합산은 약 51%였습니다.

## 084 · assistant · 2026-10-10T11:39:47.098Z final_answer

측정했습니다. **이번 CPU 측정에서는 이상치를 찾는 비용보다, limb 보정 곱셈과 패킷 생성 비용이 훨씬 컸습니다.** 원문 정책을 유지한 GPT-2, 512토큰 기준입니다.

| 대상 | 실제 limb 구성 |
|---|---|
| **Q** | main + 상위 2개. **값의 약 91%에 상위 limb가 필요** |
| K | low/high **2개를 항상 계산** |
| **P** | main + 상위 최대 2개. 상위 limb가 필요한 값은 **약 0.96%** |
| V | main 하나 |

Q는 이상치로 선택된 값이 약 12.8%인데, 상위 limb는 91%의 값에 필요했습니다. **Q의 보정은 상당히 큰 작업입니다.**

CPU에서 단계를 나눠 실행한 비중은 다음과 같습니다.

| 작업 | 측정 구간 내 시간 비중 |
|---|---:|
| 이상치 선택 관련 처리 | **0.65%** |
| Q/P limb 패킷 생성 | **6.66%** |
| 상위 limb 보정 곱셈·합산 | **약 51%** |
| P의 zero-point 보정 덧셈 | **0.11%** |

P에서는 패딩 손해도 컸습니다. **보정 패킷의 실제 유효 digit은 1.69%이고 나머지는 0**이었습니다. 희소한 값을 타일에 담으면서 비용이 늘어난 겁니다.

원문 대비 정확도를 유지하는 최적화는 같은 패킹 조건에서 **추가 GEMM이 필요하지 않습니다.** 다만 원문 자체가 이미 단일 4비트 main 대비 **QK 6배, PV 1.61배의 연산 조각**을 사용하고 있었습니다.

[측정 방법·원자료·상세 결과](/Users/chan/Projects/AISA/worktrees/potal-attn/experiments/potal_a4/ATTENTION_PROFILE.md)

시간 비중은 CPU의 NumPy int64 재생 측정입니다. 실제 Gemmini의 속도 변화나 비용 비중은 아직 측정하지 않았습니다.

## 085 · user · 2026-10-10T12:05:30.092Z

음 이거를 im2p로 돌려보고싶은데(gemmini) nano로 이동해서 하고싶어. 혹시 지금 hotfix/potal-attn 원격에 올리고 지금 우리가 이야기 한 내용을 nano에서 이어서 이야기하고싶어. 가능하니?

## 086 · assistant · 2026-10-10T12:05:32.886Z commentary

가능합니다. `hotfix/potal-attn`의 변경 사항과 Git 원격을 확인한 뒤, nano에서 이어갈 수 있도록 실험 결과·현재 결정·남은 작업을 문서로 정리하고 커밋해서 푸시하겠습니다.

## 087 · user · 2026-10-10T12:07:30.589Z

아니 potal-attn올리고 거기서 하나뺴서 거기에 우리 이야기한거 이어서 하도록옮겨줘.

## 088 · assistant · 2026-10-10T12:07:39.941Z commentary

알겠습니다. `hotfix/potal-attn`을 먼저 올리고, **거기서 nano·im2p 작업용 브랜치를 따로 만들어** 지금까지의 결정·측정 결과·다음 작업을 이어받도록 정리하겠습니다. Git 작업에는 `omo:git-master` 스킬을 적용합니다.

## 089 · assistant · 2026-10-10T12:08:38.655Z commentary

분기는 `hotfix/potal-attn` → `work/potal-attn-im2p-nano`로 잡았습니다. 기존 실험을 먼저 보존하고, 새 브랜치에는 **원문 대비 정확도 유지 조건과 아직 구현하지 않은 Gemmini 작업**을 명확히 적겠습니다. 측정 코드와 결과는 약 1.1MB라 함께 올릴 수 있습니다.

## 090 · assistant · 2026-10-10T12:09:42.379Z commentary

Python 정수 복원 141,076건과 C++ 테스트 4개가 다시 통과했고, 보관된 코드·결과의 해시 57개도 일치합니다. 이제 실험 브랜치를 올린 뒤 새 브랜치에 인계 문서를 넣겠습니다. 모델 파일과 대용량 캡처는 Git에 포함되지 않으므로, nano에서 필요한 파일과 재생성 방법도 적고 있습니다.

## 091 · assistant · 2026-10-10T12:10:47.261Z commentary

`hotfix/potal-attn` 푸시는 완료했습니다(`ba762b9`). 이제 이 커밋에서 `work/potal-attn-im2p-nano`를 만들고 있습니다. nano의 새 대화에서 바로 이어갈 수 있도록 구현 계획과 읽을 문서 순서, 첫 실행 절차를 넣겠습니다.

## 092 · assistant · 2026-10-10T12:15:03.440Z commentary

새 브랜치는 만들었습니다. 인계 문서에는 원래 알고리즘과 최신 정확도 조건을 함께 넣고, 기존 측정과 nano에서 새로 검증할 항목을 구분했습니다. 특히 실제 IM2P 실행 경로 확인, Linux 메모리 단위 처리, 논리 stripe와 하드웨어 tile 비교를 첫 작업으로 남깁니다.

## 093 · assistant · 2026-10-10T12:17:33.168Z final_answer

완료했습니다. **`hotfix/potal-attn`을 푸시하고, 거기서 `work/potal-attn-im2p-nano`를 분기해 함께 올렸습니다.**

- `cbb5dfc`: 양자화 정책·공간 실험
- `ba762b9`: attention limb·보정 비용 측정
- `dc71531`: nano 인계 문서·원래 명세·구현 계획

nano의 저장소에서:

```sh
rtk git fetch origin
rtk git switch --track origin/work/potal-attn-im2p-nano
```

새 대화에는 이렇게 말하면 됩니다:

> `POTAL_NANO_HANDOFF.md`를 읽고 이어서 작업하자. 원문 대비 정확도를 유지하면서 Gemmini/IM2P 구현과 측정을 진행해.

[인계 문서 열기](https://github.com/ajou-aisa/llama.cpp-gemmini/blob/work/potal-attn-im2p-nano/POTAL_NANO_HANDOFF.md)에 결정 사항과 다음 작업을 담았습니다. 작업 브랜치는 미커밋 변경 없이 정리됐습니다.

모델·대용량 캡처는 제외했습니다. nano에서의 실제 Gemmini 실행은 아직 진행하지 않았습니다.

## 094 · user · 2026-10-10T13:41:24.741Z

우리 대화 세션을 보내면 되는거 아닌가?

## 095 · assistant · 2026-10-10T13:41:28.984Z commentary

대화 세션 자체를 nano로 옮기자는 뜻이었군요. 현재 세션의 저장 위치와 다른 기기에서 이어 여는 방법을 확인하겠습니다.

## 096 · assistant · 2026-10-10T13:41:43.273Z commentary

제가 “대화를 이어간다”를 인계 문서로 해석했습니다. 세션 기록을 그대로 가져갈 수 있는지 `omo:coding-agent-sessions`와 OpenAI Docs 스킬로 확인하겠습니다.

## 097 · assistant · 2026-10-10T13:44:12.247Z commentary

맞습니다. **이 대화의 세션 기록을 nano로 옮겨 `codex resume`으로 이어 여는 방식**이 가능합니다. 원본 기록도 찾았습니다. 앞서는 코드와 문서만 올렸고, 세션 자체는 아직 전송하지 않았습니다. `nano2`가 맞는지 확인하는 동안 복사해야 할 기록 범위를 확인하고 있습니다.

## 098 · assistant · 2026-10-10T13:45:17.940Z final_answer

맞아요. **대화 세션 자체를 nano로 옮기면 됩니다.** 제가 앞서는 인계 문서로만 처리했습니다.

현재 세션 원본을 찾았습니다. 세션 데이터를 옮긴 뒤 nano의 작업 폴더에서 이렇게 이어 열 수 있습니다. [공식 resume 문서](https://learn.chatgpt.com/docs/cli/reference#codex-resume)

```sh
rtk proxy codex resume 01a12521-397c-7d92-b5bf-c38ac78e1886 -C .
```

아직 세션은 전송하지 않았습니다. 보내려는 nano가 SSH 설정의 **`nano2`**인지 확인이 필요합니다.

## 099 · user · 2026-10-10T14:08:41.269Z

그것도 그냥 옮길 애들이 있는 곳에 커밋해버리면 되는거 아닌가?
