# GPT-2: 큰 activation과 HP1 weight 오차

분석일: 2026-10-09. 대상: GPT-2 124M, Llama-3.2-1B.

**결론:** GPT-2의 큰 activation은 FP16 원본에도 있다. 마지막 두 transformer block에서 크게 증가하고, 마지막 LayerNorm의 학습된 배율이 소수 feature에 에너지를 더 집중시킨다. 이 feature에 연결된 LM-head weight는 원본에서 vocabulary 전체에 거의 일정하다. HP1 양자화는 이 일정한 성분을 토큰마다 다른 오차로 바꿀 수 있고, 큰 activation이 그 오차를 logits에 크게 반영한다. 고정 입력 실험에서 이 민감도를 확인했다. 현재 Full PoTal의 전체 PPL 손실 중 몇 %가 이 원인인지는 측정하지 않았다.

산출물: [2페이지 PDF](figures/activation-amplification.pdf), [레이어 추적 PNG](figures/activation-layer-trace.png), [보정 비교 PNG](figures/activation-head-corrections.png), [전체 수치·모델 SHA·실행 설정](data/activation-amplification.json), [보정 실험 CSV](data/activation-head-corrections.csv), [분석 코드](analyze_activation_amplification.py), [CPU 캡처 코드](capture-layer-channels.cpp).

## 1. 세 채널은 어느 레이어인가?

**496·430·36번은 GPT-2의 768차원 hidden vector 안의 feature 좌표다.** 첫 세 레이어, 0·1·2번 채널, attention head 번호를 뜻하지 않는다. 앞서 말한 순서는 activation 에너지 순위다.

Residual addition은 같은 좌표끼리 더하므로 이 좌표를 block 사이에서 추적할 수 있다. 그러나 QKV, attention output, FFN projection은 feature를 섞는다. `496 / 64`로 attention head 번호를 추정하면 안 된다. 각 layer의 weight는 고정돼 있고 layer마다 다르다. Layer를 지나면서 달라지는 값은 activation이다. 여기서 activation은 GEMM에 들어가는 hidden value이며, GELU·SiLU 같은 activation 함수만을 뜻하지 않는다.

### FP16 원본에서 직접 확인한 증가 위치

WikiText-2 첫 512 tokens, CPU 1 thread, FP16 weight 원본. 아래는 token 집합에서의 채널 RMS다. `blk.0`은 첫 번째, `blk.11`은 12번째이자 마지막 block이다.

| 위치 | Feature 496 | Feature 430 | Feature 36 |
|---|---:|---:|---:|
| Token + position embedding | 0.527 | 0.547 | 0.459 |
| `blk.0` 출력 | 1.220 | 1.146 | 1.220 |
| `blk.8` 출력 | 5.347 | 9.896 | 5.122 |
| `blk.9` 출력 | 9.190 | 14.698 | 8.441 |
| **`blk.10` 출력** | **47.357** | **28.663** | **22.921** |
| **`blk.11` 출력** | **167.137** | **89.397** | **83.865** |
| Final LN: gamma/beta 적용 전 | 9.628 | 5.257 | 4.965 |
| Final LN: gamma/beta 적용 후, head 입력 | 174.768 | 93.311 | 72.221 |

Branch별 token 평균 증가량도 분리했다. RMS 증가량과는 다른 통계다.

| Block | Feature | Attention이 더한 평균 | FFN이 더한 평균 |
|---|---:|---:|---:|
| `blk.10` | 496 | +5.404 | **+31.945** |
| `blk.10` | 430 | +3.823 | +9.712 |
| `blk.10` | 36 | +2.402 | +9.687 |
| `blk.11` | 496 | **+96.309** | +16.353 |
| `blk.11` | 430 | +32.763 | +17.597 |
| `blk.11` | 36 | +35.009 | +15.527 |

Attention 평균은 `mean(ffn_inp) - mean(previous residual)`로 구했고, FFN 평균은 직접 캡처했다. 둘을 더하면 실제 residual 평균 증가와 일치한다. Branch 사이의 covariance나 PPL 기여도를 측정한 것은 아니다.

Q4_0 body + 원본 F16 embedding/head 대조군에서도 496번 RMS가 `blk.9: 9.490 → blk.10: 47.432 → blk.11: 168.285`로 비슷하게 증가했다. **큰 값 자체가 양자화 오차 때문에 생겼다는 가설은 FP16 대조 실행으로 배제된다.** 학습 과정에서 왜 이 표현을 획득했는지나 feature의 의미를 규명한 것은 아니다.

### 마지막 LayerNorm

`h = gamma * (x - mean(x)) / sqrt(var(x) + epsilon) + beta`

GPT-2 gamma의 중앙값은 **1.251**인데, 세 feature의 gamma는 **17.419, 16.994, 13.804**, beta는 **7.368, 4.576, 4.504**다. 정규화 후 gamma/beta를 적용하면 모든 채널의 분산이 1이어야 하는 것은 아니다.

새 FP16 512-token 실행에서 이 세 feature의 에너지 비중은 affine 전 **18.878%**, affine 후 **95.378%**였다. 기존 8,192-row 캡처에서는 **23.944% → 96.635%**였다. 마지막 LN은 모든 feature를 같은 비율로 키우는 것이 아니라 이미 큰 feature에 특히 큰 배율을 적용한다.

Llama의 gamma는 중앙값 **2.453**, 최댓값 **2.906**이다. 기존 캡처의 최상위 세 feature 에너지 비중은 **1.346%**였다. RMSNorm에도 gamma와 채널 간 coupling이 있으므로 RMSNorm이면 이 문제가 없다는 뜻은 아니다. 이번 checkpoint의 학습된 계수와 분포가 다르다는 관측이다. LN gamma가 outlier를 확대하는 현상은 [Outlier Suppression, NeurIPS 2022](https://arxiv.org/abs/2209.13325)에서도 연구됐다.

## 2. Weight 오차와 activation 오차를 구분

Head weight를 `[vocabulary, hidden]`인 W, 입력을 H, 복원한 양자화 weight를 `Wq = W + E`라고 두면:

```text
원본 Z = H W^T
고정 H에서의 출력 오차 = H Wq^T - H W^T = H E^T
Feature c가 만드는 오차 = H[:, c] E[:, c]^T
```

입력이 1이면 weight 오차 0.015가 약 0.015의 출력 차이를 만들고, 입력이 200이면 약 3을 만든다. **H 자체에 양자화 오차가 없어도** 발생한다. 입력도 `Hq = H + D`로 바뀐다면 전체 오차는 `H E^T + D W^T + D E^T`다. 이번 head replay는 H를 고정하여 첫 항을 분리했다.

Layer 간 D의 전달은 국소적으로 대략 `(I + J_FFN)(I + J_Attention) D`와 새 오차의 주입으로 표현할 수 있다. Jacobian과 오차 방향에 따라 확대·축소가 모두 가능하다. 이번 결과를 모든 layer에서 오차가 단조롭게 증폭된다는 증거로 쓰면 안 된다.

## 3. GPT-2에서 특히 민감한 이유

큰 activation만으로는 충분한 설명이 아니다. GPT-2의 해당 head weight 열은 vocabulary 방향으로 매우 일정하다.

| Feature | FP16 weight 평균 | FP16 표준편차 | HP1 INT4 표준편차 | 양자화 오차 표준편차 |
|---|---:|---:|---:|---:|
| 496 | -0.205954 | **0.000844** | **0.015648** | 0.015459 |
| 430 | -0.256477 | 0.001264 | 0.000000 | 0.001264 |
| 36 | -0.316993 | **0.001787** | **0.011865** | 0.011963 |

496번 열의 vocabulary 간 표준편차가 약 **18.5배**가 된다. 같은 원본 값도 K32 block 안의 다른 weight들이 결정하는 scale에 따라 달라진다. 실제 `token_embd.weight`의 feature 496, block 15에서:

| Vocabulary row | FP16 weight | Carrier exponent | Column scale | 복원 block scale | HP1 INT4 weight |
|---|---:|---:|---:|---:|---:|
| 0 | -0.20556640625 | 0 | 0.03125 | 0.03125 | -0.21875 |
| 1 | -0.20556640625 | 1 | 0.03125 | 0.06250 | -0.18750 |
| 1820 | -0.20556640625 | 2 | 0.03125 | 0.12500 | -0.25000 |

496번 열 전체의 INT4 값은 `-0.25: 2,373개`, `-0.21875: 4,998개`, `-0.1875: 42,886개`다. 430번 열은 모든 50,257개 row에서 `-0.25`가 된다.

W의 한 열이 모든 단어에 정확히 같다면 그 feature는 모든 logit에 같은 상수를 더하므로 softmax 확률에 영향이 없다. **양자화가 이를 단어별로 다른 오차로 바꾸면 softmax가 제거할 수 없고, 큰 H가 그 차이를 확대한다.** 430번은 양자화 후에도 공통인 부분이 크므로 magnitude만으로 496번과 같은 손상을 가정하면 안 된다.

이번 RMSE는 입력 row마다 vocabulary 평균을 제거한 뒤 계산했다. 공통 logit 이동을 오류로 과대평가하지 않는다. 원본 FP16 head에서 세 feature를 통째로 삭제한 진단 대조군은 RMSE **0.298**, 원본 top-1과의 일치율 **93.75%**였다. 에너지 96%가 정보의 96%를 뜻하지 않는다. 다만 예측이 바뀌므로 feature 삭제는 원본과 동등한 해결책이 아니다.

반면 세 열은 전체 HP1 INT4 weight 오차 에너지의 **0.215%**에 불과한데, 이 세 열의 계산만 원본으로 복원하면 logit RMSE가 **3.377 → 0.841**로 줄었다. 일반적인 weight L2 오차는 H의 크기와 방향을 반영하지 못한다.

## 4. 다른 feature와 token에 영향이 퍼지는 연결

[GPT-2 graph](../../src/llama-model.cpp#L7494)는 다음 경로를 반복한다.

```text
h = x + Attention(QKV(LayerNorm(x)))를 output projection한 값
x_next = h + Down(GELU(Up(LayerNorm(h))))
다음 block → ... → final LayerNorm → 공유 LM head → logits
```

1. **Residual:** LN은 branch 입력에 적용된다. Identity 경로의 x를 매번 정규화해서 덮어쓰지 않는다. Branch가 같은 feature에 같은 부호의 값을 반복해서 더하면 커질 수 있다. 이번 마지막 두 block에서 이를 확인했다.
2. **Dense projection:** 한 입력 feature가 여러 출력 feature로 연결된다. Head에서는 한 feature가 vocabulary 전체 logits에 연결된다.
3. **Normalization:** LN의 평균·분산, RMSNorm의 RMS가 feature 전체에서 계산되므로 한 feature의 변화가 다른 feature에 영향을 줄 수 있다.
4. **Attention:** Q/K 변화가 score와 softmax 비중을 바꾸면 다른 위치 token의 V를 가져오는 비율도 달라진다.
5. **공유 quantization scale:** 큰 값이 정한 step 때문에 같은 block의 작은 값이 거칠게 양자화될 수 있다. 이것은 graph 연결과 별개인 추가 coupling이다.

Llama에도 이 연결 대부분이 있다. 이번 차이는 GPT-2의 연결 개수가 더 많아서가 아니라 **소수 입력 방향의 집중도와 그 방향에 대응하는 거의 공통인 head weight**에서 확인됐다. RMSNorm·SwiGLU·GQA·RoPE 각각의 기여도는 구조별 ablation 없이는 순위를 매길 수 없다. [Llama graph](../../src/llama-model.cpp#L4528).

기존 CPU Q8_K 대조군에서는 GPT-2의 nonzero 입력 중 **44.26%**, Llama의 **1.11%**가 0으로 양자화됐다. H와 Q6_K weight를 그대로 두고 activation rounding만 적용하면 GPT-2 32-chunk PPL이 **34.1601 → 52.2543**으로 변했다. 이는 별도의 activation 오차 증거다. 이번 HP1 weight 오차 실험이나 현재 Full PoTal PPL과 합쳐서 해석하면 안 된다. [이전 대조 실험](LM_HEAD.md).

## 5. 오차를 줄이는 방법: 직접 비교한 결과

**조건:** 기존 Q4_0 body / Q6_K head CPU 캡처 H를 사용한다. 앞 16 chunks의 4,096 rows로 channel·평균·방향을 결정하고, 뒤 16 chunks에서 64 rows를 골라 전체 vocabulary로 평가했다. Calibration과 평가 row는 겹치지 않지만, 둘 다 WikiText-2 test이므로 별도 corpus 일반화 검증은 아니다. W는 현재 Q4_HP1/Q8_HP1이다. Dot는 FP64이며 activation quantization·SCU·RMD·Metal 실행은 없다.

GPT-2 선택 feature는 496·430·36이다. Llama는 calibration에서 고른 1564·800·1021로, 전체 8,192 rows로 고른 이전 triple(1564·1645·1021)과 다르다. 아래는 centered logit RMSE이며 낮을수록 FP head에 가깝다.

| 방법 | GPT-2 INT4 | GPT-2 INT8 | Llama INT4 | Llama INT8 |
|---|---:|---:|---:|---:|
| 현재 HP1 head | **3.3766** | **0.2417** | **0.2962** | **0.01938** |
| 3개 입력 열을 정확하게 보정 | 0.8415 | 0.0538 | 0.2938 | 0.01920 |
| H 평균 제거 + 정확한 bias 복원 | 1.2174 | 0.0934 | 0.2459 | 0.01600 |
| 주요 입력 방향 1개를 정확하게 보정 | 0.5643 | 0.0435 | 0.2403 | 0.01567 |
| **주요 입력 방향 3개를 정확하게 보정** | **0.4019** | **0.0250** | **0.2336** | **0.01523** |
| Diagonal rescale 후 HP1 재양자화 | 1.0770 | 0.1136 | 미실행 | 미실행 |
| 고정 Hadamard 회전 후 HP1 재양자화 | 4.2853 | 0.2675 | 미실행 | 미실행 |
| **Head weight의 vocabulary 평균 제거 후 재양자화** | **0.7399** | **0.1709** | **0.2626** | **0.01627** |

GPT-2 INT4에서 3방향 보정은 RMSE **88.10% 감소**, top-1 일치율 **37.50% → 81.25%**, KL **3.48475 → 0.03622**였다. Vocabulary 평균 제거는 top-1 **81.25%**, KL **0.05952**였다. Top-1 일치율은 FP head 예측과의 비교이며 정답 accuracy나 PPL이 아니다.

Llama의 vocabulary 평균 제거는 INT4 RMSE를 줄였지만 top-1 일치율 **95.31% → 93.75%**, KL **0.02204 → 0.02646**으로 악화됐다. RMSE 감소만으로 accuracy 개선을 주장하지 않는다. 전체 지표는 CSV에 있다.

### A. 큰 방향을 projection하고 그 부분만 정확하게 계산

U는 calibration의 **uncentered** `H^T H`에서 큰 eigenvalue에 대응하는 orthonormal 방향이다. 평균을 제거한 covariance PCA와 다르다. GPT-2에서 첫 방향은 입력 에너지 **97.44%**, 세 방향은 **99.28%**를 포함했다. Llama는 각각 **44.77%, 51.13%**다.

```text
P = U U^T
Z_corrected = H(I-P) Wq^T + (H U)(W U)^T
            = H Wq^T + (H U)((W-Wq) U)^T
남는 오차 = H(I-P) (Wq-W)^T
```

큰 방향을 버리지 않고 원본 weight로 계산해서 더한다. U와 `(W-Wq)U`만 저장하면 추가 계산은 hidden→rank→vocabulary다. GPT-2 rank 3을 FP32로 저장하면 약 **0.584 MiB**지만, 이번 수치는 FP64 진단 계산이며 실제 FP32 커널 정확도·속도는 검증하지 않았다. Feature 축 3개만 선택하면 첫 번째 방법이며 3/768은 head weight의 **0.391%**다. 관련 방향: [LLM.int8()의 mixed-precision decomposition](https://arxiv.org/abs/2208.07339).

### B. 입력 평균 제거와 bias 복원

Calibration 평균을 mu라 하면 `(H-mu)Wq^T + mu W^T`를 계산한다. 남는 오차는 `(H-mu)(Wq-W)^T`다. 두 번째 항은 vocabulary별 bias로 미리 계산할 수 있다. 평균만 빼고 복원하지 않으면 함수가 달라진다. 관련 연구: [Outlier Suppression+](https://aclanthology.org/2023.emnlp-main.102/).

### C. 재스케일링과 회전

`H' = H S^-1`, `W' = W S` 또는 직교 R에 대한 `H' = H R`, `W' = W R`은 FP dot를 보존한다. 이번 scale은 `s_j = sqrt(max_calibration |H_j| / max_vocabulary |W_j|)`이며, 회전은 seed 42 sign + block-diagonal normalized Hadamard-256이다. **원본 W를 변환한 뒤 native HP1로 재양자화**했다. 이미 양자화된 Wq를 H와 함께 변환하기만 하면 기존 `H Wq^T`가 그대로여서 오차가 사라지지 않는다.

Scale은 개선됐으나 회전은 INT4 RMSE **3.377 → 4.285**로 악화됐다. Activation 양자화가 쉬워지는 것과 weight 오차가 줄어드는 것은 다른 조건이다. [SmoothQuant](https://proceedings.mlr.press/v202/xiao23c.html), [QuaRot](https://arxiv.org/abs/2404.00456)의 전체 pipeline을 재현한 실험은 아니다. GPT-2의 LN·bias·GELU·residual 전체에 임의의 회전을 그대로 통과시킬 수 없으므로 이번 변환은 마지막 head에만 적용했다.

### D. Softmax에 무관한 head weight 공통 성분을 제거

입력 평균을 제거하는 B와 구분해야 한다. 여기서는 W를 vocabulary 방향으로 평균낸다.

```text
mu_W = mean(W, axis=vocabulary)
W_centered = W - 1 mu_W^T
H W_centered^T = H W^T - (H mu_W) 1^T
softmax(H W_centered^T) = softmax(H W^T)
```

모든 logit에서 같은 값을 빼므로 FP 확률·argmax를 보존한다. 이를 **양자화 전에** 수행하면 공통 성분이 token별 quantization noise로 바뀌는 영향을 줄일 수 있다. GPT-2 INT4 RMSE는 **78.09% 감소**했다. 이미 Q(W)에 공통값을 빼는 것만으로는 예측이 바뀌지 않아 기존 오차를 고칠 수 없다.

현재 embedding/head는 공유돼 있다. Centered 행렬을 lookup에도 그대로 쓰면 입력이 달라지므로 head를 분리하거나 lookup 뒤 `+mu_W`를 더하는 등 입력 동등성도 보존해야 한다. 이번에는 head replay만 했고 GGUF는 수정하지 않았다. 새 정책의 결과를 기존 PoTal policy PPL로 보고하면 안 된다.

## 6. Residual compensation으로 해결되나?

Activation residual을 r이라 하면 개념적으로 `(q_dense+r)Wq^T`를 계산한다. r이 H를 완벽하게 복원해도 **`H(Wq-W)^T`는 남는다.** 위 projection은 weight 오차의 특정 방향을 추가로 복원하는 방법으로, 현재 activation residual 보정과 다르다.

실제 HP1의 fragment별 SCU 포화·ordered accumulation은 일반 실수 행렬곱과도 구분해야 한다. 이번 FP 항등식으로 Metal/SCU bitwise 동등성을 주장하지 않는다.

## 7. 검증·재현·한계

새 CPU trace는 GPT-2 FP16, GPT-2 Q4_0 body + F16 embedding/head, Llama Q4_0 body + F16 embedding/head에서 각각 512 tokens다. CPU 배치 로그와 88/88/115개 노드를 확인했다. Llama FP16 body 실행은 하지 않았다. Trace library 소스 commit은 `b2f6b7630da21d17f0ca34e08cfd1e2d762a47b5`이며 probe에 필요한 공개 header 다섯 개가 현재 header와 byte 단위로 같다. Library·GGUF·text·head input SHA는 JSON에 있다.

FP16 head 원본과 현재 HP1 head의 SHA는 [이전 검증](data/lm-head-diagnostics.json)과 대조했다. Llama FP16 head는 원본 head를 보존한 Q4-body archive에서 읽었으며 body까지 FP16인 파일로 취급하지 않는다.

Native producer는 `build-metal-llama-full/quality-potal4-d16/bin/libggml-base.dylib`의 `quantize_q4_hp1` / `quantize_q8_hp1`이다. GPT-2 head의 **모든 50,257 rows**, Llama의 첫 **256 rows**에서 FP 원본을 재양자화한 packed bytes가 저장된 HP1과 exact 일치했다. Llama head 전체의 producer byte 비교를 했다고 주장하지 않는다.

Self-check에서 projection·평균 보정·Hadamard 항등식과 단순 회전의 기존 오차 보존을 확인했다. 실데이터의 GPT-2 rescale/rotation 전후 FP dot 차이는 최대 `7.7e-13` 이하였다. RMSNorm의 affine 전 평균 제곱은 epsilon 때문에 정확히 1이 아니라 `mean(x^2)/(mean(x^2)+epsilon)`이며 이 범위로 검증했다. 두 페이지 PDF와 PNG의 표기·배치를 확인했다.

기존 캡처에서 분석·그림 재생성:

```bash
rtk proxy nice -n 10 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  uv run --script analysis/model/analyze_activation_amplification.py
```

완료한 것은 원인 분해, 짧은 CPU trace, 고정 입력 head 실험과 그림이다. 현재 Full PoTal/Metal 전체 PPL, 보정 커널 구현, latency·메모리 실측은 포함하지 않는다.
