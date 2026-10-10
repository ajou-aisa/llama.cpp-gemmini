# LM head: GPT-2가 HP1 INT4에 더 민감한 이유

후속 [activation 증폭 상세 분석](ACTIVATION_AMPLIFICATION.md)에서 FP16 원본의 레이어별 증가 위치, 세 feature의 거의 일정한 head weight, projection·centering 대조 실험까지 추적했다. 세 feature는 496·430·36번 좌표이며 처음 세 레이어를 뜻하지 않는다.

분석일: 2026-10-09. 대상은 GPT-2 124M과 Llama-3.2-1B다.

**확인한 현상:** 현재 HP1 head weight의 상대 오차는 두 모델 모두 약 13%지만, 저장된 실제 입력을 고정하고 head weight만 INT4로 바꾸면 GPT-2의 logit 상대 오차는 109.46%, Llama는 12.10%였다. GPT-2의 head 입력이 소수 채널에 집중되어 있어 해당 채널의 weight 오차에 매우 민감하다. Activation residual을 완전히 복원하더라도 head weight 양자화 오차는 남는다.

입력은 **2026-09-30 분석용 CPU Q4_0-body 실행에서 저장된 것**이다. 현재 Metal PoTal의 입력이나 full PPL을 새로 측정한 결과로 해석하면 안 된다. 현재 head의 민감도는 확인했지만, 진행 중인 PoTal PPL 상승에서 head가 차지하는 비중은 아직 확정하지 않았다.

## 그림과 원자료

- [LM-head FP16 / INT4 / INT8 분포](figures/lm-head-fp16-hp1-int4-int8.png): 6,800 × 4,000 PNG, 400 DPI. 전체 원소를 1/32 log2 간격으로 집계했다. 두 모델·세 precision의 축이 동일하다. 하나의 행렬을 비교하므로 2D 면적 histogram으로 표시하며 smoothing은 없다.
- [입력 집중도·head 출력 오차·과거 대조 실험](figures/lm-head-sensitivity.png): 7,200 × 2,800 PNG, 400 DPI.
- [두 페이지 PDF](figures/lm-head-comparison.pdf)
- [측정값·입력 경로·SHA·표본 위치](data/lm-head-diagnostics.json)
- [재현 스크립트](analyze_lm_head.py)

## 1. Head weight 분포만으로는 차이가 설명되지 않는다

두 모델 모두 별도 `output.weight` 없이 `token_embd.weight`를 embedding과 head에 공유한다. 현재 `models/default`의 이 행렬은 INT4 모델에서는 Q4_HP1, INT8에서는 Q8_HP1이다. GPT-2 shape는 50,257 × 768, Llama는 128,256 × 2,048이다.

| Head weight 측정 | GPT-2 | Llama-3.2-1B |
|---|---:|---:|
| FP16 RMS | 0.143696 | 0.021907 |
| FP16 최대 절댓값 | 1.785156 | 0.345703 |
| INT4 상대 L2 오차 | 13.0956% | 13.0347% |
| INT4 zero 비율 | 17.5101% | 17.2777% |
| INT8 상대 L2 오차 | 0.8184% | 0.8461% |
| INT8 zero 비율 | 1.1050% | 1.0907% |

GPT-2의 절대 weight 크기는 더 크다. 그러나 weight 크기나 전체 상대 오차만으로 PPL 민감도를 판단할 수 없다. Head는 `Z = H Wᵀ`를 계산하므로, 오차가 생긴 weight 위치에 어떤 입력 `H`가 곱해지는지까지 봐야 한다.

## 2. 실제 입력에서 GPT-2는 세 채널에 집중된다

기존 CPU 실행의 head 입력 파일을 다시 읽었다. 각 모델 8,192개 입력 행 전체를 사용했으며, 채널 에너지는 `sum_token(H[token, channel]^2)`이다. 두 모델의 tokenizer가 달라 첫 32 chunk가 덮는 원문 길이는 같지 않다.

| 입력 측정 | GPT-2 | Llama-3.2-1B |
|---|---:|---:|
| 상위 3개 채널의 에너지 비율 | **96.6346%** | **1.3456%** |
| 해당 채널 ID, 0부터 시작 | 496, 430, 36 | 1564, 1645, 1021 |
| 과거 Q8_K 입력 양자화에서 nonzero가 0이 된 비율 | 44.2605% | 1.1053% |

전체 weight 오차를 평균내면 이처럼 중요한 채널의 민감도가 드러나지 않는다. 한편 에너지 집중 자체가 항상 출력 손상을 의미하지는 않는다. 따라서 아래에서 실제 weight 오차를 입력에 곱해 확인했다. LayerNorm과 RMSNorm의 구조 차이만을 단독 원인으로 확정하지도 않는다.

## 3. 현재 HP1 head만 바꾸는 계산에서도 GPT-2가 크게 흔들린다

32개 chunk마다 저장된 입력 행의 offset 0, 64, 128, 192를 골라 총 **128개 행**을 사용했다. 모두 해당 PPL 실행의 평가 대상 위치다. 각 모델에서 동일한 `H`에 FP16·현재 Q4_HP1·현재 Q8_HP1 head weight를 각각 곱했다. Vocabulary는 GPT-2 50,257개, Llama 128,256개 전체를 사용했다.

HP1은 기존 분석에서 검증한 decoder로 F32 값으로 복원하고, dot는 FP64로 계산했다. **이 계산에는 activation 재양자화, ExSIA, residual 실행, SCU 포화, Metal kernel이 없다.** Head weight 값의 변경 효과를 분리하는 계산이다.

| Model | Head | Logit RMSE | Logit 상대 L2 오차 | FP16 head와 top-1 일치 | 평균 KL |
|---|---|---:|---:|---:|---:|
| GPT-2 | INT4 HP1 | **3.27874** | **109.46%** | **36.72%** | **3.47244** |
| GPT-2 | INT8 HP1 | 0.23394 | 7.81% | 88.28% | 0.02473 |
| Llama-3.2-1B | INT4 HP1 | **0.29666** | **12.10%** | **91.41%** | **0.02411** |
| Llama-3.2-1B | INT8 HP1 | 0.01941 | 0.79% | 100.00% | 0.000106 |

Logit RMSE와 상대 오차는 행마다 vocabulary 평균을 제거한 뒤 계산했다. 모든 logit에 같은 상수를 더해도 softmax가 달라지지 않으므로, 그 상수 이동은 오류에서 제외했다. FP16 head logit의 이 기준 RMS는 GPT-2 2.99544, Llama 2.45131이다. **Top-1 일치율은 정답 정확도나 PPL이 아니다.** KL도 FP16 head 계산의 확률분포를 기준으로 한다.

상위 세 입력 채널에 해당하는 weight 오차만 남겨 `H[:, top3] · ΔW[:, top3]ᵀ`를 계산하면 INT4 logit RMSE가 GPT-2 **3.16184**, Llama **0.03777**이었다. GPT-2에서는 세 채널의 weight 오차만으로도 전체 오차 3.27874와 비슷한 크기의 출력 변화가 생긴다. 채널 간 교차항이 있으므로 이 수치의 제곱 비율을 정확한 기여율로 해석하지 않는다.

## 4. 보상을 해도 weight 오차는 남는다

일반적인 실수 행렬 계산에서 `Wq = W + ΔW`, 보상 후 입력을 `Hq = H + ΔH`로 쓰면 다음과 같다.

```text
Hq Wqᵀ - H Wᵀ = H ΔWᵀ + ΔH Wᵀ + ΔH ΔWᵀ
입력을 완전히 복원해 ΔH = 0이 되어도 H ΔWᵀ는 남는다.
```

현재 [ExSIA folding](../../ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp#L2043)은 folded activation code를 clip하고, 선택된 outlier 위치의 차이를 residual로 저장한다. 이것은 원본 FP16 weight와 HP1 weight의 차이를 복원하는 보정이 아니다. 또한 모든 FP activation 오차를 복원한다고 보장하는 것도 아니다. 실제 HP1 실행에는 ordered SCU 포화 등 별도의 산술 조건도 있어 위 식이 전체 하드웨어 출력의 동등성 식은 아니다.

따라서 이 표본에서 확인한 GPT-2의 **head weight 자체에 대한 민감도**는 residual compensation을 켰다는 사실만으로 해결되지 않는다.

## 5. 과거 입력 양자화 대조 실험도 같은 민감도를 보였다

기존 CPU Q4_0-body / Q6_K-head 대조 실험의 raw log와 capture를 다시 확인했다. Weight 값을 동일하게 유지한 F32 head 계산과 native head 계산 사이에서 **embedding lookup과 양자화 전 head 입력이 bitwise 동일함**을 재검증했다.

| 과거 CPU head 계산 | GPT-2 PPL | Llama PPL |
|---|---:|---:|
| Native Q6_K head, 내부 Q8_K 입력 양자화 | 52.2543 | 11.1615 |
| 같은 Q6_K 복원 weight, FP 입력 | 34.1601 | 11.1605 |
| FP head에 native Q8_K로 반올림한 입력을 재주입 | 52.2543 | 미실행 |

GPT-2의 재주입 입력도 native quantize/dequantize 결과와 bitwise 일치했다. 이 대조는 과거 경로에서 head 입력 양자화가 GPT-2 PPL을 크게 악화시켰다는 근거다. **첫 32 chunks, context 512, 평가 8,160 tokens의 CPU 결과**이며 현재 HP1·Metal·full-dataset PPL 값이 아니다. 저장된 8,192개 입력에는 PPL에서 점수화하지 않는 각 chunk의 마지막 행 32개가 포함된다.

기존 weight 동일성 검증은 [과거 분석](../../output/experiment/q6-head-20260930/head-analysis.json)과 [양 모델 대조 기록](../../output/experiment/potal-model-compare-20260930/comparison.json)에 남아 있다. 과거 JSON에서 말하는 당시의 HP1 기본 정책을 현재 모델 정책으로 인용하지 않는다.

## 6. 확인 범위와 재현

현재 네 HP1 GGUF의 전체 SHA가 기존 분석 및 PPL 입력 기록과 일치했다. GPT-2는 FP16 원본을 직접 읽었다. Llama FP16 head는 보관된 `potal-ppl-20261001/llama3.2-1B.Q4_0.head-original-F16.gguf`에서 읽었다. 이 파일의 body는 계산에 사용하지 않았다. 이 head의 모든 통계·세밀한 histogram과 HP1 오차 통계가, 전체 FP16 원본 SHA를 검증했던 앞선 분석과 일치함을 재검증했다. 보관 파일 전체 SHA와 head tensor SHA는 JSON에 별도로 기록했다. 이번에 보관 파일을 전체 FP16 원본 GGUF와 byte 단위로 다시 대조한 것은 아니다.

재현 명령은 다음과 같다. CPU 1 thread, 낮은 우선순위로 저장된 데이터를 분석한다.

```bash
cd /Users/chan/Projects/AISA/llama.cpp-gemmini
rtk proxy nice -n 10 uv run --script analysis/model/analyze_lm_head.py
```

스크립트는 입력 shape·source commit·실행 종료 상태, head 입력과 embedding의 동일성, 재주입 입력, 현재 weight SHA·통계·histogram을 확인한다. Logit metric은 상수 이동 불변성과 서로 다른 logit의 양수 오류를 self-check한다. 그림은 저장 후 직접 확인했다.

**남은 인과 검증:** 현재 PoTal 실행에서 같은 token 위치의 head 직전 입력을 저장하고, body·embedding·입력을 고정한 output projection 대조가 필요하다. 그래야 현재 full PPL 증가 중 head weight, activation 양자화·보상, body 누적 오차, backend 산술 차이를 분리할 수 있다. 공유 embedding tensor를 GGUF에서 바꾸면 입력 embedding도 바뀌므로 그것만으로는 head 단독 대조가 되지 않는다. 이번 작업은 기존 PPL 큐나 모델·실행 코드를 변경하지 않았다.
