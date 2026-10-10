# GPT-2 / Llama-3.2-1B: 구조와 HP1 weight 분포 분석

분석일: 2026-10-09. 대상은 현재 실험에 사용하는 **GPT-2 124M과 Llama-3.2-1B**다. FP16 원본과 `models/default`의 **Q4_HP1 / Q8_HP1**을 비교했다. 아래 INT4·INT8은 HP1 weight code의 bit width를 뜻한다.

**핵심 결과:** 두 모델 모두 INT4의 weight 상대 L2 오차가 약 13%, INT8은 약 0.8%다. INT4에서는 약 17.5~17.8%의 weight가 0이며, RMS가 원본과 비슷해도 개별 weight의 변형은 크다. 두 모델 모두 token embedding과 LM head를 공유하고, 현재 HP1 모델에서는 이 행렬도 HP1이다.

## 1. 그림과 데이터

| 산출물 | 내용 |
|---|---|
| [GPT-2 한 페이지](figures/gpt2-fp16-hp1-int4-int8.png) | FP16의 INT4·INT8 zero map과 INT4·INT8 분포 비교 |
| [Llama 한 페이지](figures/llama-fp16-hp1-int4-int8.png) | FP16의 INT4·INT8 zero map과 INT4·INT8 분포 비교 |
| [두 페이지 PDF](figures/gpt2-llama-weight-comparison.pdf) | 위 두 그림의 벡터 PDF |
| [GPT-2 레이어별 PDF](figures/layers/gpt2-layers-fp16-hp1.pdf) | Embedding + B00~B11, 총 13페이지 |
| [Llama 레이어별 PDF](figures/layers/llama-layers-fp16-hp1.pdf) | Embedding + B00~B15, 총 17페이지 |
| [GPT-2 레이어 통계](data/gpt2-layers.csv), [Llama 레이어 통계](data/llama-layers.csv) | block별 magnitude, zero 비율, 오차 |
| [GPT-2 tensor 통계](data/gpt2-tensors.csv), [Llama tensor 통계](data/llama-tensors.csv) | 개별 행렬, norm, bias, position/RoPE까지 모든 tensor |
| [GPT-2 원자료](data/gpt2.json), [Llama 원자료](data/llama.json) | 입력 SHA, metadata, 분포 histogram, integer code 빈도, 전체 통계 |
| [GPT-2 세부 magnitude](data/gpt2-magnitude-fine.csv), [Llama 세부 magnitude](data/llama-magnitude-fine.csv) | 1/32 log2 간격의 magnitude 경계, 원소 수, 레이어 내 비율 |
| [GPT-2 zero 변환](data/gpt2-zeroed-reference-fine.csv), [Llama zero 변환](data/llama-zeroed-reference-fine.csv) | 원래 FP16 magnitude 구간별로 양자화 후 0이 된 원소 수와 비율 |
| [LM-head 비교 PDF](figures/lm-head-comparison.pdf) | Head FP16/INT4/INT8 분포와 입력·출력 민감도, 2페이지 |
| [LM-head 원인 분석](LM_HEAD.md), [측정값](data/lm-head-diagnostics.json) | 현재 HP1 head를 고정 입력에 적용한 오차와 과거 CPU 대조 실험 |
| [Activation 증폭 상세 분석](ACTIVATION_AMPLIFICATION.md), [그림 PDF](figures/activation-amplification.pdf) | FP16 레이어 추적, 세 feature의 원인, projection·centering·회전 실험 |

그림 위쪽의 3D 축은 **x = 복원된 weight 크기 `|W|`**, **y = transformer block**, **z = 해당 magnitude bin의 원소 비율(%)**이다. 실제 histogram 집계값을 꼭짓점으로 연결한 surface이며, smoothing 필터는 적용하지 않는다. 기존 0.25 log2 간격을 **0.03125 = 1/32 log2**로 줄여 **8배 촘촘하게 원본에서 재집계**했다. 인접한 magnitude 경계의 비율은 약 1.021897이다. `E`는 공유 embedding/LM head를 뜻하며 한 번만 센다. block 안의 attention/FFN 행렬은 원소 수에 비례해 합쳤다. 아래쪽은 RMS, 근사 P99, 최대 절댓값과 zero 비율이다. 각 모델의 세 precision은 같은 축 범위를 사용한다. PNG 해상도는 **8,400 × 5,080 pixels (400 DPI)**이며 PDF는 모델당 한 페이지, 총 두 페이지다.

왼쪽 두 3D panel은 **동일한 FP16 분포**를 각각 INT4·INT8의 zero 변환 비율로 색칠한다. 파란 회색은 해당 FP16 구간의 원소가 0으로 변한 비율 0%, 노랑은 중간, 빨강은 100%다. 표면 높이는 원래 FP16 빈도이며, 색은 `해당 구간에서 FP16 != 0이고 복원된 HP1 == 0인 원소 수 / 해당 구간의 FP16 원소 수`다. 원래부터 0인 값은 제외하며, 빈 구간에는 비율을 정의하지 않는다. 원본과 HP1의 **같은 위치**를 직접 비교해서 집계했다. 전체 zero 개수 차이나 단일 magnitude threshold로 추정하지 않았다. 오른쪽 두 panel은 기존 INT4·INT8 분포다.

Zero 변환 CSV에는 원래 FP16 구간의 원소 수, 새로 0이 된 수, 구간 내 비율과 레이어 전체 원소 대비 비율이 함께 있다. 빈 구간의 `zeroed_percent_of_fp_bin`은 빈칸이다. JSON의 `zeroed_reference_histogram`은 원래 FP16 magnitude로 집계한 개수, `zeroed_nonzero_fraction`은 레이어 전체 원소 대비 새로 0이 된 비율이다. 모델·레이어 전체 비율과 구간 내 비율의 분모가 다르므로 구분해서 읽는다.

Surface의 vertex는 세밀한 bin의 집계값을 모두 사용한다. bin을 다시 축소하거나 곡선 보간으로 새로운 표본을 만들지 않는다. 면 자체는 인접한 vertex를 연결한 시각적 표현이다. 전체 histogram은 160개에서 1,280개 bin으로 늘었고, 표시하는 magnitude 범위에는 기존 88개 대신 **704개 bin**이 들어간다. 각 구간의 실제 magnitude 하한·상한과 빈도는 위 세부 magnitude CSV에서 확인할 수 있다. `kind=zero`는 zero 원소를 별도로 기록한 행이다.

세밀한 bin에서 FP16의 낮은 빈도와 INT4의 높은 봉우리를 함께 읽을 수 있도록 surface 높이는 `log10(1 + bin_percent / 0.01)`로 표시한다. **Z축 눈금은 실제 비율(%)**이며 0도 그대로 표시한다. 세 precision 모두 동일한 변환과 범위를 사용한다. 저장된 histogram과 CSV는 변환 전 원소 수와 비율이다. X축은 기존 log2 배치를 유지하면서 눈금을 실제 magnitude 숫자로 표기했다.

모든 원소를 읽어 집계했다. 그림에는 embedding과 transformer의 2차원 weight 행렬을 넣었다. norm, bias, learned position, RoPE tensor는 CSV와 전체 통계에 포함하고 그림에서는 제외했다. signed mean/min/max도 CSV에 있다. 이 절의 기본 분포 그림은 weight를 대상으로 한다. 후속 [LM-head 분석](LM_HEAD.md)에는 과거에 저장된 activation과 현재 HP1 head의 출력 민감도를 추가했다.

레이어별 PDF는 이전 0.25 log2 간격의 분포를 유지하며 한 페이지에 한 transformer block의 FP16·INT4·INT8을 나란히 표시한다. 한 block의 곡선을 보기 쉽게 x=`|W|`, y=bin 비율의 2D 점·직선으로 펼쳤다. 같은 모델의 모든 페이지에서 축 범위를 고정했다. 개별 PNG는 `figures/layers/gpt2/`와 `figures/layers/llama/`의 `embedding.png`, `block-00.png` 등의 파일에 저장한다. 여기서 레이어는 transformer block이며, 해당 block의 attention/FFN 행렬을 합친 분포다.

## 2. 구조 차이

수치는 분석한 GGUF의 metadata와 tensor shape를 기준으로 확인했다. GPT-2의 기본 구성은 [원본 config](https://huggingface.co/openai-community/gpt2/blob/6c0e6080953db56375760c0471a8c5f2929baf11/config.json)와도 대조했다. 실제 연산 흐름은 현재 소스의 [GPT-2 graph](../../src/llama-model.cpp#L7494), [Llama graph](../../src/llama-model.cpp#L4528)에 따른다.

| 항목 | GPT-2 124M | Llama-3.2-1B |
|---|---|---|
| Transformer block 수 | 12 | 16 |
| Hidden dimension | 768 | 2,048 |
| FFN intermediate dimension | 3,072 | 8,192 |
| Attention | MHA: Q/K/V 각각 12 heads | GQA: Q 32 heads, K/V 각각 8 heads |
| Head dimension | 64 | 64 |
| Normalization | Pre-LayerNorm, weight + bias | Pre-RMSNorm, weight만 존재 |
| FFN | Up → GELU → Down | SiLU(Gate) × Up → Down, SwiGLU |
| Linear bias | 존재 | 현재 tensor inventory에는 없음 |
| Position | 학습된 absolute position embedding | Q/K에 RoPE |
| GGUF context length | 1,024 | 131,072 |
| Vocabulary | 50,257 | 128,256 |
| Embedding / LM head | weight 공유 | weight 공유 |
| 고유 학습 parameter 수 | 124,439,808 | 1,235,814,400 |
| 저장 tensor 원소 수 | 124,439,808 | 1,235,814,432 |

Llama의 저장 원소 수에는 학습 parameter가 아닌 `rope_freqs.weight` 32개가 포함된다. context length는 모델 metadata이며 이번 분석에서 실행한 sequence length를 뜻하지 않는다.

### Block 내부 흐름

GPT-2는 token embedding과 learned position을 더한 뒤 다음을 12회 반복한다.

```text
h = x + Attention(QKV(LayerNorm(x)))를 output projection한 값
x_next = h + Down(GELU(Up(LayerNorm(h))))
최종 LayerNorm → 공유 LM head → logits
```

Llama는 token embedding 이후 다음을 16회 반복한다.

```text
u = RMSNorm(x)
h = x + Output(GQA(RoPE(Q(u)), RoPE(K(u)), V(u)))
v = RMSNorm(h)
x_next = h + Down(SiLU(Gate(v)) * Up(v))
최종 RMSNorm → 공유 LM head → logits
```

Llama의 GQA는 K/V head 하나를 Q head 4개가 공유한다. 실제 FFN의 gate와 up 분기 결합은 [FFN builder](../../src/llama-graph.cpp#L721)에서도 확인된다.

### 주요 weight shape

아래 shape는 일반적인 `[output, input]` 표기이며 GGUF 내부 dimension 순서와 반대다.

| Weight | GPT-2 | Llama-3.2-1B |
|---|---|---|
| Token embedding / tied head | 50,257 × 768 | 128,256 × 2,048 |
| Q/K/V | 하나의 QKV: 2,304 × 768 | Q: 2,048 × 2,048; K/V 각각 512 × 2,048 |
| Attention output | 768 × 768 | 2,048 × 2,048 |
| FFN up | 3,072 × 768 | 8,192 × 2,048 |
| FFN gate | 없음 | 8,192 × 2,048 |
| FFN down | 768 × 3,072 | 2,048 × 8,192 |
| 한 block의 위 행렬 원소 합 | 7,077,888 | 60,817,408 |

**현재 파일의 quantization 범위:** GPT-2는 49개 행렬, Llama는 113개 행렬이 각각 Q4_HP1 또는 Q8_HP1이다. embedding도 포함된다. GPT-2의 나머지 99개 tensor와 Llama의 나머지 34개 tensor는 F32로 유지된다. FP16 원본도 이 행렬들만 F16이며 보조 tensor는 F32인 혼합 저장 형식이다.

두 모델의 파일에 별도 `output.weight`는 없다. loader는 이때 token embedding을 output weight로 공유한다. 따라서 현재 정책에서는 embedding의 HP1 변형이 입력 lookup과 최종 logits projection에 모두 사용된다. 근거: [Llama loader](../../src/llama-model.cpp#L1750), [GPT-2 loader](../../src/llama-model.cpp#L2684).

## 3. 전체 matrix weight의 측정 결과

비교 대상 원소 수는 GPT-2 **123,532,032개**, Llama **1,235,746,816개**다. FP16 기준으로 위치가 같은 원소끼리 비교했다.

| Model | Weight | RMS | 최대 `|W|` | Zero 비율 | FP16 대비 RMSE | 상대 L2 오차 | SQNR |
|---|---|---:|---:|---:|---:|---:|---:|
| GPT-2 | FP16 | 0.134870 | 17.109375 | 0.000031% | 0 | 0% | 기준 |
| GPT-2 | Q4_HP1 | 0.135843 | 16.000000 | 17.7647% | 0.0178163 | 13.2099% | 17.582 dB |
| GPT-2 | Q8_HP1 | 0.134921 | 17.000000 | 1.1397% | 0.00111856 | 0.8294% | 41.625 dB |
| Llama-3.2-1B | FP16 | 0.0201719 | 1.234375 | 0.000140% | 0 | 0% | 기준 |
| Llama-3.2-1B | Q4_HP1 | 0.0203663 | 1.250000 | 17.5468% | 0.00263323 | 13.0540% | 17.685 dB |
| Llama-3.2-1B | Q8_HP1 | 0.0202293 | 1.234375 | 1.1120% | 0.000171063 | 0.8480% | 41.432 dB |

정의:

```text
RMS(W) = sqrt(sum(W²) / N)
RMSE = sqrt(sum((W_HP1 - W_FP16)²) / N)
상대 L2 오차 = ||W_HP1 - W_FP16||₂ / ||W_FP16||₂
SQNR = 20 log10(||W_FP16||₂ / ||W_HP1 - W_FP16||₂)
```

### Magnitude와 분포 해석

- GPT-2의 원본 matrix RMS는 Llama의 약 **6.69배**다. 절대 RMSE도 weight의 단위를 따라 달라지므로 두 모델을 비교할 때 상대 L2 오차를 함께 봐야 한다.
- INT4는 작은 weight가 0으로 모이고 남은 값들이 제한된 단계에 집중된다. 3D 그림의 높은 봉우리는 이산화된 magnitude bin에 원소가 모인 결과다. FP16보다 weight 전체가 더 커졌다는 뜻은 아니다.
- INT8은 원본 분포와 RMS를 훨씬 가깝게 유지한다. 두 모델 모두 INT4 대비 상대 L2 오차가 약 15~16배 작다.
- FP16의 근사 P99 `|W|`는 GPT-2 **0.38877**, Llama **0.05943**이다. 최대값은 각각 **17.1094**, **1.23438**로, 평균적 크기와 큰 값의 규모가 상당히 다르다.

## 4. 레이어별 특징

### GPT-2

Block별 FP16 RMS는 B05의 **0.11553**에서 B11의 **0.16005**까지 분포한다. 중간 block보다 후반 block의 RMS가 커지는 경향이 있고, B00도 **0.14490**으로 크다.

큰 weight는 FFN down에 집중된 사례가 보인다.

| Tensor | FP16 최대 `|W|` | 해당 tensor의 FP16 RMS |
|---|---:|---:|
| `blk.3.ffn_down.weight` | 17.109375 | 0.091807 |
| `blk.2.ffn_down.weight` | 15.070313 | 0.093088 |
| `blk.1.ffn_down.weight` | 13.734375 | 0.087191 |
| `blk.10.ffn_down.weight` | 11.046875 | 0.178145 |

특히 B03 FFN down의 최대값은 해당 tensor RMS의 약 **186배**다. 이런 큰 값이 어느 K32 block에 함께 놓이는지에 따라 같은 tensor 안에서도 양자화 간격의 영향이 달라질 수 있다. 이번 통계는 큰 값의 존재를 확인하며, 특정 K32 block이 오차를 유발했다는 인과관계까지 측정한 것은 아니다.

INT4에서 상대 L2 오차가 큰 행렬은 B11 FFN down **14.070%**, B02 FFN down **13.966%**, B00 FFN down **13.943%**다. 개별 행렬의 상대 오차와 전체 오차에 대한 기여도를 구분해야 한다.

### Llama-3.2-1B

Block별 FP16 RMS는 B06 **0.018767**에서 B15 **0.021079**까지다. GPT-2보다 block 사이 magnitude 변화가 작다. 가장 큰 개별 weight는 B15 FFN up의 **1.234375**, 다음은 B05 attention K의 **1.1328125**다.

INT4에서 후반 K/V projection은 block 전체 평균보다 상대 오차가 높다.

| Tensor | INT4 상대 L2 오차 | INT4 zero 비율 |
|---|---:|---:|
| `blk.15.attn_v.weight` | 15.475% | 22.124% |
| `blk.15.attn_k.weight` | 15.150% | 21.282% |
| `blk.14.attn_v.weight` | 15.061% | 21.142% |
| `blk.14.attn_k.weight` | 14.670% | 20.844% |

K/V 행렬은 FFN보다 작아서 block 전체 분포에 합치면 이런 차이가 약해진다. 따라서 3D block 그림과 개별 tensor CSV를 함께 보는 것이 적절하다. 이것은 weight 오차의 비교이며, K/V가 실제 PPL 저하의 주원인이라는 판정은 아니다.

### Embedding / LM head

| Model | 원소 수 | INT4 상대 L2 오차 | INT8 상대 L2 오차 | 전체 INT4 weight 제곱오차 중 비중 |
|---|---:|---:|---:|---:|
| GPT-2 | 38,597,376 | 13.096% | 0.818% | 34.857% |
| Llama-3.2-1B | 262,668,288 | 13.035% | 0.846% | 24.995% |

Embedding의 상대 오차가 본체보다 특별히 큰 것은 아니다. 행렬이 커서 전체 제곱오차의 큰 몫을 차지하며, 공유 LM head이므로 logits에도 사용된다는 점이 중요하다. 위 비중은 **정적 weight 제곱오차의 비중**이지 PPL 저하 기여율이 아니다.

## 5. PoTal PPL과 연결해 해석할 때

현재 파일에서는 본체와 embedding/LM head가 모두 HP1이다. 따라서 INT4 결과를 해석할 때 activation 양자화·residual 보상 외에 **이미 변형된 HP1 weight**도 함께 고려해야 한다.

Activation residual을 사용하는 계산이 `(A_dense + R_A) W_HP1`이라면, activation이 완전히 복원되어도 `A (W_HP1 - W_FP16)`에 해당하는 weight 오차는 남는다. 이번 분석에서 그 weight 차이를 직접 확인했다. 다만 정적 weight 분석으로 activation 오차, HP1 SCU 포화, residual 구현, Metal 실행 경로의 정확성이나 최종 PPL을 판정할 수는 없다.

후속 원인 분리 실험을 한다면 현재 동일 파일을 사용해 embedding/head 정책 또는 activation/residual 한 항목만 바꿔 비교해야 한다. 이번 작업에서는 모델을 다시 양자화하거나 PPL 큐의 조건을 바꾸지 않았다.

## 6. 집계 방법과 검증

HP1은 integer code만 비교하지 않고 다음 복원식을 적용했다.

```text
m == INT16_MIN: W = 0
그 외: W = q * ldexpf(channel_scale, m)
Q4_HP1: low nibble 16개 다음 high nibble 16개, 각각 -8
Q8_HP1: signed INT8 code 32개
```

이것은 저장된 weight의 dequantization이다. activation을 곱하는 HP1 하드웨어 SCU의 fragment 포화 실행은 아니다. 기준 코드는 [Q4 decoder](../../ggml/src/ggml-quants.c#L854), [Q8 decoder](../../ggml/src/ggml-quants.c#L739)다.

- 원본 FP16 파일의 모든 bytes를 읽어 SHA-256을 검증하고 대응 tensor 이름과 shape를 확인했다.
- HP1 파일 네 개는 [실험 모델 SHA manifest](../../scripts/experiment/default-ppl-models.sha256)와 일치했다.
- 각 HP1 tensor의 첫 최대 256개 복원값을 기존 `libggml-base.dylib`의 C decoder와 bitwise 비교했다. GPT-2 **98회**, Llama **226회**, 총 **324회** 일치했다. 모든 원소에 대해 C decoder를 이중 실행한 것은 아니다.
- 모든 원소에 대해 유한값 검사와 통계 집계를 수행했다. 평균·RMS·오차는 FP64 누산이며 표본 추정이 아니다. 최솟값·최댓값·zero 개수는 원소를 직접 검사했다.
- Magnitude histogram의 전체 범위는 `log2(|W|)` 기준 **[-32, 8]**이다. 기존 통계와 P50/P99/P99.9에는 폭 **0.25**를 유지하며, 새 surface와 세부 CSV는 폭 **0.03125**를 사용한다. P50/P99/P99.9는 bin 내부 보간으로 구한 **근사값**이다.
- 3D 그림은 범위 `[-16, 6]`, 즉 `2^-16 ≤ |W| < 64`를 표시한다. 생략된 작은 nonzero 값의 비율은 각 열 하단에 기재했다. zero는 log축에 넣지 않고 아래쪽 별도 선으로 표시했다. bin 비율의 분모에는 zero도 포함한다.
- Packing·zero sentinel, 독립 GGUF reader와 header 비교, chunk 분할 전후 통계, 세밀한 bin의 독립 histogram 비교를 포함한 **4개 테스트가 통과**했다. 세밀한 bin 8개를 합하면 기존 bin 1개와 정확하게 일치하는지도 검사한다.
- 고해상도 재집계 후 전체 통계, tensor별 통계, 기존 histogram과 입력 SHA가 이전 결과와 정확하게 일치함을 확인했다. 세밀한 bin 8개를 합하면 기존 bin 1개와 모든 레이어·precision에서 일치한다.
- CPU 한 thread와 낮은 실행 우선순위로 분석했다. 세밀한 bin 재집계 시간은 GPT-2 약 **11.1초**, Llama 약 **378.1초**이며 Llama에는 원본 네트워크 수신 시간이 포함된다. 추론 성능 측정값으로 사용할 수 없다.

## 7. 원본과 재현

분석 시 저장소 HEAD: `ba42c63b08dc28b1043adc176cfe8b0111b78284`. 작업 폴더에는 기존 Metal 변경이 있는 상태였으며 이번 분석 산출물은 `analysis/model`에 저장했다.

Llama FP16은 [고정 revision의 원본 GGUF](https://huggingface.co/mradermacher/Llama-3.2-1B-GGUF/resolve/2c6e7d31cd946f66f5b23835f355455fe7954354/Llama-3.2-1B.f16.gguf)를 다시 받았다. **2,479,591,872 bytes 전체 SHA-256**이 이전 [입력 파일 기록](../../output/experiment/q4-rounding-20260930/input-and-runtime.sha256)의 `/Users/chan/Downloads/llama3.2-1B.fp16.gguf`와 일치한다. 디스크 여유 때문에 원본은 스트리밍으로 읽었고, 2.48GB 파일 자체는 다시 저장하지 않았다. 통계와 그림, 원본 URL과 SHA는 보관했다.

| Model / precision | SHA-256 |
|---|---|
| GPT-2 FP16 | `5bd663a1e1d303f9404dd2edd92e4e3ca3e24bcf80d19c655880f0e455c8f4df` |
| GPT-2 Q4_HP1 | `184e01e203452c97fb676c752998cce7842cfa1cdd32a22e09a40a73d0e3586f` |
| GPT-2 Q8_HP1 | `cf23a45e4db23736d7d1f67ab42018803a198cfcb01db36daca69ff56bdbc7d4` |
| Llama FP16 | `7d3b65ea4f88f55f258bfcceac19f254c8d063fbc7f244c7e36a216491224a50` |
| Llama Q4_HP1 | `a81449177a5e59ff0d2cec732d4d7990ef4a3faaedee61d55e35970266be9d35` |
| Llama Q8_HP1 | `283846f41922078ad91825b271e110592cfc6dbc649219e79d96460c90e56fe7` |

입력 경로는 [inputs.json](inputs.json)에 고정했다. 현재 프로젝트 루트에서 재현:

```bash
rtk proxy nice -n 10 uv run --python /opt/homebrew/bin/python3 --script analysis/model/analyze_weights.py gpt2
rtk proxy nice -n 10 uv run --python /opt/homebrew/bin/python3 --script analysis/model/analyze_weights.py llama
rtk proxy env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nice -n 10 uv run --python /opt/homebrew/bin/python3 --script analysis/model/plot_weights.py
```

레이어별 그림만 다시 생성:

```bash
rtk proxy env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nice -n 10 uv run --python /opt/homebrew/bin/python3 --script analysis/model/plot_layers.py
```

분석 스크립트의 native decoder 검사는 기존 `build-metal-llama-full/quality-potal4-d16/bin/libggml-base.dylib`를 사용한다. 이 build 경로와 실험 모델 SHA manifest가 필요하다. Plot만 다시 만들 때는 세 번째 명령만 실행하면 된다. Python 의존성은 스크립트에 버전을 고정했다.

검증 명령:

```bash
rtk proxy env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run --no-project --python /opt/homebrew/bin/python3 --with numpy==2.5.3 --with pytest python -B -m pytest -q -p no:cacheprovider analysis/model/test_weight_analysis.py
```
