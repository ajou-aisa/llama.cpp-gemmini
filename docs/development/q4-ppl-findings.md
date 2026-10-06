# GPT-2 / Llama 4bit PPL: 확인된 사실과 head 가설

정리: 2026-10-01. 소스 `b2f6b7630da21d17f0ca34e08cfd1e2d762a47b5`.
범위: GPT-2 124M / Llama-3.2-1B의 양자화·PPL 조사.

**핵심:** 이번 CPU 측정에서 **head를 원본 F16으로 맞추면 GPT·Llama 모두 PoTal PPL이 더 높다.** GPT 기본 Q4_0의 큰 손실은 **Q8_K head 입력 양자화**에서 확인됐으며, GPT head weight가 유독 많이 clipping되는 현상은 관측되지 않았다.

## 1. 이번 head 분석 전에 확인한 사실

### 모델·실행 조건

- 두 모델 모두 **embedding/head weight 공유**. GPT는 최종 LayerNorm(gain·bias), Llama는 RMSNorm(gain)을 사용한다.
- PPL 조건: CPU, WikiText-2 raw 앞 32 chunks, context 512, 스레드 4, batch/ubatch 512, KV F16, stride 0. **chunk당 255개, 총 8,160개의 다음 토큰을 채점**한다. tokenizer가 달라 모델 간 평가 텍스트 범위는 다르다.
- CPU Q4_0 body는 `Q4_0 × Q8_0`, head는 `Q6_K × Q8_K`. 파일명 Q4_0가 전체 W4A4 또는 W4A16을 의미하지 않는다.
- Q4_0는 32개 단위 signed scale `d = signed_absmax / -8`, Q8_0는 `amax / 127`을 사용한다. Q4_HP1은 `2^round(log2(amax / 7))` scale이므로 Q4_0와 같은 양자화법이 아니다.

### 배제·확인한 원인

| 검사 | GPT-2 | Llama | 판단 |
|---|---:|---:|---|
| Q4_0 반올림만 `roundf`로 변경한 PPL | 52.2543 / 52.2238 | 11.1615 / 11.1503 | 기존 / 변경. 큰 격차를 설명하지 못함 |
| Q4_0 body에서 실제 clamp된 weight | 0.4935% | 0.5042% | GPT만 특별히 많지 않음 |
| Q4_0 body 상대 제곱오차 | 0.8039% | 0.7793% | 두 모델이 비슷함 |
| 현재 HP1 body 상대 제곱오차 | 1.3912% | 1.3705% | 양쪽 모두 증가 |

상대 제곱오차(NMSE)는 `100 × Σ(w−ŵ)² / Σw²`. GPT의 범위 밖 body weight를 원본으로 되돌린 F32 대조도 PPL **34.1203 / 34.1288**로 개선되지 않았다. **Body clipping·반올림이 주원인이라는 근거는 없었다.** [반올림](../../output/experiment/q4-rounding-20260930/results.json), [clipping](../../output/experiment/q4-clipping-20260930/summary.json), [HP1 weight](../../output/experiment/potal-model-compare-20260930/hp1-weight-comparison.json).

### 가장 큰 차이는 head 실행에서 관측됨

아래는 **Q4_0 body를 고정한 대조**이며, 전체 HP1/PoTal PPL이 아니다.

| 공유 embedding/head 조건 | GPT-2 PPL | Llama PPL |
|---|---:|---:|
| 기본 Q6_K weight, Q8_K head 입력 | 52.2543 | 11.1615 |
| 동일한 Q6 복원 weight 값, F32 head 실행 | 34.1601 | 11.1605 |
| FP16 원본 embedding/head | 32.1315 | 11.1627 |

- 동일값 F32 대조에서 embedding 조회값·head 입력의 비트 동일성 확인. GPT에 Q8_K 변환을 재주입하면 **52.2543**으로 복귀했다. **Q8_K 입력 양자화 손실을 확인했고, 검사한 dot에서는 버그 증거가 없었다.**
- Q8_K에서 비영 입력이 0으로 바뀐 비율: GPT **44.26%**, Llama **1.11%**. 이는 **activation 반올림 손실**이다. 상위 3채널의 에너지 비중은 **96.63% / 1.35%**. 캡처 8,192행 통계이며 chunk 마지막 32행은 PPL에 미사용이다.
- 원본 F16 교체는 embedding과 head weight도 바꾸므로 activation만의 대조가 아니다. [head 대조](../../output/experiment/potal-model-compare-20260930/comparison.json), [재주입·dot 검사](../../output/experiment/q6-head-20260930/head-analysis.json).

추가 GPT PPL: FP16 **29.6258**, 전체 Q8_0 **31.3657**, Q4 body+공유 Q8_0 head **34.2204**, Q4 body+공유 Q4_0 head **547.7965**. 마지막 head도 CPU에서는 Q8_0 입력을 쓰며 embedding도 변경된다. 사용자 Q8_0 **27.45**는 평가 조건 미상. [Q8 대조](../../output/experiment/q8-q4-head-20260930/comparison.json), [Q4 head 대조](../../output/experiment/q8-q4-head-20260930/q4-head-control.json).

## 2. FP16에서 다시 만든 현재 기준 모델

FP16 두 원본에서 **현재 기본 옵션**으로 직접 양자화. `--pure`·head 강제 양자화·재양자화 없음.

| `models/` 파일 | body | 공유 embedding/head |
|---|---|---|
| `gpt2.Q4_0.gguf`, `llama3.2-1B.Q4_0.gguf` | Q4_0 | Q6_K |
| `gpt2.Q4_HP1.gguf`, `llama3.2-1B.Q4_HP1.gguf` | Q4_HP1 | **F16 원본 보존** |

새 Q4_0는 기존 PPL 측정 파일과 **전체 바이트 동일**. HP1 head는 원본 F16과, body는 앞서 검증한 현재 HP1 quantizer 결과와 바이트 동일하다.

구형 GPT HP1은 **head도 HP1**, 복원 weight 약 **16.4%**가 현재 방식과 달랐다. 사용자가 구형 방식임을 확인했다. 구형 GPT 파일을 교체하고 이전 Q4 실험 GGUF **14개 삭제**. 원본 FP16·측정 로그·JSON·명령·해시는 보존했다. [생성·정리 기록](../../output/experiment/q4-head-policy-20260930/models.json).

## 3. 후속 head 가설 검증

### Head도 반드시 양자화해야 하나?

**필수 아님. 정확도·저장량에 따른 선택이다.** 이 코드도 head에 별도 정밀도를 사용한다. [현재 정책](../../src/llama-quant.h), [output 선택](../../src/llama-quant.cpp). Hugging Face도 **8bit 예제**에서 `llm_int8_skip_modules=["lm_head"]`를 지원한다. 모든 모델에서 F16 head가 최적이라는 뜻은 아니다. [공식 문서](https://huggingface.co/docs/transformers/quantization/bitsandbytes#skip-module-conversion).

### 새 Q4_0 파일의 head weight를 직접 측정한 결과

모든 head block을 native C 재양자화·복원과 대조했다. `clamp`는 **반올림한 코드에 [-32,31] 제한이 실제 적용된 경우**다.

| Head weight 지표 | GPT-2 | Llama |
|---|---:|---:|
| 실제 clamp 비율 | **0.0181%** | **0.0191%** |
| 상대 제곱오차 | 0.03339% | 0.03363% |
| 전체 head weight 오차 중 clamp 위치의 제곱오차 비중 | 0.0402% | 0.0398% |

**GPT head weight가 유독 많이 잘린다는 설명은 측정과 맞지 않는다.** 큰 차이는 activation이다. 큰 값이 Q8_K block scale을 키워 작은 입력값을 0으로 반올림시키는 손실과 weight clipping을 구분해야 한다. [새 측정](../../output/experiment/q4-head-policy-20260930/head-weight-clipping.json), [코드](../../output/experiment/q4-head-policy-20260930/head-weight-clipping.py).

### 가설에 대한 결론

1. **현재 HP1의 head 생략: 확인.** CPU F16 head는 입력도 F16 경로를 사용해 Q8_K 변환을 피한다. [CPU 경로](../../ggml/src/ggml-cpu/ggml-cpu.c).
2. **GPT에 큰 이득, Llama에서는 변화가 매우 작음: 대조실험이 뒷받침.** 위 PPL 표와 Q8_K 재주입으로 head 경로의 영향을 확인했다. 아래 전체 모델 측정도 **GPT에서는 head 경로 개선 이득이 PoTal의 추가 PPL 증가를 상쇄하고, Llama에서는 head 변경 영향이 작다**는 설명과 맞는다.
3. **원래 표의 원인: 미확정.** HP1 body scale, A4 양자화, residual 보정도 달라진다. 아래 새 측정과 원래 표는 구분해야 한다. 특히 구형 HP1은 head도 HP1이었으므로 새 F16 정책으로 원래 **48.77에서 38.55** 결과를 소급 설명할 수 없다. 원래 파일·실행 조건 확인이 필요하다.

**비교 기준:** 양쪽 embedding/head 정책을 맞추고 activation 경로를 명시해야 PoTal 자체의 효과를 분리할 수 있다. 기본 preset 비교에는 head 정책 차이도 포함된다. 전체 Q4_0/HP1 파일 크기는 GPT **80.91/139.52 MiB**, Llama **735.21/1204.72 MiB**. F16 head 보존과 body metadata 크기 차이가 함께 반영된 값이다.

## 4. Head 입력 경로를 맞춘 실제 PoTal PPL

새 FP16 기반 모델, 위와 같은 c512·32 chunks·8,160개 채점 조건. **6개 실행 모두 완료·검증 통과.**

| 조건 | GPT-2 PPL | Llama PPL |
|---|---:|---:|
| Q4_0 body, 동일 Q6 복원값의 F32 head | 34.1601 | 11.1605 |
| Q4_0 body, 원본 F16 embedding/head | **32.1315** | **11.1627** |
| HP1 PoTal, 동일 원본 F16 embedding/head | **42.4047** | **15.2155** |

**동일 F16 head에서 PoTal PPL은 GPT +10.2732(+31.97%), Llama +4.0528(+36.31%)다.** 기본 preset끼리 비교하면 GPT는 52.2543에서 42.4047로 좋아지고 Llama는 11.1615에서 15.2155로 나빠지는 패턴이 나온다. Head 정책을 맞추면 두 모델 모두 PPL이 높아진다. 단, 이 비교만으로 HP1 weight·A4 activation·residual 각각의 기여를 분리하지는 못한다.

- 기준선: native CPU `Q4_0 × Q8_0` body. F32 head 대조는 Q6 복원 weight 값을 보존하며 정수 입력 변환을 제거한다. F16 대조는 공유 embedding/head weight도 원본으로 되돌린다.
- PoTal: CPU INT A4W4, ExSIA σ=2, DIM64, STRIPE_PIPELINE, LOCAL_FOLDING_PIPELINE, OMP 4 threads. Residual은 **CPU direct**, weight scaling은 host FP64이다. **하드웨어 SCU·packed limb 실행 결과가 아니다.** Head는 CPU F16 경로이며 입력의 F16 변환은 남는다.
- 같은 실행파일·CPU 모듈을 사용했다. `--no-op-offload`만으로는 Gemmini 개입이 남아 해당 시도를 제외했고, 기준선은 Gemmini 모듈 없는 별도 디렉터리에서 실행했다. 실제 dense/RMD 호출은 GPT 1,536회·Llama 3,584회였다. 전 chunk의 CPU head 배정, 종료 코드 0과 실행 실패 기록 0을 확인했다.

[전체 결과·실행 증거](../../output/experiment/potal-ppl-20261001/comparison.json), [설정·해시·제외한 시도](../../output/experiment/potal-ppl-20261001/experiment.json), [검증·집계 코드](../../output/experiment/potal-ppl-20261001/summarize.py).

## 재현 범위

Head weight 검사: `uv run --script output/experiment/q4-head-policy-20260930/head-weight-clipping.py` (`--self-check`로 경계값 검사). 이번 PPL 로그 검증·집계: `python3 output/experiment/potal-ppl-20261001/summarize.py` (Python 3.12+). 실행 명령은 각 run의 `manifest.txt`에 있다. 삭제한 과거 대조 GGUF는 기록된 명령으로 재생성해야 과거 스크립트를 실행할 수 있다. 제품 실행 코드 수정 없음.
