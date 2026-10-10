# PoTal A4: 직접 limb 분해의 공간·정확도 실험

`hotfix/potal-attn`, 기준 commit `3b2eaa3cb6f7adc36094b7f78662141825cc4ef4`.

**중간 int32 residual 배열 제거는 공간 이득이 확인됐다. 그러나 4비트 PE를 사용해도 저장량이 4비트가 되는 것은 아니며, limb를 강제로 줄이면 정확도가 크게 나빠졌다.** P의 스케일 공유 범위를 줄이면 정확도를 회복할 수 있지만 현재 DIM32 패킷 구조에서는 연산과 패딩 비용이 증가했다.

이 결과는 CPU 정책 실험과 실제 C++ packet 생성 실측이다. 제품의 Gemmini attention 실행 경로를 완성한 결과는 아니다. 재현 명령과 구현 범위는 [README](README.md), 전체 수치는 [측정 표](results-v3/TABLES.md), [CSV](results-v3/summary.csv), [native 공간 표](results-v3/NATIVE.md)에 있다.

## 1. 정확도: 전체 값을 보존해도 모델 성능이 반드시 좋아지지는 않았다

GPT-2 124M의 동일한 Q4_HP1 체크포인트를 사용했다. 실제 Q4로 저장된 48개 선형 행렬에만 입력 양자화를 적용하고, FP16 embedding과 tied output head는 보존했다. Attention은 양자화한 Q/K로 점수를 다시 구하고 softmax를 계산한 뒤 P/V를 양자화했다.

WikiText-2 valid 앞 512 tokens로 12조건을 비교했다. 별도 test 앞 4,096 tokens를 context 512의 8개 chunk로 나누고, 각 chunk 뒤 절반의 next-token 255개, 총 2,040개를 채점했다. PPL은 낮을수록 좋고, 적중률은 다음 token의 top-1 정답률이다.

| 정책 | PPL | 다음 token 적중률 | 해석 |
|---|---:|---:|---|
| Q4 weight만, activation FP32 | 30.83 | 36.27% | 같은 weight의 비교 기준 |
| 선형만 기존 selective | 35.45 | 33.92% | Attention operand는 FP32 |
| 선형만 direct limb | 35.75 | 34.71% | 일부 clipping 제거가 PPL 개선을 보장하지 않음 |
| 원문 selective-linear + a4nks | **35.38** | **34.75%** | 아래 변경의 기준 |
| 선형·Q direct, 원래 P stripe | 36.55 | 33.82% | 원문 대비 PPL +3.32% |
| direct + P 8행 단위 scale, P 8비트 | **35.36** | **34.26%** | PPL은 기준과 비슷하나 PV 비용 증가 |
| 위 조건에서 P만 6비트 | 38.00 | 34.26% | 공간을 줄이는 대신 PPL 악화 |
| P 8행·8비트, 선형·Q 상위 limb 최대 1개 | **89.74** | **26.18%** | Q 값의 15.5%가 표현 범위를 넘어 포화 |

FP16 weight + activation FP32의 별도 참고 PPL은 25.82다. 따라서 activation 정책의 변화와 이미 적용된 weight 양자화 손실을 구분해야 한다.

원문 대비 PPL 변화의 paired chunk bootstrap 95% 구간은 direct stripe가 [+0.76%, +6.00%], P 8행이 [-2.62%, +2.66%]였다. 후자를 개선이라고 판정할 근거는 부족하다. 8개 chunk의 탐색적 구간이며 전체 데이터셋이나 다른 모델로 일반화하지 않는다. Q는 원문부터 전체 보정하므로 direct 전환의 정책 차이는 선형에서 발생한다.

## 2. 공간: residual plane 제거 효과와 4비트 저장 효과는 다르다

기존 C++ `QuantizedActivationBuffer`, balanced decomposition, `RmdBitmapBuilder`를 실행했다. 512×4096개의 동일한 folded code를 stripe 32, DIM32로 처리하고, 조건마다 새 프로세스에서 3회 반복했다.

비교 기준은 **clipped main + 완전 보정 + dense int32 residual plane**이다. 변경은 **low-digit main + residual plane 생략**이다. 두 표현이 동일한 folded code를 복원하는지 모두 검사했다. 이 공간 비교의 기준은 원문 selective clipping의 정확도 정책과 다르다.

| 입력 제한 | 기존 유지 공간 | direct 유지 공간 | 감소율 | 같은 값의 FP32 저장 대비 direct |
|---|---:|---:|---:|---:|
| 모두 -7..7 | 10.00 MiB | 2.00 MiB | 80.0% | 0.25배 |
| 1,000개당 하나가 127 | 12.09 MiB | **4.08 MiB** | **66.2%** | 0.51배 |
| 전체가 -128 또는 127 | 17.09 MiB | 6.96 MiB | 59.3% | 0.87배 |
| 전체가 -65,536 또는 65,535 | 17.09 MiB | 4.83 MiB | 71.7% | 0.60배 |
| 전체가 0x77777778, 9개 digit 필요 | 27.72 MiB | **19.72 MiB** | 28.9% | **2.47배** |

여기서 유지 공간은 main, residual plane, digit payload, packet 객체와 index metadata의 크기를 합한 값이며 allocator의 여유 capacity는 제외했다. 이 모양의 FP32 값 배열은 8 MiB, FP16은 4 MiB다. 현재 byte ABI에서는 main만 있어도 2 MiB이고, nibble로 압축된 4비트 배열의 1 MiB와 다르다.

독립 프로세스 peak RSS 중앙값도 sparse 조건에서 15.47→7.44 MiB로 감소했다. RSS에는 입력 생성 및 복원 검증용 scratch가 포함된다. 모델 전체 메모리나 accelerator SRAM 절감량을 뜻하지 않는다.

Sparse 조건의 유효 상위 digit은 6,294→4,196개로 줄었지만 digit payload는 양쪽 모두 1,762,304 bytes였다. DIM 패딩 때문에 nonzero 감소가 저장 감소로 그대로 이어지지 않았다. 큰 값이라도 상위 digit 하나만 있으면 작게 표현될 수 있고, 여러 자리에서 carry가 생기는 값은 더 큰 패킷을 만든다.

## 3. 실제 모델의 operand packet도 4비트 크기는 아니었다

Context 512의 `direct_p8`에서 생성된 int8 main/digit 배열과 row/lane·K index·scale의 실제 `nbytes`를 합산했다. 여러 호출의 누적 bytes를 누적 원소 수로 나눈 평균이며 동시에 살아 있는 전체 모델 메모리가 아니다.

| Operand | bytes/값 | FP16 값 배열 2 bytes/값 대비 |
|---|---:|---:|
| 선형 입력 | 2.038 | 약 1.9% 증가 |
| Q | **3.188** | **약 59.4% 증가** |
| P, 원래 stripe | 1.611 | 약 19.4% 감소 |
| P, 8행 scale | 1.818 | 약 9.1% 감소 |
| P, 8행 scale·6비트 | 1.460 | 약 27.0% 감소 |

Q의 8비트급 정밀도를 signed 4비트 digit으로 표현할 때 carry와 stripe folding 때문에 상위 plane이 두 개 필요했고, 이 데이터에서는 main을 포함해 세 개의 byte plane에 가까운 크기가 됐다. K도 dense nibble 두 pass이므로 QK 작업량은 이상적인 단일 A4 main의 약 6배였다. 이는 packet에서 산출한 fragment 작업량이며 실행시간 측정값이 아니다.

P 8행 조건은 원문 대비 P 패킷이 약 12.8% 커졌다. PV는 원래 main 대비 1.596→4.788배로, **전체 PV fragment가 약 3배**가 됐다. 8행 묶음도 32행짜리 타일을 차지하기 때문이다. 각 정책의 증가한 main을 따로 분모로 삼으면 이 손해가 가려지므로 동일한 원래 main을 분모로 사용했다.

Python forward는 원본 FP32, 복원 operand, weight, 중간값을 함께 보유한다. 실제 프로세스 peak는 weight-only 약 1,008 MiB에서 direct_p8 약 1,148 MiB로 증가했다. 따라서 이 실험을 전체 추론 RAM이 줄었다는 증거로 사용하면 안 된다. KV cache 압축과 weight 파일 크기 변경도 포함하지 않는다.

## 4. 제약을 강하게 걸었을 때의 실패

- **P 4비트:** valid 비교에서 PPL 30.44→49.91, 확률 행 13개가 전부 0이 됐다. 두 조건 모두 direct, P 8행 scale이다.
- **상위 limb 제거:** 선형·Q를 main 하나로 제한하면 valid PPL이 89,317.72였다. 상위 limb 한 개도 test PPL 89.74로 충분하지 않았다. 높은 digit을 그대로 버리지 않고 가능한 범위로 포화시킨 결과다.
- **P 행마다 scale:** valid PPL 31.11로 8행 조건 30.44보다 좋아지지 않았고, 패킷은 9.566 bytes/값으로 FP32보다 커졌다.
- **넓은 P stripe:** Q=K=0, causal attention, V=1, T=256, d=32의 독립 반례에서 마지막 출력의 정답은 1이다. 원래 geometry의 P 8비트 양자화는 0, P scale을 32/8/1행으로 나누면 모두 1을 복원했다. 작은 확률이 큰 첫 행과 scale을 공유하며 half-even rounding으로 사라졌다.
- **Context 변화:** 원문→direct_p8의 PPL은 context 128에서 41.82→44.36, 512에서 35.38→35.36, 1024에서 30.20→29.77이었다. 길이에 따라 채점 token이 다르므로 같은 길이 안에서만 비교한다. 8행 설정이 모든 조건에서 우세하지 않았다.

Synthetic 128조건은 [stress.json](results-v3/stress.json)에 있다. Outlier 크기 1/16/256/65,536, 상위 limb 0/1/2/8개, context 128/256/512/1024, head width 32/64, P 4/6/8비트와 scale 공유 범위를 교차했다.

## 5. 이 결과로 정한 구현 순서와 검증 한계

사용자의 최신 조건은 **원문 selective-linear+a4nks 대비 정확도 저하 금지**다. 따라서 direct-all 선형, limb 제한, P 8행 세분화는 채택하지 않는다. PPL 35.36과 35.38이 비슷하다는 것만으로 정확도 유지가 증명된 것도 아니다.

먼저 **원문이 정한 값을 정확히 유지하면서 dense int32 residual plane을 없애는 변경**을 구현·검증한다. 선형은 원래 `corr = out || D>0`를 유지하고 `v = corr ? u : sat(u)`를 직접 분해한다. Q는 `v=u`를 필요한 digit 전부로 분해한다. 이렇게 하면 scalar residual을 만들지 않고도 원문의 main과 upper digit을 그대로 만들 수 있다. P 8비트와 원래 stripe, K/V 정책도 유지한다. 같은 weight·token·수치 실행 경로에서 원문 대비 코드, scale, 복원값, logits, token별 NLL/PPL의 동일성이 통과 기준이며 제품 경로에서 아직 이 검증을 마치지 않았다. 이 기준은 FP32 대비 양자화 손실까지 없어진다는 뜻은 아니다.

실험 구현에는 기존 scalar emitter로 연결하는 bridge가 남아 있다. `u-d0`가 int32 범위를 넘는 경우 명시적으로 실패하므로 full-int32 fused upper emitter 구현 완료나 준비시간 단축을 주장하지 않는다. CPU forward의 matmul은 복원된 FP32 값을 사용한다. Integer SCU 누산, 실제 Gemmini dispatch/RTL/보드의 정확도·지연시간·에너지는 아직 검증하지 않았다. 참조의 accumulator geometry를 사용한 Python fragment를 현재 하드웨어 스케줄 실측으로 취급하지 않는다.

검증 증거:

- [Python self-check](results-v3/verify.log): 141,076개 integer packet roundtrip, INT32_MIN/MAX, 9-digit carry, shift 경계, tail, zero, +8 선형 경계와 causal P 반례 통과.
- [C++ CTest](results-v3/ctest.log): 기존 bitmap LOG0/LOG1 및 native packet 비교 4/4 통과. 공간 측정의 30개 native 실행도 packet 복원을 검사했다.
- [Weight decoder 대조](results-v3/weight-parity.json): 48개 Q4_HP1 tensor의 193,536개 값을 native C decoder와 bitwise 일치 확인.
- FP16의 별도 native CPU sanity check는 PPL 19.5856, NumPy는 19.4979로 약 0.45% 차이였다. 동일 forward의 bitwise 일치를 확인한 것은 아니다. 기존 CPU 라이브러리는 Q4_HP1 모델을 compatible buffer 부재로 로드하지 못해 native Q4 전체 추론 대조는 남아 있다.
- 최종 26개 모델 조건의 JSON에 checkpoint/token/source hash를 기록했다. [artifact SHA-256](results-v3/artifact-sha256.json)과 [코드 검사](results-v3/code-check.log)를 보관했다. FP16 head까지 입력을 양자화했던 이전 탐색 결과는 최종 비교에서 제외했다. 일부 JSON의 출력층 행 수 metadata 이름만 `head_rows_per_chunk`로 바로잡았고 측정 수치는 바꾸지 않았다.

실험은 별도 worktree에 있으며 `develop`의 제품 코드를 변경하거나 merge하지 않았다.
