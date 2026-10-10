# Attention의 limb와 보정 비용 실측

원문 selective-linear+a4nks 정책을 고정하고 GPT-2의 실제 Q/K/P/V로 측정했다. **Q의 상위 limb는 희소한 이상치만의 비용이 아니었다. CPU 재생에서는 이상치 선택보다 상위 limb 곱셈과 패킷 생성 비용이 컸다.**

## 512토큰에서 limb가 얼마나 생겼나

12개 layer × 12개 head 전체를 집계했다. Main은 모든 경우 한 번 처리한다. 아래의 비율은 각 operand의 모든 원소를 분모로 하며, P는 causal mask로 0이 된 원소도 포함한다.

| Operand | 실행에 사용한 digit 면 | 실제 분포 |
|---|---|---|
| Q | main + 상위 1번 + 상위 2번 | 90.90%의 값에 상위 digit이 있음. 15.77%는 상위 2번 자리까지 필요 |
| K | low/high, dense 두 패스 | 원문이 고정한 8비트 표현. 상위 패스도 항상 실행 |
| P | main + 상위 1번 + 상위 2번 | 상위 digit이 있는 값은 0.964%. 상위 2번은 0.063% |
| V | main 하나 | 원문의 4비트 정책. 추가 limb 없음 |

Q의 main 외 **0/1/2개의 0이 아닌 상위 digit**을 갖는 원소 비율은 9.10% / 75.68% / 15.22%다. 상위 2번 자리만 있고 1번이 0인 값도 있으므로, 마지막 비율은 2번 자리가 필요한 15.77%와 다르다. 전체 1,152개 Q stripe 모두 두 상위 자리의 패킷이 있었고, 상위 1번/2번의 활성 행 비율은 100% / 99.72%였다. 따라서 행 제거로 Q 상위 패스가 크게 줄어들지 않았다.

Q의 이상치 선택 비율은 top exponent 9.23% + sigma 3.55% = **12.78%**였다. 상위 digit이 필요한 90.90%와 크게 다르다. 원문 Q는 모든 값을 보정하므로 일반적인 8비트급 값도 4비트 main에 담기지 않는 높은 자릿수를 사용한다. P/K/V에는 이 이상치 선택 단계가 없다.

P는 원소별로 희소했지만, 상위 1번이 필요한 행은 96.82%, 상위 2번이 필요한 행은 29.83%였다. 행마다 드문 큰 확률이 서로 다른 열에 나타나면, 행과 열의 합집합 및 DIM32 패딩이 큰 패킷을 만든다. 실제 0이 아닌 digit / 할당된 상위 packet bytes는 **Q 53.06%, P 1.69%**였다. P의 보정 패킷은 이 표현에서 약 98.3%가 0이었다.

## 행렬 연산의 보정 비중

아래는 원문의 DIM32 패딩 공식으로 센 fragment다. CPU 시간이나 실제 Gemmini 제출 횟수가 아니다. QK main 한 번과 PV main 한 번을 각각 1로 두면:

| 연산 | main 한 번 대비 작업량 |
|---|---:|
| QK: Q main × K low | 1 |
| QK: Q 상위 두 면 × K low | 2 |
| QK: Q main × K high | 1 |
| QK: Q 상위 두 면 × K high | 2 |
| **QK 합계** | **6.000** |
| PV: main | 1 |
| PV: P 상위 보정 | 0.607 |
| **PV 합계** | **1.607** |

두 GEMM을 합치면 단일 A4 main 두 번 대비 **3.803배**다. 전체 QK+PV fragment 중 **상위 limb 보정은 60.56%**, K-high main까지 추가 정밀도 연산으로 포함하면 **73.71%**다. K-high 전체와 Q 보정 전체는 서로 겹치므로 둘의 비율을 그대로 더하면 안 된다.

이 비용은 원문 정책 자체에 들어 있다. 정확도를 보존하며 중간 residual 배열만 없애는 변경은 같은 코드와 패킹 조건에서 **추가 GEMM을 요구하지 않는다**. 앞서 시험한 P 8행 분할의 PV 약 3배 증가는 채택안에 포함하지 않는다. 제품 경로의 변경 전후 실행시간은 아직 측정하지 않았다.

## CPU에서 이상치 선택과 compensation을 따로 실행한 시간

Layers 0/5/11 × heads 0/6/11의 9개 head를 사용했다. 각 head는 1회 warmup 후 5회 반복했다. 각 반복에서 9개 head의 평균을 구한 뒤 중앙값을 보고한다. NumPy int64로 실제 main/upper dot을 실행하고, 검사한 범위 안에서 integer shift와 누산을 수행했다.

| 구성요소 | 평균 head당 중앙값 | 측정 구간 합계 대비 |
|---|---:|---:|
| Q exponent/top 선택 + 예비 양자화 + sigma 선택 | **0.710 ms** | **0.65%** |
| 그중 sigma 통계·판정만 | 0.269 ms | 0.25% |
| Q/P digit 패킷 생성 | **7.289 ms** | **6.66%** |
| Q 상위 보정 곱셈·shift·합산, K 두 패스 포함 | **52.894 ms** | **48.11%** |
| P 상위 보정 곱셈·shift·합산 | **3.009 ms** | **2.74%** |
| P zero-point 보정: V colsum + 출력 broadcast | **0.116 ms** | **0.11%** |

Sigma 행은 첫 행의 일부이므로 더하지 않는다. Q/P 상위 보정은 합계 약 51%이며, 여기에 K-high의 main 비용은 포함하지 않았다. 보정 곱셈과 P의 zero-point 덧셈은 비용이 매우 달랐다.

양자화·패킷 준비 부분만 분모로 보면 이상치 선택은 **6.77%**, 패킷 생성은 **69.82%**였다. 이 CPU 구현에서는 이상치 선택을 없애는 것보다 패킷 생성과 보정 실행을 줄이는 쪽이 더 큰 측정 대상이다. 이상치 선택 제거는 E를 바꾸므로 정확도 유지 최적화로 간주하지 않는다.

측정 구성요소의 합계는 head당 중앙값 109.48 ms, 반복 p10~p90은 107.49~111.60 ms였다. 이것은 **전체 head 지연시간이 아니다**. 캡처, 검사, 참조용 복원/FP QK/행렬 mask/packet unpack, driver overhead는 분모에서 제외했다. 원래 FP32 BLAS forward와도 실행 방법이 다르므로 이 숫자들을 나눠 속도 저하율로 사용하지 않는다. 단일 스레드 NumPy int64 dot과 Python packet 준비의 비중을 Gemmini PE 및 native C++ 준비의 비중으로 옮길 수 없다.

## Context에 따른 변화

| Context | QK / QK main | PV / PV main | 상위 limb fragment 비중 | CPU 이상치 선택 비중 | packet 생성 / 준비 시간 |
|---:|---:|---:|---:|---:|---:|
| 128 | 6.000 | 2.023 | 62.61% | 3.36% | 55.27% |
| 512 | 6.000 | 1.607 | 60.56% | 0.65% | 69.82% |
| 1024 | 6.000 | 1.380 | 59.35% | 0.14% | 72.66% |

각 context의 첫 test chunk를 사용했다. 토큰 수와 입력 분포도 달라지므로 길이만으로 모든 차이를 설명하지 않는다. 1-token decode, Llama/GQA, 실제 Gemmini 실행은 이 측정에 포함하지 않았다.

## 검증과 재현

- 3개 context × 144 heads = **432 head 입력**의 분포와 packet을 검사했다. 캡처한 각 context의 최종 hidden이 기존 원문 정책 실행의 첫 chunk와 bitwise 일치했다.
- 프로파일용 Q/K/P/V 양자화와 packet의 main/digit/row/lane/column/scale 및 fragment를 기존 CPU 원문 정책 함수와 비교했다. 모두 일치했다.
- 시간 측정한 **27개 head의 QK/PV 54개 integer GEMM**은 복원 operand의 float64 GEMM과 bitwise 일치했다. 반복 결과도 같았다. 모든 shift·누산 전 int64 overflow 상한을 검사했고 범위를 넘으면 실패하게 했다.
- FP32 forward의 QK/softmax로 만든 P를 고정한 component replay다. Integer GEMM 출력을 모델 전체에 연결해 평가한 새로운 end-to-end PPL 결과는 아니다.
- 기존 모델/정책 코드에는 변경이 없다. 새 profiling 파일 4개는 Python 문법 및 no-excuse 코드 검사를 통과했다.

원자료와 표: [MEASUREMENTS.md](results-attention-profile/MEASUREMENTS.md), [limbs.csv](results-attention-profile/limbs.csv), [SHA-256](results-attention-profile/artifact-sha256.json). 각 JSON은 head별 수치와 반복별 시간, checkpoint/token/source hash를 담는다. 캡처 배열도 같은 디렉터리에 보관했다.

작업 디렉터리는 `/Users/chan/Projects/AISA/worktrees/potal-attn`이다. 먼저 기존 [실험 재현](README.md)으로 원문 baseline JSON, `.chunk0.npz`, `test.i32`를 생성한다. 다음은 새 결과 디렉터리로 프로파일을 재현하는 명령이다. 시간 비교를 위해 세 실행을 동시에 돌리지 않는다.

```sh
rtk proxy env PYTHONPATH=gguf-py:. OPENBLAS_NUM_THREADS=2 /Users/chan/Projects/AISA/llama.cpp-gemmini/.venv-convert311/bin/python -m experiments.potal_a4.profile_attention experiments/potal_a4/results-v3/test128-original_a4nks.json experiments/potal_a4/results-profile-new/context128.json
rtk proxy env PYTHONPATH=gguf-py:. OPENBLAS_NUM_THREADS=2 /Users/chan/Projects/AISA/llama.cpp-gemmini/.venv-convert311/bin/python -m experiments.potal_a4.profile_attention experiments/potal_a4/results-v3/test512-original_a4nks.json experiments/potal_a4/results-profile-new/context512.json
rtk proxy env PYTHONPATH=gguf-py:. OPENBLAS_NUM_THREADS=2 /Users/chan/Projects/AISA/llama.cpp-gemmini/.venv-convert311/bin/python -m experiments.potal_a4.profile_attention experiments/potal_a4/results-v3/test1024-original_a4nks.json experiments/potal_a4/results-profile-new/context1024.json
rtk proxy /Users/chan/Projects/AISA/llama.cpp-gemmini/.venv-convert311/bin/python -m experiments.potal_a4.profile_report experiments/potal_a4/results-profile-new
```
