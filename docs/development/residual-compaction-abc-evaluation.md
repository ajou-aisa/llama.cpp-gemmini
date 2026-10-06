# Residual 행·K 압축 A/B/C 평가

비교 대상은 **A: 행·K 압축 유지, B: 행 pruning만 유지, C: 행·K 압축 제거**다. A*는 A와 같은 입력 배열을 만드는 단순 구현으로, B/C와 구현 비용을 맞춰 비교하는 대조군이다. 모두 residual 보상과 limb 분해를 유지한다.

여기서 residual은 선택된 activation outlier의 clipping 손실을 보상하는 INT32 값이다. Balanced limb는 이를 W4A4에서는 16진, W8A8에서는 256진의 signed digit로 나눈 것이다. HP1 weight는 정수 code와 K32 block별 2의 거듭제곱 scale을 쓴다. GEMM의 M/N/K는 각각 limb 행 수·output column 수·축적 길이이며, DIM16/64는 systolic array 한 변의 크기다.

## 결과와 판단

**K 압축은 모든 비교 조건에서 GEMM cycles를 줄였다. 현재 A의 준비 비용 때문에 GPT-2 W8A8 prefill에서는 B가 더 빠른 조건이 있다. 전체 경로의 선택은 weight 준비 방식과 클록을 함께 봐야 한다.**

- DIM16에서 K 압축의 cycle 절감은 prefill 15.64~20.68%, decode 40.69~43.21%다. 같은 residual의 DIM64 geometry에서는 각각 5.77~15.75%, 20.54~32.42%다.
- 행 pruning은 prefill cycles를 2.24~11.87% 줄였다. 이 prompt의 decode에서는 B와 C의 행 수가 같아서 행 pruning의 GEMM 이득이 0이다.
- 매 stripe에서 weight를 다시 준비하는 DIM16 조건에서 현재 A는 8개 구간 중 7개에서 B보다 조건부 비용이 낮았다. GPT-2 W8A8 prefill은 B가 낮았다.
- 미리 decode한 호스트 weight를 재사용하면 1GHz 환산에서 A*가 B/C보다 낮았다. 이 조건은 초기 준비 비용을 상각한 이후의 prototype 비용이다.

따라서 **K 압축을 유지하면서 weight 준비·재사용을 먼저 개선하는 방향**이 이 데이터에 맞는다. GPT-2 W8A8 prefill은 현재 A/B 구현을 직접 비교할 가치가 있는 예외다. C는 이 prompt의 decode에서 B와 같은 GEMM을 만들며 row map 준비를 생략하는 단순 경로 후보가 된다. 여러 prompt에서도 행 pruning 이득이 없는지는 별도 확인이 필요하다.

## DIM16: 현재 A와 단순 B/C의 비용

아래는 측정한 CPU 구간과 추정 NPU 구간의 합이다. **prefill 행은 1회, decode 행은 3회 누계**, 단위 ms, NPU **1GHz 가정**이다. backend 호출·동기화와 공통 float merge를 제외한 부분 비용이며 전체 inference 시간은 아니다. A는 현재 builder의 재생 실측, B/C는 단순 prototype 실측이다. A의 재생에는 production이 이미 제공하는 selection bitmap을 다시 만드는 비용도 포함한다.

| 모델 | 정밀도 | 구간 | A | B | C |
|---|---|---|---:|---:|---:|
| GPT-2 | W4A4 | prefill | 336.787 | 362.801 | 383.671 |
| GPT-2 | W4A4 | decode | 137.930 | 411.600 | 410.488 |
| GPT-2 | W8A8 | prefill | 363.227 | 303.531 | 308.824 |
| GPT-2 | W8A8 | decode | 152.889 | 320.023 | 318.776 |
| Llama 3.2 1B | W4A4 | prefill | 3,737.435 | 9,946.658 | 10,071.763 |
| Llama 3.2 1B | W4A4 | decode | 1,692.292 | 13,583.503 | 13,697.568 |
| Llama 3.2 1B | W8A8 | prefill | 3,898.432 | 9,989.925 | 9,980.190 |
| Llama 3.2 1B | W8A8 | decode | 1,830.902 | 13,851.120 | 13,867.252 |

GPT-2 W8A8 prefill을 분리하면 A의 CPU 준비·복원은 254.189ms, B는 166.061ms다. GEMM은 A 109.038M cycles, B 137.470M cycles로 A가 28.431M cycles를 절약하지만 CPU에서는 88.128ms를 더 쓴다. 같은 cycle profile을 고정하면 손익분기 클록은 **약 0.323GHz**다. 이보다 낮으면 A, 높으면 B의 조건부 합계가 낮다.

추가로 bitmap allocation·zeroing·event scatter만 동일한 2회 warmup·7회 반복으로 따로 쟀다. GPT-2 W8A8 prefill의 stripe별 중앙값 합은 **0.371ms**로, 위 88.128ms 차이보다 훨씬 작았다. 전체 조건의 bitmap 준비 합계는 [bitmap-cost.csv](../../output/experiment/compaction-abc-20261004/bitmap-cost.csv)에 있다. 별도 microbenchmark이므로 기존 시간에서 단순 차감하지 않았으며, 실제 production 경로의 A/B 우열을 확정하는 측정도 아니다.

같은 단순 builder끼리 비교한 A*는 이 구간에서 CPU 92.018ms, 1GHz 합계 201.056ms다. 따라서 이 예외를 K 압축 자체의 손해로 해석할 수는 없다. 반대로 Llama W4A4 prefill의 단순 A* CPU는 4,564.897ms로 현재 A 2,409.866ms보다 느리다. 단순화만으로 준비 시간이 줄어든다는 가정도 성립하지 않는다.

## Residual GEMM cycles: DIM16·64

단위 **Mcycles**. 1GHz 환산이면 같은 숫자가 ms이고, 500MHz이면 두 배다. prefill 1회·decode 3회의 isolated request 누계다. 행·K 압축 자체의 device 효과는 현재 A와 A*에서 동일하다.

| 모델 | 정밀도 | DIM | 구간 | A | B | C | A/B cycle 절감 |
|---|---|---:|---|---:|---:|---:|---:|
| GPT-2 | W4A4 | 16 | prefill | 112.602 | 141.229 | 159.665 | 20.27% |
| GPT-2 | W4A4 | 16 | decode | 42.563 | 71.762 | 71.762 | 40.69% |
| GPT-2 | W4A4 | 64 | prefill | 86.649 | 93.525 | 106.120 | 7.35% |
| GPT-2 | W4A4 | 64 | decode | 48.191 | 60.646 | 60.646 | 20.54% |
| GPT-2 | W8A8 | 16 | prefill | 109.038 | 137.470 | 141.294 | 20.68% |
| GPT-2 | W8A8 | 16 | decode | 43.559 | 74.085 | 74.085 | 41.20% |
| GPT-2 | W8A8 | 64 | prefill | 96.303 | 114.303 | 117.108 | 15.75% |
| GPT-2 | W8A8 | 64 | decode | 49.776 | 71.024 | 71.024 | 29.92% |
| Llama 3.2 1B | W4A4 | 16 | prefill | 1,327.569 | 1,603.844 | 1,773.437 | 17.23% |
| Llama 3.2 1B | W4A4 | 16 | decode | 483.955 | 852.114 | 852.114 | 43.21% |
| Llama 3.2 1B | W4A4 | 64 | prefill | 1,151.966 | 1,320.489 | 1,465.705 | 12.76% |
| Llama 3.2 1B | W4A4 | 64 | decode | 549.698 | 813.401 | 813.401 | 32.42% |
| Llama 3.2 1B | W8A8 | 16 | prefill | 1,328.472 | 1,574.811 | 1,618.917 | 15.64% |
| Llama 3.2 1B | W8A8 | 16 | decode | 497.128 | 851.902 | 851.902 | 41.64% |
| Llama 3.2 1B | W8A8 | 64 | prefill | 1,258.167 | 1,335.252 | 1,365.884 | 5.77% |
| Llama 3.2 1B | W8A8 | 64 | decode | 571.096 | 812.991 | 812.991 | 29.75% |

Logical K의 감소는 prefill 59.45~63.87%, decode 89.53~91.91%였다. 그러나 DIM64 prefill의 padded MAC 감소는 1.89~3.52%다. K32 scale 경계마다 최소 DIM64 fragment를 실행하기 때문이다. cycle model은 memory/scale/queue 비용도 포함하므로 MAC 감소율만으로 GEMM 시간을 환산하지 않았다. 이 수치 차이는 실제 보드의 병목 검증을 대신하지 않는다.

## Weight 재사용: 초기화 이후의 DIM16 prototype 비용

아래는 A*의 compact copy를 추가 실측하고 B/C가 준비된 전체 호스트 buffer를 직접 참조하는 조건이다. 단위 ms, 1GHz 가정, prefill 1회·decode 3회 누계다. **초기 decode 비용은 제외**한다.

| 모델 | 정밀도 | 구간 | A* | B | C | A*/B 손익분기 GHz |
|---|---|---|---:|---:|---:|---:|
| GPT-2 | W4A4 | prefill | 130.612 | 147.297 | 166.570 | 2.397 |
| GPT-2 | W4A4 | decode | 48.290 | 72.313 | 72.270 | 5.642 |
| GPT-2 | W8A8 | prefill | 128.947 | 143.930 | 147.839 | 2.114 |
| GPT-2 | W8A8 | decode | 50.049 | 74.687 | 74.636 | 5.184 |
| Llama 3.2 1B | W4A4 | prefill | 1,525.610 | 1,635.795 | 1,806.450 | 1.663 |
| Llama 3.2 1B | W4A4 | decode | 540.413 | 854.741 | 854.580 | 6.839 |
| Llama 3.2 1B | W8A8 | prefill | 1,539.780 | 1,608.764 | 1,652.342 | 1.389 |
| Llama 3.2 1B | W8A8 | decode | 557.724 | 854.516 | 854.356 | 6.119 |

손익분기점보다 낮은 클록에서는 A*, 높은 클록에서는 B의 부분 비용 합계가 낮다. CPU 중앙값과 고정된 cycle profile에서 계산한 점 추정이며, 클록에 따른 실제 memory timing 변화나 통계적 신뢰구간을 반영한 값은 아니다.

재사용을 준비하는 비용도 있다. 단순 full-weight decoder의 한 번 실행을 재면 GPT-2 W4/W8은 약 0.410/0.426초, Llama W4/W8은 약 8.427/8.588초였다. 필요한 INT32 buffer 총량은 GPT-2 **324 MiB**, Llama **3,712 MiB = 3.625 GiB**다. 이 초기 비용을 이 prompt 한 번에 모두 청구하면 cached A*의 합계는 GPT-2 W4/W8 588.648/605.095ms, Llama W4/W8 10,492.827/10,685.810ms로 현재 A의 474.717/516.116ms, 5,429.728/5,729.333ms보다 높다. 첫 요청의 이득으로 주장할 수 없으며 반복 실행에서의 상각과 준비 구현의 개선이 필요하다.

동일한 residual workload를 반복하며 buffer를 유지하고 1GHz를 가정하면, `초기 비용 + 요청 수 × cached A* 비용 < 요청 수 × 현재 A 비용`은 GPT-2 W4/W8에서 **2회**, Llama W4/W8에서 **3회**부터 성립한다. 한 번 측정한 초기 비용과 반복 구간의 중앙값에 근거한 상각 계산이며, 서로 다른 prompt의 실제 요청 수 기준은 아니다.

## 비교 조건

동일한 실제 residual을 balanced limb로 분해한 뒤 아래 세 경로를 비교했다. CPU 준비·복원은 Apple M5에서 실측하고, residual GEMM은 Gemmini cycle model로 추정한다. 전체 inference latency와 residual GEMM 비용을 구분한다.

| 경로 | Zero-limb-row pruning | K compaction | Residual GEMM 입력 |
|---|---|---|---|
| A | 적용 | 적용 | 현재 production builder의 생존 행·압축 K |
| B | 적용 | 미적용 | A와 같은 생존 행·원래 K 위치 |
| C | 미적용 | 미적용 | 원래 limb 행·원래 K 위치 |

모든 경로에 residual 보상과 limb 분해를 유지한다. C의 행 수는 해당 stripe에 필요한 limb 수 × 원래 stripe 행 수다. INT32의 최대 limb 수를 무조건 실행하지 않는다. 완전히 비어 있는 residual stripe는 세 경로에서 모두 건너뛴다.

현재 A 구현의 준비 비용과 압축 자체의 비용을 구분하기 위해 **A\***도 측정했다. A*는 B/C와 같은 단순 직접 pack 코드에 행·K 압축을 적용한 대조군이다. 각 stripe에서 A*의 activation·weight 전 원소, 행 identity, K run mask를 현재 A와 대조했다. A*는 production backend에 연결한 최적화 패치가 아니다.

## 실제 입력과 측정

- 모델: GPT-2, Llama 3.2 1B. 정밀도: HP1 W4A4와 W8A8. Token embedding과 output head는 F16이다.
- 입력: 보관한 WikiText-2 test 앞부분 80단어. GPT-2는 prefill 91 tokens, Llama는 100 tokens다. 4 tokens를 생성하며 실제 decode 호출은 3회다.
- GPT-2: nonempty residual stripe 238개, prefill 94개·decode 144개. Llama: 557개, prefill 221개·decode 336개. 두 정밀도에서 stripe 개수는 같다.
- prefill의 bulk stripe는 GPT-2 80+11행, Llama 80+20행이고, 마지막 FFN은 1행이다. 설정 hint 16행을 실험의 실제 stripe 크기로 간주하지 않았다.
- 수집 시 CPU direct residual 경로에서 압축 전 sparse INT32 residual과 native HP1 weight를 dump했다. 이 값으로 A/B/C를 재생한다. dump 수집 경과 시간은 성능 지표에서 제외한다.
- 마지막 layer의 FFN은 prefill에서도 필요한 한 행만 처리한다. M=1 여부로 구간을 결정하지 않고, chronological case 순서의 layer 0 재시작으로 네 inference pass를 구분한다. `passes.tsv`에 근거를 보관한다.
- 각 stripe에서 warmup 2회 후 7회 실행한다. A*/B/C와 현재 A의 실행 순서를 섞고, 생성 배열 전체의 계산이 유지되도록 memory barrier를 둔다. 보고한 CPU 합계는 stripe별 중앙값의 합이다.
- CPU 실측을 모두 마친 뒤 scalar geometry만 cycle model에 전달한다. estimator의 wall time은 성능 수치에 포함하지 않는다.

CPU 구간은 limb 분해, 행·K 선택 및 activation pack, weight decode/gather/transpose, INT32 output의 limb 복원이다. 현재 A는 실제 `RmdBitmapBuilder`와 `build_run_aware_request`를 호출한다. 이 fixture에는 ExSIA가 원래 제공하는 selection bitmap을 재생성하는 비용이 추가된다. 파일 I/O, vector 해제, 공통 최종 float scale/merge, backend 호출과 동기화는 제외한다. 결과 복원 시간은 같은 크기의 준비된 INT32 output buffer로 측정하며, GEMM emulation은 별도 수치 검증에만 사용한다. A*·B·C는 단순 배열 builder이며 현재 A의 전체 packet/carrier 준비를 대체한 production 구현은 아니다.

## NPU 계산 방법

`im2p_cycle_estimate_runs`에 M/N/K와 원래 K32 block ID·K mask·compact offset/count를 전달했다. 각 block의 HP1 scale 경계를 유지하며, 서로 다른 K32 block을 하나의 scale group으로 합치지 않는다. 타일 선택은 현재 `gemmini_set_tile_ws`의 SP/ACC 조건과 증가 순서를 따른다.

| 항목 | 설정 |
|---|---|
| DIM | 16, 64 |
| SP / ACC 물리적 용량 | 256 KiB / 64 KiB |
| SP bank 수 | 4 |
| SP row bytes | DIM × operand bits / 8 |
| ACC row bytes | DIM × 4 |
| SP read delay / ACC latency | 4 / 2 cycles |
| Request | isolated, BLOCK_SUBMISSIONS, initial SP/ACC half 0 |
| Backing timing | cycle library의 기본 revision 1 |

Software admission 한도는 1B cycles·10M fragments로 잡았다. 기본 10M cycles를 넘는 큰 GEMM을 허용하는 설정이며 하드웨어 용량 변경이 아니다. SHA와 상세 timing 값은 provenance에 보관한다.

Logical MAC 수는 `M × N × compact_K`다. 실제 padding 비교에는 `ceil(M/DIM) × DIM`, `ceil(N/DIM) × DIM`과 **각 K32 run에서 따로 올림한 K**를 쓴다. 특히 DIM64에서는 살아 있는 K32 run마다 최소 64개의 K 자리가 필요하다. 따라서 논리적인 K 감소율과 padded MAC·cycle 감소율이 다를 수 있다.

DIM64는 DIM16에서 수집한 동일 residual·stripe의 device geometry를 바꾼 비교다. DIM64 inference를 다시 수집한 결과는 아니다. 현재 A host 함수는 DIM16으로 컴파일했으므로 DIM64의 현재 A CPU 값은 참고값이다. DIM64에서도 A*·B·C의 weight gather loop와 device geometry에는 DIM64를 적용한다.

## 호스트 weight buffer를 재사용하는 조건

매 stripe에서 원래 quantized weight를 다시 decode/transpose하는 조건 외에, `[K][N]` INT32 호스트 buffer를 미리 만들어 쓰는 DIM16 조건도 비교했다. A*는 선택한 K 행을 새 compact buffer로 복사하는 시간을 2회 warmup·7회 반복해 추가 실측했다. B/C는 원래 K를 유지하므로 전체 buffer를 직접 참조한다고 가정한다. B/C의 재사용 CPU 비용은 기존 실측의 decomposition·pack·restore 구간 합이다. A*에는 그 세 구간에 실제 compact copy 시간을 더한다.

NPU SRAM 상주나 DMA 생략을 가정하지 않는다. GEMM의 weight load cycles는 그대로 남고, 호스트 decode 초기 비용만 반복 비용에서 제외한다. 이 cache를 production A에 연결하지 않았다. 필요한 buffer의 총량과 한 번의 decode 준비 비용은 `cached-weight-setup.csv`에 남긴다. 모든 layer를 계속 재사용하려면 해당 총량을 유지해야 하며, 한 layer씩 새로 decode하면 이 반복 비용을 달성할 수 없다.

## 수치 검증

실제 dump 1,590개를 두 DIM geometry에 적용한 3,180개 비교, A/B/C **9,540개 결과**에서 sampled 수치 불일치·포화 검출은 모두 0이었다. 모든 output column의 최대 보수적 bound는 **329,082,368 < INT32_MAX**다. 독립 int64 식의 범위도 별도 확인했으며 최대 보수적 bound는 1,074,954,456,576으로 int64 한도 아래다. A*와 현재 A의 배열 대조, cached copy의 weight 대조는 전 원소를 검사했다.

검증 기준은 원본 좌표의 sparse `residual × weight_code × 2^HP1_exponent`다. A/B/C끼리의 일치만을 정답으로 사용하지 않는다. 모든 source row와 stripe당 최대 32개의 균등 선택 output column에서 각 경로의 HP1 fragment 실행·limb 복원 결과를 독립 기준과 비교했다. 각 K32 block의 최대 exponent와 signed code 범위, digit의 L1 합으로 모든 output column의 포화 부재도 별도 계산했다.

K 압축은 fragment 묶음을 바꾸므로 포화가 발생하면 A와 B/C가 달라질 수 있다. 별도 self-test는 M=N=1, K=32, DIM16에서 K 위치 0·1에 +1, 16·17에 -1 residual, weight code 7, exponent 28을 둔다. A는 네 K를 한 fragment로 압축해 0을 얻는다. B/C는 양·음 기여를 서로 다른 16칸 fragment에서 각각 Sat32한 뒤 더해 -1을 얻는다. INT32_MIN/MAX의 balanced carry 재구성도 확인했다. 이 경계 사례와 실제 dump의 검증 결과를 구분한다.

## 재현 및 근거

보관된 dump로 다시 실행하는 첫 단계는 다음과 같다. 이어지는 cache·cycle·집계 명령은 실행 방법에 있다.

```bash
rtk proxy bash scripts/eval/compaction_abc/prepare.sh
rtk proxy bash scripts/eval/compaction_abc/run-replay.sh output/experiment/compaction-abc-20261004
```

- [실행 방법과 코드](../../scripts/eval/compaction_abc/README.md)
- [stripe별 원자료와 dump](../../output/experiment/compaction-abc-20261004/)
- [전체 측정표](../../output/experiment/compaction-abc-20261004/measured-tables.md)
- [집계 CSV](../../output/experiment/compaction-abc-20261004/summary.csv)
- [weight 재사용 집계](../../output/experiment/compaction-abc-20261004/cached-weight-summary.csv)
- [검증 근거](../../output/experiment/compaction-abc-20261004/validation.json)
- [provenance](../../output/experiment/compaction-abc-20261004/provenance.json)

이 실험은 한 prompt의 residual 준비와 GEMM 비용에 대한 비교다. 실제 보드 클록, FPGA elapsed time, CPU/NPU overlap, 전체 모델 latency와 품질/PPL은 측정하지 않았다. `CPU_ms + cycles / (GHz × 10^6)`은 지정한 클록에서 직렬 실행한다는 조건부 합계다.
