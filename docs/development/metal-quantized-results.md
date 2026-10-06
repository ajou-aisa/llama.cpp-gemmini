# Metal block / HP1 ExSIA 검증 결과

2026-10-05, Apple M5 / macOS 26.4 / Apple Clang 21에서 검증했다. 기준 commit은 `bc3d7ba55e430bc984019c9ed4f50451a02fa287`이며 이 문서의 구현은 그 위의 working-tree 변경이다. 개별 실행 기록에 실제 source·binary·model hash와 CMake 설정이 있다.

## 구현 범위

| 경로 | 원본 weight | activation / residual | 계산 |
| --- | --- | --- | --- |
| BLOCK A4W4 / A8W8 | Q4_0 / Q8_0 | CPU K32 block producer | Metal integer dot와 block별 scale 복원 |
| HP1 × ExSIA A4W4 / A8W8 | Q4_HP1 / Q8_HP1 | CPU ExSIA folding과 원본 run-aware RMD 구성 | Metal ordered SCU, lane 누산, checked radix, scale·merge |

원본 GGUF의 quantized weight 포맷을 유지한다. CPU producer가 준비한 정수 code와 metadata를 shared Metal buffer로 전달한다. 앞선 GPU activation 생성과 CPU 접근 사이에는 completion fence가 있다. Apple GPU에서 native double 대신 정수 연산으로 IEEE binary64를 구현해 기준 double 연산 순서와 F32 cast를 보존한다.

HP1 fragment 길이는 `min(DIM, 32)`이며 포화 후 상쇄와 residual run 순서를 보존한다. BLOCK correction은 K32 scale로 복원하며 dense kernel 안에서 실행된다. 따라서 BLOCK의 별도 residual/merge kernel count가 0인 것은 정상이다.

A16W16과 GPU activation producer는 완료 범위에 포함하지 않는다. Gemmini backend·simulator archive 없이 빌드하지만 기존 CPU producer의 geometry를 위해 Gemmini parameter/tiling header를 사용한다. 검증된 실행 방법은 [실행 문서](metal-quantized-execution.md)에 있다.

## 수치 검증

| 검사 | 결과 | 근거 |
| --- | --- | --- |
| 최종 profile build | 9개 성공, kernel/cache 실행 18/18 PASS | [build matrix](../../.omo/metal-quantized/profile-build-matrix.json) |
| 실제 GPT-2 / Llama 모델 | 16/16 PASS | [model summary](../../.omo/metal-quantized/model-oracle-final/summary.json) |
| 독립 IM2P CPU_FUNCTIONAL provider | A4/A8 × DIM16/32/64, 6/6 PASS | [provider 검증](../../.omo/metal-quantized/provider-reference/vector4-narrowed-validation.json) |
| soft-FP64 | 262,144 fixtures × 6 API, bitwise mismatch 0 | [검증 로그](../../.omo/metal-quantized/probes/test-metal-soft-f64.log) |
| Metal API / GPU validation | no-trace 24조건 포함 PASS | [validation 기록](../../.omo/metal-quantized/performance/final/dummy-buffer-validation.json) |
| 기존 CPU / 일반 Metal | 새 feature OFF 빌드, 각각 F32·Q4_0 연산 2/2 PASS | [회귀 기록](../../.omo/metal-quantized/regression-validation.json) |

최종 실제 모델 matrix는 다음과 같다. 각 칸은 GPT-2와 Llama3.2-1B를 각각 실행한 결과다.

| Profile | DIM16 | DIM32 | DIM64 |
| --- | --- | --- | --- |
| BLOCK A4W4, RMD OFF | 2/2 | — | — |
| BLOCK A8W8, RMD OFF | 2/2 | — | — |
| HP1 ExSIA A4W4, RMD ON | 2/2 | 2/2 | 2/2 |
| HP1 ExSIA A8W8, RMD ON | 2/2 | 2/2 | 2/2 |

모델 검증은 prefill 8 tokens와 decode 2개를 같은 token trajectory로 두 번 실행했다. 실제 Metal의 quantized matmul **7,680개**, F32 출력 **73,564,160개**가 독립 CPU 정수·double oracle와 bitwise 일치했다. 해당 oracle 값을 사용하는 graph replay의 logits **14,281,040개**도 원래 실행과 bitwise 일치했다. Greedy divergence는 0이고 각 실행 전후 library hash도 동일했다.

두 번째 실행은 custom matmul을 독립 CPU reference 값으로 재생하며 나머지 graph 연산의 backend 설정은 동일하다. 따라서 이 결과의 범위는 custom matmul 및 그 결과를 사용하는 동일 graph의 logits다. CPU만으로 실행한 전체 모델과의 비교는 아니다. Producer payload 자체는 [기존 producer 직접 비교](../../.omo/metal-quantized/producer-checks/validation.json)로 별도 검증했다.

Provider 참조는 IM2P.sim commit `bc6168c5ab3cc47edc716ff2c23402bd09867b8d`의 CPU_FUNCTIONAL 실행이다. 실제 RTL/FPGA 결과를 의미하지 않는다. Provider 및 성능 측정 이후 바뀐 빈-buffer 최소 크기 수정은 arithmetic을 변경하지 않는다. 그 수정까지 반영한 최종 library로 profile 및 실제 모델 검증을 다시 수행했다.

Metal validation에서 사용하지 않는 `device long*` binding도 최소 8바이트 buffer가 필요하다는 오류를 발견했다. 빈 placeholder 크기를 수정하고 trace를 끈 24조건을 회귀 검사에 추가했다. 실패한 이전 기록은 보존했고, 최종 PASS 근거는 수정 후 실행이다.

## 성능 측정의 범위

Synthetic benchmark는 준비된 integer payload로 Metal과 독립 CPU oracle를 비교한다. Warmup 1회, 반복 5회의 median이며 CPU activation producer와 최초 shader compile은 제외한다. CPU oracle는 1 thread 기준이고 최적화된 CPU backend와의 비교가 아니다.

- 기본 54조건은 decode M=1, prefill M=256, K=768, N=768/3072, A4/A8, BLOCK 및 HP1 DIM16/64, residual density 0/6.25/100%를 포함한다. 모든 출력이 bitwise 일치했다.
- Decode attention 18조건에서는 buffer 준비와 command submission을 포함한 Metal host 시간이 serial CPU oracle보다 모두 느렸다.
- Prefill 조건의 Metal host 시간은 serial CPU oracle보다 3.61–22.29배 빨랐다. 이 배율에 CPU producer 비용은 포함하지 않는다.
- HP1 dense와 DIM16 residual에 packed-char4 연산을 적용했다. 최종 prefill 12조건의 scalar Metal 대비 관측 배율은 GPU 시간 기하평균 **1.257배**, host 시간 **1.168배**다. 11조건에서 host 시간이 줄었고 한 조건은 약 1.4% 늘었다.

같은 측정에서 코드가 그대로인 BLOCK 조건도 상당한 시간 변동을 보였다. 위 배율은 해당 paired 측정의 관측치이며, 최적화의 안정적인 속도 향상 또는 전체 모델의 CPU 대비 향상을 확정하는 수치는 아니다. [최종 비교 JSON](../../.omo/metal-quantized/performance/final/comparison.json)에 조건별 stage median과 범위가 있다.

원본 weight byte를 확인해 재사용하는 cache도 추가했다. Llama Q4_0의 실제 `blk.0.ffn_up`, M=1에서 weight 준비 median은 cache OFF **63.628 ms**, 재사용 **0.354 ms**였다. 첫 cache miss는 **65.955 ms**였다. 이 수치는 weight 준비 비용이며 전체 inference 속도가 아니다. [측정 기록](../../.omo/metal-quantized/weight-cache/benchmark-evidence.json)

## 실행 및 PPL 검증

최종 실행 matrix는 같은 16조건에서 모두 통과했다. WikiText 입력 256 tokens를 prefill하고, 첫 출력 token 뒤 decode를 한 번 실행해 총 2 tokens를 생성했다. GPT-2는 실행당 96개, Llama는 224개의 quantized matmul을 관찰했고, 총 **2,560개 모두 custom Metal kernel 완료 count와 일치**했다. Fallback과 실패 count는 0이었다. [실행 summary](../../.omo/metal-quantized/e2e-final/summary.json)

아래는 해당 단일 실행에서 관측한 시간이다. 단위는 초이며 TTFT는 첫 token까지, decode는 이후 한 번의 decode다. CPU producer, shared-memory staging, GPU 계산과 placement observer 비용을 포함한다. CPU 대비 speedup 측정이나 반복 측정의 median은 아니다.

| Profile | GPT-2 TTFT | GPT-2 decode | Llama TTFT | Llama decode |
| --- | ---: | ---: | ---: | ---: |
| BLOCK A4, DIM16 | 1.435 | 0.097 | 17.752 | 0.954 |
| BLOCK A8, DIM16 | 1.655 | 0.056 | 17.029 | 0.580 |
| HP1 ExSIA A4, DIM16 | 2.347 | 0.186 | 34.186 | 2.256 |
| HP1 ExSIA A4, DIM32 | 2.245 | 0.171 | 31.149 | 1.803 |
| HP1 ExSIA A4, DIM64 | 2.331 | 0.182 | 31.533 | 1.822 |
| HP1 ExSIA A8, DIM16 | 2.311 | 0.139 | 33.662 | 2.280 |
| HP1 ExSIA A8, DIM32 | 2.534 | 0.137 | 32.738 | 2.320 |
| HP1 ExSIA A8, DIM64 | 2.176 | 0.135 | 32.873 | 2.240 |

CPU producer·buffer 준비·dense·residual·merge·matmul 전체 시간은 [조건별 CSV](../../.omo/metal-quantized/e2e-final/summary.csv)에 있다. `transfer`는 unified shared-memory buffer 준비·복사 비용이며 PCIe H2D/D2H 시간이 아니다. `matmul_total`은 producer부터 output scatter까지이고 stage 합 외에도 allocation·submission·동기화 등의 비용을 포함한다. BLOCK과 HP1은 산술 계약이 다르므로 서로의 시간 비율을 같은 계산의 가속 배율로 해석하지 않는다.

이 2-token 검사는 prefill/decode dispatch 진단이다. Collector의 128-token 정식 E2E 수집이나 전체 corpus PPL 결과는 아니다.

실제 PPL runner도 GPT-2의 BLOCK A4/DIM16과 HP1 ExSIA A4/DIM16에서 실행했다. Context 32 / 1 chunk / warmup OFF로 각각 15 tokens를 평가했고, 두 실행 모두 관측 matmul 48개와 custom Metal 완료 count 48개가 일치했다. Source·입력·binary의 실행 전후 provenance 검사도 통과했다.

| PPL 실행 확인 | PPL 관측값 | Dense / residual / merge launches | 기록 |
| --- | ---: | --- | --- |
| BLOCK A4, RMD OFF | 137.3358 | 48 / 0 / 0 | [실행 기록](../../output/experiment/metal-ppl-gpt2-block-a4-d16-final-smoke/manifest.txt) |
| HP1 ExSIA A4, RMD ON | 136.0703 | 48 / 48 / 48 | [실행 기록](../../output/experiment/metal-ppl-gpt2-exsia-a4-d16-final-smoke/manifest.txt) |

이 짧은 입력의 PPL 값은 실행 확인용이다. 두 방식의 정확도 우열이나 전체 WikiText PPL을 판단하는 값으로 사용하지 않는다.

실패 경로는 실제 GPT-2 실행으로 별도 확인했다. `-ngl 1`에서는 48개 중 4개만 Metal이어서 exit 1로 거부했다. 입력이 짧아 warmup만 48회 실행되고 실제 평가 token이 0인 경우에도 exit 1이었다. 명시적인 `-ngl 0` CPU 실행은 Metal 인증 없이 기존 동작을 유지한다. 실제 scheduler의 Metal/CPU 혼합 graph와 기존 callback 연결도 검사했다. [실패 경로 기록](../../.omo/metal-quantized/ppl-guard/summary.json)

검증을 강제하는 실행 표면은 제공된 evaluator와 PPL runner다. 일반 `llama-cli`와 임의의 ggml scheduler client에는 같은 fallback 검사 보장이 없다. Layer offload 수만으로 custom 정수 연산 실행을 판단하지 않는다.

## 재현 기록

`.omo/metal-quantized/` 링크는 이 workspace에 저장한 실행 artifact다. Source commit만으로 uncommitted 구현과 로컬 GGUF를 재현할 수 없으므로 기록된 source hash, profile CMake cache, binary hash, 입력 GGUF hash를 함께 사용한다.

- [macOS build 기본값 및 override 검사](../../.omo/metal-quantized/build-validation.json)
- [동일 configure에서 generated file / library 불변 확인](../../.omo/metal-quantized/generation-stability.json)
- [실행 방법 및 입력 조건](metal-quantized-execution.md)
- [코드 검토](../../.omo/metal-quantized/code-review.md)
