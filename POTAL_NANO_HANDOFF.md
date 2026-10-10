# PoTal attention: nano에서 Gemmini / IM2P 작업 이어가기

## 이어받는 작업

사용자는 `hotfix/potal-attn`을 원격에 보존하고, **그 브랜치에서 새 브랜치를 만들어 nano에서 지금까지의 논의를 이어가도록** 요청했다. 새 세션은 이 문서를 먼저 읽는다.

- 원격: `https://github.com/ajou-aisa/llama.cpp-gemmini.git`.
- 실험 보존: `hotfix/potal-attn`, `ba762b951ec267aa3abed6bab9e3466431416307`.
- 이어서 작업할 브랜치: **`work/potal-attn-im2p-nano`**. 위 커밋에서 분기했다.
- 제품 코드 기준: `develop`, `3b2eaa3cb6f7adc36094b7f78662141825cc4ef4`.
- `cbb5dfc`: `test(potal): add A4 policy and storage experiments`.
- `ba762b9`: `perf(potal): measure attention limbs and compensation costs`.

완료된 것은 CPU 정책 실험, C++ packet 공간 측정, attention 구성요소 프로파일링이다. **제품의 a4nks attention dispatch, fused upper-limb emitter, exact wide accumulator는 아직 구현하지 않았다.** 이관 중에는 nano에 접속하거나 IM2P를 실행하지 않았다. 기존 Metal/CUDA 브랜치와 develop은 보존했다.

읽는 순서:

1. 이 문서의 결정과 제한 사항.
2. [처음 받은 알고리즘](experiments/potal_a4/ORIGINAL_SPEC.md).
3. [상세 구현 계획](experiments/potal_a4/IMPLEMENTATION_PLAN.md). 10개 제품 구현 항목은 미완료다. 새 테스트 이름과 A4NKS 옵션은 계획이며 현재 사용할 수 있다는 뜻이 아니다.
4. [공간·정확도 결과](experiments/potal_a4/RESULTS.md), [attention limb·보정 비용](experiments/potal_a4/ATTENTION_PROFILE.md), [실험 실행 방법](experiments/potal_a4/README.md).

## 반드시 유지할 결정

최신 조건은 **처음 제공된 selective-linear + a4nks 대비 정확도가 떨어지면 안 된다**는 것이다. FP32 대비 양자화 손실까지 없애라는 조건과는 구분한다. 정책 이름은 `potal-a4-preserve-v1`이다.

- 선형: `corr = outlier || D > 0`; `v = corr ? u : sat(u)`를 직접 balanced radix-16 digit으로 분해한다. 보정 대상이 아닌 `u=8`은 원래대로 7이다.
- Q: `rho=6`, `v=u`, 필요한 digit을 모두 유지한다. PE는 signed 4비트다.
- P: 원래 geometry stripe의 uint8, main `(q&15)-8`, upper digits `q-(q&15)`, zero point 8. Q와 동일한 양자화 방식이 아니다.
- K: `[-128,119]`의 8비트 코드, signed nibble 두 패스. V: token block / channel별 4비트.
- top exponent / strict 2-sigma / distinct E2, stripe scale, 원래 K block, mask와 FP softmax를 유지한다. FP KV 저장 형식도 유지한다.
- **줄일 대상은 dense int32 residual plane과 scalar 재분해·재수집 비용**이다. 필요한 limb를 버리지 않는다. 같은 코드와 패킹 조건이면 이 표현 최적화로 GEMM fragment가 줄어드는 것은 아니다.
- 같은 weights/tokens/실행 경로에서 main, digits, support, scales, 복원값, logits, token별 NLL/PPL이 같아야 한다. 비슷한 평균 PPL만으로 통과시키지 않는다.

초기에는 선형·Q 모두 folded code 전체를 직접 분해하기로 했지만, 실험 후 정확도 유지 조건이 우선했다. **선형 direct-all, limb 제한, P 4/6비트, P 8행 stripe는 미채택 실험**이다. `types.py`의 실험용 기본 `DIRECT`나 `direct_p8` 이름을 제품 기본 정책으로 옮기지 않는다. 기준 모델 모드는 `original_a4nks`다.

## 실측 핵심

GPT-2 Q4_HP1, WikiText-2 test 앞 4,096 tokens, context 512, 8 chunks, 2,040 scored tokens. Q4로 저장된 48개 선형 행렬만 입력을 양자화했고 F16 embedding/tied output head는 보존했다.

| 조건 | PPL |
|---|---:|
| 같은 Q4 weights, activation FP32 | 30.83377 |
| 원문 selective-linear + a4nks | **35.37806** |
| direct-all 선형 + 원래 P stripe | 36.55379 |
| direct-all + P 8행·8비트 | 35.36053 |
| 위 조건에서 P 6비트 | 38.00305 |
| 위 조건에서 선형·Q upper limb 최대 1개 | 89.74277 |

P 8행의 작은 PPL 차이는 개선의 증거가 아니다. 이 조건은 원문 대비 전체 PV fragment가 약 3배였다. 최종 결과는 `results-v3/`다. 이전 `results/`, `results-v2/`는 F16 head 입력까지 양자화했던 탐색 결과이므로 사용하지 않는다.

원문 정책의 context 512 첫 test chunk, 12 layers × 12 heads:

| 항목 | 관측 |
|---|---|
| Q | main + upper 1/2. 원소 90.90%에 upper digit, 15.77%에 upper 2번 자리 |
| P | main + upper 1/2. upper 1/2 nonzero 비율 0.964% / 0.063% |
| K / V | K dense 두 패스 / V 한 패스 |
| Q 이상치 선택 | 12.78%. Upper digit 90.90%는 이상치만의 비용이 아님 |
| DIM32 fragment | QK main 대비 6.000배, PV main 대비 1.607배 |
| 전체 QK+PV fragment | upper 보정 60.56%; K-high main까지 포함하면 73.71% |
| P upper packet | 실제 nonzero는 할당 digit bytes의 1.69%. 행·열 합집합과 패딩이 큼 |
| CPU component replay | 이상치 선택 0.710 ms/head (0.65%), Q upper 52.894 ms (48.11%), P upper 3.009 ms (2.74%) |
| 준비 과정 | Q/P packet 생성 7.289 ms, 준비 시간의 69.82% |
| P zero point | colsum+broadcast 0.116 ms, 측정 구간의 0.11% |

위 시간은 NumPy int64의 구성요소 재생이며 전체 모델 지연시간이나 Gemmini 시간이 아니다. 3 contexts × 144 heads = 432개 입력을 검사했고, 시간 측정한 27 heads의 QK/PV 54개 integer GEMM은 복원 operand의 F64 GEMM과 bitwise 일치했다. 새 end-to-end integer 모델 PPL 결과는 아니다.

C++ representation probe는 512×4096 folded codes에서 clipped main+완전 보정+dense residual과 lowdigit main+plane 생략을 비교했다. Sparse fixture 유지 공간은 12.09→4.08 MiB였지만, 9-digit carry fixture는 direct도 19.72 MiB로 FP32 배열 8 MiB보다 컸다. Main/digits는 signed byte이며 packed 4비트가 아니다. 이 probe는 원문 selective 정책 전체의 동등성 검증이나 모델 전체 RAM 절감 측정이 아니다. 기존 int32 scalar emitter bridge가 남아 있다.

이관 직전 재확인: integer roundtrip 141,076건, CTest 4/4, 소스·결과 SHA-256 57개 일치. Native Q4_HP1 decoder 193,536개 weight 대조는 이전 기록에서 bitwise 일치했다. 기존 CPU 라이브러리는 Q4_HP1 모델을 compatible buffer 부재로 로드하지 못했으며 실패 로그도 보존했다.

## nano에서 시작

저장소의 기존 작업을 확인한 뒤 첫 checkout에서 실행한다. 이미 같은 로컬 브랜치가 있으면 새로 만들지 말고 해당 브랜치로 이동한다.

```sh
rtk git status --short --branch
rtk git fetch origin
rtk git switch --track origin/work/potal-attn-im2p-nano
rtk read POTAL_NANO_HANDOFF.md
```

쉘 명령은 사용자 지침대로 `rtk`를 붙인다. 다른 변경을 reset/삭제하지 않는다. Nano의 OS/architecture, Python, 메모리, IM2P와 Gemmini header 저장소 경로/commit, selected artifact 존재 여부를 먼저 확인한다. `nano`라는 이름만으로 아키텍처나 CUDA 실행을 가정하지 않는다.

모델·데이터·가상환경·build·`.i32` token 파일·`.npz` 캡처는 Git에 포함하지 않았다. 재현에 필요한 로컬 배치는 다음과 같다.

| 저장소 아래 경로 | SHA-256 / 용도 |
|---|---|
| `models/gpt2.Q4_HP1.gguf` | `a5da4eacdb13197248bac7aea0a9d6e934004d393994833acd70b9d5ef610f1f` |
| `models/gpt2-base/tokenizer.json` | `8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6` |
| `wikitext-2-raw/wiki.test.raw` | `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08` |
| `wikitext-2-raw/wiki.valid.raw` | `4cd0f6876d07a413aa911261ff6d363c72d757d47f0fdd6015702014c89cb9c7` |
| `models/gpt2.fp16.gguf` | 별도 native FP16 sanity check용 |
| `models/llama3.2-1B.Q4_HP1.gguf` | 이후 GQA/prefill/decode 검증용, 이번 실측 미사용 |

Python 3.11, NumPy 1.26.4, tokenizers 및 저장소 `gguf-py` 의존성을 확인한다. 아래 `python3.11`은 nano의 확인된 환경 경로로 바꿀 수 있다. 모델과 데이터가 위 경로에 있어야 하며, 출력은 새 디렉터리를 사용한다.

```sh
rtk proxy python3.11 -m experiments.potal_a4.verify
rtk proxy python3.11 -m experiments.potal_a4.prepare . experiments/potal_a4/results-nano
rtk proxy env PYTHONPATH=gguf-py:. OPENBLAS_NUM_THREADS=2 python3.11 -m experiments.potal_a4.run models/gpt2.Q4_HP1.gguf experiments/potal_a4/results-nano/test.i32 512 8 original_a4nks experiments/potal_a4/results-nano/test512-original_a4nks.json
rtk proxy env PYTHONPATH=gguf-py:. OPENBLAS_NUM_THREADS=2 python3.11 -m experiments.potal_a4.profile_attention experiments/potal_a4/results-nano/test512-original_a4nks.json experiments/potal_a4/results-nano-profile/context512.json
```

과거 JSON의 Mac 절대 경로는 측정 당시 출처다. 이를 덮어쓰지 말고 nano baseline을 새로 생성한다. Profiling은 baseline JSON 옆 `test.i32`, `.chunk0.npz`와 JSON의 model 경로를 읽는다. 역사적 `.npz`는 Git에 없으며 새 baseline으로 재생성할 수 있다. OS/BLAS가 다른 실행을 기존 Mac hidden과 bitwise 동일하다고 가정하지 않는다. 같은 nano 경로 안에서 baseline/변경 후 동등성을 검증한다.

**메모리 측정 전 처리할 이식 항목:** `run.py`와 `native_probe.cpp`는 `ru_maxrss`를 macOS의 bytes로 기록한다. Nano에서 OS 단위를 확인하고 bytes로 정규화한 뒤 새 측정한다(Linux는 KiB). 과거 JSON의 의미를 바꾸지 말고 이 수정과 새 결과를 별도로 기록한다. `report.py`는 bytes를 전제로 한다. Native probe 빌드는 새 DIM32 generated header와 nano에서 빌드한 llama 라이브러리 경로도 필요하다.

## Gemmini / IM2P에서 할 일

1. [계획](experiments/potal_a4/IMPLEMENTATION_PLAN.md)의 수치·표현 계약부터 확인한다. Python 실험은 dense 임시 residual을 사용하고 native probe도 scalar bridge가 있어 제품 fused emitter가 아니다.
2. 기존 [IM2P resolver](cmake/ggml-gemmini-im2p.cmake)와 [provisioning helper](scripts/im2p-host-provision.sh)로 nano용 `GEMMINI_HP1`, A4/W4/DIM32/BK32 matched selected artifact를 확인한다. `GGML_GEMMINI_EXECUTION_BACKEND=IM2P_SIM`, `GGML_GEMMINI_OPTION=WS`, `CYCLE_SIM=0`이 계획의 raw op0 경로다. Mac archive나 CPU-functional op5 trace는 이 실행 증거가 아니다.
3. `Im2pCompactDot` raw BYPASS + K<=32와 host exact merge를 연결한다. 기존 HP1 SCU-final은 int32 saturation이 있으므로 잘린 값을 exact 결과로 재사용하지 않는다. Int64 bound를 넘는 경우의 정확한 합산도 계획에 포함되어 있다.
4. 원문 값 보존 direct emitter, 선형 통합, K/P/V 정책, QK→FP softmax→PV 그래프 연결을 진행한다. QK/PV 두 노드의 실제 Gemmini 실행을 확인하고 CPU fallback을 별도로 기록한다.
5. limb/row/K 분포, main/upper/K-high 제출 수, selection·packing·compensation·host merge 시간, packet/live peak memory를 nano에서 따로 측정한다. Fragment 수를 속도 배수로 해석하지 않는다.

Geometry 주의: 실험의 논리 stripe는 `(banks,bank_rows,acc_rows)=(4,4096,512)`다. Mac probe의 generated header는 ACC_ROWS=1024였고 probe는 stripe=32로 고정했다. 이 결과로 hardware schedule parity를 주장하지 않았다. Nano의 실제 계약을 기록하고, 논리 quantization stripe와 물리 tile 차이가 정확도를 바꾸지 않도록 먼저 대조한다.

`GGML_GEMMINI_ENABLE_A4NKS_ATTENTION`, `test-gemmini-potal-a4`, `test-gemmini-attention`은 새로 구현할 항목이다. 현재 CMake가 모르는 옵션을 전달하거나 기존 일반 inference가 동작했다는 이유만으로 a4nks 성공이라고 기록하지 않는다. KV/strided view/GQA와 single-token decode도 아직 검증하지 않았다.

## 새 대화에 붙여넣을 요청

> `work/potal-attn-im2p-nano`에서 이어서 작업하자. 먼저 `POTAL_NANO_HANDOFF.md`와 연결된 ORIGINAL_SPEC.md, IMPLEMENTATION_PLAN.md, RESULTS.md, ATTENTION_PROFILE.md를 읽어. Gemmini / IM2P를 nano에서 실제로 실행하고 limb·이상치 선택·packet 생성·compensation의 시간과 공간을 측정하려는 작업이다. 최신 조건은 원문 selective-linear+a4nks 대비 정확도 저하 금지다. Direct-all 선형, limb 제한, P 8행 실험은 채택하지 않았다. 현재 제품 attention 경로가 미완성임을 전제로 로컬 환경과 matched A4/W4/DIM32 artifact부터 확인하고, 계획의 원문 값 보존 direct emitter와 실제 QK/PV 통합을 이어가. 기존 실험 숫자는 Gemmini 지연시간으로 해석하지 말고, 같은 nano 경로에서 정확도 동등성과 실행 경로를 검증해.
