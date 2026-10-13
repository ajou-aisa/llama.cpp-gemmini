# PoTal attention: nano에서 Gemmini / IM2P 작업 이어가기

## 최신 작업: 모델의 vanilla QK/PV backend 연결

`GGML_GEMMINI_ENABLE_A4NKS_ATTENTION=ON`으로 실제 모델의 QK와 PV를 Gemmini
backend에 함께 배정하는 코드를 추가했다. 기존 ExSIA/WideQuantizer와 stacked
RMD, raw IM2P provider를 재사용한다. 모든 signed nibble은 int8로 운반하고,
BK·출력 column·lane별 scale은 raw dot 뒤 checked int64로 누적한다.
PV의 weighted colsum은 KV head마다 한 번 계산해 GQA query head에서 재사용한다.

로컬은 `OPTION=CPU` 기능 검사다. 양자화·패킹·정수곱·보정이 native backend에서
실행되지만 실제 Gemmini RTL/NPU 실행은 아니며 `raw_cycles=0`이다. 실제 IM2P
링크·실행·cycle 측정은 Nano에 남아 있다. 새 backend는 vanilla QK/PV이며,
QK tile skip은 아래의 기존 standalone 실험에만 연결되어 있다.
최신 정확도 결과는 [진행 기록](experiments/potal_a4/NANO_PROGRESS.md), Nano의
빌드·node 측정·10-chunk 모델 비교 명령은
[실험 README](experiments/potal_a4/README.md#실제-backend-qkpv-측정)에 있다.

## 이전 업로드: vanilla QK/PV의 standalone Gemmini 재생

기존 raw Gemmini provider와 CPU 묶음 누산기를 재사용해 캡처 재생 코드를 준비했다.
기본값은 masking 없는 QK/PV이며, `--experiments`로 직접 누적·QK 타일 생략을
함께 비교한다. 모든 PE 입력은 기존 signed nibble/int8, 블록·lane 보정은 checked
host int64다. 양자화·패킹·정답 생성은 실행 전에 준비한다.

로컬 CPU 빌드/CTest 7개와 실제 GPT-2/Llama QK/PV 캡처 4개의 정확도 검사는 통과했다.
새 Gemmini 공유 라이브러리의 실제 링크·실행과 RTL 측정은 원격에서 해야 한다.
이전 업로드에는 제품 attention dispatch가 없었다. 위의 후속 구현과 구분한다.
실행 명령과 결과 필드는 [실험 README](experiments/potal_a4/README.md#gemmini-qkpv-원격-검사-vanilla-기본값)에 있다.

## 최신 로컬 수정: ExSIA 공통 source-zero 정책

기존 `LocalStage`의 full/tail/reference에서 A4 한정 조건을 제거해 A4·A8·A16에
같은 정책을 적용했다. 기존 `blk.e`를 재사용하고 원본 0의 출력은 0으로 기록한다.
비영 원본이 반올림 후 0이 되는 경우는 통계에 포함한다. 새 mask·배열·순회는 없다.
Release 회귀 검사는 A4 OpenMP ON/OFF 각각 9/9, A8 14/14, A16 9/9 통과다.
전후 10-chunk PPL 측정도 완료했다. 결과와 증빙은 [최신 진행 기록](experiments/potal_a4/NANO_PROGRESS.md#local-common-exsia-source-zero-policy-2026-10-13),
전체 코드 변경은 [source-zero.patch](experiments/potal_a4/results-source-zero-common/source-zero.patch)에 있다.

## 로컬 실험 결과: native 묶음 실행·직접 누적

기존 실험 경로에 stripe 단위 native 제출과 유효 구간 직접 정수 누적을 연결했다.
int8 signed nibble 입력, BK32별 scale, 모든 lane과 K 두 패스를 보존한다.
Llama context512 ×10의 PPL은 전후 **13.801307254545433**으로 같고,
packet·scale·유효 QK·softmax·PV·logits·토큰 NLL 및 추가 mask/cache 검사도 통과했다.
QK 15개 CPU 사례는 기존 생략 **397.462→326.162 ms**, 제출은 **2,880→120회**다.
PV는 **160.280→175.864 ms**로 느려져 동일한 묶음 적용을 기본값으로 채택하지 않는다.
실제 Gemmini/IM2P dispatch·cycle 측정과 임의 범위 exact accumulator는 남아 있다.
측정 조건·남은 작업은 [최신 진행 기록](experiments/potal_a4/NANO_PROGRESS.md),
원자료는 [측정 요약](experiments/potal_a4/results-qk-batch-10chunk/component-summary.json)에 있다.

## 로컬 재개: QK 완전 마스킹 타일 생략

현재 로컬 작업 위치는 `.worktrees/potal-qk-mask`, 브랜치는
`work/potal-qk-mask-local`이다. 기존 `develop` 작업은 보존했다. nano의
`63e166da635bcbedaf9edb8cb84bd3e88851c3dc`를 받아 테스트 브랜치 기준으로 반영했다.
복원한 코드·테스트·빌드 설정 13개와 원문 명세는 모두 원본과 일치했다.
추가된 이관 문서·증빙을 반영하고 로컬 QK 변경과 측정 결과를 보존했다.
별도 압축 파일은 SSH 인증 실패로 아직 받지 못했다. 대조 근거는
[nano-sync.json](experiments/potal_a4/results-qk-mask-10chunk/nano-sync.json)에 있다.

기존 graph/native replay/정수 GEMM에 실제 mask 전달과 QK 타일 생략을 연결했다.
원래 stripe와 residual row ID를 유지하고, 완전히 `-inf`인 타일만 생략한다.
PV와 zero-point 보정은 유지했다. 로컬 Llama context512 ×10에서 전후 PPL은
**13.801307254545433으로 동일**하며 logits·softmax·토큰별 NLL도 같다.
CPU 정수곱의 DIM32 환산 main 작업량은 46.875% 감소했다. 제품 Gemmini/IM2P
가속 결과는 아니다. Apple M5의 QK 15개 사례 시간은 365.941→403.154 ms로
10.17% 증가했으므로 CPU 기본값으로 채택하지 않는다. 순수 정수곱 시간은
줄었지만 호출 수와 나머지 처리가 늘었다. 자세한 적용 범위와 근거는
[로컬 진행 기록](experiments/potal_a4/NANO_PROGRESS.md#local-continuation-qk-masked-tiles-2026-10-12)에 있다.
아래 nano 수치는 당시 환경·입력의 별도 결과다.

## 최신 nano 결정과 상태

이 문서의 이관 당시 기록에 앞서 [현재 진행 기록](experiments/potal_a4/NANO_PROGRESS.md)과
[명세-작업 대응표](experiments/potal_a4/IMPLEMENTATION_PLAN.md#specification-to-work-audit)를 확인한다.

- 사용자는 **ExSIA 지원 비트 폭에 공통으로 sigma 통계에서 원본 0을 제외하는 정책을 채택**했다. nonzero 원본이 qround 후 0이 되는 경우는 포함한다. Top/sigma와 D>0 보정 mask는 구분한다. Nano의 A4 한정 조건은 공통 정책으로 정정한다.
- 새 전후 PPL 비교는 **10 chunks**로 고정한다. 현재 context512, WikiText-2 test 앞 5,120 input / 2,550 scored tokens다. 아래의 과거 8-chunk 숫자와 직접 비교하지 않는다.
- 제품 zero-only 수정의 10-chunk PPL은 40.18882947→40.20531900, ExSIA 검사 5/5 통과다. 이 비교는 CPU main/RMD·FP attention·ACC_ROWS1024 조건이며 원문 a4nks/geometry 또는 Gemmini RTL 검증 완료가 아니다.
- 호환 IM2P `20b82f0`의 A4/W4/DIM32 archive 빌드·manifest·native link와 provider raw-dot 15개 검사가 통과했다. 일반 pointer API가 scale 정보 없이 block size를 K로 바꾸는 문제를 확인해, 기존 제품과 같은 provider 경로로 probe를 맞췄다. 설치된 이전 checkout `5da6a4c`와 혼동하지 않는다. 제품 a4nks dispatch·전체 성능 검증은 별도로 남아 있다.
- ±1 상위 digit의 덧셈 처리 후보는 Llama-3.2-1B Q4_HP1 context512·10-chunk에서 검증했다. 전후 PPL은 모두 **21.486062983190273**이며 packet/scale/attention/logits/token-NLL도 비트 단위로 같다. 다른 digit은 정확한 곱셈으로 보존했다. CPU에서 0 생략+unit 처리는 30개 attention 사례 합산 시간을 8.82% 줄였으나, 0만 생략하는 대조군 대비 unit 처리의 추가 이득은 0.42%로 반복 변동보다 작다. 현재 Gemmini raw 32×32×32는 일반 값·unit 값 모두 492 cycles다. 단순 unit 치환을 제품 기본값으로 채택할 성능 근거는 부족하다.
- 사용자의 추가 요청으로 기존 `native_measure.py`에 cuBLAS 정수 CUDA 비교 경로를 연결했다. 84개 원시곱 검사 및 실제 attention 30개 사례 모두 정확히 같았다. 개별 K32 곱마다 복사·동기화하고 host SCU를 유지한 CUDA는 CPU 기본 경로보다 29.83% 느렸다. GPU 상주·묶음 실행·SCU 통합은 아직 실험하지 않았다. CPU/CUDA 성능 근거는 [component-summary.json](experiments/potal_a4/results-unit-add-llama512/component-summary.json)이다. 전체 모델 CUDA PPL 또는 Gemmini 가속 결과로 해석하지 않는다.
- 다음 구현 대상으로 **QK의 실제 mask가 전부 -inf인 타일 생략**을 선택했다. 현재는 조사만 끝났으며 구현·전후 PPL·성능 측정은 미완료다. 원래 Q/K 양자화·stripe·residual 원본 행 번호를 유지하고, 생략한 score는 softmax가 안전하게 읽도록 처리해야 한다. PV 생략은 이번 범위에 포함하지 않는다. 상세 인수인계는 [QK 조사 기록](experiments/potal_a4/NANO_PROGRESS.md#qk-fully-masked-tile-skip-investigation--implementation-pending)에 있다.
- 2026-10-12 이관 요청은 기존 변경과 증거를 **`work/potal-attn-im2p-nano`에 커밋하고 같은 원격 브랜치에 일반 push**하는 것이다. develop/hotfix 병합, reset, 기존 변경 폐기, 새 구현·실험은 하지 않는다. hotfix 이관은 이후 작업이다.
- 전달 압축 파일은 `/home/dyndnn/Projects/llama.cpp-gemmini/output/potal-nano-transfer-20261012.tar.gz`이다(278,803,401 bytes). SHA256은 `458d8ec4eec37419d9615f3be02de8657fe30fa6555f7e5e3e2c305c608e3cc4`. 파일별 목록·해시, 실제 실행 명령, 의존성 commit·설정·artifact 경로는 [이관 증빙](experiments/potal_a4/results-nano/transfer-20261012/)에 있다. 대용량 capture·토큰·원시 로그는 압축 파일로 전달하며 모델·캐시·컴파일 산출물은 포함하지 않는다.
- 수정 전에 [기존 구현·재사용 대응표](experiments/potal_a4/IMPLEMENTATION_PLAN.md#existing-implementation-and-reuse-audit)에서 담당 함수·호출자·테스트를 확인한다. 기존 ExSIA/RMD/matmul/provider/graph/계측을 우선 수정한다. 계획에 적혀 있던 새 파일·테스트 이름은 생성 의무가 아니며, 기존 코드로 충족되지 않는 책임을 확인한 경우에만 필요한 부분을 추가한다.

이하의 실측 표와 작업 목록은 이관 당시의 역사적 기록이다. 현재 backend 연결과
검증 범위는 맨 위의 최신 작업을 기준으로 한다. 원문 선형 통합, fused emitter,
임의 정밀도 accumulator와 실제 attention RTL 검증은 여전히 남아 있다.

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

**대화 세션도 이 브랜치에 포함했다.** [세션 복원 방법](experiments/potal_a4/session/README.md)에 따라 nano에서 `codex resume`으로 이어 열 수 있다. [대화 원문](experiments/potal_a4/session/CONVERSATION.md)에는 처음 제공한 전체 코드와 이후 결정·실험 설명이 들어 있다. 원본과 충돌하지 않도록 복원용 세션 ID는 `5d07d5b8-15f4-5d29-a2d5-01e2398bb920`로 분리했다.

## 반드시 유지할 결정

최신 조건은 **처음 제공된 selective-linear + a4nks 대비 정확도가 떨어지면 안 된다**는 것이다. FP32 대비 양자화 손실까지 없애라는 조건과는 구분한다. 정책 이름은 `potal-a4-preserve-v1`이다.

- 선형: `corr = outlier || D > 0`; `v = corr ? u : sat(u)`를 직접 balanced radix-16 digit으로 분해한다. 보정 대상이 아닌 `u=8`은 원래대로 7이다.
- Q: `rho=6`, `v=u`, 필요한 digit을 모두 유지한다. PE는 signed 4비트다.
- P: 원래 geometry stripe의 uint8, main `(q&15)-8`, upper digits `q-(q&15)`, zero point 8. Q와 동일한 양자화 방식이 아니다.
- K: `[-128,119]`의 8비트 코드, signed nibble 두 패스. V: token block / channel별 4비트.
- top exponent / strict 2-sigma / distinct E2, stripe scale, 원래 K block, mask와 FP softmax를 유지한다. FP KV 저장 형식도 유지한다.
- Sigma 모집단은 source-nonzero이고 top을 제외한다. 고정 비율 선택이 아니며, 0의 D>0 보정 표시를 sigma outlier로 해석하지 않는다.
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
rtk proxy env PYTHONPATH=gguf-py:. OPENBLAS_NUM_THREADS=2 python3.11 -m experiments.potal_a4.run models/gpt2-base/gpt2.Q4_HP1.gguf experiments/potal_a4/results-nano/test.i32 512 10 original_a4nks experiments/potal_a4/results-nano/test512-original_a4nks.json
rtk proxy env PYTHONPATH=gguf-py:. OPENBLAS_NUM_THREADS=2 python3.11 -m experiments.potal_a4.profile_attention experiments/potal_a4/results-nano/test512-original_a4nks.json experiments/potal_a4/results-nano-profile/context512.json
```

과거 JSON의 Mac 절대 경로는 측정 당시 출처다. 이를 덮어쓰지 말고 nano baseline을 새로 생성한다. Profiling은 baseline JSON 옆 `test.i32`, `.chunk0.npz`와 JSON의 model 경로를 읽는다. 역사적 `.npz`는 Git에 없으며 새 baseline으로 재생성할 수 있다. OS/BLAS가 다른 실행을 기존 Mac hidden과 bitwise 동일하다고 가정하지 않는다. 같은 nano 경로 안에서 baseline/변경 후 동등성을 검증한다.

**메모리 단위 이식 완료:** nano에서 `run.py`와 `native_probe.cpp`의 Linux `ru_maxrss` KiB를 bytes로 정규화했다. Darwin의 bytes와 과거 JSON은 유지했다. 현재 결과는 [진행 기록](experiments/potal_a4/NANO_PROGRESS.md)을 참조한다. Native probe의 DIM32만으로 ACC_ROWS512 geometry 정합성을 주장하지 않는다.

## Gemmini / IM2P에서 할 일

1. [계획](experiments/potal_a4/IMPLEMENTATION_PLAN.md)의 수치·표현 계약부터 확인한다. Python 실험은 dense 임시 residual을 사용하고 native probe도 scalar bridge가 있어 제품 fused emitter가 아니다.
2. 기존 [IM2P resolver](cmake/ggml-gemmini-im2p.cmake)와 [provisioning helper](scripts/im2p-host-provision.sh)로 nano용 `GEMMINI_HP1`, A4/W4/DIM32/BK32 matched selected artifact를 확인한다. `GGML_GEMMINI_EXECUTION_BACKEND=IM2P_SIM`, `GGML_GEMMINI_OPTION=WS`, `CYCLE_SIM=0`이 계획의 raw op0 경로다. Mac archive나 CPU-functional op5 trace는 이 실행 증거가 아니다.
3. `Im2pCompactDot` raw BYPASS + K<=32와 host exact merge를 연결한다. 기존 HP1 SCU-final은 int32 saturation이 있으므로 잘린 값을 exact 결과로 재사용하지 않는다. Int64 bound를 넘는 경우의 정확한 합산도 계획에 포함되어 있다.
4. 원문 값 보존 direct emitter, 선형 통합, K/P/V 정책, QK→FP softmax→PV 그래프 연결을 진행한다. QK/PV 두 노드의 실제 Gemmini 실행을 확인하고 CPU fallback을 별도로 기록한다.
5. limb/row/K 분포, main/upper/K-high 제출 수, selection·packing·compensation·host merge 시간, packet/live peak memory를 nano에서 따로 측정한다. Fragment 수를 속도 배수로 해석하지 않는다.

Geometry 주의: 실험의 논리 stripe는 `(banks,bank_rows,acc_rows)=(4,4096,512)`다. Mac probe의 generated header는 ACC_ROWS=1024였고 probe는 stripe=32로 고정했다. 이 결과로 hardware schedule parity를 주장하지 않았다. Nano의 실제 계약을 기록하고, 논리 quantization stripe와 물리 tile 차이가 정확도를 바꾸지 않도록 먼저 대조한다.

현재 `GGML_GEMMINI_ENABLE_A4NKS_ATTENTION`은 구현된 opt-in 옵션이다.
기존 `test-gemmini-matmul`의 attention case를 확장했으며 새 실행 파일을 만들지
않았다. KV/strided view/GQA와 1-row node 검사는 로컬 CPU에서 통과했다.
실제 모델의 증분 KV decode 및 Nano RTL 검증까지 통과한 것으로 해석하지 않는다.

## 새 대화에 붙여넣을 요청

> `work/potal-attn-im2p-nano`에서 이어서 작업하자. 먼저 `POTAL_NANO_HANDOFF.md`와 연결된 ORIGINAL_SPEC.md, IMPLEMENTATION_PLAN.md, RESULTS.md, ATTENTION_PROFILE.md를 읽어. Gemmini / IM2P를 nano에서 실제로 실행하고 limb·이상치 선택·packet 생성·compensation의 시간과 공간을 측정하려는 작업이다. 최신 조건은 원문 selective-linear+a4nks 대비 정확도 저하 금지다. Direct-all 선형, limb 제한, P 8행 실험은 채택하지 않았다. 현재 제품 attention 경로가 미완성임을 전제로 로컬 환경과 matched A4/W4/DIM32 artifact부터 확인하고, 계획의 원문 값 보존 direct emitter와 실제 QK/PV 통합을 이어가. 수정 전에 재사용 대응표와 실제 호출자를 확인해서 기존 구현을 우선 수정하고 중복 구현을 추가하지 마. 원본 0 제외는 채택했고 전후 PPL은 10 chunks로 측정해. 기존 실험 숫자는 Gemmini 지연시간으로 해석하지 말고, 같은 nano 경로에서 정확도 동등성과 실행 경로를 검증해.
