# Gemmini CUDA 팀원 인계

기준일: 2026-11-09. 전달 브랜치는 `feat/mxint-qkpv-monitoring`이다.
핵심 CUDA 변경은 `d46554c`, native 정확도 실행 기반은 `584e558`에 있다.
이 문서는 팀원이 새 환경에서 빌드하고 검증 범위를 판단하기 위한 진입점이다.
보류 중인 추가 SCU 실험은 전달 대상에 넣지 않았다. PoTal 실행에 필요한
기존 SCU 구현과 수치 회귀 검사는 그대로 사용한다.

## 1. 받을 코드와 읽는 순서

저장소 전체를 checkout하고 대상 컴퓨터에서 새로 빌드한다. 기존의 깨끗한
checkout에서는 다음처럼 브랜치를 받고, 실제 받은 revision을 기록한다.

```bash
GIT_MASTER=1 git fetch origin
GIT_MASTER=1 git switch feat/mxint-qkpv-monitoring
GIT_MASTER=1 git pull --ff-only
GIT_MASTER=1 git rev-parse HEAD
```

1. 이 문서: 대상, 빌드 순서, 검증 완료 범위.
2. [native 정확도 안내](GEMMINI-CUDA-NATIVE-ACCURACY.md): scorer CLI와 데이터 준비.
3. [CUDA PPL 안내](GEMMINI-CUDA-PPL.md): profile별 runtime switch와 proof.
4. [수치 정책](ROUNDING_POLICY.md): 반올림, scale, 저장 형식과 비교 조건.

## 2. 대상과 수치 계약

| 대상 | 보존한 계산과 선택된 변경 |
| --- | --- |
| FP16 | 연속 F16 복사와 F32 scalar RNE pack. FP16 입력/FP32 누산 유지. |
| NVFP4 | block16 E2M1, E4M3FN scale, RNE. reduction과 scale encoder 최적화. |
| MXINT4/8 | block32 PoT/ties-away. 기존 F64 곱과 K 순서 합 유지. |
| PoTal LINEAR TOP3 A4/A8 | CPU EXSIA/folding, CUDA main/residual와 ordered saturation 유지. immutable weight 준비 tile 재사용. |
| PoTal LINEAR+attention TOP3 A4/A8 | 같은 SCU 계약에 QK/PV 적용. 안전한 dense window만 직접 참조. |
| BLOCK/BLOCK4/8 | Q4_0/Q8_0의 M>=128, N>=512에 2행 register tile. K32 grouping과 ordered F64 합 유지. |

MXINT clipping은 4bit `[-7,7]`, 8bit `[-127,127]`이다. HP1 저장 형식이나
파일명만으로 activation 정책을 추정하지 않는다. 모델 sidecar와 runtime proof를
함께 확인한다. 등록된 weight는 immutable이어야 하며, 변경·해제·주소 재사용에는
기존 clear/lifetime 절차가 필요하다. activation을 주소만으로 재사용하지 않는다.
PoTal 준비 단계가 모두 GPU로 옮겨진 구현은 아니다.

NVFP4의 전체 모델 비교 profile은 `fp32` 재구성/pedantic FP32 GEMM이다.
`fp16` 재구성/FP32 누산은 별도 profile이며 서로 결과를 대체하지 않는다.
이 경로는 FP16 GGUF를 동적 양자화하는 참조 구현이지 native FP4 명령의
하드웨어 처리량 측정이 아니다.

## 3. 새 환경에서 빌드

Linux 또는 CUDA 사용 가능한 WSL, CUDA toolkit/cuBLAS, Ninja, OpenMP 지원
C++ compiler, Python 3와 Gemmini 외부 헤더가 필요하다. native 아키텍처 선택은
CMake 3.24 이상을 사용한다. 확인된 GPU는 RTX 3070/SM86이다.
모델 GGUF, 대응 sidecar, 원본 corpus와 외부 헤더는 별도로 준비한다.

저장소 루트에서 실행한다. 아래 명령은 빌드만 하며 모델을 실행하지 않는다.

```bash
export GEMMINI_SW_PATH=/path/to/RISC-V-DynDNN-gemmini-include
export CUDAARCHS=native
export NVFP4_CUDA_ARCH=native
bash scripts/experiment/build-potal-cuda.sh --rtn --jobs 2 --build-only
for bits in 4 8; do
    bash scripts/experiment/build-potal-cuda.sh \
        --bits "$bits" --anchor 3 --attention --jobs 2 --build-only
done
bash scripts/experiment/build-nvfp4-reference.sh \
    "$PWD/build-potal-cuda/nvfp4-fp32" fp32
bash scripts/experiment/build-fp16-accuracy-cuda.sh
bash scripts/experiment/build-quantized-accuracy.sh
bash scripts/experiment/build-accuracy-linear.sh
```

RTN 출력은 `build-potal-cuda/rtn`, PoTal 출력은
`build-potal-cuda/top3-a4-rmdon-attention`과
`build-potal-cuda/top3-a8-rmdon-attention`이다. PoTal은 attention-capable
빌드 하나에서 scope `none`/`both`로 LINEAR/FULL을 구분한다.
CMake `CUDAARCHS`는 `86`, nvcc `NVFP4_CUDA_ARCH`는 `sm_86`처럼
서로 다른 형식을 쓴다. GPU 없는 cross-build에서는 값을 명시한다.
실행 파일과 shared library는 같은 build의 것을 사용하고, 다른 GPU/CPU에는
기존 바이너리를 복사하기보다 새 build 디렉터리에서 재빌드한다.

## 4. 검증과 모델 실행

GPU가 비어 있을 때만 수치 검사를 실행한다. RTN/PoTal build 명령에서
`--build-only`를 빼면 기존 회귀 검사와 deterministic NLL owner까지 수행한다.
NVFP4 helper는 컴파일 전용이다. 빌드 성공을 수치 검증 성공으로 읽지 않는다.

canonical CUDA 검사 진입점은 다음과 같다.

```bash
OMP_NUM_THREADS=4 ctest --test-dir build-potal-cuda/rtn \
    --output-on-failure -R '^test-cuda-rtn$'
for bits in 4 8; do
    build="build-potal-cuda/top3-a$bits-rmdon-attention"
    OMP_NUM_THREADS=4 ctest --test-dir "$build" --output-on-failure \
        -R '^test-cuda-(cpu-exact-graph|hp1-scu(-graph)?|rtn|potal-attention)$'
    OMP_NUM_THREADS=4 ctest --test-dir "$build" --output-on-failure \
        -R '^test-gemmini-attention(-mask)?$'
    GGML_GEMMINI_CUDA_POTAL_ATTENTION=1 OMP_NUM_THREADS=4 \
        ctest --test-dir "$build" --output-on-failure \
        -R '^test-gemmini-attention(-mask)?$'
done
python3 tests/test-perplexity-deterministic.py
```

PPL 비교는 원본 WikiText-2 test 전체를 사용하고 context/batch/ubatch=512,
threads=4, seed=42, stride=0, no warmup, KV F16, TF32 off를 유지한다.
Qwen은 584 chunks/148920 scored tokens, Llama는 564/143820이다.
일부 chunk나 반올림된 PPL만으로 동등성을 승인하지 않는다.
FlashAttention은 끄고 이 graph 경로에 필요한 KV offload는 유지한다.

실제 PPL 명령과 runtime 환경은 CUDA PPL 안내의 profile별 설명을 따른다.
이 host-graph 경로는 `GGML_CUDA=OFF` 및 `--gpu-layers 0`이어도 custom
Gemmini CUDA를 사용한다. 해당 옵션만 보고 CPU-only로 판정하지 않는다.
NVFP4 preload 경로에 공백이 있으면 library 디렉터리의 subshell에서
`LD_PRELOAD=./libggml-nvfp4-preload.so`로 실행하고 모델·corpus·executable은
절대 경로로 유지한다. legacy queue는 FP16 NVFP4 profile과 로컬 SM86 설정을
전제로 하므로 FP32 NVFP4/FULL 또는 다른 GPU 검증기로 그대로 사용하지 않는다.

native ACC scorer는 요청별 likelihood와 backend proof를 계산한다.
태스크 전체 정답률은 pinned harness의 prompt, continuation 경계,
`process_results`와 aggregation을 연결해야 한다. 기존 일정은 6 threads,
context 4096, batch 512, 독립 요청이며 prefix 공유나 혼합 batching을 적용하지 않는다.
초과 길이는 조용히 자르지 않고 EOF의 최종 proof와 exit status까지 확인한다.
7개 task 전체 평가기나 token/response cache runner가 완성됐다는 뜻은 아니다.

## 5. 확인된 범위와 남은 검증

기록된 원본 소스 재빌드 검증은 RTN/TOP3 A4/A8의 19 CTest, NVFP4 두 재구성
모드 각각 69-case frozen GPU/독립 CPU 비교와 deterministic NLL owner 통과다.
native scorer 6개도 재빌드했고 ordered likelihood의 CPU 비트 일치 검사가 통과했다.
아래 장부는 그 재빌드와 별도로 동결한 선택 이미지 대 frozen CUDA baseline의
전체 chunk NLL 및 최종 NLL/PPL double 비트 비교다.

| profile | Qwen2-0.5B | Llama3.2-1B |
| --- | --- | --- |
| FP16 | PASS | PASS |
| NVFP4 FP32 pedantic | PASS | 미완료 |
| MXINT4 | PASS | PASS |
| MXINT8 | PASS | PASS |
| BLOCK4 | PASS | PASS |
| BLOCK8 | PASS | PASS |
| PoTal LINEAR TOP3 A4 | PASS | PASS |
| PoTal FULL TOP3 A4 | PASS | 미완료 |
| PoTal LINEAR TOP3 A8 | PASS | 미완료 |
| PoTal FULL TOP3 A8 | PASS | 미완료 |

합계 16/20이다. 완전한 새 영수증이 나오기 전에는 미완료 행을 올리지 않는다.
기존 Llama 장시간 실행에는 6시간 상한으로 중단된 기록이 있으며 수치 불일치로
판정한 결과가 아니다. 모집단을 줄이거나 실패 영수증을 고쳐 성공 처리하지 않는다.
현재 다른 GPU의 검증과 독점 조건의 전체 모델 성능 향상 승인은 없다.
공유 GPU의 stage timing을 최종 speedup으로 전달하지 않는다.

전체 frozen pair owner와 과거 원시 증거는 이 Git 묶음에 포함된 portable
회귀 검사와 별개다. 새 환경에서는 baseline/candidate의 source, model,
corpus, binary/library, build 설정, runner와 환경을 실행 전후 hash로 결속하고
모든 chunk/proof/exit를 확인해야 같은 동등성 주장을 할 수 있다.
새 데이터나 모델에는 원래 task/population/tokenizer 기준을 별도로 확립한다.

기존 한계도 보존한다. CPU LINEAR 대조는 CUDA와 logits가 달라 CPU 대체로
승인하지 않았다. 과거 ACC 한 건은 등록 소스 변경에 따른 입력 hash 검사 실패였고,
그 저장 점수 진단을 정상 완료 ACC나 새 후보 성능 결과로 쓰지 않는다.
실행 중인 평가의 source·정책·데이터·binary를 변경하지 않는다.

## 6. 전달된 파일별 역할

경로는 저장소 루트 기준이다. 앞선 두 커밋의 25파일과 이 인계 문서를 함께 받는다.

| 파일 | 역할 |
| --- | --- |
| `ggml/src/ggml-cuda/cpu-exact/ggml-cuda-hp1-scu.cu` | immutable weight 준비 tile 재사용 |
| `ggml/src/ggml-cuda/cpu-exact/ggml-cuda-potal-attention.cu` | 안전한 QK/PV dense window 재사용 |
| `ggml/src/ggml-cuda/cpu-exact/ggml-cuda-rtn.cu` | FP16 pack과 기존 MXINT/RTN 경로 |
| `ggml/src/ggml-cuda/cpu-exact/ggml-cuda-rtn-fast.cu` | BLOCK4/8의 2행 register tile |
| `ggml/src/ggml-cuda/nvfp4-reference/ggml-nvfp4-preload.cu` | NVFP4 reduction/scale encoder/버퍼 |
| `ggml/src/ggml-gemmini/CMakeLists.txt` | canonical CUDA 소스 연결 |
| `scripts/experiment/build-potal-cuda.sh` | RTN/PoTal 빌드와 수치 검사 |
| `scripts/experiment/build-nvfp4-reference.sh` | 모드별 NVFP4 빌드 |
| `scripts/experiment/build-fp16-accuracy-cuda.sh` | FP16 scorer 빌드 |
| `scripts/experiment/build-quantized-accuracy.sh` | 양자화 scorer 빌드 |
| `scripts/experiment/build-accuracy-linear.sh` | LINEAR scorer 빌드 |
| `scripts/experiment/prepare-accuracy.py` | 공식 평가 데이터 준비 |
| `scripts/experiment/check-accuracy-tasks.py` | 고정 하네스 요청 구성 확인 |
| `tools/accuracy-fp16-cuda/score.cpp` | FP16 likelihood/proof |
| `tools/accuracy-quantized/score.cpp` | 양자화 likelihood/proof |
| `tools/accuracy-linear/score.cpp` | LINEAR likelihood/proof |
| `tools/accuracy-fast/ordered-logprobs.hpp` | 순서 보존 likelihood 계산 |
| `tools/accuracy-fast/test-ordered-logprobs.cpp` | likelihood 비트 회귀 검사 |
| `tests/test-cuda-hp1-scu.cpp` | SCU 출력·통계·수명 회귀 검사 |
| `tests/test-cuda-potal-attention.cpp` | attention window 회귀 검사 |
| `tests/test-cuda-rtn.cpp` | FP16/RTN/MXINT/tail 회귀 검사 |
| `docs/backend/GEMMINI-CUDA-PPL.md` | PPL runtime/profile/proof 안내 |
| `docs/backend/GEMMINI-CUDA-NATIVE-ACCURACY.md` | native scorer와 데이터 준비 안내 |
| `docs/backend/ROUNDING_POLICY.md` | 수치 정책 |
| `ggml/src/ggml-cuda/nvfp4-reference/README.txt` | NVFP4 모드와 loader 안내 |
| `docs/backend/GEMMINI-CUDA-OPTIMIZATION-HANDOFF.md` | 팀원 전달 진입점과 검증 범위 |
