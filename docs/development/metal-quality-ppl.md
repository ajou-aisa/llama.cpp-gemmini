# CPU 계산을 보존하는 Metal PPL 실행

`scripts/experiment/run-metal-quality-ppl.sh`는 원본 GGUF로 아래 20조건을 실행한다.

| Method | A/W | DIM | 모델 / n |
| --- | --- | --- | --- |
| RTN-W | 16/n | 해당 없음 | GPT-2, Llama-3.2-1B / 4, 8 |
| RTN-WA | n/n | 해당 없음 | GPT-2, Llama-3.2-1B / 4, 8 |
| PoTal | n/n | 16, 32, 64 | GPT-2, Llama-3.2-1B / 4, 8 |

평가 기본값은 WikiText-2 raw test 전체, context/batch/ubatch 512, threads 4, KV F16, seed 42, warmup OFF다. FP16 baseline은 실행하지 않는다. 모든 조건은 `models/default/`의 기본 양자화 GGUF만 읽는다. `default-ppl-models.sha256`과 다른 파일은 거부하고, 모델 경로 환경변수 override도 거부한다. 실행 중 모델 재양자화나 embedding/head 강제 교체는 없다.

| 모델 포맷 | GPT-2 공유 embedding/head | Llama 공유 embedding/head |
| --- | --- | --- |
| Q4_0 | Q6_K | Q6_K |
| Q8_0 | Q8_0 | Q8_0 |
| Q4_HP1 | Q4_HP1 | Q4_HP1 |
| Q8_HP1 | Q8_HP1 | Q8_HP1 |

8개 모델은 현재 `develop` 소스의 `llama-quantize INPUT OUTPUT TYPE 4` 기본 옵션으로 준비했다. `--pure`, `--token-embedding-type`, `--output-tensor-type`, `--tensor-type`, `--allow-requantize`는 사용하지 않았다. GPT-2 입력은 `models/gpt2.fp16.gguf`, Llama 입력은 `/Users/chan/Downloads/llama3.2-1B.fp16.gguf`다. 생성 기록은 `.omo/evidence/default-ppl-models-20261006/`에 있다.

이전 안내의 "Q4_HP1 원본 head는 F16" 설명은 잘못됐다. 해당 파일들은 이전 실험용 빌드의 결과였으며 현재 기본 양자화 결과가 아니다. 이번 선택에서는 제외했다. 기본 Q4_HP1 head는 두 모델 모두 Q4_HP1이다.

GPT-2 RTN-W n=4를 다시 측정하려면 다음을 실행한다.

```bash
rtk proxy caffeinate -i bash scripts/experiment/run-metal-quality-ppl.sh --case gpt2-rtnw4
```

Q4_0 조건은 본체를 Metal에서 실행하고 Q6_K head는 원본 CPU Q6_K × Q8_K 경로를 사용한다. RTN-W/RTN-WA, GPT-2/Llama 모두 해당한다. 실행 검사에 `GGML_GEMMINI_METAL_CPU_EXACT_Q6_HEAD=1`을 명시하고, Q6_K 최종 head 호출을 `cpu_q6_head_matmuls`로 별도 기록한다. 본체의 GPU 실행 검사는 그대로 유지한다. 이전 F16 head 측정값은 이번 원본 모델 결과와 합치지 않는다. 각 실행 폴더의 `head-policy.txt`와 모델 hash로 구분한다.

모델 선택은 `ppl-original-models.sh`와 `default-ppl-models.sha256`으로 고정한다. `GPT2_Q4_0` 등 이전 모델 경로 override 변수가 설정돼 있으면 오류로 처리한다. 기본 파일이 없거나 바뀌어도 실행 전 오류로 처리하며 다른 파일로 대체하지 않는다.

## 변경된 계산 경로

`GGML_GEMMINI_METAL_CPU_EXACT=ON` 빌드는 기존 Gemmini CPU 그래프 내부에서 Metal을 호출한다. 일반 Metal backend는 `GGML_METAL=OFF`, 실행 인자는 `--gpu-layers 0`이다. 실제 Metal 계산은 별도 helper에서 이루어지고, 실행 시 `GGML_GEMMINI_METAL_CPU_EXACT=1`로 켜진다. Runner가 이 환경변수를 설정한다.

| 경로 | Metal에서 실행 | 기존 CPU에서 실행 |
| --- | --- | --- |
| RTN-W | CPU와 같은 FP32 FMA/reduction 순서의 dot | 원본 weight의 F32 복원, 다른 그래프 연산 |
| RTN-WA | K 구간별 정수 dot | BLOCK producer, 기존 double scale 적용과 F32 cast, 다른 그래프 연산 |
| PoTal | K 구간별 정수 dot | ExSIA producer, CPU-direct HP1 scale 복원, 기존 residual, 다른 그래프 연산 |

RTN-W의 `16/n`은 사용자 표의 명목 표기다. 여기서는 기존 FLOAT baseline의 F32 activation과 F32로 복원한 weight를 사용한다. PoTal은 CPU-direct 산술을 재현하며, HP1 하드웨어 SCU 포화 산술과 구분한다. RTN-WA는 RMD OFF, PoTal은 RMD ON / sigma 2다.

FP32 reduction은 현재 Apple Clang 21 Release arm64 CPU 구현을 기준으로 검증한다. subnormal에 민감한 입력은 Metal의 정수 연산으로 FP32를 재현한다. 빌드 도구나 최적화 플래그를 바꾸면 작은 bitwise 회귀를 다시 확인해야 한다.

## 한 조건부터 직접 실행

Repository root에서 다음 두 명령을 실행한다. `--prepare`는 기존 `build-arm64.sh`를 호출해 빌드만 한다. 두 번째 명령이 실제 PPL 측정이다.

```bash
rtk proxy bash scripts/experiment/run-metal-quality-ppl.sh --prepare --case llama-rtnw4
rtk proxy bash scripts/experiment/run-metal-quality-ppl.sh --case llama-rtnw4
```

짧게 확인하려면 실행 명령에 `--chunks 1`을 붙인다. 전체 측정은 기본값 `--chunks -1`이다.

Case ID:

```text
{gpt2,llama}-{rtnw4,rtnw8,rtnwa4,rtnwa8,potal4-d16,potal4-d32,potal4-d64,potal8-d16,potal8-d32,potal8-d64}
```

중단 후 특정 조건부터 실행하려면 `--from llama-rtnw4`처럼 지정한다. 중단된 조건의 chunk를 이어받지는 않으며, 해당 조건은 처음부터 다시 실행한다. 결과는 새 폴더에 저장된다. 이전 측정은 head 설정이 달랐으므로 이번 원본 모델 비교는 20조건 전체를 새로 실행한다.

## 표의 20조건 전체 실행

```bash
rtk proxy bash scripts/experiment/run-metal-quality-ppl.sh --prepare
rtk proxy caffeinate -i bash scripts/experiment/run-metal-quality-ppl.sh
```

모델끼리 동일한 빌드를 공유하므로 10개 profile을 준비한다. 기본 빌드 위치는 `.omo/metal-cpu-exact/quality-<profile>`이다. Q6_K 원본 head 실행 검사가 추가됐으므로 위 `--prepare`를 먼저 한 번 실행한다. 실행 단계는 재빌드하거나 모델을 변환하지 않는다. 이미 준비한 profile을 다시 `--prepare`하면 CMake 설정 확인과 증분 빌드를 수행한다.

기존 `scripts/experiment/run-ppl-*.sh` 개별 스크립트 12개도 같은 Metal runner로 연결된다. 예를 들어 `run-ppl-gpt2-q4_0-a16.sh`는 `gpt2-rtnw4`, `run-ppl-gpt2-q4_0-a4.sh`는 `gpt2-rtnwa4`다. `--prepare`, `--dry-run`, `--chunks 1` 옵션을 그대로 전달할 수 있다. CPU 전용 실행이 필요하면 `run-one-cpu-ppl.sh`를 직접 사용하며, 이 스크립트 역시 기존 모델을 읽고 head 교체나 양자화를 수행하지 않는다.

실행 없이 명령만 보려면:

```bash
rtk proxy bash scripts/experiment/run-metal-quality-ppl.sh --prepare --case gpt2-potal4-d32 --dry-run
rtk proxy bash scripts/experiment/run-metal-quality-ppl.sh --dry-run
```

`--dry-run`은 준비 전에도 명령을 출력한다. 실제 실행은 바이너리, 모델, CMake 설정을 검사한다.

## build-arm64.sh 설정을 직접 바꾸는 경우

위 `--prepare --dry-run`이 출력하는 `build-arm64.sh` 명령의 `-D` 값을 편집해 빌드할 수 있다. 예를 들어 PoTal A4W4/DIM32:

```bash
rtk proxy env BUILD_DIR=build-metal-custom BUILD_JOBS=4 bash build-arm64.sh \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
  -DGGML_GEMMINI=ON -DGGML_GEMMINI_METAL_CPU_EXACT=ON \
  -DGGML_METAL=OFF -DGGML_METAL_QUANTIZED=OFF -DGGML_CUDA=OFF -DGGML_BACKEND_DL=OFF \
  -DGGML_BLAS=OFF -DGGML_LLAMAFILE=OFF -DGGML_OPENMP=OFF -DGGML_GEMMINI_ENABLE_OPENMP=ON \
  -DGGML_GEMMINI_OPTION=CPU -DGGML_GEMMINI_EXECUTION_BACKEND=HARDWARE \
  -DGGML_GEMMINI_COMPUTE_TYPE=INT -DGGML_GEMMINI_DEQUANT_FP_TEST=OFF \
  -DGGML_GEMMINI_ACTIVATION_QUANT=EXSIA \
  -DGGML_GEMMINI_ACTIVATION_BITS=4 -DGGML_GEMMINI_WEIGHT_BITS=4 \
  -DGGML_GEMMINI_DIM=32 -DGGML_GEMMINI_BLOCK_SIZE=32 \
  -DGGML_GEMMINI_EXSIA_SIGMA=2 -DGGML_GEMMINI_ENABLE_RMD=ON \
  -DGGML_GEMMINI_DEFAULT_RMD_BACKEND=CPU \
  -DGGML_GEMMINI_ENABLE_STRIPE_MATMUL=ON -DGGML_GEMMINI_ENABLE_STRIPE_PIPELINE=ON \
  -DGGML_GEMMINI_DEFAULT_MATMUL_MODE=STRIPE_PIPELINE \
  -DLOG_DEBUG=0 -DLOG_CYCLE=0 -DCYCLE_DETAIL=0 -DCYCLE_SIM=0

rtk proxy bash scripts/experiment/run-metal-quality-ppl.sh \
  --case gpt2-potal4-d32 --build-dir build-metal-custom
```

`--build-dir`는 단일 조건에만 쓸 수 있다. Runner는 해당 cache를 검사하고 그대로 실행한다. RTN-W는 `COMPUTE_TYPE=FLOAT`, `ACTIVATION_QUANT=BLOCK`, RMD/stripe OFF, FULL을 사용한다. RTN-WA는 같은 설정에서 `COMPUTE_TYPE=INT`다. A8W8이면 activation/weight bits를 모두 8로 바꾼다. 표의 비교 조건과 다른 설정은 runner가 거부한다.

## 검증과 결과 파일

수정 검증에는 작은 고정 입력의 실제 CPU/GPU 출력을 사용한다. 고정 입력의 bitwise 일치, 전체 corpus PPL 일치, 속도 개선은 각각 별도로 확인해야 한다. 현재 경로는 CPU scale/residual 비용과 GPU 호출 비용을 포함하므로 속도 개선을 보장하지 않는다.

`-DLLAMA_BUILD_TESTS=ON`으로 설정한 동일 profile 빌드에서:

```bash
rtk proxy cmake --build <build-dir> --target test-metal-cpu-exact-float test-metal-cpu-exact-int test-metal-cpu-exact-graph -j4
rtk proxy ctest --test-dir <build-dir> --output-on-failure -R '^test-metal-cpu-exact-'
```

기본 결과 위치는 `output/experiment/metal-quality-ppl-<timestamp>-<pid>/results.tsv`다. 각 조건의 명령, cache, source/model/binary hash, 로그, 종료 코드와 `metal-cpu-exact-proof.json`이 함께 저장된다. Q6_K 최종 head는 CPU 호출 수를 따로 검사하고, 나머지 양자화 matmul은 GPU 호출을 검사한다. 전체 chunk 및 채점 token 완료를 검사하고 성공한 조건만 TSV에 추가한다. `quality_ppl.md`와 `quality_ppl.tex`도 조건이 완료될 때마다 갱신한다. 실패하면 로그를 남기고 멈춘다.

이전에 중단한 `output/experiment/metal-quality-ppl-20261006-full`의 수치는 이전 일반 Metal/하드웨어 SCU 경로의 결과다. 이번 CPU 산술 보존 경로의 측정값으로 사용하지 않는다. 과거 표의 Llama RTN-W n=4 값 `11.51`은 그 측정의 모델·빌드·입력 조건을 확보해야 재현 여부를 판정할 수 있다.
