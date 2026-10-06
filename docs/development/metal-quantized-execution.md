# Metal block / HP1 ExSIA 실행

`scripts/experiment/run-one-metal-ppl.sh`는 이미 존재하는 GGUF를 그대로 읽어 custom Metal backend로 PPL을 실행한다. 모델 변환이나 재양자화는 하지 않는다. CPU에서 block/ExSIA activation과 residual payload를 생성하고 Metal에서 정수 dot, HP1 SCU/RMD, scale 복원과 merge를 실행한다. Apple GPU의 native FP64 대신 integer 기반 soft-FP64를 사용하며, 그 비용은 GPU 계산 시간에 포함된다.

## 입력과 profile

| 옵션 | 허용값 / 의미 |
| --- | --- |
| 첫 인자 | `gpt2`, `llama`; GGUF architecture도 검사 |
| `--model` | 필수. 기존 GGUF 경로 |
| `--bits` | `4`, `8`; activation과 weight의 bit width가 같음 |
| `--activation BLOCK` | 원본 `Q4_0` / `Q8_0`; 기본 RMD OFF |
| `--activation EXSIA` | 원본 `Q4_HP1` / `Q8_HP1`; 기본 RMD ON, sigma 2 |
| `--dim` | `16`, `32`, `64`; 기본 16, K block 32 |
| `--rmd` | `ON` / `OFF` 명시 가능. 실제 선택을 manifest에 기록 |

GGUF의 file type과 tensor dtype도 profile에 맞아야 한다. F16/F32 tensor는 허용하므로 embedding/head의 기존 정밀도는 입력 파일에 의해 정해진다. A16 profile은 지원 목록에 포함하지 않는다.

필요 환경은 macOS, CMake, Xcode Command Line Tools, `rtk`, Python 3와 numpy, libomp이다. bundled `gguf-py`가 읽기 전용 입력 검사를 수행한다. Python은 repository `.venv/bin/python`, sibling `../.venv/bin/python`, `python3` 순서로 numpy가 있는 interpreter를 선택한다. `PYTHON`으로 직접 지정할 수도 있다. CPU producer가 사용하는 `gemmini.h`/`gemmini_params.h`는 기본 sibling `../RISC-V-DynDNN-gemmini-include`에서 읽는다. 다른 경로는 `GEMMINI_SW_PATH`, libomp 위치는 `OpenMP_ROOT`로 지정한다. Gemmini backend와 simulator는 사용하지 않는다.

실제 검증 장비는 **Apple M5**이며 custom correctness shader library는 **Metal Shading Language 3.1**로 compile한다. 다른 Mac/GPU의 지원 여부는 해당 장비에서 configure·shader compile·수치 검증을 실행해 확인해야 한다.

## 실행

repository root에서 먼저 명령과 입력 profile을 확인한다. 다음 모델 경로는 예시이며 실제 보유한 파일을 지정한다.

```bash
rtk proxy bash scripts/experiment/run-one-metal-ppl.sh gpt2 \
  --model /path/to/gpt2.Q4_0.head-F16.gguf --bits 4 --activation BLOCK --dim 16 \
  --file wikitext-2-raw/wiki.test.raw --context 32 --chunks 1 --dry-run
```

`--dry-run`을 제거하면 configure, `llama-perplexity` build, 실행 순으로 진행한다. 위 context 32 / 1 chunk는 실행 확인용이며 전체 PPL 결과가 아니다. 전체 corpus는 기본 context 512 / `--chunks -1`로 실행한다.

```bash
rtk proxy bash scripts/experiment/run-one-metal-ppl.sh llama \
  --model /path/to/llama3.2-1B.Q4_HP1.head-F16.gguf --bits 4 --activation EXSIA --dim 64 \
  --file wikitext-2-raw/wiki.test.raw --context 512 --chunks -1
```

A8W8은 `--bits 8`과 같은 family의 Q8_0 또는 Q8_HP1 GGUF를 사용한다. `--threads`와 `--jobs`는 기본 4이며 CPU producer thread 수와 build 병렬도를 지정한다. `--build-dir`은 재사용 가능한 별도 build 디렉터리, `--results-dir`은 새 결과 디렉터리를 지정한다. 기본 build는 `.omo/metal-quantized/ppl-build-<profile>`, 결과는 `output/experiment/metal-ppl-<model>-<profile>-<timestamp>-<pid>`다. 소스 root에서 직접 build하지 않는다.

## Weight 준비 cache

CPU producer는 원본 packed weight와 복원된 정수 code·scale·carrier를 함께 보관한다. 기본 cache 상한은 **2 GiB / 256 entries**이며 원본 snapshot과 decoded vector의 capacity를 계산에 포함한다. 매번 shape·stride·dtype·주소와 실제 원본 weight byte를 비교한 뒤 재사용한다. Activation과 residual payload는 호출마다 CPU producer가 새로 만든다.

새 entry가 남은 byte/entry 용량에 들어갈 때만 저장하고, 초과하면 그 호출에서 cache 없이 사용한다. 기존 entry를 유지하므로 용량을 넘는 모델을 순회하면서 cache 전체를 계속 교체하지 않는다. 같은 key의 원본 byte가 바뀌면 기존 entry를 무효화한다. API로 상한을 줄이면 최근 사용이 오래된 entry부터 제거하며, 0은 재사용을 끈다. 실행 중 payload가 참조하는 weight는 cache 제거 이후에도 해당 payload 수명까지 유효하다. 따라서 상한은 cache가 보관하는 메모리의 범위다.

## Build와 검증 명령

다음은 A4W4 / EXSIA / DIM16의 별도 테스트 build 예시다. `bits`, activation mode, DIM을 바꿀 때에는 별도 build 디렉터리를 사용한다. correctness test는 backend symbol을 직접 연결하므로 **`GGML_BACKEND_DL=OFF`**가 필요하다.

```bash
rtk proxy cmake -S . -B .omo/metal-quantized/test-a4-exsia-d16 \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_COMMON=ON -DLLAMA_CURL=OFF \
  -DGGML_BACKEND_DL=OFF -DGGML_GEMMINI=OFF -DGGML_CUDA=OFF \
  -DGGML_METAL=ON -DGGML_METAL_QUANTIZED=ON -DGGML_METAL_USE_BF16=OFF \
  -DGGML_OPENMP=OFF -DGGML_GEMMINI_ENABLE_OPENMP=ON \
  -DOpenMP_ROOT="${OpenMP_ROOT:-/opt/homebrew/opt/libomp}" \
  -DGEMMINI_SW_PATH="${GEMMINI_SW_PATH:-$PWD/../RISC-V-DynDNN-gemmini-include}" \
  -DGGML_GEMMINI_ACTIVATION_BITS=4 -DGGML_GEMMINI_WEIGHT_BITS=4 \
  -DGGML_GEMMINI_ACTIVATION_QUANT=EXSIA -DGGML_GEMMINI_DIM=16 \
  -DGGML_GEMMINI_BLOCK_SIZE=32 -DGGML_GEMMINI_EXSIA_SIGMA=2 \
  -DGGML_GEMMINI_COMPUTE_TYPE=INT -DGGML_GEMMINI_ENABLE_RMD=ON \
  -DGGML_GEMMINI_OPTION=CPU -DGGML_GEMMINI_EXECUTION_BACKEND=HARDWARE \
  -DGGML_GEMMINI_DEFAULT_MATMUL_MODE=FULL \
  -DGGML_GEMMINI_ENABLE_STRIPE_MATMUL=OFF -DGGML_GEMMINI_ENABLE_STRIPE_PIPELINE=OFF \
  -DCYCLE_SIM=0 -DLOG_CYCLE=0 -DCYCLE_DETAIL=0
rtk proxy cmake --build .omo/metal-quantized/test-a4-exsia-d16 --target \
  test-metal-quantized test-metal-soft-f64 test-metal-weight-cache \
  test-metal-quantized-model benchmark-metal-quantized -j 4
rtk proxy ctest --test-dir .omo/metal-quantized/test-a4-exsia-d16 \
  --output-on-failure -R '^test-metal-(quantized|soft-f64|weight-cache)$'
```

`test-metal-quantized`는 정수 intermediate와 최종 출력·오류 처리를, `test-metal-soft-f64`는 CPU oracle에 대한 Metal soft-FP64 연산을, `test-metal-weight-cache`는 원본 byte 변경·수명·cache 상한을 검사한다. GPU가 없어 skip된 검사는 GPU PASS로 해석하지 않는다.

실제 모델 verifier는 profile과 일치하는 원본 GGUF로 prefill 뒤 decode 2개를 실행하고, 같은 token trajectory에서 각 quantized matmul과 logits를 독립 CPU oracle replay와 비교한다. CLI는 `--model`, `--prefill 8|16`, `--threads`를 받는다.

```bash
rtk proxy .omo/metal-quantized/test-a4-exsia-d16/bin/test-metal-quantized-model --help
rtk proxy .omo/metal-quantized/test-a4-exsia-d16/bin/test-metal-quantized-model \
  --model /path/to/gpt2.Q4_HP1.head-F16.gguf --prefill 16 --threads 4
rtk proxy .omo/metal-quantized/test-a4-exsia-d16/bin/benchmark-metal-quantized \
  --warmup 1 --repeats 5 --cpu-workers 4 --case-limit 1
```

benchmark의 네 옵션은 양의 정수를 받으며 `--case-limit`을 생략하면 전체 fixture를 실행한다. 현재 benchmark에 `--help` 옵션은 없다. 출력 JSON은 synthetic signed-code fixture의 CPU oracle 시간, Metal GPU stage 시간과 host 전체 시간을 제공한다. CPU activation producer와 최초 shader compile은 측정에서 제외하므로 실제 모델 pipeline 속도와 구분한다. 성능 측정 중에는 다른 GPU/model 작업을 함께 실행하지 않는다.

IM2P CPU_FUNCTIONAL provider와의 별도 비교는 선택 사항이다. `GGML_METAL_QUANTIZED_REFERENCE_ROOT`에 commit `bc6168c5ab3cc47edc716ff2c23402bd09867b8d`의 IM2P.sim checkout을 지정한다. 참조에 사용되는 파일이 수정되어 있으면 configure가 거부한다.

```bash
rtk proxy cmake -S . -B .omo/metal-quantized/test-a4-exsia-d16 \
  -DGGML_METAL_QUANTIZED_REFERENCE_ROOT=/path/to/IM2P.sim
rtk proxy cmake --build .omo/metal-quantized/test-a4-exsia-d16 \
  --target test-metal-provider-oracles -j 4
rtk proxy ctest --test-dir .omo/metal-quantized/test-a4-exsia-d16 \
  --output-on-failure -R '^test-metal-provider-oracle-'
```

provider target은 A4/A8 × DIM16/32/64의 여섯 profile을 생성한다. 이 참조 checkout은 검증용이며 일반 PPL runner의 필수 dependency는 아니다.

## 기록과 검증 범위

runner는 `GGML_GEMMINI=OFF`, `GGML_CUDA=OFF`, `GGML_METAL=ON`, `GGML_METAL_QUANTIZED=ON`을 명시한다. 테스트가 포함된 configure를 유지하며 실행 시 GPU layer 999를 요청하고 F16 KV cache, seed 42, warmup OFF를 사용한다. CPU PPL과 비교할 때는 동일 모델·dataset·context·batch·thread 설정을 맞춰야 한다.

결과 폴더의 `manifest.txt`, `model-metadata.json`, CMake cache, source diff와 hash, model/dataset/header/binary·library hash, compiler/host 정보로 입력과 설정을 확인한다. 실행 후 hash도 다시 검사한다. `ppl.log`에는 PPL과 `/usr/bin/time`의 process 시간이 포함된다. 이 시간은 모델 로딩, CPU producer, shared-memory staging, GPU 연산과 placement observer 비용을 모두 포함하며 kernel 시간과 같지 않다.

custom Metal PPL은 scheduler가 관찰한 quantized matmul의 tensor placement와 완료된 custom kernel 수를 `METAL_QUANTIZED_PROOF` JSON 한 줄로 출력한다. runner는 이를 `metal-quantized-proof.json`으로 보존하고 요청한 profile, `complete`, `placement_verified`, 실제 PPL 평가 token 수가 0보다 큼, 전체 matmul 수의 일치, failed/fallback 0을 요구한다. 증거 누락이나 CPU fallback은 실패 처리한다. 증거의 범위는 **해당 PPL 실행에서 관찰된 모든 quantized matmul**이며, 다른 executable이나 실행되지 않은 shape까지 확대하지 않는다. PPL은 decode coverage를 요구하지 않는다.

별도의 정확도 fixture와 `test-metal-quantized-model`의 동일 token trajectory 비교로 수치 계약을 검증한다. `llama-eval-workload`는 prefill/decode placement와 stage 시간을 함께 기록한다. producer·전송·dense·residual·merge 및 numerical/E2E 검증 결과는 평가 기록에서 확인한다.

실제로 완료한 검증과 측정의 범위는 [Metal 검증·측정 결과](metal-quantized-results.md)에 기록한다.
