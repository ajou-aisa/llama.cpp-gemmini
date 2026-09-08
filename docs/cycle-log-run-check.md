# Nano CPU cycle 로그의 실행 출처 확인

이 절차의 목적은 특정 commit의 코드가 **실제로 실행되어 새 로그를 만들었는지**
확인하는 것이다. Nano 실행과 로그 수집은 사용자가 수행한다.
`cpu_measurement_version=1`은 출력 계약의 표식이지 특정 commit 자체의 증명은 아니다.

## 소스와 성공한 빌드 고정

Nano의 저장소에서 전달받은 최종 commit을 checkout하고 Bash에서 다음을 함께 남긴다.

- `git rev-parse HEAD`, `git status --short`, 설명 가능한 source diff.
- 실제 Gemmini header의 경로·revision·SHA-256과 정확한 build 명령.
- 성공한 build의 `CMakeCache.txt` 및 compile definition.

빌드 중에는 source나 header를 변경하지 않는다. untracked 파일이 build 입력이면
그 내용을 별도로 보존하거나 먼저 추적 상태로 만들어야 한다.
설명되지 않은 입력이 있으면 commit과 바이너리가 일치한다고 판정하지 않는다.

아래는 **A8/W8/D64, CPU Dense/CPU_DIRECT residual, ExSIA PIPELINE** 예시다.
다른 실험의 width·route·모델을 이 값으로 임의 변경하지 않는다.
`RUN`은 매번 새 디렉터리이며 `BUILD`도 기존 바이너리가 없는 새 경로다.

```bash
REPO="$(pwd -P)"
mkdir -p "$REPO/output"
RUN="$(mktemp -d "$REPO/output/cycle-check.XXXXXX")"
BUILD="$RUN/build"
MODEL="$(readlink -f "$REPO/models/gpt2.Q8_HP1.gguf")"
git rev-parse HEAD >"$RUN/source-commit.txt"
git status --short >"$RUN/source-status.txt"
git diff --binary HEAD >"$RUN/source.patch"

build_cycle_image() {
  local target="$1"
  shift
  BUILD_DIR="$target" BUILD_JOBS=2 \
  LOG_DEBUG=1 LOG_CYCLE=1 GGML_CPU_CYCLE_LOG=1 CYCLE_DETAIL=1 \
  GGML_GEMMINI_EXSIA_PROFILE_SCOPE=STAGE \
  GGML_GEMMINI_ACTIVATION_BITS=8 GGML_GEMMINI_WEIGHT_BITS=8 \
  GGML_GEMMINI_DIM=64 GGML_GEMMINI_OPTION=CPU \
  GGML_GEMMINI_EXECUTION_BACKEND=HARDWARE \
  GGML_GEMMINI_ACTIVATION_QUANT=EXSIA \
  GGML_GEMMINI_DEFAULT_MATMUL_MODE=STRIPE_PIPELINE \
  GGML_GEMMINI_ENABLE_RMD=ON GGML_GEMMINI_DEFAULT_RMD_BACKEND=CPU \
  bash "$REPO/build-arm64.sh" "$@"
}
build_cycle_image "$BUILD" >"$RUN/build.log" 2>&1
BUILD_STATUS=$?
printf '%s\n' "$BUILD_STATUS" >"$RUN/build-exit.txt"
```

**BUILD_STATUS가 0인 경우에만 다음 단계로 간다.** 새 빌드가 실패했는데
기존 `llama-cli`를 실행하면 수정본의 증거가 아니다. 현재 로컬에서 확인된
`DenseMatmulStatus`/void header 불일치가 Nano에서도 발생하면 실패한
build.log를 남기고 중단한다. 이 절차는 해당 외부 API를 우회하지 않는다.
`BUILD_JOBS=2`는 wrapper의 macOS용 기본 job 탐색을 사용하지 않게 한다.

## 실제 실행 파일과 backend 연결

다음은 wrapper의 `GGML_BACKEND_DL=ON` 빌드를 대상으로 한다.
기존 실험의 prompt/token/thread 인자가 있다면 그 인자를 그대로 사용한다.
짧은 예제는 모든 route나 128행 stripe의 성능을 대표하지 않는다.

```bash
EXE="$BUILD/bin/llama-cli"
BACKEND="$BUILD/bin/libggml-gemmini.so"
test -x "$EXE" && test -f "$BACKEND" && test -r "$MODEL"
sha256sum "$EXE" "$BUILD"/bin/*.so* \
  "$BUILD/CMakeCache.txt" "$MODEL" >"$RUN/artifacts.sha256"
env -u LD_PRELOAD LD_LIBRARY_PATH="$BUILD/bin" ldd "$EXE" >"$RUN/executable.ldd"
env -u LD_PRELOAD LD_LIBRARY_PATH="$BUILD/bin" ldd "$BACKEND" >"$RUN/backend.ldd"
printf '%s\n' "$RUN"
```

`ldd` 결과에서 project library가 다른 build로 연결되지 않는지 확인한다.
실행 중에는 바이너리나 라이브러리를 다시 빌드·교체하지 않는다.

```bash
(
  cd "$BUILD/bin" || exit 1
  unset GGML_BACKEND_PATH LD_PRELOAD
  export LD_LIBRARY_PATH="$BUILD/bin"
  export GEMMINI_LOG_DIR="$RUN"
  export GGML_GEMMINI_CYCLE_DETAIL_LOG="$RUN/exsia-detail.jsonl"
  export GGML_GEMMINI_TELEMETRY_HASH=0
  printf '%q ' "$EXE" -m "$MODEL" -p "Hello" -n 16 -t 3 \
    --gemmini-debug-log "$RUN/debug.jsonl" \
    --gemmini-cycle-log "$RUN/cycles.jsonl"
  printf '\n'
  "$EXE" -m "$MODEL" -p "Hello" -n 16 -t 3 \
    --gemmini-debug-log "$RUN/debug.jsonl" \
    --gemmini-cycle-log "$RUN/cycles.jsonl" \
    >"$RUN/stdout.txt" 2>"$RUN/stderr.txt"
  STATUS=$?
  printf '%s\n' "$STATUS" >"$RUN/exit-status.txt"
  exit "$STATUS"
) >"$RUN/command.txt"
sha256sum -c "$RUN/artifacts.sha256" >"$RUN/artifacts-check.txt"
printf '%s\n' "$?" >"$RUN/artifacts-check-exit.txt"
```

`--log-disable`는 사용하지 않는다. 일반 stderr에 나오는 다음 성공 메시지의
경로가 위에서 hash한 backend의 실제 경로와 일치해야 한다.

```text
load_backend: loaded GEMMINI backend from /.../build/bin/libggml-gemmini.so
```

이 메시지는 library 초기화와 API 검사가 성공한 뒤 출력된다.
단순 Git HEAD나 CLI 파일 hash만으로는 다른 backend의 로드를 배제할 수 없다.

**두 로그를 같은 파일로 지정하지 않는다.** main cycle sink와 ExSIA detail
sink는 각각 처음 열 때 파일을 truncate할 수 있다. raw 파일 둘을 보존한 뒤
각각 분석한다. 기존 matrix runner의 manifest만으로는 loaded-backend 출처까지
증명되지 않는다.

`GGML_BACKEND_DL=OFF`는 linked registration을 뜻하며 반드시 정적 링크는 아니다.
그 경우 dynamic-load 메시지를 요구하지 않고, `ldd`/link command로 확인한 실제
Gemmini library 또는 정적 링크 실행 파일·archive의 provenance를 보존한다.

## 새 로그의 수락 기준

먼저 프로세스 exit 0과 artifact hash 일치를 확인하고 다음을 검사한다.

```bash
python3 scripts/utils/cycle_schema.py --check "$RUN/cycles.jsonl"
python3 scripts/utils/cycle_schema.py --check "$RUN/exsia-detail.jsonl"
```

- RMD 레코드가 발생한 route에서는 `cpu_measurement_version=1`과 timing별
  nullable 값, valid/reason/sample_reason/count가 있어야 한다.
  이 표식은 LOG_CYCLE 출력에 항상 있으며 DETAIL/scope/hash 옵션과 무관하다.
  모든 record type에 표식을 요구하지는 않는다.
- CPU PMU record의 source/unit은 `linux_perf_cpu_cycles/cycle`이어야 한다.
  invalid/null과 사유는 보존하되, 실행한 필수 CPU 작업이 모두 invalid 또는
  미수집이면 측정 완료로 판정하지 않는다. 정상적인 typed zero는 실패와 구별한다.
- ExSIA·PIPELINE의 실제 run/layer/stripe와 J-tile worker/node identity를 확인한다.
  run 0는 유효하다. non-ExSIA에 원래 없는 run이나 J-tile의 slot을 만들지 않는다.
  Tensor/Token/Stripe builder finish에는 실제 layer와 null run이 기대된다.
- 실행한 경로의 `stripe_input_capture`, `stripe_job_preparation`,
  `dense_backend_host_call`, `stripe_completion_bookkeeping` 등을 확인한다.
  FULL, empty residual, CPU_DIRECT에서 미실행되는 단계를 필수로 요구하지 않는다.
- residual이 실행되면 builder·Merge·통계 기록을 확인한다.
  CPU_DIRECT에는 packet Compose가 필요 없다.
- Hash OFF에서는 hash record가 없고 해당 stripe의 `hash_enabled`가 false여야 한다.
  Hash ON 비교는 새 RUN에서 CYCLE_DETAIL=1과
  `GGML_GEMMINI_TELEMETRY_HASH=1`을 함께 사용한다.
- 성공 sample만 있는 실제 로그로 실패 처리까지 검증했다고 하지 않는다.

## Nano의 Native 회귀 테스트

wrapper는 기본적으로 tests OFF이며 dynamic backend를 만든다. 위에서 hash한
추론 build를 재설정하지 말고, 같은 source/header/width/route로 **별도 새 test
build**를 만든다. tests ON뿐 아니라 backend-DL OFF도 필요하다.

```bash
TEST_BUILD="$RUN/native-tests"
build_cycle_image "$TEST_BUILD" -DLLAMA_BUILD_TESTS=ON -DGGML_BACKEND_DL=OFF \
  >"$RUN/native-test-build.log" 2>&1
TEST_BUILD_STATUS=$?
printf '%s\n' "$TEST_BUILD_STATUS" >"$RUN/native-test-build-exit.txt"
```

이 상태가 0일 때만 실제 test target을 빌드한다.

```bash
cmake --build "$TEST_BUILD" --target test-gemmini-cycle-reader-aarch64 \
  test-gemmini-log-boundary test-gemmini-rmd-reducer test-gemmini-exsia -j2 \
  >"$RUN/native-test-targets.log" 2>&1
TEST_TARGET_STATUS=$?
printf '%s\n' "$TEST_TARGET_STATUS" >"$RUN/native-test-targets-exit.txt"
```

target build도 0인 경우에만 다음 CTest를 실행한다.
`gemmini_exsia_non_exsia_capture_context`는 build target이 아니라 CTest 이름이다.

```bash
ctest --test-dir "$TEST_BUILD" --output-on-failure \
  -R '^(test-gemmini-cycle-reader-aarch64|test-gemmini-log-boundary(-linux-scalar)?|test-gemmini-rmd-reducer|gemmini_exsia_non_exsia_capture_context)$' \
  >"$RUN/native-tests.log" 2>&1
printf '%s\n' "$?" >"$RUN/native-tests-exit.txt"
```

`--linux-aarch64-scalar`는 scalar 정책 검사이며 native PMU 실행의 대체물이 아니다.
기본 reader 테스트도 결정적 입력 검사가 중심이다. 실제 PMU 접근·worker 측정은
별도 physical 모드의 exit status와 출력으로 확인한다.

```bash
"$TEST_BUILD/bin/test-gemmini-cycle-reader-aarch64" --physical \
  >"$RUN/native-pmu.log" 2>&1
printf '%s\n' "$?" >"$RUN/native-pmu-exit.txt"
```

실패 또는 unavailable 결과를 fixture PASS로 대신하지 않는다.

현재 builder-finish raw JSON assertion은 Linux-AArch64/LOG_CYCLE/CYCLE_DETAIL에서
실행된다. Mac의 해당 quantizer 테스트는 실제 residual 생성과 source contract를
검증하지만 Nano의 emitted builder record를 검증한 것은 아니다.

## Finalize 비용의 후속 판정

동일 실행의 Merge·통계·hash child 기록을 먼저 확인한다.
여전히 큰 차이가 있으면 이름은 **Finalize 내부 미분리 비용**으로 유지한다.
이를 cleanup이나 slot 반환 비용으로 바꾸어 부르지 않는다.
slot 반환과 Finalize의 JSON 출력은 native Finalize 종료 endpoint 밖에 있다.
필요할 때만 실제 bookkeeping 경계에 CPU pair를 추가한다.
NPU cycle은 기존 IM2P 측정을 따르며 이 절차의 새 구현 대상이 아니다.
