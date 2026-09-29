# RMD compaction 채택 및 비교

같은 빌드에서 `GGML_GEMMINI_RMD_COMPACTION`만 바꿔 실행한다.

| 값 | ExSIA의 WS packet 생성 |
| --- | --- |
| `legacy` | 기존 `RmdStripeBuilder` |
| `bitmap` | nonzero 위치 bitmap, limb 생성 중 row/K 기록, row 목록 크기 사전 확보 |

A4/A8의 기본값은 `bitmap`. A16은 기존 `legacy`를 유지하며 `bitmap`을
명시하면 실패한다. 알 수 없는 옵션도 실패한다. CPU direct 경로에는
compaction이 없으므로 이 옵션으로 성능을 비교할 수 없다.

## 채택 근거: Jetson Nano CPU 준비 실험

A4/W4 RMD의 WS packet compaction 방법으로 `bitmap`을 채택한다. Nano에서
CPU 추론을 유지하면서 동일 residual로 WS packet까지 만드는 임시 측정 코드
(`de530e2a0877c1ef43d5215eaf1c34004b43800d`)를 사용했다. GPT-2
`Q4_HP1`에서 `legacy`와 `bitmap`을 번갈아 3회씩, 두 묶음으로 실행했다.
모든 성공 실행은 같은 Git SHA와 모델을 사용했고, 각각 384개 stripe를 기록했다.
모델 경로 오류로 종료된 최초 1회는 제외했다. 원본 로그 묶음은
`rmd-cpu-results-v2.tar.gz`이다.

| 실행당 384개 stripe 시간 합계의 3회 중앙값 | 첫 묶음 legacy → bitmap | 둘째 묶음 legacy → bitmap |
| --- | ---: | ---: |
| `exsia.folding_and_pack` host 시간 | 138.2 → 113.1 ms (18.2% 감소) | 149.7 → 117.7 ms (21.4% 감소) |
| `rmd.cpu_compaction.finish` host 시간 | 44.4 → 34.1 ms (23.1% 감소) | 46.9 → 35.1 ms (25.3% 감소) |

두 묶음의 성공 실행 12개에서 생성 출력과 stripe별 packet 작업량이 같았다.
`exsia.folding_and_pack`에는 folding, CPU direct capture, packet 생성·검증이
함께 들어간다. `finish`는 그 구간에 포함되므로 두 시간을 더하지 않는다.
전체 추론은 양쪽 모두 약 32~33초였다. Nano의 native CPU cycle event는
`unavailable_event`로 무효였고, 이 CPU 실험에서는 RTL/SCU cycle을
측정하지 않았다. 따라서 채택 근거는 host 준비 시간 감소이며 하드웨어
cycle 감소는 아직 확인되지 않았다.

## 알고리즘과 동일성

ExSIA folding에서 선택된 residual을 전달받는 즉시 balanced radix limb로
나눈다. 0이 아닌 digit만 저장하면서 원래 limb 번호, row/K support를 함께
기록한다. residual별 position 복사 대신 nonzero bitmap을 사용한다.
마지막에 bitmap을 읽어 기존과 같은 group/row/K 배치의 signed-byte packet을 만든다.
최종 GEMM payload에는 기존과 같은 padding zero가 들어간다.

기존 builder도 residual을 받을 때 limb를 분해했다. 이번 변경의 차이는
중간 metadata와 compaction 순회다. 원본 weight block 식별자, limb 지수,
HP1 SCU scale, 최종 packet ABI는 유지한다. 일반 quantizer는 기존 builder를 쓴다.

## 서버 실행

먼저 이 브랜치의 코드를 서버에서 다시 빌드한다. 이미 동작하는 HP1
Gemmini(Chisel) backend 설정과 외부 산출물을 사용한다. 다음 값은 확인한다.

```text
GGML_GEMMINI_OPTION=WS
GGML_GEMMINI_DEFAULT_RMD_BACKEND=WS
GGML_GEMMINI_ACTIVATION_QUANT=EXSIA
GGML_GEMMINI_ENABLE_RMD=ON
GGML_GEMMINI_DEFAULT_MATMUL_MODE=STRIPE_PIPELINE
LOG_CYCLE=1
CYCLE_DETAIL=1
GGML_GEMMINI_EXSIA_PROFILE_SCOPE=STAGE
```

Activation/weight 폭은 A4/W4 또는 A8/W8로 맞춘다. 시뮬레이터라면
`GGML_GEMMINI_EXECUTION_BACKEND=IM2P_SIM`,
`IM2P_SIM_IMPLEMENTATION=GEMMINI_HP1`이다.
이 작업은 외부 IM2P.sim 소스를 변경하지 않는다.

아래 `build-server`, 모델 경로, 입력은 서버의 실제 값으로 바꾼다.
두 실행 사이에는 compaction 옵션만 바꾼다.

```bash
bash scripts/experiment/run-with-output.sh rmd-legacy -- \
  env GGML_GEMMINI_RMD_COMPACTION=legacy \
  ./build-server/bin/llama-cli -m models/gpt2.Q4_HP1.gguf \
  --device GEMMINI -ngl 99 --tensor-split 1 -t 1 -tb 1 \
  -c 256 -b 64 -ub 64 -n 8 --seed 42 --temp 0 --no-warmup \
  --no-conversation --simple-io --no-display-prompt \
  -p 'The history of computing began with simple machines for arithmetic.'

bash scripts/experiment/run-with-output.sh rmd-bitmap -- \
  env GGML_GEMMINI_RMD_COMPACTION=bitmap \
  ./build-server/bin/llama-cli -m models/gpt2.Q4_HP1.gguf \
  --device GEMMINI -ngl 99 --tensor-split 1 -t 1 -tb 1 \
  -c 256 -b 64 -ub 64 -n 8 --seed 42 --temp 0 --no-warmup \
  --no-conversation --simple-io --no-display-prompt \
  -p 'The history of computing began with simple machines for arithmetic.'
```

결과는 `output/experiment/rmd-{legacy,bitmap}-<UTC>/`에 남는다.

- `manifest.txt`: 실행 명령(선택한 compaction 포함), Git SHA, 종료 코드.
- `raw/stderr.txt`: 각각 `capture=ws_packet compaction=legacy` 또는
  `capture=ws_packet compaction=bitmap` 확인. `cpu_direct (compaction unused)`면
  packet compaction을 비교한 실행이 아니다.
- `raw/stdout.txt`: 생성 결과 비교.
- `raw/cycle-log.jsonl`, `raw/exsia-cycle-detail.jsonl`: 활성화된 cycle 기록.

## 무엇을 비교하나

1. 동일 모델·입력·seed·thread·warmup 정책으로 반복한다. 실행 순서도
   legacy→bitmap / bitmap→legacy로 번갈아 측정한다.
2. 결과가 같고 두 실행 모두 성공했는지 먼저 확인한다. RMD 호출이 실제로
   발생했는지도 확인한다. residual이 없는 입력은 유효한 compaction 비교가 아니다.
3. Host 준비 비용과 Gemmini hardware cycle을 분리한다. ExSIA folding에는
   residual 전달/limb 생성 비용이 포함된다. `rmd_packet_finish_cycles`와
   `capture_ns`는 **finish만** 측정하므로 전체 compaction 비용이 아니다.
4. `rmd_packet_finish_cycles`는 Linux AArch64의 유효한 native CPU counter
   기록이다. Gemmini array cycle과 합치지 않는다. counter가 유효하지 않으면
   측정 불가로 남기고 0 cycle이나 추측값으로 해석하지 않는다.
5. 두 알고리즘은 같은 packet을 만든다. 요청하는 GEMM 작업량도 같으므로
   array 연산 cycle 감소를 미리 주장하지 않는다. 줄어드는 대상은 host 준비
   비용이며, 전체 실행 시간과 hardware 대기 영향은 서버에서 따로 확인한다.

## 로컬 검증

`test-gemmini-rmd-bitmap-log0/log1`은 A4/A8 packet 전체 동일성과 원본 정수
복원, 경계값, mask 수명, 잘못된 입력 거부를 검사한다.
`gemmini_exsia_packet_legacy/bitmap`은 실제 ExSIA를 WS packet 경로로 실행하고
packet을 복원해 ExSIA residual plane과 비교한다. CPU에서도 실행 가능하며
hardware cycle 검증을 대신하지 않는다.

```bash
ctest --test-dir build-server --output-on-failure \
  -R 'test-gemmini-rmd-bitmap-log[01]|gemmini_exsia_packet_(legacy|bitmap)'
```

2026-09-29 로컬 확인: A4/A8 ExSIA 전체 검사와 두 WS packet 경로 통과,
packet 동일성 LOG_CYCLE=0/1 통과, AddressSanitizer/UndefinedBehaviorSanitizer
통과. CPU direct GPT-2 Q4_HP1 실행은 두 옵션에서 같은 출력을 냈으며
`compaction unused`를 기록했다. Hardware cycle 감소는 아직 측정하지 않았다.
