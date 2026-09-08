# Host overlap 측정

Matmul CPU 구간과 ExSIA TIMELINE 레코드의 `host_timing`은 같은 레코드의 cycle 측정에 붙는
sidecar다. 별도 join 파일 없이 `op`, `run_id`, `layer`, `stripe_id`, `worker_id`와
함께 읽는다. 기존 로그에 이 객체가 없으면 ns를 복원하거나 cycle에서 환산하지 않는다.

측정 빌드는 기존 width/backend/route를 유지하며 `LOG_CYCLE=1`, `CYCLE_DETAIL=1`,
`GGML_GEMMINI_EXSIA_PROFILE_SCOPE=STAGE`를 켠다. 빌드·실행 파일 확인 절차는
[cycle-log-run-check.md](cycle-log-run-check.md)를 따른다.

```bash
export GGML_GEMMINI_CYCLE_DETAIL_LOG="$RUN/exsia-detail.jsonl"
"$EXE" -m "$MODEL" -p "Hello" -n 16 -t 3 \
  --gemmini-cycle-log "$RUN/cycles.jsonl"

python3 scripts/utils/summarize_host_overlap.py \
  "$RUN/cycles.jsonl" "$RUN/exsia-detail.jsonl" \
  --left-op exsia.local_group \
  --right-op dense_backend_host_call \
  --right-op residual_backend_host_call >"$RUN/host-overlap.json"
```

`RUN`은 실행마다 새 디렉터리여야 한다. main과 detail 경로는 서로 다르게 둔다.
두 sink가 파일을 각각 초기화하므로 같은 경로를 공유하면 로그가 유실될 수 있다.

필터는 실제 `op` 이름을 정확히 지정하며 여러 번 지정하면 구간들의 합집합이다.
LA worker는 `exsia.local_group`, SF는 `exsia.folding`을 선택한다.
`exsia.local`과 `exsia.stripe_total`은 여러 작업을 감싼 구간이므로 세부 worker의
실행 시간과 구별한다. `exsia.stage_metric`은 누적 수치이며 구간으로 읽지 않는다.

분석 결과는 `(execution_id, run_id, layer)`마다 왼쪽·오른쪽 구간 수,
각 합집합 길이와 그 교집합인 `overlap_ns`를 정수 ns로 출력한다.
`execution_id`는 공용 로깅 라이브러리가 PID와 초기화 시각으로 자동 생성한다.
같은 프로세스의 main/detail 로그가 이 식별자를 공유하며, 사용자가 따로 지정하지 않는다.
stripe가 달라도 같은 그룹이면 비교하므로 pipeline의 stripe 간 overlap이 포함된다.
중첩 worker나 중복 레코드는 합집합 처리하여 시간을 중복 합산하지 않는다.
끝점만 닿는 구간의 overlap은 0이다.

서로 다른 실행·run·layer는 비교하지 않는다. 한쪽만 있는 그룹은
`status="missing_side", overlap_ns=null`이며 overlap 0을 관측했다는 뜻이 아니다.
식별자가 없거나 null이면 `sources[].skipped.unknown_identity`에 세고 제외한다.
run 0은 유효하다. 파일별 `left_matches/right_matches`, `left_accepted/right_accepted`와
skip 이유를 함께 확인해야 필터가 실제 측정 구간을 선택했는지 알 수 있다.
서로 다른 clock/unit 또는 잘못된 JSON은 파일명·행 번호와 함께 오류 처리한다.

`host_timing.valid`는 외부 PMU의 `valid`와 독립적이다. PMU가 실패했어도 유효한
ns 구간은 분석한다. `clock="steady_clock"`, `unit="nanosecond"`,
`start_ns/end_ns`가 공통 시간축을 정의한다. 그래프로 그릴 때도 실행마다 공통 시작값
하나만 빼고 worker별로 원점을 옮기지 않는다. `duration_ns`는 구간 길이이며
실제 교집합 계산에는 두 endpoint를 사용한다.

`worker_id`는 논리적 작업 분할 번호다. `start_tid/end_tid`는 endpoint를 기록한
실행 스레드를 `thread_id_kind`에 따라 식별한다. ExSIA의 기존
`start_thread_id/end_thread_id`는 OpenMP team 내 스레드 번호이며 OS TID와 다르다.
summary의 `host_stages`는 단계별 구간 지도이며 이 분석기는 atomic
`host_timing` 레코드만 사용한다.
`host_stages.queue`는 실제 enqueue부터 dequeue까지이며 이후 작업 준비 시간은 제외한다.
`dense`와 `residual_backend`는 backend 호출 경계다. 기존 `rmd_start_ns/end_ns`는
준비·후처리까지 포함하는 더 넓은 범위이므로 Residual 호출 시간과 구별한다.

이 수치는 **host 구간이 시간상 겹친 길이**다. CPU가 동시에 on-core로 실행됐는지는
스케줄러 추적, NPU가 동시에 연산했는지는 장치 실행 경계가 필요하다.
timestamp와 TID 수집, 로그 출력 자체에도 비용이 있으므로 상세 계측을 켠 실행과
기본 성능 실행을 구별한다.

회귀 확인: `python3 tests/test-host-overlap.py`.
