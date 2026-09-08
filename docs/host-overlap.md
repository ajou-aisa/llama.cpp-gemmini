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

## CPU_DIRECT residual 내부 시간

`LOG_CYCLE=1`, `CYCLE_DETAIL=1` 빌드에서는 CPU_DIRECT 실행 뒤 main cycle 파일에
`RESIDUAL_HOST_PROFILE`이 추가된다. 기존 빌드를 다시 빌드하고 같은 모델·입력·thread
설정으로 실행해야 새 항목이 생긴다. 이전 로그에서 세부 시간을 복원할 수는 없다.
ExSIA의 STAGE 설정은 LA/SF 관측용이며 이 residual 계측 자체는 PMU 지원과 독립적이다.

```bash
python3 scripts/utils/render_residual_profile.py "$RUN/cycles.jsonl" \
  --output-dir "$RUN/residual-profile" \
  --run-id 83 --layer blk.8.mlp.up_proj
```

`run-id`와 `layer`는 새 실행에서 비교할 호출로 바꾼다. 필터를 생략하면 파일의
모든 residual profile을 출력하며, `--stripe-id`로 한 stripe만 선택할 수 있다.
`summary.md`에 처리량·단계별 시간, `workers.csv`와 `tiles.csv`에 정수 ns,
`worker-timeline.svg`에 worker별 계산·기존 tile 로그·barrier 구간이 나온다.
SVG는 브라우저에서 열어 확대할 수 있다.

한 profile은 다음을 담는다.

- `workload`: stripe 행 범위, J/K, residual `event_count`, `active_rows`,
  `active_row_blocks`, `j_tile_count`. 활성 block은 event가 있는 `(local_row, K/32)` 쌍이다.
- `phases`: 검증, 준비·버퍼 초기화, 병렬 실행, 결과 마무리의 host/스레드 CPU 시간.
- `tiles`: 실제 실행 worker와 출력 열 범위, tile 본체의 host/스레드 CPU 시간,
  기존 tile 로그 호출의 시간 및 로거 mutex 대기·write/flush 누계.
- `workers`: 각 worker의 작업 구간과 barrier 도착부터 통과까지의 별도 구간.

`host_timing`은 기존과 같은 execution ID와 `steady_clock` 시간축을 쓴다.
`thread_cpu_timing`은 같은 OS 스레드의 CPU 사용시간 차이이며,
`CLOCK_THREAD_CPUTIME_ID`를 사용할 수 없거나 endpoint가 유효하지 않으면 null/invalid다.
스레드 CPU 시간은 공통 시간축이 아니므로 그래프 정렬이나 worker 간 overlap 계산에 쓰지 않는다.
host 경과시간과 CPU 사용시간의 차이만으로 mutex 대기와 OS 실행 대기를 구분할 수는 없다.
barrier에서 spin하는 시간은 CPU 사용시간에도 포함될 수 있다.

기존 Linux/AArch64의 tile cycle 로그는 원래 위치에서 출력하여 그 영향을 측정한다.
`log_mutex_wait_ns`는 로거 잠금 획득 구간, `log_io_ns`는 `fwrite`와 `fflush` 구간이다.
JSON 생성 등은 전체 `log_host_timing`에는 들어갈 수 있지만 이 두 누계에는 들어가지 않는다.
로그가 실행되지 않았거나 실패한 측정은 `log_valid=false`이며 0ns 관측으로 해석하지 않는다.

새 profile은 worker별 메모리에 모았다가 병렬 실행이 끝난 뒤 한 번 출력한다.
profile 자체의 JSON 생성·출력은 자신의 최상위 `host_timing`에서 제외되지만,
호출자를 감싼 기존 `residual_backend_host_call`에는 포함된다.
새 기록의 총시간을 이전 실행의 458.6ms와 같은 계측 비용이라고 가정하지 않는다.
worker들의 시간 합계도 병렬 구간의 경과시간과 직접 더하거나 빼지 않는다.

로컬에서 실제 executor의 출력과 분석 연결을 확인하려면 tests를 켠 빌드로 다음을 실행한다.

```bash
"$TEST_BUILD/bin/test-gemmini-exsia" \
  --direct-host-profile-output "$RUN/residual-fixture.jsonl"
python3 scripts/utils/render_residual_profile.py "$RUN/residual-fixture.jsonl" \
  --output-dir "$RUN/residual-fixture-report"
```

이 작은 fixture의 시간은 Nano 모델 실행 성능을 대신하지 않는다.
회귀 확인: `python3 tests/test-residual-profile.py`.

## J-tile 내부 세부 계측과 다른 출력

위 상세 빌드에서 실행할 때 `GGML_GEMMINI_RESIDUAL_DEEP_PROFILE=1`을 추가한다.
기존 모델·prompt·thread·backend 설정은 그대로 둔다.

```bash
export GGML_GEMMINI_RESIDUAL_DEEP_PROFILE=1
# 위의 실제 모델 실행 명령으로 새 cycles.jsonl을 수집한 뒤:
python3 scripts/utils/render_residual_profile.py "$RUN/cycles.jsonl" \
  --output-dir "$RUN/residual-profile"
```

새 profile의 `deep_profile=true`, 각 tile의 `stages`에서 다음 누계를 읽는다.
각 단계는 `(row, K/32 block)`마다 측정하고 J-tile별로 합쳐 저장한다.

| 단계 | 실제 측정하는 작업 |
| --- | --- |
| `event_scan` | 같은 row/block에 속하는 residual event 구간 탐색 |
| `weight_dot` | weight code 읽기·검사와 residual 곱셈·누적 |
| `scale_apply` | block scale 읽기·검사, scale 곱셈, overflow 검사, 출력 버퍼 누적 |

각 단계에 `calls`, `wall_ns`, `thread_cpu_ns`, `cycles`, `cycles_valid`,
`cycles_reason`이 나온다. cycles는 Linux/AArch64의 기존 native PMU reader로
같은 event owner/generation인지 검사한 값이다. 지원하지 않는 플랫폼이나 실패한
측정은 null과 사유를 기록한다. Mac timer tick을 CPU cycle로 대신 넣지 않는다.
ns와 cycles의 읽기 경계는 완전히 같은 시점이 아니며 서로 환산하지 않는다.

weight 읽기와 곱셈은 원래 한 루프에 붙어 있으므로 `weight_dot`으로 함께 측정한다.
명령 하나마다 타이머를 넣거나 weight를 따로 버퍼링하여 원래 계산 순서를 바꾸지 않는다.
각 span의 probe 비용과 span 사이의 제어·집계 비용 때문에 단계 합계가 tile 전체와
같지는 않다. worker들의 누적 시간은 병렬 실행의 경과시간과 구별한다.
추가 계측이 부담을 주므로 기본값은 OFF이며, 같은 조건에서 OFF/ON 반복 실행을 비교한다.

분석기는 다음 파일도 출력한다.

- `stages.csv`: run/stripe/worker/J-tile별 내부 단계의 정수 ns·cycles와 유효성.
- `summary.md`: stripe·worker별 내부 단계 누계. 미수집 값은 0으로 채우지 않는다.
- `trace.json`: [Perfetto UI](https://ui.perfetto.dev/)에서 Open trace file로 여는
  타임라인. 부모 단계·worker 계산·로그·barrier의 실제 host endpoint를 사용한다.
  내부 단계 누계에는 개별 endpoint가 없으므로 타임라인 위치를 추정해 넣지 않는다.

Chrome JSON의 `ts/dur` 단위는 microsecond이며 실행별 공통 ns 원점을 뺀 뒤 변환한다.
원래 정수 ns는 CSV에 보존한다. 서로 다른 execution ID의 시작점은 별도 기준이므로
그 사이의 overlap을 해석하지 않는다.
[Perfetto 외부 trace 형식 설명](https://perfetto.dev/docs/getting-started/other-formats).

기존 cycle 로그 전체의 단계와 worker별 cycles는 원래 도구로도 볼 수 있다.
중첩된 부모·자식 op의 cycles를 합쳐 총시간으로 해석하지 않는다.

```bash
python3 scripts/utils/render_operation_cycles.py "$RUN/cycles.jsonl" "$RUN/operations"
python3 scripts/utils/render_worker_cycles.py "$RUN/cycles.jsonl" "$RUN/workers" \
  --op rmd_direct_j_tile_interval
python3 scripts/utils/cycle_log_to_csv.py "$RUN/cycles.jsonl" "$RUN/csv"
```

## OS에서 독립적으로 측정하기

Linux에서는 같은 모델 명령을 `perf` 뒤에 두면 새 OpenMP worker도 수집한다.
아래 `PROGRAM ARGS`를 실제 실행 파일과 인자로 바꾼다. 각 명령은 별도 실행이므로
cycle 로그 경로도 실행마다 새로 지정한다. baseline 비교는 deep profile을 끄고 한다.

```bash
perf stat -o "$RUN/perf-stat.txt" \
  -e task-clock,context-switches,cpu-migrations,page-faults,cycles:u,instructions:u \
  -- PROGRAM ARGS

perf record -o "$RUN/perf.data" -F 99 -e cycles:u \
  --call-graph dwarf -- PROGRAM ARGS
perf report --stdio --no-children --sort pid,dso,symbol -i "$RUN/perf.data" \
  > "$RUN/perf-report.txt"
perf script -i "$RUN/perf.data" -F comm,pid,tid,time,event,ip,sym,dso \
  > "$RUN/perf-stacks.txt"
```

`perf stat`은 전체 실행의 CPU 사용량·문맥 전환·migration·page fault를 보여주고,
`perf record/report`는 어느 TID·함수의 스택이 자주 관측됐는지 보여준다.
stripe별 구분은 프로그램의 profile을 사용한다. `cycles:u`와 `instructions:u`는
user-space이며, PMU 권한 실패나 multiplexing의 실행 비율을 먼저 확인한다.
PMU를 쓸 수 없으면 별도 실행에서 software event만 선택할 수 있다.
이 경우 cycles를 측정했다고 표시하지 않는다. 함수·소스 줄을 자세히 보려면
debug symbol과 unwind 정보가 있는 빌드가 필요하다.
[perf stat](https://raw.githubusercontent.com/torvalds/linux/master/tools/perf/Documentation/perf-stat.txt),
[perf record](https://raw.githubusercontent.com/torvalds/linux/master/tools/perf/Documentation/perf-record.txt).

Mac에서는 모델 실행이 시작되어 첫 profile이 기록된 것을 확인한 뒤
실행 중인 프로세스의 PID로 `sample`을 사용한다.

```bash
/usr/bin/sample "$PID" 2 1 -file "$RUN/sample.txt"
```

이것은 전체 스레드의 스택을 대기 상태까지 포함하여 샘플링한 결과다.
샘플 수를 정확한 CPU ns나 cycles로 읽지 않는다. 종료 코드뿐 아니라 보고서의
`Call graph`에 실제 Thread 항목과 스택이 있는지 확인한다. 짧은 fixture를 관찰할 때는
`--direct-host-profile-repeat N`으로 동일 executor를 반복 실행할 수 있다.
반복 fixture의 JSON 출력 비용이나 Mac 결과를 Nano 모델의 병목으로 결론내리지 않는다.
