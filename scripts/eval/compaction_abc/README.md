# Residual compaction A/B/C replay

같은 실제 inference residual과 원본 HP1 weight로 압축 효과와 호스트 준비 비용을 비교한다. 결과는 residual 경로의 실측 CPU 작업량과 추정 NPU 작업량이다.

| 경로 | 생존 limb 행 선택 | K 선택 | 호스트 구현 |
|---|---|---|---|
| A | 적용 | 원본 K32 block별 nonzero K union | 현재 `RmdBitmapBuilder`와 `build_run_aware_request` 실측 |
| A* | 적용 | A와 동일 | B/C와 같은 단순 직접 pack 구현, A와 전 원소 배열 대조 |
| B | 적용 | 원본 K 전체 | 직접 pack, 원본 weight K32 경계 유지 |
| C | 미적용 | 원본 K 전체 | 원래 limb/행 위치에 scatter, 원본 weight K32 경계 유지 |

C의 원래 행 수는 `required_limb_count × source_stripe_rows`다. INT32를 담을 수 있는 최대 9/5개 plane을 무조건 실행하는 조건은 아니다. 세 경로에서 residual, required limb count, source stripe와 weight가 같다. 비어 있는 residual stripe는 모두 실행하지 않는다.

## 실행

이 저장소에는 `IM2P.sim`의 최신 cycle source가 sibling worktree `im2p-gemmini-relative`에 있다. 다른 경로라면 `IM2P_CYCLE_ROOT`를 지정한다. 수집용 소스 복사본과 빌드는 `.omo/compaction-abc/`에 만들며 production 파일을 수정하지 않는다.

```bash
rtk proxy bash scripts/eval/compaction_abc/prepare.sh
rtk proxy bash scripts/eval/compaction_abc/run-replay.sh output/experiment/compaction-abc-20261004
rtk proxy bash scripts/eval/compaction_abc/run-cached.sh output/experiment/compaction-abc-20261004
rtk proxy uv run scripts/eval/compaction_abc/fill_cycles.py \
  output/experiment/compaction-abc-20261004 \
  .omo/compaction-abc/capture-build/bin/compaction-cycle
rtk proxy uv run scripts/eval/compaction_abc/summarize.py output/experiment/compaction-abc-20261004
rtk proxy uv run scripts/eval/compaction_abc/summarize_cached.py output/experiment/compaction-abc-20261004
rtk proxy c++ -std=c++20 -O3 -Wall -Wextra -Werror \
  scripts/eval/compaction_abc/bitmap.cpp -o .omo/compaction-abc/compaction-bitmap
rtk proxy .omo/compaction-abc/compaction-bitmap \
  output/experiment/compaction-abc-20261004 \
  output/experiment/compaction-abc-20261004/bitmap-cost.csv
```

`run-replay.sh`는 보관된 네 dump(`gpt2`, `llama`, `gpt2-a8`, `llama-a8`)를 사용한다. 정확한 수집 모델·입력·명령·hash는 결과 디렉터리의 provenance에 보관한다. 새 수집에는 `GGML_GEMMINI_ABLATION_DUMP`를 기존의 빈 디렉터리로 지정한 뒤 수집 빌드의 `llama-cli`를 사용한다. `.rbin`은 원래 stripe의 sparse INT32 residual, `.wbin`은 그 layer의 원래 HP1 packed weight다. 이 포맷은 little-endian evaluation fixture이며 packet ABI가 아니다.

`label_phases.py`는 single-thread sequential capture의 case 순서에서 transformer block이 마지막 layer에서 block 0으로 돌아가는 지점으로 네 inference pass를 구분한다. 마지막 prefill FFN의 `graph_m=1`도 prefill에 포함한다. Binary가 처음 쓴 M 기반 phase는 임시값이며 집계 전에 이 script로 교정해야 한다. `run-cached.sh`가 이를 호출하고 `passes.tsv`와 각 CSV의 `execution_pass`에 근거를 남긴다. 이 수집은 prefill 1회·decode 3회 조건만 지원한다.

## 비용과 검증

- CPU는 decomposition, row/K 선택과 activation pack, weight decode/gather/transpose, INT32 output의 limb 복원을 구간별로 잰다. 2회 warmup 후 7회 실행하며 A*/B/C와 현재 A 준비 순서를 섞는다. 생성 배열은 opaque memory barrier로 전체 계산을 유지한다. 파일 I/O와 GEMM emulation 시간은 측정 구간에 없다. vector 해제와 공통 최종 float scale/merge는 포함하지 않는다.
- 현재 A의 준비 시간은 selection bitmap을 만들고 실제 builder에 emit한 뒤 실제 run-aware activation/weight/carrier 배열을 만드는 시간이다. ExSIA outlier selection bitmap을 받는 실제 producer보다 bitmap 재생성 비용이 추가된다. A*와 비교할 때 이 차이를 명시한다.
- `compaction-bitmap`은 fixture가 추가한 bitmap allocation·zeroing·event scatter를 2회 warmup·7회 반복해 따로 잰다. 독립 microbenchmark이므로 그 시간을 A에서 빼서 production latency로 해석하지 않는다.
- A* activation/weight의 모든 원소, global row map, K run mask를 현재 A와 대조한다. B/C의 결과는 원본 sparse `residual × original_weight_code × 2^HP1_exponent` 식과 source row 전체·균등 선택한 최대 32개 output column에서 대조한다. 모든 column에서의 포화 부재는 각 weight block의 최대 scale과 digit L1 합으로 별도 검증한다.
- 포화 경계에서는 A와 B/C가 수치적으로 다를 수 있다. `--self-test`는 K32의 양·음 기여가 K 압축으로 합쳐질 때와 원래 DIM16 fragment로 나뉠 때의 차이 및 INT32 balanced carry를 확인한다.
- Device는 현재 `im2p_cycle_estimate_runs`에 각 경로의 M/N/K, 원본 K32 run, runtime과 같은 tiling policy를 전달한다. SP 256 KiB, ACC 64 KiB, DIM16/64, 대응 W4A4/W8A8 packed-width profile, fresh half 0, 기본 backing timing을 사용한다. NPU output은 실제로 계산하지 않는다.
- CPU 실측을 먼저 끝내고, 값이 없는 scalar geometry의 cycle 추정만 최대 8개 프로세스로 나누어 계산한다. cycle model wall time은 성능 수치에 들어가지 않는다. cycle cache key는 M/N/K·DIM·bits와 run별 원본 block ID/compact count다. 선택된 K의 위치는 model admission에 쓰이고 timing의 물리적 배열 주소·shape는 이 key가 결정한다. 완료된 추정은 runner binary SHA가 같은 경우에만 재사용한다.
- 기본 10M cycle admission limit을 넘는 큰 GEMM에는 소프트웨어 한도를 1B cycles/10M fragments로 늘린다. 하드웨어 메모리나 queue capacity를 늘리는 변경이 아니다.

`compaction-cached`는 layer의 전체 weight를 `[K][N]` INT32 호스트 buffer로 한 번 decode/transpose한 뒤 A*의 선택 K 행을 복사하는 시간을 별도 실측한다. B/C는 그 buffer를 직접 재사용한다고 가정해 기존 실측에서 decomposition·pack·restore만 합산한다. 이것은 DIM16 prototype의 호스트 buffer 재사용 시나리오이며 NPU SRAM에 weight를 상주시킨다는 뜻은 아니다. Device weight load 비용은 그대로 남는다. Decode 초기 비용과 전체 buffer 용량은 `cached-weight-setup.csv`, 반복 실행 비용은 `cached-weight-summary.csv`에 있다. 캐시 구현을 현재 A의 production 경로에 연결하지 않았다.

DIM64는 DIM16으로 수집한 동일 residual/stripe에 대한 device geometry 비교다. DIM64로 inference를 다시 실행하면 ExSIA stripe와 residual이 바뀔 수 있으므로 그 결과와 혼동하면 안 된다. 현재 A host 함수는 DIM16으로 컴파일했고 DIM64 행의 host 값은 참고용이다.

`CPU_ms + cycles / (GHz × 10^6)`은 overlap을 가정하지 않는 조건부 서비스 시간 합계다. 목표 클록이 확인되지 않았으므로 1GHz는 환산 예시다. 실제 FPGA 시간, 전 모델 latency, 품질/PPL 또는 여러 prompt에서의 일반화는 이 실험이 측정하지 않는다. 원래 summary의 `resident_host_ms`는 단순 builder에서 weight materialization을 제외한 세 구간의 합이다. 이것만으로 A*의 K gather가 없어지는 것은 아니므로 cache 비교에는 별도 실측한 copy 비용을 더한다.
