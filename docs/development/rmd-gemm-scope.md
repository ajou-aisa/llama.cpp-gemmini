# RMD-GEMM 현재 작업 범위

이 문서는 `rmd-gemm` 작업의 **대상**을 기록한다. 저장소에 남아 있는 다른
형식과 backend의 지원 여부를 뜻하지 않는다.

- Weight 형식은 **HP1**만 사용한다. H1과 H0는 이 작업의 입력이나 검증 대상이 아니다.
- GEMM 실행 backend는 **Gemmini(Chisel)** 하나다. 시뮬레이션은
  `IM2P_SIM_IMPLEMENTATION=GEMMINI_HP1`을 대상으로 한다. FPGA 지원은 폐기했다.
  `LEGACY_BSV` 경로는 이 작업의 비교 기준이나 수정 대상이 아니다.
  RMD의 pruning, compact A/W 준비, 결과 복원 같은 호스트 작업은 계속 CPU에서 한다.
- 핵심 변경은 **RMD A4 activation 패킷의 nibble packing 제거**다.
  A4와 A8 activation은 dense frontend ABI처럼 모두 원소당 signed byte로
  저장·전달한다. A4의 값 범위는 여전히 4비트 signed 범위다. 원본 Q4 weight의
  nibble packing은 유지한다.
- 대상 수치 profile은 지원되는 matched A4/W4 또는 A8/W8 HP1이다.
- Dense와 RMD의 HP1 GEMM은 같은 Gemmini/SCU 수치 계약을 따른다.
  Descriptor는 `IM2P_VECTOR_LEFT_SHIFT`(op 5),
  `IM2P_OUTPUT_SCU_FINAL`을 사용한다. 원본 HP1 block/column의 `m` carrier를
  compact K에 대응시켜 전달하고, SCU는 각 유효 fragment를 스케일·Sat32한 뒤
  누적한다. RMD의 K compaction 때문에 물리적 fragment 수와 cycle은 dense와
  다를 수 있다.

따라서 HP1 검증에서는 dense/RMD의 **op, carrier, output domain, SCU 적용
순서**와 실제 Gemmini 실행을 확인한다. H1의 op 4 미지원이나 H0의 CPU-direct
경로를 해결하는 것은 이 작업의 완료 조건이 아니다.

ExSIA의 A4/A8 WS packet은 bitmap compaction으로 생성한다. Residual을 받는
즉시 limb로 분해하고, 0이 아닌 digit과 row/K 정보를 기록한다. 마지막에
nonzero bitmap을 읽어 row/K를 압축한 signed-byte packet을 만든다. 원본
weight block 식별자와 limb 지수, packet ABI는 유지한다. CPU direct에는
packet compaction이 없으며, 일반 quantizer와 A16, packet slicing은 기존
`RmdStripeBuilder`를 사용한다.

## 소스 확인 기록 (2026-09-28)

`IM2P.sim`의 `gemmini` 브랜치를 `fa33e5d`까지 갱신했다. 직전 커밋 이후의
변경은 `sim/cycle/`의 trace 인증과 stateful timing에 한정된다. Frontend의
op 선택, `GEMMINI_HP1` runtime의 허용 op, Gemmini RTL 수치 경로는 바뀌지 않았다.
이 pull 자체는 `rmd-gemm`의 기존 빌드 산출물을 재생성하지 않는다.
