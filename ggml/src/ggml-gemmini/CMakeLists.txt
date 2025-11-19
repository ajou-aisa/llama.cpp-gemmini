message(STATUS "GEMMINI found")

set(RISCV_TOOL_PATH $ENV{RISCV})
set(GEMMINI_SW_PATH $ENV{GEM_HOME})

ggml_add_backend_library(ggml-gemmini
                         ggml-gemmini.cpp
                         test/ggml-gemmini-test.cpp
                        )

target_compile_options(ggml-gemmini PRIVATE
  -std=gnu99
  -fno-common
  -fno-builtin-printf
  -march=rv64gc
  -Wa,-march=rv64gc
  -fno-tree-loop-distribute-patterns
  -fno-common
  -ffast-math
)

target_compile_definitions(ggml-gemmini PRIVATE
  # llama
  CYCLE_LOG=1 # print cycle log
  ERROR_COMPENSATION=0
  DEBUG=0         # ggml-gemmini-util.h에 정의된 DEBUG 매크로
  GEMMINI_DEBUG=0 # gemmini의 debug 로그 매크로
  TEST=0          # 필요 시
  ACTIVATION_BLOCK_SCALE=1 # block-scale 사용
  ACTIVATION_TENSOR_SCALE=0 # per-tensor scale 사용
  QACT_USE_PERCENTILE=0
  QACT_PCTL=0.999f

  # debug
  GEMMINI_GOLDEN_MM=0
  GEMMINI_GOLDEN_CHECK=0
  GEMMINI_KBLOCK_DEBUG=1
  GGML_GEMMINI_FORCE_GGML_OUTPUT=0
  GGML_GEMMINI_FORCE_GGML_ALL=0

  # gemmini flag
  PRINT_TILE=0    # 필요 시
  OPTION=CPU      # OS/WS/CPU
  FULL_C=1        # full_C
  LOW_D=0         # low_D

  # test switches (DUMP는 소스에서 자동 파생)
  TRANSPOSE_B=1 # B 전치 여부
  TEST_SHAPE=0 # 원본/변환 tensor shape 확인
  TEST_TYPE=0 # 타입 확인
  TEST_SORT=0 # sort 테스트
  TEST_DEC_METRICS=0 # k 중복 측정
  TEST_SLICE=0 # slicing 하여 dump
  TEST_CPU_REF=0 # 기대값 계산
  TEST_GEMMINI=0 # gemmini 호출
  TEST_COMPARE=0 # 결과 비교
  TEST_DEQUANTIZE_OUT=0 # 연산 결과 복사(dequantize)

  # slice 크기
  SLICE_I=1
  SLICE_K=4
  SLICE_J=6

  #error compensation
  SCALE_A=0.0035f
  DEC_ALPHA_RATIO=0.05
  DEC_ALPHA_MIN=1
)


target_include_directories(ggml-gemmini PRIVATE
  ${RISCV_TOOL_PATH}/sysroot/include
  ${GEMMINI_SW_PATH}/gemmini-rocc-tests
)
target_link_libraries     (ggml-gemmini PRIVATE
  # ${RISCV_TOOL_PATH}/sysroot/lib/libm.so
)
