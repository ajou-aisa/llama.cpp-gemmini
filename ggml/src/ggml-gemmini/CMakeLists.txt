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
  USE_GEMMINI_BENCH_TENSOR=0 # true: BenchTensor | false: BaselineTensor
  DEBUG=0         # ggml-gemmini-util.h에서 쓰는 DEBUG 매크로
  TEST=0          # 필요 시
  PRINT_TILE=0    # 필요 시
  OPTION=CPU      # OS/WS/CPU

  # test switches (DUMP는 소스에서 자동 파생)
  TRANSPOSE_B=1 # B 전치 여부
  TEST_SHAPE=0 # 원본/변환 tensor shape 확인
  TEST_TYPE=0 # 타입 확인
  TEST_SLICE=0 # slicing 하여 dump
  TEST_CPU_REF=0 # 기대값 계산
  TEST_GEMMINI=0 # gemmini 호출
  TEST_COMPARE=0 # 결과 비교
  TEST_DEQUANTIZE_OUT=0 # 연산 결과 복사(dequantize)

  # slice 크기
  SLICE_I=1
  SLICE_K=4
  SLICE_J=6
)


target_include_directories(ggml-gemmini PRIVATE
  ${RISCV_TOOL_PATH}/sysroot/include
  ${GEMMINI_SW_PATH}/gemmini-rocc-tests
)
target_link_libraries     (ggml-gemmini PRIVATE
  # ${RISCV_TOOL_PATH}/sysroot/lib/libm.so
)
