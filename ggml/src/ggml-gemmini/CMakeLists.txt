message(STATUS "GEMMINI found")

set(RISCV_TOOL_PATH $ENV{RISCV})
if (NOT DEFINED GEMMINI_SW_PATH OR GEMMINI_SW_PATH STREQUAL "")
  if (DEFINED ENV{GEM_HOME} AND NOT "$ENV{GEM_HOME}" STREQUAL "")
    set(GEMMINI_SW_PATH $ENV{GEM_HOME})
  else()
    # Local fallback for x86 CPU builds: repo root + ../RISC-V-DynDNN-gemmini-include
    get_filename_component(LLAMA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
    set(GEMMINI_SW_PATH "${LLAMA_ROOT}/../RISC-V-DynDNN-gemmini-include")
  endif()
endif()
if (NOT EXISTS "${GEMMINI_SW_PATH}/gemmini.h"
    AND NOT EXISTS "${GEMMINI_SW_PATH}/include/gemmini.h"
    AND NOT EXISTS "${GEMMINI_SW_PATH}/gemmini-rocc-tests/include/gemmini.h")
  get_filename_component(LLAMA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
  set(GEMMINI_SW_PATH "${LLAMA_ROOT}/../RISC-V-DynDNN-gemmini-include")
endif()

option(GGML_GEMMINI_FORCE_GGML_OUTPUT "Force Gemmini backend to produce outputs via GGML CPU path" OFF)
option(GGML_GEMMINI_FORCE_GGML_ALL "Force all Gemmini layers to GGML CPU fallback when forcing output" OFF)

set(GGML_GEMMINI_FORCE_GGML_OUTPUT_DEF $<IF:$<BOOL:${GGML_GEMMINI_FORCE_GGML_OUTPUT}>,1,0>)
set(GGML_GEMMINI_FORCE_GGML_ALL_DEF    $<IF:$<BOOL:${GGML_GEMMINI_FORCE_GGML_ALL}>,1,0>)

set(GEMMINI_COMMON_FLAGS
  -fno-common
  -fno-tree-loop-distribute-patterns
  -ffast-math
)

set(GEMMINI_RISCV_FLAGS
  -std=gnu99
  -fno-builtin-printf
  -march=rv64gc
  -Wa,-march=rv64gc
)

ggml_add_backend_library(ggml-gemmini
                         ggml-gemmini.cpp
                         quantization/ggml-gemmini-quantize.cpp
                         test/ggml-gemmini-test.cpp
                        )

if (CMAKE_SYSTEM_PROCESSOR MATCHES "riscv")
  target_compile_options(ggml-gemmini PRIVATE ${GEMMINI_COMMON_FLAGS} ${GEMMINI_RISCV_FLAGS})
else()
  message(STATUS "GEMMINI: building on ${CMAKE_SYSTEM_PROCESSOR}, skipping RISC-V specific flags")
  target_compile_options(ggml-gemmini PRIVATE ${GEMMINI_COMMON_FLAGS})
endif()

target_compile_definitions(ggml-gemmini PRIVATE
  # llama
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
  GGML_GEMMINI_FORCE_GGML_OUTPUT=${GGML_GEMMINI_FORCE_GGML_OUTPUT_DEF}
  GGML_GEMMINI_FORCE_GGML_ALL=${GGML_GEMMINI_FORCE_GGML_ALL_DEF}

  # gemmini flag
  PRINT_TILE=0    # 필요 시
  OPTION=CPU      # OS/WS/CPU
  FULL_C=1        # full_C
  LOW_D=0         # low_D

  # switches
  TRANSPOSE_B=1 # B 전치 여부

  # slice 크기
  SLICE_I=1
  SLICE_K=4
  SLICE_J=6

# error compensation
  SCALE_A=0.0035f
  DEC_ALPHA_RATIO=0.05
  DEC_ALPHA_MIN=1
)

target_include_directories(ggml-gemmini PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}
  ${RISCV_TOOL_PATH}/sysroot/include
  ${GEMMINI_SW_PATH}
  ${GEMMINI_SW_PATH}/include
  ${GEMMINI_SW_PATH}/gemmini-rocc-tests
  ${GEMMINI_SW_PATH}/gemmini-rocc-tests/include
  ${GEMMINI_SW_PATH}/rocc-software/src
)

target_link_libraries(ggml-gemmini PRIVATE
  # ${RISCV_TOOL_PATH}/sysroot/lib/libm.so
)

# orca 연동: ggml 의존이 없는 orca-ggml이 있으면 그것을 사용, 없으면 log/capi로 폴백.
if (TARGET orca-ggml)
  target_link_libraries(ggml-gemmini PRIVATE orca-ggml)
  target_compile_definitions(ggml-gemmini PRIVATE GGML_GEMMINI_WITH_ORCA=1)
elseif (TARGET orca-capi)
  target_link_libraries(ggml-gemmini PRIVATE orca-capi)
  target_compile_definitions(ggml-gemmini PRIVATE GGML_GEMMINI_WITH_ORCA=1)
elseif (TARGET orca-log)
  target_link_libraries(ggml-gemmini PRIVATE orca-log)
  target_compile_definitions(ggml-gemmini PRIVATE GGML_GEMMINI_WITH_ORCA=1)
else()
  message(STATUS "orca not found; building ggml-gemmini without orca")
endif()
