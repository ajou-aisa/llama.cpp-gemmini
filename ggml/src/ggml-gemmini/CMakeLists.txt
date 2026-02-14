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
                        )

if (CMAKE_SYSTEM_PROCESSOR MATCHES "riscv")
  target_compile_options(ggml-gemmini PRIVATE ${GEMMINI_COMMON_FLAGS} ${GEMMINI_RISCV_FLAGS})
else()
  message(STATUS "GEMMINI: building on ${CMAKE_SYSTEM_PROCESSOR}, skipping RISC-V specific flags")
  target_compile_options(ggml-gemmini PRIVATE ${GEMMINI_COMMON_FLAGS})
endif()

if (DEFINED GGML_GEMMINI_COMPILE_DEFS AND NOT "${GGML_GEMMINI_COMPILE_DEFS}" STREQUAL "")
  target_compile_definitions(ggml-gemmini PRIVATE ${GGML_GEMMINI_COMPILE_DEFS})
else()
  # Standalone/subdir safety fallback with legacy defaults.
  target_compile_definitions(ggml-gemmini PRIVATE
    PRINT_TILE=0
    OPTION=CPU
    FULL_C=1
    LOW_D=0
    TRANSPOSE_B=1
  )
endif()

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

# orca 연동:
# - orca 서브프로젝트는 ggml을 먼저 add_subdirectory() 한 다음 orca-ggml을 정의한다.
#   그 때문에 이 CMakeLists가 실행될 때는 orca-ggml 타깃이 아직 없어서, 기존 로직은
#   orca-capi/orca-log로 폴백되어 quant(orca::ggml adapter)가 비활성화되는 문제가 있었다.
# - 해결: ggml-gemmini는 "orca" INTERFACE 타깃(정의가 빠름)에 링크한다.
#   이후 orca가 orca-ggml을 orca 인터페이스에 추가하면, 최종 generate 시점에
#   ggml-gemmini에도 자동으로 반영된다.
if (TARGET orca)
  target_link_libraries(ggml-gemmini PRIVATE orca)
  target_compile_definitions(ggml-gemmini PRIVATE GGML_GEMMINI_WITH_ORCA=1)
elseif (TARGET orca-ggml)
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
