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
option(GGML_GEMMINI_ENABLE_OPENMP "Enable OpenMP for Gemmini DEC kernels" OFF)

set(GGML_GEMMINI_FORCE_GGML_OUTPUT_DEF $<IF:$<BOOL:${GGML_GEMMINI_FORCE_GGML_OUTPUT}>,1,0>)
set(GGML_GEMMINI_FORCE_GGML_ALL_DEF    $<IF:$<BOOL:${GGML_GEMMINI_FORCE_GGML_ALL}>,1,0>)

set(GEMMINI_COMMON_FLAGS
  -fno-common
  # -fno-tree-loop-distribute-patterns
  -ffast-math
)

set(GEMMINI_RISCV_FLAGS
  -std=gnu99
  -fno-tree-loop-distribute-patterns 
  -fno-builtin-printf
  -march=rv64gc
  -Wa,-march=rv64gc
)

ggml_add_backend_library(ggml-gemmini
                         ggml-gemmini.cpp
                         ggml-gemmini-ethos-config.cpp
                         quants/act/quantize.cpp
                         quants/act/ethos/types.cpp
                         quants/act/ethos/ethos.cpp
                         quants/act/tensor/tensor.cpp
                         quants/weight/quantize_Q8_0_R.cpp
                         quants/weight/unpack_Q8_0.cpp
                         quants/common/fp16_util.cpp
                         quants/common/dequant.cpp
                         quants/dec/dec.cpp
                         quants/dec/dec_stage.cpp
                         quants/dec/dec_weight.cpp
                         quants/dec/dec_kernel.cpp
                         dump/dump_common.cpp
                         dump/dump_tensor.cpp
                         dump/scale/scale_dump_block.cpp
                         dump/scale/scale_dump_dispatch.cpp
                         dump/scale/scale_dump_stub.cpp
                        )

target_compile_features(ggml-gemmini PRIVATE cxx_std_17)

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

target_include_directories(ggml-gemmini PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}
  ${CMAKE_CURRENT_LIST_DIR}/../../include
  ${CMAKE_CURRENT_LIST_DIR}/../ggml-gemmini-utils/include
  ${GEMMINI_SW_PATH}
  ${GEMMINI_SW_PATH}/include
)
target_include_directories(ggml-gemmini PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/quants
  ${CMAKE_CURRENT_LIST_DIR}/quants/act
  ${CMAKE_CURRENT_LIST_DIR}/quants/act/ethos
  ${CMAKE_CURRENT_LIST_DIR}/quants/common
  ${CMAKE_CURRENT_LIST_DIR}/quants/dec
  ${CMAKE_CURRENT_LIST_DIR}/quants/view
  ${CMAKE_CURRENT_LIST_DIR}/quants/weight
  ${CMAKE_CURRENT_LIST_DIR}/dump
  ${CMAKE_CURRENT_LIST_DIR}/dump/scale
  ${RISCV_TOOL_PATH}/sysroot/include
  ${GEMMINI_SW_PATH}/gemmini-rocc-tests
  ${GEMMINI_SW_PATH}/gemmini-rocc-tests/include
  ${GEMMINI_SW_PATH}/rocc-software/src
)

target_link_libraries(ggml-gemmini PRIVATE
  ggml-gemmini-utils
)

if (GGML_GEMMINI_ENABLE_OPENMP)
  find_package(OpenMP)
  if (OpenMP_FOUND)
    target_link_libraries(ggml-gemmini PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(ggml-gemmini PRIVATE GGML_GEMMINI_HAS_OPENMP)
  else()
    message(WARNING "OpenMP not found for ggml-gemmini")
  endif()
endif()
