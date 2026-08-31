message(STATUS "GEMMINI found")
include(CheckCXXSourceCompiles)

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

if (NOT DEFINED GGML_GEMMINI_ENABLE_OPENMP)
  set(GGML_GEMMINI_ENABLE_OPENMP OFF)
endif()
if (NOT DEFINED GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE)
  set(GGML_GEMMINI_EXSIA_PROFILE_SCOPE "OFF" CACHE STRING "ExSIA profile scope (OFF, TIMELINE, STAGE)")
  set_property(CACHE GGML_GEMMINI_EXSIA_PROFILE_SCOPE PROPERTY STRINGS OFF TIMELINE STAGE)
  string(TOUPPER "${GGML_GEMMINI_EXSIA_PROFILE_SCOPE}" _GGML_GEMMINI_EXSIA_PROFILE_SCOPE_UPPER)
  if (_GGML_GEMMINI_EXSIA_PROFILE_SCOPE_UPPER STREQUAL "OFF")
    set(GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE 0)
  elseif (_GGML_GEMMINI_EXSIA_PROFILE_SCOPE_UPPER STREQUAL "TIMELINE")
    set(GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE 1)
  elseif (_GGML_GEMMINI_EXSIA_PROFILE_SCOPE_UPPER STREQUAL "STAGE")
    set(GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE 2)
  else()
    message(FATAL_ERROR "GGML_GEMMINI_EXSIA_PROFILE_SCOPE must be OFF, TIMELINE, or STAGE, got '${GGML_GEMMINI_EXSIA_PROFILE_SCOPE}'")
  endif()
endif()

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

function(ggml_gemmini_apply_profile_compile_definitions target)
  target_compile_definitions(${target} PRIVATE
    EXSIA_VALIDATION=0
    GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE=${GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE}
    CYCLE_DETAIL=${CYCLE_DETAIL}
    LOG_CYCLE=${LOG_CYCLE})
endfunction()

function(ggml_gemmini_assert_profile_compile_definitions target)
  get_target_property(_definitions ${target} COMPILE_DEFINITIONS)
  foreach(_expected IN ITEMS
      EXSIA_VALIDATION=0
      GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE=${GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE}
      CYCLE_DETAIL=${CYCLE_DETAIL}
      LOG_CYCLE=${LOG_CYCLE})
    string(REGEX REPLACE "=.*$" "" _key "${_expected}")
    set(_actual)
    foreach(_definition IN LISTS _definitions)
      if(_definition MATCHES "^${_key}(=|$)")
        list(APPEND _actual "${_definition}")
      endif()
    endforeach()
    list(REMOVE_DUPLICATES _actual)
    if(NOT _actual STREQUAL _expected)
      string(JOIN "," _actual_text ${_actual})
      message(FATAL_ERROR
        "GEMMINI_PROFILE_CONTRACT target=${target} definition=${_key} expected=${_expected} actual=${_actual_text}")
    endif()
  endforeach()
endfunction()

ggml_add_backend_library(ggml-gemmini
                          ggml-gemmini.cpp
                          ggml-gemmini-buffer.cpp
                          ggml-gemmini-geometry.hpp
                          ggml-gemmini-matmul.cpp
                          ggml-gemmini-matmul-telemetry.cpp
                          ggml-gemmini-telemetry.cpp
                          ggml-gemmini-q8-h1-artifact.cpp
                          ggml-gemmini-q8-h1-artifact-reader.cpp
                            quants/act/dispatch.cpp
                            quants/act/quantize.cpp
                            quants/act/block/block.cpp
                            quants/act/exsia/exsia.cpp
                         quants/act/tensor/tensor.cpp
                         quants/act/stripe/stripe.cpp
                         quants/act/token/token.cpp
                         quants/weight/quantize_Q8_H1.cpp
                         quants/common/fp16_util.cpp
                         quants/common/dequant.cpp
                         quants/common/weight_reader.cpp
                         residual/direct/direct-executor.cpp
                         residual/rmd/rmd-builder.cpp
                         residual/rmd/rmd-executor.cpp
                         residual/rmd/rmd-im2p-executor.cpp
                         residual/rmd/rmd-compose.cpp
                         residual/rmd/rmd-reference.cpp
                         dump/dump_common.cpp
                         dump/dump_tensor.cpp
                         dump/scale/scale_dump_block.cpp
                         dump/scale/scale_dump_dispatch.cpp
                         dump/scale/scale_dump_stub.cpp
                        )

target_compile_features(ggml-gemmini PRIVATE cxx_std_17)
target_compile_definitions(ggml-gemmini PRIVATE
  GGML_GEMMINI_MATMUL_IMPLEMENTATION)
ggml_gemmini_apply_profile_compile_definitions(ggml-gemmini)
ggml_gemmini_assert_profile_compile_definitions(ggml-gemmini)
if (LLAMA_BUILD_TESTS AND NOT GGML_BACKEND_DL)
  target_compile_definitions(ggml-gemmini PUBLIC
    GGML_GEMMINI_TESTING=1
    GGML_GEMMINI_TEST_OBSERVER=1)
endif()

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
  ${GGML_GEMMINI_GENERATED_CONFIG_DIR}
  ${CMAKE_CURRENT_LIST_DIR}
  ${CMAKE_CURRENT_LIST_DIR}/../../include
  ${CMAKE_CURRENT_LIST_DIR}/../ggml-gemmini-utils/include
  ${GEMMINI_SW_PATH}
  ${GEMMINI_SW_PATH}/include
)
target_include_directories(ggml-gemmini PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/quants
  ${CMAKE_CURRENT_LIST_DIR}/quants/act
  ${CMAKE_CURRENT_LIST_DIR}/quants/act/exsia
  ${CMAKE_CURRENT_LIST_DIR}/quants/act/stripe
  ${CMAKE_CURRENT_LIST_DIR}/quants/act/token
  ${CMAKE_CURRENT_LIST_DIR}/quants/common
  ${CMAKE_CURRENT_LIST_DIR}/residual/rmd
  ${CMAKE_CURRENT_LIST_DIR}/quants/view
  ${CMAKE_CURRENT_LIST_DIR}/quants/weight
  ${CMAKE_CURRENT_LIST_DIR}/dump
  ${CMAKE_CURRENT_LIST_DIR}/dump/scale
  ${RISCV_TOOL_PATH}/sysroot/include
  ${GEMMINI_SW_PATH}/gemmini-rocc-tests
  ${GEMMINI_SW_PATH}/gemmini-rocc-tests/include
  ${GEMMINI_SW_PATH}/rocc-software/src
)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
    CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  target_include_directories(ggml-gemmini PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../ggml-gemmini-utils/src)
endif()

target_link_libraries(ggml-gemmini PRIVATE
  ggml-gemmini-utils
)

if (GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM")
  foreach(_IM2P_REQUIRED_VAR IN ITEMS
          IM2P_SIM_ROOT GGML_GEMMINI_IM2P_FRONTEND_ARCHIVE GGML_GEMMINI_IM2P_SIM_ARCHIVE)
    if (NOT DEFINED ${_IM2P_REQUIRED_VAR} OR "${${_IM2P_REQUIRED_VAR}}" STREQUAL "")
      message(FATAL_ERROR "Missing validated IM2P build-contract variable ${_IM2P_REQUIRED_VAR}")
    endif()
  endforeach()

  set(_GGML_GEMMINI_IM2P_LINK_FRONTEND_ARCHIVE
    "${GGML_GEMMINI_IM2P_FRONTEND_ARCHIVE}")
  if (LLAMA_BUILD_TESTS AND NOT GGML_BACKEND_DL)
    get_filename_component(_GGML_GEMMINI_IM2P_FRONTEND_DIR
      "${GGML_GEMMINI_IM2P_FRONTEND_ARCHIVE}" DIRECTORY)
    set(_GGML_GEMMINI_IM2P_TEST_FRONTEND_ARCHIVE
      "${_GGML_GEMMINI_IM2P_FRONTEND_DIR}/libim2p_gemmini_frontend_testing.a")
    if (NOT EXISTS "${_GGML_GEMMINI_IM2P_TEST_FRONTEND_ARCHIVE}")
      message(FATAL_ERROR
        "Missing IM2P frontend test archive: ${_GGML_GEMMINI_IM2P_TEST_FRONTEND_ARCHIVE}")
    endif()
    set(_GGML_GEMMINI_IM2P_LINK_FRONTEND_ARCHIVE
      "${_GGML_GEMMINI_IM2P_TEST_FRONTEND_ARCHIVE}")
  endif()

  add_library(ggml-gemmini-im2p-frontend STATIC IMPORTED GLOBAL)
  set_target_properties(ggml-gemmini-im2p-frontend PROPERTIES
    IMPORTED_LOCATION "${_GGML_GEMMINI_IM2P_LINK_FRONTEND_ARCHIVE}")
  add_library(ggml-gemmini-im2p-simulator STATIC IMPORTED GLOBAL)
  set_target_properties(ggml-gemmini-im2p-simulator PROPERTIES
    IMPORTED_LOCATION "${GGML_GEMMINI_IM2P_SIM_ARCHIVE}")

  find_package(Threads REQUIRED)
  target_sources(ggml-gemmini PRIVATE ggml-gemmini-im2p.cpp)
  if (LLAMA_BUILD_TESTS AND NOT GGML_BACKEND_DL)
    target_include_directories(ggml-gemmini PRIVATE
      "${IM2P_SIM_ROOT}/frontend/tests")
  endif()
  target_include_directories(ggml-gemmini PRIVATE
    "${IM2P_SIM_ROOT}/frontend/include"
    "${IM2P_SIM_ROOT}/sim/include")
  # Static-library order is intentional: the frontend consumes the C ABI from
  # the simulator archive, and neither archive is present in HARDWARE builds.
  target_link_libraries(ggml-gemmini PRIVATE
    ggml-gemmini-im2p-frontend
    ggml-gemmini-im2p-simulator
    Threads::Threads
    ${CMAKE_DL_LIBS})
endif()

if (GGML_GEMMINI_ENABLE_OPENMP)
  find_package(OpenMP)
  if (OpenMP_FOUND)
    set(CMAKE_REQUIRED_FLAGS "${OpenMP_CXX_FLAGS}")
    set(CMAKE_REQUIRED_LIBRARIES OpenMP::OpenMP_CXX)
    check_cxx_source_compiles("#include <omp.h>
int main() {
  int storage[8] = {0};
  int * token = storage;
  #pragma omp parallel num_threads(2)
  #pragma omp single
  {
    #pragma omp taskgroup
    {
      #pragma omp task depend(out : token[0])
      token[0] = 1;
      #pragma omp task depend(in : token[0]) depend(out : token[1])
      token[1] = token[0];
      #pragma omp task depend(in : token[0], token[1]) depend(inout : token[2]) depend(out : token[3])
      token[2] = token[0] + token[1];
    }
  }
  return token[2] == 2 ? 0 : 1;
}" GGML_GEMMINI_OPENMP_TASK_DEPEND_COMPILES)
    unset(CMAKE_REQUIRED_FLAGS)
    unset(CMAKE_REQUIRED_LIBRARIES)
    if (NOT GGML_GEMMINI_OPENMP_TASK_DEPEND_COMPILES)
      message(FATAL_ERROR "GGML_GEMMINI_ENABLE_OPENMP requires OpenMP task depend support")
    endif()
    target_link_libraries(ggml-gemmini PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(ggml-gemmini PRIVATE GGML_GEMMINI_HAS_OPENMP)
  else()
    message(FATAL_ERROR "GGML_GEMMINI_ENABLE_OPENMP=ON but OpenMP was not found; ExSIA OpenMP pipeline is BLOCKED")
  endif()
endif()
