message(STATUS "GEMMINI found")
include(CheckCXXSourceCompiles)


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

if(GGML_GEMMINI_FP32)
    include(fp32/backend.cmake)
    return()
endif()

# Standalone Gemmini API (ISA, encoders, transports) and the WS tiler.
add_subdirectory(api)

set(GGML_GEMMINI_SOURCES
    ggml-gemmini.cpp
    potal-sensitivity.cpp
    ops.cpp
    ggml-gemmini-buffer.cpp
    ggml-gemmini-geometry.hpp
    matmul/dense.cpp
    matmul/execution.cpp
    ../ggml-npu-common/matmul/options.cpp
    matmul/attention.cpp
    matmul/attention-stationary.cpp
    ../ggml-npu-common/profiling/telemetry.cpp
    ../ggml-npu-common/quantization/activation/dispatch.cpp
    ../ggml-npu-common/quantization/activation/quantize.cpp
    ../ggml-npu-common/quantization/activation/preparation-cache.cpp
    ../ggml-npu-common/quantization/activation/block/block.cpp
    ../ggml-npu-common/quantization/activation/exsia/exsia.cpp
    ../ggml-npu-common/quantization/activation/exsia/exsia-profile.cpp
    ../ggml-npu-common/quantization/activation/exsia/local.cpp
    ../ggml-npu-common/quantization/activation/exsia/folding.cpp
    ../ggml-npu-common/quantization/activation/tensor/tensor.cpp
    ../ggml-npu-common/quantization/activation/stripe/stripe.cpp
    ../ggml-npu-common/quantization/activation/token/token.cpp
    ../ggml-npu-common/quantization/common/fp16_util.cpp
    ../ggml-npu-common/quantization/common/dequant.cpp
    ../ggml-npu-common/quantization/common/weight_reader.cpp
    ../ggml-npu-common/residual/direct/direct-executor.cpp
    ../ggml-npu-common/residual/rmd/rmd-builder.cpp
    ../ggml-npu-common/residual/rmd/rmd-bitmap-builder.cpp
    residual/rmd/rmd-run-aware.cpp
    residual/rmd/rmd-executor.cpp
    ../ggml-npu-common/residual/rmd/rmd-compose.cpp
    ../ggml-npu-common/residual/rmd/rmd-reference.cpp
    dump/dump_common.cpp
    dump/dump_tensor.cpp
    dump/scale/scale_dump_block.cpp
    dump/scale/scale_dump_dispatch.cpp
    dump/scale/scale_dump_stub.cpp
    matmul/baseline.cpp
    matmul/block-quantized.cpp
    reference/cpu.cpp
)
ggml_add_backend_library(ggml-gemmini ${GGML_GEMMINI_SOURCES})
target_link_libraries(ggml-gemmini PUBLIC gemmini-tiled-api)
set_source_files_properties(matmul/attention.cpp matmul/attention-stationary.cpp potal-sensitivity.cpp PROPERTIES COMPILE_OPTIONS "-fno-fast-math")

target_compile_features(ggml-gemmini PRIVATE cxx_std_17)
if (GGML_GEMMINI_CUDA_RTN OR GGML_GEMMINI_CUDA_RTN_ATTENTION)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(ggml-cuda-rtn STATIC ../ggml-cuda/cpu-exact/ggml-cuda-rtn.cu)
    set_target_properties(ggml-cuda-rtn PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_compile_options(ggml-cuda-rtn PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ftz=false;--fmad=false;--prec-div=true>)
    target_include_directories(ggml-cuda-rtn PRIVATE ../../include ..)
    target_link_libraries(ggml-cuda-rtn PUBLIC CUDA::cudart CUDA::cublas ggml-base)
    target_link_libraries(ggml-gemmini PRIVATE ggml-cuda-rtn)
    if (GGML_GEMMINI_CUDA_RTN)
        target_compile_definitions(ggml-gemmini PUBLIC GGML_GEMMINI_CUDA_RTN=1)
    else()
        target_compile_definitions(ggml-gemmini PUBLIC GGML_GEMMINI_CUDA_RTN_ATTENTION=1)
    endif()
endif()
if (GGML_GEMMINI_CUDA_CPU_EXACT)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(ggml-gemmini-cuda-cpu-exact STATIC
        ../ggml-cuda/cpu-exact/ggml-cuda-cpu-exact.cu
        ../ggml-cuda/cpu-exact/ggml-cuda-hp1-scu.cu
        ../ggml-cuda/cpu-exact/ggml-cuda-potal-attention.cu)
    set_target_properties(ggml-gemmini-cuda-cpu-exact PROPERTIES
        POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_compile_options(ggml-gemmini-cuda-cpu-exact PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--ftz=false;--fmad=false;--prec-div=true;--prec-sqrt=true>)
    target_include_directories(ggml-gemmini-cuda-cpu-exact PRIVATE ../../include ..)
    target_link_libraries(ggml-gemmini-cuda-cpu-exact PUBLIC CUDA::cudart CUDA::cublas ggml-base)
    target_link_libraries(ggml-gemmini PRIVATE ggml-gemmini-cuda-cpu-exact)
    target_compile_definitions(ggml-gemmini PUBLIC GGML_GEMMINI_CUDA_CPU_EXACT=1)
endif()
ggml_gemmini_apply_profile_compile_definitions(ggml-gemmini)
ggml_gemmini_assert_profile_compile_definitions(ggml-gemmini)
if (GGML_GEMMINI_PB)
    target_sources(ggml-gemmini PRIVATE pb/pb-attention.cpp)
    set_source_files_properties(pb/pb-attention.cpp PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
    target_sources(ggml-gemmini PRIVATE pb/pb-exec.cpp)
    set_source_files_properties(pb/pb-exec.cpp PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
    # The target uses -ffast-math, which would fold the quantizer's non-finite input check away.
    set_source_files_properties(../ggml-npu-common/quantization/activation/block/block.cpp PROPERTIES COMPILE_OPTIONS "-fno-finite-math-only")
    target_compile_definitions(ggml-gemmini PUBLIC GGML_GEMMINI_PB=1)
    set(GGML_GEMMINI_PB_WEIGHT_FORMAT "Q8_0" CACHE STRING "Gemmini-PB weight format: Q8_0 or MXINT8")
    if (GGML_GEMMINI_PB_WEIGHT_FORMAT STREQUAL "Q8_0")
        target_compile_definitions(ggml-gemmini PUBLIC GGML_GEMMINI_PB_Q8_0=1)
    elseif (NOT GGML_GEMMINI_PB_WEIGHT_FORMAT STREQUAL "MXINT8")
        message(FATAL_ERROR "GGML_GEMMINI_PB_WEIGHT_FORMAT must be Q8_0 or MXINT8")
    endif()
    if (NPU_TRANSPORT STREQUAL "REFERENCE")
        target_compile_definitions(ggml-gemmini PUBLIC GEMMINI_ROCC_MODEL=1)
        target_sources(ggml-gemmini PRIVATE pb/pb-rocc-model.cpp)
    endif()
    if (LLAMA_BUILD_TESTS)
        target_compile_definitions(ggml-gemmini PUBLIC GGML_GEMMINI_TESTING=1)
        if (GGML_GEMMINI_PB_WEIGHT_FORMAT STREQUAL "Q8_0")
        # AArch64 enables NEON without -march. Explicitly select the scalar
        # branches in this test-owned oracle; PB production quantization uses
        # quantize_row_q8_0_ref and its rounding must not be compared to NEON.
        foreach(pb_q8_test gemmini-pb-q8-0-block-test gemmini-pb-q8-0-attention-test)
            add_executable(${pb_q8_test} tests/${pb_q8_test}.cpp ../ggml-cpu/ggml-cpu-quants.c)
            target_include_directories(${pb_q8_test} PRIVATE .. ../ggml-cpu)
            target_link_libraries(${pb_q8_test} PRIVATE ggml-gemmini ggml-cpu ggml-base)
            target_compile_features(${pb_q8_test} PRIVATE cxx_std_17)
            target_compile_options(${pb_q8_test} PRIVATE -fno-fast-math -ffp-contract=off)
            if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
                target_compile_options(${pb_q8_test} PRIVATE -U__ARM_NEON -U__ARM_NEON__)
            endif()
        endforeach()
        cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL
            add_test NAME gemmini-pb-q8-0-block-test COMMAND "$<TARGET_FILE:gemmini-pb-q8-0-block-test>")
        cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL
            add_test NAME gemmini-pb-q8-0-attention-test COMMAND "$<TARGET_FILE:gemmini-pb-q8-0-attention-test>")
        else()
        add_executable(gemmini-pb-block-test tests/gemmini-pb-block-test.cpp)
        target_link_libraries(gemmini-pb-block-test PRIVATE ggml-gemmini ggml-base)
        target_compile_features(gemmini-pb-block-test PRIVATE cxx_std_17)
        target_compile_options(gemmini-pb-block-test PRIVATE -fno-fast-math -ffp-contract=off)
        cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL
            add_test NAME gemmini-pb-block-test COMMAND "$<TARGET_FILE:gemmini-pb-block-test>")
        add_executable(gemmini-pb-attention-test tests/gemmini-pb-attention-test.cpp)
        target_link_libraries(gemmini-pb-attention-test PRIVATE ggml-gemmini ggml-cpu ggml-base)
        target_compile_features(gemmini-pb-attention-test PRIVATE cxx_std_17)
        target_compile_options(gemmini-pb-attention-test PRIVATE -fno-fast-math -ffp-contract=off)
        cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL
            add_test NAME gemmini-pb-attention-test COMMAND "$<TARGET_FILE:gemmini-pb-attention-test>")
        endif()
        add_executable(gemmini-pb-e2e-runner tests/gemmini-pb-e2e-runner.cpp)
        target_link_libraries(gemmini-pb-e2e-runner PRIVATE llama ggml ggml-gemmini ggml-base)
        target_compile_features(gemmini-pb-e2e-runner PRIVATE cxx_std_17)
    endif()
endif()
if (LLAMA_BUILD_TESTS AND NOT GGML_BACKEND_DL)
    target_compile_definitions(ggml-gemmini PUBLIC
        GGML_GEMMINI_TESTING=1
        GGML_GEMMINI_TEST_OBSERVER=1)
endif()
if (LLAMA_BUILD_TESTS AND NOT CMAKE_CROSSCOMPILING)
    # The WS tiler on the independent LoopMatmul host model + CPU reference golden.
    add_executable(gemmini-functional-test tests/gemmini-functional-test.cpp pb/pb-rocc-model.cpp)
    target_link_libraries(gemmini-functional-test PRIVATE gemmini-tiled-api)
    target_include_directories(gemmini-functional-test PRIVATE
        ${GGML_GEMMINI_GENERATED_CONFIG_DIR} ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/tests/golden)
    target_compile_definitions(gemmini-functional-test PRIVATE GGML_GEMMINI_PB=1)
    target_compile_options(gemmini-functional-test PRIVATE -fno-fast-math)
    add_test(NAME gemmini-functional-test
             COMMAND gemmini-functional-test "${CMAKE_CURRENT_LIST_DIR}/tests/golden")
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
    ${CMAKE_CURRENT_LIST_DIR}/../ggml-npu-common/profiling/include
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
)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
        CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    target_include_directories(ggml-gemmini PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/../ggml-npu-common/profiling/src)
endif()

target_link_libraries(ggml-gemmini PRIVATE
    ggml-npu-profiling
)


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
