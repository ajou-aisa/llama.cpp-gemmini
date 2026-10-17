# PoTal RoCC API (potal-api) and its tests. Two configurations, one parameter
# contract (cmake/potal-hw-params.cmake):
#   POTAL_API_ONLY=ON  added by the root CMakeLists: only the API and its tests,
#                      independent of ggml, llama, Gemmini headers and IM2P.sim.
#   GGML_POTAL=ON      added by ggml_add_backend(POTAL): additionally the GGML
#                      backend ggml-potal and its tests.

# Not PROJECT_SOURCE_DIR: under GGML_POTAL that is the nested ggml project.
include(${CMAKE_CURRENT_LIST_DIR}/../../../cmake/potal-hw-params.cmake)

set(POTAL_HW_PARAMS "" CACHE FILEPATH "RTL-generated PoTal parameter header (absolute path)")
set(POTAL_EXPECTED_DIM "" CACHE STRING "DIM the selected header must report (16|32|64)")
set(POTAL_HW_CONFIG "" CACHE STRING "Chipyard config that produced POTAL_HW_PARAMS (recorded only)")
option(POTAL_ENABLE_OPENMP "Add the OpenMP worker mode to potal-multicore-smoke (independent of GGML_OPENMP)" OFF)

potal_configure_hw_params("${CMAKE_BINARY_DIR}/generated")

add_library(potal-api STATIC potal.c)
target_include_directories(potal-api PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${POTAL_HW_GENERATED_DIR})
set_target_properties(potal-api PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF)
if (NOT POTAL_API_ONLY)
    # Linked into the (shared) ggml-potal backend, which owns the accelerator:
    # keep the potal_* symbols private to it.
    set_target_properties(potal-api PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        C_VISIBILITY_PRESET hidden)
endif()

# Field packing and validation; runs on the host and on the target.
add_executable(potal-api-test tests/potal-api-test.c)
target_link_libraries(potal-api-test PRIVATE potal-api)
set_target_properties(potal-api-test PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
if (NOT CMAKE_CROSSCOMPILING)
    add_test(NAME potal-api-test COMMAND potal-api-test)
endif()

# Hardware smoke test: RISC-V only, and only for the validated DIM16 GEMM path.
if (CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv")
    if (POTAL_HW_DIM EQUAL 16)
        add_executable(potal-api-smoke tests/potal-api-smoke.c)
        target_link_libraries(potal-api-smoke PRIVATE potal-api)
        set_target_properties(potal-api-smoke PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)

        # Multicore smoke: one hart0-pinned issuer + pinned CPU workers (Linux only).
        if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
            find_package(Threads REQUIRED)
            add_executable(potal-multicore-smoke tests/potal-multicore-smoke.c)
            target_link_libraries(potal-multicore-smoke PRIVATE potal-api Threads::Threads)
            set_target_properties(potal-multicore-smoke PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
            if (POTAL_ENABLE_OPENMP)
                # Fails configure if the target toolchain cannot compile and link OpenMP.
                find_package(OpenMP COMPONENTS C REQUIRED)
                target_link_libraries(potal-multicore-smoke PRIVATE OpenMP::OpenMP_C)
                message(STATUS "potal-multicore-smoke: OpenMP ${OpenMP_C_VERSION} flags='${OpenMP_C_FLAGS}' "
                               "libs='${OpenMP_C_LIBRARIES}' (target runtime needs libgomp.so.1 unless static)")
            endif()
        elseif (POTAL_ENABLE_OPENMP)
            message(FATAL_ERROR "POTAL_ENABLE_OPENMP=ON needs a Linux target (potal-multicore-smoke)")
        endif()
    else()
        message(STATUS "potal-api-smoke not built: GEMM execution is validated for DIM16 only (header DIM=${POTAL_HW_DIM})")
    endif()
endif()

if (POTAL_API_ONLY)
    return()
endif()

# ---- GGML backend (GGML_POTAL=ON) -------------------------------------------

if (NOT POTAL_HW_DIM EQUAL 16)
    message(FATAL_ERROR "GGML_POTAL: MUL_MAT offload is validated for DIM16 only (header DIM=${POTAL_HW_DIM})")
endif()
if (GGML_BACKEND_DL)
    message(FATAL_ERROR "GGML_POTAL supports static backend registration only (GGML_BACKEND_DL=OFF)")
endif()
if (CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv" AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "GGML_POTAL on RISC-V needs Linux (the issuer thread pins itself to the hart-0 CPU)")
endif()

# Activation quantization uses the existing options (cmake/ggml-gemmini-options.cmake).
# The PoTal hardware is A8W8 and this backend implements TENSOR only.
if (NOT _GGML_GEMMINI_ACTIVATION_QUANT_VALUE EQUAL 1)
    message(FATAL_ERROR "GGML_POTAL implements GGML_GEMMINI_ACTIVATION_QUANT=TENSOR only, got '${GGML_GEMMINI_ACTIVATION_QUANT}'")
endif()
if (NOT GGML_GEMMINI_ACTIVATION_BITS STREQUAL "8" OR NOT GGML_GEMMINI_WEIGHT_BITS STREQUAL "8")
    message(FATAL_ERROR "GGML_POTAL is A8W8: GGML_GEMMINI_ACTIVATION_BITS=8 and GGML_GEMMINI_WEIGHT_BITS=8 required "
                        "(got ${GGML_GEMMINI_ACTIVATION_BITS}/${GGML_GEMMINI_WEIGHT_BITS})")
endif()

find_package(Threads REQUIRED)

# Backend sources without the registry (shared with the hardware-free tests).
set(POTAL_BACKEND_SOURCES
    ops.cpp
    ops.hpp
    matmul/dense.cpp
    matmul/dense.hpp
    matmul/execution.cpp
    matmul/execution.hpp
    matmul/options.hpp
    matmul/pack.cpp
    matmul/pack.hpp
    matmul/tiling.cpp
    matmul/tiling.hpp
    ggml-potal-args.h
    quants/act/buffer.hpp
    quants/act/meta.hpp
    quants/act/quantize.cpp
    quants/act/quantize.hpp
    quants/act/tensor/tensor.cpp
    quants/act/tensor/tensor.hpp
    quants/act/tensor/types.hpp
    quants/common/hp1_scu.hpp
    quants/common/math.hpp
    quants/common/tensor_util.hpp
    quants/common/weight_reader.cpp
    quants/common/weight_reader.hpp
    quants/weight/hp1/hp1.cpp
    quants/weight/hp1/hp1.hpp)

# The existing Gemmini logging/cycle system (ggml-gemmini-utils, no gemmini.h).
# LOG_* / CYCLE_DETAIL must match the values the library was built with
# (cmake/ggml-gemmini-config.cmake), because its headers act on them too.
set(POTAL_LOG_DEFINITIONS
    LOG_DEBUG=${LOG_DEBUG}
    LOG_CYCLE=${LOG_CYCLE}
    CYCLE_DETAIL=${CYCLE_DETAIL}
    CYCLE_SIM=0
    GGML_GEMMINI_ACTIVATION_QUANT=${_GGML_GEMMINI_ACTIVATION_QUANT_VALUE}
    GGML_GEMMINI_ACTIVATION_BITS=${GGML_GEMMINI_ACTIVATION_BITS})

ggml_add_backend_library(ggml-potal ggml-potal.cpp ${POTAL_BACKEND_SOURCES})
target_link_libraries(ggml-potal PRIVATE potal-api ggml-gemmini-utils Threads::Threads)
target_compile_definitions(ggml-potal PRIVATE ${POTAL_LOG_DEFINITIONS})

# Hardware-free checks (host and target): registration, supports_op contract,
# packing against a CPU-backend golden, scheduler placement, no fake success.
# The backend sources are compiled in because their functions are internal.
add_executable(potal-ggml-unit-test tests/potal-ggml-unit-test.cpp ${POTAL_BACKEND_SOURCES})
target_include_directories(potal-ggml-unit-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(potal-ggml-unit-test PRIVATE ggml potal-api ggml-gemmini-utils Threads::Threads)
target_compile_definitions(potal-ggml-unit-test PRIVATE ${POTAL_LOG_DEFINITIONS})
if (NOT CMAKE_CROSSCOMPILING)
    add_test(NAME potal-ggml-unit-test COMMAND potal-ggml-unit-test)
endif()

# Hardware-free W4-5 checks (host and target): tile selection against the
# gemmini.h tiled_matmul_auto policy, tiled execution on an RTL-semantics
# emulator against the independent integer golden, HP1 validation/unpacking,
# per-tensor activation quantization, F32 reconstruction, and the F32 x Q8_HP1
# GGML contract.
add_executable(potal-tiling-unit-test tests/potal-tiling-unit-test.cpp ${POTAL_BACKEND_SOURCES})
target_include_directories(potal-tiling-unit-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(potal-tiling-unit-test PRIVATE ggml potal-api ggml-gemmini-utils Threads::Threads)
target_compile_definitions(potal-tiling-unit-test PRIVATE ${POTAL_LOG_DEFINITIONS})
if (NOT CMAKE_CROSSCOMPILING)
    add_test(NAME potal-tiling-unit-test COMMAND potal-tiling-unit-test)
endif()

# FireSim hardware test: real GGML MUL_MAT graphs computed by the POTAL backend.
if (CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv")
    add_executable(potal-ggml-smoke tests/potal-ggml-smoke.cpp)
    target_link_libraries(potal-ggml-smoke PRIVATE ggml)

    # FireSim: F32 x Q8_HP1 MUL_MAT graphs through the tiled PoTal path,
    # checked element-exactly against the CPU integer golden.
    add_executable(potal-hp1-smoke tests/potal-hp1-smoke.cpp
                   matmul/dense.cpp quants/act/quantize.cpp quants/act/tensor/tensor.cpp
                   quants/common/weight_reader.cpp quants/weight/hp1/hp1.cpp)
    target_include_directories(potal-hp1-smoke PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(potal-hp1-smoke PRIVATE ggml potal-api)
    target_compile_definitions(potal-hp1-smoke PRIVATE ${POTAL_LOG_DEFINITIONS})
endif()
