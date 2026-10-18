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

# Activation quantization and error compensation use the existing options
# (cmake/ggml-gemmini-options.cmake) and the unmodified Gemmini sources below:
# GGML_GEMMINI_ACTIVATION_QUANT = EXSIA|TENSOR|TOKEN|BLOCK|STRIPE,
# GGML_GEMMINI_ENABLE_RMD = ON|OFF. Only combinations PoTal cannot execute are refused.
if (NOT GGML_GEMMINI_ACTIVATION_BITS STREQUAL "8" OR NOT GGML_GEMMINI_WEIGHT_BITS STREQUAL "8")
    message(FATAL_ERROR "GGML_POTAL is A8W8: GGML_GEMMINI_ACTIVATION_BITS=8 and GGML_GEMMINI_WEIGHT_BITS=8 required "
                        "(got ${GGML_GEMMINI_ACTIVATION_BITS}/${GGML_GEMMINI_WEIGHT_BITS})")
endif()
if (NOT GGML_GEMMINI_BLOCK_SIZE STREQUAL "32")
    message(FATAL_ERROR "GGML_POTAL: GGML_GEMMINI_BLOCK_SIZE must be 32, the Q8_HP1 scale block (got ${GGML_GEMMINI_BLOCK_SIZE})")
endif()
if (GGML_GEMMINI_ENABLE_RMD AND NOT GGML_GEMMINI_DEFAULT_RMD_BACKEND STREQUAL "CPU")
    # The WS residual executor (rmd-executor.cpp) drives Gemmini through gemmini.h/IM2P;
    # PoTal has no compact-RMD executor yet. CPU = the existing cpu_direct route.
    message(FATAL_ERROR "GGML_POTAL: GGML_GEMMINI_DEFAULT_RMD_BACKEND=${GGML_GEMMINI_DEFAULT_RMD_BACKEND} needs the Gemmini WS "
                        "residual executor, which PoTal does not have; use CPU (cpu_direct) or GGML_GEMMINI_ENABLE_RMD=OFF")
endif()
if (NOT GGML_GEMMINI_DEFAULT_MATMUL_MODE STREQUAL "FULL")
    message(FATAL_ERROR "GGML_POTAL executes FULL matmul invocations only (GGML_GEMMINI_DEFAULT_MATMUL_MODE=${GGML_GEMMINI_DEFAULT_MATMUL_MODE}); "
                        "the stripe pipeline is not part of this backend")
endif()

find_package(Threads REQUIRED)

# ggml_gemmini_args_t needs <gemmini_params.h>: the selected PoTal header is a
# Gemmini parameter header (DIM, elem_t = int8_t, acc_t = int32_t, ...).
set(POTAL_GEMMINI_PARAMS_DIR "${POTAL_HW_GENERATED_DIR}/gemmini-params")
file(WRITE "${POTAL_GEMMINI_PARAMS_DIR}/gemmini_params.h.tmp"
     "// Generated by ggml-potal/CMakeLists.txt: the selected PoTal parameter header.\n"
     "#include \"original-potal-params.h\"\n")
configure_file("${POTAL_GEMMINI_PARAMS_DIR}/gemmini_params.h.tmp" "${POTAL_GEMMINI_PARAMS_DIR}/gemmini_params.h" COPYONLY)

# Unmodified Gemmini sources: activation quantizers and metadata, weight
# reader, cpu_direct residual compensation, matmul option resolution and the
# telemetry they report through. None of them needs gemmini.h or IM2P.sim.
set(_GEMMINI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../ggml-gemmini)
set(POTAL_GEMMINI_SHARED_SOURCES
    ${_GEMMINI_DIR}/quants/act/dispatch.cpp
    ${_GEMMINI_DIR}/quants/act/quantize.cpp
    ${_GEMMINI_DIR}/quants/act/block/block.cpp
    ${_GEMMINI_DIR}/quants/act/exsia/exsia.cpp
    ${_GEMMINI_DIR}/quants/act/exsia/exsia-profile.cpp
    ${_GEMMINI_DIR}/quants/act/exsia/local.cpp
    ${_GEMMINI_DIR}/quants/act/exsia/folding.cpp
    ${_GEMMINI_DIR}/quants/act/tensor/tensor.cpp
    ${_GEMMINI_DIR}/quants/act/stripe/stripe.cpp
    ${_GEMMINI_DIR}/quants/act/token/token.cpp
    ${_GEMMINI_DIR}/quants/common/dequant.cpp
    ${_GEMMINI_DIR}/quants/common/weight_reader.cpp
    ${_GEMMINI_DIR}/residual/direct/direct-executor.cpp
    ${_GEMMINI_DIR}/residual/rmd/rmd-builder.cpp
    ${_GEMMINI_DIR}/residual/rmd/rmd-bitmap-builder.cpp
    ${_GEMMINI_DIR}/residual/rmd/rmd-compose.cpp
    ${_GEMMINI_DIR}/matmul/options.cpp
    ${_GEMMINI_DIR}/ggml-gemmini-telemetry.cpp)

# Backend sources without the registry (shared with the hardware-free tests).
set(POTAL_BACKEND_SOURCES
    ggml-potal-buffer.cpp
    ggml-potal-buffer.hpp
    log.cpp
    log.hpp
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
    matmul/weight_cache.cpp
    matmul/weight_cache.hpp
    quants/weight/hp1/hp1.cpp
    quants/weight/hp1/hp1.hpp
    ${POTAL_GEMMINI_SHARED_SOURCES})

# Compile definitions of the Gemmini backend (cmake/ggml-gemmini-config.cmake)
# without its execution-mode settings, plus the PoTal DIM. LOG_* / CYCLE_DETAIL
# must match the values ggml-gemmini-utils was built with.
set(POTAL_LOG_DEFINITIONS
    ${GGML_GEMMINI_RUNTIME_COMPILE_DEFS}
    GGML_GEMMINI_BLOCK_SIZE=${_GGML_GEMMINI_BLOCK_SIZE_VALUE}
    GGML_GEMMINI_EXSIA_SIGMA=${GGML_GEMMINI_EXSIA_SIGMA}
    GGML_GEMMINI_CONFIGURED_DIM=${POTAL_HW_DIM}
    GGML_GEMMINI_EXECUTION_BACKEND_HARDWARE=1
    EXSIA_VALIDATION=0
    GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE=${GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE})
set(POTAL_PRIVATE_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${_GEMMINI_DIR}
    ${GGML_GEMMINI_GENERATED_CONFIG_DIR}
    ${POTAL_GEMMINI_PARAMS_DIR}
    ${POTAL_HW_GENERATED_DIR})

# ExSIA LOCAL_PARALLEL / LOCAL_FOLDING_PIPELINE run on OpenMP, as in the Gemmini backend.
if (GGML_GEMMINI_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()
function(potal_add_backend_settings target)
    target_include_directories(${target} PRIVATE ${POTAL_PRIVATE_INCLUDES})
    target_compile_definitions(${target} PRIVATE ${POTAL_LOG_DEFINITIONS})
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_link_libraries(${target} PRIVATE potal-api ggml-gemmini-utils Threads::Threads)
    if (GGML_GEMMINI_ENABLE_OPENMP)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(${target} PRIVATE GGML_GEMMINI_HAS_OPENMP)
    endif()
endfunction()

ggml_add_backend_library(ggml-potal ggml-potal.cpp ${POTAL_BACKEND_SOURCES})
potal_add_backend_settings(ggml-potal)

# Hardware-free checks (host and target): registration, supports_op contract,
# packing against a CPU-backend golden, scheduler placement, no fake success.
# The backend sources are compiled in because their functions are internal.
add_executable(potal-ggml-unit-test tests/potal-ggml-unit-test.cpp ${POTAL_BACKEND_SOURCES})
potal_add_backend_settings(potal-ggml-unit-test)
target_link_libraries(potal-ggml-unit-test PRIVATE ggml)
if (NOT CMAKE_CROSSCOMPILING)
    add_test(NAME potal-ggml-unit-test COMMAND potal-ggml-unit-test)
endif()

# Hardware-free checks (host and target): PoTal records in the existing Gemmini
# debug/cycle/semantic JSONL, no routine terminal output, failure records,
# weight-cache records, inference identity and the POTAL TARGET_NPU classification.
add_executable(potal-log-test tests/potal-log-test.cpp ${POTAL_BACKEND_SOURCES})
potal_add_backend_settings(potal-log-test)
target_include_directories(potal-log-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../common) # json.hpp
target_link_libraries(potal-log-test PRIVATE ggml)
if (NOT CMAKE_CROSSCOMPILING)
    add_test(NAME potal-log-test COMMAND potal-log-test)
endif()

# Hardware-free checks (host and target): tile selection against the gemmini.h
# tiled_matmul_auto policy, tiled execution on an RTL-semantics emulator
# against the independent integer golden, HP1 validation/unpacking, the
# activation dispatcher of the configured GGML_GEMMINI_ACTIVATION_QUANT, the
# Tensor/Token/Stripe/ExSIA/Block + block routes, the compensation switch,
# F32 reconstruction and the F32 x Q8_HP1 GGML contract.
add_executable(potal-tiling-unit-test tests/potal-tiling-unit-test.cpp ${POTAL_BACKEND_SOURCES})
potal_add_backend_settings(potal-tiling-unit-test)
target_link_libraries(potal-tiling-unit-test PRIVATE ggml)
if (NOT CMAKE_CROSSCOMPILING)
    add_test(NAME potal-tiling-unit-test COMMAND potal-tiling-unit-test)
endif()

# Benchmark (host and target, not a test): Q8_HP1 weight preparation cost per
# stage and with the backend's weight cache, synthetic or from a GGUF (--gguf).
# GGML_GEMMINI_TESTING enables the Gemmini reader's call counters in this target only.
add_executable(potal-weight-prep-bench tests/potal-weight-prep-bench.cpp ${POTAL_BACKEND_SOURCES})
potal_add_backend_settings(potal-weight-prep-bench)
target_compile_definitions(potal-weight-prep-bench PRIVATE GGML_GEMMINI_TESTING)
target_link_libraries(potal-weight-prep-bench PRIVATE ggml)

# FireSim hardware test: real GGML MUL_MAT graphs computed by the POTAL backend.
if (CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv")
    add_executable(potal-ggml-smoke tests/potal-ggml-smoke.cpp)
    target_link_libraries(potal-ggml-smoke PRIVATE ggml)

    # FireSim: F32 x Q8_HP1 MUL_MAT graphs through the routed, tiled PoTal path,
    # checked element-exactly against the CPU integer golden, plus generation
    # wrap and DMA buffer reallocation. Golden helpers come from the backend sources.
    add_executable(potal-hp1-smoke tests/potal-hp1-smoke.cpp ${POTAL_BACKEND_SOURCES})
    potal_add_backend_settings(potal-hp1-smoke)
    target_link_libraries(potal-hp1-smoke PRIVATE ggml)
endif()
