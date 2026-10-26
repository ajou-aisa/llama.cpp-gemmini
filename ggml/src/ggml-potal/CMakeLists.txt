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

# Planned-GEMM API of potal.h (potal_stripe_session, potal_tiled_matmul_stripe)
# on the Gemmini-compatible tiler: C++ inside, no ggml/Gemmini dependency. The
# one owner of matmul/tiling.cpp: the GGML backend and its tests link it.
add_library(potal-tiled-api STATIC
    matmul/tiling.cpp
    matmul/tiling.hpp
    matmul/stripe-api.cpp
    matmul/stripe-api.hpp)
target_link_libraries(potal-tiled-api PUBLIC potal-api)
target_compile_features(potal-tiled-api PUBLIC cxx_std_17)
if (NOT POTAL_API_ONLY)
    # Linked into the (shared) ggml-potal backend: private to it, as potal-api.
    set_target_properties(potal-tiled-api PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)
endif()

# Field packing and validation; runs on the host and on the target.
add_executable(potal-api-test tests/potal-api-test.c)
target_link_libraries(potal-api-test PRIVATE potal-api)
set_target_properties(potal-api-test PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
if (NOT CMAKE_CROSSCOMPILING)
    add_test(NAME potal-api-test COMMAND potal-api-test)
endif()
add_executable(potal-attention-api-test tests/potal-attention-api-test.c)
target_link_libraries(potal-attention-api-test PRIVATE potal-api)
set_target_properties(potal-attention-api-test PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
if (NOT CMAKE_CROSSCOMPILING)
    add_test(NAME potal-attention-api-test COMMAND potal-attention-api-test)
endif()

# Standalone Linux Indexed Weight diagnostic; --self-test checks C11 encoders on the host.
if (POTAL_HW_DIM EQUAL 16 AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Exercise the RISC-V trace wrapper on the host, keeping real RoCC dispatch
    # disabled there. This private test definition adds no runtime control API.
    add_executable(potal-trace-test tests/potal-trace-test.c potal.c)
    target_include_directories(potal-trace-test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include ${POTAL_HW_GENERATED_DIR})
    target_compile_definitions(potal-trace-test PRIVATE POTAL_TEST_RAW_TRACE)
    set_target_properties(potal-trace-test PROPERTIES
        C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
    if (NOT CMAKE_CROSSCOMPILING)
        add_test(NAME potal-trace-test COMMAND potal-trace-test)
    endif()
    add_executable(potal-scale-path-bench tests/potal-scale-path-bench.c tests/potal-scale-clock.cpp)
    target_link_libraries(potal-scale-path-bench PRIVATE potal-api)
    target_compile_features(potal-scale-path-bench PRIVATE cxx_std_11)
    set_target_properties(potal-scale-path-bench PROPERTIES
        C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
    add_executable(potal-paired-index-hw-probe tests/potal-paired-index-hw-probe.c)
    target_link_libraries(potal-paired-index-hw-probe PRIVATE potal-api)
    set_target_properties(potal-paired-index-hw-probe PROPERTIES
        C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
    if (NOT CMAKE_CROSSCOMPILING)
        add_test(NAME potal-paired-index-encoder-test COMMAND potal-paired-index-hw-probe --self-test)
    endif()
endif()

# potal_tiled_matmul_stripe from a C11 caller (header and link) and, on an
# RTL-semantics emulator, frozen plans, tails, K chunks, descriptor rejection
# without commands, generation wrap. Host and target; no hardware.
add_executable(potal-stripe-api-test tests/potal-stripe-api-test.cpp tests/potal-stripe-api-c.c)
target_include_directories(potal-stripe-api-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(potal-stripe-api-test PRIVATE potal-tiled-api)
set_target_properties(potal-stripe-api-test PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
if (NOT CMAKE_CROSSCOMPILING)
    add_test(NAME potal-stripe-api-test COMMAND potal-stripe-api-test)
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
# Execution modes (matmul/options.hpp): FULL runs every activation algorithm
# and, under RMD, the CPU (cpu_direct) residual; STRIPE_PIPELINE runs EXSIA
# and, under RMD, the WS residual as a PoTal Residual GEMM (potal_ws). The
# Gemmini WS executor (rmd-executor.cpp, gemmini.h/IM2P) is not linked. Build
# defaults outside that are refused here; runtime overrides
# (GEMMINI_MATMUL_MODE, GEMMINI_RMD_BACKEND, GEMMINI_STRIPE_JOB_CAPACITY) are
# refused per node, never replaced by another path.
string(TOUPPER "${GGML_GEMMINI_ACTIVATION_QUANT}" _POTAL_ACTIVATION_QUANT)
if (GGML_GEMMINI_DEFAULT_MATMUL_MODE STREQUAL "STRIPE_PIPELINE")
    if (NOT _POTAL_ACTIVATION_QUANT STREQUAL "EXSIA")
        message(FATAL_ERROR "GGML_POTAL: GGML_GEMMINI_DEFAULT_MATMUL_MODE=STRIPE_PIPELINE needs "
                            "GGML_GEMMINI_ACTIVATION_QUANT=EXSIA (got ${GGML_GEMMINI_ACTIVATION_QUANT})")
    endif()
    if (GGML_GEMMINI_ENABLE_RMD AND NOT GGML_GEMMINI_DEFAULT_RMD_BACKEND STREQUAL "WS")
        message(FATAL_ERROR "GGML_POTAL: STRIPE_PIPELINE with GGML_GEMMINI_ENABLE_RMD=ON runs the WS residual on PoTal; "
                            "GGML_GEMMINI_DEFAULT_RMD_BACKEND=${GGML_GEMMINI_DEFAULT_RMD_BACKEND} (cpu_direct) is FULL-only")
    endif()
elseif (GGML_GEMMINI_ENABLE_RMD AND NOT GGML_GEMMINI_DEFAULT_RMD_BACKEND STREQUAL "CPU")
    message(FATAL_ERROR "GGML_POTAL: FULL with GGML_GEMMINI_ENABLE_RMD=ON runs the CPU (cpu_direct) residual; "
                        "GGML_GEMMINI_DEFAULT_RMD_BACKEND=${GGML_GEMMINI_DEFAULT_RMD_BACKEND} (the PoTal WS residual) "
                        "runs in GGML_GEMMINI_DEFAULT_MATMUL_MODE=STRIPE_PIPELINE")
endif()
# Paired Main/Residual stripes (matmul/paired-auto.hpp, H2 indexed weight): the
# build default of the route; GGML_POTAL_PAIRED_AUTO=ON|OFF in the environment
# overrides it per process (matmul/options.hpp). Takes effect only in
# STRIPE_PIPELINE with compensation=ON; OFF keeps the serial Main then Residual GEMMs.
option(GGML_POTAL_PAIRED_AUTO "STRIPE_PIPELINE: run each stripe's Main and Residual GEMMs as one paired execution" OFF)
option(POTAL_PLANNING_DIAGNOSTICS "Compile Planning timers and ordinary new/delete call accounting" OFF)
if (POTAL_PLANNING_DIAGNOSTICS AND
    (NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR NOT CMAKE_SYSTEM_NAME STREQUAL "Linux"))
    message(FATAL_ERROR "POTAL_PLANNING_DIAGNOSTICS requires GNU C++ on Linux with linker --wrap")
endif()
set(POTAL_PLANNING_WRAP_OPTIONS
    "-Wl,--wrap=_Znwm" "-Wl,--wrap=_Znam"
    "-Wl,--wrap=_ZdlPv" "-Wl,--wrap=_ZdaPv"
    "-Wl,--wrap=_ZdlPvm" "-Wl,--wrap=_ZdaPvm")
if (GGML_POTAL_PAIRED_AUTO)
    set(_POTAL_PAIRED_AUTO_VALUE 1)
else()
    set(_POTAL_PAIRED_AUTO_VALUE 0)
endif()

if (NOT GGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY MATCHES "^[12]$")
    message(FATAL_ERROR "GGML_POTAL: GGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY must be 1 or 2 "
                        "(got ${GGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY})")
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
    matmul/attention.cpp
    matmul/attention-hw.cpp
    matmul/attention.hpp
    matmul/dense.cpp
    matmul/dense.hpp
    matmul/execution.cpp
    matmul/execution.hpp
    matmul/options.hpp
    matmul/pack.cpp
    matmul/pack.hpp
    matmul/paired.cpp
    matmul/paired.hpp
    matmul/paired-auto.cpp
    matmul/paired-auto.hpp
    matmul/paired-emulator.cpp
    matmul/paired-plan.cpp
    matmul/paired-schedule.cpp
    matmul/stripe.cpp
    matmul/stripe.hpp
    matmul/weight_cache.cpp
    matmul/weight_cache.hpp
    quants/weight/hp1/hp1.cpp
    quants/weight/hp1/hp1.hpp
    residual/stripe-residual.cpp
    residual/stripe-residual.hpp
    ${POTAL_GEMMINI_SHARED_SOURCES})
if (POTAL_PLANNING_DIAGNOSTICS)
    # Smoke and regression executables also compile backend sources. One shared
    # collector keeps their reset/dump calls and the registered backend's roots
    # in the same accounting instance, without relying on ELF interposition.
    add_library(potal-planning-profile SHARED
        matmul/planning-profile.cpp matmul/planning-profile.hpp)
    target_include_directories(potal-planning-profile PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_features(potal-planning-profile PRIVATE cxx_std_17)
    target_link_libraries(potal-planning-profile PRIVATE Threads::Threads)
    target_link_options(potal-planning-profile PRIVATE ${POTAL_PLANNING_WRAP_OPTIONS})
    set_target_properties(potal-planning-profile PROPERTIES
        CXX_VISIBILITY_PRESET default VISIBILITY_INLINES_HIDDEN OFF)
    set_source_files_properties(
        matmul/paired-auto.cpp matmul/planning-profile.cpp
        tests/potal-paired-smoke.cpp tests/potal-planning-profile-test.cpp
        PROPERTIES COMPILE_DEFINITIONS POTAL_PLANNING_DIAGNOSTICS)
endif()

# Compile definitions of the Gemmini backend (cmake/ggml-gemmini-config.cmake)
# without its execution-mode settings, plus the PoTal DIM. LOG_* / CYCLE_DETAIL
# must match the values ggml-gemmini-utils was built with.
set(POTAL_LOG_DEFINITIONS
    ${GGML_GEMMINI_RUNTIME_COMPILE_DEFS}
    GGML_GEMMINI_BLOCK_SIZE=${_GGML_GEMMINI_BLOCK_SIZE_VALUE}
    GGML_GEMMINI_EXSIA_SIGMA=${GGML_GEMMINI_EXSIA_SIGMA}
    GGML_GEMMINI_CONFIGURED_DIM=${POTAL_HW_DIM}
    GGML_POTAL_PAIRED_AUTO=${_POTAL_PAIRED_AUTO_VALUE}
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
    target_link_libraries(${target} PRIVATE potal-tiled-api potal-api ggml-gemmini-utils Threads::Threads)
    if (POTAL_PLANNING_DIAGNOSTICS)
        target_link_libraries(${target} PRIVATE potal-planning-profile)
        target_link_options(${target} PRIVATE ${POTAL_PLANNING_WRAP_OPTIONS})
    endif()
    if (GGML_GEMMINI_ENABLE_OPENMP)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(${target} PRIVATE GGML_GEMMINI_HAS_OPENMP)
    endif()
endfunction()

ggml_add_backend_library(ggml-potal ggml-potal.cpp ${POTAL_BACKEND_SOURCES})
potal_add_backend_settings(ggml-potal)
set_source_files_properties(matmul/attention.cpp matmul/attention-hw.cpp
    PROPERTIES COMPILE_OPTIONS "-fno-fast-math")

add_executable(potal-attention-test tests/potal-attention-test.cpp ${POTAL_BACKEND_SOURCES})
potal_add_backend_settings(potal-attention-test)
target_include_directories(potal-attention-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(potal-attention-test PRIVATE ggml)
if (NOT CMAKE_CROSSCOMPILING)
    add_test(NAME potal-attention-test COMMAND potal-attention-test)
endif()

add_executable(potal-paired-inventory tests/potal-paired-inventory.cpp ${POTAL_BACKEND_SOURCES})
potal_add_backend_settings(potal-paired-inventory)
target_include_directories(potal-paired-inventory PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(potal-paired-inventory PRIVATE ggml)

option(POTAL_E3_TESTS "Build isolated H3-6E3 acceptance executables" OFF)
if (POTAL_E3_TESTS)
    include(tests/e3/CMakeLists.txt)
endif()

option(POTAL_B5_TESTS "Build isolated QK limb reconstruction benchmark" OFF)
if (POTAL_B5_TESTS)
    option(POTAL_B5_SHARED_BASELINE "Use the checkpoint contract in the benchmark oracle" OFF)
    set(_b5_sources ${POTAL_BACKEND_SOURCES})
    list(REMOVE_ITEM _b5_sources matmul/attention.cpp)
    add_executable(potal-attention-bench tests/potal-attention-bench.cpp ${_b5_sources})
    potal_add_backend_settings(potal-attention-bench)
    target_include_directories(potal-attention-bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(potal-attention-bench PRIVATE ggml)
    set_source_files_properties(tests/potal-attention-bench.cpp PROPERTIES COMPILE_OPTIONS "-fno-fast-math")
    if (POTAL_B5_SHARED_BASELINE)
        target_compile_definitions(potal-attention-bench PRIVATE POTAL_B5_SHARED_BASELINE)
    endif()
endif()

option(POTAL_E4A_TESTS "Build isolated resident-B interface acceptance executables" OFF)
if (POTAL_E4A_TESTS)
    add_library(potal-resident-api STATIC tests/potal-resident.c)
    target_include_directories(potal-resident-api PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include ${POTAL_HW_GENERATED_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    set_target_properties(potal-resident-api PROPERTIES
        C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
    add_executable(potal-resident-test tests/potal-resident-test.cpp)
    target_compile_features(potal-resident-test PRIVATE cxx_std_17)
    target_link_libraries(potal-resident-test PRIVATE potal-resident-api)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv")
        target_sources(potal-resident-test PRIVATE ${POTAL_BACKEND_SOURCES}
            matmul/tiling.cpp matmul/stripe-api.cpp)
        target_include_directories(potal-resident-test PRIVATE ${POTAL_PRIVATE_INCLUDES})
        target_compile_definitions(potal-resident-test PRIVATE ${POTAL_LOG_DEFINITIONS})
        target_link_libraries(potal-resident-test PRIVATE ggml ggml-gemmini-utils Threads::Threads)
    else()
        add_test(NAME potal-resident-test COMMAND potal-resident-test)
    endif()
    add_executable(potal-resident-model-test tests/potal-resident-model-test.cpp ${POTAL_BACKEND_SOURCES})
    potal_add_backend_settings(potal-resident-model-test)
    target_include_directories(potal-resident-model-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(potal-resident-model-test PRIVATE ggml)
    if (NOT CMAKE_CROSSCOMPILING)
        add_test(NAME potal-resident-model-test COMMAND potal-resident-model-test)
    endif()
endif()

option(POTAL_E4B1_TESTS "Build software-only general resident contract tests" OFF)
if (POTAL_E4B1_TESTS)
    add_library(potal-e4b1-contract STATIC tests/potal-e4b1.cpp)
    target_compile_features(potal-e4b1-contract PUBLIC cxx_std_17)
    target_include_directories(potal-e4b1-contract PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(potal-e4b1-contract PUBLIC potal-api)
    add_executable(potal-e4b1-test tests/potal-e4b1-test.cpp)
    target_link_libraries(potal-e4b1-test PRIVATE potal-e4b1-contract)
    if (NOT CMAKE_CROSSCOMPILING)
        add_test(NAME potal-e4b1-test COMMAND potal-e4b1-test)
    endif()
endif()

add_executable(potal-full-inference tests/potal-full-inference.cpp)
target_include_directories(potal-full-inference PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../src ${CMAKE_CURRENT_SOURCE_DIR}/../../../common)
target_compile_features(potal-full-inference PRIVATE cxx_std_17)
target_link_libraries(potal-full-inference PRIVATE llama ggml)

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

# Hardware-free checks (host and target): the PoTal WS residual of one packet
# (run-aware rows/runs, K32 padded lowering, gather from the prepared weight)
# against an independent compact-run SCU/Sat32 golden, radix INT64
# reconstruction, the prepared checked merge against the Gemmini merge
# (bitwise) without weight revalidation, failures.
add_executable(potal-residual-unit-test tests/potal-residual-unit-test.cpp ${POTAL_BACKEND_SOURCES})
potal_add_backend_settings(potal-residual-unit-test)
target_include_directories(potal-residual-unit-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
# The Gemmini reader's call counters (this target only): the prepared merge never validates the weight.
target_compile_definitions(potal-residual-unit-test PRIVATE GGML_GEMMINI_TESTING)
target_link_libraries(potal-residual-unit-test PRIVATE ggml)
if (NOT CMAKE_CROSSCOMPILING)
    add_test(NAME potal-residual-unit-test COMMAND potal-residual-unit-test)
endif()

# Hardware-free checks (host and target): the stripe pipeline with real ExSIA
# stripe-ready events, the production queue/consumer on an emulator thread:
# overlap and backpressure gates, semantic capacity, serialized == pipelined,
# immutability of accepted stripes, failure wake-up, no partial commit, the
# event order and debug JSONL order of every run, no weight revalidation.
add_executable(potal-stripe-pipeline-test tests/potal-stripe-pipeline-test.cpp ${POTAL_BACKEND_SOURCES})
potal_add_backend_settings(potal-stripe-pipeline-test)
target_include_directories(potal-stripe-pipeline-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
# The Gemmini reader's call counters (this target only): no weight revalidation inside a run.
target_compile_definitions(potal-stripe-pipeline-test PRIVATE GGML_GEMMINI_TESTING)
target_link_libraries(potal-stripe-pipeline-test PRIVATE ggml)
if (NOT CMAKE_CROSSCOMPILING)
    add_test(NAME potal-stripe-pipeline-test COMMAND potal-stripe-pipeline-test)
endif()

# Hardware-free checks (host and target) of the paired Main/Residual GEMM
# software model (matmul/paired.hpp): no RoCC command, PAIRED_EMULATED only.
#   plan      paired SP/ACC partition, frozen t_I/t_J, K32 common windows, the
#             shared indexed-weight Residual descriptor, capacity failures
#   schedule  the micro-tile trace: 1:1 alternation, tails, K order, group
#             reversal, save/restore, determinism
#   unit      Main / Residual INT32 and radix INT64: independent golden ==
#             serialized reference == paired emulator; both merge modes,
#             failures with no partial commit, real ExSIA packets
#   auto      H3-0 / H3-1 planner of potal_paired_matmul_auto (matmul/paired-auto.hpp)
#             on the H2 contract: Main I/J kept, final K = the most restrictive
#             limit, bank / ACC / Index / scale occupancy, K panels and Index
#             Windows, resident K coordinates, refusals with diagnostics; the
#             execution plan (ops, dependencies, pair residency, Index entries,
#             scale lifetimes, ACC first/last) and its validator
#   execution H3-2B.1 potal_paired_matmul_auto on the command-level H2 emulator:
#             translation (LOOP_WS / INDEX / scale / readback counts and order),
#             residual staging, Main / Residual INT32, radix INT64 and merged
#             F32 == the serialized reference, pairs and generations, failures
foreach(_paired_test plan schedule unit auto execution)
    add_executable(potal-paired-${_paired_test}-test tests/potal-paired-${_paired_test}-test.cpp ${POTAL_BACKEND_SOURCES})
    potal_add_backend_settings(potal-paired-${_paired_test}-test)
    target_include_directories(potal-paired-${_paired_test}-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(potal-paired-${_paired_test}-test PRIVATE ggml)
    if (NOT CMAKE_CROSSCOMPILING)
        add_test(NAME potal-paired-${_paired_test}-test COMMAND potal-paired-${_paired_test}-test)
    endif()
endforeach()

if (POTAL_PLANNING_DIAGNOSTICS)
    add_executable(potal-planning-profile-test
        tests/potal-planning-profile-test.cpp)
    potal_add_backend_settings(potal-planning-profile-test)
    if (NOT CMAKE_CROSSCOMPILING)
        add_test(NAME potal-planning-profile-test COMMAND potal-planning-profile-test)
    endif()
endif()

# Counts real Residual decode calls in the fallback regression, not elapsed time.
target_compile_definitions(potal-paired-execution-test PRIVATE GGML_GEMMINI_TESTING)

# Benchmark (host and target, not a test): Q8_HP1 weight preparation cost per
# stage and with the backend's weight cache, synthetic or from a GGUF (--gguf).
# GGML_GEMMINI_TESTING enables the Gemmini reader's call counters in this target only.
add_executable(potal-weight-prep-bench tests/potal-weight-prep-bench.cpp ${POTAL_BACKEND_SOURCES})
potal_add_backend_settings(potal-weight-prep-bench)
target_compile_definitions(potal-weight-prep-bench PRIVATE GGML_GEMMINI_TESTING)
target_link_libraries(potal-weight-prep-bench PRIVATE ggml)

# Benchmark (host and target, not a test): one stripe's residual merge, the
# prepared HP1 merge against the Gemmini checked merge it replaced, on a real
# correction (EXSIA, RMD ON); merge time, reader validations per merge
# (GGML_GEMMINI_TESTING counters, this target only) and bitwise agreement.
add_executable(potal-residual-merge-bench tests/potal-residual-merge-bench.cpp ${POTAL_BACKEND_SOURCES})
potal_add_backend_settings(potal-residual-merge-bench)
target_include_directories(potal-residual-merge-bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_compile_definitions(potal-residual-merge-bench PRIVATE GGML_GEMMINI_TESTING)
target_link_libraries(potal-residual-merge-bench PRIVATE ggml)

# FireSim hardware test: real GGML MUL_MAT graphs computed by the POTAL backend.
if (CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv")
    add_executable(potal-attention-smoke tests/potal-attention-smoke.cpp ${POTAL_BACKEND_SOURCES})
    potal_add_backend_settings(potal-attention-smoke)
    target_include_directories(potal-attention-smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(potal-attention-smoke PRIVATE ggml)
    target_link_options(potal-attention-smoke PRIVATE
        "-Wl,--wrap=potal_loop_ws_issue" "-Wl,--wrap=potal_gemm_ws_issue")

    add_executable(potal-ggml-smoke tests/potal-ggml-smoke.cpp)
    target_link_libraries(potal-ggml-smoke PRIVATE ggml)

    # FireSim: F32 x Q8_HP1 MUL_MAT graphs through the routed, tiled PoTal path,
    # checked element-exactly against the CPU integer golden, plus generation
    # wrap and DMA buffer reallocation. Golden helpers come from the backend sources.
    add_executable(potal-hp1-smoke tests/potal-hp1-smoke.cpp ${POTAL_BACKEND_SOURCES})
    potal_add_backend_settings(potal-hp1-smoke)
    target_link_libraries(potal-hp1-smoke PRIVATE ggml)

    # FireSim: STRIPE_PIPELINE MUL_MAT graphs (ExSIA, Main and Residual GEMMs on
    # the hart-0 issuer), checked element-exactly against a CPU golden built
    # from the same ExSIA codes, theta and packets.
    add_executable(potal-stripe-smoke tests/potal-stripe-smoke.cpp ${POTAL_BACKEND_SOURCES})
    potal_add_backend_settings(potal-stripe-smoke)
    target_include_directories(potal-stripe-smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(potal-stripe-smoke PRIVATE ggml)

    # FireSim: the paired route (GGML_POTAL_PAIRED_AUTO) on the H2 hardware, same
    # golden as potal-stripe-smoke, each case paired and serial (measured timing).
    add_executable(potal-paired-smoke tests/potal-paired-smoke.cpp ${POTAL_BACKEND_SOURCES})
    potal_add_backend_settings(potal-paired-smoke)
    target_include_directories(potal-paired-smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(potal-paired-smoke PRIVATE ggml)
endif()
