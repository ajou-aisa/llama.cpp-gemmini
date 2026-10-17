# PoTal RoCC API (potal-api) and its tests. Two configurations, one parameter
# contract (cmake/potal-hw-params.cmake):
#   POTAL_API_ONLY=ON  added by the root CMakeLists: only the API and its tests,
#                      independent of ggml, llama, GEMMINI_SW_PATH and IM2P.sim.
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

find_package(Threads REQUIRED)
ggml_add_backend_library(ggml-potal
                         ggml-potal.cpp
                         matmul/execution.cpp
                         matmul/execution.hpp
                         matmul/pack.cpp
                         matmul/pack.hpp)
target_link_libraries(ggml-potal PRIVATE potal-api Threads::Threads)

# Hardware-free checks (host and target): registration, supports_op contract,
# packing against a CPU-backend golden, scheduler placement, no fake success.
# pack.cpp is compiled in because the packing functions are internal.
add_executable(potal-ggml-unit-test tests/potal-ggml-unit-test.cpp matmul/pack.cpp)
target_include_directories(potal-ggml-unit-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(potal-ggml-unit-test PRIVATE ggml)
if (NOT CMAKE_CROSSCOMPILING)
    add_test(NAME potal-ggml-unit-test COMMAND potal-ggml-unit-test)
endif()

# FireSim hardware test: real GGML MUL_MAT graphs computed by the POTAL backend.
if (CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv")
    add_executable(potal-ggml-smoke tests/potal-ggml-smoke.cpp)
    target_link_libraries(potal-ggml-smoke PRIVATE ggml)
endif()
