# Standalone PoTal RoCC API, configured only with POTAL_API_ONLY=ON.
# Independent of ggml, llama, GEMMINI_SW_PATH and IM2P.sim.

include(${PROJECT_SOURCE_DIR}/cmake/potal-hw-params.cmake)

set(POTAL_HW_PARAMS "" CACHE FILEPATH "RTL-generated PoTal parameter header (absolute path)")
set(POTAL_EXPECTED_DIM "" CACHE STRING "DIM the selected header must report (16|32|64)")
set(POTAL_HW_CONFIG "" CACHE STRING "Chipyard config that produced POTAL_HW_PARAMS (recorded only)")

potal_configure_hw_params("${CMAKE_BINARY_DIR}/generated")

add_library(potal-api STATIC potal.c)
target_include_directories(potal-api PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${POTAL_HW_GENERATED_DIR})
set_target_properties(potal-api PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF)

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
    else()
        message(STATUS "potal-api-smoke not built: GEMM execution is validated for DIM16 only (header DIM=${POTAL_HW_DIM})")
    endif()
endif()
