# Standalone Gemmini NPU API (C11) and the weight-stationary tiler built on it.
# Needs only the generated npu_config.h / gemmini_params.h
# (cmake/npu-config.cmake): no GGML, llama.cpp or external Gemmini SDK.
if (NOT DEFINED NPU_GENERATED_DIR OR NOT GGML_NPU STREQUAL "GEMMINI")
    message(FATAL_ERROR "gemmini-api needs cmake/npu-config.cmake with GGML_NPU=GEMMINI")
endif()

set(GEMMINI_API_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(GEMMINI_BACKEND_DIR "${CMAKE_CURRENT_LIST_DIR}/..")

add_library(gemmini-api STATIC
    gemmini.c
    encode.c
    validate.c
    counter.c
    transport/rocc.c
    transport/recording.c)
target_include_directories(gemmini-api PUBLIC
    "${GEMMINI_API_DIR}/include"
    "${NPU_GENERATED_DIR}")
set_target_properties(gemmini-api PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON)
if (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(gemmini-api PRIVATE -Wall -Wextra -Werror=implicit-function-declaration)
endif()

add_library(gemmini-tiled-api STATIC
    "${GEMMINI_BACKEND_DIR}/matmul/tiling.cpp"
    "${GEMMINI_BACKEND_DIR}/matmul/tiling.hpp")
target_link_libraries(gemmini-tiled-api PUBLIC gemmini-api)
target_include_directories(gemmini-tiled-api PUBLIC "${GEMMINI_BACKEND_DIR}/matmul")
target_compile_features(gemmini-tiled-api PUBLIC cxx_std_17)
set_target_properties(gemmini-tiled-api PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(GEMMINI_API_TEST_DIR "${GEMMINI_BACKEND_DIR}/tests")
if (NOT CMAKE_CROSSCOMPILING)
    foreach(_test IN ITEMS gemmini-api-test gemmini-encoding-test gemmini-transport-test)
        add_executable(${_test} "${GEMMINI_API_TEST_DIR}/${_test}.c")
        target_link_libraries(${_test} PRIVATE gemmini-api)
        set_target_properties(${_test} PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
        add_test(NAME ${_test} COMMAND ${_test})
    endforeach()
    add_executable(gemmini-tiling-test "${GEMMINI_API_TEST_DIR}/gemmini-tiling-test.cpp")
    target_link_libraries(gemmini-tiling-test PRIVATE gemmini-tiled-api)
    target_include_directories(gemmini-tiling-test PRIVATE "${GEMMINI_API_TEST_DIR}/golden")
    add_test(NAME gemmini-tiling-test
             COMMAND gemmini-tiling-test "${GEMMINI_API_TEST_DIR}/golden")
    get_filename_component(_npu_source_dir "${GEMMINI_BACKEND_DIR}/../../.." ABSOLUTE)
    add_test(NAME npu-config-test
             COMMAND "${CMAKE_COMMAND}" -DSOURCE_DIR=${_npu_source_dir}
                     -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/npu-config-test -DGENERATOR=${CMAKE_GENERATOR}
                     -P "${_npu_source_dir}/ggml/src/ggml-npu-common/tests/npu-config-test.cmake")
endif()

# Cross-compiled RISC-V smoke ELF: issues real RoCC commands, so it is never
# registered with the native CTest run.
if (CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv64$")
    add_executable(gemmini-api-smoke "${GEMMINI_API_TEST_DIR}/gemmini-api-smoke.c")
    target_link_libraries(gemmini-api-smoke PRIVATE gemmini-api)
    set_target_properties(gemmini-api-smoke PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
endif()
