if(NOT DEFINED TEST_SOURCE_DIR)
    message(FATAL_ERROR "TEST_SOURCE_DIR is required")
endif()

function(expect_gemmini script expected)
    execute_process(
        COMMAND env -u GGML_GEMMINI BUILD_JOBS=1 bash
            "${TEST_SOURCE_DIR}/${script}" --dry-run ${ARGN}
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "${script} rejected GGML_GEMMINI=${expected}: ${error}")
    endif()
    string(REGEX MATCHALL "-DGGML_GEMMINI(:[A-Za-z]+)?=(ON|OFF)" choices "${output}")
    list(GET choices -1 enabled)
    if(NOT enabled MATCHES "=${expected}$")
        message(FATAL_ERROR "${script} expected GGML_GEMMINI=${expected}: ${output}")
    endif()

endfunction()

expect_gemmini(build-arm64.sh OFF)
expect_gemmini(build-arm64.sh ON -DGGML_GEMMINI=ON)
expect_gemmini(build-arm64.sh OFF -DGGML_GEMMINI=OFF)
expect_gemmini(build-arm64-cpu.sh OFF -DGGML_GEMMINI=OFF)
expect_gemmini(build-x86.sh OFF -DGGML_GEMMINI=OFF)
expect_gemmini(build-riscv.sh OFF -DGGML_GEMMINI=OFF)

string(RANDOM LENGTH 8 ALPHABET 0123456789abcdef suffix)
set(build_dir "${CMAKE_CURRENT_BINARY_DIR}/gemmini-off-${suffix}")
set(stale_backend "${build_dir}/bin/libggml-gemmini.so")
file(MAKE_DIRECTORY "${build_dir}/bin")
file(WRITE "${stale_backend}" "stale")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${TEST_SOURCE_DIR}" -B "${build_dir}"
        -DGGML_GEMMINI=OFF -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON
        -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_OPENMP=OFF
        -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
        -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "GEMMINI-off configure failed: ${output}\n${error}")
endif()
if(EXISTS "${stale_backend}")
    message(FATAL_ERROR "GEMMINI-off configure retained stale backend: ${stale_backend}")
endif()
file(REMOVE_RECURSE "${build_dir}")
