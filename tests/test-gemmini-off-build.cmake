if(NOT DEFINED TEST_SOURCE_DIR)
    message(FATAL_ERROR "TEST_SOURCE_DIR is required")
endif()

function(expect_gemmini script expected)
    execute_process(
        COMMAND env -u GGML_GEMMINI BUILD_JOBS=1 bash
            "${TEST_SOURCE_DIR}/${script}" --dry-run ${ARGN}
        RESULT_VARIABLE result
        ERROR_VARIABLE error)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "${script} rejected GGML_GEMMINI=${expected}: ${error}")
    endif()
    string(REGEX MATCH "IM2P_EFFECTIVE_CONFIG=([^\n]+)" summary "${error}")
    string(JSON enabled GET "${CMAKE_MATCH_1}" effective GGML_GEMMINI)
    if(NOT enabled STREQUAL "${expected}")
        message(FATAL_ERROR "${script} expected GGML_GEMMINI=${expected}: ${summary}")
    endif()
    if(script STREQUAL "build-arm64.sh" AND expected STREQUAL "ON" AND
       CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
        string(JSON option GET "${CMAKE_MATCH_1}" effective GGML_GEMMINI_OPTION)
        string(JSON rmd GET "${CMAKE_MATCH_1}" effective GGML_GEMMINI_DEFAULT_RMD_BACKEND)
        if(NOT option STREQUAL "CPU" OR NOT rmd STREQUAL "CPU")
            message(FATAL_ERROR "macOS default requires CPU+RMD CPU, got ${option}+RMD ${rmd}")
        endif()
    endif()
endfunction()

expect_gemmini(build-arm64.sh OFF)
expect_gemmini(build-arm64.sh ON -DGGML_GEMMINI=ON)
expect_gemmini(build-arm64.sh OFF -DGGML_GEMMINI=OFF)
expect_gemmini(build-arm64-cpu.sh OFF -DGGML_GEMMINI=OFF)
expect_gemmini(build-x86.sh OFF -DGGML_GEMMINI=OFF)
expect_gemmini(build-riscv.sh OFF -DGGML_GEMMINI=OFF)

function(expect_arm64_backends expected_cuda expected_metal)
    execute_process(
        COMMAND env -u GGML_CUDA -u GGML_METAL -u GGML_CUDA_DEFAULT -u GGML_METAL_DEFAULT
            BUILD_JOBS=1 bash "${TEST_SOURCE_DIR}/build-arm64.sh" --dry-run ${ARGN}
        RESULT_VARIABLE result
        ERROR_VARIABLE error)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "ARM64 backend resolution failed: ${error}")
    endif()
    string(REGEX MATCH "IM2P_EFFECTIVE_CONFIG=([^\n]+)" summary "${error}")
    string(JSON cuda GET "${CMAKE_MATCH_1}" effective GGML_CUDA)
    string(JSON metal GET "${CMAKE_MATCH_1}" effective GGML_METAL)
    if(NOT cuda STREQUAL expected_cuda OR NOT metal STREQUAL expected_metal)
        message(FATAL_ERROR "Expected CUDA=${expected_cuda}/Metal=${expected_metal}: ${summary}")
    endif()
endfunction()

if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
    expect_arm64_backends(OFF ON)
else()
    expect_arm64_backends(ON OFF)
endif()
expect_arm64_backends(ON OFF -DGGML_CUDA:BOOL=ON -DGGML_METAL:BOOL=OFF)
expect_arm64_backends(OFF ON -DGGML_CUDA=OFF -DGGML_METAL=ON)

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
