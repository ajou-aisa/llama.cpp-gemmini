llama_add_compile_flags()
add_compile_definitions(GGML_GEMMINI_BLOCK_SIZE=${_GGML_GEMMINI_BLOCK_SIZE_VALUE})

add_executable(test-cycle-sim-reader test-cycle-sim-reader.cpp)
target_compile_definitions(test-cycle-sim-reader PRIVATE LOG_CYCLE=${LOG_CYCLE} CYCLE_SIM=${CYCLE_SIM})
target_link_libraries(test-cycle-sim-reader PRIVATE ggml-gemmini-utils)
add_test(NAME test-cycle-sim-reader COMMAND test-cycle-sim-reader)
if(LOG_CYCLE OR CYCLE_SIM)
    add_executable(test-gemmini-semantic test-gemmini-semantic.cpp)
    target_compile_features(test-gemmini-semantic PRIVATE cxx_std_17)
    target_compile_options(test-gemmini-semantic PRIVATE -UNDEBUG)
    target_compile_definitions(test-gemmini-semantic PRIVATE LOG_CYCLE=${LOG_CYCLE} CYCLE_SIM=${CYCLE_SIM})
    target_link_libraries(test-gemmini-semantic PRIVATE ggml-gemmini-utils)
    add_test(NAME test-gemmini-semantic COMMAND "${CMAKE_COMMAND}"
        "-DTEST_EXECUTABLE=$<TARGET_FILE:test-gemmini-semantic>"
        "-DTEST_OUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/test-gemmini-semantic.cmake")
endif()
add_executable(test-functional-emulation-exclusion test-functional-emulation-exclusion.cpp)
target_compile_features(test-functional-emulation-exclusion PRIVATE cxx_std_17)
target_compile_definitions(test-functional-emulation-exclusion PRIVATE
    CYCLE_SIM=${CYCLE_SIM} LOG_CYCLE=${LOG_CYCLE} CYCLE_DETAIL=${CYCLE_DETAIL})
target_compile_options(test-functional-emulation-exclusion PRIVATE -UNDEBUG)
target_link_libraries(test-functional-emulation-exclusion PRIVATE ggml-gemmini-utils Threads::Threads)
add_test(NAME test-functional-emulation-exclusion COMMAND test-functional-emulation-exclusion)
if(TARGET ggml-cpu AND NOT GGML_BACKEND_DL)
    add_executable(test-cycle-sim-cpu-smoke test-ggml-cpu-cycle-sink.cpp)
    target_compile_features(test-cycle-sim-cpu-smoke PRIVATE cxx_std_17)
    target_compile_definitions(test-cycle-sim-cpu-smoke PRIVATE
        LOG_CYCLE=${LOG_CYCLE} CYCLE_SIM=${CYCLE_SIM}
        CYCLE_DETAIL=${CYCLE_DETAIL}
        EXPECT_LOG_CYCLE=${LOG_CYCLE}
        EXPECT_CPU_CYCLE_LOG=$<IF:$<BOOL:${GGML_CPU_CYCLE_LOG}>,1,0>)
    target_link_libraries(test-cycle-sim-cpu-smoke PRIVATE ggml-base ggml-cpu ggml-gemmini-utils)
    add_test(NAME test-cycle-sim-cpu-smoke COMMAND "${CMAKE_COMMAND}"
        "-DTEST_EXECUTABLE=$<TARGET_FILE:test-cycle-sim-cpu-smoke>"
        "-DTEST_OUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/test-cycle-sim-cpu-smoke.cmake")
endif()
if(CYCLE_SIM)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_executable(test-gemmini-cycle-sim-log test-gemmini-cycle-sim-log.cpp)
    target_compile_options(test-gemmini-cycle-sim-log PRIVATE -UNDEBUG)
    target_compile_definitions(test-gemmini-cycle-sim-log PRIVATE CYCLE_SIM=1 LOG_CYCLE=${LOG_CYCLE})
    target_include_directories(test-gemmini-cycle-sim-log PRIVATE "${IM2P_SIM_ROOT}/sim/include")
    target_link_libraries(test-gemmini-cycle-sim-log PRIVATE ggml-gemmini-utils)
    add_test(NAME test-gemmini-cycle-sim-log COMMAND test-gemmini-cycle-sim-log)
    add_test(NAME test-gemmini-cycle-sim-log-failure COMMAND test-gemmini-cycle-sim-log --failure)
    add_executable(test-cycle-sim-coverage test-cycle-sim-coverage.cpp)
    target_compile_features(test-cycle-sim-coverage PRIVATE cxx_std_17)
    target_compile_options(test-cycle-sim-coverage PRIVATE -UNDEBUG)
    target_compile_definitions(test-cycle-sim-coverage PRIVATE CYCLE_SIM=1)
    target_include_directories(test-cycle-sim-coverage PRIVATE
        "${IM2P_SIM_ROOT}/sim/include" "${PROJECT_SOURCE_DIR}/ggml/src")
    target_link_libraries(test-cycle-sim-coverage PRIVATE ggml ggml-cpu ggml-gemmini-utils)
    add_test(NAME test-cycle-sim-coverage COMMAND test-cycle-sim-coverage)
    if(TARGET ggml-gemmini-cpu-functional)
        add_executable(test-gemmini-cycle-sim-dispatch test-gemmini-cycle-sim-dispatch.cpp)
        target_compile_features(test-gemmini-cycle-sim-dispatch PRIVATE cxx_std_17)
        target_compile_definitions(test-gemmini-cycle-sim-dispatch PRIVATE
            ${GGML_GEMMINI_COMPILE_DEFS} GGML_GEMMINI_TESTING=1)
        target_include_directories(test-gemmini-cycle-sim-dispatch PRIVATE
            ../ggml/src ../ggml/src/ggml-gemmini "${IM2P_SIM_ROOT}/sim/include"
            "${IM2P_SIM_ROOT}/frontend/include" "${GEMMINI_SW_PATH}" "${GEMMINI_SW_PATH}/include")
        target_link_libraries(test-gemmini-cycle-sim-dispatch PRIVATE
            ggml-gemmini ggml-base ggml-gemmini-utils ggml-gemmini-cpu-functional)
        foreach(mode IN ITEMS full pipeline residual residual-runs large-k fail-full fail-residual
                              fail-publish-full fail-publish-residual)
            add_test(NAME test-gemmini-cycle-sim-dispatch-${mode} COMMAND "${CMAKE_COMMAND}"
                "-DTEST_EXECUTABLE=$<TARGET_FILE:test-gemmini-cycle-sim-dispatch>"
                "-DTEST_MODE=${mode}" "-DTEST_OUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}"
                "-DTEST_PYTHON=${Python3_EXECUTABLE}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/test-cycle-sim-dispatch.cmake")
        endforeach()
        add_executable(test-cpu-functional-contract "${IM2P_SIM_ROOT}/frontend/tests/cpu_functional_contract.cpp")
        target_compile_features(test-cpu-functional-contract PRIVATE cxx_std_17)
        target_compile_options(test-cpu-functional-contract PRIVATE -UNDEBUG)
        target_compile_definitions(test-cpu-functional-contract PRIVATE
            IM2P_ACTIVATION_BITS=${GGML_GEMMINI_ACTIVATION_BITS}
            IM2P_WEIGHT_BITS=${GGML_GEMMINI_WEIGHT_BITS}
            IM2P_DIM=${GGML_GEMMINI_DIM})
        target_include_directories(test-cpu-functional-contract PRIVATE
            "${IM2P_SIM_ROOT}/frontend/include" "${IM2P_SIM_ROOT}/sim/include")
        target_link_libraries(test-cpu-functional-contract PRIVATE ggml-gemmini-cpu-functional)
        add_test(NAME test-cpu-functional-contract COMMAND test-cpu-functional-contract)
    endif()
endif()
if(TARGET ggml-gemmini AND TARGET ggml-cpu AND LLAMA_BUILD_TOOLS)
    add_test(NAME test-cycle-sim-build-contract COMMAND "${CMAKE_COMMAND}"
        "-DTEST_SOURCE_DIR=${CMAKE_SOURCE_DIR}" "-DTEST_BUILD_DIR=${CMAKE_BINARY_DIR}"
        "-DTEST_CYCLE_SIM=${CYCLE_SIM}" "-DTEST_LOG_CYCLE=${LOG_CYCLE}"
        "-DTEST_CPU_CYCLE_LOG=$<IF:$<BOOL:${GGML_CPU_CYCLE_LOG}>,1,0>"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/test-cycle-sim-build-contract.cmake")
endif()
if(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND
   IM2P_SIM_IMPLEMENTATION STREQUAL "GEMMINI_HP1" AND NOT CYCLE_SIM)
    add_test(NAME test-gemmini-numerical-memory-contract COMMAND "${CMAKE_COMMAND}"
        "-DTEST_MANIFEST=${GGML_GEMMINI_IM2P_MANIFEST}"
        "-DTEST_HEADER=${CMAKE_BINARY_DIR}/generated/gemmini_params.h"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/test-gemmini-memory-contract.cmake")
endif()

function(llama_build source)
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${source})
    target_link_libraries(${TEST_TARGET} PRIVATE common)
    install(TARGETS ${TEST_TARGET} RUNTIME)
endfunction()

function(llama_test target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    set(TEST_TARGET ${target})

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# Builds and runs a test source file.
# Optional args:
# - NAME: name of the executable & test target (defaults to the source file name without extension)
# - LABEL: label for the test (defaults to main)
# - ARGS: arguments to pass to the test executable
# - WORKING_DIRECTORY
function(llama_build_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${source} get-model.cpp)
    install(TARGETS ${TEST_TARGET} RUNTIME)
    target_link_libraries(${TEST_TARGET} PRIVATE common)

    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# build test-tokenizer-0 target once and add many tests
llama_build(test-tokenizer-0.cpp)

llama_test(test-tokenizer-0 NAME test-tokenizer-0-bert-bge          ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-bert-bge.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-command-r         ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-command-r.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-coder    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-deepseek-coder.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-llm      ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-deepseek-llm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-falcon            ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-falcon.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gpt-2             ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-gpt-2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-bpe         ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama-bpe.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-spm         ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama-spm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-mpt               ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-mpt.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-phi-3             ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-phi-3.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen2             ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-qwen2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-refact            ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-refact.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder         ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-starcoder.gguf)

if (LLAMA_LLGUIDANCE)
    llama_build_and_test(test-grammar-llguidance.cpp ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama-bpe.gguf)
endif ()

if (NOT WIN32)
    if (NOT DEFINED BASH_EXECUTABLE)
        find_program(BASH_EXECUTABLE NAMES bash)
    endif()
    if (BASH_EXECUTABLE AND EXISTS "${BASH_EXECUTABLE}")
        add_test(NAME test-experiment-output
            COMMAND ${BASH_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-experiment-output.sh)
    endif()

    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API
    llama_build_and_test(test-sampling.cpp)
    llama_build_and_test(test-grammar-parser.cpp)
    llama_build_and_test(test-grammar-integration.cpp)
    llama_build_and_test(test-llama-grammar.cpp)
    llama_build_and_test(test-chat.cpp)
    # TODO: disabled on loongarch64 because the ggml-ci node lacks Python 3.8
    if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "loongarch64")
        llama_build_and_test(test-json-schema-to-grammar.cpp   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
        target_include_directories(test-json-schema-to-grammar PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../tools/server)
    endif()

    if (NOT GGML_BACKEND_DL)
        llama_build(test-quantize-stats.cpp)
    endif()

    llama_build(test-gbnf-validator.cpp)

    # build test-tokenizer-1-bpe target once and add many tests
    llama_build(test-tokenizer-1-bpe.cpp)

    # TODO: disabled due to slowness
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-aquila    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-aquila.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-falcon    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-falcon.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-2     ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-gpt-2.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-neox  ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-gpt-neox.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-llama-bpe ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama-bpe.gguf --ignore-merges)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-mpt       ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-mpt.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-refact    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-refact.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-starcoder ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-starcoder.gguf)

    # build test-tokenizer-1-spm target once and add many tests
    llama_build(test-tokenizer-1-spm.cpp)

    llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-llama-spm ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama-spm.gguf)
    #llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-baichuan  ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-baichuan.gguf)

    # llama_build_and_test(test-double-float.cpp) # SLOW
endif()

llama_build_and_test(test-log.cpp)
llama_build_and_test(test-chat-template.cpp)
llama_build_and_test(test-regex-partial.cpp)

# this fails on windows (github hosted runner) due to curl DLL not found (exit code 0xc0000135)
if (NOT WIN32)
    llama_build_and_test(test-arg-parser.cpp)
endif()
if (LLAMA_BUILD_TOOLS AND LOG_CYCLE)
    add_test(NAME test-gemmini-cycle-cli
        COMMAND ${CMAKE_COMMAND}
            -DTEST_CLI=$<TARGET_FILE:llama-cli>
            -DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/gemmini-cycle-cli
            -P ${CMAKE_CURRENT_SOURCE_DIR}/test-gemmini-cycle-cli.cmake)
endif()
add_test(NAME test-gemmini-cpu-cycle-validity-contract
    COMMAND ${CMAKE_COMMAND}
        -DCPU_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-cpu/ggml-cpu.c
        -DGEMMINI_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-gemmini/ggml-gemmini.cpp
        -DCPU_CMAKE=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-cpu/CMakeLists.txt
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test-gemmini-cpu-cycle-validity-contract.cmake)
add_test(NAME test-build-arm64-log-defaults
    COMMAND ${CMAKE_COMMAND}
        -DTEST_SCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/../build-arm64.sh
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test-build-arm64-log-defaults.cmake)
add_test(NAME test-build-arm64-cpu-detection
    COMMAND ${CMAKE_COMMAND}
        -DTEST_SCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/../build-arm64.sh
        -DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/build-arm64-cpu-detection
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test-build-arm64-cpu-detection.cmake)
add_test(NAME test-im2p-cache-build-scripts
    COMMAND ${CMAKE_COMMAND}
        -DTEST_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/..
        -DTEST_IM2P_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/../../IM2P.sim
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test-im2p-cache-build-scripts.cmake)

# llama_build_and_test(test-opt.cpp) # SLOW
llama_build_and_test(test-gguf.cpp)
llama_build_and_test(test-backend-ops.cpp)
add_executable(test-gemmini-log test-gemmini-log.cpp)
target_link_libraries(test-gemmini-log PRIVATE ggml-gemmini-utils llama)
target_compile_features(test-gemmini-log PRIVATE cxx_std_17)
target_compile_definitions(test-gemmini-log PRIVATE
    EXPECT_LOG_DEBUG=$<IF:$<BOOL:${LOG_DEBUG}>,1,0>
    EXPECT_LOG_CYCLE=$<IF:$<BOOL:${LOG_CYCLE}>,1,0>
    EXPECT_CYCLE_DETAIL=$<IF:$<BOOL:${CYCLE_DETAIL}>,1,0>)
target_include_directories(test-gemmini-log PRIVATE
    ../ggml/src/ggml-gemmini-utils/include
    ../src)
add_test(NAME test-gemmini-log
    COMMAND $<TARGET_FILE:test-gemmini-log> ${CMAKE_CURRENT_BINARY_DIR}/gemmini-log-regression)
add_test(NAME test-llama-log-callback
    COMMAND $<TARGET_FILE:test-gemmini-log> --llama-callback)
add_executable(test-gemmini-optrace test-gemmini-optrace.cpp)
target_link_libraries(test-gemmini-optrace PRIVATE ggml-gemmini-utils)
target_compile_features(test-gemmini-optrace PRIVATE cxx_std_17)
target_compile_definitions(test-gemmini-optrace PRIVATE CYCLE_SIM=${CYCLE_SIM})
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(test-gemmini-optrace PRIVATE -UNDEBUG)
endif()
add_test(NAME test-gemmini-optrace
    COMMAND $<TARGET_FILE:test-gemmini-optrace> ${CMAKE_CURRENT_BINARY_DIR}/gemmini-optrace)
add_executable(test-gemmini-log-boundary
    test-gemmini-log-boundary.cpp
    test-gemmini-log-c-boundary.c)
target_link_libraries(test-gemmini-log-boundary PRIVATE ggml-gemmini-utils)
target_compile_features(test-gemmini-log-boundary PRIVATE cxx_std_17)
target_compile_definitions(test-gemmini-log-boundary PRIVATE
    EXPECT_LOG_DEBUG=$<IF:$<BOOL:${LOG_DEBUG}>,1,0>
    EXPECT_LOG_CYCLE=$<IF:$<BOOL:${LOG_CYCLE}>,1,0>
    EXPECT_CYCLE_DETAIL=$<IF:$<BOOL:${CYCLE_DETAIL}>,1,0>
    LOG_CYCLE=$<IF:$<BOOL:${LOG_CYCLE}>,1,0>
    CYCLE_DETAIL=$<IF:$<BOOL:${CYCLE_DETAIL}>,1,0>)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
    CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    target_include_directories(test-gemmini-log-boundary PRIVATE
        ../ggml/src/ggml-gemmini-utils/src)
endif()
add_test(NAME test-gemmini-log-boundary
    COMMAND $<TARGET_FILE:test-gemmini-log-boundary>)
add_executable(test-gemmini-host-timing test-gemmini-host-timing.cpp)
find_package(Threads REQUIRED)
target_link_libraries(test-gemmini-log-boundary PRIVATE Threads::Threads)
target_link_libraries(test-gemmini-host-timing PRIVATE ggml-gemmini-utils Threads::Threads)
target_compile_features(test-gemmini-host-timing PRIVATE cxx_std_17)
target_compile_definitions(test-gemmini-host-timing PRIVATE
    EXPECT_LOG_CYCLE=$<IF:$<BOOL:${LOG_CYCLE}>,1,0>
    EXPECT_CYCLE_DETAIL=$<IF:$<BOOL:${CYCLE_DETAIL}>,1,0>)
add_test(NAME test-gemmini-host-timing COMMAND $<TARGET_FILE:test-gemmini-host-timing>)
add_executable(test-gemmini-trace-context test-gemmini-trace-context.cpp)
target_link_libraries(test-gemmini-trace-context PRIVATE ggml-gemmini-utils Threads::Threads)
target_compile_features(test-gemmini-trace-context PRIVATE cxx_std_17)
target_compile_definitions(test-gemmini-trace-context PRIVATE
    EXPECT_LOG_CYCLE=$<IF:$<BOOL:${LOG_CYCLE}>,1,0>
    EXPECT_CYCLE_DETAIL=$<IF:$<BOOL:${CYCLE_DETAIL}>,1,0>)
add_test(NAME test-gemmini-trace-context COMMAND $<TARGET_FILE:test-gemmini-trace-context>
    ${CMAKE_CURRENT_BINARY_DIR}/trace-context)

find_package(Python3 3.9 COMPONENTS Interpreter QUIET)
if (Python3_Interpreter_FOUND)
    add_test(NAME test-host-overlap
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-host-overlap.py)
    add_test(NAME test-residual-profile
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-residual-profile.py)
    add_test(NAME test-cycle-timeline
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-cycle-timeline.py)
    add_test(NAME test-cycle-matrix
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-cycle-matrix.py)
endif()
if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    find_package(Threads REQUIRED)
    add_executable(test-gemmini-cycle-reader-aarch64
        test-gemmini-cycle-reader-aarch64.cpp
        ../ggml/src/ggml-gemmini-utils/src/cycle_reader_aarch64.cpp
        ../ggml/src/ggml-gemmini-utils/src/cycle_reader-capi.cpp)
    target_include_directories(test-gemmini-cycle-reader-aarch64 PRIVATE
        ../ggml/src/ggml-gemmini-utils/include)
    target_compile_definitions(test-gemmini-cycle-reader-aarch64 PRIVATE
        GGML_GEMMINI_TESTING=1 LOG_CYCLE=1 CYCLE_SIM=${CYCLE_SIM})
    target_compile_features(test-gemmini-cycle-reader-aarch64 PRIVATE cxx_std_17)
    target_link_libraries(test-gemmini-cycle-reader-aarch64 PRIVATE Threads::Threads)
    add_test(NAME test-gemmini-cycle-reader-aarch64
        COMMAND $<TARGET_FILE:test-gemmini-cycle-reader-aarch64>)
endif()
if (LOG_CYCLE AND GGML_CPU_CYCLE_LOG AND TARGET ggml-cpu AND NOT GGML_BACKEND_DL)
    add_executable(test-ggml-cpu-cycle-sink test-ggml-cpu-cycle-sink.cpp)
    target_link_libraries(test-ggml-cpu-cycle-sink PRIVATE ggml-base ggml-cpu ggml-gemmini-utils)
    target_compile_features(test-ggml-cpu-cycle-sink PRIVATE cxx_std_17)
    target_compile_definitions(test-ggml-cpu-cycle-sink PRIVATE
        LOG_CYCLE=${LOG_CYCLE} CYCLE_SIM=${CYCLE_SIM})
    target_include_directories(test-ggml-cpu-cycle-sink PRIVATE
        ../ggml/include ../ggml/src/ggml-cpu ../ggml/src/ggml-gemmini-utils/include)
    add_test(NAME test-ggml-cpu-cycle-sink
        COMMAND $<TARGET_FILE:test-ggml-cpu-cycle-sink>
            ${CMAKE_CURRENT_BINARY_DIR}/ggml-cpu-cycle-sink)
    set(_GEMMINI_REGISTERED_TEST_EXECUTABLES_APPEND_CPU_SINK TRUE)
endif()
if (LOG_DEBUG)
    add_test(NAME test-gemmini-log-explicit-debug-before-default
        COMMAND $<TARGET_FILE:test-gemmini-log> --explicit-debug-before-default)
endif()
if (LOG_CYCLE)
    foreach(_GEMMINI_LOG_FAULT IN ITEMS open write flush replacement allocation)
        add_test(NAME test-gemmini-log-fault-${_GEMMINI_LOG_FAULT}
            COMMAND $<TARGET_FILE:test-gemmini-log> --fault ${_GEMMINI_LOG_FAULT})
        set_tests_properties(test-gemmini-log-fault-${_GEMMINI_LOG_FAULT}
            PROPERTIES PASS_REGULAR_EXPRESSION "gemmini CycleLog .* failure")
    endforeach()
endif()
if (UNIX AND BUILD_SHARED_LIBS AND LOG_CYCLE AND TARGET ggml-gemmini)
    add_executable(test-gemmini-log-cross-image test-gemmini-log-cross-image.cpp)
    target_link_libraries(test-gemmini-log-cross-image PRIVATE
        ggml-gemmini-utils ${CMAKE_DL_LIBS})
    target_compile_features(test-gemmini-log-cross-image PRIVATE cxx_std_17)
    add_test(NAME test-gemmini-log-cross-image
        COMMAND $<TARGET_FILE:test-gemmini-log-cross-image>
            $<TARGET_FILE:ggml-gemmini>
            ${CMAKE_CURRENT_BINARY_DIR}/gemmini-log-cross-image/cycle-log.jsonl
            ${CMAKE_CURRENT_BINARY_DIR}/gemmini-log-cross-image/stderr.txt)
endif()
# CPU-only builds retain the common logger, host timing and CPU-operator tests
# above, but do not create fixtures that require the absent Gemmini backend or
# its generated profile helper functions.
if (TARGET ggml-gemmini)
# Direct backend symbols require a linkable backend, not a dlopen MODULE.
if (NOT GGML_BACKEND_DL)
    llama_build_and_test(test-gemmini-weight-formats.cpp NAME gemmini_weight_formats)
    target_link_libraries(gemmini_weight_formats PRIVATE ggml-base ggml-gemmini)
    target_compile_definitions(gemmini_weight_formats PRIVATE
        GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
        GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE})
endif()

add_test(NAME test-gemmini-weight-reader-callers
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/test-gemmini-weight-reader-callers.cmake)

set(_GGML_GEMMINI_BUILD_DEQUANT_FP_TARGET TRUE)
if (GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND
    NOT _GGML_GEMMINI_ENABLE_RMD_VALUE)
    set(_GGML_GEMMINI_BUILD_DEQUANT_FP_TARGET FALSE)
endif()
if (_GGML_GEMMINI_BUILD_DEQUANT_FP_TARGET)
if (_GGML_GEMMINI_ACTIVATION_BITS_VALUE EQUAL 8)
    set(_GGML_GEMMINI_DEQUANT_FP_TARGET gemmini_q8_h1_h2_dequant_fp)
else()
    set(_GGML_GEMMINI_DEQUANT_FP_TARGET gemmini_width_h1_dequant_fp)
endif()
llama_build_and_test(
    test-gemmini-q8-h1-h2-dequant-fp.cpp
    NAME ${_GGML_GEMMINI_DEQUANT_FP_TARGET})
target_sources(${_GGML_GEMMINI_DEQUANT_FP_TARGET} PRIVATE
    ../ggml/src/ggml-gemmini/quants/common/dequant.cpp
    ../ggml/src/ggml-gemmini/quants/common/weight_reader.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-compose.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp
    ../ggml/src/ggml-gemmini/quants/act/dispatch.cpp
    ../ggml/src/ggml-gemmini/quants/act/quantize.cpp
    ../ggml/src/ggml-gemmini/quants/act/block/block.cpp
    ../ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp
    ../ggml/src/ggml-gemmini/quants/act/stripe/stripe.cpp
    ../ggml/src/ggml-gemmini/quants/act/tensor/tensor.cpp
    ../ggml/src/ggml-gemmini/quants/act/token/token.cpp)
target_link_libraries(
    ${_GGML_GEMMINI_DEQUANT_FP_TARGET}
    PRIVATE ggml-base ggml-gemmini-utils)
target_compile_definitions(${_GGML_GEMMINI_DEQUANT_FP_TARGET} PRIVATE
    ${GGML_GEMMINI_COMPILE_DEFS}
    GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE=${GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE})
target_include_directories(${_GGML_GEMMINI_DEQUANT_FP_TARGET} PRIVATE
    ../ggml/src
    ../ggml/src/ggml-gemmini
    ../ggml/src/ggml-gemmini-utils/include)
if (GGML_GEMMINI_DEQUANT_FP_TEST)
    if (_GGML_GEMMINI_ACTIVATION_QUANT_UPPER STREQUAL "EXSIA")
        add_test(NAME gemmini_deferred_dequant_fp
            COMMAND ${CMAKE_COMMAND} -E env
                GEMMINI_MATMUL_MODE=STRIPE_PIPELINE
                $<TARGET_FILE:${_GGML_GEMMINI_DEQUANT_FP_TARGET}>
                --h1-pipeline)
    else()
        add_test(NAME gemmini_dequant_fp
            COMMAND ${CMAKE_COMMAND} -E env
                GEMMINI_MATMUL_MODE=FULL
                $<TARGET_FILE:${_GGML_GEMMINI_DEQUANT_FP_TARGET}>
                --h1-full)
    endif()
endif()
unset(_GGML_GEMMINI_DEQUANT_FP_TARGET)
endif()
unset(_GGML_GEMMINI_BUILD_DEQUANT_FP_TARGET)

add_executable(test-gemmini-act-quant-i32-saturation
    test-gemmini-act-quant-i32-saturation.cpp
    ../ggml/src/ggml-gemmini/quants/common/dequant.cpp
    ../ggml/src/ggml-gemmini/quants/common/weight_reader.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-compose.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp
    ../ggml/src/ggml-gemmini/quants/act/dispatch.cpp
    ../ggml/src/ggml-gemmini/quants/act/quantize.cpp
    ../ggml/src/ggml-gemmini/quants/act/block/block.cpp
    ../ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp
    ../ggml/src/ggml-gemmini/quants/act/tensor/tensor.cpp
    ../ggml/src/ggml-gemmini/quants/act/token/token.cpp
    ../ggml/src/ggml-gemmini/quants/act/stripe/stripe.cpp)
target_link_libraries(test-gemmini-act-quant-i32-saturation PRIVATE ggml-base ggml-gemmini-utils)
target_compile_features(test-gemmini-act-quant-i32-saturation PRIVATE cxx_std_17)
set(_GGML_GEMMINI_SATURATION_TEST_ACTIVATION_QUANT ${_GGML_GEMMINI_ACTIVATION_QUANT_VALUE})
if (_GGML_GEMMINI_ACTIVATION_QUANT_UPPER STREQUAL "BLOCK")
    # Keep the existing ExSIA merge fixture; BLOCK saturation is covered by test-gemmini-block-quant.
    set(_GGML_GEMMINI_SATURATION_TEST_ACTIVATION_QUANT 0)
endif()
target_compile_definitions(test-gemmini-act-quant-i32-saturation PRIVATE
    GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
    GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE}
    GGML_GEMMINI_ACTIVATION_QUANT=${_GGML_GEMMINI_SATURATION_TEST_ACTIVATION_QUANT}
    GGML_GEMMINI_EXSIA_SIGMA=${GGML_GEMMINI_EXSIA_SIGMA}
    GGML_GEMMINI_ENABLE_RMD=1)
target_include_directories(test-gemmini-act-quant-i32-saturation PRIVATE
    ../ggml/include
    ../ggml/src
    ../ggml/src/ggml-gemmini
    ../ggml/src/ggml-gemmini-utils/include)
add_test(NAME test-gemmini-act-quant-i32-saturation
    COMMAND $<TARGET_FILE:test-gemmini-act-quant-i32-saturation>)

if (_GGML_GEMMINI_ACTIVATION_QUANT_UPPER STREQUAL "BLOCK")
add_executable(test-gemmini-block-quant
    test-gemmini-block-quant.cpp
    ../ggml/src/ggml-gemmini/quants/common/dequant.cpp
    ../ggml/src/ggml-gemmini/quants/common/weight_reader.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-compose.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp
    ../ggml/src/ggml-gemmini/quants/act/dispatch.cpp
    ../ggml/src/ggml-gemmini/quants/act/quantize.cpp
    ../ggml/src/ggml-gemmini/quants/act/block/block.cpp
    ../ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp
    ../ggml/src/ggml-gemmini/quants/act/tensor/tensor.cpp
    ../ggml/src/ggml-gemmini/quants/act/token/token.cpp
    ../ggml/src/ggml-gemmini/quants/act/stripe/stripe.cpp)
target_link_libraries(test-gemmini-block-quant PRIVATE ggml-base ggml-gemmini-utils)
target_compile_features(test-gemmini-block-quant PRIVATE cxx_std_17)
target_compile_definitions(test-gemmini-block-quant PRIVATE
    GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
    GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE}
    GGML_GEMMINI_ACTIVATION_QUANT=${_GGML_GEMMINI_ACTIVATION_QUANT_VALUE}
    GGML_GEMMINI_EXSIA_SIGMA=${GGML_GEMMINI_EXSIA_SIGMA}
    GGML_GEMMINI_ENABLE_RMD=${_GGML_GEMMINI_ENABLE_RMD_VALUE}
    LOG_CYCLE=${LOG_CYCLE}
    CYCLE_DETAIL=${CYCLE_DETAIL})
target_include_directories(test-gemmini-block-quant PRIVATE
    ../ggml/include
    ../ggml/src
    ../ggml/src/ggml-gemmini
    ../ggml/src/ggml-gemmini-utils/include)
add_test(NAME gemmini_block_metadata
    COMMAND $<TARGET_FILE:test-gemmini-block-quant> --metadata)
add_test(NAME gemmini_block_quant
    COMMAND $<TARGET_FILE:test-gemmini-block-quant> --quant)

add_executable(test-gemmini-block-dense test-gemmini-block-dense.cpp)
target_link_libraries(test-gemmini-block-dense PRIVATE
    ggml-gemmini ggml-base ggml-gemmini-utils)
target_compile_features(test-gemmini-block-dense PRIVATE cxx_std_17)
target_compile_definitions(test-gemmini-block-dense PRIVATE
    ${GGML_GEMMINI_COMPILE_DEFS})
add_test(NAME gemmini_block_dense COMMAND $<TARGET_FILE:test-gemmini-block-dense>)

if (NOT GGML_BACKEND_DL)
    add_executable(test-gemmini-block-end-to-end test-gemmini-block-end-to-end.cpp)
    target_link_libraries(test-gemmini-block-end-to-end PRIVATE
        common ggml-gemmini ggml-base ggml-gemmini-utils)
    target_compile_features(test-gemmini-block-end-to-end PRIVATE cxx_std_17)
    target_compile_definitions(test-gemmini-block-end-to-end PRIVATE ${GGML_GEMMINI_COMPILE_DEFS})
    target_include_directories(test-gemmini-block-end-to-end PRIVATE
        ../ggml/include
        ../ggml/src
        ../ggml/src/ggml-gemmini
        ../ggml/src/ggml-gemmini-utils/include)
    add_test(NAME gemmini_block_end_to_end
        COMMAND $<TARGET_FILE:test-gemmini-block-end-to-end>)
endif()

endif()

set(TEST_GEMMINI_EXSIA_SW_PATH "")
if (DEFINED GEMMINI_SW_PATH AND NOT GEMMINI_SW_PATH STREQUAL "")
    set(TEST_GEMMINI_EXSIA_SW_PATH "${GEMMINI_SW_PATH}")
elseif (DEFINED ENV{GEM_HOME} AND NOT "$ENV{GEM_HOME}" STREQUAL "")
    set(TEST_GEMMINI_EXSIA_SW_PATH "$ENV{GEM_HOME}")
else()
    get_filename_component(_test_gemmini_llama_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
    set(TEST_GEMMINI_EXSIA_SW_PATH "${_test_gemmini_llama_root}/../RISC-V-DynDNN-gemmini-include")
endif()
if (NOT EXISTS "${TEST_GEMMINI_EXSIA_SW_PATH}/gemmini.h"
    AND NOT EXISTS "${TEST_GEMMINI_EXSIA_SW_PATH}/include/gemmini.h"
    AND NOT EXISTS "${TEST_GEMMINI_EXSIA_SW_PATH}/gemmini-rocc-tests/include/gemmini.h")
    message(FATAL_ERROR "test-gemmini-exsia requires a supported Gemmini header layout under ${TEST_GEMMINI_EXSIA_SW_PATH}")
endif()

add_executable(test-gemmini-exsia
    test-gemmini-exsia.cpp
    ../ggml/src/ggml-gemmini/quants/common/dequant.cpp
    ../ggml/src/ggml-gemmini/quants/common/weight_reader.cpp
    ../ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-compose.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-reference.cpp
    ../ggml/src/ggml-gemmini/quants/act/dispatch.cpp
    ../ggml/src/ggml-gemmini/quants/act/quantize.cpp
    ../ggml/src/ggml-gemmini/quants/act/block/block.cpp
    ../ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp
    ../ggml/src/ggml-gemmini/quants/act/stripe/stripe.cpp
    ../ggml/src/ggml-gemmini/quants/act/tensor/tensor.cpp
    ../ggml/src/ggml-gemmini/quants/act/token/token.cpp)
target_link_libraries(test-gemmini-exsia PRIVATE ggml-base ggml-gemmini-utils)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "riscv")
  target_link_libraries(test-gemmini-exsia PRIVATE atomic)
endif()
target_compile_features(test-gemmini-exsia PRIVATE cxx_std_17)
target_compile_definitions(test-gemmini-exsia PRIVATE
    GGML_GEMMINI_TESTING=1
    GGML_GEMMINI_DIRECT_METRICS_TESTING=1
    GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
    GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE}
    GGML_GEMMINI_EXSIA_SIGMA=${GGML_GEMMINI_EXSIA_SIGMA}
    GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE=${GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE}
    GGML_GEMMINI_ENABLE_RMD=${_GGML_GEMMINI_ENABLE_RMD_VALUE}
    CYCLE_DETAIL=${CYCLE_DETAIL}
    LOG_CYCLE=${LOG_CYCLE}
    EXSIA_VALIDATION=1)
if (GGML_GEMMINI_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    target_compile_definitions(test-gemmini-exsia PRIVATE GGML_GEMMINI_HAS_OPENMP)
    target_link_libraries(test-gemmini-exsia PRIVATE OpenMP::OpenMP_CXX)
endif()
target_include_directories(test-gemmini-exsia PRIVATE
    ../ggml/include
    ../ggml/src
    ../ggml/src/ggml-gemmini
    ../ggml/src/ggml-gemmini-utils/include
    ../ggml/src/ggml-gemmini/quants
    ../ggml/src/ggml-gemmini/quants/act
    ../ggml/src/ggml-gemmini/quants/act/exsia
    ../ggml/src/ggml-gemmini/quants/common
    ${TEST_GEMMINI_EXSIA_SW_PATH}
    ${TEST_GEMMINI_EXSIA_SW_PATH}/include
    ${TEST_GEMMINI_EXSIA_SW_PATH}/gemmini-rocc-tests
    ${TEST_GEMMINI_EXSIA_SW_PATH}/gemmini-rocc-tests/include
    ${TEST_GEMMINI_EXSIA_SW_PATH}/rocc-software/src)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
    CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    add_library(test-gemmini-direct-linux-aarch64-strict OBJECT
        ../ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp)
    set_target_properties(test-gemmini-direct-linux-aarch64-strict PROPERTIES COMPILE_OPTIONS "")
    target_compile_features(test-gemmini-direct-linux-aarch64-strict PRIVATE cxx_std_17)
    target_compile_definitions(test-gemmini-direct-linux-aarch64-strict PRIVATE
        CYCLE_SIM=${CYCLE_SIM}
        CYCLE_DETAIL=1
        LOG_CYCLE=1
        GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
        GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE})
    target_compile_options(test-gemmini-direct-linux-aarch64-strict PRIVATE
        -Wmissing-declarations
        -Wmissing-noreturn
        -Wall
        -Wextra
        -Wpedantic
        -Wcast-qual
        -Wextra-semi
        -Wunused-parameter
        -Werror)
    target_include_directories(test-gemmini-direct-linux-aarch64-strict PRIVATE
        ../ggml/include
        ../ggml/src
        ../ggml/src/ggml-gemmini
        ../ggml/src/ggml-gemmini-utils/include
        ${TEST_GEMMINI_EXSIA_SW_PATH}
        ${TEST_GEMMINI_EXSIA_SW_PATH}/include)
    add_dependencies(test-gemmini-exsia test-gemmini-direct-linux-aarch64-strict)
endif()
if (LOG_CYCLE AND CYCLE_DETAIL AND Python3_Interpreter_FOUND)
    add_test(NAME test-residual-profile-runtime
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-residual-profile.py
            --runtime-fixture $<TARGET_FILE:test-gemmini-exsia>)
endif()
add_test(NAME gemmini_direct_metrics_source_contract
    COMMAND ${CMAKE_COMMAND}
        -DDIRECT_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp
        -DDIRECT_HEADER=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-gemmini/residual/direct/direct-executor.hpp
        -DTEST_CMAKE=${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
        -DPROJECT_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/..
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test-gemmini-direct-metrics-source-contract.cmake)
# These fixtures deliberately compile the ExSIA profiling implementation with
# LOG_CYCLE=1/CYCLE_DETAIL=1 and assert both main and detail sink records. They
# must not link against an OFF logger; OFF behavior is tested by the log suite.
if (LOG_CYCLE AND CYCLE_DETAIL)
foreach(_worker_count IN ITEMS 3 4)
    add_executable(test-gemmini-exsia-workers${_worker_count}
        test-gemmini-exsia-profile.cpp
        ../ggml/src/ggml-gemmini/quants/common/dequant.cpp
        ../ggml/src/ggml-gemmini/quants/common/weight_reader.cpp
        ../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp
        ../ggml/src/ggml-gemmini/residual/rmd/rmd-compose.cpp
        ../ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp
        ../ggml/src/ggml-gemmini/residual/rmd/rmd-reference.cpp
        ../ggml/src/ggml-gemmini/quants/act/dispatch.cpp
        ../ggml/src/ggml-gemmini/quants/act/quantize.cpp
        ../ggml/src/ggml-gemmini/quants/act/block/block.cpp
        ../ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp
        ../ggml/src/ggml-gemmini/quants/act/stripe/stripe.cpp
        ../ggml/src/ggml-gemmini/quants/act/tensor/tensor.cpp
        ../ggml/src/ggml-gemmini/quants/act/token/token.cpp)
    target_link_libraries(test-gemmini-exsia-workers${_worker_count} PRIVATE
        ggml-base ggml-gemmini-utils)
    target_compile_features(test-gemmini-exsia-workers${_worker_count} PRIVATE cxx_std_17)
    target_compile_definitions(test-gemmini-exsia-workers${_worker_count} PRIVATE
        CYCLE_SIM=${CYCLE_SIM}
        GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
        GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE}
        GGML_GEMMINI_EXSIA_SIGMA=${GGML_GEMMINI_EXSIA_SIGMA}
        GGML_GEMMINI_EXSIA_LOCAL_WORKERS=${_worker_count}
        GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE=2
        GGML_GEMMINI_ENABLE_RMD=0
        CYCLE_DETAIL=1
        LOG_CYCLE=1
        EXSIA_VALIDATION=1)
    target_include_directories(test-gemmini-exsia-workers${_worker_count} PRIVATE
        ../ggml/include ../ggml/src ../ggml/src/ggml-gemmini
        ../ggml/src/ggml-gemmini-utils/include ../ggml/src/ggml-gemmini/quants
        ../ggml/src/ggml-gemmini/quants/act ../ggml/src/ggml-gemmini/quants/act/exsia
        ../ggml/src/ggml-gemmini/quants/common ${TEST_GEMMINI_EXSIA_SW_PATH}
        ${TEST_GEMMINI_EXSIA_SW_PATH}/include)
    if (GGML_GEMMINI_ENABLE_OPENMP)
        target_compile_definitions(test-gemmini-exsia-workers${_worker_count} PRIVATE
            GGML_GEMMINI_HAS_OPENMP)
        target_link_libraries(test-gemmini-exsia-workers${_worker_count} PRIVATE
            OpenMP::OpenMP_CXX)
    endif()
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
        CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        add_library(test-gemmini-exsia-workers${_worker_count}-strict OBJECT
            ../ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp)
        set_target_properties(test-gemmini-exsia-workers${_worker_count}-strict
            PROPERTIES COMPILE_OPTIONS "")
        target_compile_features(test-gemmini-exsia-workers${_worker_count}-strict PRIVATE cxx_std_17)
        target_compile_definitions(test-gemmini-exsia-workers${_worker_count}-strict PRIVATE
            CYCLE_SIM=${CYCLE_SIM}
            GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
            GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE}
            GGML_GEMMINI_EXSIA_SIGMA=${GGML_GEMMINI_EXSIA_SIGMA}
            GGML_GEMMINI_EXSIA_LOCAL_WORKERS=${_worker_count}
            GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE=2
            GGML_GEMMINI_ENABLE_RMD=0 CYCLE_DETAIL=1 LOG_CYCLE=1 EXSIA_VALIDATION=1)
        target_compile_options(test-gemmini-exsia-workers${_worker_count}-strict PRIVATE
            -Wall -Wextra -Wcast-qual -Wextra-semi -Wunused-parameter -Werror)
        target_include_directories(test-gemmini-exsia-workers${_worker_count}-strict PRIVATE
            ../ggml/include ../ggml/src ../ggml/src/ggml-gemmini
            ../ggml/src/ggml-gemmini-utils/include ../ggml/src/ggml-gemmini/quants
            ../ggml/src/ggml-gemmini/quants/act ../ggml/src/ggml-gemmini/quants/act/exsia
            ../ggml/src/ggml-gemmini/quants/common)
        target_include_directories(test-gemmini-exsia-workers${_worker_count}-strict SYSTEM PRIVATE
            ${TEST_GEMMINI_EXSIA_SW_PATH} ${TEST_GEMMINI_EXSIA_SW_PATH}/include)
        if (GGML_GEMMINI_ENABLE_OPENMP)
            target_compile_definitions(test-gemmini-exsia-workers${_worker_count}-strict PRIVATE
                GGML_GEMMINI_HAS_OPENMP)
            target_link_libraries(test-gemmini-exsia-workers${_worker_count}-strict PRIVATE
                OpenMP::OpenMP_CXX)
        endif()
        add_dependencies(test-gemmini-exsia-workers${_worker_count}
            test-gemmini-exsia-workers${_worker_count}-strict)
    endif()
    add_test(NAME test-gemmini-exsia-workers${_worker_count}
        COMMAND $<TARGET_FILE:test-gemmini-exsia-workers${_worker_count}>)
endforeach()
endif()
add_test(NAME test-gemmini-exsia-profile-contract
    COMMAND ${CMAKE_COMMAND}
        -DEXSIA_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp
        -DEXSIA_HEADER=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-gemmini/quants/act/exsia/exsia.hpp
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test-gemmini-exsia-profile-contract.cmake)

set(_GGML_GEMMINI_EXSIA_CASES
    all baseline dispatch direct-executor stripe-geometry non-exsia-capture-context)
if (_GGML_GEMMINI_ACTIVATION_BITS_VALUE EQUAL 8 AND
    _GGML_GEMMINI_WEIGHT_BITS_VALUE EQUAL 8)
    list(APPEND _GGML_GEMMINI_EXSIA_CASES
        rmd-routes q8-srmd-software-ws hp1-srmd-software-ws
        rmd-direct-parity rmd-gather)
endif()
foreach(_case IN LISTS _GGML_GEMMINI_EXSIA_CASES)
    string(REPLACE "-" "_" _case_test_name "${_case}")
    add_test(NAME gemmini_exsia_${_case_test_name}
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXECUTABLE=$<TARGET_FILE:test-gemmini-exsia>
            -DEXPECTED_CASE=${_case}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/test-gemmini-exsia-case.cmake)
endforeach()
set_tests_properties(gemmini_exsia_all gemmini_exsia_direct_executor PROPERTIES
    ENVIRONMENT "OMP_DYNAMIC=FALSE;OMP_NUM_THREADS=3;OMP_THREAD_LIMIT=3"
    TIMEOUT 30)
unset(_GGML_GEMMINI_EXSIA_CASES)
add_test(NAME gemmini_exsia_unknown_case
    COMMAND $<TARGET_FILE:test-gemmini-exsia> --case=unknown)
set_tests_properties(gemmini_exsia_unknown_case PROPERTIES WILL_FAIL TRUE)

add_executable(test-gemmini-exsia-writer
    test-gemmini-exsia.cpp
    ../ggml/src/ggml-gemmini/quants/common/dequant.cpp
    ../ggml/src/ggml-gemmini/quants/common/weight_reader.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-compose.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp
    ../ggml/src/ggml-gemmini/quants/act/dispatch.cpp
    ../ggml/src/ggml-gemmini/quants/act/quantize.cpp
    ../ggml/src/ggml-gemmini/quants/act/block/block.cpp
    ../ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp
    ../ggml/src/ggml-gemmini/quants/act/tensor/tensor.cpp
    ../ggml/src/ggml-gemmini/quants/act/token/token.cpp)
target_link_libraries(test-gemmini-exsia-writer PRIVATE ggml-base ggml-gemmini-utils)
target_compile_features(test-gemmini-exsia-writer PRIVATE cxx_std_17)
target_compile_definitions(test-gemmini-exsia-writer PRIVATE
    GEMMINI_EXSIA_WRITER_TEST_ONLY=1
    GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
    GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE}
    GGML_GEMMINI_EXSIA_SIGMA=${GGML_GEMMINI_EXSIA_SIGMA}
    GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE=1
    GGML_GEMMINI_ENABLE_RMD=${_GGML_GEMMINI_ENABLE_RMD_VALUE}
    GGML_GEMMINI_EXSIA_DEFAULT_MODE_VALUE=${_GGML_GEMMINI_EXSIA_DEFAULT_MODE_VALUE}
    GGML_GEMMINI_EXSIA_LOCAL_WORKERS=${GGML_GEMMINI_EXSIA_LOCAL_WORKERS}
    CYCLE_DETAIL=1
    LOG_CYCLE=1
    EXSIA_VALIDATION=1)
if (GGML_GEMMINI_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    target_compile_definitions(test-gemmini-exsia-writer PRIVATE
        GGML_GEMMINI_HAS_OPENMP)
    target_link_libraries(test-gemmini-exsia-writer PRIVATE OpenMP::OpenMP_CXX)
endif()
target_include_directories(test-gemmini-exsia-writer PRIVATE
    ../ggml/include ../ggml/src ../ggml/src/ggml-gemmini
    ../ggml/src/ggml-gemmini-utils/include ../ggml/src/ggml-gemmini/quants
    ../ggml/src/ggml-gemmini/quants/act ../ggml/src/ggml-gemmini/quants/act/exsia
    ../ggml/src/ggml-gemmini/quants/common ${GEMMINI_SW_PATH} ${GEMMINI_SW_PATH}/include)

set(GEMMINI_WRITER_TEST_ROOT "${CMAKE_CURRENT_BINARY_DIR}/gemmini-writer-output")
foreach(_case IN ITEMS exsia-default exsia-dir exsia-relative exsia-absolute exsia-empty exsia-invalid)
    file(MAKE_DIRECTORY "${GEMMINI_WRITER_TEST_ROOT}/${_case}")
endforeach()
add_test(NAME gemmini-exsia-output-default
    COMMAND ${CMAKE_COMMAND} -E env --unset=GEMMINI_LOG_DIR --unset=GGML_GEMMINI_CYCLE_DETAIL_LOG
        $<TARGET_FILE:test-gemmini-exsia-writer> --profile-output output/log/exsia-cycle-detail.jsonl)
set_tests_properties(gemmini-exsia-output-default PROPERTIES WORKING_DIRECTORY "${GEMMINI_WRITER_TEST_ROOT}/exsia-default")
add_test(NAME gemmini-exsia-output-dir
    COMMAND ${CMAKE_COMMAND} -E env GEMMINI_LOG_DIR=env-log --unset=GGML_GEMMINI_CYCLE_DETAIL_LOG
        $<TARGET_FILE:test-gemmini-exsia-writer> --profile-output env-log/exsia-cycle-detail.jsonl)
set_tests_properties(gemmini-exsia-output-dir PROPERTIES WORKING_DIRECTORY "${GEMMINI_WRITER_TEST_ROOT}/exsia-dir")
add_test(NAME gemmini-exsia-output-relative
    COMMAND ${CMAKE_COMMAND} -E env GEMMINI_LOG_DIR=env-log GGML_GEMMINI_CYCLE_DETAIL_LOG=custom/detail.jsonl
        $<TARGET_FILE:test-gemmini-exsia-writer> --profile-output env-log/custom/detail.jsonl)
set_tests_properties(gemmini-exsia-output-relative PROPERTIES WORKING_DIRECTORY "${GEMMINI_WRITER_TEST_ROOT}/exsia-relative")
add_test(NAME gemmini-exsia-output-absolute
    COMMAND ${CMAKE_COMMAND} -E env GEMMINI_LOG_DIR=env-log GGML_GEMMINI_CYCLE_DETAIL_LOG=${GEMMINI_WRITER_TEST_ROOT}/exsia-absolute/absolute-log/detail.jsonl
        $<TARGET_FILE:test-gemmini-exsia-writer> --profile-output ${GEMMINI_WRITER_TEST_ROOT}/exsia-absolute/absolute-log/detail.jsonl)
set_tests_properties(gemmini-exsia-output-absolute PROPERTIES WORKING_DIRECTORY "${GEMMINI_WRITER_TEST_ROOT}/exsia-absolute")
add_test(NAME gemmini-exsia-output-empty
    COMMAND ${CMAKE_COMMAND} -E env GEMMINI_LOG_DIR=env-log GGML_GEMMINI_CYCLE_DETAIL_LOG=
        $<TARGET_FILE:test-gemmini-exsia-writer> --profile-output env-log/exsia-cycle-detail.jsonl)
set_tests_properties(gemmini-exsia-output-empty PROPERTIES WORKING_DIRECTORY "${GEMMINI_WRITER_TEST_ROOT}/exsia-empty")
add_test(NAME gemmini-exsia-output-invalid-parent
    COMMAND ${CMAKE_COMMAND} -E env --unset=GEMMINI_LOG_DIR GGML_GEMMINI_CYCLE_DETAIL_LOG=blocked/detail.jsonl
        $<TARGET_FILE:test-gemmini-exsia-writer> --profile-output blocked/detail.jsonl --invalid-parent)
set_tests_properties(gemmini-exsia-output-invalid-parent PROPERTIES WORKING_DIRECTORY "${GEMMINI_WRITER_TEST_ROOT}/exsia-invalid")

# Direct backend symbols require a linkable backend, not a dlopen MODULE.
if (NOT GGML_BACKEND_DL)
    add_executable(test-gemmini-rmd
        test-gemmini-rmd.cpp
        ../ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp
        ../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp)
    target_link_libraries(test-gemmini-rmd PRIVATE ggml-base ggml-gemmini ggml-gemmini-utils)
    target_compile_features(test-gemmini-rmd PRIVATE cxx_std_17)
    target_compile_definitions(test-gemmini-rmd PRIVATE
        LOG_CYCLE=${LOG_CYCLE}
        GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
        GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE})
    if (GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "FPGA_UART")
        target_compile_definitions(test-gemmini-rmd PRIVATE
            GGML_GEMMINI_EXECUTION_BACKEND_FPGA_UART=1)
    endif()
    target_include_directories(test-gemmini-rmd PRIVATE
        ../ggml/src
        ../ggml/src/ggml-gemmini
        ../ggml/src/ggml-gemmini/residual/rmd
        ${GEMMINI_SW_PATH}
        ${GEMMINI_SW_PATH}/include)
    add_test(NAME test-gemmini-rmd COMMAND $<TARGET_FILE:test-gemmini-rmd>)
    add_test(NAME test-gemmini-rmd-direct-happy
        COMMAND $<TARGET_FILE:test-gemmini-rmd> --case=direct-happy)
    add_test(NAME test-gemmini-rmd-direct-failure
        COMMAND $<TARGET_FILE:test-gemmini-rmd> --case=direct-failure)
    add_test(NAME test-gemmini-rmd-compact-happy
        COMMAND $<TARGET_FILE:test-gemmini-rmd> --case=compact-happy)
    add_test(NAME test-gemmini-rmd-compact-failure
        COMMAND $<TARGET_FILE:test-gemmini-rmd> --case=compact-failure)
    add_test(NAME test-gemmini-rmd-radix-happy
        COMMAND $<TARGET_FILE:test-gemmini-rmd> --case=radix-happy)
    add_test(NAME test-gemmini-rmd-radix-failure
        COMMAND $<TARGET_FILE:test-gemmini-rmd> --case=radix-failure)
endif()

add_executable(test-gemmini-rmd-packet
    test-gemmini-rmd-packet.cpp
    ../ggml/src/ggml-gemmini/quants/common/weight_reader.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp)
target_link_libraries(test-gemmini-rmd-packet PRIVATE
    ggml-base ggml-gemmini-utils)
target_compile_features(test-gemmini-rmd-packet PRIVATE cxx_std_17)
target_compile_definitions(test-gemmini-rmd-packet PRIVATE
    GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
    GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE})
target_include_directories(test-gemmini-rmd-packet PRIVATE
    ../ggml/src
    ../ggml/src/ggml-gemmini
    ../ggml/src/ggml-gemmini/residual/rmd
    ${GEMMINI_SW_PATH}
    ${GEMMINI_SW_PATH}/include)
add_test(NAME test-gemmini-rmd-packet
    COMMAND $<TARGET_FILE:test-gemmini-rmd-packet>)

add_executable(test-gemmini-rmd-run-aware
    test-gemmini-rmd-run-aware.cpp)
target_link_libraries(test-gemmini-rmd-run-aware PRIVATE
    ggml-base ggml-gemmini ggml-gemmini-utils)
target_compile_features(test-gemmini-rmd-run-aware PRIVATE cxx_std_17)
target_compile_definitions(test-gemmini-rmd-run-aware PRIVATE
    GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
    GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE})
target_include_directories(test-gemmini-rmd-run-aware PRIVATE
    ../ggml/src
    ../ggml/src/ggml-gemmini
    ../ggml/src/ggml-gemmini/residual/rmd
    ${GEMMINI_SW_PATH}
    ${GEMMINI_SW_PATH}/include)
add_test(NAME test-gemmini-rmd-run-aware
    COMMAND $<TARGET_FILE:test-gemmini-rmd-run-aware>)

add_executable(test-gemmini-rmd-capture
    test-gemmini-rmd-capture.cpp
    ../ggml/src/ggml-gemmini/quants/common/weight_reader.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp)
target_link_libraries(test-gemmini-rmd-capture PRIVATE
    ggml-base ggml-gemmini-utils)
target_compile_features(test-gemmini-rmd-capture PRIVATE cxx_std_17)
target_compile_definitions(test-gemmini-rmd-capture PRIVATE
    CYCLE_SIM=${CYCLE_SIM}
    GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
    GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE}
    LOG_CYCLE=${LOG_CYCLE}
    CYCLE_DETAIL=${CYCLE_DETAIL})
target_include_directories(test-gemmini-rmd-capture PRIVATE
    ../ggml/src
    ../ggml/src/ggml-gemmini
    ../ggml/src/ggml-gemmini/residual/rmd
    ${GEMMINI_SW_PATH}
    ${GEMMINI_SW_PATH}/include)
add_test(NAME test-gemmini-rmd-capture
    COMMAND $<TARGET_FILE:test-gemmini-rmd-capture>)
add_test(NAME test-gemmini-rmd-capture-finish-boundary
    COMMAND ${CMAKE_COMMAND}
        -DTEST_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/..
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test-gemmini-rmd-capture-finish-boundary.cmake)

add_executable(test-gemmini-production-host-backend
    test-gemmini-production-host-backend.cpp)
target_compile_features(test-gemmini-production-host-backend PRIVATE cxx_std_17)
target_compile_definitions(test-gemmini-production-host-backend PRIVATE
    GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
    GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE})
target_include_directories(test-gemmini-production-host-backend PRIVATE
    ../ggml/include
    ../ggml/src
    ../ggml/src/ggml-gemmini
    ../ggml/src/ggml-gemmini-utils/include
    ${GGML_GEMMINI_GENERATED_CONFIG_DIR}
    ${GEMMINI_SW_PATH}
    ${GEMMINI_SW_PATH}/include)
add_test(NAME test-gemmini-production-host
    COMMAND $<TARGET_FILE:test-gemmini-production-host-backend>)

# Direct backend symbols require a linkable backend, not a dlopen MODULE.
if (NOT GGML_BACKEND_DL)
    add_executable(test-gemmini-rmd-telemetry
        test-gemmini-rmd-telemetry.cpp)
    target_link_libraries(test-gemmini-rmd-telemetry PRIVATE
        ggml-gemmini ggml-base ggml-gemmini-utils)
    target_compile_features(test-gemmini-rmd-telemetry PRIVATE cxx_std_17)
    target_compile_definitions(test-gemmini-rmd-telemetry PRIVATE
        GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
        GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE}
        LOG_DEBUG=${LOG_DEBUG}
        LOG_CYCLE=${LOG_CYCLE}
        CYCLE_DETAIL=${CYCLE_DETAIL})
    target_include_directories(test-gemmini-rmd-telemetry PRIVATE
        ../ggml/include ../ggml/src ../ggml/src/ggml-gemmini
        ../ggml/src/ggml-gemmini-utils/include ${GGML_GEMMINI_GENERATED_CONFIG_DIR}
        ${GEMMINI_SW_PATH} ${GEMMINI_SW_PATH}/include)
    # Plain HARDWARE/CPU builds do not compile the IM2P adapter and must not
    # require simulator headers merely to test the common telemetry schema.
    if (GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" OR
        GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "FPGA_UART")
        target_compile_definitions(test-gemmini-rmd-telemetry PRIVATE
            GGML_GEMMINI_TELEMETRY_HAS_IM2P=1)
        target_include_directories(test-gemmini-rmd-telemetry PRIVATE
            ${IM2P_SIM_ROOT}/frontend/include ${IM2P_SIM_ROOT}/sim/include)
    endif()
    add_test(NAME test-gemmini-rmd-telemetry
        COMMAND $<TARGET_FILE:test-gemmini-rmd-telemetry>)
    add_test(NAME test-gemmini-aggregate-cycle-driver
        COMMAND $<TARGET_FILE:test-gemmini-rmd-telemetry> --aggregate-driver
                ${CMAKE_CURRENT_BINARY_DIR}/gemmini-aggregate-driver/cycle-log.jsonl)
endif()

add_executable(test-gemmini-rmd-reducer
    test-gemmini-rmd-telemetry.cpp
    ../ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp)
target_link_libraries(test-gemmini-rmd-reducer PRIVATE ggml-gemmini-utils)
target_compile_features(test-gemmini-rmd-reducer PRIVATE cxx_std_17)
target_compile_definitions(test-gemmini-rmd-reducer PRIVATE
    GGML_GEMMINI_REDUCER_TEST_ONLY=1
    GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
    GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE}
    LOG_DEBUG=${LOG_DEBUG} LOG_CYCLE=${LOG_CYCLE} CYCLE_DETAIL=${CYCLE_DETAIL})
target_include_directories(test-gemmini-rmd-reducer PRIVATE
    ../ggml/include ../ggml/src ../ggml/src/ggml-gemmini
    ../ggml/src/ggml-gemmini-utils/include ${GGML_GEMMINI_GENERATED_CONFIG_DIR}
    ${GEMMINI_SW_PATH} ${GEMMINI_SW_PATH}/include)
add_test(NAME test-gemmini-rmd-reducer
    COMMAND $<TARGET_FILE:test-gemmini-rmd-reducer>)
if (UNIX AND EXISTS "${IM2P_SIM_ROOT}/frontend/include/im2p_gemmini_frontend.hpp")
    add_executable(test-gemmini-im2p-host-cycle test-gemmini-im2p-routing.cpp
        ../ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp)
    target_link_libraries(test-gemmini-im2p-host-cycle PRIVATE ggml-base ggml-gemmini-utils)
    target_compile_features(test-gemmini-im2p-host-cycle PRIVATE cxx_std_17)
    target_compile_definitions(test-gemmini-im2p-host-cycle PRIVATE
        GGML_GEMMINI_IM2P_HOST_CYCLE_TEST=1
        GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM=1
        GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
        GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE}
        LOG_CYCLE=${LOG_CYCLE} CYCLE_DETAIL=${CYCLE_DETAIL})
    target_include_directories(test-gemmini-im2p-host-cycle PRIVATE
        ../ggml/include ../ggml/src ../ggml/src/ggml-gemmini
        ${GGML_GEMMINI_GENERATED_CONFIG_DIR}
        ${GEMMINI_SW_PATH} ${GEMMINI_SW_PATH}/include
        ${IM2P_SIM_ROOT}/frontend/include ${IM2P_SIM_ROOT}/sim/include)
    target_compile_options(test-gemmini-im2p-host-cycle PRIVATE -ffunction-sections -fdata-sections)
    if (APPLE)
        target_link_options(test-gemmini-im2p-host-cycle PRIVATE -Wl,-dead_strip)
    else()
        target_link_options(test-gemmini-im2p-host-cycle PRIVATE -Wl,--gc-sections)
    endif()
    add_test(NAME test-gemmini-im2p-host-cycle COMMAND $<TARGET_FILE:test-gemmini-im2p-host-cycle>)
endif()
add_test(NAME test-gemmini-telemetry-clock-gate
    COMMAND ${CMAKE_COMMAND}
            -DTEST_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/..
            -P ${CMAKE_CURRENT_SOURCE_DIR}/test-gemmini-telemetry-clock-gate.cmake)

option(GEMMINI_BUILD_SRMD_RAW_WS_TESTS
    "Build the raw-WS SRMD tests against the configured Gemmini headers"
    OFF)
if (GEMMINI_BUILD_SRMD_RAW_WS_TESTS AND GGML_BACKEND_DL)
    message(FATAL_ERROR "Raw WS tests link backend symbols directly; use GGML_BACKEND_DL=OFF")
endif()
if (GEMMINI_BUILD_SRMD_RAW_WS_TESTS)
    set(GEMMINI_SRMD_RAW_WS_HEADER "")
    foreach(_header IN ITEMS
            "${GEMMINI_SW_PATH}/gemmini.h"
            "${GEMMINI_SW_PATH}/include/gemmini.h"
            "${GEMMINI_SW_PATH}/gemmini-rocc-tests/include/gemmini.h")
        if (NOT GEMMINI_SRMD_RAW_WS_HEADER AND EXISTS "${_header}")
            set(GEMMINI_SRMD_RAW_WS_HEADER "${_header}")
        endif()
    endforeach()
    if (NOT GEMMINI_SRMD_RAW_WS_HEADER)
        message(FATAL_ERROR "test-gemmini-srmd-raw-ws requires a supported Gemmini header layout")
    endif()
    get_filename_component(GEMMINI_SRMD_RAW_WS_INCLUDE_DIR
        "${GEMMINI_SRMD_RAW_WS_HEADER}" DIRECTORY)
    add_executable(test-gemmini-srmd-raw-ws
        test-gemmini-srmd-raw-ws.cpp)
    target_link_libraries(test-gemmini-srmd-raw-ws PRIVATE
        ggml-base ggml-gemmini ggml-gemmini-utils)
    target_compile_features(test-gemmini-srmd-raw-ws PRIVATE cxx_std_17)
    target_include_directories(test-gemmini-srmd-raw-ws PRIVATE
        ${GEMMINI_SRMD_RAW_WS_INCLUDE_DIR}
        ${GEMMINI_SW_PATH}
        ${GEMMINI_SW_PATH}/include
        ../ggml/src
        ../ggml/src/ggml-gemmini
        ../ggml/src/ggml-gemmini/residual/rmd)
    add_test(NAME test-gemmini-srmd-raw-ws-cpu
        COMMAND $<TARGET_FILE:test-gemmini-srmd-raw-ws> --cpu)
endif()

if (_GGML_GEMMINI_ACTIVATION_BITS_VALUE EQUAL 8 AND
    _GGML_GEMMINI_WEIGHT_BITS_VALUE EQUAL 8)
    add_executable(test-gemmini-matmul-options test-gemmini-matmul-options.cpp)
    target_compile_features(test-gemmini-matmul-options PRIVATE cxx_std_17)
    target_compile_definitions(test-gemmini-matmul-options PRIVATE
        GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
        GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE}
        GGML_GEMMINI_ENABLE_RMD=${_GGML_GEMMINI_ENABLE_RMD_VALUE}
        GGML_GEMMINI_EXSIA_DEFAULT_MODE_VALUE=${_GGML_GEMMINI_EXSIA_DEFAULT_MODE_VALUE}
        GGML_GEMMINI_EXSIA_LOCAL_WORKERS=${GGML_GEMMINI_EXSIA_LOCAL_WORKERS})
    ggml_gemmini_apply_profile_compile_definitions(test-gemmini-matmul-options)
    ggml_gemmini_assert_profile_compile_definitions(test-gemmini-matmul-options)
    target_include_directories(test-gemmini-matmul-options PRIVATE
        ../ggml/include
        ../ggml/src
        ../ggml/src/ggml-gemmini
        ../ggml/src/ggml-gemmini-utils/include
        ../ggml/src/ggml-gemmini/quants
        ${GGML_GEMMINI_GENERATED_CONFIG_DIR}
        ${GEMMINI_SW_PATH}
        ${GEMMINI_SW_PATH}/include)
    if (NOT GGML_BACKEND_DL)
        target_link_libraries(test-gemmini-matmul-options PRIVATE ggml-gemmini ggml-base ggml-gemmini-utils)
        target_compile_definitions(test-gemmini-matmul-options PRIVATE GGML_GEMMINI_OPTIONS_TEST_BACKEND=1)
    endif()
    add_test(NAME test-gemmini-matmul-options COMMAND $<TARGET_FILE:test-gemmini-matmul-options>)
endif()
add_test(NAME test-gemmini-geometry-contract
    COMMAND ${CMAKE_COMMAND}
        -DTEST_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/..
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test-gemmini-geometry-contract.cmake)
add_test(NAME test-gemmini-geometry-contract-adversarial
    COMMAND ${CMAKE_COMMAND}
        -DTEST_CMAKE_COMMAND=${CMAKE_COMMAND}
        -DTEST_SCANNER=${CMAKE_CURRENT_SOURCE_DIR}/test-gemmini-geometry-contract.cmake
        -DTEST_BINARY_ROOT=${CMAKE_CURRENT_BINARY_DIR}/gemmini-geometry-contract-adversarial
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test-gemmini-geometry-contract-adversarial.cmake)
get_filename_component(LLAMA_TEST_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
if (EXISTS "${CMAKE_COMMAND}" AND EXISTS "${LLAMA_TEST_SOURCE_DIR}/CMakeLists.txt")
    add_test(
        NAME test-gemmini-build-contract
        COMMAND ${CMAKE_COMMAND}
            -DTEST_CMAKE_COMMAND=${CMAKE_COMMAND}
            -DTEST_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DTEST_AR=${CMAKE_AR}
            -DTEST_REAL_IM2P_ROOT=${LLAMA_TEST_SOURCE_DIR}/../IM2P.sim
            -DTEST_SOURCE_DIR=${LLAMA_TEST_SOURCE_DIR}
            -DTEST_BINARY_ROOT=${CMAKE_CURRENT_BINARY_DIR}/gemmini-build-contract
            -DTEST_GENERATOR=${CMAKE_GENERATOR}
            -DTEST_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}
            -DTEST_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/test-gemmini-build-contract.cmake)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME test-gemmini-hp1-fpga-contract
            COMMAND ${CMAKE_COMMAND}
                -DTEST_CMAKE=${CMAKE_COMMAND}
                -DTEST_PYTHON=${Python3_EXECUTABLE}
                -DTEST_SOURCE_DIR=${LLAMA_TEST_SOURCE_DIR}
                -DTEST_BINARY_ROOT=${CMAKE_CURRENT_BINARY_DIR}/gemmini-hp1-fpga-contract
                -P ${CMAKE_CURRENT_SOURCE_DIR}/test-gemmini-hp1-fpga-contract.cmake)
    endif()
endif()

if (GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND NOT GGML_BACKEND_DL)
    add_executable(test-gemmini-im2p-routing test-gemmini-im2p-routing.cpp)
    target_link_libraries(test-gemmini-im2p-routing PRIVATE ggml-gemmini ggml-base ggml-gemmini-utils)
    target_compile_features(test-gemmini-im2p-routing PRIVATE cxx_std_17)
    target_compile_definitions(test-gemmini-im2p-routing PRIVATE
        GGML_GEMMINI_TESTING=1
        GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
        GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE}
        GGML_GEMMINI_ACTIVATION_QUANT=${_GGML_GEMMINI_ACTIVATION_QUANT_VALUE}
        GGML_GEMMINI_ENABLE_RMD=${_GGML_GEMMINI_ENABLE_RMD_VALUE}
        GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE=${GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE}
        GGML_GEMMINI_TEST_IM2P_DIM=${GGML_GEMMINI_DIM}
        LOG_DEBUG=${LOG_DEBUG})
    ggml_gemmini_apply_profile_compile_definitions(test-gemmini-im2p-routing)
    ggml_gemmini_assert_profile_compile_definitions(test-gemmini-im2p-routing)
    target_include_directories(test-gemmini-im2p-routing PRIVATE
        ../ggml/src
        ../ggml/src/ggml-gemmini
        ../ggml/src/ggml-gemmini/quants
        ${IM2P_SIM_ROOT}/frontend/include
        ${IM2P_SIM_ROOT}/sim/include
        ${GEMMINI_SW_PATH}
        ${GEMMINI_SW_PATH}/include)
    add_test(NAME test-gemmini-im2p-routing
        COMMAND $<TARGET_FILE:test-gemmini-im2p-routing>)
    add_test(NAME test-gemmini-im2p-route-matrix
        COMMAND $<TARGET_FILE:test-gemmini-im2p-routing> --route-matrix)
    if (_GGML_GEMMINI_ACTIVATION_QUANT_UPPER STREQUAL "EXSIA" AND GGML_GEMMINI_ENABLE_RMD)
        add_test(NAME test-gemmini-im2p-routing-multiwidth-matrix
            COMMAND $<TARGET_FILE:test-gemmini-im2p-routing> --case multiwidth-matrix)
        add_test(NAME test-gemmini-im2p-routing-boundary-rejections
            COMMAND $<TARGET_FILE:test-gemmini-im2p-routing> --case boundary-rejections)
    endif()
    add_test(NAME test-gemmini-im2p-routing-geometry
        COMMAND $<TARGET_FILE:test-gemmini-im2p-routing> --geometry-contract)
    add_test(NAME test-gemmini-im2p-routing-stats-translation
        COMMAND $<TARGET_FILE:test-gemmini-im2p-routing> --stats-translation)
    if (LOG_CYCLE AND
        GGML_GEMMINI_ACTIVATION_QUANT STREQUAL "EXSIA" AND
        GGML_GEMMINI_ENABLE_RMD)
        add_test(NAME test-gemmini-im2p-cycle-overlap
            COMMAND ${CMAKE_COMMAND}
                -DTEST_BINARY=$<TARGET_FILE:test-gemmini-im2p-routing>
                -DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/gemmini-im2p-cycle-overlap
                -DTEST_DIM=${GGML_GEMMINI_DIM}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/test-gemmini-im2p-cycle-overlap.cmake)
    endif()

    add_executable(test-gemmini-im2p-lifecycle
        test-gemmini-im2p-lifecycle.cpp)
    target_link_libraries(test-gemmini-im2p-lifecycle PRIVATE
        ggml-gemmini ggml-base)
    target_compile_features(test-gemmini-im2p-lifecycle PRIVATE cxx_std_17)
    target_compile_options(test-gemmini-im2p-lifecycle PRIVATE -Werror)
    add_dependencies(test-gemmini-im2p-lifecycle test-gemmini-im2p-routing)
    target_compile_definitions(test-gemmini-im2p-lifecycle PRIVATE
        GGML_GEMMINI_ROUTING_BINARY="$<TARGET_FILE:test-gemmini-im2p-routing>"
        GGML_GEMMINI_TESTING=1
        GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
        GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE}
        GGML_GEMMINI_ACTIVATION_QUANT=${_GGML_GEMMINI_ACTIVATION_QUANT_VALUE}
        GGML_GEMMINI_ENABLE_RMD=${_GGML_GEMMINI_ENABLE_RMD_VALUE}
        GGML_GEMMINI_TEST_IM2P_DIM=${GGML_GEMMINI_DIM})
    target_include_directories(test-gemmini-im2p-lifecycle PRIVATE
        ../ggml/src
        ../ggml/src/ggml-gemmini
        ../ggml/src/ggml-gemmini/quants
        ${IM2P_SIM_ROOT}/frontend/include
        ${IM2P_SIM_ROOT}/sim/include
        ${GEMMINI_SW_PATH}
        ${GEMMINI_SW_PATH}/include)
    add_test(NAME test-gemmini-im2p-lifecycle
        COMMAND $<TARGET_FILE:test-gemmini-im2p-lifecycle> --case all)
    add_test(NAME test-gemmini-im2p-lifecycle-stage-failures
        COMMAND $<TARGET_FILE:test-gemmini-im2p-lifecycle>
            --case stage-failures)
    add_test(NAME test-gemmini-im2p-lifecycle-invalid-reuse
        COMMAND $<TARGET_FILE:test-gemmini-im2p-lifecycle>
            --case invalid-reuse)
    add_test(NAME test-gemmini-im2p-lifecycle-stats-invariants
        COMMAND $<TARGET_FILE:test-gemmini-im2p-lifecycle>
            --case stats-invariants)

    add_executable(test-gemmini-rmd-im2p-provider
        test-gemmini-rmd-im2p-provider.cpp)
    target_link_libraries(test-gemmini-rmd-im2p-provider PRIVATE
        ggml-gemmini ggml-base)
    target_compile_features(test-gemmini-rmd-im2p-provider PRIVATE cxx_std_17)
    target_compile_definitions(test-gemmini-rmd-im2p-provider PRIVATE
        GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM=1
        GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
        GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE})
    target_include_directories(test-gemmini-rmd-im2p-provider PRIVATE
        ../ggml/src
        ../ggml/src/ggml-gemmini
        ${IM2P_SIM_ROOT}/sim/include
        ${IM2P_SIM_ROOT}/frontend/include
        ${GEMMINI_SW_PATH}
        ${GEMMINI_SW_PATH}/include)
    if(TARGET ggml-gemmini-cpu-functional)
        target_link_libraries(test-gemmini-rmd-im2p-provider PRIVATE ggml-gemmini-cpu-functional)
    endif()
    add_test(NAME test-gemmini-rmd-im2p-provider-dispatch-parity
        COMMAND ${CMAKE_COMMAND} -E env
            GEMMINI_LOG_DIR=${CMAKE_CURRENT_BINARY_DIR}/rmd-im2p-dispatch-parity
            $<TARGET_FILE:test-gemmini-rmd-im2p-provider> --case dispatch-parity)
    if(IM2P_SIM_IMPLEMENTATION STREQUAL "GEMMINI_HP1" AND NOT CYCLE_SIM)
        add_test(NAME test-gemmini-rmd-im2p-provider-dispatch-parity-callback-failure
            COMMAND $<TARGET_FILE:test-gemmini-rmd-im2p-provider>
                --case dispatch-parity-callback-failure)
        add_test(NAME test-gemmini-rmd-im2p-provider-production-fixtures
            COMMAND ${CMAKE_COMMAND} -E env
                GEMMINI_LOG_DIR=${CMAKE_CURRENT_BINARY_DIR}/rmd-im2p-production-fixtures
                $<TARGET_FILE:test-gemmini-rmd-im2p-provider> --case production-fixtures)
        add_test(NAME test-gemmini-rmd-im2p-provider-production-oracle-mutations
            COMMAND ${CMAKE_COMMAND} -E env
                GEMMINI_LOG_DIR=${CMAKE_CURRENT_BINARY_DIR}/rmd-im2p-production-oracle-mutations
                $<TARGET_FILE:test-gemmini-rmd-im2p-provider> --case production-oracle-mutations)
    endif()
    if(CYCLE_SIM)
        add_test(NAME test-gemmini-rmd-im2p-provider-run-aware-functional
            COMMAND $<TARGET_FILE:test-gemmini-rmd-im2p-provider> --case cross-block-one-work)
        add_test(NAME test-gemmini-rmd-im2p-provider-sat32-functional
            COMMAND $<TARGET_FILE:test-gemmini-rmd-im2p-provider> --case sat32-cross-run)
    else()
        add_test(NAME test-gemmini-rmd-im2p-provider
            COMMAND $<TARGET_FILE:test-gemmini-rmd-im2p-provider> --case success)
        add_test(NAME test-gemmini-rmd-im2p-provider-failures
            COMMAND $<TARGET_FILE:test-gemmini-rmd-im2p-provider> --case all)
    endif()

    if (IM2P_SIM_IMPLEMENTATION STREQUAL "GEMMINI_HP1" AND GGML_GEMMINI_ENABLE_RMD)
        add_executable(test-gemmini-optrace-dispatch test-gemmini-optrace-dispatch.cpp)
        target_link_libraries(test-gemmini-optrace-dispatch PRIVATE
            ggml-gemmini ggml-base ggml-gemmini-utils)
        target_compile_features(test-gemmini-optrace-dispatch PRIVATE cxx_std_17)
        target_compile_definitions(test-gemmini-optrace-dispatch PRIVATE GGML_GEMMINI_TESTING=1)
        ggml_gemmini_apply_profile_compile_definitions(test-gemmini-optrace-dispatch)
        target_include_directories(test-gemmini-optrace-dispatch PRIVATE
            ../ggml/src ../ggml/src/ggml-gemmini ${IM2P_SIM_ROOT}/sim/include
            ${IM2P_SIM_ROOT}/frontend/include ${GEMMINI_SW_PATH} ${GEMMINI_SW_PATH}/include)
        foreach(mode IN ITEMS full pipeline residual)
            add_test(NAME test-gemmini-optrace-dispatch-${mode}
                COMMAND $<TARGET_FILE:test-gemmini-optrace-dispatch> ${mode})
        endforeach()
        add_test(NAME test-gemmini-optrace-transaction
            COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/test-gemmini-optrace-transaction.sh
                $<TARGET_FILE:test-gemmini-optrace-dispatch> ${CMAKE_CURRENT_BINARY_DIR})
    endif()

    add_executable(test-gemmini-rmd-backend-gate
        test-gemmini-rmd-backend-gate.cpp)
    target_link_libraries(test-gemmini-rmd-backend-gate PRIVATE
        ggml-gemmini ggml-base)
    target_compile_features(test-gemmini-rmd-backend-gate PRIVATE cxx_std_17)
    add_test(
        NAME test-gemmini-rmd-backend-denylist
        COMMAND ${CMAKE_COMMAND}
            -DTEST_GATE_EXECUTABLE=$<TARGET_FILE:test-gemmini-rmd-backend-gate>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/test-gemmini-rmd-backend-denylist.cmake)
endif()

if (NOT GGML_BACKEND_DL)
    # The façade test uses C++ symbols from the statically linked backend.
    # Dynamic backends are MODULE libraries and cannot be linked into a test.
    add_executable(test-gemmini-matmul test-gemmini-matmul.cpp)
    target_link_libraries(test-gemmini-matmul PRIVATE ggml-gemmini ggml-base ggml-gemmini-utils)
    target_compile_features(test-gemmini-matmul PRIVATE cxx_std_17)
    target_compile_definitions(test-gemmini-matmul PRIVATE
        GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
        GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE}
        GGML_GEMMINI_ENABLE_RMD=${_GGML_GEMMINI_ENABLE_RMD_VALUE}
        LOG_CYCLE=${LOG_CYCLE} CYCLE_DETAIL=${CYCLE_DETAIL})
    add_test(NAME test-gemmini-matmul COMMAND $<TARGET_FILE:test-gemmini-matmul>)
    add_test(
        NAME test-gemmini-matmul-finite-output
        COMMAND $<TARGET_FILE:test-gemmini-matmul> --case=finite-output)
    find_package(Threads REQUIRED)
    add_executable(test-gemmini-buffer-artifacts test-gemmini-buffer-artifacts.cpp)
    target_link_libraries(test-gemmini-buffer-artifacts PRIVATE ggml-gemmini ggml-base Threads::Threads)
    target_compile_features(test-gemmini-buffer-artifacts PRIVATE cxx_std_17)
    add_test(NAME test-gemmini-buffer-artifacts COMMAND $<TARGET_FILE:test-gemmini-buffer-artifacts>)
endif()

endif() # TARGET ggml-gemmini

llama_build_and_test(test-model-load-cancel.cpp  LABEL "model")
llama_build_and_test(test-autorelease.cpp        LABEL "model")

if (NOT GGML_BACKEND_DL)
    # these tests use the backends directly and cannot be built with dynamic loading
    llama_build_and_test(test-barrier.cpp)
    llama_build_and_test(test-quantize-fns.cpp)
    llama_build_and_test(test-quantize-perf.cpp)
    llama_build_and_test(test-rope.cpp)
endif()

if (TARGET ggml-gemmini)
add_executable(gemmini-quant-dispatch-smoke
    gemmini-quant-dispatch-smoke.cpp
    ../ggml/src/ggml-gemmini/quants/common/dequant.cpp
    ../ggml/src/ggml-gemmini/quants/common/weight_reader.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-compose.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp
    ../ggml/src/ggml-gemmini/quants/act/dispatch.cpp
    ../ggml/src/ggml-gemmini/quants/act/block/block.cpp
    ../ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp
    ../ggml/src/ggml-gemmini/quants/act/tensor/tensor.cpp
    ../ggml/src/ggml-gemmini/quants/act/token/token.cpp)
target_link_libraries(gemmini-quant-dispatch-smoke PRIVATE ggml-base ggml-gemmini-utils)
target_compile_features(gemmini-quant-dispatch-smoke PRIVATE cxx_std_17)
target_compile_definitions(gemmini-quant-dispatch-smoke PRIVATE
    GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
    GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE}
    GGML_GEMMINI_ACTIVATION_QUANT=2
    GGML_GEMMINI_ENABLE_RMD=${_GGML_GEMMINI_ENABLE_RMD_VALUE}
    GGML_GEMMINI_EXSIA_SIGMA=${GGML_GEMMINI_EXSIA_SIGMA})
target_include_directories(gemmini-quant-dispatch-smoke PRIVATE
    ../ggml/src
    ../ggml/src/ggml-gemmini
    ../ggml/src/ggml-gemmini/quants
    ../ggml/src/ggml-gemmini/quants/act
    ../ggml/src/ggml-gemmini/quants/act/exsia
    ../ggml/src/ggml-gemmini/quants/act/token
    ../ggml/src/ggml-gemmini/quants/common
    ${GEMMINI_SW_PATH} ${GEMMINI_SW_PATH}/include)
llama_test(gemmini-quant-dispatch-smoke NAME gemmini_quant_dispatch_fp          ARGS fp)
llama_test(gemmini-quant-dispatch-smoke NAME gemmini_quant_dispatch_exsia       ARGS exsia)
llama_test(gemmini-quant-dispatch-smoke NAME gemmini_quant_dispatch_tensor     ARGS tensor)
llama_test(gemmini-quant-dispatch-smoke NAME gemmini_quant_dispatch_token      ARGS token)
add_test(
    NAME gemmini_quant_dispatch_unsupported
    COMMAND ${CMAKE_COMMAND}
        -DTEST_BINARY=$<TARGET_FILE:gemmini-quant-dispatch-smoke>
        -P ${CMAKE_CURRENT_SOURCE_DIR}/gemmini-quant-dispatch-unsupported.cmake)
set_property(TEST gemmini_quant_dispatch_unsupported PROPERTY LABELS main)

add_test(NAME test-gemmini-matmul-cpu-boundaries
    COMMAND ${CMAKE_COMMAND}
        -DMATMUL_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-gemmini/ggml-gemmini-matmul.cpp
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test-gemmini-matmul-cpu-boundaries.cmake)

add_test(NAME test-gemmini-profiled-domain-contract
    COMMAND ${CMAKE_COMMAND}
        -DMATMUL_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-gemmini/ggml-gemmini-matmul.cpp
        -DMATMUL_HEADER=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-gemmini/ggml-gemmini-matmul.hpp
        -DMATMUL_TELEMETRY=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp
        -DGEMMINI_ENTRY_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-gemmini/ggml-gemmini.cpp
        -DGEMMINI_SOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-gemmini
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test-gemmini-profiled-domain-contract.cmake)

endif() # TARGET ggml-gemmini

# Keep the executable side of Gemmini's CTest registration explicit. Building
# this target before CTest prevents a stale tree from hiding missing artifacts.
set(_GEMMINI_REGISTERED_TEST_EXECUTABLES
    test-gemmini-log
    test-gemmini-host-timing
    test-gemmini-trace-context
    test-gemmini-log-boundary)
if (TARGET ggml-gemmini)
list(APPEND _GEMMINI_REGISTERED_TEST_EXECUTABLES
    test-gemmini-rmd-capture
    test-gemmini-act-quant-i32-saturation
    test-gemmini-exsia
    test-gemmini-exsia-writer
    test-gemmini-rmd-packet
    test-gemmini-production-host-backend
    test-gemmini-rmd-reducer
    gemmini-quant-dispatch-smoke)
endif()
foreach(_GEMMINI_OPTIONAL_TEST_EXECUTABLE IN ITEMS
        test-gemmini-exsia-workers3
        test-gemmini-exsia-workers4
        test-gemmini-block-quant
        test-gemmini-block-dense
        test-gemmini-block-end-to-end
        test-gemmini-cycle-reader-aarch64
        gemmini_weight_formats
        test-gemmini-rmd
        test-gemmini-rmd-telemetry
        gemmini_q8_h1_h2_dequant_fp
        gemmini_width_h1_dequant_fp
        test-gemmini-srmd-raw-ws
        test-gemmini-matmul-options
        test-gemmini-im2p-routing
        test-gemmini-im2p-host-cycle
        test-gemmini-im2p-lifecycle
        test-gemmini-rmd-im2p-provider
        test-gemmini-rmd-backend-gate
        test-gemmini-matmul
        test-gemmini-buffer-artifacts
        test-gemmini-log-cross-image)
    if (TARGET ${_GEMMINI_OPTIONAL_TEST_EXECUTABLE})
        list(APPEND _GEMMINI_REGISTERED_TEST_EXECUTABLES
            ${_GEMMINI_OPTIONAL_TEST_EXECUTABLE})
    endif()
endforeach()
if (_GEMMINI_REGISTERED_TEST_EXECUTABLES_APPEND_CPU_SINK)
    list(APPEND _GEMMINI_REGISTERED_TEST_EXECUTABLES
        test-ggml-cpu-cycle-sink
    )
endif()
foreach(_GEMMINI_TEST_EXECUTABLE IN LISTS _GEMMINI_REGISTERED_TEST_EXECUTABLES)
    if (NOT TARGET ${_GEMMINI_TEST_EXECUTABLE})
        message(FATAL_ERROR
            "Registered Gemmini test executable has no target: ${_GEMMINI_TEST_EXECUTABLE}")
    endif()
    target_compile_definitions(${_GEMMINI_TEST_EXECUTABLE} PRIVATE CYCLE_SIM=${CYCLE_SIM})
endforeach()
add_custom_target(gemmini-test-executables ALL
    DEPENDS ${_GEMMINI_REGISTERED_TEST_EXECUTABLES})

# libmtmd
set(LLAMA_TEST_NAME test-mtmd-c-api)
llama_build_and_test(test-mtmd-c-api.c)
target_link_libraries(${LLAMA_TEST_NAME} PRIVATE mtmd)

# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)

# This test deliberately links no simulator archive: injected compact execution
# must work with a null simulator and reject incomplete or duplicate outputs.
if (TARGET ggml-gemmini AND GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "FPGA_UART")
    add_executable(test-gemmini-rmd-executor
        test-gemmini-rmd-executor.cpp
        ../ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp)
    target_compile_features(test-gemmini-rmd-executor PRIVATE cxx_std_17)
    target_compile_definitions(test-gemmini-rmd-executor PRIVATE ${GGML_GEMMINI_COMPILE_DEFS})
    target_include_directories(test-gemmini-rmd-executor PRIVATE
        "${GGML_GEMMINI_GENERATED_CONFIG_DIR}"
        "${IM2P_SIM_ROOT}/sim/include"
        "${GEMMINI_SW_PATH}" "${GEMMINI_SW_PATH}/include"
        ../ggml/src/ggml-gemmini ../ggml/src/ggml-gemmini-utils/include ../ggml/include)
    add_test(NAME test-gemmini-rmd-executor COMMAND $<TARGET_FILE:test-gemmini-rmd-executor>)
endif()
