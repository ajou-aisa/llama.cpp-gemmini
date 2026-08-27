llama_add_compile_flags()
add_compile_definitions(GGML_GEMMINI_BLOCK_SIZE=${_GGML_GEMMINI_BLOCK_SIZE_VALUE})

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
add_test(NAME test-build-arm64-log-defaults
    COMMAND ${CMAKE_COMMAND}
        -DTEST_SCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/../build-arm64.sh
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test-build-arm64-log-defaults.cmake)

# llama_build_and_test(test-opt.cpp) # SLOW
llama_build_and_test(test-gguf.cpp)
llama_build_and_test(test-backend-ops.cpp)
add_executable(test-gemmini-log test-gemmini-log.cpp)
target_link_libraries(test-gemmini-log PRIVATE ggml-gemmini-utils llama)
target_compile_features(test-gemmini-log PRIVATE cxx_std_17)
target_compile_definitions(test-gemmini-log PRIVATE
    EXPECT_LOG_DEBUG=$<IF:$<BOOL:${LOG_DEBUG}>,1,0>
    EXPECT_LOG_CYCLE=$<IF:$<BOOL:${LOG_CYCLE}>,1,0>)
target_include_directories(test-gemmini-log PRIVATE
    ../ggml/src/ggml-gemmini-utils/include
    ../src)
add_test(NAME test-gemmini-log
    COMMAND $<TARGET_FILE:test-gemmini-log> ${CMAKE_CURRENT_BINARY_DIR}/gemmini-log-regression)
add_test(NAME test-llama-log-callback
    COMMAND $<TARGET_FILE:test-gemmini-log> --llama-callback)
add_executable(test-gemmini-log-boundary
    test-gemmini-log-boundary.cpp
    test-gemmini-log-c-boundary.c)
target_link_libraries(test-gemmini-log-boundary PRIVATE ggml-gemmini-utils)
target_compile_features(test-gemmini-log-boundary PRIVATE cxx_std_17)
target_compile_definitions(test-gemmini-log-boundary PRIVATE
    EXPECT_LOG_DEBUG=$<IF:$<BOOL:${LOG_DEBUG}>,1,0>
    EXPECT_LOG_CYCLE=$<IF:$<BOOL:${LOG_CYCLE}>,1,0>)
add_test(NAME test-gemmini-log-boundary
    COMMAND $<TARGET_FILE:test-gemmini-log-boundary>)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    find_package(Threads REQUIRED)
    add_executable(test-gemmini-cycle-reader-aarch64
        test-gemmini-cycle-reader-aarch64.cpp
        ../ggml/src/ggml-gemmini-utils/src/cycle_reader_aarch64.cpp
        ../ggml/src/ggml-gemmini-utils/src/cycle_reader-capi.cpp)
    target_include_directories(test-gemmini-cycle-reader-aarch64 PRIVATE
        ../ggml/src/ggml-gemmini-utils/include)
    target_compile_definitions(test-gemmini-cycle-reader-aarch64 PRIVATE
        GGML_GEMMINI_TESTING=1 LOG_CYCLE=1)
    target_compile_features(test-gemmini-cycle-reader-aarch64 PRIVATE cxx_std_17)
    target_link_libraries(test-gemmini-cycle-reader-aarch64 PRIVATE Threads::Threads)
    add_test(NAME test-gemmini-cycle-reader-aarch64
        COMMAND $<TARGET_FILE:test-gemmini-cycle-reader-aarch64>)
endif()
if (LOG_CYCLE AND GGML_CPU_CYCLE_LOG AND TARGET ggml-cpu)
    add_executable(test-ggml-cpu-cycle-sink test-ggml-cpu-cycle-sink.cpp)
    target_link_libraries(test-ggml-cpu-cycle-sink PRIVATE ggml-base ggml-cpu ggml-gemmini-utils)
    target_compile_features(test-ggml-cpu-cycle-sink PRIVATE cxx_std_17)
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
llama_build_and_test(test-gemmini-weight-formats.cpp NAME gemmini_weight_formats)
target_link_libraries(gemmini_weight_formats PRIVATE ggml-base ggml-gemmini)
target_compile_definitions(gemmini_weight_formats PRIVATE
    GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
    GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE})
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
    # BLOCK intentionally has no residual/RMD path; keep this RMD saturation test on ExSIA.
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
    GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
    GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE}
    GGML_GEMMINI_EXSIA_SIGMA=${GGML_GEMMINI_EXSIA_SIGMA}
    GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE=${GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE}
    GGML_GEMMINI_ENABLE_RMD=${_GGML_GEMMINI_ENABLE_RMD_VALUE}
    CYCLE_DETAIL=${CYCLE_DETAIL}
    LOG_CYCLE=${LOG_CYCLE}
    EXSIA_VALIDATION=1)
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
set(_GGML_GEMMINI_EXSIA_CASES
    all baseline dispatch direct-executor stripe-geometry)
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

add_executable(test-gemmini-rmd
    test-gemmini-rmd.cpp
    ../ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp)
target_link_libraries(test-gemmini-rmd PRIVATE ggml-base ggml-gemmini ggml-gemmini-utils)
target_compile_features(test-gemmini-rmd PRIVATE cxx_std_17)
target_compile_definitions(test-gemmini-rmd PRIVATE
    GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
    GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE})
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

add_executable(test-gemmini-rmd-packet
    test-gemmini-rmd-packet.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp)
target_link_libraries(test-gemmini-rmd-packet PRIVATE
    ggml-base ggml-gemmini ggml-gemmini-utils)
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

add_executable(test-gemmini-production-host-backend
    test-gemmini-production-host-backend.cpp)
target_compile_features(test-gemmini-production-host-backend PRIVATE cxx_std_17)
target_compile_definitions(test-gemmini-production-host-backend PRIVATE
    GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
    GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE})
target_include_directories(test-gemmini-production-host-backend PRIVATE
    ../ggml/src
    ../ggml/src/ggml-gemmini
    ${GGML_GEMMINI_GENERATED_CONFIG_DIR}
    ${GEMMINI_SW_PATH}
    ${GEMMINI_SW_PATH}/include)
add_test(NAME test-gemmini-production-host
    COMMAND $<TARGET_FILE:test-gemmini-production-host-backend>)

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
    ${IM2P_SIM_ROOT}/frontend/include ${IM2P_SIM_ROOT}/sim/include
    ${GEMMINI_SW_PATH} ${GEMMINI_SW_PATH}/include)
add_test(NAME test-gemmini-rmd-telemetry
    COMMAND $<TARGET_FILE:test-gemmini-rmd-telemetry>)
add_test(NAME test-gemmini-aggregate-cycle-driver
    COMMAND $<TARGET_FILE:test-gemmini-rmd-telemetry> --aggregate-driver
            ${CMAKE_CURRENT_BINARY_DIR}/gemmini-aggregate-driver/cycle-log.jsonl)
add_test(NAME test-gemmini-telemetry-clock-gate
    COMMAND ${CMAKE_COMMAND}
            -DTEST_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/..
            -P ${CMAKE_CURRENT_SOURCE_DIR}/test-gemmini-telemetry-clock-gate.cmake)

option(GEMMINI_BUILD_SRMD_RAW_WS_TESTS
    "Build the raw-WS SRMD tests against the configured Gemmini headers"
    OFF)
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
    add_test(NAME test-gemmini-srmd-layouts-cpu
        COMMAND $<TARGET_FILE:test-gemmini-srmd-raw-ws> --bench-layouts)
endif()

if (_GGML_GEMMINI_ACTIVATION_BITS_VALUE EQUAL 8 AND
    _GGML_GEMMINI_WEIGHT_BITS_VALUE EQUAL 8)
    add_executable(test-gemmini-matmul-options test-gemmini-matmul-options.cpp)
    target_compile_features(test-gemmini-matmul-options PRIVATE cxx_std_17)
    target_compile_definitions(test-gemmini-matmul-options PRIVATE
        GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
        GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE})
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
        target_link_libraries(test-gemmini-matmul-options PRIVATE ggml-gemmini ggml-base)
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
endif()

if (GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND NOT GGML_BACKEND_DL)
    add_executable(test-gemmini-im2p-routing test-gemmini-im2p-routing.cpp)
    target_link_libraries(test-gemmini-im2p-routing PRIVATE ggml-gemmini ggml-base)
    target_compile_features(test-gemmini-im2p-routing PRIVATE cxx_std_17)
    target_compile_definitions(test-gemmini-im2p-routing PRIVATE
        GGML_GEMMINI_TESTING=1
        GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
        GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE}
        GGML_GEMMINI_ACTIVATION_QUANT=${_GGML_GEMMINI_ACTIVATION_QUANT_VALUE}
        GGML_GEMMINI_ENABLE_RMD=${_GGML_GEMMINI_ENABLE_RMD_VALUE}
        GGML_GEMMINI_TEST_IM2P_DIM=${GGML_GEMMINI_DIM}
        LOG_CYCLE=${LOG_CYCLE}
        LOG_DEBUG=${LOG_DEBUG})
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
    add_test(NAME test-gemmini-im2p-routing-multiwidth-matrix
        COMMAND $<TARGET_FILE:test-gemmini-im2p-routing> --case multiwidth-matrix)
    add_test(NAME test-gemmini-im2p-routing-boundary-rejections
        COMMAND $<TARGET_FILE:test-gemmini-im2p-routing> --case boundary-rejections)
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
        GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
        GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE})
    target_include_directories(test-gemmini-rmd-im2p-provider PRIVATE
        ../ggml/src
        ../ggml/src/ggml-gemmini
        ${IM2P_SIM_ROOT}/sim/include
        ${GEMMINI_SW_PATH}
        ${GEMMINI_SW_PATH}/include)
    add_test(NAME test-gemmini-rmd-im2p-provider
        COMMAND $<TARGET_FILE:test-gemmini-rmd-im2p-provider> --case success)
    add_test(NAME test-gemmini-rmd-im2p-provider-failures
        COMMAND $<TARGET_FILE:test-gemmini-rmd-im2p-provider> --case all)

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
    target_link_libraries(test-gemmini-matmul PRIVATE ggml-gemmini ggml-base)
    target_compile_features(test-gemmini-matmul PRIVATE cxx_std_17)
    target_compile_definitions(test-gemmini-matmul PRIVATE
        GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
        GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE}
        GGML_GEMMINI_ENABLE_RMD=${_GGML_GEMMINI_ENABLE_RMD_VALUE})
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

llama_build_and_test(test-model-load-cancel.cpp  LABEL "model")
llama_build_and_test(test-autorelease.cpp        LABEL "model")

if (NOT GGML_BACKEND_DL)
    # these tests use the backends directly and cannot be built with dynamic loading
    llama_build_and_test(test-barrier.cpp)
    llama_build_and_test(test-quantize-fns.cpp)
    llama_build_and_test(test-quantize-perf.cpp)
    llama_build_and_test(test-rope.cpp)
endif()

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
    ../ggml/src/ggml-gemmini/quants/common)
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

# Keep the executable side of Gemmini's CTest registration explicit. Building
# this target before CTest prevents a stale tree from hiding missing artifacts.
set(_GEMMINI_REGISTERED_TEST_EXECUTABLES
    test-gemmini-log
    test-gemmini-log-boundary
    gemmini_weight_formats
    test-gemmini-act-quant-i32-saturation
    test-gemmini-exsia
    test-gemmini-exsia-writer
    test-gemmini-rmd
    test-gemmini-rmd-packet
    test-gemmini-production-host-backend
    test-gemmini-rmd-telemetry
    gemmini-quant-dispatch-smoke)
foreach(_GEMMINI_OPTIONAL_TEST_EXECUTABLE IN ITEMS
        gemmini_q8_h1_h2_dequant_fp
        gemmini_width_h1_dequant_fp
        test-gemmini-srmd-raw-ws
        test-gemmini-matmul-options
        test-gemmini-im2p-routing
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
