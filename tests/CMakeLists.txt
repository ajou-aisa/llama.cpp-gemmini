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

# llama_build_and_test(test-opt.cpp) # SLOW
llama_build_and_test(test-gguf.cpp)
llama_build_and_test(test-backend-ops.cpp)
add_executable(test-gemmini-log test-gemmini-log.cpp)
target_link_libraries(test-gemmini-log PRIVATE ggml-gemmini-utils)
target_compile_features(test-gemmini-log PRIVATE cxx_std_17)
target_compile_definitions(test-gemmini-log PRIVATE
    EXPECT_LOG_DEBUG=$<IF:$<BOOL:${LOG_DEBUG}>,1,0>
    EXPECT_LOG_CYCLE=$<IF:$<BOOL:${LOG_CYCLE}>,1,0>)
target_include_directories(test-gemmini-log PRIVATE
    ../ggml/src/ggml-gemmini-utils/include)
add_test(NAME test-gemmini-log
    COMMAND $<TARGET_FILE:test-gemmini-log> ${CMAKE_CURRENT_BINARY_DIR}/gemmini-log-regression)
if (LOG_DEBUG)
    add_test(NAME test-gemmini-log-explicit-debug-before-default
        COMMAND $<TARGET_FILE:test-gemmini-log> --explicit-debug-before-default)
endif()
llama_build_and_test(test-gemmini-q8-h1-h2-dequant-fp.cpp NAME gemmini_q8_h1_h2_dequant_fp)
target_sources(gemmini_q8_h1_h2_dequant_fp PRIVATE
    ../ggml/src/ggml-gemmini/quants/common/dequant.cpp
    ../ggml/src/ggml-gemmini/quants/act/dispatch.cpp
    ../ggml/src/ggml-gemmini/quants/act/quantize.cpp
    ../ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp
    ../ggml/src/ggml-gemmini/quants/act/tensor/tensor.cpp
    ../ggml/src/ggml-gemmini/quants/act/token/token.cpp)
target_link_libraries(gemmini_q8_h1_h2_dequant_fp PRIVATE ggml-base ggml-gemmini-utils)
target_compile_definitions(gemmini_q8_h1_h2_dequant_fp PRIVATE
    GGML_GEMMINI_ENABLE_RMD=${_GGML_GEMMINI_ENABLE_RMD_VALUE})
target_include_directories(gemmini_q8_h1_h2_dequant_fp PRIVATE
    ../ggml/src
    ../ggml/src/ggml-gemmini
    ../ggml/src/ggml-gemmini-utils/include)

add_executable(test-gemmini-act-quant-i32-saturation
    test-gemmini-act-quant-i32-saturation.cpp
    ../ggml/src/ggml-gemmini/quants/common/dequant.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-compose.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp
    ../ggml/src/ggml-gemmini/quants/act/dispatch.cpp
    ../ggml/src/ggml-gemmini/quants/act/quantize.cpp
    ../ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp
    ../ggml/src/ggml-gemmini/quants/act/tensor/tensor.cpp
    ../ggml/src/ggml-gemmini/quants/act/token/token.cpp
    ../ggml/src/ggml-gemmini/quants/act/stripe/stripe.cpp)
target_link_libraries(test-gemmini-act-quant-i32-saturation PRIVATE ggml-base ggml-gemmini-utils)
target_compile_features(test-gemmini-act-quant-i32-saturation PRIVATE cxx_std_17)
target_compile_definitions(test-gemmini-act-quant-i32-saturation PRIVATE
    GGML_GEMMINI_ACTIVATION_QUANT=${_GGML_GEMMINI_ACTIVATION_QUANT_VALUE}
    GGML_GEMMINI_EXSIA_SIGMA=${GGML_GEMMINI_EXSIA_SIGMA}
    GGML_GEMMINI_ENABLE_RMD=1)
target_include_directories(test-gemmini-act-quant-i32-saturation PRIVATE
    ../ggml/include
    ../ggml/src
    ../ggml/src/ggml-gemmini
    ../ggml/src/ggml-gemmini-utils/include)
add_test(NAME test-gemmini-act-quant-i32-saturation
    COMMAND $<TARGET_FILE:test-gemmini-act-quant-i32-saturation>)

add_executable(test-gemmini-exsia
    test-gemmini-exsia.cpp
    ../ggml/src/ggml-gemmini/quants/common/dequant.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-compose.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp
    ../ggml/src/ggml-gemmini/quants/act/dispatch.cpp
    ../ggml/src/ggml-gemmini/quants/act/quantize.cpp
    ../ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp
    ../ggml/src/ggml-gemmini/quants/act/tensor/tensor.cpp
    ../ggml/src/ggml-gemmini/quants/act/token/token.cpp)
target_link_libraries(test-gemmini-exsia PRIVATE ggml-base ggml-gemmini-utils)
target_compile_features(test-gemmini-exsia PRIVATE cxx_std_17)
target_compile_definitions(test-gemmini-exsia PRIVATE
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
    ../ggml/src/ggml-gemmini/quants/common)
foreach(_case IN ITEMS all baseline dispatch rmd-routes rmd-direct-parity)
    string(REPLACE "-" "_" _case_test_name "${_case}")
    add_test(NAME gemmini_exsia_${_case_test_name}
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXECUTABLE=$<TARGET_FILE:test-gemmini-exsia>
            -DEXPECTED_CASE=${_case}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/test-gemmini-exsia-case.cmake)
endforeach()
add_test(NAME gemmini_exsia_unknown_case
    COMMAND $<TARGET_FILE:test-gemmini-exsia> --case=unknown)
set_tests_properties(gemmini_exsia_unknown_case PROPERTIES WILL_FAIL TRUE)

add_executable(test-gemmini-exsia-writer
    test-gemmini-exsia.cpp
    ../ggml/src/ggml-gemmini/quants/common/dequant.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-compose.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp
    ../ggml/src/ggml-gemmini/quants/act/dispatch.cpp
    ../ggml/src/ggml-gemmini/quants/act/quantize.cpp
    ../ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp
    ../ggml/src/ggml-gemmini/quants/act/tensor/tensor.cpp
    ../ggml/src/ggml-gemmini/quants/act/token/token.cpp)
target_link_libraries(test-gemmini-exsia-writer PRIVATE ggml-base ggml-gemmini-utils)
target_compile_features(test-gemmini-exsia-writer PRIVATE cxx_std_17)
target_compile_definitions(test-gemmini-exsia-writer PRIVATE
    GEMMINI_EXSIA_WRITER_TEST_ONLY=1
    GGML_GEMMINI_EXSIA_SIGMA=${GGML_GEMMINI_EXSIA_SIGMA}
    GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE=1
    GGML_GEMMINI_ENABLE_RMD=${_GGML_GEMMINI_ENABLE_RMD_VALUE}
    CYCLE_DETAIL=1
    LOG_CYCLE=1
    EXSIA_VALIDATION=1)
target_include_directories(test-gemmini-exsia-writer PRIVATE
    ../ggml/include ../ggml/src ../ggml/src/ggml-gemmini
    ../ggml/src/ggml-gemmini-utils/include ../ggml/src/ggml-gemmini/quants
    ../ggml/src/ggml-gemmini/quants/act ../ggml/src/ggml-gemmini/quants/act/exsia
    ../ggml/src/ggml-gemmini/quants/common ${GEMMINI_SW_PATH} ${GEMMINI_SW_PATH}/include)

set(GEMMINI_WRITER_TEST_ROOT "${CMAKE_CURRENT_BINARY_DIR}/gemmini-writer-output")
foreach(_case IN ITEMS exsia-default exsia-dir exsia-relative exsia-absolute exsia-empty exsia-invalid
                       pipeline-default pipeline-dir pipeline-empty pipeline-invalid)
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
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp)
target_link_libraries(test-gemmini-rmd PRIVATE ggml-base ggml-gemmini-utils)
target_compile_features(test-gemmini-rmd PRIVATE cxx_std_17)
target_include_directories(test-gemmini-rmd PRIVATE
    ../ggml/src
    ../ggml/src/ggml-gemmini
    ../ggml/src/ggml-gemmini/residual/rmd
    ${GEMMINI_SW_PATH}
    ${GEMMINI_SW_PATH}/include)
add_test(NAME test-gemmini-rmd COMMAND $<TARGET_FILE:test-gemmini-rmd>)

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
execute_process(
    COMMAND ${GIT_EXECUTABLE} -C "${GEMMINI_SW_PATH}" rev-parse HEAD
    RESULT_VARIABLE GEMMINI_SRMD_RAW_WS_GIT_RESULT
    OUTPUT_VARIABLE GEMMINI_SRMD_RAW_WS_GIT_HEAD
    OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(
    COMMAND ${GIT_EXECUTABLE} -C "${GEMMINI_SW_PATH}" status --porcelain
    RESULT_VARIABLE GEMMINI_SRMD_RAW_WS_STATUS_RESULT
    OUTPUT_VARIABLE GEMMINI_SRMD_RAW_WS_GIT_STATUS
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if (NOT GEMMINI_SRMD_RAW_WS_GIT_RESULT EQUAL 0 OR
    NOT GEMMINI_SRMD_RAW_WS_STATUS_RESULT EQUAL 0 OR
    NOT GEMMINI_SRMD_RAW_WS_GIT_HEAD STREQUAL
        "2416775820be7da60a310570964b3c97743b9007" OR
    GEMMINI_SRMD_RAW_WS_GIT_STATUS)
    message(FATAL_ERROR "test-gemmini-srmd-raw-ws requires the pinned clean Gemmini checkout")
endif()
get_filename_component(GEMMINI_SRMD_RAW_WS_INCLUDE_DIR
    "${GEMMINI_SRMD_RAW_WS_HEADER}" DIRECTORY)
add_executable(test-gemmini-srmd-raw-ws
    test-gemmini-srmd-raw-ws.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp)
target_link_libraries(test-gemmini-srmd-raw-ws PRIVATE ggml-base ggml-gemmini-utils)
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

add_executable(test-gemmini-matmul-options test-gemmini-matmul-options.cpp)
target_compile_features(test-gemmini-matmul-options PRIVATE cxx_std_17)
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
get_filename_component(LLAMA_TEST_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
if (EXISTS "${CMAKE_COMMAND}" AND EXISTS "${LLAMA_TEST_SOURCE_DIR}/CMakeLists.txt")
    add_test(
        NAME test-gemmini-build-contract
        COMMAND ${CMAKE_COMMAND}
            -DTEST_CMAKE_COMMAND=${CMAKE_COMMAND}
            -DTEST_SOURCE_DIR=${LLAMA_TEST_SOURCE_DIR}
            -DTEST_BINARY_ROOT=${CMAKE_CURRENT_BINARY_DIR}/gemmini-build-contract
            -DTEST_GENERATOR=${CMAKE_GENERATOR}
            -DTEST_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}
            -DTEST_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/test-gemmini-build-contract.cmake)
endif()

if (NOT GGML_BACKEND_DL)
    # The façade test uses C++ symbols from the statically linked backend.
    # Dynamic backends are MODULE libraries and cannot be linked into a test.
    add_executable(test-gemmini-matmul test-gemmini-matmul.cpp)
    target_link_libraries(test-gemmini-matmul PRIVATE ggml-gemmini ggml-base)
    target_compile_features(test-gemmini-matmul PRIVATE cxx_std_17)
    add_test(NAME test-gemmini-matmul COMMAND $<TARGET_FILE:test-gemmini-matmul>)
    add_executable(test-gemmini-pipeline-writer
        test-gemmini-matmul.cpp
        ../ggml/src/ggml-gemmini/ggml-gemmini-matmul.cpp)
    target_link_libraries(test-gemmini-pipeline-writer PRIVATE ggml-gemmini-utils)
    target_compile_features(test-gemmini-pipeline-writer PRIVATE cxx_std_17)
    target_compile_definitions(test-gemmini-pipeline-writer PRIVATE
        GGML_GEMMINI_PIPELINE_WRITER_TEST_ONLY=1)
    target_include_directories(test-gemmini-pipeline-writer PRIVATE
        ../ggml/include ../ggml/src ../ggml/src/ggml-gemmini
        ../ggml/src/ggml-gemmini-utils/include ${GGML_GEMMINI_GENERATED_CONFIG_DIR}
        ${GEMMINI_SW_PATH} ${GEMMINI_SW_PATH}/include)
    add_test(NAME gemmini-matmul-output-default
        COMMAND ${CMAKE_COMMAND} -E env --unset=GEMMINI_LOG_DIR
            $<TARGET_FILE:test-gemmini-pipeline-writer> --pipeline-output output/log/debug-log.jsonl)
    set_tests_properties(gemmini-matmul-output-default PROPERTIES WORKING_DIRECTORY "${GEMMINI_WRITER_TEST_ROOT}/pipeline-default")
    add_test(NAME gemmini-matmul-output-dir
        COMMAND ${CMAKE_COMMAND} -E env GEMMINI_LOG_DIR=env-log
            $<TARGET_FILE:test-gemmini-pipeline-writer> --pipeline-output env-log/debug-log.jsonl)
    set_tests_properties(gemmini-matmul-output-dir PROPERTIES WORKING_DIRECTORY "${GEMMINI_WRITER_TEST_ROOT}/pipeline-dir")
    add_test(NAME gemmini-matmul-output-empty
        COMMAND ${CMAKE_COMMAND} -E env GEMMINI_LOG_DIR=
            $<TARGET_FILE:test-gemmini-pipeline-writer> --pipeline-output output/log/debug-log.jsonl)
    set_tests_properties(gemmini-matmul-output-empty PROPERTIES WORKING_DIRECTORY "${GEMMINI_WRITER_TEST_ROOT}/pipeline-empty")
    add_test(NAME gemmini-matmul-output-invalid-parent
        COMMAND ${CMAKE_COMMAND} -E env GEMMINI_LOG_DIR=blocked
            $<TARGET_FILE:test-gemmini-pipeline-writer> --pipeline-output blocked/debug-log.jsonl --invalid-parent)
    set_tests_properties(gemmini-matmul-output-invalid-parent PROPERTIES WORKING_DIRECTORY "${GEMMINI_WRITER_TEST_ROOT}/pipeline-invalid")

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
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-compose.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp
    ../ggml/src/ggml-gemmini/quants/act/dispatch.cpp
    ../ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp
    ../ggml/src/ggml-gemmini/quants/act/tensor/tensor.cpp
    ../ggml/src/ggml-gemmini/quants/act/token/token.cpp)
target_link_libraries(gemmini-quant-dispatch-smoke PRIVATE ggml-base ggml-gemmini-utils)
target_compile_features(gemmini-quant-dispatch-smoke PRIVATE cxx_std_17)
target_compile_definitions(gemmini-quant-dispatch-smoke PRIVATE
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

# libmtmd
set(LLAMA_TEST_NAME test-mtmd-c-api)
llama_build_and_test(test-mtmd-c-api.c)
target_link_libraries(${LLAMA_TEST_NAME} PRIVATE mtmd)

# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)
