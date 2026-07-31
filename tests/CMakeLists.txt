llama_add_compile_flags()

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
    EXPECT_LOG_DEBUG=$<IF:$<BOOL:${LOG_DEBUG}>,1,0>)
target_include_directories(test-gemmini-log PRIVATE
    ../ggml/src/ggml-gemmini-utils/include)
add_test(NAME test-gemmini-log
    COMMAND $<TARGET_FILE:test-gemmini-log> ${CMAKE_CURRENT_BINARY_DIR}/gemmini-log-regression)
llama_build_and_test(test-gemmini-q8-h1-h2-dequant-fp.cpp NAME gemmini_q8_h1_h2_dequant_fp)
target_sources(gemmini_q8_h1_h2_dequant_fp PRIVATE
    ../ggml/src/ggml-gemmini/quants/common/dequant.cpp
    ../ggml/src/ggml-gemmini/quants/act/dispatch.cpp
    ../ggml/src/ggml-gemmini/quants/act/quantize.cpp
    ../ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp
    ../ggml/src/ggml-gemmini/quants/act/tensor/tensor.cpp
    ../ggml/src/ggml-gemmini/quants/act/token/token.cpp)
target_link_libraries(gemmini_q8_h1_h2_dequant_fp PRIVATE ggml-base ggml-gemmini-utils)
target_include_directories(gemmini_q8_h1_h2_dequant_fp PRIVATE
    ../ggml/src
    ../ggml/src/ggml-gemmini
    ../ggml/src/ggml-gemmini-utils/include)

add_executable(test-gemmini-exsia
    test-gemmini-exsia.cpp
    ../ggml/src/ggml-gemmini/quants/common/dequant.cpp
    ../ggml/src/ggml-gemmini/quants/act/dispatch.cpp
    ../ggml/src/ggml-gemmini/quants/act/quantize.cpp
    ../ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp
    ../ggml/src/ggml-gemmini/quants/act/tensor/tensor.cpp
    ../ggml/src/ggml-gemmini/quants/act/token/token.cpp)
target_link_libraries(test-gemmini-exsia PRIVATE ggml-base ggml-gemmini-utils)
target_compile_features(test-gemmini-exsia PRIVATE cxx_std_17)
target_compile_definitions(test-gemmini-exsia PRIVATE
    GGML_GEMMINI_EXSIA_SIGMA=${GGML_GEMMINI_EXSIA_SIGMA}
    CYCLE_DETAIL=${CYCLE_DETAIL})
target_include_directories(test-gemmini-exsia PRIVATE
    ../ggml/include
    ../ggml/src
    ../ggml/src/ggml-gemmini
    ../ggml/src/ggml-gemmini-utils/include
    ../ggml/src/ggml-gemmini/quants
    ../ggml/src/ggml-gemmini/quants/act
    ../ggml/src/ggml-gemmini/quants/act/exsia
    ../ggml/src/ggml-gemmini/quants/common)
add_test(NAME test-gemmini-exsia COMMAND $<TARGET_FILE:test-gemmini-exsia>)
add_test(NAME test-gemmini-exsia-stream-parity
    COMMAND $<TARGET_FILE:test-gemmini-exsia> --case=stream-parity --rows=129 --cols=37)
add_test(NAME test-gemmini-exsia-stream-invalid-shape
    COMMAND $<TARGET_FILE:test-gemmini-exsia> --case=stream-invalid-shape --rows=0 --cols=37)
add_test(NAME gemmini_exsia_all
    COMMAND $<TARGET_FILE:test-gemmini-exsia> --case=all --rows=65 --cols=37 --compiled-tau=${GGML_GEMMINI_EXSIA_SIGMA})
add_test(NAME gemmini_exsia_shift_extremes
    COMMAND $<TARGET_FILE:test-gemmini-exsia> --case=shift-extremes --compiled-tau=${GGML_GEMMINI_EXSIA_SIGMA})
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
    ../ggml/src/ggml-gemmini/quants/act/dispatch.cpp
    ../ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp
    ../ggml/src/ggml-gemmini/quants/act/tensor/tensor.cpp
    ../ggml/src/ggml-gemmini/quants/act/token/token.cpp)
target_link_libraries(gemmini-quant-dispatch-smoke PRIVATE ggml-base ggml-gemmini-utils)
target_compile_features(gemmini-quant-dispatch-smoke PRIVATE cxx_std_17)
target_compile_definitions(gemmini-quant-dispatch-smoke PRIVATE
    GGML_GEMMINI_ACTIVATION_QUANT=2
    GGML_GEMMINI_EXSIA_SIGMA=${GGML_GEMMINI_EXSIA_SIGMA})
target_include_directories(gemmini-quant-dispatch-smoke PRIVATE
    ../ggml/src
    ../ggml/src/ggml-gemmini
    ../ggml/src/ggml-gemmini/quants
    ../ggml/src/ggml-gemmini/quants/act
    ../ggml/src/ggml-gemmini/quants/act/exsia
    ../ggml/src/ggml-gemmini/quants/act/token
    ../ggml/src/ggml-gemmini/quants/common
    ../ggml/src/ggml-gemmini/quants/dec)
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
