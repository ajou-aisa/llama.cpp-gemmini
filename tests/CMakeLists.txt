llama_add_compile_flags()
add_compile_definitions(GGML_GEMMINI_BLOCK_SIZE=${_GGML_GEMMINI_BLOCK_SIZE_VALUE})
add_compile_definitions(GGML_GEMMINI_EXSIA_STRIPE_ANCHOR=${GGML_GEMMINI_EXSIA_STRIPE_ANCHOR})

add_executable(test-cycle-sim-reader test-cycle-sim-reader.cpp)
target_compile_definitions(test-cycle-sim-reader PRIVATE LOG_CYCLE=${LOG_CYCLE} )
target_link_libraries(test-cycle-sim-reader PRIVATE ggml-npu-profiling)
add_test(NAME test-cycle-sim-reader COMMAND test-cycle-sim-reader)
if(LOG_CYCLE)
    add_executable(test-gemmini-semantic test-gemmini-semantic.cpp)
    target_compile_features(test-gemmini-semantic PRIVATE cxx_std_17)
    target_compile_options(test-gemmini-semantic PRIVATE -UNDEBUG)
    target_compile_definitions(test-gemmini-semantic PRIVATE LOG_CYCLE=${LOG_CYCLE} )
    target_link_libraries(test-gemmini-semantic PRIVATE ggml-npu-profiling)
    add_test(NAME test-gemmini-semantic COMMAND "${CMAKE_COMMAND}"
        "-DTEST_EXECUTABLE=$<TARGET_FILE:test-gemmini-semantic>"
        "-DTEST_OUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/test-gemmini-semantic.cmake")
endif()
add_executable(test-functional-emulation-exclusion test-functional-emulation-exclusion.cpp)
target_compile_features(test-functional-emulation-exclusion PRIVATE cxx_std_17)
target_compile_definitions(test-functional-emulation-exclusion PRIVATE
    LOG_CYCLE=${LOG_CYCLE} CYCLE_DETAIL=${CYCLE_DETAIL})
target_compile_options(test-functional-emulation-exclusion PRIVATE -UNDEBUG)
target_link_libraries(test-functional-emulation-exclusion PRIVATE ggml-npu-profiling Threads::Threads)
add_test(NAME test-functional-emulation-exclusion COMMAND test-functional-emulation-exclusion)
if(TARGET ggml-cpu AND NOT GGML_BACKEND_DL)
    add_executable(test-cycle-sim-cpu-smoke test-ggml-cpu-cycle-sink.cpp)
    target_compile_features(test-cycle-sim-cpu-smoke PRIVATE cxx_std_17)
    target_compile_definitions(test-cycle-sim-cpu-smoke PRIVATE
        LOG_CYCLE=${LOG_CYCLE}
        CYCLE_DETAIL=${CYCLE_DETAIL}
        EXPECT_LOG_CYCLE=${LOG_CYCLE}
        EXPECT_CPU_CYCLE_LOG=$<IF:$<BOOL:${GGML_CPU_CYCLE_LOG}>,1,0>)
    target_link_libraries(test-cycle-sim-cpu-smoke PRIVATE ggml-base ggml-cpu ggml-npu-profiling)
    add_test(NAME test-cycle-sim-cpu-smoke COMMAND "${CMAKE_COMMAND}"
        "-DTEST_EXECUTABLE=$<TARGET_FILE:test-cycle-sim-cpu-smoke>"
        "-DTEST_OUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/test-cycle-sim-cpu-smoke.cmake")
endif()
if(TARGET ggml-gemmini AND TARGET ggml-cpu AND LLAMA_BUILD_TOOLS)
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
        -DOPS_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-gemmini/ops.cpp
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
if(UNIX)
    add_test(NAME test-gemmini-off-build
        COMMAND ${CMAKE_COMMAND}
            -DTEST_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/..
            -P ${CMAKE_CURRENT_SOURCE_DIR}/test-gemmini-off-build.cmake)
endif()
add_test(NAME test-evaluation-build-options
    COMMAND ${CMAKE_COMMAND} -DTEST_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/..
        "-DTEST_GENERATOR=${CMAKE_GENERATOR}"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test-evaluation-build-options.cmake)
add_executable(test-evaluation-workload test-evaluation-workload.cpp)
target_link_libraries(test-evaluation-workload PRIVATE common llama)
target_compile_options(test-evaluation-workload PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>)
add_test(NAME test-evaluation-workload COMMAND test-evaluation-workload)
add_executable(test-evaluation-trace test-evaluation-trace.cpp ../tools/eval/evaluation-trace.cpp)
target_include_directories(test-evaluation-trace PRIVATE ../tools/eval)
target_link_libraries(test-evaluation-trace PRIVATE common llama ggml-npu-profiling)
target_compile_definitions(test-evaluation-trace PRIVATE ${GGML_GEMMINI_RUNTIME_COMPILE_DEFS})
target_compile_options(test-evaluation-trace PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>)
add_test(NAME test-evaluation-trace
    COMMAND ${CMAKE_COMMAND} -DTEST_BINARY=$<TARGET_FILE:test-evaluation-trace>
        -DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test-evaluation-metric-sink.cmake)
add_executable(test-evaluation-lifecycle test-evaluation-lifecycle.cpp)
target_include_directories(test-evaluation-lifecycle PRIVATE ../common)
target_compile_features(test-evaluation-lifecycle PRIVATE cxx_std_17)
target_compile_options(test-evaluation-lifecycle PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>)
add_test(NAME test-evaluation-lifecycle COMMAND test-evaluation-lifecycle)
add_executable(test-gemmini-evaluation-metrics test-gemmini-evaluation-metrics.cpp)
target_link_libraries(test-gemmini-evaluation-metrics PRIVATE ggml-npu-profiling)
target_compile_definitions(test-gemmini-evaluation-metrics PRIVATE
    GGML_GEMMINI_SCALE_METRICS=${GGML_GEMMINI_SCALE_METRICS}
    GGML_GEMMINI_ACT_QUANT_METRICS=${GGML_GEMMINI_ACT_QUANT_METRICS}
    GGML_GEMMINI_RESIDUAL_METRICS=${GGML_GEMMINI_RESIDUAL_METRICS})
target_compile_options(test-gemmini-evaluation-metrics PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>)
add_test(NAME test-gemmini-evaluation-metrics
    COMMAND ${CMAKE_COMMAND} -DTEST_BINARY=$<TARGET_FILE:test-gemmini-evaluation-metrics>
        -DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test-evaluation-metric-sink.cmake)
add_test(NAME test-potal-target-profile
    COMMAND ${CMAKE_COMMAND} -DTEST_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/..
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test-potal-target-profile.cmake)

# llama_build_and_test(test-opt.cpp) # SLOW
llama_build_and_test(test-gguf.cpp)
llama_build_and_test(test-backend-ops.cpp)
add_executable(test-gemmini-log test-gemmini-log.cpp)
target_link_libraries(test-gemmini-log PRIVATE ggml-npu-profiling llama)
target_compile_features(test-gemmini-log PRIVATE cxx_std_17)
target_compile_definitions(test-gemmini-log PRIVATE
    EXPECT_LOG_DEBUG=$<IF:$<BOOL:${LOG_DEBUG}>,1,0>
    EXPECT_LOG_CYCLE=$<IF:$<BOOL:${LOG_CYCLE}>,1,0>
    EXPECT_CYCLE_DETAIL=$<IF:$<BOOL:${CYCLE_DETAIL}>,1,0>)
target_include_directories(test-gemmini-log PRIVATE
    ../ggml/src/ggml-npu-common/profiling/include
    ../src)
add_test(NAME test-gemmini-log
    COMMAND $<TARGET_FILE:test-gemmini-log> ${CMAKE_CURRENT_BINARY_DIR}/gemmini-log-regression)
add_test(NAME test-llama-log-callback
    COMMAND $<TARGET_FILE:test-gemmini-log> --llama-callback)
add_executable(test-gemmini-optrace test-gemmini-optrace.cpp)
target_link_libraries(test-gemmini-optrace PRIVATE ggml-npu-profiling)
target_compile_features(test-gemmini-optrace PRIVATE cxx_std_17)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(test-gemmini-optrace PRIVATE -UNDEBUG)
endif()
add_test(NAME test-gemmini-optrace
    COMMAND $<TARGET_FILE:test-gemmini-optrace> ${CMAKE_CURRENT_BINARY_DIR}/gemmini-optrace)
add_executable(test-gemmini-log-boundary
    test-gemmini-log-boundary.cpp
    test-gemmini-log-c-boundary.c)
target_link_libraries(test-gemmini-log-boundary PRIVATE ggml-npu-profiling)
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
        ../ggml/src/ggml-npu-common/profiling/src)
endif()
add_test(NAME test-gemmini-log-boundary
    COMMAND $<TARGET_FILE:test-gemmini-log-boundary>)
if (GGML_GEMMINI AND LOG_DUMP AND NOT GGML_BACKEND_DL AND
    GGML_GEMMINI_OPTION STREQUAL "CPU" AND GGML_GEMMINI_COMPUTE_TYPE STREQUAL "FLOAT" AND
    NOT GGML_GEMMINI_DEQUANT_FP_TEST)
    add_executable(test-gemmini-dump-phase test-gemmini-dump-phase.cpp)
    target_link_libraries(test-gemmini-dump-phase PRIVATE ggml-gemmini ggml-base ggml-npu-profiling)
    target_compile_features(test-gemmini-dump-phase PRIVATE cxx_std_17)
    target_compile_definitions(test-gemmini-dump-phase PRIVATE LOG_DUMP=1 LOG_CYCLE=${LOG_CYCLE} LOG_DUMP_SCALE=${LOG_DUMP_SCALE})
    target_include_directories(test-gemmini-dump-phase PRIVATE ../common ../ggml/src/ggml-gemmini)
    add_test(NAME test-gemmini-dump-phase COMMAND test-gemmini-dump-phase)
    if (LOG_DUMP_SCALE AND UNIX)
        add_test(NAME test-gemmini-dump-output COMMAND test-gemmini-dump-phase --outputs)
    endif()
endif()
add_executable(test-gemmini-host-timing test-gemmini-host-timing.cpp)
find_package(Threads REQUIRED)
target_link_libraries(test-gemmini-log-boundary PRIVATE Threads::Threads)
target_link_libraries(test-gemmini-host-timing PRIVATE ggml-npu-profiling Threads::Threads)
target_compile_features(test-gemmini-host-timing PRIVATE cxx_std_17)
target_compile_definitions(test-gemmini-host-timing PRIVATE
    EXPECT_LOG_CYCLE=$<IF:$<BOOL:${LOG_CYCLE}>,1,0>
    EXPECT_CYCLE_DETAIL=$<IF:$<BOOL:${CYCLE_DETAIL}>,1,0>)
add_test(NAME test-gemmini-host-timing COMMAND $<TARGET_FILE:test-gemmini-host-timing>)
add_executable(test-gemmini-trace-context test-gemmini-trace-context.cpp)
target_link_libraries(test-gemmini-trace-context PRIVATE ggml-npu-profiling Threads::Threads)
target_compile_features(test-gemmini-trace-context PRIVATE cxx_std_17)
target_compile_definitions(test-gemmini-trace-context PRIVATE
    EXPECT_LOG_CYCLE=$<IF:$<BOOL:${LOG_CYCLE}>,1,0>
    EXPECT_CYCLE_DETAIL=$<IF:$<BOOL:${CYCLE_DETAIL}>,1,0>)
add_test(NAME test-gemmini-trace-context COMMAND $<TARGET_FILE:test-gemmini-trace-context>
    ${CMAKE_CURRENT_BINARY_DIR}/trace-context)

find_package(Python3 3.9 COMPONENTS Interpreter QUIET)
if (Python3_Interpreter_FOUND)
    if(Python3_VERSION VERSION_GREATER_EQUAL "3.11")
        add_test(NAME test-evaluation-metric-framework
            COMMAND ${Python3_EXECUTABLE} -B -m evaluation.tests.test_framework)
        set_tests_properties(test-evaluation-metric-framework PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/..")
    endif()
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
        ../ggml/src/ggml-npu-common/profiling/src/cycle_reader_aarch64.cpp
        ../ggml/src/ggml-npu-common/profiling/src/cycle_reader-capi.cpp)
    target_include_directories(test-gemmini-cycle-reader-aarch64 PRIVATE
        ../ggml/src/ggml-npu-common/profiling/include)
    target_compile_definitions(test-gemmini-cycle-reader-aarch64 PRIVATE
        GGML_GEMMINI_TESTING=1 LOG_CYCLE=1 )
    target_compile_features(test-gemmini-cycle-reader-aarch64 PRIVATE cxx_std_17)
    target_link_libraries(test-gemmini-cycle-reader-aarch64 PRIVATE ggml-npu-profiling Threads::Threads)
    add_test(NAME test-gemmini-cycle-reader-aarch64
        COMMAND $<TARGET_FILE:test-gemmini-cycle-reader-aarch64>)
endif()
if (LOG_CYCLE AND GGML_CPU_CYCLE_LOG AND TARGET ggml-cpu AND NOT GGML_BACKEND_DL)
    add_executable(test-ggml-cpu-cycle-sink test-ggml-cpu-cycle-sink.cpp)
    target_link_libraries(test-ggml-cpu-cycle-sink PRIVATE ggml-base ggml-cpu ggml-npu-profiling)
    target_compile_features(test-ggml-cpu-cycle-sink PRIVATE cxx_std_17)
    target_compile_definitions(test-ggml-cpu-cycle-sink PRIVATE
        LOG_CYCLE=${LOG_CYCLE} )
    target_include_directories(test-ggml-cpu-cycle-sink PRIVATE
        ../ggml/include ../ggml/src/ggml-cpu ../ggml/src/ggml-npu-common/profiling/include)
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
        ggml-npu-profiling ${CMAKE_DL_LIBS})
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
set(_GEMMINI_TEST_FULL_SOURCES
    ../ggml/src/ggml-gemmini/matmul/dense.cpp
    ../ggml/src/ggml-gemmini/matmul/execution.cpp
    ../ggml/src/ggml-npu-common/matmul/options.cpp
    ../ggml/src/ggml-npu-common/profiling/telemetry.cpp
    ../ggml/src/ggml-npu-common/residual/direct/direct-executor.cpp
    ../ggml/src/ggml-npu-common/quantization/common/dequant.cpp
    ../ggml/src/ggml-npu-common/quantization/common/weight_reader.cpp
    ../ggml/src/ggml-npu-common/residual/rmd/rmd-builder.cpp
    ../ggml/src/ggml-npu-common/residual/rmd/rmd-bitmap-builder.cpp
    ../ggml/src/ggml-npu-common/residual/rmd/rmd-compose.cpp
    ../ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp
    ../ggml/src/ggml-npu-common/quantization/activation/dispatch.cpp
    ../ggml/src/ggml-npu-common/quantization/activation/quantize.cpp
    ../ggml/src/ggml-npu-common/quantization/activation/block/block.cpp
    ../ggml/src/ggml-npu-common/quantization/activation/exsia/exsia.cpp
    ../ggml/src/ggml-npu-common/quantization/activation/exsia/exsia-profile.cpp
    ../ggml/src/ggml-npu-common/quantization/activation/exsia/local.cpp
    ../ggml/src/ggml-npu-common/quantization/activation/exsia/folding.cpp
    ../ggml/src/ggml-npu-common/quantization/activation/stripe/stripe.cpp
    ../ggml/src/ggml-npu-common/quantization/activation/tensor/tensor.cpp
    ../ggml/src/ggml-npu-common/quantization/activation/token/token.cpp
    ../ggml/src/ggml-gemmini/matmul/baseline.cpp
    ../ggml/src/ggml-gemmini/matmul/block-quantized.cpp
    ../ggml/src/ggml-gemmini/reference/cpu.cpp
)
# Tests that compile backend sources directly also need the internal API.
link_libraries(gemmini-tiled-api)
set(_GEMMINI_TEST_WRITER_SOURCES ${_GEMMINI_TEST_FULL_SOURCES})
list(REMOVE_ITEM _GEMMINI_TEST_WRITER_SOURCES
    ../ggml/src/ggml-npu-common/quantization/activation/stripe/stripe.cpp)
set(_GEMMINI_TEST_BASE_SOURCES ${_GEMMINI_TEST_WRITER_SOURCES})
set(_GEMMINI_TEST_EXSIA_SOURCES ${_GEMMINI_TEST_FULL_SOURCES})
list(INSERT _GEMMINI_TEST_EXSIA_SOURCES 7
    ../ggml/src/ggml-npu-common/residual/rmd/rmd-reference.cpp)
set(_GEMMINI_TEST_WORKER_SOURCES ${_GEMMINI_TEST_FULL_SOURCES})
list(INSERT _GEMMINI_TEST_WORKER_SOURCES 6
    ../ggml/src/ggml-npu-common/residual/rmd/rmd-reference.cpp)

# Direct backend symbols require a linkable backend, not a dlopen MODULE.
if (NOT GGML_BACKEND_DL)
    llama_build_and_test(test-gemmini-weight-formats.cpp NAME gemmini_weight_formats)
    add_test(NAME gemmini_mxint_weight_routes_activation
        COMMAND $<TARGET_FILE:gemmini_weight_formats> --case=mxint-graph)
    target_link_libraries(gemmini_weight_formats PRIVATE ggml-base ggml-gemmini)
    target_compile_definitions(gemmini_weight_formats PRIVATE
        GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
        GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE})
endif()

add_test(NAME test-gemmini-weight-reader-callers
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/test-gemmini-weight-reader-callers.cmake)

set(_GGML_GEMMINI_BUILD_DEQUANT_FP_TARGET TRUE)
if (_GGML_GEMMINI_BUILD_DEQUANT_FP_TARGET)
set(_GGML_GEMMINI_DEQUANT_FP_TARGET gemmini_native_dequant_fp)
llama_build_and_test(
    test-gemmini-native-dequant-fp.cpp
    NAME ${_GGML_GEMMINI_DEQUANT_FP_TARGET})
target_sources(${_GGML_GEMMINI_DEQUANT_FP_TARGET} PRIVATE
    ${_GEMMINI_TEST_FULL_SOURCES})
target_link_libraries(
    ${_GGML_GEMMINI_DEQUANT_FP_TARGET}
    PRIVATE ggml-base ggml-npu-profiling)
target_compile_definitions(${_GGML_GEMMINI_DEQUANT_FP_TARGET} PRIVATE
    ${GGML_GEMMINI_COMPILE_DEFS}
    GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE=${GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE})
target_include_directories(${_GGML_GEMMINI_DEQUANT_FP_TARGET} PRIVATE
    ../ggml/src
    ../ggml/src/ggml-gemmini
    ../ggml/src/ggml-npu-common/profiling/include)
if (GGML_GEMMINI_DEQUANT_FP_TEST)
    if (_GGML_GEMMINI_ACTIVATION_QUANT_UPPER STREQUAL "EXSIA")
        add_test(NAME gemmini_deferred_dequant_fp
            COMMAND ${CMAKE_COMMAND} -E env
                GEMMINI_MATMUL_MODE=STRIPE_PIPELINE
                $<TARGET_FILE:${_GGML_GEMMINI_DEQUANT_FP_TARGET}>
                --native-pipeline)
    else()
        add_test(NAME gemmini_dequant_fp
            COMMAND ${CMAKE_COMMAND} -E env
                GEMMINI_MATMUL_MODE=FULL
                $<TARGET_FILE:${_GGML_GEMMINI_DEQUANT_FP_TARGET}>
                --native-full)
    endif()
endif()
unset(_GGML_GEMMINI_DEQUANT_FP_TARGET)
endif()
unset(_GGML_GEMMINI_BUILD_DEQUANT_FP_TARGET)

add_executable(test-gemmini-act-quant-i32-saturation
    test-gemmini-act-quant-i32-saturation.cpp
    ${_GEMMINI_TEST_WRITER_SOURCES}
    ../ggml/src/ggml-npu-common/quantization/activation/stripe/stripe.cpp)
target_link_libraries(test-gemmini-act-quant-i32-saturation PRIVATE ggml-base ggml-npu-profiling)
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
    ../ggml/src/ggml-npu-common/profiling/include)
add_test(NAME test-gemmini-act-quant-i32-saturation
    COMMAND $<TARGET_FILE:test-gemmini-act-quant-i32-saturation>)

if (_GGML_GEMMINI_ACTIVATION_QUANT_UPPER STREQUAL "BLOCK")
add_executable(test-gemmini-block-quant
    test-gemmini-block-quant.cpp
    ${_GEMMINI_TEST_WRITER_SOURCES}
    ../ggml/src/ggml-npu-common/quantization/activation/stripe/stripe.cpp)
target_link_libraries(test-gemmini-block-quant PRIVATE ggml-base ggml-npu-profiling)
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
    ../ggml/src/ggml-npu-common/profiling/include)
add_test(NAME gemmini_block_metadata
    COMMAND $<TARGET_FILE:test-gemmini-block-quant> --metadata)
add_test(NAME gemmini_block_quant
    COMMAND $<TARGET_FILE:test-gemmini-block-quant> --quant)

add_executable(test-gemmini-block-dense test-gemmini-block-dense.cpp)
target_link_libraries(test-gemmini-block-dense PRIVATE
    ggml-gemmini ggml-base ggml-npu-profiling)
target_compile_features(test-gemmini-block-dense PRIVATE cxx_std_17)
target_compile_definitions(test-gemmini-block-dense PRIVATE
    ${GGML_GEMMINI_COMPILE_DEFS})
add_test(NAME gemmini_block_dense COMMAND $<TARGET_FILE:test-gemmini-block-dense>)

if (NOT GGML_BACKEND_DL)
    add_executable(test-gemmini-block-end-to-end test-gemmini-block-end-to-end.cpp)
    target_link_libraries(test-gemmini-block-end-to-end PRIVATE
        common ggml-gemmini ggml-base ggml-npu-profiling)
    target_compile_features(test-gemmini-block-end-to-end PRIVATE cxx_std_17)
    target_compile_definitions(test-gemmini-block-end-to-end PRIVATE ${GGML_GEMMINI_COMPILE_DEFS})
    target_include_directories(test-gemmini-block-end-to-end PRIVATE
        ../ggml/include
        ../ggml/src
        ../ggml/src/ggml-gemmini
        ../ggml/src/ggml-npu-common/profiling/include)
    add_test(NAME gemmini_block_end_to_end
        COMMAND $<TARGET_FILE:test-gemmini-block-end-to-end>)
endif()

endif()


add_executable(test-gemmini-exsia-sigma
    test-gemmini-exsia-sigma.cpp
    ../ggml/src/ggml-npu-common/quantization/activation/exsia/local.cpp)
target_link_libraries(test-gemmini-exsia-sigma PRIVATE ggml-base ggml-npu-profiling)
target_compile_features(test-gemmini-exsia-sigma PRIVATE cxx_std_17)
target_compile_definitions(test-gemmini-exsia-sigma PRIVATE
    GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
    GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE}
    GGML_GEMMINI_EXSIA_SIGMA=${GGML_GEMMINI_EXSIA_SIGMA}
    GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE=0
    LOG_CYCLE=0 CYCLE_DETAIL=0 EXSIA_VALIDATION=1)
target_include_directories(test-gemmini-exsia-sigma PRIVATE
    ../ggml/include ../ggml/src ../ggml/src/ggml-npu-common/profiling/include)
add_test(NAME test-gemmini-exsia-sigma COMMAND test-gemmini-exsia-sigma)

add_executable(test-gemmini-exsia
    test-gemmini-exsia.cpp
    ${_GEMMINI_TEST_EXSIA_SOURCES})
target_link_libraries(test-gemmini-exsia PRIVATE ggml-base ggml-npu-profiling)
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
    ../ggml/src/ggml-npu-common/profiling/include
    ../ggml/src/ggml-npu-common/quantization
    ../ggml/src/ggml-npu-common/quantization/activation
    ../ggml/src/ggml-npu-common/quantization/activation/exsia
    ../ggml/src/ggml-npu-common/quantization/common)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
    CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    add_library(test-gemmini-direct-linux-aarch64-strict OBJECT
        ../ggml/src/ggml-npu-common/residual/direct/direct-executor.cpp)
    set_target_properties(test-gemmini-direct-linux-aarch64-strict PROPERTIES COMPILE_OPTIONS "")
    target_compile_features(test-gemmini-direct-linux-aarch64-strict PRIVATE cxx_std_17)
    target_compile_definitions(test-gemmini-direct-linux-aarch64-strict PRIVATE

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
        ../ggml/src/ggml-npu-common/profiling/include)
    add_dependencies(test-gemmini-exsia test-gemmini-direct-linux-aarch64-strict)
endif()
if (LOG_CYCLE AND CYCLE_DETAIL AND Python3_Interpreter_FOUND)
    add_test(NAME test-residual-profile-runtime
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-residual-profile.py
            --runtime-fixture $<TARGET_FILE:test-gemmini-exsia>)
endif()
add_test(NAME gemmini_direct_metrics_source_contract
    COMMAND ${CMAKE_COMMAND}
        -DDENSE_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-gemmini/matmul/dense.cpp
        -DEXECUTION_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-gemmini/matmul/execution.cpp
        -DDIRECT_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-npu-common/residual/direct/direct-executor.cpp
        -DDIRECT_HEADER=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-npu-common/residual/direct/direct-executor.hpp
        -DTEST_CMAKE=${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
        -DTEST_TARGET_CONTRACT=${CMAKE_CURRENT_BINARY_DIR}/gemmini-test-targets.cmake
        -DPROJECT_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/..
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test-gemmini-direct-metrics-source-contract.cmake)
# These fixtures deliberately compile the ExSIA profiling implementation with
# LOG_CYCLE=1/CYCLE_DETAIL=1 and assert both main and detail sink records. They
# must not link against an OFF logger; OFF behavior is tested by the log suite.
if (LOG_CYCLE AND CYCLE_DETAIL)
foreach(_worker_count IN ITEMS 3 4)
    add_executable(test-gemmini-exsia-workers${_worker_count}
        test-gemmini-exsia-profile.cpp
        ${_GEMMINI_TEST_WORKER_SOURCES})
    target_link_libraries(test-gemmini-exsia-workers${_worker_count} PRIVATE
        ggml-base ggml-npu-profiling)
    target_compile_features(test-gemmini-exsia-workers${_worker_count} PRIVATE cxx_std_17)
    target_compile_definitions(test-gemmini-exsia-workers${_worker_count} PRIVATE

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
        ../ggml/src/ggml-npu-common/profiling/include ../ggml/src/ggml-npu-common/quantization
        ../ggml/src/ggml-npu-common/quantization/activation ../ggml/src/ggml-npu-common/quantization/activation/exsia
        ../ggml/src/ggml-npu-common/quantization/common)
    if (GGML_GEMMINI_ENABLE_OPENMP)
        target_compile_definitions(test-gemmini-exsia-workers${_worker_count} PRIVATE
            GGML_GEMMINI_HAS_OPENMP)
        target_link_libraries(test-gemmini-exsia-workers${_worker_count} PRIVATE
            OpenMP::OpenMP_CXX)
    endif()
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
        CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        add_library(test-gemmini-exsia-workers${_worker_count}-strict OBJECT
            ../ggml/src/ggml-npu-common/quantization/activation/exsia/exsia.cpp
            ../ggml/src/ggml-npu-common/quantization/activation/exsia/exsia-profile.cpp
            ../ggml/src/ggml-npu-common/quantization/activation/exsia/local.cpp
            ../ggml/src/ggml-npu-common/quantization/activation/exsia/folding.cpp)
        set_target_properties(test-gemmini-exsia-workers${_worker_count}-strict
            PROPERTIES COMPILE_OPTIONS "")
        target_compile_features(test-gemmini-exsia-workers${_worker_count}-strict PRIVATE cxx_std_17)
        target_compile_definitions(test-gemmini-exsia-workers${_worker_count}-strict PRIVATE

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
            ../ggml/src/ggml-npu-common/profiling/include ../ggml/src/ggml-npu-common/quantization
            ../ggml/src/ggml-npu-common/quantization/activation ../ggml/src/ggml-npu-common/quantization/activation/exsia
            ../ggml/src/ggml-npu-common/quantization/common)
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
    add_test(NAME test-gemmini-exsia-workers${_worker_count}-origin-order
        COMMAND $<TARGET_FILE:test-gemmini-exsia-workers${_worker_count}>
            --origin-order ${CMAKE_CURRENT_BINARY_DIR}/profile-origin-workers${_worker_count})
endforeach()
endif()
add_test(NAME test-gemmini-exsia-profile-contract
    COMMAND ${CMAKE_COMMAND}
        -DEXSIA_RUNTIME_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-npu-common/quantization/activation/exsia/exsia.cpp
        -DEXSIA_LOCAL_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-npu-common/quantization/activation/exsia/local.cpp
        -DEXSIA_FOLDING_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-npu-common/quantization/activation/exsia/folding.cpp
        -DEXSIA_PROFILE_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-npu-common/quantization/activation/exsia/exsia-profile.cpp
        -DEXSIA_PROFILE_HEADER=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-npu-common/quantization/activation/exsia/exsia-profile.hpp
        -DEXSIA_STATE_HEADER=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-npu-common/quantization/activation/exsia/exsia-state.hpp
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
    ${_GEMMINI_TEST_WRITER_SOURCES})
target_link_libraries(test-gemmini-exsia-writer PRIVATE ggml-base ggml-npu-profiling)
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
    ../ggml/src/ggml-npu-common/profiling/include ../ggml/src/ggml-npu-common/quantization
    ../ggml/src/ggml-npu-common/quantization/activation ../ggml/src/ggml-npu-common/quantization/activation/exsia
    ../ggml/src/ggml-npu-common/quantization/common)

set(GEMMINI_WRITER_TEST_ROOT "${CMAKE_CURRENT_BINARY_DIR}/gemmini-writer-output")
add_test(NAME test-gemmini-exsia-writer-profile-context
    COMMAND $<TARGET_FILE:test-gemmini-exsia-writer>
        --profile-context ${CMAKE_CURRENT_BINARY_DIR}/profile-context/detail.jsonl)
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
        ../ggml/src/ggml-npu-common/residual/direct/direct-executor.cpp
        ../ggml/src/ggml-npu-common/residual/rmd/rmd-builder.cpp)
    target_link_libraries(test-gemmini-rmd PRIVATE ggml-base ggml-gemmini ggml-npu-profiling)
    target_compile_features(test-gemmini-rmd PRIVATE cxx_std_17)
    target_compile_definitions(test-gemmini-rmd PRIVATE
        LOG_CYCLE=${LOG_CYCLE}
        GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
        GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE})
    target_include_directories(test-gemmini-rmd PRIVATE
        ../ggml/src
        ../ggml/src/ggml-gemmini
        ../ggml/src/ggml-gemmini/residual/rmd)
    add_test(NAME test-gemmini-rmd COMMAND $<TARGET_FILE:test-gemmini-rmd>)
endif()

add_executable(test-gemmini-rmd-packet
    test-gemmini-rmd-packet.cpp
    ../ggml/src/ggml-npu-common/quantization/common/weight_reader.cpp
    ../ggml/src/ggml-npu-common/residual/rmd/rmd-builder.cpp)
target_link_libraries(test-gemmini-rmd-packet PRIVATE
    ggml-base ggml-npu-profiling)
target_compile_features(test-gemmini-rmd-packet PRIVATE cxx_std_17)
target_compile_definitions(test-gemmini-rmd-packet PRIVATE
    GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
    GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE})
target_include_directories(test-gemmini-rmd-packet PRIVATE
    ../ggml/src
    ../ggml/src/ggml-gemmini
    ../ggml/src/ggml-gemmini/residual/rmd)
add_test(NAME test-gemmini-rmd-packet
    COMMAND $<TARGET_FILE:test-gemmini-rmd-packet>)

foreach(_bitmap_log IN ITEMS 0 1)
    add_executable(test-gemmini-rmd-bitmap-log${_bitmap_log}
        test-gemmini-rmd-bitmap.cpp
        ../ggml/src/ggml-npu-common/residual/rmd/rmd-builder.cpp
        ../ggml/src/ggml-npu-common/residual/rmd/rmd-bitmap-builder.cpp)
    target_compile_features(test-gemmini-rmd-bitmap-log${_bitmap_log} PRIVATE cxx_std_17)
    target_compile_definitions(test-gemmini-rmd-bitmap-log${_bitmap_log} PRIVATE LOG_CYCLE=${_bitmap_log})
    target_include_directories(test-gemmini-rmd-bitmap-log${_bitmap_log} PRIVATE
        ../ggml/src)
    add_test(NAME test-gemmini-rmd-bitmap-log${_bitmap_log}
        COMMAND $<TARGET_FILE:test-gemmini-rmd-bitmap-log${_bitmap_log}>)
endforeach()

add_executable(test-gemmini-rmd-run-aware
    test-gemmini-rmd-run-aware.cpp)
if(GGML_BACKEND_DL)
    set(_GEMMINI_RUN_AWARE_SOURCES ${_GEMMINI_TEST_FULL_SOURCES})
    target_sources(test-gemmini-rmd-run-aware PRIVATE
        ${_GEMMINI_RUN_AWARE_SOURCES}
        ../ggml/src/ggml-gemmini/residual/rmd/rmd-run-aware.cpp)
    target_link_libraries(test-gemmini-rmd-run-aware PRIVATE ggml-base ggml-npu-profiling)
    set(_GEMMINI_RUN_AWARE_DEFS ${GGML_GEMMINI_COMPILE_DEFS})
    list(FILTER _GEMMINI_RUN_AWARE_DEFS EXCLUDE REGEX "^GGML_GEMMINI_TEST_OBSERVER(=|$)")
    target_compile_definitions(test-gemmini-rmd-run-aware PRIVATE ${_GEMMINI_RUN_AWARE_DEFS})
    ggml_gemmini_apply_profile_compile_definitions(test-gemmini-rmd-run-aware)
    ggml_gemmini_assert_profile_compile_definitions(test-gemmini-rmd-run-aware)
    target_include_directories(test-gemmini-rmd-run-aware BEFORE PRIVATE
        "${GGML_GEMMINI_GENERATED_CONFIG_DIR}")
else()
    target_link_libraries(test-gemmini-rmd-run-aware PRIVATE
        ggml-base ggml-gemmini ggml-npu-profiling)
endif()
target_compile_features(test-gemmini-rmd-run-aware PRIVATE cxx_std_17)
target_compile_definitions(test-gemmini-rmd-run-aware PRIVATE
    GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
    GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE})
target_include_directories(test-gemmini-rmd-run-aware PRIVATE
    ../ggml/src
    ../ggml/src/ggml-gemmini
    ../ggml/src/ggml-gemmini/residual/rmd)
add_test(NAME test-gemmini-rmd-run-aware
    COMMAND $<TARGET_FILE:test-gemmini-rmd-run-aware>)

if((GGML_GEMMINI_ACT_QUANT_METRICS OR GGML_GEMMINI_RESIDUAL_METRICS OR GGML_GEMMINI_SCALE_METRICS)
   AND GGML_GEMMINI_ENABLE_RMD AND GGML_GEMMINI_ACTIVATION_BITS MATCHES "^(4|8)$")
    add_executable(test-gemmini-evaluation-observer test-gemmini-evaluation-observer.cpp)
    target_link_libraries(test-gemmini-evaluation-observer PRIVATE ggml-base ggml-gemmini ggml-npu-profiling)
    target_compile_features(test-gemmini-evaluation-observer PRIVATE cxx_std_17)
    target_compile_definitions(test-gemmini-evaluation-observer PRIVATE ${GGML_GEMMINI_RUNTIME_COMPILE_DEFS})
    target_compile_options(test-gemmini-evaluation-observer PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>)
    target_include_directories(test-gemmini-evaluation-observer PRIVATE ../ggml/src ../ggml/src/ggml-gemmini)
    add_test(NAME test-gemmini-evaluation-observer
        COMMAND ${CMAKE_COMMAND} -DTEST_BINARY=$<TARGET_FILE:test-gemmini-evaluation-observer>
            -DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/test-evaluation-metric-sink.cmake)
endif()

add_executable(test-gemmini-rmd-capture
    test-gemmini-rmd-capture.cpp
    ../ggml/src/ggml-npu-common/quantization/common/weight_reader.cpp
    ../ggml/src/ggml-npu-common/residual/rmd/rmd-builder.cpp
    ../ggml/src/ggml-npu-common/residual/rmd/rmd-bitmap-builder.cpp)
target_link_libraries(test-gemmini-rmd-capture PRIVATE
    ggml-base ggml-npu-profiling)
target_compile_features(test-gemmini-rmd-capture PRIVATE cxx_std_17)
target_compile_definitions(test-gemmini-rmd-capture PRIVATE

    GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
    GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE}
    LOG_CYCLE=${LOG_CYCLE}
    CYCLE_DETAIL=${CYCLE_DETAIL})
target_include_directories(test-gemmini-rmd-capture PRIVATE
    ../ggml/src
    ../ggml/src/ggml-gemmini
    ../ggml/src/ggml-gemmini/residual/rmd)
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
    ../ggml/src/ggml-npu-common/profiling/include
    ${GGML_GEMMINI_GENERATED_CONFIG_DIR})
add_test(NAME test-gemmini-production-host
    COMMAND $<TARGET_FILE:test-gemmini-production-host-backend>)

# Direct backend symbols require a linkable backend, not a dlopen MODULE.
if (NOT GGML_BACKEND_DL)
    add_executable(test-gemmini-rmd-telemetry
        test-gemmini-rmd-telemetry.cpp)
    target_link_libraries(test-gemmini-rmd-telemetry PRIVATE
        ggml-gemmini ggml-base ggml-npu-profiling)
    target_compile_features(test-gemmini-rmd-telemetry PRIVATE cxx_std_17)
    target_compile_definitions(test-gemmini-rmd-telemetry PRIVATE
        GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
        GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE}
        LOG_DEBUG=${LOG_DEBUG}
        LOG_CYCLE=${LOG_CYCLE}
        CYCLE_DETAIL=${CYCLE_DETAIL})
    target_include_directories(test-gemmini-rmd-telemetry PRIVATE
        ../ggml/include ../ggml/src ../ggml/src/ggml-gemmini
        ../ggml/src/ggml-npu-common/profiling/include ${GGML_GEMMINI_GENERATED_CONFIG_DIR})
    add_test(NAME test-gemmini-rmd-telemetry
        COMMAND $<TARGET_FILE:test-gemmini-rmd-telemetry>)
    if (LOG_CYCLE)
        add_test(NAME test-gemmini-rmd-telemetry-typed-metadata
            COMMAND $<TARGET_FILE:test-gemmini-rmd-telemetry>
                --typed-metadata ${CMAKE_CURRENT_BINARY_DIR}/typed-telemetry/rows.jsonl)
    endif()
    add_test(NAME test-gemmini-aggregate-cycle-driver
        COMMAND $<TARGET_FILE:test-gemmini-rmd-telemetry> --aggregate-driver
                ${CMAKE_CURRENT_BINARY_DIR}/gemmini-aggregate-driver/cycle-log.jsonl)
endif()

add_executable(test-gemmini-rmd-reducer
    test-gemmini-rmd-telemetry.cpp
    ../ggml/src/ggml-npu-common/profiling/telemetry.cpp
    ../ggml/src/ggml-npu-common/residual/rmd/rmd-builder.cpp)
target_link_libraries(test-gemmini-rmd-reducer PRIVATE ggml-npu-profiling)
target_compile_features(test-gemmini-rmd-reducer PRIVATE cxx_std_17)
target_compile_definitions(test-gemmini-rmd-reducer PRIVATE
    GGML_GEMMINI_REDUCER_TEST_ONLY=1
    GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
    GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE}
    LOG_DEBUG=${LOG_DEBUG} LOG_CYCLE=${LOG_CYCLE} CYCLE_DETAIL=${CYCLE_DETAIL})
target_include_directories(test-gemmini-rmd-reducer PRIVATE
    ../ggml/include ../ggml/src ../ggml/src/ggml-gemmini
    ../ggml/src/ggml-npu-common/profiling/include ${GGML_GEMMINI_GENERATED_CONFIG_DIR})
add_test(NAME test-gemmini-rmd-reducer
    COMMAND $<TARGET_FILE:test-gemmini-rmd-reducer>)
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
        ggml-base ggml-gemmini ggml-npu-profiling)
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
        ../ggml/src/ggml-npu-common/profiling/include
        ../ggml/src/ggml-npu-common/quantization
        ${GGML_GEMMINI_GENERATED_CONFIG_DIR})
    if (NOT GGML_BACKEND_DL)
        target_link_libraries(test-gemmini-matmul-options PRIVATE ggml-gemmini ggml-base ggml-npu-profiling)
        target_compile_definitions(test-gemmini-matmul-options PRIVATE GGML_GEMMINI_OPTIONS_TEST_BACKEND=1)
    else()
        target_sources(test-gemmini-matmul-options PRIVATE
            ../ggml/src/ggml-npu-common/matmul/options.cpp)
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
            -DTEST_SOURCE_DIR=${LLAMA_TEST_SOURCE_DIR}
            -DTEST_GEMMINI_SW_PATH=${GEMMINI_SW_PATH}
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
    target_link_libraries(test-gemmini-matmul PRIVATE ggml-gemmini ggml-base ggml-npu-profiling)
    if(TARGET ggml-cpu)
        target_link_libraries(test-gemmini-matmul PRIVATE ggml-cpu)
        target_compile_definitions(test-gemmini-matmul PRIVATE GGML_GEMMINI_TEST_SOFTMAX=1)
    endif()
    target_compile_features(test-gemmini-matmul PRIVATE cxx_std_17)
    target_compile_definitions(test-gemmini-matmul PRIVATE
        GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
        GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE}
        GGML_GEMMINI_ENABLE_RMD=${_GGML_GEMMINI_ENABLE_RMD_VALUE}
        LOG_CYCLE=${LOG_CYCLE} CYCLE_DETAIL=${CYCLE_DETAIL})
    add_test(NAME test-gemmini-matmul COMMAND $<TARGET_FILE:test-gemmini-matmul>)
    if(GGML_GEMMINI_ENABLE_A4NKS_ATTENTION)
        add_test(NAME test-gemmini-attention COMMAND $<TARGET_FILE:test-gemmini-matmul> --case=attention)
        add_test(NAME test-gemmini-attention-mask COMMAND $<TARGET_FILE:test-gemmini-matmul> --case=attention-mask)
    endif()
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
    ${_GEMMINI_TEST_BASE_SOURCES})
target_link_libraries(gemmini-quant-dispatch-smoke PRIVATE ggml-base ggml-npu-profiling)
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
    ../ggml/src/ggml-npu-common/quantization
    ../ggml/src/ggml-npu-common/quantization/activation
    ../ggml/src/ggml-npu-common/quantization/activation/exsia
    ../ggml/src/ggml-npu-common/quantization/activation/token
    ../ggml/src/ggml-npu-common/quantization/common)
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
        -DDENSE_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-gemmini/matmul/dense.cpp
        -DEXECUTION_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-gemmini/matmul/execution.cpp
        -DTYPES_HEADER=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-gemmini/matmul/types.hpp
        -DDENSE_HEADER=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-gemmini/matmul/dense.hpp
        -DEXECUTION_HEADER=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-gemmini/matmul/execution.hpp
        -DMATMUL_TELEMETRY=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-npu-common/profiling/telemetry.cpp
        -DGEMMINI_SOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-gemmini
        -DGEMMINI_UTILS_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-npu-common/profiling
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test-gemmini-matmul-cpu-boundaries.cmake)

add_test(NAME test-gemmini-profiled-domain-contract
    COMMAND ${CMAKE_COMMAND}
        -DDENSE_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-gemmini/matmul/dense.cpp
        -DEXECUTION_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-gemmini/matmul/execution.cpp
        -DTYPES_HEADER=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-gemmini/matmul/types.hpp
        -DDENSE_HEADER=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-gemmini/matmul/dense.hpp
        -DEXECUTION_HEADER=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-gemmini/matmul/execution.hpp
        -DMATMUL_HEADER=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-gemmini/ggml-gemmini-matmul.hpp
        -DMATMUL_TELEMETRY=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-npu-common/profiling/telemetry.cpp
        -DGEMMINI_ENTRY_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-gemmini/ggml-gemmini.cpp
        -DOPS_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-gemmini/ops.cpp
        -DGEMMINI_SOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-gemmini
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test-gemmini-profiled-domain-contract.cmake)

endif() # TARGET ggml-gemmini

# This test deliberately links no simulator archive: injected compact execution
# must work with a null simulator and reject incomplete or duplicate outputs.

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
    test-gemmini-rmd-bitmap-log0
    test-gemmini-rmd-bitmap-log1
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
        gemmini_native_dequant_fp
        test-gemmini-srmd-raw-ws
        test-gemmini-matmul-options
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
add_custom_target(gemmini-test-executables ALL)
add_dependencies(gemmini-test-executables ${_GEMMINI_REGISTERED_TEST_EXECUTABLES})
get_target_property(_gemmini_test_dependencies gemmini-test-executables MANUALLY_ADDED_DEPENDENCIES)
set(_gemmini_test_contract "${CMAKE_CURRENT_BINARY_DIR}/gemmini-test-targets.cmake")
file(WRITE "${_gemmini_test_contract}"
    "set(GEMMINI_TEST_EXECUTABLES [==[${_GEMMINI_REGISTERED_TEST_EXECUTABLES}]==])\n"
    "set(GEMMINI_TEST_DEPENDENCIES [==[${_gemmini_test_dependencies}]==])\n")
set(_gemmini_cpu_sink_required OFF)
if(LOG_CYCLE AND GGML_CPU_CYCLE_LOG AND TARGET ggml-cpu AND NOT GGML_BACKEND_DL)
    set(_gemmini_cpu_sink_required ON)
endif()
file(APPEND "${_gemmini_test_contract}"
    "set(GEMMINI_CPU_SINK_REQUIRED ${_gemmini_cpu_sink_required})\n")
foreach(_target IN ITEMS test-ggml-cpu-cycle-sink test-gemmini-exsia)
    if(TARGET ${_target})
        get_target_property(_definitions ${_target} COMPILE_DEFINITIONS)
        get_target_property(_interface ${_target} INTERFACE_COMPILE_DEFINITIONS)
        file(APPEND "${_gemmini_test_contract}"
            "set(${_target}_EXISTS ON)\n"
            "set(${_target}_DEFINITIONS [==[${_definitions}]==])\n"
            "set(${_target}_INTERFACE [==[${_interface}]==])\n")
    else()
        file(APPEND "${_gemmini_test_contract}" "set(${_target}_EXISTS OFF)\n")
    endif()
endforeach()

# libmtmd
set(LLAMA_TEST_NAME test-mtmd-c-api)
llama_build_and_test(test-mtmd-c-api.c)
target_link_libraries(${LLAMA_TEST_NAME} PRIVATE mtmd)

# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)

add_compile_definitions(
    GGML_GEMMINI_SCALE_METRICS=${GGML_GEMMINI_SCALE_METRICS}
    GGML_GEMMINI_ACT_QUANT_METRICS=${GGML_GEMMINI_ACT_QUANT_METRICS}
    GGML_GEMMINI_RESIDUAL_METRICS=${GGML_GEMMINI_RESIDUAL_METRICS})

# POTAL: the real GPT-2 Q8_HP1 GGUF through the production model loader (buffer
# placement only; no graph is computed). Skipped (77) when the model is absent.
if (GGML_POTAL)
    add_executable(potal-gguf-load-test test-potal-gguf-load.cpp)
    target_link_libraries(potal-gguf-load-test PRIVATE llama ggml)
    if (NOT CMAKE_CROSSCOMPILING)
        add_test(NAME potal-gguf-load-test
                 COMMAND potal-gguf-load-test ${CMAKE_CURRENT_SOURCE_DIR}/../models/gpt2/gpt2.Q8_HP1.gguf)
        set_property(TEST potal-gguf-load-test PROPERTY SKIP_RETURN_CODE 77)
    endif()
endif()

# CUDA kernels used by the opt-in Gemmini host-graph path.
if (GGML_GEMMINI_CUDA_CPU_EXACT)
    foreach(_target test-cuda-cpu-exact-graph test-cuda-hp1-scu test-cuda-potal-attention test-potal-sensitivity)
        add_executable(${_target} ${_target}.cpp)
        target_compile_features(${_target} PRIVATE cxx_std_17)
        target_compile_definitions(${_target} PRIVATE ${GGML_GEMMINI_COMPILE_DEFS}
            TEST_CPU_EXACT_FLOAT=0
            TEST_CPU_EXACT_HP1=$<STREQUAL:${GGML_GEMMINI_ACTIVATION_QUANT},EXSIA>
            EVALUATION_EXACT_COMPUTE="${GGML_GEMMINI_COMPUTE_TYPE}"
            EVALUATION_EXACT_ACTIVATION="${GGML_GEMMINI_ACTIVATION_QUANT}"
            EVALUATION_EXACT_BITS=${GGML_GEMMINI_ACTIVATION_BITS}
            EVALUATION_EXACT_DIM=${GGML_GEMMINI_DIM})
        target_include_directories(${_target} PRIVATE ../ggml/src/ggml-gemmini ../ggml/src)
        target_link_libraries(${_target} PRIVATE common ggml ggml-gemmini
            ggml-gemmini-cuda-cpu-exact ggml-cpu ggml-base ggml-npu-profiling)
        add_test(NAME ${_target} COMMAND ${_target})
    endforeach()
    add_test(NAME test-potal-sensitivity-invalid COMMAND test-potal-sensitivity 4)
    add_test(NAME test-cuda-hp1-scu-graph
        COMMAND ${CMAKE_COMMAND} -E env GGML_GEMMINI_HP1_SCU=1 $<TARGET_FILE:test-cuda-cpu-exact-graph>)
endif()
if (GGML_GEMMINI_CUDA_RTN OR GGML_GEMMINI_CUDA_RTN_ATTENTION)
    add_executable(test-cuda-rtn test-cuda-rtn.cpp)
    target_compile_features(test-cuda-rtn PRIVATE cxx_std_17)
    target_compile_options(test-cuda-rtn PRIVATE -fno-fast-math -ffp-contract=off)
    target_include_directories(test-cuda-rtn PRIVATE ../ggml/src)
    target_link_libraries(test-cuda-rtn PRIVATE ggml-cuda-rtn ggml-base)
    add_test(NAME test-cuda-rtn COMMAND test-cuda-rtn)
endif()
