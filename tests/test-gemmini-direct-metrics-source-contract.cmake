if (NOT DEFINED DIRECT_SOURCE OR NOT DEFINED DIRECT_HEADER OR NOT DEFINED TEST_CMAKE)
    message(FATAL_ERROR "DIRECT_SOURCE, DIRECT_HEADER, and TEST_CMAKE are required")
endif()

file(READ "${DIRECT_SOURCE}" direct_source)
file(READ "${DIRECT_HEADER}" direct_header)
file(READ "${TEST_CMAKE}" test_cmake)
set(structural_failures)

function(require_count text regex expected label)
    string(REGEX MATCHALL "${regex}" matches "${text}")
    list(LENGTH matches actual)
    if (NOT actual EQUAL expected)
        message(FATAL_ERROR "${label}: expected ${expected}, got ${actual}")
    endif()
endfunction()

if (NOT DIRECT_CONTRACT_LEGACY_ACCEPTANCE)
    set(forbidden_source_patterns
        "\\(void\\)[ \t]+[A-Za-z_]"
        "static_cast<void>"
        "\\[\\[maybe_unused\\]\\]"
        "#[ \t]*pragma[ \t]+GCC[ \t]+diagnostic"
        "#[ \t]*pragma[ \t]+clang[ \t]+diagnostic")
    foreach(pattern IN LISTS forbidden_source_patterns)
        if (direct_source MATCHES "${pattern}")
            message(FATAL_ERROR "suppression-free source contract rejected ${pattern}")
        endif()
    endforeach()
    if (direct_source MATCHES "__int128|__builtin_[A-Za-z0-9_]+")
        list(APPEND structural_failures
            "portable arithmetic contract rejects compiler integer extensions")
    endif()
    if (test_cmake MATCHES "-Wno-[A-Za-z0-9_=+-]+")
        list(APPEND structural_failures "strict CMake contract rejects arbitrary -Wno")
    endif()
    foreach(pattern IN ITEMS "-Wno-error" "__linux__=1"
                             "add_compile_definitions(GGML_GEMMINI_DIRECT_METRICS_TESTING"
                             "target_compile_definitions(ggml-gemmini PRIVATE GGML_GEMMINI_DIRECT_METRICS_TESTING")
        string(FIND "${test_cmake}" "${pattern}" found)
        if (NOT found EQUAL -1)
            message(FATAL_ERROR "CMake contract rejected ${pattern}")
        endif()
    endforeach()
endif()

require_count("${direct_source}" "CpuSample read_cpu_sample\\(\\)" 1
              "zero-argument production reader definition")
require_count("${direct_source}" "read_cpu_sample\\(\\)" 7
              "production reader definition plus six boundary calls")
require_count("${direct_source}" "enum class CpuSamplePoint" 0
              "duplicate internal sample-point enum")

set(test_points
    serial_pre_start serial_pre_end tile_start tile_end serial_post_start serial_post_end)
foreach(point IN LISTS test_points)
    require_count("${direct_source}"
        "read_cpu_sample\\(\\*hooks,[ \t\r\n]+testing::DirectCpuSamplePoint::${point},"
        1 "test dispatch for ${point}")
endforeach()
require_count("${direct_source}"
    "CpuSample read_cpu_sample\\(const testing::DirectExecutionTestHooks & hooks,[ \t\r\n]+testing::DirectCpuSamplePoint point,[ \t\r\n]+size_t tile_index\\)"
    1 "dedicated test reader overload")

string(REGEX MATCH "struct DirectExecutionMetrics \\{[^}]*\\};" metrics_block
             "${direct_header}")
if (metrics_block STREQUAL "")
    message(FATAL_ERROR "DirectExecutionMetrics block is unavailable")
endif()
if (metrics_block MATCHES "TestHooks|SampleReader|sample_reader|context")
    message(FATAL_ERROR "production metrics layout contains conditional test hooks")
endif()
foreach(token IN ITEMS "struct DirectExecutionTestHooks" "DirectCpuSampleReader sample_reader"
                       "void * context")
    string(FIND "${direct_header}" "${token}" found)
    if (found EQUAL -1)
        message(FATAL_ERROR "missing test-only hook token ${token}")
    endif()
endforeach()

if (NOT DIRECT_CONTRACT_LEGACY_ACCEPTANCE)
    require_count("${test_cmake}" "GGML_GEMMINI_DIRECT_METRICS_TESTING=1" 1
                  "target-private direct metrics testing macro")
endif()
string(REGEX MATCH
    "target_compile_definitions\\(test-gemmini-exsia PRIVATE[^)]*EXSIA_VALIDATION=1\\)"
    test_defs "${test_cmake}")
if (test_defs STREQUAL "")
    message(FATAL_ERROR "test-gemmini-exsia private definitions block is unavailable")
endif()
if (NOT test_defs MATCHES "GGML_GEMMINI_DIRECT_METRICS_TESTING=1")
    message(FATAL_ERROR "testing macro is not confined to the exact target-private block")
endif()

set(strict_guard
    "if (CMAKE_SYSTEM_NAME STREQUAL \"Linux\" AND\n    CMAKE_SYSTEM_PROCESSOR MATCHES \"^(aarch64|arm64)$\")")
string(FIND "${test_cmake}" "${strict_guard}" strict_guard_index)
if (strict_guard_index EQUAL -1)
    message(FATAL_ERROR "strict target lacks the genuine Linux-AArch64 platform guard")
endif()
foreach(token IN ITEMS
        "add_library(test-gemmini-direct-linux-aarch64-strict OBJECT\n        ../ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp)"
        "CYCLE_DETAIL=1")
    string(FIND "${test_cmake}" "${token}" found)
    if (found EQUAL -1)
        message(FATAL_ERROR "strict target contract is missing ${token}")
    endif()
endforeach()
if (NOT DIRECT_CONTRACT_LEGACY_ACCEPTANCE)
    string(REGEX MATCH
        "add_library\\(test-gemmini-direct-linux-aarch64-strict OBJECT[^#]*add_dependencies\\(test-gemmini-exsia test-gemmini-direct-linux-aarch64-strict\\)"
        strict_target_block "${test_cmake}")
    if (strict_target_block STREQUAL "")
        list(APPEND structural_failures "strict target block is unavailable")
    else()
        string(REGEX MATCHALL
            "set_target_properties\\(test-gemmini-direct-linux-aarch64-strict PROPERTIES COMPILE_OPTIONS \"\"\\)"
            reset_matches "${strict_target_block}")
        list(LENGTH reset_matches reset_count)
        if (NOT reset_count EQUAL 1)
            list(APPEND structural_failures
                "strict target requires one inherited COMPILE_OPTIONS reset")
        endif()
        foreach(flag IN ITEMS
                -Wmissing-declarations -Wmissing-noreturn -Wall -Wextra -Wpedantic
                -Wcast-qual -Wextra-semi -Wunused-parameter -Werror)
            string(REGEX MATCHALL "${flag}([ \t\r\n]|\\))" flag_matches
                "${strict_target_block}")
            list(LENGTH flag_matches flag_count)
            if (NOT flag_count EQUAL 1)
                list(APPEND structural_failures
                    "strict target requires exactly one positive warning ${flag}")
            endif()
        endforeach()
    endif()
    if (structural_failures)
        list(JOIN structural_failures "\n" structural_failure_report)
        message(FATAL_ERROR "direct metrics structural contract failed:\n${structural_failure_report}")
    endif()
endif()

if (DEFINED CXX_COMPILER AND DEFINED GENERATED_DIR AND DEFINED PROJECT_ROOT)
    execute_process(
        COMMAND "${CXX_COMPILER}" -E -std=gnu++17
            -DCYCLE_DETAIL=0 -DLOG_CYCLE=0
            -DGGML_GEMMINI_ACTIVATION_BITS=8
            -DGGML_GEMMINI_WEIGHT_BITS=8
            -DGGML_GEMMINI_BLOCK_SIZE=32
            "-I${GENERATED_DIR}"
            "-I${PROJECT_ROOT}/ggml/include"
            "-I${PROJECT_ROOT}/ggml/src"
            "-I${PROJECT_ROOT}/ggml/src/ggml-gemmini"
            "-I${PROJECT_ROOT}/ggml/src/ggml-gemmini-utils/include"
            "-I${GEMMINI_SW_PATH}"
            "-I${GEMMINI_SW_PATH}/include"
            "-I${GEMMINI_SW_PATH}/gemmini-rocc-tests"
            "-I${GEMMINI_SW_PATH}/gemmini-rocc-tests/include"
            "-I${GEMMINI_SW_PATH}/rocc-software/src"
            "${DIRECT_SOURCE}"
        RESULT_VARIABLE preprocess_result
        OUTPUT_VARIABLE production_output
        ERROR_VARIABLE preprocess_error)
    if (NOT preprocess_result EQUAL 0)
        message(FATAL_ERROR "production preprocessing failed: ${preprocess_error}")
    endif()
    if (production_output MATCHES
        "DirectExecutionTestHooks|DirectCpuSamplePoint|test_cpu_sample|read_cpu_sample")
        message(FATAL_ERROR "non-Linux/detail-off production preprocessing retains private sampling symbols")
    endif()
endif()

message(STATUS "direct CPU metrics source contract passed")
