if(NOT DEFINED TEST_CLI OR NOT EXISTS "${TEST_CLI}")
    message(FATAL_ERROR "TEST_CLI must name the built llama-cli binary")
endif()
if(NOT DEFINED TEST_ROOT)
    message(FATAL_ERROR "TEST_ROOT is required")
endif()

file(REMOVE_RECURSE "${TEST_ROOT}")
file(MAKE_DIRECTORY "${TEST_ROOT}/traversal")

function(run_invalid_sink name working_directory path diagnostic forbidden_path)
    execute_process(
        COMMAND "${TEST_CLI}" --gemmini-cycle-log "${path}"
                -m "${TEST_ROOT}/missing.gguf" -n 0
        WORKING_DIRECTORY "${working_directory}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
        TIMEOUT 30)
    set(combined "${output}${error}")
    if(result EQUAL 0)
        message(FATAL_ERROR "${name}: invalid sink exited successfully")
    endif()
    if(combined MATCHES "load the model|llama_model_load")
        message(FATAL_ERROR "${name}: invalid sink continued into model loading:\n${combined}")
    endif()
    string(REGEX MATCHALL "${diagnostic}" diagnostics "${combined}")
    list(LENGTH diagnostics diagnostic_count)
    if(NOT diagnostic_count EQUAL 1)
        message(FATAL_ERROR "${name}: expected one setup diagnostic, got ${diagnostic_count}:\n${combined}")
    endif()
    if(EXISTS "${forbidden_path}")
        message(FATAL_ERROR "${name}: invalid sink created ${forbidden_path}")
    endif()
    if(EXISTS "${working_directory}/output/log/cycle-log.jsonl")
        message(FATAL_ERROR "${name}: invalid explicit sink created the default cycle file")
    endif()
endfunction()

run_invalid_sink(
    traversal
    "${TEST_ROOT}/traversal"
    "../escape.jsonl"
    "gemmini CycleLog [A-Za-z]+ failure"
    "${TEST_ROOT}/escape.jsonl")

file(WRITE "${TEST_ROOT}/blocked" "not a directory\n")
run_invalid_sink(
    blocked-parent
    "${TEST_ROOT}"
    "${TEST_ROOT}/blocked/cycle.jsonl"
    "gemmini CycleLog [A-Za-z]+ failure"
    "${TEST_ROOT}/blocked/cycle.jsonl")

run_invalid_sink(
    empty-path
    "${TEST_ROOT}"
    ""
    "Gemmini cycle log path must not be empty"
    "${TEST_ROOT}/output/log/cycle-log.jsonl")

file(REMOVE_RECURSE "${TEST_ROOT}")
