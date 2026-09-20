string(RANDOM LENGTH 16 run_suffix)
set(log_dir "${TEST_OUTPUT_DIR}/cycle-sim-${TEST_MODE}-${run_suffix}")
file(MAKE_DIRECTORY "${log_dir}")
execute_process(COMMAND "${CMAKE_COMMAND}" -E env "GEMMINI_LOG_DIR=${log_dir}"
    "${TEST_EXECUTABLE}" "${TEST_MODE}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
message(STATUS "NPU trace fixture: ${log_dir}/npu-cycle-trace.jsonl")
if(NOT result EQUAL 0)
    message(FATAL_ERROR "${output}\n${error}")
endif()
