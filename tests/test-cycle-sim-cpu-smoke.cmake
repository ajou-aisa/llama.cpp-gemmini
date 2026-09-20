string(RANDOM LENGTH 16 run_suffix)
set(log_dir "${TEST_OUTPUT_DIR}/cpu-smoke-${run_suffix}")
execute_process(COMMAND "${TEST_EXECUTABLE}" "${log_dir}" --preserve
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
message(STATUS "CPU operation smoke evidence: ${log_dir}")
if(NOT result EQUAL 0)
    message(FATAL_ERROR "${output}\n${error}")
endif()
