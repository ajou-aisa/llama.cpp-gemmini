string(RANDOM LENGTH 16 run_suffix)
set(log_dir "${TEST_OUTPUT_DIR}/semantic-${run_suffix}")
execute_process(COMMAND "${TEST_EXECUTABLE}" "${log_dir}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "${output}\n${error}")
endif()
