execute_process(
    COMMAND "${TEST_BINARY}" unsupported
    RESULT_VARIABLE test_rc
    OUTPUT_VARIABLE test_stdout
    ERROR_VARIABLE test_stderr)

set(test_output "${test_stdout}${test_stderr}")

if(NOT test_rc EQUAL 0)
    message(FATAL_ERROR
        "Expected typed unsupported baseline status, got rc=${test_rc}:\n${test_output}")
endif()
