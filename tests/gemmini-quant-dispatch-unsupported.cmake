execute_process(
    COMMAND "${TEST_BINARY}" unsupported
    RESULT_VARIABLE test_rc
    OUTPUT_VARIABLE test_stdout
    ERROR_VARIABLE test_stderr)

set(test_output "${test_stdout}${test_stderr}")

if(test_rc EQUAL 0)
    message(FATAL_ERROR "Expected unsupported Gemmini baseline quantization pair to fail")
endif()

if(NOT test_output MATCHES "unsupported Gemmini baseline quantization pair")
    message(FATAL_ERROR "Expected unsupported baseline assertion, got:\n${test_output}")
endif()
