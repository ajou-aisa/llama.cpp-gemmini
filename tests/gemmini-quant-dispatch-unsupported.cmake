file(REMOVE_RECURSE "${BINARY_DIR}")

execute_process(
    COMMAND "${CMAKE_COMMAND}"
        -S "${SOURCE_DIR}"
        -B "${BINARY_DIR}"
        -DGGML_GEMMINI_WEIGHT_QUANT=UNKNOWN
        -DLLAMA_BUILD_TESTS=OFF
    RESULT_VARIABLE configure_rc
    OUTPUT_VARIABLE configure_stdout
    ERROR_VARIABLE configure_stderr)

set(configure_output "${configure_stdout}${configure_stderr}")

if(configure_rc EQUAL 0)
    message(FATAL_ERROR "Expected GGML_GEMMINI_WEIGHT_QUANT=UNKNOWN configure to fail")
endif()

if(NOT configure_output MATCHES "GGML_GEMMINI_WEIGHT_QUANT must be PER_TENSOR")
    message(FATAL_ERROR "Expected unsupported-weight-quant error, got:\n${configure_output}")
endif()
