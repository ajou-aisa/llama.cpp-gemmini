if (NOT DEFINED TEST_GATE_EXECUTABLE OR
    NOT EXISTS "${TEST_GATE_EXECUTABLE}")
    message(FATAL_ERROR "TEST_GATE_EXECUTABLE must name the built backend gate test")
endif()

execute_process(
    COMMAND "${TEST_GATE_EXECUTABLE}"
    RESULT_VARIABLE gate_result
    OUTPUT_VARIABLE gate_stdout
    ERROR_VARIABLE gate_stderr)

if (NOT gate_result EQUAL 0)
    message(FATAL_ERROR
        "Machine-consumed backend gate failed with ${gate_result}\n${gate_stdout}${gate_stderr}")
endif()

message(STATUS "${gate_stdout}")
