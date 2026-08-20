if (NOT DEFINED TEST_EXECUTABLE OR NOT DEFINED EXPECTED_CASE)
    message(FATAL_ERROR "TEST_EXECUTABLE and EXPECTED_CASE are required")
endif()

execute_process(
    COMMAND "${TEST_EXECUTABLE}" "--case=${EXPECTED_CASE}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr)
if (NOT result EQUAL 0)
    message(FATAL_ERROR "case ${EXPECTED_CASE} exited ${result}\n${stdout}${stderr}")
endif()

string(REGEX MATCHALL "TEST_CASE_BEGIN name=[^\r\n]+" markers "${stdout}${stderr}")
list(LENGTH markers marker_count)
if (NOT marker_count EQUAL 1 OR
    NOT markers STREQUAL "TEST_CASE_BEGIN name=${EXPECTED_CASE}")
    message(FATAL_ERROR "case ${EXPECTED_CASE} emitted invalid markers: ${markers}")
endif()

message(STATUS "${stdout}${stderr}")
