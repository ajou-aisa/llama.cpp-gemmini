if(NOT DEFINED RUNNER OR NOT DEFINED TEST_OUTPUT_DIR)
    message(FATAL_ERROR "RUNNER and TEST_OUTPUT_DIR are required")
endif()
execute_process(COMMAND "${RUNNER}" --build-info RESULT_VARIABLE code OUTPUT_VARIABLE build)
if(NOT code EQUAL 0)
    message(FATAL_ERROR "cannot query runner build flags")
endif()
string(JSON activation GET "${build}" activation_metrics)
string(JSON residual GET "${build}" residual_metrics)
if(activation OR residual)
    message(FATAL_ERROR "test requires both metric flags compiled OFF")
endif()
foreach(metric IN ITEMS activation residual)
    execute_process(COMMAND "${RUNNER}"
        --model "${TEST_OUTPUT_DIR}/missing-model.gguf"
        --file "${TEST_OUTPUT_DIR}/missing-dataset.txt"
        --output-dir "${TEST_OUTPUT_DIR}/native"
        "--${metric}-output" "${TEST_OUTPUT_DIR}/${metric}.jsonl"
        RESULT_VARIABLE code OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
    if(code EQUAL 0 OR NOT stderr MATCHES "${metric} metrics are compiled out")
        message(FATAL_ERROR "disabled ${metric} must reject before opening any input: ${stdout}${stderr}")
    endif()
endforeach()
if(EXISTS "${TEST_OUTPUT_DIR}")
    message(FATAL_ERROR "disabled metric preflight created an output path")
endif()
message(STATUS "OFF metric options reject before dataset/model/output access")
