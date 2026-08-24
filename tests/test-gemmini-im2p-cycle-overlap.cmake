if(NOT DEFINED TEST_BINARY OR NOT DEFINED TEST_ROOT)
    message(FATAL_ERROR "TEST_BINARY and TEST_ROOT are required")
endif()

file(REMOVE_RECURSE "${TEST_ROOT}")
file(MAKE_DIRECTORY "${TEST_ROOT}")

execute_process(
    COMMAND "${TEST_BINARY}" --case pipeline
    WORKING_DIRECTORY "${TEST_ROOT}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR
        "Pipeline cycle fixture failed (${result})\nstdout:\n${output}\nstderr:\n${error}")
endif()

set(cycle_path "${TEST_ROOT}/output/log/cycle-log.jsonl")
if(NOT EXISTS "${cycle_path}")
    message(FATAL_ERROR "Pipeline cycle fixture did not create ${cycle_path}")
endif()
file(READ "${cycle_path}" cycle_log)

string(FIND "${cycle_log}"
    "\"name\":\"cpu.Quantize activation\"" cpu_quant_at)
if(NOT cpu_quant_at EQUAL -1)
    message(FATAL_ERROR
        "Pipeline cycle sink contains overlapping CPU activation quantization")
endif()

string(FIND "${cycle_log}"
    "\"record_type\":\"IM2P_EXECUTION_TELEMETRY\"" rtl_at)
string(FIND "${cycle_log}"
    "\"rtl_work_total_cycles\":" rtl_total_at)
if(rtl_at EQUAL -1 OR rtl_total_at EQUAL -1)
    message(FATAL_ERROR
        "Pipeline cycle sink is missing RTL end-to-end work cycles")
endif()
