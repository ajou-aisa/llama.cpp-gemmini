if(NOT DEFINED TEST_SOURCE_DIR)
    message(FATAL_ERROR "TEST_SOURCE_DIR required")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(resolver "${TEST_SOURCE_DIR}/scripts/im2p-build-options.py")
foreach(flag IN ITEMS GGML_GEMMINI_ACT_QUANT_METRICS GGML_GEMMINI_RESIDUAL_METRICS)
    foreach(value IN ITEMS 2 ON -1 true)
        execute_process(COMMAND "${Python3_EXECUTABLE}" "${resolver}"
            unused build-arm64.sh GGML_GEMMINI=OFF -- "-D${flag}=${value}"
            RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
        if(result EQUAL 0 OR NOT error MATCHES "${flag} must be 0 or 1")
            message(FATAL_ERROR "invalid ${flag}=${value} admitted: ${output} ${error}")
        endif()
    endforeach()
endforeach()
foreach(act IN ITEMS 0 1)
    foreach(res IN ITEMS 0 1)
        execute_process(COMMAND "${Python3_EXECUTABLE}" "${resolver}"
            unused build-arm64.sh GGML_GEMMINI=OFF LOG_CYCLE=0 CYCLE_SIM=0
            CYCLE_DETAIL=0 GGML_CPU_CYCLE_LOG=0 --
            "-DGGML_GEMMINI_ACT_QUANT_METRICS=${act}"
            "-DGGML_GEMMINI_RESIDUAL_METRICS=${res}"
            RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
        if(NOT result EQUAL 0)
            message(FATAL_ERROR "independent metric options rejected: ${error}")
        endif()
        foreach(pair IN ITEMS "GGML_GEMMINI_ACT_QUANT_METRICS=${act}"
                "GGML_GEMMINI_RESIDUAL_METRICS=${res}" "LOG_CYCLE=0"
                "CYCLE_SIM=0" "CYCLE_DETAIL=0" "GGML_CPU_CYCLE_LOG=0")
            string(FIND "${output}" "-D${pair}" found)
            if(found EQUAL -1)
                message(FATAL_ERROR "resolved option absent or changed: ${pair}")
            endif()
        endforeach()
    endforeach()
endforeach()
