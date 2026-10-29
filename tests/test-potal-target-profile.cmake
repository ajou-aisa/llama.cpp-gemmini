if(NOT DEFINED TEST_SOURCE_DIR)
    message(FATAL_ERROR "TEST_SOURCE_DIR is required")
endif()

foreach(profile IN ITEMS functional performance)
    foreach(dim IN ITEMS 16 32)
        execute_process(COMMAND env -i "PATH=$ENV{PATH}" BUILD_JOBS=1
            bash "${TEST_SOURCE_DIR}/build-potal-firesim.sh" "${profile}" "${dim}" --dry-run
            RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
        if(NOT result EQUAL 0)
            message(FATAL_ERROR "${profile}/DIM${dim} failed: ${output}\n${error}")
        endif()
        string(REGEX MATCH "IM2P_EFFECTIVE_CONFIG=([^\n]+)" summary "${error}")
        set(config "${CMAKE_MATCH_1}")
        set(debug 1)
        if(profile STREQUAL performance)
            set(debug 0)
        endif()
        foreach(pair IN ITEMS
                "GGML_POTAL=ON" "GGML_GEMMINI=OFF" "GGML_BACKEND_DL=OFF"
                "GGML_GEMMINI_EXECUTION_BACKEND=HARDWARE" "GGML_GEMMINI_OPTION=WS"
                "GGML_GEMMINI_ACTIVATION_QUANT=EXSIA" "GGML_GEMMINI_ENABLE_RMD=ON"
                "GGML_GEMMINI_DEFAULT_RMD_BACKEND=WS"
                "GGML_GEMMINI_DEFAULT_MATMUL_MODE=STRIPE_PIPELINE"
                "GGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY=2" "GGML_POTAL_PAIRED_AUTO=ON"
                "GGML_GEMMINI_EXSIA_DEFAULT_MODE=SEQUENTIAL"
                "GGML_GEMMINI_ACTIVATION_BITS=8" "GGML_GEMMINI_WEIGHT_BITS=8"
                "GGML_GEMMINI_BLOCK_SIZE=32" "CYCLE_SIM=0" "CYCLE_DETAIL=0"
                "POTAL_PLANNING_DIAGNOSTICS=OFF" "LOG_DEBUG=${debug}" "LOG_CYCLE=1"
                "GGML_CPU_CYCLE_LOG=0" "GGML_GEMMINI_ENABLE_A4NKS_ATTENTION=OFF"
                "GGML_GEMMINI_OFFLOAD_A4NKS_ATTENTION=OFF"
                "POTAL_EXPECTED_DIM=${dim}" "GGML_GEMMINI_DIM=${dim}" "POTAL_TARGET_DIM=${dim}"
                "POTAL_HW_CONFIG=FireSimQuadRocketSinglePoTalD${dim}Config")
            string(REPLACE "=" ";" fields "${pair}")
            list(GET fields 0 key)
            list(GET fields 1 expected)
            string(JSON actual GET "${config}" effective "${key}")
            if(NOT actual STREQUAL expected)
                message(FATAL_ERROR "${profile}/DIM${dim}: ${key}=${actual}, expected ${expected}")
            endif()
        endforeach()
        string(JSON header GET "${config}" effective POTAL_HW_PARAMS)
        string(JSON target_header GET "${config}" effective POTAL_TARGET_PARAMS)
        if(NOT header STREQUAL target_header)
            message(FATAL_ERROR "Profile must bind the selected header to its generated target parameters")
        endif()
        file(STRINGS "${header}" header_dim REGEX "^#define DIM [0-9]+$")
        if(NOT header_dim STREQUAL "#define DIM ${dim}")
            message(FATAL_ERROR "${profile}/DIM${dim} selected wrong header: ${header}")
        endif()
        string(JSON build_dir GET "${config}" build_dir)
        if(NOT build_dir MATCHES "/build-potal-firesim-d${dim}-${profile}$")
            message(FATAL_ERROR "Profile build directory is not isolated: ${build_dir}")
        endif()
    endforeach()
endforeach()

foreach(profile IN ITEMS functional performance)
    foreach(precedence IN ITEMS default environment command)
        set(overrides GGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY_DEFAULT=3)
        set(arguments)
        set(expected 3)
        if(NOT precedence STREQUAL default)
            list(APPEND overrides GGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY=4)
            set(expected 4)
        endif()
        if(precedence STREQUAL command)
            list(APPEND arguments -DGGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY=5)
            set(expected 5)
        endif()
        execute_process(COMMAND env -i "PATH=$ENV{PATH}" BUILD_JOBS=1 ${overrides}
            bash "${TEST_SOURCE_DIR}/build-potal-firesim.sh" "${profile}" 32 --dry-run ${arguments}
            RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
        if(NOT result EQUAL 0)
            message(FATAL_ERROR "${profile}/${precedence} failed: ${output}\n${error}")
        endif()
        string(REGEX MATCH "IM2P_EFFECTIVE_CONFIG=([^\n]+)" summary "${error}")
        string(JSON actual GET "${CMAKE_MATCH_1}" effective GGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY)
        if(NOT actual STREQUAL expected)
            message(FATAL_ERROR "${profile}/${precedence}: capacity ${actual}, expected ${expected}")
        endif()
    endforeach()
endforeach()

foreach(args IN ITEMS "functional" "functional;64" "unknown;32")
    execute_process(COMMAND bash "${TEST_SOURCE_DIR}/build-potal-firesim.sh" ${args} --dry-run
        RESULT_VARIABLE result OUTPUT_QUIET ERROR_QUIET)
    if(result EQUAL 0)
        message(FATAL_ERROR "Invalid target profile accepted: ${args}")
    endif()
endforeach()

string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef suffix)
set(test_root "${CMAKE_CURRENT_BINARY_DIR}/potal-target-profile-${suffix}")
get_filename_component(params "${TEST_SOURCE_DIR}/../../gemmini-rocc-tests/include" ABSOLUTE)
foreach(case IN ITEMS software-dim expected-dim target-dim config-dim)
    set(expected 16)
    set(software 16)
    set(target 16)
    set(config_dim 16)
    if(case STREQUAL software-dim)
        set(software 32)
    elseif(case STREQUAL expected-dim)
        set(expected 32)
    elseif(case STREQUAL target-dim)
        set(target 32)
    else()
        set(config_dim 32)
    endif()
    execute_process(COMMAND "${CMAKE_COMMAND}" -S "${TEST_SOURCE_DIR}" -B "${test_root}/${case}"
        -DPOTAL_API_ONLY=ON -DGGML_POTAL=OFF -DGGML_GEMMINI=OFF
        "-DPOTAL_HW_PARAMS:FILEPATH=${params}/gemmini_potal_params.h"
        "-DPOTAL_EXPECTED_DIM=${expected}" "-DGGML_GEMMINI_DIM=${software}" "-DPOTAL_TARGET_DIM=${target}"
        "-DPOTAL_HW_CONFIG=FireSimQuadRocketSinglePoTalD${config_dim}Config"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(result EQUAL 0 OR NOT error MATCHES "DIM mismatch")
        message(FATAL_ERROR "${case} did not reject conflicting DIM values: ${output}\n${error}")
    endif()
endforeach()
file(READ "${params}/gemmini_potal_params.h" original)
string(REPLACE "#define BANK_ROWS 4096" "#define BANK_ROWS 8192" wrong_geometry "${original}")
file(WRITE "${test_root}/wrong-geometry.h" "${wrong_geometry}")
execute_process(COMMAND "${CMAKE_COMMAND}" -S "${TEST_SOURCE_DIR}" -B "${test_root}/geometry"
    -DPOTAL_API_ONLY=ON -DGGML_POTAL=OFF -DGGML_GEMMINI=OFF
    "-DPOTAL_HW_PARAMS:FILEPATH=${test_root}/wrong-geometry.h"
    "-DPOTAL_TARGET_PARAMS:FILEPATH=${params}/gemmini_potal_params.h"
    -DPOTAL_EXPECTED_DIM=16 -DGGML_GEMMINI_DIM=16 -DPOTAL_TARGET_DIM=16
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(result EQUAL 0 OR NOT error MATCHES "target parameter mismatch")
    message(FATAL_ERROR "Same-DIM mismatched hardware geometry was accepted: ${output}\n${error}")
endif()
file(REMOVE_RECURSE "${test_root}")
message(STATUS "PoTal FireSim target profile and DIM mismatch tests PASS")
