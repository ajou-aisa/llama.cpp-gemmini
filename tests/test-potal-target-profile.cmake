if(NOT DEFINED TEST_SOURCE_DIR)
    message(FATAL_ERROR "TEST_SOURCE_DIR is required")
endif()
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef suffix)
set(root "${CMAKE_CURRENT_BINARY_DIR}/potal-target-profile-${suffix}")
file(MAKE_DIRECTORY "${root}/bin")
file(WRITE "${root}/bin/cmake" [=[#!/bin/bash
if [[ "${1:-}" == --build ]]; then exit 86; fi
exec "$REAL_CMAKE" "$@"
]=])
execute_process(COMMAND chmod +x "${root}/bin/cmake" COMMAND_ERROR_IS_FATAL ANY)
function(configure_profile profile dim name environment arguments)
    set(dir "${root}/${name}")
    execute_process(COMMAND env -i "PATH=${root}/bin:$ENV{PATH}" "REAL_CMAKE=${CMAKE_COMMAND}"
        "BUILD_DIR=${dir}" BUILD_JOBS=1 ${environment}
        bash "${TEST_SOURCE_DIR}/build-potal-firesim.sh" "${profile}" "${dim}"
        -DCMAKE_TOOLCHAIN_FILE:FILEPATH= -DNPU_TRANSPORT=REFERENCE -DGGML_NATIVE=OFF
        -DGGML_OPENMP=OFF -DGGML_METAL=OFF -DLLAMA_BUILD_TESTS=OFF ${arguments}
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(NOT result EQUAL 86)
        message(FATAL_ERROR "${profile}/DIM${dim} failed: ${output}\n${error}")
    endif()
    file(STRINGS "${dir}/CMakeCache.txt" cache)
    set(PROFILE_CACHE "${cache}" PARENT_SCOPE)
    set(PROFILE_DIR "${dir}" PARENT_SCOPE)
endfunction()
foreach(profile IN ITEMS functional performance)
    foreach(dim IN ITEMS 16 32)
        configure_profile("${profile}" "${dim}" "${profile}-${dim}" "" "")
        if(profile STREQUAL functional)
            set(debug 1)
        else()
            set(debug 0)
        endif()
        foreach(pair IN ITEMS "GGML_POTAL=ON" "GGML_GEMMINI=OFF" "GGML_BACKEND_DL=OFF"
                "GGML_GEMMINI_ACTIVATION_QUANT=EXSIA" "GGML_GEMMINI_ENABLE_RMD=ON"
                "GGML_GEMMINI_DEFAULT_RMD_BACKEND=WS" "GGML_GEMMINI_DEFAULT_MATMUL_MODE=STRIPE_PIPELINE"
                "GGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY=2" "GGML_POTAL_PAIRED_AUTO=ON"
                "GGML_GEMMINI_ACTIVATION_BITS=8" "GGML_GEMMINI_WEIGHT_BITS=8"
                "GGML_GEMMINI_BLOCK_SIZE=32" "CYCLE_DETAIL=0" "LOG_DEBUG=${debug}" "LOG_CYCLE=1"
                "POTAL_EXPECTED_DIM=${dim}" "GGML_GEMMINI_DIM=${dim}" "POTAL_TARGET_DIM=${dim}"
                "POTAL_HW_CONFIG=FireSimQuadRocketSinglePoTalD${dim}Config")
            string(REPLACE "=" ";" fields "${pair}")
            list(GET fields 0 key)
            list(GET fields 1 expected)
            if(NOT PROFILE_CACHE MATCHES "${key}:[^=;]+=${expected}(;|$)")
                message(FATAL_ERROR "${profile}/DIM${dim} wrong ${pair}")
            endif()
        endforeach()
        file(READ "${PROFILE_DIR}/generated/npu_config.json" geometry)
        string(JSON actual_dim GET "${geometry}" geometry d)
        if(NOT actual_dim STREQUAL dim)
            message(FATAL_ERROR "profile generated wrong DIM: ${geometry}")
        endif()
        execute_process(COMMAND "${CMAKE_COMMAND}" -S "${TEST_SOURCE_DIR}" -B "${root}/mismatch-${profile}-${dim}"
            --preset "potal-int8-d${dim}-reference" -DPOTAL_TARGET_DIM=64 -DLLAMA_CURL=OFF
            RESULT_VARIABLE mismatch OUTPUT_QUIET ERROR_VARIABLE diagnostic)
        if(mismatch EQUAL 0 OR NOT diagnostic MATCHES "DIM mismatch")
            message(FATAL_ERROR "profile accepted conflicting target DIM: ${diagnostic}")
        endif()
    endforeach()
    foreach(precedence IN ITEMS default environment command)
        set(environment GGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY_DEFAULT=1)
        set(arguments)
        set(expected 1)
        if(NOT precedence STREQUAL default)
            list(APPEND environment GGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY=2)
            set(expected 2)
        endif()
        if(precedence STREQUAL command)
            set(arguments -DGGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY=1)
            set(expected 1)
        endif()
        configure_profile("${profile}" 32 "${profile}-${precedence}" "${environment}" "${arguments}")
        if(NOT PROFILE_CACHE MATCHES "GGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY:[^=;]+=${expected}(;|$)")
            message(FATAL_ERROR "profile environment precedence changed")
        endif()
    endforeach()
endforeach()
foreach(args IN ITEMS "functional" "functional;64" "unknown;32")
    execute_process(COMMAND bash "${TEST_SOURCE_DIR}/build-potal-firesim.sh" ${args} --dry-run
        RESULT_VARIABLE result OUTPUT_QUIET ERROR_QUIET)
    if(result EQUAL 0)
        message(FATAL_ERROR "invalid target profile accepted: ${args}")
    endif()
endforeach()
