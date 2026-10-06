if(NOT DEFINED TEST_SCRIPT OR NOT EXISTS "${TEST_SCRIPT}")
    message(FATAL_ERROR "TEST_SCRIPT must point to build-arm64.sh")
endif()

if(NOT DEFINED TEST_ROOT)
    set(TEST_ROOT "${CMAKE_CURRENT_BINARY_DIR}/arm64-log-defaults")
endif()
string(RANDOM LENGTH 12 ALPHABET abcdef0123456789 run_id)
set(run_root "${TEST_ROOT}/${run_id}")
file(MAKE_DIRECTORY "${run_root}/bin")
foreach(tool IN ITEMS cmake make brew)
    file(WRITE "${run_root}/bin/${tool}" [=[#!/bin/bash
printf '%s\n' "$0 $*" >> "$CONTRACT_LOG"
exit 97
]=])
    execute_process(COMMAND chmod +x "${run_root}/bin/${tool}"
        RESULT_VARIABLE chmod_result)
    if(NOT chmod_result EQUAL 0)
        message(FATAL_ERROR "Could not install ${tool} dry-run tripwire")
    endif()
endforeach()

get_filename_component(source_root "${TEST_SCRIPT}" DIRECTORY)
set(facade "${run_root}/sdk-absent/llama")
file(MAKE_DIRECTORY "${facade}" "${run_root}/configure-bin")
file(COPY "${source_root}/CMakeLists.txt" "${TEST_SCRIPT}" DESTINATION "${facade}")
foreach(directory IN ITEMS cmake ggml src common include tools examples pocs vendor scripts)
    file(CREATE_LINK "${source_root}/${directory}" "${facade}/${directory}" SYMBOLIC)
endforeach()
file(WRITE "${run_root}/configure-bin/cmake" [=[#!/bin/bash
if [[ "${1:-}" == --build ]]; then
    printf '%s\n' "$*" >> "$BUILD_STUB_LOG"
    exit 0
fi
exec "$REAL_CMAKE" "$@"
]=])
file(WRITE "${run_root}/configure-bin/make" [=[#!/bin/bash
printf '%s\n' "$*" >> "$PROVISION_LOG"
exit 97
]=])
execute_process(COMMAND chmod +x "${run_root}/configure-bin/cmake" "${run_root}/configure-bin/make"
    RESULT_VARIABLE chmod_result)
if(NOT chmod_result EQUAL 0)
    message(FATAL_ERROR "Could not install OFF configure fixture wrappers")
endif()
set(off_build "${run_root}/sdk-absent/build")
execute_process(COMMAND env -i "PATH=${run_root}/configure-bin:$ENV{PATH}"
    "REAL_CMAKE=${CMAKE_COMMAND}" "BUILD_STUB_LOG=${run_root}/build-stub.log"
    "PROVISION_LOG=${run_root}/provision.log" "BUILD_DIR=${off_build}" BUILD_JOBS=2
    CMAKE_GENERATOR=Ninja PYTHONDONTWRITEBYTECODE=1
    bash "${facade}/build-arm64.sh"
    -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_OPENMP=OFF
    -DGGML_GEMMINI_ENABLE_OPENMP=OFF -DGGML_GEMMINI_EXSIA_DEFAULT_MODE=SEQUENTIAL
    -DLLAMA_CURL=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
    WORKING_DIRECTORY "${run_root}/sdk-absent"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
file(WRITE "${run_root}/sdk-absent-configure.log" "exit=${result}\n${output}\n${error}")
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Default Gemmini OFF wrapper must configure without an SDK: ${output}\n${error}")
endif()
file(READ "${off_build}/CMakeCache.txt" off_cache)
if(NOT off_cache MATCHES "GGML_GEMMINI:[^=]+=OFF" OR
   NOT off_cache MATCHES "GEMMINI_SW_PATH:[^=]+=.*sdk-absent/llama/../RISC-V-DynDNN-gemmini-include")
    message(FATAL_ERROR "OFF wrapper must retain its declared missing header path and backend selection")
endif()
if(EXISTS "${run_root}/sdk-absent/RISC-V-DynDNN-gemmini-include" OR
   EXISTS "${run_root}/sdk-absent/IM2P.sim" OR EXISTS "${run_root}/provision.log" OR
   NOT EXISTS "${run_root}/build-stub.log" OR NOT EXISTS "${off_build}/build.ninja" OR
   EXISTS "${off_build}/generated/gemmini_params.h")
    message(FATAL_ERROR "OFF SDK-absent configure must avoid provisioning and Gemmini headers")
endif()
message(STATUS "sdk-absent: actual configure succeeded with default Gemmini OFF; no SDK/provisioning")

foreach(scenario IN ITEMS defaults environment typed-cli simulator)
    set(environment)
    set(arguments)
    set(expected 0)
    if(scenario STREQUAL "environment" OR scenario STREQUAL "typed-cli")
        set(environment LOG_DEBUG=1 LOG_CYCLE=1)
        set(expected 1)
    endif()
    if(scenario STREQUAL "typed-cli")
        set(arguments -DLOG_DEBUG:STRING=0 -DLOG_CYCLE:STRING=0)
        set(expected 0)
    elseif(scenario STREQUAL "simulator")
        set(arguments -DGGML_GEMMINI=ON -DGGML_GEMMINI_EXECUTION_BACKEND=IM2P_SIM
            -DGGML_GEMMINI_OPTION=WS -DIM2P_SIM_IMPLEMENTATION=LEGACY_BSV)
    endif()
    set(build_dir "${run_root}/${scenario}-build")
    execute_process(
        COMMAND env -i "PATH=${run_root}/bin:$ENV{PATH}"
            "CONTRACT_LOG=${run_root}/commands.log" "BUILD_DIR=${build_dir}"
            BUILD_JOBS=2 PYTHONDONTWRITEBYTECODE=1 ${environment}
            bash "${TEST_SCRIPT}" --dry-run ${arguments}
        WORKING_DIRECTORY "${run_root}"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    file(WRITE "${run_root}/${scenario}.log"
        "exit=${result}\n${output}\n${error}")
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "${scenario} dry-run failed: ${output}\n${error}")
    endif()
    if(EXISTS "${run_root}/commands.log" OR EXISTS "${build_dir}")
        message(FATAL_ERROR "${scenario} dry-run invoked provisioning/configure/build")
    endif()
    if(NOT output MATCHES "Dry run: no provisioning, configure, build, or device access")
        message(FATAL_ERROR "${scenario} did not complete the dry-run path")
    endif()
    string(REGEX MATCH "\"effective\": *\\{[^}]*\\}" effective "${error}")
    foreach(variable IN ITEMS LOG_DEBUG LOG_CYCLE GGML_CPU_CYCLE_LOG)
        if(NOT effective MATCHES "\"${variable}\": *\"${expected}\"")
            message(FATAL_ERROR
                "${scenario} expected effective ${variable}=${expected}: ${error}")
        endif()
    endforeach()
    if(scenario STREQUAL "simulator" AND
       NOT effective MATCHES "\"GGML_GEMMINI_EXECUTION_BACKEND\": *\"IM2P_SIM\"")
        message(FATAL_ERROR
            "simulator dry-run did not exercise the provisioning backend: ${error}")
    endif()
    message(STATUS "${scenario}: effective logs=${expected}, no provisioning/configure/build")
endforeach()
