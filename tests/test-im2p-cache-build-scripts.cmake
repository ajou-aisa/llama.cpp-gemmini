if(NOT DEFINED TEST_SOURCE_DIR OR
   NOT EXISTS "${TEST_SOURCE_DIR}/build-arm64.sh")
    message(FATAL_ERROR "TEST_SOURCE_DIR must point to llama.cpp-gemmini")
endif()
if(NOT DEFINED TEST_IM2P_ROOT OR
   NOT EXISTS "${TEST_IM2P_ROOT}/Makefile")
    message(FATAL_ERROR "TEST_IM2P_ROOT must point to IM2P.sim")
endif()

get_filename_component(TEST_SOURCE_DIR "${TEST_SOURCE_DIR}" REALPATH)
get_filename_component(TEST_IM2P_ROOT "${TEST_IM2P_ROOT}" REALPATH)

set(test_root "${CMAKE_CURRENT_BINARY_DIR}/im2p-cache-build-scripts")
file(REMOVE_RECURSE "${test_root}")
file(MAKE_DIRECTORY "${test_root}/bin")

file(WRITE "${test_root}/bin/make" [=[#!/bin/bash
printf 'make:%s\n' "$*" >> "$CONTRACT_LOG"
]=])
file(WRITE "${test_root}/bin/cmake" [=[#!/bin/bash
printf 'cmake:%s\n' "$*" >> "$CONTRACT_LOG"
]=])
execute_process(
    COMMAND chmod +x "${test_root}/bin/make" "${test_root}/bin/cmake")

function(run_host_all_matched script)
    set(log "${test_root}/${script}.log")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
            "PATH=${test_root}/bin:$ENV{PATH}"
            "CONTRACT_LOG=${log}"
            "BUILD_DIR=${test_root}/${script}-build"
            "BUILD_JOBS=1"
            "IM2P_CACHE_JOBS=3"
            "IM2P_SIM_ROOT=${TEST_IM2P_ROOT}"
            "IM2P_ARTIFACT_SET=ALL_MATCHED"
            "GGML_GEMMINI_EXECUTION_BACKEND=IM2P_SIM"
            "GGML_GEMMINI_OPTION=WS"
            bash "${TEST_SOURCE_DIR}/${script}"
        WORKING_DIRECTORY "${TEST_SOURCE_DIR}"
        RESULT_VARIABLE rc
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr)
    if(NOT rc EQUAL 0)
        message(FATAL_ERROR
            "${script} ALL_MATCHED failed: ${stdout}\n${stderr}")
    endif()

    file(READ "${log}" commands)
    string(FIND "${commands}"
        "make:-C ${TEST_IM2P_ROOT} -j3 IM2P_CACHE_JOBS=3" jobs_at)
    string(FIND "${commands}"
        "gemmini-frontend-real-lib-all" aggregate_at)
    string(FIND "${commands}" "cmake:-B" configure_at)
    if(jobs_at EQUAL -1 OR aggregate_at EQUAL -1 OR
       configure_at EQUAL -1 OR NOT aggregate_at LESS configure_at)
        message(FATAL_ERROR
            "${script} must populate all matched caches before CMake:\n${commands}")
    endif()

    string(REGEX MATCH "cmake:-B[^\n]+" configure_command "${commands}")
    foreach(selected_arg IN ITEMS
            "-DGGML_GEMMINI_ACTIVATION_BITS="
            "-DGGML_GEMMINI_WEIGHT_BITS="
            "-DGGML_GEMMINI_DIM="
            "-DGGML_GEMMINI_BLOCK_SIZE=")
        string(REGEX MATCHALL "${selected_arg}" selected_occurrences
            "${configure_command}")
        list(LENGTH selected_occurrences selected_count)
        if(NOT selected_count EQUAL 1)
            message(FATAL_ERROR
                "${script} must pass exactly one selected ${selected_arg} value to CMake:\n${configure_command}")
        endif()
    endforeach()
endfunction()

run_host_all_matched(build-arm64.sh)
run_host_all_matched(build-x86.sh)

set(default_all_log "${test_root}/build-x86-default-all.log")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
        "PATH=${test_root}/bin:$ENV{PATH}"
        "CONTRACT_LOG=${default_all_log}"
        "BUILD_DIR=${test_root}/build-x86-default-all-build"
        "BUILD_JOBS=7"
        "IM2P_SIM_ROOT=${TEST_IM2P_ROOT}"
        "IM2P_ARTIFACT_SET=ALL_MATCHED"
        "GGML_GEMMINI_EXECUTION_BACKEND=IM2P_SIM"
        "GGML_GEMMINI_OPTION=WS"
        bash "${TEST_SOURCE_DIR}/build-x86.sh"
    WORKING_DIRECTORY "${TEST_SOURCE_DIR}"
    RESULT_VARIABLE default_all_rc
    OUTPUT_VARIABLE default_all_stdout
    ERROR_VARIABLE default_all_stderr)
if(NOT default_all_rc EQUAL 0)
    message(FATAL_ERROR
        "build-x86.sh default ALL_MATCHED failed: ${default_all_stdout}\n${default_all_stderr}")
endif()
file(READ "${default_all_log}" default_all_commands)
string(FIND "${default_all_commands}"
    "make:-C ${TEST_IM2P_ROOT} -j1 IM2P_CACHE_JOBS=1" default_all_jobs_at)
if(default_all_jobs_at EQUAL -1)
    message(FATAL_ERROR
        "ALL_MATCHED must default IM2P_CACHE_JOBS to 1, independent of BUILD_JOBS:\n${default_all_commands}")
endif()

function(run_host_selected script)
    set(log "${test_root}/${script}-selected.log")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
            "PATH=${test_root}/bin:$ENV{PATH}"
            "CONTRACT_LOG=${log}"
            "BUILD_DIR=${test_root}/${script}-selected-build"
            "BUILD_JOBS=4"
            "IM2P_SIM_ROOT=${TEST_IM2P_ROOT}"
            "GGML_GEMMINI_EXECUTION_BACKEND=IM2P_SIM"
            "GGML_GEMMINI_OPTION=WS"
            bash "${TEST_SOURCE_DIR}/${script}"
        WORKING_DIRECTORY "${TEST_SOURCE_DIR}"
        RESULT_VARIABLE rc
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr)
    if(NOT rc EQUAL 0)
        message(FATAL_ERROR "${script} SELECTED failed: ${stdout}\n${stderr}")
    endif()
    file(READ "${log}" commands)
    string(FIND "${commands}" "gemmini-frontend-real-lib-all" aggregate_at)
    string(FIND "${commands}" "gemmini-frontend-real-lib" selected_at)
    string(FIND "${commands}"
        "make:-C ${TEST_IM2P_ROOT} -j4 IM2P_CACHE_JOBS=4" selected_jobs_at)
    if(selected_at EQUAL -1 OR selected_jobs_at EQUAL -1 OR
       NOT aggregate_at EQUAL -1)
        message(FATAL_ERROR
            "${script} must default to the selected target:\n${commands}")
    endif()
endfunction()

run_host_selected(build-arm64.sh)
run_host_selected(build-x86.sh)

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
        "PATH=${test_root}/bin:$ENV{PATH}"
        "CONTRACT_LOG=${test_root}/invalid-jobs.log"
        "BUILD_JOBS=1"
        "IM2P_CACHE_JOBS=0"
        "IM2P_SIM_ROOT=${TEST_IM2P_ROOT}"
        "GGML_GEMMINI_EXECUTION_BACKEND=IM2P_SIM"
        "GGML_GEMMINI_OPTION=WS"
        bash "${TEST_SOURCE_DIR}/build-x86.sh"
    WORKING_DIRECTORY "${TEST_SOURCE_DIR}"
    RESULT_VARIABLE invalid_jobs_rc
    OUTPUT_VARIABLE invalid_jobs_stdout
    ERROR_VARIABLE invalid_jobs_stderr)
if(invalid_jobs_rc EQUAL 0 OR EXISTS "${test_root}/invalid-jobs.log")
    message(FATAL_ERROR "Invalid IM2P_CACHE_JOBS must fail before Make/CMake")
endif()
string(CONCAT invalid_jobs_output
    "${invalid_jobs_stdout}" "\n" "${invalid_jobs_stderr}")
if(NOT invalid_jobs_output MATCHES "IM2P_CACHE_JOBS")
    message(FATAL_ERROR
        "Invalid cache-job rejection must identify IM2P_CACHE_JOBS")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
        "PATH=${test_root}/bin:$ENV{PATH}"
        "CONTRACT_LOG=${test_root}/build-riscv.log"
        "IM2P_ARTIFACT_SET=ALL_MATCHED"
        bash "${TEST_SOURCE_DIR}/build-riscv.sh"
    WORKING_DIRECTORY "${TEST_SOURCE_DIR}"
    RESULT_VARIABLE riscv_rc
    OUTPUT_VARIABLE riscv_stdout
    ERROR_VARIABLE riscv_stderr)
if(riscv_rc EQUAL 0)
    message(FATAL_ERROR "build-riscv.sh must reject ALL_MATCHED")
endif()
string(CONCAT riscv_output "${riscv_stdout}" "\n" "${riscv_stderr}")
string(FIND "${riscv_output}" "ALL_MATCHED" scope_at)
if(scope_at EQUAL -1)
    message(FATAL_ERROR
        "RISC-V rejection must identify ALL_MATCHED:\n${riscv_output}")
endif()


file(READ "${TEST_SOURCE_DIR}/scripts/im2p-host-provision.sh" helper_source)
string(FIND "${helper_source}" "cargo" direct_cargo_at)
if(NOT direct_cargo_at EQUAL -1)
    message(FATAL_ERROR "Host provisioning must delegate to Make, not Cargo")
endif()

file(READ "${TEST_SOURCE_DIR}/CMakeLists.txt" root_cmake)
string(FIND "${root_cmake}"
    [=[/build/selected/${GGML_GEMMINI_IM2P_ARTIFACT_ID}/current]=]
    manifest_at)
string(FIND "${root_cmake}"
    [=[GGML_GEMMINI_IM2P_GENERATION]=] generation_at)
string(FIND "${root_cmake}" "REALPATH" realpath_at)
string(FIND "${root_cmake}" "real_lib_cache.py" verifier_at)
string(FIND "${root_cmake}"
    "find_package(Python3 REQUIRED COMPONENTS Interpreter)" python_find_at)
string(FIND "${root_cmake}"
    [=[COMMAND "${Python3_EXECUTABLE}"]=] verify_command_at)
string(FIND "${root_cmake}" "COMMAND python3" bare_python_at)
string(FIND "${root_cmake}"
    "try_run(_GGML_GEMMINI_IM2P_PROBE_RESULT" first_probe_at)
if(manifest_at EQUAL -1 OR generation_at EQUAL -1 OR realpath_at EQUAL -1 OR
   verifier_at EQUAL -1 OR
   python_find_at EQUAL -1 OR verify_command_at EQUAL -1 OR
   first_probe_at EQUAL -1 OR NOT bare_python_at EQUAL -1 OR
   NOT python_find_at LESS verify_command_at OR
   NOT verify_command_at LESS first_probe_at)
    message(FATAL_ERROR
        "CMake must verify the selected real-lib manifest before IM2P try_run")
endif()
foreach(expected_verifier_arg IN ITEMS
        "--expected-identity"
        "--expected-block-size"
        "--expected-platform"
        "--expected-platform-release"
        "--expected-arch"
        "--artifact-kind")
    string(FIND "${root_cmake}" "${expected_verifier_arg}" expected_at)
    if(expected_at EQUAL -1)
        message(FATAL_ERROR
            "CMake verifier must enforce ${expected_verifier_arg}")
    endif()
endforeach()

file(READ
    "${TEST_SOURCE_DIR}/ggml/src/ggml-gemmini/CMakeLists.txt"
    gemmini_cmake)
foreach(build_verify_contract IN ITEMS
        "add_custom_target(ggml-gemmini-im2p-verify"
        "add_dependencies(ggml-gemmini ggml-gemmini-im2p-verify)")
    string(FIND "${gemmini_cmake}" "${build_verify_contract}" contract_at)
    if(contract_at EQUAL -1)
        message(FATAL_ERROR
            "IM2P archive integrity must be rechecked at build time")
    endif()
endforeach()
foreach(selected_archive IN ITEMS
        [=[${GGML_GEMMINI_IM2P_GENERATION}/libim2p_gemmini_frontend.a]=]
        [=[${GGML_GEMMINI_IM2P_GENERATION}/libim2p_sim.a]=])
    string(FIND "${root_cmake}" "${selected_archive}" archive_at)
    if(archive_at EQUAL -1)
        message(FATAL_ERROR
            "CMake must retain selected-pair archive linking: ${selected_archive}")
    endif()
endforeach()
