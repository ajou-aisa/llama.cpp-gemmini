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
printf 'cmake-arg:%s\n' "$@" >> "$CONTRACT_LOG"
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
            "GGML_GEMMINI=ON"
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
        "GGML_GEMMINI=ON"
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
            "GGML_GEMMINI=ON"
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
    if(script STREQUAL "build-arm64.sh")
        string(FIND "${commands}" "IM2P_SIM_IMPLEMENTATION=GEMMINI_HP1" implementation_at)
        if(implementation_at EQUAL -1)
            message(FATAL_ERROR
                "build-arm64.sh must provision GEMMINI_HP1 integrated IM2P artifacts:\n${commands}")
        endif()
    endif()
endfunction()

run_host_selected(build-arm64.sh)
run_host_selected(build-x86.sh)

foreach(scenario IN ITEMS default-environment ordinary-environment)
    set(control_log "${test_root}/controls-${scenario}.log")
    set(control_env IM2P_ARTIFACT_SET_DEFAULT=ALL_MATCHED IM2P_CACHE_JOBS_DEFAULT=3)
    set(expected_target gemmini-frontend-real-lib-all)
    set(expected_jobs 3)
    if(scenario STREQUAL "ordinary-environment")
        list(APPEND control_env IM2P_ARTIFACT_SET=SELECTED IM2P_CACHE_JOBS=2)
        set(expected_target gemmini-frontend-real-lib)
        set(expected_jobs 2)
    endif()
    execute_process(COMMAND env -i "PATH=${test_root}/bin:$ENV{PATH}"
        "CONTRACT_LOG=${control_log}" "BUILD_DIR=${test_root}/controls-${scenario}-build"
        BUILD_JOBS=4 "IM2P_SIM_ROOT=${TEST_IM2P_ROOT}" ${control_env}
        bash "${TEST_SOURCE_DIR}/build-x86.sh"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "Resolved shell controls failed ${scenario}: ${output}\n${error}")
    endif()
    file(READ "${control_log}" commands)
    if(NOT commands MATCHES "make:-C ${TEST_IM2P_ROOT} -j${expected_jobs} IM2P_CACHE_JOBS=${expected_jobs} [^\n]* ${expected_target}\n")
        message(FATAL_ERROR "Incorrect shell control precedence for ${scenario}: ${commands}")
    endif()
endforeach()

foreach(script IN ITEMS build-arm64.sh build-arm64-cpu.sh build-x86.sh build-riscv.sh)
    set(dry_log "${test_root}/${script}-dry.log")
    execute_process(COMMAND "${CMAKE_COMMAND}" -E env
        "PATH=${test_root}/bin:$ENV{PATH}" "CONTRACT_LOG=${dry_log}"
        "BUILD_DIR=${test_root}/dry-build" "BUILD_JOBS=1"
        bash "${TEST_SOURCE_DIR}/${script}" --dry-run
        WORKING_DIRECTORY "${TEST_SOURCE_DIR}"
        RESULT_VARIABLE rc OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
    if(NOT rc EQUAL 0 OR EXISTS "${dry_log}")
        message(FATAL_ERROR "${script} dry-run must succeed without provisioning: ${stdout}\n${stderr}")
    endif()
    if(script STREQUAL "build-x86.sh")
        string(REGEX MATCH "IM2P_EFFECTIVE_CONFIG=([^\n]+)" summary "${stderr}")
        set(config "${CMAKE_MATCH_1}")
        foreach(pair IN ITEMS
                "GGML_GEMMINI_EXECUTION_BACKEND=IM2P_SIM"
                "IM2P_SIM_IMPLEMENTATION=GEMMINI_HP1"
                "GGML_GEMMINI_OPTION=WS" "GGML_GEMMINI_ENABLE_RMD=ON"
                "GGML_GEMMINI_ACTIVATION_BITS=8" "GGML_GEMMINI_WEIGHT_BITS=8"
                "GGML_GEMMINI_DIM=16" "GGML_GEMMINI_DEFAULT_RMD_BACKEND=WS"
                "GGML_GEMMINI_DEFAULT_MATMUL_MODE=STRIPE_PIPELINE")
            string(REPLACE "=" ";" fields "${pair}")
            list(GET fields 0 key)
            list(GET fields 1 expected)
            string(JSON actual GET "${config}" effective "${key}")
            if(NOT actual STREQUAL expected)
                message(FATAL_ERROR "x86 default ${key}=${actual}; expected ${expected}")
            endif()
        endforeach()
    endif()
    foreach(enabled IN ITEMS ON OFF)
        foreach(source IN ITEMS environment typed-cli)
            set(reject_log "${test_root}/${script}-${enabled}-${source}.log")
            set(reject_env "GGML_GEMMINI_EXECUTION_BACKEND=FPGA_UART")
            set(reject_cli)
            if(source STREQUAL "typed-cli")
                set(reject_env "GGML_GEMMINI_EXECUTION_BACKEND=HARDWARE")
                set(reject_cli -DGGML_GEMMINI_EXECUTION_BACKEND:STRING=FPGA_UART)
            endif()
            execute_process(COMMAND "${CMAKE_COMMAND}" -E env
                "PATH=${test_root}/bin:$ENV{PATH}" "CONTRACT_LOG=${reject_log}"
                "BUILD_DIR=${test_root}/reject-build" "BUILD_JOBS=1"
                "GGML_GEMMINI=${enabled}" ${reject_env}
                bash "${TEST_SOURCE_DIR}/${script}" ${reject_cli}
                WORKING_DIRECTORY "${TEST_SOURCE_DIR}"
                RESULT_VARIABLE rc OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
            if(rc EQUAL 0 OR EXISTS "${reject_log}" OR
               NOT stderr MATCHES "FPGA_UART is deprecated and unsupported")
                message(FATAL_ERROR "${script}/${enabled}/${source} must reject before provisioning: ${stdout}\n${stderr}")
            endif()
        endforeach()
    endforeach()
endforeach()

foreach(script IN ITEMS build-arm64.sh build-arm64-cpu.sh build-x86.sh build-riscv.sh)
    set(argv_log "${test_root}/${script}-argv.log")
    set(argv_build "${test_root}/${script}-argv-build")
    execute_process(COMMAND env -i "PATH=${test_root}/bin:$ENV{PATH}"
        "CONTRACT_LOG=${argv_log}" "BUILD_DIR=${argv_build}" BUILD_JOBS=2
        GGML_GEMMINI=OFF GGML_GEMMINI_DIM_DEFAULT=16 GGML_GEMMINI_DIM=32
        LOG_DEBUG_DEFAULT=1 LOG_DEBUG=0
        bash "${TEST_SOURCE_DIR}/${script}"
        -DLOG_DEBUG:BOOL=0 -DLOG_DEBUG:STRING=1 -DIM2P_DIM:STRING=64
        -DUPSTREAM_OPTION:STRING=first "-DUPSTREAM_OPTION:PATH=last value"
        -DBUILD_DIR:PATH=cmake-only -DBUILD_JOBS:STRING=99 -Wno-dev
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "${script} final argv failed: ${output}\n${error}")
    endif()
    file(STRINGS "${argv_log}" argv_rows REGEX "^cmake-arg:")
    set(keys)
    foreach(row IN LISTS argv_rows)
        if(row MATCHES "^cmake-arg:-D([^:=]+)(:[^=]+)?=")
            set(key "${CMAKE_MATCH_1}")
            if(key IN_LIST keys)
                message(FATAL_ERROR "${script} sent duplicate ${key}: ${argv_rows}")
            endif()
            list(APPEND keys "${key}")
        endif()
    endforeach()
    foreach(argument IN ITEMS -DLOG_DEBUG:STRING=1 -DGGML_GEMMINI_DIM:STRING=64
            "-DUPSTREAM_OPTION:PATH=last value" -DBUILD_DIR:PATH=cmake-only
            -DBUILD_JOBS:STRING=99 -Wno-dev)
        if(NOT "cmake-arg:${argument}" IN_LIST argv_rows)
            message(FATAL_ERROR "${script} lost final argument ${argument}: ${argv_rows}")
        endif()
    endforeach()
    file(READ "${argv_log}" commands)
    if(commands MATCHES "(^|\n)make:" OR NOT commands MATCHES "cmake:-B ${argv_build} " OR
       NOT commands MATCHES "cmake:--build ${argv_build} [^\n]* -j2")
        message(FATAL_ERROR "${script} reinterpreted CMake -D shell-control names: ${commands}")
    endif()
    foreach(control IN ITEMS IM2P_CACHE_JOBS IM2P_ARTIFACT_SET APPLE_SILICON_ARCH LIBOMP_PREFIX)
        if(control IN_LIST keys)
            message(FATAL_ERROR "${script} emitted shell control ${control} as a cache option")
        endif()
    endforeach()
    foreach(invalid IN ITEMS -DGGML_GEMMINI_SCALE_METRICS=2 -DGGML_GEMMINI_ACT_METRICS=ON
            -DGGML_GEMMINI_EXSIA_PROFILE_SCOPE=BAD "-DCYCLE_DETAIL=1;-DLOG_CYCLE=0"
            "-DIM2P_DIM=32;-DGGML_GEMMINI_DIM=64")
        set(reject_log "${test_root}/${script}-invalid-options.log")
        execute_process(COMMAND env -i "PATH=${test_root}/bin:$ENV{PATH}"
            "CONTRACT_LOG=${reject_log}" "BUILD_DIR=${argv_build}" BUILD_JOBS=2
            GGML_GEMMINI=OFF bash "${TEST_SOURCE_DIR}/${script}" ${invalid}
            RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
        if(result EQUAL 0 OR EXISTS "${reject_log}")
            message(FATAL_ERROR "${script} admitted ${invalid} before configure: ${output}\n${error}")
        endif()
    endforeach()
endforeach()

execute_process(COMMAND env -i "PATH=${test_root}/bin:$ENV{PATH}"
    "CONTRACT_LOG=${test_root}/static-dry.log" "BUILD_DIR=${test_root}/static-dry-build"
    BUILD_JOBS=2 bash "${TEST_SOURCE_DIR}/build-riscv.sh" static --dry-run
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0 OR EXISTS "${test_root}/static-dry.log" OR EXISTS "${test_root}/static-dry-build")
    message(FATAL_ERROR "RISC-V static dry-run invoked configure: ${output}\n${error}")
endif()
string(REGEX MATCH "IM2P_EFFECTIVE_CONFIG=([^\n]+)" summary "${error}")
set(config "${CMAKE_MATCH_1}")
foreach(pair IN ITEMS CMAKE_BUILD_TYPE=DEBUG BUILD_SHARED_LIBS=OFF GGML_NATIVE=OFF GGML_OPENMP=OFF)
    string(REPLACE "=" ";" fields "${pair}")
    list(GET fields 0 key)
    list(GET fields 1 expected)
    string(JSON actual GET "${config}" effective "${key}")
    if(NOT actual STREQUAL expected)
        message(FATAL_ERROR "Static policy ${key}=${actual}; expected ${expected}")
    endif()
endforeach()

# Make has a file-owned BUILD_DIR assignment, so an environment override alone
# is insufficient. Provisioning and configure must receive the same path.
foreach(cache_source IN ITEMS fallback environment typed-cli)
    set(cache_log "${test_root}/cache-${cache_source}.log")
    set(cache_env)
    set(cache_cli)
    set(expected_cache "${TEST_IM2P_ROOT}/build")
    if(NOT cache_source STREQUAL "fallback")
        set(expected_cache "${test_root}/cache-${cache_source}")
        list(APPEND cache_env "IM2P_SIM_BUILD_DIR=${expected_cache}")
    endif()
    if(cache_source STREQUAL "typed-cli")
        set(expected_cache "${test_root}/cache-cli-selected")
        list(APPEND cache_cli "-DIM2P_SIM_BUILD_DIR:PATH=${expected_cache}")
    endif()
    execute_process(COMMAND "${CMAKE_COMMAND}" -E env
        "PATH=${test_root}/bin:$ENV{PATH}" "CONTRACT_LOG=${cache_log}"
        "BUILD_DIR=${test_root}/cache-${cache_source}-build" "BUILD_JOBS=1"
        "IM2P_SIM_ROOT=${TEST_IM2P_ROOT}"
        "GGML_GEMMINI_EXECUTION_BACKEND=IM2P_SIM"
        ${cache_env} bash "${TEST_SOURCE_DIR}/build-x86.sh" ${cache_cli}
        WORKING_DIRECTORY "${TEST_SOURCE_DIR}"
        RESULT_VARIABLE cache_rc OUTPUT_VARIABLE cache_out ERROR_VARIABLE cache_err)
    if(NOT cache_rc EQUAL 0)
        message(FATAL_ERROR "Cache ${cache_source} failed: ${cache_out}\n${cache_err}")
    endif()
    file(READ "${cache_log}" cache_commands)
    set(cache_cmake_type "")
    if(cache_source STREQUAL "typed-cli")
        set(cache_cmake_type ":PATH")
    endif()
    foreach(required IN ITEMS " BUILD_DIR=${expected_cache} " "-DIM2P_SIM_BUILD_DIR${cache_cmake_type}=${expected_cache}")
        string(FIND "${cache_commands}" "${required}" cache_at)
        if(cache_at EQUAL -1)
            message(FATAL_ERROR "Make/configure cache mismatch (${cache_source}): ${cache_commands}")
        endif()
    endforeach()
    if(NOT cache_source STREQUAL "fallback" AND cache_commands MATCHES " BUILD_DIR=${TEST_IM2P_ROOT}/build ")
        message(FATAL_ERROR "Explicit cache must never write the sibling build cache")
    endif()
endforeach()

# A reused host build directory must not override the ARM64 script defaults.
set(stale_root "${test_root}/arm64-stale-cache")
file(MAKE_DIRECTORY "${stale_root}")
file(WRITE "${stale_root}/CMakeCache.txt"
    "GGML_GEMMINI:BOOL=ON\n"
    "GGML_GEMMINI_OPTION:STRING=WS\n"
    "GGML_GEMMINI_EXECUTION_BACKEND:STRING=IM2P_SIM\n"
    "CMAKE_TOOLCHAIN_FILE:FILEPATH=/stale/toolchain.cmake\n"
    "CMAKE_PREFIX_PATH:PATH=/stale/prefix\n"
    "OpenMP_ROOT:PATH=/stale/openmp\n"
    "CMAKE_BUILD_TYPE:STRING=Debug\n"
    "GGML_GEMMINI_DEFAULT_RMD_BACKEND:STRING=WS\n"
    "GGML_GEMMINI_ENABLE_RMD:BOOL=OFF\n"
    "LOG_DEBUG:STRING=1\n"
    "LOG_CYCLE:STRING=1\n")
set(stale_log "${test_root}/build-arm64-stale-cache.log")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
        "PATH=${test_root}/bin:$ENV{PATH}"
        "CONTRACT_LOG=${stale_log}"
        "BUILD_DIR=${stale_root}"
        "BUILD_JOBS=1"
        "IM2P_SIM_ROOT=${TEST_IM2P_ROOT}"
        bash "${TEST_SOURCE_DIR}/build-arm64.sh"
    WORKING_DIRECTORY "${TEST_SOURCE_DIR}"
    RESULT_VARIABLE stale_rc
    OUTPUT_VARIABLE stale_stdout
    ERROR_VARIABLE stale_stderr)
if(NOT stale_rc EQUAL 0)
    message(FATAL_ERROR
        "build-arm64.sh stale simulator cache recovery failed:\n${stale_stdout}\n${stale_stderr}")
endif()
file(READ "${stale_log}" stale_commands)
if(stale_commands MATCHES "(^|\n)make:")
    message(FATAL_ERROR "build-arm64.sh defaults must not provision a stale simulator cache")
endif()
foreach(stale_arg IN ITEMS
        "-U GGML_*"
        "-U IM2P_*"
        "-U LLAMA_*"
        "-U LOG_*"
        "-U CYCLE_*"
        "-U BUILD_SHARED_LIBS"
        "-U CMAKE_TOOLCHAIN_FILE"
        "-U CMAKE_PREFIX_PATH"
        "-U OpenMP_ROOT"
        "-DGGML_GEMMINI=OFF"
        "-DGGML_GEMMINI_OPTION=CPU"
        "-DGGML_GEMMINI_EXECUTION_BACKEND=HARDWARE"
        "-DIM2P_SIM_IMPLEMENTATION="
        "-DCMAKE_BUILD_TYPE=Release"
        "-DGGML_GEMMINI_DEFAULT_RMD_BACKEND=CPU"
        "-DGGML_GEMMINI_ENABLE_RMD=ON"
        "-DLOG_DEBUG=0"
        "-DLOG_CYCLE=0")
    string(FIND "${stale_commands}" "${stale_arg}" stale_arg_at)
    if(stale_arg_at EQUAL -1)
        message(FATAL_ERROR
            "build-arm64.sh must override stale cache with ${stale_arg}:\n${stale_commands}")
    endif()
endforeach()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
        "PATH=${test_root}/bin:$ENV{PATH}"
        "CONTRACT_LOG=${test_root}/invalid-jobs.log"
        "BUILD_DIR=${test_root}/invalid-jobs-build"
        "BUILD_JOBS=1"
        "IM2P_CACHE_JOBS=0"
        "IM2P_SIM_ROOT=${TEST_IM2P_ROOT}"
        "GGML_GEMMINI=ON"
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

file(READ "${TEST_SOURCE_DIR}/cmake/ggml-gemmini-im2p.cmake" root_cmake)
string(FIND "${root_cmake}"
    [=[${_GGML_GEMMINI_IM2P_BUILD_DIR}/selected/${IM2P_SIM_IMPLEMENTATION}/${GGML_GEMMINI_IM2P_ARTIFACT_ID}/current]=]
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
    "try_run(_GGML_GEMMINI_IM2P_PAIR_PROBE_RESULT" first_probe_at)
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
        "--expected-implementation"
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
string(FIND "${root_cmake}" "im2p_sim_implementation()" simulator_implementation_at)
string(FIND "${root_cmake}" "gemmini-hp1-integrated-v1" hp1_implementation_at)
file(READ "${TEST_SOURCE_DIR}/cmake/ggml-gemmini-config.cmake" config_cmake)
string(FIND "${config_cmake}" "IM2P_SIM_IMPLEMENTATION_GEMMINI_HP1=1" simulator_hp1_define_at)
if(simulator_implementation_at EQUAL -1 OR hp1_implementation_at EQUAL -1)
    message(FATAL_ERROR "CMake must validate selected IM2P simulator implementation identity")
endif()
if(simulator_hp1_define_at EQUAL -1)
    message(FATAL_ERROR
        "CMake must select SCU numerical contract for integrated GEMMINI_HP1 simulator")
endif()

file(READ
    "${TEST_SOURCE_DIR}/ggml/src/ggml-gemmini/CMakeLists.txt"
    gemmini_cmake)
file(READ
    "${TEST_IM2P_ROOT}/frontend/src/im2p_gemmini_frontend.cpp"
    im2p_frontend_source)
file(READ "${TEST_IM2P_ROOT}/Makefile" im2p_makefile)
foreach(build_verify_contract IN ITEMS
        "add_custom_target(ggml-gemmini-im2p-verify"
        "add_dependencies(ggml-gemmini ggml-gemmini-im2p-verify)")
    string(FIND "${gemmini_cmake}" "${build_verify_contract}" contract_at)
    if(contract_at EQUAL -1)
        message(FATAL_ERROR
            "IM2P archive integrity must be rechecked at build time")
    endif()
endforeach()
string(FIND "${gemmini_cmake}" "--expected-implementation" target_implementation_at)
if(target_implementation_at EQUAL -1)
    message(FATAL_ERROR "IM2P build-time verification must enforce implementation identity")
endif()
string(FIND "${im2p_frontend_source}" "IM2P_SIM_IMPLEMENTATION_GEMMINI_HP1" frontend_hp1_residual_at)
if(frontend_hp1_residual_at EQUAL -1)
    message(FATAL_ERROR
        "provisioned IM2P frontend must allow integrated GEMMINI_HP1 residual callback")
endif()
string(FIND "${im2p_makefile}" "IM2P_SIM_IMPLEMENTATION_GEMMINI_HP1=1" makefile_hp1_residual_at)
if(makefile_hp1_residual_at EQUAL -1)
    message(FATAL_ERROR
        "IM2P Makefile must compile integrated GEMMINI_HP1 frontend contract")
endif()
foreach(selected_archive IN ITEMS
        [=[${GGML_GEMMINI_IM2P_GENERATION}/libim2p_gemmini_frontend.a]=]
        [=[${GGML_GEMMINI_IM2P_GENERATION}/libim2p_sim.a]=])
    string(FIND "${root_cmake}" "${selected_archive}" archive_at)
    if(archive_at EQUAL -1)
        message(FATAL_ERROR
            "CMake must retain selected-pair archive linking: ${selected_archive}")
    endif()
endforeach()
