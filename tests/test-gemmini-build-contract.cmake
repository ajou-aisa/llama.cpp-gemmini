set(CMAKE_EXECUTE_PROCESS_COMMAND_ECHO STDOUT)

function(record_configure name expected result output)
    message(STATUS "CONTRACT_CONFIGURE ${name} expected_success=${expected} exit=${result}")
    file(APPEND "${TEST_BINARY_ROOT}/configure-results.tsv" "${name}\t${expected}\t${result}\n")
    file(WRITE "${TEST_BINARY_ROOT}/${name}.log" "exit=${result}\n${output}")
endfunction()

function(expect_contains path needle)
    file(READ "${path}" content)
    string(FIND "${content}" "${needle}" found_at)
    if (found_at EQUAL -1)
        message(FATAL_ERROR "Expected '${needle}' in ${path}")
    endif()
endfunction()

function(expect_not_contains path needle)
    file(READ "${path}" content)
    string(FIND "${content}" "${needle}" found_at)
    if (NOT found_at EQUAL -1)
        message(FATAL_ERROR "Did not expect '${needle}' in ${path}")
    endif()
endfunction()

function(expect_configured_test_targets path)
    if(NOT EXISTS "${path}")
        message(FATAL_ERROR "Missing configured test target contract: ${path}")
    endif()
    include("${path}")
    foreach(target IN LISTS GEMMINI_TEST_EXECUTABLES)
        if(NOT target IN_LIST GEMMINI_TEST_DEPENDENCIES)
            message(FATAL_ERROR "Registered test is absent from aggregator dependencies: ${target}")
        endif()
    endforeach()
    if(NOT "test-gemmini-log-boundary" IN_LIST GEMMINI_TEST_EXECUTABLES)
        message(FATAL_ERROR "Configured log-boundary test is missing")
    endif()
    if(NOT GEMMINI_CPU_SINK_REQUIRED STREQUAL test-ggml-cpu-cycle-sink_EXISTS)
        message(FATAL_ERROR "Configured CPU cycle sink does not match enabled instrumentation")
    endif()
    foreach(target IN ITEMS test-ggml-cpu-cycle-sink test-gemmini-rmd-executor)
        if(${target}_EXISTS AND NOT target IN_LIST GEMMINI_TEST_EXECUTABLES)
            message(FATAL_ERROR "Configured optional test is missing from aggregator: ${target}")
        endif()
    endforeach()
    if(test-ggml-cpu-cycle-sink_EXISTS AND
       NOT "LOG_CYCLE=1" IN_LIST test-ggml-cpu-cycle-sink_DEFINITIONS)
        message(FATAL_ERROR "Configured CPU cycle sink requires LOG_CYCLE=1")
    endif()
endfunction()

function(run_test_registration_case name dynamic_backend gemmini)
    set(build_dir "${TEST_BINARY_ROOT}/${name}")
    execute_process(COMMAND "${TEST_CMAKE_COMMAND}"
        -S "${TEST_SOURCE_DIR}" -B "${build_dir}"
        -DGGML_GEMMINI=${gemmini} -DGGML_GEMMINI_OPTION=CPU
        -DGGML_BACKEND_DL=${dynamic_backend} -DLLAMA_BUILD_TESTS=ON
        -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
        -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_BLAS=OFF -DGGML_OPENMP=OFF
        -DLOG_CYCLE=1 -DCYCLE_DETAIL=1 -DGGML_CPU_CYCLE_LOG=ON -DLLAMA_CURL=OFF
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    record_configure("${name}" TRUE "${result}" "${output}\n${error}")
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "Test registration configure failed: ${output}\n${error}")
    endif()
    expect_configured_test_targets("${build_dir}/tests/gemmini-test-targets.cmake")
endfunction()

function(expect_ws_cycle_build_script script_name)
    set(script_path "${TEST_SOURCE_DIR}/${script_name}")
    if(NOT EXISTS "${script_path}")
        message(FATAL_ERROR "Missing production build script ${script_path}")
    endif()
    foreach(value IN ITEMS default 0 1)
        set(arguments)
        set(expected 0)
        if(NOT value STREQUAL "default")
            set(arguments "-DGGML_GEMMINI_WS_LOOP_CYCLE=${value}")
            set(expected "${value}")
        endif()
        execute_process(
            COMMAND env -i "PATH=$ENV{PATH}" BUILD_JOBS=2 PYTHONDONTWRITEBYTECODE=1
                bash "${script_path}" --dry-run ${arguments}
            RESULT_VARIABLE rc OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
        file(WRITE "${TEST_BINARY_ROOT}/${script_name}-ws-cycle-${value}.log"
            "exit=${rc}\n${stdout}\n${stderr}")
        if(NOT rc EQUAL 0)
            message(FATAL_ERROR "${script_name} dry-run failed: ${stdout}\n${stderr}")
        endif()
        if(value STREQUAL "default")
            if(stdout MATCHES "-DGGML_GEMMINI_WS_LOOP_CYCLE(:[A-Za-z]+)?=1")
                message(FATAL_ERROR "${script_name} overrides the CMake WS loop logging default")
            endif()
        elseif(NOT stdout MATCHES "-DGGML_GEMMINI_WS_LOOP_CYCLE=${expected}([ \n]|$)")
            message(FATAL_ERROR "${script_name} lost WS_LOOP_CYCLE=${value}: ${stdout}")
        endif()
    endforeach()
    string(CONCAT deprecated_option "GGML_GEMMINI_WS_LOOP_" "DE" "BUG")
    expect_not_contains("${script_path}" "${deprecated_option}")
endfunction()

function(expect_dynamic_backend_production_script script_name)
    execute_process(COMMAND env -i "PATH=$ENV{PATH}" BUILD_JOBS=2 bash "${TEST_SOURCE_DIR}/${script_name}" --dry-run
        RESULT_VARIABLE rc OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(NOT rc EQUAL 0 OR NOT output MATCHES "-DGGML_BACKEND_DL=ON" OR NOT output MATCHES "-DLLAMA_BUILD_TESTS=OFF")
        message(FATAL_ERROR "${script_name} production policy failed: ${output} ${error}")
    endif()
endfunction()

if(DEFINED TEST_TARGET_CONTRACT)
    expect_configured_test_targets("${TEST_TARGET_CONTRACT}")
    return()
endif()

function(mode_index mode out_var)
    set(modes FULL STRIPE_PIPELINE)
    list(FIND modes "${mode}" index)
    if (index EQUAL -1)
        message(FATAL_ERROR "Unknown matmul mode '${mode}'")
    endif()
    set(${out_var} "${index}" PARENT_SCOPE)
endfunction()

function(run_configure_case name stripe pipeline mode local_workers rmd rmd_backend block_size expect_success)
    set(build_dir "${TEST_BINARY_ROOT}/${name}")
    file(REMOVE_RECURSE "${build_dir}")
    file(MAKE_DIRECTORY "${build_dir}")

    set(cmake_args
        -S "${TEST_SOURCE_DIR}"
        -B "${build_dir}"
        -DGGML_GEMMINI_ENABLE_STRIPE_MATMUL=${stripe}
        -DGGML_GEMMINI_ENABLE_STRIPE_PIPELINE=${pipeline}
        -DGGML_GEMMINI_DEFAULT_MATMUL_MODE=${mode}
        -DGGML_GEMMINI_EXSIA_LOCAL_WORKERS=${local_workers}
        -DGGML_GEMMINI_ENABLE_RMD=${rmd}
        -DGGML_GEMMINI_BLOCK_SIZE=${block_size}
        -DLLAMA_BUILD_TESTS=OFF)
    if (ARGC GREATER 10)
        set(expected_activation_quant "${ARGV10}")
        list(APPEND cmake_args "-DGGML_GEMMINI_ACTIVATION_QUANT=${expected_activation_quant}")
    else()
        set(expected_activation_quant EXSIA)
    endif()
    if (rmd_backend STREQUAL "DEFAULT")
        set(expected_rmd_backend CPU)
    else()
        set(expected_rmd_backend "${rmd_backend}")
        list(APPEND cmake_args "-DGGML_GEMMINI_DEFAULT_RMD_BACKEND=${rmd_backend}")
    endif()
    if (TEST_GENERATOR)
        list(APPEND cmake_args -G "${TEST_GENERATOR}")
    endif()
    if (TEST_GENERATOR_PLATFORM)
        list(APPEND cmake_args -A "${TEST_GENERATOR_PLATFORM}")
    endif()
    if (TEST_GENERATOR_TOOLSET)
        list(APPEND cmake_args -T "${TEST_GENERATOR_TOOLSET}")
    endif()

    execute_process(
        COMMAND "${TEST_CMAKE_COMMAND}" ${cmake_args}
        RESULT_VARIABLE rc
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr)

    record_configure("${name}" "${expect_success}" "${rc}" "${stdout}\n${stderr}")
    if (expect_success)
        if (NOT rc EQUAL 0)
            message(FATAL_ERROR
                "Expected configure success for ${name}, got ${rc}\nstdout:\n${stdout}\nstderr:\n${stderr}")
        endif()
        set(cache_path "${build_dir}/CMakeCache.txt")
        set(config_path "${build_dir}/generated/npu-matmul-config.hpp")
        if (NOT EXISTS "${cache_path}")
            message(FATAL_ERROR "Missing CMakeCache.txt for ${name}")
        endif()
        if (NOT EXISTS "${config_path}")
            message(FATAL_ERROR "Missing generated config for ${name}")
        endif()
        expect_contains("${cache_path}" "GGML_GEMMINI_ENABLE_STRIPE_MATMUL:BOOL=${stripe}")
        expect_contains("${cache_path}" "GGML_GEMMINI_ENABLE_STRIPE_PIPELINE:BOOL=${pipeline}")
        expect_contains("${cache_path}" "GGML_GEMMINI_DEFAULT_MATMUL_MODE:STRING=${mode}")
        expect_contains("${cache_path}" "GGML_GEMMINI_ACTIVATION_QUANT:STRING=${expected_activation_quant}")
        expect_contains("${cache_path}" "GGML_GEMMINI_EXSIA_LOCAL_WORKERS:STRING=${local_workers}")
        expect_contains("${cache_path}" "GGML_GEMMINI_ENABLE_RMD:BOOL=${rmd}")
        expect_contains("${cache_path}" "GGML_GEMMINI_DEFAULT_RMD_BACKEND:STRING=${expected_rmd_backend}")
        expect_contains("${cache_path}" "GGML_GEMMINI_BLOCK_SIZE:STRING=${block_size}")
        expect_contains("${cache_path}" "GGML_GEMMINI_WEIGHT_BITS:STRING=8")
        expect_contains("${cache_path}" "GGML_GEMMINI_EXECUTION_BACKEND:STRING=HARDWARE")
        set(gemmini_link_path
            "${build_dir}/ggml/src/ggml-gemmini/CMakeFiles/ggml-gemmini.dir/link.txt")
        if(EXISTS "${gemmini_link_path}")
        endif()
        string(CONCAT retired_stripe_rows
               "GGML_GEMMINI_DEFAULT_STRIPE_" "ROWS")
        foreach(removed_option IN ITEMS
                GGML_GEMMINI_FORCE_GGML_OUTPUT
                GGML_GEMMINI_FORCE_GGML_ALL
                GGML_GEMMINI_WEIGHT_QUANT
                GGML_GEMMINI_EXSIA_SUPERBLOCKS
                GGML_GEMMINI_EXSIA_PIPELINE_SLOTS
                GGML_GEMMINI_EXSIA_OMP_THREADS_DEFAULT
                "${retired_stripe_rows}")
            expect_not_contains("${cache_path}" "${removed_option}:")
        endforeach()
        if (stripe STREQUAL "ON")
            set(stripe_value 1)
        else()
            set(stripe_value 0)
        endif()
        if (pipeline STREQUAL "ON")
            set(pipeline_value 1)
        else()
            set(pipeline_value 0)
        endif()
        mode_index("${mode}" mode_value)
        expect_contains("${config_path}" "inline constexpr bool ENABLE_STRIPE_MATMUL = ${stripe_value};")
        expect_contains("${config_path}" "inline constexpr bool ENABLE_STRIPE_PIPELINE = ${pipeline_value};")
        if (expected_activation_quant STREQUAL "EXSIA")
            set(activation_quant_value 0)
        elseif (expected_activation_quant STREQUAL "TENSOR")
            set(activation_quant_value 1)
        else()
            message(FATAL_ERROR "Unsupported test activation quant '${expected_activation_quant}'")
        endif()
        expect_contains("${config_path}" "inline constexpr int ACTIVATION_QUANT = ${activation_quant_value};")
        expect_contains("${config_path}" "inline constexpr int DEFAULT_MATMUL_MODE = ${mode_value};")
        if (expected_rmd_backend STREQUAL "CPU")
            set(rmd_backend_value 0)
        else()
            set(rmd_backend_value 1)
        endif()
        expect_contains("${config_path}" "inline constexpr int DEFAULT_RMD_BACKEND = ${rmd_backend_value};")
        return()
    endif()

    if (rc EQUAL 0)
        message(FATAL_ERROR "Expected configure failure for ${name}")
    endif()

    if (ARGC GREATER 9)
        string(CONCAT output "${stdout}" "\n" "${stderr}")
        string(FIND "${output}" "${ARGV9}" failure_needle_at)
        if (failure_needle_at EQUAL -1)
            message(FATAL_ERROR
                "Expected configure failure for ${name} to mention '${ARGV9}'\nstdout:\n${stdout}\nstderr:\n${stderr}")
        endif()
    endif()
endfunction()








function(run_option_backend_case name option backend root expect_success failure_needle)
    set(build_dir "${TEST_BINARY_ROOT}/${name}")
    file(REMOVE_RECURSE "${build_dir}")
    set(cmake_args
        -S "${TEST_SOURCE_DIR}" -B "${build_dir}"
        -DGGML_GEMMINI=ON
        -DGGML_GEMMINI_OPTION=${option}
        -DGGML_GEMMINI_EXECUTION_BACKEND=${backend}
        -DGGML_GEMMINI_ACTIVATION_BITS=8
        -DGGML_GEMMINI_WEIGHT_BITS=8
        -DGGML_GEMMINI_BLOCK_SIZE=32
            -DGGML_GEMMINI_DIM=16
        -DLLAMA_BUILD_COMMON=OFF
        -DLLAMA_BUILD_TESTS=OFF
        -DLLAMA_BUILD_TOOLS=OFF
        -DLLAMA_BUILD_EXAMPLES=OFF
        -DLLAMA_BUILD_SERVER=OFF
        -DLLAMA_CURL=OFF)
    if(TEST_GENERATOR)
        list(APPEND cmake_args -G "${TEST_GENERATOR}")
    endif()
    execute_process(
        COMMAND "${TEST_CMAKE_COMMAND}" ${cmake_args}
        RESULT_VARIABLE rc OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
    string(CONCAT output "${stdout}" "\n" "${stderr}")
    string(REGEX REPLACE "[ \t\r\n]+" " " normalized_output "${output}")

    record_configure("${name}" "${expect_success}" "${rc}" "${output}")
    if(expect_success)
        if(NOT rc EQUAL 0)
            message(FATAL_ERROR
                "Expected option/backend success for ${name}\n${output}")
        endif()
        string(TOUPPER "${option}" expected_option)
        expect_contains("${build_dir}/CMakeCache.txt"
            "GGML_GEMMINI_OPTION:STRING=${expected_option}")
        return()
    endif()

    if(rc EQUAL 0)
        message(FATAL_ERROR "Expected option/backend failure for ${name}")
    endif()
    string(FIND "${normalized_output}" "${failure_needle}" failure_at)
    if(failure_at EQUAL -1)
        message(FATAL_ERROR
            "Expected ${name} failure to mention '${failure_needle}'\n${output}")
    endif()

endfunction()

function(run_hardware_width_case name option activation_bits weight_bits mode expect_success)
    set(build_dir "${TEST_BINARY_ROOT}/${name}")
    file(REMOVE_RECURSE "${build_dir}")
    execute_process(
        COMMAND "${TEST_CMAKE_COMMAND}"
            -S "${TEST_SOURCE_DIR}" -B "${build_dir}"
            -DGGML_GEMMINI=ON
            -DGGML_GEMMINI_OPTION=${option}
            -DGGML_GEMMINI_EXECUTION_BACKEND=HARDWARE
            -DGGML_GEMMINI_ACTIVATION_BITS=${activation_bits}
            -DGGML_GEMMINI_WEIGHT_BITS=${weight_bits}
            -DGGML_GEMMINI_DEFAULT_MATMUL_MODE=${mode}
            -DGGML_GEMMINI_ENABLE_STRIPE_MATMUL=ON
            -DGGML_GEMMINI_ENABLE_STRIPE_PIPELINE=ON
            -DLLAMA_BUILD_COMMON=OFF
            -DLLAMA_BUILD_TESTS=OFF
            -DLLAMA_BUILD_TOOLS=OFF
            -DLLAMA_BUILD_EXAMPLES=OFF
            -DLLAMA_BUILD_SERVER=OFF
            -DLLAMA_CURL=OFF
        RESULT_VARIABLE rc OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
    string(CONCAT output "${stdout}" "\n" "${stderr}")
    record_configure("${name}" "${expect_success}" "${rc}" "${output}")
    if(expect_success)
        if(NOT rc EQUAL 0)
            message(FATAL_ERROR
                "Expected matched HARDWARE route ${name} to configure\n${output}")
        endif()
        return()
    endif()
    if(rc EQUAL 0)
        message(FATAL_ERROR "Mixed HARDWARE route ${name} configured successfully")
    endif()
    string(FIND "${output}"
        "Gemmini requires matched activation and weight widths" failure_at)
    if(failure_at EQUAL -1)
        message(FATAL_ERROR
            "Expected universal matched-width rejection for ${name}\n${output}")
    endif()
endfunction()

function(run_host_dim_case name dim)
    set(build_dir "${TEST_BINARY_ROOT}/${name}")
    file(REMOVE_RECURSE "${build_dir}")
    execute_process(
        COMMAND "${TEST_CMAKE_COMMAND}"
            -S "${TEST_SOURCE_DIR}" -B "${build_dir}"
            -DGGML_GEMMINI=ON
            -DGGML_GEMMINI_OPTION=CPU
            -DGGML_GEMMINI_EXECUTION_BACKEND=HARDWARE
            -DGGML_GEMMINI_DIM=${dim}
            -DGGML_GEMMINI_ACTIVATION_BITS=8
            -DGGML_GEMMINI_WEIGHT_BITS=8
            -DGGML_GEMMINI_WS_LOOP_CYCLE=1
            -DLLAMA_BUILD_COMMON=OFF
            -DLLAMA_BUILD_TESTS=OFF
            -DLLAMA_BUILD_TOOLS=OFF
            -DLLAMA_BUILD_EXAMPLES=OFF
            -DLLAMA_BUILD_SERVER=OFF
            -DLLAMA_CURL=OFF
        RESULT_VARIABLE rc OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
    record_configure("${name}" TRUE "${rc}" "${stdout}\n${stderr}")
    if(NOT rc EQUAL 0)
        message(FATAL_ERROR
            "Expected host HARDWARE DIM ${dim} to configure\n${stdout}\n${stderr}")
    endif()
    expect_contains("${build_dir}/generated/gemmini_params.h"
        "#define DIM ${dim}")
    expect_contains("${build_dir}/CMakeCache.txt"
        "GGML_GEMMINI_WS_LOOP_CYCLE:STRING=1")
    expect_contains("${build_dir}/compile_commands.json"
        "-DGEMMINI_WS_LOOP_CYCLE=1")
endfunction()

function(run_riscv_dim_mismatch_case)
    set(root "${TEST_BINARY_ROOT}/riscv-dim-mismatch")
    set(toolchain "${root}/toolchain.cmake")
    file(MAKE_DIRECTORY "${root}")
    file(WRITE "${toolchain}"
        "set(CMAKE_SYSTEM_NAME Linux)\n"
        "set(CMAKE_SYSTEM_PROCESSOR riscv64)\n")
    execute_process(
        COMMAND "${TEST_CMAKE_COMMAND}"
            -S "${TEST_SOURCE_DIR}" -B "${root}/build"
            -DCMAKE_TOOLCHAIN_FILE=${toolchain}
            -DGGML_GEMMINI=ON
            -DGGML_GEMMINI_OPTION=WS
            -DGGML_GEMMINI_EXECUTION_BACKEND=HARDWARE
            -DGGML_GEMMINI_DIM=64
            -DGGML_GEMMINI_ACTIVATION_BITS=8
            -DGGML_GEMMINI_WEIGHT_BITS=8
            -DLLAMA_BUILD_COMMON=OFF
            -DLLAMA_BUILD_TESTS=OFF
            -DLLAMA_BUILD_TOOLS=OFF
            -DLLAMA_BUILD_EXAMPLES=OFF
            -DLLAMA_BUILD_SERVER=OFF
            -DLLAMA_CURL=OFF
        RESULT_VARIABLE rc OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
    string(CONCAT output "${stdout}" "\n" "${stderr}")
    record_configure(riscv_dim_mismatch FALSE "${rc}" "${output}")
    if(rc EQUAL 0)
        message(FATAL_ERROR
            "Expected RISC-V physical DIM mismatch to fail configure")
    endif()
    string(FIND "${output}" "Physical Gemmini DIM mismatch" mismatch_at)
    string(FIND "${output}" "hardware header reports 16" hardware_at)
    if(mismatch_at EQUAL -1 OR hardware_at EQUAL -1)
        message(FATAL_ERROR
            "Expected explicit RISC-V physical DIM mismatch\n${output}")
    endif()
endfunction()


function(run_host_script_dim_contract script dim)
    set(root "${TEST_BINARY_ROOT}/${script}-dim-contract")
    set(bin "${root}/bin")
    set(log "${root}/commands.log")
    file(MAKE_DIRECTORY "${bin}")
    file(WRITE "${bin}/cmake" [=[#!/bin/bash
printf 'cmake:%s\n' "$*" >> "$CONTRACT_LOG"
]=])
    execute_process(COMMAND chmod +x "${bin}/cmake")
    execute_process(
        COMMAND "${TEST_CMAKE_COMMAND}" -E env
            "PATH=${bin}:$ENV{PATH}"
            "CONTRACT_LOG=${log}"
            "BUILD_DIR=${root}/build"
            "BUILD_JOBS=1"
            "GGML_GEMMINI_EXECUTION_BACKEND=HARDWARE"
            "GGML_GEMMINI_DIM=${dim}"
            bash "${TEST_SOURCE_DIR}/${script}"
        WORKING_DIRECTORY "${TEST_SOURCE_DIR}"
        RESULT_VARIABLE rc OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
    if(NOT rc EQUAL 0)
        message(FATAL_ERROR
            "${script} DIM contract failed: ${stdout}\n${stderr}")
    endif()
    file(READ "${log}" commands)
    string(FIND "${commands}" "-DGGML_GEMMINI_DIM=${dim}" dim_at)
    if(dim_at EQUAL -1)
        message(FATAL_ERROR
            "${script} did not forward GGML_GEMMINI_DIM=${dim}:\n${commands}")
    endif()
endfunction()

function(run_invalid_log_case name option_name option_value)
    set(build_dir "${TEST_BINARY_ROOT}/${name}")
    execute_process(
        COMMAND "${TEST_CMAKE_COMMAND}"
            -S "${TEST_SOURCE_DIR}"
            -B "${build_dir}"
            -D${option_name}=${option_value}
            -DLLAMA_BUILD_TESTS=OFF
        RESULT_VARIABLE rc
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr)
    record_configure("${name}" FALSE "${rc}" "${stdout}\n${stderr}")
    if (rc EQUAL 0)
        message(FATAL_ERROR "Expected ${option_name}=${option_value} to fail configure")
    endif()
    string(CONCAT output "${stdout}" "\n" "${stderr}")
    string(FIND "${output}" "${option_name}" option_name_at)
    if (option_name_at EQUAL -1)
        message(FATAL_ERROR "Expected invalid log failure to mention ${option_name}")
    endif()
endfunction()

if (NOT DEFINED TEST_CMAKE_COMMAND OR NOT EXISTS "${TEST_CMAKE_COMMAND}")
    message(FATAL_ERROR "TEST_CMAKE_COMMAND must point to a usable cmake binary")
endif()
if (NOT DEFINED TEST_SOURCE_DIR OR NOT EXISTS "${TEST_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "TEST_SOURCE_DIR must point to the project source dir")
endif()
if (NOT DEFINED TEST_BINARY_ROOT)
    message(FATAL_ERROR "TEST_BINARY_ROOT must be set")
endif()
if(NOT DEFINED TEST_CXX_COMPILER OR NOT EXISTS "${TEST_CXX_COMPILER}")
    message(FATAL_ERROR "TEST_CXX_COMPILER must point to a usable C++ compiler")
endif()
if(NOT DEFINED TEST_AR OR NOT EXISTS "${TEST_AR}")
    message(FATAL_ERROR "TEST_AR must point to a usable archive tool")
endif()

file(REMOVE_RECURSE "${TEST_BINARY_ROOT}")
file(MAKE_DIRECTORY "${TEST_BINARY_ROOT}")
find_package(Python3 REQUIRED COMPONENTS Interpreter)

foreach(gemmini_build_script IN ITEMS build-arm64.sh build-riscv.sh build-x86.sh)
    expect_ws_cycle_build_script("${gemmini_build_script}")
endforeach()
expect_dynamic_backend_production_script(build-arm64.sh)

run_configure_case(default_rmd_backend ON ON FULL 4 ON DEFAULT 32 TRUE)
run_configure_case(full_with_features ON ON FULL 4 ON CPU 32 TRUE)
run_configure_case(rmd_ws_default ON ON FULL 4 ON WS 32 TRUE)
run_configure_case(removed_stripe_sequential ON ON STRIPE_SEQUENTIAL 4 ON CPU 32 FALSE
    "GGML_GEMMINI_DEFAULT_MATMUL_MODE must be FULL or STRIPE_PIPELINE")
run_configure_case(stripe_pipeline_with_features ON ON STRIPE_PIPELINE 4 ON CPU 32 TRUE)
run_configure_case(exsia_local_workers_three ON ON FULL 3 ON CPU 32 TRUE)
run_configure_case(rmd_ablation ON ON FULL 4 OFF CPU 32 TRUE)
run_configure_case(block_size_64 ON ON FULL 4 ON CPU 64 TRUE)
run_configure_case(block_size_128 ON ON FULL 4 ON CPU 128 TRUE)
run_configure_case(stripe_pipeline_without_pipeline ON OFF STRIPE_PIPELINE 4 ON CPU 32 FALSE)
run_configure_case(exsia_local_workers_two ON ON FULL 2 ON CPU 32 FALSE)
foreach(invalid_rmd_backend IN ITEMS "" cpu " CPU" 0 AUTO INHERIT OS DEC)
    string(MAKE_C_IDENTIFIER "${invalid_rmd_backend}" invalid_rmd_backend_id)
    run_configure_case("invalid_rmd_backend_${invalid_rmd_backend_id}"
        ON ON FULL 4 ON "${invalid_rmd_backend}" 32 FALSE
        "GGML_GEMMINI_DEFAULT_RMD_BACKEND")
endforeach()
run_invalid_log_case(invalid_log_debug LOG_DEBUG ON)
run_invalid_log_case(invalid_log_dump LOG_DUMP YES)
run_invalid_log_case(invalid_log_dump_scale LOG_DUMP_SCALE garbage)
foreach(host_dim IN ITEMS 16 32 64)
    run_host_dim_case("host_hardware_dim${host_dim}" "${host_dim}")
endforeach()
run_riscv_dim_mismatch_case()

run_option_backend_case(option_cpu_hardware CPU HARDWARE "" TRUE "")
run_option_backend_case(option_ws_hardware ws HARDWARE "" TRUE "")
run_option_backend_case(option_os_hardware OS HARDWARE "" FALSE "Gemmini OPTION=OS is unsupported")
foreach(hardware_option IN ITEMS CPU WS)
    foreach(hardware_mode IN ITEMS FULL STRIPE_PIPELINE)
        foreach(activation_bits IN ITEMS 4 8 16)
            foreach(weight_bits IN ITEMS 4 8 16)
                if(activation_bits STREQUAL weight_bits)
                    set(expect_hardware_success TRUE)
                else()
                    set(expect_hardware_success FALSE)
                endif()
                run_hardware_width_case(
                    "hardware_${hardware_option}_${hardware_mode}_a${activation_bits}_w${weight_bits}"
                    "${hardware_option}" "${activation_bits}" "${weight_bits}"
                    "${hardware_mode}" "${expect_hardware_success}")
            endforeach()
        endforeach()
    endforeach()
endforeach()
foreach(enabled IN ITEMS ON OFF)
    execute_process(COMMAND "${TEST_CMAKE_COMMAND}"
        -S "${TEST_SOURCE_DIR}" -B "${TEST_BINARY_ROOT}/deprecated-backend-${enabled}"
        -DGGML_GEMMINI=${enabled} -DGGML_GEMMINI_EXECUTION_BACKEND=FPGA_UART
        -DLLAMA_CURL=OFF
        RESULT_VARIABLE rc OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
    record_configure("deprecated-backend-${enabled}" FALSE "${rc}" "${stdout}\n${stderr}")
    if(rc EQUAL 0 OR NOT stderr MATCHES "FPGA_UART is deprecated and unsupported")
        message(FATAL_ERROR "Deprecated backend must reject with Gemmini ${enabled}: ${stdout}\n${stderr}")
    endif()
    execute_process(COMMAND "${TEST_CMAKE_COMMAND}"
        -S "${TEST_SOURCE_DIR}" -B "${TEST_BINARY_ROOT}/invalid-header-${enabled}"
        -DGGML_GEMMINI=${enabled}
        "-DGEMMINI_SW_PATH=${TEST_BINARY_ROOT}/missing-header-tree" -DLLAMA_CURL=OFF
        -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_BLAS=OFF -DGGML_OPENMP=OFF
        RESULT_VARIABLE rc OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
    if(enabled STREQUAL "ON")
        set(expect_header_success FALSE)
    else()
        set(expect_header_success TRUE)
    endif()
    record_configure("invalid-header-${enabled}" ${expect_header_success} "${rc}" "${stdout}\n${stderr}")
    string(REGEX REPLACE "[ \t\r\n]+" " " diagnostic "${stderr}")
    if(enabled STREQUAL "ON" AND (rc EQUAL 0 OR NOT diagnostic MATCHES "Invalid explicit GEMMINI_SW_PATH"))
        message(FATAL_ERROR "Invalid explicit header path must reject with Gemmini ${enabled}: ${stdout}\n${stderr}")
    elseif(enabled STREQUAL "OFF" AND NOT rc EQUAL 0)
        message(FATAL_ERROR "Gemmini OFF must ignore its unused header path: ${stdout}\n${stderr}")
    endif()
endforeach()
run_host_script_dim_contract(build-x86.sh 64)
run_host_script_dim_contract(build-riscv.sh 16)

foreach(build_script IN ITEMS build-arm64.sh build-x86.sh build-riscv.sh)
    expect_contains("${TEST_SOURCE_DIR}/${build_script}"
        "GGML_GEMMINI_EXECUTION_BACKEND_DEFAULT")
    expect_contains("${TEST_SOURCE_DIR}/${build_script}"
        "GGML_GEMMINI_WEIGHT_BITS_DEFAULT")
    expect_contains("${TEST_SOURCE_DIR}/${build_script}"
        "GGML_GEMMINI_DIM_DEFAULT")
endforeach()

expect_contains("${TEST_SOURCE_DIR}/cmake/ggml-gemmini-options.cmake"
    "GGML_GEMMINI_EXECUTION_BACKEND")
expect_not_contains("${TEST_SOURCE_DIR}/cmake/ggml-gemmini-options.cmake"
    "ExSIA A4/Q4 and A16/Q16 remain TODO pending RMD scale integration")
run_test_registration_case(registered_linked_tests OFF ON)
run_test_registration_case(registered_dynamic_tests ON OFF)
run_test_registration_case(registered_gemmini_module_tests ON ON)
file(READ "${TEST_SOURCE_DIR}/ggml/src/ggml-gemmini/ops.cpp"
    product_route_source)
string(FIND "${product_route_source}"
    "if (product_weight_bits != GGML_GEMMINI_ACTIVATION_BITS" width_gate_at)
string(FIND "${product_route_source}"
    "args.A.allocate(args.I, args.K, GGML_GEMMINI_ACTIVATION_BITS)" allocation_at)
string(FIND "${product_route_source}"
    "quants::quantize_activation(src1, args)" quantization_at)
if(width_gate_at EQUAL -1 OR allocation_at EQUAL -1 OR
   quantization_at EQUAL -1 OR width_gate_at GREATER allocation_at OR
   width_gate_at GREATER quantization_at)
    message(FATAL_ERROR
        "Universal product width gate must precede activation allocation and quantization")
endif()
