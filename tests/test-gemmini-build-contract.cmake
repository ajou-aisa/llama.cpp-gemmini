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

function(mode_index mode out_var)
    set(modes FULL STRIPE_SEQUENTIAL STRIPE_PIPELINE)
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
    if (rmd_backend STREQUAL "DEFAULT")
        set(expected_rmd_backend WS)
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

    if (expect_success)
        if (NOT rc EQUAL 0)
            message(FATAL_ERROR
                "Expected configure success for ${name}, got ${rc}\nstdout:\n${stdout}\nstderr:\n${stderr}")
        endif()
        set(cache_path "${build_dir}/CMakeCache.txt")
        set(config_path "${build_dir}/generated/ggml-gemmini-matmul-config.hpp")
        if (NOT EXISTS "${cache_path}")
            message(FATAL_ERROR "Missing CMakeCache.txt for ${name}")
        endif()
        if (NOT EXISTS "${config_path}")
            message(FATAL_ERROR "Missing generated config for ${name}")
        endif()
        expect_contains("${cache_path}" "GGML_GEMMINI_ENABLE_STRIPE_MATMUL:BOOL=${stripe}")
        expect_contains("${cache_path}" "GGML_GEMMINI_ENABLE_STRIPE_PIPELINE:BOOL=${pipeline}")
        expect_contains("${cache_path}" "GGML_GEMMINI_DEFAULT_MATMUL_MODE:STRING=${mode}")
        expect_contains("${cache_path}" "GGML_GEMMINI_EXSIA_LOCAL_WORKERS:STRING=${local_workers}")
        expect_contains("${cache_path}" "GGML_GEMMINI_ENABLE_RMD:BOOL=${rmd}")
        expect_contains("${cache_path}" "GGML_GEMMINI_DEFAULT_RMD_BACKEND:STRING=${expected_rmd_backend}")
        expect_contains("${cache_path}" "GGML_GEMMINI_BLOCK_SIZE:STRING=${block_size}")
        foreach(removed_option IN ITEMS
                GGML_GEMMINI_FORCE_GGML_OUTPUT
                GGML_GEMMINI_FORCE_GGML_ALL
                GGML_GEMMINI_WEIGHT_QUANT
                GGML_GEMMINI_EXSIA_SUPERBLOCKS
                GGML_GEMMINI_EXSIA_PIPELINE_SLOTS
                GGML_GEMMINI_EXSIA_OMP_THREADS_DEFAULT)
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

file(REMOVE_RECURSE "${TEST_BINARY_ROOT}")
file(MAKE_DIRECTORY "${TEST_BINARY_ROOT}")

run_configure_case(default_rmd_backend ON ON FULL 4 ON DEFAULT 32 TRUE)
run_configure_case(full_with_features ON ON FULL 4 ON CPU 32 TRUE)
run_configure_case(rmd_ws_default ON ON FULL 4 ON WS 32 TRUE)
run_configure_case(stripe_sequential_without_pipeline ON OFF STRIPE_SEQUENTIAL 4 ON CPU 32 TRUE)
run_configure_case(stripe_pipeline_with_features ON ON STRIPE_PIPELINE 4 ON CPU 32 TRUE)
run_configure_case(exsia_local_workers_three ON ON FULL 3 ON CPU 32 TRUE)
run_configure_case(rmd_ablation ON ON FULL 4 OFF CPU 32 TRUE)
run_configure_case(block_size_64 ON ON FULL 4 ON CPU 64 TRUE)
run_configure_case(block_size_128 ON ON FULL 4 ON CPU 128 TRUE)
run_configure_case(stripe_sequential_without_stripe OFF OFF STRIPE_SEQUENTIAL 4 ON CPU 32 FALSE)
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

foreach(build_script IN ITEMS build-arm64.sh build-x86.sh build-riscv.sh)
    expect_contains("${TEST_SOURCE_DIR}/${build_script}"
        "GGML_GEMMINI_DEFAULT_RMD_BACKEND:-WS")
endforeach()
