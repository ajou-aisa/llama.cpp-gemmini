function(expect_contains path needle)
    file(READ "${path}" content)
    string(FIND "${content}" "${needle}" found_at)
    if (found_at EQUAL -1)
        message(FATAL_ERROR "Expected '${needle}' in ${path}")
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

function(run_configure_case name stripe pipeline mode local_workers expect_success)
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
        -DLLAMA_BUILD_TESTS=OFF)
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
        return()
    endif()

    if (rc EQUAL 0)
        message(FATAL_ERROR "Expected configure failure for ${name}")
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

run_configure_case(full_with_features ON ON FULL 4 TRUE)
run_configure_case(stripe_sequential_without_pipeline ON OFF STRIPE_SEQUENTIAL 4 TRUE)
run_configure_case(stripe_pipeline_with_features ON ON STRIPE_PIPELINE 4 TRUE)
run_configure_case(exsia_local_workers_three ON ON FULL 3 TRUE)
run_configure_case(stripe_sequential_without_stripe OFF OFF STRIPE_SEQUENTIAL 4 FALSE)
run_configure_case(stripe_pipeline_without_pipeline ON OFF STRIPE_PIPELINE 4 FALSE)
run_configure_case(exsia_local_workers_two ON ON FULL 2 FALSE)
