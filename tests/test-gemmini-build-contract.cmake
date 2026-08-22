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
        expect_contains("${cache_path}" "GGML_GEMMINI_EXECUTION_BACKEND:STRING=HARDWARE")
        expect_not_contains("${cache_path}" "GGML_GEMMINI_IM2P_FRONTEND_ARCHIVE")
        expect_not_contains("${cache_path}" "GGML_GEMMINI_IM2P_SIM_ARCHIVE")
        set(gemmini_link_path
            "${build_dir}/ggml/src/ggml-gemmini/CMakeFiles/ggml-gemmini.dir/link.txt")
        if(EXISTS "${gemmini_link_path}")
            expect_not_contains("${gemmini_link_path}" "libim2p")
        endif()
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

function(make_fake_im2p_root name path_bits path_dim reported_abi reported_bits reported_dim out_var)
    if(NOT DEFINED TEST_CXX_COMPILER OR NOT EXISTS "${TEST_CXX_COMPILER}")
        message(FATAL_ERROR "TEST_CXX_COMPILER must point to a usable C++ compiler")
    endif()
    if(NOT DEFINED TEST_AR OR NOT EXISTS "${TEST_AR}")
        message(FATAL_ERROR "TEST_AR must point to a usable archive tool")
    endif()

    set(root "${TEST_BINARY_ROOT}/fake-im2p/${name}")
    set(artifact_id "a${path_bits}-w8-d${path_dim}")
    set(frontend_dir "${root}/build/lib/${artifact_id}")
    set(sim_dir "${root}/build/cargo/${artifact_id}/release")
    file(MAKE_DIRECTORY
        "${root}/frontend/include" "${root}/sim/include"
        "${frontend_dir}" "${sim_dir}")
    file(WRITE "${root}/frontend/include/im2p_gemmini_frontend.hpp" "#pragma once\n")
    file(WRITE "${root}/sim/include/im2p_sim.h" "#pragma once\n")
    set(source "${root}/identity.cpp")
    set(object "${root}/identity.o")
    file(WRITE "${source}"
        "extern \"C\" unsigned im2p_sim_abi_version(){return ${reported_abi};}\nextern \"C\" unsigned im2p_sim_activation_bits(){return ${reported_bits};}\nextern \"C\" unsigned im2p_sim_dim(){return ${reported_dim};}\n")
    execute_process(
        COMMAND "${TEST_CXX_COMPILER}" -c "${source}" -o "${object}"
        RESULT_VARIABLE compile_rc ERROR_VARIABLE compile_stderr)
    if(NOT compile_rc EQUAL 0)
        message(FATAL_ERROR "Failed to compile fake IM2P archive: ${compile_stderr}")
    endif()
    foreach(archive IN ITEMS
            "${frontend_dir}/libim2p_gemmini_frontend.a"
            "${sim_dir}/libim2p_sim.a")
        execute_process(
            COMMAND "${TEST_AR}" rcs "${archive}" "${object}"
            RESULT_VARIABLE ar_rc ERROR_VARIABLE ar_stderr)
        if(NOT ar_rc EQUAL 0)
            message(FATAL_ERROR "Failed to create fake IM2P archive: ${ar_stderr}")
        endif()
    endforeach()
    set(${out_var} "${root}" PARENT_SCOPE)
endfunction()

function(make_stale_frontend_root out_var)
    if(NOT DEFINED TEST_REAL_IM2P_ROOT OR
       NOT EXISTS "${TEST_REAL_IM2P_ROOT}/build/lib/a8-w8-d16/libim2p_gemmini_frontend.a" OR
       NOT EXISTS "${TEST_REAL_IM2P_ROOT}/build/cargo/a16-w8-d32/release/libim2p_sim.a")
        message(FATAL_ERROR "TEST_REAL_IM2P_ROOT must contain real A8/DIM16 frontend and A16/DIM32 simulator archives")
    endif()
    set(root "${TEST_BINARY_ROOT}/stale-frontend-root")
    file(MAKE_DIRECTORY
        "${root}/frontend" "${root}/sim"
        "${root}/build/lib/a16-w8-d32"
        "${root}/build/cargo/a16-w8-d32/release")
    file(COPY "${TEST_REAL_IM2P_ROOT}/frontend/include" DESTINATION "${root}/frontend")
    file(COPY "${TEST_REAL_IM2P_ROOT}/sim/include" DESTINATION "${root}/sim")
    configure_file(
        "${TEST_REAL_IM2P_ROOT}/build/lib/a8-w8-d16/libim2p_gemmini_frontend.a"
        "${root}/build/lib/a16-w8-d32/libim2p_gemmini_frontend.a" COPYONLY)
    configure_file(
        "${TEST_REAL_IM2P_ROOT}/build/cargo/a16-w8-d32/release/libim2p_sim.a"
        "${root}/build/cargo/a16-w8-d32/release/libim2p_sim.a" COPYONLY)
    set(${out_var} "${root}" PARENT_SCOPE)
endfunction()

function(run_im2p_configure_case name root bits dim backend expect_success failure_needle)
    set(build_dir "${TEST_BINARY_ROOT}/${name}")
    file(REMOVE_RECURSE "${build_dir}")
    set(cmake_args
        -S "${TEST_SOURCE_DIR}" -B "${build_dir}"
        -DGGML_GEMMINI=ON
        -DGGML_GEMMINI_EXECUTION_BACKEND=${backend}
        -DGGML_GEMMINI_ACTIVATION_BITS=${bits}
        -DIM2P_DIM=${dim}
        -DIM2P_SIM_ROOT=${root}
        -DLLAMA_BUILD_COMMON=OFF
        -DLLAMA_BUILD_TESTS=OFF
        -DLLAMA_BUILD_TOOLS=OFF
        -DLLAMA_BUILD_EXAMPLES=OFF
        -DLLAMA_BUILD_SERVER=OFF
        -DLLAMA_CURL=OFF)
    if(TEST_GENERATOR)
        list(APPEND cmake_args -G "${TEST_GENERATOR}")
    endif()
    if(TEST_GENERATOR_PLATFORM)
        list(APPEND cmake_args -A "${TEST_GENERATOR_PLATFORM}")
    endif()
    if(TEST_GENERATOR_TOOLSET)
        list(APPEND cmake_args -T "${TEST_GENERATOR_TOOLSET}")
    endif()
    execute_process(
        COMMAND "${TEST_CMAKE_COMMAND}" ${cmake_args}
        RESULT_VARIABLE rc OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
    string(CONCAT output "${stdout}" "\n" "${stderr}")
    if(expect_success)
        if(NOT rc EQUAL 0)
            message(FATAL_ERROR "Expected IM2P configure success for ${name}\n${output}")
        endif()
        set(artifact_id "a${bits}-w8-d${dim}")
        string(FIND "${output}"
            "build/lib/${artifact_id}/libim2p_gemmini_frontend.a" frontend_at)
        string(FIND "${output}"
            "build/cargo/${artifact_id}/release/libim2p_sim.a" simulator_at)
        if(frontend_at EQUAL -1 OR simulator_at EQUAL -1)
            message(FATAL_ERROR "${name} did not report the exact pair archives\n${output}")
        endif()
        return()
    endif()
    if(rc EQUAL 0)
        message(FATAL_ERROR "Expected IM2P configure failure for ${name}")
    endif()
    string(FIND "${output}" "${failure_needle}" failure_at)
    if(failure_at EQUAL -1)
        message(FATAL_ERROR
            "Expected ${name} failure to mention '${failure_needle}'\n${output}")
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
if(NOT DEFINED TEST_CXX_COMPILER OR NOT EXISTS "${TEST_CXX_COMPILER}")
    message(FATAL_ERROR "TEST_CXX_COMPILER must point to a usable C++ compiler")
endif()
if(NOT DEFINED TEST_AR OR NOT EXISTS "${TEST_AR}")
    message(FATAL_ERROR "TEST_AR must point to a usable archive tool")
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

set(matching_root "${TEST_REAL_IM2P_ROOT}")
make_fake_im2p_root(width_mismatch 16 32 2 8 32 width_mismatch_root)
make_fake_im2p_root(dim_mismatch 16 32 2 16 16 dim_mismatch_root)
make_fake_im2p_root(abi_mismatch 16 32 1 16 32 abi_mismatch_root)
set(real_pair_count 0)
foreach(real_bits IN ITEMS 4 8 16)
    foreach(real_dim IN ITEMS 16 32)
        set(real_id "a${real_bits}-w8-d${real_dim}")
        if(EXISTS "${matching_root}/build/lib/${real_id}/libim2p_gemmini_frontend.a" AND
           EXISTS "${matching_root}/build/cargo/${real_id}/release/libim2p_sim.a")
            math(EXPR real_pair_count "${real_pair_count} + 1")
            run_im2p_configure_case("im2p_matching_a${real_bits}_d${real_dim}"
                "${matching_root}" "${real_bits}" "${real_dim}" IM2P_SIM TRUE "")
        endif()
    endforeach()
endforeach()
if(real_pair_count EQUAL 0)
    message(FATAL_ERROR "Expected at least one complete real IM2P frontend/simulator pair")
endif()
run_im2p_configure_case(im2p_missing_root "" 16 32 IM2P_SIM FALSE "IM2P_SIM_ROOT is required")
run_im2p_configure_case(im2p_invalid_backend "" 16 32 simulator FALSE
    "GGML_GEMMINI_EXECUTION_BACKEND must be exactly HARDWARE or IM2P_SIM")
run_im2p_configure_case(im2p_malformed_backend "" 16 32 " IM2P_SIM" FALSE
    "GGML_GEMMINI_EXECUTION_BACKEND must be exactly HARDWARE or IM2P_SIM")
run_im2p_configure_case(im2p_invalid_width "${matching_root}" 5 32 IM2P_SIM FALSE
    "GGML_GEMMINI_ACTIVATION_BITS must be one of")
run_im2p_configure_case(im2p_invalid_dim "${matching_root}" 16 17 IM2P_SIM FALSE
    "IM2P_DIM must be 16 or 32")
run_im2p_configure_case(im2p_width_mismatch "${width_mismatch_root}" 16 32 IM2P_SIM FALSE
    "IM2P activation width mismatch: llama requests 16, archive reports 8")
run_im2p_configure_case(im2p_dim_mismatch "${dim_mismatch_root}" 16 32 IM2P_SIM FALSE
    "IM2P DIM mismatch: llama requests 32, archive reports 16")
run_im2p_configure_case(im2p_abi_mismatch "${abi_mismatch_root}" 16 32 IM2P_SIM FALSE
    "IM2P simulator ABI mismatch: llama requires 2, archive reports 1")
run_im2p_configure_case(im2p_missing_pair "${matching_root}" 8 32 IM2P_SIM FALSE
    "Missing matching IM2P a8-w8-d32 archive")
make_stale_frontend_root(stale_frontend_root)
run_im2p_configure_case(im2p_stale_frontend "${stale_frontend_root}" 16 32 IM2P_SIM FALSE
    "IM2P frontend configuration mismatch")

foreach(build_script IN ITEMS build-arm64.sh build-x86.sh build-riscv.sh)
    expect_contains("${TEST_SOURCE_DIR}/${build_script}"
        "GGML_GEMMINI_EXECUTION_BACKEND_DEFAULT")
endforeach()

expect_contains("${TEST_SOURCE_DIR}/CMakeLists.txt"
    "GGML_GEMMINI_EXECUTION_BACKEND")
expect_contains("${TEST_SOURCE_DIR}/CMakeLists.txt"
    "IM2P_SIM_ROOT")
expect_contains("${TEST_SOURCE_DIR}/ggml/src/ggml-gemmini/CMakeLists.txt"
    "GGML_GEMMINI_IM2P_FRONTEND_ARCHIVE")
expect_contains("${TEST_SOURCE_DIR}/ggml/src/ggml-gemmini/CMakeLists.txt"
    "GGML_GEMMINI_IM2P_SIM_ARCHIVE")
