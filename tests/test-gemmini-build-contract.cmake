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

function(read_registered_test_executable_block path out_var)
    file(STRINGS "${path}" lines)
    set(in_block FALSE)
    set(block "")
    foreach(line IN LISTS lines)
        string(STRIP "${line}" line_trimmed)
        if (NOT in_block)
            if (line_trimmed STREQUAL "set(_GEMMINI_REGISTERED_TEST_EXECUTABLES")
                set(in_block TRUE)
                string(APPEND block "${line_trimmed}\n")
            endif()
        else()
            string(APPEND block "${line_trimmed}\n")
            if (line_trimmed MATCHES "\\)$")
                set(${out_var} "${block}" PARENT_SCOPE)
                return()
            endif()
        endif()
    endforeach()
    message(FATAL_ERROR
        "Missing exact _GEMMINI_REGISTERED_TEST_EXECUTABLES block in ${path}")
endfunction()

function(read_cycle_sink_creation_block path out_var)
    file(STRINGS "${path}" lines)
    set(in_block FALSE)
    set(block "")
    foreach(line IN LISTS lines)
        string(STRIP "${line}" line_trimmed)
        if (NOT in_block)
            if (line_trimmed STREQUAL "if (LOG_CYCLE AND GGML_CPU_CYCLE_LOG AND TARGET ggml-cpu)")
                set(in_block TRUE)
                string(APPEND block "${line_trimmed}\n")
            endif()
        else()
            string(APPEND block "${line_trimmed}\n")
            if (line_trimmed STREQUAL "endif()")
                set(${out_var} "${block}" PARENT_SCOPE)
                return()
            endif()
        endif()
    endforeach()
    message(FATAL_ERROR
        "Missing exact CPU cycle sink creation block in ${path}")
endfunction()

function(read_cycle_sink_append_block path out_var)
    file(STRINGS "${path}" lines)
    set(in_block FALSE)
    set(block "")
    foreach(line IN LISTS lines)
        string(STRIP "${line}" line_trimmed)
        if (NOT in_block)
            if (line_trimmed STREQUAL "if (_GEMMINI_REGISTERED_TEST_EXECUTABLES_APPEND_CPU_SINK)")
                set(in_block TRUE)
                string(APPEND block "${line_trimmed}\n")
            endif()
        else()
            string(APPEND block "${line_trimmed}\n")
            if (line_trimmed STREQUAL "endif()")
                set(${out_var} "${block}" PARENT_SCOPE)
                return()
            endif()
        endif()
    endforeach()
    message(FATAL_ERROR
        "Missing exact CPU cycle sink append block in ${path}")
endfunction()

function(expect_registered_test_executable path target)
    read_registered_test_executable_block("${path}" block)
    string(REPLACE "\n" ";" block_lines "${block}")
    foreach(line IN LISTS block_lines)
        string(STRIP "${line}" stripped_line)
        if (stripped_line STREQUAL "${target}")
            return()
        endif()
    endforeach()
    message(FATAL_ERROR
        "Expected '${target}' in exact _GEMMINI_REGISTERED_TEST_EXECUTABLES block in ${path}")
endfunction()

function(expect_cycle_sink_append path target)
    read_cycle_sink_creation_block("${path}" creation_block)
    string(REPLACE "\n" ";" creation_lines "${creation_block}")
    set(found_flag FALSE)
    foreach(line IN LISTS creation_lines)
        string(STRIP "${line}" stripped_line)
        if (stripped_line STREQUAL "set(_GEMMINI_REGISTERED_TEST_EXECUTABLES_APPEND_CPU_SINK TRUE)")
            set(found_flag TRUE)
            break()
        endif()
    endforeach()
    if (NOT found_flag)
        message(FATAL_ERROR
            "Expected CPU sink append flag in exact creation block in ${path}")
    endif()

    read_cycle_sink_append_block("${path}" append_block)
    string(REPLACE "\n" ";" append_lines "${append_block}")
    set(found_append FALSE)
    set(found_target FALSE)
    foreach(line IN LISTS append_lines)
        string(STRIP "${line}" stripped_line)
        if (stripped_line STREQUAL "list(APPEND _GEMMINI_REGISTERED_TEST_EXECUTABLES")
            set(found_append TRUE)
        elseif (stripped_line STREQUAL "${target}")
            set(found_target TRUE)
        endif()
    endforeach()
    if (NOT found_append OR NOT found_target)
        message(FATAL_ERROR
            "Expected '${target}' in exact CPU cycle sink append block in ${path}")
    endif()
endfunction()

function(expect_ws_cycle_build_script script_name)
    set(script_path "${TEST_SOURCE_DIR}/${script_name}")
    if(NOT EXISTS "${script_path}")
        message(FATAL_ERROR "Missing production build script ${script_path}")
    endif()
    expect_contains("${script_path}"
        "GGML_GEMMINI_WS_LOOP_CYCLE_DEFAULT=\${GGML_GEMMINI_WS_LOOP_CYCLE:-0}")
    expect_contains("${script_path}"
        "-DGGML_GEMMINI_WS_LOOP_CYCLE=\"\${GGML_GEMMINI_WS_LOOP_CYCLE_DEFAULT}\"")
    string(CONCAT deprecated_option "GGML_GEMMINI_WS_LOOP_" "DE" "BUG")
    expect_not_contains("${script_path}" "${deprecated_option}")
endfunction()

function(expect_dynamic_backend_production_script script_name)
    set(script_path "${TEST_SOURCE_DIR}/${script_name}")
    expect_contains("${script_path}" "-DGGML_BACKEND_DL=ON")
    expect_contains("${script_path}" "-DLLAMA_BUILD_TESTS=OFF")
endfunction()

if (DEFINED REPO_ROOT)
    if (NOT EXISTS "${REPO_ROOT}/tests/CMakeLists.txt")
        message(FATAL_ERROR "REPO_ROOT must contain tests/CMakeLists.txt")
    endif()
    expect_registered_test_executable("${REPO_ROOT}/tests/CMakeLists.txt"
        test-gemmini-log-boundary)
    expect_cycle_sink_append("${REPO_ROOT}/tests/CMakeLists.txt"
        test-ggml-cpu-cycle-sink)
    return()
endif()

foreach(gemmini_build_script IN ITEMS build-arm64.sh build-riscv.sh build-x86.sh)
    expect_ws_cycle_build_script("${gemmini_build_script}")
endforeach()
expect_dynamic_backend_production_script(build-arm64.sh)

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
        expect_contains("${cache_path}" "GGML_GEMMINI_ACTIVATION_QUANT:STRING=${expected_activation_quant}")
        expect_contains("${cache_path}" "GGML_GEMMINI_EXSIA_LOCAL_WORKERS:STRING=${local_workers}")
        expect_contains("${cache_path}" "GGML_GEMMINI_ENABLE_RMD:BOOL=${rmd}")
        expect_contains("${cache_path}" "GGML_GEMMINI_DEFAULT_RMD_BACKEND:STRING=${expected_rmd_backend}")
        expect_contains("${cache_path}" "GGML_GEMMINI_BLOCK_SIZE:STRING=${block_size}")
        expect_contains("${cache_path}" "GGML_GEMMINI_WEIGHT_BITS:STRING=8")
        expect_contains("${cache_path}" "GGML_GEMMINI_EXECUTION_BACKEND:STRING=HARDWARE")
        expect_not_contains("${cache_path}" "GGML_GEMMINI_IM2P_FRONTEND_ARCHIVE")
        expect_not_contains("${cache_path}" "GGML_GEMMINI_IM2P_SIM_ARCHIVE")
        set(gemmini_link_path
            "${build_dir}/ggml/src/ggml-gemmini/CMakeFiles/ggml-gemmini.dir/link.txt")
        if(EXISTS "${gemmini_link_path}")
            expect_not_contains("${gemmini_link_path}" "libim2p")
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

function(make_fake_im2p_root name path_activation_bits path_weight_bits path_dim
         reported_abi reported_activation_bits reported_weight_bits reported_dim
         out_var)
    if(NOT DEFINED TEST_CXX_COMPILER OR NOT EXISTS "${TEST_CXX_COMPILER}")
        message(FATAL_ERROR "TEST_CXX_COMPILER must point to a usable C++ compiler")
    endif()
    if(NOT DEFINED TEST_AR OR NOT EXISTS "${TEST_AR}")
        message(FATAL_ERROR "TEST_AR must point to a usable archive tool")
    endif()

    set(root "${TEST_BINARY_ROOT}/fake-im2p/${name}")
    set(artifact_id
        "a${path_activation_bits}-w${path_weight_bits}-d${path_dim}")
    set(frontend_dir "${root}/build/lib/${artifact_id}")
    set(sim_dir "${root}/build/cargo/${artifact_id}/release")
    file(MAKE_DIRECTORY
        "${root}/frontend/include" "${root}/sim/include"
        "${frontend_dir}" "${sim_dir}")
    file(WRITE "${root}/frontend/include/im2p_gemmini_frontend.hpp" "#pragma once\n")
    file(WRITE "${root}/sim/include/im2p_sim.h" "#pragma once\n")
    set(source "${root}/identity.cpp")
    set(object "${root}/identity.o")
    if(reported_activation_bits STREQUAL "16")
        set(reported_activation_storage_bytes 2)
    else()
        set(reported_activation_storage_bytes 1)
    endif()
    if(reported_weight_bits STREQUAL "16")
        set(reported_weight_storage_bytes 2)
    else()
        set(reported_weight_storage_bytes 1)
    endif()
    file(WRITE "${source}"
        "extern \"C\" unsigned im2p_sim_abi_version(){return ${reported_abi};}\nextern \"C\" unsigned im2p_sim_activation_bits(){return ${reported_activation_bits};}\nextern \"C\" unsigned im2p_sim_activation_storage_bytes(){return ${reported_activation_storage_bytes};}\nextern \"C\" unsigned im2p_sim_weight_bits(){return ${reported_weight_bits};}\nextern \"C\" unsigned im2p_sim_weight_storage_bytes(){return ${reported_weight_storage_bytes};}\nextern \"C\" unsigned im2p_sim_dim(){return ${reported_dim};}\n")
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

function(make_matching_fake_im2p_root out_var)
    set(root "${TEST_BINARY_ROOT}/fake-im2p/matching-a8-w8-d16")
    file(MAKE_DIRECTORY
        "${root}/frontend" "${root}/sim"
        "${root}/build/lib/a8-w8-d16"
        "${root}/build/cargo/a8-w8-d16/release")
    file(COPY "${TEST_REAL_IM2P_ROOT}/frontend/include" DESTINATION "${root}/frontend")
    file(COPY "${TEST_REAL_IM2P_ROOT}/sim/include" DESTINATION "${root}/sim")
    configure_file(
        "${TEST_REAL_IM2P_ROOT}/build/lib/a8-w8-d16/libim2p_gemmini_frontend.a"
        "${root}/build/lib/a8-w8-d16/libim2p_gemmini_frontend.a" COPYONLY)
    configure_file(
        "${TEST_REAL_IM2P_ROOT}/build/cargo/a8-w8-d16/release/libim2p_sim.a"
        "${root}/build/cargo/a8-w8-d16/release/libim2p_sim.a" COPYONLY)
    set(${out_var} "${root}" PARENT_SCOPE)
endfunction()

function(make_stale_frontend_root out_var)
    if(NOT DEFINED TEST_REAL_IM2P_ROOT OR
       NOT EXISTS "${TEST_REAL_IM2P_ROOT}/build/lib/a8-w8-d16/libim2p_gemmini_frontend.a" OR
       NOT EXISTS "${TEST_REAL_IM2P_ROOT}/build/cargo/a8-w8-d32/release/libim2p_sim.a")
        message(FATAL_ERROR "TEST_REAL_IM2P_ROOT must contain real A8/DIM16 frontend and A8/DIM32 simulator archives")
    endif()
    set(root "${TEST_BINARY_ROOT}/stale-frontend-root")
    file(MAKE_DIRECTORY
        "${root}/frontend" "${root}/sim"
        "${root}/build/lib/a8-w8-d32"
        "${root}/build/cargo/a8-w8-d32/release")
    file(COPY "${TEST_REAL_IM2P_ROOT}/frontend/include" DESTINATION "${root}/frontend")
    file(COPY "${TEST_REAL_IM2P_ROOT}/sim/include" DESTINATION "${root}/sim")
    configure_file(
        "${TEST_REAL_IM2P_ROOT}/build/lib/a8-w8-d16/libim2p_gemmini_frontend.a"
        "${root}/build/lib/a8-w8-d32/libim2p_gemmini_frontend.a" COPYONLY)
    configure_file(
        "${TEST_REAL_IM2P_ROOT}/build/cargo/a8-w8-d32/release/libim2p_sim.a"
        "${root}/build/cargo/a8-w8-d32/release/libim2p_sim.a" COPYONLY)
    set(${out_var} "${root}" PARENT_SCOPE)
endfunction()

function(make_extended_poll_stale_simulator_root out_var)
    if(NOT DEFINED TEST_REAL_IM2P_ROOT OR
       NOT EXISTS "${TEST_REAL_IM2P_ROOT}/build/lib/a8-w8-d16/libim2p_gemmini_frontend.a")
        message(FATAL_ERROR "TEST_REAL_IM2P_ROOT must contain the fresh A8/W8/DIM16 frontend archive")
    endif()
    set(root "${TEST_BINARY_ROOT}/extended-poll-stale-simulator-root")
    file(MAKE_DIRECTORY
        "${root}/frontend/include" "${root}/sim/include"
        "${root}/build/lib/a8-w8-d16"
        "${root}/build/cargo/a8-w8-d16/release")
    file(COPY "${TEST_REAL_IM2P_ROOT}/frontend/include/" DESTINATION "${root}/frontend/include")
    file(COPY "${TEST_REAL_IM2P_ROOT}/sim/include/" DESTINATION "${root}/sim/include")
    configure_file(
        "${TEST_REAL_IM2P_ROOT}/build/lib/a8-w8-d16/libim2p_gemmini_frontend.a"
        "${root}/build/lib/a8-w8-d16/libim2p_gemmini_frontend.a" COPYONLY)
    set(source "${root}/old-simulator.cpp")
    set(object "${root}/old-simulator.o")
    file(WRITE "${source}" [=[
#include "im2p_sim.h"
extern "C" {
im2p_sim_t *im2p_sim_create(void) { return reinterpret_cast<im2p_sim_t *>(1); }
void im2p_sim_destroy(im2p_sim_t *) {}
uint32_t im2p_sim_abi_version(void) { return 4; }
uint32_t im2p_sim_activation_bits(void) { return 8; }
uint32_t im2p_sim_activation_storage_bytes(void) { return 1; }
uint32_t im2p_sim_weight_bits(void) { return 8; }
uint32_t im2p_sim_weight_storage_bytes(void) { return 1; }
uint32_t im2p_sim_dim(void) { return 16; }
int im2p_execute_matmul_extended(im2p_sim_t *, const im2p_matmul_desc_t *, im2p_work_stats_extended_t *) { return IM2P_ERROR; }
int im2p_begin_striped_matmul(im2p_sim_t *, const im2p_stripe_work_desc_t *, im2p_stream_t **) { return IM2P_ERROR; }
int im2p_publish_stripe(im2p_stream_t *, const im2p_activation_stripe_t *) { return IM2P_INVALID_LAYOUT; }
int im2p_progress_stream(im2p_stream_t *, uint64_t) { return IM2P_ERROR; }
uint64_t im2p_stream_progress_count(const im2p_stream_t *) { return 0; }
int im2p_finish_stream_extended(im2p_stream_t *, im2p_work_stats_extended_t *) { return IM2P_ERROR; }
void im2p_destroy_stream(im2p_stream_t *) {}
}
]=])
    execute_process(
        COMMAND "${TEST_CXX_COMPILER}" -std=c++20
            -I "${root}/sim/include" -c "${source}" -o "${object}"
        RESULT_VARIABLE compile_rc ERROR_VARIABLE compile_stderr)
    if(NOT compile_rc EQUAL 0)
        message(FATAL_ERROR "Failed to compile stale simulator: ${compile_stderr}")
    endif()
    execute_process(
        COMMAND "${TEST_AR}" rcs
            "${root}/build/cargo/a8-w8-d16/release/libim2p_sim.a" "${object}"
        RESULT_VARIABLE ar_rc ERROR_VARIABLE ar_stderr)
    if(NOT ar_rc EQUAL 0)
        message(FATAL_ERROR "Failed to archive stale simulator: ${ar_stderr}")
    endif()
    set(${out_var} "${root}" PARENT_SCOPE)
endfunction()

function(make_abi_stale_frontend_root out_var)
    if(NOT DEFINED TEST_REAL_IM2P_ROOT OR
       NOT EXISTS "${TEST_REAL_IM2P_ROOT}/build/cargo/a8-w8-d16/release/libim2p_sim.a")
        message(FATAL_ERROR "TEST_REAL_IM2P_ROOT must contain the real A8/W8/DIM16 simulator archive")
    endif()
    set(root "${TEST_BINARY_ROOT}/abi-stale-frontend-root")
    file(MAKE_DIRECTORY
        "${root}/frontend/include" "${root}/sim/include"
        "${root}/build/lib/a8-w8-d16"
        "${root}/build/cargo/a8-w8-d16/release")
    file(COPY "${TEST_REAL_IM2P_ROOT}/frontend/include/" DESTINATION "${root}/frontend/include")
    file(COPY "${TEST_REAL_IM2P_ROOT}/sim/include/" DESTINATION "${root}/sim/include")
    configure_file(
        "${TEST_REAL_IM2P_ROOT}/build/cargo/a8-w8-d16/release/libim2p_sim.a"
        "${root}/build/cargo/a8-w8-d16/release/libim2p_sim.a" COPYONLY)
    set(source "${root}/stale-identity.cpp")
    set(object "${root}/stale-identity.o")
    file(WRITE "${source}" [=[
#include <cstdint>
namespace im2p::gemmini {
struct ArgsLayoutFingerprint {
    std::uint64_t size, native_weight_bytes, col_stride_f_out,
        stride_f_out, tile_i;
};
std::uint32_t compiled_activation_bits() noexcept { return 8; }
std::uint32_t compiled_weight_bits() noexcept { return 8; }
std::uint32_t compiled_dim() noexcept { return 16; }
ArgsLayoutFingerprint compiled_args_layout_fingerprint() noexcept { return {}; }
}
]=])
    execute_process(
        COMMAND "${TEST_CXX_COMPILER}" -std=c++20 -c "${source}" -o "${object}"
        RESULT_VARIABLE compile_rc ERROR_VARIABLE compile_stderr)
    if(NOT compile_rc EQUAL 0)
        message(FATAL_ERROR "Failed to compile ABI-stale frontend: ${compile_stderr}")
    endif()
    execute_process(
        COMMAND "${TEST_AR}" rcs
            "${root}/build/lib/a8-w8-d16/libim2p_gemmini_frontend.a" "${object}"
        RESULT_VARIABLE ar_rc ERROR_VARIABLE ar_stderr)
    if(NOT ar_rc EQUAL 0)
        message(FATAL_ERROR "Failed to archive ABI-stale frontend: ${ar_stderr}")
    endif()
    set(${out_var} "${root}" PARENT_SCOPE)
endfunction()

function(run_im2p_configure_case name root activation_bits weight_bits dim
         backend expect_success failure_needle)
    set(build_dir "${TEST_BINARY_ROOT}/${name}")
    file(REMOVE_RECURSE "${build_dir}")
    set(cmake_args
        -S "${TEST_SOURCE_DIR}" -B "${build_dir}"
        -DGGML_GEMMINI=ON
        -DGGML_GEMMINI_OPTION=WS
        -DGGML_GEMMINI_EXECUTION_BACKEND=${backend}
        -DGGML_GEMMINI_ACTIVATION_BITS=${activation_bits}
        -DGGML_GEMMINI_WEIGHT_BITS=${weight_bits}
        -DGGML_GEMMINI_DIM=${dim}
        -DIM2P_SIM_ROOT=${root}
        -DLLAMA_BUILD_COMMON=OFF
        -DLLAMA_BUILD_TESTS=OFF
        -DLLAMA_BUILD_TOOLS=OFF
        -DLLAMA_BUILD_EXAMPLES=OFF
        -DLLAMA_BUILD_SERVER=OFF
        -DLLAMA_CURL=OFF)
    if(ARGC GREATER 8)
        set(block_size "${ARGV8}")
    else()
        set(block_size 32)
    endif()
    list(APPEND cmake_args -DGGML_GEMMINI_BLOCK_SIZE=${block_size})
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
        set(artifact_id "a${activation_bits}-w${weight_bits}-d${dim}")
        string(FIND "${output}"
            "build/lib/${artifact_id}/libim2p_gemmini_frontend.a" frontend_at)
        string(FIND "${output}"
            "build/cargo/${artifact_id}/release/libim2p_sim.a" simulator_at)
        if(frontend_at EQUAL -1 OR simulator_at EQUAL -1)
            message(FATAL_ERROR "${name} did not report the exact pair archives\n${output}")
        endif()
        expect_contains("${build_dir}/generated/gemmini_params.h" "#define DIM ${dim}")
        expect_contains("${build_dir}/CMakeCache.txt"
            "GGML_GEMMINI_BLOCK_SIZE:STRING=${block_size}")
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
    execute_process(
        COMMAND "${TEST_CMAKE_COMMAND}" ${cmake_args}
        RESULT_VARIABLE rc OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
    string(CONCAT output "${stdout}" "\n" "${stderr}")
    string(REGEX REPLACE "[ \t\r\n]+" " " normalized_output "${output}")

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

    set(artifact_probe_count 0)
    set(runtime_dispatch_count 0)
    if(EXISTS "${build_dir}/im2p-build-contract-probe.cpp")
        math(EXPR artifact_probe_count "${artifact_probe_count} + 1")
    endif()
    if(EXISTS "${build_dir}/im2p-frontend-pair-probe.cpp")
        math(EXPR runtime_dispatch_count "${runtime_dispatch_count} + 1")
    endif()
    if(NOT artifact_probe_count EQUAL 0 OR NOT runtime_dispatch_count EQUAL 0)
        message(FATAL_ERROR
            "${name} reached backend probes: artifact=${artifact_probe_count}, dispatch=${runtime_dispatch_count}")
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
    foreach(probe IN ITEMS im2p-build-contract-probe.cpp im2p-frontend-pair-probe.cpp)
        if(EXISTS "${build_dir}/${probe}")
            message(FATAL_ERROR "${name} reached forbidden artifact probe ${probe}")
        endif()
    endforeach()
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
            -DLLAMA_BUILD_COMMON=OFF
            -DLLAMA_BUILD_TESTS=OFF
            -DLLAMA_BUILD_TOOLS=OFF
            -DLLAMA_BUILD_EXAMPLES=OFF
            -DLLAMA_BUILD_SERVER=OFF
            -DLLAMA_CURL=OFF
        RESULT_VARIABLE rc OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
    if(NOT rc EQUAL 0)
        message(FATAL_ERROR
            "Expected host HARDWARE DIM ${dim} to configure\n${stdout}\n${stderr}")
    endif()
    expect_contains("${build_dir}/generated/gemmini_params.h"
        "#define DIM ${dim}")
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

function(run_arm64_bootstrap_contract)
    set(root "${TEST_BINARY_ROOT}/arm64-bootstrap-contract")
    set(bin "${root}/bin")
    set(log "${root}/commands.log")
    file(MAKE_DIRECTORY "${bin}")
    file(WRITE "${bin}/make" [=[#!/bin/bash
printf 'make:%s\n' "$*" >> "$CONTRACT_LOG"
]=])
    file(WRITE "${bin}/cargo" [=[#!/bin/bash
printf 'cargo:repo=%s build=%s a=%s w=%s d=%s target=%s args=%s\n' \
    "$IM2P_REPO_ROOT" "$IM2P_BUILD_DIR" "$IM2P_ACTIVATION_BITS" \
    "$IM2P_WEIGHT_BITS" "$IM2P_DIM" "$CARGO_TARGET_DIR" "$*" >> "$CONTRACT_LOG"
]=])
    file(WRITE "${bin}/cmake" [=[#!/bin/bash
printf 'cmake:%s\n' "$*" >> "$CONTRACT_LOG"
]=])
    execute_process(COMMAND chmod +x "${bin}/make" "${bin}/cargo" "${bin}/cmake")
    execute_process(
        COMMAND "${TEST_CMAKE_COMMAND}" -E env
            "PATH=${bin}:$ENV{PATH}"
            "CONTRACT_LOG=${log}"
            "BUILD_DIR=${root}/llama-build"
            "BUILD_JOBS=1"
            "IM2P_SIM_ROOT=${TEST_REAL_IM2P_ROOT}"
            "LOG_CYCLE=0"
            "GGML_GEMMINI_OPTION=WS"
            "GGML_GEMMINI_EXECUTION_BACKEND=IM2P_SIM"
            "GGML_GEMMINI_DIM=64"
            bash "${TEST_SOURCE_DIR}/build-arm64.sh"
        WORKING_DIRECTORY "${TEST_SOURCE_DIR}"
        RESULT_VARIABLE rc OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
    if(NOT rc EQUAL 0)
        message(FATAL_ERROR "build-arm64 bootstrap contract failed: ${stdout}\n${stderr}")
    endif()
    file(READ "${log}" commands)
    string(FIND "${commands}"
        "GEMMINI_FRONTEND_BLOCK_SIZE=32" frontend_block_size_at)
    string(FIND "${commands}" "gemmini-frontend-real-lib" cache_target_at)
    string(FIND "${commands}" "cargo:" direct_cargo_at)
    string(FIND "${commands}" "cmake:-B" configure_at)
    string(FIND "${commands}" "-DGGML_GEMMINI_OPTION=WS" option_at)
    string(FIND "${commands}"
        "-DGGML_GEMMINI_EXECUTION_BACKEND=IM2P_SIM" backend_at)
    string(FIND "${commands}" "-DGGML_GEMMINI_DIM=64" dim_at)
    string(FIND "${commands}" "-DGGML_GEMMINI_BLOCK_SIZE=32" cmake_block_size_at)
    string(FIND "${commands}" "cmake:--build" build_at)
    if(frontend_block_size_at EQUAL -1 OR cache_target_at EQUAL -1 OR
       NOT direct_cargo_at EQUAL -1 OR
       configure_at EQUAL -1 OR option_at EQUAL -1 OR
       backend_at EQUAL -1 OR
       dim_at EQUAL -1 OR
       cmake_block_size_at EQUAL -1 OR build_at EQUAL -1 OR
       NOT cache_target_at LESS configure_at OR NOT configure_at LESS build_at)
        message(FATAL_ERROR
            "build-arm64 WS/IM2P DIM64 overrides must delegate matching B32/DIM64 cache selection before configure/build without direct Cargo orchestration:\n${commands}")
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

set(matching_root "${TEST_REAL_IM2P_ROOT}")
make_matching_fake_im2p_root(option_backend_fake_root)
set(legal_pairs_text
    "legal combinations are CPU+HARDWARE, WS+HARDWARE, WS+IM2P_SIM")
run_option_backend_case(option_cpu_hardware CPU HARDWARE "" TRUE "")
run_option_backend_case(option_ws_hardware ws HARDWARE "" TRUE "")
run_option_backend_case(option_ws_im2p WS IM2P_SIM "${option_backend_fake_root}" TRUE "")
run_option_backend_case(option_os_hardware OS HARDWARE "" FALSE
    "Illegal Gemmini option/backend combination: OS+HARDWARE; ${legal_pairs_text}")
run_option_backend_case(option_os_im2p OS IM2P_SIM "${option_backend_fake_root}" FALSE
    "Illegal Gemmini option/backend combination: OS+IM2P_SIM; ${legal_pairs_text}")
run_option_backend_case(option_cpu_im2p CPU IM2P_SIM "${option_backend_fake_root}" FALSE
    "Illegal Gemmini option/backend combination: CPU+IM2P_SIM; ${legal_pairs_text}")
run_option_backend_case(option_malformed_im2p " WS" IM2P_SIM "${matching_root}" FALSE
    "GGML_GEMMINI_OPTION must be one of CPU|OS|WS")
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
make_fake_im2p_root(width_mismatch 16 16 32 4 8 16 32
    width_mismatch_root)
make_fake_im2p_root(weight_mismatch 16 16 32 4 16 8 32
    weight_mismatch_root)
make_fake_im2p_root(dim_mismatch 16 16 32 4 16 16 16
    dim_mismatch_root)
make_fake_im2p_root(abi_mismatch 16 16 32 3 16 16 32
    abi_mismatch_root)
foreach(real_dim IN ITEMS 16 32 64)
    set(real_id "a8-w8-d${real_dim}")
    if(NOT EXISTS "${matching_root}/build/lib/${real_id}/libim2p_gemmini_frontend.a" OR
       NOT EXISTS "${matching_root}/build/cargo/${real_id}/release/libim2p_sim.a")
        message(FATAL_ERROR "Expected complete real IM2P ${real_id} frontend/simulator pair")
    endif()
    run_im2p_configure_case("im2p_matching_a8_d${real_dim}"
        "${matching_root}" 8 8 "${real_dim}" IM2P_SIM TRUE "")
endforeach()
foreach(matched_width IN ITEMS 4 16)
    set(real_id "a${matched_width}-w${matched_width}-d16")
    if(NOT EXISTS "${matching_root}/build/lib/${real_id}/libim2p_gemmini_frontend.a" OR
       NOT EXISTS "${matching_root}/build/cargo/${real_id}/release/libim2p_sim.a")
        message(FATAL_ERROR "Expected complete real IM2P ${real_id} frontend/simulator pair")
    endif()
    run_im2p_configure_case("im2p_matching_a${matched_width}_w${matched_width}_d16"
        "${matching_root}" "${matched_width}" "${matched_width}" 16
        IM2P_SIM TRUE "")
endforeach()
run_im2p_configure_case(im2p_dim64_block32 "" 8 8 64 IM2P_SIM FALSE
    "IM2P_SIM_ROOT is required" 32)
run_im2p_configure_case(im2p_missing_root "" 8 8 32 IM2P_SIM FALSE "IM2P_SIM_ROOT is required")
run_im2p_configure_case(im2p_invalid_backend "" 8 8 32 simulator FALSE
    "GGML_GEMMINI_EXECUTION_BACKEND must be exactly HARDWARE or IM2P_SIM")
run_im2p_configure_case(im2p_malformed_backend "" 8 8 32 " IM2P_SIM" FALSE
    "GGML_GEMMINI_EXECUTION_BACKEND must be exactly HARDWARE or IM2P_SIM")
run_im2p_configure_case(im2p_invalid_width "${matching_root}" 5 8 32 IM2P_SIM FALSE
    "GGML_GEMMINI_ACTIVATION_BITS must be one of")
run_im2p_configure_case(im2p_invalid_weight_width "${matching_root}" 8 5 32 IM2P_SIM FALSE
    "GGML_GEMMINI_WEIGHT_BITS must be one of")
run_im2p_configure_case(im2p_invalid_dim "${matching_root}" 8 8 17 IM2P_SIM FALSE
    "GGML_GEMMINI_DIM must be 16, 32, or 64")
run_im2p_configure_case(im2p_width_mismatch "${width_mismatch_root}" 16 16 32 IM2P_SIM FALSE
    "IM2P activation width mismatch: llama requests 16, archive reports 8")
run_im2p_configure_case(im2p_mixed_a4_w8 "${matching_root}" 4 8 16 IM2P_SIM FALSE
    "Gemmini requires matched activation and weight widths")
run_im2p_configure_case(im2p_mixed_a8_w4 "${matching_root}" 8 4 16 IM2P_SIM FALSE
    "Gemmini requires matched activation and weight widths")
run_im2p_configure_case(im2p_weight_mismatch "${weight_mismatch_root}" 16 16 32 IM2P_SIM FALSE
    "IM2P weight width mismatch: llama requests 16, archive reports 8")
run_im2p_configure_case(im2p_dim_mismatch "${dim_mismatch_root}" 16 16 32 IM2P_SIM FALSE
    "IM2P DIM mismatch: llama requests 32, archive reports 16")
run_im2p_configure_case(im2p_abi_mismatch "${abi_mismatch_root}" 16 16 32 IM2P_SIM FALSE
    "IM2P simulator ABI mismatch: llama requires 4, archive reports 3")
run_im2p_configure_case(im2p_missing_pair "${width_mismatch_root}" 8 8 32 IM2P_SIM FALSE
    "Missing matching IM2P a8-w8-d32 archive")
make_stale_frontend_root(stale_frontend_root)
run_im2p_configure_case(im2p_stale_frontend "${stale_frontend_root}" 8 8 32 IM2P_SIM FALSE
    "im2p_poll_completed_extended")
make_abi_stale_frontend_root(abi_stale_frontend_root)
run_im2p_configure_case(im2p_abi_stale_frontend "${abi_stale_frontend_root}"
    8 8 16 IM2P_SIM FALSE "IM2P frontend args layout mismatch")
make_extended_poll_stale_simulator_root(extended_poll_stale_simulator_root)
run_im2p_configure_case(im2p_extended_poll_stale_simulator
    "${extended_poll_stale_simulator_root}" 8 8 16 IM2P_SIM FALSE
    "im2p_poll_completed_extended")
run_arm64_bootstrap_contract()
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

expect_contains("${TEST_SOURCE_DIR}/CMakeLists.txt"
    "GGML_GEMMINI_EXECUTION_BACKEND")
expect_contains("${TEST_SOURCE_DIR}/CMakeLists.txt"
    "IM2P_SIM_ROOT")
expect_not_contains("${TEST_SOURCE_DIR}/CMakeLists.txt"
    "ExSIA A4/Q4 and A16/Q16 remain TODO pending RMD scale integration")
expect_contains("${TEST_SOURCE_DIR}/CMakeLists.txt"
    "im2p_begin_striped_matmul")
expect_registered_test_executable("${TEST_SOURCE_DIR}/tests/CMakeLists.txt"
    test-gemmini-log-boundary)
expect_cycle_sink_append("${TEST_SOURCE_DIR}/tests/CMakeLists.txt"
    test-ggml-cpu-cycle-sink)
file(READ "${TEST_SOURCE_DIR}/ggml/src/ggml-gemmini/ggml-gemmini.cpp"
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
expect_contains("${TEST_SOURCE_DIR}/ggml/src/ggml-gemmini/CMakeLists.txt"
    "GGML_GEMMINI_IM2P_FRONTEND_ARCHIVE")
expect_contains("${TEST_SOURCE_DIR}/ggml/src/ggml-gemmini/CMakeLists.txt"
    "GGML_GEMMINI_IM2P_SIM_ARCHIVE")
