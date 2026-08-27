if (NOT DEFINED DIRECT_SOURCE OR NOT DEFINED DIRECT_HEADER OR
    NOT DEFINED TEST_CMAKE OR NOT DEFINED CONTRACT_SCRIPT)
    message(FATAL_ERROR "DIRECT_SOURCE, DIRECT_HEADER, TEST_CMAKE, and CONTRACT_SCRIPT are required")
endif()

file(READ "${DIRECT_SOURCE}" base_source)
file(READ "${TEST_CMAKE}" base_cmake)
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef run_id)
set(work_dir "${CMAKE_CURRENT_BINARY_DIR}/direct-contract-adversarial-${run_id}")
file(MAKE_DIRECTORY "${work_dir}")
set(mutated_source_path "${work_dir}/direct-executor.cpp")
set(mutated_cmake_path "${work_dir}/CMakeLists.txt")

function(expect_accepted_before_rejected_after name)
    if (mutation_source STREQUAL base_source AND mutation_cmake STREQUAL base_cmake)
        message(FATAL_ERROR "mutation ${name} changed no input")
    endif()
    file(WRITE "${mutated_source_path}" "${mutation_source}")
    file(WRITE "${mutated_cmake_path}" "${mutation_cmake}")

    execute_process(
        COMMAND "${CMAKE_COMMAND}"
            "-DDIRECT_SOURCE=${mutated_source_path}"
            "-DDIRECT_HEADER=${DIRECT_HEADER}"
            "-DTEST_CMAKE=${mutated_cmake_path}"
            -DDIRECT_CONTRACT_LEGACY_ACCEPTANCE=ON
            -P "${CONTRACT_SCRIPT}"
        RESULT_VARIABLE before_result
        OUTPUT_VARIABLE before_output
        ERROR_VARIABLE before_error)
    if (NOT before_result EQUAL 0)
        message(FATAL_ERROR
            "mutation ${name} was not accepted before its adversarial rule:\n${before_output}${before_error}")
    endif()

    execute_process(
        COMMAND "${CMAKE_COMMAND}"
            "-DDIRECT_SOURCE=${mutated_source_path}"
            "-DDIRECT_HEADER=${DIRECT_HEADER}"
            "-DTEST_CMAKE=${mutated_cmake_path}"
            -P "${CONTRACT_SCRIPT}"
        RESULT_VARIABLE after_result
        OUTPUT_VARIABLE after_output
        ERROR_VARIABLE after_error)
    if (after_result EQUAL 0)
        message(FATAL_ERROR "mutation ${name} was accepted after its adversarial rule")
    endif()
    message(STATUS "mutation=${name} accepted_before=true rejected_after=true")
endfunction()

set(source_anchor "CpuSample read_cpu_sample() {\n")
set(mutation_cmake "${base_cmake}")

set(mutation_source "${base_source}")
string(REPLACE "${source_anchor}" "${source_anchor}    (void) ignored;\n"
       mutation_source "${mutation_source}")
expect_accepted_before_rejected_after(void_cast)

set(mutation_source "${base_source}")
string(REPLACE "${source_anchor}" "${source_anchor}    static_cast<void>(ignored);\n"
       mutation_source "${mutation_source}")
expect_accepted_before_rejected_after(static_cast_void)

set(mutation_source "${base_source}")
string(REPLACE "${source_anchor}" "${source_anchor}    [[maybe_unused]] int ignored = 0;\n"
       mutation_source "${mutation_source}")
expect_accepted_before_rejected_after(maybe_unused)

set(mutation_source "${base_source}")
string(REPLACE "${source_anchor}" "${source_anchor}#pragma GCC diagnostic ignored \"-Wunused-parameter\"\n"
       mutation_source "${mutation_source}")
expect_accepted_before_rejected_after(gcc_diagnostic)

set(mutation_source "${base_source}")
string(REPLACE "${source_anchor}" "${source_anchor}#pragma clang diagnostic ignored \"-Wunused-parameter\"\n"
       mutation_source "${mutation_source}")
expect_accepted_before_rejected_after(clang_diagnostic)

set(mutation_source "${base_source}")
string(REPLACE "${source_anchor}" "${source_anchor}    __int128 ignored = 0;\n"
       mutation_source "${mutation_source}")
expect_accepted_before_rejected_after(int128_extension)

set(mutation_source "${base_source}")
set(mutation_cmake "${base_cmake}\ntarget_compile_options(test-gemmini-exsia PRIVATE -Wno-unused-parameter)\n")
expect_accepted_before_rejected_after(wno_unused)

set(mutation_cmake "${base_cmake}\ntarget_compile_options(test-gemmini-exsia PRIVATE -Wno-error=unused-parameter)\n")
expect_accepted_before_rejected_after(wno_error)

set(mutation_cmake "${base_cmake}")
string(REPLACE
    "    set_target_properties(test-gemmini-direct-linux-aarch64-strict PROPERTIES COMPILE_OPTIONS \"\")\n"
    "" mutation_cmake "${mutation_cmake}")
expect_accepted_before_rejected_after(remove_compile_options_reset)

set(mutation_cmake "${base_cmake}")
string(REPLACE "        -Wpedantic\n" "" mutation_cmake "${mutation_cmake}")
expect_accepted_before_rejected_after(remove_wpedantic)

set(mutation_cmake "${base_cmake}")
string(REPLACE "        -Wunused-parameter\n" "" mutation_cmake "${mutation_cmake}")
expect_accepted_before_rejected_after(remove_wunused_parameter)

set(mutation_cmake "${base_cmake}\ntarget_compile_options(test-gemmini-exsia PRIVATE -Wno-array-bounds)\n")
expect_accepted_before_rejected_after(wno_array_bounds)

set(mutation_cmake "${base_cmake}\ntarget_compile_options(test-gemmini-exsia PRIVATE -Wno-shadow)\n")
expect_accepted_before_rejected_after(arbitrary_wno)

set(mutation_cmake "${base_cmake}\nif (APPLE)\n    target_compile_definitions(test-gemmini-direct-linux-aarch64-strict PRIVATE __linux__=1)\nendif()\n")
expect_accepted_before_rejected_after(apple_linux)

set(mutation_cmake "${base_cmake}\nadd_compile_definitions(GGML_GEMMINI_DIRECT_METRICS_TESTING=1)\n")
expect_accepted_before_rejected_after(global_testing_macro)

set(mutation_cmake "${base_cmake}\ntarget_compile_definitions(ggml-gemmini PRIVATE GGML_GEMMINI_DIRECT_METRICS_TESTING=1)\n")
expect_accepted_before_rejected_after(ggml_gemmini_testing_macro)

file(REMOVE_RECURSE "${work_dir}")
message(STATUS "direct CPU metrics adversarial contract passed")
