function(expect_contains text needle)
    string(FIND "${text}" "${needle}" found_at)
    if(found_at EQUAL -1)
        message(FATAL_ERROR "Expected '${needle}' in:\n${text}")
    endif()
endfunction()

function(expect_not_contains text needle)
    string(FIND "${text}" "${needle}" found_at)
    if(NOT found_at EQUAL -1)
        message(FATAL_ERROR "Did not expect '${needle}' in:\n${text}")
    endif()
endfunction()

function(write_profile path bits dim)
    if(bits EQUAL 4)
        set(packing signed-int4-low-nibble-first)
    else()
        set(packing signed-int8)
    endif()
    math(EXPR scratchpad_row_bytes "${dim} * ${bits} / 8")
    math(EXPR accumulator_row_bytes "${dim} * 4")
    math(EXPR bank_rows "262144 / 4 / ${scratchpad_row_bytes}")
    math(EXPR accumulator_rows "65536 / ${accumulator_row_bytes}")
    math(EXPR ws_scratchpad_rows "4 * ${bank_rows} / 2")
    math(EXPR ws_accumulator_rows "${accumulator_rows} / 2")
    file(WRITE "${path}" "{\n"
        "  \"schema_version\": 1,\n"
        "  \"profile\": \"a${bits}w${bits}-d${dim}-hp1\",\n"
        "  \"implementation\": \"gemmini-hp1\",\n"
        "  \"activation_bits\": ${bits},\n"
        "  \"weight_bits\": ${bits},\n"
        "  \"dim\": ${dim},\n"
        "  \"accumulator_bits\": 32,\n"
        "  \"block_size\": 32,\n"
        "  \"scu\": \"hp1-left-shift\",\n"
        "  \"packing\": \"${packing}\",\n"
        "  \"numerical_revision\": \"hp1-fragment-sat32-v1\",\n"
        "  \"memory\": {\"bank_count\": 4, \"bank_rows\": ${bank_rows}, "
        "\"accumulator_rows\": ${accumulator_rows}, \"scratchpad_row_bytes\": ${scratchpad_row_bytes}, "
        "\"accumulator_row_bytes\": ${accumulator_row_bytes}, \"scratchpad_total_bytes\": 262144, "
        "\"accumulator_total_bytes\": 65536, \"ws_double_buffered\": true, "
        "\"ws_scratchpad_rows_per_buffer\": ${ws_scratchpad_rows}, "
        "\"ws_accumulator_rows_per_buffer\": ${ws_accumulator_rows}},\n"
        "  \"host_contract_source\": \"fixture\",\n"
        "  \"host_contract_sha256\": \"0000000000000000000000000000000000000000000000000000000000000000\"\n"
        "}\n")
endfunction()

function(run_resolver name expect_success out_var)
    execute_process(
        COMMAND "${TEST_PYTHON}"
            "${TEST_SOURCE_DIR}/scripts/im2p-build-options.py"
            "${TEST_BINARY_ROOT}/${name}" build-arm64.sh
            GGML_GEMMINI_EXECUTION_BACKEND=FPGA_UART
            IM2P_SIM_ROOT=${TEST_BINARY_ROOT}/im2p
            -- ${ARGN}
        RESULT_VARIABLE rc OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
    string(CONCAT output "${stdout}" "\n" "${stderr}")
    if(expect_success AND NOT rc EQUAL 0)
        message(FATAL_ERROR "${name} unexpectedly failed:\n${output}")
    elseif(NOT expect_success AND rc EQUAL 0)
        message(FATAL_ERROR "${name} unexpectedly succeeded:\n${output}")
    endif()
    set(${out_var} "${output}" PARENT_SCOPE)
endfunction()

if(NOT DEFINED TEST_SOURCE_DIR OR NOT EXISTS "${TEST_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "TEST_SOURCE_DIR must point to llama.cpp-gemmini")
endif()
if(NOT DEFINED TEST_BINARY_ROOT OR NOT DEFINED TEST_PYTHON OR
   NOT DEFINED TEST_CMAKE)
    message(FATAL_ERROR "TEST_BINARY_ROOT, TEST_PYTHON, and TEST_CMAKE are required")
endif()

file(REMOVE_RECURSE "${TEST_BINARY_ROOT}")
file(MAKE_DIRECTORY "${TEST_BINARY_ROOT}")

foreach(bits IN ITEMS 4 8)
    foreach(dim IN ITEMS 16 32 64)
        set(profile "${TEST_BINARY_ROOT}/a${bits}w${bits}-d${dim}-hp1.json")
        write_profile("${profile}" ${bits} ${dim})
        run_resolver("a${bits}w${bits}-d${dim}" TRUE output
            -DIM2P_FPGA_ARCH=GEMMINI_HP1
            -DIM2P_GEMMINI_RESOLVED_PROFILE=${profile}
            -DGGML_GEMMINI_ENABLE_RMD=OFF)
        expect_contains("${output}" "GGML_GEMMINI_ACTIVATION_BITS_DEFAULT=${bits}")
        expect_contains("${output}" "GGML_GEMMINI_WEIGHT_BITS_DEFAULT=${bits}")
        expect_contains("${output}" "GGML_GEMMINI_DIM_DEFAULT=${dim}")
        expect_contains("${output}" "IM2P_FPGA_ARCH_DEFAULT=GEMMINI_HP1")
    endforeach()
endforeach()

set(invalid_memory "${TEST_BINARY_ROOT}/invalid-memory.json")
write_profile("${invalid_memory}" 8 16)
file(READ "${invalid_memory}" invalid_memory_text)
string(REPLACE "\"bank_count\": 4" "\"bank_count\": 3" invalid_memory_text "${invalid_memory_text}")
file(WRITE "${invalid_memory}" "${invalid_memory_text}")
run_resolver(invalid-memory FALSE invalid_memory_output
    -DIM2P_FPGA_ARCH=GEMMINI_HP1
    -DIM2P_GEMMINI_RESOLVED_PROFILE=${invalid_memory}
    -DGGML_GEMMINI_ENABLE_RMD=OFF)
expect_contains("${invalid_memory_output}" "memory contract is inconsistent")

run_resolver(legacy TRUE legacy_output -DGGML_GEMMINI_ENABLE_RMD=ON)
expect_contains("${legacy_output}" "GGML_GEMMINI_ACTIVATION_BITS_DEFAULT=8")
expect_contains("${legacy_output}" "GGML_GEMMINI_WEIGHT_BITS_DEFAULT=8")
expect_contains("${legacy_output}" "GGML_GEMMINI_DIM_DEFAULT=16")
expect_contains("${legacy_output}" "GGML_GEMMINI_ENABLE_RMD_DEFAULT=ON")

set(profile "${TEST_BINARY_ROOT}/a4w4-d16-hp1.json")
run_resolver(rmd-on FALSE rmd_output
    -DIM2P_FPGA_ARCH=GEMMINI_HP1
    -DIM2P_GEMMINI_RESOLVED_PROFILE=${profile}
    -DGGML_GEMMINI_ENABLE_RMD=ON)
expect_contains("${rmd_output}" "GEMMINI_HP1 requires GGML_GEMMINI_ENABLE_RMD=OFF")

run_resolver(profile-mismatch FALSE mismatch_output
    -DIM2P_FPGA_ARCH=GEMMINI_HP1
    -DIM2P_GEMMINI_RESOLVED_PROFILE=${profile}
    -DGGML_GEMMINI_ACTIVATION_BITS=8
    -DGGML_GEMMINI_WEIGHT_BITS=8
    -DGGML_GEMMINI_ENABLE_RMD=OFF)
expect_contains("${mismatch_output}" "resolved profile requires GGML_GEMMINI_ACTIVATION_BITS=4")

run_resolver(no-manifest FALSE missing_output
    -DIM2P_FPGA_ARCH=GEMMINI_HP1
    -DGGML_GEMMINI_ENABLE_RMD=OFF)
expect_contains("${missing_output}" "GEMMINI_HP1 requires IM2P_GEMMINI_RESOLVED_PROFILE")

file(READ "${TEST_SOURCE_DIR}/cmake/ggml-gemmini-fpga.cmake" fpga_cmake)
expect_contains("${fpga_cmake}" "FPGA_UART currently supports Linux hosts only")
expect_contains("${fpga_cmake}" "GEMMINI_HP1")
expect_contains("${fpga_cmake}" "hp1-fragment-sat32-v1")

file(READ "${TEST_SOURCE_DIR}/ggml/src/ggml-gemmini/CMakeLists.txt" backend_cmake)
expect_contains("${backend_cmake}" "IM2P_GEMMINI_FRONTEND_EXPECTED_DIM=\${GGML_GEMMINI_DIM}")
expect_contains("${backend_cmake}" "IM2P_GEMMINI_FRONTEND_ACTIVATION_BITS=\${GGML_GEMMINI_ACTIVATION_BITS}")
expect_contains("${backend_cmake}" "IM2P_FPGA_ARCH_GEMMINI_HP1=1")
string(FIND "${backend_cmake}"
    "if (GGML_GEMMINI_EXECUTION_BACKEND STREQUAL \"FPGA_UART\")" fpga_at)
string(FIND "${backend_cmake}"
    "if (GGML_GEMMINI_EXECUTION_BACKEND STREQUAL \"IM2P_SIM\")" simulator_at)
math(EXPR fpga_length "${simulator_at} - ${fpga_at}")
string(SUBSTRING "${backend_cmake}" ${fpga_at} ${fpga_length} fpga_block)
expect_not_contains("${fpga_block}" "ggml-gemmini-im2p-simulator")

execute_process(
    COMMAND "${TEST_CMAKE}"
        -S "${TEST_SOURCE_DIR}"
        -B "${TEST_BINARY_ROOT}/cmake-mismatch"
        -DGGML_GEMMINI=ON
        -DGGML_GEMMINI_OPTION=WS
        -DGGML_GEMMINI_EXECUTION_BACKEND=FPGA_UART
        -DIM2P_FPGA_ARCH=GEMMINI_HP1
        -DIM2P_GEMMINI_RESOLVED_PROFILE=${profile}
        -DGGML_GEMMINI_ACTIVATION_BITS=8
        -DGGML_GEMMINI_WEIGHT_BITS=8
        -DGGML_GEMMINI_DIM=16
        -DGGML_GEMMINI_BLOCK_SIZE=32
        -DGGML_GEMMINI_ENABLE_RMD=OFF
        -DIM2P_SIM_ROOT=${TEST_BINARY_ROOT}/im2p
        -DLLAMA_BUILD_COMMON=OFF
        -DLLAMA_BUILD_TESTS=OFF
        -DLLAMA_BUILD_TOOLS=OFF
        -DLLAMA_BUILD_EXAMPLES=OFF
        -DLLAMA_BUILD_SERVER=OFF
        -DLLAMA_CURL=OFF
    RESULT_VARIABLE cmake_rc OUTPUT_VARIABLE cmake_stdout ERROR_VARIABLE cmake_stderr)
if(cmake_rc EQUAL 0)
    message(FATAL_ERROR "Mismatched CMake/profile identity unexpectedly configured")
endif()
string(CONCAT cmake_output "${cmake_stdout}" "\n" "${cmake_stderr}")
string(REGEX REPLACE "[ \t\r\n]+" " " cmake_output "${cmake_output}")
expect_contains("${cmake_output}"
    "GEMMINI_HP1 CMake/profile mismatch: requested (8, 8, 16, 32), profile (4, 4, 16, 32)")
