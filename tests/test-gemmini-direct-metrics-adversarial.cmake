if (NOT DEFINED DIRECT_SOURCE OR NOT DEFINED DIRECT_HEADER OR
    NOT DEFINED TEST_CMAKE OR NOT DEFINED CONTRACT_SCRIPT)
    message(FATAL_ERROR "DIRECT_SOURCE, DIRECT_HEADER, TEST_CMAKE, and CONTRACT_SCRIPT are required")
endif()

file(READ "${DIRECT_SOURCE}" base_source)
file(READ "${DIRECT_HEADER}" base_header)
file(READ "${TEST_CMAKE}" base_cmake)
if(NOT DEFINED PROJECT_ROOT)
    get_filename_component(direct_source_dir "${DIRECT_SOURCE}" DIRECTORY)
    get_filename_component(PROJECT_ROOT "${direct_source_dir}/../../../../.." ABSOLUTE)
endif()
set(im2p_source_path "${PROJECT_ROOT}/ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp")
set(matmul_source_path "${PROJECT_ROOT}/ggml/src/ggml-gemmini/ggml-gemmini-matmul.cpp")
if(NOT EXISTS "${im2p_source_path}" OR NOT EXISTS "${matmul_source_path}")
    message(FATAL_ERROR "F1b: PROJECT_ROOT does not contain Gemmini production sources: ${PROJECT_ROOT}")
endif()
file(READ "${im2p_source_path}" im2p_source)
file(READ "${matmul_source_path}" matmul_source)

# IM2P is protected origin/develop surface and must not inject event.run_id into
# direct metrics. Those two historical assignments were unauthorized. The
# matmul job pipeline separately retains its real, authorized job run ID.
string(FIND "${im2p_source}" "direct_metrics.run_id" forbidden_im2p_run_id)
if(NOT forbidden_im2p_run_id EQUAL -1)
    message(FATAL_ERROR
        "F1b: ggml-gemmini-im2p must contain zero direct_metrics.run_id uses")
endif()
set(im2p_remainder "${im2p_source}")
foreach(path_index RANGE 1 2)
    string(FIND "${im2p_remainder}"
        "residual::DirectExecutionMetrics direct_metrics{};" metrics_begin)
    if(metrics_begin EQUAL -1)
        message(FATAL_ERROR "F1b: IM2P direct path ${path_index} has no metrics object")
    endif()
    string(SUBSTRING "${im2p_remainder}" ${metrics_begin} -1 metrics_tail)
    string(FIND "${metrics_tail}" "} else if (event.rmd_packet != nullptr)" metrics_end)
    if(metrics_end EQUAL -1)
        message(FATAL_ERROR "F1b: IM2P direct path ${path_index} boundary is unavailable")
    endif()
    string(SUBSTRING "${metrics_tail}" 0 ${metrics_end} metrics_block)
    string(FIND "${metrics_block}" "status = residual::execute_direct_stripe(" direct_call)
    string(FIND "${metrics_block}" "direct_metrics.run_id" forbidden_path_run_id)
    if(direct_call EQUAL -1)
        message(FATAL_ERROR "F1b: IM2P direct path ${path_index} dispatch is unavailable")
    endif()
    if(NOT forbidden_path_run_id EQUAL -1)
        message(FATAL_ERROR
            "F1b: IM2P direct path ${path_index} must not fabricate direct run identity")
    endif()
    math(EXPR remainder_begin "${metrics_end} + 1")
    string(SUBSTRING "${metrics_tail}" ${remainder_begin} -1 im2p_remainder)
endforeach()
string(FIND "${im2p_remainder}"
    "residual::DirectExecutionMetrics direct_metrics{};" extra_im2p_metrics)
if(NOT extra_im2p_metrics EQUAL -1)
    message(FATAL_ERROR "F1b: unexpected third IM2P direct metrics path")
endif()

string(FIND "${matmul_source}"
    "residual::DirectExecutionMetrics direct_metrics{};" matmul_metrics_begin)
if(matmul_metrics_begin EQUAL -1)
    message(FATAL_ERROR "F1b: matmul direct metrics block is unavailable")
endif()
string(SUBSTRING "${matmul_source}" ${matmul_metrics_begin} -1 matmul_metrics_tail)
string(FIND "${matmul_metrics_tail}"
    "metrics.direct_event_count = direct_metrics.event_count;" matmul_metrics_end)
if(matmul_metrics_end EQUAL -1)
    message(FATAL_ERROR "F1b: matmul direct metrics block end is unavailable")
endif()
string(SUBSTRING "${matmul_metrics_tail}" 0 ${matmul_metrics_end} matmul_metrics_block)
string(FIND "${matmul_metrics_block}"
    "direct_metrics.run_id = job.metrics_.run_id;" matmul_run_assignment)
string(FIND "${matmul_metrics_block}"
    "residual::execute_direct_stripe(" matmul_direct_call)
if(matmul_run_assignment EQUAL -1 OR matmul_direct_call EQUAL -1 OR
   NOT matmul_run_assignment LESS matmul_direct_call)
    message(FATAL_ERROR "F1b: matmul direct pipeline must copy job run_id before dispatch")
endif()

string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef run_id)
set(work_dir "${CMAKE_CURRENT_BINARY_DIR}/direct-contract-adversarial-${run_id}")
file(MAKE_DIRECTORY "${work_dir}")
set(mutated_source_path "${work_dir}/direct-executor.cpp")
set(mutated_header_path "${work_dir}/direct-executor.hpp")
set(mutated_cmake_path "${work_dir}/CMakeLists.txt")

function(run_contract source_text header_text result_var output_var)
    file(WRITE "${mutated_source_path}" "${source_text}")
    file(WRITE "${mutated_header_path}" "${header_text}")
    file(WRITE "${mutated_cmake_path}" "${base_cmake}")
    execute_process(
        COMMAND "${CMAKE_COMMAND}"
            "-DDIRECT_SOURCE=${mutated_source_path}"
            "-DDIRECT_HEADER=${mutated_header_path}"
            "-DTEST_CMAKE=${mutated_cmake_path}"
            "-DPROJECT_ROOT=${PROJECT_ROOT}"
            -P "${CONTRACT_SCRIPT}"
        RESULT_VARIABLE contract_result
        OUTPUT_VARIABLE contract_output
        ERROR_VARIABLE contract_error)
    set(${result_var} "${contract_result}" PARENT_SCOPE)
    set(${output_var} "${contract_output}${contract_error}" PARENT_SCOPE)
endfunction()

# The unmodified implementation must satisfy the corrected contract before
# mutation testing is meaningful. This is intentionally RED until production
# removes aggregate metrics and provides standalone dynamic tile records.
run_contract("${base_source}" "${base_header}" base_result base_output)
if (NOT base_result EQUAL 0)
    file(REMOVE_RECURSE "${work_dir}")
    message(FATAL_ERROR "base standalone J-tile contract failed:\n${base_output}")
endif()

function(expect_rejected name source_text header_text)
    if (source_text STREQUAL base_source AND header_text STREQUAL base_header)
        message(FATAL_ERROR "mutation ${name} changed no input")
    endif()
    run_contract("${source_text}" "${header_text}" result output)
    if (result EQUAL 0)
        file(REMOVE_RECURSE "${work_dir}")
        message(FATAL_ERROR "standalone J-tile contract accepted mutation ${name}")
    endif()
    message(STATUS "mutation=${name} rejected=true")
endfunction()

set(mutated_header "${base_header}")
string(REPLACE "struct DirectCpuTileRecord {"
    "struct DirectCpuTileRecord {\n    uint64_t serial_pre_cycles = 0;"
    mutated_header "${mutated_header}")
expect_rejected(serial_boundary "${base_source}" "${mutated_header}")

set(mutated_header "${base_header}")
string(REPLACE "std::vector<DirectCpuTileRecord> cpu_tiles"
    "std::array<DirectCpuTileRecord, 3> cpu_tiles"
    mutated_header "${mutated_header}")
expect_rejected(fixed_three_tiles "${base_source}" "${mutated_header}")

set(mutated_header "${base_header}")
string(REPLACE "std::optional<uint64_t> delta_cycles;"
    "std::optional<uint64_t> delta_cycles;\n    std::optional<uint64_t> total_cycles;"
    mutated_header "${mutated_header}")
expect_rejected(tile_sum "${base_source}" "${mutated_header}")

set(mutated_header "${base_header}")
string(REPLACE "uint64_t run_id" "uint64_t omitted_identity"
    mutated_header "${mutated_header}")
expect_rejected(missing_run_identity "${base_source}" "${mutated_header}")

set(mutated_header "${base_header}")
string(REPLACE "uint64_t owner_event_token" "uint64_t omitted_owner_token"
    mutated_header "${mutated_header}")
expect_rejected(missing_owner_provenance "${base_source}" "${mutated_header}")

set(mutated_source "${base_source}")
string(REPLACE "uint32_t identity_mask" "uint64_t identity_mask"
    mutated_source "${mutated_source}")
expect_rejected(wide_identity_mask_narrowing "${mutated_source}" "${base_header}")

set(mutated_source "${base_source}")
string(REPLACE "testing::DirectCpuSamplePoint::tile_end"
    "testing::DirectCpuSamplePoint::tile_start"
    mutated_source "${mutated_source}")
expect_rejected(missing_tile_end "${mutated_source}" "${base_header}")

set(mutated_source "${base_source}")
string(REPLACE "cycle::read_sample()" "cycle::NativeCycleSample{}"
    mutated_source "${mutated_source}")
expect_rejected(missing_production_reader "${mutated_source}" "${base_header}")

set(mutated_source "${base_source}")
string(REPLACE "cycle::evaluate_interval(" "cycle::omitted_interval_evaluator("
    mutated_source "${mutated_source}")
expect_rejected(missing_production_evaluator "${mutated_source}" "${base_header}")

set(mutated_source "${base_source}")
string(REPLACE "gemmini_log_cycle_record_v2_checked_internal("
    "omitted_direct_tile_emitter("
    mutated_source "${mutated_source}")
expect_rejected(missing_production_emitter "${mutated_source}" "${base_header}")

set(mutated_source "${base_source}")
string(REPLACE "record.worker_id = direct_worker_id()"
    "record.worker_id = 0"
    mutated_source "${mutated_source}")
expect_rejected(constant_worker_identity "${mutated_source}" "${base_header}")

set(mutated_source "${base_source}")
string(REPLACE "hooks != nullptr && hooks->sample_reader != nullptr"
    "hooks != nullptr"
    mutated_source "${mutated_source}")
expect_rejected(hook_enables_sampling "${mutated_source}" "${base_header}")

file(REMOVE_RECURSE "${work_dir}")
message(STATUS "dynamic standalone direct J-tile adversarial contract passed")
