if (NOT DEFINED DIRECT_SOURCE OR NOT DEFINED DIRECT_HEADER OR NOT DEFINED TEST_CMAKE)
    message(FATAL_ERROR "DIRECT_SOURCE, DIRECT_HEADER, and TEST_CMAKE are required")
endif()

file(READ "${DIRECT_SOURCE}" direct_source)
file(READ "${DIRECT_HEADER}" direct_header)
file(READ "${TEST_CMAKE}" test_cmake)
if(NOT DEFINED PROJECT_ROOT)
    get_filename_component(direct_source_dir "${DIRECT_SOURCE}" DIRECTORY)
    get_filename_component(PROJECT_ROOT "${direct_source_dir}/../../../../.." ABSOLUTE)
endif()
set(matmul_source_path
    "${PROJECT_ROOT}/ggml/src/ggml-gemmini/ggml-gemmini-matmul.cpp")
if(NOT EXISTS "${matmul_source_path}")
    message(FATAL_ERROR "MatMul production source is unavailable")
endif()
file(READ "${matmul_source_path}" matmul_source)

function(require_count text regex expected label)
    string(REGEX MATCHALL "${regex}" matches "${text}")
    list(LENGTH matches actual)
    if (NOT actual EQUAL expected)
        message(FATAL_ERROR "${label}: expected ${expected}, got ${actual}")
    endif()
endfunction()

# The direct executor has exactly one start/end sampling site inside the J-tile
# invocation. The sites execute dynamically once for every execute_j_tile call.
require_count("${direct_source}"
    "testing::DirectCpuSamplePoint::tile_start,[ \t\r\n]+tile_index" 1
    "one test start sample site per dynamic J tile")
require_count("${direct_source}"
    "testing::DirectCpuSamplePoint::tile_end,[ \t\r\n]+tile_index" 1
    "one test end sample site per dynamic J tile")
require_count("${direct_source}" "const CpuSample tile_start" 1
    "one J-tile start boundary declaration")

# Hooks only substitute endpoint samples. The normal Linux-AArch64 detail path
# samples and emits for every dynamic tile invocation, including production
# callers which do not request DirectExecutionMetrics.
require_count("${direct_source}" "cycle::read_sample\\(\\)" 1
    "production J-tile endpoint reader")
require_count("${direct_source}" "cycle::evaluate_interval\\(" 1
    "production J-tile provenance evaluator")
string(FIND "${direct_source}" "CpuInterval cpu_interval" interval_start)
string(FIND "${direct_source}" "size_t direct_worker_id" interval_end)
if(interval_start EQUAL -1 OR interval_end EQUAL -1 OR interval_end LESS interval_start)
    message(FATAL_ERROR "J-tile interval evaluator boundary is unavailable")
endif()
math(EXPR interval_length "${interval_end} - ${interval_start}")
string(SUBSTRING "${direct_source}" ${interval_start} ${interval_length} interval_block)
if(NOT interval_block MATCHES "cycle::evaluate_interval\\(")
    message(FATAL_ERROR "J-tile record validity must come from the native evaluator")
endif()
require_count("${direct_source}" "gemmini_log_cycle_record_v2_checked_internal\\(" 1
    "production J-tile checked emitter")
require_count("${direct_source}" "rmd_direct_j_tile_interval" 1
    "standalone J-tile operation identity")
if(direct_source MATCHES
    "const bool sample_cpu =[^;]*hooks")
    message(FATAL_ERROR "test hooks must not enable J-tile production records")
endif()
string(FIND "${direct_source}" "CpuSample read_cpu_sample" sample_helper_start)
string(FIND "${direct_source}" "CpuInterval cpu_interval" sample_helper_end)
if(sample_helper_start EQUAL -1 OR sample_helper_end EQUAL -1 OR
   sample_helper_end LESS sample_helper_start)
    message(FATAL_ERROR "J-tile endpoint reader boundary is unavailable")
endif()
math(EXPR sample_helper_length "${sample_helper_end} - ${sample_helper_start}")
string(SUBSTRING "${direct_source}" ${sample_helper_start} ${sample_helper_length}
       sample_helper_block)
if(NOT sample_helper_block MATCHES
    "hooks != nullptr && hooks->sample_reader != nullptr")
    message(FATAL_ERROR "test hook reader must be an optional endpoint substitute")
endif()
if(NOT DEFINED SEMANTIC_CASE OR SEMANTIC_CASE STREQUAL "F1a")
    if(direct_source MATCHES "const bool sample_cpu = metrics != nullptr")
        message(FATAL_ERROR
            "F1a: Linux-AArch64 J-tile sampling must not depend on DirectExecutionMetrics")
    endif()
    if(direct_source MATCHES "next_direct_run_id")
        message(FATAL_ERROR "F1a: unavailable run identity must not be fabricated")
    endif()
    string(FIND "${matmul_source}" "MatMulResult MatMul::run_full()" run_full_begin)
    string(FIND "${matmul_source}" "MatMulStatus MatMul::begin_stripes()" run_full_end)
    if(run_full_begin EQUAL -1 OR run_full_end EQUAL -1 OR
       run_full_end LESS run_full_begin)
        message(FATAL_ERROR "F1a: bounded MatMul::run_full body is unavailable")
    endif()
    math(EXPR run_full_length "${run_full_end} - ${run_full_begin}")
    string(SUBSTRING "${matmul_source}" ${run_full_begin} ${run_full_length} run_full_body)
    string(FIND "${run_full_body}"
        "residual::execute_direct_stripe(args(), *payload, correction)"
        run_full_direct_call)
    if(run_full_direct_call EQUAL -1)
        message(FATAL_ERROR
            "F1a: MatMul::run_full null-metrics direct execution must remain covered")
    endif()

    string(FIND "${direct_source}" "uint64_t identity_mask" wide_identity_mask)
    string(FIND "${direct_source}" "static_cast<uint32_t>(identity_mask)"
        narrowed_identity_mask)
    if(NOT wide_identity_mask EQUAL -1 OR NOT narrowed_identity_mask EQUAL -1)
        message(FATAL_ERROR
            "F1a: identity_mask must match the uint32_t checked-record ABI without narrowing")
    endif()
    string(FIND "${direct_source}" "uint32_t identity_mask" identity_begin)
    string(FIND "${direct_source}"
        "gemmini_log_cycle_record_v2_checked_internal(" identity_end)
    if(identity_begin EQUAL -1 OR identity_end EQUAL -1 OR
       identity_end LESS identity_begin)
        message(FATAL_ERROR "F1a: bounded checked identity block is unavailable")
    endif()
    math(EXPR identity_length "${identity_end} - ${identity_begin}")
    string(SUBSTRING "${direct_source}" ${identity_begin} ${identity_length} identity_block)
    string(FIND "${identity_block}" "GEMMINI_CYCLE_HAS_STRIPE_ID" stripe_identity)
    string(FIND "${identity_block}" "GEMMINI_CYCLE_HAS_NODE_ID" node_identity)
    string(FIND "${identity_block}" "GEMMINI_CYCLE_HAS_WORKER_ID" worker_identity)
    string(FIND "${identity_block}" "if (direct_run_id != 0)" conditional_run)
    string(FIND "${identity_block}" "GEMMINI_CYCLE_HAS_RUN_ID" run_identity)
    string(FIND "${identity_block}" "identity_mask, direct_run_id" direct_identity_use)
    if(stripe_identity EQUAL -1 OR node_identity EQUAL -1 OR worker_identity EQUAL -1 OR
       conditional_run EQUAL -1 OR run_identity EQUAL -1 OR direct_identity_use EQUAL -1 OR
       NOT conditional_run LESS run_identity)
        message(FATAL_ERROR
            "F1a: uint32 identity must retain stripe/node/worker and conditional RUN_ID bits")
    endif()
endif()
if(NOT direct_source MATCHES
    "tile_cpu_records\\.assign\\(j_tile_count, DirectCpuTileRecord\\{\\}\\)" OR
   NOT direct_source MATCHES "metrics->cpu_tiles = std::move\\(tile_cpu_records\\)")
    message(FATAL_ERROR "every dynamic execute_j_tile must publish its standalone record")
endif()
foreach(identity_flag IN ITEMS GEMMINI_CYCLE_HAS_RUN_ID GEMMINI_CYCLE_HAS_STRIPE_ID
                               GEMMINI_CYCLE_HAS_NODE_ID GEMMINI_CYCLE_HAS_WORKER_ID)
    if(NOT direct_source MATCHES "${identity_flag}")
        message(FATAL_ERROR "standalone production emitter lacks ${identity_flag}")
    endif()
endforeach()

foreach(forbidden IN ITEMS
        "serial_pre" "serial_post" "tile_cycles" "total_cycles"
        "checked_add_u64" "DirectCpuDetailMetrics" "algorithm_cpu_leaves")
    if (direct_source MATCHES "${forbidden}" OR direct_header MATCHES "${forbidden}")
        message(FATAL_ERROR "standalone J-tile contract rejects ${forbidden}")
    endif()
endforeach()

# Records are dynamic and standalone. They identify the invocation and retain
# its own validity/provenance; they are not a fixed worker array or an aggregate.
if (NOT direct_header MATCHES "struct DirectCpuTileRecord")
    message(FATAL_ERROR "DirectCpuTileRecord is unavailable")
endif()
foreach(field IN ITEMS run_id stripe_id worker_id tile_index j_begin j_end
                       delta_cycles valid reason owner_event_token generation)
    if (NOT direct_header MATCHES "${field}")
        message(FATAL_ERROR "DirectCpuTileRecord is missing ${field}")
    endif()
endforeach()
string(REGEX MATCH "struct DirectCpuTileRecord \\{[^}]*\\};" direct_record_block
             "${direct_header}")
if(direct_record_block STREQUAL "" OR NOT direct_record_block MATCHES "sample_reason")
    message(FATAL_ERROR
        "F1c: DirectCpuTileRecord must preserve the exact endpoint sample_reason")
endif()
if (NOT direct_header MATCHES
    "std::vector<DirectCpuTileRecord>[ \t\r\n]+cpu_tiles")
    message(FATAL_ERROR "J-tile records must use a dynamic vector")
endif()
if(NOT direct_source MATCHES "record\\.worker_id = direct_worker_id\\(\\)" OR
   NOT direct_source MATCHES "omp_get_thread_num\\(\\)")
    message(FATAL_ERROR "published worker_id must come from the executing worker")
endif()
if(direct_source MATCHES "record\\.worker_id = [0-9]+")
    message(FATAL_ERROR "published worker_id must not be a constant")
endif()
if (direct_header MATCHES
    "(array|DirectCpuTileRecord[ \t]+[A-Za-z_][A-Za-z0-9_]*\\[[0-9]+\\]).*(worker|tile)")
    message(FATAL_ERROR "J-tile records must not encode a fixed worker/tile count")
endif()
foreach(point IN ITEMS serial_pre_start serial_pre_end serial_post_start serial_post_end)
    if (direct_header MATCHES "${point}" OR direct_source MATCHES "${point}")
        message(FATAL_ERROR "test sampling seam retains forbidden ${point}")
    endif()
endforeach()
require_count("${direct_header}" "tile_start" 1
    "test sampling seam has one tile-start point")
require_count("${direct_header}" "tile_end" 1
    "test sampling seam has one tile-end point")

# Production owns no test hook state. The hook remains target-private.
string(REGEX MATCH "struct DirectExecutionMetrics \\{[^}]*\\};" metrics_block
             "${direct_header}")
if (metrics_block STREQUAL "")
    message(FATAL_ERROR "DirectExecutionMetrics block is unavailable")
endif()
if (metrics_block MATCHES "TestHooks|SampleReader|sample_reader|context")
    message(FATAL_ERROR "production metrics layout contains test hook state")
endif()
foreach(reason IN ITEMS unavailable_event unavailable_direct_mapping multiplexed
                        seqlock_exhausted invalid_start invalid_end source_mismatch
                        event_owner_mismatch event_generation_mismatch
                        structurally_cross_task counter_regression)
    if(NOT direct_header MATCHES "${reason}")
        message(FATAL_ERROR "F1c: direct sample/record reason domain is missing ${reason}")
    endif()
endforeach()
string(REGEX MATCH "struct DirectCpuSample \\{[^}]*\\};" direct_sample_block
             "${direct_header}")
if(direct_sample_block STREQUAL "" OR NOT direct_sample_block MATCHES "reason")
    message(FATAL_ERROR "F1c: DirectCpuSample must preserve the exact endpoint reason")
endif()
string(FIND "${direct_source}" "GEMMINI_NATIVE_CYCLE_REASON_UNAVAILABLE_EVENT"
       blanket_unavailable_reason)
if(NOT blanket_unavailable_reason EQUAL -1)
    message(FATAL_ERROR
        "F1c: checked direct samples must not reconstruct every failure as unavailable_event")
endif()
foreach(token IN ITEMS "struct DirectExecutionTestHooks" "DirectCpuSampleReader sample_reader"
                       "void * context")
    string(FIND "${direct_header}" "${token}" found)
    if (found EQUAL -1)
        message(FATAL_ERROR "missing test-only hook token ${token}")
    endif()
endforeach()
string(REGEX MATCH
    "target_compile_definitions\\(test-gemmini-exsia PRIVATE[^)]*EXSIA_VALIDATION=1\\)"
    test_defs "${test_cmake}")
if (test_defs STREQUAL "" OR NOT test_defs MATCHES
    "GGML_GEMMINI_DIRECT_METRICS_TESTING=1")
    message(FATAL_ERROR "direct sampling seam must be exact-target private")
endif()

# Keep the existing suppression-free portability contract.
foreach(pattern IN ITEMS "\\(void\\)[ \t]+[A-Za-z_]" "static_cast<void>"
                         "\\[\\[maybe_unused\\]\\]"
                         "#[ \t]*pragma[ \t]+GCC[ \t]+diagnostic"
                         "#[ \t]*pragma[ \t]+clang[ \t]+diagnostic"
                         "__int128" "__builtin_[A-Za-z0-9_]+")
    if (direct_source MATCHES "${pattern}")
        message(FATAL_ERROR "suppression-free source contract rejected ${pattern}")
    endif()
endforeach()
if (test_cmake MATCHES "-Wno-[A-Za-z0-9_=+-]+")
    message(FATAL_ERROR "strict CMake contract rejects arbitrary -Wno")
endif()

message(STATUS "dynamic standalone direct J-tile source contract passed")
