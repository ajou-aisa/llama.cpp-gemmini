if(NOT DEFINED MATMUL_SOURCE)
    message(FATAL_ERROR "MATMUL_SOURCE is required")
endif()
file(READ "${MATMUL_SOURCE}" source)
get_filename_component(gemmini_source_dir "${MATMUL_SOURCE}" DIRECTORY)
set(log_capi_source "${gemmini_source_dir}/../ggml-gemmini-utils/src/log-capi.cpp")
set(cycle_serialization_source
    "${gemmini_source_dir}/../ggml-gemmini-utils/src/cycle-serialization.cpp")
file(READ "${log_capi_source}" log_capi)
file(READ "${cycle_serialization_source}" cycle_serialization)

function(extract_between output begin_marker end_marker)
    string(FIND "${source}" "${begin_marker}" begin)
    string(FIND "${source}" "${end_marker}" end)
    if(begin EQUAL -1 OR end EQUAL -1 OR end LESS_EQUAL begin)
        message(FATAL_ERROR "cannot extract ${begin_marker}")
    endif()
    math(EXPR length "${end} - ${begin}")
    string(SUBSTRING "${source}" ${begin} ${length} value)
    set(${output} "${value}" PARENT_SCOPE)
endfunction()

function(require_count value needle expected label)
    set(rest "${value}")
    set(count 0)
    while(1)
        string(FIND "${rest}" "${needle}" position)
        if(position EQUAL -1)
            break()
        endif()
        string(LENGTH "${needle}" needle_length)
        math(EXPR next "${position} + ${needle_length}")
        string(SUBSTRING "${rest}" ${next} -1 rest)
        math(EXPR count "${count} + 1")
    endwhile()
    if(NOT count EQUAL expected)
        message(FATAL_ERROR "${label}: expected ${expected}, found ${count}")
    endif()
endfunction()

function(require_order value label)
    set(previous -1)
    foreach(token IN LISTS ARGN)
        string(FIND "${value}" "${token}" position)
        if(position EQUAL -1 OR (NOT previous EQUAL -1 AND NOT previous LESS position))
            message(FATAL_ERROR "${label}: ordering failed at ${token}")
        endif()
        set(previous ${position})
    endforeach()
endfunction()

function(require_absent value token label)
    string(FIND "${value}" "${token}" position)
    if(NOT position EQUAL -1)
        message(FATAL_ERROR "${label}: unexpected ${token}")
    endif()
endfunction()

function(require_token value token label)
    string(FIND "${value}" "${token}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "${label}: missing ${token}")
    endif()
endfunction()

function(require_checked_publication value operation eligibility label)
    require_count("${value}" "gemmini_log_cycle_record_v2_checked_internal" 1
        "${label} checked publication")
    string(FIND "${value}" "const gemmini_cycle_record_v2" record_begin)
    if(record_begin EQUAL -1)
        message(FATAL_ERROR "${label}: standalone v2 record is missing")
    endif()
    string(SUBSTRING "${value}" ${record_begin} -1 publication_tail)
    string(FIND "${publication_tail}" "gemmini_log_cycle_record_v2_checked_internal" sink)
    if(sink EQUAL -1)
        message(FATAL_ERROR "${label}: standalone checked sink is missing")
    endif()
    math(EXPR publication_length "${sink} + 256")
    string(LENGTH "${publication_tail}" tail_length)
    if(publication_length GREATER tail_length)
        set(publication_length ${tail_length})
    endif()
    string(SUBSTRING "${publication_tail}" 0 ${publication_length} publication)
    if(NOT value MATCHES "\"[^\"]*${operation}[^\"]*\"" OR
       NOT publication MATCHES "${operation}")
        message(FATAL_ERROR "${label}: standalone op must identify ${operation}")
    endif()
    foreach(token IN ITEMS GEMMINI_CYCLE_HAS_RUN_ID GEMMINI_CYCLE_HAS_STRIPE_ID
                           job.metrics_.run_id job.metrics_.stripe_id "${eligibility}")
        require_token("${publication}" "${token}" "${label} standalone publication")
    endforeach()
    require_absent("${publication}" "PIPELINE_STRIPE_SUMMARY"
        "${label} publication must remain outside the canonical summary")
endfunction()

function(require_cycle_only_detail_blocks value label)
    set(rest "${value}")
    while(1)
        string(FIND "${rest}" "#if CYCLE_DETAIL && defined(__linux__) && defined(__aarch64__)" begin)
        if(begin EQUAL -1)
            break()
        endif()
        string(SUBSTRING "${rest}" ${begin} -1 tail)
        string(FIND "${tail}" "#endif" end)
        if(end EQUAL -1)
            message(FATAL_ERROR "${label}: unterminated Linux-AArch64 detail block")
        endif()
        string(SUBSTRING "${tail}" 0 ${end} block)
        if(block MATCHES "now_ns\\(|timestamp_ns\\(")
            message(FATAL_ERROR "${label}: Linux-AArch64 detail block must be cycle-only")
        endif()
        math(EXPR next "${end} + 6")
        string(SUBSTRING "${tail}" ${next} -1 rest)
    endwhile()
endfunction()

extract_between(run_full "MatMulResult MatMul::run_full" "MatMulStatus MatMul::begin_stripes")
extract_between(compose "MatmulStatus compose_rmd_stripe" "MatmulStatus finalize_stripe")
extract_between(finalize "MatmulStatus finalize_stripe" "MatmulStatus finish_execution")

foreach(token IN ITEMS "serialize_checked_cycle_record" "delta.valid"
                       "reason_name(delta.reason)" "interval.op")
    require_token("${log_capi}" "${token}" "checked sink projection contract")
endforeach()
foreach(token IN ITEMS "add_nullable_string(\"op\"" "add_identity(\"run_id\""
                       "add_identity(\"stripe_id\"" "add_null(\"delta\")"
                       "add_key(\"valid\")" "add_string(\"reason\"")
    require_token("${cycle_serialization}" "${token}"
        "checked sink machine-consumed nullable fields contract")
endforeach()

# One full packet Compose pair. Direct/no-packet Compose exits structurally
# before the start sample; operation failures still close the pair before they
# can publish anything.
require_count("${compose}" "cycle::read_sample()" 2 "one full Compose pair")
require_count("${compose}" "telemetry_compose_start_sample" 2 "Compose start endpoint storage/use")
require_count("${compose}" "telemetry_compose_end_sample" 2 "Compose end endpoint storage/use")
require_order("${compose}" "Compose detail boundary"
    "job.direct_residual_ != nullptr || packet == nullptr"
    "telemetry_compose_start_sample = cycle::read_sample()"
    "rmd::compose_rmd_output"
    "telemetry_compose_end_sample = cycle::read_sample()"
    "if (status != rmd::RmdStatus::success)")
require_absent("${compose}" "compose_cpu_work" "Compose is standalone, not canonical CPU work")
require_absent("${compose}" "CpuWorkCoverage" "Compose has no coverage framework")
require_checked_publication("${compose}" "compose" "status == rmd::RmdStatus::success" "Compose")
require_cycle_only_detail_blocks("${compose}" "Compose")

# Finalize owns exactly one native pair. The nested Merge remains the exact
# origin/develop scalar pair and is never selected into or arithmetically
# combined with Finalize.
require_count("${finalize}" "cycle::read_sample()" 2 "one full Finalize native pair")
require_count("${finalize}" "telemetry_finalize_start_sample = cycle::read_sample()" 1 "Finalize start")
require_count("${finalize}" "telemetry_finalize_end_sample = cycle::read_sample()" 1 "Finalize end")
require_count("${finalize}" "telemetry_merge_start = cycle::read()" 1 "scalar Merge start")
require_count("${finalize}" "telemetry_merge_end = cycle::read()" 1 "scalar Merge end")
require_order("${finalize}" "Finalize and scalar Merge boundary"
    "telemetry_finalize_start_sample = cycle::read_sample()"
    "telemetry_merge_start = cycle::read()"
    "rmd::merge_rmd_correction"
    "telemetry_merge_end = cycle::read()"
    "telemetry_finalize_end_sample = cycle::read_sample()")
foreach(token IN ITEMS cpu_work CpuWorkCoverage additive profiled_stripe checked_sum
                       telemetry_merge_start_sample telemetry_merge_end_sample)
    require_absent("${finalize}" "${token}" "Finalize/Merge never enters canonical totals or native Merge detail")
endforeach()
string(REGEX REPLACE "[ \t\r\n]" "" finalize_compact "${finalize}")
if(finalize_compact MATCHES "finalize[^;]*[+-][^;]*merge|merge[^;]*[+-][^;]*finalize")
    message(FATAL_ERROR "Finalize and Merge must never be added or subtracted")
endif()
require_absent("${finalize}" "telemetry_residual_end_sample = cycle::read_sample()"
    "Finalize has no third residual-total endpoint")
require_checked_publication("${finalize}" "finalize" "merge_failure.ok()" "Finalize")
require_cycle_only_detail_blocks("${finalize}" "Finalize")

message(STATUS "approved Compose/Finalize/Merge boundary contract passed")
