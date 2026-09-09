if(NOT DEFINED MATMUL_SOURCE)
    message(FATAL_ERROR "MATMUL_SOURCE is required")
endif()
file(READ "${MATMUL_SOURCE}" source)
get_filename_component(gemmini_source_dir "${MATMUL_SOURCE}" DIRECTORY)
set(log_capi_source "${gemmini_source_dir}/../ggml-gemmini-utils/src/log-capi.cpp")
set(cycle_source
    "${gemmini_source_dir}/../ggml-gemmini-utils/src/cycle.cpp")
file(READ "${log_capi_source}" log_capi)
file(READ "${cycle_source}" cycle_source_text)

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

function(require_checked_publication value operation operation_success label)
    require_count("${value}" "emit_matmul_native_interval" 1 "${label} one publication")
    require_token("${value}" "${operation}" "${label} operation ID")
    require_token("${value}" "${operation_success}" "${label} separate algorithm result")
    require_absent("${value}" "gemmini_log_cycle_record_v2_checked_internal"
        "${label} algorithm status must not become structural eligibility")
endfunction()

function(require_run_only_checked_pair value operation target label)
    require_count("${value}" "cycle::read_sample()" 2 "${label} native endpoints")
    require_count("${value}" "emit_matmul_native_interval" 1
        "${label} checked publication")
    require_count("${value}" "\"${operation}\"" 1 "${label} operation label")
    require_order("${value}" "${label} exact boundary"
        "cycle::read_sample()" "${target}" "commit_end_sample = cycle::read_sample()"
        "emit_matmul_native_interval")
    foreach(token IN ITEMS "args().matmul_layer.c_str()" "matmul_cpu_run_id(args())"
                           "commit_start_sample, commit_end_sample"
                           "commit_start_ns, commit_end_ns, commit_start_tid, commit_end_tid"
                           "true, nullptr, matmul_cpu_run_id(args())")
        require_token("${value}" "${token}" "${label} publication contract")
    endforeach()
    foreach(token IN ITEMS GEMMINI_CYCLE_HAS_STRIPE_ID
                           GEMMINI_CYCLE_HAS_SLOT GEMMINI_CYCLE_HAS_NODE_ID
                           GEMMINI_CYCLE_HAS_WORKER_ID stripe_id worker_id
                           "cycle::read()" "gemmini_log_cycle_record_v2_checked_internal")
        require_absent("${value}" "${token}" "${label} synthetic identity/domain")
    endforeach()
endfunction()

function(require_native_detail_gate value label)
    require_token("${value}"
        "#if LOG_CYCLE && CYCLE_DETAIL && defined(__linux__) && defined(__aarch64__)"
        "${label} native reads require enabled detail collection")
    require_absent("${value}"
        "#if CYCLE_DETAIL && defined(__linux__) && defined(__aarch64__)"
        "${label} OFF cannot bypass the collection gate")
endfunction()

extract_between(commit "void MatMul::commit_output_transaction" "void MatMul::discard_output_transaction")
extract_between(run_full "MatMulResult MatMul::run_full" "MatMulStatus MatMul::begin_stripes")
extract_between(finish_stripes "MatMulStatus MatMul::finish_stripes" "MatMulCapability MatMul::stripe_capability")
extract_between(compose "MatmulStatus compose_rmd_stripe" "MatmulStatus finalize_stripe")
extract_between(finalize "MatmulStatus finalize_stripe" "MatmulStatus finish_execution")

foreach(token IN ITEMS "serialize_checked_cycle_record" "delta.valid"
                       "reason_name(delta.reason)" "interval.op")
    require_token("${log_capi}" "${token}" "checked sink projection contract")
endforeach()
foreach(token IN ITEMS "add_nullable_string(\"op\"" "add_identity(\"run_id\""
                       "add_identity(\"stripe_id\"" "add_null(\"delta\")"
                       "add_key(\"valid\")" "add_string(\"reason\"")
    require_token("${cycle_source_text}" "${token}"
        "checked sink machine-consumed nullable fields contract")
endforeach()

# Preserve the single legacy FULL CPU_DIRECT Merge pair and label. New packet
# and stripe correction intervals identify their own actual boundaries.
string(FIND "${run_full}" "if (args().residual_route == residual::ResidualRoute::cpu_direct)" direct_begin)
string(FIND "${run_full}" "    } else {" packet_begin)
if(direct_begin EQUAL -1 OR packet_begin EQUAL -1 OR packet_begin LESS_EQUAL direct_begin)
    message(FATAL_ERROR "cannot isolate normal FULL CPU_DIRECT route")
endif()
math(EXPR direct_length "${packet_begin} - ${direct_begin}")
string(SUBSTRING "${run_full}" ${direct_begin} ${direct_length} direct_full)
require_count("${direct_full}" "cycle::read_sample()" 2 "one legacy FULL Merge pair")
require_count("${direct_full}" "\"rmd_merge_cycles\"" 1 "one legacy FULL Merge label")
require_order("${direct_full}" "U12 success-only callsite guard"
    "if (residual_status == rmd::RmdStatus::success)"
    "merge_start_sample = cycle::read_sample()" "rmd::merge_rmd_correction"
    "merge_end_sample = cycle::read_sample()"
    "emit_matmul_native_interval"
    "if (residual_status != rmd::RmdStatus::success)")
require_absent("${finalize}" "\"rmd_merge_cycles\""
    "U12 must not duplicate stripe finalize Merge")

# U14 and U15 retain the origin/develop direct finite checks. Only U12 owns
# native sample endpoints in run_full; finish_stripes owns none.
require_count("${source}" "\"matmul_finite_output_validate_cycles\"" 0
    "U14/U15 finite validation label/site count")
require_count("${source}" "if (!finite_output(args()))" 2
    "U14/U15 direct finite validation count")
require_count("${run_full}" "cycle::read_sample()" 2
    "run_full U12-only native endpoints")
require_count("${finish_stripes}" "cycle::read_sample()" 0
    "finish-stripes native endpoints")

string(FIND "${run_full}" "if (!finite_output(args()))" full_validate)
if(full_validate EQUAL -1)
    message(FATAL_ERROR "U14 FULL direct finite validation is missing")
endif()
string(SUBSTRING "${run_full}" ${full_validate} -1 full_epilogue)
require_order("${full_epilogue}" "U14 FULL failure/commit/state ordering"
    "if (!finite_output(args()))" "discard_output_transaction()"
    "return {MatMulStatus::invalid_contract, MatMulCapability::unsupported}"
    "commit_output_transaction()" "state_ = MatMulState::completed"
    "return { MatMulStatus::success, MatMulCapability::supported }")

string(FIND "${finish_stripes}" "if (!finite_output(args()))" stripe_validate)
if(stripe_validate EQUAL -1)
    message(FATAL_ERROR "U15 stripe direct finite validation is missing")
endif()
string(SUBSTRING "${finish_stripes}" ${stripe_validate} -1 stripe_epilogue)
require_order("${stripe_epilogue}" "U15 stripe failure/commit/state ordering"
    "if (!finite_output(args()))" "discard_output_transaction()"
    "state_ = MatMulState::idle" "return MatMulStatus::invalid_contract"
    "commit_output_transaction()" "state_ = MatMulState::completed"
    "return MatMulStatus::success")

# U16 wraps only the normal facade's logical transaction copy. Early bypass
# returns before the first endpoint; state cleanup remains after publication.
require_run_only_checked_pair("${commit}" "matmul_output_commit_cycles"
    "for (size_t row = 0; row < args().I; ++row)" "U16 output commit copy")
require_order("${commit}" "U16 success-only commit boundary"
    "if (output_destination_ == nullptr || args_ptr_ == nullptr) return"
    "commit_start_sample = cycle::read_sample()"
    "commit_start_ns = cycle::timestamp_ns()" "commit_start_tid = cycle::host_thread_id()"
    "for (size_t row = 0; row < args().I; ++row)"
    "commit_end_sample = cycle::read_sample()"
    "commit_end_ns = cycle::timestamp_ns()" "commit_end_tid = cycle::host_thread_id()"
    "emit_matmul_native_interval"
    "args().f_out = output_destination_")
require_native_detail_gate("${commit}" "U16")
require_count("${source}" "\"rmd_merge_cycles\"" 1 "exact U12 label/site count")
require_count("${source}" "\"matmul_output_commit_cycles\"" 1
    "exact U16 label/site count")

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
require_order("${compose}" "Compose captures host endpoints before publication"
    "compose_start_ns = now_ns()" "compose_start_tid = cycle::host_thread_id()"
    "rmd::compose_rmd_output" "compose_end_ns = now_ns()"
    "compose_end_tid = cycle::host_thread_id()" "emit_matmul_native_interval"
    "job.metrics_.compose_end_ns = compose_end_ns")
require_token("${compose}" "job.metrics_.compose_start_ns, compose_end_ns"
    "Compose publishes captured host ns")
require_token("${compose}" "job.metrics_.compose_start_tid, compose_end_tid"
    "Compose publishes executing thread identity")
require_native_detail_gate("${compose}" "Compose")

# Keep one legacy inclusive Finalize pair. Its checked children are never
# added to or subtracted from their inclusive parent.
require_count("${finalize}" "cycle::read_sample()" 2 "one full Finalize native pair")
require_count("${finalize}" "telemetry_finalize_start_sample = cycle::read_sample()" 1 "Finalize start")
require_count("${finalize}" "telemetry_finalize_end_sample = cycle::read_sample()" 1 "Finalize end")
require_count("${finalize}" "merge_start = read_matmul_cpu_sample()" 1 "checked Merge start")
require_count("${finalize}" "merge_end = read_matmul_cpu_sample()" 1 "checked Merge end")
require_order("${finalize}" "Finalize contains Merge/diagnostics, not completion"
    "telemetry_finalize_start_sample = cycle::read_sample()"
    "merge_start = read_matmul_cpu_sample()" "rmd::merge_rmd_correction"
    "merge_end = read_matmul_cpu_sample()"
    "stats_start = read_matmul_cpu_sample()" "std::count_if"
    "stats_end = read_matmul_cpu_sample()"
    "matmul_telemetry_hash_enabled()" "hash_start = read_matmul_cpu_sample()"
    "rmd_input_hash" "hash_end = read_matmul_cpu_sample()"
    "telemetry_finalize_end_sample = cycle::read_sample()"
    "completion_start = read_matmul_cpu_sample()" "finalized_rows_ +="
    "job.release_slot()" "completion_end = read_matmul_cpu_sample()")
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
require_order("${finalize}" "Finalize captures host endpoints before publication"
    "finalize_start_ns = now_ns()" "finalize_start_tid = cycle::host_thread_id()"
    "merge_start = read_matmul_cpu_sample()" "finalize_end_ns = now_ns()"
    "finalize_end_tid = cycle::host_thread_id()" "emit_matmul_native_interval")
require_token("${finalize}" "job.metrics_.finalize_start_ns, job.metrics_.finalize_end_ns"
    "Finalize publishes captured host ns")
require_token("${finalize}" "job.metrics_.finalize_start_tid, job.metrics_.finalize_end_tid"
    "Finalize publishes executing thread identity")
require_native_detail_gate("${finalize}" "Finalize")

extract_between(dense "MatmulStatus execute_dense_stripe" "MatmulStatus accept_external_dense_completion")
extract_between(external "MatmulStatus accept_external_dense_completion" "MatmulStatus execute_rmd_stripe")
extract_between(residual "MatmulStatus execute_rmd_stripe" "MatmulStatus compose_rmd_stripe")
extract_between(capture "bool MatmulStripeCollector::on_ready" "MatmulStripeJob::MatmulStripeJob")
extract_between(worker "void MatmulStripeCollector::worker_loop" "const quants::act::exsia::StripeReadySink * MatmulStripeCollector::sink")
require_order("${dense}" "Dense samples exactly its facade host call"
    "dense_start = read_matmul_cpu_sample()" "facade_.run_staged_stripe"
    "dense_end = read_matmul_cpu_sample()" "to_public_status")
require_count("${dense}" "read_matmul_cpu_sample()" 2 "matching Dense gates")
require_absent("${dense}" "#if CYCLE_DETAIL" "Dense SUMMARY collects both endpoints")
require_absent("${external}" "cycle::read()" "external completion is unmeasured")
require_token("${external}" "unavailable(\"external_completion\")" "external marker status")
require_order("${residual}" "observer and metrics are outside executor pair"
    "observe_backend_dispatch" "backend_start = read_matmul_cpu_sample()"
    "residual::execute_direct_stripe" "backend_end = read_matmul_cpu_sample()"
    "metrics.direct_event_count")
require_order("${capture}" "input capture samples metadata/handle acquisition"
    "capture_start = read_matmul_cpu_sample()" "detail::capture_collector_event"
    "capture_end = read_matmul_cpu_sample()" "record_metric(captured.timing.capture_copy")
require_order("${worker}" "worker job preparation has its own same-thread pair"
    "pending_.pop_front()" "captured.timing.dequeued_ns = now_ns()"
    "captured.timing.dequeue_tid = cycle::host_thread_id()"
    "preparation_start = read_matmul_cpu_sample()" "std::make_shared<MatmulStripeJob>"
    "std::make_unique<quants::act::Meta>" "preparation_end = read_matmul_cpu_sample()")
foreach(pair IN ITEMS compose merge)
    require_token("${run_full}" "${pair}_start = read_matmul_cpu_sample()" "FULL packet ${pair} start")
    require_token("${run_full}" "${pair}_end = read_matmul_cpu_sample()" "FULL packet ${pair} end")
endforeach()
file(READ "${gemmini_source_dir}/ggml-gemmini-matmul.hpp" header)
require_order("${header}" "reader collection gate"
    "inline MatmulCpuSample read_matmul_cpu_sample()" "#if LOG_CYCLE"
    "result.collected = true" "result.native = cycle::read_sample()")
extract_between(emitter "void emit_matmul_cpu_interval" "void emit_matmul_native_interval")
require_order("${emitter}" "nonthrowing CPU telemetry boundary"
    "noexcept" "try {" "project_matmul_cpu_identity"
    "log::cycle.write_json" "serialize_matmul_cpu_interval" "catch (...)"
    "log::cycle.report_failure")
file(READ "${gemmini_source_dir}/ggml-gemmini-im2p.cpp" im2p)
require_order("${im2p}" "IM2P retains its measurement and failure boundary"
    "void finish(" "const auto end = read_matmul_cpu_sample()" "active_ = false"
    "try {" "log::cycle.write_json" "serialize_matmul_cpu_interval"
    "record_, start_, end, operation_success" "catch (...)" "log::cycle.report_failure")
extract_between(native_adapter "void emit_matmul_native_interval" "class ProofHash64")
require_token("${native_adapter}" "{start.value, true, start, start_ns, start_tid}"
    "native adapter preserves CPU sample and host start independently")
require_token("${native_adapter}" "{end.value, true, end, end_ns, end_tid}"
    "native adapter preserves CPU sample and host end independently")
message(STATUS "checked CPU validity, identity presence, and lifecycle boundaries passed")
