if(NOT DEFINED TEST_SOURCE_DIR)
    message(FATAL_ERROR "TEST_SOURCE_DIR is required")
endif()
if(NOT DEFINED RESIDUAL_CAPTURE_PATH)
    set(RESIDUAL_CAPTURE_PATH
        "${TEST_SOURCE_DIR}/ggml/src/ggml-gemmini/residual/residual-capture.hpp")
endif()

file(READ "${RESIDUAL_CAPTURE_PATH}" source)
string(FIND "${source}" "ResidualStripePayload finish()" finish_begin)
string(FIND "${source}" "    bool holds_cpu_sink" finish_end)
if(finish_begin EQUAL -1 OR finish_end EQUAL -1 OR finish_end LESS finish_begin)
    message(FATAL_ERROR "TimedResidualCapture::finish boundary is unavailable")
endif()
string(SUBSTRING "${source}" ${finish_begin} ${finish_end} finish_body)

# reset() receives the real stripe identity. The capture object must retain it
# and the standalone finish record must advertise stripe_id only; this boundary
# has no real run_id and must not invent one.
string(FIND "${source}" "stripe_id_ = stripe_id;" reset_stripe_assignment)
if(reset_stripe_assignment EQUAL -1)
    message(FATAL_ERROR "F3: TimedResidualCapture::reset must retain its real stripe_id")
endif()
string(FIND "${finish_body}" "const gemmini_cycle_record_v2 record" record_begin)
string(FIND "${finish_body}"
    "gemmini_log_cycle_record_v2_checked_internal" record_end)
if(record_begin EQUAL -1 OR record_end EQUAL -1 OR record_end LESS record_begin)
    message(FATAL_ERROR "F3: bounded capture_finish checked record is unavailable")
endif()
math(EXPR record_length "${record_end} - ${record_begin}")
string(SUBSTRING "${finish_body}" ${record_begin} ${record_length} record_block)
string(FIND "${record_block}" "GEMMINI_CYCLE_HAS_STRIPE_ID" stripe_mask)
string(FIND "${record_block}" "0, stripe_id_, 0, 0, 0" stripe_value)
string(FIND "${record_block}" "GEMMINI_CYCLE_HAS_RUN_ID" run_mask)
string(FIND "${record_block}" "next_" generated_identity)
if(stripe_mask EQUAL -1 OR stripe_value EQUAL -1)
    message(FATAL_ERROR
        "F3: capture_finish identity mask/value must contain the retained stripe_id only")
endif()
if(NOT run_mask EQUAL -1 OR NOT generated_identity EQUAL -1)
    message(FATAL_ERROR "F3: capture_finish must not invent run_id")
endif()

string(REGEX MATCHALL "cycle::read_sample\\(\\)" finish_samples "${finish_body}")
list(LENGTH finish_samples finish_sample_count)
if(NOT finish_sample_count EQUAL 2)
    message(FATAL_ERROR "capture_finish must have one route-selected sample pair")
endif()

string(FIND "${finish_body}" "if (empty()) return result;" empty_return)
string(FIND "${finish_body}" "finish_start = cycle::read_sample();" sample_start)
string(FIND "${finish_body}" "const uint64_t start = cycle::timestamp_ns();" ns_start)
string(FIND "${finish_body}" "result.direct = cpu->finish();" direct_finish)
string(FIND "${finish_body}" "result.packet = std::get<rmd::RmdStripeBuilder>(sink_).finish();" packet_finish)
string(FIND "${finish_body}" "const uint64_t end = cycle::timestamp_ns();" ns_end)
string(FIND "${finish_body}" "finish_end = cycle::read_sample();" sample_end)
string(FIND "${finish_body}" "cycle::evaluate_interval(finish_start, finish_end)" evaluation)
string(FIND "${finish_body}" "gemmini_log_cycle_record_v2_checked_internal" logger)
if(empty_return EQUAL -1 OR sample_start EQUAL -1 OR ns_start EQUAL -1 OR
   direct_finish EQUAL -1 OR packet_finish EQUAL -1 OR ns_end EQUAL -1 OR
   sample_end EQUAL -1 OR evaluation EQUAL -1 OR logger EQUAL -1 OR
   NOT empty_return LESS ns_start OR NOT ns_start LESS sample_start OR
   NOT sample_start LESS direct_finish OR NOT sample_start LESS packet_finish OR
   NOT direct_finish LESS sample_end OR NOT packet_finish LESS sample_end OR
   NOT sample_end LESS evaluation OR NOT evaluation LESS logger OR
   NOT logger LESS ns_end)
    message(FATAL_ERROR
        "legacy capture_ns timestamps must enclose the cycle-only builder.finish pair and checked publication")
endif()

math(EXPR sample_span "${sample_end} - ${sample_start}")
string(SUBSTRING "${finish_body}" ${sample_start} ${sample_span} sample_body)
if(sample_body MATCHES "now_ns\\(|timestamp_ns\\(|evaluate_interval|gemmini_log_cycle_record_v2_checked_internal")
    message(FATAL_ERROR "capture_finish PMU pair must enclose builder.finish only")
endif()

# Existing capture_ns is preserved, but the Linux-AArch64 detail blocks may not
# introduce another now_ns/timestamp_ns pair.
set(rest "${finish_body}")
while(1)
    string(FIND "${rest}" "#if defined(__linux__) && defined(__aarch64__) && CYCLE_DETAIL" detail_begin)
    if(detail_begin EQUAL -1)
        break()
    endif()
    string(SUBSTRING "${rest}" ${detail_begin} -1 tail)
    string(FIND "${tail}" "#endif" detail_end)
    if(detail_end EQUAL -1)
        message(FATAL_ERROR "unterminated capture_finish Linux-AArch64 detail block")
    endif()
    string(SUBSTRING "${tail}" 0 ${detail_end} detail_block)
    if(detail_block MATCHES "now_ns\\(|timestamp_ns\\(")
        message(FATAL_ERROR "capture_finish Linux-AArch64 detail path must be cycle-only")
    endif()
    math(EXPR next "${detail_end} + 6")
    string(SUBSTRING "${tail}" ${next} -1 rest)
endwhile()

foreach(op IN ITEMS rmd_direct_finish_cycles rmd_packet_finish_cycles)
    string(FIND "${finish_body}" "${op}" found)
    if(found EQUAL -1)
        message(FATAL_ERROR "missing standalone ${op} record")
    endif()
endforeach()
# The checked sink owns mismatch projection: it serializes a null delta and the
# exact reader reason, while never accepting a partial endpoint as valid.
string(FIND "${finish_body}" "gemmini_log_cycle_record_v2_checked_internal" checked_sink)
if(checked_sink EQUAL -1)
    message(FATAL_ERROR "capture_finish mismatch must use the checked standalone sink")
endif()

message(STATUS "standalone capture_finish boundary contract passed")
