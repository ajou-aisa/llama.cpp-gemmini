if (NOT DEFINED TEST_SOURCE_DIR)
    message(FATAL_ERROR "TEST_SOURCE_DIR is required")
endif()
if (NOT DEFINED RESIDUAL_CAPTURE_PATH)
    set(RESIDUAL_CAPTURE_PATH
        "${TEST_SOURCE_DIR}/ggml/src/ggml-gemmini/residual/residual-capture.hpp")
endif()

file(READ "${RESIDUAL_CAPTURE_PATH}" source)
string(FIND "${source}" "ResidualStripePayload finish()" finish_begin)
string(FIND "${source}" "    bool holds_cpu_sink" finish_end)
if (finish_begin EQUAL -1 OR finish_end EQUAL -1 OR finish_end LESS finish_begin)
    message(FATAL_ERROR "TimedResidualCapture::finish boundary is unavailable")
endif()
string(SUBSTRING "${source}" ${finish_begin} ${finish_end} finish_body)

string(REGEX MATCHALL "cycle::read_sample\\(\\)" source_samples "${source}")
string(REGEX MATCHALL "cycle::read_sample\\(\\)" finish_samples "${finish_body}")
list(LENGTH source_samples source_sample_count)
list(LENGTH finish_samples finish_sample_count)
if (NOT "${source_sample_count}" STREQUAL "2" OR
    NOT "${finish_sample_count}" STREQUAL "2")
    message(FATAL_ERROR "PMU samples must be one pair immediately outside the original timestamp pair")
endif()

string(FIND "${finish_body}" "finish_start = cycle::read_sample();" pmu_start)
string(FIND "${finish_body}" "const uint64_t start = cycle::timestamp_ns();" timestamp_start)
string(FIND "${finish_body}" "result.direct = cpu->finish();" direct_finish)
string(FIND "${finish_body}" "result.packet = std::get<rmd::RmdStripeBuilder>(sink_).finish();" packet_finish)
string(FIND "${finish_body}" "const uint64_t end = cycle::timestamp_ns();" timestamp_end)
string(FIND "${finish_body}" "finish_end = cycle::read_sample();" pmu_end)
string(FIND "${finish_body}" "cycle::evaluate_interval(finish_start, finish_end)" evaluation)
string(FIND "${finish_body}" "gemmini_log_cycle_record_v2_checked_internal" logger)
if (pmu_start EQUAL -1 OR timestamp_start EQUAL -1 OR direct_finish EQUAL -1 OR
    packet_finish EQUAL -1 OR timestamp_end EQUAL -1 OR pmu_end EQUAL -1 OR
    evaluation EQUAL -1 OR logger EQUAL -1 OR
    NOT pmu_start LESS timestamp_start OR NOT timestamp_start LESS direct_finish OR
    NOT timestamp_start LESS packet_finish OR NOT direct_finish LESS timestamp_end OR
    NOT packet_finish LESS timestamp_end OR NOT timestamp_end LESS pmu_end OR
    NOT pmu_end LESS evaluation OR NOT evaluation LESS logger)
    message(FATAL_ERROR "PMU samples must immediately enclose the original timestamp pair; evaluation/logging must follow it")
endif()

math(EXPR timestamp_span "${timestamp_end} - ${timestamp_start}")
string(SUBSTRING "${finish_body}" ${timestamp_start} ${timestamp_span} timestamp_body)
if (timestamp_body MATCHES "cycle::read_sample\\(\\)|cycle::evaluate_interval|gemmini_log_cycle_record_v2_checked_internal")
    message(FATAL_ERROR "the original capture_ns timestamp pair must contain only builder finish")
endif()

string(REGEX MATCHALL "rmd_[A-Za-z_]*finish_cycles" finish_operations "${finish_body}")
list(LENGTH finish_operations finish_operation_count)
if (NOT "${finish_operation_count}" STREQUAL "2")
    message(FATAL_ERROR "exactly two route-specific finish operations are required")
endif()
string(REGEX MATCHALL "GEMMINI_NATIVE_CYCLE_SOURCE_LINUX_PERF_CPU_CYCLES" source_projections "${finish_body}")
list(LENGTH source_projections source_projection_count)
if (NOT "${source_projection_count}" STREQUAL "2" OR
    finish_body MATCHES "static_cast<uint8_t>\\(finish_(start|end)\\.source\\)")
    message(FATAL_ERROR "checked Linux finish samples must use the typed Linux perf source constant")
endif()
