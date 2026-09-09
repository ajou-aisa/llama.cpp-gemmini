#include "ggml-gemmini-telemetry.hpp"
#include "ggml-gemmini-matmul.hpp"
#include <gemmini/log.hpp>

#include <sstream>
#include <limits>
#include <string_view>

namespace ggml::gemmini {

std::string serialize_matmul_cpu_interval(log::CycleRecord record,
        const MatmulCpuSample & start, const MatmulCpuSample & end,
        bool operation_success, const MatmulCpuInterval * explicit_interval) {
    const auto interval = explicit_interval != nullptr ? *explicit_interval :
        evaluate_matmul_cpu_interval(start, end);
    record.start = start.value;
    record.end = end.value;
    std::string json = log::serialize_checked_cycle_record(record, interval.cycles.has_value(),
        interval.reason.empty() ? nullptr : interval.reason.c_str(),
        interval.sample_reason.empty() ? nullptr : interval.sample_reason.c_str());
    json.insert(json.rfind('}'),
        std::string(",\"cpu_measurement_version\":1,\"operation_success\":") +
        (operation_success ? "true" : "false") + ",\"additive\":false,\"host_timing\":" +
        cycle::serialize_host_timing(start.ns, end.ns, start.tid, end.tid));
    return json;
}

void project_matmul_cpu_identity(log::CycleRecord & record,
        const MatmulJobMetrics * profile, std::optional<uint64_t> invocation_run_id) {
    if (profile != nullptr) {
        record.identity_mask = profile->cpu_identity_mask;
        record.stripe_id = profile->stripe_id;
        record.run_id = profile->run_id;
        record.slot = profile->slot;
    } else if (invocation_run_id.has_value()) {
        record.identity_mask = GEMMINI_CYCLE_HAS_RUN_ID;
        record.run_id = *invocation_run_id;
    }
}

namespace {
void json_string(std::ostringstream & out, std::string_view value);
const char * telemetry_backend_name(RmdBackend backend) {
    return backend == RmdBackend::cpu_direct ? "cpu_direct" : "gemmini_ws_compact";
}
const char * telemetry_clock_source() {
#ifdef __riscv
    return "riscv_cycle";
#elif defined(__linux__) && defined(__aarch64__)
    return "linux_perf_cpu_cycles";
#else
    return "host_tick";
#endif
}
const char * telemetry_unit_name(std::string_view units) {
    return units == "cycles" ? "cycle" : "tick";
}
const char * telemetry_source_name(MatmulOptionSource source) {
    switch (source) {
        case MatmulOptionSource::build_default: return "build_default";
        case MatmulOptionSource::environment: return "environment";
        case MatmulOptionSource::explicit_override: return "explicit_override";
    }
    return "invalid";
}

MatmulCpuInterval aggregate_cpu_intervals(
        const std::vector<MatmulJobMetrics> & profiles,
        MatmulCpuInterval MatmulJobMetrics::* member) {
    MatmulCpuInterval result{{}, {}, {}, 0, 0, 0};
    uint64_t sum = 0;
    for (const auto & profile : profiles) {
        const auto & item = profile.*member;
        ++result.count;
        if (item.reason == "not_applicable") {
            ++result.not_applicable_count;
        } else if (!item.cycles.has_value()) {
            if (result.reason.empty()) {
                result.reason = item.reason;
                result.sample_reason = item.sample_reason;
            }
        } else {
            ++result.valid_count;
            if (*item.cycles > std::numeric_limits<uint64_t>::max() - sum) {
                result.reason = "aggregate_overflow";
            } else {
                sum += *item.cycles;
            }
        }
    }
    if (result.reason.empty()) {
        if (result.valid_count != 0) result.cycles = sum;
        else result.reason = "not_applicable";
    }
    return result;
}

void cpu_interval_json(std::ostringstream & out, const char * key,
                       const MatmulCpuInterval & interval, bool comma = true) {
    if (comma) out << ',';
    out << '"' << key << "\":";
    if (interval.cycles) out << *interval.cycles; else out << "null";
    out << ",\"" << key << "_valid\":" << (interval.cycles ? "true" : "false")
        << ",\"" << key << "_reason\":";
    if (interval.reason.empty()) out << "null";
    else json_string(out, interval.reason);
    out << ",\"" << key << "_sample_reason\":";
    if (interval.sample_reason.empty()) out << "null";
    else json_string(out, interval.sample_reason);
    out << ",\"" << key << "_count\":" << interval.count
        << ",\"" << key << "_valid_count\":" << interval.valid_count
        << ",\"" << key << "_not_applicable_count\":" << interval.not_applicable_count;
}
}

RmdTelemetryRecord make_rmd_telemetry_record(
        RmdBackend backend, MatmulOptionSource source,
        std::string runtime_bundle_id, std::string model_id, std::string layer,
        uint64_t run_id,
        const MatmulCpuInterval & invocation_total,
        const std::vector<MatmulJobMetrics> & profiles) {
    RmdTelemetryRecord record{};
    record.runtime_bundle_id = std::move(runtime_bundle_id);
    record.model_id = std::move(model_id);
    record.layer = std::move(layer);
    record.run_id = run_id;
    record.backend = backend;
    record.source = source;
    record.units = cycle::units();
    record.invocation_total = invocation_total;
    record.timing.prep = aggregate_cpu_intervals(profiles, &MatmulJobMetrics::cpu_prep);
    record.timing.backend_service = aggregate_cpu_intervals(profiles, &MatmulJobMetrics::cpu_backend);
    record.timing.merge = aggregate_cpu_intervals(profiles, &MatmulJobMetrics::cpu_merge);
    record.timing.residual_total = aggregate_cpu_intervals(profiles, &MatmulJobMetrics::cpu_residual_total);
#if defined(__linux__) && defined(__aarch64__)
    record.timing.queue = MatmulCpuInterval::unavailable("structurally_cross_task");
#else
    record.timing.queue = MatmulCpuInterval::measured(0);
#endif
    for (const MatmulJobMetrics & profile : profiles) {
        record.counters.direct_events += profile.rmd.direct_event_count;
        record.counters.direct_calls += profile.rmd.direct_call_count;
        record.counters.packet_calls += profile.rmd.packet_call_count;
        record.counters.ws_calls += profile.rmd.ws_call_count;
        record.geometry.packet_count += profile.rmd.packet_call_count;
        record.geometry.active_blocks += profile.rmd.active_blocks;
        record.geometry.compact_k_count += profile.rmd.compact_k_count;
        record.geometry.padded_k_count += profile.rmd.padded_k_count;
        record.geometry.physical_tile_count += profile.rmd.physical_tile_count;
#if !(defined(__linux__) && defined(__aarch64__))
        // Preserve the legacy auxiliary host-tick queue, never a PMU interval.
        if (profile.telemetry_queue_tick != 0 &&
            profile.telemetry_residual_start >= profile.telemetry_queue_tick) {
            const uint64_t elapsed = profile.telemetry_residual_start - profile.telemetry_queue_tick;
            if (record.timing.queue.cycles &&
                elapsed <= std::numeric_limits<uint64_t>::max() - *record.timing.queue.cycles) {
                *record.timing.queue.cycles += elapsed;
            } else {
                record.timing.queue = MatmulCpuInterval::unavailable("aggregate_overflow");
            }
        }
#endif
        if (record.timing.dense_end == 0 || profile.telemetry_dense_end < record.timing.dense_end)
            record.timing.dense_end = profile.telemetry_dense_end;
        if (record.timing.residual_start == 0 || profile.telemetry_residual_start < record.timing.residual_start)
            record.timing.residual_start = profile.telemetry_residual_start;
#if CYCLE_DETAIL
        record.stripes.push_back({profile.stripe_id, profile.row_begin, profile.row_end,
            {profile.telemetry_dense_start, profile.telemetry_dense_end,
             profile.telemetry_residual_start, profile.telemetry_backend_start,
             profile.telemetry_backend_end, profile.telemetry_merge_start,
             profile.telemetry_merge_end, profile.telemetry_residual_end},
            profile.telemetry_input_hash,
            profile.telemetry_correction_hash,
            profile.telemetry_output_hash,
            profile.telemetry_correction_nonzero_count,
            profile.telemetry_hash_enabled, profile.cpu_dense});
#endif
    }
    record.work = backend == RmdBackend::cpu_direct
        ? record.counters.direct_calls != 0 : record.counters.packet_calls != 0;
    return record;
}

std::string serialize_rmd_telemetry(const RmdTelemetryRecord & record) {
#if !LOG_CYCLE
    (void) record;
    return {};
#else
    std::ostringstream out;
    out << "{\"schema\":"; json_string(out, record.schema);
    out << ",\"version\":" << record.version << ",\"record_type\":\"RMD_BACKEND_TELEMETRY\"";
    out << ",\"source\":"; json_string(out, telemetry_clock_source());
    out << ",\"unit\":"; json_string(out, telemetry_unit_name(record.units));
    out << ",\"op\":\"rmd.execute\",\"layer\":";
    if (record.layer.empty()) out << "null"; else json_string(out, record.layer);
    out << ",\"run_id\":" << record.run_id
        << ",\"stripe_id\":null,\"slot\":null,\"node_id\":null,\"worker_id\":null"
        << ",\"runtime_bundle_id\":";
    json_string(out, record.runtime_bundle_id);
    out << ",\"model_id\":"; json_string(out, record.model_id);
    out << ",\"backend\":"; json_string(out, telemetry_backend_name(record.backend));
    out << ",\"option_source\":"; json_string(out, telemetry_source_name(record.source));
    out << ",\"cpu_measurement_version\":1,\"work\":" << (record.work ? "true" : "false");
    cpu_interval_json(out, "invocation_total", record.invocation_total);
    out << ",\"dispatch\":{\"direct_events\":" << record.counters.direct_events
        << ",\"direct_calls\":" << record.counters.direct_calls
        << ",\"packet_calls\":" << record.counters.packet_calls
        << ",\"ws_calls\":" << record.counters.ws_calls << "}"
        << ",\"timing\":{";
    cpu_interval_json(out, "prep", record.timing.prep, false);
    cpu_interval_json(out, "backend_service", record.timing.backend_service);
    cpu_interval_json(out, "merge", record.timing.merge);
    cpu_interval_json(out, "residual_total", record.timing.residual_total);
    cpu_interval_json(out, "queue", record.timing.queue);
#if defined(__linux__) && defined(__aarch64__)
    out << ",\"dense_end\":null,\"residual_start\":null";
#else
    out << ",\"dense_end\":" << record.timing.dense_end
        << ",\"residual_start\":" << record.timing.residual_start;
#endif
    out << "},\"geometry\":{\"packet_count\":" << record.geometry.packet_count
        << ",\"active_blocks\":" << record.geometry.active_blocks
        << ",\"compact_k_count\":" << record.geometry.compact_k_count
        << ",\"padded_k_count\":" << record.geometry.padded_k_count
        << ",\"physical_tile_count\":" << record.geometry.physical_tile_count << "}";
#if CYCLE_DETAIL
    out << ",\"stripes\":[";
    for (size_t i = 0; i < record.stripes.size(); ++i) {
        const auto & stripe = record.stripes[i];
        if (i != 0) out << ',';
        out << "{\"stripe_id\":" << stripe.stripe_id << ",\"row_begin\":" << stripe.row_begin
            << ",\"row_end\":" << stripe.row_end
            << ",\"stages\":{\"dense_start\":" << stripe.ordered_ticks[0]
            << ",\"dense_end\":" << stripe.ordered_ticks[1]
            << ",\"residual_start\":" << stripe.ordered_ticks[2]
            << ",\"backend_start\":" << stripe.ordered_ticks[3]
            << ",\"backend_end\":" << stripe.ordered_ticks[4]
            << ",\"merge_start\":" << stripe.ordered_ticks[5]
            << ",\"merge_end\":" << stripe.ordered_ticks[6]
            << ",\"residual_end\":" << stripe.ordered_ticks[7] << '}'
            << ",\"input_hash\":"; json_string(out, stripe.input_hash);
        out << ",\"correction_hash\":"; json_string(out, stripe.correction_hash);
        out << ",\"correction_nonzero_count\":" << stripe.correction_nonzero_count;
        out << ",\"output_hash\":"; json_string(out, stripe.output_hash);
        out << ",\"hash_enabled\":" << (stripe.hash_enabled ? "true" : "false");
        cpu_interval_json(out, "dense", stripe.dense);
        out << '}';
    }
    out << ']';
#endif
    out << '}';
    return out.str();
#endif
}

namespace {

void json_string(std::ostringstream & out, std::string_view value) {
    out << '"';
    for (const char c : value) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    static constexpr char hex[] = "0123456789abcdef";
                    out << "\\u00" << hex[(static_cast<unsigned char>(c) >> 4) & 0xf]
                        << hex[static_cast<unsigned char>(c) & 0xf];
                } else out << c;
        }
    }
    out << '"';
}

void field(std::ostringstream & out, const char * name, uint64_t value) {
    out << ",\"" << name << "\":" << value;
}
void string_field(std::ostringstream & out, const char * name, std::string_view value) {
    out << ",\"" << name << "\":";
    json_string(out, value);
}
void nullable_string_field(std::ostringstream & out, const char * name, std::string_view value) {
    if (value.empty()) out << ",\"" << name << "\":null";
    else string_field(out, name, value);
}
void null_field(std::ostringstream & out, const char * name) {
    out << ",\"" << name << "\":null";
}

void prefix(std::ostringstream & out, const char * type,
            std::string_view source, std::string_view unit) {
    out << "{\"schema\":\"" << kCycleTelemetrySchema
        << "\",\"version\":" << kCycleTelemetryVersion
        << ",\"record_type\":\"" << type << "\",\"source\":";
    json_string(out, source);
    out << ",\"unit\":";
    json_string(out, unit);
}

#if LOG_DEBUG
void debug_field(std::ostringstream & out, const char * name, std::uint64_t value) {
    out << ' ' << name << '=' << value;
}

std::string serialize_im2p_debug_detail(const Im2pExecutionTelemetry & record) {
    std::ostringstream out;
    out << "IM2P_EXECUTION_TELEMETRY_DETAIL mode=" << record.mode;
    debug_field(out, "activation_bits", record.activation_bits);
    debug_field(out, "weight_bits", record.weight_bits);
    debug_field(out, "dim", record.dim);
    debug_field(out, "problem_i", record.problem_i);
    debug_field(out, "problem_j", record.problem_j);
    debug_field(out, "problem_k", record.problem_k);
    debug_field(out, "tile_i", record.tile_i);
    debug_field(out, "tile_j", record.tile_j);
    debug_field(out, "tile_k", record.tile_k);
    debug_field(out, "rtl_work_total_cycles", record.rtl_work_total_cycles);
    debug_field(out, "rtl_compute_cycles", record.rtl_compute_cycles);
    debug_field(out, "rtl_drain_cycles", record.rtl_drain_cycles);
    debug_field(out, "rtl_activation_wait_cycles", record.rtl_activation_wait_cycles);
    debug_field(out, "rtl_weight_wait_cycles", record.rtl_weight_wait_cycles);
    debug_field(out, "rtl_scale_wait_cycles", record.rtl_scale_wait_cycles);
    debug_field(out, "rtl_output_wait_cycles", record.rtl_output_wait_cycles);
    debug_field(out, "rtl_overlap_cycles", record.rtl_overlap_cycles);
    debug_field(out, "rtl_activation_overlap_cycles", record.rtl_activation_overlap_cycles);
    debug_field(out, "rtl_weight_overlap_cycles", record.rtl_weight_overlap_cycles);
    debug_field(out, "rtl_scale_overlap_cycles", record.rtl_scale_overlap_cycles);
    debug_field(out, "rtl_completed_output_works", record.rtl_completed_output_works);
    debug_field(out, "rtl_completed_fragments", record.rtl_completed_fragments);
    debug_field(out, "rtl_scheduler_groups_completed", record.rtl_scheduler_groups_completed);
    debug_field(out, "rtl_stripes_published", record.rtl_stripes_published);
    debug_field(out, "rtl_stripe_rows_published", record.rtl_stripe_rows_published);
    return out.str();
}
#endif

} // namespace

std::string serialize_cycle_telemetry(const CycleIntervalTelemetry & record) {
#if !LOG_CYCLE
    (void) record;
    return {};
#else
    std::string json = log::serialize_cycle_record(
        {record.layer.c_str(), record.op.c_str(), record.start, record.end, nullptr, 0, nullptr,
         record.source.c_str(), record.unit.c_str()});
    if (!json.empty() && json.back() == '\n') json.pop_back();
    return json;
#endif
}

std::string serialize_cycle_telemetry(const WsLoopTelemetry & record) {
#if !LOG_CYCLE
    (void) record;
    return {};
#else
    return log::serialize_ws_cycle_record({
        record.containing_interval_cycles, record.load_occupancy_cycles,
        record.execute_occupancy_cycles, record.store_occupancy_cycles,
        record.loop_occupancy_cycles, record.problem_i, record.problem_j, record.problem_k,
        record.tile_i, record.tile_j, record.tile_k, record.gemmini_outer_i,
        record.gemmini_outer_j, record.gemmini_outer_k, record.ws_inner_calls});
#endif
}

std::string serialize_cycle_telemetry(const Im2pExecutionTelemetry & record) {
#if !LOG_CYCLE
    (void) record;
    return {};
#else
    std::ostringstream out;
    if (record.residual_domain) {
        prefix(out, record.residual_aggregate
                        ? "IM2P_RMD_EXECUTION_TELEMETRY"
                        : "IM2P_RMD_STRIPE_TELEMETRY",
               "im2p_rmd_rtl", "rtl_cycle");
        string_field(out, "op", "rmd.im2p.execute");
        nullable_string_field(out, "layer", record.layer);
        field(out, "run_id", record.run_id);
        if (record.residual_aggregate) {
            null_field(out, "stripe_id");
            null_field(out, "slot");
        } else {
            field(out, "stripe_id", record.stripe_id);
            field(out, "slot", record.slot);
        }
        null_field(out, "node_id");
        null_field(out, "worker_id");
        if (!record.residual_aggregate) {
            field(out, "row_begin", record.row_begin);
            field(out, "row_end", record.row_end);
        }
        field(out, "rmd_dot_calls", record.rmd_dot_calls);
        field(out, "rmd_work_total_cycles", record.rtl_work_total_cycles);
        string_field(out, "clock_domain", "independent_rmd_simulator");
        out << ",\"additive\":false}";
        return out.str();
    }
    prefix(out, "IM2P_EXECUTION_TELEMETRY", "im2p_rtl", "rtl_cycle");
    string_field(out, "op", "im2p.execute");
    nullable_string_field(out, "layer", record.layer);
    field(out, "run_id", record.run_id);
    null_field(out, "stripe_id");
    null_field(out, "slot");
    null_field(out, "node_id");
    null_field(out, "worker_id");
    field(out, "rtl_work_total_cycles", record.rtl_work_total_cycles);
    out << '}';
    return out.str();
#endif
}

std::string serialize_cycle_telemetry(const Im2pStripeTelemetry & record) {
#if !LOG_CYCLE
    (void) record;
    return {};
#else
    std::ostringstream out;
    prefix(out, "IM2P_STRIPE_TELEMETRY", "im2p_rtl", "rtl_cycle");
    string_field(out, "op", "im2p.execute");
    nullable_string_field(out, "layer", record.layer);
    field(out, "run_id", record.run_id);
    field(out, "stripe_id", record.stripe_id);
    field(out, "slot", record.slot);
    null_field(out, "node_id");
    null_field(out, "worker_id");
    field(out, "row_begin", record.row_begin);
    field(out, "row_end", record.row_end);
    field(out, "publish_cycle", record.publish_cycle);
    field(out, "completion_cycle", record.completion_cycle);
    field(out, "latency_cycles", record.completion_cycle - record.publish_cycle);
    out << ",\"additive\":false}";
    return out.str();
#endif
}

std::string serialize_cycle_telemetry(const QuantizationStripeTelemetry & record) {
#if !LOG_CYCLE
    (void) record;
    return {};
#else
    std::ostringstream out;
    prefix(out, "QUANTIZATION_STRIPE_TELEMETRY", kNativeCycleSource, kNativeCycleUnit);
    string_field(out, "op", "exsia.quantize");
    nullable_string_field(out, "layer", record.layer);
    field(out, "run_id", record.run_id);
    field(out, "stripe_id", record.stripe_id);
    field(out, "slot", record.slot);
    null_field(out, "node_id");
    null_field(out, "worker_id");
    field(out, "row_begin", record.row_begin);
    field(out, "row_end", record.row_end);
    null_field(out, "start");
    null_field(out, "end");
    null_field(out, "delta");
    out << ",\"valid\":false";
    string_field(out, "reason", "structurally_cross_task");
    field(out, "start_ns", record.start_ns);
    field(out, "end_ns", record.end_ns);
    field(out, "duration_ns", record.end_ns - record.start_ns);
    out << ",\"overlaps_rtl\":true,\"additive\":false}";
    return out.str();
#endif
}

std::string serialize_cycle_telemetry(const RmdTelemetryRecord & record) {
    return serialize_rmd_telemetry(record);
}

std::string serialize_cycle_telemetry(const PipelineStripeTelemetry & record) {
#if !LOG_CYCLE
    (void) record;
    return {};
#else
    const bool valid = record.row_end >= record.row_begin &&
        record.queue_end_ns >= record.queue_start_ns &&
        record.dense_end_ns >= record.dense_start_ns &&
        record.rmd_end_ns >= record.rmd_start_ns &&
        record.residual_backend_end_ns >= record.residual_backend_start_ns &&
        record.compose_end_ns >= record.compose_start_ns &&
        record.finalize_end_ns >= record.finalize_start_ns;
    std::ostringstream out;
    out << "{\"schema\":\"" << kCycleTelemetrySchema
        << "\",\"version\":" << kCycleTelemetryVersion
        << ",\"record_type\":\"PIPELINE_STRIPE_SUMMARY\",\"source\":\"steady_clock\",\"unit\":\"nanosecond\"";
    string_field(out, "op", "matmul.pipeline");
    nullable_string_field(out, "layer", record.layer);
    field(out, "run_id", record.run_id);
    field(out, "stripe_id", record.stripe_id);
    field(out, "slot", record.slot);
    null_field(out, "node_id");
    null_field(out, "worker_id");
    field(out, "row_begin", record.row_begin);
    field(out, "row_end", record.row_end);
    field(out, "queue_start_ns", record.queue_start_ns);
    field(out, "queue_end_ns", record.queue_end_ns);
    field(out, "dense_start_ns", record.dense_start_ns);
    field(out, "dense_end_ns", record.dense_end_ns);
    field(out, "rmd_start_ns", record.rmd_start_ns);
    field(out, "rmd_end_ns", record.rmd_end_ns);
    field(out, "residual_backend_start_ns", record.residual_backend_start_ns);
    field(out, "residual_backend_end_ns", record.residual_backend_end_ns);
    field(out, "compose_start_ns", record.compose_start_ns);
    field(out, "compose_end_ns", record.compose_end_ns);
    field(out, "finalize_start_ns", record.finalize_start_ns);
    field(out, "finalize_end_ns", record.finalize_end_ns);
    out << ",\"host_stages\":{\"queue\":"
        << cycle::serialize_host_timing(record.queue_start_ns, record.queue_end_ns,
                                       record.queue_start_tid, record.queue_end_tid)
        << ",\"dense\":"
        << cycle::serialize_host_timing(record.dense_start_ns, record.dense_end_ns,
                                       record.dense_start_tid, record.dense_end_tid)
        << ",\"residual_backend\":"
        << cycle::serialize_host_timing(record.residual_backend_start_ns, record.residual_backend_end_ns,
                                       record.residual_backend_start_tid, record.residual_backend_end_tid)
        << ",\"compose\":"
        << cycle::serialize_host_timing(record.compose_start_ns, record.compose_end_ns,
                                       record.compose_start_tid, record.compose_end_tid)
        << ",\"finalize\":"
        << cycle::serialize_host_timing(record.finalize_start_ns, record.finalize_end_ns,
                                       record.finalize_start_tid, record.finalize_end_tid) << '}';
    out << ",\"valid\":" << (valid ? "true" : "false") << '}';
    return out.str();
#endif
}

void emit_cycle_telemetry(const CycleIntervalTelemetry & record) { log::cycle.write_json(serialize_cycle_telemetry(record)); }
void emit_cycle_telemetry(const WsLoopTelemetry & record) { log::cycle.write_json(serialize_cycle_telemetry(record)); }
void emit_cycle_telemetry(const Im2pExecutionTelemetry & record) {
    log::cycle.write_json(serialize_cycle_telemetry(record));
#if LOG_DEBUG
    if (!record.residual_domain) {
        const std::string detail = serialize_im2p_debug_detail(record);
        log::debug(record.layer.c_str(), "%s", detail.c_str());
    }
#endif
}
void emit_cycle_telemetry(const Im2pStripeTelemetry & record) { log::cycle.write_json(serialize_cycle_telemetry(record)); }
void emit_cycle_telemetry(const QuantizationStripeTelemetry & record) { log::cycle.write_json(serialize_cycle_telemetry(record)); }
void emit_cycle_telemetry(const PipelineStripeTelemetry & record) { log::cycle.write_json(serialize_cycle_telemetry(record)); }
void emit_cycle_telemetry(const RmdTelemetryRecord & record) { log::cycle.write_json(serialize_cycle_telemetry(record)); }

} // namespace ggml::gemmini
