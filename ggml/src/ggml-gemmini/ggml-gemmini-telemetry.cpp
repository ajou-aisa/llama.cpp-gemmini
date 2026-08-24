#include "ggml-gemmini-telemetry.hpp"
#include "ggml-gemmini-matmul.hpp"
#include <gemmini/log.hpp>

#include <sstream>
#include <string_view>

namespace ggml::gemmini {
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
                } else {
                    out << c;
                }
        }
    }
    out << '"';
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

void field(std::ostringstream & out, const char * name, std::uint64_t value) {
    out << ",\"" << name << "\":" << value;
}

void string_field(std::ostringstream & out, const char * name, std::string_view value) {
    out << ",\"" << name << "\":";
    json_string(out, value);
}

bool ordered(std::uint64_t start, std::uint64_t end) {
    return end >= start;
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
        {record.layer.c_str(), record.name.c_str(), record.start, record.end, nullptr, 0, nullptr,
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
    prefix(out, "IM2P_EXECUTION_TELEMETRY", "im2p_rtl", "rtl_cycle");
    string_field(out, "layer", record.layer);
    field(out, "rtl_work_total_cycles", record.rtl_work_total_cycles);
    out << '}';
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
        ordered(record.queue_start_ns, record.queue_end_ns) &&
        ordered(record.dense_start_ns, record.dense_end_ns) &&
        ordered(record.rmd_start_ns, record.rmd_end_ns) &&
        ordered(record.compose_start_ns, record.compose_end_ns) &&
        ordered(record.finalize_start_ns, record.finalize_end_ns);
    std::ostringstream out;
    prefix(out, "PIPELINE_STRIPE_SUMMARY", "steady_clock", "nanosecond");
    string_field(out, "layer", record.layer);
    field(out, "run_id", record.run_id);
    field(out, "stripe_id", record.stripe_id);
    field(out, "slot", record.slot);
    field(out, "row_begin", record.row_begin);
    field(out, "row_end", record.row_end);
    field(out, "queue_start_ns", record.queue_start_ns);
    field(out, "queue_end_ns", record.queue_end_ns);
    field(out, "dense_start_ns", record.dense_start_ns);
    field(out, "dense_end_ns", record.dense_end_ns);
    field(out, "rmd_start_ns", record.rmd_start_ns);
    field(out, "rmd_end_ns", record.rmd_end_ns);
    field(out, "compose_start_ns", record.compose_start_ns);
    field(out, "compose_end_ns", record.compose_end_ns);
    field(out, "finalize_start_ns", record.finalize_start_ns);
    field(out, "finalize_end_ns", record.finalize_end_ns);
    out << ",\"valid\":" << (valid ? "true" : "false") << '}';
    return out.str();
#endif
}

void emit_cycle_telemetry(const CycleIntervalTelemetry & record) {
    log::cycle.write_json(serialize_cycle_telemetry(record));
}
void emit_cycle_telemetry(const WsLoopTelemetry & record) {
    log::cycle.write_json(serialize_cycle_telemetry(record));
}
void emit_cycle_telemetry(const Im2pExecutionTelemetry & record) {
    log::cycle.write_json(serialize_cycle_telemetry(record));
#if LOG_DEBUG
    const std::string detail = serialize_im2p_debug_detail(record);
    log::debug(record.layer.c_str(), "%s", detail.c_str());
#endif
}
void emit_cycle_telemetry(const PipelineStripeTelemetry & record) {
    log::cycle.write_json(serialize_cycle_telemetry(record));
}
void emit_cycle_telemetry(const RmdTelemetryRecord & record) {
    log::cycle.write_json(serialize_cycle_telemetry(record));
}

} // namespace ggml::gemmini
