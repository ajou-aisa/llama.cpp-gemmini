#include "ggml-gemmini-telemetry.hpp"
#include "ggml-gemmini-matmul.hpp"
#include <gemmini/log.hpp>

#include <sstream>
#include <string_view>

namespace ggml::gemmini {
namespace {

void prefix(std::ostringstream & out, const char * type,
            std::string_view source, std::string_view unit) {
    out << "{\"schema\":\"" << kCycleTelemetrySchema
        << "\",\"version\":" << kCycleTelemetryVersion
        << ",\"record_type\":\"" << type << "\",\"source\":";
    detail::json_string(out, source);
    out << ",\"unit\":";
    detail::json_string(out, unit);
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
        detail::string_field(out, "op", "rmd.im2p.execute");
        detail::nullable_string_field(out, "layer", record.layer);
        detail::field(out, "run_id", record.run_id);
        if (record.residual_aggregate) {
            detail::null_field(out, "stripe_id");
            detail::null_field(out, "slot");
        } else {
            detail::field(out, "stripe_id", record.stripe_id);
            detail::field(out, "slot", record.slot);
        }
        detail::null_field(out, "node_id");
        detail::null_field(out, "worker_id");
        if (!record.residual_aggregate) {
            detail::field(out, "row_begin", record.row_begin);
            detail::field(out, "row_end", record.row_end);
        }
        detail::field(out, "rmd_dot_calls", record.rmd_dot_calls);
        detail::field(out, "rmd_work_total_cycles", record.rtl_work_total_cycles);
        detail::string_field(out, "clock_domain", "independent_rmd_simulator");
        out << ",\"additive\":false}";
        return out.str();
    }
    prefix(out, "IM2P_EXECUTION_TELEMETRY", "im2p_rtl", "rtl_cycle");
    detail::string_field(out, "op", "im2p.execute");
    detail::nullable_string_field(out, "layer", record.layer);
    detail::field(out, "run_id", record.run_id);
    detail::null_field(out, "stripe_id");
    detail::null_field(out, "slot");
    detail::null_field(out, "node_id");
    detail::null_field(out, "worker_id");
    detail::field(out, "rtl_work_total_cycles", record.rtl_work_total_cycles);
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
    detail::string_field(out, "op", "im2p.execute");
    detail::nullable_string_field(out, "layer", record.layer);
    detail::field(out, "run_id", record.run_id);
    detail::field(out, "stripe_id", record.stripe_id);
    detail::field(out, "slot", record.slot);
    detail::null_field(out, "node_id");
    detail::null_field(out, "worker_id");
    detail::field(out, "row_begin", record.row_begin);
    detail::field(out, "row_end", record.row_end);
    detail::field(out, "publish_cycle", record.publish_cycle);
    detail::field(out, "completion_cycle", record.completion_cycle);
    detail::field(out, "latency_cycles", record.completion_cycle - record.publish_cycle);
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
    detail::string_field(out, "op", "exsia.quantize");
    detail::nullable_string_field(out, "layer", record.layer);
    detail::field(out, "run_id", record.run_id);
    detail::field(out, "stripe_id", record.stripe_id);
    detail::field(out, "slot", record.slot);
    detail::null_field(out, "node_id");
    detail::null_field(out, "worker_id");
    detail::field(out, "row_begin", record.row_begin);
    detail::field(out, "row_end", record.row_end);
    detail::null_field(out, "start");
    detail::null_field(out, "end");
    detail::null_field(out, "delta");
    out << ",\"valid\":false";
    detail::string_field(out, "reason", "structurally_cross_task");
    detail::field(out, "start_ns", record.start_ns);
    detail::field(out, "end_ns", record.end_ns);
    detail::field(out, "duration_ns", record.end_ns - record.start_ns);
    out << ",\"overlaps_rtl\":true,\"additive\":false}";
    return out.str();
#endif
}

std::string serialize_cycle_telemetry(const RmdTelemetryRecord & record) {
    return serialize_rmd_telemetry(record);
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
