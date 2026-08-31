#include "ggml-gemmini-matmul.hpp"

#include <sstream>
#include <string_view>
#include <utility>

namespace ggml::gemmini {
namespace detail {

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

MatmulCapturedStripe capture_collector_event(
        const quants::act::exsia::StripeReadyEvent & event,
        MatmulCaptureTiming timing) {
    MatmulCapturedStripe captured{};
    captured.run_id = event.run_id;
    captured.stripe_id = event.stripe_id;
    captured.slot = event.slot;
    captured.row_begin = event.row_begin;
    captured.row_end = event.row_end;
    captured.activation_metadata = event.activation_metadata;
#if GGML_GEMMINI_ENABLE_RMD
    captured.rmd_packet = event.rmd_packet;
    captured.direct_residual = event.direct_residual;
    if (event.rmd_packet != nullptr || event.direct_residual != nullptr) {
        timing.rmd_pack.nanoseconds = event.rmd_pack_ns;
        timing.rmd_pack.count = 1;
    }
#endif
    captured.la3_ns = event.local_end_ns >= event.local_start_ns ?
        event.local_end_ns - event.local_start_ns : 0;
    captured.sf1_ns = event.folding_end_ns >= event.folding_start_ns ?
        event.folding_end_ns - event.folding_start_ns : 0;
    captured.sf_mask_start_ns = event.mask_assembly_start_ns;
    captured.sf_mask_end_ns = event.mask_assembly_end_ns;
    captured.sf_exponent_start_ns = event.exponent_reduction_start_ns;
    captured.sf_exponent_end_ns = event.exponent_reduction_end_ns;
    captured.sf_folding_start_ns = event.folding_start_ns;
    captured.sf_folding_end_ns = event.folding_end_ns;
    captured.sf_commit_ns = event.folding_commit_ns;
    captured.timing = std::move(timing);
    return captured;
}

void apply_captured_stripe(
        const MatmulCapturedStripe & captured, MatmulJobMetrics & profile) {
    profile.run_id = captured.run_id;
    profile.stripe_id = captured.stripe_id;
    profile.slot = captured.slot;
    profile.row_begin = captured.row_begin;
    profile.row_end = captured.row_end;
    profile.la3_ns = captured.la3_ns;
    profile.sf1_ns = captured.sf1_ns;
    profile.sf_mask_start_ns = captured.sf_mask_start_ns;
    profile.sf_mask_end_ns = captured.sf_mask_end_ns;
    profile.sf_exponent_start_ns = captured.sf_exponent_start_ns;
    profile.sf_exponent_end_ns = captured.sf_exponent_end_ns;
    profile.sf_folding_start_ns = captured.sf_folding_start_ns;
    profile.sf_folding_end_ns = captured.sf_folding_end_ns;
    profile.sf_commit_ns = captured.sf_commit_ns;
    profile.la = {captured.la3_ns, captured.la3_ns != 0 ? 1U : 0U};
    profile.sf = {captured.sf1_ns, captured.sf1_ns != 0 ? 1U : 0U};
    profile.capture_copy = captured.timing.capture_copy;
    profile.producer_wait = captured.timing.producer_wait;
    profile.queue_insert = captured.timing.queue_insert;
    profile.rmd_pack = captured.timing.rmd_pack;
    profile.producer_wait_start_ns = captured.timing.producer_wait_start_ns;
    profile.producer_wait_end_ns = captured.timing.producer_wait_end_ns;
    profile.capture_queue_enqueue_ns = captured.timing.queued_ns;
    profile.telemetry_queue_tick = captured.timing.telemetry_queued_tick;
    profile.sf_handoff.nanoseconds = captured.sf1_ns + profile.handoff.nanoseconds;
    profile.sf_handoff.count = 1;
}

PipelineStripeTelemetry pipeline_stripe_telemetry(
        const char * layer, const MatmulJobMetrics & profile) {
    PipelineStripeTelemetry record{};
    record.layer = layer != nullptr ? layer : "";
    record.run_id = profile.run_id;
    record.stripe_id = profile.stripe_id;
    record.slot = profile.slot;
    record.row_begin = profile.row_begin;
    record.row_end = profile.row_end;
    record.queue_start_ns = profile.capture_queue_enqueue_ns;
    record.queue_end_ns = profile.ws_start_ns;
    record.dense_start_ns = profile.ws_start_ns;
    record.dense_end_ns = profile.ws_end_ns;
    record.rmd_start_ns = profile.rmd_start_ns;
    record.rmd_end_ns = profile.rmd_end_ns;
    record.compose_start_ns = profile.compose_start_ns;
    record.compose_end_ns = profile.compose_end_ns;
    record.finalize_start_ns = profile.finalize_start_ns;
    record.finalize_end_ns = profile.finalize_end_ns;
    return record;
}

} // namespace detail

std::string serialize_cycle_telemetry(const PipelineStripeTelemetry & record) {
#if !LOG_CYCLE
    (void) record;
    return {};
#else
    const bool valid = record.row_end >= record.row_begin &&
        record.queue_end_ns >= record.queue_start_ns &&
        record.dense_end_ns >= record.dense_start_ns &&
        record.rmd_end_ns >= record.rmd_start_ns &&
        record.compose_end_ns >= record.compose_start_ns &&
        record.finalize_end_ns >= record.finalize_start_ns;
    std::ostringstream out;
    out << "{\"schema\":\"" << kCycleTelemetrySchema
        << "\",\"version\":" << kCycleTelemetryVersion
        << ",\"record_type\":\"PIPELINE_STRIPE_SUMMARY\",\"source\":\"steady_clock\",\"unit\":\"nanosecond\"";
    detail::string_field(out, "op", "matmul.pipeline");
    detail::nullable_string_field(out, "layer", record.layer);
    detail::field(out, "run_id", record.run_id);
    detail::field(out, "stripe_id", record.stripe_id);
    detail::field(out, "slot", record.slot);
    detail::null_field(out, "node_id");
    detail::null_field(out, "worker_id");
    detail::field(out, "row_begin", record.row_begin);
    detail::field(out, "row_end", record.row_end);
    detail::field(out, "queue_start_ns", record.queue_start_ns);
    detail::field(out, "queue_end_ns", record.queue_end_ns);
    detail::field(out, "dense_start_ns", record.dense_start_ns);
    detail::field(out, "dense_end_ns", record.dense_end_ns);
    detail::field(out, "rmd_start_ns", record.rmd_start_ns);
    detail::field(out, "rmd_end_ns", record.rmd_end_ns);
    detail::field(out, "compose_start_ns", record.compose_start_ns);
    detail::field(out, "compose_end_ns", record.compose_end_ns);
    detail::field(out, "finalize_start_ns", record.finalize_start_ns);
    detail::field(out, "finalize_end_ns", record.finalize_end_ns);
    out << ",\"valid\":" << (valid ? "true" : "false") << '}';
    return out.str();
#endif
}

} // namespace ggml::gemmini
