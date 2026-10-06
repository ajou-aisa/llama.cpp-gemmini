#include <gemmini/trace-context.hpp>
#include "../ggml-gemmini-utils/src/trace-metadata.hpp"
#include "ggml-gemmini-telemetry.hpp"
#include "matmul/types.hpp"
#include "matmul/detail.hpp"
#include "residual/rmd/rmd-executor.hpp"
#include <gemmini/log.hpp>

#include <gemmini/performance.hpp>
#include <algorithm>
#include <cstring>
#include <tuple>
#include <sstream>
#include <iterator>
#include <limits>
#include <string_view>
#include <utility>

namespace ggml::gemmini {

namespace detail {

MatmulCycleDrain::~MatmulCycleDrain() {
#if LOG_CYCLE && CYCLE_DETAIL
    (void)log::cycle.drain();
#endif
}

void record_matmul_cpu_wall(const MatmulCpuSample & start, const MatmulCpuSample & end) {
#if LOG_CYCLE && CYCLE_DETAIL
    performance::record_cpu_wall(start.ns, end.ns);
#else
    (void)start;
    (void)end;
#endif
}

void emit_matmul_cpu_interval(const char *              layer,
                              const char *              op,
                              const MatmulCpuSample &   start,
                              const MatmulCpuSample &   end,
                              bool                      operation_success,
                              const MatmulJobMetrics *  profile,
                              const MatmulCpuInterval * explicit_interval,
                              std::optional<uint64_t>   invocation_run_id) noexcept {
#if LOG_CYCLE
    try {
        log::CycleRecord record{
            layer, op, 0, 0, nullptr, 0, nullptr, kNativeCycleSource, kNativeCycleUnit};
        project_matmul_cpu_identity(record, profile, invocation_run_id);
        log::cycle.write_decorated(serialize_matmul_cpu_interval(
            record, start, end, operation_success, explicit_interval));
    } catch (...) {
        log::cycle.report_failure("matmul CPU interval");
    }
#else
    (void)layer;
    (void)op;
    (void)start;
    (void)end;
    (void)operation_success;
    (void)profile;
    (void)explicit_interval;
    (void)invocation_run_id;
#endif
}

void emit_rmd_stripe_metrics(const std::string &              layer,
                             const MatmulJobMetrics &         profile,
                             RmdBackend                       backend,
                             bool                             success,
                             const char *                     reason,
                             const rmd::RmdExecutionMetrics * metrics) noexcept {
#if LOG_CYCLE
    try {
        RmdStripeTelemetry record;
        record.layer = layer;
        if (profile.cpu_identity_mask & GEMMINI_CYCLE_HAS_RUN_ID)
            record.run_id = profile.run_id;
        if (profile.cpu_identity_mask & GEMMINI_CYCLE_HAS_SLOT)
            record.slot = profile.slot;
        record.stripe_id = profile.stripe_id;
        record.row_begin = profile.row_begin;
        record.row_end   = profile.row_end;
        record.backend   = backend == RmdBackend::cpu_direct ? "cpu_direct" : "compact";
        record.success   = success;
        record.reason    = reason;
        record.metrics   = metrics;
        emit_cycle_telemetry(record);
    } catch (...) {
        log::cycle.report_failure("RMD stripe metrics");
    }
#else
    (void)layer;
    (void)profile;
    (void)backend;
    (void)success;
    (void)reason;
    (void)metrics;
#endif
}

PipelineStripeTelemetry pipeline_stripe_telemetry(const char *             layer,
                                                  const MatmulJobMetrics & profile) {
    PipelineStripeTelemetry record{};
    record.layer                      = layer != nullptr ? layer : "";
    record.run_id                     = profile.run_id;
    record.stripe_id                  = profile.stripe_id;
    record.slot                       = profile.slot;
    record.row_begin                  = profile.row_begin;
    record.row_end                    = profile.row_end;
    record.queue_start_ns             = profile.capture_queue_enqueue_ns;
    record.queue_end_ns               = profile.capture_queue_dequeue_ns;
    record.queue_start_tid            = profile.queue_enqueue_tid;
    record.queue_end_tid              = profile.queue_dequeue_tid;
    record.dense_start_ns             = profile.ws_start_ns;
    record.dense_end_ns               = profile.ws_end_ns;
    record.dense_start_tid            = profile.ws_start_tid;
    record.dense_end_tid              = profile.ws_end_tid;
    record.rmd_start_ns               = profile.rmd_start_ns;
    record.rmd_end_ns                 = profile.rmd_end_ns;
    record.residual_backend_start_ns  = profile.backend_start_ns;
    record.residual_backend_end_ns    = profile.backend_end_ns;
    record.residual_backend_start_tid = profile.backend_start_tid;
    record.residual_backend_end_tid   = profile.backend_end_tid;
    record.compose_start_ns           = profile.merge_start_ns;
    record.compose_end_ns             = profile.merge_end_ns;
    record.compose_start_tid          = profile.merge_start_tid;
    record.compose_end_tid            = profile.merge_end_tid;
    record.finalize_start_ns          = profile.finalize_start_ns;
    record.finalize_end_ns            = profile.finalize_end_ns;
    record.finalize_start_tid         = profile.finalize_start_tid;
    record.finalize_end_tid           = profile.finalize_end_tid;
    return record;
}

} // namespace detail

namespace {
class ProofHash64 {
  public:
    void u8(uint8_t value) {
        value_ ^= value;
        value_ *= 1099511628211ULL;
    }
    void u32(uint32_t value) {
        for (unsigned shift = 0; shift < 32; shift += 8)
            u8(static_cast<uint8_t>(value >> shift));
    }
    void u64(uint64_t value) {
        for (unsigned shift = 0; shift < 64; shift += 8)
            u8(static_cast<uint8_t>(value >> shift));
    }
    std::string finish() const {
        static constexpr char hex[] = "0123456789abcdef";
        std::string           result(16, '0');
        for (size_t i = 0; i < result.size(); ++i) {
            result[15 - i] = hex[(value_ >> (i * 4)) & 0x0f];
        }
        return result;
    }

  private:
    uint64_t value_ = 1469598103934665603ULL;
};

using CanonicalResidual = std::tuple<size_t, size_t, int32_t>;

std::string hash_canonical_residuals(std::vector<CanonicalResidual> events) {
    std::sort(events.begin(), events.end());
    ProofHash64 hash;
    for (const auto & [row, k, residual] : events) {
        hash.u64(row);
        hash.u64(k);
        hash.u32(static_cast<uint32_t>(residual));
    }
    return hash.finish();
}
} // namespace

std::string rmd_input_hash(const residual::DirectStripePayload & payload) {
    std::vector<CanonicalResidual> events;
    events.reserve(payload.events.size());
    for (const residual::ResidualEvent & event : payload.events) {
        events.emplace_back(payload.row_begin + event.local_row, event.original_k, event.residual);
    }
    return hash_canonical_residuals(std::move(events));
}

std::string rmd_input_hash(const rmd::StripePacket & packet) {
    if (rmd::validate_packet(packet) != rmd::RmdStatus::success) {
        return {};
    }
    const rmd::BalancedRadixContract contract = rmd::balanced_radix_contract(packet.digit_bits);
    std::vector<CanonicalResidual>   events;
    for (size_t row = 0; row < packet.row_count; ++row) {
        for (const rmd::BlockDescriptor & block : packet.blocks) {
            for (size_t compact_k = 0; compact_k < block.compact_k_count; ++compact_k) {
                rmd::NativeBalancedDigits digits{};
                digits.radix         = contract.radix;
                digits.lane_capacity = contract.lane_capacity;
                for (uint8_t position = 0; position < block.active_lane_count; ++position) {
                    const uint8_t lane  = block.lane_ids[position];
                    int32_t       digit = 0;
                    if (lane >= contract.lane_capacity ||
                        rmd::read_packet_digit(packet, block, position, row, compact_k, digit) !=
                            rmd::RmdStatus::success) {
                        return {};
                    }
                    digits.digits[lane] = digit;
                    if (digit != 0) {
                        digits.active_lane_count = static_cast<uint8_t>(lane + 1);
                    }
                }
                int64_t residual = 0;
                if (rmd::compose_balanced_radix(digits, residual) != rmd::RmdStatus::success) {
                    return {};
                }
                if (residual != 0) {
                    const size_t k =
                        block.global_k_begin + packet.k_indices[block.k_index_offset + compact_k];
                    events.emplace_back(packet.row_begin + row, k, static_cast<int32_t>(residual));
                }
            }
        }
    }
    return hash_canonical_residuals(std::move(events));
}

std::string rmd_correction_hash(const rmd::Correction & correction) {
    ProofHash64 hash;
    if (const auto * integer = std::get_if<rmd::BlockScaledInt64Correction>(&correction)) {
        hash.u8(0);
        for (const rmd::OutputValue value : integer->values) {
            hash.u64(static_cast<uint64_t>(value));
        }
    } else {
        const auto * fully_scaled = std::get_if<rmd::FullyScaledFloat64Correction>(&correction);
        hash.u8(fully_scaled != nullptr ? 2 : 1);
        const auto & values = fully_scaled != nullptr
                                  ? fully_scaled->values
                                  : std::get<rmd::PreScaledFloat64Correction>(correction).values;
        for (const double value : values) {
            uint64_t bits = 0;
            static_assert(sizeof(bits) == sizeof(value), "FP64 proof hash requires 64-bit double");
            std::memcpy(&bits, &value, sizeof(bits));
            hash.u64(bits);
        }
    }
    return hash.finish();
}

std::string rmd_output_hash(const ggml_gemmini_args_t & args, size_t row_begin, size_t row_end) {
    ProofHash64  hash;
    const size_t row_stride = args.stride_f_out != 0 ? args.stride_f_out : args.J;
    const size_t col_stride = args.col_stride_f_out != 0 ? args.col_stride_f_out : 1;
    hash.u64(row_stride);
    hash.u64(col_stride);
    for (size_t row = row_begin; row < row_end; ++row) {
        for (size_t column = 0; column < args.J; ++column) {
            uint32_t    bits  = 0;
            const float value = args.f_out[row * row_stride + column * col_stride];
            static_assert(sizeof(bits) == sizeof(value), "FP32 proof hash requires 32-bit float");
            std::memcpy(&bits, &value, sizeof(bits));
            hash.u32(bits);
        }
    }
    return hash.finish();
}

std::string resolve_rmd_model_id(const char * environment_model_id, std::string_view model_arch) {
    return environment_model_id != nullptr ? std::string(environment_model_id)
                                           : std::string(model_arch);
}

RmdTelemetryCheckResult check_rmd_telemetry(const RmdTelemetryRecord & record,
                                            std::string_view           expected_units,
                                            bool                       comparison_mode) {
    if (record.schema != kRmdTelemetrySchema)
        return {RmdTelemetryCheckCode::malformed_schema, "malformed RMD telemetry schema"};
    if (record.version != kRmdTelemetryVersion)
        return {RmdTelemetryCheckCode::unsupported_version, "unsupported RMD telemetry version"};
    if ((record.units != "ticks" && record.units != "cycles") || record.units != expected_units)
        return {RmdTelemetryCheckCode::wrong_units, "telemetry timing units differ"};
    if (!record.work) {
        return comparison_mode ? RmdTelemetryCheckResult{RmdTelemetryCheckCode::zero_work,
                                                         "zero-work record is not comparable"}
                               : RmdTelemetryCheckResult{};
    }
    const bool cpu = record.backend == RmdBackend::cpu_direct;
    const bool exclusive =
        cpu ? record.counters.direct_events != 0 && record.counters.direct_calls != 0 &&
                  record.counters.packet_calls == 0 && record.counters.ws_calls == 0
            : record.counters.direct_events == 0 && record.counters.direct_calls == 0 &&
                  record.counters.packet_calls != 0 && record.counters.ws_calls != 0 &&
                  record.geometry.packet_count != 0;
    if (!exclusive)
        return {RmdTelemetryCheckCode::route_not_exclusive,
                "backend dispatch counters are not exclusive"};
    if (comparison_mode &&
        (!record.invocation_total.cycles || !record.timing.backend_service.cycles ||
         !record.timing.residual_total.cycles))
        return {RmdTelemetryCheckCode::invalid_timing,
                "comparison requires collected valid CPU intervals"};
    // Independent task/event counters are not an absolute timeline. Neither
    // backend service nor worker sums are bounded by the caller's CPU total.
#if CYCLE_DETAIL
    if (record.stripes.empty())
        return {RmdTelemetryCheckCode::missing_detail, "detail telemetry requires stripes"};
    for (const RmdTelemetryStripe & stripe : record.stripes) {
        if (stripe.row_begin >= stripe.row_end ||
            (comparison_mode &&
             (stripe.input_hash.size() != 16 || stripe.correction_hash.size() != 16 ||
              stripe.output_hash.size() != 16)))
            return {RmdTelemetryCheckCode::missing_detail,
                    "stripe attribution requires three fixed-width hashes"};
        // Raw attribution endpoints can belong to different event owners.
    }
#endif
    return {};
}

RmdTelemetryCheckResult compare_rmd_telemetry_proofs(const RmdTelemetryRecord & lhs,
                                                     const RmdTelemetryRecord & rhs) {
#if !CYCLE_DETAIL
    (void)lhs;
    (void)rhs;
    return {RmdTelemetryCheckCode::missing_detail, "proof comparison requires DETAIL"};
#else
    if (lhs.stripes.size() != rhs.stripes.size())
        return {RmdTelemetryCheckCode::input_hash_mismatch, "stripe proof cardinality differs"};
    for (size_t i = 0; i < lhs.stripes.size(); ++i) {
        const auto & a = lhs.stripes[i];
        const auto & b = rhs.stripes[i];
        if (a.input_hash.size() != 16 || a.correction_hash.size() != 16 ||
            a.output_hash.size() != 16 || b.input_hash.size() != 16 ||
            b.correction_hash.size() != 16 || b.output_hash.size() != 16)
            return {RmdTelemetryCheckCode::missing_detail,
                    "proof comparison requires three fixed-width hashes"};
        if (a.stripe_id != b.stripe_id || a.row_begin != b.row_begin || a.row_end != b.row_end ||
            a.input_hash != b.input_hash)
            return {RmdTelemetryCheckCode::input_hash_mismatch, "input hashes differ"};
        if (a.correction_hash != b.correction_hash)
            return {RmdTelemetryCheckCode::correction_hash_mismatch, "correction hashes differ"};
        if (a.correction_nonzero_count != b.correction_nonzero_count)
            return {RmdTelemetryCheckCode::correction_nonzero_count_mismatch,
                    "correction nonzero counts differ"};
        if (a.output_hash != b.output_hash)
            return {RmdTelemetryCheckCode::output_hash_mismatch, "output hashes differ"};
    }
    return {};
#endif
}

std::string serialize_matmul_cpu_interval(log::CycleRecord          record,
                                          const MatmulCpuSample &   start,
                                          const MatmulCpuSample &   end,
                                          bool                      operation_success,
                                          const MatmulCpuInterval * explicit_interval) {
    const auto interval = explicit_interval != nullptr ? *explicit_interval
                                                       : evaluate_matmul_cpu_interval(start, end);
    record.start        = start.value;
    record.end          = end.value;
    record.correlation  = start.correlation;
#if CYCLE_SIM
    if (start.exclusion.active || end.exclusion.active ||
        start.exclusion.epoch != end.exclusion.epoch) {
        record.cpu_service_exclusion = "functional_emulation";
        record.timing_interval_class = cycle::TimingIntervalClass::functional_emulation;
    }
#endif
    const auto cpu_sample = [](const MatmulCpuSample & sample) {
        gemmini_cpu_sample result{};
        result.trace            = sample.trace;
        result.ns               = sample.ns;
        result.tid              = sample.tid;
        result.thread_cpu_ns    = sample.thread_cpu_ns;
        result.thread_cpu_valid = sample.thread_cpu_valid;
        result.cpu_core         = sample.cpu_core;
        result.cpu_core_valid   = sample.cpu_core >= 0 ? 1 : 0;
#if defined(__linux__) && defined(__aarch64__)
        result.counter       = sample.native.value;
        result.native_valid  = sample.collected && sample.native.valid;
        result.native_reason = static_cast<uint8_t>(sample.native.reason);
        result.native_source = GEMMINI_CPU_COUNTER_THREAD_PERF;
        result.owner_token   = sample.native.owner_event_token;
        result.generation    = sample.native.generation;
#endif
        return result;
    };
#if CYCLE_DETAIL
    std::string tail =
        std::string(",\"cpu_measurement_version\":1,\"operation_success\":") +
        (operation_success ? "true" : "false") + ",\"additive\":" +
        (record.timing_interval_class == cycle::TimingIntervalClass::canonical_additive ? "true"
                                                                                        : "false") +
        ",\"host_timing\":" + cycle::serialize_host_timing(start.ns, end.ns, start.tid, end.tid) +
        ",\"native_cycles\":" + cycle::serialize_cpu_native(cpu_sample(start), cpu_sample(end)) +
        ",\"thread_cpu_timing\":" +
        cycle::serialize_thread_cpu_timing(
            {start.ns, start.tid, start.thread_cpu_ns, start.thread_cpu_valid},
            {end.ns, end.tid, end.thread_cpu_ns, end.thread_cpu_valid}) +
        cycle::serialize_cpu_timing_contract(cpu_sample(start), cpu_sample(end));
#else
    std::string tail =
        std::string(",\"operation_success\":") + (operation_success ? "true" : "false") +
        ",\"ns_start\":" + std::to_string(start.ns) + ",\"ns_end\":" + std::to_string(end.ns);
    if (start.tid != 0 && start.tid == end.tid) {
        tail += ",\"tid\":" + std::to_string(start.tid);
    } else {
        tail += ",\"tid_start\":" + std::to_string(start.tid) +
                ",\"tid_end\":" + std::to_string(end.tid);
    }
    tail += cycle::serialize_cpu_timing_contract(cpu_sample(start), cpu_sample(end));
#endif
    nlohmann::json facts;
#if CYCLE_DETAIL
    facts                = trace::cpu_timing_facts(cpu_sample(start), cpu_sample(end));
    facts["record_type"] = "CYCLE_INTERVAL";
    facts["source"]      = record.source ? record.source : "linux_perf_cpu_cycles";
    if (record.source && !*record.source)
        facts.erase("source");
#else
    facts = record.correlation.present || record.correlation.semantic_context
                ? nlohmann::json{{"record_type", "CYCLE_INTERVAL"}}
                : nlohmann::json{{"kind", "cycle"}};
#endif
    const auto origin =
        start.trace.flags & GEMMINI_TRACE_CAPTURED ? start.trace : gemmini_trace_capture();
    tail +=
        trace::metadata_suffix(trace::origin_fields(facts,
                                                    origin,
                                                    gemmini_trace_reserve_ids(1),
                                                    false,
                                                    (origin.flags & GEMMINI_TRACE_CAPTURED) != 0),
                               facts.contains("kind"));
    return log::serialize_checked_cycle_record(
        record,
        interval.cycles.has_value(),
        interval.reason.empty() ? nullptr : interval.reason.c_str(),
        interval.sample_reason.empty() ? nullptr : interval.sample_reason.c_str(),
        tail);
}

void project_matmul_cpu_identity(log::CycleRecord &       record,
                                 const MatmulJobMetrics * profile,
                                 std::optional<uint64_t>  invocation_run_id) {
    if (profile != nullptr) {
        record.identity_mask = profile->cpu_identity_mask;
        record.stripe_id     = profile->stripe_id;
        record.run_id        = profile->run_id;
        record.slot          = profile->slot;
    } else if (invocation_run_id.has_value()) {
        record.identity_mask = GEMMINI_CYCLE_HAS_RUN_ID;
        record.run_id        = *invocation_run_id;
    }
}

namespace {
void         json_string(std::ostringstream & out, std::string_view value);
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
    case MatmulOptionSource::build_default:
        return "build_default";
    case MatmulOptionSource::environment:
        return "environment";
    case MatmulOptionSource::explicit_override:
        return "explicit_override";
    }
    return "invalid";
}

MatmulCpuInterval aggregate_cpu_intervals(const std::vector<MatmulJobMetrics> & profiles,
                                          MatmulCpuInterval MatmulJobMetrics::* member) {
    MatmulCpuInterval result{{}, {}, {}, 0, 0, 0};
    uint64_t          sum = 0;
    for (const auto & profile : profiles) {
        const auto & item = profile.*member;
        ++result.count;
        if (item.reason == "not_applicable") {
            ++result.not_applicable_count;
        } else if (!item.cycles.has_value()) {
            if (result.reason.empty()) {
                result.reason        = item.reason;
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
        if (result.valid_count != 0)
            result.cycles = sum;
        else
            result.reason = "not_applicable";
    }
    return result;
}

void cpu_interval_json(std::ostringstream &      out,
                       const char *              key,
                       const MatmulCpuInterval & interval,
                       bool                      comma = true) {
    if (comma)
        out << ',';
    out << '"' << key << "\":";
    if (interval.cycles)
        out << *interval.cycles;
    else
        out << "null";
    out << ",\"" << key << "_valid\":" << (interval.cycles ? "true" : "false") << ",\"" << key
        << "_reason\":";
    if (interval.reason.empty())
        out << "null";
    else
        json_string(out, interval.reason);
    out << ",\"" << key << "_sample_reason\":";
    if (interval.sample_reason.empty())
        out << "null";
    else
        json_string(out, interval.sample_reason);
    out << ",\"" << key << "_count\":" << interval.count << ",\"" << key
        << "_valid_count\":" << interval.valid_count << ",\"" << key
        << "_not_applicable_count\":" << interval.not_applicable_count;
}
} // namespace

RmdTelemetryRecord make_rmd_telemetry_record(RmdBackend                backend,
                                             MatmulOptionSource        source,
                                             std::string               runtime_bundle_id,
                                             std::string               model_id,
                                             std::string               layer,
                                             uint64_t                  run_id,
                                             const MatmulCpuInterval & invocation_total,
                                             const std::vector<MatmulJobMetrics> & profiles) {
    RmdTelemetryRecord record{};
    record.runtime_bundle_id = std::move(runtime_bundle_id);
    record.model_id          = std::move(model_id);
    record.layer             = std::move(layer);
    record.run_id            = run_id;
    record.backend           = backend;
    record.source            = source;
    record.units             = cycle::units();
    record.invocation_total  = invocation_total;
    record.timing.prep       = aggregate_cpu_intervals(profiles, &MatmulJobMetrics::cpu_prep);
    record.timing.backend_service =
        aggregate_cpu_intervals(profiles, &MatmulJobMetrics::cpu_backend);
    record.timing.merge = aggregate_cpu_intervals(profiles, &MatmulJobMetrics::cpu_merge);
    record.timing.residual_total =
        aggregate_cpu_intervals(profiles, &MatmulJobMetrics::cpu_residual_total);
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
            const uint64_t elapsed =
                profile.telemetry_residual_start - profile.telemetry_queue_tick;
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
        if (record.timing.residual_start == 0 ||
            profile.telemetry_residual_start < record.timing.residual_start)
            record.timing.residual_start = profile.telemetry_residual_start;
#if CYCLE_DETAIL
        record.stripes.push_back({profile.stripe_id,
                                  profile.row_begin,
                                  profile.row_end,
                                  {profile.telemetry_dense_start,
                                   profile.telemetry_dense_end,
                                   profile.telemetry_residual_start,
                                   profile.telemetry_backend_start,
                                   profile.telemetry_backend_end,
                                   profile.telemetry_merge_start,
                                   profile.telemetry_merge_end,
                                   profile.telemetry_residual_end},
                                  profile.telemetry_input_hash,
                                  profile.telemetry_correction_hash,
                                  profile.telemetry_output_hash,
                                  profile.telemetry_correction_nonzero_count,
                                  profile.telemetry_hash_enabled,
                                  profile.cpu_dense});
#endif
    }
    record.work = backend == RmdBackend::cpu_direct ? record.counters.direct_calls != 0
                                                    : record.counters.packet_calls != 0;
    return record;
}

static std::string serialize_rmd_telemetry(const RmdTelemetryRecord & record,
                                           const std::string &        tail) {
#if !LOG_CYCLE
    (void)record;
    (void)tail;
    return {};
#else
    std::ostringstream out;
    out << "{\"schema\":";
    json_string(out, record.schema);
    out << ",\"version\":" << record.version << ",\"record_type\":\"RMD_BACKEND_TELEMETRY\"";
    out << ",\"source\":";
    json_string(out, telemetry_clock_source());
    out << ",\"unit\":";
    json_string(out, telemetry_unit_name(record.units));
    out << ",\"op\":\"rmd.execute\",\"layer\":";
    if (record.layer.empty())
        out << "null";
    else
        json_string(out, record.layer);
    out << ",\"run_id\":" << record.run_id
        << ",\"stripe_id\":null,\"slot\":null,\"node_id\":null,\"worker_id\":null"
        << ",\"runtime_bundle_id\":";
    json_string(out, record.runtime_bundle_id);
    out << ",\"model_id\":";
    json_string(out, record.model_id);
    out << ",\"backend\":";
    json_string(out, telemetry_backend_name(record.backend));
    out << ",\"option_source\":";
    json_string(out, telemetry_source_name(record.source));
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
        if (i != 0)
            out << ',';
        out << "{\"stripe_id\":" << stripe.stripe_id << ",\"row_begin\":" << stripe.row_begin
            << ",\"row_end\":" << stripe.row_end
            << ",\"stages\":{\"dense_start\":" << stripe.ordered_ticks[0]
            << ",\"dense_end\":" << stripe.ordered_ticks[1]
            << ",\"residual_start\":" << stripe.ordered_ticks[2]
            << ",\"backend_start\":" << stripe.ordered_ticks[3]
            << ",\"backend_end\":" << stripe.ordered_ticks[4]
            << ",\"merge_start\":" << stripe.ordered_ticks[5]
            << ",\"merge_end\":" << stripe.ordered_ticks[6]
            << ",\"residual_end\":" << stripe.ordered_ticks[7] << '}' << ",\"input_hash\":";
        json_string(out, stripe.input_hash);
        out << ",\"correction_hash\":";
        json_string(out, stripe.correction_hash);
        out << ",\"correction_nonzero_count\":" << stripe.correction_nonzero_count;
        out << ",\"output_hash\":";
        json_string(out, stripe.output_hash);
        out << ",\"hash_enabled\":" << (stripe.hash_enabled ? "true" : "false");
        cpu_interval_json(out, "dense", stripe.dense);
        out << '}';
    }
    out << ']';
#endif
    out << log::serialize_cpu_service_metadata("rmd.execute", "nonadditive_summary") << tail << '}';
    return out.str();
#endif
}

std::string serialize_rmd_telemetry(const RmdTelemetryRecord & record) {
    return serialize_rmd_telemetry(record, {});
}

namespace {

void json_string(std::ostringstream & out, std::string_view value) {
    out << '"';
    for (const char c : value) {
        switch (c) {
        case '\\':
            out << "\\\\";
            break;
        case '"':
            out << "\\\"";
            break;
        case '\b':
            out << "\\b";
            break;
        case '\f':
            out << "\\f";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                static constexpr char hex[] = "0123456789abcdef";
                out << "\\u00" << hex[(static_cast<unsigned char>(c) >> 4) & 0xf]
                    << hex[static_cast<unsigned char>(c) & 0xf];
            } else
                out << c;
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
    if (value.empty())
        out << ",\"" << name << "\":null";
    else
        string_field(out, name, value);
}
void null_field(std::ostringstream & out, const char * name) {
    out << ",\"" << name << "\":null";
}

void prefix(std::ostringstream & out,
            const char *         type,
            std::string_view     source,
            std::string_view     unit) {
    out << "{\"schema\":\"" << kCycleTelemetrySchema << "\",\"version\":" << kCycleTelemetryVersion
        << ",\"record_type\":\"" << type << "\",\"source\":";
    json_string(out, source);
    out << ",\"unit\":";
    json_string(out, unit);
}

void device_diagnostics(std::ostringstream & out, const Im2pExecutionTelemetry & record) {
    if (!record.provider_stats)
        return;
    string_field(out, "execution_id", cycle::host_execution_id());
    const auto & stats = *record.provider_stats;
    out << ",\"device\":{\"additive\":false,\"counter_bits\":64";
    string_field(out, "backend", record.backend);
    string_field(out, "clock_domain", record.clock_domain);
    if (!record.numerical_contract.empty())
        string_field(out, "numerical_contract", record.numerical_contract);
    if (!record.scale_mode.empty())
        string_field(out, "scale_mode", record.scale_mode);
    if (record.vector_op)
        field(out, "vector_op", *record.vector_op);
    if (record.output_domain)
        field(out, "output_domain", *record.output_domain);
    field(out, "activation_bits", record.activation_bits);
    field(out, "weight_bits", record.weight_bits);
    field(out, "dim", record.dim);
    field(out, "problem_i", record.problem_i);
    field(out, "problem_j", record.problem_j);
    field(out, "problem_k", record.problem_k);
    string_field(out, "mode", record.mode);
    string_field(out, "counter_semantics", "independent_observations");
    out << ",\"counters\":{\"additive\":false";
#define IM2P_COUNTER(member) field(out, #member, stats.member)
    IM2P_COUNTER(rtl_work_total_cycles);
    IM2P_COUNTER(rtl_activation_read_requests);
    IM2P_COUNTER(rtl_weight_read_requests);
    IM2P_COUNTER(rtl_scale_read_requests);
    IM2P_COUNTER(rtl_output_write_requests);
    IM2P_COUNTER(rtl_output_write_responses);
    IM2P_COUNTER(rtl_activation_wait_cycles);
    IM2P_COUNTER(rtl_weight_wait_cycles);
    IM2P_COUNTER(rtl_scale_wait_cycles);
    IM2P_COUNTER(rtl_output_wait_cycles);
    IM2P_COUNTER(rtl_stripe_host_wait_cycles);
    IM2P_COUNTER(rtl_drain_cycles);
    IM2P_COUNTER(rtl_weight_preload_cycles);
    IM2P_COUNTER(rtl_same_block_scale_hits);
    IM2P_COUNTER(rtl_next_scale_hits);
    IM2P_COUNTER(rtl_scale_demand_misses);
    IM2P_COUNTER(rtl_compute_cycles);
    IM2P_COUNTER(rtl_overlap_cycles);
    IM2P_COUNTER(rtl_activation_overlap_cycles);
    IM2P_COUNTER(rtl_weight_overlap_cycles);
    IM2P_COUNTER(rtl_scale_overlap_cycles);
    IM2P_COUNTER(rtl_completed_fragments);
    IM2P_COUNTER(rtl_completed_output_works);
    IM2P_COUNTER(rtl_scheduler_groups_completed);
    IM2P_COUNTER(rtl_stripes_published);
    IM2P_COUNTER(rtl_stripe_rows_published);
    IM2P_COUNTER(rtl_weight_bank_activations);
    IM2P_COUNTER(rtl_cross_stripe_overlap_cycles);
    IM2P_COUNTER(rtl_lookahead_prepared);
    IM2P_COUNTER(rtl_first_publish_cycle);
    IM2P_COUNTER(rtl_first_activation_read_cycle);
    IM2P_COUNTER(rtl_first_weight_read_cycle);
    IM2P_COUNTER(rtl_weight_preload_cycle);
    IM2P_COUNTER(rtl_lookahead_weight_requests);
    IM2P_COUNTER(rtl_lookahead_weight_reuse_hits);
    IM2P_COUNTER(rtl_first_scale_read_cycle);
    IM2P_COUNTER(rtl_lookahead_scale_requests);
    IM2P_COUNTER(rtl_lookahead_scale_reuses);
    IM2P_COUNTER(rtl_current_scheduler_group_completion_cycle);
    IM2P_COUNTER(rtl_lookahead_ready_cycle);
    IM2P_COUNTER(rtl_lookahead_start_cycle);
#undef IM2P_COUNTER
    for (const char * name :
         {"scu_execution_cycles", "saturation_events", "overflow_events", "metadata_bytes"}) {
        null_field(out, name);
        string_field(out, (std::string(name) + "_reason").c_str(), "provider_counter_unavailable");
    }
    out << "}}";
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

static std::string serialize_cycle_telemetry_impl(const CycleIntervalTelemetry & record,
                                                  const std::string &            tail) {
#if !LOG_CYCLE
    (void)record;
    (void)tail;
    return {};
#else
    std::string json = log::serialize_cycle_record({record.layer.c_str(),
                                                    record.op.c_str(),
                                                    record.start,
                                                    record.end,
                                                    nullptr,
                                                    0,
                                                    nullptr,
                                                    record.source.c_str(),
                                                    record.unit.c_str()},
                                                   tail);
    if (!json.empty() && json.back() == '\n')
        json.pop_back();
    return json;
#endif
}

static std::string serialize_cycle_telemetry_impl(const WsLoopTelemetry & record,
                                                  const std::string &     tail) {
#if !LOG_CYCLE
    (void)record;
    (void)tail;
    return {};
#else
    return log::serialize_ws_cycle_record({record.containing_interval_cycles,
                                           record.load_occupancy_cycles,
                                           record.execute_occupancy_cycles,
                                           record.store_occupancy_cycles,
                                           record.loop_occupancy_cycles,
                                           record.problem_i,
                                           record.problem_j,
                                           record.problem_k,
                                           record.tile_i,
                                           record.tile_j,
                                           record.tile_k,
                                           record.gemmini_outer_i,
                                           record.gemmini_outer_j,
                                           record.gemmini_outer_k,
                                           record.ws_inner_calls},
                                          tail);
#endif
}

static std::string serialize_cycle_telemetry_impl(const Im2pExecutionTelemetry & record,
                                                  const std::string &            tail) {
#if !LOG_CYCLE
    (void)record;
    (void)tail;
    return {};
#else
    std::ostringstream out;
    if (record.residual_domain) {
        prefix(out,
               record.residual_aggregate ? "IM2P_RMD_EXECUTION_TELEMETRY"
                                         : "IM2P_RMD_STRIPE_TELEMETRY",
               record.backend.empty() || record.backend == "im2p_sim" ? "im2p_rmd_rtl"
                                                                      : record.backend,
               "rtl_cycle");
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
        string_field(out,
                     "clock_domain",
                     record.clock_domain.empty() ? "independent_rmd_simulator"
                                                 : record.clock_domain);
        device_diagnostics(out, record);
        out << ",\"additive\":false" << tail << '}';
        return out.str();
    }
    prefix(out,
           "IM2P_EXECUTION_TELEMETRY",
           record.backend.empty() || record.backend == "im2p_sim" ? "im2p_rtl" : record.backend,
           "rtl_cycle");
    string_field(out, "op", "im2p.execute");
    nullable_string_field(out, "layer", record.layer);
    field(out, "run_id", record.run_id);
    null_field(out, "stripe_id");
    null_field(out, "slot");
    null_field(out, "node_id");
    null_field(out, "worker_id");
    field(out, "rtl_work_total_cycles", record.rtl_work_total_cycles);
    device_diagnostics(out, record);
    out << tail << '}';
    return out.str();
#endif
}

static std::string serialize_cycle_telemetry_impl(const Im2pStripeTelemetry & record,
                                                  const std::string &         tail) {
#if !LOG_CYCLE
    (void)record;
    (void)tail;
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
    out << ",\"additive\":false" << tail << '}';
    return out.str();
#endif
}

static std::string serialize_cycle_telemetry_impl(const QuantizationStripeTelemetry & record,
                                                  const std::string &                 tail) {
#if !LOG_CYCLE || !CYCLE_DETAIL
    (void)record;
    (void)tail;
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
    string_field(out, "execution_id", cycle::host_execution_id());
    out << ",\"host_clock\":\"steady_clock\",\"overlaps_rtl\":null,"
           "\"overlaps_rtl_reason\":\"independent_clock_domains\",\"additive\":false"
        << tail << '}';
    return out.str();
#endif
}

static std::string serialize_cycle_telemetry_impl(const RmdTelemetryRecord & record,
                                                  const std::string &        tail) {
    return serialize_rmd_telemetry(record, tail);
}

static std::string serialize_cycle_telemetry_impl(const RmdStripeTelemetry & record,
                                                  const std::string &        tail) {
#if !LOG_CYCLE
    (void)record;
    (void)tail;
    return {};
#else
    std::ostringstream out;
    prefix(out, "RMD_STRIPE_TELEMETRY", "host_observation", "mixed");
    string_field(out, "execution_id", cycle::host_execution_id());
    string_field(out, "op", "rmd.stripe");
    nullable_string_field(out, "layer", record.layer);
    if (record.run_id)
        field(out, "run_id", *record.run_id);
    else
        null_field(out, "run_id");
    field(out, "stripe_id", record.stripe_id);
    if (record.slot)
        field(out, "slot", *record.slot);
    else
        null_field(out, "slot");
    null_field(out, "node_id");
    null_field(out, "worker_id");
    field(out, "row_begin", record.row_begin);
    field(out, "row_end", record.row_end);
    string_field(out, "backend", record.backend);
    out << ",\"additive\":false,\"operation_success\":" << (record.success ? "true" : "false")
        << ",\"valid\":" << (record.success && record.metrics ? "true" : "false");
    nullable_string_field(out, "reason", record.metrics ? record.reason : "missing_metrics");
    if (!record.metrics) {
        out << ",\"metrics\":null" << tail << '}';
        return out.str();
    }
    const auto & m = *record.metrics;
    out << ",\"metrics\":{\"additive\":false";
    string_field(out, "digit_observation_scope", "signed_radix_input_decomposition");
    const auto compact_field = [&](const char * name, uint64_t value) {
        if (record.backend != "cpu_direct")
            field(out, name, value);
        else {
            null_field(out, name);
            string_field(out, (std::string(name) + "_reason").c_str(), "not_applicable_cpu_direct");
        }
    };
#define RMD_FIELD(member) field(out, #member, m.member)
#define RMD_COMPACT_FIELD(member) compact_field(#member, m.member)
    RMD_FIELD(residual_nnz);
    RMD_FIELD(digit_bits);
    const auto observed = [&](const char * name, uint64_t value) {
        if (m.residual_observations_valid)
            field(out, name, value);
        else {
            null_field(out, name);
            string_field(
                out, (std::string(name) + "_reason").c_str(), "input_observation_unavailable");
        }
    };
    for (const auto & entry :
         {std::pair{"residual_min", m.residual_min}, std::pair{"residual_max", m.residual_max}}) {
        if (m.residual_observations_valid && m.residual_nnz)
            out << ",\"" << entry.first << "\":" << entry.second;
        else {
            null_field(out, entry.first);
            string_field(out,
                         (std::string(entry.first) + "_reason").c_str(),
                         m.residual_observations_valid ? "no_nonzero_residual"
                                                       : "input_observation_unavailable");
        }
    }
    observed("required_planes", m.required_planes);
    observed("digit_nnz", m.digit_nnz);
    if (m.active_original_rows_valid)
        field(out, "active_original_rows", m.active_original_rows);
    else {
        null_field(out, "active_original_rows");
        string_field(out, "active_original_rows_reason", "input_observation_unavailable");
    }
    RMD_FIELD(original_rows);
    RMD_FIELD(logical_k);
    RMD_FIELD(logical_j);
    RMD_FIELD(array_dim);
    RMD_FIELD(original_rows_after_pruning);
    RMD_COMPACT_FIELD(lane_rows_before_pruning);
    RMD_COMPACT_FIELD(lane_rows_after_pruning);
    RMD_COMPACT_FIELD(group_rows_padded);
    RMD_COMPACT_FIELD(group_active_k_count);
    RMD_COMPACT_FIELD(group_padded_k_count);
    RMD_FIELD(source_residual_macs);
    RMD_COMPACT_FIELD(useful_digit_macs);
    RMD_FIELD(issued_mac_capacity);
    RMD_FIELD(activation_payload_bytes);
    RMD_FIELD(metadata_host_bytes);
    RMD_FIELD(gathered_weight_host_bytes);
    RMD_FIELD(correction_bytes);
    RMD_FIELD(logical_dot_result_bytes);
    RMD_COMPACT_FIELD(block_scale_values_bytes);
    RMD_FIELD(final_scale_values_bytes);
    RMD_FIELD(final_output_store_bytes);
    RMD_FIELD(direct_event_count);
    RMD_FIELD(direct_call_count);
    RMD_FIELD(packet_call_count);
    RMD_FIELD(ws_call_count);
    RMD_FIELD(im2p_dot_calls);
    RMD_COMPACT_FIELD(active_blocks);
    RMD_COMPACT_FIELD(active_lanes);
    RMD_COMPACT_FIELD(compact_k_count);
    RMD_COMPACT_FIELD(padded_k_count);
    RMD_FIELD(physical_tile_count);
    RMD_FIELD(matmul_call_count);
    RMD_FIELD(lane_group_count);
    RMD_FIELD(baseline_stacked_i_tile_count);
    RMD_FIELD(stacked_i_tile_count);
    RMD_FIELD(packet_bytes);
    RMD_FIELD(compressed_output_values);
    RMD_FIELD(block_padding_zeros);
    RMD_FIELD(row_padding_zeros);
    RMD_FIELD(j_padding_zeros);
    RMD_FIELD(weight_values_gathered);
    RMD_FIELD(weight_baseline_address_resolutions);
    RMD_FIELD(weight_address_resolutions);
#undef RMD_FIELD
#undef RMD_COMPACT_FIELD
    out << "},\"host_stages\":{";
    constexpr const char * names[] = {"preparation",
                                      "weight_gather",
                                      "block_scale_metadata",
                                      "dot_output_accumulate",
                                      "block_scale_apply",
                                      "radix_reconstruct_combine",
                                      "final_metadata",
                                      "final_scale_combine_stage",
                                      "output_store"};
    static_assert(std::size(names) == static_cast<size_t>(rmd::RmdHostStage::count));
    for (size_t i = 0; i < std::size(names); ++i) {
        const auto & stage = m.host_stages[i];
        if (i)
            out << ',';
        out << '"' << names[i] << "\":{\"additive\":false,\"calls\":" << stage.calls;
        const auto sample =
            [&](const char * name, uint64_t value, bool valid, const char * reason) {
                if (valid)
                    field(out, name, value);
                else
                    null_field(out, name);
                out << ",\"" << name << "_valid\":" << (valid ? "true" : "false");
                nullable_string_field(out,
                                      (std::string(name) + "_reason").c_str(),
                                      valid              ? ""
                                      : stage.calls == 0 ? "no_samples"
                                      : reason           ? reason
                                                         : "invalid_sample");
            };
#if CYCLE_DETAIL
        sample("wall_ns", stage.wall_ns, stage.calls && stage.wall_valid, "invalid_host_interval");
        sample("thread_cpu_ns",
               stage.cpu.thread_cpu_ns,
               stage.calls && stage.cpu.interval_count == stage.calls &&
                   stage.cpu.thread_cpu_valid_count == stage.calls && !stage.cpu.thread_cpu_reason,
               stage.cpu.thread_cpu_reason);
#endif
        sample("native_cycles",
               stage.cpu.cycles,
               stage.calls && stage.cpu.interval_count == stage.calls &&
                   stage.cpu.cycles_valid_count == stage.calls && !stage.cpu.cycles_reason,
               stage.cpu.cycles_reason);
        out << '}';
    }
    out << '}' << tail << '}';
    return out.str();
#endif
}

static std::string serialize_cycle_telemetry_impl(const PipelineStripeTelemetry & record,
                                                  const std::string &             tail) {
#if !LOG_CYCLE || !CYCLE_DETAIL
    (void)record;
    (void)tail;
    return {};
#else
    const bool valid =
        record.row_end >= record.row_begin && record.queue_end_ns >= record.queue_start_ns &&
        record.dense_end_ns >= record.dense_start_ns && record.rmd_end_ns >= record.rmd_start_ns &&
        record.residual_backend_end_ns >= record.residual_backend_start_ns &&
        record.compose_end_ns >= record.compose_start_ns &&
        record.finalize_end_ns >= record.finalize_start_ns;
    std::ostringstream out;
    out << "{\"schema\":\"" << kCycleTelemetrySchema << "\",\"version\":" << kCycleTelemetryVersion
        << ",\"record_type\":\"PIPELINE_STRIPE_SUMMARY\",\"source\":\"steady_clock\",\"unit\":"
           "\"nanosecond\"";
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
        << cycle::serialize_host_timing(record.queue_start_ns,
                                        record.queue_end_ns,
                                        record.queue_start_tid,
                                        record.queue_end_tid)
        << ",\"dense\":"
        << cycle::serialize_host_timing(record.dense_start_ns,
                                        record.dense_end_ns,
                                        record.dense_start_tid,
                                        record.dense_end_tid)
        << ",\"residual_backend\":"
        << cycle::serialize_host_timing(record.residual_backend_start_ns,
                                        record.residual_backend_end_ns,
                                        record.residual_backend_start_tid,
                                        record.residual_backend_end_tid)
        << ",\"compose\":"
        << cycle::serialize_host_timing(record.compose_start_ns,
                                        record.compose_end_ns,
                                        record.compose_start_tid,
                                        record.compose_end_tid)
        << ",\"finalize\":"
        << cycle::serialize_host_timing(record.finalize_start_ns,
                                        record.finalize_end_ns,
                                        record.finalize_start_tid,
                                        record.finalize_end_tid)
        << '}';
    out << ",\"valid\":" << (valid ? "true" : "false")
        << log::serialize_cpu_service_metadata("matmul.pipeline", "nonadditive_summary") << tail
        << '}';
    return out.str();
#endif
}
std::string serialize_cycle_telemetry(const CycleIntervalTelemetry & record) {
    return serialize_cycle_telemetry_impl(record, {});
}
std::string serialize_cycle_telemetry(const WsLoopTelemetry & record) {
    return serialize_cycle_telemetry_impl(record, {});
}
std::string serialize_cycle_telemetry(const Im2pExecutionTelemetry & record) {
    return serialize_cycle_telemetry_impl(record, {});
}
std::string serialize_cycle_telemetry(const Im2pStripeTelemetry & record) {
    return serialize_cycle_telemetry_impl(record, {});
}
std::string serialize_cycle_telemetry(const QuantizationStripeTelemetry & record) {
    return serialize_cycle_telemetry_impl(record, {});
}
std::string serialize_cycle_telemetry(const RmdTelemetryRecord & record) {
    return serialize_cycle_telemetry_impl(record, {});
}
std::string serialize_cycle_telemetry(const RmdStripeTelemetry & record) {
    return serialize_cycle_telemetry_impl(record, {});
}
std::string serialize_cycle_telemetry(const PipelineStripeTelemetry & record) {
    return serialize_cycle_telemetry_impl(record, {});
}

[[maybe_unused]] static std::string telemetry_tail(const nlohmann::json & facts) {
    const auto origin = gemmini_trace_capture();
    return trace::metadata_suffix(trace::origin_fields(facts, origin, gemmini_trace_reserve_ids(1)),
                                  facts.contains("kind"));
}

void emit_cycle_telemetry(const CycleIntervalTelemetry & record) {
#if LOG_CYCLE
    const auto origin = gemmini_trace_capture();
    const auto id     = gemmini_trace_reserve_ids(1);
    log::cycle.write_decorated(log::serialize_cycle_record({record.layer.c_str(),
                                                            record.op.c_str(),
                                                            record.start,
                                                            record.end,
                                                            nullptr,
                                                            0,
                                                            nullptr,
                                                            record.source.c_str(),
                                                            record.unit.c_str()},
                                                           origin,
                                                           id));
#else
    (void)record;
#endif
}
void emit_cycle_telemetry(const WsLoopTelemetry & record) {
#if LOG_CYCLE
    log::cycle.write_decorated(
        serialize_cycle_telemetry_impl(record,
                                       telemetry_tail({{"record_type", "WS_LOOP_TELEMETRY"},
                                                       {"source", "gemmini_hw_counter"},
                                                       {"domain", "gemmini_hw_unknown"}})));
#else
    (void)record;
#endif
}
void emit_cycle_telemetry(const Im2pExecutionTelemetry & record) {
#if LOG_CYCLE
    nlohmann::json facts = {{"record_type",
                             record.residual_domain
                                 ? (record.residual_aggregate ? "IM2P_RMD_EXECUTION_TELEMETRY"
                                                              : "IM2P_RMD_STRIPE_TELEMETRY")
                                 : "IM2P_EXECUTION_TELEMETRY"},
                            {"source",
                             record.backend.empty() || record.backend == "im2p_sim"
                                 ? (record.residual_domain ? "im2p_rmd_rtl" : "im2p_rtl")
                                 : record.backend}};
    facts[record.residual_domain ? "rmd_work_total_cycles" : "rtl_work_total_cycles"] =
        record.rtl_work_total_cycles;
    if (record.residual_domain)
        facts["clock_domain"] =
            record.clock_domain.empty() ? "independent_rmd_simulator" : record.clock_domain;
    log::cycle.write_decorated(serialize_cycle_telemetry_impl(record, telemetry_tail(facts)));
#endif
#if LOG_DEBUG
    if (!record.residual_domain) {
        const std::string detail = serialize_im2p_debug_detail(record);
        log::debug(record.layer.c_str(), "%s", detail.c_str());
    }
#endif
}
void emit_cycle_telemetry(const Im2pStripeTelemetry & record) {
#if LOG_CYCLE
    log::cycle.write_decorated(serialize_cycle_telemetry_impl(
        record,
        telemetry_tail({{"record_type", "IM2P_STRIPE_TELEMETRY"},
                        {"source", "im2p_rtl"},
                        {"publish_cycle", record.publish_cycle},
                        {"completion_cycle", record.completion_cycle},
                        {"latency_cycles", record.completion_cycle - record.publish_cycle}})));
#else
    (void)record;
#endif
}
void emit_cycle_telemetry(const QuantizationStripeTelemetry & record) {
#if LOG_CYCLE && CYCLE_DETAIL
    log::cycle.write_decorated(serialize_cycle_telemetry_impl(
        record,
        telemetry_tail({{"record_type", "QUANTIZATION_STRIPE_TELEMETRY"},
                        {"source", kNativeCycleSource},
                        {"delta", nullptr},
                        {"start", nullptr},
                        {"end", nullptr},
                        {"valid", false},
                        {"reason", "structurally_cross_task"}})));
#else
    (void)record;
#endif
}
void emit_cycle_telemetry(const PipelineStripeTelemetry & record) {
#if LOG_CYCLE && CYCLE_DETAIL
    log::cycle.write_decorated(serialize_cycle_telemetry_impl(
        record, telemetry_tail({{"record_type", "PIPELINE_STRIPE_SUMMARY"}})));
#else
    (void)record;
#endif
}
void emit_cycle_telemetry(const RmdTelemetryRecord & record) {
#if LOG_CYCLE
    log::cycle.write_decorated(serialize_cycle_telemetry_impl(
        record,
        telemetry_tail({{"record_type", "RMD_BACKEND_TELEMETRY"},
                        {"source", telemetry_clock_source()},
                        {"backend", telemetry_backend_name(record.backend)}})));
#else
    (void)record;
#endif
}
void emit_cycle_telemetry(const RmdStripeTelemetry & record) {
#if LOG_CYCLE
    log::cycle.write_decorated(
        serialize_cycle_telemetry_impl(record,
                                       telemetry_tail({{"record_type", "RMD_STRIPE_TELEMETRY"},
                                                       {"source", "host_observation"},
                                                       {"backend", record.backend}})));
#else
    (void)record;
#endif
}

} // namespace ggml::gemmini
