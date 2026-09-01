#define GGML_GEMMINI_MATMUL_IMPLEMENTATION 1
#include "ggml-gemmini-matmul.hpp"
#include "quants/common/weight_reader.hpp"

#include <gemmini/log.hpp>
#include <gemmini/cycle_reader.hpp>
#if CYCLE_DETAIL && defined(__linux__) && defined(__aarch64__)
#include <gemmini/log.h>
#include "../ggml-gemmini-utils/src/cycle_reader_internal.h"
#endif

#include <cstdio>
#include <algorithm>
#include <string>
#include <sstream>
#include <cstring>
#include <tuple>

namespace ggml::gemmini {
namespace {
void telemetry_json_string(std::ostringstream & out, std::string_view value) {
    out << '"';
    for (const char c : value) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << c; break;
        }
    }
    out << '"';
}
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
}

namespace {
class ProofHash64 {
public:
    void u8(uint8_t value) {
        value_ ^= value;
        value_ *= 1099511628211ULL;
    }
    void u32(uint32_t value) {
        for (unsigned shift = 0; shift < 32; shift += 8) u8(static_cast<uint8_t>(value >> shift));
    }
    void u64(uint64_t value) {
        for (unsigned shift = 0; shift < 64; shift += 8) u8(static_cast<uint8_t>(value >> shift));
    }
    std::string finish() const {
        static constexpr char hex[] = "0123456789abcdef";
        std::string result(16, '0');
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
}

std::string rmd_input_hash(const residual::DirectStripePayload & payload) {
    std::vector<CanonicalResidual> events;
    events.reserve(payload.events.size());
    for (const residual::ResidualEvent & event : payload.events) {
        events.emplace_back(payload.row_begin + event.local_row,
                            event.original_k, event.residual);
    }
    return hash_canonical_residuals(std::move(events));
}

std::string rmd_input_hash(const rmd::StripePacket & packet) {
    if (rmd::validate_packet(packet) != rmd::RmdStatus::success) {
        return {};
    }
    const rmd::BalancedRadixContract contract =
        rmd::balanced_radix_contract(packet.digit_bits);
    std::vector<CanonicalResidual> events;
    for (size_t row = 0; row < packet.row_count; ++row) {
        for (const rmd::BlockDescriptor & block : packet.blocks) {
            for (size_t compact_k = 0; compact_k < block.compact_k_count; ++compact_k) {
                rmd::NativeBalancedDigits digits{};
                digits.radix = contract.radix;
                digits.lane_capacity = contract.lane_capacity;
                for (uint8_t position = 0; position < block.active_lane_count; ++position) {
                    const uint8_t lane = block.lane_ids[position];
                    int32_t digit = 0;
                    if (lane >= contract.lane_capacity ||
                        rmd::read_packet_digit(packet, block, position, row,
                                               compact_k, digit) != rmd::RmdStatus::success) {
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
                    const size_t k = block.global_k_begin +
                        packet.k_indices[block.k_index_offset + compact_k];
                    events.emplace_back(packet.row_begin + row, k,
                                        static_cast<int32_t>(residual));
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
        hash.u8(1);
        for (const double value : std::get<rmd::PreScaledFloat64Correction>(correction).values) {
            uint64_t bits = 0;
            static_assert(sizeof(bits) == sizeof(value), "FP64 proof hash requires 64-bit double");
            std::memcpy(&bits, &value, sizeof(bits));
            hash.u64(bits);
        }
    }
    return hash.finish();
}

std::string rmd_output_hash(const ggml_gemmini_args_t & args,
                            size_t row_begin, size_t row_end) {
    ProofHash64 hash;
    const size_t row_stride = args.stride_f_out != 0 ? args.stride_f_out : args.J;
    const size_t col_stride = args.col_stride_f_out != 0 ? args.col_stride_f_out : 1;
    hash.u64(row_stride);
    hash.u64(col_stride);
    for (size_t row = row_begin; row < row_end; ++row) {
        for (size_t column = 0; column < args.J; ++column) {
            uint32_t bits = 0;
            const float value = args.f_out[row * row_stride + column * col_stride];
            static_assert(sizeof(bits) == sizeof(value), "FP32 proof hash requires 32-bit float");
            std::memcpy(&bits, &value, sizeof(bits));
            hash.u32(bits);
        }
    }
    return hash.finish();
}

std::string resolve_rmd_model_id(const char * environment_model_id,
                                 std::string_view model_arch) {
    return environment_model_id != nullptr ? std::string(environment_model_id)
                                           : std::string(model_arch);
}

RmdTelemetryCheckResult check_rmd_telemetry(const RmdTelemetryRecord & record,
                                             std::string_view expected_units,
                                             bool comparison_mode) {
    if (record.schema != kRmdTelemetrySchema)
        return {RmdTelemetryCheckCode::malformed_schema, "malformed RMD telemetry schema"};
    if (record.version != kRmdTelemetryVersion)
        return {RmdTelemetryCheckCode::unsupported_version, "unsupported RMD telemetry version"};
    if ((record.units != "ticks" && record.units != "cycles") || record.units != expected_units)
        return {RmdTelemetryCheckCode::wrong_units, "telemetry timing units differ"};
    if (!record.work) {
        return comparison_mode
            ? RmdTelemetryCheckResult{RmdTelemetryCheckCode::zero_work, "zero-work record is not comparable"}
            : RmdTelemetryCheckResult{};
    }
    const bool cpu = record.backend == RmdBackend::cpu_direct;
    const bool exclusive = cpu
        ? record.counters.direct_events != 0 && record.counters.direct_calls != 0 &&
          record.counters.packet_calls == 0 && record.counters.ws_calls == 0
        : record.counters.direct_events == 0 && record.counters.direct_calls == 0 &&
          record.counters.packet_calls != 0 && record.counters.ws_calls != 0 &&
          record.geometry.packet_count != 0;
    if (!exclusive)
        return {RmdTelemetryCheckCode::route_not_exclusive, "backend dispatch counters are not exclusive"};
    if (record.timing.residual_total < record.timing.backend_service)
        return {RmdTelemetryCheckCode::invalid_timing, "residual total does not contain backend service"};
    if (record.timing.dense_end > record.timing.residual_start)
        return {RmdTelemetryCheckCode::ordering_violation, "dense end follows residual start"};
#if CYCLE_DETAIL
    if (record.stripes.empty())
        return {RmdTelemetryCheckCode::missing_detail, "detail telemetry requires stripes"};
    for (const RmdTelemetryStripe & stripe : record.stripes) {
        if (stripe.row_begin >= stripe.row_end || stripe.input_hash.size() != 16 ||
            stripe.correction_hash.size() != 16 || stripe.output_hash.size() != 16)
            return {RmdTelemetryCheckCode::missing_detail,
                    "stripe attribution requires three fixed-width hashes"};
        for (size_t i = 1; i < stripe.ordered_ticks.size(); ++i) {
            if (stripe.ordered_ticks[i] < stripe.ordered_ticks[i - 1])
                return {RmdTelemetryCheckCode::ordering_violation, "stripe stages are not ordered"};
        }
    }
#endif
    return {};
}

RmdTelemetryCheckResult compare_rmd_telemetry_proofs(
        const RmdTelemetryRecord & lhs, const RmdTelemetryRecord & rhs) {
#if !CYCLE_DETAIL
    (void) lhs; (void) rhs;
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

RmdTelemetryRecord make_rmd_telemetry_record(
        RmdBackend backend, MatmulOptionSource source,
        std::string runtime_bundle_id, std::string model_id, std::string layer,
        uint64_t run_id,
        uint64_t invocation_total, const std::vector<MatmulJobMetrics> & profiles) {
    RmdTelemetryRecord record{};
    record.runtime_bundle_id = std::move(runtime_bundle_id);
    record.model_id = std::move(model_id);
    record.layer = std::move(layer);
    record.run_id = run_id;
    record.backend = backend;
    record.source = source;
    record.units = cycle::units();
    record.invocation_total = invocation_total;
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
        const auto elapsed = [](uint64_t begin, uint64_t end) {
            return end >= begin ? end - begin : uint64_t{0};
        };
        record.timing.prep += elapsed(
            profile.telemetry_residual_start, profile.telemetry_backend_start);
        record.timing.backend_service += elapsed(
            profile.telemetry_backend_start, profile.telemetry_backend_end);
        record.timing.merge += elapsed(profile.telemetry_merge_start, profile.telemetry_merge_end);
#if !(defined(__linux__) && defined(__aarch64__))
        if (profile.telemetry_queue_tick != 0) {
            record.timing.queue += elapsed(
                profile.telemetry_queue_tick, profile.telemetry_residual_start);
        }
#endif
        record.timing.residual_total += elapsed(
            profile.telemetry_residual_start, profile.telemetry_residual_end);
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
            profile.telemetry_correction_nonzero_count});
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
    out << "{\"schema\":"; telemetry_json_string(out, record.schema);
    out << ",\"version\":" << record.version << ",\"record_type\":\"RMD_BACKEND_TELEMETRY\"";
    out << ",\"source\":"; telemetry_json_string(out, telemetry_clock_source());
    out << ",\"unit\":"; telemetry_json_string(out, telemetry_unit_name(record.units));
    out << ",\"op\":\"rmd.execute\",\"layer\":";
    if (record.layer.empty()) out << "null"; else telemetry_json_string(out, record.layer);
    out << ",\"run_id\":" << record.run_id
        << ",\"stripe_id\":null,\"slot\":null,\"node_id\":null,\"worker_id\":null"
        << ",\"runtime_bundle_id\":";
    telemetry_json_string(out, record.runtime_bundle_id);
    out << ",\"model_id\":"; telemetry_json_string(out, record.model_id);
    out << ",\"backend\":"; telemetry_json_string(out, telemetry_backend_name(record.backend));
    out << ",\"option_source\":"; telemetry_json_string(out, telemetry_source_name(record.source));
    out << ",\"work\":" << (record.work ? "true" : "false")
        << ",\"invocation_total\":" << record.invocation_total
        << ",\"dispatch\":{\"direct_events\":" << record.counters.direct_events
        << ",\"direct_calls\":" << record.counters.direct_calls
        << ",\"packet_calls\":" << record.counters.packet_calls
        << ",\"ws_calls\":" << record.counters.ws_calls << "}"
        << ",\"timing\":{\"prep\":" << record.timing.prep
        << ",\"backend_service\":" << record.timing.backend_service
        << ",\"merge\":" << record.timing.merge
        << ",\"residual_total\":" << record.timing.residual_total
#if defined(__linux__) && defined(__aarch64__)
        << ",\"queue\":null,\"queue_reason\":\"structurally_cross_task\""
#else
        << ",\"queue\":" << record.timing.queue
#endif
        << ",\"dense_end\":" << record.timing.dense_end
        << ",\"residual_start\":" << record.timing.residual_start << "}"
        << ",\"geometry\":{\"packet_count\":" << record.geometry.packet_count
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
            << ",\"input_hash\":"; telemetry_json_string(out, stripe.input_hash);
        out << ",\"correction_hash\":"; telemetry_json_string(out, stripe.correction_hash);
        out << ",\"correction_nonzero_count\":" << stripe.correction_nonzero_count;
        out << ",\"output_hash\":"; telemetry_json_string(out, stripe.output_hash); out << '}';
    }
    out << ']';
#endif
    out << '}';
    return out.str();
#endif
}

}

#include "quants/act/quantize.hpp"
#include "quants/act/dispatch.hpp"
#include "residual/rmd/rmd-builder.hpp"
#include "residual/direct/direct-builder.hpp"
#include "residual/direct/direct-executor.hpp"

#include <gemmini.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <new>
#include <sstream>
#include <cstring>
#include <tuple>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace ggml::gemmini {

#if CYCLE_DETAIL && defined(__linux__) && defined(__aarch64__)
namespace {
gemmini_native_cycle_sample_internal project_native_sample(
        const cycle::NativeCycleSample & sample) {
    return {sample.value, static_cast<uint8_t>(sample.valid),
            static_cast<uint8_t>(sample.reason),
            GEMMINI_NATIVE_CYCLE_SOURCE_LINUX_PERF_CPU_CYCLES,
            sample.owner_event_token, sample.generation};
}
}
#endif

namespace test_detail {
#if defined(GGML_GEMMINI_TESTING)
struct AtomicMatmulTestCounters {
    std::atomic<uint64_t> execution_constructions{0};
    std::atomic<uint64_t> allocation_attempts{0};
    std::atomic<uint64_t> dense_dispatches{0};
    std::atomic<uint64_t> residual_dispatches{0};
    std::atomic<uint64_t> hardware_dispatches{0};
    std::atomic<uint64_t> fallback_dispatches{0};
    std::atomic<bool> fail_output_stage_allocation{false};
};

AtomicMatmulTestCounters counters;

static void increment(std::atomic<uint64_t> & counter) {
    counter.fetch_add(1, std::memory_order_relaxed);
}

static void observe_execution_construction() { increment(counters.execution_constructions); }
static void observe_allocation_attempt() { increment(counters.allocation_attempts); }
static void observe_dense_dispatch() { increment(counters.dense_dispatches); }
static void observe_residual_dispatch() { increment(counters.residual_dispatches); }
static void observe_backend_dispatch(bool fallback) {
    increment(fallback ? counters.fallback_dispatches : counters.hardware_dispatches);
}
#else
static void observe_execution_construction() {}
static void observe_allocation_attempt() {}
static void observe_dense_dispatch() {}
static void observe_residual_dispatch() {}
static void observe_backend_dispatch(bool) {}
#endif
}

#if defined(GGML_GEMMINI_TESTING)
void test_reset_matmul_counters() {
    test_detail::counters.execution_constructions.store(0, std::memory_order_relaxed);
    test_detail::counters.allocation_attempts.store(0, std::memory_order_relaxed);
    test_detail::counters.dense_dispatches.store(0, std::memory_order_relaxed);
    test_detail::counters.residual_dispatches.store(0, std::memory_order_relaxed);
    test_detail::counters.hardware_dispatches.store(0, std::memory_order_relaxed);
    test_detail::counters.fallback_dispatches.store(0, std::memory_order_relaxed);
    test_detail::counters.fail_output_stage_allocation.store(false, std::memory_order_relaxed);
}

void test_inject_output_stage_allocation_failure() {
    test_detail::counters.fail_output_stage_allocation.store(true, std::memory_order_relaxed);
}

MatmulTestCounters test_matmul_counters() {
    return {
        test_detail::counters.execution_constructions.load(std::memory_order_relaxed),
        test_detail::counters.allocation_attempts.load(std::memory_order_relaxed),
        test_detail::counters.dense_dispatches.load(std::memory_order_relaxed),
        test_detail::counters.residual_dispatches.load(std::memory_order_relaxed),
        test_detail::counters.hardware_dispatches.load(std::memory_order_relaxed),
        test_detail::counters.fallback_dispatches.load(std::memory_order_relaxed),
    };
}
#endif

namespace {

bool checked_offset(size_t row, size_t stride, size_t & offset) {
    if (stride != 0 && row > std::numeric_limits<size_t>::max() / stride) {
        return false;
    }
    offset = row * stride;
    return true;
}

bool row_invariant_activation(const ggml_gemmini_args_t & args) {
    const auto & storage = args.act_quant.storage();
    return std::holds_alternative<quants::act::NoneMeta>(storage) ||
        std::holds_alternative<quants::act::tensor::Meta>(storage);
}

bool supports_row_slice_activation(const ggml_gemmini_args_t & args) {
    const auto & storage = args.act_quant.storage();
    return row_invariant_activation(args) ||
        std::holds_alternative<quants::act::exsia::Meta>(storage) ||
        std::holds_alternative<quants::act::token::Meta>(storage) ||
        std::holds_alternative<quants::act::block::Meta>(storage) ||
        std::holds_alternative<quants::act::stripe::Meta>(storage);
}

bool uses_baseline_channel_route(const ggml_gemmini_args_t & args) {
    return args.weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_channel ||
        args.weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_channel_dense_sidecar;
}

GemminiGeometryResult resolve_geometry(ggml_gemmini_args_t & args) {
    if (args.tile_I == 0 || args.tile_J == 0 || args.tile_K == 0) {
        gemmini_set_tile_ws(&args);
    }
    return make_gemmini_geometry(
        {{args.I, args.J, args.K}, {args.tile_I, args.tile_J, args.tile_K}, DIM});
}

bool valid_matmul_shape(const ggml_gemmini_args_t & args) {
    return args.I != 0 && args.J != 0 && args.K != 0 && args.f_out != nullptr &&
        (args.A.valid() || args.A_fp32 != nullptr) &&
        ((args.A_fp32 == nullptr) == (args.B_fp32 == nullptr));
}

bool valid_activation_metadata(const ggml_gemmini_args_t & args) {
    if (std::holds_alternative<quants::act::NoneMeta>(args.act_quant.storage())) {
        return !args.A.valid() && args.A_fp32 != nullptr && args.B_fp32 != nullptr;
    }
    if (args.I > std::numeric_limits<size_t>::max() - args.activation_row_offset) {
        return false;
    }
    const quants::act::ActivationMetadataView metadata(
        args, args.activation_row_offset, args.activation_row_offset + args.I);
    float scale = 0.0f;
    for (size_t row = 0; metadata.valid() && row < args.I; ++row) {
        if (!metadata.scale(row, scale)) {
            return false;
        }
    }
    return metadata.valid();
}

bool finite_float_bits(const float * value) {
    static_assert(sizeof(float) == sizeof(uint32_t));
    static_assert(std::numeric_limits<float>::is_iec559);
    uint32_t bits = 0;
    std::memcpy(&bits, static_cast<const void *>(value), sizeof(bits));
    constexpr uint32_t exponent_mask = UINT32_C(0x7f800000);
    return (bits & exponent_mask) != exponent_mask;
}

bool finite_output(const ggml_gemmini_args_t & args) {
    const size_t row_stride = args.stride_f_out != 0 ? args.stride_f_out : args.J;
    const size_t col_stride = args.col_stride_f_out != 0 ? args.col_stride_f_out : 1;
    for (size_t row = 0; row < args.I; ++row) {
        for (size_t col = 0; col < args.J; ++col) {
            if (!finite_float_bits(
                    &args.f_out[row * row_stride + col * col_stride])) {
                return false;
            }
        }
    }
    return true;
}

baseline_activation_quant_t baseline_activation_for(const ggml_gemmini_args_t & args) {
    const auto & storage = args.act_quant.storage();
    if (std::holds_alternative<quants::act::tensor::Meta>(storage)) {
        return baseline_activation_quant_t::TENSOR;
    }
    if (std::holds_alternative<quants::act::token::Meta>(storage)) {
        return baseline_activation_quant_t::TOKEN;
    }
    if (std::holds_alternative<quants::act::block::Meta>(storage) ||
        std::holds_alternative<quants::act::stripe::Meta>(storage)) {
        return baseline_activation_quant_t::BLOCK;
    }
    return baseline_activation_quant_t::EXSIA;
}

MatMulStatus execute_native_matched_cpu_dense(ggml_gemmini_args_t & args) {
    using namespace quants::wreader;
    using namespace quants::wroute;

    const WeightRoutePlan plan =
        resolve_weight_route_plan(args, WeightScaleInfoMode::CommonOutput);
    if (!plan.valid ||
        weight_route_status(plan, WeightExecutionPath::CpuDirect) !=
            WeightRouteStatus::Success ||
        validate(args, plan) != WeightReaderStatus::Success ||
        args.f_out == nullptr || !args.A.valid() ||
        !route_covers_k(plan, args.K)) {
        return MatMulStatus::invalid_contract;
    }
    if (args.I != 0 && args.K > std::numeric_limits<size_t>::max() / args.I) {
        return MatMulStatus::invalid_arguments;
    }

    std::vector<float> activation;
    try {
        activation.resize(args.I * args.K);
    } catch (const std::bad_alloc &) {
        return MatMulStatus::invalid_arguments;
    } catch (const std::length_error &) {
        return MatMulStatus::invalid_arguments;
    }

    ggml_gemmini_args_t dense_args = args;
    std::visit([](auto & meta) {
        using T = std::decay_t<decltype(meta)>;
        if constexpr (!std::is_same_v<T, quants::act::NoneMeta>) {
            meta.rmd_packets.clear();
            meta.direct_residuals.clear();
        }
    }, dense_args.act_quant.storage());
    if (!quants::dequantize_activation(
            activation.data(), args.K, 1, args.I, args.K, dense_args)) {
        return MatMulStatus::invalid_contract;
    }

    const size_t block_size =
        plan.scales.block_size != 0 ? plan.scales.block_size : args.block_size_k;
    if (block_size == 0 ||
        (args.J != 0 && args.K > std::numeric_limits<size_t>::max() / args.J)) {
        return MatMulStatus::invalid_contract;
    }

    std::vector<float> weights;
    try {
        weights.resize(args.J * args.K);
    } catch (const std::bad_alloc &) {
        return MatMulStatus::invalid_arguments;
    } catch (const std::length_error &) {
        return MatMulStatus::invalid_arguments;
    }

    for (size_t j = 0; j < args.J; ++j) {
        for (size_t block_begin = 0; block_begin < args.K;
             block_begin += block_size) {
            const size_t block_index = block_begin / block_size;
            const WeightScaleResult scale =
                read_scale_validated(args, plan, j, block_index);
            if (!scale.ok()) {
                return MatMulStatus::invalid_contract;
            }
            double weight_scale = 0.0;
            if (scale.domain == WeightScaleDomain::FloatingBlock) {
                weight_scale = scale.floating_block_scale;
            } else if (scale.domain ==
                       WeightScaleDomain::IntegerBlockTimesColumn) {
                weight_scale =
                    static_cast<double>(scale.integer_block_scale) *
                    static_cast<double>(scale.column_scale);
            } else {
                return MatMulStatus::invalid_contract;
            }
            const size_t block_end =
                std::min(args.K, block_begin + block_size);
            for (size_t k = block_begin; k < block_end; ++k) {
                const WeightCodeResult code =
                    read_code_validated(args, plan, j, k);
                if (!code.ok()) {
                    return MatMulStatus::invalid_contract;
                }
                weights[j * args.K + k] =
                    static_cast<float>(static_cast<double>(code.value) *
                                       weight_scale);
            }
        }
    }

    std::vector<float> bias;
    const float * bias_data = nullptr;
    if (args.D != nullptr) {
        if (args.I != 0 &&
            args.J > std::numeric_limits<size_t>::max() / args.I) {
            return MatMulStatus::invalid_arguments;
        }
        const size_t source_stride = args.sD != 0 ? args.sD : args.J;
        if (!args.repeating_bias && args.I > 1 &&
            source_stride >
                (std::numeric_limits<size_t>::max() - (args.J - 1)) /
                    (args.I - 1)) {
            return MatMulStatus::invalid_arguments;
        }
        try {
            bias.resize(args.I * args.J);
        } catch (const std::bad_alloc &) {
            return MatMulStatus::invalid_arguments;
        } catch (const std::length_error &) {
            return MatMulStatus::invalid_arguments;
        }
        for (size_t i = 0; i < args.I; ++i) {
            const size_t source_row = args.repeating_bias ? 0 : i;
            for (size_t j = 0; j < args.J; ++j) {
                const size_t source_index = source_row * source_stride + j;
                bias[i * args.J + j] = args.low_D
                    ? static_cast<float>(
                          static_cast<const elem_t *>(args.D)[source_index]) *
                          static_cast<float>(args.scale_D)
                    : static_cast<float>(
                          static_cast<const acc_t *>(args.D)[source_index]) *
                          static_cast<float>(args.scale_D);
            }
        }
        bias_data = bias.data();
    }

    const size_t output_row_stride =
        args.stride_f_out != 0 ? args.stride_f_out : args.J;
    const size_t output_col_stride =
        args.col_stride_f_out != 0 ? args.col_stride_f_out : 1;

    if (output_col_stride == 1) {
        matmul_cpu_fp(false, true, args.I, args.J, args.K,
                      activation.data(), weights.data(), bias_data, args.f_out,
                      args.K, args.K, args.J, output_row_stride);
        return MatMulStatus::success;
    }

    if (args.I != 0 && args.J > std::numeric_limits<size_t>::max() / args.I) {
        return MatMulStatus::invalid_arguments;
    }
    std::vector<float> contiguous_output;
    try {
        contiguous_output.resize(args.I * args.J);
    } catch (const std::bad_alloc &) {
        return MatMulStatus::invalid_arguments;
    } catch (const std::length_error &) {
        return MatMulStatus::invalid_arguments;
    }
    matmul_cpu_fp(false, true, args.I, args.J, args.K,
                  activation.data(), weights.data(), bias_data,
                  contiguous_output.data(), args.K, args.K, args.J, args.J);
    for (size_t i = 0; i < args.I; ++i) {
        for (size_t j = 0; j < args.J; ++j) {
            args.f_out[i * output_row_stride + j * output_col_stride] =
                contiguous_output[i * args.J + j];
        }
    }
    return MatMulStatus::success;
}

MatMulStatus execute_dense(ggml_gemmini_args_t &args) {
    if (args.A_fp32 != nullptr || args.B_fp32 != nullptr) {
        if (args.A_fp32 == nullptr || args.B_fp32 == nullptr || args.f_out == nullptr) {
            return MatMulStatus::invalid_contract;
        }
        test_detail::observe_dense_dispatch();
        test_detail::observe_backend_dispatch(true);
        matmul_cpu_fp(false, true, args.I, args.J, args.K,
                      args.A_fp32, args.B_fp32, nullptr, args.f_out,
                      args.sA, args.sB, args.col_stride_f_out, args.stride_f_out);
        return MatMulStatus::success;
    }
    test_detail::observe_dense_dispatch();
    test_detail::observe_backend_dispatch(args.tiled_matmul_type == CPU);
    if (quants::wroute::is_native_matched_width_format(args)) {
        if (args.tiled_matmul_type != CPU) {
            return MatMulStatus::unsupported;
        }
        return execute_native_matched_cpu_dense(args);
    } else if (uses_baseline_channel_route(args)) {
        tiled_matmul_auto_baseline(&args, baseline_activation_for(args),
                                   baseline_weight_quant_t::CHANNEL);
    } else if (args.weight_i8_scale_active) {
        tiled_matmul_auto_baseline(&args, baseline_activation_for(args),
                                   baseline_weight_quant_t::TENSOR);
    } else {
        using Format = ggml_gemmini_args_t::im2p_weight_format_t;
        switch (args.weight_format) {
            case Format::q8_h1:
            case Format::q8_h2:
            case Format::q8_hp1:
            case Format::q8_hp2:
                break;
            case Format::q8_h0:
            case Format::q8_channel:
            case Format::q8_channel_dense_sidecar:
            default:
                return MatMulStatus::unsupported;
        }
        if (args.tiled_matmul_type != CPU && args.tiled_matmul_type != WS) {
            return MatMulStatus::unsupported;
        }
        tiled_matmul_auto_im2p(&args);
    }
    return MatMulStatus::success;
}

MatMulStatus execute_stripe(ggml_gemmini_args_t args, MatMulStripe stripe, size_t stripe_id,
                            bool metadata_is_local = false) {
    const size_t input_stride = args.sA ? args.sA : args.K;
    const size_t output_stride = args.stride_f_out ? args.stride_f_out : args.J;
    size_t input_offset = 0;
    size_t output_offset = 0;
    if (!checked_offset(stripe.row_begin, input_stride, input_offset) ||
        !checked_offset(stripe.row_begin, output_stride, output_offset)) {
        return MatMulStatus::invalid_arguments;
    }

    if (args.tile_I == 0 || args.tile_J == 0 || args.tile_K == 0) {
        gemmini_set_tile_ws(&args);
    }
    const size_t metadata_tile_I = args.tile_I;
    if (!metadata_is_local &&
        stripe.row_begin > std::numeric_limits<size_t>::max() - args.activation_row_offset) {
        return MatMulStatus::invalid_arguments;
    }
    auto original_A = args.A;
    if (metadata_is_local) {
        args.activation_row_offset = 0;
    } else {
        args.activation_row_offset += stripe.row_begin;
    }

    args.I = stripe.row_end - stripe.row_begin;
    gemmini_set_tile_ws(&args);
    args.tile_I = metadata_tile_I;
    if (args.A.valid()) {
        args.A = original_A.slice_rows(stripe.row_begin, args.I);
    }
    if (args.A_fp32 != nullptr) {
        args.A_fp32 += input_offset;
    }
    args.f_out += output_offset;

    (void) stripe_id;

    return execute_dense(args);
}

MatmulStatus to_public_status(MatMulStatus status, MatMulCapability capability,
                              const ggml_gemmini_args_t * args = nullptr) {
    MatmulStatusCode code = MatmulStatusCode::invalid_state;
    const char * message = "invalid state";
    switch (status) {
        case MatMulStatus::success:
            code = MatmulStatusCode::success;
            message = "success";
            break;
        case MatMulStatus::empty_stripes:
            code = MatmulStatusCode::invalid_contract;
            message = "missing stripes";
            break;
        case MatMulStatus::malformed_stripe:
            code = MatmulStatusCode::invalid_argument;
            message = "invalid stripe bounds";
            break;
        case MatMulStatus::duplicate_stripe:
            code = MatmulStatusCode::invalid_contract;
            message = "duplicate stripe";
            break;
        case MatMulStatus::overlapping_stripe:
            code = MatmulStatusCode::invalid_contract;
            message = "overlapping stripe";
            break;
        case MatMulStatus::missing_stripes:
            code = MatmulStatusCode::invalid_contract;
            message = "missing stripes";
            break;
        case MatMulStatus::unsupported:
            if (args != nullptr) {
                const auto backend = detail::normalize_route(*args).backend;
                if (backend == detail::BackendRoute::gemmini_os ||
                    backend == detail::BackendRoute::ws_sim) {
                    code = MatmulStatusCode::unsupported_backend;
                    message = "unsupported Gemmini backend";
                    break;
                }
            }
            code = MatmulStatusCode::unsupported_route;
            message = "unsupported route";
            break;
        case MatMulStatus::invalid_contract:
            code = MatmulStatusCode::invalid_contract;
            message = "invalid route contract";
            break;
        case MatMulStatus::invalid_state:
            break;
        case MatMulStatus::invalid_arguments:
            code = MatmulStatusCode::invalid_argument;
            message = "invalid argument";
            break;
    }
    return { code, message, capability };
}

MatmulStatus make_status(MatmulStatusCode code, const char * message,
                         MatMulCapability capability = MatMulCapability::supported) {
    return { code, message, capability };
}

MatmulStatus invalid_state(const char * message = "invalid state") {
    return make_status(MatmulStatusCode::invalid_state, message);
}

MatmulStatus invalid_contract(const char * message) {
    return make_status(MatmulStatusCode::invalid_contract, message);
}

MatmulStatus unsupported_backend(const char * message) {
    return make_status(MatmulStatusCode::unsupported_backend, message, MatMulCapability::unsupported);
}

bool residual_backend_available(RmdBackend backend) {
#if defined(__riscv) || defined(GGML_GEMMINI_TESTING)
    (void) backend;
    return true;
#else
    constexpr bool im2p_build =
#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM)
        true;
#else
        false;
#endif
    return backend == RmdBackend::cpu_direct ||
           (backend == RmdBackend::gemmini_ws_compact &&
            rmd::compact_rmd_backend_available(false, im2p_build));
#endif
}

residual::ResidualRoute residual_route_for(RmdBackend backend) {
    return backend == RmdBackend::cpu_direct
        ? residual::ResidualRoute::cpu_direct
        : residual::ResidualRoute::ws_packet;
}

MatmulStatus validate_exsia_residual_route(
        const ggml_gemmini_args_t & args, const MatmulOptions & options) {
    if (detail::normalize_route(args).activation != detail::ActivationRoute::exsia) {
        return {};
    }
    using Format = ggml_gemmini_args_t::im2p_weight_format_t;
    const auto format = args.weight_format;
    if (format == Format::q8_h2 || format == Format::q8_hp2) {
        return make_status(MatmulStatusCode::unsupported_route,
                           "H2/HP2 ExSIA residual formats are unsupported",
                           MatMulCapability::unsupported);
    }
    const bool h0 = format == Format::q4_h0 || format == Format::q8_h0 ||
                    format == Format::q16_h0;
    if (h0 && options.rmd_backend == RmdBackend::gemmini_ws_compact) {
        return make_status(MatmulStatusCode::unsupported_route,
                           "H0 ExSIA requires CPU-direct residual execution",
                           MatMulCapability::unsupported);
    }
    const std::uint8_t weight_bits =
        format == Format::q4_h0 || format == Format::q4_h1 ||
                format == Format::q4_hp1
            ? 4
            : format == Format::q16_h0 || format == Format::q16_h1 ||
                      format == Format::q16_hp1
                  ? 16
                  : 8;
    if (args.A.valid() && args.A.bits != weight_bits) {
        return make_status(MatmulStatusCode::unsupported_route,
                           "ExSIA requires matched activation and weight widths",
                           MatMulCapability::unsupported);
    }
    return {};
}

MatmulStatus from_rmd_status(rmd::RmdStatus status) {
    switch (status) {
        case rmd::RmdStatus::success:
            return {};
        case rmd::RmdStatus::unsupported_route:
            return unsupported_backend(rmd::rmd_status_message(status));
        case rmd::RmdStatus::invalid_arguments:
        case rmd::RmdStatus::invalid_packet:
            return make_status(MatmulStatusCode::invalid_argument,
                               rmd::rmd_status_message(status), MatMulCapability::unsupported);
        case rmd::RmdStatus::allocation_failure:
            return make_status(MatmulStatusCode::out_of_memory, rmd::rmd_status_message(status));
        case rmd::RmdStatus::residual_too_wide:
        case rmd::RmdStatus::overflow:
        case rmd::RmdStatus::execution_failed:
            return make_status(MatmulStatusCode::execution_failure,
                               rmd::rmd_status_message(status));
    }
    return make_status(MatmulStatusCode::execution_failure, "rmd: unknown status");
}

struct Clock {
    using time_point = std::chrono::steady_clock::time_point;
    static time_point now() {
        return time_point(std::chrono::nanoseconds(cycle::timestamp_ns()));
    }
};

void record_metric(MatmulStageMetrics & metric, bool enabled, Clock::time_point start) {
    if (!enabled) {
        return;
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now() - start).count();
    metric.nanoseconds += static_cast<uint64_t>(std::max<int64_t>(1, elapsed));
    ++metric.count;
}

uint64_t now_ns() {
    return cycle::timestamp_ns();
}

}

namespace detail {

namespace {

struct RouteDescriptor {
    bool legacy_full;
    bool facade_full;
    bool facade_sequential;
    bool facade_pipeline;
    bool deprecated;
};

struct WeightDescriptor {
    bool legacy_full;
    bool facade_full;
    bool facade_sliced;
    bool deprecated;
};

constexpr size_t activation_route_count = static_cast<size_t>(ActivationRoute::stripe) + 1;
constexpr size_t weight_route_count = static_cast<size_t>(WeightRoute::q8_h0) + 1;

constexpr std::array<WeightDescriptor, weight_route_count> weight_descriptors = {{
    { false, false, false, false },
    { true,  true,  true,  false },
    { true,  true,  true,  false },
    { true,  false, false, false },
    { true,  false, false, false },
    { true,  true,  true,  false },
    { true,  true,  true,  false },
    { true,  true,  false, true  },
    { true,  true,  false, true  },
    { true,  true,  true,  false },
    { true,  true,  true,  false },
    { true,  true,  true,  false },
}};

constexpr auto make_route_descriptors() {
    std::array<std::array<RouteDescriptor, weight_route_count>, activation_route_count> matrix{};
    for (size_t activation = 0; activation < activation_route_count; ++activation) {
        for (size_t weight = 0; weight < weight_route_count; ++weight) {
            const auto & base = weight_descriptors[weight];
            const bool known_activation = activation != static_cast<size_t>(ActivationRoute::unknown);
            const bool channel = weight == static_cast<size_t>(WeightRoute::q8_channel_direct) ||
                weight == static_cast<size_t>(WeightRoute::q8_channel_sidecar);
            const bool exsia_or_fp32_channel = channel &&
                (activation == static_cast<size_t>(ActivationRoute::exsia) ||
                 activation == static_cast<size_t>(ActivationRoute::fp32));
            const bool full = known_activation && base.facade_full && !exsia_or_fp32_channel;
            const bool activation_is_sliceable =
                activation == static_cast<size_t>(ActivationRoute::fp32) ||
                activation == static_cast<size_t>(ActivationRoute::exsia) ||
                activation == static_cast<size_t>(ActivationRoute::tensor);
            const bool sequential = full && base.facade_sliced && activation_is_sliceable &&
                (!channel || activation == static_cast<size_t>(ActivationRoute::tensor));
            matrix[activation][weight] = {
                known_activation && base.legacy_full,
                full,
                sequential,
                sequential && activation == static_cast<size_t>(ActivationRoute::exsia),
                base.deprecated,
            };
        }
    }
    return matrix;
}

constexpr auto route_descriptors = make_route_descriptors();

const RouteDescriptor & route_descriptor(const RouteKey & key) {
    return route_descriptors[static_cast<size_t>(key.activation)][static_cast<size_t>(key.weight)];
}

}

RouteKey normalize_route(const ggml_gemmini_args_t & args) {
    RouteKey key{};
    switch (args.tiled_matmul_type) {
        case CPU: key.backend = BackendRoute::cpu; break;
        case WS: key.backend = BackendRoute::gemmini_ws; break;
        case OS: key.backend = BackendRoute::gemmini_os; break;
        default: key.backend = BackendRoute::ws_sim; break;
    }
    switch (args.act_quant.kind()) {
        case quants::act::MetaKind::exsia: key.activation = ActivationRoute::exsia; break;
        case quants::act::MetaKind::tensor: key.activation = ActivationRoute::tensor; break;
        case quants::act::MetaKind::token: key.activation = ActivationRoute::token; break;
        case quants::act::MetaKind::block: key.activation = ActivationRoute::block; break;
        case quants::act::MetaKind::stripe: key.activation = ActivationRoute::stripe; break;
        case quants::act::MetaKind::none: key.activation = ActivationRoute::fp32; break;
    }

    if (args.A_fp32 != nullptr || args.B_fp32 != nullptr) {
        key.activation = ActivationRoute::fp32;
        key.weight = WeightRoute::fp32;
        return key;
    }

    using Format = ggml_gemmini_args_t::im2p_weight_format_t;
    switch (args.weight_format) {
        case Format::q8_0_unpacked_to_h1:
            key.weight = args.weight_i8_scale_active ? WeightRoute::tensor_i8 : WeightRoute::q8_h1;
            break;
        case Format::q4_h0:
        case Format::q8_h0:
        case Format::q16_h0: key.weight = WeightRoute::q8_h0; break;
        case Format::q4_h1:
        case Format::q8_h1:
        case Format::q16_h1: key.weight = WeightRoute::q8_h1; break;
        case Format::q4_hp1:
        case Format::q8_hp1:
        case Format::q16_hp1: key.weight = WeightRoute::q8_hp1; break;
        case Format::q8_h2: key.weight = WeightRoute::q8_h2; break;
        case Format::q8_hp2: key.weight = WeightRoute::q8_hp2; break;
        case Format::q8_channel: key.weight = WeightRoute::q8_channel_direct; break;
        case Format::q8_channel_dense_sidecar: key.weight = WeightRoute::q8_channel_sidecar; break;
    }
    return key;
}

RouteCapabilities route_capabilities(const ggml_gemmini_args_t & args) {
    const RouteKey key = normalize_route(args);
    const RouteDescriptor & descriptor = route_descriptor(key);
    RouteCapabilities caps{};
    caps.legacy_full = descriptor.legacy_full;
    caps.deprecated = descriptor.deprecated;
    caps.full = descriptor.facade_full;
    caps.sliced_dense = descriptor.facade_sequential;
    caps.sliced_compensation = descriptor.facade_sequential;
    caps.live_stripe_producer = descriptor.facade_pipeline;
    caps.internal_parallel_dense = caps.full;
    if (key.backend == BackendRoute::gemmini_os || key.backend == BackendRoute::ws_sim) {
        caps = {};
        caps.legacy_full = descriptor.legacy_full;
        caps.deprecated = descriptor.deprecated;
    }
    return caps;
}

const char * activation_route_name(ActivationRoute route) {
    switch (route) {
        case ActivationRoute::fp32: return "fp32";
        case ActivationRoute::exsia: return "exsia";
        case ActivationRoute::tensor: return "tensor";
        case ActivationRoute::token: return "token";
        case ActivationRoute::block: return "block";
        case ActivationRoute::stripe: return "stripe";
        case ActivationRoute::unknown: return "unknown";
    }
    return "unknown";
}

const char * weight_route_name(WeightRoute route) {
    switch (route) {
        case WeightRoute::fp32: return "fp32";
        case WeightRoute::tensor_i8: return "tensor_i8";
        case WeightRoute::channel_i8: return "channel_i8";
        case WeightRoute::block_i8: return "block_i8";
        case WeightRoute::q8_h1: return "q8_h1";
        case WeightRoute::q8_hp1: return "q8_hp1";
        case WeightRoute::q8_h2: return "q8_h2";
        case WeightRoute::q8_hp2: return "q8_hp2";
        case WeightRoute::q8_channel_direct: return "q8_channel_direct";
        case WeightRoute::q8_channel_sidecar: return "q8_channel_sidecar";
        case WeightRoute::q8_h0: return "q8_h0";
        case WeightRoute::unknown: return "unknown";
    }
    return "unknown";
}

const char * backend_route_name(BackendRoute route) {
    switch (route) {
        case BackendRoute::cpu: return "cpu";
        case BackendRoute::gemmini_ws: return "gemmini_ws";
        case BackendRoute::gemmini_os: return "gemmini_os";
        case BackendRoute::ws_sim: return "ws_sim";
    }
    return "unknown";
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

}

MatMul::MatMul(ggml_gemmini_args_t args) : owned_args_(std::move(args)), args_ptr_(&owned_args_) {}

MatMul::MatMul(ggml_gemmini_args_t * args) : args_ptr_(args) {}

MatMul::MatMul(MatMul && other) noexcept
    : owned_args_(std::move(other.owned_args_)),
      args_ptr_(other.args_ptr_ == &other.owned_args_ ? &owned_args_ : other.args_ptr_),
      first_row_(other.first_row_), last_row_begin_(other.last_row_begin_),
      last_row_end_(other.last_row_end_), covered_rows_(other.covered_rows_),
      has_stripes_(other.has_stripes_), state_(other.state_),
      output_destination_(other.output_destination_),
      output_row_stride_(other.output_row_stride_),
      output_col_stride_(other.output_col_stride_),
      output_stage_(std::move(other.output_stage_)) {
    if (output_destination_ != nullptr && args_ptr_ != nullptr) {
        args().f_out = output_stage_.data();
    }
    other.output_destination_ = nullptr;
    other.output_row_stride_ = 0;
    other.output_col_stride_ = 0;
}

MatMul & MatMul::operator=(MatMul && other) noexcept {
    if (this != &other) {
        discard_output_transaction();
        owned_args_ = std::move(other.owned_args_);
        args_ptr_ = other.args_ptr_ == &other.owned_args_ ? &owned_args_ : other.args_ptr_;
        first_row_ = other.first_row_;
        last_row_begin_ = other.last_row_begin_;
        last_row_end_ = other.last_row_end_;
        covered_rows_ = other.covered_rows_;
        has_stripes_ = other.has_stripes_;
        state_ = other.state_;
        output_destination_ = other.output_destination_;
        output_row_stride_ = other.output_row_stride_;
        output_col_stride_ = other.output_col_stride_;
        output_stage_ = std::move(other.output_stage_);
        if (output_destination_ != nullptr && args_ptr_ != nullptr) {
            args().f_out = output_stage_.data();
        }
        other.output_destination_ = nullptr;
        other.output_row_stride_ = 0;
        other.output_col_stride_ = 0;
    }
    return *this;
}

MatMul::~MatMul() {
    discard_output_transaction();
}

ggml_gemmini_args_t & MatMul::args() { return *args_ptr_; }
const ggml_gemmini_args_t & MatMul::args() const { return *args_ptr_; }

MatMulStatus MatMul::begin_output_transaction() {
    if (output_destination_ != nullptr || args_ptr_ == nullptr || args().f_out == nullptr ||
        args().I == 0 || args().J == 0) {
        return MatMulStatus::invalid_state;
    }
    const size_t row_stride = args().stride_f_out != 0 ? args().stride_f_out : args().J;
    const size_t col_stride = args().col_stride_f_out != 0 ? args().col_stride_f_out : 1;
    size_t row_offset = 0;
    size_t column_offset = 0;
    size_t final_offset = 0;
    size_t output_span = 0;
    if (__builtin_mul_overflow(args().I - 1, row_stride, &row_offset) ||
        __builtin_mul_overflow(args().J - 1, col_stride, &column_offset) ||
        __builtin_add_overflow(row_offset, column_offset, &final_offset) ||
        __builtin_add_overflow(final_offset, size_t{1}, &output_span) ||
        output_span > output_stage_.max_size()) {
        return MatMulStatus::invalid_arguments;
    }

    std::vector<float> staged;
    try {
        test_detail::observe_allocation_attempt();
#if defined(GGML_GEMMINI_TESTING)
        if (test_detail::counters.fail_output_stage_allocation.exchange(
                false, std::memory_order_relaxed)) {
            throw std::bad_alloc();
        }
#endif
        staged.assign(args().f_out, args().f_out + output_span);
    } catch (const std::bad_alloc &) {
        return MatMulStatus::invalid_arguments;
    } catch (const std::length_error &) {
        return MatMulStatus::invalid_arguments;
    }
    output_destination_ = args().f_out;
    output_row_stride_ = row_stride;
    output_col_stride_ = col_stride;
    output_stage_ = std::move(staged);
    args().f_out = output_stage_.data();
    return MatMulStatus::success;
}

void MatMul::commit_output_transaction() {
    if (output_destination_ == nullptr || args_ptr_ == nullptr) return;
#if CYCLE_DETAIL && defined(__linux__) && defined(__aarch64__)
    const cycle::NativeCycleSample commit_start_sample = cycle::read_sample();
#endif
    for (size_t row = 0; row < args().I; ++row) {
        for (size_t column = 0; column < args().J; ++column) {
            const size_t offset = row * output_row_stride_ + column * output_col_stride_;
            output_destination_[offset] = output_stage_[offset];
        }
    }
#if CYCLE_DETAIL && defined(__linux__) && defined(__aarch64__)
    const cycle::NativeCycleSample commit_end_sample = cycle::read_sample();
    const gemmini_native_cycle_sample_internal commit_start =
        project_native_sample(commit_start_sample);
    const gemmini_native_cycle_sample_internal commit_end =
        project_native_sample(commit_end_sample);
    const gemmini_cycle_record_v2 commit_detail{{
        args().matmul_layer.empty() ? nullptr : args().matmul_layer.c_str(),
        "matmul_output_commit_cycles", commit_start.value, commit_end.value,
        nullptr, 0, nullptr}, 0, 0, 0, 0, 0, 0};
    gemmini_log_cycle_record_v2_checked_internal(
        &commit_detail, &commit_start, &commit_end, true);
#endif
    args().f_out = output_destination_;
    output_destination_ = nullptr;
    output_row_stride_ = 0;
    output_col_stride_ = 0;
    output_stage_.clear();
}

void MatMul::discard_output_transaction() {
    if (output_destination_ == nullptr) return;
    if (args_ptr_ != nullptr) args().f_out = output_destination_;
    output_destination_ = nullptr;
    output_row_stride_ = 0;
    output_col_stride_ = 0;
    output_stage_.clear();
}

MatMulResult MatMul::run_dense() {
    return run_dense(false);
}

MatMulResult MatMul::run_dense(bool transactional) {
    if (state_ != MatMulState::idle) {
        return { MatMulStatus::invalid_state, MatMulCapability::supported };
    }
    if (!valid_matmul_shape(args())) {
        return { MatMulStatus::invalid_arguments, MatMulCapability::unsupported };
    }
    if (!valid_activation_metadata(args())) {
        return { MatMulStatus::invalid_contract, MatMulCapability::unsupported };
    }
    if (!transactional &&
        (!quants::activation_rmd_packets(args()).empty() ||
         !quants::activation_direct_residuals(args()).empty())) {
        return { MatMulStatus::invalid_contract, MatMulCapability::unsupported };
    }
    if (!detail::route_capabilities(args()).full) {
        return { MatMulStatus::unsupported, MatMulCapability::unsupported };
    }
    const auto format = args().weight_format;
    const bool metadata_weight =
        format == ggml_gemmini_args_t::im2p_weight_format_t::q4_h0 ||
        format == ggml_gemmini_args_t::im2p_weight_format_t::q4_h1 ||
        format == ggml_gemmini_args_t::im2p_weight_format_t::q4_hp1 ||
        format == ggml_gemmini_args_t::im2p_weight_format_t::q8_h0 ||
        format == ggml_gemmini_args_t::im2p_weight_format_t::q8_h1 ||
        format == ggml_gemmini_args_t::im2p_weight_format_t::q8_hp1 ||
        format == ggml_gemmini_args_t::im2p_weight_format_t::q8_h2 ||
        format == ggml_gemmini_args_t::im2p_weight_format_t::q8_hp2 ||
        format == ggml_gemmini_args_t::im2p_weight_format_t::q16_h0 ||
        format == ggml_gemmini_args_t::im2p_weight_format_t::q16_h1 ||
        format == ggml_gemmini_args_t::im2p_weight_format_t::q16_hp1;
    if (args().B == nullptr && args().B_fp32 == nullptr && !metadata_weight) {
        return { MatMulStatus::invalid_contract, MatMulCapability::unsupported };
    }
    switch (args().weight_format) {
        case ggml_gemmini_args_t::im2p_weight_format_t::q8_channel:
            if (!args().has_q8_channel_direct_read_contract()) {
                return { MatMulStatus::invalid_contract, MatMulCapability::unsupported };
            }
            break;
        case ggml_gemmini_args_t::im2p_weight_format_t::q8_channel_dense_sidecar:
            if (!args().has_q8_channel_dense_sidecar_contract()) {
                return { MatMulStatus::invalid_contract, MatMulCapability::unsupported };
            }
            break;
        case ggml_gemmini_args_t::im2p_weight_format_t::q4_h0:
        case ggml_gemmini_args_t::im2p_weight_format_t::q4_h1:
        case ggml_gemmini_args_t::im2p_weight_format_t::q4_hp1:
        case ggml_gemmini_args_t::im2p_weight_format_t::q16_h0:
        case ggml_gemmini_args_t::im2p_weight_format_t::q16_h1:
        case ggml_gemmini_args_t::im2p_weight_format_t::q16_hp1:
            if (!args().has_native_matched_width_contract()) {
                return { MatMulStatus::invalid_contract, MatMulCapability::unsupported };
            }
            break;
        case ggml_gemmini_args_t::im2p_weight_format_t::q8_h1:
            if (!args().has_q8_h1_im2p_contract()) {
                return { MatMulStatus::invalid_contract, MatMulCapability::unsupported };
            }
            break;
        case ggml_gemmini_args_t::im2p_weight_format_t::q8_hp1:
            if (!args().has_q8_hp1_im2p_contract()) {
                return { MatMulStatus::invalid_contract, MatMulCapability::unsupported };
            }
            break;
        case ggml_gemmini_args_t::im2p_weight_format_t::q8_h2:
            if (!args().has_q8_h2_im2p_contract()) {
                return { MatMulStatus::invalid_contract, MatMulCapability::unsupported };
            }
            break;
        case ggml_gemmini_args_t::im2p_weight_format_t::q8_hp2:
            if (!args().has_q8_hp2_im2p_contract()) {
                return { MatMulStatus::invalid_contract, MatMulCapability::unsupported };
            }
            break;
        default:
            break;
    }

    if (transactional) {
        const MatMulStatus transaction = begin_output_transaction();
        if (transaction != MatMulStatus::success) {
            return {transaction, MatMulCapability::unsupported};
        }
    }
    const MatMulStatus dense_status = execute_dense(args());
    if (dense_status != MatMulStatus::success) {
        if (transactional) {
            discard_output_transaction();
        }
        return {dense_status, MatMulCapability::unsupported};
    }
    return { MatMulStatus::success, MatMulCapability::supported };
}

MatMulResult MatMul::run_full() {
    const MatMulResult dense = run_dense(true);
    if (dense.status != MatMulStatus::success) {
        discard_output_transaction();
        return dense;
    }
#if GGML_GEMMINI_ENABLE_RMD
    rmd::RmdStatus residual_status = rmd::RmdStatus::success;
    if (args().residual_route == residual::ResidualRoute::cpu_direct) {
        for (const auto & payload : quants::activation_direct_residuals(args())) {
            if (payload == nullptr) continue;
            size_t row_end = 0;
            rmd::Correction correction = rmd::BlockScaledInt64Correction{};
            residual_status = __builtin_add_overflow(
                payload->row_begin, payload->row_count, &row_end)
                ? rmd::RmdStatus::invalid_arguments
                : ([&] {
                    test_detail::observe_residual_dispatch();
                    test_detail::observe_backend_dispatch(true);
                    return residual::execute_direct_stripe(args(), *payload, correction);
                })();
            if (residual_status == rmd::RmdStatus::success) {
#if CYCLE_DETAIL && defined(__linux__) && defined(__aarch64__)
                const cycle::NativeCycleSample merge_start_sample = cycle::read_sample();
#endif
                residual_status = rmd::merge_rmd_correction(
                    args(), payload->row_begin, row_end, correction);
#if CYCLE_DETAIL && defined(__linux__) && defined(__aarch64__)
                const cycle::NativeCycleSample merge_end_sample = cycle::read_sample();
                const gemmini_native_cycle_sample_internal merge_start =
                    project_native_sample(merge_start_sample);
                const gemmini_native_cycle_sample_internal merge_end =
                    project_native_sample(merge_end_sample);
                const gemmini_cycle_record_v2 merge_detail{{
                    args().matmul_layer.empty() ? nullptr : args().matmul_layer.c_str(),
                    "rmd_merge_cycles", merge_start.value, merge_end.value,
                    nullptr, 0, nullptr}, 0, 0, 0, 0, 0, 0};
                gemmini_log_cycle_record_v2_checked_internal(
                    &merge_detail, &merge_start, &merge_end, true);
#endif
            }
            if (residual_status != rmd::RmdStatus::success) break;
        }
    } else {
        for (const auto & packet : quants::activation_rmd_packets(args())) {
            if (packet == nullptr) continue;
            rmd::CompressedOutput compressed;
            rmd::Correction correction = rmd::BlockScaledInt64Correction{};
            test_detail::observe_residual_dispatch();
            test_detail::observe_backend_dispatch(false);
            residual_status = rmd::execute_rmd_stripe_ws(args(), *packet, compressed);
            if (residual_status == rmd::RmdStatus::success) {
                residual_status = rmd::compose_rmd_output(*packet, compressed, correction);
            }
            if (residual_status == rmd::RmdStatus::success) {
                residual_status = rmd::merge_rmd_correction(
                    args(), *packet, correction);
            }
            if (residual_status != rmd::RmdStatus::success) break;
        }
    }
    if (residual_status != rmd::RmdStatus::success) {
        discard_output_transaction();
        return { residual_status == rmd::RmdStatus::unsupported_route
                     ? MatMulStatus::unsupported : MatMulStatus::invalid_arguments,
                 MatMulCapability::unsupported };
    }
#endif
#if CYCLE_DETAIL && defined(__linux__) && defined(__aarch64__)
    const cycle::NativeCycleSample finite_start_sample = cycle::read_sample();
#endif
    const bool finite = finite_output(args());
#if CYCLE_DETAIL && defined(__linux__) && defined(__aarch64__)
    const cycle::NativeCycleSample finite_end_sample = cycle::read_sample();
    const gemmini_native_cycle_sample_internal finite_start =
        project_native_sample(finite_start_sample);
    const gemmini_native_cycle_sample_internal finite_end =
        project_native_sample(finite_end_sample);
    const gemmini_cycle_record_v2 finite_detail{{
        args().matmul_layer.empty() ? nullptr : args().matmul_layer.c_str(),
        "matmul_finite_output_validate_cycles", finite_start.value, finite_end.value,
        nullptr, 0, nullptr}, 0, 0, 0, 0, 0, 0};
    gemmini_log_cycle_record_v2_checked_internal(
        &finite_detail, &finite_start, &finite_end, true);
#endif
    if (!finite) {
        discard_output_transaction();
        return {MatMulStatus::invalid_contract, MatMulCapability::unsupported};
    }
    commit_output_transaction();
    state_ = MatMulState::completed;
    return { MatMulStatus::success, MatMulCapability::supported };
}

MatMulStatus MatMul::begin_stripes() {
    if (state_ != MatMulState::idle) {
        return MatMulStatus::invalid_state;
    }
    if (!valid_matmul_shape(args())) {
        return MatMulStatus::invalid_arguments;
    }
    const bool live_exsia_metadata = std::holds_alternative<quants::act::exsia::Meta>(
        args().act_quant.storage());
    if (!live_exsia_metadata && !valid_activation_metadata(args())) {
        return MatMulStatus::invalid_contract;
    }
    if (stripe_capability(args()) == MatMulCapability::unsupported) {
        return MatMulStatus::unsupported;
    }
    const MatMulStatus transaction = begin_output_transaction();
    if (transaction != MatMulStatus::success) {
        return transaction;
    }
    first_row_ = 0;
    last_row_begin_ = 0;
    last_row_end_ = 0;
    covered_rows_ = 0;
    has_stripes_ = false;
    state_ = MatMulState::accepting_stripes;
    return MatMulStatus::success;
}

MatMulStatus MatMul::run_stripe(MatMulStripe stripe) {
    return run_stripe(stripe, 0);
}

MatMulStatus MatMul::run_stripe(MatMulStripe stripe, size_t stripe_id) {
    if (state_ != MatMulState::accepting_stripes) {
        return MatMulStatus::invalid_state;
    }
    if (stripe.row_begin >= stripe.row_end || stripe.row_end > args().I) {
        discard_output_transaction();
        return MatMulStatus::malformed_stripe;
    }

    if (has_stripes_) {
        if (last_row_begin_ == stripe.row_begin && last_row_end_ == stripe.row_end) {
            discard_output_transaction();
            return MatMulStatus::duplicate_stripe;
        }
        if (stripe.row_begin < last_row_end_) {
            discard_output_transaction();
            return MatMulStatus::overlapping_stripe;
        }
    }

    const MatMulStatus status = execute_stripe(args(), stripe, stripe_id);
    if (status == MatMulStatus::success) {
        if (!has_stripes_) {
            first_row_ = stripe.row_begin;
        }
        last_row_begin_ = stripe.row_begin;
        last_row_end_ = stripe.row_end;
        covered_rows_ += stripe.row_end - stripe.row_begin;
        has_stripes_ = true;
    } else {
        discard_output_transaction();
    }
    return status;
}

MatMulStatus MatMul::run_staged_stripe(
        MatMulStripe stripe, size_t stripe_id,
        const quants::act::Meta & activation_metadata) {
    if (state_ != MatMulState::accepting_stripes) {
        return MatMulStatus::invalid_state;
    }
    if (stripe.row_begin >= stripe.row_end || stripe.row_end > args().I) {
        discard_output_transaction();
        return MatMulStatus::malformed_stripe;
    }
    ggml_gemmini_args_t staged_args = args();
    staged_args.act_quant = activation_metadata;
    const MatMulStatus status =
        execute_stripe(std::move(staged_args), stripe, stripe_id, true);
    if (status == MatMulStatus::success) {
        if (!has_stripes_) {
            first_row_ = stripe.row_begin;
        }
        last_row_begin_ = stripe.row_begin;
        last_row_end_ = stripe.row_end;
        covered_rows_ += stripe.row_end - stripe.row_begin;
        has_stripes_ = true;
    } else {
        discard_output_transaction();
    }
    return status;
}

MatMulStatus MatMul::finish_stripes() {
    if (state_ != MatMulState::accepting_stripes) {
        return MatMulStatus::invalid_state;
    }
    if (!has_stripes_) {
        discard_output_transaction();
        state_ = MatMulState::idle;
        return MatMulStatus::empty_stripes;
    }
    if (first_row_ != 0 || last_row_end_ != args().I || covered_rows_ != args().I) {
        discard_output_transaction();
        state_ = MatMulState::idle;
        return MatMulStatus::missing_stripes;
    }
#if CYCLE_DETAIL && defined(__linux__) && defined(__aarch64__)
    const cycle::NativeCycleSample finite_start_sample = cycle::read_sample();
#endif
    const bool finite = finite_output(args());
#if CYCLE_DETAIL && defined(__linux__) && defined(__aarch64__)
    const cycle::NativeCycleSample finite_end_sample = cycle::read_sample();
    const gemmini_native_cycle_sample_internal finite_start =
        project_native_sample(finite_start_sample);
    const gemmini_native_cycle_sample_internal finite_end =
        project_native_sample(finite_end_sample);
    const gemmini_cycle_record_v2 finite_detail{{
        args().matmul_layer.empty() ? nullptr : args().matmul_layer.c_str(),
        "matmul_finite_output_validate_cycles", finite_start.value, finite_end.value,
        nullptr, 0, nullptr}, 0, 0, 0, 0, 0, 0};
    gemmini_log_cycle_record_v2_checked_internal(
        &finite_detail, &finite_start, &finite_end, true);
#endif
    if (!finite) {
        discard_output_transaction();
        state_ = MatMulState::idle;
        return MatMulStatus::invalid_contract;
    }
    commit_output_transaction();
    state_ = MatMulState::completed;
    return MatMulStatus::success;
}

MatMulCapability MatMul::stripe_capability(const ggml_gemmini_args_t & args) {
    const auto format = args.weight_format;
    const auto capabilities = detail::route_capabilities(args);
    if (format == ggml_gemmini_args_t::im2p_weight_format_t::q8_h2 ||
        format == ggml_gemmini_args_t::im2p_weight_format_t::q8_hp2 ||
        !capabilities.sliced_compensation ||
        args.transpose_A || (args.D != nullptr && !args.repeating_bias) ||
        !supports_row_slice_activation(args)) {
        return MatMulCapability::unsupported;
    }
    if (uses_baseline_channel_route(args) &&
        !std::holds_alternative<quants::act::tensor::Meta>(args.act_quant.storage())) {
        return MatMulCapability::unsupported;
    }
    return MatMulCapability::supported;
}

MatMulState MatMul::state() const {
    return state_;
}

MatmulStripeInput::MatmulStripeInput(size_t row_begin, size_t row_end)
    : row_begin_(row_begin), row_end_(row_end), stripe_id_(row_begin),
      residual_(nullptr), residual_count_(0) {}

MatmulStripeInput::MatmulStripeInput(size_t row_begin, size_t row_end, size_t stripe_id,
                                    const int32_t * residual, size_t residual_count)
    : row_begin_(row_begin), row_end_(row_end), stripe_id_(stripe_id),
      residual_(residual), residual_count_(residual_count) {}

size_t MatmulStripeInput::row_begin() const {
    return row_begin_;
}

size_t MatmulStripeInput::row_end() const {
    return row_end_;
}

size_t MatmulStripeInput::stripe_id() const {
    return stripe_id_;
}

const int32_t * MatmulStripeInput::residual() const {
    return residual_;
}

size_t MatmulStripeInput::residual_count() const {
    return residual_count_;
}

MatmulExecution::MatmulExecution(ggml_gemmini_args_t args, MatmulOptions options)
    : total_rows_(args.I), facade_(std::move(args)), options_(options) {
    test_detail::observe_execution_construction();
    state_ = MatmulExecutionState::prepared;
    if (!resolve_geometry(facade_.args()).ok()) {
        status_ = invalid_contract("invalid Gemmini geometry");
        state_ = MatmulExecutionState::failed;
        return;
    }
    status_ = validate_exsia_residual_route(facade_.args(), options_);
    if (!status_.ok()) {
        state_ = MatmulExecutionState::failed;
        return;
    }
    facade_.args().residual_route = residual_route_for(options_.rmd_backend);
    if (!residual_backend_available(options_.rmd_backend)) {
        status_ = unsupported_backend("RMD WS backend is unavailable on this host");
        state_ = MatmulExecutionState::failed;
        return;
    }
    if (options_.dense_threads > 1) {
        status_ = make_status(MatmulStatusCode::unsupported_invocation,
                              "dense stripe execution has one owner lane");
        state_ = MatmulExecutionState::failed;
        return;
    }
    if (options_.mode != MatmulInvocationMode::full && options_.job_capacity == 0) {
        status_ = make_status(MatmulStatusCode::invalid_argument, "job capacity must be nonzero");
        state_ = MatmulExecutionState::failed;
        return;
    }
    if (options_.mode == MatmulInvocationMode::stripe_pipeline &&
        !std::holds_alternative<quants::act::NoneMeta>(facade_.args().act_quant.storage()) &&
        !detail::route_capabilities(facade_.args()).live_stripe_producer) {
        status_ = make_status(MatmulStatusCode::unsupported_invocation,
                              "stripe pipeline requires an ExSIA live producer route");
        state_ = MatmulExecutionState::failed;
        return;
    }
    const bool defer_pipeline_route_validation =
        options_.mode == MatmulInvocationMode::stripe_pipeline &&
        std::holds_alternative<quants::act::NoneMeta>(facade_.args().act_quant.storage());
    if (options_.mode == MatmulInvocationMode::stripe_pipeline &&
        !defer_pipeline_route_validation) {
        const MatMulStatus status = facade_.begin_stripes();
        status_ = to_public_status(
            status, status == MatMulStatus::unsupported ? MatMulCapability::unsupported : MatMulCapability::supported,
            &facade_.args());
        if (!status_.ok()) {
            state_ = MatmulExecutionState::failed;
        }
    }
}

MatmulExecution::MatmulExecution()
    : total_rows_(0), facade_(static_cast<ggml_gemmini_args_t *>(nullptr)) {
    status_ = invalid_state("execution is not prepared");
}

MatmulExecution::MatmulExecution(MatmulStatus status)
    : total_rows_(0), facade_(static_cast<ggml_gemmini_args_t *>(nullptr)), status_(status),
      state_(status.ok() ? MatmulExecutionState::prepared : MatmulExecutionState::failed) {}

MatmulExecution::MatmulExecution(MatmulExecution && other) noexcept
    : MatmulExecution() {
    *this = std::move(other);
}

MatmulExecution::MatmulExecution(ggml_gemmini_args_t * args, MatmulOptions options)
    : total_rows_(args != nullptr ? args->I : 0), facade_(args), options_(options) {
    test_detail::observe_execution_construction();
    state_ = MatmulExecutionState::prepared;
    if (args == nullptr) {
        status_ = make_status(MatmulStatusCode::invalid_argument, "null execution args");
        state_ = MatmulExecutionState::failed;
        return;
    }
    if (!resolve_geometry(facade_.args()).ok()) {
        status_ = invalid_contract("invalid Gemmini geometry");
        state_ = MatmulExecutionState::failed;
        return;
    }
    status_ = validate_exsia_residual_route(facade_.args(), options_);
    if (!status_.ok()) {
        state_ = MatmulExecutionState::failed;
        return;
    }
    facade_.args().residual_route = residual_route_for(options_.rmd_backend);
    if (!residual_backend_available(options_.rmd_backend)) {
        status_ = unsupported_backend("RMD WS backend is unavailable on this host");
        state_ = MatmulExecutionState::failed;
        return;
    }
    if (options_.dense_threads > 1) {
        status_ = make_status(MatmulStatusCode::unsupported_invocation,
                              "dense stripe execution has one owner lane");
        state_ = MatmulExecutionState::failed;
        return;
    }
    if (options_.mode != MatmulInvocationMode::full && options_.job_capacity == 0) {
        status_ = make_status(MatmulStatusCode::invalid_argument, "job capacity must be nonzero");
        state_ = MatmulExecutionState::failed;
        return;
    }
    if (options_.mode == MatmulInvocationMode::stripe_pipeline &&
        !std::holds_alternative<quants::act::NoneMeta>(facade_.args().act_quant.storage()) &&
        !detail::route_capabilities(facade_.args()).live_stripe_producer) {
        status_ = make_status(MatmulStatusCode::unsupported_invocation,
                              "stripe pipeline requires an ExSIA live producer route");
        state_ = MatmulExecutionState::failed;
        return;
    }
    const bool defer_pipeline_route_validation =
        options_.mode == MatmulInvocationMode::stripe_pipeline &&
        std::holds_alternative<quants::act::NoneMeta>(facade_.args().act_quant.storage());
    if (options_.mode == MatmulInvocationMode::stripe_pipeline &&
        !defer_pipeline_route_validation) {
        const MatMulStatus status = facade_.begin_stripes();
        status_ = to_public_status(
            status, status == MatMulStatus::unsupported ? MatMulCapability::unsupported : MatMulCapability::supported,
            &facade_.args());
        if (!status_.ok()) {
            state_ = MatmulExecutionState::failed;
        }
    }
}

MatmulExecution & MatmulExecution::operator=(MatmulExecution && other) noexcept {
    if (this == &other) {
        return *this;
    }
    assert_pipeline_detached();
    other.assert_pipeline_detached();
    total_rows_ = other.total_rows_;
    facade_ = std::move(other.facade_);
    options_ = other.options_;
    status_ = other.status_;
    state_ = other.state_;
    state_mutex_ = std::move(other.state_mutex_);
    active_jobs_ = other.active_jobs_;
    captured_rows_ = other.captured_rows_;
    finalized_rows_ = other.finalized_rows_;
    first_row_ = other.first_row_;
    last_row_begin_ = other.last_row_begin_;
    last_row_end_ = other.last_row_end_;
    has_captures_ = other.has_captures_;
    captured_stripe_ids_ = std::move(other.captured_stripe_ids_);
    pipeline_attached_ = other.pipeline_attached_;
    other.total_rows_ = 0;
    other.active_jobs_ = 0;
    other.captured_rows_ = 0;
    other.finalized_rows_ = 0;
    other.first_row_ = 0;
    other.last_row_begin_ = 0;
    other.last_row_end_ = 0;
    other.has_captures_ = false;
    other.pipeline_attached_ = false;
    return *this;
}

MatmulExecution::~MatmulExecution() {
    assert_pipeline_detached();
}

MatmulInvocationMode MatmulExecution::mode() const {
    return options_.mode;
}

MatmulExecutionState MatmulExecution::state() const {
    std::lock_guard<std::mutex> lock(*state_mutex_);
    return state_;
}

MatmulStatus MatmulExecution::status() const {
    std::lock_guard<std::mutex> lock(*state_mutex_);
    return status_;
}

void MatmulExecution::assert_pipeline_detached() const {
    if (!state_mutex_) {
        return;
    }
    std::lock_guard<std::mutex> lock(*state_mutex_);
    GGML_ASSERT(!pipeline_attached_);
    GGML_ASSERT(active_jobs_ == 0);
}

#if defined(GGML_GEMMINI_TEST_OBSERVER)
bool MatmulExecution::test_pipeline_attached() const {
    std::lock_guard<std::mutex> lock(*state_mutex_);
    return pipeline_attached_;
}
#endif

namespace {
#if defined(GGML_GEMMINI_TEST_OBSERVER)
MatmulStripeCollector * test_residual_failure_collector = nullptr;
MatmulStatus test_residual_failure;
MatmulStripeCollector * test_dense_observer_collector = nullptr;
MatmulDenseState observed_dense_state_at_release = MatmulDenseState::idle;
#endif

bool dense_state_is_terminal(MatmulDenseState state) {
    return state == MatmulDenseState::complete ||
        state == MatmulDenseState::failed ||
        state == MatmulDenseState::cancelled;
}
}

MatmulStripeCollector::MatmulStripeCollector(size_t capacity)
    : capacity_(capacity), sink_{this, &MatmulStripeCollector::on_ready} {
    if (capacity == 0) {
        status_ = make_status(MatmulStatusCode::invalid_argument, "collector capacity must be nonzero");
    }
}

MatmulStripeCollector::~MatmulStripeCollector() {
    finish();
#if defined(GGML_GEMMINI_TEST_OBSERVER)
    std::lock_guard<std::mutex> lock(mutex_);
    if (test_residual_failure_collector == this) {
        test_residual_failure_collector = nullptr;
        test_residual_failure = {};
    }
    if (test_dense_observer_collector == this) {
        test_dense_observer_collector = nullptr;
        observed_dense_state_at_release = MatmulDenseState::idle;
    }
#endif
}

bool MatmulStripeCollector::start(MatmulExecution & execution) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!status_ || worker_started_ || startup_in_progress_ ||
            execution.mode() != MatmulInvocationMode::stripe_pipeline) {
            return false;
        }
        worker_started_ = true;
        startup_in_progress_ = true;
    }
    {
        std::lock_guard<std::mutex> execution_lock(*execution.state_mutex_);
        if (execution.pipeline_attached_) {
            std::lock_guard<std::mutex> lock(mutex_);
            status_ = invalid_state("execution already has a live stripe collector");
            worker_started_ = false;
            startup_in_progress_ = false;
            condition_.notify_all();
            return false;
        }
        execution.pipeline_attached_ = true;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        execution_ = &execution;
        dense_done_ = false;
        stop_requested_ = false;
#if defined(GGML_GEMMINI_TEST_OBSERVER)
        test_thread_start_attempts_ = 0;
#endif
    }
#if defined(GGML_GEMMINI_TEST_OBSERVER)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (test_pause_startup_) {
            condition_.notify_all();
            condition_.wait(lock, [this] { return !test_pause_startup_; });
        }
    }
#endif
    const auto fail_start = [&](MatmulStatus failure) {
        std::vector<std::shared_ptr<MatmulStripeJob>> jobs;
        MatmulExecution * attached_execution = nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            status_ = failure;
            stop_requested_ = true;
            dense_done_ = true;
            pending_.clear();
            attached_execution = execution_;
            for (const auto & weak_job : jobs_) {
                if (auto job = weak_job.lock()) {
                    jobs.push_back(std::move(job));
                }
            }
            execution_ = nullptr;
            worker_started_ = false;
            startup_in_progress_ = false;
        }
        for (const auto & job : jobs) {
            job->cancel(failure);
            release_in_flight_once(job);
        }
        condition_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
        if (attached_execution != nullptr) {
            std::lock_guard<std::mutex> execution_lock(*attached_execution->state_mutex_);
            attached_execution->facade_.discard_output_transaction();
            attached_execution->status_ = failure;
            attached_execution->state_ = MatmulExecutionState::failed;
            attached_execution->pipeline_attached_ = false;
        }
        condition_.notify_all();
        return false;
    };
    try {
#if defined(GGML_GEMMINI_TEST_OBSERVER)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (test_fail_thread_start_attempt_ != 0 &&
                ++test_thread_start_attempts_ == test_fail_thread_start_attempt_) {
                throw std::system_error(
                    std::make_error_code(std::errc::resource_unavailable_try_again),
                    "injected thread start failure");
            }
        }
#endif
        worker_ = std::thread(&MatmulStripeCollector::worker_loop, this);
    } catch (const std::bad_alloc &) {
        return fail_start(make_status(
            MatmulStatusCode::out_of_memory, "collector startup allocation failed"));
    } catch (const std::system_error &) {
        return fail_start(make_status(
            MatmulStatusCode::execution_failure, "collector worker thread creation failed"));
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        startup_in_progress_ = false;
    }
    condition_.notify_all();
    return true;
}

MatmulStatus MatmulStripeCollector::cancel() {
    std::vector<std::shared_ptr<MatmulStripeJob>> jobs;
    MatmulStatus cancelled = make_status(MatmulStatusCode::cancelled, "stripe pipeline cancelled");
    {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !startup_in_progress_; });
        if (!worker_started_) {
            return make_status(MatmulStatusCode::invalid_state, "stripe pipeline is not running");
        }
        if (status_.ok()) {
            status_ = cancelled;
        } else {
            cancelled = status_;
        }
        stop_requested_ = true;
        pending_.clear();
        for (const auto & weak_job : jobs_) {
            if (auto job = weak_job.lock()) {
                jobs.push_back(std::move(job));
            }
        }
    }
    for (const auto & job : jobs) {
        job->cancel(cancelled);
        release_in_flight_once(job);
    }
    condition_.notify_all();
    return cancelled;
}

void MatmulStripeCollector::fail(MatmulStatus failure) {
    std::vector<std::shared_ptr<MatmulStripeJob>> jobs;
    MatmulExecution * execution = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (status_.ok()) {
            status_ = failure;
        } else {
            failure = status_;
        }
        stop_requested_ = true;
        pending_.clear();
        for (const auto & weak_job : jobs_) {
            if (auto job = weak_job.lock()) {
                jobs.push_back(std::move(job));
            }
        }
        execution = execution_;
    }
    for (const auto & job : jobs) {
        job->cancel(failure);
        release_in_flight_once(job);
    }
    if (execution != nullptr) {
        std::lock_guard<std::mutex> execution_lock(*execution->state_mutex_);
        execution->status_ = failure;
        execution->state_ = MatmulExecutionState::failed;
    }
    condition_.notify_all();
}

void MatmulStripeCollector::release_in_flight_once(
        const std::shared_ptr<MatmulStripeJob> & job) {
#if defined(GGML_GEMMINI_TEST_OBSERVER)
    const MatmulDenseState dense_state = job->snapshot().dense;
#endif
    std::lock_guard<std::mutex> lock(mutex_);
    if (job->collector_slot_released_) {
        return;
    }
    job->collector_slot_released_ = true;
#if defined(GGML_GEMMINI_TEST_OBSERVER)
    if (test_dense_observer_collector == this) {
        observed_dense_state_at_release = dense_state;
    }
#endif
    --in_flight_;
}

MatmulStatus MatmulStripeCollector::finish() {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !startup_in_progress_; });
        if (!worker_started_) {
            return status_;
        }
        stop_requested_ = true;
    }
    condition_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        dense_done_ = true;
    }
    condition_.notify_all();
    MatmulExecution * execution = nullptr;
    MatmulStatus status = {};
    {
        std::lock_guard<std::mutex> lock(mutex_);
        execution = execution_;
        status = status_;
        execution_ = nullptr;
        worker_started_ = false;
    }
    if (execution != nullptr) {
        std::lock_guard<std::mutex> execution_lock(*execution->state_mutex_);
        if (!status) {
            execution->facade_.discard_output_transaction();
            execution->status_ = status;
            execution->state_ = MatmulExecutionState::failed;
        }
        execution->pipeline_attached_ = false;
    }
    return status;
}

// One NPU stream: dense WS and RMD run back to back in this worker, in that order.
void MatmulStripeCollector::worker_loop() {
    try {
#if defined(GGML_GEMMINI_TEST_OBSERVER)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (test_thread_exception_ == MatmulCollectorThread::worker) {
                const auto failure = test_thread_exception_failure_;
                test_thread_exception_.reset();
                if (failure == MatmulCollectorThreadFailure::out_of_memory) {
                    throw std::bad_alloc();
                }
                throw std::runtime_error("injected worker thread failure");
            }
        }
#endif
        for (;;) {
            CapturedStripe captured{};
            MatmulExecution * execution = nullptr;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock, [this] {
                    return stop_requested_ || (!pending_.empty() && in_flight_ < capacity_);
                });
                if (pending_.empty()) {
                    break;
                }
                execution = execution_;
                captured = std::move(pending_.front());
                pending_.pop_front();
                ++in_flight_;
                condition_.notify_all();
            }

            std::shared_ptr<MatmulStripeJob> job;
            try {
                job = std::make_shared<MatmulStripeJob>(capture_stripe(
                    *execution,
                    MatmulStripeInput(captured.row_begin, captured.row_end, captured.stripe_id),
                    std::move(captured.direct_residual), std::move(captured.rmd_packet)));
                if (job->status().ok() && captured.activation_metadata.has_value()) {
                    job->staged_activation_meta_ =
                        std::make_unique<quants::act::Meta>();
                    auto & local = job->staged_activation_meta_->storage()
                        .emplace<quants::act::exsia::Meta>();
                    local.e_s = captured.activation_metadata->e_s;
                    local.rho = captured.activation_metadata->rho;
                    local.sigma = captured.activation_metadata->sigma;
                    local.theta = {captured.activation_metadata->theta};
                }
            } catch (const std::exception &) {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    --in_flight_;
                }
                condition_.notify_all();
                throw;
            }
            {
                std::lock_guard<std::mutex> job_lock(*job->job_mutex_);
                detail::apply_captured_stripe(captured, job->metrics_);
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                jobs_.push_back(job);
            }
            MatmulStatus status = job->status();
            if (!status) {
                release_in_flight_once(job);
                fail(status);
                break;
            }

            {
                std::lock_guard<std::mutex> job_lock(*job->job_mutex_);
                job->metrics_.ws_start_ns = now_ns();
                job->rmd_queued_ns_ = captured.timing.queued_ns;
                job->metrics_.rmd_enqueue_ns = captured.timing.queued_ns;
                if (job->execution_->options_.profiling) {
                    job->metrics_.ws_queue.nanoseconds =
                        job->metrics_.ws_start_ns - captured.timing.queued_ns;
                    job->metrics_.ws_queue.count = 1;
                }
            }
#if defined(GGML_GEMMINI_TEST_OBSERVER)
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock, [this] { return !test_pause_dense_ || stop_requested_; });
            }
#endif
            MatmulStatus collector_status = this->status();
            if (!collector_status) {
                job->cancel(collector_status);
                release_in_flight_once(job);
                break;
            }

            status = execute_dense_stripe(*job);
#if defined(GGML_GEMMINI_TEST_OBSERVER)
            if (status && test_residual_failure_collector == this && !test_residual_failure) {
                status = test_residual_failure;
                job->record_failure(status, false);
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    test_residual_failure_observed_ = true;
                }
                condition_.notify_all();
            }
#endif
            if (status) status = execute_rmd_stripe(*job);
#if GGML_GEMMINI_ENABLE_RMD
            if (status) status = compose_rmd_stripe(*job);
#endif
            if (status) status = finalize_stripe(*job);
            if (status) {
                const MatmulJobMetrics profile = job->metrics();
                std::lock_guard<std::mutex> lock(mutex_);
                profiles_.push_back(profile);
            }
            release_in_flight_once(job);
            condition_.notify_all();
            if (!status) {
                fail(status);
                break;
            }
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            dense_done_ = true;
        }
        condition_.notify_all();
    } catch (const std::bad_alloc &) {
        fail(make_status(MatmulStatusCode::out_of_memory, "collector worker thread allocation failed"));
    } catch (const std::exception &) {
        fail(make_status(MatmulStatusCode::execution_failure, "collector worker thread failed"));
    }
}

const quants::act::exsia::StripeReadySink * MatmulStripeCollector::sink() const {
    return &sink_;
}

MatmulStatus MatmulStripeCollector::status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return status_;
}

MatmulCollectorSnapshot MatmulStripeCollector::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {status_, capacity_, pending_.size(), in_flight_, worker_started_};
}

std::vector<MatmulJobMetrics> MatmulStripeCollector::profiles() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return profiles_;
}

rmd::StripePacketHandle MatmulStripeCollector::captured_packet(size_t stripe) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stripes_.at(stripe).rmd_packet;
}

#if defined(GGML_GEMMINI_TEST_OBSERVER)
void MatmulStripeCollector::test_inject_residual_failure(MatmulStatus failure) {
    std::lock_guard<std::mutex> lock(mutex_);
    test_residual_failure_collector = this;
    test_residual_failure = failure;
    test_dense_observer_collector = this;
    observed_dense_state_at_release = MatmulDenseState::idle;
}

void MatmulStripeCollector::test_inject_thread_start_failure(size_t attempt) {
    std::lock_guard<std::mutex> lock(mutex_);
    test_fail_thread_start_attempt_ = attempt;
}

void MatmulStripeCollector::test_inject_thread_exception(
        MatmulCollectorThread thread, MatmulCollectorThreadFailure failure) {
    std::lock_guard<std::mutex> lock(mutex_);
    test_thread_exception_ = thread;
    test_thread_exception_failure_ = failure;
}

void MatmulStripeCollector::test_pause_dense_before_execute() {
    std::lock_guard<std::mutex> lock(mutex_);
    test_pause_dense_ = true;
}

void MatmulStripeCollector::test_pause_startup_after_attachment() {
    std::lock_guard<std::mutex> lock(mutex_);
    test_pause_startup_ = true;
}

void MatmulStripeCollector::test_resume_startup() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        test_pause_startup_ = false;
    }
    condition_.notify_all();
}

void MatmulStripeCollector::test_wait_for_residual_failure() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return test_residual_failure_observed_; });
}

size_t MatmulStripeCollector::test_in_flight() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return in_flight_;
}

MatmulDenseState MatmulStripeCollector::test_dense_state_at_release() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return test_dense_observer_collector == this ?
        observed_dense_state_at_release : MatmulDenseState::idle;
}

#endif

bool MatmulStripeCollector::on_ready(
        void * user_data, const quants::act::exsia::StripeReadyEvent & event) {
    auto & collector = *static_cast<MatmulStripeCollector *>(user_data);
    if (event.row_begin >= event.row_end ||
        (event.activation_metadata.has_value() &&
         event.activation_metadata->theta == std::numeric_limits<int16_t>::min())) {
        {
            std::lock_guard<std::mutex> lock(collector.mutex_);
            collector.status_ = make_status(MatmulStatusCode::invalid_argument, "invalid stripe event");
            collector.stop_requested_ = true;
        }
        collector.condition_.notify_all();
        return false;
    }
    const auto make_captured = [&event](MatmulStageMetrics capture_copy,
                                        MatmulStageMetrics producer_wait,
                                        uint64_t producer_wait_start_ns,
                                        uint64_t producer_wait_end_ns) {
        detail::MatmulCaptureTiming timing{};
        timing.capture_copy = capture_copy;
        timing.producer_wait = producer_wait;
        timing.producer_wait_start_ns = producer_wait_start_ns;
        timing.producer_wait_end_ns = producer_wait_end_ns;
        return detail::capture_collector_event(event, std::move(timing));
    };
    try {
        const auto copy_start = Clock::now();
        MatmulStageMetrics capture_copy;
        record_metric(capture_copy, true, copy_start);
        std::unique_lock<std::mutex> lock(collector.mutex_);
        if (!collector.status_ || collector.stop_requested_) {
            return false;
        }
        if (collector.worker_started_) {
            const auto wait_start = Clock::now();
            const uint64_t producer_wait_start_ns = now_ns();
            collector.condition_.wait(lock, [&collector] {
                return collector.stop_requested_ || !collector.status_ ||
                    collector.pending_.size() + collector.in_flight_ < collector.capacity_;
            });
            if (collector.stop_requested_ || !collector.status_) {
                return false;
            }
            MatmulStageMetrics producer_wait;
            record_metric(producer_wait, true, wait_start);
            CapturedStripe captured = make_captured(
                capture_copy, producer_wait, producer_wait_start_ns, now_ns());
            const auto insert_start = Clock::now();
            collector.pending_.push_back(std::move(captured));
            record_metric(collector.pending_.back().timing.queue_insert, true, insert_start);
            collector.pending_.back().timing.queued_ns = now_ns();
#if LOG_CYCLE
            collector.pending_.back().timing.telemetry_queued_tick = cycle::read();
#endif
            lock.unlock();
            collector.condition_.notify_all();
            return true;
        }
        if (collector.stripes_.size() >= collector.capacity_) {
            collector.status_ = make_status(MatmulStatusCode::out_of_memory, "collector capacity exhausted");
            collector.stop_requested_ = true;
            lock.unlock();
            collector.condition_.notify_all();
            return false;
        }
        CapturedStripe captured = make_captured(capture_copy, {}, 0, 0);
        const auto insert_start = Clock::now();
        collector.stripes_.push_back(std::move(captured));
        record_metric(collector.stripes_.back().timing.queue_insert, true, insert_start);
        collector.stripes_.back().timing.queued_ns = now_ns();
#if LOG_CYCLE
        collector.stripes_.back().timing.telemetry_queued_tick = cycle::read();
#endif
    } catch (const std::bad_alloc &) {
        {
            std::lock_guard<std::mutex> lock(collector.mutex_);
            collector.status_ = make_status(MatmulStatusCode::out_of_memory, "stripe capture allocation failed");
            collector.stop_requested_ = true;
        }
        collector.condition_.notify_all();
        return false;
    }
    return true;
}

MatmulStripeJob::MatmulStripeJob(
        MatmulExecution * execution, MatmulStripeInput input, MatmulStatus status,
        residual::DirectStripePayloadHandle direct_residual,
        rmd::StripePacketHandle rmd_packet)
    : execution_(execution), input_(std::move(input)), status_(status),
      direct_residual_(std::move(direct_residual)), rmd_packet_(std::move(rmd_packet)),
      captured_(status.ok()) {}

MatmulStripeJob::MatmulStripeJob()
    : execution_(nullptr), input_(0, 0), status_(invalid_state("job is not captured")), captured_(false) {}

MatmulStripeJob::MatmulStripeJob(MatmulStripeJob && other) noexcept
    : execution_(other.execution_), input_(std::move(other.input_)), status_(other.status_),
      metrics_(other.metrics_),
      staged_activation_meta_(std::move(other.staged_activation_meta_)),
      owns_slot_(other.owns_slot_),
      released_(other.released_), collector_slot_released_(other.collector_slot_released_),
      rmd_queued_ns_(other.rmd_queued_ns_), job_mutex_(std::move(other.job_mutex_)),
      direct_residual_(std::move(other.direct_residual_)),
      rmd_packet_(std::move(other.rmd_packet_)),
      rmd_output_(std::move(other.rmd_output_)),
      rmd_correction_(std::move(other.rmd_correction_)),
      dense_state_(other.dense_state_), residual_state_(other.residual_state_),
      captured_(other.captured_), finalized_(other.finalized_) {
    other.execution_ = nullptr;
    other.owns_slot_ = false;
    other.released_ = true;
    other.collector_slot_released_ = true;
}

MatmulStripeJob & MatmulStripeJob::operator=(MatmulStripeJob && other) noexcept {
    if (this != &other) {
        release_slot();
        execution_ = other.execution_;
        input_ = std::move(other.input_);
        status_ = other.status_;
        metrics_ = other.metrics_;
        staged_activation_meta_ = std::move(other.staged_activation_meta_);
        direct_residual_ = std::move(other.direct_residual_);
        rmd_packet_ = std::move(other.rmd_packet_);
        rmd_output_ = std::move(other.rmd_output_);
        rmd_correction_ = std::move(other.rmd_correction_);
        owns_slot_ = other.owns_slot_;
        released_ = other.released_;
        collector_slot_released_ = other.collector_slot_released_;
        rmd_queued_ns_ = other.rmd_queued_ns_;
        dense_state_ = other.dense_state_;
        residual_state_ = other.residual_state_;
        captured_ = other.captured_;
        finalized_ = other.finalized_;
        job_mutex_ = std::move(other.job_mutex_);
        other.execution_ = nullptr;
        other.owns_slot_ = false;
        other.released_ = true;
        other.collector_slot_released_ = true;
    }
    return *this;
}

MatmulStripeJob::~MatmulStripeJob() {
    release_slot();
}

void MatmulStripeJob::release_slot() {
    if (!owns_slot_ || released_ || execution_ == nullptr || job_mutex_ == nullptr) {
        return;
    }
    MatmulExecution * execution = nullptr;
    {
        std::lock_guard<std::mutex> lock(*job_mutex_);
        if (!owns_slot_ || released_ || execution_ == nullptr) {
            return;
        }
        owns_slot_ = false;
        released_ = true;
        execution = execution_;
    }
    std::lock_guard<std::mutex> state_lock(*execution->state_mutex_);
    if (execution->active_jobs_ != 0) {
        --execution->active_jobs_;
    }
}

void MatmulStripeJob::cancel(MatmulStatus status) {
    {
        std::lock_guard<std::mutex> lock(*job_mutex_);
        if (finalized_) {
            return;
        }
        if (status_.ok()) {
            status_ = status;
        }
        if (dense_state_ != MatmulDenseState::complete && dense_state_ != MatmulDenseState::failed) {
            dense_state_ = MatmulDenseState::cancelled;
        }
        if (residual_state_ != MatmulResidualState::complete &&
            residual_state_ != MatmulResidualState::failed) {
            residual_state_ = MatmulResidualState::cancelled;
        }
    }
    lifecycle_condition_.notify_all();
}

void MatmulStripeJob::record_failure(MatmulStatus status, bool dense_branch) {
    {
        std::lock_guard<std::mutex> lock(*job_mutex_);
        if (status_.ok()) {
            status_ = status;
            if (dense_branch) {
                dense_state_ = MatmulDenseState::failed;
            } else {
                residual_state_ = MatmulResidualState::failed;
            }
        }
    }
    lifecycle_condition_.notify_all();
}

MatmulStatus MatmulStripeJob::status() const {
    std::lock_guard<std::mutex> lock(*job_mutex_);
    return status_;
}

MatmulJobMetrics MatmulStripeJob::metrics() const {
    std::lock_guard<std::mutex> lock(*job_mutex_);
    return metrics_;
}

MatmulStripeJobSnapshot MatmulStripeJob::snapshot() const {
    std::lock_guard<std::mutex> lock(*job_mutex_);
    return {status_, metrics_, dense_state_, residual_state_, captured_, finalized_, released_};
}

MatmulExecution prepare_execution(const ggml_gemmini_args_t & args, MatmulOptions options) {
    return MatmulExecution(args, options);
}

MatmulExecution prepare_execution(ggml_gemmini_args_t * args, MatmulOptions options) {
    return MatmulExecution(args, options);
}

MatmulStatus prepare_execution(ggml_gemmini_args_t & args, const MatmulOptions & options,
                               MatmulExecution & execution) {
    execution = MatmulExecution(&args, options);
    return execution.status();
}

MatmulStatus execute_full(MatmulExecution & execution) {
    if (execution.options_.mode != MatmulInvocationMode::full) {
        execution.status_ = make_status(
            MatmulStatusCode::unsupported_invocation, "full execution requires full mode");
        execution.state_ = MatmulExecutionState::failed;
        return execution.status_;
    }
    const MatMulResult result = execution.facade_.run_full();
    execution.status_ = to_public_status(result.status, result.capability, &execution.facade_.args());
    execution.state_ = execution.status_.ok() ? MatmulExecutionState::completed : MatmulExecutionState::failed;
    return execution.status_;
}

MatmulStripeJob capture_stripe(MatmulExecution & execution, MatmulStripeInput input) {
    if (input.residual_count() != 0 && input.residual() == nullptr) {
        return MatmulStripeJob(
            &execution,
            std::move(input),
            make_status(MatmulStatusCode::invalid_argument, "null stripe capture payload"));
    }
    if (input.residual_count() != 0) {
        const size_t rows = input.row_end() > input.row_begin() ?
            input.row_end() - input.row_begin() : 0;
        if (rows == 0 || execution.facade_.args().K == 0 ||
            rows > std::numeric_limits<size_t>::max() / execution.facade_.args().K ||
            input.residual_count() != rows * execution.facade_.args().K) {
            return MatmulStripeJob(
                &execution,
                std::move(input),
                make_status(MatmulStatusCode::invalid_argument,
                            "invalid dense residual cardinality"));
        }
        return MatmulStripeJob(
            &execution,
            std::move(input),
            make_status(MatmulStatusCode::unsupported_route,
                        "raw residual stripe payload is unsupported",
                        MatMulCapability::unsupported));
    }
#if GGML_GEMMINI_ENABLE_RMD
    // No live packet was handed over (sequential stripe mode): rebuild one for exactly
    // this row range out of the packets the quantizer produced.
    const size_t row_begin = input.row_begin();
    const size_t row_end = input.row_end();
    const size_t stripe_id = input.stripe_id();
    rmd::RmdStatus slice_status = rmd::RmdStatus::success;
    residual::DirectStripePayloadHandle direct;
    rmd::StripePacketHandle packet;
    try {
        test_detail::observe_allocation_attempt();
        if (execution.options_.rmd_backend == RmdBackend::cpu_direct) {
            direct = residual::slice_direct_payloads(
                quants::activation_direct_residuals(execution.facade_.args()),
                row_begin, row_end, stripe_id, slice_status);
        } else {
            packet = rmd::slice_packets(
                quants::activation_rmd_packets(execution.facade_.args()),
                row_begin, row_end, stripe_id, slice_status);
        }
    } catch (const std::bad_alloc &) {
        return MatmulStripeJob(
            &execution, std::move(input),
            make_status(MatmulStatusCode::out_of_memory, "stripe capture allocation failed"));
    }
    if (slice_status != rmd::RmdStatus::success) {
        return MatmulStripeJob(&execution, std::move(input), from_rmd_status(slice_status));
    }
    return capture_stripe(execution, std::move(input), std::move(direct), std::move(packet));
#else
    return capture_stripe(execution, std::move(input), nullptr);
#endif
}

MatmulStripeJob capture_stripe(MatmulExecution & execution, MatmulStripeInput input,
                               rmd::StripePacketHandle rmd_packet) {
    return capture_stripe(execution, std::move(input), nullptr, std::move(rmd_packet));
}

MatmulStripeJob capture_stripe(MatmulExecution & execution, MatmulStripeInput input,
                               residual::DirectStripePayloadHandle direct_residual,
                               rmd::StripePacketHandle rmd_packet) {
#if !GGML_GEMMINI_ENABLE_RMD
    direct_residual.reset();
    rmd_packet.reset();
#endif
    const auto start = Clock::now();
    MatmulStatus status{};
    std::lock_guard<std::mutex> state_lock(*execution.state_mutex_);
    if (!execution.status_.ok()) {
        status = execution.status_;
    } else if (execution.options_.mode == MatmulInvocationMode::full) {
        status = make_status(MatmulStatusCode::unsupported_invocation, "stripe capture requires stripe mode");
    } else if (execution.options_.mode == MatmulInvocationMode::stripe_pipeline &&
               execution.facade_.state() == MatMulState::idle) {
        const MatMulStatus begin_status = execution.facade_.begin_stripes();
        if (begin_status != MatMulStatus::success) {
            status = to_public_status(
                begin_status,
                begin_status == MatMulStatus::unsupported ? MatMulCapability::unsupported :
                    MatMulCapability::supported);
        }
    } else if (execution.facade_.state() != MatMulState::accepting_stripes) {
        status = invalid_state("execution is not accepting stripes");
    } else if (input.row_begin() >= input.row_end() || input.row_end() > execution.total_rows_ ||
               input.stripe_id() >= execution.total_rows_ ||
               ((input.residual() == nullptr) != (input.residual_count() == 0))) {
        status = make_status(MatmulStatusCode::invalid_argument, "invalid stripe input or id");
    } else if (execution.captured_stripe_ids_.find(input.stripe_id()) !=
               execution.captured_stripe_ids_.end()) {
        status = invalid_contract("duplicate stripe id");
    } else if (execution.active_jobs_ >= execution.options_.job_capacity) {
        status = make_status(MatmulStatusCode::out_of_memory, "job capacity exhausted");
    } else if (execution.has_captures_ && input.row_begin() == execution.last_row_begin_ &&
               input.row_end() == execution.last_row_end_) {
        status = invalid_contract("duplicate stripe");
    } else if (execution.has_captures_ && input.row_begin() < execution.last_row_end_) {
        status = invalid_contract("overlapping stripe");
    } else if ((execution.options_.rmd_backend == RmdBackend::cpu_direct && rmd_packet != nullptr) ||
               (execution.options_.rmd_backend == RmdBackend::gemmini_ws_compact && direct_residual != nullptr) ||
               (direct_residual != nullptr && rmd_packet != nullptr)) {
        status = invalid_contract("residual payload does not match selected backend");
    } else if (direct_residual != nullptr &&
               (direct_residual->row_begin != input.row_begin() ||
                direct_residual->row_count != input.row_end() - input.row_begin())) {
        status = invalid_contract("direct residual does not cover the stripe rows");
    } else if (rmd_packet != nullptr &&
               (rmd_packet->row_begin != input.row_begin() ||
                rmd_packet->row_count != input.row_end() - input.row_begin())) {
        status = invalid_contract("rmd packet does not cover the stripe rows");
    }

    MatmulStripeJob job(&execution, std::move(input), status,
                        std::move(direct_residual), std::move(rmd_packet));
    if (job.status_.ok()) {
        if (!execution.has_captures_) {
            execution.first_row_ = job.input_.row_begin();
        }
        execution.last_row_begin_ = job.input_.row_begin();
        execution.last_row_end_ = job.input_.row_end();
        execution.captured_rows_ += job.input_.row_end() - job.input_.row_begin();
        execution.captured_stripe_ids_.insert(job.input_.stripe_id());
        execution.has_captures_ = true;
        ++execution.active_jobs_;
        job.owns_slot_ = true;
        if (execution.state_ == MatmulExecutionState::prepared) {
            execution.state_ = MatmulExecutionState::running;
        }
        record_metric(job.metrics_.handoff, execution.options_.profiling, start);
    }
    return job;
}

MatmulStatus capture_stripe(MatmulExecution & execution, const MatmulStripeInput & input,
                            MatmulStripeJob & job) {
    MatmulStripeJob captured = capture_stripe(execution, MatmulStripeInput(
        input.row_begin(), input.row_end(), input.stripe_id(),
        input.residual(), input.residual_count()));
    job = std::move(captured);
    return job.status();
}

MatmulStatus execute_dense_stripe(MatmulStripeJob & job) {
    const quants::act::Meta * staged_activation_meta = nullptr;
    {
        std::lock_guard<std::mutex> lock(*job.job_mutex_);
        if (job.execution_ == nullptr || !job.captured_ || job.finalized_ ||
            job.dense_state_ != MatmulDenseState::idle) {
            return invalid_state("dense execution requires captured Dense idle state");
        }
        job.dense_state_ = MatmulDenseState::running;
        staged_activation_meta = job.staged_activation_meta_.get();
    }
    const auto start = Clock::now();
#if CYCLE_DETAIL
    job.metrics_.telemetry_dense_start = cycle::read();
#endif
    const MatMulStatus status = staged_activation_meta != nullptr
        ? job.execution_->facade_.run_staged_stripe(
              {job.input_.row_begin(), job.input_.row_end()},
              job.input_.stripe_id(), *staged_activation_meta)
        : job.execution_->facade_.run_stripe(
              {job.input_.row_begin(), job.input_.row_end()},
              job.input_.stripe_id());
    const MatmulStatus dense_status = to_public_status(
        status, status == MatMulStatus::unsupported ? MatMulCapability::unsupported : MatMulCapability::supported,
        &job.execution_->facade_.args());
    {
        std::lock_guard<std::mutex> lock(*job.job_mutex_);
        job.metrics_.ws_end_ns = now_ns();
#if LOG_CYCLE
        job.metrics_.telemetry_dense_end = cycle::read();
#endif
    }
    if (!dense_status) {
        job.record_failure(dense_status, true);
        return dense_status;
    }
    MatmulStatus result;
    {
        std::lock_guard<std::mutex> lock(*job.job_mutex_);
        job.dense_state_ = MatmulDenseState::complete;
        record_metric(job.metrics_.ws, job.execution_->options_.profiling, start);
        job.metrics_.ws_service = job.metrics_.ws;
        result = dense_status;
    }
    job.lifecycle_condition_.notify_all();
    return result;
}

MatmulStatus accept_external_dense_completion(MatmulStripeJob &job) {
  if (job.job_mutex_ == nullptr) {
    return invalid_state("dense state unavailable");
  }
  {
    std::lock_guard<std::mutex> lock(*job.job_mutex_);
    if (job.execution_ == nullptr || !job.captured_ || job.finalized_ ||
        !job.status_ || job.dense_state_ != MatmulDenseState::idle) {
      return invalid_state(
          "external dense completion requires captured Dense idle state");
    }
    MatMul &facade = job.execution_->facade_;
    const MatMulStripe stripe{job.input_.row_begin(), job.input_.row_end()};
    MatMulStatus accepted = MatMulStatus::success;
    if (facade.state_ != MatMulState::accepting_stripes) {
      accepted = MatMulStatus::invalid_state;
    } else if (stripe.row_begin >= stripe.row_end ||
               stripe.row_end > facade.args().I) {
      accepted = MatMulStatus::malformed_stripe;
    } else if (facade.has_stripes_ &&
               facade.last_row_begin_ == stripe.row_begin &&
               facade.last_row_end_ == stripe.row_end) {
      accepted = MatMulStatus::duplicate_stripe;
    } else if (facade.has_stripes_ && stripe.row_begin < facade.last_row_end_) {
      accepted = MatMulStatus::overlapping_stripe;
    }
    if (accepted != MatMulStatus::success) {
      return to_public_status(accepted, MatMulCapability::supported,
                              &facade.args());
    }
    if (!facade.has_stripes_) {
      facade.first_row_ = stripe.row_begin;
    }
    facade.last_row_begin_ = stripe.row_begin;
    facade.last_row_end_ = stripe.row_end;
    facade.covered_rows_ += stripe.row_end - stripe.row_begin;
    facade.has_stripes_ = true;
    job.dense_state_ = MatmulDenseState::complete;
#if LOG_CYCLE
    job.metrics_.telemetry_dense_start = cycle::read();
    job.metrics_.telemetry_dense_end = job.metrics_.telemetry_dense_start;
#endif
  }
  job.lifecycle_condition_.notify_all();
  return {};
}

// Executes the stripe's RMD packet on the NPU stream and produces the canonical
// block-scaled INT64 compressed output.
MatmulStatus execute_rmd_stripe(MatmulStripeJob & job) {
    if (job.job_mutex_ == nullptr) {
        return invalid_state("residual state unavailable");
    }
    residual::DirectStripePayloadHandle direct;
    rmd::StripePacketHandle packet;
    {
        std::lock_guard<std::mutex> lock(*job.job_mutex_);
        if (job.execution_ == nullptr || !job.captured_ || job.finalized_ || !job.status_ ||
            job.residual_state_ != MatmulResidualState::idle) {
            return invalid_state("residual execution requires captured idle state");
        }
        direct = job.direct_residual_;
        packet = job.rmd_packet_;
        job.residual_state_ = MatmulResidualState::running;
        job.metrics_.rmd_start_ns = now_ns();
        if (job.execution_->options_.profiling && job.rmd_queued_ns_ != 0) {
            job.metrics_.rmd_queue.nanoseconds = job.metrics_.rmd_start_ns - job.rmd_queued_ns_;
            job.metrics_.rmd_queue.count = 1;
        }
    }
    if (direct == nullptr && packet == nullptr) {
        std::lock_guard<std::mutex> lock(*job.job_mutex_);
        job.rmd_output_ = {};
        job.rmd_correction_ = rmd::BlockScaledInt64Correction{};
        job.residual_state_ = MatmulResidualState::complete;
        job.lifecycle_condition_.notify_all();
        return {};
    }

    const auto start = Clock::now();
#if LOG_CYCLE
    const uint64_t residual_start_tick = cycle::read();
#else
    constexpr uint64_t residual_start_tick = 0;
#endif
    rmd::CompressedOutput output;
    rmd::Correction direct_correction = rmd::BlockScaledInt64Correction{};
    rmd::RmdExecutionMetrics metrics{};
    residual::DirectExecutionMetrics direct_metrics{};
    direct_metrics.run_id = job.metrics_.run_id;
#if LOG_CYCLE
    const uint64_t backend_start_tick = cycle::read();
#else
    constexpr uint64_t backend_start_tick = 0;
#endif
    test_detail::observe_residual_dispatch();
    test_detail::observe_backend_dispatch(direct != nullptr);
    const rmd::RmdStatus status = direct != nullptr
        ? residual::execute_direct_stripe(
              job.execution_->facade_.args(), *direct, direct_correction, &direct_metrics)
        : rmd::execute_rmd_stripe_ws(
              job.execution_->facade_.args(), *packet, output, &metrics);
    metrics.direct_event_count = direct_metrics.event_count;
    metrics.direct_call_count = direct_metrics.call_count;
#if LOG_CYCLE
    const uint64_t backend_end_tick = cycle::read();
#else
    constexpr uint64_t backend_end_tick = 0;
#endif
    if (status != rmd::RmdStatus::success) {
        const MatmulStatus failure = from_rmd_status(status);
        job.record_failure(failure, false);
        return failure;
    }
    {
        std::lock_guard<std::mutex> lock(*job.job_mutex_);
        if (!job.status_) return job.status_;
        job.rmd_output_ = std::move(output);
        job.rmd_correction_ = std::move(direct_correction);
        job.metrics_.telemetry_residual_start = residual_start_tick;
        job.metrics_.telemetry_backend_start = backend_start_tick;
        job.metrics_.telemetry_backend_end = backend_end_tick;
        job.metrics_.rmd = metrics;
        record_metric(job.metrics_.rmd_execute, job.execution_->options_.profiling, start);
    }
    job.lifecycle_condition_.notify_all();
    return {};
}

// Reads the canonical compressed output and performs the radix composition.
MatmulStatus compose_rmd_stripe(MatmulStripeJob & job) {
    if (job.job_mutex_ == nullptr) {
        return invalid_state("residual state unavailable");
    }
    rmd::StripePacketHandle packet;
    {
        std::lock_guard<std::mutex> lock(*job.job_mutex_);
        if (job.execution_ == nullptr || !job.captured_ || job.finalized_ || !job.status_ ||
            (job.residual_state_ != MatmulResidualState::running &&
             job.residual_state_ != MatmulResidualState::complete)) {
            return invalid_state("compose requires an executed residual stripe");
        }
        packet = job.rmd_packet_;
        if (job.direct_residual_ != nullptr || packet == nullptr) {
            job.residual_state_ = MatmulResidualState::complete;
            job.metrics_.rmd_end_ns = now_ns();
            return {};
        }
    }

    const auto read_start = Clock::now();
    {
        std::lock_guard<std::mutex> lock(*job.job_mutex_);
#if CYCLE_DETAIL && defined(__linux__) && defined(__aarch64__)
        job.metrics_.telemetry_compose_start_sample = cycle::read_sample();
#endif
        job.metrics_.compose_start_ns = now_ns();
    }
    rmd::Correction correction = rmd::BlockScaledInt64Correction{};
    const rmd::RmdStatus status = rmd::compose_rmd_output(*packet, job.rmd_output_, correction);
#if CYCLE_DETAIL && defined(__linux__) && defined(__aarch64__)
    job.metrics_.telemetry_compose_end_sample = cycle::read_sample();
    const gemmini_native_cycle_sample_internal compose_start =
        project_native_sample(job.metrics_.telemetry_compose_start_sample);
    const gemmini_native_cycle_sample_internal compose_end =
        project_native_sample(job.metrics_.telemetry_compose_end_sample);
    const gemmini_cycle_record_v2 compose_detail{{nullptr, "rmd_packet_compose_cycles",
        compose_start.value, compose_end.value, nullptr, 0, nullptr},
        GEMMINI_CYCLE_HAS_RUN_ID | GEMMINI_CYCLE_HAS_STRIPE_ID,
        job.metrics_.run_id, job.metrics_.stripe_id, 0, 0, 0};
    gemmini_log_cycle_record_v2_checked_internal(
        &compose_detail, &compose_start, &compose_end,
        status == rmd::RmdStatus::success);
#endif
    if (status != rmd::RmdStatus::success) {
        const MatmulStatus failure = from_rmd_status(status);
        job.record_failure(failure, false);
        return failure;
    }
    {
        std::lock_guard<std::mutex> lock(*job.job_mutex_);
        if (!job.status_) {
            return job.status_;
        }
        job.rmd_correction_ = std::move(correction);
        record_metric(job.metrics_.rmd_compose, job.execution_->options_.profiling, read_start);
        job.metrics_.rmd_output_read = job.metrics_.rmd_compose;
        job.metrics_.compose_end_ns = now_ns();
        job.residual_state_ = MatmulResidualState::complete;
        job.metrics_.rmd_end_ns = job.metrics_.compose_end_ns;
    }
    job.lifecycle_condition_.notify_all();
    return {};
}

MatmulStatus finalize_stripe(MatmulStripeJob & job) {
    const auto start = Clock::now();
    MatmulStatus merge_failure{};
    {
        std::lock_guard<std::mutex> lock(*job.job_mutex_);
        if (job.execution_ == nullptr || !job.captured_ || job.finalized_) {
            return invalid_state("stripe is not finalizable");
        }
        if (!job.status_) {
            return job.status_;
        }
        if (job.dense_state_ != MatmulDenseState::complete ||
            job.residual_state_ != MatmulResidualState::complete) {
            return invalid_state("finalize requires dense and residual completion");
        }
        job.finalized_ = true;
#if CYCLE_DETAIL && defined(__linux__) && defined(__aarch64__)
        job.metrics_.telemetry_finalize_start_sample = cycle::read_sample();
#endif
        job.metrics_.finalize_start_ns = now_ns();
        job.metrics_.stripe_id = job.input_.stripe_id();
        job.metrics_.row_begin = job.input_.row_begin();
        job.metrics_.row_end = job.input_.row_end();
        if (!rmd::correction_empty(job.rmd_correction_)) {
            job.metrics_.merge_start_ns = now_ns();
#if LOG_CYCLE
            job.metrics_.telemetry_merge_start = cycle::read();
#endif
            const rmd::RmdStatus status = job.direct_residual_ != nullptr
                ? rmd::merge_rmd_correction(
                      job.execution_->facade_.args(), job.input_.row_begin(),
                      job.input_.row_end(), job.rmd_correction_)
                : rmd::merge_rmd_correction(
                      job.execution_->facade_.args(), *job.rmd_packet_,
                      job.rmd_correction_);
            job.metrics_.merge_end_ns = now_ns();
#if LOG_CYCLE
            job.metrics_.telemetry_merge_end = cycle::read();
            job.metrics_.telemetry_residual_end = job.metrics_.telemetry_merge_end;
#endif
            job.metrics_.telemetry_correction_nonzero_count = std::visit(
                [](const auto & typed) {
                    return static_cast<uint64_t>(std::count_if(
                        typed.values.begin(), typed.values.end(),
                        [](const auto value) { return value != 0; }));
                },
                job.rmd_correction_);
#if CYCLE_DETAIL
            job.metrics_.telemetry_input_hash = job.direct_residual_ != nullptr
                ? rmd_input_hash(*job.direct_residual_) : rmd_input_hash(*job.rmd_packet_);
            job.metrics_.telemetry_correction_hash = rmd_correction_hash(job.rmd_correction_);
            job.metrics_.telemetry_output_hash = rmd_output_hash(
                job.execution_->facade_.args(), job.input_.row_begin(), job.input_.row_end());
#endif
            if (status != rmd::RmdStatus::success) {
                merge_failure = from_rmd_status(status);
            }
        }
#if LOG_CYCLE
        if (job.metrics_.telemetry_residual_start != 0 &&
            job.metrics_.telemetry_residual_end == 0) {
            job.metrics_.telemetry_residual_end = cycle::read();
        }
#endif
        record_metric(job.metrics_.rmd_finalize, job.execution_->options_.profiling, start);
        job.metrics_.finalize_end_ns = now_ns();
#if CYCLE_DETAIL && defined(__linux__) && defined(__aarch64__)
        job.metrics_.telemetry_finalize_end_sample = cycle::read_sample();
        const gemmini_native_cycle_sample_internal finalize_start =
            project_native_sample(job.metrics_.telemetry_finalize_start_sample);
        const gemmini_native_cycle_sample_internal finalize_end =
            project_native_sample(job.metrics_.telemetry_finalize_end_sample);
        const gemmini_cycle_record_v2 finalize_detail{{nullptr,
            job.direct_residual_ != nullptr ? "rmd_cpu_direct_finalize_cycles" :
                (job.rmd_packet_ != nullptr ? "rmd_packet_finalize_cycles" :
                                              "dense_finalize_cycles"),
            finalize_start.value, finalize_end.value, nullptr, 0, nullptr},
            GEMMINI_CYCLE_HAS_RUN_ID | GEMMINI_CYCLE_HAS_STRIPE_ID,
            job.metrics_.run_id, job.metrics_.stripe_id, 0, 0, 0};
        gemmini_log_cycle_record_v2_checked_internal(
            &finalize_detail, &finalize_start, &finalize_end, merge_failure.ok());
#endif
    }
    if (!merge_failure.ok()) {
        job.record_failure(merge_failure, false);
        return merge_failure;
    }
    {
        std::lock_guard<std::mutex> state_lock(*job.execution_->state_mutex_);
        job.execution_->finalized_rows_ += job.input_.row_end() - job.input_.row_begin();
    }
    job.release_slot();
    job.lifecycle_condition_.notify_all();
    return {};
}

MatmulStatus finish_execution(MatmulExecution & execution) {
    std::lock_guard<std::mutex> state_lock(*execution.state_mutex_);
    if (!execution.status_.ok()) {
        execution.facade_.discard_output_transaction();
        execution.state_ = MatmulExecutionState::failed;
        return execution.status_;
    }
    if (execution.options_.mode == MatmulInvocationMode::full) {
        return execution.facade_.state() == MatMulState::completed ? execution.status_ : invalid_state();
    }
    if (execution.pipeline_attached_) {
        return invalid_state("cannot finish while stripe collector is attached");
    }
    execution.state_ = MatmulExecutionState::finishing;
    if (execution.active_jobs_ != 0) {
        execution.state_ = MatmulExecutionState::running;
        return invalid_state("cannot finish with live jobs");
    }
    if (!execution.has_captures_ || execution.first_row_ != 0 ||
        execution.last_row_end_ != execution.total_rows_ ||
        execution.captured_rows_ != execution.total_rows_ ||
        execution.finalized_rows_ != execution.total_rows_) {
        execution.facade_.discard_output_transaction();
        execution.state_ = MatmulExecutionState::failed;
        return invalid_contract("missing stripes");
    }
    const MatMulStatus status = execution.facade_.finish_stripes();
    execution.status_ = to_public_status(status, MatMulCapability::supported);
    execution.state_ = execution.status_.ok() ? MatmulExecutionState::completed : MatmulExecutionState::failed;
    return execution.status_;
}

static MatmulStatus matmul_impl(MatmulExecution execution, MatmulOptions options) {
    if (!execution.status()) {
        return execution.status();
    }
    if (options.mode == MatmulInvocationMode::full) {
        return execute_full(execution);
    }
    return make_status(MatmulStatusCode::unsupported_invocation,
                       "pipeline mode requires externally staged stripes");
}

MatmulStatus matmul(ggml_gemmini_args_t & args, MatmulOptions options) {
    return matmul_impl(prepare_execution(&args, options), options);
}

MatmulStatus matmul(const ggml_gemmini_args_t & args, MatmulOptions options) {
    return matmul_impl(prepare_execution(args, options), options);
}

MatmulStatus execute_post_fold_pipeline(
        const ggml_gemmini_args_t & args, MatmulStripeCollector & collector) {
    if (!collector.status_) {
        return collector.status_;
    }
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_pipeline;
    options.job_capacity = 1;
    options.rmd_backend = args.residual_route == residual::ResidualRoute::cpu_direct
        ? RmdBackend::cpu_direct : RmdBackend::gemmini_ws_compact;
    MatmulExecution execution = prepare_execution(args, options);
    if (!execution.status()) {
        return execution.status();
    }
    for (auto & captured : collector.stripes_) {
        MatmulStripeJob job = capture_stripe(
            execution,
            MatmulStripeInput(captured.row_begin, captured.row_end, captured.stripe_id),
            std::move(captured.direct_residual), std::move(captured.rmd_packet));
        detail::apply_captured_stripe(captured, job.metrics_);
        MatmulStatus status = job.status();
        if (status) status = execute_dense_stripe(job);
        if (status) status = execute_rmd_stripe(job);
#if GGML_GEMMINI_ENABLE_RMD
        if (status) status = compose_rmd_stripe(job);
#endif
        if (status) status = finalize_stripe(job);
        if (!status) {
            return status;
        }
    }
    return finish_execution(execution);
}

}
