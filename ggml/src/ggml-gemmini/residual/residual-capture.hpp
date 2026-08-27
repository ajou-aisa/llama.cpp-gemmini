#pragma once

#include "direct/direct-builder.hpp"
#include "rmd/rmd-builder.hpp"
#include <gemmini/cycle_reader.hpp>

#if defined(__linux__) && defined(__aarch64__) && CYCLE_DETAIL
#include <gemmini/log.h>
#include "../ggml-gemmini-utils/src/cycle_reader_internal.h"
#endif

#include <optional>
#include <variant>

namespace ggml::gemmini::residual {

enum class ResidualRoute : uint8_t {
    cpu_direct,
    ws_packet,
};

struct ResidualStripePayload {
    DirectStripePayloadHandle direct;
    rmd::StripePacketHandle packet;
    uint64_t capture_ns = 0;
    std::optional<uint64_t> capture_finish_cycles;
    std::optional<ResidualRoute> capture_finish_route;
    bool capture_finish_valid = false;

    bool empty() const { return !direct && !packet; }
};

class TimedResidualCapture {
public:
    TimedResidualCapture() : TimedResidualCapture(ResidualRoute::ws_packet) {}

    explicit TimedResidualCapture(ResidualRoute route)
        : sink_(route == ResidualRoute::cpu_direct
                    ? Sink(std::in_place_type<DirectStripeBuilder>)
                    : Sink(std::in_place_type<rmd::RmdStripeBuilder>)) {}

    void select(ResidualRoute route) {
        if (route == ResidualRoute::cpu_direct) {
            sink_.emplace<DirectStripeBuilder>();
        } else {
            sink_.emplace<rmd::RmdStripeBuilder>();
        }
    }

    void reset(size_t stripe_id, size_t row_begin, size_t row_count,
               size_t logical_k, size_t logical_j) {
        std::visit([&](auto &sink) {
            sink.reset(stripe_id, row_begin, row_count, logical_k, logical_j);
        }, sink_);
    }

    bool add_residual(size_t local_row, size_t original_k, int32_t residual) {
        return std::visit([&](auto &sink) {
            return sink.add_residual(local_row, original_k, residual);
        }, sink_);
    }

    bool empty() const {
        return std::visit([](const auto &sink) { return sink.empty(); }, sink_);
    }

    rmd::RmdStatus status() const {
        return std::visit([](const auto &sink) { return sink.status(); }, sink_);
    }

    ResidualStripePayload finish() {
        ResidualStripePayload result;
        if (empty()) return result;
#if defined(__linux__) && defined(__aarch64__) && CYCLE_DETAIL
        cycle::NativeCycleSample finish_start{};
        cycle::NativeCycleSample finish_end{};
        const char *finish_op = nullptr;
        if (std::holds_alternative<DirectStripeBuilder>(sink_)) {
            result.capture_finish_route = ResidualRoute::cpu_direct;
            finish_op = "rmd_direct_finish_cycles";
        } else {
            result.capture_finish_route = ResidualRoute::ws_packet;
            finish_op = "rmd_packet_finish_cycles";
        }
        finish_start = cycle::read_sample();
#endif
#if LOG_CYCLE
        const uint64_t start = cycle::timestamp_ns();
#endif
        if (auto *cpu = std::get_if<DirectStripeBuilder>(&sink_)) {
            result.direct = cpu->finish();
        } else {
            result.packet = std::get<rmd::RmdStripeBuilder>(sink_).finish();
        }
#if LOG_CYCLE
        const uint64_t end = cycle::timestamp_ns();
        result.capture_ns = end >= start ? end - start : 0;
#endif
#if defined(__linux__) && defined(__aarch64__) && CYCLE_DETAIL
        finish_end = cycle::read_sample();
        const cycle::NativeCycleDelta finish_delta =
            cycle::evaluate_interval(finish_start, finish_end);
        result.capture_finish_valid = finish_delta.valid;
        if (finish_delta.valid) result.capture_finish_cycles = finish_delta.value;
        const gemmini_native_cycle_sample_internal start_sample{
            finish_start.value, static_cast<uint8_t>(finish_start.valid),
            static_cast<uint8_t>(finish_start.reason), GEMMINI_NATIVE_CYCLE_SOURCE_LINUX_PERF_CPU_CYCLES,
            finish_start.owner_event_token, finish_start.generation};
        const gemmini_native_cycle_sample_internal end_sample{
            finish_end.value, static_cast<uint8_t>(finish_end.valid),
            static_cast<uint8_t>(finish_end.reason), GEMMINI_NATIVE_CYCLE_SOURCE_LINUX_PERF_CPU_CYCLES,
            finish_end.owner_event_token, finish_end.generation};
        const gemmini_cycle_record_v2 record{{nullptr, finish_op, finish_start.value, finish_end.value,
                                               nullptr, 0, nullptr}, 0, 0, 0, 0, 0, 0};
        gemmini_log_cycle_record_v2_checked_internal(&record, &start_sample, &end_sample, 1);
#endif
        return result;
    }

    bool holds_cpu_sink() const { return std::holds_alternative<DirectStripeBuilder>(sink_); }
    bool holds_ws_sink() const { return std::holds_alternative<rmd::RmdStripeBuilder>(sink_); }

private:
    using Sink = std::variant<DirectStripeBuilder, rmd::RmdStripeBuilder>;
    Sink sink_;
};

}
