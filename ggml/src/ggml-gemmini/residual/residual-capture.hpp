#pragma once

#include "direct/direct-builder.hpp"
#include "rmd/rmd-builder.hpp"

#include <chrono>
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
        const auto start = std::chrono::steady_clock::now();
        if (auto *cpu = std::get_if<DirectStripeBuilder>(&sink_)) {
            result.direct = cpu->finish();
        } else {
            result.packet = std::get<rmd::RmdStripeBuilder>(sink_).finish();
        }
        const auto end = std::chrono::steady_clock::now();
        result.capture_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        return result;
    }

    bool holds_cpu_sink() const { return std::holds_alternative<DirectStripeBuilder>(sink_); }
    bool holds_ws_sink() const { return std::holds_alternative<rmd::RmdStripeBuilder>(sink_); }

private:
    using Sink = std::variant<DirectStripeBuilder, rmd::RmdStripeBuilder>;
    Sink sink_;
};

}
