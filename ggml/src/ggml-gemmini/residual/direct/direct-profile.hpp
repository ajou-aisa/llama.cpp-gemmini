#pragma once

#include "direct-types.hpp"
#include "direct-stage-profile.hpp"
#include "../rmd/rmd-types.hpp"

#include <gemmini/host-timing.hpp>
#include <gemmini/log.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace ggml::gemmini::residual::detail {

struct DirectHostSpan {
    cycle::HostSample start{};
    cycle::HostSample end{};

    std::string host_json() const {
        return cycle::serialize_host_timing(start.ns, end.ns, start.tid, end.tid);
    }

    std::string cpu_json() const {
        return cycle::serialize_thread_cpu_timing(start, end);
    }
};

struct DirectHostTile {
    size_t worker_id = 0;
    size_t j_begin = 0;
    size_t j_end = 0;
    DirectHostSpan compute;
    DirectHostSpan logging;
    log::CycleWriteTiming writes{};
    std::array<DirectStageTotals, 3> stages{};
};

struct DirectHostWorker {
    bool active = false;
    DirectHostSpan work;
    DirectHostSpan barrier;
};

class DirectHostProfile {
public:
    DirectHostProfile(const DirectStripePayload & payload, const std::string & layer,
                      std::optional<uint64_t> run_id, bool enabled = true) noexcept :
        payload_(payload), layer_(layer), run_id_(run_id), enabled_(enabled) {
        if (enabled_) phases_[0].start = cycle::read_host_sample();
        const char * deep = std::getenv("GGML_GEMMINI_RESIDUAL_DEEP_PROFILE");
        deep_profile = enabled_ && deep != nullptr && std::strcmp(deep, "1") == 0;
    }

    ~DirectHostProfile() noexcept {
        if (!enabled_) return;
        const cycle::HostSample end = cycle::read_host_sample();
        phases_[phase_].end = end;
        try {
            // Serialization and the sidecar write are outside all measured spans.
            log::cycle.write_json(serialize(end));
        } catch (...) {
            log::cycle.report_failure("residual host profile");
        }
    }

    void next_phase(size_t phase) noexcept {
        if (!enabled_) return;
        const cycle::HostSample sample = cycle::read_host_sample();
        phases_[phase_].end = sample;
        phases_[phase].start = sample;
        phase_ = phase;
    }

    void prepare(size_t tile_count, size_t worker_count) noexcept {
        if (!enabled_) return;
        j_tile_count_ = tile_count;
        for (size_t index = 0; index < payload_.events.size(); ++index) {
            const auto & event = payload_.events[index];
            const bool new_row = index == 0 ||
                payload_.events[index - 1].local_row != event.local_row;
            if (new_row) ++active_rows_;
            if (new_row || payload_.events[index - 1].original_k / rmd::kBlockSize !=
                           event.original_k / rmd::kBlockSize) {
                ++active_row_blocks_;
            }
        }
        workload_valid_ = true;
        try {
            tiles.resize(tile_count);
            workers.resize(worker_count);
            ready = true;
        } catch (...) {
            // Profiling allocation failure must not change numerical execution.
            tiles.clear();
            workers.clear();
        }
    }

    bool ready = false;
    bool success = false;
    bool deep_profile = false;
    std::vector<DirectHostTile> tiles;
    std::vector<DirectHostWorker> workers;

private:
    static void quote(std::ostream & out, const std::string & value) {
        constexpr char hex[] = "0123456789abcdef";
        out << '"';
        for (const unsigned char ch : value) {
            if (ch == '"' || ch == '\\') {
                out << '\\' << static_cast<char>(ch);
            } else if (ch < 0x20) {
                out << "\\u00" << hex[ch >> 4] << hex[ch & 15];
            } else {
                out << static_cast<char>(ch);
            }
        }
        out << '"';
    }

    std::string serialize(const cycle::HostSample & end) const {
        const DirectHostSpan total{phases_[0].start, end};
        const bool valid = success && ready && total.start.tid != 0 &&
            total.start.tid == end.tid && end.ns >= total.start.ns;
        std::ostringstream out;
        out << "{\"schema\":\"gemmini.cycle\",\"version\":2,"
            << "\"record_type\":\"RESIDUAL_HOST_PROFILE\",\"source\":\"steady_clock\","
            << "\"unit\":\"nanosecond\",\"op\":\"rmd.cpu_direct.profile\",\"layer\":";
        if (layer_.empty()) out << "null"; else quote(out, layer_);
        out << ",\"run_id\":" << (run_id_ ? std::to_string(*run_id_) : "null")
            << ",\"stripe_id\":" << payload_.stripe_id
            << ",\"slot\":null,\"node_id\":null,\"worker_id\":null,\"valid\":"
            << (valid ? "true" : "false")
            << ",\"deep_profile\":" << (deep_profile ? "true" : "false")
            << ",\"host_timing\":" << total.host_json()
            << ",\"workload\":{\"event_count\":" << payload_.events.size()
            << ",\"active_rows\":" << (workload_valid_ ? std::to_string(active_rows_) : "null")
            << ",\"active_row_blocks\":"
            << (workload_valid_ ? std::to_string(active_row_blocks_) : "null")
            << ",\"row_begin\":" << payload_.row_begin
            << ",\"row_count\":" << payload_.row_count
            << ",\"logical_j\":" << payload_.logical_j
            << ",\"logical_k\":" << payload_.logical_k
            << ",\"j_tile_count\":" << (workload_valid_ ? std::to_string(j_tile_count_) : "null")
            << "},\"phases\":{";
        constexpr std::array<const char *, 4> names{
            "validation", "preparation", "parallel", "finalization"};
        for (size_t index = 0; index < phases_.size(); ++index) {
            if (index != 0) out << ',';
            out << '"' << names[index] << "\":{\"host_timing\":"
                << phases_[index].host_json() << ",\"thread_cpu_timing\":"
                << phases_[index].cpu_json() << '}';
        }
        out << "},\"tiles\":[";
        for (size_t index = 0; index < tiles.size(); ++index) {
            const auto & tile = tiles[index];
            const bool log_valid = tile.writes.calls != 0 && tile.writes.valid;
            if (index != 0) out << ',';
            out << "{\"node_id\":" << index << ",\"worker_id\":" << tile.worker_id
                << ",\"j_begin\":" << tile.j_begin
                << ",\"j_end\":" << tile.j_end
                << ",\"host_timing\":" << tile.compute.host_json()
                << ",\"thread_cpu_timing\":" << tile.compute.cpu_json()
                << ",\"log_host_timing\":" << tile.logging.host_json()
                << ",\"log_thread_cpu_timing\":" << tile.logging.cpu_json()
                << ",\"log_calls\":" << (log_valid ? std::to_string(tile.writes.calls) : "null")
                << ",\"log_mutex_wait_ns\":"
                << (log_valid ? std::to_string(tile.writes.mutex_wait_ns) : "null")
                << ",\"log_io_ns\":" << (log_valid ? std::to_string(tile.writes.io_ns) : "null")
                << ",\"log_valid\":" << (log_valid ? "true" : "false");
            if (deep_profile) {
                constexpr std::array<const char *, 3> stage_names{
                    "event_scan", "weight_dot", "scale_apply"};
                out << ",\"stages\":{";
                for (size_t stage = 0; stage < stage_names.size(); ++stage) {
                    if (stage != 0) out << ',';
                    out << '"' << stage_names[stage] << "\":";
                    tile.stages[stage].write_json(out);
                }
                out << '}';
            }
            out << '}';
        }
        out << "],\"workers\":[";
        bool first = true;
        for (size_t index = 0; index < workers.size(); ++index) {
            const auto & worker = workers[index];
            if (!worker.active) continue;
            if (!first) out << ',';
            first = false;
            out << "{\"worker_id\":" << index << ",\"tid\":"
                << (worker.work.start.tid != 0 ? std::to_string(worker.work.start.tid) : "null")
                << ",\"host_timing\":" << worker.work.host_json()
                << ",\"thread_cpu_timing\":" << worker.work.cpu_json()
                << ",\"barrier_host_timing\":" << worker.barrier.host_json()
                << ",\"barrier_thread_cpu_timing\":" << worker.barrier.cpu_json() << '}';
        }
        out << "]}\n";
        return out.str();
    }

    const DirectStripePayload & payload_;
    const std::string & layer_;
    std::optional<uint64_t> run_id_;
    bool enabled_;
    bool workload_valid_ = false;
    size_t phase_ = 0;
    size_t active_rows_ = 0;
    size_t active_row_blocks_ = 0;
    size_t j_tile_count_ = 0;
    std::array<DirectHostSpan, 4> phases_{};
};

}
