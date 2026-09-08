#include "../include/gemmini/log.hpp"
#include "../include/gemmini/host-timing.hpp"
#if defined(__linux__) && defined(__aarch64__)
#include "cycle_reader_internal.h"
#endif

#include <limits>
#include <atomic>
#include <chrono>
#include <ctime>
#include <exception>
#include <mutex>
#include <string>

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <pthread.h>
#include <unistd.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace ggml::gemmini::cycle {

uint64_t host_thread_id() noexcept {
#if defined(__linux__)
    return static_cast<uint64_t>(syscall(SYS_gettid));
#elif defined(__APPLE__)
    uint64_t tid = 0;
    return pthread_threadid_np(nullptr, &tid) == 0 ? tid : 0;
#elif defined(_WIN32)
    return static_cast<uint64_t>(GetCurrentThreadId());
#else
    static std::atomic<uint64_t> next_id{1};
    thread_local const uint64_t tid = next_id.fetch_add(1, std::memory_order_relaxed);
    return tid;
#endif
}

HostSample read_host_sample() noexcept {
    HostSample sample;
    sample.ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
    sample.tid = host_thread_id();
#if (defined(__linux__) || defined(__APPLE__)) && defined(CLOCK_THREAD_CPUTIME_ID)
    timespec cpu_time{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_time) == 0) {
        sample.thread_cpu_ns = static_cast<uint64_t>(cpu_time.tv_sec) * 1000000000ULL +
            static_cast<uint64_t>(cpu_time.tv_nsec);
        sample.thread_cpu_valid = true;
    }
#endif
    return sample;
}

std::string serialize_thread_cpu_timing(const HostSample &start, const HostSample &end) {
    const bool start_valid = start.thread_cpu_valid && start.tid != 0;
    const bool end_valid = end.thread_cpu_valid && end.tid != 0;
    const bool valid = start_valid && end_valid && start.tid == end.tid &&
        end.thread_cpu_ns >= start.thread_cpu_ns;
    return std::string("{\"clock\":\"thread_cpu\",\"unit\":\"nanosecond\",\"start_ns\":") +
        (start_valid ? std::to_string(start.thread_cpu_ns) : "null") +
        ",\"end_ns\":" + (end_valid ? std::to_string(end.thread_cpu_ns) : "null") +
        ",\"duration_ns\":" + (valid ? std::to_string(end.thread_cpu_ns - start.thread_cpu_ns) : "null") +
        ",\"valid\":" + (valid ? "true}" : "false}");
}

std::string serialize_host_timing(uint64_t start_ns, uint64_t end_ns,
                                  uint64_t start_tid, uint64_t end_tid) {
    static const std::string epoch = std::to_string(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
#if defined(__linux__) || defined(__APPLE__)
    const uint64_t pid = static_cast<uint64_t>(getpid());
    const char * const thread_kind = "os_tid";
#elif defined(_WIN32)
    const uint64_t pid = static_cast<uint64_t>(GetCurrentProcessId());
    const char * const thread_kind = "os_tid";
#else
    const uint64_t pid = 0;
    const char * const thread_kind = "process_thread_token";
#endif
    const bool valid = start_tid != 0 && end_tid != 0 && end_ns >= start_ns;
    return std::string("{\"execution_id\":\"") + std::to_string(pid) + "-" + epoch +
        "\",\"clock\":\"steady_clock\",\"unit\":\"nanosecond\",\"thread_id_kind\":\"" +
        thread_kind + "\",\"start_ns\":" + (start_tid != 0 ? std::to_string(start_ns) : "null") +
        ",\"end_ns\":" + (end_tid != 0 ? std::to_string(end_ns) : "null") +
        ",\"start_tid\":" + (start_tid != 0 ? std::to_string(start_tid) : "null") +
        ",\"end_tid\":" + (end_tid != 0 ? std::to_string(end_tid) : "null") +
        ",\"duration_ns\":" + (valid ? std::to_string(end_ns - start_ns) : "null") +
        ",\"valid\":" + (valid ? "true}" : "false}");
}

}

namespace ggml::gemmini::log
{
    CycleLog cycle;

    namespace
    {
        thread_local CycleWriteTiming *active_cycle_write_timing = nullptr;

        void append_json_escaped(std::string &out, const char *s)
        {
            if (!s)
            {
                return;
            }
            for (const unsigned char *p = reinterpret_cast<const unsigned char *>(s); *p; ++p)
            {
                const unsigned char c = *p;
                switch (c)
                {
                case '\\': out += "\\\\"; break;
                case '"':  out += "\\\""; break;
                case '\b': out += "\\b"; break;
                case '\f': out += "\\f"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                default:
                    if (c < 0x20)
                    {
                        char buf[7];
                        std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned int>(c));
                        out += buf;
                    }
                    else
                    {
                        out.push_back(static_cast<char>(c));
                    }
                    break;
                }
            }
        }
    } // namespace

    ScopedCycleWriteTiming::ScopedCycleWriteTiming(CycleWriteTiming &timing) noexcept
        : timing_(timing), previous_(active_cycle_write_timing), initial_calls_(timing.calls),
          initial_exceptions_(std::uncaught_exceptions())
    {
        active_cycle_write_timing = &timing;
#if !LOG_CYCLE
        timing.valid = false;
#endif
    }

    ScopedCycleWriteTiming::~ScopedCycleWriteTiming() noexcept
    {
        if (timing_.calls == initial_calls_ || std::uncaught_exceptions() > initial_exceptions_)
        {
            timing_.valid = false;
        }
        active_cycle_write_timing = previous_;
    }

    static std::string serialize_cycle_record_impl(
            const CycleRecord & record, bool linux_aarch64,
            bool provenance_available = false, bool checked_valid = false,
            const char * checked_reason = nullptr, const char * sample_reason = nullptr
    ) {
#if defined(__riscv)
        const char * const default_source = linux_aarch64 ? "linux_perf_cpu_cycles" : "riscv_cycle";
        const char * const default_unit = "cycle";
#else
        const char * const default_source = linux_aarch64 ? "linux_perf_cpu_cycles" : "host_tick";
        const char * const default_unit = linux_aarch64 ? "cycle" : "tick";
#endif
        const char * const source = record.source ? record.source : default_source;
        const char * const unit = record.unit ? record.unit : default_unit;
        bool valid = record.end >= record.start;
        const char * reason = nullptr;
        if (provenance_available) {
            valid = checked_valid;
            reason = checked_reason;
        } else
        if (linux_aarch64) {
            if (record.start == 0) {
                valid = false;
                reason = "invalid_start";
            } else if (record.end == 0) {
                valid = false;
                reason = "invalid_end";
            } else if (record.end < record.start) {
                valid = false;
                reason = "counter_regression";
            }
        }
        const uint64_t cycles = valid ? record.end - record.start : 0;
        std::string json;
        json.reserve(192);
        bool first = true;
        auto add_key = [&](const char *key) {
            if (!first) json.push_back(',');
            first = false;
            json.push_back('"');
            json += key;
            json += "\":";
        };
        auto add_string = [&](const char *key, const char *value) {
            if (!value || *value == '\0') return;
            add_key(key);
            json.push_back('"');
            append_json_escaped(json, value);
            json.push_back('"');
        };
        auto add_u64 = [&](const char *key, uint64_t value) {
            add_key(key);
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%llu", static_cast<unsigned long long>(value));
            json += buf;
        };
        auto add_null = [&](const char *key) {
            add_key(key);
            json += "null";
        };
        auto add_nullable_string = [&](const char *key, const char *value) {
            if (!value || *value == '\0') {
                add_null(key);
                return;
            }
            add_string(key, value);
        };
        auto add_identity = [&](const char *key, uint32_t flag, uint64_t value) {
            if ((record.identity_mask & flag) != 0) {
                add_u64(key, value);
            } else {
                add_null(key);
            }
        };
#if LOG_DETAIL
        auto add_i32 = [&](const char *key, int value) {
            add_key(key);
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%d", value);
            json += buf;
        };
#endif

        json.push_back('{');
        add_string("schema", "gemmini.cycle");
        add_u64("version", 2);
        add_string("record_type", "CYCLE_INTERVAL");
        add_string("source", source);
        add_string("unit", unit);
        add_nullable_string("op", record.op);
        add_nullable_string("layer", record.layer);
        add_identity("run_id", GEMMINI_CYCLE_HAS_RUN_ID, record.run_id);
        add_identity("stripe_id", GEMMINI_CYCLE_HAS_STRIPE_ID, record.stripe_id);
        add_identity("slot", GEMMINI_CYCLE_HAS_SLOT, record.slot);
        add_identity("node_id", GEMMINI_CYCLE_HAS_NODE_ID, record.node_id);
        add_identity("worker_id", GEMMINI_CYCLE_HAS_WORKER_ID, record.worker_id);
        add_u64("start", record.start);
        add_u64("end", record.end);
        if (linux_aarch64 && !valid) add_null("delta"); else add_u64("delta", cycles);
        add_key("valid");
        json += valid ? "true" : "false";
        if (linux_aarch64 && !valid) {
            add_string("reason", reason ? reason : "counter_regression");
            add_string("sample_reason", sample_reason);
        }
#if LOG_DETAIL
        add_string("file", record.file);
        if (record.file) add_i32("line", record.line);
        add_string("func", record.func);
#endif
        json += "}\n";
        return json;
    }

    std::string serialize_cycle_record(const CycleRecord & record)
    {
#if defined(__linux__) && defined(__aarch64__)
        return serialize_cycle_record_impl(record, true, false, false, nullptr);
#else
        return serialize_cycle_record_impl(record, false);
#endif
    }

    std::string serialize_checked_cycle_record(const CycleRecord & record, bool valid,
                                               const char * reason, const char * sample_reason)
    {
        return serialize_cycle_record_impl(record, true, true, valid, reason, sample_reason);
    }

    namespace testing
    {
        std::string serialize_linux_aarch64_scalar_cycle_record_for_test(const CycleRecord & record)
        {
#if defined(__linux__) && defined(__aarch64__)
            return serialize_cycle_record_impl(record, true, false, false, nullptr);
#else
            return serialize_cycle_record_impl(record, true);
#endif
        }
    }

    std::string serialize_ws_cycle_record(const WsCycleRecord &record)
    {
#if !LOG_CYCLE
        (void) record;
        return {};
#else
        std::string json =
            "{\"schema\":\"gemmini.cycle\",\"version\":2,"
            "\"record_type\":\"WS_LOOP_TELEMETRY\","
            "\"source\":\"gemmini_hw_counter\",\"unit\":\"cycle\","
            "\"op\":\"gemmini.ws_loop\",\"layer\":null,\"run_id\":null,"
            "\"stripe_id\":null,\"slot\":null,\"node_id\":null,\"worker_id\":null";
        auto add = [&json](const char *name, uint64_t value) {
            json += ",\"";
            json += name;
            json += "\":";
            json += std::to_string(value);
        };
        add("problem_i", record.problem_i); add("problem_j", record.problem_j); add("problem_k", record.problem_k);
        add("tile_i", record.tile_i); add("tile_j", record.tile_j); add("tile_k", record.tile_k);
        add("gemmini_outer_i", record.gemmini_outer_i);
        add("gemmini_outer_j", record.gemmini_outer_j);
        add("gemmini_outer_k", record.gemmini_outer_k);
        add("ws_inner_calls", record.ws_inner_calls);
        add("containing_interval_cycles", record.containing_interval_cycles);
        add("containing_interval_counter_bits", 64);
        add("load_occupancy_cycles", record.load_occupancy_cycles);
        add("execute_occupancy_cycles", record.execute_occupancy_cycles);
        add("store_occupancy_cycles", record.store_occupancy_cycles);
        add("loop_occupancy_cycles", record.loop_occupancy_cycles);
        add("occupancy_counter_bits", 32);
        const bool valid = record.containing_interval_cycles <= std::numeric_limits<uint32_t>::max() &&
            record.load_occupancy_cycles <= record.containing_interval_cycles &&
            record.execute_occupancy_cycles <= record.containing_interval_cycles &&
            record.store_occupancy_cycles <= record.containing_interval_cycles &&
            record.loop_occupancy_cycles <= record.containing_interval_cycles;
        json += valid ? ",\"valid\":true}" : ",\"valid\":false}";
        return json;
#endif
    }

    void CycleLog::warn_once_unlocked(const char *operation)
    {
        if (warned_)
        {
            return;
        }
        warned_ = true;
        std::fprintf(stderr, "gemmini CycleLog %s failure\n", operation);
        std::fflush(stderr);
    }

    void CycleLog::emit(const char *path, const std::string &json)
    {
#if LOG_CYCLE
        CycleWriteTiming * const timing = active_cycle_write_timing;
        const bool previously_valid = timing && timing->valid;
        if (timing)
        {
            ++timing->calls;
            timing->valid = false;
        }
        const auto wait_start = timing ? std::chrono::steady_clock::now() :
            std::chrono::steady_clock::time_point{};
        std::lock_guard<std::mutex> lock(detail::output_mutex());
        if (timing)
        {
            timing->mutex_wait_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - wait_start).count());
        }
        if (disabled_)
        {
            return;
        }

        FILE *output = out_;
        bool owns_call_output = false;
        if (path && *path)
        {
            const std::filesystem::path resolved = resolve_output_path(path);
            if (resolved.empty() || !prepare_output_parent(resolved) ||
                detail::consume_fault(testing::LogFault::open))
            {
                disabled_ = true;
                disable_output_unlocked();
                warn_once_unlocked("open");
                return;
            }
            output = std::fopen(resolved.string().c_str(), "a");
            owns_call_output = output != nullptr;
            if (!output)
            {
                disabled_ = true;
                disable_output_unlocked();
                warn_once_unlocked("open");
                return;
            }
        }
        if (!output)
        {
            return;
        }

        const bool write_fault = detail::consume_fault(testing::LogFault::write);
        const auto io_start = timing ? std::chrono::steady_clock::now() :
            std::chrono::steady_clock::time_point{};
        const std::size_t written = write_fault ? 0 : std::fwrite(json.data(), 1, json.size(), output);
        const bool flush_fault = detail::consume_fault(testing::LogFault::flush);
        const int flushed = flush_fault ? EOF : std::fflush(output);
        if (timing)
        {
            timing->io_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - io_start).count());
            timing->valid = previously_valid && written == json.size() && flushed == 0;
        }
        if (owns_call_output)
        {
            const int closed = std::fclose(output);
            if (timing && closed != 0) timing->valid = false;
        }
        if (written != json.size() || flushed != 0)
        {
            disabled_ = true;
            disable_output_unlocked();
            warn_once_unlocked(written != json.size() ? "write" : "flush");
        }
#else
        (void)path;
        (void)json;
#endif
    }

    void CycleLog::set_output(FILE *out)
    {
        std::lock_guard<std::mutex> lock(detail::output_mutex());
        set_output_unlocked(out);
        disabled_ = false;
        warned_ = false;
    }

    bool CycleLog::set_output_path(const char *path, bool truncate)
    {
#if LOG_CYCLE
        std::lock_guard<std::mutex> lock(detail::output_mutex());
        const char *failure = nullptr;
        if (!set_output_path_unlocked(path, truncate, &failure))
        {
            warn_once_unlocked(failure ? failure : "setup");
            return false;
        }
        disabled_ = false;
        warned_ = false;
        return true;
#else
        (void)path;
        (void)truncate;
        return true;
#endif
    }

    void CycleLog::report_failure(const char * operation) noexcept
    {
#if LOG_CYCLE
        if (active_cycle_write_timing) active_cycle_write_timing->valid = false;
        try
        {
            std::lock_guard<std::mutex> lock(detail::output_mutex());
            warn_once_unlocked(operation);
        }
        catch (...)
        {
            // A C API failure path must never propagate through the language boundary.
        }
#else
        (void) operation;
#endif
    }

    void CycleLog::write(const CycleRecord &record)
    {
        emit(nullptr, serialize_cycle_record(record));
    }

    void CycleLog::write_json(std::string_view json_record)
    {
#if LOG_CYCLE
        if (json_record.empty())
        {
            return;
        }
        std::string line(json_record);
        if (line.back() != '\n')
        {
            line.push_back('\n');
        }
        emit(nullptr, line);
#else
        (void)json_record;
#endif
    }

    void CycleLog::operator()(const char *layer, const char *op, uint64_t start, uint64_t end)
    {
        write(CycleRecord{layer, op, start, end, nullptr, 0, nullptr});
    }

    void CycleLog::operator()(const char *file, int line, const char *func, const char *layer, const char *op,
                              uint64_t start, uint64_t end)
    {
        write(CycleRecord{layer, op, start, end, file, line, func});
    }

    void CycleLog::operator()(LogTarget target, const char *layer, const char *op, uint64_t start, uint64_t end)
    {
        emit(target.path, serialize_cycle_record(CycleRecord{layer, op, start, end, nullptr, 0, nullptr}));
    }

    void CycleLog::operator()(LogTarget target, const char *file, int line, const char *func, const char *layer,
                              const char *op, uint64_t start, uint64_t end)
    {
        emit(target.path, serialize_cycle_record(CycleRecord{layer, op, start, end, file, line, func}));
    }

    void CycleLog::cycle(const char *layer, const char *op, uint64_t start, uint64_t end)
    {
        write(CycleRecord{layer, op, start, end, nullptr, 0, nullptr});
    }

} // namespace ggml::gemmini::log
