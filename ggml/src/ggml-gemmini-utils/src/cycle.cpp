#include "../include/gemmini/log.hpp"
#include "../include/gemmini/host-timing.hpp"
#include "../include/gemmini/cycle_reader.hpp"
#include "../include/gemmini/performance.hpp"
#include "../include/gemmini/trace-context.hpp"

#include <limits>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <ctime>
#include <cstring>
#include <exception>
#include <mutex>
#include <new>
#include <string>
#include <type_traits>
#include <variant>
#include <sys/stat.h>

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <pthread.h>
#include <unistd.h>
#elif defined(_WIN32)
#include <io.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {
std::string serialize_cpu_record(const gemmini_cycle_record_v2 &identity,
    const gemmini_cpu_sample &start, const gemmini_cpu_sample &end, bool raw_segment);
}

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

uint64_t timeline_now_ns() noexcept {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

HostSample read_host_sample() noexcept {
    HostSample sample;
    sample.ns = timeline_now_ns();
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

const std::string &host_execution_id() {
    static const std::string id = [] {
        const auto epoch = std::to_string(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
#if defined(__linux__) || defined(__APPLE__)
        const uint64_t pid = static_cast<uint64_t>(getpid());
#elif defined(_WIN32)
        const uint64_t pid = static_cast<uint64_t>(GetCurrentProcessId());
#else
        const uint64_t pid = 0;
#endif
        return std::to_string(pid) + "-" + epoch;
    }();
    return id;
}

std::string serialize_host_timing(uint64_t start_ns, uint64_t end_ns,
                                  uint64_t start_tid, uint64_t end_tid) {
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    const char * const thread_kind = "os_tid";
#else
    const char * const thread_kind = "process_thread_token";
#endif
    const bool valid = start_tid != 0 && end_tid != 0 && end_ns >= start_ns;
    return std::string("{\"execution_id\":\"") + host_execution_id() +
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
#if CYCLE_DETAIL
#if defined(__riscv)
        const char * const default_source = linux_aarch64 ? "linux_perf_cpu_cycles" : "riscv_cycle";
        const char * const default_unit = "cycle";
#else
        const char * const default_source = linux_aarch64 ? "linux_perf_cpu_cycles" : "host_tick";
        const char * const default_unit = linux_aarch64 ? "cycle" : "tick";
#endif
        const char * const source = record.source ? record.source : default_source;
        const char * const unit = record.unit ? record.unit : default_unit;
#endif
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
#if CYCLE_DETAIL
        auto add_identity = [&](const char *key, uint32_t flag, uint64_t value) {
            if ((record.identity_mask & flag) != 0) {
                add_u64(key, value);
            } else {
                add_null(key);
            }
        };
#endif
#if LOG_DETAIL
        auto add_i32 = [&](const char *key, int value) {
            add_key(key);
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%d", value);
            json += buf;
        };
#endif

        json.push_back('{');
#if CYCLE_DETAIL
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
#else
        add_nullable_string("op", record.op);
        add_string("kind", "cycle");
        add_nullable_string("layer", record.layer);
        if (record.identity_mask & GEMMINI_CYCLE_HAS_RUN_ID) add_u64("run_id", record.run_id);
        if (record.identity_mask & GEMMINI_CYCLE_HAS_STRIPE_ID) add_u64("stripe_id", record.stripe_id);
        if (record.identity_mask & GEMMINI_CYCLE_HAS_SLOT) add_u64("slot", record.slot);
        if (record.identity_mask & GEMMINI_CYCLE_HAS_NODE_ID) add_u64("node_id", record.node_id);
        if (record.identity_mask & GEMMINI_CYCLE_HAS_WORKER_ID) add_u64("worker_id", record.worker_id);
        if (record.host_timing_valid) {
            add_u64("ns_start", record.ns_start);
            add_u64("ns_end", record.ns_end);
            if (record.tid_start != 0 && record.tid_start == record.tid_end) {
                add_u64("tid", record.tid_start);
            } else {
                add_u64("tid_start", record.tid_start);
                add_u64("tid_end", record.tid_end);
            }
        }
#endif
        add_u64("start", record.start);
        add_u64("end", record.end);
#if CYCLE_DETAIL
        if (linux_aarch64 && !valid) add_null("delta"); else add_u64("delta", cycles);
#else
        if (!valid) add_null("delta"); else add_u64("delta", cycles);
#endif
        add_key("valid");
        json += valid ? "true" : "false";
#if CYCLE_DETAIL
        if (linux_aarch64 && !valid) {
            add_string("reason", reason ? reason : "counter_regression");
            add_string("sample_reason", sample_reason);
        }
#else
        if (!valid) {
            add_string("reason", reason ? reason : "counter_regression");
            add_string("sample_reason", sample_reason);
        }
#endif
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
            "\"op\":\"gemmini.ws_loop\"";
        auto add = [&json](const char *name, uint64_t value) {
            json += ",\"";
            json += name;
            json += "\":";
            json += std::to_string(value);
        };
        auto add_text = [&json](const char *name, const char *value) {
            json += ",\"";
            json += name;
            json += "\":";
            if (value != nullptr) {
                json += '"';
                append_json_escaped(json, value);
                json += '"';
            } else {
                json += "null";
            }
        };
        auto add_identity = [&](const char *name, uint32_t bit, uint64_t value) {
            if (record.identity.identity_mask & bit) add(name, value);
            else add_text(name, nullptr);
        };
        add_text("layer", record.identity.interval.layer);
        add_text("domain", record.domain);
        add_identity("run_id", GEMMINI_CYCLE_HAS_RUN_ID, record.identity.run_id);
        add_identity("stripe_id", GEMMINI_CYCLE_HAS_STRIPE_ID, record.identity.stripe_id);
        add_identity("slot", GEMMINI_CYCLE_HAS_SLOT, record.identity.slot);
        add_identity("node_id", GEMMINI_CYCLE_HAS_NODE_ID, record.identity.node_id);
        add_identity("worker_id", GEMMINI_CYCLE_HAS_WORKER_ID, record.identity.worker_id);
        add("problem_i", record.problem_i); add("problem_j", record.problem_j); add("problem_k", record.problem_k);
        add("tile_i", record.tile_i); add("tile_j", record.tile_j); add("tile_k", record.tile_k);
        add("gemmini_outer_i", record.gemmini_outer_i);
        add("gemmini_outer_j", record.gemmini_outer_j);
        add("gemmini_outer_k", record.gemmini_outer_k);
        add("ws_inner_calls", record.ws_inner_calls);
        add("containing_interval_cycles", record.containing_interval_cycles);
        add("containing_interval_counter_bits", 64);
        add_text("containing_interval_source", record.containing_interval_source);
        add_text("containing_interval_domain", "cpu_counter");
        add_text("containing_interval_unit", "cycle");
        add("load_occupancy_cycles", record.load_occupancy_cycles);
        add("execute_occupancy_cycles", record.execute_occupancy_cycles);
        add("store_occupancy_cycles", record.store_occupancy_cycles);
        add("loop_occupancy_cycles", record.loop_occupancy_cycles);
        add("occupancy_counter_bits", 32);
        add_text("occupancy_counter_semantics", "raw_modulo_2^32_readings");
        json += ",\"valid\":false,\"reason\":\"device_counter_window_and_wrap_unverified\","
                "\"aggregation_role\":\"diagnostic\",\"additive\":false}";
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

    struct CycleLog::Entry
    {
        struct Cpu
        {
            gemmini_cycle_record_v2 identity;
            gemmini_cpu_sample start, end;
            std::string layer, op;
            uint64_t sequence = 0;
            std::optional<bool> operation_success;
            bool raw_segment = false;
            bool structural_envelope = false;
        };
        performance::Context context;
        std::variant<std::string, Cpu, performance::Measurement> value;
        gemmini_trace_context trace_context{};
        uint64_t segment_id = 0;

        std::size_t owned_bytes() const
        {
            return std::visit([](const auto &record) -> std::size_t {
                using T = std::decay_t<decltype(record)>;
                if constexpr (std::is_same_v<T, std::string>) return record.capacity() + 1;
                else if constexpr (std::is_same_v<T, Cpu>)
                    return record.layer.capacity() + record.op.capacity() + 2;
                else return record.cycles_reason.capacity() + record.thread_reason.capacity() +
                    record.backend.capacity() + record.domain.capacity() + record.metric.capacity() +
                    record.reason.capacity() + 6;
            }, value);
        }

        std::string serialize() const
        {
            if (detail::consume_fault(testing::LogFault::format)) throw std::bad_alloc();
            std::string json = std::visit([](const auto &record) -> std::string {
                using T = std::decay_t<decltype(record)>;
                if constexpr (std::is_same_v<T, std::string>) return record;
                else if constexpr (std::is_same_v<T, Cpu>) {
                    auto identity = record.identity;
                    identity.interval.layer = record.layer.c_str();
                    identity.interval.op = record.op.c_str();
                    auto json = serialize_cpu_record(identity, record.start, record.end, record.raw_segment);
                    if (record.operation_success.has_value()) json.insert(json.size() - 1,
                        std::string(",\"operation_success\":") + (*record.operation_success ? "true" : "false"));
                    if (record.sequence) json.insert(json.size() - 1,
                        ",\"cpu_interval_sequence\":" + std::to_string(record.sequence));
                    return json;
                } else return performance::serialize_measurement(record);
            }, value);
            if (json.empty()) return json;
            if ((trace_context.flags & GEMMINI_TRACE_OPERATOR) || trace_context.task_id) {
                std::string decorated;
                for (std::size_t begin = 0; begin < json.size();) {
                    const auto next = json.find('\n', begin);
                    const auto end = next == std::string::npos ? json.size() : next;
                    if (end > begin) decorated += trace::append_metadata(json.substr(begin, end-begin),
                        trace_context, begin == 0 ? segment_id : gemmini_trace_reserve_ids(1),
                        begin == 0 && std::holds_alternative<Cpu>(value) &&
                            std::get<Cpu>(value).structural_envelope);
                    if (next != std::string::npos) decorated += '\n';
                    begin = end+1;
                }
                json = std::move(decorated);
            }
            if (json.back() != '\n') json += '\n';
            const std::string context_json = performance::serialize_context(context);
            if (context_json.empty()) return json;
            std::string contextual;
            for (std::size_t begin = 0; begin < json.size();) {
                const auto newline = json.find('\n', begin);
                const auto end = newline == std::string::npos ? json.size() : newline;
                const auto close = end > begin ? json.rfind('}', end - 1) : std::string::npos;
                const auto open = json.find_first_not_of(" \t\r", begin);
                if (open < end && json[open] == '{' && close != std::string::npos && close > open &&
                    json.find("\"inference_context\":", begin) >= end) {
                    contextual.append(json, begin, close - begin);
                    if (json.find_first_not_of(" \t\r", open + 1) != close) contextual += ',';
                    contextual += "\"inference_context\":" + context_json;
                    contextual.append(json, close, end - close);
                } else contextual.append(json, begin, end - begin);
                if (newline != std::string::npos) contextual += '\n';
                begin = end + 1;
            }
            return contextual;
        }
    };

    struct CycleLog::WorkerBuffer
    {
        // Operations needing both locks always take the output mutex before this mutex.
        std::mutex mutex;
        std::mutex *output_mutex = nullptr;
        CycleLog *owner = nullptr;
        std::vector<Entry> entries;
        std::size_t bytes = 0, peak_entries = 0, peak_bytes = 0;

        ~WorkerBuffer()
        {
            if (!output_mutex) return;
            std::lock_guard<std::mutex> output_lock(*output_mutex);
            std::lock_guard<std::mutex> worker_lock(mutex);
            if (owner) {
                owner->drain_worker_unlocked(*this);
                auto &workers = owner->workers_;
                workers.erase(std::find(workers.begin(), workers.end(), this));
            }
        }
    };

    void CycleLog::update_queue_enabled_unlocked()
    {
        queue_enabled_.store(buffered_ && regular_output_ && !disabled_ && !lost_records_,
                             std::memory_order_release);
    }

    bool CycleLog::enqueue(Entry &entry)
    {
        if (!queue_enabled_.load(std::memory_order_acquire)) return false;
        thread_local WorkerBuffer worker;
        const std::size_t owned = entry.owned_bytes();
        const auto fits = [&] {
            return worker.bytes <= BufferStats::max_bytes &&
                worker.entries.size() < BufferStats::max_entries &&
                owned <= BufferStats::max_bytes - worker.bytes;
        };
        const auto push = [&] {
            worker.entries.push_back(std::move(entry));
            worker.bytes += owned;
            worker.peak_entries = std::max(worker.peak_entries, worker.entries.size());
            worker.peak_bytes = std::max(worker.peak_bytes, worker.bytes);
        };
        {
            std::lock_guard<std::mutex> lock(worker.mutex);
            if (worker.owner == this && queue_enabled_.load(std::memory_order_acquire) && fits()) {
                push();
                return true;
            }
        }
        auto &output_mutex = detail::output_mutex();
        std::lock_guard<std::mutex> output_lock(output_mutex);
        std::lock_guard<std::mutex> worker_lock(worker.mutex);
        if (worker.owner && worker.owner != this) {
            worker.owner->drain_worker_unlocked(worker);
            auto &workers = worker.owner->workers_;
            workers.erase(std::find(workers.begin(), workers.end(), &worker));
            worker.owner = nullptr;
        }
        if (!queue_enabled_.load(std::memory_order_acquire)) return false;
        if (!worker.owner) {
            worker.entries.reserve(BufferStats::max_entries);
            workers_.push_back(&worker);
            worker.owner = this;
            worker.output_mutex = &output_mutex;
            worker.bytes = worker.entries.capacity() * sizeof(Entry);
            worker.peak_entries = 0;
            worker.peak_bytes = worker.bytes;
        }
        if (!fits()) drain_worker_unlocked(worker);
        if (!queue_enabled_.load(std::memory_order_acquire) || !fits()) return false;
        push();
        return true;
    }

    bool CycleLog::drain_worker_unlocked(WorkerBuffer &worker)
    {
        bool ok = true;
        for (const auto &entry : worker.entries) {
            try {
                if (!emit_unlocked(nullptr, entry.serialize())) {
                    ok = false;
                    break;
                }
            }
            catch (...) {
                ok = false;
                lost_records_ = true;
                update_queue_enabled_unlocked();
                warn_once_unlocked("serialization");
            }
        }
        peak_entries_ = std::max(peak_entries_, worker.peak_entries);
        peak_bytes_ = std::max(peak_bytes_, worker.peak_bytes);
        worker.entries.clear();
        worker.bytes = worker.entries.capacity() * sizeof(Entry);
        return ok;
    }

    bool CycleLog::drain_unlocked()
    {
        bool ok = true;
        for (auto *worker : workers_) {
            std::lock_guard<std::mutex> lock(worker->mutex);
            if (!drain_worker_unlocked(*worker)) ok = false;
        }
        return ok;
    }

    CycleLog::~CycleLog()
    {
        std::lock_guard<std::mutex> lock(detail::output_mutex());
        queue_enabled_.store(false, std::memory_order_release);
        drain_unlocked();
        if (buffered_ && owns_output_unlocked()) flush_unlocked();
        for (auto *worker : workers_) {
            std::lock_guard<std::mutex> worker_lock(worker->mutex);
            worker->owner = nullptr;
        }
    }

    void CycleLog::submit(Entry entry, const char *path)
    {
#if LOG_CYCLE
        if (detail::consume_fault(testing::LogFault::allocation)) throw std::bad_alloc();
        if ((!path || !*path) && !active_cycle_write_timing && enqueue(entry)) return;
        const std::string json = entry.serialize();
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
        drain_unlocked();
        const bool ok = emit_unlocked(path, json, timing);
        if (timing) timing->valid = previously_valid && ok;
#else
        (void)path;
        (void)entry;
#endif
    }

    bool CycleLog::emit_unlocked(const char *path, const std::string &json, CycleWriteTiming *timing)
    {
        if (disabled_) return false;
        FILE *output = out_;
        bool owns_call_output = false;
        if (path && *path)
        {
            if (buffered_ && owns_output_unlocked() && !flush_unlocked()) return false;
            const std::filesystem::path resolved = resolve_output_path(path);
            if (resolved.empty() || !prepare_output_parent(resolved) ||
                detail::consume_fault(testing::LogFault::open))
            {
                disabled_ = true;
                update_queue_enabled_unlocked();
                disable_output_unlocked();
                warn_once_unlocked("open");
                return false;
            }
            output = std::fopen(resolved.string().c_str(), "a");
            owns_call_output = output != nullptr;
            if (!output)
            {
                disabled_ = true;
                update_queue_enabled_unlocked();
                disable_output_unlocked();
                warn_once_unlocked("open");
                return false;
            }
        }
        if (!output)
        {
            return false;
        }

        const bool write_fault = detail::consume_fault(testing::LogFault::write);
        const auto io_start = timing ? std::chrono::steady_clock::now() :
            std::chrono::steady_clock::time_point{};
        const std::size_t written = write_fault ? 0 : std::fwrite(json.data(), 1, json.size(), output);
        const bool flush_now = owns_call_output || !buffered_ || !regular_output_ ||
            !owns_output_unlocked();
        const int flushed = flush_now ?
            (detail::consume_fault(testing::LogFault::flush) ? EOF : std::fflush(output)) : 0;
        if (timing) {
            timing->io_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - io_start).count());
        }
        const int closed = owns_call_output ? std::fclose(output) : 0;
        if (written != json.size() || flushed != 0 || closed != 0)
        {
            disabled_ = true;
            update_queue_enabled_unlocked();
            disable_output_unlocked();
            warn_once_unlocked(written != json.size() ? "write" : "flush");
            return false;
        }
        return true;
    }

    void CycleLog::emit(const char *path, const std::string &json) try
    {
        const auto captured = gemmini_trace_capture();
        submit({trace::inference_context(captured), json, captured, gemmini_trace_reserve_ids(1)}, path);
    }
    catch (...) { report_failure("capture"); throw; }

    void CycleLog::set_output(FILE *out)
    {
        std::lock_guard<std::mutex> lock(detail::output_mutex());
        queue_enabled_.store(false, std::memory_order_release);
        drain_unlocked();
        if (buffered_ && owns_output_unlocked()) flush_unlocked();
        set_output_unlocked(out);
        output_path_.clear();
        regular_output_ = false;
        disabled_ = false;
        warned_ = false;
        lost_records_ = false;
    }

    bool CycleLog::set_output_path(const char *path, bool truncate)
    {
#if LOG_CYCLE
        std::lock_guard<std::mutex> lock(detail::output_mutex());
        const auto resolved = resolve_output_path(path);
        queue_enabled_.store(false, std::memory_order_release);
        if (!drain_unlocked()) return false;
        if (buffered_ && owns_output_unlocked() && !flush_unlocked()) return false;
        const char *failure = nullptr;
        if (!set_output_path_unlocked(path, truncate, &failure))
        {
            warn_once_unlocked(failure ? failure : "setup");
            update_queue_enabled_unlocked();
            return false;
        }
        disabled_ = false;
        warned_ = false;
        lost_records_ = false;
#if defined(_WIN32)
        struct _stat status{};
        regular_output_ = owns_output_unlocked() && _fstat(_fileno(out_), &status) == 0 &&
            (status.st_mode & _S_IFMT) == _S_IFREG;
#else
        struct stat status{};
        regular_output_ = owns_output_unlocked() && fstat(fileno(out_), &status) == 0 &&
            S_ISREG(status.st_mode);
#endif
        output_path_ = regular_output_ ? resolved : std::filesystem::path{};
        update_queue_enabled_unlocked();
        return true;
#else
        (void)path;
        (void)truncate;
        return true;
#endif
    }

    void CycleLog::set_buffered(bool buffered)
    {
#if LOG_CYCLE
        std::lock_guard<std::mutex> lock(detail::output_mutex());
        queue_enabled_.store(false, std::memory_order_release);
        drain_unlocked();
        if (buffered_ && !buffered) flush_unlocked();
        buffered_ = buffered;
        update_queue_enabled_unlocked();
#else
        (void)buffered;
#endif
    }

    bool CycleLog::flush_unlocked()
    {
        if (disabled_) return false;
        if (!out_) return true;
        if (!detail::consume_fault(testing::LogFault::flush) && std::fflush(out_) == 0) return true;
        disabled_ = true;
        update_queue_enabled_unlocked();
        disable_output_unlocked();
        warn_once_unlocked("flush");
        return false;
    }

    bool CycleLog::flush()
    {
#if LOG_CYCLE
        std::lock_guard<std::mutex> lock(detail::output_mutex());
        drain_unlocked();
        return flush_unlocked() && !lost_records_;
#else
        return true;
#endif
    }

    bool CycleLog::healthy() const
    {
        std::lock_guard<std::mutex> lock(detail::output_mutex());
        return !disabled_ && !lost_records_;
    }

    std::filesystem::path CycleLog::output_path() const
    {
        std::lock_guard<std::mutex> lock(detail::output_mutex());
        return output_path_;
    }

    CycleLog::BufferStats CycleLog::buffer_stats_for_test() const
    {
        std::lock_guard<std::mutex> lock(detail::output_mutex());
        BufferStats stats{workers_.size(), peak_entries_, peak_bytes_};
        for (auto *worker : workers_) {
            std::lock_guard<std::mutex> worker_lock(worker->mutex);
            stats.peak_entries = std::max(stats.peak_entries, worker->peak_entries);
            stats.peak_bytes = std::max(stats.peak_bytes, worker->peak_bytes);
        }
        return stats;
    }

    void CycleLog::report_failure(const char * operation) noexcept
    {
#if LOG_CYCLE
        if (active_cycle_write_timing) active_cycle_write_timing->valid = false;
        try
        {
            std::lock_guard<std::mutex> lock(detail::output_mutex());
            lost_records_ = true;
            update_queue_enabled_unlocked();
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

    void CycleLog::write(const CycleRecord &record) try
    {
        emit(nullptr, serialize_cycle_record(record));
    }
    catch (...) { report_failure("serialization"); throw; }

    void CycleLog::write_json(std::string_view json_record) try
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
    catch (...) { report_failure("capture"); throw; }

    void CycleLog::write_cpu(const gemmini_cycle_record_v2 &identity,
                            const gemmini_cpu_sample &start, const gemmini_cpu_sample &end,
                            std::optional<bool> operation_success, bool raw_segment,
                            bool structural_envelope) try
    {
#if LOG_CYCLE
        const auto captured = start.trace.flags & GEMMINI_TRACE_CAPTURED ? start.trace : gemmini_trace_capture();
        const auto context = trace::inference_context(captured);
        Entry::Cpu cpu{identity, start, end,
            identity.interval.layer ? identity.interval.layer : "",
            identity.interval.op ? identity.interval.op : "",
            0, std::nullopt, false, false};
        if (!raw_segment && this == &ggml::gemmini::log::cycle)
            cpu.sequence = performance::next_cpu_interval_sequence(context);
        cpu.operation_success = operation_success;
        cpu.raw_segment = raw_segment;
        cpu.structural_envelope = structural_envelope;
        cpu.identity.interval.layer = cpu.identity.interval.op = nullptr;
        cpu.identity.interval.file = cpu.identity.interval.func = nullptr;
        submit({context, std::move(cpu), captured, gemmini_trace_reserve_ids(1)});
#else
        (void)identity; (void)start; (void)end; (void)operation_success; (void)raw_segment;
        (void)structural_envelope;
#endif
    }
    catch (...) { report_failure("CPU interval"); throw; }

    void CycleLog::write_measurement(const performance::Measurement &measurement) try
    {
#if LOG_CYCLE
        const auto captured = measurement.trace.flags & GEMMINI_TRACE_CAPTURED ? measurement.trace : gemmini_trace_capture();
        submit({trace::inference_context(captured), measurement, captured, gemmini_trace_reserve_ids(1)});
#else
        (void)measurement;
#endif
    }
    catch (...) { report_failure("resource measurement"); throw; }

    void CycleLog::operator()(const char *layer, const char *op, uint64_t start, uint64_t end)
    {
        write(CycleRecord{layer, op, start, end, nullptr, 0, nullptr});
    }

    void CycleLog::operator()(const char *file, int line, const char *func, const char *layer, const char *op,
                              uint64_t start, uint64_t end)
    {
        write(CycleRecord{layer, op, start, end, file, line, func});
    }

    void CycleLog::operator()(LogTarget target, const char *layer, const char *op, uint64_t start, uint64_t end) try
    {
        emit(target.path, serialize_cycle_record(CycleRecord{layer, op, start, end, nullptr, 0, nullptr}));
    }
    catch (...) { report_failure("serialization"); throw; }

    void CycleLog::operator()(LogTarget target, const char *file, int line, const char *func, const char *layer,
                              const char *op, uint64_t start, uint64_t end) try
    {
        emit(target.path, serialize_cycle_record(CycleRecord{layer, op, start, end, file, line, func}));
    }
    catch (...) { report_failure("serialization"); throw; }

    void CycleLog::cycle(const char *layer, const char *op, uint64_t start, uint64_t end)
    {
        write(CycleRecord{layer, op, start, end, nullptr, 0, nullptr});
    }

} // namespace ggml::gemmini::log

namespace {
void cpu_sum(uint64_t &total, uint64_t value, const char *&reason) {
    if (value > std::numeric_limits<uint64_t>::max() - total) {
        reason = "aggregate_overflow";
    } else {
        total += value;
    }
}
}

extern "C" gemmini_cpu_sample gemmini_cpu_timing_read(void) {
    gemmini_cpu_sample result{};
#if LOG_CYCLE
    result.trace = gemmini_trace_capture();
    const auto host = ggml::gemmini::cycle::read_host_sample();
    result.ns = host.ns;
    result.tid = host.tid;
    result.thread_cpu_ns = host.thread_cpu_ns;
    result.thread_cpu_valid = host.thread_cpu_valid;
#if defined(__linux__) && defined(__aarch64__)
    const auto native = ggml::gemmini::cycle::read_sample();
    result.counter = native.value;
    result.owner_token = native.owner_event_token;
    result.generation = native.generation;
    result.native_valid = native.valid;
    result.native_reason = static_cast<uint8_t>(native.reason);
    result.native_source = GEMMINI_CPU_COUNTER_THREAD_PERF;
#endif
#endif
    return result;
}

namespace {
std::string cpu_json_string(const char *value) {
    if (!value || !*value) return "null";
    std::string json = "\"";
    ggml::gemmini::log::append_json_escaped(json, value);
    return json + '"';
}

gemmini_cpu_totals evaluate_cpu_interval(const gemmini_cpu_sample *start,
                                        const gemmini_cpu_sample *end) {
    gemmini_cpu_totals interval{};
    interval.interval_count = 1;
    if (start->trace.task_id != 0 && end->trace.task_id != 0 &&
        start->trace.task_id != end->trace.task_id) {
        interval.cycles_reason = interval.thread_cpu_reason = "structurally_cross_task";
        return interval;
    }
    const bool same_thread = start->tid != 0 && start->tid == end->tid;
    if (!same_thread) {
        interval.thread_cpu_reason = "thread_mismatch";
    } else if (!start->thread_cpu_valid || !end->thread_cpu_valid) {
        interval.thread_cpu_reason = "unavailable_thread_cpu_time";
    } else if (end->thread_cpu_ns < start->thread_cpu_ns) {
        interval.thread_cpu_reason = "counter_regression";
    } else {
        interval.thread_cpu_ns = end->thread_cpu_ns - start->thread_cpu_ns;
        interval.thread_cpu_valid_count = 1;
    }
#if defined(__linux__) && defined(__aarch64__)
    using namespace ggml::gemmini::cycle;
    if (!same_thread) {
        interval.cycles_reason = "thread_mismatch";
    } else if (start->native_source != GEMMINI_CPU_COUNTER_THREAD_PERF ||
               end->native_source != GEMMINI_CPU_COUNTER_THREAD_PERF) {
        interval.cycles_reason = "source_mismatch";
    } else {
        const NativeCycleSample native_start{start->counter, start->native_valid != 0,
            static_cast<NativeCycleReason>(start->native_reason), NativeCycleSource::perf_cpu_cycles,
            start->owner_token, start->generation};
        const NativeCycleSample native_end{end->counter, end->native_valid != 0,
            static_cast<NativeCycleReason>(end->native_reason), NativeCycleSource::perf_cpu_cycles,
            end->owner_token, end->generation};
        const auto delta = evaluate_interval(native_start, native_end);
        if (delta.valid) {
            interval.cycles = delta.value;
            interval.cycles_valid_count = 1;
        } else {
            interval.cycles_reason = reason_name(delta.sample_reason != NativeCycleReason::none
                ? delta.sample_reason : delta.reason);
        }
    }
#else
    interval.cycles_reason = "not_thread_cpu_counter";
#endif
    return interval;
}
}

extern "C" void gemmini_cpu_timing_add(gemmini_cpu_totals *totals,
        const gemmini_cpu_sample *start, const gemmini_cpu_sample *end) {
    const auto interval = evaluate_cpu_interval(start, end);
    gemmini_cpu_timing_merge(totals, &interval);
#if LOG_CYCLE
    ggml::gemmini::performance::record_cpu(*start, *end, interval);
#endif
}

extern "C" void gemmini_cpu_timing_merge(gemmini_cpu_totals *totals,
                                         const gemmini_cpu_totals *other) {
    if (!totals->cycles_reason) totals->cycles_reason = other->cycles_reason;
    if (!totals->thread_cpu_reason) totals->thread_cpu_reason = other->thread_cpu_reason;
    const char *count_reason = nullptr;
    cpu_sum(totals->interval_count, other->interval_count, count_reason);
    cpu_sum(totals->cycles, other->cycles, totals->cycles_reason);
    cpu_sum(totals->cycles_valid_count, other->cycles_valid_count, totals->cycles_reason);
    cpu_sum(totals->thread_cpu_ns, other->thread_cpu_ns, totals->thread_cpu_reason);
    cpu_sum(totals->thread_cpu_valid_count, other->thread_cpu_valid_count, totals->thread_cpu_reason);
    if (count_reason) totals->cycles_reason = totals->thread_cpu_reason = count_reason;
}

namespace ggml::gemmini::cycle {
std::string serialize_cpu_totals(const gemmini_cpu_totals &totals) {
    const auto reason_json = [](const char *reason) {
        if (!reason) return std::string("null");
        std::string value = "\"";
        log::append_json_escaped(value, reason);
        return value + '"';
    };
    const bool cycles_valid = totals.interval_count != 0 && !totals.cycles_reason &&
        totals.cycles_valid_count == totals.interval_count;
    const bool thread_valid = totals.interval_count != 0 && !totals.thread_cpu_reason &&
        totals.thread_cpu_valid_count == totals.interval_count;
    const char *cycles_reason = cycles_valid ? nullptr :
        (totals.cycles_reason ? totals.cycles_reason : "not_collected");
    const char *thread_reason = thread_valid ? nullptr :
        (totals.thread_cpu_reason ? totals.thread_cpu_reason : "not_collected");
    return std::string("{\"interval_count\":") + std::to_string(totals.interval_count) +
#if defined(__linux__) && defined(__aarch64__)
        ",\"cycles_source\":\"linux_perf_cpu_cycles\"" +
#else
        ",\"cycles_source\":null" +
#endif
        ",\"cycles\":" + (cycles_valid ? std::to_string(totals.cycles) : "null") +
        ",\"cycles_valid_count\":" + std::to_string(totals.cycles_valid_count) +
        ",\"cycles_reason\":" + reason_json(cycles_reason) +
        ",\"thread_cpu_ns\":" + (thread_valid ? std::to_string(totals.thread_cpu_ns) : "null") +
        ",\"thread_cpu_valid_count\":" + std::to_string(totals.thread_cpu_valid_count) +
        ",\"thread_cpu_reason\":" + reason_json(thread_reason) + '}';
}

std::string serialize_cpu_native(const gemmini_cpu_sample &start, const gemmini_cpu_sample &end) {
    const auto sample_json = [](const gemmini_cpu_sample &sample) {
        const bool source_valid = sample.native_source == GEMMINI_CPU_COUNTER_THREAD_PERF;
        const bool valid = source_valid && sample.native_valid;
        const char *reason = !source_valid ? "not_thread_cpu_counter" :
            (valid ? nullptr : "unavailable_sample");
#if defined(__linux__) && defined(__aarch64__)
        if (source_valid && sample.native_reason) {
            reason = reason_name(static_cast<NativeCycleReason>(sample.native_reason));
        }
#endif
        return std::string("{\"value\":") + (source_valid ? std::to_string(sample.counter) : "null") +
            ",\"valid\":" + (valid ? "true" : "false") +
            ",\"source\":" + (source_valid ? "\"linux_perf_cpu_cycles\"" : "null") +
            ",\"reason\":" + cpu_json_string(reason) +
            ",\"owner_token\":" + (source_valid ? std::to_string(sample.owner_token) : "null") +
            ",\"generation\":" + (source_valid ? std::to_string(sample.generation) : "null") + '}';
    };
    const auto interval = evaluate_cpu_interval(&start, &end);
    const bool valid = interval.cycles_valid_count == 1 && !interval.cycles_reason;
    return std::string("{\"start\":") + sample_json(start) +
        ",\"end\":" + sample_json(end) +
        ",\"delta\":" + (valid ? std::to_string(interval.cycles) : "null") +
        ",\"valid\":" + (valid ? "true" : "false") +
        ",\"reason\":" + cpu_json_string(interval.cycles_reason) + '}';
}

void WorkerCpuTiming::observe(void *context, bool begin) noexcept {
    auto &timing = *static_cast<WorkerCpuTiming *>(context);
    if (begin) {
        timing.previous_context = gemmini_trace_bind(gemmini_trace_fork(timing.origin));
        timing.start = gemmini_cpu_timing_read();
        timing.started = true;
        timing.finished = false;
    } else {
        timing.end = gemmini_cpu_timing_read();
        timing.finished = true;
        gemmini_trace_restore(timing.previous_context);
    }
}

void WorkerCpuTiming::emit(const char *layer, const char *scope, std::optional<uint64_t> run_id,
                           bool success) const noexcept {
    if (!started) return;
    trace::ScopedContext origin_scope(start.trace);
    gemmini_cycle_record_v2 identity{};
    identity.interval.layer = layer;
    identity.interval.op = scope;
    if (run_id) {
        identity.identity_mask = GEMMINI_CYCLE_HAS_RUN_ID;
        identity.run_id = *run_id;
    }
    gemmini_cpu_timing_record_envelope(&identity, &start, &end);
    gemmini_cpu_totals totals{};
    gemmini_cpu_timing_add(&totals, &start, &end);
    gemmini_cpu_timing_emit(layer, scope, run_id ? &*run_id : nullptr,
                           success && finished, &start, &end, &totals);
}
}

namespace {
std::string serialize_cpu_record(const gemmini_cycle_record_v2 &record,
        const gemmini_cpu_sample &start_sample, const gemmini_cpu_sample &end_sample,
        bool raw_segment) {
    using namespace ggml::gemmini;
    const auto *identity = &record;
    const auto *start = &start_sample;
    const auto *end = &end_sample;
#if CYCLE_DETAIL
    std::string json = "{\"schema\":\"gemmini.cycle\",\"version\":2,"
        "\"record_type\":\"CPU_INTERVAL\",\"unit\":\"cycle\",\"source\":";
    json += start->native_source == GEMMINI_CPU_COUNTER_THREAD_PERF &&
            end->native_source == GEMMINI_CPU_COUNTER_THREAD_PERF ?
        "\"linux_perf_cpu_cycles\"" : "null";
    json += ",\"layer\":" + cpu_json_string(identity->interval.layer) +
        ",\"op\":" + cpu_json_string(identity->interval.op);
    const auto add_identity = [&](const char *key, uint32_t field, uint64_t value) {
        json += std::string(",\"") + key + "\":" +
            ((identity->identity_mask & field) ? std::to_string(value) : "null");
    };
    add_identity("run_id", GEMMINI_CYCLE_HAS_RUN_ID, identity->run_id);
    add_identity("stripe_id", GEMMINI_CYCLE_HAS_STRIPE_ID, identity->stripe_id);
    add_identity("slot", GEMMINI_CYCLE_HAS_SLOT, identity->slot);
    add_identity("node_id", GEMMINI_CYCLE_HAS_NODE_ID, identity->node_id);
    add_identity("worker_id", GEMMINI_CYCLE_HAS_WORKER_ID, identity->worker_id);
    json += ",\"native_cycles\":" + cycle::serialize_cpu_native(*start, *end) +
        ",\"host_timing\":" + cycle::serialize_host_timing(start->ns, end->ns, start->tid, end->tid) +
        ",\"thread_cpu_timing\":" + cycle::serialize_thread_cpu_timing(
            {start->ns, start->tid, start->thread_cpu_ns, start->thread_cpu_valid != 0},
            {end->ns, end->tid, end->thread_cpu_ns, end->thread_cpu_valid != 0}) +
        ",\"additive\":false}";
    if (raw_segment) {
        const auto marker = json.find("\"record_type\":\"CPU_INTERVAL\"");
        if (marker != std::string::npos)
            json.replace(marker, std::strlen("\"record_type\":\"CPU_INTERVAL\""),
                         "\"record_type\":\"OPERATOR_SEGMENT\"");
    }
    return json;
#else
    const auto interval = evaluate_cpu_interval(start, end);
    const bool valid = interval.cycles_valid_count == 1 && !interval.cycles_reason;
    std::string json = "{\"op\":" + cpu_json_string(identity->interval.op) +
        ",\"kind\":\"" + (raw_segment ? std::string("segment") : std::string("cpu")) + "\"";
    if (identity->interval.layer && *identity->interval.layer)
        json += ",\"layer\":" + cpu_json_string(identity->interval.layer);
    const auto add_identity = [&](const char *key, uint32_t field, uint64_t value) {
        if (identity->identity_mask & field)
            json += std::string(",\"") + key + "\":" + std::to_string(value);
    };
    add_identity("run_id", GEMMINI_CYCLE_HAS_RUN_ID, identity->run_id);
    add_identity("stripe_id", GEMMINI_CYCLE_HAS_STRIPE_ID, identity->stripe_id);
    add_identity("slot", GEMMINI_CYCLE_HAS_SLOT, identity->slot);
    add_identity("node_id", GEMMINI_CYCLE_HAS_NODE_ID, identity->node_id);
    add_identity("worker_id", GEMMINI_CYCLE_HAS_WORKER_ID, identity->worker_id);
    json += ",\"start\":" + std::to_string(start->counter) +
        ",\"end\":" + std::to_string(end->counter) +
        ",\"delta\":" + (valid ? std::to_string(interval.cycles) : std::string("null")) +
        ",\"ns_start\":" + std::to_string(start->ns) +
        ",\"ns_end\":" + std::to_string(end->ns);
    if (start->tid != 0 && start->tid == end->tid) {
        json += ",\"tid\":" + std::to_string(start->tid);
    } else {
        json += ",\"tid_start\":" + std::to_string(start->tid) +
            ",\"tid_end\":" + std::to_string(end->tid);
    }
    json += ",\"valid\":";
    json += valid ? "true" : "false";
    if (interval.cycles_reason)
        json += ",\"reason\":" + cpu_json_string(interval.cycles_reason);
    json += '}';
    return json;
#endif
}
}

extern "C" void gemmini_cpu_timing_record(const gemmini_cycle_record_v2 *identity,
        const gemmini_cpu_sample *start, const gemmini_cpu_sample *end) noexcept {
#if LOG_CYCLE
    using namespace ggml::gemmini;
    if (!identity || !start || !end) return;
    try {
        log::cycle.write_cpu(*identity, *start, *end);
    } catch (...) {
        log::cycle.report_failure("CPU interval");
    }
#else
    (void) identity; (void) start; (void) end;
#endif
}

extern "C" void gemmini_cpu_timing_record_segment(const gemmini_cycle_record_v2 *identity,
        const gemmini_cpu_sample *start, const gemmini_cpu_sample *end) noexcept {
#if LOG_CYCLE
    if (!identity || !start || !end) return;
    try { ggml::gemmini::log::cycle.write_cpu(*identity, *start, *end, {}, true); }
    catch (...) { ggml::gemmini::log::cycle.report_failure("operator segment"); }
#else
    (void)identity; (void)start; (void)end;
#endif
}

extern "C" void gemmini_cpu_timing_record_envelope(const gemmini_cycle_record_v2 *identity,
        const gemmini_cpu_sample *start, const gemmini_cpu_sample *end) noexcept {
#if LOG_CYCLE
    if (!identity || !start || !end) return;
    try { ggml::gemmini::log::cycle.write_cpu(*identity, *start, *end, {}, true, true); }
    catch (...) { ggml::gemmini::log::cycle.report_failure("structural envelope"); }
#else
    (void)identity; (void)start; (void)end;
#endif
}

extern "C" void gemmini_cpu_timing_emit(const char *layer, const char *scope, const uint64_t *run_id,
        int operation_success, const gemmini_cpu_sample *start, const gemmini_cpu_sample *end,
        const gemmini_cpu_totals *totals) {
#if LOG_CYCLE && CYCLE_DETAIL
    using namespace ggml::gemmini;
    try {
        if (scope && std::string_view(scope) == "cpu.graph_workers") {
            performance::record_cpu_wall(start->ns, end->ns);
        }
        std::string json = "{\"schema\":\"gemmini.cycle\",\"version\":2,"
            "\"record_type\":\"CPU_WORK_SUMMARY\",\"source\":\"steady_clock\","
            "\"unit\":\"nanosecond\",\"op\":\"";
        log::append_json_escaped(json, scope);
        json += "\",\"layer\":";
        if (layer && *layer) {
            json += '"';
            log::append_json_escaped(json, layer);
            json += '"';
        } else {
            json += "null";
        }
        json += ",\"run_id\":" + (run_id ? std::to_string(*run_id) : "null") +
            ",\"host_timing\":" + cycle::serialize_host_timing(start->ns, end->ns, start->tid, end->tid) +
            ",\"cpu_workers\":" + cycle::serialize_cpu_totals(*totals) +
            ",\"operation_success\":" + (operation_success ? "true" : "false") +
            ",\"additive\":false}";
        log::cycle.write_json(json);
        if (scope && std::string_view(scope) == "cpu.graph_workers") log::cycle.flush();
    } catch (...) {
        log::cycle.report_failure("CPU worker summary");
    }
#else
    (void) layer; (void) scope; (void) run_id; (void) operation_success;
    (void) start; (void) end; (void) totals;
#endif
}
