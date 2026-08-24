#include "../include/gemmini/log.hpp"

#include <limits>
#include <mutex>
#include <string>

namespace ggml::gemmini::log
{
    CycleLog cycle;

    namespace
    {
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

    std::string serialize_cycle_record(const CycleRecord &record)
    {
        const uint64_t cycles = record.end >= record.start ? record.end - record.start : 0;
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
        add_u64("version", 1);
        add_string("record_type", "CYCLE_INTERVAL");
#ifdef __riscv
        add_string("source", record.source ? record.source : "riscv_cycle");
        add_string("unit", record.unit ? record.unit : "cycle");
#else
        add_string("source", record.source ? record.source : "host_tick");
        add_string("unit", record.unit ? record.unit : "tick");
#endif
        add_string("layer", record.layer);
        add_string("name", record.op);
        add_u64("start", record.start);
        add_u64("end", record.end);
        add_u64("delta", cycles);
        add_key("valid");
        json += record.end >= record.start ? "true" : "false";
#if LOG_DETAIL
        add_string("file", record.file);
        if (record.file) add_i32("line", record.line);
        add_string("func", record.func);
#endif
        json += "}\n";
        return json;
    }

    std::string serialize_ws_cycle_record(const WsCycleRecord &record)
    {
#if !LOG_CYCLE
        (void) record;
        return {};
#else
        std::string json =
            "{\"schema\":\"gemmini.cycle\",\"version\":1,"
            "\"record_type\":\"WS_LOOP_TELEMETRY\","
            "\"source\":\"gemmini_hw_counter\",\"unit\":\"cycle\"";
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
        std::lock_guard<std::mutex> lock(detail::output_mutex());
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
        const std::size_t written = write_fault ? 0 : std::fwrite(json.data(), 1, json.size(), output);
        const bool flush_fault = detail::consume_fault(testing::LogFault::flush);
        const int flushed = flush_fault ? EOF : std::fflush(output);
        if (owns_call_output)
        {
            std::fclose(output);
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
