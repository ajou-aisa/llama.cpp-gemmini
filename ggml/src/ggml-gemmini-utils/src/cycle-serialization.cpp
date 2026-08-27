#include "../include/gemmini/log.hpp"
#include "cycle_reader_internal.h"

#include <cstring>
#include <limits>
#include <string>

namespace ggml::gemmini::log
{
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

    static std::string serialize_cycle_record_impl(const CycleRecord &record,
                                                   bool provenance_available,
                                                   bool checked_valid,
                                                   const char * checked_reason)
    {
#if defined(__riscv)
        const char * const default_source = "riscv_cycle";
        const char * const default_unit = "cycle";
#elif defined(__linux__) && defined(__aarch64__)
        const char * const default_source = "linux_perf_cpu_cycles";
        const char * const default_unit = "cycle";
#else
        const char * const default_source = "host_tick";
        const char * const default_unit = "tick";
#endif
        const char * const source = record.source ? record.source : default_source;
        const char * const unit = record.unit ? record.unit : default_unit;
        const bool scalar_jetson = !provenance_available &&
            std::strcmp(source, "linux_perf_cpu_cycles") == 0;
        const bool valid = provenance_available ? checked_valid :
            (!scalar_jetson && record.end >= record.start);
        const char * const reason = scalar_jetson ? "scalar_provenance_unavailable" : checked_reason;
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
        if (valid) add_u64("delta", cycles); else add_null("delta");
        add_key("valid");
        json += valid ? "true" : "false";
        if (!valid) add_string("reason", reason ? reason : "counter_regression");
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
        return serialize_cycle_record_impl(record, false, false, nullptr);
    }

    std::string serialize_checked_cycle_record(const CycleRecord & record, bool valid,
                                               const char * reason)
    {
        return serialize_cycle_record_impl(record, true, valid, reason);
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

} // namespace ggml::gemmini::log
