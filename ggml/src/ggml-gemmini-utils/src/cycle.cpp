#include "../include/gemmini/log.hpp"

#include <cinttypes>
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
                case '\\':
                    out += "\\\\";
                    break;
                case '"':
                    out += "\\\"";
                    break;
                case '\b':
                    out += "\\b";
                    break;
                case '\f':
                    out += "\\f";
                    break;
                case '\n':
                    out += "\\n";
                    break;
                case '\r':
                    out += "\\r";
                    break;
                case '\t':
                    out += "\\t";
                    break;
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

        std::mutex &write_mutex()
        {
            static std::mutex mutex;
            return mutex;
        }
    } // namespace

    void CycleLog::write(FILE *out, const char *file, int line, const char *func, const char *layer, const char *op,
                         uint64_t start, uint64_t end)
    {
#if LOG_CYCLE
        if (!out)
        {
            return;
        }

        const uint64_t cycles = (end >= start) ? (end - start) : 0;
        const char *safe_layer = layer ? layer : "";
        const char *safe_op = op ? op : "";

        std::string json;
        json.reserve(160);

        bool first = true;
        auto add_key = [&](const char *k)
        {
            if (!first)
            {
                json.push_back(',');
            }
            first = false;
            json.push_back('"');
            json += k;
            json += "\":";
        };
        auto add_str = [&](const char *k, const char *v)
        {
            if (!v || *v == '\0')
            {
                return;
            }
            add_key(k);
            json.push_back('"');
            append_json_escaped(json, v);
            json.push_back('"');
        };
        auto add_u64 = [&](const char *k, uint64_t v)
        {
            add_key(k);
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%" PRIu64, v);
            json += buf;
        };
#if LOG_DETAIL
        auto add_i32 = [&](const char *k, int v)
        {
            add_key(k);
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%d", v);
            json += buf;
        };
#endif

        json.push_back('{');
        add_str("layer", safe_layer);
        add_str("op", safe_op);
        add_u64("start", start);
        add_u64("end", end);
        add_u64("cycles", cycles);

#if LOG_DETAIL
        add_str("file", file);
        if (file)
        {
            add_i32("line", line);
        }
        add_str("func", func);
#else
        (void)file;
        (void)line;
        (void)func;
#endif

        json.push_back('}');
        json.push_back('\n');

        std::lock_guard<std::mutex> lock(write_mutex());
        std::fwrite(json.data(), 1, json.size(), out);
        std::fflush(out);
#else
        (void)out;
        (void)file;
        (void)line;
        (void)func;
        (void)layer;
        (void)op;
        (void)start;
        (void)end;
#endif
    }

    void CycleLog::operator()(const char *layer, const char *op, uint64_t start, uint64_t end)
    {
        write(out_, nullptr, 0, nullptr, layer, op, start, end);
    }

    void CycleLog::operator()(const char *file, int line, const char *func, const char *layer, const char *op,
                              uint64_t start, uint64_t end)
    {
        write(out_, file, line, func, layer, op, start, end);
    }

    void CycleLog::operator()(LogTarget target, const char *layer, const char *op, uint64_t start, uint64_t end)
    {
#if LOG_CYCLE
        bool owns = false;
        FILE *out = select_output(target.path, &owns);
        write(out, nullptr, 0, nullptr, layer, op, start, end);
        if (owns && out)
        {
            std::fclose(out);
        }
#else
        (void)target;
        (void)layer;
        (void)op;
        (void)start;
        (void)end;
#endif
    }

    void CycleLog::operator()(LogTarget target, const char *file, int line, const char *func, const char *layer, const char *op,
                              uint64_t start, uint64_t end)
    {
#if LOG_CYCLE
        bool owns = false;
        FILE *out = select_output(target.path, &owns);
        write(out, file, line, func, layer, op, start, end);
        if (owns && out)
        {
            std::fclose(out);
        }
#else
        (void)target;
        (void)file;
        (void)line;
        (void)func;
        (void)layer;
        (void)op;
        (void)start;
        (void)end;
#endif
    }

    void CycleLog::cycle(const char *layer, const char *op, uint64_t start, uint64_t end)
    {
        (*this)(layer, op, start, end);
    }

} // namespace ggml::gemmini::log
