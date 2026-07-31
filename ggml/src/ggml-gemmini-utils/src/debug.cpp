#include "../include/gemmini/log.hpp"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <system_error>

namespace ggml::gemmini::log
{
    namespace
    {
        bool path_starts_with_log_dir(const char *path)
        {
            if (!path || *path == '\0')
            {
                return false;
            }
            if (path[0] != 'l' || path[1] != 'o' || path[2] != 'g')
            {
                return false;
            }
            const char c = path[3];
            return c == '/' || c == '\\' || c == '\0';
        }

        std::filesystem::path resolve_output_path(const char *path)
        {
            if (!path || *path == '\0')
            {
                return {};
            }

            std::filesystem::path p(path);
            if (p.is_absolute())
            {
                return p;
            }

            // Highest priority: force all relative paths under GEMMINI_LOG_DIR.
            // For callers that use `log/<name>`, strip the leading `log/` so
            // `log/out.jsonl` becomes `$GEMMINI_LOG_DIR/out.jsonl`.
            if (const char *env_dir = std::getenv("GEMMINI_LOG_DIR"))
            {
                if (*env_dir)
                {
                    std::filesystem::path base(env_dir);
                    std::filesystem::path rel = p;
                    if (path_starts_with_log_dir(path))
                    {
                        rel = p.lexically_relative("log");
                        if (rel.empty() || rel == ".")
                        {
                            rel.clear();
                        }
                    }
                    return rel.empty() ? base : (base / rel);
                }
            }

            if (path_starts_with_log_dir(path))
            {
                std::filesystem::path rel = p.lexically_relative("log");
                if (rel.empty() || rel == ".")
                {
                    rel.clear();
                }

                std::error_code ec;
                std::filesystem::path cwd = std::filesystem::current_path(ec);
                if (ec || cwd.empty())
                {
                    cwd = ".";
                }
                std::filesystem::path base = cwd / "log";
                return rel.empty() ? base : (base / rel);
            }

            return p;
        }

        void ensure_parent_dir_exists(const std::filesystem::path &path)
        {
            std::error_code ec;
            std::filesystem::path parent = path.parent_path();
            if (parent.empty())
            {
                return;
            }
            std::filesystem::create_directories(parent, ec);
        }

        FILE *open_output_file(const std::filesystem::path &path)
        {
            const std::string key = path.string();
            return std::fopen(key.c_str(), "a");
        }

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

        std::string vformat_printf(const char *fmt, va_list ap)
        {
            const char *safe_fmt = fmt ? fmt : "";
            va_list ap_copy;
            va_copy(ap_copy, ap);
            const int needed = std::vsnprintf(nullptr, 0, safe_fmt, ap_copy);
            va_end(ap_copy);
            if (needed <= 0)
            {
                return std::string();
            }

            std::string buf;
            buf.resize(static_cast<std::size_t>(needed) + 1);
            va_copy(ap_copy, ap);
            std::vsnprintf(buf.data(), buf.size(), safe_fmt, ap_copy);
            va_end(ap_copy);
            buf.resize(static_cast<std::size_t>(needed));
            return buf;
        }

        void trim_trailing_newlines(std::string &s)
        {
            while (!s.empty())
            {
                const char c = s.back();
                if (c == '\n' || c == '\r')
                {
                    s.pop_back();
                    continue;
                }
                break;
            }
        }

        void write_jsonl_debug(FILE *out, const char *file, int line, const char *func,
                               const char *layer, const std::string &msg)
        {
            if (!out)
            {
                return;
            }

            std::string json;
            json.reserve(128 + msg.size());

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
            auto add_u64 = [&](const char *k, unsigned long long v)
            {
                add_key(k);
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%llu", v);
                json += buf;
            };
            auto add_i32 = [&](const char *k, int v)
            {
                add_key(k);
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%d", v);
                json += buf;
            };

            json.push_back('{');
            add_str("layer", layer);

            add_key("msg");
            json.push_back('"');
            append_json_escaped(json, msg.c_str());
            json.push_back('"');

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
            std::fwrite(json.data(), 1, json.size(), out);
            std::fputc('\n', out);
            std::fflush(out);
        }

        void write_jsonl_ws_loop(FILE *out, uint64_t wall, uint64_t load, uint64_t exe, uint64_t store, uint64_t loop,
                                 uint64_t dim_I, uint64_t dim_J, uint64_t dim_K,
                                 uint64_t tile_I, uint64_t tile_J, uint64_t tile_K,
                                 uint64_t I0, uint64_t J0, uint64_t K0,
                                 uint64_t a_reuse, uint64_t b_reuse)
        {
            if (!out)
            {
                return;
            }

            std::string json;
            json.reserve(256);
            bool first = true;
            auto add_key = [&](const char *key)
            {
                if (!first)
                {
                    json.push_back(',');
                }
                first = false;
                json.push_back('"');
                json += key;
                json += "\":";
            };
            auto add_u64 = [&](const char *key, uint64_t value)
            {
                add_key(key);
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%llu", static_cast<unsigned long long>(value));
                json += buf;
            };

            json.push_back('{');
            add_key("event");
            json += "\"ws_loop\"";
            add_u64("wall", wall);
            add_u64("load", load);
            add_u64("exe", exe);
            add_u64("store", store);
            add_u64("loop", loop);
            add_u64("dim_I", dim_I);
            add_u64("dim_J", dim_J);
            add_u64("dim_K", dim_K);
            add_u64("tile_I", tile_I);
            add_u64("tile_J", tile_J);
            add_u64("tile_K", tile_K);
            add_u64("I0", I0);
            add_u64("J0", J0);
            add_u64("K0", K0);
            add_u64("a_reuse", a_reuse);
            add_u64("b_reuse", b_reuse);
            add_u64("valid", 1);
            json += "}\n";
            static std::mutex write_mutex;
            std::lock_guard<std::mutex> lock(write_mutex);
            std::fwrite(json.data(), 1, json.size(), out);
            std::fflush(out);
        }
    } // namespace

    LogTarget file(const char *path)
    {
        return LogTarget{path};
    }

    bool truncate_file(const char *path)
    {
        if (path == nullptr || *path == '\0')
        {
            return false;
        }

        std::filesystem::path resolved = resolve_output_path(path);
        if (resolved.empty())
        {
            return false;
        }
        ensure_parent_dir_exists(resolved);

        FILE *file = std::fopen(resolved.string().c_str(), "w");
        if (!file)
        {
            return false;
        }
        std::fclose(file);
        return true;
    }

    Log::Log(FILE *out) : out_(out), owns_(false) {}

    Log::~Log()
    {
        close_owned();
    }

    void Log::set_output(FILE *out)
    {
        has_explicit_output_ = true;

        if (out == out_ && !owns_)
        {
            return;
        }
        close_owned();
        out_ = out;
        owns_ = false;
    }

    bool Log::set_output_path(const char *path)
    {
        has_explicit_output_ = true;

        if (path == nullptr || *path == '\0')
        {
            set_output(stderr);
            return true;
        }

        std::filesystem::path resolved = resolve_output_path(path);
        if (resolved.empty())
        {
            return false;
        }
        ensure_parent_dir_exists(resolved);

        FILE *file = open_output_file(resolved);
        if (!file)
        {
            return false;
        }
        close_owned();
        out_ = file;
        owns_ = true;
        return true;
    }

    FILE *Log::select_output(const char *path, bool *owns) const
    {
        if (path && *path)
        {
            std::filesystem::path resolved = resolve_output_path(path);
            if (!resolved.empty())
            {
                ensure_parent_dir_exists(resolved);
                FILE *file = open_output_file(resolved);
                if (file)
                {
                    if (owns)
                    {
                        *owns = true;
                    }
                    return file;
                }
            }
        }
        if (owns)
        {
            *owns = false;
        }
        return out_;
    }

    void Log::close_owned()
    {
        if (owns_ && out_)
        {
            std::fclose(out_);
            out_ = nullptr;
            owns_ = false;
        }
    }

    DebugLog debug;

    DebugLog::DebugLog(FILE *out, bool add_newline)
        : Log(out), add_newline_(add_newline) {}

    bool DebugLog::set_output_path(const char *path, bool truncate)
    {
#if LOG_DEBUG
        if (truncate && path && *path && !truncate_file(path)) {
            return false;
        }
        return Log::set_output_path(path);
#else
        (void)path;
        (void)truncate;
        return true;
#endif
    }

    void DebugLog::vwrite(FILE *out, const char *file, int line, const char *func, const char *fmt, va_list ap)
    {
#if LOG_DEBUG
        if (!out)
        {
            return;
        }
        std::string msg = vformat_printf(fmt, ap);
        trim_trailing_newlines(msg);
        write_jsonl_debug(out, file, line, func, nullptr, msg);
#else
        (void)file;
        (void)line;
        (void)func;
        (void)fmt;
        (void)ap;
#endif
    }

    void DebugLog::vwrite_layer_fmt(FILE *out, const char *file, int line, const char *func, const char *layer, const char *fmt, va_list ap)
    {
#if LOG_DEBUG
        if (!out)
        {
            return;
        }
        std::string msg = vformat_printf(fmt, ap);
        trim_trailing_newlines(msg);
        write_jsonl_debug(out, file, line, func, layer, msg);
#else
        (void)file;
        (void)line;
        (void)func;
        (void)layer;
        (void)fmt;
        (void)ap;
#endif
    }

    void DebugLog::operator()(const char *fmt, ...)
    {
        va_list ap;
        va_start(ap, fmt);
        vwrite(out_, nullptr, 0, nullptr, fmt, ap);
        va_end(ap);
    }

    void DebugLog::operator()(const char *file, int line, const char *func, const char *fmt, ...)
    {
        va_list ap;
        va_start(ap, fmt);
        vwrite(out_, file, line, func, fmt, ap);
        va_end(ap);
    }

    void DebugLog::operator()(LogTarget target, const char *fmt, ...)
    {
#if LOG_DEBUG
        bool owns = false;
        FILE *out = select_output(target.path, &owns);
        va_list ap;
        va_start(ap, fmt);
        vwrite(out, nullptr, 0, nullptr, fmt, ap);
        va_end(ap);
        if (owns && out)
        {
            std::fclose(out);
        }
#else
        (void)target;
        (void)fmt;
#endif
    }

    void DebugLog::operator()(LogTarget target, const char *file, int line, const char *func, const char *fmt, ...)
    {
#if LOG_DEBUG
        bool owns = false;
        FILE *out = select_output(target.path, &owns);
        va_list ap;
        va_start(ap, fmt);
        vwrite(out, file, line, func, fmt, ap);
        va_end(ap);
        if (owns && out)
        {
            std::fclose(out);
        }
#else
        (void)target;
        (void)file;
        (void)line;
        (void)func;
        (void)fmt;
#endif
    }

    void DebugLog::operator()(const char *layer, const char *fmt, ...)
    {
        va_list ap;
        va_start(ap, fmt);
        vwrite_layer_fmt(out_, nullptr, 0, nullptr, layer, fmt, ap);
        va_end(ap);
    }

    void DebugLog::operator()(LogTarget target, const char *layer, const char *fmt, ...)
    {
#if LOG_DEBUG

        bool owns = false;
        FILE *out = select_output(target.path, &owns);
        va_list ap;
        va_start(ap, fmt);
        vwrite_layer_fmt(out, nullptr, 0, nullptr, layer, fmt, ap);
        va_end(ap);
        if (owns && out)
        {
            std::fclose(out);
        }
#else
        (void)target;
        (void)layer;
        (void)fmt;
#endif
    }

    void DebugLog::v(const char *fmt, va_list ap)
    {
        vwrite(out_, nullptr, 0, nullptr, fmt, ap);
    }

    void DebugLog::v_layer(const char *layer, const char *fmt, va_list ap)
    {
        vwrite_layer_fmt(out_, nullptr, 0, nullptr, layer, fmt, ap);
    }

    void DebugLog::v_loc(const char *file, int line, const char *func, const char *fmt, va_list ap)
    {
        vwrite(out_, file, line, func, fmt, ap);
    }

    void DebugLog::v_target(LogTarget target, const char *fmt, va_list ap)
    {
#if LOG_DEBUG
        bool owns = false;
        FILE *out = select_output(target.path, &owns);
        vwrite(out, nullptr, 0, nullptr, fmt, ap);
        if (owns && out)
        {
            std::fclose(out);
        }
#else
        (void)target;
        (void)fmt;
        (void)ap;
#endif
    }

    void DebugLog::v_target_layer(LogTarget target, const char *layer, const char *fmt, va_list ap)
    {
#if LOG_DEBUG
        bool owns = false;
        FILE *out = select_output(target.path, &owns);
        vwrite_layer_fmt(out, nullptr, 0, nullptr, layer, fmt, ap);
        if (owns && out)
        {
            std::fclose(out);
        }
#else
        (void)target;
        (void)layer;
        (void)fmt;
        (void)ap;
#endif
    }

    void DebugLog::ws_loop(LogTarget target, uint64_t wall, uint64_t load, uint64_t exe, uint64_t store, uint64_t loop,
                           uint64_t dim_I, uint64_t dim_J, uint64_t dim_K,
                           uint64_t tile_I, uint64_t tile_J, uint64_t tile_K,
                           uint64_t I0, uint64_t J0, uint64_t K0,
                           uint64_t a_reuse, uint64_t b_reuse)
    {
        bool owns = false;
        FILE *out = select_output(target.path, &owns);
        write_jsonl_ws_loop(out, wall, load, exe, store, loop,
                            dim_I, dim_J, dim_K,
                            tile_I, tile_J, tile_K,
                            I0, J0, K0, a_reuse, b_reuse);
        if (owns && out)
        {
            std::fclose(out);
        }
    }

    void DebugLog::v_target_loc(LogTarget target, const char *file, int line, const char *func,
                                const char *fmt, va_list ap)
    {
#if LOG_DEBUG
        bool owns = false;
        FILE *out = select_output(target.path, &owns);
        vwrite(out, file, line, func, fmt, ap);
        if (owns && out)
        {
            std::fclose(out);
        }
#else
        (void)target;
        (void)file;
        (void)line;
        (void)func;
        (void)fmt;
        (void)ap;
#endif
    }

} // namespace ggml::gemmini::log
