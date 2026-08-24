#include "../include/gemmini/log.hpp"

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <new>
#include <string>
#include <system_error>

namespace ggml::gemmini::log
{
    namespace detail
    {
        std::mutex &output_mutex()
        {
            if (consume_fault(testing::LogFault::mutex))
            {
                throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again));
            }
            static std::mutex *mutex = new std::mutex;
            return *mutex;
        }

        static std::atomic<testing::LogFault> injected_fault{testing::LogFault::none};
        static std::atomic<testing::TargetLockHook> target_lock_hook{nullptr};
        static std::atomic<void *> target_lock_hook_user_data{nullptr};

        bool consume_fault(testing::LogFault expected)
        {
            testing::LogFault value = expected;
            return injected_fault.compare_exchange_strong(value, testing::LogFault::none);
        }

        void invoke_target_lock_hook(testing::TargetWriteKind kind)
        {
            const testing::TargetLockHook hook = target_lock_hook.load(std::memory_order_acquire);
            if (hook) hook(kind, target_lock_hook_user_data.load(std::memory_order_relaxed));
        }
    } // namespace detail

    namespace testing
    {
        void set_target_lock_hook(TargetLockHook hook, void *user_data)
        {
            detail::target_lock_hook_user_data.store(user_data, std::memory_order_relaxed);
            detail::target_lock_hook.store(hook, std::memory_order_release);
        }

        void clear_target_lock_hook()
        {
            detail::target_lock_hook.store(nullptr, std::memory_order_release);
            detail::target_lock_hook_user_data.store(nullptr, std::memory_order_relaxed);
        }

        void set_log_fault(LogFault fault)
        {
            detail::injected_fault.store(fault);
        }

        void clear_log_fault()
        {
            detail::injected_fault.store(LogFault::none);
        }
    } // namespace testing

    std::filesystem::path resolve_output_path(const char *path)
    {
        if (detail::consume_fault(testing::LogFault::filesystem))
        {
            throw std::filesystem::filesystem_error("injected logger filesystem failure", std::error_code{});
        }
        if (!path || *path == '\0')
        {
            return {};
        }

        const std::filesystem::path requested(path);
        if (requested.is_absolute())
        {
            return requested;
        }

        std::filesystem::path relative;
        bool first = true;
        for (const std::filesystem::path &component : requested)
        {
            if (component == "..")
            {
                return {};
            }
            if (component.empty() || component == ".")
            {
                continue;
            }
            if (first && component == "log")
            {
                first = false;
                continue;
            }
            first = false;
            relative /= component;
        }
        if (relative.empty())
        {
            return {};
        }

        std::error_code ec;
        std::filesystem::path base;
        if (const char *env_dir = std::getenv("GEMMINI_LOG_DIR"); env_dir && *env_dir)
        {
            base = std::filesystem::absolute(env_dir, ec);
        }
        else
        {
            base = std::filesystem::current_path(ec);
            if (!ec) base /= "output/log";
        }
        if (ec || base.empty())
        {
            return {};
        }

        const std::filesystem::path confined_base = std::filesystem::weakly_canonical(base, ec);
        if (ec) return {};
        const std::filesystem::path candidate = std::filesystem::weakly_canonical(confined_base / relative, ec);
        if (ec) return {};
        auto base_it = confined_base.begin();
        auto candidate_it = candidate.begin();
        for (; base_it != confined_base.end(); ++base_it, ++candidate_it)
        {
            if (candidate_it == candidate.end() || *candidate_it != *base_it)
            {
                return {};
            }
        }
        return candidate;
    }

    bool prepare_output_parent(const std::filesystem::path &path)
    {
        const std::filesystem::path parent = path.parent_path();
        if (parent.empty())
        {
            return true;
        }
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        return !ec;
    }

    namespace
    {
        class ScopedFile
        {
        public:
            explicit ScopedFile(FILE *file) : file_(file) {}
            ~ScopedFile() { if (file_) std::fclose(file_); }
            ScopedFile(const ScopedFile &) = delete;
            ScopedFile &operator=(const ScopedFile &) = delete;

        private:
            FILE *file_;
        };

        FILE *open_output_file(const std::filesystem::path &path, const char *mode = "a")
        {
            if (detail::consume_fault(testing::LogFault::open))
            {
                return nullptr;
            }
            const std::string key = path.string();
            return std::fopen(key.c_str(), mode);
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
            if (detail::consume_fault(testing::LogFault::format) ||
                detail::consume_fault(testing::LogFault::allocation))
            {
                throw std::bad_alloc();
            }
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

    } // namespace

    LogTarget file(const char *path)
    {
        return LogTarget{path};
    }

    bool truncate_file(const char *path)
    {
        std::lock_guard<std::mutex> lock(detail::output_mutex());
        const std::filesystem::path resolved = resolve_output_path(path);
        if (resolved.empty() || !prepare_output_parent(resolved))
        {
            return false;
        }
        FILE *file = open_output_file(resolved, "w");
        if (!file)
        {
            return false;
        }
        return std::fclose(file) == 0;
    }

    Log::Log(FILE *out) : out_(out), owns_(false) {}

    Log::~Log()
    {
        std::lock_guard<std::mutex> lock(detail::output_mutex());
        close_owned_unlocked();
    }

    void Log::set_output_unlocked(FILE *out)
    {
        has_explicit_output_ = true;
        if (out == out_ && !owns_)
        {
            return;
        }
        close_owned_unlocked();
        out_ = out;
        owns_ = false;
    }

    void Log::set_output(FILE *out)
    {
        std::lock_guard<std::mutex> lock(detail::output_mutex());
        set_output_unlocked(out);
    }

    bool Log::set_output_path_unlocked(const char *path, bool truncate, const char **failure)
    {
        if (failure) *failure = nullptr;
        has_explicit_output_ = true;
        if (path == nullptr || *path == '\0')
        {
            set_output_unlocked(stderr);
            return true;
        }

        const std::filesystem::path resolved = resolve_output_path(path);
        if (resolved.empty() || !prepare_output_parent(resolved))
        {
            if (failure) *failure = "setup";
            return false;
        }
        std::error_code exists_error;
        const bool existed = std::filesystem::exists(resolved, exists_error);
        if (truncate)
        {
            FILE *truncated = open_output_file(resolved, "w");
            if (!truncated)
            {
                if (failure) *failure = "open";
                return false;
            }
            if (std::fclose(truncated) != 0)
            {
                if (failure) *failure = "setup";
                return false;
            }
        }
        FILE *file = open_output_file(resolved, "a");
        if (!file)
        {
            if (!existed && !exists_error)
            {
                std::error_code remove_error;
                std::filesystem::remove(resolved, remove_error);
            }
            if (failure) *failure = "open";
            return false;
        }
        if (detail::consume_fault(testing::LogFault::replacement))
        {
            std::fclose(file);
            if (!existed && !exists_error)
            {
                std::error_code remove_error;
                std::filesystem::remove(resolved, remove_error);
            }
            if (failure) *failure = "replacement";
            return false;
        }
        close_owned_unlocked();
        out_ = file;
        owns_ = true;
        return true;
    }

    bool Log::set_output_path(const char *path)
    {
        std::lock_guard<std::mutex> lock(detail::output_mutex());
        return set_output_path_unlocked(path, false);
    }

    bool Log::has_explicit_output() const
    {
        std::lock_guard<std::mutex> lock(detail::output_mutex());
        return has_explicit_output_;
    }

    FILE *Log::select_output_unlocked(const char *path, bool *owns) const
    {
        if (path && *path)
        {
            const std::filesystem::path resolved = resolve_output_path(path);
            if (!resolved.empty() && prepare_output_parent(resolved))
            {
                FILE *file = open_output_file(resolved);
                if (file)
                {
                    if (owns) *owns = true;
                    return file;
                }
            }
            if (owns) *owns = false;
            return nullptr;
        }
        if (owns) *owns = false;
        return out_;
    }

    void Log::close_owned_unlocked()
    {
        if (owns_ && out_)
        {
            std::fclose(out_);
        }
        out_ = nullptr;
        owns_ = false;
    }

    void Log::disable_output_unlocked()
    {
        close_owned_unlocked();
    }

    DebugLog debug;

    DefaultOutputSetupResult setup_default_outputs()
    {
        static std::once_flag once;
        static DefaultOutputSetupResult result{true, true};
        std::call_once(once, [] {
            if (!cycle.has_explicit_output())
            {
                result.cycle = cycle.set_output_path(GEMMINI_LOG_DEFAULT_CYCLE_PATH, true);
            }
            if (!debug.has_explicit_output())
            {
                result.debug = debug.set_output_path(GEMMINI_LOG_DEFAULT_DEBUG_PATH, true);
            }
        });
        return result;
    }

    DebugLog::DebugLog(FILE *out) : Log(out) {}

    bool DebugLog::set_output_path(const char *path, bool truncate)
    {
#if LOG_DEBUG
        std::lock_guard<std::mutex> lock(detail::output_mutex());
        return set_output_path_unlocked(path, truncate);
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
        (void)out;
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
        (void)out;
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
        v(fmt, ap);
        va_end(ap);
    }

    void DebugLog::operator()(const char *file, int line, const char *func, const char *fmt, ...)
    {
        va_list ap;
        va_start(ap, fmt);
        v_loc(file, line, func, fmt, ap);
        va_end(ap);
    }

    void DebugLog::operator()(LogTarget target, const char *fmt, ...)
    {
#if LOG_DEBUG
        va_list ap;
        va_start(ap, fmt);
        v_target(target, fmt, ap);
        va_end(ap);
#else
        (void)target;
        (void)fmt;
#endif
    }

    void DebugLog::operator()(LogTarget target, const char *file, int line, const char *func, const char *fmt, ...)
    {
#if LOG_DEBUG
        va_list ap;
        va_start(ap, fmt);
        v_target_loc(target, file, line, func, fmt, ap);
        va_end(ap);
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
        v_layer(layer, fmt, ap);
        va_end(ap);
    }

    void DebugLog::operator()(LogTarget target, const char *layer, const char *fmt, ...)
    {
#if LOG_DEBUG
        va_list ap;
        va_start(ap, fmt);
        v_target_layer(target, layer, fmt, ap);
        va_end(ap);
#else
        (void)target;
        (void)layer;
        (void)fmt;
#endif
    }

    void DebugLog::v(const char *fmt, va_list ap)
    {
        std::lock_guard<std::mutex> lock(detail::output_mutex());
        vwrite(out_, nullptr, 0, nullptr, fmt, ap);
    }

    void DebugLog::v_layer(const char *layer, const char *fmt, va_list ap)
    {
        std::lock_guard<std::mutex> lock(detail::output_mutex());
        vwrite_layer_fmt(out_, nullptr, 0, nullptr, layer, fmt, ap);
    }

    void DebugLog::v_loc(const char *file, int line, const char *func, const char *fmt, va_list ap)
    {
        std::lock_guard<std::mutex> lock(detail::output_mutex());
        vwrite(out_, file, line, func, fmt, ap);
    }

    void DebugLog::v_target(LogTarget target, const char *fmt, va_list ap)
    {
#if LOG_DEBUG
        std::lock_guard<std::mutex> lock(detail::output_mutex());
        detail::invoke_target_lock_hook(testing::TargetWriteKind::plain);
        bool owns = false;
        FILE *out = select_output_unlocked(target.path, &owns);
        ScopedFile owned_output(owns ? out : nullptr);
        vwrite(out, nullptr, 0, nullptr, fmt, ap);
#else
        (void)target;
        (void)fmt;
        (void)ap;
#endif
    }

    void DebugLog::v_target_layer(LogTarget target, const char *layer, const char *fmt, va_list ap)
    {
#if LOG_DEBUG
        std::lock_guard<std::mutex> lock(detail::output_mutex());
        detail::invoke_target_lock_hook(testing::TargetWriteKind::layer);
        bool owns = false;
        FILE *out = select_output_unlocked(target.path, &owns);
        ScopedFile owned_output(owns ? out : nullptr);
        vwrite_layer_fmt(out, nullptr, 0, nullptr, layer, fmt, ap);
#else
        (void)target;
        (void)layer;
        (void)fmt;
        (void)ap;
#endif
    }

    void DebugLog::v_target_loc(LogTarget target, const char *file, int line, const char *func,
                                const char *fmt, va_list ap)
    {
#if LOG_DEBUG
        std::lock_guard<std::mutex> lock(detail::output_mutex());
        detail::invoke_target_lock_hook(testing::TargetWriteKind::location);
        bool owns = false;
        FILE *out = select_output_unlocked(target.path, &owns);
        ScopedFile owned_output(owns ? out : nullptr);
        vwrite(out, file, line, func, fmt, ap);
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
