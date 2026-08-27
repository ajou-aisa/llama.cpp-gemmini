#include "../include/gemmini/log.hpp"
#include <mutex>
#include <string>

namespace ggml::gemmini::log
{
    CycleLog cycle;

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
