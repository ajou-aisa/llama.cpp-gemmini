/*
 * Logging API usage (no macros):
 * - Compile-time toggles: LOG_DEBUG, LOG_CYCLE
 * - Default output (stderr unless set_output_path):
 *   - ggml::gemmini::log::debug("x=%d", x);
 *   - ggml::gemmini::log::debug("layer", "x=%d", x);
 *   - ggml::gemmini::log::cycle("layer", "op", start, end);
 * - File output (per-call target, filepath is first argument):
 *   - ggml::gemmini::log::debug(ggml::gemmini::log::file("out.log"), "x=%d", x);
 *   - ggml::gemmini::log::debug(ggml::gemmini::log::file("out.log"), "layer", "x=%d", x);
 *   - ggml::gemmini::log::cycle(ggml::gemmini::log::file("out.log"), "layer", "op, start, end);
 * - Persistent output routing (changes default output):
 *   - ggml::gemmini::log::debug.set_output_path("out.log");
 *   - ggml::gemmini::log::cycle.set_output_path("out.log");
 *
 * Path resolution notes:
 * - Absolute paths are used as-is.
 * - Relative paths are normally resolved against the process working directory (CWD).
 * - If `GEMMINI_LOG_DIR` is set, all relative paths are resolved under it.
 *   For paths starting with `log/`, the `log/` prefix is stripped
 *   (so `log/out.jsonl` -> `$GEMMINI_LOG_DIR/out.jsonl`).
 * - Otherwise, relative paths are resolved under `./output/log/` (CWD).
 * - Relative traversal is rejected.
 *
 * Output format:
 * - Logs are emitted as JSON Lines (JSONL): 1 JSON object per line.
 * - JSON fields include only the data that was present in the previous text format.
 */
#pragma once

#include "log.h"

#include <cstdarg>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <string_view>

#ifndef LOG_DEBUG
#define LOG_DEBUG 1
#endif
#ifndef LOG_CYCLE
#define LOG_CYCLE 1
#endif
#ifndef CYCLE_DETAIL
#define CYCLE_DETAIL 0
#endif
#ifndef LOG_DETAIL
#define LOG_DETAIL CYCLE_DETAIL
#endif

namespace ggml::gemmini::log
{
    struct LogTarget
    {
        const char *path;
    };

    LogTarget file(const char *path);

    // Resolves null/empty or traversing relative paths to empty, preserves absolute paths,
    // and confines every accepted relative path below GEMMINI_LOG_DIR or CWD/output/log.
    std::filesystem::path resolve_output_path(const char *path);
    bool prepare_output_parent(const std::filesystem::path &path);

    bool truncate_file(const char *path);

    struct DefaultOutputSetupResult
    {
        bool debug;
        bool cycle;
    };

    DefaultOutputSetupResult setup_default_outputs();

    class Log
    {
    public:
        explicit Log(FILE *out = stderr);
        Log(const Log &) = delete;
        Log &operator=(const Log &) = delete;
        virtual ~Log();

        void set_output(FILE *out);

        bool set_output_path(const char *path);

        bool has_explicit_output() const;

    protected:
        void set_output_unlocked(FILE *out);
        bool set_output_path_unlocked(const char *path, bool truncate, const char **failure = nullptr);
        FILE *select_output_unlocked(const char *path, bool *owns) const;
        void close_owned_unlocked();
        void disable_output_unlocked();

        FILE *out_;

    private:
        bool owns_;
        bool has_explicit_output_ = false;
    };

    struct CycleRecord
    {
        const char *layer = nullptr;
        const char *op = nullptr;
        uint64_t start = 0;
        uint64_t end = 0;
        const char *file = nullptr;
        int line = 0;
        const char *func = nullptr;
        const char *source = nullptr;
        const char *unit = nullptr;
    };

    struct WsCycleRecord
    {
        uint64_t containing_interval_cycles = 0;
        uint32_t load_occupancy_cycles = 0;
        uint32_t execute_occupancy_cycles = 0;
        uint32_t store_occupancy_cycles = 0;
        uint32_t loop_occupancy_cycles = 0;
        uint64_t problem_i = 0;
        uint64_t problem_j = 0;
        uint64_t problem_k = 0;
        uint64_t tile_i = 0;
        uint64_t tile_j = 0;
        uint64_t tile_k = 0;
        uint64_t gemmini_outer_i = 0;
        uint64_t gemmini_outer_j = 0;
        uint64_t gemmini_outer_k = 0;
        uint64_t ws_inner_calls = 0;
    };

    std::string serialize_cycle_record(const CycleRecord &record);
    std::string serialize_ws_cycle_record(const WsCycleRecord &record);

    class DebugLog : public Log
    {
    public:
        explicit DebugLog(FILE *out = stderr);
        bool set_output_path(const char *path, bool truncate = false);

        void operator()(const char *fmt, ...);
        void operator()(const char *file, int line, const char *func, const char *fmt, ...);
        void operator()(LogTarget target, const char *fmt, ...);
        void operator()(LogTarget target, const char *file, int line, const char *func, const char *fmt, ...);
        void operator()(const char *layer, const char *fmt, ...);
        void operator()(LogTarget target, const char *layer, const char *fmt, ...);

        void v(const char *fmt, va_list ap);
        void v_layer(const char *layer, const char *fmt, va_list ap);
        void v_loc(const char *file, int line, const char *func, const char *fmt, va_list ap);
        void v_target(LogTarget target, const char *fmt, va_list ap);
        void v_target_layer(LogTarget target, const char *layer, const char *fmt, va_list ap);
        void v_target_loc(LogTarget target, const char *file, int line, const char *func,
                          const char *fmt, va_list ap);

    private:
        void vwrite(FILE *out, const char *file, int line, const char *func, const char *fmt, va_list ap);
        void vwrite_layer_fmt(FILE *out, const char *file, int line, const char *func, const char *layer, const char *fmt, va_list ap);
    };

    class HardwareCounterLease
    {
    public:
        HardwareCounterLease();
        ~HardwareCounterLease();
        HardwareCounterLease(const HardwareCounterLease &) = delete;
        HardwareCounterLease &operator=(const HardwareCounterLease &) = delete;
    };

    class CycleLog : public Log
    {
    public:
        explicit CycleLog(FILE *out = stderr) : Log(out) {}
        void set_output(FILE *out);
        bool set_output_path(const char *path, bool truncate = false);

        void write(const CycleRecord &record);
        void write_json(std::string_view json_record);
        void report_failure(const char * operation) noexcept;

        void operator()(const char *layer, const char *op,
                        uint64_t start, uint64_t end);
        void operator()(const char *file, int line, const char *func, const char *layer, const char *op,
                        uint64_t start, uint64_t end);
        void operator()(LogTarget target, const char *layer, const char *op,
                        uint64_t start, uint64_t end);
        void operator()(LogTarget target, const char *file, int line, const char *func, const char *layer, const char *op,
                        uint64_t start, uint64_t end);

        void cycle(const char *layer, const char *op,
                   uint64_t start, uint64_t end);

    private:
        void emit(const char *path, const std::string &json);
        void warn_once_unlocked(const char *operation);

        bool disabled_ = false;
        bool warned_ = false;
    };

    namespace testing
    {
        enum class LogFault
        {
            none,
            open,
            write,
            flush,
            replacement,
            allocation,
            filesystem,
            format,
            mutex,
        };

        enum class TargetWriteKind
        {
            plain,
            layer,
            location,
        };

        using TargetLockHook = void (*)(TargetWriteKind kind, void *user_data);
        void set_target_lock_hook(TargetLockHook hook, void *user_data);
        void clear_target_lock_hook();
        void set_log_fault(LogFault fault);
        void clear_log_fault();
    }

    namespace detail
    {
        std::mutex &output_mutex();
        bool consume_fault(testing::LogFault expected);
        void invoke_target_lock_hook(testing::TargetWriteKind kind);
    }

    extern DebugLog debug;
    extern CycleLog cycle;

} // namespace ggml::gemmini::log
