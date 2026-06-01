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
 * Path resolution notes:
 * - Absolute paths are used as-is.
 * - If `GEMMINI_LOG_DIR` is set, all relative paths are resolved under it.
 *   For paths starting with `log/`, the `log/` prefix is stripped
 *   (so `log/out.jsonl` -> `$GEMMINI_LOG_DIR/out.jsonl`).
 * - Otherwise, paths starting with `log/` are resolved under `./log/` (CWD),
 *   and the `log/` directory is created when needed.
 *
 * Output format:
 * - Logs are emitted as JSON Lines (JSONL): 1 JSON object per line.
 * - JSON fields include only the data that was present in the previous text format.
 */
#pragma once

#include <cstdarg>
#include <cstdio>
#include <cstdint>

#ifndef LOG_DEBUG
#define LOG_DEBUG 1
#endif
#ifndef LOG_CYCLE
#define LOG_CYCLE 1
#endif

namespace ggml::gemmini::log
{
    struct LogTarget
    {
        const char *path;
    };

    LogTarget file(const char *path);

    bool truncate_file(const char *path);

    class Log
    {
    public:
        explicit Log(FILE *out = stderr);
        Log(const Log &) = delete;
        Log &operator=(const Log &) = delete;
        virtual ~Log();

        void set_output(FILE *out);

        bool set_output_path(const char *path);

    protected:
        FILE *select_output(const char *path, bool *owns) const;

        FILE *out_;

    private:
        void close_owned();

        bool owns_;
    };

    class DebugLog : public Log
    {
    public:
        explicit DebugLog(FILE *out = stderr, bool add_newline = false);

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

        bool add_newline_;
    };

    class CycleLog : public Log
    {
    public:
        using Log::Log;

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
        void write(FILE *out, const char *file, int line, const char *func, const char *layer, const char *op,
                   uint64_t start, uint64_t end);
    };

    extern DebugLog debug;
    extern CycleLog cycle;

} // namespace ggml::gemmini::log
