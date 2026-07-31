#include "../include/gemmini/log.hpp"
#include "../include/gemmini/log.h"

#include <cstdarg>
#include <cstdint>

extern "C"
{

    gemmini_log_target gemmini_log_file(const char *path)
    {
        gemmini_log_target t;
        t.path = path;
        return t;
    }

    int gemmini_log_truncate_file(const char *path)
    {
        return ggml::gemmini::log::truncate_file(path) ? 1 : 0;
    }

    int gemmini_log_debug_set_output_path(const char *path)
    {
        return ggml::gemmini::log::debug.set_output_path(path) ? 1 : 0;
    }

    void gemmini_log_debug_set_output(FILE *out)
    {
        ggml::gemmini::log::debug.set_output(out);
    }

    int gemmini_log_cycle_set_output_path(const char *path)
    {
        return ggml::gemmini::log::cycle.set_output_path(path) ? 1 : 0;
    }

    void gemmini_log_cycle_set_output(FILE *out)
    {
        ggml::gemmini::log::cycle.set_output(out);
    }

    void gemmini_log_debug(const char *fmt, ...)
    {
        va_list ap;
        va_start(ap, fmt);
        ggml::gemmini::log::debug.v(fmt, ap);
        va_end(ap);
    }

    void gemmini_log_debug_layer(const char *layer, const char *fmt, ...)
    {
        va_list ap;
        va_start(ap, fmt);
        ggml::gemmini::log::debug.v_layer(layer, fmt, ap);
        va_end(ap);
    }

    void gemmini_log_debug_loc(const char *file, int line, const char *func, const char *fmt, ...)
    {
        va_list ap;
        va_start(ap, fmt);
        ggml::gemmini::log::debug.v_loc(file, line, func, fmt, ap);
        va_end(ap);
    }

    void gemmini_log_debug_to(gemmini_log_target target, const char *fmt, ...)
    {
        va_list ap;
        va_start(ap, fmt);
        ggml::gemmini::log::debug.v_target(ggml::gemmini::log::file(target.path), fmt, ap);
        va_end(ap);
    }

    void gemmini_log_debug_to_layer(gemmini_log_target target, const char *layer, const char *fmt, ...)
    {
        va_list ap;
        va_start(ap, fmt);
        ggml::gemmini::log::debug.v_target_layer(ggml::gemmini::log::file(target.path), layer, fmt, ap);
        va_end(ap);
    }

    void gemmini_log_debug_to_loc(gemmini_log_target target, const char *file, int line, const char *func, const char *fmt, ...)
    {
        va_list ap;
        va_start(ap, fmt);
        ggml::gemmini::log::debug.v_target_loc(ggml::gemmini::log::file(target.path), file, line, func, fmt, ap);
        va_end(ap);
    }

    void gemmini_log_ws_loop(uint64_t wall, uint64_t load, uint64_t exe, uint64_t store, uint64_t loop,
                             uint64_t dim_I, uint64_t dim_J, uint64_t dim_K,
                             uint64_t tile_I, uint64_t tile_J, uint64_t tile_K,
                             uint64_t I0, uint64_t J0, uint64_t K0,
                             uint64_t a_reuse, uint64_t b_reuse)
    {
        const gemmini_log_target target = gemmini_log_file("log/log-ws-loop.jsonl");
        ggml::gemmini::log::debug.ws_loop(ggml::gemmini::log::file(target.path),
                                          wall, load, exe, store, loop,
                                          dim_I, dim_J, dim_K,
                                          tile_I, tile_J, tile_K,
                                          I0, J0, K0, a_reuse, b_reuse);
    }

    void gemmini_log_cycle(const char *layer, const char *op, uint64_t start, uint64_t end)
    {
        ggml::gemmini::log::cycle(layer, op, start, end);
    }

    void gemmini_log_cycle_loc(const char *file, int line, const char *func,
                               const char *layer, const char *op, uint64_t start, uint64_t end)
    {
        ggml::gemmini::log::cycle(file, line, func, layer, op, start, end);
    }

    void gemmini_log_cycle_to(gemmini_log_target target, const char *layer, const char *op,
                              uint64_t start, uint64_t end)
    {
        ggml::gemmini::log::cycle(ggml::gemmini::log::file(target.path), layer, op, start, end);
    }

    void gemmini_log_cycle_to_loc(gemmini_log_target target, const char *file, int line, const char *func,
                                  const char *layer, const char *op, uint64_t start, uint64_t end)
    {
        ggml::gemmini::log::cycle(ggml::gemmini::log::file(target.path),
                                  file, line, func, layer, op,
                                  start, end);
    }
} // extern "C"
