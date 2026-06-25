/*
 * Logging API usage (no macros):
 * - Compile-time toggles: LOG_DEBUG, LOG_CYCLE
 * - Default output (stderr unless set_output_path):
 *   - gemmini_log_debug("x=%d", x);
 *   - gemmini_log_debug_layer("layer", "x=%d", x);
 *   - gemmini_log_cycle("layer", "op", start, end);
 * - File output (per-call target, filepath is first argument):
 *   - gemmini_log_debug_to(gemmini_log_file("out.log"), "x=%d", x);
 *   - gemmini_log_debug_to_layer(gemmini_log_file("out.log"), "layer", "x=%d", x);
 *   - gemmini_log_cycle_to(gemmini_log_file("out.log"), "layer", "op", start, end);
 * - Persistent output routing (changes default output):
 *   - gemmini_log_debug_set_output_path("out.log");
 *   - gemmini_log_cycle_set_output_path("out.log");
 *   - gemmini_log_debug_set_output(stderr);
 *   - gemmini_log_cycle_set_output(stderr);
 *
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

#include <stdint.h> // uint64_t
#include <stdio.h>  // FILE

#ifdef __cplusplus
extern "C"
{
#endif
    typedef struct gemmini_log_target
    {
        const char *path;
    } gemmini_log_target;

    gemmini_log_target gemmini_log_file(const char *path);
    int gemmini_log_truncate_file(const char *path); // 1: 성공, 0: 실패

    int gemmini_log_debug_set_output_path(const char *path); // 1/0
    int gemmini_log_cycle_set_output_path(const char *path); // 1/0
    void gemmini_log_debug_set_output(FILE *out);
    void gemmini_log_cycle_set_output(FILE *out);

    void gemmini_log_debug(const char *fmt, ...);

    void gemmini_log_debug_layer(const char *layer, const char *fmt, ...);
    void gemmini_log_debug_loc(const char *file, int line, const char *func, const char *fmt, ...);

    void gemmini_log_debug_to(gemmini_log_target target, const char *fmt, ...);
    void gemmini_log_debug_to_layer(gemmini_log_target target, const char *layer, const char *fmt, ...);
    void gemmini_log_debug_to_loc(gemmini_log_target target, const char *file, int line, const char *func, const char *fmt, ...);

    void gemmini_log_cycle(const char *layer, const char *op, uint64_t start, uint64_t end);
    void gemmini_log_cycle_loc(const char *file, int line, const char *func,
                               const char *layer, const char *op, uint64_t start, uint64_t end);

    void gemmini_log_cycle_to(gemmini_log_target target, const char *layer, const char *op,
                              uint64_t start, uint64_t end);
    void gemmini_log_cycle_to_loc(gemmini_log_target target, const char *file, int line, const char *func,
                                  const char *layer, const char *op, uint64_t start, uint64_t end);

#ifdef __cplusplus
}
#endif
