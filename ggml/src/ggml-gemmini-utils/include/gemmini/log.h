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
 * - Structured cycle records:
 *   - gemmini_log_cycle_record(&record);
 *
 * Path resolution notes:
 * - Absolute paths are used as-is.
 * - If `GEMMINI_LOG_DIR` is set, all relative paths are resolved under it.
 *   For paths starting with `log/`, the `log/` prefix is stripped
 *   (so `log/out.jsonl` -> `$GEMMINI_LOG_DIR/out.jsonl`).
 * - Otherwise, all relative paths are resolved under `./output/log/` (CWD).
 * - Relative traversal is rejected.
 *
 * Output format:
 * - Logs are emitted as JSON Lines (JSONL): 1 JSON object per line.
 * - JSON fields include only the data that was present in the previous text format.
 */
#pragma once

#include <stdint.h> // uint64_t
#include <stdio.h>  // FILE

#define GEMMINI_LOG_DEFAULT_DEBUG_PATH "log/debug-log.jsonl"
#define GEMMINI_LOG_DEFAULT_CYCLE_PATH "log/cycle-log.jsonl"
#define GEMMINI_LOG_DEFAULT_EXSIA_DETAIL_PATH "log/exsia-cycle-detail.jsonl"

#ifdef __cplusplus
#define GEMMINI_LOG_C_BOUNDARY_NOEXCEPT noexcept
extern "C"
{
#else
#define GEMMINI_LOG_C_BOUNDARY_NOEXCEPT
#endif
    typedef struct gemmini_log_target
    {
        const char *path;
    } gemmini_log_target;

    typedef struct gemmini_cycle_record
    {
        const char *layer;
        const char *op;
        uint64_t start;
        uint64_t end;
        const char *file;
        int line;
        const char *func;
    } gemmini_cycle_record;

    gemmini_log_target gemmini_log_file(const char *path) GEMMINI_LOG_C_BOUNDARY_NOEXCEPT;
    int gemmini_log_truncate_file(const char *path) GEMMINI_LOG_C_BOUNDARY_NOEXCEPT; // 1: success, 0: failure

    int gemmini_log_debug_set_output_path(const char *path) GEMMINI_LOG_C_BOUNDARY_NOEXCEPT; // 1/0
    int gemmini_log_cycle_set_output_path(const char *path) GEMMINI_LOG_C_BOUNDARY_NOEXCEPT; // 1/0
    void gemmini_log_debug_set_output(FILE *out) GEMMINI_LOG_C_BOUNDARY_NOEXCEPT;
    void gemmini_log_cycle_set_output(FILE *out) GEMMINI_LOG_C_BOUNDARY_NOEXCEPT;

    void gemmini_log_debug(const char *fmt, ...) GEMMINI_LOG_C_BOUNDARY_NOEXCEPT;

    void gemmini_log_debug_layer(const char *layer, const char *fmt, ...) GEMMINI_LOG_C_BOUNDARY_NOEXCEPT;
    void gemmini_log_debug_loc(const char *file, int line, const char *func, const char *fmt, ...) GEMMINI_LOG_C_BOUNDARY_NOEXCEPT;

    void gemmini_log_debug_to(gemmini_log_target target, const char *fmt, ...) GEMMINI_LOG_C_BOUNDARY_NOEXCEPT;
    void gemmini_log_debug_to_layer(gemmini_log_target target, const char *layer, const char *fmt, ...) GEMMINI_LOG_C_BOUNDARY_NOEXCEPT;
    void gemmini_log_debug_to_loc(gemmini_log_target target, const char *file, int line, const char *func, const char *fmt, ...) GEMMINI_LOG_C_BOUNDARY_NOEXCEPT;
    void gemmini_hardware_counter_lease_acquire(void) GEMMINI_LOG_C_BOUNDARY_NOEXCEPT;
    void gemmini_hardware_counter_lease_release(void) GEMMINI_LOG_C_BOUNDARY_NOEXCEPT;

    void gemmini_log_ws_cycle(uint64_t containing_interval_cycles,
                              uint32_t load_occupancy_cycles,
                              uint32_t execute_occupancy_cycles,
                              uint32_t store_occupancy_cycles,
                              uint32_t loop_occupancy_cycles,
                              uint64_t dim_I, uint64_t dim_J, uint64_t dim_K,
                              uint64_t tile_I, uint64_t tile_J, uint64_t tile_K,
                              uint64_t I0, uint64_t J0, uint64_t K0,
                              uint64_t a_reuse, uint64_t b_reuse) GEMMINI_LOG_C_BOUNDARY_NOEXCEPT;

    void gemmini_log_cycle_record(const gemmini_cycle_record *record) GEMMINI_LOG_C_BOUNDARY_NOEXCEPT;
    void gemmini_log_cycle(const char *layer, const char *op, uint64_t start, uint64_t end) GEMMINI_LOG_C_BOUNDARY_NOEXCEPT;
    void gemmini_log_cycle_loc(const char *file, int line, const char *func,
                               const char *layer, const char *op, uint64_t start, uint64_t end) GEMMINI_LOG_C_BOUNDARY_NOEXCEPT;

    void gemmini_log_cycle_to(gemmini_log_target target, const char *layer, const char *op,
                              uint64_t start, uint64_t end) GEMMINI_LOG_C_BOUNDARY_NOEXCEPT;
    void gemmini_log_cycle_to_loc(gemmini_log_target target, const char *file, int line, const char *func,
                                  const char *layer, const char *op, uint64_t start, uint64_t end) GEMMINI_LOG_C_BOUNDARY_NOEXCEPT;

#ifdef __cplusplus
}
#endif
#undef GEMMINI_LOG_C_BOUNDARY_NOEXCEPT
