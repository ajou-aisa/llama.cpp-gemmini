#include <gemmini/log.h>

int gemmini_log_c_boundary_call(int operation, const char * path);

int gemmini_log_c_boundary_call(int operation, const char * path) {
    switch (operation) {
        case 0: return gemmini_log_cycle_set_output_path(path);
        case 1: gemmini_log_debug("c-format-boundary-%d", 1); return 1;
        case 2: gemmini_log_debug("c-mutex-boundary-%d", 2); return 1;
        case 3:
            gemmini_log_ws_cycle(100, 10, 20, 30, 40,
                                 2, 3, 4, 1, 1, 1, 2, 3, 4, 0, 1);
            return 1;
        case 4: gemmini_log_debug_to(gemmini_log_file(path), "c-target-format-%d", 4); return 1;
        case 5: gemmini_log_debug_to_layer(gemmini_log_file(path), "c-layer", "c-target-layer-%d", 5); return 1;
        case 6: gemmini_log_debug_to_loc(gemmini_log_file(path), "c-file", 6, "c-func", "c-target-loc-%d", 6); return 1;
        default: return 0;
    }
}
