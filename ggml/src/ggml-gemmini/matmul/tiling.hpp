// Gemmini weight-stationary tiler on the standalone API.
//
// Port of the SDK tiler the backend used before Phase 1 (gemmini.h develop
// fcfcf5ce: gemmini_set_tile_ws, tiled_matmul_auto's tile choice,
// tiled_matmul's argument checks, tiled_matmul_outer, sp_tiled_matmul_ws).
// I/J/K keep the SDK meaning: matrix dimensions in elements, tiles in DIM
// blocks. Commands, their order and the final fence are unchanged; geometry
// comes from a gemmini_hw_config_t instead of DIM/BANK_NUM/BANK_ROWS/ACC_ROWS
// macros, so one binary can plan for several configurations.
#pragma once

#include <cstddef>
#include <cstdint>

#include "gemmini.h"

namespace ggml::gemmini::tiling {

struct Tiles {
    size_t i = 0;
    size_t j = 0;
    size_t k = 0;
};

// gemmini_set_tile_ws: double-buffered WS plan (half SPAD, half ACC).
Tiles select_ws_tiles(const gemmini_hw_config_t &cfg, size_t dim_i, size_t dim_j, size_t dim_k, int act);
// tiled_matmul_auto's choice; double_buffered is true for WS.
Tiles select_tiles(const gemmini_hw_config_t &cfg, size_t dim_i, size_t dim_j, size_t dim_k, int act,
                   bool double_buffered);

size_t total_spad_rows(const gemmini_hw_config_t &cfg, size_t i, size_t j, size_t k);
size_t total_acc_rows(const gemmini_hw_config_t &cfg, size_t i, size_t j);

// Callbacks at the points where the SDK logged or sampled counters; any may
// be null. They run between commands and must not issue to the transport
// except through the same transport, as the SDK's WS counter sampling did.
struct Observer {
    void *context = nullptr;
    void (*before_loops)(void *context, size_t i0_count, size_t j0_count, size_t k0_count, bool a_reuse,
                         bool b_reuse) = nullptr;
    void (*before_inner)(void *context, size_t index, size_t i0, size_t j0, size_t k0, uint64_t pre,
                         const gemmini_loop_ws_desc_t &loop) = nullptr;
    void (*after_inner)(void *context, size_t index, size_t i0, size_t j0, size_t k0) = nullptr;
    void (*after_fence)(void *context) = nullptr;
};

// One matmul in SDK argument order. Pointers are DRAM byte addresses
// (0: absent); strides are in elements. Scale operands are the IEEE-754 bits
// the SDK encoded: scale_t_to_scale_t_bits(A/B/D scale factor) and
// acc_scale_t_to_acc_scale_t_bits(scale).
struct Matmul {
    size_t   dim_i = 0, dim_j = 0, dim_k = 0;
    uint64_t a = 0, b = 0, d = 0, c = 0;
    size_t   stride_a = 0, stride_b = 0, stride_d = 0, stride_c = 0;
    uint32_t a_scale_bits = 0, b_scale_bits = 0, d_scale_bits = 0;
    int      act = GEMMINI_ACT_NONE;
    uint32_t acc_scale_bits = 0;
    uint32_t config_ex_acc_scale_bits = 0; // ACC_SCALE_IDENTITY: CONFIG_EX never carries `scale`
    bool     repeating_bias = false;
    bool     a_transpose = false;
    bool     b_transpose = false;
    bool     full_c = false;
    bool     low_d = false;
    bool     disable_ws_reuse = false; // GEMMINI_DISABLE_WS_REUSE
};

// tiled_matmul's checks for a WS run, in its order. The SDK printed the
// returned message and exit(1); `message` receives it (never null on error).
gemmini_status_t check_ws(const gemmini_hw_config_t &cfg, const Matmul &mm, const Tiles &tiles, const char **message);

// tiled_matmul_outer + sp_tiled_matmul_ws (WEIGHT_STATIONARY): CONFIG_EX,
// CONFIG_ST, three CONFIG_LD, one LOOP_WS per (i0, j0, k0) tile, then drain.
// NO_ACTIVATION and RELU only: the SDK's LAYERNORM/SOFTMAX/IGELU branches
// (CONFIG_NORM) have no caller in this backend and are not ported.
gemmini_status_t run_ws(const gemmini_transport_t *transport, const gemmini_hw_config_t &cfg, const Matmul &mm,
                        const Tiles &tiles, const Observer *observer = nullptr);

} // namespace ggml::gemmini::tiling
