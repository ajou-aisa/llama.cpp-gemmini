#include "tiling.hpp"

#include <cmath>

namespace ggml::gemmini::tiling {

namespace {

size_t padded(size_t dim, size_t d) {
    return (dim / d + (dim % d != 0)) * d;
}

gemmini_status_t issue(const gemmini_transport_t *transport, const gemmini_command_t &command) {
    return gemmini_issue(transport, &command, nullptr);
}

} // namespace

size_t total_spad_rows(const gemmini_hw_config_t &cfg, size_t i, size_t j, size_t k) {
    return (i * k + k * j) * cfg.dim;
}

size_t total_acc_rows(const gemmini_hw_config_t &cfg, size_t i, size_t j) {
    return (i * j) * cfg.dim;
}

Tiles select_tiles(const gemmini_hw_config_t &cfg, size_t dim_i, size_t dim_j, size_t dim_k, int act,
                   bool double_buffered) {
    const size_t d             = cfg.dim;
    const size_t spad_rows     = gemmini_spad_rows(&cfg);
    const size_t max_spad_rows = double_buffered ? spad_rows / 2 : spad_rows;
    const size_t max_acc_rows  = double_buffered ? cfg.acc_rows / 2 : cfg.acc_rows;
    // tiled_matmul_auto's partition_rows / db_partition_rows macros.
    const size_t partition_rows   = double_buffered ? (spad_rows / 2) / 2 : spad_rows / 2;
    const size_t mats_in_partition = partition_rows / d;
    const size_t mats_in_acc       = (double_buffered ? cfg.acc_rows / 2 : cfg.acc_rows) / d;
    const size_t max_tile_i_j      = (size_t) std::sqrt(mats_in_acc);
    const size_t max_tile_k        = mats_in_partition / max_tile_i_j;

    const size_t dim_i_padded = padded(dim_i, d);
    const size_t dim_j_padded = padded(dim_j, d);
    const size_t dim_k_padded = padded(dim_k, d);

    Tiles t;
    if (act == GEMMINI_ACT_LAYERNORM || act == GEMMINI_ACT_SOFTMAX) {
        t.i = 1;
        t.j = dim_j_padded / d;
        t.k = 1;
    } else {
        t.i = dim_i_padded / d < max_tile_i_j ? dim_i_padded / d : max_tile_i_j;
        t.j = dim_j_padded / d < max_tile_i_j ? dim_j_padded / d : max_tile_i_j;
        t.k = dim_k_padded / d < max_tile_k ? dim_k_padded / d : max_tile_k;
    }

    // Fill the scratchpad as much as possible, growing J, then I, then K.
    while (true) {
        bool increased = false;
        if (total_spad_rows(cfg, t.i, t.j + 1, t.k) <= max_spad_rows &&
            total_acc_rows(cfg, t.i, t.j + 1) <= max_acc_rows && (t.j + 1) * d <= dim_j_padded) {
            t.j++;
            increased = true;
        }
        if (total_spad_rows(cfg, t.i + 1, t.j, t.k) <= max_spad_rows &&
            total_acc_rows(cfg, t.i + 1, t.j) <= max_acc_rows && (t.i + 1) * d <= dim_i_padded) {
            t.i++;
            increased = true;
        }
        if (total_spad_rows(cfg, t.i, t.j, t.k + 1) <= max_spad_rows && (t.k + 1) * d <= dim_k_padded) {
            t.k++;
            increased = true;
        }
        if (!increased) {
            break;
        }
    }
    return t;
}

Tiles select_ws_tiles(const gemmini_hw_config_t &cfg, size_t dim_i, size_t dim_j, size_t dim_k, int act) {
    return select_tiles(cfg, dim_i, dim_j, dim_k, act, true);
}

gemmini_status_t check_ws(const gemmini_hw_config_t &cfg, const Matmul &mm, const Tiles &tiles,
                          const char **message) {
    const char *unused;
    const char **out = message ? message : &unused;
    const size_t d   = cfg.dim;
    if (tiles.i <= 0) {
        *out = "tile_I is non-positive\n";
    } else if (tiles.j <= 0) {
        *out = "tile_J is non-positive\n";
    } else if (tiles.k <= 0) {
        *out = "tile_K is non-positive\n";
    } else if (tiles.i * d > padded(mm.dim_i, d)) {
        *out = "tile_I is too large (tile_I * DIM > dim_I_padded)\n";
    } else if (tiles.j * d > padded(mm.dim_j, d)) {
        *out = "tile_J is too large (tile_J * DIM > dim_J_padded)\n";
    } else if (tiles.k * d > padded(mm.dim_k, d)) {
        *out = "tile_K is too large (tile_K * DIM > dim_K_padded)\n";
    } else if (total_spad_rows(cfg, tiles.i, tiles.j, tiles.k) > gemmini_spad_rows(&cfg) / 2) {
        *out = "Not enough space in scratchpad to store A and B matrices\n";
        return GEMMINI_ERR_CAPACITY;
    } else if (total_acc_rows(cfg, tiles.i, tiles.j) > cfg.acc_rows / 2) {
        *out = "Not enough space in accumulator to store C\n";
        return GEMMINI_ERR_CAPACITY;
    } else if (tiles.i > 65535 || tiles.j > 65535 || tiles.k > 65535) {
        *out = "I, J, and K tiling factors must be less than 65535, to fit within the bounds of the LOOP_WS function";
    } else if (mm.a_transpose && mm.b_transpose) {
        *out = "Not implemented: WS matmul, a_transpose=1, b_transpose=1\n";
        return GEMMINI_ERR_UNSUPPORTED;
    } else if (mm.act != GEMMINI_ACT_NONE && mm.act != GEMMINI_ACT_RELU) {
        *out = "Not implemented in the standalone tiler: LAYERNORM/SOFTMAX/IGELU (CONFIG_NORM)\n";
        return GEMMINI_ERR_UNSUPPORTED;
    } else {
        return GEMMINI_OK;
    }
    return GEMMINI_ERR_INVALID_ARGUMENT;
}

gemmini_status_t run_ws(const gemmini_transport_t *transport, const gemmini_hw_config_t &cfg, const Matmul &mm,
                        const Tiles &tiles, const Observer *observer) {
    const gemmini_status_t checked = check_ws(cfg, mm, tiles, nullptr);
    if (checked != GEMMINI_OK) {
        return checked;
    }
    const size_t d            = cfg.dim;
    const size_t dim_i_padded = padded(mm.dim_i, d);
    const size_t dim_j_padded = padded(mm.dim_j, d);
    const size_t dim_k_padded = padded(mm.dim_k, d);

    const size_t i0_count = dim_i_padded / (tiles.i * d) + (dim_i_padded % (tiles.i * d) != 0);
    const size_t j0_count = dim_j_padded / (tiles.j * d) + (dim_j_padded % (tiles.j * d) != 0);
    const size_t k0_count = dim_k_padded / (tiles.k * d) + (dim_k_padded % (tiles.k * d) != 0);

    const size_t last_i = dim_i_padded % (tiles.i * d) == 0 ? tiles.i : (dim_i_padded / d) % tiles.i;
    const size_t last_j = dim_j_padded % (tiles.j * d) == 0 ? tiles.j : (dim_j_padded / d) % tiles.j;
    const size_t last_k = dim_k_padded % (tiles.k * d) == 0 ? tiles.k : (dim_k_padded / d) % tiles.k;

    const size_t padding_i = dim_i_padded - mm.dim_i;
    const size_t padding_j = dim_j_padded - mm.dim_j;
    const size_t padding_k = dim_k_padded - mm.dim_k;

    const bool no_bias = mm.d == 0;
    // The SDK replaced a NULL bias by the dummy address 1 so the first K tile
    // still passes a non-NULL D to sp_tiled_matmul_ws (ex_accumulate = 0).
    const uint64_t d_base = no_bias ? 1 : mm.d;

    const size_t elem    = cfg.input_bytes;
    const size_t sizeof_d = mm.low_d ? cfg.input_bytes : cfg.acc_bytes;
    const size_t sizeof_c = mm.full_c ? cfg.acc_bytes : cfg.input_bytes;
    const uint32_t act_low = (uint32_t) (mm.act & 3);

    gemmini_command_t command;
    gemmini_status_t  status;
    const gemmini_config_ex_desc_t ex = { GEMMINI_DATAFLOW_WS, act_low, 0, mm.config_ex_acc_scale_bits, 1, 1,
                                          mm.a_transpose, mm.b_transpose, false };
    if ((status = gemmini_encode_config_ex(&ex, &command)) != GEMMINI_OK || (status = issue(transport, command)) != GEMMINI_OK) {
        return status;
    }
    const gemmini_config_st_desc_t st = { (uint32_t) (mm.stride_c * sizeof_c), act_low, mm.acc_scale_bits };
    if ((status = gemmini_encode_config_st(&st, &command)) != GEMMINI_OK || (status = issue(transport, command)) != GEMMINI_OK) {
        return status;
    }
    const uint32_t                 block_stride = (uint32_t) d;
    const gemmini_config_ld_desc_t ld[3] = {
        { mm.stride_a * elem, mm.a_scale_bits, false, block_stride, 1, 0 },
        { mm.stride_b * elem, mm.b_scale_bits, false, block_stride, 1, 1 },
        { mm.repeating_bias ? 0 : mm.stride_d * sizeof_d, mm.d_scale_bits, mm.low_d, block_stride, 1, 2 },
    };
    for (const auto &desc : ld) {
        if ((status = gemmini_encode_config_ld(&desc, &command)) != GEMMINI_OK || (status = issue(transport, command)) != GEMMINI_OK) {
            return status;
        }
    }

    bool b_reuse = j0_count * k0_count <= 2;
    bool a_reuse = i0_count * k0_count <= 2;
    if (mm.disable_ws_reuse) {
        a_reuse = false;
        b_reuse = false;
    }
    if (observer && observer->before_loops) {
        observer->before_loops(observer->context, i0_count, j0_count, k0_count, a_reuse, b_reuse);
    }

    uint32_t a_spad_id = 0;
    uint32_t b_spad_id = 0;
    size_t   index     = 0;
    for (size_t i0 = 0; i0 < i0_count; i0++) {
        for (size_t j0 = 0; j0 < j0_count; j0++) {
            for (size_t k0 = 0; k0 < k0_count; k0++) {
                if (a_reuse) {
                    a_spad_id = ((i0 + k0) == 0) ? 1 : 2;
                }
                if (b_reuse) {
                    b_spad_id = ((j0 + k0) == 0) ? 1 : 2;
                }
                uint64_t pre = 0;
                if (k0 == 0) {
                    const size_t bias_row = mm.repeating_bias ? 0 : i0 * tiles.i * d;
                    pre = d_base + (bias_row * mm.stride_d + j0 * tiles.j * d) * sizeof_d;
                }
                const uint64_t out =
                    k0 == k0_count - 1 ? mm.c + (i0 * tiles.i * d * mm.stride_c + j0 * tiles.j * d) * sizeof_c : 0;

                uint64_t a = mm.a_transpose ? mm.a + (k0 * tiles.k * d * mm.stride_a + i0 * tiles.i * d) * elem
                                            : mm.a + (i0 * tiles.i * d * mm.stride_a + k0 * tiles.k * d) * elem;
                uint64_t b = mm.b_transpose ? mm.b + (j0 * tiles.j * d * mm.stride_b + k0 * tiles.k * d) * elem
                                            : mm.b + (k0 * tiles.k * d * mm.stride_b + j0 * tiles.j * d) * elem;
                if (a_reuse && j0 >= 1) {
                    a = 0;
                }
                if (b_reuse && i0 >= 1) {
                    b = 0;
                }

                // sp_tiled_matmul_ws: one combined LOOP_WS.
                gemmini_loop_ws_desc_t loop = {};
                loop.i             = (uint32_t) (i0 < i0_count - 1 ? tiles.i : last_i);
                loop.j             = (uint32_t) (j0 < j0_count - 1 ? tiles.j : last_j);
                loop.k             = (uint32_t) (k0 < k0_count - 1 ? tiles.k : last_k);
                loop.pad_i         = (uint32_t) (i0 == i0_count - 1 ? padding_i : 0);
                loop.pad_j         = (uint32_t) (j0 == j0_count - 1 ? padding_j : 0);
                loop.pad_k         = (uint32_t) (k0 == k0_count - 1 ? padding_k : 0);
                loop.a             = a;
                loop.b             = b;
                loop.d             = no_bias ? 0 : pre;
                loop.c             = out;
                loop.a_stride      = mm.stride_a;
                loop.b_stride      = mm.stride_b;
                loop.d_stride      = mm.repeating_bias ? 0 : mm.stride_d;
                loop.c_stride      = mm.stride_c;
                loop.a_transpose   = mm.a_transpose;
                loop.b_transpose   = mm.b_transpose;
                loop.full_c        = mm.full_c;
                loop.low_d         = mm.low_d;
                loop.ex_accumulate = !no_bias || pre == 0;
                loop.activation    = (uint32_t) mm.act;
                loop.a_spad_id     = a_spad_id;
                loop.b_spad_id     = b_spad_id;
                loop.is_resadd     = false;

                if (observer && observer->before_inner) {
                    observer->before_inner(observer->context, index, i0, j0, k0, pre, loop);
                }
                gemmini_command_t commands[GEMMINI_LOOP_WS_COMMANDS];
                if ((status = gemmini_encode_loop_ws(&loop, commands)) != GEMMINI_OK ||
                    (status = gemmini_issue_all(transport, commands, GEMMINI_LOOP_WS_COMMANDS)) != GEMMINI_OK) {
                    return status;
                }
                if (observer && observer->after_inner) {
                    observer->after_inner(observer->context, index, i0, j0, k0);
                }
                ++index;
            }
        }
    }

    if ((status = gemmini_drain(transport)) != GEMMINI_OK) {
        return status;
    }
    if (observer && observer->after_fence) {
        observer->after_fence(observer->context);
    }
    return GEMMINI_OK;
}

} // namespace ggml::gemmini::tiling
