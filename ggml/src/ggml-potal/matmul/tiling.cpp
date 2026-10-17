#include "tiling.hpp"

#include "options.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace ggml::potal {

namespace {

using clock_type = std::chrono::steady_clock;

constexpr size_t DIM = POTAL_HW_DIM;

double seconds_since(clock_type::time_point t0) {
    return std::chrono::duration<double>(clock_type::now() - t0).count();
}

// gemmini.h tiled_matmul_total_spad_rows / tiled_matmul_total_acc_rows.
size_t total_spad_rows(size_t I, size_t J, size_t K) {
    return (I * K + K * J) * DIM;
}

size_t total_acc_rows(size_t I, size_t J) {
    return (I * J) * DIM;
}

size_t padded(size_t dim) {
    return (dim / DIM + (dim % DIM != 0)) * DIM;
}

// WS: one loop's half of SPAD and ACC (tiled_matmul_auto, double_buffered).
constexpr size_t max_spad_rows() {
    return (size_t) POTAL_HW_BANK_NUM * POTAL_HW_BANK_ROWS / 2;
}

constexpr size_t max_acc_rows() {
    return (size_t) POTAL_HW_ACC_ROWS / 2;
}

// gemmini.h fits plus the PoTal limits a single LOOP_WS must respect.
bool fits(size_t I, size_t J, size_t K, uint32_t passes_per_block) {
    return total_spad_rows(I, J, K) <= max_spad_rows() && total_acc_rows(I, J) <= max_acc_rows() &&
           I * J <= POTAL_WORK_IDS_PER_LOOP && (K / passes_per_block) * J <= options::scale_window_slots;
}

// The requests the PoTal LOOP_WS path implements (see tiling.hpp).
potal_status check_request(const void * D, potal_scale_t A_scale_factor, potal_scale_t B_scale_factor,
                           potal_scale_acc_t D_scale_factor, int act, potal_acc_scale_t scale,
                           potal_acc_scale_t bert_scale, bool repeating_bias, bool transpose_A, bool transpose_B,
                           bool full_C, bool low_D, int tiled_matmul_type) {
    if (tiled_matmul_type != POTAL_WS || D != nullptr || repeating_bias || act != POTAL_NO_ACTIVATION ||
        A_scale_factor != POTAL_MVIN_SCALE_IDENTITY || B_scale_factor != POTAL_MVIN_SCALE_IDENTITY ||
        D_scale_factor != 0 || scale != POTAL_ACC_SCALE_IDENTITY || bert_scale != 0 || transpose_A ||
        transpose_B || !full_C || low_D) {
        return POTAL_ERR_UNSUPPORTED;
    }
    return POTAL_OK;
}

// Carriers cover whole scale blocks of a DIM-padded K (block_k is a multiple of DIM).
potal_status check_scales(size_t dim_J, size_t dim_K, const potal_scale_operand & scales,
                          uint32_t * passes_per_block) {
    if (scales.carriers == nullptr || scales.block_k == 0 || scales.block_k % DIM != 0 || scales.stride < dim_J ||
        dim_K % scales.block_k != 0) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    *passes_per_block = scales.block_k / DIM;
    return POTAL_OK;
}

potal_status fail(potal_ws_report & r, potal_status status, const char * stage, bool fatal) {
    r.status = status;
    r.stage  = stage;
    r.fatal  = r.fatal || fatal;
    return status;
}

potal_status hw_scale_write(void *, uint32_t slot, uint32_t generation, uint32_t column, uint32_t carrier) {
    return potal_scale_write(slot, generation, column, carrier);
}

potal_status hw_scale_release(void *, uint32_t slot, uint32_t generation, uint32_t column) {
    return potal_scale_release(slot, generation, column);
}

potal_status hw_gemm_ws_issue(void *, const potal_gemm_ws * gemm) {
    return potal_gemm_ws_issue(gemm);
}

potal_status hw_loop_ws_issue(void *, const potal_loop_ws * loop) {
    return potal_loop_ws_issue(loop);
}

potal_status hw_fence(void *) {
    return potal_fence();
}

} // namespace

const potal_ws_ops & potal_hw_ops() {
    static const potal_ws_ops ops = { hw_scale_write, hw_scale_release, hw_gemm_ws_issue, hw_loop_ws_issue,
                                      hw_fence, nullptr };
    return ops;
}

potal_status potal_tile_select(size_t dim_I, size_t dim_J, size_t dim_K, uint32_t passes_per_block,
                               potal_tile_plan * plan) {
    if (plan == nullptr || dim_I == 0 || dim_J == 0 || dim_K == 0 || passes_per_block == 0) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    const size_t dim_I_padded = padded(dim_I);
    const size_t dim_J_padded = padded(dim_J);
    const size_t dim_K_padded = padded(dim_K);
    const size_t ppb          = passes_per_block;
    if ((dim_K_padded / DIM) % ppb != 0) {
        return POTAL_ERR_INVALID_ARGUMENT; // K is not a whole number of scale blocks
    }

    // "db_" means "double-buffered" (gemmini.h tiled_matmul_auto).
    const size_t db_partition_rows    = ((size_t) POTAL_HW_BANK_NUM * POTAL_HW_BANK_ROWS / 2) / 2;
    const size_t db_mats_in_partition = db_partition_rows / DIM;
    const size_t db_mats_in_acc       = ((size_t) POTAL_HW_ACC_ROWS / 2) / DIM;
    const size_t db_max_tile_i_j      = (size_t) std::sqrt((double) db_mats_in_acc);
    const size_t db_max_tile_k        = db_mats_in_partition / db_max_tile_i_j;

    size_t tile_I = dim_I_padded / DIM < db_max_tile_i_j ? dim_I_padded / DIM : db_max_tile_i_j;
    size_t tile_J = dim_J_padded / DIM < db_max_tile_i_j ? dim_J_padded / DIM : db_max_tile_i_j;
    size_t tile_K = dim_K_padded / DIM < db_max_tile_k ? dim_K_padded / DIM : db_max_tile_k;

    // PoTal: whole scale blocks per invocation, and the start must satisfy the
    // PoTal limits gemmini.h does not have (shrink K, then J, then I).
    tile_K = std::max(ppb, tile_K - tile_K % ppb);
    while (!fits(tile_I, tile_J, tile_K, passes_per_block) && tile_K > ppb) {
        tile_K -= ppb;
    }
    while (!fits(tile_I, tile_J, tile_K, passes_per_block) && tile_J > 1) {
        tile_J--;
    }
    while (!fits(tile_I, tile_J, tile_K, passes_per_block) && tile_I > 1) {
        tile_I--;
    }
    if (!fits(tile_I, tile_J, tile_K, passes_per_block)) {
        return POTAL_ERR_CAPACITY;
    }

    // Fill scratchpad as much as possible (gemmini.h order: J, I, K).
    while (true) {
        bool increased = false;

        if (fits(tile_I, tile_J + 1, tile_K, passes_per_block) && (tile_J + 1) * DIM <= dim_J_padded) {
            tile_J++;
            increased = true;
        }

        if (fits(tile_I + 1, tile_J, tile_K, passes_per_block) && (tile_I + 1) * DIM <= dim_I_padded) {
            tile_I++;
            increased = true;
        }

        if (fits(tile_I, tile_J, tile_K + ppb, passes_per_block) && (tile_K + ppb) * DIM <= dim_K_padded) {
            tile_K += ppb;
            increased = true;
        }

        if (!increased) {
            break;
        }
    }

    plan->tile_I           = tile_I;
    plan->tile_J           = tile_J;
    plan->tile_K           = tile_K;
    plan->I0               = dim_I_padded / (tile_I * DIM) + (dim_I_padded % (tile_I * DIM) != 0);
    plan->J0               = dim_J_padded / (tile_J * DIM) + (dim_J_padded % (tile_J * DIM) != 0);
    plan->K0               = dim_K_padded / (tile_K * DIM) + (dim_K_padded % (tile_K * DIM) != 0);
    plan->pad_I            = dim_I_padded - dim_I;
    plan->pad_J            = dim_J_padded - dim_J;
    plan->pad_K            = dim_K_padded - dim_K;
    plan->spad_rows        = total_spad_rows(tile_I, tile_J, tile_K);
    plan->max_spad_rows    = max_spad_rows();
    plan->acc_rows         = total_acc_rows(tile_I, tile_J);
    plan->max_acc_rows     = max_acc_rows();
    plan->passes_per_block = passes_per_block;
    plan->scale_slots      = (uint32_t) ((tile_K / ppb) * tile_J);
    return POTAL_OK;
}

potal_status potal_tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
                                     const int8_t * A, const int8_t * B,
                                     const void * D, int32_t * C,
                                     size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
                                     potal_scale_t A_scale_factor, potal_scale_t B_scale_factor,
                                     potal_scale_acc_t D_scale_factor,
                                     int act, potal_acc_scale_t scale, potal_acc_scale_t bert_scale,
                                     bool repeating_bias,
                                     bool transpose_A, bool transpose_B,
                                     bool full_C, bool low_D,
                                     uint8_t weightA,
                                     potal_tiled_matmul_type_t tiled_matmul_type,
                                     const potal_scale_operand & scales, potal_ws_session & session,
                                     potal_tile_plan * plan_out) {
    potal_status status = check_request(D, A_scale_factor, B_scale_factor, D_scale_factor, act, scale, bert_scale,
                                        repeating_bias, transpose_A, transpose_B, full_C, low_D, tiled_matmul_type);
    uint32_t passes_per_block = 0;
    if (status == POTAL_OK) {
        status = check_scales(dim_J, dim_K, scales, &passes_per_block);
    }
    potal_tile_plan plan;
    if (status == POTAL_OK) {
        status = potal_tile_select(dim_I, dim_J, dim_K, passes_per_block, &plan);
    }
    if (status != POTAL_OK) {
        return fail(session.report, status, "tile_select", false);
    }
    if (plan_out != nullptr) {
        *plan_out = plan;
    }

    return potal_tiled_matmul(dim_I, dim_J, dim_K,
                              A, B, D, C,
                              stride_A, stride_B, stride_D, stride_C,
                              A_scale_factor, B_scale_factor, D_scale_factor,
                              act, scale, bert_scale, repeating_bias,
                              plan.tile_I, plan.tile_J, plan.tile_K,
                              transpose_A, transpose_B,
                              full_C, low_D,
                              weightA,
                              tiled_matmul_type,
                              scales, session);
}

potal_status potal_tiled_matmul(size_t dim_I, size_t dim_J, size_t dim_K,
                                const int8_t * A, const int8_t * B,
                                const void * D, int32_t * C,
                                size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
                                potal_scale_t A_scale_factor, potal_scale_t B_scale_factor,
                                potal_scale_acc_t D_scale_factor,
                                int act, potal_acc_scale_t scale, potal_acc_scale_t bert_scale,
                                bool repeating_bias,
                                size_t tile_I, size_t tile_J, size_t tile_K,
                                bool transpose_A, bool transpose_B,
                                bool full_C, bool low_D,
                                uint8_t weightA,
                                potal_tiled_matmul_type_t tiled_matmul_type,
                                const potal_scale_operand & scales, potal_ws_session & session) {
    potal_ws_report & r = session.report;
    if (r.fatal) {
        return fail(r, r.status != POTAL_OK ? r.status : POTAL_ERR_BUSY, "session_disabled", true);
    }
    potal_status status = check_request(D, A_scale_factor, B_scale_factor, D_scale_factor, act, scale, bert_scale,
                                        repeating_bias, transpose_A, transpose_B, full_C, low_D, tiled_matmul_type);
    if (status != POTAL_OK) {
        return fail(r, status, "request", false);
    }
    uint32_t passes_per_block = 0;
    if (session.ops == nullptr || A == nullptr || B == nullptr || C == nullptr || dim_I == 0 || dim_J == 0 ||
        dim_K == 0 || stride_A < dim_K || stride_B < dim_J || stride_C < dim_J ||
        check_scales(dim_J, dim_K, scales, &passes_per_block) != POTAL_OK) {
        return fail(r, POTAL_ERR_INVALID_ARGUMENT, "request", false);
    }

    // GEMMINI_ASSERTIONS of tiled_matmul, returned as statuses, plus the PoTal limits.
    const size_t dim_I_padded = padded(dim_I);
    const size_t dim_J_padded = padded(dim_J);
    const size_t dim_K_padded = padded(dim_K);
    if (tile_I == 0 || tile_J == 0 || tile_K == 0 || tile_I * DIM > dim_I_padded || tile_J * DIM > dim_J_padded ||
        tile_K * DIM > dim_K_padded || tile_I > 65535 || tile_J > 65535 || tile_K > 65535 ||
        tile_K % passes_per_block != 0) {
        return fail(r, POTAL_ERR_INVALID_ARGUMENT, "tiles", false);
    }
    if (!fits(tile_I, tile_J, tile_K, passes_per_block)) {
        return fail(r, POTAL_ERR_CAPACITY, "tiles", false);
    }

    return potal_tiled_matmul_outer(dim_I, dim_J, dim_K,
                                    A, B, D, C,
                                    stride_A, stride_B, stride_D, stride_C,
                                    A_scale_factor, B_scale_factor, D_scale_factor,
                                    tile_I, tile_J, tile_K,
                                    act, scale, bert_scale, repeating_bias,
                                    transpose_A, transpose_B,
                                    full_C, low_D,
                                    weightA,
                                    (int) tiled_matmul_type,
                                    scales, session);
}

potal_status potal_tiled_matmul_outer(size_t dim_I, size_t dim_J, size_t dim_K,
                                      const int8_t * A, const int8_t * B,
                                      const void * D, int32_t * C,
                                      size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
                                      potal_scale_t A_scale_factor, potal_scale_t B_scale_factor,
                                      potal_scale_acc_t D_scale_factor,
                                      size_t tile_I, size_t tile_J, size_t tile_K,
                                      int act, potal_acc_scale_t scale, potal_acc_scale_t bert_scale,
                                      bool repeating_bias,
                                      bool a_transpose, bool b_transpose,
                                      bool full_C, bool low_D,
                                      uint8_t weightA,
                                      int dataflow,
                                      const potal_scale_operand & scales, potal_ws_session & session) {
    (void) scale;
    (void) bert_scale;
    (void) weightA; // unused by gemmini.h tiled_matmul_outer as well

    const size_t dim_I_padded = padded(dim_I);
    const size_t dim_J_padded = padded(dim_J);
    const size_t dim_K_padded = padded(dim_K);

    const size_t I0 = dim_I_padded / (tile_I * DIM) + (dim_I_padded % (tile_I * DIM) != 0);
    const size_t J0 = dim_J_padded / (tile_J * DIM) + (dim_J_padded % (tile_J * DIM) != 0);
    const size_t K0 = dim_K_padded / (tile_K * DIM) + (dim_K_padded % (tile_K * DIM) != 0);

    // These lines here are supposed to help us deal with when the dimensions of
    // the systolic array aren't divisible by the tiling factors
    const size_t last_I = dim_I_padded % (tile_I * DIM) == 0 ? tile_I : (dim_I_padded / DIM) % tile_I;
    const size_t last_J = dim_J_padded % (tile_J * DIM) == 0 ? tile_J : (dim_J_padded / DIM) % tile_J;
    const size_t last_K = dim_K_padded % (tile_K * DIM) == 0 ? tile_K : (dim_K_padded / DIM) % tile_K;

    // These lines are supposed to figure out how much padding the hardware is
    // supposed to add for the final tile
    const size_t padding_I = dim_I_padded - dim_I;
    const size_t padding_J = dim_J_padded - dim_J;
    const size_t padding_K = dim_K_padded - dim_K;

    const bool no_bias = D == nullptr;

    if (no_bias) {
        D = (const void *) 1; // Dummy address which isn't NULL
    }

    // CONFIG_EX/ST/LD (gemmini_extended_config_*) are part of every PoTal
    // invocation's command sequence (potal.c encode_program), so nothing is
    // configured here.

    // Operand reuse (a_spad_id/b_spad_id with a NULL tile origin) is not
    // implemented: every invocation loads its A and B, as gemmini.h with
    // GEMMINI_DISABLE_WS_REUSE=1.
    const int a_spad_id = 0;
    const int b_spad_id = 0;

    const uint32_t ppb = scales.block_k / DIM;

    for (size_t i0 = 0; i0 < I0; i0++) {
        for (size_t j0 = 0; j0 < J0; j0++) {
            for (size_t k0 = 0; k0 < K0; k0++) {
                // gemmini.h passes the bias only with the first K chunk and C
                // only with the last; no bias: the dummy pointer marks k0 == 0.
                const void * pre = k0 != 0 ? nullptr : D;

                int32_t * out = k0 == K0 - 1 ? C + (i0 * tile_I * DIM * stride_C + j0 * tile_J * DIM) : nullptr;

                const size_t I = i0 < I0 - 1 ? tile_I : last_I;
                const size_t J = j0 < J0 - 1 ? tile_J : last_J;
                const size_t K = k0 < K0 - 1 ? tile_K : last_K;

                const size_t pad_I = i0 == I0 - 1 ? padding_I : 0;
                const size_t pad_J = j0 == J0 - 1 ? padding_J : 0;
                const size_t pad_K = k0 == K0 - 1 ? padding_K : 0;

                const int8_t * a = A + i0 * tile_I * DIM * stride_A + k0 * tile_K * DIM;
                const int8_t * b = B + k0 * tile_K * DIM * stride_B + j0 * tile_J * DIM;

                // PoTal: carriers of this chunk's first scale block and J tile group.
                const uint32_t * carriers =
                    scales.carriers + (k0 * tile_K / ppb) * scales.stride + j0 * tile_J * DIM;

                const potal_status status = potal_sp_tiled_matmul_ws(
                    a, b, pre, out,
                    A_scale_factor, B_scale_factor, D_scale_factor,
                    I, J, K,
                    pad_I, pad_J, pad_K,
                    stride_A, stride_B, stride_D, stride_C,
                    a_transpose, b_transpose,
                    full_C, low_D,
                    no_bias, repeating_bias,
                    act, a_spad_id, b_spad_id,
                    carriers, scales.stride, ppb,
                    session);
                if (status != POTAL_OK) {
                    return status; // fail-stop: no later chunk is issued
                }
            }
        }
    }

    // gemmini_fence(): every invocation was fenced by potal_sp_tiled_matmul_ws.
    (void) dataflow;
    return POTAL_OK;
}

potal_status potal_sp_tiled_matmul_ws(const int8_t * A, const int8_t * B,
                                      const void * D, int32_t * C,
                                      potal_scale_t A_scale_factor, potal_scale_t B_scale_factor,
                                      potal_scale_acc_t D_scale_factor,
                                      size_t I, size_t J, size_t K, size_t pad_I, size_t pad_J, size_t pad_K,
                                      size_t A_row_stride, size_t B_row_stride, size_t D_row_stride,
                                      size_t C_row_stride,
                                      bool a_transpose, bool b_transpose,
                                      bool full_C, bool low_D,
                                      bool no_bias, bool repeating_bias,
                                      int act,
                                      int a_spad_id, int b_spad_id,
                                      const uint32_t * carriers, size_t carrier_stride, uint32_t passes_per_block,
                                      potal_ws_session & session) {
    (void) D_row_stride;
    potal_ws_report & r = session.report;
    if (r.fatal) {
        return fail(r, r.status != POTAL_OK ? r.status : POTAL_ERR_BUSY, "session_disabled", true);
    }
    if (!no_bias || repeating_bias || a_transpose || b_transpose || !full_C || low_D ||
        act != POTAL_NO_ACTIVATION || a_spad_id != 0 || b_spad_id != 0 ||
        A_scale_factor != POTAL_MVIN_SCALE_IDENTITY || B_scale_factor != POTAL_MVIN_SCALE_IDENTITY ||
        D_scale_factor != 0) {
        return fail(r, POTAL_ERR_UNSUPPORTED, "sp_request", false);
    }
    if (session.ops == nullptr || carriers == nullptr || passes_per_block == 0 || K % passes_per_block != 0 ||
        I > 0xFFFF || J > 0xFFFF || K > 0xFFFF) {
        return fail(r, POTAL_ERR_INVALID_ARGUMENT, "sp_request", false);
    }

    // gemmini.h: ex_accumulate = !no_bias || D == NULL (D is NULL for k0 > 0).
    const bool     accumulate = !no_bias || D == nullptr;
    const uint32_t blocks     = (uint32_t) (K / passes_per_block);
    const uint32_t slots      = blocks * (uint32_t) J;
    const uint32_t base       = options::scale_window_base;
    const uint32_t generation = session.generation % POTAL_GENERATION_MAX + 1; // 1..255

    potal_loop_ws loop = {};
    loop.a                = A;
    loop.b                = B;
    loop.c                = C;
    loop.lda              = A_row_stride;
    loop.ldb              = B_row_stride;
    loop.ldc              = C != nullptr ? C_row_stride : 0;
    loop.i_tiles          = (uint32_t) I;
    loop.j_tiles          = (uint32_t) J;
    loop.k_tiles          = (uint32_t) K;
    loop.pad_i            = (uint32_t) pad_I;
    loop.pad_j            = (uint32_t) pad_J;
    loop.pad_k            = (uint32_t) pad_K;
    loop.scale_base       = base;
    loop.scale_generation = generation;
    loop.passes_per_block = passes_per_block;
    loop.flags            = accumulate ? POTAL_GEMM_ACCUMULATE : 0;

    // Software checks first: nothing reaches the accelerator when they fail.
    potal_status status = potal_loop_ws_check(&loop);
    if (status == POTAL_OK && slots > options::scale_window_slots) {
        status = POTAL_ERR_CAPACITY;
    }
    const auto valid_columns = [&](size_t j) { return (uint32_t) (DIM - (j == J - 1 ? pad_J : 0)); };
    for (uint32_t b = 0; b < blocks && status == POTAL_OK; ++b) {
        for (size_t j = 0; j < J && status == POTAL_OK; ++j) {
            for (uint32_t col = 0; col < valid_columns(j); ++col) {
                if (!potal_carrier_is_valid(carriers[b * carrier_stride + j * DIM + col])) {
                    status = POTAL_ERR_INVALID_ARGUMENT;
                    break;
                }
            }
        }
    }
    if (status != POTAL_OK) {
        return fail(r, status, "sp_check", false);
    }

    // The validated single-tile request goes through the validated API.
    potal_gemm_ws gemm    = {};
    gemm.a                = A;
    gemm.b                = B;
    gemm.c                = C;
    gemm.lda              = A_row_stride;
    gemm.ldb              = B_row_stride;
    gemm.ldc              = C_row_stride;
    gemm.m                = (uint32_t) (I * DIM - pad_I);
    gemm.n                = (uint32_t) (J * DIM - pad_J);
    gemm.k                = (uint32_t) (K * DIM - pad_K);
    gemm.scale_base       = base;
    gemm.scale_generation = generation;
    gemm.passes_per_block = passes_per_block;
    gemm.flags            = 0;
    const bool validated  = !accumulate && C != nullptr && pad_I == 0 && pad_J == 0 && pad_K == 0 &&
                           potal_gemm_ws_check(&gemm) == POTAL_OK;

    session.generation = generation;
    r.last_generation  = generation;
    r.max_scale_slots  = std::max(r.max_scale_slots, slots);
    const potal_ws_ops & ops = *session.ops;

    // Release, in write order, the first `count` entries written by this invocation.
    const auto release = [&](uint64_t count) -> potal_status {
        uint64_t     done = 0;
        potal_status s    = POTAL_OK;
        for (uint32_t b = 0; b < blocks && done < count; ++b) {
            for (size_t j = 0; j < J && done < count; ++j) {
                for (uint32_t col = 0; col < valid_columns(j) && done < count; ++col, ++done) {
                    const potal_status rs = ops.scale_release(ops.ctx, base + b * (uint32_t) J + (uint32_t) j,
                                                              generation, col);
                    r.scale_releases += rs == POTAL_OK ? 1 : 0;
                    s = s == POTAL_OK ? rs : s;
                }
            }
        }
        return s;
    };

    // 1. SCALE_WRITE: slot = base + block * J + j, one carrier per valid column.
    auto     t       = clock_type::now();
    uint64_t written = 0;
    for (uint32_t b = 0; b < blocks && status == POTAL_OK; ++b) {
        for (size_t j = 0; j < J && status == POTAL_OK; ++j) {
            for (uint32_t col = 0; col < valid_columns(j); ++col) {
                status = ops.scale_write(ops.ctx, base + b * (uint32_t) J + (uint32_t) j, generation, col,
                                         carriers[b * carrier_stride + j * DIM + col]);
                if (status != POTAL_OK) {
                    break;
                }
                ++written;
            }
        }
    }
    r.scale_writes += written;
    r.scale_s += seconds_since(t);
    if (status != POTAL_OK) {
        (void) release(written); // nothing issued: the entries are idle
        return fail(r, status, "scale_write", true);
    }

    // 2. LOOP_WS.
    t         = clock_type::now();
    status    = validated ? ops.gemm_ws_issue(ops.ctx, &gemm) : ops.loop_ws_issue(ops.ctx, &loop);
    r.issue_s += seconds_since(t);
    if (status != POTAL_OK) {
        (void) release(written); // refused before any GEMM command was sent
        return fail(r, status, "issue", true);
    }
    r.invocations++;
    r.validated_gemms += validated ? 1 : 0;

    // 3. POTAL_FENCE: the invocation and its C store drained.
    t         = clock_type::now();
    status    = ops.fence(ops.ctx);
    r.fence_s += seconds_since(t);
    if (status != POTAL_OK) {
        // The GEMM may still run: leave its buffers and scale entries alone.
        return fail(r, status, "fence", true);
    }
    r.fences++;

    // 4. SCALE_RELEASE every entry, so the next invocation can write the window.
    t      = clock_type::now();
    status = release(written);
    r.scale_s += seconds_since(t);
    if (status != POTAL_OK) {
        return fail(r, status, "scale_release", true);
    }
    return POTAL_OK;
}

} // namespace ggml::potal
