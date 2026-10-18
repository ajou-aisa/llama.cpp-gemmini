#include "stripe-api.hpp"

#include <cstdint>
#include <new>

namespace {

using namespace ggml::potal;

constexpr uint64_t DIM = POTAL_HW_DIM;

// LoopMatmul.castDramOffset keeps 32 bits of every tile-origin DRAM offset
// (potal.c); a whole operand within it keeps every tile origin within it.
constexpr uint64_t DRAM_OFFSET_LIMIT = (uint64_t) 1 << 32;

// Bytes spanned by `rows` rows of `cols` elements with row stride `ld`; 0 on overflow.
uint64_t extent(uint64_t rows, uint64_t cols, uint64_t ld, uint64_t elem_size) {
    const uint64_t max = UINT64_MAX / elem_size;
    if (rows == 0 || cols == 0 || ld < cols || rows - 1 > (max - cols) / ld) {
        return 0;
    }
    return ((rows - 1) * ld + cols) * elem_size;
}

bool operand_ok(const void * p, uint64_t bytes, uint64_t align) {
    const uintptr_t addr = (uintptr_t) p;
    return p != nullptr && addr % align == 0 && bytes != 0 && bytes <= DRAM_OFFSET_LIMIT &&
           (uint64_t) addr <= UINT64_MAX - bytes;
}

bool overlap(const void * a, uint64_t a_bytes, const void * b, uint64_t b_bytes) {
    const uint64_t x = (uintptr_t) a;
    const uint64_t y = (uintptr_t) b;
    return x < y + b_bytes && y < x + a_bytes;
}

// Everything potal_tiled_matmul() and the invocations below it would check,
// on the whole request, so a rejected descriptor never issues a command.
potal_status check(const potal_stripe_gemm_desc & d, potal_tile_plan & plan, const char *& stage) {
    stage = "request";
    if (d.work_kind > POTAL_STRIPE_WORK_RESIDUAL || d.m == 0 || d.n == 0 || d.k == 0 || d.a == nullptr ||
        d.b == nullptr || d.c == nullptr || d.carriers == nullptr || d.lda < d.k || d.ldb < d.n || d.ldc < d.n ||
        d.carrier_stride < d.n || d.ldc > UINT32_MAX / sizeof(int32_t) || d.scale_block_k == 0 ||
        d.scale_block_k % DIM != 0 || d.k % d.scale_block_k != 0) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }

    stage = "tiles";
    const potal_status s = potal_frozen_plan(d.m, d.n, d.k, d.tile_i, d.tile_j, d.tile_k,
                                             (uint32_t) (d.scale_block_k / DIM), &plan);
    if (s != POTAL_OK) {
        return s;
    }

    stage = "operands";
    const uint64_t a_bytes = extent(d.m, d.k, d.lda, sizeof(int8_t));
    const uint64_t b_bytes = extent(d.k, d.n, d.ldb, sizeof(int8_t));
    const uint64_t c_bytes = extent(d.m, d.n, d.ldc, sizeof(int32_t));
    if (!operand_ok(d.a, a_bytes, DIM * sizeof(int8_t)) || !operand_ok(d.b, b_bytes, DIM * sizeof(int8_t)) ||
        !operand_ok(d.c, c_bytes, DIM * sizeof(int32_t)) || overlap(d.c, c_bytes, d.a, a_bytes) ||
        overlap(d.c, c_bytes, d.b, b_bytes)) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }

    stage = "carriers";
    const size_t blocks = d.k / d.scale_block_k;
    if (blocks > SIZE_MAX / d.carrier_stride) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    for (size_t blk = 0; blk < blocks; ++blk) {
        for (uint32_t col = 0; col < d.n; ++col) {
            if (!potal_carrier_is_valid(d.carriers[blk * d.carrier_stride + col])) {
                return POTAL_ERR_INVALID_ARGUMENT;
            }
        }
    }
    stage = nullptr;
    return POTAL_OK;
}

} // namespace

potal_status potal_stripe_session_create(potal_stripe_session ** out) {
    if (out == nullptr) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    *out = nullptr;
    potal_stripe_session * s = new (std::nothrow) potal_stripe_session{ { &potal_hw_ops(), 0, {} } };
    if (s == nullptr) {
        return POTAL_ERR_CAPACITY; // host memory for the session
    }
    *out = s;
    return POTAL_OK;
}

void potal_stripe_session_destroy(potal_stripe_session * session) {
    delete session;
}

potal_status potal_tiled_matmul_stripe(potal_stripe_session * session, const potal_stripe_gemm_desc * desc,
                                       potal_stripe_gemm_report * report) {
    if (report == nullptr) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    *report = potal_stripe_gemm_report{};
    const auto reject = [report](potal_status s, const char * stage, bool fatal) {
        report->status = s;
        report->stage  = stage;
        report->fatal  = fatal ? 1 : 0;
        return s;
    };
    if (session == nullptr || desc == nullptr || session->ws.ops == nullptr) {
        return reject(POTAL_ERR_INVALID_ARGUMENT, "request", false);
    }
    potal_ws_session & ws = session->ws;
    if (ws.report.fatal) {
        return reject(ws.report.status != POTAL_OK ? ws.report.status : POTAL_ERR_BUSY, "session_disabled", true);
    }

    potal_tile_plan    plan;
    const char *       stage  = nullptr;
    const potal_status status = check(*desc, plan, stage);
    if (status != POTAL_OK) {
        return reject(status, stage, false);
    }
    report->tile_i = (uint32_t) plan.tile_I;
    report->tile_j = (uint32_t) plan.tile_J;
    report->tile_k = (uint32_t) plan.tile_K;
    report->i0     = (uint32_t) plan.I0;
    report->j0     = (uint32_t) plan.J0;
    report->k0     = (uint32_t) plan.K0;
    report->pad_i  = (uint32_t) plan.pad_I;
    report->pad_j  = (uint32_t) plan.pad_J;
    report->pad_k  = (uint32_t) plan.pad_K;

    // tiled_matmul with the frozen tiles: no tile search, the ACC carries the
    // K chunks of every output tile, every invocation is fenced and released.
    const potal_ws_report before     = ws.report;
    ws.report.status                 = POTAL_OK;
    ws.report.stage                  = nullptr;
    const potal_scale_operand scales = { desc->carriers, desc->carrier_stride, desc->scale_block_k };
    report->status = potal_tiled_matmul(desc->m, desc->n, desc->k,
                                        desc->a, desc->b, nullptr, desc->c,
                                        desc->lda, desc->ldb, 0, desc->ldc,
                                        POTAL_MVIN_SCALE_IDENTITY, POTAL_MVIN_SCALE_IDENTITY, 0,
                                        POTAL_NO_ACTIVATION, POTAL_ACC_SCALE_IDENTITY, 0,
                                        /* repeating_bias */ false,
                                        plan.tile_I, plan.tile_J, plan.tile_K,
                                        /* transpose_A */ false, /* transpose_B */ false,
                                        /* full_C */ true, /* low_D */ false,
                                        /* weightA */ 0,
                                        POTAL_WS,
                                        scales, ws);
    const potal_ws_report w = work_since(ws.report, before);
    report->stage           = report->status == POTAL_OK ? nullptr : ws.report.stage;
    report->fatal           = ws.report.fatal ? 1 : 0;
    report->loop_ws         = w.invocations;
    report->validated_gemms = w.validated_gemms;
    report->fences          = w.fences;
    report->scale_writes    = w.scale_writes;
    report->scale_releases  = w.scale_releases;
    report->last_generation = w.generations != 0 ? ws.report.last_generation : 0;
    report->scale_s         = w.scale_s;
    report->issue_s         = w.issue_s;
    report->fence_s         = w.fence_s;
    return report->status;
}
