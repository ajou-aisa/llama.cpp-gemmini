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

potal_status refuse(potal_stripe_gemm_report & report, potal_status s, const char * stage, bool fatal) {
    report.status = s;
    report.stage  = stage;
    report.fatal  = fatal ? 1 : 0;
    return s;
}

// The checks of both entry points before their first command: a session that
// may still issue work, then the preflight. On success the report holds the
// geometry that will run.
potal_status admit(potal_stripe_session & session, const potal_stripe_gemm_desc & desc, prepared_stripe_gemm & g,
                   potal_stripe_gemm_report & report) {
    const potal_ws_session & ws = session.ws;
    if (ws.ops == nullptr) {
        return refuse(report, POTAL_ERR_INVALID_ARGUMENT, "request", false);
    }
    if (ws.report.fatal) {
        return refuse(report, ws.report.status != POTAL_OK ? ws.report.status : POTAL_ERR_BUSY, "session_disabled",
                      true);
    }
    const char *       stage  = nullptr;
    const potal_status status = prepare_stripe_gemm(desc, g, &stage);
    if (status != POTAL_OK) {
        return refuse(report, status, stage, false);
    }
    report.tile_i = (uint32_t) g.plan.tile_I;
    report.tile_j = (uint32_t) g.plan.tile_J;
    report.tile_k = (uint32_t) g.plan.tile_K;
    report.i0     = (uint32_t) g.plan.I0;
    report.j0     = (uint32_t) g.plan.J0;
    report.k0     = (uint32_t) g.plan.K0;
    report.pad_i  = (uint32_t) g.plan.pad_I;
    report.pad_j  = (uint32_t) g.plan.pad_J;
    report.pad_k  = (uint32_t) g.plan.pad_K;
    return POTAL_OK;
}

// tiled_matmul with the frozen tiles: no tile search, the ACC carries the K
// chunks of every output tile, every invocation is fenced and released.
potal_status run_prepared(potal_ws_session & ws, const prepared_stripe_gemm & g, potal_stripe_gemm_report & report) {
    const potal_stripe_gemm_desc & d      = g.desc;
    const potal_ws_report          before = ws.report;
    ws.report.status                      = POTAL_OK;
    ws.report.stage                       = nullptr;
    const potal_scale_operand scales      = { d.carriers, d.carrier_stride, d.scale_block_k };
    report.status = potal_tiled_matmul(d.m, d.n, d.k,
                                       d.a, d.b, nullptr, d.c,
                                       d.lda, d.ldb, 0, d.ldc,
                                       POTAL_MVIN_SCALE_IDENTITY, POTAL_MVIN_SCALE_IDENTITY, 0,
                                       POTAL_NO_ACTIVATION, POTAL_ACC_SCALE_IDENTITY, 0,
                                       /* repeating_bias */ false,
                                       g.plan.tile_I, g.plan.tile_J, g.plan.tile_K,
                                       /* transpose_A */ false, /* transpose_B */ false,
                                       /* full_C */ true, /* low_D */ false,
                                       /* weightA */ 0,
                                       POTAL_WS,
                                       scales, ws);
    const potal_ws_report w = work_since(ws.report, before);
    report.stage           = report.status == POTAL_OK ? nullptr : ws.report.stage;
    report.fatal           = ws.report.fatal ? 1 : 0;
    report.loop_ws         = w.invocations;
    report.validated_gemms = w.validated_gemms;
    report.fences          = w.fences;
    report.scale_writes    = w.scale_writes;
    report.scale_releases  = w.scale_releases;
    report.last_generation = w.generations != 0 ? ws.report.last_generation : 0;
    report.scale_s         = w.scale_s;
    report.issue_s         = w.issue_s;
    report.fence_s         = w.fence_s;
    return report.status;
}

} // namespace

namespace ggml::potal {

potal_status prepare_stripe_gemm(const potal_stripe_gemm_desc & d, prepared_stripe_gemm & out, const char ** stage) {
    const char *  unused = nullptr;
    const char *& at     = stage != nullptr ? *stage : unused;

    at = "request";
    if (d.work_kind > POTAL_STRIPE_WORK_RESIDUAL || d.m == 0 || d.n == 0 || d.k == 0 || d.a == nullptr ||
        d.b == nullptr || d.c == nullptr || d.carriers == nullptr || d.lda < d.k || d.ldb < d.n || d.ldc < d.n ||
        d.carrier_stride < d.n || d.ldc > UINT32_MAX / sizeof(int32_t) || d.scale_block_k == 0 ||
        d.scale_block_k % DIM != 0 || d.k % d.scale_block_k != 0) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }

    at = "tiles";
    potal_tile_plan    plan;
    const potal_status s = potal_frozen_plan(d.m, d.n, d.k, d.tile_i, d.tile_j, d.tile_k,
                                             (uint32_t) (d.scale_block_k / DIM), &plan);
    if (s != POTAL_OK) {
        return s;
    }

    at = "operands";
    const uint64_t a_bytes = extent(d.m, d.k, d.lda, sizeof(int8_t));
    const uint64_t b_bytes = extent(d.k, d.n, d.ldb, sizeof(int8_t));
    const uint64_t c_bytes = extent(d.m, d.n, d.ldc, sizeof(int32_t));
    if (!operand_ok(d.a, a_bytes, DIM * sizeof(int8_t)) || !operand_ok(d.b, b_bytes, DIM * sizeof(int8_t)) ||
        !operand_ok(d.c, c_bytes, DIM * sizeof(int32_t)) || overlap(d.c, c_bytes, d.a, a_bytes) ||
        overlap(d.c, c_bytes, d.b, b_bytes)) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }

    at = "carriers";
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
    at       = nullptr;
    out.desc = d;
    out.plan = plan;
    return POTAL_OK;
}

planned_result run_planned_stripe_gemm(potal_stripe_session & session, const potal_stripe_gemm_desc & desc) {
    planned_result       r;
    prepared_stripe_gemm g;
    r.status = admit(session, desc, g, r.report);
    if (r.status != POTAL_OK) {
        r.fatal = r.report.fatal != 0;
        r.stage = r.fatal ? r.report.stage : "preflight";
        return r;
    }
    potal_ws_session &   ws  = session.ws;
    const potal_ws_ops & ops = *ws.ops;
    r.status                 = ops.tlb_flush(ops.ctx);
    if (r.status == POTAL_OK) {
        r.status = ops.fence(ops.ctx);
    }
    if (r.status != POTAL_OK) {
        ws.report.status = r.status;
        ws.report.stage  = "tlb_flush";
        ws.report.fatal  = true;
        r.fatal          = true;
        r.stage          = "tlb_flush";
        (void) refuse(r.report, r.status, "tlb_flush", true);
        return r;
    }
    ws.report.tlb_flushes++;
    r.tlb_flushes = 1;
    r.status      = run_prepared(ws, g, r.report);
    r.stage       = r.report.stage;
    r.fatal       = r.report.fatal != 0;
    r.ok          = r.status == POTAL_OK;
    return r;
}

} // namespace ggml::potal

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
    if (session == nullptr || desc == nullptr) {
        return refuse(*report, POTAL_ERR_INVALID_ARGUMENT, "request", false);
    }
    prepared_stripe_gemm g;
    const potal_status   status = admit(*session, *desc, g, *report);
    if (status != POTAL_OK) {
        return status;
    }
    return run_prepared(session->ws, g, *report);
}
