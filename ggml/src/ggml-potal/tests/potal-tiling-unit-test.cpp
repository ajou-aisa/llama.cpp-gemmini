// Hardware-free W4-5 tests of the PoTal tiled path (host and target):
//
//   A  tile selection: potal_tile_select() vs the gemmini.h tiled_matmul_auto
//      policy (verbatim port below), PoTal limits, capacity reporting
//   B  tiled execution on an emulator of the RTL LOOP_WS/SCU/Sat32 semantics
//      (LoopMatmul, PoTalSCU, PoTalScaleMemory, AccPipe) vs the independent
//      integer golden reference_gemm_i32(): multi-I/J/K tiles, K chunks,
//      padding, zero carriers, saturation, scale slot release, fail-stop
//   C  Q8_HP1 validation and unpacking (weight reader copied from Gemmini)
//   D  per-tensor activation quantization (TENSOR quantizer copied from Gemmini)
//   E  F32 reconstruction, compensation OFF, end-to-end against the golden
//   F  GGML contract: supports_op, no CPU fallback for Q8_HP1, no fake success
//
// The emulator is a test double for the accelerator; passing here proves the
// software contract, not the hardware. Hardware: potal-hp1-smoke on FireSim.

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"
#include "ggml-potal.h"
#include "matmul/dense.hpp"
#include "quants/act/quantize.hpp"
#include "quants/common/hp1_scu.hpp"
#include "matmul/options.hpp"
#include "matmul/tiling.hpp"
#include "ops.hpp"
#include "potal.h"
#include "potal_hw_params.h"

#include "ggml-quants.h" // block_q8_hp1 (ggml-common.h), Q8_HP1 reference quant/dequant

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

using namespace ggml::potal;
namespace hp1 = ggml::potal::quants::weight::hp1;

static_assert(!options::compensation, "W4-5: the PoTal path has no error compensation");

namespace {

int failures = 0;

void check(const std::string & name, bool ok) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", name.c_str());
    failures += ok ? 0 : 1;
}

constexpr size_t DIM = POTAL_HW_DIM;

// ---- A: gemmini.h tiled_matmul_auto tile choice (WS), verbatim policy ------

size_t gm_spad(size_t I, size_t J, size_t K) { return (I * K + K * J) * DIM; }
size_t gm_acc(size_t I, size_t J) { return (I * J) * DIM; }

void gemmini_reference_tiles(size_t dim_I, size_t dim_J, size_t dim_K, size_t & tile_I, size_t & tile_J,
                             size_t & tile_K) {
    const size_t db_partition_rows    = ((size_t) POTAL_HW_BANK_NUM * POTAL_HW_BANK_ROWS / 2) / 2;
    const size_t db_mats_in_partition = db_partition_rows / DIM;
    const size_t db_mats_in_acc       = ((size_t) POTAL_HW_ACC_ROWS / 2) / DIM;
    const size_t db_max_tile_i_j      = (size_t) std::sqrt((double) db_mats_in_acc);
    const size_t db_max_tile_k        = db_mats_in_partition / db_max_tile_i_j;
    const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
    const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
    const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;
    const size_t max_spad_rows = (size_t) POTAL_HW_BANK_NUM * POTAL_HW_BANK_ROWS / 2;
    const size_t max_acc_rows  = (size_t) POTAL_HW_ACC_ROWS / 2;
    tile_I = dim_I_padded / DIM < db_max_tile_i_j ? dim_I_padded / DIM : db_max_tile_i_j;
    tile_J = dim_J_padded / DIM < db_max_tile_i_j ? dim_J_padded / DIM : db_max_tile_i_j;
    tile_K = dim_K_padded / DIM < db_max_tile_k ? dim_K_padded / DIM : db_max_tile_k;
    while (true) {
        bool increased = false;
        if (gm_spad(tile_I, tile_J + 1, tile_K) <= max_spad_rows && gm_acc(tile_I, tile_J + 1) <= max_acc_rows &&
            (tile_J + 1) * DIM <= dim_J_padded) {
            tile_J++;
            increased = true;
        }
        if (gm_spad(tile_I + 1, tile_J, tile_K) <= max_spad_rows && gm_acc(tile_I + 1, tile_J) <= max_acc_rows &&
            (tile_I + 1) * DIM <= dim_I_padded) {
            tile_I++;
            increased = true;
        }
        if (gm_spad(tile_I, tile_J, tile_K + 1) <= max_spad_rows && (tile_K + 1) * DIM <= dim_K_padded) {
            tile_K++;
            increased = true;
        }
        if (!increased) {
            break;
        }
    }
}

bool potal_limits_hold(size_t I, size_t J, size_t K, size_t ppb) {
    return gm_spad(I, J, K) <= (size_t) POTAL_HW_BANK_NUM * POTAL_HW_BANK_ROWS / 2 &&
           gm_acc(I, J) <= (size_t) POTAL_HW_ACC_ROWS / 2 && I * J <= POTAL_WORK_IDS_PER_LOOP && K % ppb == 0 &&
           (K / ppb) * J <= options::scale_window_slots;
}

void test_tile_select() {
    const size_t shapes[][3] = {
        { 16, 16, 16 }, { 16, 16, 32 }, { 1, 768, 768 }, { 7, 50, 96 },  { 128, 768, 3072 },
        { 1, 50257, 768 }, { 32, 2304, 768 }, { 5, 40, 64 }, { 300, 17, 2048 }, { 1, 16, 4096 },
    };
    for (const auto & sh : shapes) {
        for (uint32_t ppb : { 1u, 2u }) {
            if ((sh[2] + DIM - 1) / DIM % ppb != 0) {
                continue;
            }
            char name[160];
            std::snprintf(name, sizeof(name), "A tiles M=%zu N=%zu K=%zu ppb=%u", sh[0], sh[1], sh[2], ppb);
            potal_tile_plan plan;
            const potal_status st = potal_tile_select(sh[0], sh[1], sh[2], ppb, &plan);
            size_t gi, gj, gk;
            gemmini_reference_tiles(sh[0], sh[1], sh[2], gi, gj, gk);
            const bool ref_fits = potal_limits_hold(gi, gj, gk, ppb);
            bool       ok       = st == POTAL_OK && potal_limits_hold(plan.tile_I, plan.tile_J, plan.tile_K, ppb);
            if (ref_fits && ppb == 1) {
                // Where the PoTal limits do not bind, the choice is gemmini.h's
                // (ppb 1: same start and K step; ppb > 1 steps K by whole blocks).
                ok = ok && plan.tile_I == gi && plan.tile_J == gj && plan.tile_K == gk;
            }
            std::printf("     tiles=(%zu,%zu,%zu) gemmini=(%zu,%zu,%zu)%s iter=(%zu,%zu,%zu) slots=%u spad=%zu/%zu "
                        "acc=%zu/%zu\n",
                        plan.tile_I, plan.tile_J, plan.tile_K, gi, gj, gk, ref_fits ? "" : " (PoTal-limited)",
                        plan.I0, plan.J0, plan.K0, plan.scale_slots, plan.spad_rows, plan.max_spad_rows,
                        plan.acc_rows, plan.max_acc_rows);
            check(name, ok);
        }
    }
    potal_tile_plan plan;
    check("A K not whole scale blocks rejected", potal_tile_select(4, 16, 48, 2, &plan) == POTAL_ERR_INVALID_ARGUMENT);
    check("A zero dimension rejected", potal_tile_select(0, 16, 32, 2, &plan) == POTAL_ERR_INVALID_ARGUMENT);
}

// ---- B: emulator of the RTL LOOP_WS + SCU + Sat32 semantics ---------------

struct emulator {
    struct entry {
        uint32_t generation;
        uint32_t carrier;
    };
    std::map<std::pair<uint32_t, uint32_t>, entry> scales; // (slot, column) -> valid entry
    std::vector<int32_t> acc[2];
    int                  half          = 0;
    bool                 in_flight     = false;
    bool                 scale_miss    = false; // hardware would hang
    int                  fail_write_at = -1;    // inject a SCALE_WRITE failure on this write
    bool                 fail_fence    = false;
    uint64_t             writes = 0, releases = 0, loops = 0, gemms = 0, fences = 0;

    emulator() {
        acc[0].assign((size_t) POTAL_HW_ACC_ROWS / 2 * DIM, 0);
        acc[1].assign((size_t) POTAL_HW_ACC_ROWS / 2 * DIM, 0);
    }

    // PoTalSCU.scala: left shift with Sat32, zero carrier -> 0.
    static int32_t scu(int32_t partial, uint32_t carrier) {
        if (carrier == POTAL_CARRIER_ZERO || partial == 0) {
            return 0;
        }
        if (carrier >= 32) {
            return partial < 0 ? INT32_MIN : INT32_MAX;
        }
        const int64_t pos = (int64_t) INT32_MAX >> carrier;
        const int64_t neg = (int64_t) INT32_MIN >> carrier;
        if (partial > pos) {
            return INT32_MAX;
        }
        if (partial < neg) {
            return INT32_MIN;
        }
        return (int32_t) ((int64_t) partial << carrier);
    }

    // AccumulatorMem saturateInt32.
    static int32_t sat_add(int32_t a, int32_t b) {
        const int64_t s = (int64_t) a + b;
        return s > INT32_MAX ? INT32_MAX : s < INT32_MIN ? INT32_MIN : (int32_t) s;
    }

    potal_status write(uint32_t slot, uint32_t gen, uint32_t col, uint32_t carrier) {
        if ((int64_t) writes == fail_write_at) {
            return POTAL_HW_INVALID_SCALE;
        }
        if (slot >= POTAL_SCALE_ENTRIES || col >= DIM || gen > POTAL_GENERATION_MAX ||
            !potal_carrier_is_valid(carrier)) {
            return POTAL_ERR_INVALID_ARGUMENT;
        }
        if (scales.count({ slot, col })) {
            return POTAL_HW_SCALE_CONFLICT;
        }
        scales[{ slot, col }] = { gen, carrier };
        writes++;
        return POTAL_OK;
    }

    potal_status release(uint32_t slot, uint32_t gen, uint32_t col) {
        auto it = scales.find({ slot, col });
        if (it == scales.end() || it->second.generation != gen) {
            return POTAL_HW_RELEASE_MISS;
        }
        scales.erase(it);
        releases++;
        return POTAL_OK;
    }

    potal_status run(const potal_loop_ws & l) {
        if (in_flight) {
            return POTAL_ERR_BUSY;
        }
        const potal_status st = potal_loop_ws_check(&l);
        if (st != POTAL_OK) {
            return st;
        }
        const bool accumulate = (l.flags & POTAL_GEMM_ACCUMULATE) != 0;
        auto &     a_cc       = acc[half];
        // LoopMatmul EX order: k outer, then j, then i.
        for (uint32_t k = 0; k < l.k_tiles; ++k) {
            const uint32_t k_valid = DIM - (k == l.k_tiles - 1 ? l.pad_k : 0);
            for (uint32_t j = 0; j < l.j_tiles; ++j) {
                const uint32_t cols = DIM - (j == l.j_tiles - 1 ? l.pad_j : 0);
                for (uint32_t i = 0; i < l.i_tiles; ++i) {
                    const uint32_t rows = DIM - (i == l.i_tiles - 1 ? l.pad_i : 0);
                    for (uint32_t c = 0; c < cols; ++c) {
                        const uint32_t slot = l.scale_base + (k / l.passes_per_block) * l.j_tiles + j;
                        auto           e    = scales.find({ slot, c });
                        if (e == scales.end() || e->second.generation != l.scale_generation) {
                            scale_miss = true;
                            continue;
                        }
                        for (uint32_t r = 0; r < rows; ++r) {
                            int32_t partial = 0;
                            for (uint32_t kk = 0; kk < k_valid; ++kk) {
                                partial += (int32_t) l.a[(size_t) (i * DIM + r) * l.lda + k * DIM + kk] *
                                           l.b[(size_t) (k * DIM + kk) * l.ldb + j * DIM + c];
                            }
                            const int32_t v   = scu(partial, e->second.carrier);
                            int32_t &     out = a_cc[((size_t) (i * l.j_tiles + j) * DIM + r) * DIM + c];
                            out               = (!accumulate && k == 0) ? v : sat_add(out, v);
                        }
                    }
                }
            }
        }
        if (l.c != nullptr) {
            for (uint32_t i = 0; i < l.i_tiles; ++i) {
                const uint32_t rows = DIM - (i == l.i_tiles - 1 ? l.pad_i : 0);
                for (uint32_t j = 0; j < l.j_tiles; ++j) {
                    const uint32_t cols = DIM - (j == l.j_tiles - 1 ? l.pad_j : 0);
                    for (uint32_t r = 0; r < rows; ++r) {
                        for (uint32_t c = 0; c < cols; ++c) {
                            l.c[(size_t) (i * DIM + r) * l.ldc + j * DIM + c] =
                                a_cc[((size_t) (i * l.j_tiles + j) * DIM + r) * DIM + c];
                        }
                    }
                }
            }
            half ^= 1; // the ACC half advances only when C is stored
        }
        in_flight = true;
        loops++;
        return POTAL_OK;
    }

    static potal_status op_write(void * ctx, uint32_t s, uint32_t g, uint32_t c, uint32_t v) {
        return static_cast<emulator *>(ctx)->write(s, g, c, v);
    }
    static potal_status op_release(void * ctx, uint32_t s, uint32_t g, uint32_t c) {
        return static_cast<emulator *>(ctx)->release(s, g, c);
    }
    static potal_status op_gemm(void * ctx, const potal_gemm_ws * g) {
        auto *             e  = static_cast<emulator *>(ctx);
        const potal_status st = potal_gemm_ws_check(g);
        if (st != POTAL_OK) {
            return st;
        }
        const potal_loop_ws l = { g->a, g->b, g->c, g->lda, g->ldb, g->ldc, g->m / (uint32_t) DIM,
                                  g->n / (uint32_t) DIM, g->k / (uint32_t) DIM, 0, 0, 0, g->scale_base,
                                  g->scale_generation, g->passes_per_block, 0 };
        e->gemms++;
        return e->run(l);
    }
    static potal_status op_loop(void * ctx, const potal_loop_ws * l) {
        return static_cast<emulator *>(ctx)->run(*l);
    }
    static potal_status op_fence(void * ctx) {
        auto * e = static_cast<emulator *>(ctx);
        if (e->fail_fence) {
            return POTAL_HW_BAD_COMMAND;
        }
        e->in_flight = false;
        e->fences++;
        return POTAL_OK;
    }

    potal_ws_ops ops() { return { op_write, op_release, op_gemm, op_loop, op_fence, this }; }
};

// gemmini.h argument list of the one request the PoTal path implements.
potal_status tiled(size_t M, size_t N, size_t K, const hp1_operands & o, potal_ws_session & session,
                   potal_tile_plan * plan, const void * D = nullptr, int act = POTAL_NO_ACTIVATION,
                   bool transpose_B = false, bool full_C = true) {
    const potal_scale_operand scales = { o.carriers, o.cstride, hp1::block_k };
    return potal_tiled_matmul_auto(M, N, K, o.a, o.b, D, o.c, o.lda, o.ldb, 0, o.ldc, POTAL_MVIN_SCALE_IDENTITY,
                                   POTAL_MVIN_SCALE_IDENTITY, 0, act, POTAL_ACC_SCALE_IDENTITY, 0, false, false,
                                   transpose_B, full_C, false, 0, POTAL_WS, scales, session, plan);
}

void fill_operands(hp1_operands & o, size_t M, size_t N, size_t K, std::mt19937 & rng, int carrier_mode) {
    std::uniform_int_distribution<int> q(-128, 127);
    std::uniform_int_distribution<int> shift(0, 6);
    for (size_t m = 0; m < M; ++m) {
        for (size_t k = 0; k < K; ++k) {
            o.a[m * o.lda + k] = (int8_t) q(rng);
        }
    }
    for (size_t k = 0; k < K; ++k) {
        for (size_t n = 0; n < N; ++n) {
            o.b[k * o.ldb + n] = (int8_t) q(rng);
        }
    }
    for (size_t b = 0; b < K / hp1::block_k; ++b) {
        for (size_t n = 0; n < N; ++n) {
            uint32_t c = (uint32_t) shift(rng);
            if (carrier_mode == 1 && (b + n) % 7 == 0) {
                c = POTAL_CARRIER_ZERO;
            }
            if (carrier_mode == 2) {
                c = (b + n) % 5 == 0 ? 31u : (b + n) % 5 == 1 ? 40u : 20u; // saturating shifts
            }
            o.carriers[b * o.cstride + n] = c;
        }
    }
}

void test_tiled_execution() {
    std::mt19937 rng(1234);
    struct shape {
        size_t m, n, k;
        int    carriers;
    };
    const shape shapes[] = {
        { 16, 16, 32, 0 },   // the validated single GEMM
        { 16, 16, 64, 1 },   // one tile, K=64: through potal_loop_ws (unvalidated scope)
        { 5, 40, 96, 1 },    // padding in I and J
        { 37, 100, 640, 0 }, // multi-I/J, padded
        { 3, 160, 4096, 1 }, // K chunks (accumulate, C NULL)
        { 40, 160, 1600, 0 },
        { 17, 48, 256, 2 },  // Sat32 SCU and ACC saturation
        { 1, 300, 768, 1 },  // decode-like: one token
    };
    for (const shape & s : shapes) {
        hp1_operands o;
        const bool   alloc = o.allocate({ (int64_t) s.m, (int64_t) s.n, (int64_t) s.k });
        fill_operands(o, s.m, s.n, s.k, rng, s.carriers);
        std::vector<int32_t> want(s.m * o.ldc, 0);
        reference_gemm_i32(o.a, o.lda, o.b, o.ldb, o.carriers, o.cstride, hp1::block_k, s.m, s.n, s.k, want.data(),
                           o.ldc);

        emulator         emu;
        potal_ws_ops     ops     = emu.ops();
        potal_ws_session session = { &ops, 0, {} };
        potal_tile_plan  plan;
        const potal_status st = tiled(s.m, s.n, s.k, o, session, &plan);

        bool equal = true;
        for (size_t m = 0; m < s.m && equal; ++m) {
            for (size_t n = 0; n < s.n; ++n) {
                if (o.c[m * o.ldc + n] != want[m * o.ldc + n]) {
                    std::printf("     mismatch m=%zu n=%zu got=%d want=%d\n", m, n, o.c[m * o.ldc + n],
                                want[m * o.ldc + n]);
                    equal = false;
                    break;
                }
            }
        }
        char name[160];
        std::snprintf(name, sizeof(name), "B tiled == integer golden M=%zu N=%zu K=%zu (tiles %zu,%zu,%zu x %zu,%zu,%zu)",
                      s.m, s.n, s.k, plan.tile_I, plan.tile_J, plan.tile_K, plan.I0, plan.J0, plan.K0);
        check(name, alloc && st == POTAL_OK && equal && !emu.scale_miss);
        std::snprintf(name, sizeof(name), "B work M=%zu N=%zu K=%zu: invocations, fences, every scale released", s.m,
                      s.n, s.k);
        check(name, session.report.invocations == plan.I0 * plan.J0 * plan.K0 &&
                        session.report.fences == session.report.invocations && emu.scales.empty() &&
                        session.report.scale_writes == session.report.scale_releases && !session.report.fatal);
        if (s.m == 16 && s.n == 16 && s.k == 32) {
            check("B validated shape issued through potal_gemm_ws_issue",
                  emu.gemms == 1 && session.report.validated_gemms == 1);
        }
        if (s.k == 4096) {
            check("B K chunks: K0 > 1 accumulated across invocations", plan.K0 > 1 && equal);
        }
    }

    // Fail-stop: SCALE_WRITE failure -> nothing issued, written entries released, session disabled.
    {
        hp1_operands o;
        o.allocate({ 16, 64, 64 });
        fill_operands(o, 16, 64, 64, rng, 0);
        emulator emu;
        emu.fail_write_at        = 5;
        potal_ws_ops     ops     = emu.ops();
        potal_ws_session session = { &ops, 0, {} };
        const potal_status st    = tiled(16, 64, 64, o, session, nullptr);
        check("B scale_write failure is fatal, nothing issued, entries released",
              st == POTAL_HW_INVALID_SCALE && session.report.fatal && emu.loops == 0 && emu.scales.empty() &&
                  std::string(session.report.stage) == "scale_write");
        emu.fail_write_at = -1;
        check("B disabled session refuses later work", tiled(16, 64, 64, o, session, nullptr) != POTAL_OK &&
                                                           emu.loops == 0 && emu.writes == 5);
    }
    // Fail-stop: FENCE failure -> fatal, scales kept (the GEMM may still run).
    {
        hp1_operands o;
        o.allocate({ 16, 16, 64 });
        fill_operands(o, 16, 16, 64, rng, 0);
        emulator emu;
        emu.fail_fence           = true;
        potal_ws_ops     ops     = emu.ops();
        potal_ws_session session = { &ops, 0, {} };
        const potal_status st    = tiled(16, 16, 64, o, session, nullptr);
        check("B fence failure is fatal and leaves the scale entries alone",
              st == POTAL_HW_BAD_COMMAND && session.report.fatal && emu.loops == 1 && !emu.scales.empty() &&
                  std::string(session.report.stage) == "fence");
    }
    // Unsupported Gemmini features are refused before any command.
    {
        hp1_operands o;
        o.allocate({ 16, 16, 32 });
        fill_operands(o, 16, 16, 32, rng, 0);
        emulator         emu;
        potal_ws_ops     ops     = emu.ops();
        potal_ws_session session = { &ops, 0, {} };
        const int32_t    bias[16] = {};
        const bool ok = tiled(16, 16, 32, o, session, nullptr, bias) == POTAL_ERR_UNSUPPORTED &&
                        tiled(16, 16, 32, o, session, nullptr, nullptr, 1) == POTAL_ERR_UNSUPPORTED &&
                        tiled(16, 16, 32, o, session, nullptr, nullptr, 0, true) == POTAL_ERR_UNSUPPORTED &&
                        tiled(16, 16, 32, o, session, nullptr, nullptr, 0, false, false) == POTAL_ERR_UNSUPPORTED;
        check("B bias/activation/transpose/low-precision C refused, nothing issued",
              ok && emu.writes == 0 && emu.loops == 0 && !session.report.fatal);
    }
    // Invalid carrier: refused before any SCALE_WRITE.
    {
        hp1_operands o;
        o.allocate({ 16, 16, 32 });
        fill_operands(o, 16, 16, 32, rng, 0);
        o.carriers[3] = 32768;
        emulator         emu;
        potal_ws_ops     ops     = emu.ops();
        potal_ws_session session = { &ops, 0, {} };
        check("B invalid carrier refused before any command",
              tiled(16, 16, 32, o, session, nullptr) == POTAL_ERR_INVALID_ARGUMENT && emu.writes == 0 &&
                  !session.report.fatal);
    }
}

// ---- C: Q8_HP1 (weight reader copied from Gemmini + PoTal row checks) ----------

std::vector<block_q8_hp1> quantize_rows(const std::vector<float> & w, size_t N, size_t K) {
    std::vector<block_q8_hp1> blocks(N * K / QK8_HP);
    for (size_t n = 0; n < N; ++n) {
        quantize_row_q8_hp1_ref(w.data() + n * K, blocks.data() + n * (K / QK8_HP), (int64_t) K);
    }
    return blocks;
}

ggml_potal_args_t hp1_args(const std::vector<block_q8_hp1> & blocks, size_t N, size_t K) {
    ggml_potal_args_t args;
    args.J                     = N;
    args.K                     = K;
    args.q8_hp1_blocks         = blocks.data();
    args.q8_hp1_blocks_per_row = K / QK8_HP;
    args.q8_hp1_block_count    = N * (K / QK8_HP);
    args.native_weight_bytes   = blocks.size() * sizeof(block_q8_hp1);
    return args;
}

void test_hp1() {
    const size_t       N = 24, K = 128;
    std::vector<float> w(N * K);
    std::mt19937       rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (size_t n = 0; n < N; ++n) {
        for (size_t k = 0; k < K; ++k) {
            // Per-block magnitudes 1, 4, 64, 0.25 -> different exponents per block.
            const float mag[] = { 1.0f, 4.0f, 64.0f, 0.25f };
            w[n * K + k]      = nd(rng) * mag[(k / 32 + n) % 4] * (1.0f + (float) n);
        }
    }
    for (size_t k = 64; k < 96; ++k) {
        w[2 * K + k] = 0.0f; // zero block in row 2
    }
    for (size_t k = 0; k < K; ++k) {
        w[5 * K + k] = 0.0f; // zero row 5
    }
    std::vector<block_q8_hp1> blocks = quantize_rows(w, N, K);

    const size_t          ldb = 64, cs = 32;
    std::vector<int8_t>   B(K * ldb, 0);
    std::vector<uint32_t> carriers((K / 32) * cs, 0);
    std::vector<float>    scale(N, 0.0f);
    hp1::stats            st;
    const hp1::status     s = hp1::unpack(hp1_args(blocks, N, K), B.data(), ldb, carriers.data(), cs, scale.data(), &st);
    bool                  fields = s == hp1::status::ok;
    bool                  dequant = true;
    bool                  distinct_m = false;
    for (size_t n = 0; n < N && fields; ++n) {
        std::vector<float> ref(K);
        dequantize_row_q8_hp1(blocks.data() + n * (K / 32), ref.data(), (int64_t) K);
        for (size_t b = 0; b < K / 32; ++b) {
            const block_q8_hp1 & blk = blocks[n * (K / 32) + b];
            const uint32_t       c   = carriers[b * cs + n];
            fields = fields && (blk.m == INT16_MIN ? c == POTAL_CARRIER_ZERO : c == (uint32_t) blk.m);
            distinct_m = distinct_m || (b > 0 && blk.m != blocks[n * (K / 32)].m);
            for (size_t i = 0; i < 32; ++i) {
                fields = fields && B[(b * 32 + i) * ldb + n] == blk.qs[i];
                // qs * channel_scale * 2^m, the HP1 value the SCU path computes.
                const float v = c == POTAL_CARRIER_ZERO ? 0.0f :
                                    (float) blk.qs[i] * std::ldexp(scale[n], (int) c);
                dequant = dequant && v == ref[b * 32 + i];
            }
        }
        fields = fields && std::memcmp(&scale[n], &blocks[n * (K / 32)].channel_scale, sizeof(float)) == 0;
    }
    check("C HP1 unpack: qs -> B[k][n], m -> carrier, INT16_MIN -> zero carrier, channel_scale per column", fields);
    check("C HP1 qs * channel_scale * 2^carrier == dequantize_row_q8_hp1", dequant);
    check("C HP1 rows carry different exponents across blocks", distinct_m && st.max_m > st.min_m);
    check("C HP1 stats: zero blocks and zero row counted", st.zero_rows == 1 && st.zero_blocks >= 1 + K / 32);

    const auto expect = [&](const char * name, hp1::status want, quants::wreader::WeightReaderStatus want_reader,
                            auto mutate) {
        std::vector<block_q8_hp1> bad = blocks;
        mutate(bad);
        hp1::stats                          ignored;
        quants::wreader::WeightReaderStatus reader = quants::wreader::WeightReaderStatus::Success;
        const hp1::status got = hp1::unpack(hp1_args(bad, N, K), B.data(), ldb, carriers.data(), cs, scale.data(),
                                            &ignored, &reader);
        check(std::string("C HP1 rejects ") + name, got == want && reader == want_reader);
    };
    using RS = quants::wreader::WeightReaderStatus;
    expect("non-zero padding (reader)", hp1::status::metadata, RS::InvalidMetadata,
           [](auto & b) { b[9].padding[1] = 1; });
    expect("negative exponent (reader)", hp1::status::metadata, RS::InvalidMetadata, [](auto & b) { b[4].m = -1; });
    expect("NaN channel scale (reader)", hp1::status::metadata, RS::InvalidMetadata,
           [](auto & b) { b[2].channel_scale = std::nanf(""); });
    expect("channel scale mismatch inside a row", hp1::status::scale_mismatch, RS::Success,
           [](auto & b) { b[1].channel_scale *= 2.0f; });
    expect("negative channel scale", hp1::status::channel_scale, RS::Success,
           [](auto & b) { for (size_t i = 0; i < 4; ++i) b[i].channel_scale = -1.0f; });
    expect("zero channel scale with a non-zero block", hp1::status::channel_scale, RS::Success,
           [](auto & b) { b[5 * 4 + 1].m = 0; });
    hp1::stats        ignored;
    ggml_potal_args_t short_k = hp1_args(blocks, N, K);
    short_k.K                 = 100;
    check("C HP1 rejects K not a multiple of 32 (storage contract)",
          hp1::unpack(short_k, B.data(), ldb, carriers.data(), cs, scale.data(), &ignored) == hp1::status::metadata);
    uint32_t carrier = 0;
    check("C carrier encoding (hp1_scu.hpp)",
          quants::hp1::encode_carrier(0, carrier) && carrier == 0 && quants::hp1::encode_carrier(32767, carrier) &&
              carrier == 32767 && quants::hp1::encode_carrier(INT16_MIN, carrier) &&
              carrier == POTAL_CARRIER_ZERO && !quants::hp1::encode_carrier(-5, carrier));
}

// ---- D: per-tensor activation quantization (copied TENSOR quantizer) ---------

// One F32 tensor ne = {K, M} through quants::quantize_activation.
bool quantize_f32(const std::vector<float> & values, int64_t K, int64_t M, ggml_potal_args_t & args) {
    static ggml_context_ptr ctx;
    ggml_init_params        ip = { 1024 * 1024, nullptr, false };
    ctx.reset(ggml_init(ip));
    ggml_tensor * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, K, M);
    std::memcpy(x->data, values.data(), values.size() * sizeof(float));
    args   = ggml_potal_args_t{};
    args.I = (size_t) M;
    args.K = (size_t) K;
    return args.A.allocate(args.I, args.K, 8) && quants::quantize_activation(x, args);
}

float tensor_scale(const ggml_potal_args_t & args) {
    const auto * m = std::get_if<quants::act::tensor::Meta>(&args.act_quant.storage());
    return m ? m->scale : -1.0f;
}

void test_act_quant() {
    // Row 0 small, row 1 holds the tensor maximum: row 0 must use the tensor scale.
    const int64_t      K = 8;
    std::vector<float> x = { 0.5f, -0.25f, 1.0f, 0.0f, 0.125f, -1.0f, 0.75f, 0.3f,
                             12.7f, -3.0f, 6.35f, 0.0f, -12.7f, 2.0f, 1.0f, 0.05f };
    ggml_potal_args_t args;
    const bool        ok    = quantize_f32(x, K, 2, args);
    const float       scale = (float) (12.7 / (float) 127);
    bool              codes = ok && tensor_scale(args) == scale;
    for (int64_t r = 0; r < 2 && codes; ++r) {
        for (int64_t k = 0; k < K; ++k) {
            const long want = std::lrint((double) x[r * K + k] / (double) scale);
            codes = codes && args.A.get(r, k) == std::max(-128L, std::min(127L, want));
        }
    }
    check("D tensor scale = max_abs / 127 over the whole tensor, not per tile/row", codes);
    check("D row with small values uses the tensor scale (0.5 -> 5)", args.A.get(0, 0) == 5);

    // Scale 1 (max 127): x / scale = 2.5 -> 2, 3.5 -> 4, -2.5 -> -2 (lrint, half to even).
    check("D lrint half-to-even", quantize_f32({ 127.0f, 2.5f, 3.5f, -2.5f }, 4, 1, args) &&
                                      tensor_scale(args) == 1.0f && args.A.get(0, 1) == 2 && args.A.get(0, 2) == 4 &&
                                      args.A.get(0, 3) == -2);

    check("D zero tensor: scale 1.0, codes 0", quantize_f32(std::vector<float>(16, 0.0f), K, 2, args) &&
                                                   tensor_scale(args) == 1.0f && args.A.get(0, 0) == 0 &&
                                                   args.A.get(1, 7) == 0);

    // Gemmini semantics: non-finite values leave the statistic and become code 0.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    check("D NaN/Inf excluded from the scale and quantized to 0",
          quantize_f32({ 127.0f, nan, inf, -inf }, 4, 1, args) && tensor_scale(args) == 1.0f &&
              args.A.get(0, 0) == 127 && args.A.get(0, 1) == 0 && args.A.get(0, 2) == 0 && args.A.get(0, 3) == 0);
    check("D tensor without a finite value fails", !quantize_f32({ nan, inf }, 2, 1, args));

    // dequantize_activation: code * scale.
    std::vector<float> back(16, 0.0f);
    quantize_f32(x, K, 2, args);
    check("D dequantize_activation = code * scale",
          quants::dequantize_activation(back.data(), K, 1, 2, K, args) &&
              back[8] == (float) args.A.get(1, 0) * scale);
}

// ---- E: reconstruction and the full pipeline on the emulator ----------------

void test_end_to_end() {
    std::mt19937                    rng(99);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int64_t                   M = 7, N = 48, K = 256;

    ggml_init_params ip  = { 64 * 1024 * 1024, nullptr, false };
    ggml_context_ptr ctx(ggml_init(ip));
    ggml_tensor *    w   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_HP1, K, N);
    ggml_tensor *    x   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, K, M);
    ggml_tensor *    y   = ggml_mul_mat(ctx.get(), w, x);
    std::vector<float> wf(N * K);
    for (auto & v : wf) {
        v = nd(rng) * 0.05f;
    }
    for (int64_t k = 0; k < K; ++k) {
        wf[3 * K + k] = 0.0f; // zero row
    }
    ggml_quantize_chunk(GGML_TYPE_Q8_HP1, wf.data(), w->data, 0, N, K, nullptr);
    float * xf = static_cast<float *>(x->data);
    for (int64_t i = 0; i < M * K; ++i) {
        xf[i] = nd(rng);
    }

    hp1_shape s;
    check("E F32 x Q8_HP1 accepted by the contract", check_mul_mat_hp1(y, true, &s) == GGML_POTAL_REJECT_NONE &&
                                                         s.m == M && s.n == N && s.k == K);
    hp1_operands      o;
    ggml_potal_args_t args;
    float             act_scale = 0.0f;
    hp1::stats        ws;
    const bool        prepared = o.allocate(s) && init_args(w, x, y, s, "gpt2", "test", args) &&
                          quantize_activations(x, args, o, &act_scale) &&
                          unpack_weights(args, o, &ws) == hp1::status::ok;

    std::vector<int32_t> want(o.m_pad * o.ldc, 0);
    reference_gemm_i32(o.a, o.lda, o.b, o.ldb, o.carriers, o.cstride, hp1::block_k, M, N, K, want.data(), o.ldc);

    emulator         emu;
    potal_ws_ops     ops     = emu.ops();
    potal_ws_session session = { &ops, 0, {} };
    const bool       ran     = tiled(M, N, K, o, session, nullptr) == POTAL_OK;
    reconstruct(o, args, act_scale);

    bool   exact = prepared && ran;
    double err2 = 0, ref2 = 0;
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            const float got  = static_cast<float *>(y->data)[m * N + n];
            const float gold = (float) ((double) want[m * o.ldc + n] * (double) act_scale * (double) o.scale[n]);
            exact            = exact && got == gold && o.c[m * o.ldc + n] == want[m * o.ldc + n];
            double fp = 0; // FP32 model reference: dequantized HP1 weight x F32 activation
            std::vector<float> row(K);
            dequantize_row_q8_hp1(static_cast<const block_q8_hp1 *>(w->data) + n * (K / 32), row.data(), K);
            for (int64_t k = 0; k < K; ++k) {
                fp += (double) row[k] * xf[m * K + k];
            }
            err2 += (got - fp) * (got - fp);
            ref2 += fp * fp;
        }
    }
    check("E integer domain: C == golden, dst == C * activation_scale * channel_scale exactly (no compensation)",
          exact);
    check("E zero weight row gives exact zeros", static_cast<float *>(y->data)[3] == 0.0f);
    std::printf("     model-quality (informational, compensation OFF): rel_l2_error=%.6f vs FP32(dequant W) x x\n",
                ref2 > 0 ? std::sqrt(err2 / ref2) : 0.0);
}

// ---- F: GGML contract --------------------------------------------------------

void test_ggml_contract() {
    ggml_init_params ip = { 16 * 1024 * 1024, nullptr, true };
    ggml_context_ptr ctx(ggml_init(ip));
    ggml_context *   c = ctx.get();
    const auto mm = [&](ggml_type wt, ggml_type xt, int64_t k, int64_t n, int64_t m) {
        return ggml_mul_mat(c, ggml_new_tensor_2d(c, wt, k, n), ggml_new_tensor_2d(c, xt, k, m));
    };
    ggml_tensor * hp = mm(GGML_TYPE_Q8_HP1, GGML_TYPE_F32, 768, 2304, 5);
    check("F Q8_HP1 x F32 2-D accepted", check_op(hp) == GGML_POTAL_REJECT_NONE);
    check("F Q8_HP1 x F32 odd M/N accepted", check_op(mm(GGML_TYPE_Q8_HP1, GGML_TYPE_F32, 64, 37, 3)) ==
                                                 GGML_POTAL_REJECT_NONE);
    check("F Q8_HP1 x F16 rejected (type)", check_op(mm(GGML_TYPE_Q8_HP1, GGML_TYPE_F16, 64, 16, 16)) ==
                                                GGML_POTAL_REJECT_TYPE);
    check("F Q8_HP1 x I8 rejected (type)", check_op(mm(GGML_TYPE_Q8_HP1, GGML_TYPE_I8, 64, 16, 16)) ==
                                               GGML_POTAL_REJECT_TYPE);
    check("F Q8_0 x F32 rejected (type)", check_op(mm(GGML_TYPE_Q8_0, GGML_TYPE_F32, 64, 16, 16)) ==
                                              GGML_POTAL_REJECT_TYPE);
    check("F batched F32 activations rejected (shape)",
          check_op(ggml_mul_mat(c, ggml_new_tensor_2d(c, GGML_TYPE_Q8_HP1, 64, 16),
                                ggml_new_tensor_3d(c, GGML_TYPE_F32, 64, 4, 2))) == GGML_POTAL_REJECT_SHAPE);
    check("F strided (non-contiguous) activation rows rejected (layout)",
          check_op(ggml_mul_mat(c, ggml_new_tensor_2d(c, GGML_TYPE_Q8_HP1, 64, 16),
                                ggml_view_2d(c, ggml_new_tensor_2d(c, GGML_TYPE_F32, 96, 4), 64, 4, 96 * sizeof(float),
                                             0))) == GGML_POTAL_REJECT_LAYOUT);
    check("F transposed activations rejected (layout)",
          check_op(ggml_mul_mat(c, ggml_new_tensor_2d(c, GGML_TYPE_Q8_HP1, 64, 16),
                                ggml_transpose(c, ggml_new_tensor_2d(c, GGML_TYPE_F32, 4, 64)))) ==
              GGML_POTAL_REJECT_LAYOUT);
    ggml_tensor * tagged = mm(GGML_TYPE_Q8_HP1, GGML_TYPE_F32, 64, 16, 16);
    tagged->op_params[1] = 1;
    check("F tagged MUL_MAT rejected (op_params)", check_op(tagged) == GGML_POTAL_REJECT_OP_PARAMS);
    check("F I8 x I8 16x16x32 still accepted", check_op(mm(GGML_TYPE_I8, GGML_TYPE_I8, 32, 16, 16)) ==
                                                   GGML_POTAL_REJECT_NONE);

    // The CPU backend refuses Q8_HP1 MUL_MAT: a PoTal reject can never become
    // a silent CPU computation of an HP1 GEMM.
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    check("F CPU backend does not run Q8_HP1 MUL_MAT (no silent fallback)", !ggml_backend_supports_op(cpu.get(), hp));

    // Host: real backend, no accelerator -> failure, NaN dst, no success counted.
    ggml_backend_ptr potal(ggml_backend_potal_init());
    ggml_init_params dp = { 16 * 1024 * 1024, nullptr, false };
    ggml_context_ptr dctx(ggml_init(dp));
    ggml_tensor *    w = ggml_new_tensor_2d(dctx.get(), GGML_TYPE_Q8_HP1, 64, 16);
    ggml_tensor *    x = ggml_new_tensor_2d(dctx.get(), GGML_TYPE_F32, 64, 4);
    std::vector<float> wf(16 * 64, 0.5f);
    ggml_quantize_chunk(GGML_TYPE_Q8_HP1, wf.data(), w->data, 0, 16, 64, nullptr);
    std::fill(static_cast<float *>(x->data), static_cast<float *>(x->data) + 4 * 64, 1.0f);
    ggml_tensor * y = ggml_mul_mat(dctx.get(), w, x);
    ggml_cgraph * g = ggml_new_graph(dctx.get());
    ggml_build_forward_expand(g, y);
    ggml_backend_potal_stats before = {}, after = {};
    ggml_backend_potal_get_stats(potal.get(), &before);
    const ggml_status st = ggml_backend_graph_compute(potal.get(), g);
    ggml_backend_potal_get_stats(potal.get(), &after);
#if defined(__riscv)
    std::printf("     (target: graph_compute status=%d, hp1_completed=%llu)\n", (int) st,
                (unsigned long long) after.hp1_completed);
    check("F target: HP1 node counted", after.hp1_nodes == before.hp1_nodes + 1);
#else
    check("F host: no accelerator -> FAILED, dst NaN, no completion counted",
          st == GGML_STATUS_FAILED && std::isnan(static_cast<float *>(y->data)[0]) &&
              after.hp1_completed == before.hp1_completed && after.hp1_failed == before.hp1_failed + 1 &&
              after.tiled_fences == before.tiled_fences);
#endif
}

} // namespace

int main() {
    std::printf("POTAL_TILING_UNIT_TEST DIM=%u\n", (unsigned) POTAL_HW_DIM);
    test_tile_select();
    test_tiled_execution();
    test_hp1();
    test_act_quant();
    test_end_to_end();
    test_ggml_contract();
    if (failures != 0) {
        std::printf("POTAL_TILING_UNIT_TEST FAIL (%d)\n", failures);
        return 1;
    }
    std::printf("POTAL_TILING_UNIT_TEST PASS\n");
    return 0;
}
