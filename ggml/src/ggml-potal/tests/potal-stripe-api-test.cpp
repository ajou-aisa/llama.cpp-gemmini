// Tests of the planned stripe GEMM API (potal.h, library potal-tiled-api)
// without hardware (host and target):
//
//   A  C11 caller: the header compiles as C, the C++ library links into a C
//      translation unit, refused descriptors issue nothing; C and C++ agree
//      on the struct layout
//   B  frozen plans on the RTL-semantics emulator vs an independent integer
//      reference: every stripe of one plan, the tail stripe lowering tile_i
//      only, an N tail, K chunks accumulated in the ACC, the validated
//      single-GEMM path; no tile search while executing
//   C  every refused descriptor issues no command
//   D  generation wrap 255 -> 1 over real invocations; WRITE == RELEASE
//   E  a fatal report disables the session; the legacy tiled API shares the
//      session's generation sequence
//   F  host only: the hardware session reports POTAL_ERR_NO_HARDWARE
//
// The emulator is a test double: passing proves the software contract, not
// the hardware (potal-stripe-smoke on FireSim).

#include "matmul/stripe-api.hpp"
#include "matmul/tiling.hpp"
#include "potal.h"
#include "potal_hw_params.h"
#include "support/potal-emulator.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern "C" {
size_t potal_stripe_api_c_layout(size_t * out, size_t count);
int    potal_stripe_api_c_caller(void);
}

using namespace ggml::potal;
using potal_test::emulator;

namespace {

int failures = 0;

void check(const std::string & name, bool ok) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name.c_str());
    failures += ok ? 0 : 1;
}

constexpr size_t D = POTAL_HW_DIM;

// ---- A ------------------------------------------------------------------------

void test_c_caller() {
    check("A C11 caller: create/destroy, refused descriptors issue nothing", potal_stripe_api_c_caller() == 0);
    size_t       c_view[32] = {};
    const size_t count      = potal_stripe_api_c_layout(c_view, 32);
    const size_t cpp_view[] = {
        sizeof(potal_stripe_gemm_desc),
        offsetof(potal_stripe_gemm_desc, work_kind),
        offsetof(potal_stripe_gemm_desc, m),
        offsetof(potal_stripe_gemm_desc, a),
        offsetof(potal_stripe_gemm_desc, lda),
        offsetof(potal_stripe_gemm_desc, carriers),
        offsetof(potal_stripe_gemm_desc, scale_block_k),
        offsetof(potal_stripe_gemm_desc, tile_k),
        sizeof(potal_stripe_gemm_report),
        offsetof(potal_stripe_gemm_report, fatal),
        offsetof(potal_stripe_gemm_report, loop_ws),
        offsetof(potal_stripe_gemm_report, last_generation),
        offsetof(potal_stripe_gemm_report, tile_i),
        offsetof(potal_stripe_gemm_report, pad_k),
        offsetof(potal_stripe_gemm_report, fence_s),
    };
    bool same = count == sizeof(cpp_view) / sizeof(cpp_view[0]);
    for (size_t i = 0; same && i < count; ++i) {
        same = c_view[i] == cpp_view[i];
    }
    check("A C and C++ see the same descriptor/report layout", same);
}

#if POTAL_HW_DIM == 16 // B..F use the DIM16 HP1 layout: scale_block_k 32 = 2 passes of DIM

size_t up(size_t v, size_t to) {
    return (v + to - 1) / to * to;
}

// One zeroed, 64-byte aligned buffer (the DMA layout of the backend).
template <class T> class buffer {
  public:
    explicit buffer(size_t count) : bytes_(up(std::max<size_t>(count, 1) * sizeof(T), 64)) {
        p_ = static_cast<T *>(std::aligned_alloc(64, bytes_));
        std::memset(p_, 0, bytes_);
    }
    ~buffer() { std::free(p_); }
    buffer(const buffer &)             = delete;
    buffer & operator=(const buffer &) = delete;
    T *       get() { return p_; }
    const T * get() const { return p_; }

  private:
    size_t bytes_;
    T *    p_;
};

// C[m][n] = A[m][k] * B[k][n] with carriers per (32-wide K block, column).
struct problem {
    size_t           m, n, k, lda, ldb, ldc, cstride;
    buffer<int8_t>   a;
    buffer<int8_t>   b;
    buffer<uint32_t> carriers;
    buffer<int32_t>  c;

    problem(size_t m_, size_t n_, size_t k_, uint32_t seed, int carrier_mode) :
        m(m_), n(n_), k(k_), lda(up(k_, 64)), ldb(up(n_, 64)), ldc(up(n_, D)), cstride(up(n_, D)),
        a(up(m_, D) * up(k_, 64)), b(k_ * up(n_, 64)), carriers(k_ / 32 * up(n_, D)), c(up(m_, D) * up(n_, D)) {
        std::mt19937                       rng(seed);
        std::uniform_int_distribution<int> q(-128, 127);
        std::uniform_int_distribution<int> shift(0, 6);
        for (size_t i = 0; i < m; ++i) {
            for (size_t kk = 0; kk < k; ++kk) {
                a.get()[i * lda + kk] = (int8_t) q(rng);
            }
        }
        for (size_t kk = 0; kk < k; ++kk) {
            for (size_t j = 0; j < n; ++j) {
                b.get()[kk * ldb + j] = (int8_t) q(rng);
            }
        }
        for (size_t blk = 0; blk < k / 32; ++blk) {
            for (size_t j = 0; j < n; ++j) {
                uint32_t v = (uint32_t) shift(rng);
                if (carrier_mode == 1 && (blk + j) % 7 == 0) {
                    v = POTAL_CARRIER_ZERO;
                }
                if (carrier_mode == 2) {
                    v = (blk + j) % 3 == 0 ? 31u : (blk + j) % 3 == 1 ? 40u : 20u; // saturating shifts
                }
                carriers.get()[blk * cstride + j] = v;
            }
        }
    }

    // potal.h's formula without tiling: K passes of DIM in order, SCU, Sat32.
    std::vector<int32_t> reference() const {
        std::vector<int32_t> out(m * n);
        const size_t         passes = (k + D - 1) / D;
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                int32_t acc = 0;
                for (size_t p = 0; p < passes; ++p) {
                    int32_t partial = 0;
                    for (size_t kk = p * D; kk < k && kk < (p + 1) * D; ++kk) {
                        partial += (int32_t) a.get()[i * lda + kk] * b.get()[kk * ldb + j];
                    }
                    const int32_t v = emulator::scu(partial, carriers.get()[(p * D / 32) * cstride + j]);
                    acc             = p == 0 ? v : emulator::sat_add(acc, v);
                }
                out[i * n + j] = acc;
            }
        }
        return out;
    }

    bool equals(const std::vector<int32_t> & want) const {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (c.get()[i * ldc + j] != want[i * n + j]) {
                    return false;
                }
            }
        }
        return true;
    }

    // The descriptor of rows [row_begin, row_begin + rows) with a frozen plan.
    potal_stripe_gemm_desc stripe(size_t row_begin, size_t rows, const potal_tile_plan & plan) {
        potal_stripe_gemm_desc d = {};
        d.run_id                 = 7;
        d.stripe_id              = row_begin / std::max<size_t>(1, plan.tile_I * D);
        d.row_begin              = row_begin;
        d.work_kind              = POTAL_STRIPE_WORK_MAIN;
        d.m                      = (uint32_t) rows;
        d.n                      = (uint32_t) n;
        d.k                      = (uint32_t) k;
        d.a                      = a.get() + row_begin * lda;
        d.b                      = b.get();
        d.c                      = c.get() + row_begin * ldc;
        d.lda                    = lda;
        d.ldb                    = ldb;
        d.ldc                    = ldc;
        d.carriers               = carriers.get();
        d.carrier_stride         = cstride;
        d.scale_block_k          = 32;
        d.tile_i                 = (uint32_t) plan.tile_I;
        d.tile_j                 = (uint32_t) plan.tile_J;
        d.tile_k                 = (uint32_t) plan.tile_K;
        return d;
    }
};

struct emu_session {
    emulator             emu;
    potal_ws_ops         ops     = emu.ops();
    potal_stripe_session session = { { &ops, 0, {} } };
};

// ---- B ------------------------------------------------------------------------

// Every stripe of a plan selected for the whole M, executed with the frozen tiles.
void run_stripes(const char * name, size_t M, size_t N, size_t K, int carrier_mode, bool expect_k_chunks) {
    problem         p(M, N, K, (uint32_t) (M * 131 + N * 7 + K), carrier_mode);
    potal_tile_plan plan;
    const bool      planned = potal_tile_select(M, N, K, 32 / D, &plan) == POTAL_OK;
    emu_session     es;
    const uint64_t  selects_before = potal_tile_select_count();
    const size_t    rows_per       = plan.tile_I * D;
    bool            ok             = planned;
    size_t          stripes = 0, tail_rows = 0;
    uint64_t        loops = 0, fences = 0;
    potal_stripe_gemm_report last = {};
    for (size_t row = 0; ok && row < M; row += rows_per) {
        const size_t                 rows = std::min(rows_per, M - row);
        const potal_stripe_gemm_desc d    = p.stripe(row, rows, plan);
        potal_stripe_gemm_report     r;
        ok = potal_tiled_matmul_stripe(&es.session, &d, &r) == POTAL_OK && r.stage == nullptr && r.fatal == 0;
        ok = ok && r.tile_j == plan.tile_J && r.tile_k == plan.tile_K &&
             r.tile_i == std::min<uint32_t>((uint32_t) plan.tile_I, (uint32_t) ((rows + D - 1) / D)) &&
             r.loop_ws == (uint64_t) r.i0 * r.j0 * r.k0 && r.fences == r.loop_ws &&
             r.scale_writes == r.scale_releases && r.last_generation == es.session.ws.generation;
        loops += r.loop_ws;
        fences += r.fences;
        tail_rows = rows;
        last      = r;
        stripes++;
    }
    const bool exact = ok && p.equals(p.reference());
    const std::string tag = std::string("B ") + name + " (" + std::to_string(M) + "x" + std::to_string(N) + "x" +
                            std::to_string(K) + ", " + std::to_string(stripes) + " stripes of " +
                            std::to_string(rows_per) + " rows, tail " + std::to_string(tail_rows) + ")";
    check(tag + ": C == independent reference, every report consistent", exact);
    check(tag + ": frozen plan, no tile search while executing", potal_tile_select_count() == selects_before);
    check(tag + ": every invocation fenced, scale entries released",
          fences == loops && es.emu.fences == loops && es.emu.scales.empty() && !es.emu.scale_miss &&
              es.emu.writes == es.emu.releases);
    if ((tail_rows + D - 1) / D < plan.tile_I) {
        check(tag + ": tail stripe lowers tile_i only", last.tile_i == (tail_rows + D - 1) / D &&
                                                            last.tile_j == plan.tile_J && last.tile_k == plan.tile_K);
    }
    if (expect_k_chunks) {
        check(tag + ": K chunks accumulate in the ACC (k0 > 1)", ok && last.k0 > 1);
    }
}

void test_frozen_plans() {
    run_stripes("tail stripe and N tail", 200, 312, 128, 0, false);
    run_stripes("one stripe, no tail lowering", 150, 40, 256, 0, false);
    run_stripes("zero carriers", 211, 72, 128, 1, false);
    run_stripes("saturating carriers", 77, 48, 256, 2, false);
    run_stripes("K chunks", 48, 160, 4096, 0, true);
    run_stripes("residual-shaped (K32 runs)", 37, 40, 96, 2, false);

    // 16x16x32 with a whole plan goes through the validated potal_gemm_ws_issue path.
    problem                      p(16, 16, 32, 5, 0);
    potal_tile_plan              plan;
    emu_session                  es;
    const bool                   planned = potal_tile_select(16, 16, 32, 2, &plan) == POTAL_OK;
    const potal_stripe_gemm_desc d       = p.stripe(0, 16, plan);
    potal_stripe_gemm_report     r;
    const potal_status           st = planned ? potal_tiled_matmul_stripe(&es.session, &d, &r) : POTAL_ERR_UNSUPPORTED;
    check("B validated 16x16x32 stripe through potal_gemm_ws_issue",
          st == POTAL_OK && r.validated_gemms == 1 && es.emu.gemms == 1 && p.equals(p.reference()));
}

// ---- C ------------------------------------------------------------------------

void test_rejections() {
    problem         p(40, 48, 96, 9, 0);
    potal_tile_plan plan;
    (void) potal_tile_select(40, 48, 96, 2, &plan);
    struct mutation {
        const char * name;
        void (*apply)(potal_stripe_gemm_desc &, problem &);
    };
    const mutation mutations[] = {
        { "A NULL", [](potal_stripe_gemm_desc & d, problem &) { d.a = nullptr; } },
        { "B NULL", [](potal_stripe_gemm_desc & d, problem &) { d.b = nullptr; } },
        { "C NULL", [](potal_stripe_gemm_desc & d, problem &) { d.c = nullptr; } },
        { "carriers NULL", [](potal_stripe_gemm_desc & d, problem &) { d.carriers = nullptr; } },
        { "m = 0", [](potal_stripe_gemm_desc & d, problem &) { d.m = 0; } },
        { "work_kind 2", [](potal_stripe_gemm_desc & d, problem &) { d.work_kind = 2; } },
        { "lda < k", [](potal_stripe_gemm_desc & d, problem &) { d.lda = d.k - 1; } },
        { "ldb < n", [](potal_stripe_gemm_desc & d, problem &) { d.ldb = d.n - 1; } },
        { "ldc < n", [](potal_stripe_gemm_desc & d, problem &) { d.ldc = d.n - 1; } },
        { "ldc beyond CONFIG_ST", [](potal_stripe_gemm_desc & d, problem &) { d.ldc = (size_t) UINT32_MAX; } },
        { "carrier_stride < n", [](potal_stripe_gemm_desc & d, problem &) { d.carrier_stride = d.n - 1; } },
        { "A misaligned", [](potal_stripe_gemm_desc & d, problem &) { d.a = d.a + 1; } },
        { "B misaligned", [](potal_stripe_gemm_desc & d, problem &) { d.b = d.b + 8; } },
        { "C misaligned", [](potal_stripe_gemm_desc & d, problem &) { d.c = d.c + 1; } },
        { "scale_block_k 0", [](potal_stripe_gemm_desc & d, problem &) { d.scale_block_k = 0; } },
        { "scale_block_k not a DIM multiple", [](potal_stripe_gemm_desc & d, problem &) { d.scale_block_k = 24; } },
        { "k not whole scale blocks", [](potal_stripe_gemm_desc & d, problem &) { d.k = 80; } },
        { "tile_i 0", [](potal_stripe_gemm_desc & d, problem &) { d.tile_i = 0; } },
        { "tile_j beyond N", [](potal_stripe_gemm_desc & d, problem &) { d.tile_j = 4; } },
        { "tile_k beyond K", [](potal_stripe_gemm_desc & d, problem &) { d.tile_k = 8; } },
        { "tile_k not whole scale blocks", [](potal_stripe_gemm_desc & d, problem &) { d.tile_k = 3; } },
        { "invalid carrier value",
          [](potal_stripe_gemm_desc & d, problem & p) {
              p.carriers.get()[(d.k / 32 - 1) * d.carrier_stride + d.n - 1] = 40000;
          } },
        { "C overlaps A", [](potal_stripe_gemm_desc & d, problem & p) { d.c = (int32_t *) (void *) p.a.get(); } },
        { "A extent beyond the 32-bit DRAM offset",
          [](potal_stripe_gemm_desc & d, problem &) { d.lda = (size_t) 1 << 30; } },
    };
    for (const mutation & mu : mutations) {
        emu_session            es;
        potal_stripe_gemm_desc d = p.stripe(0, 40, plan);
        mu.apply(d, p);
        potal_stripe_gemm_report r;
        const potal_status       st = potal_tiled_matmul_stripe(&es.session, &d, &r);
        check(std::string("C refused without a command: ") + mu.name,
              st != POTAL_OK && r.status == st && r.stage != nullptr && r.fatal == 0 && r.loop_ws == 0 &&
                  r.scale_writes == 0 && es.emu.writes == 0 && es.emu.loops == 0 && es.emu.fences == 0 &&
                  !es.session.ws.report.fatal);
        p.carriers.get()[(96 / 32 - 1) * p.cstride + 48 - 1] = 1; // undo the carrier mutation
    }
    {
        // 8 x 8 output tiles need 1024 ACC rows: over one LOOP_WS half (and WorkIdsPerLoop).
        problem                  big(128, 128, 64, 17, 0);
        emu_session              es;
        potal_tile_plan          wide = plan;
        wide.tile_I = wide.tile_J = 8;
        wide.tile_K                  = 2;
        const potal_stripe_gemm_desc d  = big.stripe(0, 128, wide);
        potal_stripe_gemm_report     r;
        const potal_status           st = potal_tiled_matmul_stripe(&es.session, &d, &r);
        check("C refused without a command: tiles beyond the ACC half",
              st == POTAL_ERR_CAPACITY && std::string(r.stage) == "tiles" && es.emu.writes == 0 && es.emu.loops == 0);
    }
    emu_session              es;
    potal_stripe_gemm_desc   d = p.stripe(0, 40, plan);
    potal_stripe_gemm_report r;
    check("C NULL report refused", potal_tiled_matmul_stripe(&es.session, &d, nullptr) == POTAL_ERR_INVALID_ARGUMENT);
    check("C NULL session refused", potal_tiled_matmul_stripe(nullptr, &d, &r) == POTAL_ERR_INVALID_ARGUMENT);
    check("C NULL descriptor refused", potal_tiled_matmul_stripe(&es.session, nullptr, &r) ==
                                           POTAL_ERR_INVALID_ARGUMENT);
    potal_stripe_session unbound = {};
    check("C session without ops refused", potal_tiled_matmul_stripe(&unbound, &d, &r) == POTAL_ERR_INVALID_ARGUMENT);
    check("C refusals left the session usable", potal_tiled_matmul_stripe(&es.session, &d, &r) == POTAL_OK &&
                                                    p.equals(p.reference()));
}

// ---- D ------------------------------------------------------------------------

void test_generation_wrap() {
    problem         p(16, 16, 32, 11, 0);
    potal_tile_plan plan;
    (void) potal_tile_select(16, 16, 32, 2, &plan);
    emu_session es;
    bool        sequence = true;
    uint32_t    previous = 0;
    const int   calls    = 2 * (int) POTAL_GENERATION_MAX + 7;
    for (int i = 0; i < calls && sequence; ++i) {
        const potal_stripe_gemm_desc d = p.stripe(0, 16, plan);
        potal_stripe_gemm_report     r;
        sequence = potal_tiled_matmul_stripe(&es.session, &d, &r) == POTAL_OK && r.loop_ws == 1 &&
                   r.last_generation == (uint32_t) (i % (int) POTAL_GENERATION_MAX) + 1 &&
                   (i == 0 || r.last_generation == previous % POTAL_GENERATION_MAX + 1);
        previous = r.last_generation;
    }
    const auto & g    = es.emu.loop_generations;
    const bool   wrap = g.size() == (size_t) calls && g[POTAL_GENERATION_MAX - 1] == POTAL_GENERATION_MAX &&
                      g[POTAL_GENERATION_MAX] == 1 && std::find(g.begin(), g.end(), 0u) == g.end();
    check("D generation 1..255 then 1 again, per real invocation", sequence && wrap);
    check("D SCALE_WRITE == SCALE_RELEASE over the wrap, no entry left",
          es.emu.writes == es.emu.releases && es.emu.scales.empty() &&
              es.session.ws.report.scale_writes == es.session.ws.report.scale_releases);

    // One call of exactly 255 invocations ends on the generation it started
    // from: its report still names the last invocation's generation, not 0.
    problem         wide(16, 4080, 32, 19, 0);
    potal_tile_plan one_tile = plan;
    one_tile.tile_I = one_tile.tile_J = 1;
    one_tile.tile_K                   = 2;
    emu_session                  es255;
    const potal_stripe_gemm_desc d0 = p.stripe(0, 16, plan);
    const potal_stripe_gemm_desc d1 = wide.stripe(0, 16, one_tile);
    potal_stripe_gemm_report     r0, r1;
    const bool ran = potal_tiled_matmul_stripe(&es255.session, &d0, &r0) == POTAL_OK &&
                     potal_tiled_matmul_stripe(&es255.session, &d1, &r1) == POTAL_OK;
    check("D a call of 255 invocations reports the generation of its last one (not 0)",
          ran && r1.loop_ws == POTAL_GENERATION_MAX && r1.last_generation == r0.last_generation &&
              r1.last_generation == es255.session.ws.generation && r1.last_generation != 0 &&
              wide.equals(wide.reference()));
}

// ---- E ------------------------------------------------------------------------

void test_session() {
    problem         p(40, 48, 96, 13, 0);
    potal_tile_plan plan;
    (void) potal_tile_select(40, 48, 96, 2, &plan);

    emu_session es;
    es.emu.fail_fence              = true;
    potal_stripe_gemm_desc   d     = p.stripe(0, 40, plan);
    potal_stripe_gemm_report r;
    const potal_status       st    = potal_tiled_matmul_stripe(&es.session, &d, &r);
    const uint64_t           loops = es.emu.loops;
    check("E fence failure is fatal and leaves the scale entries (GEMM may still run)",
          st == POTAL_HW_BAD_COMMAND && r.fatal == 1 && r.stage != nullptr && std::string(r.stage) == "fence" &&
              loops == 1 && !es.emu.scales.empty());
    es.emu.fail_fence = false;
    const potal_status again = potal_tiled_matmul_stripe(&es.session, &d, &r);
    check("E a disabled session refuses later work without a command",
          again != POTAL_OK && r.fatal == 1 && std::string(r.stage) == "session_disabled" && es.emu.loops == loops);

    // The legacy auto API and the stripe API on one session: one generation sequence.
    emu_session               shared;
    const potal_scale_operand scales = { p.carriers.get(), p.cstride, 32 };
    potal_tile_plan           auto_plan;
    const potal_status        legacy = potal_tiled_matmul_auto(
        40, 48, 96, p.a.get(), p.b.get(), nullptr, p.c.get(), p.lda, p.ldb, 0, p.ldc, POTAL_MVIN_SCALE_IDENTITY,
        POTAL_MVIN_SCALE_IDENTITY, 0, POTAL_NO_ACTIVATION, POTAL_ACC_SCALE_IDENTITY, 0, false, false, false, true,
        false, 0, POTAL_WS, scales, shared.session.ws, &auto_plan);
    const uint32_t after_legacy = shared.session.ws.generation;
    const potal_status stripe   = potal_tiled_matmul_stripe(&shared.session, &d, &r);
    check("E legacy tiled API and stripe API share the session's generations",
          legacy == POTAL_OK && stripe == POTAL_OK && after_legacy > 0 &&
              r.last_generation == after_legacy + r.loop_ws && p.equals(p.reference()));
}

// ---- F ------------------------------------------------------------------------

void test_hardware_session() {
#if !defined(__riscv)
    potal_stripe_session * s = nullptr;
    const bool             created = potal_stripe_session_create(&s) == POTAL_OK && s != nullptr;
    problem                p(16, 16, 32, 3, 0);
    potal_tile_plan        plan;
    (void) potal_tile_select(16, 16, 32, 2, &plan);
    const potal_stripe_gemm_desc d = p.stripe(0, 16, plan);
    potal_stripe_gemm_report     r;
    const potal_status           st = created ? potal_tiled_matmul_stripe(s, &d, &r) : POTAL_OK;
    check("F host build: a valid stripe GEMM stops at its first hardware command (NO_HARDWARE)",
          created && st == POTAL_ERR_NO_HARDWARE && r.stage != nullptr && std::string(r.stage) == "scale_write" &&
              r.loop_ws == 0);
    potal_stripe_session_destroy(s);
#else
    std::printf("[SKIP] F hardware session: on RISC-V it would issue RoCC commands from an unpinned thread\n");
#endif
}

#endif // POTAL_HW_DIM == 16

} // namespace

int main() {
    std::printf("potal-stripe-api-test DIM=%u backend=EMULATED (no hardware)\n", (unsigned) POTAL_HW_DIM);
    test_c_caller();
#if POTAL_HW_DIM == 16
    test_frozen_plans();
    test_rejections();
    test_generation_wrap();
    test_session();
    test_hardware_session();
#else
    std::printf("[SKIP] B-F: the planned GEMM cases use the DIM16 HP1 layout (scale_block_k 32 = 2 passes)\n");
#endif
    if (failures != 0) {
        std::printf("POTAL_STRIPE_API_TEST FAIL (%d)\n", failures);
        return 1;
    }
    std::printf("POTAL_STRIPE_API_TEST PASS\n");
    return 0;
}
