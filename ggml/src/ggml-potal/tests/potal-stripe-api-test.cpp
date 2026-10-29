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
//   C  every refused descriptor issues no command, through the C API and
//      through the accelerator owner's planned step (run_planned_stripe_gemm,
//      what the hart-0 issuer runs after its affinity check): not even its
//      TLB FLUSH; stage "preflight", not fatal, the session stays usable
//   D  generation wrap 255 -> 1 over real invocations; WRITE == RELEASE
//   E  a fatal report disables the session; the legacy tiled API shares the
//      session's generation sequence
//   F  host only: the hardware session reports POTAL_ERR_NO_HARDWARE; the
//      owner step reaches potal_tlb_flush() only for an accepted descriptor
//   G  the owner step on accepted descriptors: TLB FLUSH -> FENCE -> per
//      invocation SCALE_WRITE -> LOOP_WS -> FENCE -> SCALE_RELEASE, the same
//      work and C as the C API; a failed TLB FLUSH is fatal and disables the
//      session; a disabled session issues nothing; generation wrap
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
#include <utility>
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

static_assert(D == 16 || D == 32, "stripe fixtures require DIM16 or DIM32");

size_t up(size_t v, size_t to) {
    return (v + to - 1) / to * to;
}

// One zeroed buffer, aligned for both DMA and a full INT32 accumulator row.
template <class T> class buffer {
  public:
    explicit buffer(size_t count) : bytes_(up(std::max<size_t>(count, 1) * sizeof(T), std::max<size_t>(64, D * 4))) {
        p_ = static_cast<T *>(std::aligned_alloc(std::max<size_t>(64, D * 4), bytes_));
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

// The emulator's command record of one accepted planned GEMM: "TF" (TLB FLUSH,
// FENCE), then per invocation W..W (L|G) F R..R with as many releases as
// writes. Returns the invocations, -1 when the record has another shape.
long owner_invocations(const std::string & c) {
    if (c.size() < 2 || c[0] != 'T' || c[1] != 'F') {
        return -1;
    }
    size_t i = 2;
    long   n = 0;
    while (i < c.size()) {
        size_t writes = 0, releases = 0;
        for (; i < c.size() && c[i] == 'W'; ++i) {
            writes++;
        }
        if (writes == 0 || i + 1 >= c.size() || (c[i] != 'L' && c[i] != 'G') || c[i + 1] != 'F') {
            return -1;
        }
        for (i += 2; i < c.size() && c[i] == 'R'; ++i) {
            releases++;
        }
        if (releases != writes) {
            return -1;
        }
        n++;
    }
    return n;
}

// One descriptor through both entry points, each on a fresh emulator session.
struct both_runs {
    potal_status             api_status = POTAL_OK;
    potal_stripe_gemm_report api        = {};
    std::string              api_commands;
    planned_result           owner;
    std::string              owner_commands;
    bool                     owner_session_fatal = false;
    uint64_t                 owner_tlb_flushes   = 0; // the session's count
};

both_runs run_both(const potal_stripe_gemm_desc & d) {
    both_runs   b;
    emu_session api;
    b.api_status   = potal_tiled_matmul_stripe(&api.session, &d, &b.api);
    b.api_commands = api.emu.commands;
    emu_session owner;
    b.owner               = run_planned_stripe_gemm(owner.session, d);
    b.owner_commands      = owner.emu.commands;
    b.owner_session_fatal = owner.session.ws.report.fatal;
    b.owner_tlb_flushes   = owner.session.ws.report.tlb_flushes;
    return b;
}

// Refused by both before any command: the C API with its own stage, the owner
// step with stage "preflight" (report.stage the same check), neither fatal,
// no TLB FLUSH, no LOOP_WS, no command at all, the session still usable.
bool refused_by_both(const potal_stripe_gemm_desc & d, std::string * stage = nullptr) {
    const both_runs b = run_both(d);
    const bool api = b.api_status != POTAL_OK && b.api.status == b.api_status && b.api.stage != nullptr &&
                     b.api.fatal == 0 && b.api.loop_ws == 0 && b.api.scale_writes == 0 && b.api_commands.empty();
    const bool owner = api && !b.owner.ok && b.owner.status == b.api_status && !b.owner.fatal &&
                       b.owner.stage != nullptr && std::string(b.owner.stage) == "preflight" &&
                       b.owner.report.stage != nullptr && std::string(b.owner.report.stage) == b.api.stage &&
                       b.owner.report.fatal == 0 && b.owner.tlb_flushes == 0 && b.owner.report.loop_ws == 0 &&
                       b.owner.report.scale_writes == 0 && b.owner_commands.empty() && b.owner_tlb_flushes == 0 &&
                       !b.owner_session_fatal;
    if (stage != nullptr) {
        *stage = b.api.stage != nullptr ? b.api.stage : "-";
    }
    return api && owner;
}

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

    // DIM x DIM x K32 with a whole plan takes the validated single-GEMM path.
    problem                      p(D, D, 32, 5, 0);
    potal_tile_plan              plan;
    emu_session                  es;
    const bool                   planned = potal_tile_select(D, D, 32, 32 / D, &plan) == POTAL_OK;
    const potal_stripe_gemm_desc d       = p.stripe(0, D, plan);
    potal_stripe_gemm_report     r;
    const potal_status           st = planned ? potal_tiled_matmul_stripe(&es.session, &d, &r) : POTAL_ERR_UNSUPPORTED;
    check("B validated DIM x DIM x K32 stripe through potal_gemm_ws_issue",
          st == POTAL_OK && r.validated_gemms == 1 && es.emu.gemms == 1 && p.equals(p.reference()));
}

// ---- C ------------------------------------------------------------------------

void test_rejections() {
    problem         p(40, 48, 96, 9, 0);
    potal_tile_plan plan;
    (void) potal_tile_select(40, 48, 96, 32 / D, &plan);
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
        { "n = 0", [](potal_stripe_gemm_desc & d, problem &) { d.n = 0; } },
        { "k = 0", [](potal_stripe_gemm_desc & d, problem &) { d.k = 0; } },
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
        { "tile_j 0", [](potal_stripe_gemm_desc & d, problem &) { d.tile_j = 0; } },
        { "tile_k 0", [](potal_stripe_gemm_desc & d, problem &) { d.tile_k = 0; } },
        { "tile_j beyond N", [](potal_stripe_gemm_desc & d, problem &) { d.tile_j = 4; } },
        { "tile_k beyond K", [](potal_stripe_gemm_desc & d, problem &) { d.tile_k = 8; } },
        { "tile_k not whole scale blocks",
          [](potal_stripe_gemm_desc & d, problem &) { d.scale_block_k = 2 * D; d.k = 4 * D; d.tile_k = 3; } },
        { "invalid carrier value",
          [](potal_stripe_gemm_desc & d, problem & p) {
              p.carriers.get()[(d.k / 32 - 1) * d.carrier_stride + d.n - 1] = 40000;
          } },
        { "C overlaps A", [](potal_stripe_gemm_desc & d, problem & p) { d.c = (int32_t *) (void *) p.a.get(); } },
        { "C overlaps B", [](potal_stripe_gemm_desc & d, problem & p) { d.c = (int32_t *) (void *) p.b.get(); } },
    };
    for (const mutation & mu : mutations) {
        potal_stripe_gemm_desc d = p.stripe(0, 40, plan);
        mu.apply(d, p);
        std::string stage;
        const bool  refused = refused_by_both(d, &stage);
        check(std::string("C refused without a command (C API and owner step, stage ") + stage + "): " + mu.name,
              refused);
        p.carriers.get()[(96 / 32 - 1) * p.cstride + 48 - 1] = 1; // undo the carrier mutation
    }
    {
        // A, B and C copied to chosen offsets of one buffer: exact overlap and
        // adjacency, and an over-limit operand placed above the other two, so
        // that each refusal can only come from the check its case names.
        problem         q(40, 48, 128, 41, 0); // A/B extents unchanged; C has a DIM-padded stride.
        potal_tile_plan qp;
        (void) potal_tile_select(40, 48, 128, 32 / D, &qp);
        buffer<int8_t>             mem(1 << 16);
        const std::vector<int32_t> want = q.reference();
        const auto place = [&](size_t a_at, size_t b_at, size_t c_at) {
            potal_stripe_gemm_desc d = q.stripe(0, 40, qp);
            std::memset(mem.get(), 0, 1 << 16);
            for (size_t i = 0; i < q.m; ++i) {
                std::memcpy(mem.get() + a_at + i * q.lda, q.a.get() + i * q.lda, q.k);
            }
            for (size_t kk = 0; kk < q.k; ++kk) {
                std::memcpy(mem.get() + b_at + kk * q.ldb, q.b.get() + kk * q.ldb, q.n);
            }
            d.a = mem.get() + a_at;
            d.b = mem.get() + b_at;
            d.c = (int32_t *) (void *) (mem.get() + c_at);
            return d;
        };
        const auto bytes = [](size_t rows, size_t cols, size_t ld, size_t elem) { return ((rows - 1) * ld + cols) * elem; };
        const auto disjoint = [](const void * x, size_t xb, const void * y, size_t yb) {
            return (uintptr_t) x + xb <= (uintptr_t) y || (uintptr_t) y + yb <= (uintptr_t) x;
        };
        // Accepted: C right behind A's last byte and right before B's first.
        const size_t c_bytes = bytes(q.m, q.n, q.ldc, sizeof(int32_t));
        const potal_stripe_gemm_desc adjacent = place(0, 5120 + c_bytes, 5120);
        bool                         runs     = c_bytes == (D == 32 ? 10176u : 7680u) &&
                            disjoint(adjacent.a, 5120, adjacent.c, c_bytes) &&
                            (uintptr_t) adjacent.a + 5120 == (uintptr_t) adjacent.c &&
                            (uintptr_t) adjacent.c + c_bytes == (uintptr_t) adjacent.b;
        for (const bool owner_step : { false, true }) {
            std::memset((void *) adjacent.c, 0, c_bytes);
            emu_session es;
            potal_stripe_gemm_report r;
            const bool ok = owner_step ? run_planned_stripe_gemm(es.session, adjacent).ok
                                       : potal_tiled_matmul_stripe(&es.session, &adjacent, &r) == POTAL_OK;
            for (size_t i = 0; ok && i < q.m; ++i) {
                for (size_t j = 0; j < q.n; ++j) {
                    runs = runs && adjacent.c[i * q.ldc + j] == want[i * q.n + j];
                }
            }
            runs = runs && ok;
        }
        check("C accepted: C adjacent to the end of A and the start of B (disjoint) runs == reference, both entry points",
              runs);
        const potal_stripe_gemm_desc in_a = place(0, 32768, 5120 - D * sizeof(int32_t));
        const potal_stripe_gemm_desc in_b = place(0, 5120 + c_bytes - 64, 5120);
        std::string                  stage_a, stage_b;
        check("C refused without a command (C API and owner step): C starts inside A's last row",
              !disjoint(in_a.a, 5120, in_a.c, c_bytes) && disjoint(in_a.b, 8176, in_a.c, c_bytes) &&
                  refused_by_both(in_a, &stage_a) && stage_a == "operands");
        check("C refused without a command (C API and owner step): C ends inside B's first row",
              !disjoint(in_b.b, 8176, in_b.c, c_bytes) && disjoint(in_b.a, 5120, in_b.c, c_bytes) &&
                  refused_by_both(in_b, &stage_b) && stage_b == "operands");
        // Over the 32-bit DRAM offset, placed above the other operands (no overlap).
        potal_stripe_gemm_desc big_a = place(40960, 16384, 0);
        big_a.lda                    = (size_t) 1 << 30;
        potal_stripe_gemm_desc big_b = place(0, 40960, 8192);
        big_b.ldb                    = (size_t) 1 << 30;
        potal_stripe_gemm_desc big_c = place(0, 8192, 40960);
        big_c.ldc                    = (size_t) 1 << 28;
        const size_t a_big = bytes(40, 128, big_a.lda, 1), b_big = bytes(128, 48, big_b.ldb, 1),
                     c_big = bytes(40, 48, big_c.ldc, 4);
        const struct {
            const char *                   name;
            const potal_stripe_gemm_desc & d;
            bool                           apart; // no overlap: only the DRAM-offset limit can refuse it
        } over[] = {
            { "A", big_a, a_big > ((uint64_t) 1 << 32) && disjoint(big_a.a, a_big, big_a.c, c_bytes) &&
                              disjoint(big_a.b, 8176, big_a.c, c_bytes) },
            { "B", big_b, b_big > ((uint64_t) 1 << 32) && disjoint(big_b.b, b_big, big_b.c, c_bytes) },
            { "C", big_c, c_big > ((uint64_t) 1 << 32) && disjoint(big_c.c, c_big, big_c.a, 5120) &&
                              disjoint(big_c.c, c_big, big_c.b, 8176) },
        };
        for (const auto & o : over) {
            std::string stage;
            check(std::string("C refused without a command (C API and owner step): ") + o.name +
                      " extent beyond the 32-bit DRAM offset, disjoint from the other operands",
                  o.apart && refused_by_both(o.d, &stage) && stage == "operands");
        }
    }
    {
        // Per-invocation capacity (potal_frozen_plan): POTAL_ERR_CAPACITY at
        // "tiles". DIM16 isolates each limit. DIM32 cannot isolate Scale:
        // for I,J,K >= 1, (I+J)*K*32 <= 4096 implies K*J < 128,
        // hence (K/ppb)*J < 128 < 256 for every admitted ppb >= 1.
        // Keep actual generated geometry and test that dominance explicitly.
        const size_t spad_k = D == 32 ? 65 : 258;
        problem         spad(D, D, spad_k * D, 23, 0);
        potal_tile_plan t = plan;
        t.tile_I = t.tile_J = 1;
        t.tile_K            = spad_k;
        const potal_stripe_gemm_desc d_spad = spad.stripe(0, D, t);
        // ACC: DIM16 8x8 needs 1024 rows > 512; DIM32 3x3 needs 288 > 256.
        const size_t acc_ij = D == 32 ? 3 : 8;
        problem         acc(acc_ij * D, acc_ij * D, 64, 17, 0);
        potal_tile_plan wide = plan;
        wide.tile_I = wide.tile_J = acc_ij;
        wide.tile_K               = 2;
        const potal_stripe_gemm_desc d_acc = acc.stripe(0, acc_ij * D, wide);
        for (const auto & c : { std::make_pair("scratchpad half (SPAD)", d_spad),
                                std::make_pair("accumulator half (ACC)", d_acc) }) {
            const both_runs b = run_both(c.second);
            check(std::string("C refused without a command (C API and owner step): tiles beyond the ") + c.first,
                  refused_by_both(c.second) && b.api_status == POTAL_ERR_CAPACITY && std::string(b.api.stage) == "tiles");
        }
        if (D == 32) {
            const size_t half = (size_t) POTAL_HW_BANK_NUM * POTAL_HW_BANK_ROWS / 2;
            check("C DIM32 physical SPAD strictly dominates Scale for every positive tile shape and ppb",
                  half == 4096 && POTAL_HW_SCALE_ENTRIES == 256 &&
                      half / D <= POTAL_HW_SCALE_ENTRIES &&
                      2 * spad_k * D > half && spad_k < POTAL_HW_SCALE_ENTRIES &&
                      acc_ij * acc_ij * D > POTAL_HW_ACC_ROWS / 2 &&
                      acc_ij * acc_ij <= POTAL_WORK_IDS_PER_LOOP && 2 * acc_ij * 2 * D <= half &&
                      2 * acc_ij <= POTAL_HW_SCALE_ENTRIES);
            // At I=J=1, K64 exactly fills SPAD; K65 exceeds it alone.
            t.tile_K = 64;
            problem edge(D, D, 2048, 29, 0);
            const auto edge_desc = edge.stripe(0, D, t);
            const both_runs fit = run_both(edge_desc);
            check("C DIM32 SPAD boundary control: I1/J1/K64 uses 4096 rows and 64 Scale slots, runs exactly",
                  fit.api_status == POTAL_OK && fit.owner.ok && edge.equals(edge.reference()) &&
                      fit.api.scale_writes == 64 * D && fit.api.scale_releases == 64 * D);
            // A real Scale-overflow descriptor is still refused, but SPAD
            // must also overflow; do not mislabel this as an isolated check.
            problem over(D, 2 * D, 130 * D, 30, 0);
            t.tile_J = 2;
            t.tile_K = 130;
            const auto over_desc = over.stripe(0, D, t);
            const both_runs over_run = run_both(over_desc);
            check("C DIM32 260 Scale slots necessarily exceed SPAD too; both entry points refuse without commands",
                  130 * 2 > POTAL_HW_SCALE_ENTRIES && (1 + 2) * 130 * D > half &&
                      refused_by_both(over_desc) && over_run.api_status == POTAL_ERR_CAPACITY &&
                      std::string(over_run.api.stage) == "tiles");
        } else {
            // Scale window: 16-wide scale blocks (1 pass each), 1 x 2 x 130 tiles
            // fit SPAD (6240 rows) and ACC but need 2 * 130 = 260 > 256 slots.
            problem          window(16, 32, 2080, 29, 0);
            buffer<uint32_t> narrow(2080 / 16 * window.cstride);
            potal_tile_plan  u = plan;
            u.tile_I = 1;
            u.tile_J = 2;
            u.tile_K = 130;
            potal_stripe_gemm_desc d_window = window.stripe(0, 16, u);
            d_window.carriers               = narrow.get();
            d_window.scale_block_k          = 16;
            const both_runs b = run_both(d_window);
            check("C DIM16 isolated ScaleMemory overflow refused without a command",
                  refused_by_both(d_window) && b.api_status == POTAL_ERR_CAPACITY && std::string(b.api.stage) == "tiles");
            // The same tiles with 32-wide blocks (2 passes per block) need 130
            // slots: they fit and run, so the refusal above is the window's.
            const both_runs fit_b = run_both(window.stripe(0, 16, u));
            check("C control: the same tiles with 32-wide scale blocks fit the window (130 slots) and run",
                  fit_b.api_status == POTAL_OK && fit_b.owner.ok);
        }
    }
    emu_session              es;
    potal_stripe_gemm_desc   d = p.stripe(0, 40, plan);
    potal_stripe_gemm_report r;
    // The owner step takes the descriptor by reference: a NULL descriptor
    // exists for the C API only (its NULL operands are refused above).
    check("C NULL report refused", potal_tiled_matmul_stripe(&es.session, &d, nullptr) == POTAL_ERR_INVALID_ARGUMENT);
    check("C NULL session refused", potal_tiled_matmul_stripe(nullptr, &d, &r) == POTAL_ERR_INVALID_ARGUMENT);
    check("C NULL descriptor refused", potal_tiled_matmul_stripe(&es.session, nullptr, &r) ==
                                           POTAL_ERR_INVALID_ARGUMENT);
    potal_stripe_session unbound = {};
    check("C session without ops refused", potal_tiled_matmul_stripe(&unbound, &d, &r) == POTAL_ERR_INVALID_ARGUMENT);
    const planned_result unbound_owner = run_planned_stripe_gemm(unbound, d);
    check("C owner step: session without ops refused, not fatal",
          unbound_owner.status == POTAL_ERR_INVALID_ARGUMENT && !unbound_owner.fatal &&
              std::string(unbound_owner.stage) == "preflight" && unbound_owner.tlb_flushes == 0);
    check("C refusals left the session usable", potal_tiled_matmul_stripe(&es.session, &d, &r) == POTAL_OK &&
                                                    p.equals(p.reference()));

    // Every refusal above on one owner session, then an accepted descriptor:
    // a refused preflight never disables the owner.
    emu_session owner;
    size_t      refused = 0;
    for (const mutation & mu : mutations) {
        potal_stripe_gemm_desc bad = p.stripe(0, 40, plan);
        mu.apply(bad, p);
        refused += !run_planned_stripe_gemm(owner.session, bad).ok;
        p.carriers.get()[(96 / 32 - 1) * p.cstride + 48 - 1] = 1;
    }
    std::fill(p.c.get(), p.c.get() + up(40, D) * p.ldc, 0);
    const planned_result after = run_planned_stripe_gemm(owner.session, d);
    check("C owner step: " + std::to_string(refused) + " refusals on one session, nothing issued, then the next "
          "descriptor runs",
          refused == sizeof(mutations) / sizeof(mutations[0]) && after.ok && after.tlb_flushes == 1 &&
              owner_invocations(owner.emu.commands) == (long) after.report.loop_ws && p.equals(p.reference()));
}

// ---- D ------------------------------------------------------------------------

void test_generation_wrap() {
    problem         p(D, D, 32, 11, 0);
    potal_tile_plan plan;
    (void) potal_tile_select(D, D, 32, 32 / D, &plan);
    emu_session es;
    bool        sequence = true;
    uint32_t    previous = 0;
    const int   calls    = 2 * (int) POTAL_GENERATION_MAX + 7;
    for (int i = 0; i < calls && sequence; ++i) {
        const potal_stripe_gemm_desc d = p.stripe(0, D, plan);
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
    problem         wide(D, POTAL_GENERATION_MAX * D, 32, 19, 0);
    potal_tile_plan one_tile = plan;
    one_tile.tile_I = one_tile.tile_J = 1;
    one_tile.tile_K                   = 32 / D;
    emu_session                  es255;
    const potal_stripe_gemm_desc d0 = p.stripe(0, D, plan);
    const potal_stripe_gemm_desc d1 = wide.stripe(0, D, one_tile);
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
    (void) potal_tile_select(40, 48, 96, 32 / D, &plan);

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
    problem                p(D, D, 32, 3, 0);
    potal_tile_plan        plan;
    (void) potal_tile_select(D, D, 32, 32 / D, &plan);
    const potal_stripe_gemm_desc d = p.stripe(0, D, plan);
    potal_stripe_gemm_report     r;
    const potal_status           st = created ? potal_tiled_matmul_stripe(s, &d, &r) : POTAL_OK;
    check("F host build: a valid stripe GEMM stops at its first hardware command (NO_HARDWARE)",
          created && st == POTAL_ERR_NO_HARDWARE && r.stage != nullptr && std::string(r.stage) == "scale_write" &&
              r.loop_ws == 0);
    potal_stripe_session_destroy(s);

    // The owner step on the real potal.h binding: a refused descriptor never
    // reaches potal_tlb_flush() (which answers NO_HARDWARE on the host), an
    // accepted one reaches it first.
    potal_stripe_session * hw = nullptr;
    const bool             made = potal_stripe_session_create(&hw) == POTAL_OK && hw != nullptr;
    potal_stripe_gemm_desc bad  = d;
    bad.tile_k                  = 3;
    const planned_result refused = made ? run_planned_stripe_gemm(*hw, bad) : planned_result{};
    check("F owner step, hardware binding: a refused descriptor stops before the TLB FLUSH (no NO_HARDWARE)",
          made && refused.status == POTAL_ERR_INVALID_ARGUMENT && std::string(refused.stage) == "preflight" &&
              !refused.fatal && refused.tlb_flushes == 0 && !hw->ws.report.fatal);
    const planned_result first = made ? run_planned_stripe_gemm(*hw, d) : planned_result{};
    check("F owner step, hardware binding: an accepted descriptor's first command is the TLB FLUSH",
          made && first.status == POTAL_ERR_NO_HARDWARE && std::string(first.stage) == "tlb_flush" && first.fatal &&
              first.report.loop_ws == 0 && first.report.scale_writes == 0 && hw->ws.report.fatal);
    potal_stripe_session_destroy(hw);
#else
    std::printf("[SKIP] F hardware session: on RISC-V it would issue RoCC commands from an unpinned thread\n");
#endif
}

// ---- G ------------------------------------------------------------------------

// The owner step (the issuer's planned GEMM after its affinity check) on
// accepted descriptors: the command order, and the same work and result as
// the C API on a session of its own.
void owner_matches_api(const char * name, size_t M, size_t N, size_t K, size_t rows, int carrier_mode) {
    problem         p(M, N, K, (uint32_t) (M * 7 + N + K), carrier_mode);
    potal_tile_plan plan;
    const bool      planned = potal_tile_select(M, N, K, 32 / D, &plan) == POTAL_OK;
    // The last stripe of the plan: a tail when M is not a multiple of its rows.
    const size_t                 row = planned ? (M - 1) / (plan.tile_I * D) * (plan.tile_I * D) : 0;
    const potal_stripe_gemm_desc d    = p.stripe(row, std::min(rows, M - row), plan);
    const std::vector<int32_t>   want = p.reference();
    both_runs                    b;
    // Each run writes the stripe's rows of C from zero; both must be the reference.
    const auto rows_ok = [&] {
        bool ok = true;
        for (size_t i = row; i < row + d.m; ++i) {
            for (size_t j = 0; j < N; ++j) {
                ok = ok && p.c.get()[i * p.ldc + j] == want[i * N + j];
                p.c.get()[i * p.ldc + j] = 0;
            }
        }
        return ok;
    };
    bool same_c = planned;
    if (planned) {
        emu_session api;
        b.api_status   = potal_tiled_matmul_stripe(&api.session, &d, &b.api);
        b.api_commands = api.emu.commands;
        same_c         = rows_ok();
        emu_session owner;
        b.owner             = run_planned_stripe_gemm(owner.session, d);
        b.owner_commands    = owner.emu.commands;
        b.owner_tlb_flushes = owner.session.ws.report.tlb_flushes;
        same_c              = same_c && rows_ok();
    }
    const long invocations = owner_invocations(b.owner_commands);
    check(std::string("G owner step ") + name + " (" + std::to_string(d.m) + "x" + std::to_string(N) + "x" +
              std::to_string(K) + ", tile_i " + std::to_string(b.owner.report.tile_i) +
              "): TLB FLUSH, FENCE, then per invocation SCALE_WRITE..LOOP_WS..FENCE..SCALE_RELEASE",
          planned && b.owner.ok && b.owner.tlb_flushes == 1 && b.owner_tlb_flushes == 1 && invocations > 0 &&
              (uint64_t) invocations == b.owner.report.loop_ws);
    check(std::string("G owner step ") + name + ": same LOOP_WS/FENCE/WRITE/RELEASE as the C API, C == reference",
          b.api_status == POTAL_OK && b.owner.report.loop_ws == b.api.loop_ws &&
              b.owner.report.fences == b.api.fences && b.owner.report.fences == b.owner.report.loop_ws &&
              b.owner.report.scale_writes == b.api.scale_writes &&
              b.owner.report.scale_releases == b.api.scale_releases &&
              b.owner.report.validated_gemms == b.api.validated_gemms && b.owner.report.tile_i == b.api.tile_i &&
              b.owner_commands.substr(2) == b.api_commands && same_c);
}

void test_owner_step() {
    owner_matches_api("tail stripe", 200, 312, 128, 200, 0);
    owner_matches_api("saturating carriers", 77, 48, 256, 77, 2);
    owner_matches_api("K chunks", 48, 160, 4096, 48, 0);
    owner_matches_api("zero carriers, one stripe", 150, 40, 256, 150, 1);
    owner_matches_api("validated DIM x DIM x K32", D, D, 32, D, 0);

    problem         p(40, 48, 96, 31, 0);
    potal_tile_plan plan;
    (void) potal_tile_select(40, 48, 96, 32 / D, &plan);
    const potal_stripe_gemm_desc d = p.stripe(0, 40, plan);
    {
        emu_session es;
        es.emu.fail_tlb            = true;
        const planned_result first = run_planned_stripe_gemm(es.session, d);
        check("G a failed TLB FLUSH is fatal: stage tlb_flush, no SCALE_WRITE/LOOP_WS, the session disabled",
              first.status == POTAL_HW_BAD_COMMAND && first.fatal && std::string(first.stage) == "tlb_flush" &&
                  first.report.fatal == 1 && first.tlb_flushes == 0 && es.emu.commands == "T" &&
                  es.session.ws.report.fatal);
        es.emu.fail_tlb             = false;
        const planned_result second = run_planned_stripe_gemm(es.session, d);
        check("G after it the session issues nothing, not even a TLB FLUSH (session_disabled, fatal)",
              !second.ok && second.fatal && std::string(second.stage) == "session_disabled" &&
                  es.emu.commands == "T");
    }
    {
        emu_session es;
        es.emu.fail_fence = true;
        const planned_result failed = run_planned_stripe_gemm(es.session, d);
        check("G a failed TLB FENCE is fatal at tlb_flush too", failed.fatal && std::string(failed.stage) ==
                                                                                 "tlb_flush" &&
                                                                    es.emu.commands == "TF");
    }
    {
        // Hardware failures inside the GEMM, after the TLB FLUSH + FENCE: fatal,
        // with the leaf's stage, exactly as through the C API; the session is
        // disabled and the next owner step issues nothing.
        const struct {
            const char * stage;
            void (*arm)(emulator &, bool owner_step);
        } injections[] = {
            { "scale_write", [](emulator & e, bool) { e.fail_write_at = 0; } },
            { "issue", [](emulator & e, bool) { e.fail_issue = true; } },
            // The owner step's FENCE 0 is the TLB FENCE: fail its first GEMM FENCE.
            { "fence", [](emulator & e, bool owner_step) { e.fail_fence_at = owner_step ? 1 : 0; } },
            { "scale_release", [](emulator & e, bool) { e.fail_release = true; } },
        };
        for (const auto & inj : injections) {
            emu_session api, owner;
            inj.arm(api.emu, false);
            inj.arm(owner.emu, true);
            potal_stripe_gemm_report ar;
            const potal_status       ast      = potal_tiled_matmul_stripe(&api.session, &d, &ar);
            const planned_result     o        = run_planned_stripe_gemm(owner.session, d);
            const size_t             commands = owner.emu.commands.size();
            const planned_result     next     = run_planned_stripe_gemm(owner.session, d);
            check(std::string("G a ") + inj.stage + " failure inside the owner step's GEMM: fatal at " + inj.stage +
                      " as through the C API, the session disabled, nothing issued after it",
                  ast != POTAL_OK && ar.fatal == 1 && ar.stage != nullptr && std::string(ar.stage) == inj.stage &&
                      o.status == ast && o.fatal && o.report.fatal == 1 && o.stage != nullptr &&
                      std::string(o.stage) == inj.stage && o.tlb_flushes == 1 &&
                      owner.emu.commands.compare(0, 2, "TF") == 0 && owner.session.ws.report.fatal && next.fatal &&
                      std::string(next.stage) == "session_disabled" && owner.emu.commands.size() == commands);
        }
    }
    {
        // A GEMM fence failure disables the session (here through the C API,
        // which issues no TLB FLUSH); the owner step then issues nothing.
        emu_session es;
        es.emu.fail_fence = true;
        potal_stripe_gemm_report broken;
        const potal_status       st = potal_tiled_matmul_stripe(&es.session, &d, &broken);
        es.emu.fail_fence           = false;
        const size_t         commands = es.emu.commands.size();
        const planned_result refused  = run_planned_stripe_gemm(es.session, d);
        check("G a GEMM fence failure disables the session; the next owner step issues no command, no TLB FLUSH",
              st == POTAL_HW_BAD_COMMAND && broken.fatal == 1 && std::string(broken.stage) == "fence" &&
                  refused.fatal && std::string(refused.stage) == "session_disabled" && refused.tlb_flushes == 0 &&
                  es.emu.commands.size() == commands && es.emu.tlb_flushes == 0);
    }
    {
        // Generation wrap through the owner step: 2 * 255 + 7 calls of one invocation each.
        problem         q(D, D, 32, 37, 0);
        potal_tile_plan one;
        (void) potal_tile_select(D, D, 32, 32 / D, &one);
        emu_session                  es;
        const potal_stripe_gemm_desc g     = q.stripe(0, D, one);
        const int                    calls = 2 * (int) POTAL_GENERATION_MAX + 7;
        bool                         ok    = true;
        for (int i = 0; i < calls && ok; ++i) {
            const planned_result r = run_planned_stripe_gemm(es.session, g);
            ok = r.ok && r.tlb_flushes == 1 && r.report.loop_ws == 1 &&
                 r.report.last_generation == (uint32_t) (i % (int) POTAL_GENERATION_MAX) + 1;
        }
        const auto & gens = es.emu.loop_generations;
        check("G owner step: generation 1..255 then 1 again over " + std::to_string(calls) +
                  " LOOP_WS, one TLB FLUSH each, WRITE == RELEASE",
              ok && gens.size() == (size_t) calls && gens[POTAL_GENERATION_MAX - 1] == POTAL_GENERATION_MAX &&
                  gens[POTAL_GENERATION_MAX] == 1 && es.emu.tlb_flushes == (uint64_t) calls &&
                  es.session.ws.report.tlb_flushes == (uint64_t) calls && es.emu.writes == es.emu.releases &&
                  es.emu.scales.empty() && q.equals(q.reference()));
    }
}

} // namespace

int main() {
    std::printf("potal-stripe-api-test DIM=%u backend=EMULATED (no hardware)\n", (unsigned) POTAL_HW_DIM);
    test_c_caller();
    test_frozen_plans();
    test_rejections();
    test_generation_wrap();
    test_session();
    test_hardware_session();
    test_owner_step();
    if (failures != 0) {
        std::printf("POTAL_STRIPE_API_TEST FAIL (%d)\n", failures);
        return 1;
    }
    std::printf("POTAL_STRIPE_API_TEST PASS\n");
    return 0;
}
