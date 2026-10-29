// Hardware-free tests of the PoTal tiled path (host and target):
//
//   A  tile selection: potal_tile_select() vs the gemmini.h tiled_matmul_auto
//      policy (verbatim port below), PoTal limits, capacity reporting
//   B  tiled execution on an emulator of the RTL LOOP_WS/SCU/Sat32 semantics
//      (LoopMatmul, PoTalSCU, PoTalScaleMemory, AccPipe) vs the independent
//      integer golden reference_gemm_i32(): multi-I/J/K tiles, K chunks,
//      padding, zero carriers, saturation, scale slot release, fail-stop
//   C  Q8_HP1 validation and unpacking (Gemmini weight reader, HP1 SCU domain)
//   D  the configured activation dispatcher (GGML_GEMMINI_ACTIVATION_QUANT):
//      metadata kind and route, never replaced by TENSOR; TENSOR scale rules
//      when TENSOR with compensation OFF is configured
//   E  the configured pipeline end to end on the emulator vs the golden
//   G  every activation algorithm (TENSOR, TOKEN, BLOCK, STRIPE, EXSIA) through
//      its own quantizer and route: integer-domain golden from the raw
//      metadata, and the algorithm's own dequantization x dequantized HP1
//   H  error compensation as configured (GGML_GEMMINI_ENABLE_RMD): OFF captures
//      and executes nothing; ON captures, corrects and merges (cpu_direct),
//      and a ws_packet payload fails instead of being dropped
//   I  scale generation wrap over more than POTAL_GENERATION_MAX invocations
//   F  GGML contract: supports_op, no CPU fallback for Q8_HP1, no fake success
//
// The emulator is a test double for the accelerator; passing here proves the
// software contract, not the hardware. Hardware: potal-hp1-smoke on FireSim.

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"
#include "ggml-potal-buffer.hpp"
#include "ggml-potal.h"
#include "matmul/dense.hpp"
#include "matmul/options.hpp"
#include "matmul/tiling.hpp"
#include "matmul/weight_cache.hpp"
#include "ops.hpp"
#include "potal.h"
#include "potal_hw_params.h"
#include "support/potal-emulator.hpp"

// Unmodified Gemmini quantizers, metadata and residual code (shared sources).
#include "quants/act/block/block.hpp"
#include "quants/act/dispatch.hpp"
#include "quants/act/exsia/exsia.hpp"
#include "quants/act/quantize.hpp"
#include "quants/act/stripe/stripe.hpp"
#include "quants/act/tensor/tensor.hpp"
#include "quants/act/token/token.hpp"
#include "quants/common/hp1_scu.hpp"

#include "ggml-quants.h" // block_q8_hp1 (ggml-common.h), Q8_HP1 reference quant/dequant

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <variant>
#include <vector>

using namespace ggml::potal;
namespace hp1 = ggml::potal::quants::weight::hp1;
namespace gq  = ggml::gemmini::quants;

// DIM is the <gemmini_params.h> macro, i.e. the selected PoTal parameter header.
static_assert(DIM == POTAL_HW_DIM, "gemmini_params.h and potal_hw_params.h disagree on DIM");

namespace {

int failures = 0;

void check(const std::string & name, bool ok) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", name.c_str());
    failures += ok ? 0 : 1;
}

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
    check("A K not whole scale blocks rejected", potal_tile_select(4, DIM, 3 * DIM, 2, &plan) == POTAL_ERR_INVALID_ARGUMENT);
    check("A zero dimension rejected", potal_tile_select(0, 16, 32, 2, &plan) == POTAL_ERR_INVALID_ARGUMENT);
}

// ---- B: emulator of the RTL LOOP_WS + SCU + Sat32 semantics ---------------

using potal_test::emulator;

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
            o.own.b[k * o.ldb + n] = (int8_t) q(rng);
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
            o.own.carriers[b * o.cstride + n] = c;
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
        { DIM, DIM, 32, 0 }, // the validated single GEMM on either HP1 geometry
        { DIM, DIM, 96, 1 }, // beyond the bounded two-DIM legacy API: LOOP_WS
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
        if (s.m == DIM && s.n == DIM && s.k == 32) {
            check("B validated shape issued through potal_gemm_ws_issue",
                  emu.gemms == 1 && session.report.validated_gemms == 1);
        }
        if (s.m == DIM && s.n == DIM && s.k == 96) {
            check("B K beyond two DIM uses LOOP_WS, not the bounded single GEMM",
                  emu.gemms == 0 && session.report.validated_gemms == 0 && emu.loops == 1);
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
        o.own.carriers[3] = 32768;
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

ggml_gemmini_args_t hp1_args(const std::vector<block_q8_hp1> & blocks, size_t N, size_t K) {
    ggml_gemmini_args_t args;
    args.weight_format         = ggml_gemmini_args_t::im2p_weight_format_t::q8_hp1;
    args.block_size_k          = QK8_HP;
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

    const auto expect = [&](const char * name, hp1::status want, gq::wreader::WeightReaderStatus want_reader,
                            auto mutate) {
        std::vector<block_q8_hp1> bad = blocks;
        mutate(bad);
        hp1::stats                          ignored;
        gq::wreader::WeightReaderStatus reader = gq::wreader::WeightReaderStatus::Success;
        const hp1::status got = hp1::unpack(hp1_args(bad, N, K), B.data(), ldb, carriers.data(), cs, scale.data(),
                                            &ignored, &reader);
        if (got != want || reader != want_reader) {
            std::fprintf(stderr, "[HP1_DIAG] case=%s got=%s expected=%s reader=%s expected_reader=%s\n", name,
                         hp1::status_name(got), hp1::status_name(want),
                         gq::wreader::weight_reader_status_name(reader),
                         gq::wreader::weight_reader_status_name(want_reader));
        }
        check(std::string("C HP1 rejects ") + name, got == want && reader == want_reader);
    };
    using RS = gq::wreader::WeightReaderStatus;
    expect("non-zero padding (reader)", hp1::status::metadata, RS::InvalidMetadata,
           [](auto & b) { b[9].padding[1] = 1; });
    expect("negative exponent (reader)", hp1::status::metadata, RS::InvalidMetadata, [](auto & b) { b[4].m = -1; });
    expect("NaN channel scale (reader)", hp1::status::metadata, RS::InvalidMetadata,
           [](auto & b) { b[2].channel_scale = std::nanf(""); });
    expect("Inf channel scale (reader)", hp1::status::metadata, RS::InvalidMetadata,
           [](auto & b) { b[3].channel_scale = std::numeric_limits<float>::infinity(); });
    expect("carrier below range, m = INT16_MIN + 1 (reader)", hp1::status::metadata, RS::InvalidMetadata,
           [](auto & b) { b[6].m = INT16_MIN + 1; });
    expect("accepts m = 0 and m = 32767 carriers", hp1::status::ok, RS::Success, [](auto & b) {
        b[0].m = 0;
        b[1].m = 32767;
    });
    expect("channel scale mismatch inside a row", hp1::status::scale_mismatch, RS::Success,
           [](auto & b) { b[1].channel_scale *= 2.0f; });
    expect("negative channel scale", hp1::status::channel_scale, RS::Success,
           [](auto & b) { for (size_t i = 0; i < 4; ++i) b[i].channel_scale = -1.0f; });
    expect("zero channel scale with a non-zero block", hp1::status::channel_scale, RS::Success,
           [](auto & b) { b[5 * 4 + 1].m = 0; });
    hp1::stats        ignored;
    ggml_gemmini_args_t short_k = hp1_args(blocks, N, K);
    short_k.K                   = 100;
    const hp1::status   short_s = hp1::unpack(short_k, B.data(), ldb, carriers.data(), cs, scale.data(), &ignored);
    check("C HP1 rejects K not a multiple of 32 (storage contract)",
          short_s == hp1::status::route || short_s == hp1::status::metadata);
    uint32_t carrier = 0;
    check("C carrier encoding (hp1_scu.hpp)",
          gq::hp1::encode_carrier(0, carrier) && carrier == 0 && gq::hp1::encode_carrier(32767, carrier) &&
              carrier == 32767 && gq::hp1::encode_carrier(INT16_MIN, carrier) &&
              carrier == POTAL_CARRIER_ZERO && !gq::hp1::encode_carrier(-5, carrier));
}

// ---- shared fixtures for D, E, G, H ------------------------------------------

namespace act = gq::act;
using ResidualRoute = ggml::gemmini::residual::ResidualRoute;

enum class algo { tensor, token, block, stripe, exsia };

const char * algo_name(algo a) {
    switch (a) {
        case algo::tensor: return "TENSOR";
        case algo::token:  return "TOKEN";
        case algo::block:  return "BLOCK";
        case algo::stripe: return "STRIPE";
        case algo::exsia:  return "EXSIA";
    }
    return "?";
}

act::MetaKind kind_of(algo a) {
    switch (a) {
        case algo::tensor: return act::MetaKind::tensor;
        case algo::token:  return act::MetaKind::token;
        case algo::block:  return act::MetaKind::block;
        case algo::stripe: return act::MetaKind::stripe;
        case algo::exsia:  return act::MetaKind::exsia;
    }
    return act::MetaKind::none;
}

algo configured_algo() {
    using A = options::ActivationQuantAlgo;
    switch (options::activation_quant) {
        case A::EXSIA:  return algo::exsia;
        case A::TENSOR: return algo::tensor;
        case A::TOKEN:  return algo::token;
        case A::BLOCK:  return algo::block;
        case A::STRIPE: return algo::stripe;
    }
    return algo::tensor;
}

// One F32 x Q8_HP1 MUL_MAT with staged weights and args as the backend sets them.
struct mm_case {
    ggml_context_ptr ctx;
    ggml_tensor *    w = nullptr;
    ggml_tensor *    x = nullptr;
    ggml_tensor *    y = nullptr;
    hp1_shape        s = {};
    hp1_operands     o;
    ggml_gemmini_args_t args;
    hp1::stats       ws;
    bool             ok = false;

    // xf: M x K activations; weights N x K random with per-block magnitudes (distinct exponents).
    mm_case(int64_t M, int64_t N, int64_t K, const std::vector<float> & xf, ResidualRoute route, uint32_t seed) {
        ggml_init_params ip = { (size_t) (M * K + N * K) * 8 + 4 * 1024 * 1024, nullptr, false };
        ctx.reset(ggml_init(ip));
        w = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_HP1, K, N);
        x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, K, M);
        y = ggml_mul_mat(ctx.get(), w, x);
        std::mt19937                    rng(seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float>              wf((size_t) (N * K));
        const float                     mag[] = { 0.02f, 0.08f, 0.3f, 0.01f };
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t k = 0; k < K; ++k) {
                wf[n * K + k] = nd(rng) * mag[(k / 32 + n) % 4];
            }
        }
        ggml_quantize_chunk(GGML_TYPE_Q8_HP1, wf.data(), w->data, 0, N, K, nullptr);
        std::memcpy(x->data, xf.data(), xf.size() * sizeof(float));
        potal_tile_plan plan;
        ok = check_mul_mat_hp1(y, true, &s) == GGML_POTAL_REJECT_NONE &&
             potal_tile_select((size_t) M, (size_t) N, (size_t) K, hp1::block_k / DIM, &plan) == POTAL_OK &&
             o.allocate(s) && init_args(w, y, s, "gpt2", "test", route, plan, args) &&
             unpack_weights(args, o, &ws) == hp1::status::ok;
    }

    const float * out() const { return static_cast<const float *>(y->data); }
};

std::vector<float> random_activations(int64_t M, int64_t K, uint32_t seed, float outlier = 0.0f) {
    std::mt19937                    rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float>              v((size_t) (M * K));
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            // Row and K-group magnitudes differ, so the algorithms choose different scales.
            v[m * K + k] = nd(rng) * (1.0f + 0.5f * (float) (m % 3)) * (k / 32 % 2 ? 4.0f : 1.0f);
        }
    }
    if (outlier != 0.0f) {
        for (int64_t m = 0; m < M; m += 2) {
            v[m * K + (m * 7) % K] = outlier * (m % 4 ? -1.0f : 1.0f);
        }
    }
    return v;
}

// The unmodified per-algorithm quantizer, exactly as quants/act/dispatch.cpp calls it.
bool quantize_with(algo a, const ggml_tensor * x, ggml_gemmini_args_t & args, hp1_operands & o) {
    gq::reset_activation_quant_state(args);
    bool ok = false;
    switch (a) {
        case algo::tensor: args.act_quant.storage().emplace<act::tensor::Meta>(); ok = act::tensor::quantize(x, args); break;
        case algo::token:  args.act_quant.storage().emplace<act::token::Meta>();  ok = act::token::quantize(x, args);  break;
        case algo::block:  args.act_quant.storage().emplace<act::block::Meta>();  ok = act::block::quantize(x, args);  break;
        case algo::stripe: args.act_quant.storage().emplace<act::stripe::Meta>(); ok = act::stripe::quantize(x, args); break;
        case algo::exsia: {
            auto &          meta = args.act_quant.storage().emplace<act::exsia::Meta>();
            act::exsia::ExSIA q;
            ok = q.run(meta, x, args, args.exsia_stripe_ready_sink);
            break;
        }
    }
    const int8_t * codes = args.A;
    if (!ok || codes == nullptr) {
        return false;
    }
    for (size_t m = 0; m < args.I; ++m) {
        std::memcpy(o.a + m * o.lda, codes + m * args.K, args.K);
    }
    return true;
}

// The algorithm's own dequantization (residuals included when it captured any).
bool dequantize_with(algo a, const ggml_gemmini_args_t & args, std::vector<float> & out) {
    out.assign(args.I * args.K, 0.0f);
    switch (a) {
        case algo::tensor: return act::tensor::dequantize_activation(out.data(), args.K, 1, args.I, args.K, args);
        case algo::token:  return act::token::dequantize_activation(out.data(), args.K, 1, args.I, args.K, args);
        case algo::block:  return act::block::dequantize_activation(out.data(), args.K, 1, args.I, args.K, args);
        case algo::stripe: return act::stripe::dequantize_activation(out.data(), args.K, 1, args.I, args.K, args);
        case algo::exsia:  return act::exsia::dequantize_activation(out.data(), args.K, 1, args.I, args.K, args);
    }
    return false;
}

// Activation scale of element (m, k) read directly from the metadata the
// quantizer stored (independent of ActivationMetadataView and the route code).
bool raw_scale(const ggml_gemmini_args_t & args, size_t m, size_t k, double & scale) {
    const auto & st  = args.act_quant.storage();
    const size_t rps = args.activation_rows_per_stripe;
    if (const auto * t = std::get_if<act::tensor::Meta>(&st)) {
        scale = t->scale;
    } else if (const auto * t = std::get_if<act::token::Meta>(&st)) {
        if (m >= t->scales.size()) return false;
        scale = t->scales[m];
    } else if (const auto * t = std::get_if<act::stripe::Meta>(&st)) {
        if (rps == 0 || m / rps >= t->scales.size()) return false;
        scale = t->scales[m / rps];
    } else if (const auto * t = std::get_if<act::exsia::Meta>(&st)) {
        if (rps == 0 || m / rps >= t->theta.size()) return false;
        scale = std::ldexp(1.0, t->theta[m / rps]);
    } else if (const auto * t = std::get_if<act::block::Meta>(&st)) {
        const size_t bpr = (t->cols + 31) / 32;
        if (m * bpr + k / 32 >= t->scales.size()) return false;
        scale = t->scales[m * bpr + k / 32];
    } else {
        return false;
    }
    return std::isfinite(scale) && scale > 0.0;
}

// Integer-domain golden of a route: per 32-wide K group, the SCU/Sat32 golden
// C_g; constant-along-K routes use the whole-K golden instead (that is what the
// ACC accumulates).
bool route_golden(const route_plan & route, const mm_case & c, std::vector<float> & want) {
    const ggml_gemmini_args_t & a = c.args;
    const hp1_operands &        o = c.o;
    want.assign(a.I * a.J, 0.0f);
    std::vector<int32_t> g(o.m_pad * o.ldc, 0);
    if (!route.k_groups) {
        reference_gemm_i32(o.a, o.lda, o.b, o.ldb, o.carriers, o.cstride, hp1::block_k, a.I, a.J, a.K, g.data(), o.ldc);
        for (size_t m = 0; m < a.I; ++m) {
            double s = 0;
            if (!raw_scale(a, m, 0, s)) return false;
            for (size_t n = 0; n < a.J; ++n) {
                want[m * a.J + n] = (float) ((double) g[m * o.ldc + n] * s * (double) o.scale[n]);
            }
        }
        return true;
    }
    std::vector<double> acc(a.I * a.J, 0.0);
    for (size_t k0 = 0; k0 < a.K; k0 += 32) {
        reference_gemm_i32(o.a + k0, o.lda, o.b + k0 * o.ldb, o.ldb, o.carriers + (k0 / 32) * o.cstride, o.cstride,
                           hp1::block_k, a.I, a.J, 32, g.data(), o.ldc);
        for (size_t m = 0; m < a.I; ++m) {
            double s = 0;
            if (!raw_scale(a, m, k0, s)) return false;
            for (size_t n = 0; n < a.J; ++n) {
                acc[m * a.J + n] += (double) g[m * o.ldc + n] * s * (double) o.scale[n];
            }
        }
    }
    for (size_t i = 0; i < acc.size(); ++i) {
        want[i] = (float) acc[i];
    }
    return true;
}

// sum_k xhat[m][k] * dequant(W)[n][k] in double.
std::vector<double> fp_reference(const mm_case & c, const std::vector<float> & xhat) {
    const size_t I = c.args.I, J = c.args.J, K = c.args.K;
    std::vector<double> r(I * J, 0.0);
    std::vector<float>  row(K);
    for (size_t n = 0; n < J; ++n) {
        dequantize_row_q8_hp1(static_cast<const block_q8_hp1 *>(c.w->data) + n * (K / 32), row.data(), (int64_t) K);
        for (size_t m = 0; m < I; ++m) {
            double v = 0;
            for (size_t k = 0; k < K; ++k) {
                v += (double) xhat[m * K + k] * row[k];
            }
            r[m * J + n] = v;
        }
    }
    return r;
}

double rel_l2(const float * got, const std::vector<double> & ref) {
    double e = 0, n = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        e += ((double) got[i] - ref[i]) * ((double) got[i] - ref[i]);
        n += ref[i] * ref[i];
    }
    return n > 0 ? std::sqrt(e / n) : std::sqrt(e);
}

// The emulator as the job runner (a fresh session per runner).
struct emu_runner {
    emulator         emu;
    potal_ws_ops     ops     = emu.ops();
    potal_ws_session session = { &ops, 0, {} };
    tiled_runner     fn() {
        return [this](const tiled_job & job) { return run_tiled_in_session(job, session); };
    }
};

// ---- D: the configured activation dispatcher ------------------------------------

void test_configured_dispatcher() {
    const algo cfg = configured_algo();
    std::printf("     configured: GGML_GEMMINI_ACTIVATION_QUANT=%s GGML_GEMMINI_ENABLE_RMD=%s\n",
                options::activation_quant_name(), options::compensation_name);
    const std::vector<float> xf = random_activations(9, 128, 5);
    mm_case                  c(9, 32, 128, xf, ResidualRoute::cpu_direct, 11);
    const bool               q  = c.ok && quantize_activations(c.x, c.args, c.o);
    route_plan               route;
    check(std::string("D dispatcher runs the configured quantizer, metadata kind ") + algo_name(cfg),
          q && c.args.act_quant.kind() == kind_of(cfg));
    check("D configured non-TENSOR algorithm is not replaced by TENSOR",
          cfg == algo::tensor || c.args.act_quant.kind() != act::MetaKind::tensor);
    check("D route follows the metadata", q && select_route(c.args, route) &&
                                              route.activation == static_cast<act_route>(cfg == algo::tensor ? 0 :
                                                                                         cfg == algo::token  ? 1 :
                                                                                         cfg == algo::stripe ? 2 :
                                                                                         cfg == algo::exsia  ? 3 : 4));
    route_plan none;
    ggml_gemmini_args_t empty;
    check("D no metadata -> no route", !select_route(empty, none));

    if constexpr (options::activation_quant == options::ActivationQuantAlgo::TENSOR && !options::compensation) {
        // Gemmini TENSOR with RMD OFF: scale = max_abs / 127 over the whole tensor.
        const int64_t            K  = 32;
        std::vector<float>       x2(2 * K, 0.0f);
        x2[0] = 0.5f;
        x2[K] = 12.7f;
        x2[K + 3] = -2.5f;
        mm_case    t(2, 16, K, x2, ResidualRoute::cpu_direct, 3);
        const bool ok    = t.ok && quantize_activations(t.x, t.args, t.o);
        const auto * m   = std::get_if<act::tensor::Meta>(&t.args.act_quant.storage());
        const float scale = (float) (12.7 / (float) 127);
        check("D TENSOR scale = max_abs / 127 over the whole tensor (RMD OFF)", ok && m && m->scale == scale);
        check("D TENSOR row with small values uses the tensor scale (0.5 -> 5)", ok && t.args.A.get(0, 0) == 5);
        check("D TENSOR codes staged into the DMA buffer", ok && t.o.a[0] == 5 && t.o.a[t.o.lda] == 127);
    }
}

// ---- E: configured pipeline end to end on the emulator -----------------------

void test_end_to_end() {
    const int64_t            M = 7, N = 48, K = 256;
    const std::vector<float> xf = random_activations(M, K, 99);
    mm_case                  c(M, N, K, xf, ResidualRoute::cpu_direct, 98);
    route_plan               route;
    const bool prepared = c.ok && quantize_activations(c.x, c.args, c.o) && select_route(c.args, route);
    std::vector<float> want;
    const bool         golden = prepared && route_golden(route, c, want);
    emu_runner         er;
    const route_result r = prepared ? execute_route(route, c.args, c.o, er.fn()) : route_result{};
    bool exact = golden && r.ok;
    for (size_t i = 0; exact && i < want.size(); ++i) {
        exact = c.out()[i] == want[i];
    }
    check(std::string("E configured route ") + (route.name ? route.name : "-") +
              ": dst == integer-domain golden exactly, through potal_tiled_matmul_auto",
          exact && r.jobs == (route.k_groups ? (uint64_t) (K / 32) : 1u) && r.work.invocations > 0);
    check("E every invocation fenced and its scale slots released",
          r.work.fences == r.work.invocations && r.work.scale_writes == r.work.scale_releases && er.emu.scales.empty());
}

// ---- G: every activation algorithm through its own route ----------------------

void test_routes() {
    const int64_t M = 37, N = 40, K = 192; // padding in I and J, 6 K groups
    for (algo a : { algo::tensor, algo::token, algo::block, algo::stripe, algo::exsia }) {
        const std::string        name = std::string("G ") + algo_name(a);
        const std::vector<float> xf   = random_activations(M, K, 21);
        mm_case                  c(M, N, K, xf, ResidualRoute::cpu_direct, 22);
        route_plan               route;
        const bool q = c.ok && quantize_with(a, c.x, c.args, c.o) && select_route(c.args, route);
        check(name + " quantizer produced its own metadata and route", q && c.args.act_quant.kind() == kind_of(a));
        if (!q) {
            continue;
        }
        std::vector<float> want;
        const bool         golden = route_golden(route, c, want);
        emu_runner         er;
        const route_result r = execute_route(route, c.args, c.o, er.fn());
        bool               exact = golden && r.ok;
        for (size_t i = 0; exact && i < want.size(); ++i) {
            exact = c.out()[i] == want[i];
        }
        check(name + " " + route.name + ": dst == integer-domain golden from the raw metadata", exact);
        check(name + " " + route.name + ": " +
                  (route.k_groups ? "one GEMM per 32-wide K group" : "one GEMM over the whole K"),
              r.ok && r.jobs == (route.k_groups ? (uint64_t) (K / 32) : 1u) &&
                  r.k_groups == (route.k_groups ? (size_t) (K / 32) : 0u));
        // BLOCK and EXSIA dequantization expand cpu_direct residuals as well
        // as the dense codes. Compare like with like: the Main result above
        // remains checked independently, then complete its captured residuals.
        // Tensor/token/stripe dequantizers expand only ws_packet residuals,
        // not the cpu_direct payloads captured by this fixture.
        const bool residual_dequant = a == algo::block || a == algo::exsia;
        const compensation_result comp = r.ok && residual_dequant ? apply_compensation(c.args) : compensation_result{};
        const size_t payloads = gq::activation_direct_residuals(c.args).size();
        if (residual_dequant) {
            check(name + " complete output executes every captured direct residual",
                  r.ok && comp.ok && comp.payloads == payloads && comp.executed == payloads);
        }
        std::vector<float>        xhat;
        const bool                dq   = dequantize_with(a, c.args, xhat);
        const double              err  = dq ? rel_l2(c.out(), fp_reference(c, xhat)) : 1.0;
        std::printf("     %s %s rel_l2(dst, dequant(A) x dequant(W)) = %.3g\n", algo_name(a), route.name, err);
        check(name + " dst == the algorithm's dequantized activation x dequantized HP1 (rel_l2 < 1e-5)",
              comp.ok && dq && err < 1e-5);
    }
}

// ---- H: error compensation as configured ---------------------------------------

void test_compensation() {
    const int64_t            M = 6, N = 32, K = 128;
    const std::vector<float> xf = random_activations(M, K, 41, 400.0f); // strong outliers
    mm_case                  c(M, N, K, xf, ResidualRoute::cpu_direct, 42);
    route_plan               route;
    const bool q = c.ok && quantize_activations(c.x, c.args, c.o) && select_route(c.args, route);
    emu_runner         er;
    const route_result r = q ? execute_route(route, c.args, c.o, er.fn()) : route_result{};
    const std::vector<float> main_out(c.out(), c.out() + M * N);
    const compensation_result comp = r.ok ? apply_compensation(c.args) : compensation_result{};
    const size_t direct  = gq::activation_direct_residuals(c.args).size();
    const size_t packets = gq::activation_rmd_packets(c.args).size();
    std::printf("     compensation=%s residual_payloads=%zu corrections=%zu residual_events=%zu\n",
                options::compensation_name, comp.payloads, comp.executed, comp.residuals);
    if constexpr (!options::compensation) {
        check("H compensation OFF: the quantizer captured no residual payload",
              q && r.ok && direct == 0 && packets == 0);
        check("H compensation OFF: no correction executed, dst unchanged",
              comp.ok && comp.payloads == 0 && comp.executed == 0 &&
                  std::equal(main_out.begin(), main_out.end(), c.out()));
    } else {
        const std::vector<double> exact = fp_reference(c, xf);
        const double              before = rel_l2(main_out.data(), exact);
        const double              after  = rel_l2(c.out(), exact);
        std::printf("     rel_l2 vs F32 activations: main=%.4g compensated=%.4g\n", before, after);
        check("H compensation ON: residuals captured and every payload corrected (cpu_direct)",
              q && r.ok && comp.ok && comp.payloads > 0 && comp.executed == comp.payloads && comp.residuals > 0);
        check("H compensation ON: correction merged into dst and reduces the error", after < before);

        // ws_packet residual payloads need the Gemmini WS executor: refused, never dropped.
        mm_case    w(M, N, K, xf, ResidualRoute::ws_packet, 42);
        const bool wq = w.ok && quantize_activations(w.x, w.args, w.o);
        const compensation_result wc = wq ? apply_compensation(w.args) : compensation_result{};
        check("H ws_packet payload fails explicitly",
              wq && (gq::activation_rmd_packets(w.args).empty() || (!wc.ok && wc.detail != nullptr)));
    }
}

// ---- I: scale generation wrap ---------------------------------------------------

void test_generation_wrap() {
    hp1_operands     o;
    const hp1_shape  s = { 16, 16, 32 };
    std::mt19937     rng(5);
    const bool       alloc = o.allocate(s);
    if (alloc) {
        fill_operands(o, 16, 16, 32, rng, 0);
    }
    std::vector<int32_t> want(o.m_pad * o.ldc, 0);
    reference_gemm_i32(o.a, o.lda, o.b, o.ldb, o.carriers, o.cstride, hp1::block_k, 16, 16, 32, want.data(), o.ldc);
    emulator         emu;
    potal_ws_ops     ops     = emu.ops();
    potal_ws_session session = { &ops, 0, {} };
    const size_t     runs    = (size_t) POTAL_GENERATION_MAX + 5;
    bool             all_ok  = alloc;
    for (size_t i = 0; i < runs && all_ok; ++i) {
        std::fill(o.c, o.c + o.m_pad * o.ldc, 0);
        all_ok = tiled(16, 16, 32, o, session, nullptr) == POTAL_OK &&
                 std::equal(o.c, o.c + o.m_pad * o.ldc, want.begin());
    }
    const auto & g      = emu.loop_generations;
    bool         seq    = all_ok && g.size() == runs;
    bool         wraps  = false;
    for (size_t i = 0; seq && i < g.size(); ++i) {
        seq   = g[i] == (uint32_t) (i % POTAL_GENERATION_MAX) + 1;
        wraps = wraps || (i > 0 && g[i - 1] == POTAL_GENERATION_MAX && g[i] == 1);
    }
    std::printf("     generation: invocations=%zu first=%u last=%u\n", g.size(), g.empty() ? 0u : g.front(),
                g.empty() ? 0u : g.back());
    check("I generation runs 1..255 and wraps to 1 (observed per LOOP_WS), results exact across the wrap",
          seq && wraps && g.back() == (uint32_t) ((runs - 1) % POTAL_GENERATION_MAX) + 1);
    check("I every scale entry released across the wrap", emu.scales.empty() && emu.writes == emu.releases &&
                                                             session.report.scale_writes == emu.writes);
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
    check("F I8 x I8 DIM x DIM x K32 still accepted", check_op(mm(GGML_TYPE_I8, GGML_TYPE_I8, 32, DIM, DIM)) ==
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

// ---- W: prepared-weight cache (matmul/weight_cache.hpp) -------------------------

namespace {

// Q8_HP1 weights in a CPU backend buffer: w[0], w[1] (same shape, different
// values) and a row view of w[0] at an offset.
struct weight_set {
    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf;
    ggml_tensor *           w[2] = {};
    ggml_tensor *           view = nullptr;
    int64_t                 view_row0 = 0;

    weight_set(ggml_backend_t cpu, int64_t N, int64_t K, ggml_backend_buffer_usage usage) {
        ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, true };
        ctx.reset(ggml_init(ip));
        w[0]      = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_HP1, K, N);
        w[1]      = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_HP1, K, N);
        view_row0 = N / 4;
        view      = ggml_view_2d(ctx.get(), w[0], K, N / 2, w[0]->nb[1], (size_t) view_row0 * w[0]->nb[1]);
        buf.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), cpu));
        ggml_backend_buffer_set_usage(buf.get(), usage);
        for (int i = 0; i < 2; ++i) {
            std::mt19937                    rng(500 + i);
            std::normal_distribution<float> nd(0.0f, 1.0f);
            std::vector<float>              f((size_t) (N * K));
            const float                     mag[] = { 0.02f, 0.08f, 0.3f, 0.01f };
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t k = 0; k < K; ++k) {
                    f[n * K + k] = nd(rng) * mag[(k / 32 + n + i) % 4];
                }
            }
            if (N > 3) {
                std::fill(f.begin() + 3 * K, f.begin() + 4 * K, 0.0f); // zero row
            }
            std::vector<uint8_t> q(ggml_nbytes(w[i]));
            ggml_quantize_chunk(GGML_TYPE_Q8_HP1, f.data(), q.data(), 0, N, K, nullptr);
            ggml_backend_tensor_set(w[i], q.data(), 0, q.size());
        }
    }
};

// One MUL_MAT of src0 with M activation rows, as mul_mat_hp1 sets it up.
struct mm_args {
    ggml_context_ptr    ctx;
    ggml_tensor *       x = nullptr;
    ggml_tensor *       y = nullptr;
    hp1_shape           s = {};
    ggml_gemmini_args_t args;
    bool                ok = false;

    mm_args(ggml_tensor * src0, int64_t M, uint32_t seed) {
        const int64_t    K  = src0->ne[0];
        ggml_init_params ip = { (size_t) (M * K + M * src0->ne[1]) * 4 + 1024 * 1024, nullptr, false };
        ctx.reset(ggml_init(ip));
        x                         = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, K, M);
        const std::vector<float> xf = random_activations(M, K, seed);
        std::memcpy(x->data, xf.data(), xf.size() * sizeof(float));
        y = ggml_mul_mat(ctx.get(), src0, x);
        potal_tile_plan plan;
        ok = check_mul_mat_hp1(y, true, &s) == GGML_POTAL_REJECT_NONE &&
             potal_tile_select((size_t) s.m, (size_t) s.n, (size_t) s.k, hp1::block_k / DIM, &plan) == POTAL_OK &&
             init_args(src0, y, s, "gpt2", "test", ResidualRoute::cpu_direct, plan, args);
    }
};

// The prepared operands against the Gemmini reader's per-element accessors
// (the pre-cache unpacking), padding included.
bool same_as_reader(const ggml_gemmini_args_t & args, const hp1_weight & w) {
    namespace wroute  = ggml::gemmini::quants::wroute;
    namespace wreader = ggml::gemmini::quants::wreader;
    const auto plan   = wroute::resolve_weight_route_plan(args, wroute::WeightScaleInfoMode::ResidualHp1Scu);
    if (!plan.valid || w.n != args.J || w.k != args.K) {
        return false;
    }
    for (size_t n = 0; n < w.ldb; ++n) {
        for (size_t k = 0; k < args.K; ++k) {
            const int8_t want = n < args.J ? (int8_t) wreader::read_code_validated(args, plan, n, k).value : 0;
            if (w.b[k * w.ldb + n] != want) {
                return false;
            }
        }
    }
    for (size_t n = 0; n < w.n_pad; ++n) {
        const float scale = n < args.J ? wreader::read_column_scale_validated(args, plan, n, 0).column_scale : 0.0f;
        if (std::memcmp(&w.scale[n], &scale, sizeof(float)) != 0) {
            return false;
        }
        for (size_t b = 0; b < args.K / hp1::block_k; ++b) {
            const uint32_t want = n < args.J ? wreader::read_hp1_carrier_validated(args, plan, n, b).carrier : 0;
            if (w.carriers[b * w.cstride + n] != want) {
                return false;
            }
        }
    }
    return true;
}

bool same_weight(const hp1_weight & a, const hp1_weight & b) {
    return a.n == b.n && a.k == b.k && a.ldb == b.ldb && a.cstride == b.cstride && a.n_pad == b.n_pad &&
           std::memcmp(a.b, b.b, a.k * a.ldb) == 0 &&
           std::memcmp(a.carriers, b.carriers, a.k / hp1::block_k * a.cstride * sizeof(uint32_t)) == 0 &&
           std::memcmp(a.scale, b.scale, a.n_pad * sizeof(float)) == 0;
}

// dst of one MUL_MAT on the emulator: per-call operands (the pre-cache path)
// or the prepared weight w.
bool route_output(algo a, ggml_tensor * src0, int64_t M, uint32_t seed, const hp1_weight * w, std::vector<float> & out,
                  route_plan & route) {
    mm_args      m(src0, M, seed);
    hp1_operands o;
    hp1::stats   ws;
    bool ok = m.ok && (w != nullptr ? o.allocate(m.s, *w)
                                    : o.allocate(m.s) && unpack_weights(m.args, o, &ws) == hp1::status::ok) &&
              quantize_with(a, m.x, m.args, o) && select_route(m.args, route);
    emu_runner er;
    ok = ok && execute_route(route, m.args, o, er.fn()).ok;
    const float * y = static_cast<const float *>(m.y->data);
    out.assign(y, y + ggml_nelements(m.y));
    return ok;
}

bool bitwise_equal(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

void test_weight_cache() {
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    const int64_t    N = 40, K = 192; // N not a multiple of DIM: padded columns
    weight_set       ws(cpu.get(), N, K, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    weight_cache     cache(size_t(64) << 20);
    const hp1_weight * first = nullptr;

    // W1-W3: miss, reader-exact operands, then a hit without any preparation.
    {
        mm_args       m(ws.w[0], 5, 1);
        weight_lookup l = cache.get(ws.w[0], m.args);
        first           = l.weight;
        const auto c    = cache.snapshot();
        check("W1 first use of a weights-buffer weight: MISS, cached", m.ok && l.outcome == weight_cache_outcome::miss &&
                                                                          l.weight && !l.transient && c.entries == 1 &&
                                                                          c.misses == 1 && c.bytes == l.weight->bytes());
        check("W2 prepared B / carriers / channel_scale == the reader's per-element values, padding zero",
              l.weight && same_as_reader(m.args, *l.weight));
        hp1_operands o;
        hp1::stats   st;
        check("W2 prepared operands == the per-call unpacking (B, carriers, scale, stats)",
              l.weight && o.allocate(m.s) && unpack_weights(m.args, o, &st) == hp1::status::ok &&
                  same_weight(*l.weight, o.own) && st.blocks == l.weight->stats.blocks &&
                  st.zero_blocks == l.weight->stats.zero_blocks && st.shift_blocks == l.weight->stats.shift_blocks &&
                  l.weight->stats.zero_rows == 1);
        mm_args       m2(ws.w[0], 1, 2); // another node, other activations
        weight_lookup h = cache.get(ws.w[0], m2.args);
        check("W3 second use: HIT, same operands, nothing allocated or prepared",
              h.outcome == weight_cache_outcome::hit && h.weight == first && h.alloc_s == 0 && h.prepare_s == 0 &&
                  cache.snapshot().hits == 1 && cache.snapshot().entries == 1);
    }

    // W4: hit == miss == per-call path, bitwise, for every activation algorithm,
    // with different activations reusing the one entry.
    for (algo a : { algo::tensor, algo::token, algo::block, algo::stripe, algo::exsia }) {
        bool same = true;
        for (uint32_t seed : { 31u, 32u }) {
            std::vector<float> want, got;
            route_plan         r1, r2;
            mm_args            m(ws.w[0], 7, seed);
            weight_lookup      l = cache.get(ws.w[0], m.args);
            same = same && l.outcome == weight_cache_outcome::hit && route_output(a, ws.w[0], 7, seed, nullptr, want, r1) &&
                   route_output(a, ws.w[0], 7, seed, l.weight, got, r2) && r1.activation == r2.activation &&
                   bitwise_equal(want, got);
        }
        check(std::string("W4 ") + algo_name(a) + "+block: cached-weight dst == per-call dst bitwise (2 activations)",
              same);
    }

    // W5-W7: distinct weights, views and non-weight buffers.
    {
        mm_args       m(ws.w[1], 3, 3);
        weight_lookup l = cache.get(ws.w[1], m.args);
        check("W5 other weight, same shape: MISS, own entry, other contents",
              l.outcome == weight_cache_outcome::miss && l.weight != first && same_as_reader(m.args, *l.weight) &&
                  !same_weight(*l.weight, *first) && cache.snapshot().entries == 2);
        mm_args       mv(ws.view, 3, 4);
        weight_lookup v = cache.get(ws.view, mv.args);
        bool rows = v.weight != nullptr && v.weight->n == (size_t) (N / 2);
        for (size_t k = 0; rows && k < (size_t) K; ++k) {
            for (size_t n = 0; rows && n < (size_t) (N / 2); ++n) {
                rows = v.weight->b[k * v.weight->ldb + n] == first->b[k * first->ldb + n + (size_t) ws.view_row0];
            }
        }
        check("W6 row view at an offset: own entry, == the viewed rows of the base weight",
              v.outcome == weight_cache_outcome::miss && rows && same_as_reader(mv.args, *v.weight) &&
                  cache.get(ws.view, mv.args).outcome == weight_cache_outcome::hit);

        weight_set    other(cpu.get(), N, K, GGML_BACKEND_BUFFER_USAGE_ANY);
        mm_args       mo(other.w[0], 3, 5);
        weight_lookup u1 = cache.get(other.w[0], mo.args);
        weight_lookup u2 = cache.get(other.w[0], mo.args);
        mm_case       plain(3, N, K, random_activations(3, K, 6), ResidualRoute::cpu_direct, 7); // no buffer
        weight_lookup u3 = cache.get(plain.w, plain.args);
        check("W7 compute-buffer and buffer-less weights: UNCACHED every time, prepared per call",
              u1.outcome == weight_cache_outcome::uncached && u2.outcome == weight_cache_outcome::uncached &&
                  u3.outcome == weight_cache_outcome::uncached && u1.transient && u1.weight == u1.transient.get() &&
                  same_as_reader(mo.args, *u1.weight) && cache.snapshot().entries == 3 && cache.snapshot().uncached == 3);
    }

    // W8-W9: invalid weights are never cached; mutation needs clear().
    {
        weight_set   bad(cpu.get(), N, K, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        weight_cache c2(size_t(64) << 20);
        auto *       blocks = static_cast<block_q8_hp1 *>(bad.w[0]->data);
        blocks[5].padding[0] = 1;
        mm_args       m(bad.w[0], 3, 8);
        weight_lookup f1 = c2.get(bad.w[0], m.args);
        weight_lookup f2 = c2.get(bad.w[0], m.args);
        check("W8 invalid weight (non-zero padding): rejected as metadata, not cached, rejected again",
              f1.outcome == weight_cache_outcome::failed && f1.status == hp1::status::metadata && !f1.weight &&
                  f2.outcome == weight_cache_outcome::failed && f2.status == hp1::status::metadata &&
                  c2.snapshot().entries == 0 && c2.snapshot().failures == 2);
        blocks[5].padding[0] = 0;
        weight_lookup ok = c2.get(bad.w[0], m.args);
        check("W8 the repaired weight is validated (MISS) and cached", ok.outcome == weight_cache_outcome::miss);
        // Documented contract: a hit does not re-read the weight, so a mutation
        // is invisible until clear(); after clear() the weight is validated again.
        blocks[7].padding[1] = 3;
        weight_lookup h = c2.get(bad.w[0], m.args);
        c2.clear();
        weight_lookup after = c2.get(bad.w[0], m.args);
        check("W9 mutated weight: HIT before clear() (no re-read), rejected after clear(), nothing cached",
              h.outcome == weight_cache_outcome::hit && after.outcome == weight_cache_outcome::failed &&
                  after.status == hp1::status::metadata && c2.snapshot().entries == 0 && c2.snapshot().bytes == 0);
        blocks[7].padding[1] = 0;
    }

    // W10: budget.
    {
        mm_args       m(ws.w[0], 3, 9);
        weight_cache  small(first->bytes() - 1);
        weight_lookup b1 = small.get(ws.w[0], m.args);
        weight_lookup b2 = small.get(ws.w[0], m.args);
        weight_cache  off(0);
        weight_lookup o1 = off.get(ws.w[0], m.args);
        weight_cache  exact(first->bytes());
        weight_lookup e1 = exact.get(ws.w[0], m.args);
        weight_lookup e2 = exact.get(ws.w[1], m.args);
        check("W10 over the budget: BYPASS each time, prepared per call, nothing cached",
              b1.outcome == weight_cache_outcome::bypass && b2.outcome == weight_cache_outcome::bypass && b1.transient &&
                  same_weight(*b1.weight, *first) && small.snapshot().entries == 0 && small.snapshot().bypasses == 2);
        check("W10 budget 0 disables the cache", o1.outcome == weight_cache_outcome::uncached &&
                                                     off.snapshot().entries == 0);
        check("W10 full cache: the next weight bypasses, the cached one stays",
              e1.outcome == weight_cache_outcome::miss && e2.outcome == weight_cache_outcome::bypass &&
                  exact.get(ws.w[0], m.args).outcome == weight_cache_outcome::hit && exact.snapshot().entries == 1);
    }

    // W11: concurrent first use prepares once.
    {
        weight_cache                c3(size_t(64) << 20);
        mm_args                     m(ws.w[0], 3, 10);
        std::vector<std::thread>    threads;
        std::vector<const hp1_weight *> seen(8, nullptr);
        for (size_t i = 0; i < seen.size(); ++i) {
            threads.emplace_back([&, i] { seen[i] = c3.get(ws.w[0], m.args).weight; });
        }
        for (auto & t : threads) {
            t.join();
        }
        const auto c = c3.snapshot();
        check("W11 8 threads, one weight: 1 MISS + 7 HIT, one entry, every thread got it",
              c.misses == 1 && c.hits == 7 && c.entries == 1 &&
                  std::all_of(seen.begin(), seen.end(), [&](const hp1_weight * p) { return p && p == seen[0]; }));
    }

    // W12: allocation failure and operand shape checks.
    {
        hp1_weight huge;
        check("W12 impossible weight allocation fails cleanly", !huge.allocate(int64_t(1) << 40, int64_t(1) << 20));
        hp1_operands o;
        check("W12 operands refuse a prepared weight of another shape", !o.allocate({ 3, N, K + 32 }, *first) &&
                                                                         !o.allocate({ 3, N + 1, K }, *first));
    }

    // W13: through the backend. Legacy FULL / serial-pipeline hardware
    // failures are fatal and abandon the prepared weight. On the host a
    // Paired request instead gets NO_HARDWARE before any command takes
    // effect; its recoverable refusal safely retains the cached weight.
    {
        ggml_backend_ptr potal(ggml_backend_potal_init());
        const auto selected = ggml::gemmini::resolve_matmul_options();
        bool invalid_paired = false;
        const bool paired = options::paired_auto_requested(&invalid_paired);
        const bool retain = selected.ok() && !invalid_paired &&
                            selected.options.mode == options::MatmulInvocationMode::stripe_pipeline &&
                            options::compensation && paired;
        check("W13 selected mode is valid", selected.ok() && !invalid_paired);
        const auto       compute = [&](ggml_tensor * w0, ggml_backend_potal_stats & st) {
            mm_args     m(w0, 2, 11);
            ggml_cgraph * g = ggml_new_graph(m.ctx.get());
            ggml_build_forward_expand(g, m.y);
            const ggml_status status = ggml_backend_graph_compute(potal.get(), g);
            ggml_backend_potal_get_stats(potal.get(), &st);
            return status;
        };
        ggml_backend_potal_stats s1 = {}, s2 = {}, s3 = {};
        const ggml_status        r1 = compute(ws.w[0], s1);
        const ggml_status        r2 = compute(ws.w[0], s2);
        mm_case                  plain(2, N, K, random_activations(2, K, 12), ResidualRoute::cpu_direct, 13);
        const ggml_status        r3 = compute(plain.w, s3);
        check("W13 backend: weights-buffer node MISS, validated and packed once, counted in weight_pack_s",
              r1 == GGML_STATUS_FAILED && s1.weight_cache_misses == 1 && s1.weight_validations == 1 &&
                  s1.weight_packs == 1 && s1.weight_pack_bytes == first->bytes() && s1.weight_pack_s > 0 &&
                  (std::getenv("GGML_POTAL_WEIGHT_CACHE_MB") != nullptr ||
                   s1.weight_cache_budget == (uint64_t(1024) << 20)));
        check("W13 backend: no hardware completes no work",
              s1.last_status == POTAL_ERR_NO_HARDWARE && s1.hp1_completed == 0 && s2.hp1_completed == 0 &&
                  s1.tiled_invocations == 0 && s1.paired_main_loop_ws == 0 && s1.scale_writes == 0);
        check(retain ? "W13 backend: Paired no-effect refusal retains the entry; the next node is a HIT"
                     : "W13 backend: fatal hardware failure abandons the entry; the next node prepares again",
              s1.weight_cache_entries == (retain ? 1u : 0u) && r2 == GGML_STATUS_FAILED &&
                  s2.weight_cache_entries == (retain ? 1u : 0u) && s2.weight_cache_misses == (retain ? 1u : 2u) &&
                  s2.weight_cache_hits == (retain ? 1u : 0u) && s2.weight_validations == (retain ? 1u : 2u));
        check("W13 backend: buffer-less weight UNCACHED", r3 == GGML_STATUS_FAILED && s3.weight_cache_uncached == 1 &&
                  s3.weight_cache_entries == (retain ? 1u : 0u) && s3.weight_validations == (retain ? 2u : 3u));
        ggml_backend_potal_clear_weight_cache(potal.get());
    }

    // W14: weights in the POTAL buffer type (what llama.cpp now allocates for
    // POTAL MUL_MAT weights). Same MISS/HIT contract; a buffer released and
    // replaced over the same bytes (same data address, possibly the same
    // buffer address) is a new generation, so its first use is a MISS.
    {
        ggml_backend_ptr potal(ggml_backend_potal_init());
        weight_set       pw(potal.get(), N, K, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        weight_cache     c4(size_t(64) << 20);
        mm_args          m(pw.w[0], 5, 14);
        const weight_lookup l1 = c4.get(pw.w[0], m.args);
        const weight_lookup l2 = c4.get(pw.w[0], m.args);
        check("W14 POTAL weights buffer: eligible, MISS then HIT, reader-exact",
              m.ok && ggml::potal::is_potal_buffer(pw.buf.get()) && weight_cache::eligible(pw.w[0]) &&
                  l1.outcome == weight_cache_outcome::miss && l2.outcome == weight_cache_outcome::hit &&
                  l2.weight == l1.weight && same_as_reader(m.args, *l1.weight));

        ggml_backend_dev_t dev   = ggml_backend_get_device(potal.get());
        const size_t       bytes = ggml_nbytes(pw.w[0]);
        const void *       data  = pw.w[0]->data;
        uint64_t           gen[2] = {};
        const void *       addr[2] = {};
        weight_cache_outcome first_use[2] = { weight_cache_outcome::failed, weight_cache_outcome::failed };
        weight_cache_outcome second_use   = weight_cache_outcome::failed;
        bool                 same_operands = true;
        for (int i = 0; i < 2; ++i) {
            ggml_init_params      ip = { 2 * ggml_tensor_overhead(), nullptr, true };
            ggml_context_ptr      ctx(ggml_init(ip));
            ggml_tensor *         t     = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_HP1, K, N);
            ggml_backend_buffer_t alias = ggml_backend_dev_buffer_from_host_ptr(dev, const_cast<void *>(data), bytes, bytes);
            if (alias == nullptr || ggml_backend_tensor_alloc(alias, t, const_cast<void *>(data)) != GGML_STATUS_SUCCESS) {
                same_operands = false;
                if (alias) {
                    ggml_backend_buffer_free(alias);
                }
                break;
            }
            ggml_backend_buffer_set_usage(alias, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            gen[i]                 = ggml::potal::potal_buffer_generation(alias);
            addr[i]                = alias;
            mm_args             ma(t, 5, 14);
            const weight_lookup a  = c4.get(t, ma.args);
            first_use[i]           = a.outcome;
            same_operands          = same_operands && ma.ok && a.weight && same_weight(*a.weight, *l1.weight);
            if (i == 1) {
                second_use = c4.get(t, ma.args).outcome;
            }
            ggml_backend_buffer_free(alias);
        }
        std::printf("INFO W14 alias buffers at %p then %p (address %s)\n", addr[0], addr[1],
                    addr[0] == addr[1] ? "reused" : "not reused");
        check("W14 released and replaced POTAL buffer over the same bytes: new generation, MISS, then HIT",
              gen[0] != 0 && gen[1] != 0 && gen[0] != gen[1] && first_use[0] == weight_cache_outcome::miss &&
                  first_use[1] == weight_cache_outcome::miss && second_use == weight_cache_outcome::hit);
        check("W14 every entry prepared the same operands", same_operands);
        c4.clear();
    }
}

} // namespace

int main() {
    std::printf("POTAL_TILING_UNIT_TEST DIM=%u activation_quant=%s compensation=%s\n", (unsigned) POTAL_HW_DIM,
                options::activation_quant_name(), options::compensation_name);
    test_tile_select();
    test_tiled_execution();
    test_hp1();
    test_configured_dispatcher();
    test_end_to_end();
    test_routes();
    test_compensation();
    test_generation_wrap();
    test_ggml_contract();
    test_weight_cache();
    if (failures != 0) {
        std::printf("POTAL_TILING_UNIT_TEST FAIL (%d)\n", failures);
        return 1;
    }
    std::printf("POTAL_TILING_UNIT_TEST PASS\n");
    return 0;
}
