// FireSim hardware test of the W4-5 path: GGML MUL_MAT F32 x Q8_HP1 graphs
// computed by the POTAL backend (per-tensor activation quantization, HP1
// carriers, potal_tiled_matmul_auto), each output compared element-exactly
// against the CPU integer golden (dense.hpp reference_gemm_i32 + the same
// reconstruction). No error compensation exists on either side.
//
//   potal-hp1-smoke            all cases
//   potal-hp1-smoke --quick    the validated single GEMM and one padded case
//
// PASS needs, per case: graph_compute OK, dst == golden for every element,
// hp1_completed +1, every invocation fenced, every scale entry released.
// Exit 0 only on POTAL_HP1_SMOKE PASS.

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-potal.h"
#include "ggml-quants.h"
#include "matmul/dense.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace ggml::potal;
namespace hp1 = ggml::potal::quants::weight::hp1;

namespace {

int failures = 0;

ggml_backend_potal_stats stats_of(ggml_backend_t potal) {
    ggml_backend_potal_stats st = {};
    ggml_backend_potal_get_stats(potal, &st);
    return st;
}

// Golden of one F32 x Q8_HP1 MUL_MAT into `expected` (F32, same shape as dst).
bool golden(const ggml_tensor * w, const ggml_tensor * x, ggml_tensor * expected) {
    hp1_shape         s = { x->ne[1], w->ne[1], w->ne[0] };
    hp1_operands      o;
    ggml_potal_args_t args;
    float             act_scale = 0.0f;
    hp1::stats        ws;
    if (!o.allocate(s) || !init_args(w, x, expected, s, "", "golden", args) ||
        !quantize_activations(x, args, o, &act_scale) || unpack_weights(args, o, &ws) != hp1::status::ok) {
        return false;
    }
    reference_gemm_i32(o.a, o.lda, o.b, o.ldb, o.carriers, o.cstride, hp1::block_k, s.m, s.n, s.k, o.c, o.ldc);
    reconstruct(o, args, act_scale);
    return true;
}

ggml_tensor * hp1_weight(ggml_context * ctx, int64_t K, int64_t N, std::mt19937 & rng, int mode) {
    ggml_tensor *                   w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_HP1, K, N);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float>              f(N * K);
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t k = 0; k < K; ++k) {
            const float mag[] = { 1.0f, 8.0f, 0.125f, 64.0f };
            f[n * K + k]      = nd(rng) * 0.02f * mag[(k / 32 + n) % 4];
        }
    }
    if (N > 3) {
        std::fill(f.begin() + 3 * K, f.begin() + 4 * K, 0.0f); // zero row
    }
    if (K >= 64) {
        std::fill(f.begin() + 32, f.begin() + 64, 0.0f); // zero block in row 0
    }
    ggml_quantize_chunk(GGML_TYPE_Q8_HP1, f.data(), w->data, 0, N, K, nullptr);
    if (mode == 1) {
        // Large exponents: SCU and ACC Sat32 saturate (still valid HP1 metadata).
        auto * b = static_cast<block_q8_hp1 *>(w->data);
        for (int64_t i = 0; i < N * K / 32; ++i) {
            if (b[i].m != INT16_MIN) {
                b[i].m = (int16_t) (20 + i % 14); // 20..33
            }
        }
    }
    return w;
}

ggml_tensor * activations(ggml_context * ctx, int64_t K, int64_t M, std::mt19937 & rng) {
    ggml_tensor *                   x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    float *                         d = static_cast<float *>(x->data);
    for (int64_t i = 0; i < K * M; ++i) {
        d[i] = nd(rng);
    }
    return x;
}

bool same(const ggml_tensor * got, const ggml_tensor * want, const char * name) {
    const float * g = static_cast<const float *>(got->data);
    const float * w = static_cast<const float *>(want->data);
    for (int64_t i = 0; i < ggml_nelements(got); ++i) {
        if (std::memcmp(&g[i], &w[i], sizeof(float)) != 0) {
            std::printf("POTAL_HP1_MISMATCH case=%s index=%lld got=%.9g want=%.9g\n", name, (long long) i,
                        (double) g[i], (double) w[i]);
            return false;
        }
    }
    return true;
}

void run_case(ggml_backend_t potal, const char * name, int64_t M, int64_t N, int64_t K, int mode,
              std::mt19937 & rng) {
    ggml_init_params ip = { (size_t) (4 * (N * K + 2 * M * N + M * K) + 64 * 1024 * 1024), nullptr, false };
    ggml_context_ptr ctx(ggml_init(ip));
    ggml_tensor *    w        = hp1_weight(ctx.get(), K, N, rng, mode);
    ggml_tensor *    x        = activations(ctx.get(), K, M, rng);
    ggml_tensor *    y        = ggml_mul_mat(ctx.get(), w, x);
    ggml_tensor *    expected = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, N, M);
    ggml_cgraph *    g        = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(g, y);

    const bool                     have_golden = golden(w, x, expected);
    const ggml_backend_potal_stats before      = stats_of(potal);
    const ggml_status              status      = ggml_backend_graph_compute(potal, g);
    const ggml_backend_potal_stats after       = stats_of(potal);
    const uint64_t                 loops       = after.tiled_invocations - before.tiled_invocations;
    const bool ok = have_golden && status == GGML_STATUS_SUCCESS && same(y, expected, name) &&
                    after.hp1_completed == before.hp1_completed + 1 && after.hp1_failed == before.hp1_failed &&
                    loops > 0 && after.tiled_fences - before.tiled_fences == loops &&
                    after.scale_writes - before.scale_writes == after.scale_releases - before.scale_releases;
    std::printf("POTAL_HP1_CASE %s case=%s M=%lld N=%lld K=%lld invocations=%llu validated_api=%llu scale_writes=%llu "
                "generation=%u act_q_ms=%.3f pack_ms=%.3f scale_ms=%.3f issue_ms=%.3f fence_ms=%.3f recon_ms=%.3f\n",
                ok ? "PASS" : "FAIL", name, (long long) M, (long long) N, (long long) K, (unsigned long long) loops,
                (unsigned long long) (after.tiled_validated - before.tiled_validated),
                (unsigned long long) (after.scale_writes - before.scale_writes), after.last_generation,
                (after.act_quant_s - before.act_quant_s) * 1e3, (after.weight_pack_s - before.weight_pack_s) * 1e3,
                (after.scale_s - before.scale_s) * 1e3, (after.issue_s - before.issue_s) * 1e3,
                (after.fence_s - before.fence_s) * 1e3, (after.reconstruct_s - before.reconstruct_s) * 1e3);
    failures += ok ? 0 : 1;
}

// Two MUL_MATs in one graph: the second consumes the first's hardware output.
void run_chain(ggml_backend_t potal, std::mt19937 & rng) {
    const int64_t    M = 4, K = 128, H = 96, N = 64;
    ggml_init_params ip = { 64 * 1024 * 1024, nullptr, false };
    ggml_context_ptr ctx(ggml_init(ip));
    ggml_tensor *    w1 = hp1_weight(ctx.get(), K, H, rng, 0);
    ggml_tensor *    w2 = hp1_weight(ctx.get(), H, N, rng, 0);
    ggml_tensor *    x  = activations(ctx.get(), K, M, rng);
    ggml_tensor *    h  = ggml_mul_mat(ctx.get(), w1, x);
    ggml_tensor *    y  = ggml_mul_mat(ctx.get(), w2, h);
    ggml_tensor *    eh = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, H, M);
    ggml_tensor *    ey = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, N, M);
    ggml_cgraph *    g  = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(g, y);

    const bool have_golden = golden(w1, x, eh) && golden(w2, eh, ey);
    const auto before      = stats_of(potal);
    const auto status      = ggml_backend_graph_compute(potal, g);
    const auto after       = stats_of(potal);
    const bool ok          = have_golden && status == GGML_STATUS_SUCCESS && same(h, eh, "chain.layer1") &&
                    same(y, ey, "chain.layer2") && after.hp1_completed == before.hp1_completed + 2;
    std::printf("POTAL_HP1_CASE %s case=chain2 M=%lld K=%lld H=%lld N=%lld\n", ok ? "PASS" : "FAIL", (long long) M,
                (long long) K, (long long) H, (long long) N);
    failures += ok ? 0 : 1;
}

} // namespace

int main(int argc, char ** argv) {
    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    std::mt19937 rng(20261017);

    ggml_backend_ptr potal(ggml_backend_potal_init());
    if (!potal) {
        std::printf("POTAL_HP1_SMOKE FAIL backend init\n");
        return 1;
    }

    run_case(potal.get(), "validated_16x16x32", 16, 16, 32, 0, rng); // potal_gemm_ws_issue path
    run_case(potal.get(), "padded_5x40x96", 5, 40, 96, 0, rng);
    if (!quick) {
        run_case(potal.get(), "one_tile_k64", 16, 16, 64, 0, rng);
        run_case(potal.get(), "multi_tile_37x100x640", 37, 100, 640, 0, rng);
        run_case(potal.get(), "k_chunks_3x160x4096", 3, 160, 4096, 0, rng);
        run_case(potal.get(), "decode_1x300x768", 1, 300, 768, 0, rng);
        run_case(potal.get(), "sat32_17x48x256", 17, 48, 256, 1, rng);
        run_chain(potal.get(), rng);
        // Generation wrap: more than 255 invocations in one process.
        for (int i = 0; i < 4; ++i) {
            run_case(potal.get(), "repeat_40x160x1600", 40, 160, 1600, 0, rng);
        }
    }

    const ggml_backend_potal_stats st = stats_of(potal.get());
    std::printf("POTAL_HP1_STATS hp1_nodes=%llu completed=%llu failed=%llu invocations=%llu validated_api=%llu "
                "fences=%llu scale_writes=%llu scale_releases=%llu zero_carriers=%llu "
                "issuer_ready=%d issuer_cpu=%d\n",
                (unsigned long long) st.hp1_nodes, (unsigned long long) st.hp1_completed,
                (unsigned long long) st.hp1_failed, (unsigned long long) st.tiled_invocations,
                (unsigned long long) st.tiled_validated, (unsigned long long) st.tiled_fences,
                (unsigned long long) st.scale_writes, (unsigned long long) st.scale_releases,
                (unsigned long long) st.zero_carriers, st.issuer_ready ? 1 : 0,
                st.issuer_cpu);
    if (failures != 0 || st.hp1_failed != 0) {
        std::printf("POTAL_HP1_SMOKE FAIL (%d)\n", failures);
        return 1;
    }
    std::printf("POTAL_HP1_SMOKE PASS\n");
    return 0;
}
